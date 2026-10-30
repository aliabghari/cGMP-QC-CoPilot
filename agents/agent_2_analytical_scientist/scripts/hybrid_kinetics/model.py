"""
Phase 3: Hybrid Mechanistic + Neural-Network Kinetic Model (`model.py`).

Architecture:
- `BoundedKineticMLP`: Compact 2-hidden-layer MLP (`[32, 32]`, SiLU activations) that maps
  standardized observable lot covariates (`LOT_COVARIATE_COLUMNS` + quadratic `(pH - 6.0)^2`)
  to sigmoid-bounded thermodynamic parameters inside the configured physical bounds of
  `hybrid_config.json`.
- Never predicts `% HMW`, `% Monomer`, or unconstrained rate constants `k` directly.
- Couples predicted thermodynamic parameters directly into the deterministic Lumry-Eyring +
  Arrhenius physics layer (`physics.py`) and 4th-order Runge-Kutta (RK4) ODE integrator (`ode.py`).
- Supports both PyTorch (`torch.nn.Module` with full autograd graph) and pure NumPy inference
  from exported `.npz` weight dictionaries with `< 1e-6` relative parity.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False

from .config import load_hybrid_config
from .ode import integrate_pathway_rk4
from .physics import (
    HOURS_PER_MONTH,
    apparent_activation_energy,
    celsius_to_kelvin,
    evaluate_pathway_rates,
    lumry_eyring_rates,
)
from .synth import LOT_COVARIATE_COLUMNS

# The 8 bounded thermodynamic parameters predicted by the ANN
PREDICTED_PARAM_KEYS: Tuple[str, ...] = (
    "dG_eff_ref_kj_mol",
    "dH_agg_ref_kj_mol",
    "ku_ref_5c",
    "dH_u_kj_mol",
    "dG_frag_ref_kj_mol",
    "ea_frag_kj_mol",
    "dG_acidic_ref_kj_mol",
    "ea_acidic_kj_mol",
)


def extract_feature_matrix(
    df_or_dict: Any,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the 11-dimensional design matrix X from observable lot covariates:
    the 10 `LOT_COVARIATE_COLUMNS` plus the raw quadratic feature `ph ** 2`
    (standardized separately alongside `ph` so any pH optimum must be learned from data,
    never hard-coded from the generator's true optimum).
    Strictly refuses any ground-truth or `structural_group` column.
    """
    if isinstance(df_or_dict, dict):
        raw_cols = [float(df_or_dict[c]) for c in LOT_COVARIATE_COLUMNS]
        x_base = np.asarray(raw_cols, dtype=np.float64)[None, :]
    else:
        for col in LOT_COVARIATE_COLUMNS:
            if col not in df_or_dict.columns:
                raise KeyError(f"Missing required observable lot covariate '{col}'.")
        x_base = df_or_dict[list(LOT_COVARIATE_COLUMNS)].to_numpy(dtype=np.float64)

    ph_col = x_base[:, 0:1]
    ph_sq = ph_col ** 2
    x_raw = np.hstack([x_base, ph_sq])

    if feature_mean is None:
        feature_mean = np.mean(x_raw, axis=0)
    if feature_std is None:
        feature_std = np.std(x_raw, axis=0)
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    x_norm = (x_raw - feature_mean) / feature_std
    return x_norm, feature_mean, feature_std


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def map_raw_logits_to_bounded_params_numpy(
    raw_out: np.ndarray,
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, np.ndarray | float]:
    """
    Maps unconstrained ANN output logits `(..., 9)` to physically bounded thermodynamic
    parameters using smooth sigmoid transformations (and log-space sigmoid for `ku_ref_5c`).
    """
    if cfg is None:
        cfg = load_hybrid_config()
    bounds = cfg["parameter_bounds"]

    squeeze_scalar = raw_out.ndim == 1
    logits = np.atleast_2d(np.asarray(raw_out, dtype=np.float64))
    u = _sigmoid_np(logits)

    params: Dict[str, np.ndarray | float] = {}
    for idx, key in enumerate(PREDICTED_PARAM_KEYS):
        b_min = float(bounds[key]["min"])
        b_max = float(bounds[key]["max"])
        if key == "ku_ref_5c":
            log_min = np.log(b_min)
            log_max = np.log(b_max)
            val = np.exp(log_min + u[:, idx] * (log_max - log_min))
        else:
            val = b_min + u[:, idx] * (b_max - b_min)
        params[key] = float(val[0]) if squeeze_scalar else val

    # 9th output head: bounded log-scale aleatoric noise adjustment factor in [-0.35, 0.50]
    log_sigma_adj = -0.35 + u[:, 8] * (0.50 - (-0.35))
    params["log_sigma_adj"] = float(log_sigma_adj[0]) if squeeze_scalar else log_sigma_adj

    # Fixed structural assumptions in the fitted model (dCp_agg = 0, dCp_u = 0)
    params["dCp_agg_kj_mol_k"] = 0.0 if squeeze_scalar else np.zeros(logits.shape[0], dtype=np.float64)
    params["dCp_u_kj_mol_k"] = 0.0 if squeeze_scalar else np.zeros(logits.shape[0], dtype=np.float64)

    # Compute derived intrinsic dG_agg_ref and dS_agg_ref at T_ref = 278.15 K
    r_ref = lumry_eyring_rates(278.15, **params)
    params["dG_agg_ref_kj_mol"] = r_ref["dG_agg_ref_kj_mol"]
    params["dS_agg_ref_j_mol_k"] = r_ref["dS_agg_ref_j_mol_k"]
    return params


def predict_params_from_npz_member(
    x_norm: np.ndarray,
    member_weights: Dict[str, np.ndarray],
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, np.ndarray | float]:
    """
    Pure NumPy forward pass through a single MLP ensemble member's exported weights.
    """
    x = np.asarray(x_norm, dtype=np.float64)
    h1 = _silu_np(x @ member_weights["w1"].T + member_weights["b1"])
    h2 = _silu_np(h1 @ member_weights["w2"].T + member_weights["b2"])
    logits = h2 @ member_weights["w3"].T + member_weights["b3"]
    return map_raw_logits_to_bounded_params_numpy(logits, cfg=cfg)


if HAS_TORCH:

    class BoundedKineticMLP(nn.Module):
        """
        Differentiable PyTorch MLP that predicts bounded thermodynamic parameters
        (`dG_eff_ref_kj_mol`, `dH_agg_ref_kj_mol`, `ku_ref_5c`, `dH_u_kj_mol`,
        plus secondary pathway parameters and log-scale aleatoric noise adjustment)
        from standardized observable lot covariates.
        """

        def __init__(
            self,
            input_dim: int = 11,
            hidden_dims: Sequence[int] = (32, 32),
            cfg: Dict[str, Any] | None = None,
            seed: int = 20261023,
        ) -> None:
            super().__init__()
            if cfg is None:
                cfg = load_hybrid_config()
            self.cfg = cfg
            self.bounds = cfg["parameter_bounds"]

            torch.manual_seed(seed)
            h1_dim, h2_dim = int(hidden_dims[0]), int(hidden_dims[1])
            self.fc1 = nn.Linear(input_dim, h1_dim, dtype=torch.float64)
            self.fc2 = nn.Linear(h1_dim, h2_dim, dtype=torch.float64)
            self.fc3 = nn.Linear(h2_dim, len(PREDICTED_PARAM_KEYS) + 1, dtype=torch.float64)
            self.act = nn.SiLU()

            self._init_near_nominal()

        def _init_near_nominal(self) -> None:
            """
            Initializes final layer biases so initial predictions start near the physical
            nominal values in `hybrid_config.json`, with small weight variance for diversity.
            """
            nn.init.xavier_normal_(self.fc1.weight, gain=0.6)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_normal_(self.fc2.weight, gain=0.6)
            nn.init.zeros_(self.fc2.bias)
            nn.init.normal_(self.fc3.weight, mean=0.0, std=0.05)

            bias_vals = []
            for key in PREDICTED_PARAM_KEYS:
                b_min = float(self.bounds[key]["min"])
                b_max = float(self.bounds[key]["max"])
                b_nom = float(self.bounds[key]["nominal"])
                if key == "ku_ref_5c":
                    frac = (np.log(b_nom) - np.log(b_min)) / (np.log(b_max) - np.log(b_min))
                else:
                    frac = (b_nom - b_min) / (b_max - b_min)
                frac = float(np.clip(frac, 0.05, 0.95))
                bias_vals.append(float(np.log(frac / (1.0 - frac))))
            # 9th head (log_sigma_adj = 0.0 inside [-0.35, 0.50])
            frac_sig = (0.0 - (-0.35)) / (0.50 - (-0.35))
            bias_vals.append(float(np.log(frac_sig / (1.0 - frac_sig))))

            with torch.no_grad():
                self.fc3.bias.copy_(torch.tensor(bias_vals, dtype=torch.float64))

        def forward_logits(self, x_norm: torch.Tensor) -> torch.Tensor:
            h1 = self.act(self.fc1(x_norm))
            h2 = self.act(self.fc2(h1))
            return self.fc3(h2)

        def forward(self, x_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
            squeeze_scalar = x_norm.ndim == 1
            x_in = x_norm.unsqueeze(0) if squeeze_scalar else x_norm
            logits = self.forward_logits(x_in)
            u = torch.sigmoid(logits)

            params: Dict[str, torch.Tensor] = {}
            for idx, key in enumerate(PREDICTED_PARAM_KEYS):
                b_min = float(self.bounds[key]["min"])
                b_max = float(self.bounds[key]["max"])
                if key == "ku_ref_5c":
                    log_min = float(np.log(b_min))
                    log_max = float(np.log(b_max))
                    val = torch.exp(log_min + u[:, idx] * (log_max - log_min))
                else:
                    val = b_min + u[:, idx] * (b_max - b_min)
                params[key] = val[0] if squeeze_scalar else val

            log_sigma_adj = -0.35 + u[:, 8] * (0.50 - (-0.35))
            params["log_sigma_adj"] = log_sigma_adj[0] if squeeze_scalar else log_sigma_adj

            zeros = torch.zeros_like(params["dG_eff_ref_kj_mol"])
            params["dCp_agg_kj_mol_k"] = zeros
            params["dCp_u_kj_mol_k"] = zeros

            r_ref = lumry_eyring_rates(278.15, **params)
            params["dG_agg_ref_kj_mol"] = r_ref["dG_agg_ref_kj_mol"]
            params["dS_agg_ref_j_mol_k"] = r_ref["dS_agg_ref_j_mol_k"]
            return params

        def export_numpy_weights(self) -> Dict[str, np.ndarray]:
            """Exports layer weights and biases as float64 NumPy arrays for PyTorch-free inference."""
            return {
                "w1": self.fc1.weight.detach().cpu().numpy().astype(np.float64),
                "b1": self.fc1.bias.detach().cpu().numpy().astype(np.float64),
                "w2": self.fc2.weight.detach().cpu().numpy().astype(np.float64),
                "b2": self.fc2.bias.detach().cpu().numpy().astype(np.float64),
                "w3": self.fc3.weight.detach().cpu().numpy().astype(np.float64),
                "b3": self.fc3.bias.detach().cpu().numpy().astype(np.float64),
            }

        def load_numpy_weights(self, weights: Dict[str, np.ndarray]) -> None:
            with torch.no_grad():
                self.fc1.weight.copy_(torch.tensor(weights["w1"], dtype=torch.float64))
                self.fc1.bias.copy_(torch.tensor(weights["b1"], dtype=torch.float64))
                self.fc2.weight.copy_(torch.tensor(weights["w2"], dtype=torch.float64))
                self.fc2.bias.copy_(torch.tensor(weights["b2"], dtype=torch.float64))
                self.fc3.weight.copy_(torch.tensor(weights["w3"], dtype=torch.float64))
                self.fc3.bias.copy_(torch.tensor(weights["b3"], dtype=torch.float64))

else:
    class BoundedKineticMLP:  # type: ignore
        """Fallback stub when PyTorch is not installed in the deployment environment."""
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is not installed in the active environment. "
                "BoundedKineticMLP is only needed for training or autograd parity tests. "
                "Use predict_params_from_npz_member() for runtime NumPy inference."
            )


def integrate_isothermal_rk4_grid(
    temp_c: float,
    max_months: float,
    step_months: float,
    params: Dict[str, Any],
    initial_hmw_fraction: Any,
    initial_lmw_fraction: Any,
    include_lmw: bool = True,
) -> Dict[str, Any]:
    """
    Integrates isothermal degradation over `[0, max_months]` with fixed RK4 step `step_months`
    (`dt_h = step_months * 730.5`). Works identically on batched PyTorch tensors `(B,)` or
    scalar/NumPy arrays, matching `integrate_pathway_rk4` to machine precision (`< 1e-14`)
    while allowing vectorized batch training across all lots simultaneously.
    """
    n_steps = int(round(max_months / step_months))
    dt_h = float(step_months * HOURS_PER_MONTH)
    temp_k = celsius_to_kelvin(float(temp_c))

    pws = ("SEC_HMW", "SEC_LMW") if include_lmw else ("SEC_HMW",)
    pw_rates = evaluate_pathway_rates(temp_k, params, pathways=pws)
    k_hmw = pw_rates["SEC_HMW"]["k_eff_fraction_per_h"]
    k_lmw = pw_rates["SEC_LMW"]["k_eff_fraction_per_h"] if include_lmw else 0.0 * k_hmw
    k_tot = k_hmw + k_lmw

    curr_hmw = initial_hmw_fraction
    curr_lmw = initial_lmw_fraction if include_lmw else 0.0 * initial_hmw_fraction
    curr_m = 1.0 - curr_hmw - curr_lmw

    m_steps = [curr_m]
    hmw_steps = [curr_hmw]
    lmw_steps = [curr_lmw]

    for _ in range(n_steps):
        # 4-stage RK4 at constant temperature T
        dm_1 = -k_tot * curr_m
        dhmw_1 = k_hmw * curr_m
        dlmw_1 = k_lmw * curr_m

        m_k2 = curr_m + 0.5 * dt_h * dm_1
        dm_2 = -k_tot * m_k2
        dhmw_2 = k_hmw * m_k2
        dlmw_2 = k_lmw * m_k2

        m_k3 = curr_m + 0.5 * dt_h * dm_2
        dm_3 = -k_tot * m_k3
        dhmw_3 = k_hmw * m_k3
        dlmw_3 = k_lmw * m_k3

        m_k4 = curr_m + dt_h * dm_3
        dm_4 = -k_tot * m_k4
        dhmw_4 = k_hmw * m_k4
        dlmw_4 = k_lmw * m_k4

        curr_m = curr_m + (dt_h / 6.0) * (dm_1 + 2.0 * dm_2 + 2.0 * dm_3 + dm_4)
        curr_hmw = curr_hmw + (dt_h / 6.0) * (dhmw_1 + 2.0 * dhmw_2 + 2.0 * dhmw_3 + dhmw_4)
        curr_lmw = curr_lmw + (dt_h / 6.0) * (dlmw_1 + 2.0 * dlmw_2 + 2.0 * dlmw_3 + dlmw_4)

        m_steps.append(curr_m)
        hmw_steps.append(curr_hmw)
        lmw_steps.append(curr_lmw)

    if HAS_TORCH and isinstance(curr_m, torch.Tensor):
        m_traj = torch.stack(m_steps, dim=-1)
        hmw_traj = torch.stack(hmw_steps, dim=-1)
        lmw_traj = torch.stack(lmw_steps, dim=-1)
    else:
        m_traj = np.stack(m_steps, axis=-1)
        hmw_traj = np.stack(hmw_steps, axis=-1)
        lmw_traj = np.stack(lmw_steps, axis=-1)

    return {
        "monomer_fraction": m_traj,
        "hmw_fraction": hmw_traj,
        "lmw_fraction": lmw_traj,
        "k_eff_fraction_per_h": k_hmw,
        "f_U": pw_rates["SEC_HMW"]["f_U"],
    }


def compute_lot_apparent_ea(params: Dict[str, Any]) -> Dict[str, Any]:
    """Computes effective rates at 5/25/40 °C and apparent Ea (5-25 °C and 5-40 °C) for predicted params."""
    t5 = celsius_to_kelvin(5.0)
    t25 = celsius_to_kelvin(25.0)
    t40 = celsius_to_kelvin(40.0)
    r5 = lumry_eyring_rates(t5, **params)
    r25 = lumry_eyring_rates(t25, **params)
    r40 = lumry_eyring_rates(t40, **params)
    k5 = r5["k_eff_fraction_per_h"]
    k25 = r25["k_eff_fraction_per_h"]
    k40 = r40["k_eff_fraction_per_h"]
    return {
        "k_eff_5c_per_h": k5,
        "k_eff_25c_per_h": k25,
        "k_eff_40c_per_h": k40,
        "f_U_5c": r5["f_U"],
        "f_U_25c": r25["f_U"],
        "f_U_40c": r40["f_U"],
        "apparent_ea_5_25c_kj_mol": apparent_activation_energy(k5, k25, t5, t25),
        "apparent_ea_5_40c_kj_mol": apparent_activation_energy(k5, k40, t5, t40),
    }
