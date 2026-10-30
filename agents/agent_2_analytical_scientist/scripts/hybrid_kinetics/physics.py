"""
Pure-function deterministic biophysical kinetics module (SI CODATA constants).
Supports both NumPy arrays / Python floats and PyTorch Tensors (preserving autograd).

Unit Conventions (Amendment A):
- Internal state variables (Monomer, Native, Unfolded, HMW, LMW, Acidic): fraction in [0.0, 1.0].
  Percentage (% = 100 * fraction) is used strictly for display.
- Internal rate constants k(T): fraction per hour (h^-1).
- Temperature T: Kelvin (K) internally; helper conversions provided for Celsius.
- Energy (dG‡, dH‡, Ea, dH_u): kJ/mol.
- Entropy (dS‡, dS_u): J/(mol*K).
- Heat capacity (dCp‡, dCp_u): kJ/(mol*K).
- Reference temperature T_ref = 278.15 K (5.0 °C).

Identifiability & Reparametrization Note:
Over the biological stability temperature range (2-45 °C, 275.15-318.15 K), 1/T varies by only ~14%.
In dG‡(T) = dH‡ - T*dS‡, shifts in dH‡ and dS‡ are >99% collinear if parameterized independently
(entropy-enthalpy compensation). Therefore, we parameterize around T_ref = 278.15 K using bounded
(dG‡_ref, dH‡_ref), where dG‡_ref directly pins ln k(T_ref) and dH‡_ref governs the temperature slope,
and convert deterministically to dS‡_ref = 1000 * (dH‡_ref - dG‡_ref) / T_ref for reporting.

Explicit Definition of dG_eff_ref_kj_mol vs dG_agg_ref_kj_mol:
In the Lumry-Eyring aggregation pathway (Native <-> Unfolded -> HMW):
  k_eff(T) = k_agg(T) * f_U(T), where f_U(T) = K_u(T) / (1 + K_u(T)).
- `dG_eff_ref_kj_mol` (~121.07 kJ/mol at 5 °C) is the effective Eyring free-energy barrier
  governing the observable rate k_eff(T_ref) in fraction/hour:
    k_eff(T_ref) = (kB * T_ref / h) * 3600 * exp(-dG_eff_ref * 1000 / (R * T_ref))
- `dG_agg_ref_kj_mol` (~105.09 kJ/mol at 5 °C for f_U(5 °C) = 1e-3) is the intrinsic Eyring
  free-energy barrier of the U -> HMW step k_agg(T_ref), linked by exact thermodynamics:
    dG_agg_ref = dG_eff_ref + 1e-3 * R * T_ref * ln(f_U(T_ref))
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    HAS_TORCH = False

from .config import load_hybrid_config, get_nominal_parameters

# CODATA 2018/2022 SI Constants
KB_SI: float = 1.380649e-23        # Boltzmann constant (J / K)
H_PLANCK_SI: float = 6.62607015e-34  # Planck constant (J * s)
R_GAS_SI: float = 8.314462618      # Universal gas constant (J / (mol * K))
SECONDS_PER_HOUR: float = 3600.0   # s / h
HOURS_PER_MONTH: float = 730.5     # 365.25 * 24 / 12 h / month
T_REF_K: float = 278.15            # 5.0 °C reference temperature in Kelvin


def _is_tensor(x: Any) -> bool:
    return HAS_TORCH and isinstance(x, torch.Tensor)


def _exp(x: Any) -> Any:
    if _is_tensor(x):
        return torch.exp(x)
    return np.exp(x)


def _log(x: Any) -> Any:
    if _is_tensor(x):
        return torch.log(x)
    return np.log(x)


def celsius_to_kelvin(temp_c: Any) -> Any:
    """Converts temperature from degrees Celsius to Kelvin."""
    return temp_c + 273.15


def kelvin_to_celsius(temp_k: Any) -> Any:
    """Converts temperature from Kelvin to degrees Celsius."""
    return temp_k - 273.15


def eyring_prefactor_per_hour(temp_k: Any) -> Any:
    """
    Computes the universal Eyring frequency factor (kB * T / h) * 3600 in h^-1.
    At T_ref = 278.15 K (5 °C), this equals 2.08613135e16 h^-1.
    """
    return (KB_SI / H_PLANCK_SI) * SECONDS_PER_HOUR * temp_k


def dS_from_dG_dH(
    dG_ref_kj_mol: Any,
    dH_ref_kj_mol: Any,
    t_ref_k: float = T_REF_K,
) -> Any:
    """
    Deterministically computes activation entropy dS‡_ref in J/(mol*K) from
    reparametrized (dG‡_ref, dH‡_ref) in kJ/mol at T_ref:
        dS‡_ref = 1000 * (dH‡_ref - dG‡_ref) / T_ref
    """
    return 1000.0 * (dH_ref_kj_mol - dG_ref_kj_mol) / t_ref_k


def dG_from_dH_dS(
    dH_kj_mol: Any,
    dS_j_mol_k: Any,
    temp_k: Any = T_REF_K,
) -> Any:
    """
    Computes activation free energy dG‡(T) in kJ/mol from dH‡(T) [kJ/mol] and dS‡(T) [J/(mol*K)]:
        dG‡(T) = dH‡(T) - 1e-3 * T * dS‡(T)
    """
    return dH_kj_mol - 1e-3 * temp_k * dS_j_mol_k


def eyring_rate_from_dG(temp_k: Any, dG_kj_mol: Any) -> Any:
    """
    Computes Eyring rate constant k(T) in fraction/hour (h^-1) from dG‡(T) in kJ/mol:
        k(T) = (kB * T / h) * 3600 * exp(-dG‡(T) * 1000 / (R * T))
    """
    nu = eyring_prefactor_per_hour(temp_k)
    exponent = -(dG_kj_mol * 1000.0) / (R_GAS_SI * temp_k)
    return nu * _exp(exponent)


def dG_from_eyring_rate(temp_k: Any, k_per_hour: Any) -> Any:
    """
    Inverse of `eyring_rate_from_dG`: computes dG‡(T) in kJ/mol for a rate k(T) in fraction/hour:
        dG‡(T) = -1e-3 * R * T * ln(k(T) / ((kB * T / h) * 3600))
    """
    nu = eyring_prefactor_per_hour(temp_k)
    return -1e-3 * R_GAS_SI * temp_k * _log(k_per_hour / nu)


def eyring_thermodynamics(
    temp_k: Any,
    dH_ref_kj_mol: Any,
    dS_ref_j_mol_k: Any,
    dCp_kj_mol_k: Any = 0.0,
    t_ref_k: float = T_REF_K,
) -> Dict[str, Any]:
    """
    Full first-principles Eyring-Polanyi equation with optional activation heat capacity dCp‡:
        dH‡(T) = dH‡_ref + dCp‡ * (T - T_ref)
        dS‡(T) = dS‡_ref + 1000 * dCp‡ * ln(T / T_ref)
        dG‡(T) = dH‡(T) - 1e-3 * T * dS‡(T)
        k(T)   = (kB * T / h) * 3600 * exp(-dG‡(T) * 1000 / (R * T))   [fraction / hour]
        Ea(T)  = dH‡(T) + 1e-3 * R * T                                 [kJ / mol]

    Note: Unlike the legacy placeholder where dS‡ cancelled out, dS‡_ref directly governs dG‡(T)
    and therefore measurably scales k(T).
    """
    dH_t = dH_ref_kj_mol + dCp_kj_mol_k * (temp_k - t_ref_k)
    dS_t = dS_ref_j_mol_k + 1000.0 * dCp_kj_mol_k * _log(temp_k / t_ref_k)
    dG_t = dG_from_dH_dS(dH_t, dS_t, temp_k)
    k_t = eyring_rate_from_dG(temp_k, dG_t)
    ea_t = eyring_to_arrhenius_ea(dH_t, temp_k)
    return {
        "dH_kj_mol": dH_t,
        "dS_j_mol_k": dS_t,
        "dG_kj_mol": dG_t,
        "k_fraction_per_h": k_t,
        "ea_kj_mol": ea_t,
    }


def eyring_to_arrhenius_ea(dH_kj_mol: Any, temp_k: Any) -> Any:
    """
    Eyring-consistent conversion between activation enthalpy dH‡(T) and Arrhenius activation energy Ea(T):
        Ea(T) = dH‡(T) + 1e-3 * R * T   [kJ/mol]
    """
    return dH_kj_mol + 1e-3 * R_GAS_SI * temp_k


def arrhenius_rate(
    temp_k: Any,
    ea_kj_mol: Any,
    k_ref_per_h: Any | None = None,
    dG_ref_kj_mol: Any | None = None,
    pre_exp_a_per_h: Any | None = None,
    t_ref_k: float = T_REF_K,
) -> Dict[str, Any]:
    """
    Arrhenius rate law for chemical degradation pathways (e.g., SEC_LMW clipping, CEX_ACIDIC deamidation):
        k(T) = A * exp(-Ea * 1000 / (R * T)) = k(T_ref) * exp(-(Ea * 1000 / R) * (1/T - 1/T_ref))
    Also returns Eyring-consistent dH‡(T) = Ea - R*T and dG‡(T_ref).
    """
    if pre_exp_a_per_h is not None:
        k_t = pre_exp_a_per_h * _exp(-(ea_kj_mol * 1000.0) / (R_GAS_SI * temp_k))
        k_ref = pre_exp_a_per_h * _exp(-(ea_kj_mol * 1000.0) / (R_GAS_SI * t_ref_k))
        A_val = pre_exp_a_per_h
    else:
        if k_ref_per_h is None:
            if dG_ref_kj_mol is None:
                raise ValueError("Specify one of pre_exp_a_per_h, k_ref_per_h, or dG_ref_kj_mol.")
            k_ref = eyring_rate_from_dG(t_ref_k, dG_ref_kj_mol)
        else:
            k_ref = k_ref_per_h
        k_t = k_ref * _exp(-(ea_kj_mol * 1000.0 / R_GAS_SI) * ((1.0 / temp_k) - (1.0 / t_ref_k)))
        A_val = k_ref * _exp((ea_kj_mol * 1000.0) / (R_GAS_SI * t_ref_k))

    dH_t = ea_kj_mol - 1e-3 * R_GAS_SI * temp_k
    dG_ref_val = dG_from_eyring_rate(t_ref_k, k_ref) if dG_ref_kj_mol is None else dG_ref_kj_mol
    dH_ref_val = ea_kj_mol - 1e-3 * R_GAS_SI * t_ref_k
    dS_ref_val = dS_from_dG_dH(dG_ref_val, dH_ref_val, t_ref_k=t_ref_k)

    return {
        "k_fraction_per_h": k_t,
        "k_ref_fraction_per_h": k_ref,
        "pre_exp_a_per_h": A_val,
        "ea_kj_mol": ea_kj_mol,
        "dH_kj_mol": dH_t,
        "dG_ref_kj_mol": dG_ref_val,
        "dH_ref_kj_mol": dH_ref_val,
        "dS_ref_j_mol_k": dS_ref_val,
    }


def apparent_activation_energy(
    k_t1: Any,
    k_t2: Any,
    t1_k: float,
    t2_k: float,
) -> Any:
    """
    Computes apparent Arrhenius activation energy Ea_app (kJ/mol) between two temperatures (T1, T2):
        Ea_app = 1e-3 * R * ln(k(T2) / k(T1)) / (1/T1 - 1/T2)
    """
    inv_diff = (1.0 / t1_k) - (1.0 / t2_k)
    return 1e-3 * R_GAS_SI * _log(k_t2 / k_t1) / inv_diff


def vant_hoff_unfolding(
    temp_k: Any,
    ku_ref_5c: Any = 0.0010,
    dH_u_kj_mol: Any = 34.0,
    dCp_u_kj_mol_k: Any = 0.0,
    t_ref_k: float = T_REF_K,
) -> Dict[str, Any]:
    """
    Van 't Hoff reversible conformational equilibrium between Native (N) and
    aggregation-competent partially unfolded state (U):
        N <-> U,   K_u(T) = [U]/[N],   f_U(T) = K_u(T) / (1 + K_u(T))
    """
    dG_u_ref = -1e-3 * R_GAS_SI * t_ref_k * _log(ku_ref_5c)
    dS_u_ref = 1000.0 * (dH_u_kj_mol - dG_u_ref) / t_ref_k

    dH_u_t = dH_u_kj_mol + dCp_u_kj_mol_k * (temp_k - t_ref_k)
    dS_u_t = dS_u_ref + 1000.0 * dCp_u_kj_mol_k * _log(temp_k / t_ref_k)
    dG_u_t = dH_u_t - 1e-3 * temp_k * dS_u_t

    ku_t = _exp(-(dG_u_t * 1000.0) / (R_GAS_SI * temp_k))
    fu_t = ku_t / (1.0 + ku_t)
    fu_ref = ku_ref_5c / (1.0 + ku_ref_5c)

    return {
        "K_u": ku_t,
        "f_U": fu_t,
        "f_U_ref": fu_ref,
        "dG_u_kj_mol": dG_u_t,
        "dH_u_kj_mol": dH_u_t,
        "dS_u_j_mol_k": dS_u_t,
        "dG_u_ref_kj_mol": dG_u_ref,
        "dS_u_ref_j_mol_k": dS_u_ref,
    }


def lumry_eyring_rates(
    temp_k: Any,
    dG_eff_ref_kj_mol: Any | None = 121.066,
    dH_agg_ref_kj_mol: Any = 38.0,
    ku_ref_5c: Any = 0.0010,
    dH_u_kj_mol: Any = 34.0,
    dCp_agg_kj_mol_k: Any = 0.0,
    dCp_u_kj_mol_k: Any = 0.0,
    dG_agg_ref_kj_mol: Any | None = None,
    dS_agg_ref_j_mol_k: Any | None = None,
    t_ref_k: float = T_REF_K,
    **_extra_kwargs: Any,
) -> Dict[str, Any]:
    """
    Computes the Lumry-Eyring aggregation pathway quantities at temperature(s) `temp_k`:
        N <-> U (fast reversible equilibrium K_u(T), unfolded fraction f_U(T) = K_u(T)/(1+K_u(T)))
        U -> HMW (irreversible aggregation step with Eyring rate constant k_agg(T))
        Effective monomer loss rate to HMW: k_eff(T) = k_agg(T) * f_U(T)   [fraction / hour]

    Supports three ways of specifying the reference aggregation barrier at T_ref:
    1. `dG_eff_ref_kj_mol` (default, ~121.07 kJ/mol): directly pins k_eff(T_ref) and derives
       dG_agg_ref = dG_eff_ref + 1e-3 * R * T_ref * ln(f_U(T_ref)).
    2. `dG_agg_ref_kj_mol` (~105.09 kJ/mol): directly specifies the U -> HMW step barrier.
    3. `dS_agg_ref_j_mol_k`: directly specifies the transition-state entropy of U -> HMW.
    """
    uf = vant_hoff_unfolding(
        temp_k=temp_k,
        ku_ref_5c=ku_ref_5c,
        dH_u_kj_mol=dH_u_kj_mol,
        dCp_u_kj_mol_k=dCp_u_kj_mol_k,
        t_ref_k=t_ref_k,
    )
    fu_ref = uf["f_U_ref"]
    rt_ln_fu_ref = 1e-3 * R_GAS_SI * t_ref_k * _log(fu_ref)

    if dS_agg_ref_j_mol_k is not None:
        dS_agg_ref = dS_agg_ref_j_mol_k
        dG_agg_ref = dG_from_dH_dS(dH_agg_ref_kj_mol, dS_agg_ref, t_ref_k)
        dG_eff_ref = dG_agg_ref - rt_ln_fu_ref
    elif dG_agg_ref_kj_mol is not None and dG_eff_ref_kj_mol is None:
        dG_agg_ref = dG_agg_ref_kj_mol
        dG_eff_ref = dG_agg_ref - rt_ln_fu_ref
        dS_agg_ref = dS_from_dG_dH(dG_agg_ref, dH_agg_ref_kj_mol, t_ref_k=t_ref_k)
    else:
        if dG_eff_ref_kj_mol is None:
            raise ValueError("Must provide dG_eff_ref_kj_mol, dG_agg_ref_kj_mol, or dS_agg_ref_j_mol_k.")
        dG_eff_ref = dG_eff_ref_kj_mol
        dG_agg_ref = dG_eff_ref + rt_ln_fu_ref
        dS_agg_ref = dS_from_dG_dH(dG_agg_ref, dH_agg_ref_kj_mol, t_ref_k=t_ref_k)

    agg_eyring = eyring_thermodynamics(
        temp_k=temp_k,
        dH_ref_kj_mol=dH_agg_ref_kj_mol,
        dS_ref_j_mol_k=dS_agg_ref,
        dCp_kj_mol_k=dCp_agg_kj_mol_k,
        t_ref_k=t_ref_k,
    )

    k_agg_t = agg_eyring["k_fraction_per_h"]
    fu_t = uf["f_U"]
    k_eff_t = k_agg_t * fu_t

    # Effective thermodynamic quantities for k_eff = k_agg * f_U
    dG_eff_t = dG_from_eyring_rate(temp_k, k_eff_t)
    # Effective enthalpy dH_eff(T) = dH_agg(T) + (1 - f_U(T)) * dH_u(T)
    dH_eff_t = agg_eyring["dH_kj_mol"] + (1.0 - fu_t) * uf["dH_u_kj_mol"]
    dS_eff_t = 1000.0 * (dH_eff_t - dG_eff_t) / temp_k
    ea_eff_t = eyring_to_arrhenius_ea(dH_eff_t, temp_k)

    return {
        "k_eff_fraction_per_h": k_eff_t,
        "k_agg_fraction_per_h": k_agg_t,
        "f_U": fu_t,
        "K_u": uf["K_u"],
        "dG_eff_ref_kj_mol": dG_eff_ref,
        "dG_agg_ref_kj_mol": dG_agg_ref,
        "dH_agg_ref_kj_mol": dH_agg_ref_kj_mol,
        "dS_agg_ref_j_mol_k": dS_agg_ref,
        "dG_eff_kj_mol": dG_eff_t,
        "dH_eff_kj_mol": dH_eff_t,
        "dS_eff_j_mol_k": dS_eff_t,
        "ea_eff_kj_mol": ea_eff_t,
        "dH_u_kj_mol": uf["dH_u_kj_mol"],
        "dS_u_j_mol_k": uf["dS_u_j_mol_k"],
        "dG_u_kj_mol": uf["dG_u_kj_mol"],
    }


# ---------------------------------------------------------------------------
# EXTENSIBLE PATHWAY REGISTRY (Physics Requirement #5)
# ---------------------------------------------------------------------------
PathwayRateFn = Callable[[Any, Dict[str, Any]], Dict[str, Any]]
PATHWAY_REGISTRY: Dict[str, PathwayRateFn] = {}


def register_pathway(name: str, fn: PathwayRateFn) -> None:
    """Registers a degradation pathway rate calculator without touching the ODE solver."""
    PATHWAY_REGISTRY[name] = fn


def _pathway_sec_hmw(temp_k: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    has_dg_eff = "dG_eff_ref_kj_mol" in params and params["dG_eff_ref_kj_mol"] is not None
    return lumry_eyring_rates(
        temp_k=temp_k,
        dG_eff_ref_kj_mol=params.get("dG_eff_ref_kj_mol", 121.066),
        dH_agg_ref_kj_mol=params.get("dH_agg_ref_kj_mol", 38.0),
        ku_ref_5c=params.get("ku_ref_5c", 0.0010),
        dH_u_kj_mol=params.get("dH_u_kj_mol", 34.0),
        dCp_agg_kj_mol_k=params.get("dCp_agg_kj_mol_k", 0.0),
        dCp_u_kj_mol_k=params.get("dCp_u_kj_mol_k", 0.0),
        dG_agg_ref_kj_mol=None if has_dg_eff else params.get("dG_agg_ref_kj_mol", None),
        dS_agg_ref_j_mol_k=None if has_dg_eff else params.get("dS_agg_ref_j_mol_k", None),
    )


def _pathway_sec_lmw(temp_k: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    dG_frag_ref = params.get("dG_frag_ref_kj_mol", 125.8)
    ea_frag = params.get("ea_frag_kj_mol", 78.0)
    res = arrhenius_rate(temp_k=temp_k, ea_kj_mol=ea_frag, dG_ref_kj_mol=dG_frag_ref)
    res["k_eff_fraction_per_h"] = res["k_fraction_per_h"]
    return res


def _pathway_cex_acidic(temp_k: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    dG_acidic_ref = params.get("dG_acidic_ref_kj_mol", 118.4)
    ea_acidic = params.get("ea_acidic_kj_mol", 72.0)
    res = arrhenius_rate(temp_k=temp_k, ea_kj_mol=ea_acidic, dG_ref_kj_mol=dG_acidic_ref)
    res["k_eff_fraction_per_h"] = res["k_fraction_per_h"]
    return res


register_pathway("SEC_HMW", _pathway_sec_hmw)
register_pathway("SEC_LMW", _pathway_sec_lmw)
register_pathway("CEX_ACIDIC", _pathway_cex_acidic)


def evaluate_pathway_rates(
    temp_k: Any,
    params: Dict[str, Any],
    pathways: tuple[str, ...] = ("SEC_HMW", "SEC_LMW", "CEX_ACIDIC"),
) -> Dict[str, Dict[str, Any]]:
    """Evaluates registered degradation pathways at `temp_k`."""
    out: Dict[str, Dict[str, Any]] = {}
    for pw in pathways:
        if pw not in PATHWAY_REGISTRY:
            raise KeyError(f"Unknown pathway '{pw}'. Registered: {list(PATHWAY_REGISTRY.keys())}")
        out[pw] = PATHWAY_REGISTRY[pw](temp_k, params)
    return out


def compute_excursion_detection_gate_pct(
    sec_repeatability_sd_pct: float = 0.035,
    z_multiplier: float = 2.0,
) -> float:
    """
    Computes the minimum detectable difference gate (in % HMW points) for a pre/post
    excursion comparison of two independent measurements:
        SD_diff = sqrt(2) * SD_assay
        gate    = z_multiplier * sqrt(2) * SD_assay
    For SD_assay = 0.035% and z_multiplier = 2.0, gate = 2 * sqrt(2) * 0.035% = 0.098995% (~0.099%).
    """
    return float(z_multiplier * math.sqrt(2.0) * sec_repeatability_sd_pct)


def classify_excursion_verdict(
    delta_hmw_lower95_pct: float,
    delta_hmw_upper95_pct: float,
    sec_repeatability_sd_pct: float = 0.035,
    z_multiplier: float = 2.0,
) -> Dict[str, Any]:
    """
    Classifies excursion impact against configured SEC repeatability using the 95% interval
    of excursion-attributable dHMW (% points):
    - "no detectable change": upper 95% bound < sqrt(2) * z * SD_assay
    - "detectable change likely": lower 95% bound > sqrt(2) * z * SD_assay
    - "inconclusive": 95% interval straddles the detection gate

    Never outputs a release or disposition statement. Explicitly notes that SEC alone cannot
    separate reversible conformational stress from sub-detectable irreversible nucleation,
    and that f_U(t) is a prior-driven latent quantity.
    """
    gate_pct = compute_excursion_detection_gate_pct(sec_repeatability_sd_pct, z_multiplier)
    if delta_hmw_upper95_pct < gate_pct:
        verdict = "no detectable change"
        detail = (
            f"No detectable excursion-attributable change versus assay variability "
            f"(95% upper bound +{delta_hmw_upper95_pct:.4f} pp < 2*sqrt(2)*SD gate +{gate_pct:.3f} pp). "
            "Note: SEC alone cannot separate reversible conformational stress from sub-detectable nucleation, "
            "and f_U(t) is a prior-driven latent quantity (relaxes in <1 min at default k_-1). "
            "Supports technical evaluation only; requires human QA review."
        )
    elif delta_hmw_lower95_pct > gate_pct:
        verdict = "detectable change likely"
        detail = (
            f"Detectable excursion-attributable HMW increase likely "
            f"(95% lower bound +{delta_hmw_lower95_pct:.4f} pp > 2*sqrt(2)*SD gate +{gate_pct:.3f} pp). "
            "Note: SEC alone cannot separate reversible conformational stress from irreversible aggregation mechanisms "
            "without orthogonal characterization (e.g., SEC-MALS, DLS, sub-visible particles), and f_U(t) is a "
            "prior-driven latent quantity. Does not support automatic clearance; requires human QA review."
        )
    else:
        verdict = "inconclusive"
        detail = (
            f"Inconclusive versus assay variability: excursion-attributable dHMW 95% interval "
            f"[+{delta_hmw_lower95_pct:.4f} pp, +{delta_hmw_upper95_pct:.4f} pp] straddles the "
            f"2*sqrt(2)*SD detection gate (+{gate_pct:.3f} pp). "
            "Note: SEC alone cannot separate reversible stress from irreversible change, and f_U(t) is a "
            "prior-driven latent quantity. Additional replicate/orthogonal testing and human QA review required."
        )
    return {
        "verdict": verdict,
        "detection_gate_pct": gate_pct,
        "sec_repeatability_sd_pct": sec_repeatability_sd_pct,
        "delta_hmw_lower95_pct": delta_hmw_lower95_pct,
        "delta_hmw_upper95_pct": delta_hmw_upper95_pct,
        "assessment_statement": detail,
    }


def compute_amendment_ab_summary(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Computes the exact numerical diagnostic summary required for Amendments A and B:
    - Universal Eyring frequency factor nu_Eyring at 278.15 K (in 1/h)
    - Effective and intrinsic dG‡ at 5 °C (in kJ/mol) for k in fraction/hour
    - k_eff at 5 °C, 25 °C, 28 °C, 38 °C, 40 °C (in fraction/hour and %/month)
    - Ratios k_eff(25 °C)/k_eff(5 °C), k_eff(28 °C)/k_eff(5 °C), k_eff(40 °C)/k_eff(5 °C)
    - Apparent Ea (5-25 °C) and Apparent Ea (5-40 °C) in kJ/mol
    - Excursion outcomes for nominal model:
      * 36h @ 28 °C
      * 2h @ 28 °C
      * 48h @ 38 °C
      * 14d (336h) @ 40 °C
      reporting delta_HMW (using M0 = 1 - HMW0 - LMW0 = 0.9938, in fraction and pp), peak f_U (%), and 5 °C-equivalent days consumed.
    """
    cfg = load_hybrid_config()
    if params is None:
        params = get_nominal_parameters(cfg)

    sd_sec = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    gate_2sigma_diff = compute_excursion_detection_gate_pct(sd_sec, 2.0)

    t5 = celsius_to_kelvin(5.0)
    t25 = celsius_to_kelvin(25.0)
    t28 = celsius_to_kelvin(28.0)
    t38 = celsius_to_kelvin(38.0)
    t40 = celsius_to_kelvin(40.0)

    r5 = lumry_eyring_rates(t5, **params)
    r25 = lumry_eyring_rates(t25, **params)
    r28 = lumry_eyring_rates(t28, **params)
    r38 = lumry_eyring_rates(t38, **params)
    r40 = lumry_eyring_rates(t40, **params)

    k5 = float(r5["k_eff_fraction_per_h"])
    k25 = float(r25["k_eff_fraction_per_h"])
    k28 = float(r28["k_eff_fraction_per_h"])
    k38 = float(r38["k_eff_fraction_per_h"])
    k40 = float(r40["k_eff_fraction_per_h"])

    ea_5_25 = float(apparent_activation_energy(k5, k25, t5, t25))
    ea_5_40 = float(apparent_activation_energy(k5, k40, t5, t40))

    hmw0 = 0.0062
    lmw0 = 0.0
    m0 = 1.0 - hmw0 - lmw0  # Single starting-monomer convention: M0 = 1 - HMW0 - LMW0 = 0.9938
    scenarios = {
        "36h_at_28C": (36.0, k28, float(r28["f_U"])),
        "2h_at_28C": (2.0, k28, float(r28["f_U"])),
        "48h_at_38C": (48.0, k38, float(r38["f_U"])),
        "72h_at_40C": (72.0, k40, float(r40["f_U"])),
        "7d_at_40C": (7.0 * 24.0, k40, float(r40["f_U"])),
        "14d_at_40C": (14.0 * 24.0, k40, float(r40["f_U"])),
    }
    excursion_outcomes: Dict[str, Dict[str, Any]] = {}
    for name, (dur_h, k_spike, fu_spike) in scenarios.items():
        d_hmw_frac = m0 * (1.0 - math.exp(-k_spike * dur_h))
        d_hmw_pp = d_hmw_frac * 100.0
        # 5C-equivalent days: time at 5C producing the same monomer decay exp(-k5 * t_eq) = exp(-k_spike * dur_h)
        eq_5c_days = (k_spike * dur_h) / (k5 * 24.0)
        # PLACEHOLDER PRE-ENSEMBLE INTERVAL: ±15% around deterministic nominal point estimate.
        # Real L95/U95 excursion verdicts are produced only after Phase 4 (25-member deep ensemble + conformal calibration).
        verdict_info = classify_excursion_verdict(
            delta_hmw_lower95_pct=d_hmw_pp * 0.85,
            delta_hmw_upper95_pct=d_hmw_pp * 1.15,
            sec_repeatability_sd_pct=sd_sec,
        )
        excursion_outcomes[name] = {
            "duration_hours": dur_h,
            "initial_monomer_fraction": m0,
            "delta_hmw_fraction": d_hmw_frac,
            "delta_hmw_pp": d_hmw_pp,
            "peak_f_U_pct": fu_spike * 100.0,
            "equivalent_5c_days": eq_5c_days,
            "placeholder_l95_pp": d_hmw_pp * 0.85,
            "placeholder_u95_pp": d_hmw_pp * 1.15,
            "verdict_interval_source": "PLACEHOLDER_PRE_ENSEMBLE_PLUS_MINUS_15PCT (real ensemble/conformal verdicts after Phase 4)",
            "verdict": verdict_info["verdict"],
        }

    return {
        "eyring_prefactor_5c_per_h": float(eyring_prefactor_per_hour(t5)),
        "dG_eff_ref_5c_kj_mol": float(r5["dG_eff_ref_kj_mol"]),
        "dG_agg_ref_5c_kj_mol": float(r5["dG_agg_ref_kj_mol"]),
        "dH_agg_ref_5c_kj_mol": float(r5["dH_agg_ref_kj_mol"]),
        "dS_agg_ref_5c_j_mol_k": float(r5["dS_agg_ref_j_mol_k"]),
        "dS_agg_ref_5c_sme_note": "ASSUMPTION - SME to confirm (-241.19 J/mol/K noted for SME review in LIMITATIONS.md; no hard gate)",
        "dH_u_kj_mol": float(r5["dH_u_kj_mol"]),
        "ku_5c": float(r5["K_u"]),
        "fu_5c_pct": float(r5["f_U"]) * 100.0,
        "fu_25c_pct": float(r25["f_U"]) * 100.0,
        "fu_28c_pct": float(r28["f_U"]) * 100.0,
        "fu_38c_pct": float(r38["f_U"]) * 100.0,
        "fu_40c_pct": float(r40["f_U"]) * 100.0,
        "k_eff_5c_fraction_per_h": k5,
        "k_eff_5c_pct_per_month": k5 * HOURS_PER_MONTH * 100.0,
        "k_eff_25c_fraction_per_h": k25,
        "k_eff_25c_pct_per_month": k25 * HOURS_PER_MONTH * 100.0,
        "k_eff_28c_fraction_per_h": k28,
        "k_eff_38c_fraction_per_h": k38,
        "k_eff_40c_fraction_per_h": k40,
        "ratio_k25_over_k5": k25 / k5,
        "ratio_k28_over_k5": k28 / k5,
        "ratio_k40_over_k5": k40 / k5,
        "apparent_ea_5_25c_kj_mol": ea_5_25,
        "apparent_ea_5_40c_kj_mol": ea_5_40,
        "sec_hmw_repeatability_sd_pct": sd_sec,
        "sec_hmw_diff_2sigma_gate_pct": gate_2sigma_diff,
        "excursion_outcomes": excursion_outcomes,
    }
