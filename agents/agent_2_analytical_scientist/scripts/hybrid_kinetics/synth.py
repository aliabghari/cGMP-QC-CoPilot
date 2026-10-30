"""
Phase 2: Synthetic Stability & Thermal Excursion Generator (`synth.py`).

Generates 240 synthetic biologics lots (strictly split by lot: 100 train / 40 validation / 100 test)
with realistic ICH Q1A/Q5C stability pull schedules:
- 5 °C (long-term): 0, 1, 3, 6, 9, 12, 18, 24 months
- 25 °C (accelerated): 0, 1, 3, 6 months (plus 9, 12 months on 40% of lots per split)
- 40 °C (stress): 0, 0.5, 1, 2, 3 months
plus 120 thermal excursion logger profiles (validation/test lots only) and the illustrative seeded
demo lot `NS-2026-004` (4 timepoints: 0, 1, 3, 6M at 5 °C and 25 °C).

Key anti-inverse-crime design:
1. Ground-truth trajectories are integrated using the full 3-state stiff `Radau` ODE solver at finite
   reverse conformational relaxation rate `k_-1`, NEVER the rapid pre-equilibrium RK4 solver used
   by the fitted hybrid model.
2. Each split (train/val/test) has:
   - 70% `matched` lots (`structural_group="matched"`, `structural_subgroup="matched"`): finite `k_-1` in
     [75, 150] h^-1, `dCp_agg = 0`, no PS80 depletion.
   - 15% `mismatched_ps80_only` ablation lots (`structural_group="mismatched"`,
     `structural_subgroup="mismatched_ps80_only"`): ONLY late-time PS80 depletion (`ps80_alpha > 0`),
     with `dCp_agg = 0` and fast `k_-1` in [75, 150] h^-1 so mismatch errors can be attributed cleanly.
   - 15% `mismatched_full` lots (`structural_group="mismatched"`, `structural_subgroup="mismatched_full"`):
     late-time PS80 depletion (`ps80_alpha > 0`) + non-zero `dCp_agg` in [0.05, 0.12] kJ/(mol*K) +
     slower `k_-1` in [18, 55] h^-1.
3. Neither `structural_group`, `structural_subgroup`, nor any latent parameter appears in
   `synth_observations.csv` or `synth_excursions.csv`. They exist ONLY in
   `_ground_truth_do_not_read_in_train.json`, and `load_observations_for_training()` blocks them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .config import get_config_sha256, load_hybrid_config
from .ode import build_excursion_profile_knots
from .physics import (
    HOURS_PER_MONTH,
    R_GAS_SI,
    apparent_activation_energy,
    celsius_to_kelvin,
    compute_excursion_detection_gate_pct,
    evaluate_pathway_rates,
    lumry_eyring_rates,
)

SYNTH_STAMP = "SYNTHETIC - NOT GMP DATA"
GENERATOR_VERSION = "2.1.0-radau-3state-ps80-realistic-pulls-non-gmp"

DEFAULT_SYNTH_DIR = (
    Path(__file__).resolve().parents[2] / "assets" / "synthetic_stability"
)

# Feature columns allowed to be read by the ANN training pipeline
LOT_COVARIATE_COLUMNS: Tuple[str, ...] = (
    "ph",
    "ionic_strength_mm",
    "surfactant_pct",
    "protein_conc_mg_ml",
    "headspace_index",
    "container_syringe_flag",
    "initial_afuc_glycan_pct",
    "initial_acidic_pct",
    "initial_hmw_pct",
    "initial_lmw_pct",
)

# Forbidden ground-truth column substrings that must NEVER appear in training CSVs
FORBIDDEN_GROUND_TRUTH_COLUMNS: Tuple[str, ...] = (
    "structural_group",
    "structural_subgroup",
    "dG_eff_ref",
    "dG_agg_ref",
    "dH_agg_ref",
    "dS_agg_ref",
    "ku_ref_5c",
    "dH_u_kj_mol",
    "dCp_agg",
    "k_eff_5c_true",
    "k_eff_25c_true",
    "k_eff_40c_true",
    "ea_app_5_40_true",
    "ps80_alpha_true",
    "hmw_true",
    "monomer_true",
    "f_U_true",
)


def compute_file_sha256(path: Path) -> str:
    """Computes hex SHA-256 digest of a file on disk."""
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def _sample_lot_covariates_and_truth(
    rng: np.random.Generator,
    lot_index: int,
    split: str,
    subgroup: str,
    extended_25c_pulls: bool,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Samples physical formulation/container/initial-state covariates and computes
    ground-truth latent kinetic parameters with lot random effects.
    """
    bounds = cfg["parameter_bounds"]

    # 1. Sample observable lot covariates
    ph = float(rng.uniform(5.5, 6.8))
    ionic_strength_mm = float(rng.uniform(50.0, 220.0))
    surfactant_pct = float(rng.uniform(0.005, 0.050))
    protein_conc_mg_ml = float(rng.uniform(20.0, 150.0))
    container_syringe_flag = int(rng.random() < 0.40)
    headspace_index = float(
        rng.uniform(0.45, 0.95) if container_syringe_flag == 1 else rng.uniform(0.10, 0.65)
    )
    initial_afuc_glycan_pct = float(rng.uniform(2.0, 12.0))
    initial_acidic_pct = float(rng.uniform(11.5, 18.5))
    initial_hmw_pct = float(rng.uniform(0.35, 0.85))
    initial_lmw_pct = float(rng.uniform(0.05, 0.22))

    # Normalized covariate deviations around nominal formulation
    d_ph = ph - 6.0  # optimum pH = 6.0
    z_ionic = (ionic_strength_mm - 120.0) / 50.0
    z_surf = (surfactant_pct - 0.020) / 0.015
    z_conc = (protein_conc_mg_ml - 50.0) / 40.0
    z_head = (headspace_index - 0.40) / 0.30
    z_glyc = (initial_afuc_glycan_pct - 6.0) / 3.0
    z_acid = (initial_acidic_pct - 14.5) / 2.5

    # 2. Latent mechanistic parameters = smooth physical response + lot random effects
    # Centered around nominal 121.07 kJ/mol (~0.028%/month at 5 °C)
    eps_dg = float(rng.normal(0.0, 0.14))
    dG_eff_ref = (
        121.35
        - 1.50 * ((d_ph ** 2) - 0.16)
        - 0.28 * z_ionic
        + 0.28 * z_surf
        - 0.32 * z_conc
        - 0.18 * z_head
        - 0.12 * (container_syringe_flag - 0.40)
        + eps_dg
    )
    dG_eff_ref = float(
        np.clip(
            dG_eff_ref,
            bounds["dG_eff_ref_kj_mol"]["min"] + 0.5,
            bounds["dG_eff_ref_kj_mol"]["max"] - 0.5,
        )
    )

    eps_dh_agg = float(rng.normal(0.0, 0.9))
    dH_agg_ref = (
        38.0
        + 2.6 * ((d_ph ** 2) - 0.16)
        + 1.4 * z_ionic
        - 1.2 * z_surf
        + 1.0 * z_conc
        + eps_dh_agg
    )
    dH_agg_ref = float(
        np.clip(
            dH_agg_ref,
            bounds["dH_agg_ref_kj_mol"]["min"] + 2.0,
            bounds["dH_agg_ref_kj_mol"]["max"] - 2.0,
        )
    )

    eps_log_ku = float(rng.normal(0.0, 0.05))
    log_ku_5c = (
        np.log(0.0010)
        + 0.32 * ((d_ph ** 2) - 0.16)
        + 0.12 * z_head
        + 0.10 * (container_syringe_flag - 0.40)
        + 0.08 * z_glyc
        - 0.12 * z_surf
        + eps_log_ku
    )
    ku_ref_5c = float(
        np.clip(
            np.exp(log_ku_5c),
            bounds["ku_ref_5c"]["min"] * 1.1,
            bounds["ku_ref_5c"]["max"] * 0.9,
        )
    )

    eps_dh_u = float(rng.normal(0.0, 0.8))
    dH_u = (
        34.0
        + 2.0 * ((d_ph ** 2) - 0.16)
        + 1.1 * z_glyc
        + 0.8 * z_acid
        - 1.0 * z_surf
        + eps_dh_u
    )
    dH_u = float(
        np.clip(
            dH_u,
            bounds["dH_u_kj_mol"]["min"] + 2.0,
            bounds["dH_u_kj_mol"]["max"] - 2.0,
        )
    )

    # Secondary pathways (LMW fragmentation & Acidic deamidation)
    dG_frag_ref = float(
        np.clip(
            125.8 - 1.2 * (d_ph ** 2) - 0.25 * z_acid + rng.normal(0.0, 0.15),
            bounds["dG_frag_ref_kj_mol"]["min"] + 0.3,
            bounds["dG_frag_ref_kj_mol"]["max"] - 0.3,
        )
    )
    ea_frag = float(
        np.clip(
            78.0 + 2.0 * d_ph + rng.normal(0.0, 1.0),
            bounds["ea_frag_kj_mol"]["min"] + 1.0,
            bounds["ea_frag_kj_mol"]["max"] - 1.0,
        )
    )
    dG_acidic_ref = float(
        np.clip(
            118.4 - 0.9 * max(0.0, d_ph) - 0.20 * z_ionic + rng.normal(0.0, 0.15),
            bounds["dG_acidic_ref_kj_mol"]["min"] + 0.3,
            bounds["dG_acidic_ref_kj_mol"]["max"] - 0.3,
        )
    )
    ea_acidic = float(
        np.clip(
            72.0 + 1.8 * d_ph + rng.normal(0.0, 1.0),
            bounds["ea_acidic_kj_mol"]["min"] + 1.0,
            bounds["ea_acidic_kj_mol"]["max"] - 1.0,
        )
    )

    # 3. Structural match vs mismatch parameters (anti-inverse-crime + ablation)
    if subgroup == "matched":
        structural_group = "matched"
        k_unfold_relax = float(rng.uniform(75.0, 150.0))
        dCp_agg = 0.0
        ps80_alpha = 0.0
        ps80_deg_rate_25c_per_h = 0.0
    elif subgroup == "mismatched_ps80_only":
        structural_group = "mismatched"
        k_unfold_relax = float(rng.uniform(75.0, 150.0))
        dCp_agg = 0.0
        ps80_alpha = float(
            rng.uniform(1.10, 1.85) * (1.0 + 0.30 * max(0.0, -z_surf) + 0.20 * max(0.0, z_head))
        )
        ps80_deg_rate_25c_per_h = float(
            rng.uniform(1.8e-4, 2.8e-4) * (1.0 + 0.25 * max(0.0, z_head))
        )
    elif subgroup == "mismatched_full":
        structural_group = "mismatched"
        k_unfold_relax = float(rng.uniform(18.0, 55.0))
        dCp_agg = float(rng.uniform(0.05, 0.12))
        ps80_alpha = float(
            rng.uniform(1.10, 1.85) * (1.0 + 0.30 * max(0.0, -z_surf) + 0.20 * max(0.0, z_head))
        )
        ps80_deg_rate_25c_per_h = float(
            rng.uniform(1.8e-4, 2.8e-4) * (1.0 + 0.25 * max(0.0, z_head))
        )
    else:
        raise ValueError(f"Unknown subgroup: {subgroup}")

    params_eval = {
        "dG_eff_ref_kj_mol": dG_eff_ref,
        "dH_agg_ref_kj_mol": dH_agg_ref,
        "dCp_agg_kj_mol_k": dCp_agg,
        "ku_ref_5c": ku_ref_5c,
        "dH_u_kj_mol": dH_u,
        "dCp_u_kj_mol_k": 0.0,
        "dG_frag_ref_kj_mol": dG_frag_ref,
        "ea_frag_kj_mol": ea_frag,
        "dG_acidic_ref_kj_mol": dG_acidic_ref,
        "ea_acidic_kj_mol": ea_acidic,
        "k_unfold_relax_per_hour": k_unfold_relax,
    }

    t5 = celsius_to_kelvin(5.0)
    t25 = celsius_to_kelvin(25.0)
    t40 = celsius_to_kelvin(40.0)
    r5 = lumry_eyring_rates(t5, **params_eval)
    r25 = lumry_eyring_rates(t25, **params_eval)
    r40 = lumry_eyring_rates(t40, **params_eval)

    k5_true = float(r5["k_eff_fraction_per_h"])
    k25_true = float(r25["k_eff_fraction_per_h"])
    k40_true = float(r40["k_eff_fraction_per_h"])
    ea_5_25_true = float(apparent_activation_energy(k5_true, k25_true, t5, t25))
    ea_5_40_true = float(apparent_activation_energy(k5_true, k40_true, t5, t40))

    lot_id = f"SYN-{split.upper()[:3]}-{lot_index:03d}"

    # Observable covariates ONLY (no structural_group or structural_subgroup)
    covariates = {
        "lot_id": lot_id,
        "split": split,
        "ph": round(ph, 4),
        "ionic_strength_mm": round(ionic_strength_mm, 3),
        "surfactant_pct": round(surfactant_pct, 5),
        "protein_conc_mg_ml": round(protein_conc_mg_ml, 3),
        "headspace_index": round(headspace_index, 4),
        "container_syringe_flag": int(container_syringe_flag),
        "container_closure": "prefilled_syringe" if container_syringe_flag == 1 else "glass_vial",
        "initial_afuc_glycan_pct": round(initial_afuc_glycan_pct, 4),
        "initial_acidic_pct": round(initial_acidic_pct, 4),
        "initial_hmw_pct": round(initial_hmw_pct, 4),
        "initial_lmw_pct": round(initial_lmw_pct, 4),
        "initial_monomer_pct": round(100.0 - initial_hmw_pct - initial_lmw_pct, 4),
    }

    ground_truth = {
        **covariates,
        "structural_group": structural_group,
        "structural_subgroup": subgroup,
        "extended_25c_pulls": bool(extended_25c_pulls),
        **params_eval,
        "dG_agg_ref_kj_mol": float(r5["dG_agg_ref_kj_mol"]),
        "dS_agg_ref_j_mol_k": float(r5["dS_agg_ref_j_mol_k"]),
        "k_eff_5c_true_per_h": k5_true,
        "k_eff_25c_true_per_h": k25_true,
        "k_eff_40c_true_per_h": k40_true,
        "f_U_5c_true": float(r5["f_U"]),
        "f_U_25c_true": float(r25["f_U"]),
        "f_U_40c_true": float(r40["f_U"]),
        "apparent_ea_5_25c_true_kj_mol": ea_5_25_true,
        "apparent_ea_5_40c_true_kj_mol": ea_5_40_true,
        "ps80_alpha_true": ps80_alpha,
        "ps80_deg_rate_25c_per_h": ps80_deg_rate_25c_per_h,
    }
    return covariates, ground_truth


def integrate_ground_truth_3state_radau(
    time_grid_h: np.ndarray | Sequence[float],
    time_knots_h: Sequence[float] | np.ndarray,
    temp_knots_c: Sequence[float] | np.ndarray,
    lot_truth: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Ground-truth 3-state stiff Radau ODE integrator at finite `k_-1` with optional late-time
    polysorbate-80 (PS80) depletion acceleration (`ps80_alpha_true > 0` for structurally
    mismatched lots).

    State vector y = [N, U, HMW, LMW, Acidic, S_ps80]:
        dS_ps80/dt = -k_ps80(T) * S_ps80
        boost(t)   = 1 + alpha_ps80 * max(0, 0.78 - S_ps80(t))^2 / (0.78^2)
        k_agg_eff  = k_agg(T) * boost(t)
        dN/dt      = -k_1(T)*N + k_-1*U - k_frag(T)*N
        dU/dt      =  k_1(T)*N - k_-1*U - (k_agg_eff + k_frag(T))*U
        dHMW/dt    =  k_agg_eff * U
        dLMW/dt    =  k_frag(T) * (N + U)
        dAcidic/dt =  k_acidic(T) * (1 - Acidic)
    """
    t_arr = np.asarray(time_grid_h, dtype=np.float64)
    tk = np.asarray(time_knots_h, dtype=np.float64)
    tc = np.asarray(temp_knots_c, dtype=np.float64)

    k_rev = float(lot_truth["k_unfold_relax_per_hour"])
    ps80_alpha = float(lot_truth.get("ps80_alpha_true", 0.0))
    k_ps80_25c = float(lot_truth.get("ps80_deg_rate_25c_per_h", 0.0))
    ea_ps80_j_mol = 68000.0  # Arrhenius activation energy for PS80 hydrolysis/oxidation
    t25_k = 298.15

    hmw0 = float(lot_truth["initial_hmw_pct"]) / 100.0
    lmw0 = float(lot_truth["initial_lmw_pct"]) / 100.0
    ac0 = float(lot_truth["initial_acidic_pct"]) / 100.0
    m0 = 1.0 - hmw0 - lmw0

    t0_c = float(np.interp(t_arr[0], tk, tc))
    r0 = lumry_eyring_rates(celsius_to_kelvin(t0_c), **lot_truth)
    fu0 = float(r0["f_U"])
    n0 = (1.0 - fu0) * m0
    u0 = fu0 * m0

    y0 = np.array([n0, u0, hmw0, lmw0, ac0, 1.0], dtype=np.float64)

    is_isothermal = bool(np.allclose(tc, tc[0], atol=1e-9))
    if is_isothermal:
        const_k = celsius_to_kelvin(float(tc[0]))
        r_const = lumry_eyring_rates(const_k, **lot_truth)
        ku_const = float(r_const["K_u"])
        k_agg_const = float(r_const["k_agg_fraction_per_h"])
        pw_const = evaluate_pathway_rates(const_k, lot_truth, pathways=("SEC_LMW", "CEX_ACIDIC"))
        k_frag_const = float(pw_const["SEC_LMW"]["k_eff_fraction_per_h"])
        k_ac_const = float(pw_const["CEX_ACIDIC"]["k_eff_fraction_per_h"])
        k_ps80_const = (
            k_ps80_25c * np.exp(-(ea_ps80_j_mol / R_GAS_SI) * (1.0 / const_k - 1.0 / t25_k))
            if k_ps80_25c > 0.0
            else 0.0
        )

        def rhs_iso(_t: float, y: np.ndarray) -> np.ndarray:
            n_v, u_v, _, _, ac_v, s_v = y
            if ps80_alpha > 0.0 and s_v < 0.78:
                depl = (0.78 - max(0.0, s_v)) / 0.78
                boost = 1.0 + ps80_alpha * (depl * depl)
            else:
                boost = 1.0
            k_agg_t = k_agg_const * boost
            k_fwd = ku_const * k_rev
            dn = -k_fwd * n_v + k_rev * u_v - k_frag_const * n_v
            du = k_fwd * n_v - k_rev * u_v - (k_agg_t + k_frag_const) * u_v
            dhmw = k_agg_t * u_v
            dlmw = k_frag_const * (n_v + u_v)
            dac = k_ac_const * (1.0 - ac_v)
            ds = -k_ps80_const * s_v
            return np.array([dn, du, dhmw, dlmw, dac, ds], dtype=np.float64)

        sol = solve_ivp(
            rhs_iso,
            t_span=(float(t_arr[0]), float(t_arr[-1])),
            y0=y0,
            t_eval=t_arr,
            method="Radau",
            rtol=1e-8,
            atol=1e-10,
        )
    else:
        def rhs_var(t_val: float, y: np.ndarray) -> np.ndarray:
            n_v, u_v, _, _, ac_v, s_v = y
            curr_c = float(np.interp(t_val, tk, tc))
            curr_k = celsius_to_kelvin(curr_c)
            pw_out = evaluate_pathway_rates(
                curr_k, lot_truth, pathways=("SEC_HMW", "SEC_LMW", "CEX_ACIDIC")
            )
            ku_val = float(pw_out["SEC_HMW"]["K_u"])
            k_agg_val = float(pw_out["SEC_HMW"]["k_agg_fraction_per_h"])
            k_frag_val = float(pw_out["SEC_LMW"]["k_eff_fraction_per_h"])
            k_ac_val = float(pw_out["CEX_ACIDIC"]["k_eff_fraction_per_h"])
            k_ps80_val = (
                k_ps80_25c * np.exp(-(ea_ps80_j_mol / R_GAS_SI) * (1.0 / curr_k - 1.0 / t25_k))
                if k_ps80_25c > 0.0
                else 0.0
            )
            if ps80_alpha > 0.0 and s_v < 0.78:
                depl = (0.78 - max(0.0, s_v)) / 0.78
                boost = 1.0 + ps80_alpha * (depl * depl)
            else:
                boost = 1.0
            k_agg_t = k_agg_val * boost
            k_fwd = ku_val * k_rev
            dn = -k_fwd * n_v + k_rev * u_v - k_frag_val * n_v
            du = k_fwd * n_v - k_rev * u_v - (k_agg_t + k_frag_val) * u_v
            dhmw = k_agg_t * u_v
            dlmw = k_frag_val * (n_v + u_v)
            dac = k_ac_val * (1.0 - ac_v)
            ds = -k_ps80_val * s_v
            return np.array([dn, du, dhmw, dlmw, dac, ds], dtype=np.float64)

        sol = solve_ivp(
            rhs_var,
            t_span=(float(t_arr[0]), float(t_arr[-1])),
            y0=y0,
            t_eval=t_arr,
            method="Radau",
            rtol=1e-8,
            atol=1e-10,
            max_step=1.0,
        )

    if not sol.success:
        raise RuntimeError(f"3-state Radau ground-truth solver failed: {sol.message}")

    n_traj = sol.y[0]
    u_traj = sol.y[1]
    hmw_traj = sol.y[2]
    lmw_traj = sol.y[3]
    ac_traj = sol.y[4]
    s_traj = sol.y[5]
    m_traj = n_traj + u_traj

    return {
        "time_hours": t_arr,
        "monomer_fraction": m_traj,
        "native_fraction": n_traj,
        "unfolded_fraction": u_traj,
        "hmw_fraction": hmw_traj,
        "lmw_fraction": lmw_traj,
        "acidic_fraction": ac_traj,
        "ps80_intact_fraction": s_traj,
        "f_U": u_traj / np.maximum(m_traj, 1e-15),
        "mass_balance_fraction": n_traj + u_traj + hmw_traj + lmw_traj,
    }


def heteroscedastic_sec_sd_fraction(
    hmw_true_fraction: np.ndarray | float,
    base_sd_pct: float = 0.035,
    rel_rsd_slope: float = 0.03,
) -> np.ndarray | float:
    """
    Heteroscedastic SEC assay noise SD in fraction units:
        sigma_SEC(HMW) = sqrt((base_sd_pct / 100)^2 + (rel_rsd_slope * HMW)^2)
    where `base_sd_pct = 0.035` pp is configured in `hybrid_config.json` and marked
    "ASSUMPTION - SME to confirm; should come from method-validation intermediate precision".
    """
    sigma0_frac = base_sd_pct / 100.0
    return np.sqrt(sigma0_frac ** 2 + (rel_rsd_slope * np.asarray(hmw_true_fraction)) ** 2)


def generate_synthetic_datasets(
    output_dir: Path | str = DEFAULT_SYNTH_DIR,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Generates all Phase 2 synthetic datasets with realistic pull schedules and writes them to `output_dir`:
    - `synth_observations.csv`: 240 lots (100 train / 40 val / 100 test) with realistic pull schedules:
        * 5 °C: 0, 1, 3, 6, 9, 12, 18, 24 months (8 pulls)
        * 25 °C: 0, 1, 3, 6 months (4 pulls; 40% of lots per split also have 9, 12 months = 6 pulls)
        * 40 °C: 0, 0.5, 1, 2, 3 months (5 pulls)
      Contains ONLY observable covariates, pull conditions, and noisy assay measurements (NO `structural_group`).
    - `synth_excursions.csv`: 120 thermal excursion logger scenarios across validation/test lots with
      balanced representation clearly below, near, and clearly above the 0.099 pp detection gate.
    - `illustrative_demo_lot_NS-2026-004.csv`: 4-timepoint seeded demo lot NS-2026-004 (0, 1, 3, 6M
      at 5 °C and 25 °C), clearly stamped `ILLUSTRATIVE_DEMO_LOT - NOT USED IN TRAINING - NOT GMP DATA`.
    - `_ground_truth_do_not_read_in_train.json`: Separate ground-truth file containing `structural_group`,
      `structural_subgroup`, all hidden thermodynamic parameters, rate constants, PS80 depletion parameters,
      and noiseless Radau trajectories.
    - `dataset_manifest.json`: SHA-256 hashes, summary statistics by group and condition, excursion gate
      distribution, and structural mismatch note.
    """
    cfg = load_hybrid_config()
    train_cfg = cfg["training_and_uncertainty"]
    if seed is None:
        seed = int(train_cfg["random_seed"])

    rng = np.random.default_rng(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_train = int(train_cfg["n_train_lots"])  # 100
    n_val = int(train_cfg["n_val_lots"])      # 40
    n_test = int(train_cfg["n_test_lots"])    # 100

    base_sec_sd_pct = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    cex_sd_pct = float(cfg["assay_and_decision_thresholds"]["cex_acidic_repeatability_sd_pct"])
    gate_2sigma_pp = compute_excursion_detection_gate_pct(base_sec_sd_pct, 2.0)

    pulls_5c = np.array([0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0], dtype=np.float64)
    pulls_25c_std = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64)
    pulls_25c_ext = np.array([0.0, 1.0, 3.0, 6.0, 9.0, 12.0], dtype=np.float64)
    pulls_40c = np.array([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float64)

    # Split specification: (split_name, total_lots, n_matched, n_ps80_only, n_full_mismatch)
    splits_spec: List[Tuple[str, int, int, int, int]] = [
        ("train", n_train, 70, 15, 15),
        ("val", n_val, 28, 6, 6),
        ("test", n_test, 70, 15, 15),
    ]

    obs_rows: List[Dict[str, Any]] = []
    ground_truth_lots: Dict[str, Any] = {}
    lot_counter = 0

    for split_name, n_lots_split, n_mat, n_ps80, n_full in splits_spec:
        subgroups = (
            ["matched"] * n_mat
            + ["mismatched_ps80_only"] * n_ps80
            + ["mismatched_full"] * n_full
        )
        rng.shuffle(subgroups)

        # 40% of lots in each split receive extended 25 °C pulls (0,1,3,6,9,12M)
        n_ext_25 = int(round(0.40 * n_lots_split))
        ext_25_flags = [True] * n_ext_25 + [False] * (n_lots_split - n_ext_25)
        rng.shuffle(ext_25_flags)

        for idx_in_split in range(n_lots_split):
            lot_counter += 1
            sgrp = str(subgroups[idx_in_split])
            has_ext_25 = bool(ext_25_flags[idx_in_split])
            cov, truth = _sample_lot_covariates_and_truth(
                rng=rng,
                lot_index=lot_counter,
                split=split_name,
                subgroup=sgrp,
                extended_25c_pulls=has_ext_25,
                cfg=cfg,
            )
            lot_id = cov["lot_id"]
            truth["noiseless_trajectories"] = {}

            cond_schedules = [
                (5.0, pulls_5c),
                (25.0, pulls_25c_ext if has_ext_25 else pulls_25c_std),
                (40.0, pulls_40c),
            ]

            for temp_c, pull_months in cond_schedules:
                pull_hours = pull_months * HOURS_PER_MONTH
                traj = integrate_ground_truth_3state_radau(
                    time_grid_h=pull_hours,
                    time_knots_h=[0.0, float(pull_hours[-1])],
                    temp_knots_c=[temp_c, temp_c],
                    lot_truth=truth,
                )
                hmw_true_frac = traj["hmw_fraction"]
                lmw_true_frac = traj["lmw_fraction"]
                ac_true_frac = traj["acidic_fraction"]
                ps80_frac = traj["ps80_intact_fraction"]

                truth["noiseless_trajectories"][f"{int(temp_c)}C"] = {
                    "time_months": pull_months.tolist(),
                    "hmw_true_pct": (hmw_true_frac * 100.0).tolist(),
                    "lmw_true_pct": (lmw_true_frac * 100.0).tolist(),
                    "monomer_true_pct": (traj["monomer_fraction"] * 100.0).tolist(),
                    "acidic_true_pct": (ac_true_frac * 100.0).tolist(),
                    "ps80_intact_fraction": ps80_frac.tolist(),
                }

                # Add heteroscedastic SEC noise & CEX noise
                sigma_hmw_frac = heteroscedastic_sec_sd_fraction(
                    hmw_true_frac, base_sd_pct=base_sec_sd_pct, rel_rsd_slope=0.03
                )
                sigma_lmw_frac = base_sec_sd_pct * 0.65 / 100.0
                sigma_ac_frac = cex_sd_pct / 100.0

                hmw_obs_frac = np.clip(
                    hmw_true_frac + rng.normal(0.0, sigma_hmw_frac),
                    0.0005,
                    0.9990,
                )
                lmw_obs_frac = np.clip(
                    lmw_true_frac + rng.normal(0.0, sigma_lmw_frac),
                    0.0001,
                    0.9990,
                )
                hmw_obs_pct = hmw_obs_frac * 100.0
                lmw_obs_pct = lmw_obs_frac * 100.0
                monomer_obs_pct = np.clip(100.0 - hmw_obs_pct - lmw_obs_pct, 0.0, 100.0)
                acidic_obs_pct = np.clip(
                    (ac_true_frac + rng.normal(0.0, sigma_ac_frac, size=len(pull_months))) * 100.0,
                    1.0,
                    95.0,
                )

                for m_idx, m_val in enumerate(pull_months):
                    obs_rows.append(
                        {
                            "data_stamp": SYNTH_STAMP,
                            "generator_version": GENERATOR_VERSION,
                            "lot_id": lot_id,
                            "split": split_name,
                            "ph": cov["ph"],
                            "ionic_strength_mm": cov["ionic_strength_mm"],
                            "surfactant_pct": cov["surfactant_pct"],
                            "protein_conc_mg_ml": cov["protein_conc_mg_ml"],
                            "headspace_index": cov["headspace_index"],
                            "container_syringe_flag": cov["container_syringe_flag"],
                            "container_closure": cov["container_closure"],
                            "initial_afuc_glycan_pct": cov["initial_afuc_glycan_pct"],
                            "initial_acidic_pct": cov["initial_acidic_pct"],
                            "initial_hmw_pct": cov["initial_hmw_pct"],
                            "initial_lmw_pct": cov["initial_lmw_pct"],
                            "initial_monomer_pct": cov["initial_monomer_pct"],
                            "temperature_c": float(temp_c),
                            "time_months": float(m_val),
                            "time_hours": float(pull_hours[m_idx]),
                            "hmw_obs_pct": round(float(hmw_obs_pct[m_idx]), 5),
                            "lmw_obs_pct": round(float(lmw_obs_pct[m_idx]), 5),
                            "monomer_obs_pct": round(float(monomer_obs_pct[m_idx]), 5),
                            "acidic_obs_pct": round(float(acidic_obs_pct[m_idx]), 4),
                        }
                    )

            ground_truth_lots[lot_id] = truth

    # 4. Generate thermal excursion logger dataset (`synth_excursions.csv`) across val and test lots ONLY
    # Balanced across clearly below (<0.065 pp), near ([0.065, 0.145] pp), and clearly above (>0.145 pp) the 0.099 pp gate
    excursion_rows: List[Dict[str, Any]] = []
    excursion_ground_truth: Dict[str, Any] = {}
    val_test_ids = [
        lid for lid, lmeta in ground_truth_lots.items() if lmeta["split"] in ("val", "test")
    ]
    excursion_templates = [
        ("short_2h_28C", 28.0, 2.0, 0.5, 1),          # clearly below gate (~0.001 pp)
        ("cold_chain_36h_28C", 28.0, 36.0, 2.0, 1),   # clearly below gate (~0.017 pp)
        ("near_gate_60h_40C", 40.0, 60.0, 1.5, 1),    # near gate (~0.090 pp)
        ("near_gate_72h_40C", 40.0, 72.0, 1.5, 1),    # near gate (~0.108 pp)
        ("prolonged_7d_40C", 40.0, 168.0, 3.0, 1),    # clearly above gate (~0.25 pp)
        ("prolonged_14d_40C", 40.0, 336.0, 4.0, 1),   # clearly above gate (~0.51 pp)
    ]
    for exc_idx, lot_id in enumerate(val_test_ids[:120]):
        truth = ground_truth_lots[lot_id]
        tpl_name, peak_c, dur_h, ramp_h, n_rep = excursion_templates[
            exc_idx % len(excursion_templates)
        ]
        baseline_c = 5.0
        total_window_h = max(120.0, dur_h + 48.0)
        tk, tc = build_excursion_profile_knots(
            baseline_temp_c=baseline_c,
            excursion_temp_c=peak_c,
            excursion_duration_hours=dur_h,
            ramp_hours=ramp_h,
            n_spikes=n_rep,
            total_hours=total_window_h,
            start_hour=12.0,
        )
        t_eval_exc = np.linspace(0.0, total_window_h, int(total_window_h) + 1)
        traj_exc = integrate_ground_truth_3state_radau(
            time_grid_h=t_eval_exc,
            time_knots_h=tk,
            temp_knots_c=tc,
            lot_truth=truth,
        )
        hmw_pre_true_pct = float(traj_exc["hmw_fraction"][0]) * 100.0
        hmw_post_true_pct = float(traj_exc["hmw_fraction"][-1]) * 100.0
        dhmw_true_pp = hmw_post_true_pct - hmw_pre_true_pct
        peak_fu_true_pct = float(np.max(traj_exc["f_U"])) * 100.0

        sd_pre_pct = float(
            heteroscedastic_sec_sd_fraction(
                hmw_pre_true_pct / 100.0, base_sd_pct=base_sec_sd_pct
            )
            * 100.0
        )
        sd_post_pct = float(
            heteroscedastic_sec_sd_fraction(
                hmw_post_true_pct / 100.0, base_sd_pct=base_sec_sd_pct
            )
            * 100.0
        )
        hmw_pre_obs_pct = float(np.clip(hmw_pre_true_pct + rng.normal(0.0, sd_pre_pct), 0.05, 25.0))
        hmw_post_obs_pct = float(np.clip(hmw_post_true_pct + rng.normal(0.0, sd_post_pct), 0.05, 25.0))

        if dhmw_true_pp < 0.065:
            gate_regime = "clearly_below_gate"
        elif dhmw_true_pp <= 0.145:
            gate_regime = "near_gate"
        else:
            gate_regime = "clearly_above_gate"

        excursion_id = f"EXC-{exc_idx + 1:03d}"
        # Note: NO structural_group or true dHMW in synth_excursions.csv
        excursion_rows.append(
            {
                "data_stamp": SYNTH_STAMP,
                "excursion_id": excursion_id,
                "lot_id": lot_id,
                "split": truth["split"],
                "scenario_label": tpl_name,
                "baseline_temp_c": baseline_c,
                "peak_temp_c": peak_c,
                "duration_hours": dur_h,
                "ramp_hours": ramp_h,
                "n_repeats": n_rep,
                "total_window_hours": total_window_h,
                "hmw_pre_obs_pct": round(hmw_pre_obs_pct, 5),
                "hmw_post_obs_pct": round(hmw_post_obs_pct, 5),
                "delta_hmw_obs_pp": round(hmw_post_obs_pct - hmw_pre_obs_pct, 5),
            }
        )
        excursion_ground_truth[excursion_id] = {
            "excursion_id": excursion_id,
            "lot_id": lot_id,
            "split": truth["split"],
            "structural_group": truth["structural_group"],
            "structural_subgroup": truth["structural_subgroup"],
            "scenario_label": tpl_name,
            "gate_regime": gate_regime,
            "hmw_pre_true_pct": hmw_pre_true_pct,
            "hmw_post_true_pct": hmw_post_true_pct,
            "delta_hmw_true_pp": dhmw_true_pp,
            "peak_f_U_true_pct": peak_fu_true_pct,
        }

    # 5. Seeded illustrative demo lot `NS-2026-004` (4 timepoints: 0, 1, 3, 6M at 5C and 25C)
    demo_months = [0.0, 1.0, 3.0, 6.0]
    demo_5c_hmw = [0.62, 0.65, 0.70, 0.79]
    demo_25c_hmw = [0.62, 0.72, 0.88, 1.58]
    demo_rows: List[Dict[str, Any]] = []
    for temp_c, hmw_series in ((5.0, demo_5c_hmw), (25.0, demo_25c_hmw)):
        for m_val, hmw_val in zip(demo_months, hmw_series):
            demo_rows.append(
                {
                    "data_stamp": "ILLUSTRATIVE_DEMO_LOT - NOT USED IN TRAINING - NOT GMP DATA",
                    "lot_id": "NS-2026-004",
                    "split": "illustrative_demo",
                    "ph": 5.8,
                    "ionic_strength_mm": 150.0,
                    "surfactant_pct": 0.012,
                    "protein_conc_mg_ml": 100.0,
                    "headspace_index": 0.75,
                    "container_syringe_flag": 0,
                    "container_closure": "glass_vial",
                    "initial_afuc_glycan_pct": 6.5,
                    "initial_acidic_pct": 15.2,
                    "initial_hmw_pct": 0.62,
                    "initial_lmw_pct": 0.10,
                    "initial_monomer_pct": 99.28,
                    "temperature_c": temp_c,
                    "time_months": m_val,
                    "time_hours": m_val * HOURS_PER_MONTH,
                    "hmw_obs_pct": hmw_val,
                    "lmw_obs_pct": round(0.10 + 0.008 * m_val * (2.5 if temp_c > 10 else 1.0), 4),
                    "monomer_obs_pct": round(100.0 - hmw_val - (0.10 + 0.008 * m_val * (2.5 if temp_c > 10 else 1.0)), 4),
                    "acidic_obs_pct": round(15.2 + 0.18 * m_val * (3.2 if temp_c > 10 else 1.0), 4),
                }
            )

    df_obs = pd.DataFrame(obs_rows)
    df_exc = pd.DataFrame(excursion_rows)
    df_demo = pd.DataFrame(demo_rows)

    obs_csv_path = out_path / "synth_observations.csv"
    exc_csv_path = out_path / "synth_excursions.csv"
    demo_csv_path = out_path / "illustrative_demo_lot_NS-2026-004.csv"
    truth_json_path = out_path / "_ground_truth_do_not_read_in_train.json"
    manifest_json_path = out_path / "dataset_manifest.json"

    df_obs.to_csv(obs_csv_path, index=False)
    df_exc.to_csv(exc_csv_path, index=False)
    df_demo.to_csv(demo_csv_path, index=False)

    truth_payload = {
        "data_stamp": SYNTH_STAMP,
        "warning": "GROUND TRUTH FILE FOR PHASE 4/8 EVALUATION ONLY. DO NOT READ IN TRAINING.",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "config_sha256": get_config_sha256(),
        "lots": ground_truth_lots,
        "excursions": excursion_ground_truth,
    }
    truth_json_path.write_text(json.dumps(truth_payload, indent=2), encoding="utf-8")

    # Compute summary statistics for manifest
    dG_vals = np.array([v["dG_eff_ref_kj_mol"] for v in ground_truth_lots.values()])
    ea_vals = np.array([v["apparent_ea_5_40c_true_kj_mol"] for v in ground_truth_lots.values()])
    k5_vals = np.array([v["k_eff_5c_true_per_h"] for v in ground_truth_lots.values()])
    k25_vals = np.array([v["k_eff_25c_true_per_h"] for v in ground_truth_lots.values()])
    k40_vals = np.array([v["k_eff_40c_true_per_h"] for v in ground_truth_lots.values()])

    # Join structural_group from ground_truth ONLY for manifest summary reporting
    df_eval = df_obs.copy()
    df_eval["structural_group"] = df_eval["lot_id"].map(
        lambda lid: ground_truth_lots[lid]["structural_group"]
    )
    df_eval["structural_subgroup"] = df_eval["lot_id"].map(
        lambda lid: ground_truth_lots[lid]["structural_subgroup"]
    )

    hmw_by_group_cond: Dict[str, Any] = {}
    for sgrp in ("matched", "mismatched", "mismatched_ps80_only", "mismatched_full"):
        if sgrp in ("matched", "mismatched"):
            sub_df = df_eval[df_eval["structural_group"] == sgrp]
        else:
            sub_df = df_eval[df_eval["structural_subgroup"] == sgrp]
        hmw_by_group_cond[sgrp] = {}
        for cond_c, end_m in ((5.0, 24.0), (25.0, 6.0), (25.0, 12.0), (40.0, 3.0)):
            slice_df = sub_df[
                (sub_df["temperature_c"] == cond_c) & (sub_df["time_months"] == end_m)
            ]
            vals = slice_df["hmw_obs_pct"].to_numpy(dtype=float)
            hmw_by_group_cond[sgrp][f"{int(cond_c)}C_{int(end_m)}M"] = {
                "count": int(len(vals)),
                "mean": round(float(np.mean(vals)), 4) if len(vals) else 0.0,
                "std": round(float(np.std(vals)), 4) if len(vals) else 0.0,
                "min": round(float(np.min(vals)), 4) if len(vals) else 0.0,
                "median": round(float(np.median(vals)), 4) if len(vals) else 0.0,
                "max": round(float(np.max(vals)), 4) if len(vals) else 0.0,
            }

    # Excursion true dHMW distribution relative to 0.099 pp gate
    exc_dhmw = np.array([v["delta_hmw_true_pp"] for v in excursion_ground_truth.values()])
    exc_regimes = [v["gate_regime"] for v in excursion_ground_truth.values()]
    excursion_summary = {
        "total_excursions": len(excursion_ground_truth),
        "detection_gate_2sigma_pp": round(gate_2sigma_pp, 5),
        "clearly_below_gate_lt_0.065pp": int(sum(1 for r in exc_regimes if r == "clearly_below_gate")),
        "near_gate_0.065_to_0.145pp": int(sum(1 for r in exc_regimes if r == "near_gate")),
        "clearly_above_gate_gt_0.145pp": int(sum(1 for r in exc_regimes if r == "clearly_above_gate")),
        "strictly_below_0.099pp_count": int(np.sum(exc_dhmw < gate_2sigma_pp)),
        "strictly_above_0.099pp_count": int(np.sum(exc_dhmw >= gate_2sigma_pp)),
        "by_scenario": {},
    }
    for tpl_name, _, _, _, _ in excursion_templates:
        tpl_vals = np.array(
            [v["delta_hmw_true_pp"] for v in excursion_ground_truth.values() if v["scenario_label"] == tpl_name]
        )
        excursion_summary["by_scenario"][tpl_name] = {
            "count": int(len(tpl_vals)),
            "min_pp": round(float(np.min(tpl_vals)), 5),
            "median_pp": round(float(np.median(tpl_vals)), 5),
            "mean_pp": round(float(np.mean(tpl_vals)), 5),
            "max_pp": round(float(np.max(tpl_vals)), 5),
        }

    split_summary = {}
    for sp in ("train", "val", "test"):
        sp_lots = [v for v in ground_truth_lots.values() if v["split"] == sp]
        split_summary[sp] = {
            "total_lots": len(sp_lots),
            "matched_lots": sum(1 for v in sp_lots if v["structural_group"] == "matched"),
            "mismatched_lots": sum(1 for v in sp_lots if v["structural_group"] == "mismatched"),
            "mismatched_ps80_only_lots": sum(1 for v in sp_lots if v["structural_subgroup"] == "mismatched_ps80_only"),
            "mismatched_full_lots": sum(1 for v in sp_lots if v["structural_subgroup"] == "mismatched_full"),
            "extended_25c_12m_lots": sum(1 for v in sp_lots if v["extended_25c_pulls"]),
            "observation_rows": int((df_obs["split"] == sp).sum()),
        }

    manifest = {
        "data_stamp": SYNTH_STAMP,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "config_sha256": get_config_sha256(),
        "max_observed_hmw_pct_overall": round(float(df_obs["hmw_obs_pct"].max()), 4),
        "files": {
            "synth_observations.csv": {
                "sha256": compute_file_sha256(obs_csv_path),
                "rows": len(df_obs),
                "columns": list(df_obs.columns),
            },
            "synth_excursions.csv": {
                "sha256": compute_file_sha256(exc_csv_path),
                "rows": len(df_exc),
                "columns": list(df_exc.columns),
            },
            "illustrative_demo_lot_NS-2026-004.csv": {
                "sha256": compute_file_sha256(demo_csv_path),
                "rows": len(df_demo),
                "columns": list(df_demo.columns),
            },
            "_ground_truth_do_not_read_in_train.json": {
                "sha256": compute_file_sha256(truth_json_path),
                "lots_count": len(ground_truth_lots),
                "excursions_count": len(excursion_ground_truth),
            },
        },
        "splits": split_summary,
        "hmw_obs_pct_summary_by_group_and_condition": hmw_by_group_cond,
        "excursion_dhmw_distribution_vs_gate": excursion_summary,
        "ground_truth_parameter_summary": {
            "dG_eff_ref_kj_mol": {
                "min": float(np.min(dG_vals)),
                "median": float(np.median(dG_vals)),
                "mean": float(np.mean(dG_vals)),
                "max": float(np.max(dG_vals)),
            },
            "apparent_ea_5_40c_kj_mol": {
                "min": float(np.min(ea_vals)),
                "median": float(np.median(ea_vals)),
                "mean": float(np.mean(ea_vals)),
                "max": float(np.max(ea_vals)),
            },
            "k_eff_5c_per_h": {
                "min": float(np.min(k5_vals)),
                "median": float(np.median(k5_vals)),
                "max": float(np.max(k5_vals)),
            },
            "k_eff_25c_per_h": {
                "min": float(np.min(k25_vals)),
                "median": float(np.median(k25_vals)),
                "max": float(np.max(k25_vals)),
            },
            "k_eff_40c_per_h": {
                "min": float(np.min(k40_vals)),
                "median": float(np.median(k40_vals)),
                "max": float(np.max(k40_vals)),
            },
        },
        "anti_inverse_crime_design_note": (
            "All 240 synthetic lots are integrated with the 3-state stiff Radau ODE solver at finite "
            "reverse relaxation rate k_-1 (never the rapid pre-equilibrium RK4 solver used for fitting). "
            "70% of each split ('matched': 70 train / 28 val / 70 test) use finite k_-1 in [75, 150] h^-1 "
            "with dCp_agg = 0 and no surfactant depletion. 15% of each split ('mismatched_ps80_only': "
            "15 train / 6 val / 15 test) include ONLY late-time PS80 depletion (with dCp_agg = 0 and fast k_-1) "
            "as a clean ablation set. 15% of each split ('mismatched_full': 15 train / 6 val / 15 test) combine "
            "late-time PS80 depletion + dCp_agg in [0.05, 0.12] kJ/(mol*K) + slower k_-1 in [18, 55] h^-1. "
            "Neither structural_group nor structural_subgroup appears in synth_observations.csv or synth_excursions.csv."
        ),
    }
    manifest_json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_observations_for_training(
    csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_observations.csv",
    allowed_splits: Sequence[str] = ("train", "val"),
) -> pd.DataFrame:
    """
    Safe training data loader that strictly reads ONLY observable stability CSV data
    and blocks any attempt to read `_ground_truth_do_not_read_in_train.json`, excursion files,
    or any DataFrame containing ground-truth or `structural_group` columns.
    """
    path = Path(csv_path)
    name_lower = path.name.lower()
    if (
        "ground_truth" in name_lower
        or "excursion" in name_lower
        or path.suffix.lower() == ".json"
    ):
        raise PermissionError(
            f"Leakage guard triggered: Training loader is forbidden from reading '{path.name}'."
        )

    df = pd.read_csv(path)
    for col in df.columns:
        for forbidden in FORBIDDEN_GROUND_TRUTH_COLUMNS:
            if forbidden.lower() in col.lower():
                raise PermissionError(
                    f"Leakage guard triggered: Forbidden ground-truth column '{col}' found in '{path.name}'."
                )

    if allowed_splits:
        df = df[df["split"].isin(allowed_splits)].copy()

    # Ensure illustrative demo lot is never included in training
    df = df[df["lot_id"] != "NS-2026-004"].copy()
    return df.reset_index(drop=True)


def load_ground_truth_for_verification_only(
    truth_path: Path | str = DEFAULT_SYNTH_DIR / "_ground_truth_do_not_read_in_train.json",
) -> Dict[str, Any]:
    """
    Explicit evaluation-only helper to load `_ground_truth_do_not_read_in_train.json`
    for Phase 3/4/8 stratified error, parameter recovery, and coverage reporting.
    Must NEVER be imported or called by `model.py` or `train.py` during optimization.
    """
    return json.loads(Path(truth_path).read_text(encoding="utf-8"))


def generate_calibration_dataset(
    output_dir: Path | str = DEFAULT_SYNTH_DIR,
    cal_seed: int | None = None,
    n_cal_lots: int = 60,
) -> Dict[str, Any]:
    """
    Generates 60 extra synthetic calibration lots (`SYN-CAL-241` .. `SYN-CAL-300`) using an
    independent RNG seed substream (`seed + 99999 = 20361022`) with the exact same generator,
    70% matched / 15% mismatched_ps80_only / 15% mismatched_full mix, and realistic pull schedules.

    Leaves the existing `synth_observations.csv`, `synth_excursions.csv`,
    `illustrative_demo_lot_NS-2026-004.csv`, and `_ground_truth_do_not_read_in_train.json`
    files and their SHA-256 hashes completely untouched.
    """
    cfg = load_hybrid_config()
    if cal_seed is None:
        cal_seed = int(cfg["training_and_uncertainty"]["random_seed"]) + 99999

    rng = np.random.default_rng(cal_seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_sec_sd_pct = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    cex_sd_pct = float(cfg["assay_and_decision_thresholds"]["cex_acidic_repeatability_sd_pct"])

    pulls_5c = np.array([0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0], dtype=np.float64)
    pulls_25c_std = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64)
    pulls_25c_ext = np.array([0.0, 1.0, 3.0, 6.0, 9.0, 12.0], dtype=np.float64)
    pulls_40c = np.array([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float64)

    n_mat = int(round(0.70 * n_cal_lots))   # 42
    n_ps80 = int(round(0.15 * n_cal_lots))  # 9
    n_full = n_cal_lots - n_mat - n_ps80    # 9

    subgroups = (
        ["matched"] * n_mat
        + ["mismatched_ps80_only"] * n_ps80
        + ["mismatched_full"] * n_full
    )
    rng.shuffle(subgroups)

    n_ext_25 = int(round(0.40 * n_cal_lots))  # 24
    ext_25_flags = [True] * n_ext_25 + [False] * (n_cal_lots - n_ext_25)
    rng.shuffle(ext_25_flags)

    cal_obs_rows: List[Dict[str, Any]] = []
    cal_ground_truth_lots: Dict[str, Any] = {}

    for idx in range(n_cal_lots):
        lot_num = 241 + idx
        sgrp = str(subgroups[idx])
        has_ext_25 = bool(ext_25_flags[idx])
        cov, truth = _sample_lot_covariates_and_truth(
            rng=rng,
            lot_index=lot_num,
            split="cal",
            subgroup=sgrp,
            extended_25c_pulls=has_ext_25,
            cfg=cfg,
        )
        lot_id = cov["lot_id"]
        truth["noiseless_trajectories"] = {}

        cond_schedules = [
            (5.0, pulls_5c),
            (25.0, pulls_25c_ext if has_ext_25 else pulls_25c_std),
            (40.0, pulls_40c),
        ]

        for temp_c, pull_months in cond_schedules:
            pull_hours = pull_months * HOURS_PER_MONTH
            traj = integrate_ground_truth_3state_radau(
                time_grid_h=pull_hours,
                time_knots_h=[0.0, float(pull_hours[-1])],
                temp_knots_c=[temp_c, temp_c],
                lot_truth=truth,
            )
            hmw_true_frac = traj["hmw_fraction"]
            lmw_true_frac = traj["lmw_fraction"]
            ac_true_frac = traj["acidic_fraction"]
            ps80_frac = traj["ps80_intact_fraction"]

            truth["noiseless_trajectories"][f"{int(temp_c)}C"] = {
                "time_months": pull_months.tolist(),
                "hmw_true_pct": (hmw_true_frac * 100.0).tolist(),
                "lmw_true_pct": (lmw_true_frac * 100.0).tolist(),
                "monomer_true_pct": (traj["monomer_fraction"] * 100.0).tolist(),
                "acidic_true_pct": (ac_true_frac * 100.0).tolist(),
                "ps80_intact_fraction": ps80_frac.tolist(),
            }

            sigma_hmw_frac = heteroscedastic_sec_sd_fraction(
                hmw_true_frac, base_sd_pct=base_sec_sd_pct, rel_rsd_slope=0.03
            )
            sigma_lmw_frac = base_sec_sd_pct * 0.65 / 100.0
            sigma_ac_frac = cex_sd_pct / 100.0

            hmw_obs_frac = np.clip(
                hmw_true_frac + rng.normal(0.0, sigma_hmw_frac),
                0.0005,
                0.9990,
            )
            lmw_obs_frac = np.clip(
                lmw_true_frac + rng.normal(0.0, sigma_lmw_frac),
                0.0001,
                0.9990,
            )
            hmw_obs_pct = hmw_obs_frac * 100.0
            lmw_obs_pct = lmw_obs_frac * 100.0
            monomer_obs_pct = np.clip(100.0 - hmw_obs_pct - lmw_obs_pct, 0.0, 100.0)
            acidic_obs_pct = np.clip(
                (ac_true_frac + rng.normal(0.0, sigma_ac_frac, size=len(pull_months))) * 100.0,
                1.0,
                95.0,
            )

            for m_idx, m_val in enumerate(pull_months):
                cal_obs_rows.append(
                    {
                        "data_stamp": SYNTH_STAMP,
                        "generator_version": GENERATOR_VERSION,
                        "lot_id": lot_id,
                        "split": "cal",
                        "ph": cov["ph"],
                        "ionic_strength_mm": cov["ionic_strength_mm"],
                        "surfactant_pct": cov["surfactant_pct"],
                        "protein_conc_mg_ml": cov["protein_conc_mg_ml"],
                        "headspace_index": cov["headspace_index"],
                        "container_syringe_flag": cov["container_syringe_flag"],
                        "container_closure": cov["container_closure"],
                        "initial_afuc_glycan_pct": cov["initial_afuc_glycan_pct"],
                        "initial_acidic_pct": cov["initial_acidic_pct"],
                        "initial_hmw_pct": cov["initial_hmw_pct"],
                        "initial_lmw_pct": cov["initial_lmw_pct"],
                        "initial_monomer_pct": cov["initial_monomer_pct"],
                        "temperature_c": float(temp_c),
                        "time_months": float(m_val),
                        "time_hours": float(pull_hours[m_idx]),
                        "hmw_obs_pct": round(float(hmw_obs_pct[m_idx]), 5),
                        "lmw_obs_pct": round(float(lmw_obs_pct[m_idx]), 5),
                        "monomer_obs_pct": round(float(monomer_obs_pct[m_idx]), 5),
                        "acidic_obs_pct": round(float(acidic_obs_pct[m_idx]), 4),
                    }
                )

        cal_ground_truth_lots[lot_id] = truth

    df_cal = pd.DataFrame(cal_obs_rows)
    cal_csv_path = out_path / "synth_calibration_observations.csv"
    cal_truth_path = out_path / "_calibration_ground_truth_do_not_read_in_train.json"

    df_cal.to_csv(cal_csv_path, index=False)
    cal_truth_payload = {
        "data_stamp": SYNTH_STAMP,
        "warning": "CALIBRATION GROUND TRUTH FILE FOR PHASE 4 CONFORMAL CALIBRATION ONLY. DO NOT READ IN TRAINING.",
        "generator_version": GENERATOR_VERSION,
        "cal_seed": cal_seed,
        "config_sha256": get_config_sha256(),
        "lots": cal_ground_truth_lots,
    }
    cal_truth_path.write_text(json.dumps(cal_truth_payload, indent=2), encoding="utf-8")

    return {
        "cal_seed": cal_seed,
        "n_cal_lots": n_cal_lots,
        "matched_lots": n_mat,
        "mismatched_ps80_only_lots": n_ps80,
        "mismatched_full_lots": n_full,
        "extended_25c_12m_lots": n_ext_25,
        "observation_rows": len(df_cal),
        "synth_calibration_observations_sha256": compute_file_sha256(cal_csv_path),
        "calibration_ground_truth_sha256": compute_file_sha256(cal_truth_path),
    }
