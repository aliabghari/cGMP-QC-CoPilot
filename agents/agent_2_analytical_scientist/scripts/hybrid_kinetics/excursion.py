"""
Phase 5: Thermal Excursion Evaluator (`excursion.py`).

Evaluates arbitrary or benchmark thermal excursion profiles using the trained 25-member
hybrid Lumry-Eyring + ANN ensemble (`hybrid_ensemble_weights.npz`) and validation-lot
conformal calibration (`calibration_bundle.json`).

Mandatory regulatory & scientific guardrails:
- Verdict wording is strictly limited to:
  - `"no detectable change"` (upper 95% bound of excursion-attributable dHMW < 2*sqrt(2)*SD gate)
  - `"detectable change likely"` (lower 95% bound > 2*sqrt(2)*SD gate)
  - `"inconclusive"` (95% interval straddles the 2*sqrt(2)*SD gate)
- Never outputs release, quarantine, or disposition recommendations (`"RELEASE FOR USE"` is banned).
- Always includes two explicit methodological caveats in outputs and memos:
  1. Standard aqueous SEC-HPLC measures irreversible soluble aggregates (HMW) and cannot separate
     transient reversible unfolded monomer (`U`) from native monomer (`N`) once returned to 5 °C.
  2. Peak unfolded fraction `f_U(T)` is prior-driven (`K_u(5°C)`, `dH_u` are not uniquely
     identifiable from SEC HMW trajectories alone) and relaxes rapidly (`t_1/2 < 1 min` at the
     default refolding rate `k_-1 = 100 h^-1`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import load_hybrid_config
from .model import extract_feature_matrix, predict_params_from_npz_member
from .ode import build_excursion_profile_knots, integrate_pathway_rk4
from .physics import (
    celsius_to_kelvin,
    classify_excursion_verdict,
    compute_excursion_detection_gate_pct,
    eyring_thermodynamics,
    lumry_eyring_rates,
)
from .train import DEFAULT_MODEL_DIR, load_exported_ensemble_npz

SEC_REVERSIBLE_VS_IRREVERSIBLE_CAVEAT = (
    "SEC-HPLC measures irreversible soluble HMW aggregates and cannot separate reversible "
    "partially unfolded intermediate (U) from native monomer (N) after cooling back to 5 °C."
)

PRIOR_DRIVEN_FU_CAVEAT = (
    "Peak unfolded fraction f_U(T) is prior-driven (K_u(5 °C) and dH_u are not uniquely identifiable "
    "from SEC HMW trajectories alone) and relaxes in < 1 min upon cooling at default k_-1 = 100 h^-1."
)

NOMINAL_LOT_COVARIATES: Dict[str, Any] = {
    "ph": 6.0,
    "ionic_strength_mm": 120.0,
    "surfactant_pct": 0.020,
    "protein_conc_mg_ml": 50.0,
    "headspace_index": 0.40,
    "container_syringe_flag": 0,
    "initial_afuc_glycan_pct": 6.0,
    "initial_acidic_pct": 14.5,
    "initial_hmw_pct": 0.62,
    "initial_lmw_pct": 0.0,
}


def evaluate_thermal_excursion(
    baseline_temp_c: float = 5.0,
    excursion_temp_c: float = 28.0,
    excursion_duration_hours: float = 36.0,
    ramp_hours: float = 0.0,
    n_spikes: int = 1,
    total_window_hours: float = 120.0,
    lot_covariates: Dict[str, Any] | None = None,
    spec_limit_hmw_pct: float = 2.00,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Evaluates a cold-chain thermal excursion profile using the 25-member trained hybrid ensemble
    and the validation-excursion conformal interval calibration.

    Uses the single starting-monomer convention:
        M_0 = 1 - HMW_0 - LMW_0
    and computes excursion-attributable `delta_hmw_pp` strictly above the 5 °C baseline over the
    excursion duration (`excursion_duration_hours * n_spikes`).
    """
    if cfg is None:
        cfg = load_hybrid_config()
    sd_sec = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    gate_pp = compute_excursion_detection_gate_pct(sd_sec, 2.0)

    cov = dict(NOMINAL_LOT_COVARIATES)
    if lot_covariates is not None:
        cov.update(lot_covariates)

    hmw0_pct = float(cov.get("initial_hmw_pct", 0.62))
    lmw0_pct = float(cov.get("initial_lmw_pct", 0.0))
    hmw0_frac = hmw0_pct / 100.0
    lmw0_frac = lmw0_pct / 100.0
    m0_frac = 1.0 - hmw0_frac - lmw0_frac

    mdir = Path(model_dir)
    weights_list, f_mean, f_std = load_exported_ensemble_npz(mdir / "hybrid_ensemble_weights.npz")
    cal_path = mdir / "calibration_bundle.json"
    if cal_path.exists():
        import json
        cal_bundle = json.loads(cal_path.read_text(encoding="utf-8"))
        exc_cal = cal_bundle.get("excursion_calibration_from_val_lots", {})
        rel_floor = float(exc_cal.get("lot_rel_sd_floor", 0.05))
        q_exc_mult = float(exc_cal.get("conformal_multiplier_q95_val_excursions", 1.96))
    else:
        rel_floor = 0.05
        q_exc_mult = 1.96

    x_norm, _, _ = extract_feature_matrix(cov, feature_mean=f_mean, feature_std=f_std)

    eff_window_h = max(total_window_hours, 24.0 + n_spikes * (excursion_duration_hours + 2.0 * ramp_hours) + 12.0)
    tk, tc = build_excursion_profile_knots(
        baseline_temp_c=baseline_temp_c,
        excursion_temp_c=excursion_temp_c,
        excursion_duration_hours=excursion_duration_hours,
        ramp_hours=ramp_hours,
        n_spikes=n_spikes,
        total_hours=eff_window_h,
        start_hour=24.0,
    )
    dt_h = 0.5
    time_grid = np.arange(0.0, eff_window_h + 0.5 * dt_h, dt_h)

    member_trajs_pct: List[np.ndarray] = []
    member_exc_dhmw_pp: List[float] = []
    member_eq_5c_days: List[float] = []
    member_peak_fu_pct: List[float] = []
    member_k_ratio: List[float] = []

    active_exc_hours = float(n_spikes) * float(excursion_duration_hours)

    for mw in weights_list:
        mp = predict_params_from_npz_member(x_norm[0], mw, cfg=cfg)
        single_p = {k: float(np.asarray(mp[k]).item()) for k in mp}

        # Full window trajectory for UI plotting
        traj = integrate_pathway_rk4(
            time_grid_h=time_grid,
            time_knots_h=tk,
            temp_knots_c=tc,
            params=single_p,
            initial_hmw_fraction=hmw0_frac,
            initial_lmw_fraction=lmw0_frac,
            include_lmw=True,
        )
        member_trajs_pct.append(traj["hmw_fraction"] * 100.0)

        # Excursion-attributable dHMW over the excursion window
        r_ref = lumry_eyring_rates(celsius_to_kelvin(5.0), **single_p)
        r_peak = lumry_eyring_rates(celsius_to_kelvin(excursion_temp_c), **single_p)
        k_5c = float(np.asarray(r_ref["k_eff_fraction_per_h"]).item())
        k_peak = float(np.asarray(r_peak["k_eff_fraction_per_h"]).item())
        fu_peak_pct = float(np.asarray(r_peak["f_U"]).item()) * 100.0

        if ramp_hours <= 1e-9 and n_spikes == 1:
            d_pp = m0_frac * (1.0 - math.exp(-k_peak * active_exc_hours)) * 100.0
        else:
            # Integrate strictly across the excursion spike window
            spike_end = 24.0 + n_spikes * (excursion_duration_hours + 2.0 * ramp_hours + max(0, n_spikes - 1) * 4.0)
            t_sp = np.linspace(24.0, spike_end, 121)
            tr_sp = integrate_pathway_rk4(
                time_grid_h=t_sp,
                time_knots_h=tk,
                temp_knots_c=tc,
                params=single_p,
                initial_hmw_fraction=hmw0_frac,
                initial_lmw_fraction=lmw0_frac,
                include_lmw=True,
            )
            d_pp = float(tr_sp["hmw_fraction"][-1] - tr_sp["hmw_fraction"][0]) * 100.0

        eq_days = (d_pp / 100.0) / max(m0_frac * k_5c * 24.0, 1e-15)
        member_exc_dhmw_pp.append(d_pp)
        member_eq_5c_days.append(eq_days)
        member_peak_fu_pct.append(fu_peak_pct)
        member_k_ratio.append(k_peak / max(k_5c, 1e-15))

    traj_mat = np.stack(member_trajs_pct, axis=0)
    traj_mean_pct = np.mean(traj_mat, axis=0)
    traj_l95_pct = np.quantile(traj_mat, 0.025, axis=0)
    traj_u95_pct = np.quantile(traj_mat, 0.975, axis=0)

    # Nominal single-step Eyring reference trajectory (without unfolded intermediate U)
    nom_p = {k: float(cfg["parameter_bounds"][k]["nominal"]) for k in cfg["parameter_bounds"]}
    r_5c_nom = lumry_eyring_rates(celsius_to_kelvin(5.0), **nom_p)
    k_5c_nom = float(np.asarray(r_5c_nom["k_eff_fraction_per_h"]).item())
    dg_eff_ref_nom = float(nom_p["dG_eff_ref_kj_mol"])
    dh_agg_nom = float(nom_p["dH_agg_ref_kj_mol"])
    ds_eff_nom = (dh_agg_nom * 1000.0 - dg_eff_ref_nom * 1000.0) / 278.15

    temp_interp_c = np.interp(time_grid, tk, tc)
    single_step_hmw_pct = np.zeros_like(time_grid)
    single_step_hmw_pct[0] = hmw0_pct
    for i in range(1, len(time_grid)):
        tc_prev = float(temp_interp_c[i - 1])
        k_ss = float(
            np.asarray(
                eyring_thermodynamics(
                    celsius_to_kelvin(tc_prev),
                    dH_ref_kj_mol=dh_agg_nom,
                    dS_ref_j_mol_k=ds_eff_nom,
                )["k_fraction_per_h"]
            ).item()
        )
        m_rem = max(0.0, 1.0 - single_step_hmw_pct[i - 1] / 100.0 - lmw0_frac)
        single_step_hmw_pct[i] = single_step_hmw_pct[i - 1] + (k_ss * m_rem * dt_h) * 100.0

    exc_arr = np.asarray(member_exc_dhmw_pp, dtype=np.float64)
    mean_dhmw_pp = float(np.mean(exc_arr))
    ens_sd_pp = float(np.std(exc_arr, ddof=1))
    tot_sd_pp = math.sqrt(ens_sd_pp ** 2 + (rel_floor * mean_dhmw_pp) ** 2)
    l95_dhmw_pp = max(0.0, mean_dhmw_pp - q_exc_mult * tot_sd_pp)
    u95_dhmw_pp = mean_dhmw_pp + q_exc_mult * tot_sd_pp

    verdict_info = classify_excursion_verdict(
        l95_dhmw_pp, u95_dhmw_pp, sec_repeatability_sd_pct=sd_sec
    )
    verdict_str = verdict_info["verdict"]

    eq_5c_days_mean = float(np.mean(member_eq_5c_days))
    eq_5c_months_mean = eq_5c_days_mean / 30.4375
    peak_fu_mean_pct = float(np.mean(member_peak_fu_pct))
    peak_fu_l95_pct = float(np.quantile(member_peak_fu_pct, 0.025))
    peak_fu_u95_pct = float(np.quantile(member_peak_fu_pct, 0.975))
    mean_k_ratio = float(np.mean(member_k_ratio))

    post_hmw_mean_pct = hmw0_pct + mean_dhmw_pp
    remaining_margin_pp = max(0.0, spec_limit_hmw_pct - post_hmw_mean_pct)
    k_5c_per_month_pp = m0_frac * k_5c_nom * 730.5 * 100.0
    rem_shelf_life_months = round(remaining_margin_pp / max(k_5c_per_month_pp, 1e-9), 1)

    df_trajectory = pd.DataFrame(
        {
            "Time_Hours": time_grid,
            "Temperature_C": np.round(temp_interp_c, 3),
            "HMW_Hybrid_Eyring_ANN_Pct": np.round(traj_mean_pct, 5),
            "HMW_Hybrid_L95_Pct": np.round(traj_l95_pct, 5),
            "HMW_Hybrid_U95_Pct": np.round(traj_u95_pct, 5),
            "HMW_Naive_Eyring_Only_Pct": np.round(single_step_hmw_pct, 5),
            "ANN_TVP_Acceleration_Factor": np.round(np.where(temp_interp_c > 8.0, mean_k_ratio, 1.0), 3),
            "Spec_Limit_Pct": spec_limit_hmw_pct,
        }
    )

    ruling_statement = (
        f"Draft assessment for QA review. Kinetic model result: '{verdict_str}' "
        f"(Excursion dHMW = +{mean_dhmw_pp:.4f} pp, 95% interval [{l95_dhmw_pp:.4f}, {u95_dhmw_pp:.4f}] pp "
        f"vs. 2*sqrt(2)*SD pre/post detection gate {gate_pp:.4f} pp; "
        f"consumes {eq_5c_days_mean:.2f} 5°C-equivalent days; "
        f"prior-driven peak f_U = {peak_fu_mean_pct:.4f}%). "
        f"Disposition decision rests with QA. "
        f"Note: {SEC_REVERSIBLE_VS_IRREVERSIBLE_CAVEAT}"
    )

    return {
        "df_trajectory": df_trajectory,
        "summary": {
            "starting_monomer_fraction_m0": round(m0_frac, 4),
            "initial_hmw_pct": round(hmw0_pct, 4),
            "post_excursion_hmw_hybrid_pct": round(post_hmw_mean_pct, 4),
            "post_excursion_hmw_naive_pct": round(float(single_step_hmw_pct[-1]), 4),
            "delta_hmw_excursion_pp": round(mean_dhmw_pp, 4),
            "delta_hmw_excursion_l95_pp": round(l95_dhmw_pp, 4),
            "delta_hmw_excursion_u95_pp": round(u95_dhmw_pp, 4),
            # Backward-compatible alias used by existing app.py / comms_generator.py
            "delta_hmw_excursion_pct": round(mean_dhmw_pp, 4),
            "detection_gate_2sigma_pp": round(gate_pp, 5),
            "verdict": verdict_str,
            "equivalent_5c_days_consumed": round(eq_5c_days_mean, 2),
            # Backward-compatible alias
            "equivalent_4c_days_consumed": round(eq_5c_days_mean, 2),
            "shelf_life_deduction_months": round(eq_5c_months_mean, 2),
            "projected_remaining_shelf_life_months": rem_shelf_life_months,
            "peak_f_u_prior_driven_pct": round(peak_fu_mean_pct, 4),
            "peak_f_u_prior_driven_l95_pct": round(peak_fu_l95_pct, 4),
            "peak_f_u_prior_driven_u95_pct": round(peak_fu_u95_pct, 4),
            "max_ann_tvp_factor": round(mean_k_ratio, 2),
            "sec_reversible_vs_irreversible_caveat": SEC_REVERSIBLE_VS_IRREVERSIBLE_CAVEAT,
            "prior_driven_f_u_caveat": PRIOR_DRIVEN_FU_CAVEAT,
            # Backward-compatible key with non-disposition wording
            "disposition_recommendation": ruling_statement,
        },
    }
