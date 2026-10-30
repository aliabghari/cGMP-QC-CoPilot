"""
Phase 4: Uncertainty Quantification, Split-Conformal Calibration, Lot-Level Bayesian Adaptation,
Baseline Comparisons, Physical-Validity Audits, and Held-Out Test Coverage & Excursion Verdict
Evaluation (`uncertainty.py`).

Key capabilities:
1. Distinguishes the **Mean Confidence Band** (uncertainty in the latent noiseless degradation curve)
   from the **Observation Prediction Interval** (which adds heteroscedastic SEC assay noise and
   conformal calibration, reported both per-lot simultaneously as the primary pass/fail gate and
   per-observation descriptively, plus stratified by observable `temperature_c x time_months` cells).
2. Calibrates observation-PI nonconformity quantiles on the 60 independent calibration lots
   (`synth_calibration_observations.csv`, `split="cal"`).
   NOTE ON LATENT PARAMETER INTERVALS (`dG_eff_ref`, `k_eff(5/25/40C)`, `apparent_ea_5_40c_kj_mol`):
   **CALIBRATION POSSIBLE ONLY IN SYNTHETIC DATA; ON REAL LOTS THESE PARAMETERS ARE UNOBSERVABLE.**
3. Calibrates excursion `dHMW` conformal intervals exclusively on the **validation-lot excursions**
   (`48` excursions on `val` lots) because the 60 isothermal calibration lots have no excursions,
   and evaluates the verdict confusion matrix, false-reassurance rate, and false-alarm rate
   (with exact Clopper-Pearson 95% upper bounds) strictly on the **test-lot excursions** (`72` excursions).
4. Provides `adapt_lot_from_early_pulls()` (low-dimensional lot offset `[delta_dG_eff, delta_dH_agg]`
   fitted by penalized nonlinear least squares through the ODE with a prior from the ANN ensemble)
   and evaluates `6M / 9M / 12M` forecast HMW RMSE and 95% PI coverage on validation and test lots
   both **without adaptation** (prior from covariates only) and **with adaptation from `0/1/3M` pulls**.
5. Trains and evaluates three comparison baselines (`pooled_mean`, `ridge` with `pH` and `pH^2`,
   and a tuned PyTorch `blackbox_mlp` with hyperparameter search and early stopping on `val` lots).
6. Runs the **Physical-Validity Audit** (training both the Hybrid ODE model and the tuned Black-Box MLP
   on pulls `<= 3M` only, then evaluating temporal extrapolation `> 3M` and temperature interpolation
   at `15 °C`, counting purity-increase / negative-degradation-rate violations and non-monotone
   temperature-rate violations).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import beta as scipy_beta

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False

from .config import get_config_sha256, load_hybrid_config
from .model import (
    PREDICTED_PARAM_KEYS,
    BoundedKineticMLP,
    compute_lot_apparent_ea,
    extract_feature_matrix,
    integrate_isothermal_rk4_grid,
    predict_params_from_npz_member,
)
from .ode import (
    build_excursion_profile_knots,
    closed_form_isothermal_trajectory,
    integrate_pathway_rk4,
)
from .physics import (
    HOURS_PER_MONTH,
    apparent_activation_energy,
    celsius_to_kelvin,
    classify_excursion_verdict,
    compute_amendment_ab_summary,
    compute_excursion_detection_gate_pct,
    lumry_eyring_rates,
)
from .synth import (
    DEFAULT_SYNTH_DIR,
    LOT_COVARIATE_COLUMNS,
    generate_calibration_dataset,
    heteroscedastic_sec_sd_fraction,
    integrate_ground_truth_3state_radau,
    load_ground_truth_for_verification_only,
    load_observations_for_training,
)
from .train import (
    DEFAULT_MODEL_DIR,
    RK4_STEP_MONTHS,
    _build_split_lot_tensors,
    compute_split_loss_and_predictions_torch,
    evaluate_member_or_ensemble_numpy,
    load_exported_ensemble_npz,
)

IDENTIFIABLE_METRIC_KEYS: Tuple[str, ...] = (
    "dG_eff_ref_kj_mol",
    "k_eff_5c_per_h",
    "k_eff_25c_per_h",
    "k_eff_40c_per_h",
    "apparent_ea_5_40c_kj_mol",
)

TRUTH_KEY_MAP: Dict[str, str] = {
    "dG_eff_ref_kj_mol": "dG_eff_ref_kj_mol",
    "k_eff_5c_per_h": "k_eff_5c_true_per_h",
    "k_eff_25c_per_h": "k_eff_25c_true_per_h",
    "k_eff_40c_per_h": "k_eff_40c_true_per_h",
    "apparent_ea_5_40c_kj_mol": "apparent_ea_5_40c_true_kj_mol",
}

PRIOR_DRIVEN_NONIDENTIFIABLE_KEYS: Tuple[str, ...] = (
    "dH_agg_ref_kj_mol",
    "ku_ref_5c",
    "dH_u_kj_mol",
)

SYNTHETIC_ONLY_PARAM_CALIBRATION_NOTE = (
    "calibration possible only in synthetic data; on real lots these parameters are unobservable"
)


def clopper_pearson_interval(
    k: int,
    n: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Computes the exact Clopper-Pearson (1 - alpha) binomial confidence interval for k successes
    out of n trials using Beta distribution quantiles.
    """
    if n <= 0:
        raise ValueError("Number of trials n must be positive.")
    if k < 0 or k > n:
        raise ValueError(f"Invalid successes k={k} for n={n}.")
    lower = 0.0 if k == 0 else float(scipy_beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(scipy_beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def evaluate_coverage_verdict(
    k: int,
    n: int,
    nominal_target: float = 0.95,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Evaluates empirical coverage k/n using the exact Clopper-Pearson 95% binomial interval:
    - FAIL only if upper bound < 95% (statistically significant under-coverage)
    - "conservative / intervals too wide" (not a failure) if lower bound > 95%
    - PASS otherwise (95% lies inside the Clopper-Pearson interval)
    """
    obs_cov = float(k) / float(n)
    l95, u95 = clopper_pearson_interval(k, n, alpha=alpha)
    if u95 < nominal_target:
        verdict = "FAIL (under-coverage)"
    elif l95 > nominal_target:
        verdict = "conservative / intervals too wide"
    else:
        verdict = "PASS"
    return {
        "k": int(k),
        "n": int(n),
        "k_over_n": f"{k}/{n}",
        "observed_coverage": round(obs_cov, 4),
        "observed_coverage_pct": round(obs_cov * 100.0, 2),
        "clopper_pearson_l95": round(l95, 4),
        "clopper_pearson_u95": round(u95, 4),
        "verdict": verdict,
    }


def _finite_sample_conformal_quantile(scores: np.ndarray, alpha: float = 0.05) -> float:
    """
    Computes the exact finite-sample split-conformal quantile at level (1 - alpha):
        index k = ceil((n + 1) * (1 - alpha))
    clipped to n, returning the k-th smallest score (1-indexed).
    """
    arr = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(arr)
    if n == 0:
        return 1.96
    k_idx = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
    k_idx = max(0, min(n - 1, k_idx))
    return float(arr[k_idx])


def calibrate_uncertainty_on_calibration_lots(
    ensemble_weights: List[Dict[str, np.ndarray]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    cal_obs_csv: Path | str = DEFAULT_SYNTH_DIR / "synth_calibration_observations.csv",
    cal_truth_json: Path | str = DEFAULT_SYNTH_DIR / "_calibration_ground_truth_do_not_read_in_train.json",
    obs_csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_observations.csv",
    exc_csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_excursions.csv",
    truth_json_path: Path | str = DEFAULT_SYNTH_DIR / "_ground_truth_do_not_read_in_train.json",
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Calibrates:
    1. Observation Prediction Interval nonconformity multipliers (`q_obs_pointwise_95`,
       `q_obs_simultaneous_lot_95`, plus HMW-level-scaled multiplier `q_obs_level_scaled_95`)
       on the 60 calibration lots (`split="cal"`).
    2. Latent identifiable parameter intervals (`dG_eff_ref`, `k_eff(5/25/40C)`, `apparent_ea_5_40c_kj_mol`)
       on the 60 calibration lots, explicitly labeled:
       `"calibration possible only in synthetic data; on real lots these parameters are unobservable"`.
    3. Excursion `dHMW` interval multiplier (`conformal_multiplier_q95_val_excursions`) exclusively
       on the **validation-lot excursions** (`split="val"`, 48 excursions), because the 60 isothermal
       calibration lots have no excursions and test-lot excursions (`72`) must remain strictly held-out.
    """
    if cfg is None:
        cfg = load_hybrid_config()

    cal_csv_path = Path(cal_obs_csv)
    cal_truth_path = Path(cal_truth_json)
    if not cal_csv_path.exists() or not cal_truth_path.exists():
        generate_calibration_dataset(output_dir=cal_csv_path.parent)

    df_cal = load_observations_for_training(cal_csv_path, allowed_splits=("cal",))
    cal_truth = load_ground_truth_for_verification_only(cal_truth_path)["lots"]

    cal_eval = evaluate_member_or_ensemble_numpy(
        ensemble_weights, feature_mean, feature_std, df_cal, cfg=cfg
    )
    param_summaries = cal_eval["lot_param_summaries"]
    obs_df = cal_eval["observations_df"].copy()

    base_sec_sd_pct = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    alpha = float(cfg["training_and_uncertainty"]["clopper_pearson_alpha"])

    # 1. Latent identifiable parameter calibration (SYNTHETIC-ONLY: uses true latent parameters of the 60 cal lots)
    param_calibration: Dict[str, Dict[str, Any]] = {}
    for m_key in IDENTIFIABLE_METRIC_KEYS:
        t_key = TRUTH_KEY_MAP[m_key]
        raw_res = []
        ens_sds = []
        for lid, ps in param_summaries.items():
            lt = cal_truth[lid]
            mu_val = float(ps[f"{m_key}_mean"])
            sd_ens = float(ps[f"{m_key}_std"])
            true_val = float(lt[t_key])
            if "k_eff" in m_key:
                raw_res.append(abs(true_val - mu_val) / max(mu_val, 1e-12))
                ens_sds.append(sd_ens / max(mu_val, 1e-12))
            else:
                raw_res.append(abs(true_val - mu_val))
                ens_sds.append(sd_ens)

        raw_res_arr = np.asarray(raw_res, dtype=np.float64)
        ens_sd_arr = np.asarray(ens_sds, dtype=np.float64)
        rms_res = float(np.sqrt(np.mean(raw_res_arr ** 2)))
        mean_ens_var = float(np.mean(ens_sd_arr ** 2))
        lot_effect_sd = float(math.sqrt(max(1e-12, rms_res ** 2 - mean_ens_var)))

        comb_scale = np.sqrt(ens_sd_arr ** 2 + lot_effect_sd ** 2)
        nonconf_scores = raw_res_arr / np.maximum(comb_scale, 1e-12)
        q_conf = _finite_sample_conformal_quantile(nonconf_scores, alpha=alpha)

        param_calibration[m_key] = {
            "observability_note": SYNTHETIC_ONLY_PARAM_CALIBRATION_NOTE,
            "scale_mode": "relative" if "k_eff" in m_key else "absolute",
            "lot_effect_sd": lot_effect_sd,
            "conformal_multiplier_q95": q_conf,
            "cal_rms_error": rms_res,
        }

    # 2. Trajectory & Observation Prediction Interval calibration on the 60 calibration lots (uses observable HMW%)
    hmw0_map = df_cal.groupby("lot_id")["initial_hmw_pct"].first().to_dict()
    log_sig_map = {lid: float(ps["log_sigma_adj_mean"]) for lid, ps in param_summaries.items()}

    obs_df["hmw0_pct"] = obs_df["lot_id"].map(hmw0_map)
    obs_df["log_sigma_adj"] = obs_df["lot_id"].map(log_sig_map)
    obs_df["delta_hmw_pred_pp"] = np.maximum(0.0, obs_df["hmw_pred_mean_pct"] - obs_df["hmw0_pct"])
    obs_df["sec_sd_pct"] = (
        heteroscedastic_sec_sd_fraction(
            obs_df["hmw_pred_mean_pct"].to_numpy(dtype=np.float64) / 100.0,
            base_sd_pct=base_sec_sd_pct,
            rel_rsd_slope=0.03,
        )
        * 100.0
        * np.exp(obs_df["log_sigma_adj"].to_numpy(dtype=np.float64))
    )

    active_mask = obs_df["delta_hmw_pred_pp"] > 0.15
    if np.any(active_mask):
        sub_active = obs_df[active_mask]
        excess_var = np.maximum(
            0.0,
            (sub_active["residual_pp"] ** 2)
            - (sub_active["hmw_pred_epistemic_sd_pct"] ** 2)
            - (sub_active["sec_sd_pct"] ** 2),
        )
        gamma_lot_traj = float(
            np.sqrt(np.median(excess_var / np.maximum(sub_active["delta_hmw_pred_pp"] ** 2, 1e-6)))
        )
        gamma_lot_traj = float(np.clip(gamma_lot_traj, 0.06, 0.25))
    else:
        gamma_lot_traj = 0.10

    obs_df["traj_sd_pp"] = np.sqrt(
        (obs_df["hmw_pred_epistemic_sd_pct"] ** 2)
        + ((gamma_lot_traj * obs_df["delta_hmw_pred_pp"]) ** 2)
    )
    obs_df["obs_total_sd_pp"] = np.sqrt(
        (obs_df["traj_sd_pp"] ** 2) + (obs_df["sec_sd_pct"] ** 2)
    )
    obs_df["abs_std_score"] = np.abs(obs_df["residual_pp"]) / np.maximum(obs_df["obs_total_sd_pp"], 1e-9)

    q_obs_pointwise = _finite_sample_conformal_quantile(
        obs_df["abs_std_score"].to_numpy(dtype=np.float64), alpha=alpha
    )
    lot_max_scores = obs_df.groupby("lot_id")["abs_std_score"].max().to_numpy(dtype=np.float64)
    q_obs_simultaneous_lot = _finite_sample_conformal_quantile(lot_max_scores, alpha=alpha)

    # Optional HMW-level-scaled nonconformity score: scales total SD by (1 + 0.35 * delta_hmw_pred_pp)
    level_scale_beta = 0.35
    obs_df["obs_level_scaled_sd_pp"] = obs_df["obs_total_sd_pp"] * (
        1.0 + level_scale_beta * obs_df["delta_hmw_pred_pp"]
    )
    obs_df["abs_level_scaled_score"] = np.abs(obs_df["residual_pp"]) / np.maximum(
        obs_df["obs_level_scaled_sd_pp"], 1e-9
    )
    q_obs_level_scaled = _finite_sample_conformal_quantile(
        obs_df["abs_level_scaled_score"].to_numpy(dtype=np.float64), alpha=alpha
    )
    lot_max_level_scores = (
        obs_df.groupby("lot_id")["abs_level_scaled_score"].max().to_numpy(dtype=np.float64)
    )
    q_obs_level_scaled_sim_lot = _finite_sample_conformal_quantile(lot_max_level_scores, alpha=alpha)

    # 3. Calibrate adapted-forecast prediction intervals on the 60 calibration lots (0/1/3M -> 6/9/12M)
    cal_adapted_scores: List[float] = []
    cal_adapted_scores_matched: List[float] = []
    for lid_c, l_df_c in df_cal.groupby("lot_id"):
        lid_c_str = str(lid_c)
        cov_c = l_df_c.iloc[0].to_dict()
        hmw0_c_pct = float(cov_c["initial_hmw_pct"])
        hmw0_c_frac = hmw0_c_pct / 100.0
        lmw0_c_frac = float(cov_c["initial_lmw_pct"]) / 100.0
        early_c = l_df_c[
            (l_df_c["time_months"] <= 3.0) & (l_df_c["temperature_c"].isin([5.0, 25.0]))
        ].copy()
        adapt_c = adapt_lot_from_early_pulls(
            cov_c, early_c, ensemble_weights, feature_mean, feature_std, cfg=cfg
        )
        post_c = adapt_c["posterior_members"]
        fut_c = l_df_c[l_df_c["time_months"].isin([6.0, 9.0, 12.0])].copy()
        for _, frow_c in fut_c.iterrows():
            tc_c = float(frow_c["temperature_c"])
            th_c = float(frow_c["time_hours"])
            y_obs_c = float(frow_c["hmw_obs_pct"])
            po_preds_c, log_sigs_c = [], []
            for m_po in post_c:
                c_po = closed_form_isothermal_trajectory(
                    [0.0, th_c], tc_c, params=m_po,
                    initial_hmw_fraction=hmw0_c_frac, initial_lmw_fraction=lmw0_c_frac, include_lmw=True
                )["hmw_fraction"][-1] * 100.0
                po_preds_c.append(float(c_po))
                log_sigs_c.append(float(m_po.get("log_sigma_adj", 0.0)))
            mean_po_c = float(np.mean(po_preds_c))
            sd_po_c = float(np.std(po_preds_c, ddof=1))
            mean_lsig_c = float(np.mean(log_sigs_c))
            d_po_c = max(0.0, mean_po_c - hmw0_c_pct)
            sec_po_c = (
                float(np.asarray(heteroscedastic_sec_sd_fraction(mean_po_c / 100.0, base_sd_pct=base_sec_sd_pct)).item())
                * 100.0
                * math.exp(mean_lsig_c)
            )
            # After 0/1/3M adaptation, lot random-effect residual scale is 0.25 * gamma_lot_traj
            tot_sd_po_c = math.sqrt(sd_po_c ** 2 + (0.25 * gamma_lot_traj * d_po_c) ** 2 + sec_po_c ** 2)
            score_c = abs(y_obs_c - mean_po_c) / max(tot_sd_po_c, 1e-9)
            cal_adapted_scores.append(score_c)
            if cal_truth[lid_c_str]["structural_group"] == "matched":
                cal_adapted_scores_matched.append(score_c)

    q_adapted_future_95 = _finite_sample_conformal_quantile(
        np.asarray(cal_adapted_scores, dtype=np.float64), alpha=alpha
    )
    q_adapted_future_matched_95 = _finite_sample_conformal_quantile(
        np.asarray(cal_adapted_scores_matched, dtype=np.float64), alpha=alpha
    )

    # 4. Calibrate excursion dHMW conformal multiplier exclusively on validation-lot excursions (n=40)
    df_all_obs = load_observations_for_training(obs_csv_path, allowed_splits=("train", "val", "test"))
    lot_cov_df = df_all_obs.groupby("lot_id", as_index=False).first()
    lot_cov_map = {row["lot_id"]: row.to_dict() for _, row in lot_cov_df.iterrows()}

    df_exc = pd.read_csv(exc_csv_path)
    df_exc_val = df_exc[df_exc["split"] == "val"].copy()
    exc_truth = load_ground_truth_for_verification_only(truth_json_path)["excursions"]

    lot_rel_sd_floor = 0.05
    val_exc_scores: List[float] = []
    for _, erow in df_exc_val.iterrows():
        exc_id = str(erow["excursion_id"])
        lid = str(erow["lot_id"])
        cov = lot_cov_map[lid]
        x_norm, _, _ = extract_feature_matrix(cov, feature_mean=feature_mean, feature_std=feature_std)

        tk, tc = build_excursion_profile_knots(
            baseline_temp_c=float(erow["baseline_temp_c"]),
            excursion_temp_c=float(erow["peak_temp_c"]),
            excursion_duration_hours=float(erow["duration_hours"]),
            ramp_hours=float(erow["ramp_hours"]),
            n_spikes=int(erow["n_repeats"]),
            total_hours=float(erow["total_window_hours"]),
            start_hour=12.0,
        )
        tot_h = float(erow["total_window_hours"])
        t_grid = np.linspace(0.0, tot_h, max(61, int(tot_h // 2) + 1))
        hmw0 = float(cov["initial_hmw_pct"]) / 100.0
        lmw0 = float(cov["initial_lmw_pct"]) / 100.0

        member_dhmw_pp = []
        for mw in ensemble_weights:
            mp = predict_params_from_npz_member(x_norm[0], mw, cfg=cfg)
            traj = integrate_pathway_rk4(
                time_grid_h=t_grid,
                time_knots_h=tk,
                temp_knots_c=tc,
                params=mp,
                initial_hmw_fraction=hmw0,
                initial_lmw_fraction=lmw0,
                include_lmw=True,
            )
            member_dhmw_pp.append(float(traj["hmw_fraction"][-1] - traj["hmw_fraction"][0]) * 100.0)

        arr_pp = np.asarray(member_dhmw_pp, dtype=np.float64)
        mean_pp = float(np.mean(arr_pp))
        ens_sd_pp = float(np.std(arr_pp, ddof=1))
        scale_pp = math.sqrt(ens_sd_pp ** 2 + (lot_rel_sd_floor * mean_pp) ** 2)
        true_pp = float(exc_truth[exc_id]["delta_hmw_true_pp"])
        val_exc_scores.append(abs(true_pp - mean_pp) / max(scale_pp, 1e-12))

    q_exc_val = _finite_sample_conformal_quantile(np.asarray(val_exc_scores, dtype=np.float64), alpha=alpha)

    calibration_bundle = {
        "calibration_lots_count": len(param_summaries),
        "calibration_observations_count": len(obs_df),
        "alpha": alpha,
        "gamma_lot_trajectory": round(gamma_lot_traj, 5),
        "q_obs_pointwise_95": round(q_obs_pointwise, 5),
        "q_obs_simultaneous_lot_95": round(q_obs_simultaneous_lot, 5),
        "adapted_future_pi_calibration": {
            "calibration_lots_count": 60,
            "future_6m_9m_12m_observations_count": len(cal_adapted_scores),
            "residual_gamma_scale_factor": 0.25,
            "q_adapted_future_overall_95": round(q_adapted_future_95, 5),
            "q_adapted_future_matched_95": round(q_adapted_future_matched_95, 5),
        },
        "level_scaled_pi": {
            "level_scale_beta": level_scale_beta,
            "q_obs_level_scaled_pointwise_95": round(q_obs_level_scaled, 5),
            "q_obs_level_scaled_simultaneous_lot_95": round(q_obs_level_scaled_sim_lot, 5),
        },
        "identifiable_parameters_note": SYNTHETIC_ONLY_PARAM_CALIBRATION_NOTE,
        "identifiable_parameters": param_calibration,
        "excursion_calibration_from_val_lots": {
            "calibration_source": "validation_lots_excursions_only (split='val', n=40; test lots strictly held-out, n=80)",
            "note": (
                "The 60 isothermal calibration lots have no excursions; excursion dHMW intervals [L95, U95] "
                "are calibrated on the 40 validation-lot excursions using score |true_dhmw_pp - mean_pp| / "
                "sqrt(ens_sd_pp^2 + (lot_rel_sd_floor * mean_pp)^2), and evaluated strictly on the 80 held-out "
                "test-lot excursions."
            ),
            "val_excursions_count": len(val_exc_scores),
            "lot_rel_sd_floor": lot_rel_sd_floor,
            "conformal_multiplier_q95_val_excursions": round(q_exc_val, 5),
        },
    }
    return calibration_bundle


def predict_lot_intervals_and_trajectories(
    lot_df: pd.DataFrame,
    ensemble_weights: List[Dict[str, np.ndarray]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    calibration_bundle: Dict[str, Any],
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Computes calibrated 95% intervals for identifiable quantities (synthetic-only calibration),
    the Mean Confidence Band (uncertainty in the latent noiseless curve), and the Observation
    Prediction Interval (both standard and HMW-level-scaled, pointwise and simultaneous lot-level)
    on `lot_df`.
    """
    if cfg is None:
        cfg = load_hybrid_config()
    base_sec_sd_pct = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])

    eval_out = evaluate_member_or_ensemble_numpy(
        ensemble_weights, feature_mean, feature_std, lot_df, cfg=cfg
    )
    param_summaries = eval_out["lot_param_summaries"]
    obs_df = eval_out["observations_df"].copy()

    p_cal = calibration_bundle["identifiable_parameters"]
    for lid, ps in param_summaries.items():
        for m_key in IDENTIFIABLE_METRIC_KEYS:
            c_info = p_cal[m_key]
            mu_val = float(ps[f"{m_key}_mean"])
            sd_ens = float(ps[f"{m_key}_std"])
            lot_sd = float(c_info["lot_effect_sd"])
            q_mult = float(c_info["conformal_multiplier_q95"])
            if c_info["scale_mode"] == "relative":
                rel_sd = math.sqrt((sd_ens / max(mu_val, 1e-12)) ** 2 + lot_sd ** 2)
                half_w = q_mult * rel_sd * mu_val
            else:
                tot_sd = math.sqrt(sd_ens ** 2 + lot_sd ** 2)
                half_w = q_mult * tot_sd
            ps[f"{m_key}_cal_l95"] = mu_val - half_w
            ps[f"{m_key}_cal_u95"] = mu_val + half_w

    hmw0_map = lot_df.groupby("lot_id")["initial_hmw_pct"].first().to_dict()
    log_sig_map = {lid: float(ps["log_sigma_adj_mean"]) for lid, ps in param_summaries.items()}

    gamma_lot = float(calibration_bundle["gamma_lot_trajectory"])
    q_pt = float(calibration_bundle["q_obs_pointwise_95"])
    q_sim = float(calibration_bundle["q_obs_simultaneous_lot_95"])

    lvl_cfg = calibration_bundle.get("level_scaled_pi", {})
    lvl_beta = float(lvl_cfg.get("level_scale_beta", 0.35))
    q_lvl_pt = float(lvl_cfg.get("q_obs_level_scaled_pointwise_95", q_pt))
    q_lvl_sim = float(lvl_cfg.get("q_obs_level_scaled_simultaneous_lot_95", q_sim))

    obs_df["hmw0_pct"] = obs_df["lot_id"].map(hmw0_map)
    obs_df["log_sigma_adj"] = obs_df["lot_id"].map(log_sig_map)
    obs_df["delta_hmw_pred_pp"] = np.maximum(0.0, obs_df["hmw_pred_mean_pct"] - obs_df["hmw0_pct"])
    obs_df["sec_sd_pct"] = (
        heteroscedastic_sec_sd_fraction(
            obs_df["hmw_pred_mean_pct"].to_numpy(dtype=np.float64) / 100.0,
            base_sd_pct=base_sec_sd_pct,
            rel_rsd_slope=0.03,
        )
        * 100.0
        * np.exp(obs_df["log_sigma_adj"].to_numpy(dtype=np.float64))
    )

    # Mean Confidence Band (noiseless latent trajectory uncertainty: epistemic ensemble spread only)
    obs_df["mean_traj_sd_pp"] = obs_df["hmw_pred_epistemic_sd_pct"]
    obs_df["mean_ci95_lower_pct"] = np.maximum(
        0.0, obs_df["hmw_pred_mean_pct"] - q_pt * obs_df["mean_traj_sd_pp"]
    )
    obs_df["mean_ci95_upper_pct"] = obs_df["hmw_pred_mean_pct"] + q_pt * obs_df["mean_traj_sd_pp"]

    # Total trajectory + lot-effect SD and Observation Prediction Interval
    obs_df["traj_sd_pp"] = np.sqrt(
        (obs_df["hmw_pred_epistemic_sd_pct"] ** 2)
        + ((gamma_lot * obs_df["delta_hmw_pred_pp"]) ** 2)
    )
    obs_df["obs_total_sd_pp"] = np.sqrt(
        (obs_df["traj_sd_pp"] ** 2) + (obs_df["sec_sd_pct"] ** 2)
    )
    obs_df["obs_pi95_pointwise_lower_pct"] = np.maximum(
        0.0, obs_df["hmw_pred_mean_pct"] - q_pt * obs_df["obs_total_sd_pp"]
    )
    obs_df["obs_pi95_pointwise_upper_pct"] = (
        obs_df["hmw_pred_mean_pct"] + q_pt * obs_df["obs_total_sd_pp"]
    )
    obs_df["obs_pi95_simultaneous_lower_pct"] = np.maximum(
        0.0, obs_df["hmw_pred_mean_pct"] - q_sim * obs_df["obs_total_sd_pp"]
    )
    obs_df["obs_pi95_simultaneous_upper_pct"] = (
        obs_df["hmw_pred_mean_pct"] + q_sim * obs_df["obs_total_sd_pp"]
    )

    # HMW-level-scaled Observation Prediction Interval
    obs_df["obs_level_scaled_sd_pp"] = obs_df["obs_total_sd_pp"] * (
        1.0 + lvl_beta * obs_df["delta_hmw_pred_pp"]
    )
    obs_df["obs_pi95_level_scaled_pointwise_lower_pct"] = np.maximum(
        0.0, obs_df["hmw_pred_mean_pct"] - q_lvl_pt * obs_df["obs_level_scaled_sd_pp"]
    )
    obs_df["obs_pi95_level_scaled_pointwise_upper_pct"] = (
        obs_df["hmw_pred_mean_pct"] + q_lvl_pt * obs_df["obs_level_scaled_sd_pp"]
    )
    obs_df["obs_pi95_level_scaled_simultaneous_lower_pct"] = np.maximum(
        0.0, obs_df["hmw_pred_mean_pct"] - q_lvl_sim * obs_df["obs_level_scaled_sd_pp"]
    )
    obs_df["obs_pi95_level_scaled_simultaneous_upper_pct"] = (
        obs_df["hmw_pred_mean_pct"] + q_lvl_sim * obs_df["obs_level_scaled_sd_pp"]
    )

    return {
        "lot_param_summaries": param_summaries,
        "observations_df": obs_df,
    }


def adapt_lot_from_early_pulls(
    lot_covariates: Dict[str, Any],
    early_pulls_df: pd.DataFrame,
    ensemble_weights: List[Dict[str, np.ndarray]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    cfg: Dict[str, Any] | None = None,
    prior_scale_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """
    Low-dimensional Bayesian MAP / penalized nonlinear least-squares adaptation of a lot's
    kinetic parameters (`[delta_dG_eff, delta_dH_agg]`) through the ODE from its early stability
    pulls (e.g., 0, 1, 3 months at 5/25/40 °C), regularized toward the ANN ensemble prediction.

    `prior_scale_multiplier` scales the default prior standard deviations
    (`sigma_prior_dg = 0.28 * prior_scale_multiplier`, `sigma_prior_dh = 2.20 * prior_scale_multiplier`)
    so sensitivity to the prior (`x0.5, x1, x2, x4`) can be audited transparently.
    """
    if cfg is None:
        cfg = load_hybrid_config()
    bounds = cfg["parameter_bounds"]
    base_sec_sd_pct = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])

    x_norm, _, _ = extract_feature_matrix(
        lot_covariates, feature_mean=feature_mean, feature_std=feature_std
    )
    hmw0_frac = float(lot_covariates["initial_hmw_pct"]) / 100.0
    lmw0_frac = float(lot_covariates.get("initial_lmw_pct", 0.10)) / 100.0

    prior_members: List[Dict[str, float]] = []
    for mw in ensemble_weights:
        mp = predict_params_from_npz_member(x_norm[0], mw, cfg=cfg)
        prior_members.append({k: float(np.asarray(mp[k]).item()) for k in mp})

    if len(early_pulls_df) == 0:
        return {
            "prior_members": prior_members,
            "posterior_members": prior_members,
            "adapted": False,
        }

    # Filter out t=0 pulls where HMW == HMW0 exactly (no rate information at t=0)
    active_early = early_pulls_df[early_pulls_df["time_hours"] > 0.0].copy()
    if len(active_early) == 0:
        return {
            "prior_members": prior_members,
            "posterior_members": prior_members,
            "adapted": False,
        }

    t_hours = active_early["time_hours"].to_numpy(dtype=np.float64)
    temps_c = active_early["temperature_c"].to_numpy(dtype=np.float64)
    hmw_obs_frac = active_early["hmw_obs_pct"].to_numpy(dtype=np.float64) / 100.0
    sigma_obs_frac = heteroscedastic_sec_sd_fraction(
        hmw_obs_frac, base_sd_pct=base_sec_sd_pct, rel_rsd_slope=0.03
    )

    dg_min = float(bounds["dG_eff_ref_kj_mol"]["min"])
    dg_max = float(bounds["dG_eff_ref_kj_mol"]["max"])
    dh_min = float(bounds["dH_agg_ref_kj_mol"]["min"])
    dh_max = float(bounds["dH_agg_ref_kj_mol"]["max"])

    # Representative prior mean member
    mean_prior_p: Dict[str, float] = {}
    for k in prior_members[0]:
        mean_prior_p[k] = float(np.mean([m[k] for m in prior_members]))
    mean_prior_p.pop("dG_agg_ref_kj_mol", None)
    mean_prior_p.pop("dS_agg_ref_j_mol_k", None)

    dg0 = float(mean_prior_p["dG_eff_ref_kj_mol"])
    dh0 = float(mean_prior_p["dH_agg_ref_kj_mol"])

    # Prior regularization scales (lot random-effect SDs scaled by prior_scale_multiplier)
    sigma_prior_dg = 0.28 * float(prior_scale_multiplier)
    sigma_prior_dh = 2.20 * float(prior_scale_multiplier)

    def _res_fn(theta: np.ndarray) -> np.ndarray:
        dg_try, dh_try = float(theta[0]), float(theta[1])
        p_try = dict(mean_prior_p)
        p_try["dG_eff_ref_kj_mol"] = dg_try
        p_try["dH_agg_ref_kj_mol"] = dh_try
        preds = []
        for tc_i, th_i in zip(temps_c, t_hours):
            cf = closed_form_isothermal_trajectory(
                time_grid_h=[0.0, float(th_i)],
                temp_c=float(tc_i),
                params=p_try,
                initial_hmw_fraction=hmw0_frac,
                initial_lmw_fraction=lmw0_frac,
                include_lmw=True,
            )
            preds.append(float(cf["hmw_fraction"][-1]))
        preds_arr = np.asarray(preds, dtype=np.float64)
        data_res = (preds_arr - hmw_obs_frac) / sigma_obs_frac
        prior_res = np.array(
            [(dg_try - dg0) / sigma_prior_dg, (dh_try - dh0) / sigma_prior_dh],
            dtype=np.float64,
        )
        return np.concatenate([data_res, prior_res])

    opt = least_squares(
        _res_fn,
        x0=np.array([dg0, dh0], dtype=np.float64),
        bounds=([dg_min, dh_min], [dg_max, dh_max]),
        method="trf",
        ftol=1e-7,
        xtol=1e-7,
    )
    delta_dg = float(opt.x[0] - dg0)
    delta_dh = float(opt.x[1] - dh0)

    posterior_members: List[Dict[str, float]] = []
    for mp in prior_members:
        post_p = dict(mp)
        post_p.pop("dG_agg_ref_kj_mol", None)
        post_p.pop("dS_agg_ref_j_mol_k", None)
        post_p["dG_eff_ref_kj_mol"] = float(np.clip(mp["dG_eff_ref_kj_mol"] + delta_dg, dg_min, dg_max))
        post_p["dH_agg_ref_kj_mol"] = float(np.clip(mp["dH_agg_ref_kj_mol"] + delta_dh, dh_min, dh_max))
        r_ref = lumry_eyring_rates(278.15, **post_p)
        post_p["dG_agg_ref_kj_mol"] = float(np.asarray(r_ref["dG_agg_ref_kj_mol"]).item())
        post_p["dS_agg_ref_j_mol_k"] = float(np.asarray(r_ref["dS_agg_ref_j_mol_k"]).item())
        posterior_members.append(post_p)

    return {
        "prior_members": prior_members,
        "posterior_members": posterior_members,
        "adapted": True,
        "delta_dG_eff_kj_mol": round(delta_dg, 5),
        "delta_dH_agg_kj_mol": round(delta_dh, 5),
        "prior_dG_eff_mean": float(np.mean([m["dG_eff_ref_kj_mol"] for m in prior_members])),
        "posterior_dG_eff_mean": float(np.mean([m["dG_eff_ref_kj_mol"] for m in posterior_members])),
        "prior_dH_agg_mean": float(np.mean([m["dH_agg_ref_kj_mol"] for m in prior_members])),
        "posterior_dH_agg_mean": float(np.mean([m["dH_agg_ref_kj_mol"] for m in posterior_members])),
    }


def evaluate_early_pull_lot_adaptation_on_splits(
    ensemble_weights: List[Dict[str, np.ndarray]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    calibration_bundle: Dict[str, Any],
    obs_csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_observations.csv",
    truth_json_path: Path | str = DEFAULT_SYNTH_DIR / "_ground_truth_do_not_read_in_train.json",
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Follow-up #3 Deliverable:
    Evaluates `adapt_lot_from_early_pulls()` (fitting `0, 1, 3M` pulls at 5/25 °C) when forecasting
    future pulls at `6M, 9M, 12M` across the 40 validation lots and 100 test lots, reporting HMW RMSE
    and 95% PI coverage **with and without adaptation**, stratified by `overall`, `matched`, and `mismatched`.
    """
    if cfg is None:
        cfg = load_hybrid_config()
    base_sec_sd_pct = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    gamma_lot = float(calibration_bundle["gamma_lot_trajectory"])
    q_pt = float(calibration_bundle["q_obs_pointwise_95"])
    adapt_cal = calibration_bundle.get("adapted_future_pi_calibration", {})
    q_adapt_all = float(adapt_cal.get("q_adapted_future_overall_95", q_pt))
    q_adapt_mat = float(adapt_cal.get("q_adapted_future_matched_95", q_pt))

    truth_lots = load_ground_truth_for_verification_only(truth_json_path)["lots"]

    def _eval_split(split_name: str) -> Dict[str, Any]:
        df_sp = load_observations_for_training(obs_csv_path, allowed_splits=(split_name,))
        records = []
        for lid, l_df in df_sp.groupby("lot_id"):
            lid_str = str(lid)
            cov = l_df.iloc[0].to_dict()
            hmw0_pct = float(cov["initial_hmw_pct"])
            hmw0_frac = hmw0_pct / 100.0
            lmw0_frac = float(cov["initial_lmw_pct"]) / 100.0

            # Adapt using only 0, 1, 3M pulls at 5C and 25C
            early_df = l_df[
                (l_df["time_months"] <= 3.0) & (l_df["temperature_c"].isin([5.0, 25.0]))
            ].copy()
            adapt_out = adapt_lot_from_early_pulls(
                cov, early_df, ensemble_weights, feature_mean, feature_std, cfg=cfg
            )
            prior_m = adapt_out["prior_members"]
            post_m = adapt_out["posterior_members"]

            # Classic linear 0-3M fit at 25C for OOT operating characteristics comparison
            early_25 = early_df[early_df["temperature_c"] == 25.0].sort_values("time_months")
            if len(early_25) >= 2:
                sl_25, int_25 = np.polyfit(
                    early_25["time_months"].to_numpy(dtype=np.float64),
                    early_25["hmw_obs_pct"].to_numpy(dtype=np.float64),
                    1,
                )
            else:
                sl_25, int_25 = 0.08, hmw0_pct

            # Evaluate on future 6M, 9M, 12M pulls
            future_df = l_df[l_df["time_months"].isin([6.0, 9.0, 12.0])].copy()
            for _, frow in future_df.iterrows():
                tc = float(frow["temperature_c"])
                th = float(frow["time_hours"])
                tm = float(frow["time_months"])
                y_obs = float(frow["hmw_obs_pct"])

                pr_preds, po_preds, log_sigs = [], [], []
                for m_pr, m_po in zip(prior_m, post_m):
                    c_pr = closed_form_isothermal_trajectory(
                        [0.0, th], tc, params=m_pr,
                        initial_hmw_fraction=hmw0_frac, initial_lmw_fraction=lmw0_frac, include_lmw=True
                    )["hmw_fraction"][-1] * 100.0
                    c_po = closed_form_isothermal_trajectory(
                        [0.0, th], tc, params=m_po,
                        initial_hmw_fraction=hmw0_frac, initial_lmw_fraction=lmw0_frac, include_lmw=True
                    )["hmw_fraction"][-1] * 100.0
                    pr_preds.append(float(c_pr))
                    po_preds.append(float(c_po))
                    log_sigs.append(float(m_pr.get("log_sigma_adj", 0.0)))

                mean_pr = float(np.mean(pr_preds))
                sd_pr = float(np.std(pr_preds, ddof=1))
                mean_po = float(np.mean(po_preds))
                sd_po = float(np.std(po_preds, ddof=1))
                mean_lsig = float(np.mean(log_sigs))

                # Prior PI95
                d_pr = max(0.0, mean_pr - hmw0_pct)
                sec_pr = float(np.asarray(heteroscedastic_sec_sd_fraction(mean_pr / 100.0, base_sd_pct=base_sec_sd_pct)).item()) * 100.0 * math.exp(mean_lsig)
                tot_sd_pr = math.sqrt(sd_pr ** 2 + (gamma_lot * d_pr) ** 2 + sec_pr ** 2)
                in_pr = (y_obs >= max(0.0, mean_pr - q_pt * tot_sd_pr)) and (y_obs <= mean_pr + q_pt * tot_sd_pr)

                # Posterior (adapted) PI95 — both uncalibrated (old 0.65 * gamma_lot, q_pt) and recalibrated on 60 cal lots
                d_po = max(0.0, mean_po - hmw0_pct)
                sec_po = float(np.asarray(heteroscedastic_sec_sd_fraction(mean_po / 100.0, base_sd_pct=base_sec_sd_pct)).item()) * 100.0 * math.exp(mean_lsig)
                tot_sd_po_old = math.sqrt(sd_po ** 2 + (0.65 * gamma_lot * d_po) ** 2 + sec_po ** 2)
                in_po_old = (y_obs >= max(0.0, mean_po - q_pt * tot_sd_po_old)) and (y_obs <= mean_po + q_pt * tot_sd_po_old)

                tot_sd_po_recal = math.sqrt(sd_po ** 2 + (0.25 * gamma_lot * d_po) ** 2 + sec_po ** 2)
                in_po_recal_all = (
                    (y_obs >= max(0.0, mean_po - q_adapt_all * tot_sd_po_recal))
                    and (y_obs <= mean_po + q_adapt_all * tot_sd_po_recal)
                )
                in_po_recal_mat = (
                    (y_obs >= max(0.0, mean_po - q_adapt_mat * tot_sd_po_recal))
                    and (y_obs <= mean_po + q_adapt_mat * tot_sd_po_recal)
                )

                # OOT upper-bound flags at 25C:
                # (a) Recalibrated adapted hybrid PI (overall calibration quantile q_adapt_all)
                # (b) Proper classic regression prediction interval (including slope uncertainty, one-sided 95%)
                # (c) Classic linear 0-3M fit + 3*SD_SEC upper limit with SD stated (SD=0.035 and SD=0.050)
                lin_pred_25 = int_25 + sl_25 * tm
                n_pts = 3.0
                t_bar = 4.0 / 3.0
                s_tt = 14.0 / 3.0
                se_pred_lin = base_sec_sd_pct * math.sqrt(1.0 + (1.0 / n_pts) + ((tm - t_bar) ** 2) / s_tt)
                lin_reg_pi95_upper_25 = lin_pred_25 + 1.6448536 * se_pred_lin
                lin_3sd_035_upper_25 = lin_pred_25 + 3.0 * 0.035
                lin_3sd_050_upper_25 = lin_pred_25 + 3.0 * 0.050

                flag_hybrid_recal_mat = bool(y_obs > mean_po + q_adapt_mat * tot_sd_po_recal)
                flag_hybrid_recal_all = bool(y_obs > mean_po + q_adapt_all * tot_sd_po_recal)
                flag_classic_reg_pi = bool(y_obs > lin_reg_pi95_upper_25)
                flag_classic_3sd_035 = bool(y_obs > lin_3sd_035_upper_25)
                flag_classic_3sd_050 = bool(y_obs > lin_3sd_050_upper_25)

                records.append(
                    {
                        "lot_id": lid_str,
                        "structural_group": truth_lots[lid_str]["structural_group"],
                        "structural_subgroup": truth_lots[lid_str]["structural_subgroup"],
                        "temperature_c": tc,
                        "time_months": tm,
                        "hmw_obs_pct": y_obs,
                        "prior_mean_pct": mean_pr,
                        "posterior_mean_pct": mean_po,
                        "prior_res_pp": y_obs - mean_pr,
                        "posterior_res_pp": y_obs - mean_po,
                        "inside_prior_pi95": bool(in_pr),
                        "inside_posterior_pi95_old": bool(in_po_old),
                        "inside_posterior_pi95_recal_overall": bool(in_po_recal_all),
                        "inside_posterior_pi95_recal_matched": bool(in_po_recal_mat),
                        "oot_flag_hybrid_recal_matched": flag_hybrid_recal_mat,
                        "oot_flag_hybrid_recal_overall": flag_hybrid_recal_all,
                        "oot_flag_classic_linear_reg_pi95": flag_classic_reg_pi,
                        "oot_flag_classic_linear_3sd_035": flag_classic_3sd_035,
                        "oot_flag_classic_linear_3sd_050": flag_classic_3sd_050,
                        "oot_flag_classic_linear_3sd": flag_classic_3sd_035,
                    }
                )

        df_r = pd.DataFrame(records)
        out_groups: Dict[str, Any] = {}
        for g_name, sub in (
            ("overall", df_r),
            ("matched", df_r[df_r["structural_group"] == "matched"]),
            ("mismatched", df_r[df_r["structural_group"] == "mismatched"]),
        ):
            n_obs = len(sub)
            k_pr = int(sub["inside_prior_pi95"].sum())
            k_po_old = int(sub["inside_posterior_pi95_old"].sum())
            k_po_recal_all = int(sub["inside_posterior_pi95_recal_overall"].sum())
            k_po_recal_mat = int(sub["inside_posterior_pi95_recal_matched"].sum())
            out_groups[g_name] = {
                "future_6m_9m_12m_observations_count": n_obs,
                "without_adaptation_hmw_rmse_pp": round(float(np.sqrt(np.mean(sub["prior_res_pp"] ** 2))), 5),
                "with_0_1_3m_adaptation_hmw_rmse_pp": round(float(np.sqrt(np.mean(sub["posterior_res_pp"] ** 2))), 5),
                "without_adaptation_pi95_coverage": evaluate_coverage_verdict(k_pr, n_obs),
                "with_0_1_3m_adaptation_pi95_coverage_before_recal": evaluate_coverage_verdict(k_po_old, n_obs),
                "with_0_1_3m_adaptation_pi95_coverage_recal_overall_cal": evaluate_coverage_verdict(k_po_recal_all, n_obs),
                "with_0_1_3m_adaptation_pi95_coverage_recal_matched_cal": evaluate_coverage_verdict(k_po_recal_mat, n_obs),
            }

        # Check #3: OOT operating characteristics on 25C pulls at 6M, 9M, 12M (and combined 6/9/12M)
        df_25 = df_r[df_r["temperature_c"] == 25.0].copy()
        oot_operating_characteristics: Dict[str, Any] = {}
        for m_label, sub_m in (
            ("6M_25C", df_25[df_25["time_months"] == 6.0]),
            ("9M_25C", df_25[df_25["time_months"] == 9.0]),
            ("12M_25C", df_25[df_25["time_months"] == 12.0]),
            ("all_6M_9M_12M_25C", df_25),
        ):
            if len(sub_m) == 0:
                continue
            oot_operating_characteristics[m_label] = {}
            for grp_label, sub_g in (
                ("matched_false_alarm_rate", sub_m[sub_m["structural_group"] == "matched"]),
                ("mismatched_all_detection_rate", sub_m[sub_m["structural_group"] == "mismatched"]),
                ("mismatched_ps80_only_detection_rate", sub_m[sub_m["structural_subgroup"] == "mismatched_ps80_only"]),
                ("mismatched_full_detection_rate", sub_m[sub_m["structural_subgroup"] == "mismatched_full"]),
            ):
                n_g = len(sub_g)
                if n_g == 0:
                    continue
                k_hyb_mat = int(sub_g["oot_flag_hybrid_recal_matched"].sum())
                k_hyb_all = int(sub_g["oot_flag_hybrid_recal_overall"].sum())
                k_reg = int(sub_g["oot_flag_classic_linear_reg_pi95"].sum())
                k_3sd_035 = int(sub_g["oot_flag_classic_linear_3sd_035"].sum())
                k_3sd_050 = int(sub_g["oot_flag_classic_linear_3sd_050"].sum())

                l_hm, u_hm = clopper_pearson_interval(k_hyb_mat, n_g)
                l_ha, u_ha = clopper_pearson_interval(k_hyb_all, n_g)
                l_reg, u_reg = clopper_pearson_interval(k_reg, n_g)
                l_35, u_35 = clopper_pearson_interval(k_3sd_035, n_g)
                l_50, u_50 = clopper_pearson_interval(k_3sd_050, n_g)

                oot_operating_characteristics[m_label][grp_label] = {
                    "n": n_g,
                    "recalibrated_adapted_hybrid_pi_overall_cal": {
                        "k": k_hyb_all,
                        "k_over_n": f"{k_hyb_all}/{n_g}",
                        "rate_pct": round(100.0 * k_hyb_all / n_g, 2),
                        "clopper_pearson_95_ci_pct": [round(100.0 * l_ha, 2), round(100.0 * u_ha, 2)],
                    },
                    "recalibrated_adapted_hybrid_pi_matched_cal": {
                        "k": k_hyb_mat,
                        "k_over_n": f"{k_hyb_mat}/{n_g}",
                        "rate_pct": round(100.0 * k_hyb_mat / n_g, 2),
                        "clopper_pearson_95_ci_pct": [round(100.0 * l_hm, 2), round(100.0 * u_hm, 2)],
                    },
                    "classic_linear_reg_pi95_with_slope_uncertainty": {
                        "k": k_reg,
                        "k_over_n": f"{k_reg}/{n_g}",
                        "rate_pct": round(100.0 * k_reg / n_g, 2),
                        "clopper_pearson_95_ci_pct": [round(100.0 * l_reg, 2), round(100.0 * u_reg, 2)],
                    },
                    "classic_linear_0_3m_plus_3sd_sd_035": {
                        "k": k_3sd_035,
                        "k_over_n": f"{k_3sd_035}/{n_g}",
                        "rate_pct": round(100.0 * k_3sd_035 / n_g, 2),
                        "clopper_pearson_95_ci_pct": [round(100.0 * l_35, 2), round(100.0 * u_35, 2)],
                    },
                    "classic_linear_0_3m_plus_3sd_sd_050": {
                        "k": k_3sd_050,
                        "k_over_n": f"{k_3sd_050}/{n_g}",
                        "rate_pct": round(100.0 * k_3sd_050 / n_g, 2),
                        "clopper_pearson_95_ci_pct": [round(100.0 * l_50, 2), round(100.0 * u_50, 2)],
                    },
                    # Backward-compatible key matching SD=0.035
                    "classic_linear_0_3m_plus_3sd": {
                        "k": k_3sd_035,
                        "k_over_n": f"{k_3sd_035}/{n_g}",
                        "rate_pct": round(100.0 * k_3sd_035 / n_g, 2),
                        "clopper_pearson_95_ci_pct": [round(100.0 * l_35, 2), round(100.0 * u_35, 2)],
                    },
                }

        return {
            "coverage_and_rmse_by_group": out_groups,
            "oot_25c_operating_characteristics": oot_operating_characteristics,
        }

    return {
        "validation_lots_6m_9m_12m_forecasts": _eval_split("val"),
        "test_lots_6m_9m_12m_forecasts": _eval_split("test"),
    }


def evaluate_test_coverage_and_recovery(
    ensemble_weights: List[Dict[str, np.ndarray]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    calibration_bundle: Dict[str, Any],
    obs_csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_observations.csv",
    truth_json_path: Path | str = DEFAULT_SYNTH_DIR / "_ground_truth_do_not_read_in_train.json",
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Evaluates parameter/rate recovery and exact Clopper-Pearson 95% binomial coverage on the
    100 held-out test lots (`split="test"`):
    - Uses **per-lot simultaneous coverage** (`observation_pi_per_lot_simultaneous_primary_gate`)
      for pass/fail statements, and marks **per-observation coverage** as descriptive (since
      observations within a lot are correlated).
    - Reports conditional observation-PI coverage stratified by observable `temperature_c x time_months`
      cells on the 100 test lots, comparing standard vs. HMW-level-scaled intervals.
    """
    if cfg is None:
        cfg = load_hybrid_config()

    df_test = load_observations_for_training(obs_csv_path, allowed_splits=("test",))
    truth_data = load_ground_truth_for_verification_only(truth_json_path)
    lots_truth = truth_data["lots"]

    pred_out = predict_lot_intervals_and_trajectories(
        df_test,
        ensemble_weights,
        feature_mean,
        feature_std,
        calibration_bundle,
        cfg=cfg,
    )
    param_summaries = pred_out["lot_param_summaries"]
    obs_df = pred_out["observations_df"].copy()

    obs_df["structural_group"] = obs_df["lot_id"].map(lambda lid: lots_truth[lid]["structural_group"])
    obs_df["structural_subgroup"] = obs_df["lot_id"].map(lambda lid: lots_truth[lid]["structural_subgroup"])
    obs_df["inside_pointwise_pi95"] = (
        (obs_df["hmw_obs_pct"] >= obs_df["obs_pi95_pointwise_lower_pct"])
        & (obs_df["hmw_obs_pct"] <= obs_df["obs_pi95_pointwise_upper_pct"])
    )
    obs_df["inside_simultaneous_pi95"] = (
        (obs_df["hmw_obs_pct"] >= obs_df["obs_pi95_simultaneous_lower_pct"])
        & (obs_df["hmw_obs_pct"] <= obs_df["obs_pi95_simultaneous_upper_pct"])
    )
    obs_df["inside_level_scaled_pointwise_pi95"] = (
        (obs_df["hmw_obs_pct"] >= obs_df["obs_pi95_level_scaled_pointwise_lower_pct"])
        & (obs_df["hmw_obs_pct"] <= obs_df["obs_pi95_level_scaled_pointwise_upper_pct"])
    )
    obs_df["inside_level_scaled_simultaneous_pi95"] = (
        (obs_df["hmw_obs_pct"] >= obs_df["obs_pi95_level_scaled_simultaneous_lower_pct"])
        & (obs_df["hmw_obs_pct"] <= obs_df["obs_pi95_level_scaled_simultaneous_upper_pct"])
    )

    test_lot_ids = sorted(param_summaries.keys())
    groups_map = {
        "overall": test_lot_ids,
        "matched": [lid for lid in test_lot_ids if lots_truth[lid]["structural_group"] == "matched"],
        "mismatched": [lid for lid in test_lot_ids if lots_truth[lid]["structural_group"] == "mismatched"],
        "mismatched_ps80_only": [
            lid for lid in test_lot_ids if lots_truth[lid]["structural_subgroup"] == "mismatched_ps80_only"
        ],
        "mismatched_full": [
            lid for lid in test_lot_ids if lots_truth[lid]["structural_subgroup"] == "mismatched_full"
        ],
    }

    coverage_by_group: Dict[str, Any] = {}
    for g_name, g_lids in groups_map.items():
        g_obs = obs_df[obs_df["lot_id"].isin(g_lids)]
        n_lots = len(g_lids)
        n_obs = len(g_obs)

        metrics_dict: Dict[str, Any] = {}
        for m_key in IDENTIFIABLE_METRIC_KEYS:
            t_key = TRUTH_KEY_MAP[m_key]
            hits = 0
            abs_errs = []
            rel_errs = []
            for lid in g_lids:
                ps = param_summaries[lid]
                true_v = float(lots_truth[lid][t_key])
                mu_v = float(ps[f"{m_key}_mean"])
                l95 = float(ps[f"{m_key}_cal_l95"])
                u95 = float(ps[f"{m_key}_cal_u95"])
                if l95 <= true_v <= u95:
                    hits += 1
                abs_errs.append(abs(mu_v - true_v))
                rel_errs.append(abs(mu_v - true_v) / max(abs(true_v), 1e-12))

            cov_info = evaluate_coverage_verdict(hits, n_lots)
            cov_info["mae"] = round(float(np.mean(abs_errs)), 5)
            cov_info["mape_pct"] = round(float(np.mean(rel_errs)) * 100.0, 2)
            cov_info["observability_note"] = SYNTHETIC_ONLY_PARAM_CALIBRATION_NOTE
            metrics_dict[m_key] = cov_info

        # Primary pass/fail gate: simultaneous lot-level coverage (independent across lots)
        lot_all_in_sim = g_obs.groupby("lot_id")["inside_simultaneous_pi95"].all()
        k_lot_sim = int(lot_all_in_sim.sum())
        lot_sim_cov = evaluate_coverage_verdict(k_lot_sim, n_lots)
        lot_sim_cov["role"] = "PRIMARY_PASS_FAIL_GATE (independent lot-level binomial trials)"

        # Descriptive per-observation coverage (correlations within a lot exist)
        k_obs_pt = int(g_obs["inside_pointwise_pi95"].sum())
        obs_pt_cov = evaluate_coverage_verdict(k_obs_pt, n_obs)
        obs_pt_cov["role"] = "DESCRIPTIVE_ONLY (observations within a lot are correlated; use lot-level result for pass/fail)"

        lot_all_in_lvl_sim = g_obs.groupby("lot_id")["inside_level_scaled_simultaneous_pi95"].all()
        lot_lvl_sim_cov = evaluate_coverage_verdict(int(lot_all_in_lvl_sim.sum()), n_lots)
        obs_lvl_pt_cov = evaluate_coverage_verdict(int(g_obs["inside_level_scaled_pointwise_pi95"].sum()), n_obs)

        lot_all_in_pt = g_obs.groupby("lot_id")["inside_pointwise_pi95"].all()
        res_pp = g_obs["residual_pp"].to_numpy(dtype=np.float64)

        coverage_by_group[g_name] = {
            "lots_count": n_lots,
            "observations_count": n_obs,
            "hmw_rmse_pp": round(float(np.sqrt(np.mean(res_pp ** 2))), 5),
            "hmw_mae_pp": round(float(np.mean(np.abs(res_pp))), 5),
            "identifiable_quantities_note": SYNTHETIC_ONLY_PARAM_CALIBRATION_NOTE,
            "identifiable_quantities": metrics_dict,
            "observation_pi_per_lot_simultaneous": lot_sim_cov,
            "observation_pi_per_observation_pointwise": obs_pt_cov,
            "observation_pi_level_scaled_per_lot_simultaneous": lot_lvl_sim_cov,
            "observation_pi_level_scaled_per_observation_descriptive": obs_lvl_pt_cov,
            "lots_with_all_observations_inside_pointwise_pi": {
                "k": int(lot_all_in_pt.sum()),
                "n": n_lots,
                "k_over_n": f"{int(lot_all_in_pt.sum())}/{n_lots}",
                "observed_pct": round(float(lot_all_in_pt.mean()) * 100.0, 2),
            },
        }

    # Follow-up #7: Observable strata coverage table (temperature_c x time_months) on the 100 test lots
    strata_rows: List[Dict[str, Any]] = []
    for (tc, tm), cell_df in obs_df.groupby(["temperature_c", "time_months"]):
        if tm == 0.0:
            continue
        n_cell = len(cell_df)
        k_std = int(cell_df["inside_pointwise_pi95"].sum())
        k_lvl = int(cell_df["inside_level_scaled_pointwise_pi95"].sum())
        mat_cell = cell_df[cell_df["structural_group"] == "matched"]
        mis_cell = cell_df[cell_df["structural_group"] == "mismatched"]
        strata_rows.append(
            {
                "temperature_c": float(tc),
                "time_months": float(tm),
                "n_lots_in_stratum": n_cell,
                "mean_obs_hmw_pct": round(float(cell_df["hmw_obs_pct"].mean()), 3),
                "standard_pi95_coverage_pct": round(100.0 * k_std / n_cell, 2),
                "standard_pi95_k_over_n": f"{k_std}/{n_cell}",
                "level_scaled_pi95_coverage_pct": round(100.0 * k_lvl / n_cell, 2),
                "level_scaled_pi95_k_over_n": f"{k_lvl}/{n_cell}",
                "matched_standard_pi95_coverage_pct": round(100.0 * float(mat_cell["inside_pointwise_pi95"].mean()), 2) if len(mat_cell) else 100.0,
                "mismatched_standard_pi95_coverage_pct": round(100.0 * float(mis_cell["inside_pointwise_pi95"].mean()), 2) if len(mis_cell) else 100.0,
                "mismatched_level_scaled_pi95_coverage_pct": round(100.0 * float(mis_cell["inside_level_scaled_pointwise_pi95"].mean()), 2) if len(mis_cell) else 100.0,
            }
        )

    # Prior-driven non-identifiable parameters spread + dH_agg vs dS_agg correlation
    test_lot_df = (
        df_test.groupby("lot_id", as_index=False)
        .first()
        .sort_values("lot_id")
        .reset_index(drop=True)
    )
    x_test, _, _ = extract_feature_matrix(
        test_lot_df, feature_mean=feature_mean, feature_std=feature_std
    )
    member_preds_test = [
        predict_params_from_npz_member(x_test, mw, cfg=cfg) for mw in ensemble_weights
    ]

    prior_driven_summary: Dict[str, Any] = {
        "note": (
            "PRIOR-DRIVEN / NON-IDENTIFIABLE FROM SEC HMW DATA ALONE: "
            "dH_agg_ref, K_u(5C), and dH_u trade off inside k_eff(T) = f_U(T) * k_agg(T). "
            "Reported for diagnostic transparency only; never gated."
        ),
        "parameters": {},
    }
    for p_key in PRIOR_DRIVEN_NONIDENTIFIABLE_KEYS:
        all_vals = np.concatenate([mp[p_key] for mp in member_preds_test])  # type: ignore
        within_lot_sds = [
            float(param_summaries[lid][f"{p_key}_std"]) for lid in test_lot_ids
        ]
        prior_driven_summary["parameters"][p_key] = {
            "ensemble_mean": round(float(np.mean(all_vals)), 5),
            "ensemble_median": round(float(np.median(all_vals)), 5),
            "ensemble_p025": round(float(np.quantile(all_vals, 0.025)), 5),
            "ensemble_p975": round(float(np.quantile(all_vals, 0.975)), 5),
            "mean_within_lot_ensemble_sd": round(float(np.mean(within_lot_sds)), 5),
        }

    dh_ds_corrs = []
    for l_i in range(len(test_lot_ids)):
        dh_vec = np.array([mp["dH_agg_ref_kj_mol"][l_i] for mp in member_preds_test], dtype=np.float64)
        ds_vec = np.array([mp["dS_agg_ref_j_mol_k"][l_i] for mp in member_preds_test], dtype=np.float64)
        if np.std(dh_vec) > 1e-9 and np.std(ds_vec) > 1e-9:
            dh_ds_corrs.append(float(np.corrcoef(dh_vec, ds_vec)[0, 1]))
    prior_driven_summary["dH_agg_vs_dS_agg_ensemble_correlation"] = {
        "mean_pearson_r": round(float(np.mean(dh_ds_corrs)), 4),
        "median_pearson_r": round(float(np.median(dh_ds_corrs)), 4),
        "min_pearson_r": round(float(np.min(dh_ds_corrs)), 4),
        "max_pearson_r": round(float(np.max(dh_ds_corrs)), 4),
        "explanation": (
            "Strong positive enthalpy-entropy compensation (r ~ +0.99) arises physically because SEC data "
            "tightly constrain dG_agg_ref = dH_agg_ref - T_ref * dS_agg_ref at 5 °C while leaving the "
            "partitioning between dH_agg_ref and dH_u partially collinear."
        ),
    }

    return {
        "coverage_by_group": coverage_by_group,
        "observable_strata_coverage_temp_x_time": strata_rows,
        "prior_driven_nonidentifiable_diagnostics": prior_driven_summary,
    }


def evaluate_excursion_verdicts_with_ensemble(
    ensemble_weights: List[Dict[str, np.ndarray]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    calibration_bundle: Dict[str, Any],
    obs_csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_observations.csv",
    exc_csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_excursions.csv",
    truth_json_path: Path | str = DEFAULT_SYNTH_DIR / "_ground_truth_do_not_read_in_train.json",
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Follow-up #5:
    - Calibrates excursion `dHMW` intervals `[L95, U95]` ONLY on validation-lot excursions (`split="val"`, `n=48`).
    - Evaluates the verdict confusion matrix, false-reassurance rate, and false-alarm rate
      (with exact Clopper-Pearson 95% confidence intervals / upper bounds) strictly on the
      **held-out test-lot excursions** (`split="test"`, `n=72`), broken down by `overall`,
      `matched`, and `mismatched`.
    """
    if cfg is None:
        cfg = load_hybrid_config()
    sd_sec = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    gate_pp = compute_excursion_detection_gate_pct(sd_sec, 2.0)

    df_all_obs = load_observations_for_training(obs_csv_path, allowed_splits=("train", "val", "test"))
    lot_cov_df = df_all_obs.groupby("lot_id", as_index=False).first()
    lot_cov_map = {row["lot_id"]: row.to_dict() for _, row in lot_cov_df.iterrows()}

    df_exc_all = pd.read_csv(exc_csv_path)
    # Strictly evaluate confusion matrix on held-out TEST excursions only (split == 'test')
    df_exc_test = df_exc_all[df_exc_all["split"] == "test"].copy()
    truth_data = load_ground_truth_for_verification_only(truth_json_path)
    exc_truth = truth_data["excursions"]

    exc_cal = calibration_bundle["excursion_calibration_from_val_lots"]
    lot_rel_sd_floor = float(exc_cal["lot_rel_sd_floor"])
    q_exc_mult = float(exc_cal["conformal_multiplier_q95_val_excursions"])

    exc_records: List[Dict[str, Any]] = []
    for _, erow in df_exc_test.iterrows():
        exc_id = str(erow["excursion_id"])
        lid = str(erow["lot_id"])
        cov = lot_cov_map[lid]
        x_norm, _, _ = extract_feature_matrix(cov, feature_mean=feature_mean, feature_std=feature_std)

        peak_c = float(erow["peak_temp_c"])
        dur_h = float(erow["duration_hours"])
        ramp_h = float(erow["ramp_hours"])
        n_rep = int(erow["n_repeats"])
        base_c = float(erow["baseline_temp_c"])
        tot_h = float(erow["total_window_hours"])

        tk, tc = build_excursion_profile_knots(
            baseline_temp_c=base_c,
            excursion_temp_c=peak_c,
            excursion_duration_hours=dur_h,
            ramp_hours=ramp_h,
            n_spikes=n_rep,
            total_hours=tot_h,
            start_hour=12.0,
        )
        t_grid = np.linspace(0.0, tot_h, max(61, int(tot_h // 2) + 1))
        hmw0 = float(cov["initial_hmw_pct"]) / 100.0
        lmw0 = float(cov["initial_lmw_pct"]) / 100.0

        member_dhmw_pp = []
        for mw in ensemble_weights:
            mp = predict_params_from_npz_member(x_norm[0], mw, cfg=cfg)
            traj = integrate_pathway_rk4(
                time_grid_h=t_grid,
                time_knots_h=tk,
                temp_knots_c=tc,
                params=mp,
                initial_hmw_fraction=hmw0,
                initial_lmw_fraction=lmw0,
                include_lmw=True,
            )
            d_pp = float(traj["hmw_fraction"][-1] - traj["hmw_fraction"][0]) * 100.0
            member_dhmw_pp.append(d_pp)

        arr_pp = np.asarray(member_dhmw_pp, dtype=np.float64)
        mean_pp = float(np.mean(arr_pp))
        ens_sd_pp = float(np.std(arr_pp, ddof=1))

        tot_sd_pp = math.sqrt(ens_sd_pp ** 2 + (lot_rel_sd_floor * mean_pp) ** 2)
        l95_pp = max(0.0, mean_pp - q_exc_mult * tot_sd_pp)
        u95_pp = mean_pp + q_exc_mult * tot_sd_pp

        v_info = classify_excursion_verdict(l95_pp, u95_pp, sec_repeatability_sd_pct=sd_sec)
        et = exc_truth[exc_id]
        true_pp = float(et["delta_hmw_true_pp"])
        true_above_gate = bool(true_pp >= gate_pp)

        exc_records.append(
            {
                "excursion_id": exc_id,
                "lot_id": lid,
                "structural_group": et["structural_group"],
                "structural_subgroup": et["structural_subgroup"],
                "scenario_label": str(erow["scenario_label"]),
                "true_dhmw_pp": true_pp,
                "true_class": "true_above_gate" if true_above_gate else "true_below_gate",
                "pred_mean_dhmw_pp": mean_pp,
                "pred_l95_dhmw_pp": l95_pp,
                "pred_u95_dhmw_pp": u95_pp,
                "inside_dhmw_interval": bool(l95_pp <= true_pp <= u95_pp),
                "verdict": v_info["verdict"],
            }
        )

    df_res = pd.DataFrame(exc_records)

    def _summarize_exc_group(sub: pd.DataFrame) -> Dict[str, Any]:
        below = sub[sub["true_class"] == "true_below_gate"]
        above = sub[sub["true_class"] == "true_above_gate"]
        n_below = len(below)
        n_above = len(above)

        cm = {
            "true_below_gate": {
                "total": int(n_below),
                "pred_no_detectable_change": int((below["verdict"] == "no detectable change").sum()),
                "pred_inconclusive": int((below["verdict"] == "inconclusive").sum()),
                "pred_detectable_change_likely": int((below["verdict"] == "detectable change likely").sum()),
            },
            "true_above_gate": {
                "total": int(n_above),
                "pred_no_detectable_change": int((above["verdict"] == "no detectable change").sum()),
                "pred_inconclusive": int((above["verdict"] == "inconclusive").sum()),
                "pred_detectable_change_likely": int((above["verdict"] == "detectable change likely").sum()),
            },
        }
        false_reassurance_count = cm["true_above_gate"]["pred_no_detectable_change"]
        false_alarm_count = cm["true_below_gate"]["pred_detectable_change_likely"]

        fr_l95, fr_u95 = clopper_pearson_interval(false_reassurance_count, max(n_above, 1), alpha=0.05)
        fa_l95, fa_u95 = clopper_pearson_interval(false_alarm_count, max(n_below, 1), alpha=0.05)
        k_cov = int(sub["inside_dhmw_interval"].sum())

        return {
            "total_test_excursions": int(len(sub)),
            "dhmw_interval_coverage": evaluate_coverage_verdict(k_cov, len(sub)),
            "confusion_matrix": cm,
            "false_reassurance_count": false_reassurance_count,
            "false_reassurance_k_over_n": f"{false_reassurance_count}/{n_above}",
            "false_reassurance_rate": round(false_reassurance_count / max(n_above, 1), 4),
            "false_reassurance_clopper_pearson_95_ci": [round(fr_l95, 4), round(fr_u95, 4)],
            "false_reassurance_clopper_pearson_u95_pct": round(fr_u95 * 100.0, 2),
            "false_alarm_count": false_alarm_count,
            "false_alarm_k_over_n": f"{false_alarm_count}/{n_below}",
            "false_alarm_rate": round(false_alarm_count / max(n_below, 1), 4),
            "false_alarm_clopper_pearson_95_ci": [round(fa_l95, 4), round(fa_u95, 4)],
            "false_alarm_clopper_pearson_u95_pct": round(fa_u95 * 100.0, 2),
            "inconclusive_count": int((sub["verdict"] == "inconclusive").sum()),
            "inconclusive_rate": round(float((sub["verdict"] == "inconclusive").mean()), 4),
        }

    # Evaluate the 6 nominal benchmark scenarios with the trained ensemble (at nominal formulation covariates)
    nom_cov = {
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
    x_nom, _, _ = extract_feature_matrix(nom_cov, feature_mean=feature_mean, feature_std=feature_std)
    nom_phase1 = compute_amendment_ab_summary()["excursion_outcomes"]
    benchmark_specs = {
        "36h_at_28C": (28.0, 36.0),
        "2h_at_28C": (28.0, 2.0),
        "48h_at_38C": (38.0, 48.0),
        "72h_at_40C": (40.0, 72.0),
        "7d_at_40C": (40.0, 168.0),
        "14d_at_40C": (40.0, 336.0),
    }
    benchmark_comparison: Dict[str, Any] = {}
    m0 = 0.9938
    for b_name, (b_temp_c, b_dur_h) in benchmark_specs.items():
        member_pp = []
        for mw in ensemble_weights:
            mp = predict_params_from_npz_member(x_nom[0], mw, cfg=cfg)
            r_sp = lumry_eyring_rates(celsius_to_kelvin(b_temp_c), **mp)
            k_sp = float(np.asarray(r_sp["k_eff_fraction_per_h"]).item())
            d_pp = m0 * (1.0 - math.exp(-k_sp * b_dur_h)) * 100.0
            member_pp.append(d_pp)
        arr_b = np.asarray(member_pp, dtype=np.float64)
        mean_b = float(np.mean(arr_b))
        sd_b = float(np.std(arr_b, ddof=1))
        tot_sd_b = math.sqrt(sd_b ** 2 + (lot_rel_sd_floor * mean_b) ** 2)
        l95_b = max(0.0, mean_b - q_exc_mult * tot_sd_b)
        u95_b = mean_b + q_exc_mult * tot_sd_b
        v_ens = classify_excursion_verdict(l95_b, u95_b, sec_repeatability_sd_pct=sd_sec)["verdict"]
        v_nom = nom_phase1[b_name]["verdict"]
        benchmark_comparison[b_name] = {
            "nominal_dhmw_pp": round(float(nom_phase1[b_name]["delta_hmw_pp"]), 6),
            "nominal_placeholder_verdict": v_nom,
            "ensemble_mean_dhmw_pp": round(mean_b, 6),
            "ensemble_sd_dhmw_pp": round(sd_b, 6),
            "ensemble_l95_dhmw_pp": round(l95_b, 6),
            "ensemble_u95_dhmw_pp": round(u95_b, 6),
            "ensemble_verdict": v_ens,
            "differs_from_nominal": bool(v_ens != v_nom),
        }

    return {
        "evaluation_subset": "test_lots_excursions_only (split='test', n=80; calibrated on val_lots_excursions, n=40)",
        "detection_gate_2sigma_pp": round(gate_pp, 5),
        "by_group": {
            "overall": _summarize_exc_group(df_res),
            "matched": _summarize_exc_group(df_res[df_res["structural_group"] == "matched"]),
            "mismatched": _summarize_exc_group(df_res[df_res["structural_group"] == "mismatched"]),
        },
        "nominal_benchmark_comparison": benchmark_comparison,
    }


def _train_tuned_blackbox_mlp(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    seed: int = 20261023,
) -> Dict[str, Any]:
    """
    Follow-up #2:
    Properly tunes a PyTorch Black-Box MLP predicting `% HMW` directly from standardized
    `[LOT_COVARIATE_COLUMNS, pH^2, temperature_c, time_months]` (13 inputs, same `pH` and `pH^2`
    features as the Hybrid ANN and Ridge) using early stopping on the 40 validation lots across
    multiple architectures and learning rates.
    """
    def _build_xy(df_in: pd.DataFrame, mean_vec: np.ndarray | None = None, std_vec: np.ndarray | None = None):
        x_cov, _, _ = extract_feature_matrix(df_in, feature_mean=np.zeros(11), feature_std=np.ones(11))
        # Undo dummy standardization to get raw 11 features (10 covariates + pH^2)
        x_base = df_in[list(LOT_COVARIATE_COLUMNS)].to_numpy(dtype=np.float64)
        ph_sq = x_base[:, 0:1] ** 2
        t_c = df_in[["temperature_c"]].to_numpy(dtype=np.float64)
        t_m = df_in[["time_months"]].to_numpy(dtype=np.float64)
        x_raw = np.hstack([x_base, ph_sq, t_c, t_m])
        y = df_in["hmw_obs_pct"].to_numpy(dtype=np.float64)
        if mean_vec is None:
            mean_vec = np.mean(x_raw, axis=0)
        if std_vec is None:
            std_vec = np.where(np.std(x_raw, axis=0) < 1e-8, 1.0, np.std(x_raw, axis=0))
        x_norm = (x_raw - mean_vec) / std_vec
        return x_norm, y, mean_vec, std_vec

    x_tr, y_tr, bb_mean, bb_std = _build_xy(df_train)
    x_val, y_val, _, _ = _build_xy(df_val, bb_mean, bb_std)

    x_tr_t = torch.tensor(x_tr, dtype=torch.float64)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float64).unsqueeze(1)
    x_val_t = torch.tensor(x_val, dtype=torch.float64)
    y_val_t = torch.tensor(y_val, dtype=torch.float64).unsqueeze(1)

    candidates = [
        {"hidden_dims": (32, 32), "lr": 0.010, "weight_decay": 1e-4},
        {"hidden_dims": (64, 64), "lr": 0.008, "weight_decay": 1e-4},
        {"hidden_dims": (64, 32), "lr": 0.012, "weight_decay": 5e-4},
    ]

    best_overall = None
    search_log = []
    for c_idx, cand in enumerate(candidates):
        torch.manual_seed(seed + c_idx * 100)
        h1, h2 = cand["hidden_dims"]
        net = nn.Sequential(
            nn.Linear(x_tr.shape[1], h1, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(h1, h2, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(h2, 1, dtype=torch.float64),
        )
        opt = torch.optim.AdamW(net.parameters(), lr=cand["lr"], weight_decay=cand["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=cand["lr"] * 0.05)

        best_v_rmse = float("inf")
        best_state = None
        best_ep = 0
        for ep in range(1, 301):
            net.train()
            opt.zero_grad()
            pred_tr = net(x_tr_t)
            loss = torch.mean((pred_tr - y_tr_t) ** 2)
            loss.backward()
            opt.step()
            sched.step()

            net.eval()
            with torch.no_grad():
                pred_v = net(x_val_t)
                v_rmse = float(torch.sqrt(torch.mean((pred_v - y_val_t) ** 2)).item())
            if v_rmse < best_v_rmse - 1e-5:
                best_v_rmse = v_rmse
                best_state = copy.deepcopy(net.state_dict())
                best_ep = ep

        search_log.append(
            {
                "hidden_dims": list(cand["hidden_dims"]),
                "lr": cand["lr"],
                "weight_decay": cand["weight_decay"],
                "best_epoch": best_ep,
                "best_val_rmse_pp": round(best_v_rmse, 5),
            }
        )
        if best_overall is None or best_v_rmse < best_overall["best_val_rmse"]:
            net.load_state_dict(best_state)
            best_overall = {
                "net": net,
                "config": cand,
                "best_epoch": best_ep,
                "best_val_rmse": best_v_rmse,
                "bb_mean": bb_mean,
                "bb_std": bb_std,
                "w1": net[0].weight.detach().cpu().numpy().copy(),
                "b1": net[0].bias.detach().cpu().numpy().copy(),
                "w2": net[2].weight.detach().cpu().numpy().copy(),
                "b2": net[2].bias.detach().cpu().numpy().copy(),
                "w3": net[4].weight.detach().cpu().numpy().copy(),
                "b3": net[4].bias.detach().cpu().numpy().copy(),
            }

    assert best_overall is not None
    best_overall["search_log"] = search_log
    return best_overall


def _predict_blackbox_mlp_numpy(
    df_in: pd.DataFrame,
    bb_model: Dict[str, Any],
) -> np.ndarray:
    x_base = df_in[list(LOT_COVARIATE_COLUMNS)].to_numpy(dtype=np.float64)
    ph_sq = x_base[:, 0:1] ** 2
    t_c = df_in[["temperature_c"]].to_numpy(dtype=np.float64)
    t_m = df_in[["time_months"]].to_numpy(dtype=np.float64)
    x_raw = np.hstack([x_base, ph_sq, t_c, t_m])
    x_norm = (x_raw - bb_model["bb_mean"]) / bb_model["bb_std"]

    def _silu(z: np.ndarray) -> np.ndarray:
        return z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    h1 = _silu(x_norm @ bb_model["w1"].T + bb_model["b1"])
    h2 = _silu(h1 @ bb_model["w2"].T + bb_model["b2"])
    out = h2 @ bb_model["w3"].T + bb_model["b3"]
    return out[:, 0]


def evaluate_physical_validity_extrapolation_and_15c(
    obs_csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_observations.csv",
    truth_json_path: Path | str = DEFAULT_SYNTH_DIR / "_ground_truth_do_not_read_in_train.json",
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Follow-up #8 Deliverable (Physical-Validity Test):
    1. Trains both the **Hybrid Lumry-Eyring + RK4 model** and the **Tuned Black-Box MLP**
       exclusively on stability pulls up to `3 months` (`time_months <= 3.0`) on the 100 training
       lots (with early stopping on `<= 3M` pulls of the 40 validation lots).
    2. Predicts later pulls (`6, 9, 12, 18, 24M`) on the 100 held-out test lots and reports
       future-pull extrapolation HMW RMSE (`pp`) for both models.
    3. Predicts dense trajectories across `0..24M` at `5 °C, 15 °C (unseen intermediate temperature),
       25 °C, and 40 °C` on all 100 test lots and counts physical violations:
       - **Purity-increase / negative-degradation violations:** time steps where predicted `% HMW`
         decreases over time (`HMW(t_{k+1}) < HMW(t_k) - 1e-9`), or where predicted `t=0` HMW
         violates the known initial condition `HMW(0) = HMW_0`.
       - **Non-monotone temperature-rate violations:** lot/time pairs where predicted degradation
         at `15 °C` is outside `[HMW(5°C), HMW(25°C)]` or `HMW(25°C) > HMW(40°C)`.
    """
    if cfg is None:
        cfg = load_hybrid_config()
    base_sec_sd_pct = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    cex_sd_pct = float(cfg["assay_and_decision_thresholds"]["cex_acidic_repeatability_sd_pct"])

    df_train_full = load_observations_for_training(obs_csv_path, allowed_splits=("train",))
    df_val_full = load_observations_for_training(obs_csv_path, allowed_splits=("val",))
    df_test_full = load_observations_for_training(obs_csv_path, allowed_splits=("test",))
    lots_truth = load_ground_truth_for_verification_only(truth_json_path)["lots"]

    df_train_3m = df_train_full[df_train_full["time_months"] <= 3.0].copy()
    df_val_3m = df_val_full[df_val_full["time_months"] <= 3.0].copy()

    # 1. Train 5-member Hybrid Ensemble on <= 3M pulls only
    tr_3m_data = _build_split_lot_tensors(df_train_3m, base_sec_sd_pct=base_sec_sd_pct, cex_sd_pct=cex_sd_pct)
    val_3m_data = _build_split_lot_tensors(
        df_val_3m,
        feature_mean=tr_3m_data["feature_mean"],
        feature_std=tr_3m_data["feature_std"],
        base_sec_sd_pct=base_sec_sd_pct,
        cex_sd_pct=cex_sd_pct,
    )
    hyb_3m_weights: List[Dict[str, np.ndarray]] = []
    for m_i in range(5):
        model_3m = BoundedKineticMLP(
            input_dim=tr_3m_data["x_norm"].shape[1],
            hidden_dims=(32, 32),
            cfg=cfg,
            seed=20261023 + 500 + m_i,
        )
        opt = torch.optim.AdamW(model_3m.parameters(), lr=0.015, weight_decay=2e-4)
        best_v = float("inf")
        best_w = model_3m.export_numpy_weights()
        for _ in range(120):
            model_3m.train()
            opt.zero_grad()
            out_tr = compute_split_loss_and_predictions_torch(model_3m, tr_3m_data, smoothness_lambda=0.02)
            out_tr["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model_3m.parameters(), 2.0)
            opt.step()
            model_3m.eval()
            with torch.no_grad():
                v_l = float(compute_split_loss_and_predictions_torch(model_3m, val_3m_data)["loss"].item())
            if v_l < best_v:
                best_v = v_l
                best_w = model_3m.export_numpy_weights()
        hyb_3m_weights.append(best_w)

    # 2. Train Tuned Black-Box MLP on <= 3M pulls only
    bb_3m_model = _train_tuned_blackbox_mlp(df_train_3m, df_val_3m, seed=20261023 + 900)

    # 3. Evaluate extrapolation RMSE on test pulls > 3M (6, 9, 12, 18, 24M)
    df_test_late = df_test_full[df_test_full["time_months"] > 3.0].copy()
    df_test_late["structural_group"] = df_test_late["lot_id"].map(lambda lid: lots_truth[lid]["structural_group"])

    hyb_late_eval = evaluate_member_or_ensemble_numpy(
        hyb_3m_weights, tr_3m_data["feature_mean"], tr_3m_data["feature_std"], df_test_late, cfg=cfg
    )["observations_df"]
    hyb_late_eval["structural_group"] = hyb_late_eval["lot_id"].map(lambda lid: lots_truth[lid]["structural_group"])

    bb_late_preds = _predict_blackbox_mlp_numpy(df_test_late, bb_3m_model)
    bb_late_res = df_test_late["hmw_obs_pct"].to_numpy(dtype=np.float64) - bb_late_preds
    is_mat_late = (df_test_late["structural_group"] == "matched").to_numpy(dtype=bool)

    # 4. Physical violation audit across all 100 test lots on dense time grid [0, 1, 2, 3, 6, 9, 12, 18, 24M]
    # and temperatures [5, 15, 25, 40C]
    test_lot_df = df_test_full.groupby("lot_id", as_index=False).first().sort_values("lot_id")
    time_grid_m = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0], dtype=np.float64)
    temps_audit = [5.0, 15.0, 25.0, 40.0]

    hyb_neg_rate_steps = 0
    hyb_total_time_steps = 0
    hyb_temp_order_violations = 0
    hyb_total_temp_comparisons = 0

    bb_neg_rate_steps = 0
    bb_total_time_steps = 0
    bb_temp_order_violations = 0
    bb_total_temp_comparisons = 0

    # Full-data models for 15C interpolation accuracy comparison against generator 3-state Radau truth
    full_weights, full_f_mean, full_f_std = load_exported_ensemble_npz(DEFAULT_MODEL_DIR / "hybrid_ensemble_weights.npz")
    bb_full_model = _train_tuned_blackbox_mlp(df_train_full, df_val_full, seed=20261023)

    x_test_3m, _, _ = extract_feature_matrix(
        test_lot_df, feature_mean=tr_3m_data["feature_mean"], feature_std=tr_3m_data["feature_std"]
    )
    x_test_full, _, _ = extract_feature_matrix(
        test_lot_df, feature_mean=full_f_mean, feature_std=full_f_std
    )

    err_15c_hyb_3m_all, err_15c_hyb_3m_mat, err_15c_hyb_3m_mis = [], [], []
    err_15c_bb_3m_all, err_15c_bb_3m_mat, err_15c_bb_3m_mis = [], [], []
    err_15c_hyb_full_all, err_15c_hyb_full_mat, err_15c_hyb_full_mis = [], [], []
    err_15c_bb_full_all, err_15c_bb_full_mat, err_15c_bb_full_mis = [], [], []

    for l_idx, (_, lrow) in enumerate(test_lot_df.iterrows()):
        lid_str = str(lrow["lot_id"])
        lt = lots_truth[lid_str]
        is_mat = lt["structural_group"] == "matched"
        hmw0 = float(lrow["initial_hmw_pct"]) / 100.0
        lmw0 = float(lrow["initial_lmw_pct"]) / 100.0

        # Ground-truth 3-state Radau trajectory at 15C across [0, 1, 2, 3, 6, 9, 12, 18, 24M]
        hours_audit = time_grid_m * 730.5
        true_15c_pct = (
            integrate_ground_truth_3state_radau(
                hours_audit, [0.0, float(hours_audit[-1])], [15.0, 15.0], lt
            )["hmw_fraction"]
            * 100.0
        )

        # Full-data Hybrid Ensemble at 15C
        full_15c_curves = []
        for mw in full_weights:
            mp = predict_params_from_npz_member(x_test_full[l_idx], mw, cfg=cfg)
            single_p = {k: float(np.asarray(mp[k]).item()) for k in mp}
            cf_f = closed_form_isothermal_trajectory(
                hours_audit, 15.0, params=single_p,
                initial_hmw_fraction=hmw0, initial_lmw_fraction=lmw0, include_lmw=True
            )["hmw_fraction"] * 100.0
            full_15c_curves.append(cf_f)
        hyb_full_15c = np.mean(full_15c_curves, axis=0)

        # Full-data Tuned Black-Box MLP at 15C
        grid_15c_df = pd.DataFrame([lrow.to_dict()] * len(time_grid_m))
        grid_15c_df["temperature_c"] = 15.0
        grid_15c_df["time_months"] = time_grid_m
        bb_full_15c = _predict_blackbox_mlp_numpy(grid_15c_df, bb_full_model)

        # Hybrid predictions (<= 3M trained) across [5, 15, 25, 40C] x time_grid_m
        hyb_curves_by_temp: Dict[float, np.ndarray] = {}
        for tc in temps_audit:
            m_curves = []
            for mw in hyb_3m_weights:
                mp = predict_params_from_npz_member(x_test_3m[l_idx], mw, cfg=cfg)
                single_p = {k: float(np.asarray(mp[k]).item()) for k in mp}
                cf = closed_form_isothermal_trajectory(
                    hours_audit, tc, params=single_p,
                    initial_hmw_fraction=hmw0, initial_lmw_fraction=lmw0, include_lmw=True
                )["hmw_fraction"] * 100.0
                m_curves.append(cf)
            hyb_c = np.mean(m_curves, axis=0)
            hyb_curves_by_temp[tc] = hyb_c
            diffs = np.diff(hyb_c)
            hyb_neg_rate_steps += int(np.sum(diffs < -1e-9))
            hyb_total_time_steps += len(diffs)

        # Black-Box MLP predictions (<= 3M trained) across [5, 15, 25, 40C] x time_grid_m
        bb_curves_by_temp: Dict[float, np.ndarray] = {}
        for tc in temps_audit:
            grid_rows = pd.DataFrame([lrow.to_dict()] * len(time_grid_m))
            grid_rows["temperature_c"] = tc
            grid_rows["time_months"] = time_grid_m
            bb_c = _predict_blackbox_mlp_numpy(grid_rows, bb_3m_model)
            bb_curves_by_temp[tc] = bb_c
            diffs_bb = np.diff(bb_c)
            bb_neg_rate_steps += int(np.sum(diffs_bb < -1e-9))
            bb_total_time_steps += len(diffs_bb)

        # Record 15C residuals vs generator 3-state Radau truth (for t > 0 months)
        res_h3 = (hyb_curves_by_temp[15.0][1:] - true_15c_pct[1:]).tolist()
        res_b3 = (bb_curves_by_temp[15.0][1:] - true_15c_pct[1:]).tolist()
        res_hf = (hyb_full_15c[1:] - true_15c_pct[1:]).tolist()
        res_bf = (bb_full_15c[1:] - true_15c_pct[1:]).tolist()

        err_15c_hyb_3m_all.extend(res_h3)
        err_15c_bb_3m_all.extend(res_b3)
        err_15c_hyb_full_all.extend(res_hf)
        err_15c_bb_full_all.extend(res_bf)
        if is_mat:
            err_15c_hyb_3m_mat.extend(res_h3)
            err_15c_bb_3m_mat.extend(res_b3)
            err_15c_hyb_full_mat.extend(res_hf)
            err_15c_bb_full_mat.extend(res_bf)
        else:
            err_15c_hyb_3m_mis.extend(res_h3)
            err_15c_bb_3m_mis.extend(res_b3)
            err_15c_hyb_full_mis.extend(res_hf)
            err_15c_bb_full_mis.extend(res_bf)

        # Temperature monotonicity checks across (5C <= 15C <= 25C <= 40C) at each t > 0
        for t_i in range(1, len(time_grid_m)):
            for t_low, t_high in ((5.0, 15.0), (15.0, 25.0), (25.0, 40.0)):
                hyb_total_temp_comparisons += 1
                if hyb_curves_by_temp[t_high][t_i] < hyb_curves_by_temp[t_low][t_i] - 1e-9:
                    hyb_temp_order_violations += 1
                bb_total_temp_comparisons += 1
                if bb_curves_by_temp[t_high][t_i] < bb_curves_by_temp[t_low][t_i] - 1e-9:
                    bb_temp_order_violations += 1

    def _pack_15c_err(e_all: List[float], e_mat: List[float], e_mis: List[float]) -> Dict[str, Any]:
        return {
            "overall_rmse_pp": round(float(np.sqrt(np.mean(np.square(e_all)))), 5),
            "overall_mae_pp": round(float(np.mean(np.abs(e_all))), 5),
            "matched_rmse_pp": round(float(np.sqrt(np.mean(np.square(e_mat)))), 5),
            "matched_mae_pp": round(float(np.mean(np.abs(e_mat))), 5),
            "mismatched_rmse_pp": round(float(np.sqrt(np.mean(np.square(e_mis)))), 5),
            "mismatched_mae_pp": round(float(np.mean(np.abs(e_mis))), 5),
        }

    return {
        "training_protocol": "Models trained strictly on pulls <= 3.0 months (0, 0.5, 1, 2, 3M) of the 100 training lots",
        "late_time_extrapolation_test_6m_to_24m_hmw_rmse_pp": {
            "hybrid_lumry_eyring_ode": {
                "overall": round(float(np.sqrt(np.mean(hyb_late_eval["residual_pp"] ** 2))), 5),
                "matched": round(float(np.sqrt(np.mean(hyb_late_eval[hyb_late_eval["structural_group"] == "matched"]["residual_pp"] ** 2))), 5),
                "mismatched": round(float(np.sqrt(np.mean(hyb_late_eval[hyb_late_eval["structural_group"] == "mismatched"]["residual_pp"] ** 2))), 5),
            },
            "tuned_blackbox_mlp": {
                "overall": round(float(np.sqrt(np.mean(bb_late_res ** 2))), 5),
                "matched": round(float(np.sqrt(np.mean(bb_late_res[is_mat_late] ** 2))), 5),
                "mismatched": round(float(np.sqrt(np.mean(bb_late_res[~is_mat_late] ** 2))), 5),
            },
        },
        "unseen_15c_accuracy_vs_generator_3state_radau_truth_1m_to_24m": {
            "trained_on_early_pulls_le_3m": {
                "hybrid_lumry_eyring_ode": _pack_15c_err(err_15c_hyb_3m_all, err_15c_hyb_3m_mat, err_15c_hyb_3m_mis),
                "tuned_blackbox_mlp": _pack_15c_err(err_15c_bb_3m_all, err_15c_bb_3m_mat, err_15c_bb_3m_mis),
            },
            "trained_on_full_schedule_0_to_24m": {
                "hybrid_25_member_ensemble": _pack_15c_err(err_15c_hyb_full_all, err_15c_hyb_full_mat, err_15c_hyb_full_mis),
                "tuned_blackbox_mlp": _pack_15c_err(err_15c_bb_full_all, err_15c_bb_full_mat, err_15c_bb_full_mis),
            },
        },
        "physical_violation_counts_across_100_test_lots_5c_15c_25c_40c": {
            "hybrid_lumry_eyring_ode": {
                "purity_increase_or_negative_rate_violations": hyb_neg_rate_steps,
                "total_time_steps_checked": hyb_total_time_steps,
                "non_monotone_temperature_rate_violations_incl_15c": hyb_temp_order_violations,
                "total_temperature_order_pairs_checked": hyb_total_temp_comparisons,
                "note": "Zero physical violations by construction (strictly positive Lumry-Eyring rate constants and bounded positive dH_agg / dH_u).",
            },
            "tuned_blackbox_mlp": {
                "purity_increase_or_negative_rate_violations": bb_neg_rate_steps,
                "total_time_steps_checked": bb_total_time_steps,
                "non_monotone_temperature_rate_violations_incl_15c": bb_temp_order_violations,
                "total_temperature_order_pairs_checked": bb_total_temp_comparisons,
            },
        },
    }


def train_and_evaluate_baselines_and_phase3_followups(
    ensemble_weights: List[Dict[str, np.ndarray]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    obs_csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_observations.csv",
    truth_json_path: Path | str = DEFAULT_SYNTH_DIR / "_ground_truth_do_not_read_in_train.json",
    output_dir: Path | str = DEFAULT_MODEL_DIR,
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Implements Follow-ups #1 and #2 and exports `baseline_models.npz`:
    - Uses `pH` and `pH^2` as separate standardized features (11 features total, no `(pH - 6.0)^2`)
      across the Hybrid ANN, Linear Ridge regression, and the Tuned Black-Box MLP.
    - Tunes the PyTorch Black-Box MLP with early stopping on validation lots across multiple
      architectures/learning rates.
    """
    if cfg is None:
        cfg = load_hybrid_config()

    df_train = load_observations_for_training(obs_csv_path, allowed_splits=("train",))
    df_val = load_observations_for_training(obs_csv_path, allowed_splits=("val",))
    truth_data = load_ground_truth_for_verification_only(truth_json_path)
    lots_truth = truth_data["lots"]

    tr_ens_eval = evaluate_member_or_ensemble_numpy(
        ensemble_weights, feature_mean, feature_std, df_train, cfg=cfg
    )
    val_ens_eval = evaluate_member_or_ensemble_numpy(
        ensemble_weights, feature_mean, feature_std, df_val, cfg=cfg
    )

    tr_obs = tr_ens_eval["observations_df"].copy()
    val_obs = val_ens_eval["observations_df"].copy()
    tr_obs["structural_group"] = tr_obs["lot_id"].map(lambda lid: lots_truth[lid]["structural_group"])
    val_obs["structural_group"] = val_obs["lot_id"].map(lambda lid: lots_truth[lid]["structural_group"])

    per_member_tr_rmse, per_member_val_rmse = [], []
    for mw in ensemble_weights:
        m_tr = evaluate_member_or_ensemble_numpy([mw], feature_mean, feature_std, df_train, cfg=cfg)["observations_df"]
        m_val = evaluate_member_or_ensemble_numpy([mw], feature_mean, feature_std, df_val, cfg=cfg)["observations_df"]
        per_member_tr_rmse.append(float(np.sqrt(np.mean(m_tr["residual_pp"].to_numpy(dtype=np.float64) ** 2))))
        per_member_val_rmse.append(float(np.sqrt(np.mean(m_val["residual_pp"].to_numpy(dtype=np.float64) ** 2))))

    def _compute_noise_and_structural_floors(df_split: pd.DataFrame) -> Dict[str, Any]:
        sec_errs_mat, sec_errs_mis = [], []
        oracle_preeq_errs_mat, oracle_preeq_errs_mis = [], []
        cov_only_oracle_errs_mat = []

        for lid, group_df in df_split.groupby("lot_id"):
            lt = lots_truth[lid]
            is_mat = lt["structural_group"] == "matched"
            hmw0 = float(lt["initial_hmw_pct"]) / 100.0
            lmw0 = float(lt["initial_lmw_pct"]) / 100.0

            d_ph = float(lt["ph"]) - 6.0
            z_ion = (float(lt["ionic_strength_mm"]) - 120.0) / 50.0
            z_surf = (float(lt["surfactant_pct"]) - 0.020) / 0.015
            z_conc = (float(lt["protein_conc_mg_ml"]) - 50.0) / 40.0
            z_head = (float(lt["headspace_index"]) - 0.40) / 0.30
            syr = int(lt["container_syringe_flag"])
            z_glyc = (float(lt["initial_afuc_glycan_pct"]) - 6.0) / 3.0
            z_acid = (float(lt["initial_acidic_pct"]) - 14.5) / 2.5

            cov_oracle_params = {
                "dG_eff_ref_kj_mol": 121.35 - 1.50 * (d_ph ** 2 - 0.16) - 0.28 * z_ion + 0.28 * z_surf - 0.32 * z_conc - 0.18 * z_head - 0.12 * (syr - 0.40),
                "dH_agg_ref_kj_mol": 38.0 + 2.6 * (d_ph ** 2 - 0.16) + 1.4 * z_ion - 1.2 * z_surf + 1.0 * z_conc,
                "ku_ref_5c": float(np.exp(np.log(0.0010) + 0.32 * (d_ph ** 2 - 0.16) + 0.12 * z_head + 0.10 * (syr - 0.40) + 0.08 * z_glyc - 0.12 * z_surf)),
                "dH_u_kj_mol": 34.0 + 2.0 * (d_ph ** 2 - 0.16) + 1.1 * z_glyc + 0.8 * z_acid - 1.0 * z_surf,
                "dCp_agg_kj_mol_k": 0.0,
                "dG_frag_ref_kj_mol": float(lt["dG_frag_ref_kj_mol"]),
                "ea_frag_kj_mol": float(lt["ea_frag_kj_mol"]),
            }

            exact_lot_preeq_params = {
                "dG_eff_ref_kj_mol": float(lt["dG_eff_ref_kj_mol"]),
                "dH_agg_ref_kj_mol": float(lt["dH_agg_ref_kj_mol"]),
                "ku_ref_5c": float(lt["ku_ref_5c"]),
                "dH_u_kj_mol": float(lt["dH_u_kj_mol"]),
                "dCp_agg_kj_mol_k": 0.0,
                "dG_frag_ref_kj_mol": float(lt["dG_frag_ref_kj_mol"]),
                "ea_frag_kj_mol": float(lt["ea_frag_kj_mol"]),
            }

            for cond_c, c_df in group_df.groupby("temperature_c"):
                c_key = f"{int(cond_c)}C"
                radau_true_pct = np.asarray(lt["noiseless_trajectories"][c_key]["hmw_true_pct"], dtype=np.float64)
                obs_pct = c_df.sort_values("time_months")["hmw_obs_pct"].to_numpy(dtype=np.float64)
                t_hours = c_df.sort_values("time_months")["time_hours"].to_numpy(dtype=np.float64)

                cf_exact = closed_form_isothermal_trajectory(
                    t_hours, float(cond_c), params=exact_lot_preeq_params,
                    initial_hmw_fraction=hmw0, initial_lmw_fraction=lmw0, include_lmw=True
                )["hmw_fraction"] * 100.0
                cf_cov = closed_form_isothermal_trajectory(
                    t_hours, float(cond_c), params=cov_oracle_params,
                    initial_hmw_fraction=hmw0, initial_lmw_fraction=lmw0, include_lmw=True
                )["hmw_fraction"] * 100.0

                if is_mat:
                    sec_errs_mat.extend((obs_pct - radau_true_pct).tolist())
                    oracle_preeq_errs_mat.extend((obs_pct - cf_exact).tolist())
                    cov_only_oracle_errs_mat.extend((obs_pct - cf_cov).tolist())
                else:
                    sec_errs_mis.extend((obs_pct - radau_true_pct).tolist())
                    oracle_preeq_errs_mis.extend((obs_pct - cf_exact).tolist())

        return {
            "pure_sec_assay_noise_rmse_pp_matched": round(float(np.sqrt(np.mean(np.square(sec_errs_mat)))), 5),
            "pure_sec_assay_noise_rmse_pp_mismatched": round(float(np.sqrt(np.mean(np.square(sec_errs_mis)))), 5),
            "covariate_oracle_plus_lot_random_effect_and_sec_noise_rmse_pp_matched": round(
                float(np.sqrt(np.mean(np.square(cov_only_oracle_errs_mat)))), 5
            ),
            "exact_lot_param_preeq_oracle_vs_obs_rmse_pp_matched": round(
                float(np.sqrt(np.mean(np.square(oracle_preeq_errs_mat)))), 5
            ),
            "exact_lot_param_preeq_oracle_vs_obs_rmse_pp_mismatched": round(
                float(np.sqrt(np.mean(np.square(oracle_preeq_errs_mis)))), 5
            ),
        }

    train_floors = _compute_noise_and_structural_floors(df_train)
    val_floors = _compute_noise_and_structural_floors(df_val)

    train_vs_val_footing = {
        "ensemble_mean_prediction_rmse_pp": {
            "train_overall": round(float(np.sqrt(np.mean(tr_obs["residual_pp"] ** 2))), 5),
            "val_overall": round(float(np.sqrt(np.mean(val_obs["residual_pp"] ** 2))), 5),
            "train_matched": round(float(np.sqrt(np.mean(tr_obs[tr_obs["structural_group"] == "matched"]["residual_pp"] ** 2))), 5),
            "val_matched": round(float(np.sqrt(np.mean(val_obs[val_obs["structural_group"] == "matched"]["residual_pp"] ** 2))), 5),
            "train_mismatched": round(float(np.sqrt(np.mean(tr_obs[tr_obs["structural_group"] == "mismatched"]["residual_pp"] ** 2))), 5),
            "val_mismatched": round(float(np.sqrt(np.mean(val_obs[val_obs["structural_group"] == "mismatched"]["residual_pp"] ** 2))), 5),
        },
        "per_member_average_rmse_pp": {
            "train_overall_mean": round(float(np.mean(per_member_tr_rmse)), 5),
            "val_overall_mean": round(float(np.mean(per_member_val_rmse)), 5),
        },
        "irreducible_floors_comparison": {
            "train_splits": train_floors,
            "val_splits": val_floors,
        },
    }

    # --- Fit Baselines (`pooled_mean`, `ridge` with pH & pH^2, and `tuned_blackbox_mlp`) on the 100 training lots ---
    train_lot_df = df_train.groupby("lot_id", as_index=False).first().sort_values("lot_id")
    val_lot_df = df_val.groupby("lot_id", as_index=False).first().sort_values("lot_id")

    # Estimate empirical per-lot parameters on the 100 training lots from their noisy observations ONLY
    emp_dg_train, emp_dh_train = [], []
    nominal_prior_member = {k: float(cfg["parameter_bounds"][k]["nominal"]) for k in PREDICTED_PARAM_KEYS}
    nominal_prior_member["log_sigma_adj"] = 0.0
    nominal_prior_member["dCp_agg_kj_mol_k"] = 0.0
    nominal_prior_member["dCp_u_kj_mol_k"] = 0.0

    bounds = cfg["parameter_bounds"]
    dg_min, dg_max = float(bounds["dG_eff_ref_kj_mol"]["min"]), float(bounds["dG_eff_ref_kj_mol"]["max"])
    dh_min, dh_max = float(bounds["dH_agg_ref_kj_mol"]["min"]), float(bounds["dH_agg_ref_kj_mol"]["max"])
    base_sec_sd_pct = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])

    for _, lrow in train_lot_df.iterrows():
        lid = str(lrow["lot_id"])
        sub_l = df_train[(df_train["lot_id"] == lid) & (df_train["time_hours"] > 0.0)]
        t_h = sub_l["time_hours"].to_numpy(dtype=np.float64)
        t_c = sub_l["temperature_c"].to_numpy(dtype=np.float64)
        y_f = sub_l["hmw_obs_pct"].to_numpy(dtype=np.float64) / 100.0
        sig_f = heteroscedastic_sec_sd_fraction(y_f, base_sd_pct=base_sec_sd_pct, rel_rsd_slope=0.03)
        hmw0 = float(lrow["initial_hmw_pct"]) / 100.0
        lmw0 = float(lrow["initial_lmw_pct"]) / 100.0

        def _lot_ols_res(theta: np.ndarray) -> np.ndarray:
            p_try = dict(nominal_prior_member)
            p_try["dG_eff_ref_kj_mol"] = float(theta[0])
            p_try["dH_agg_ref_kj_mol"] = float(theta[1])
            preds = []
            for tc_i, th_i in zip(t_c, t_h):
                cf = closed_form_isothermal_trajectory(
                    [0.0, float(th_i)], float(tc_i), params=p_try,
                    initial_hmw_fraction=hmw0, initial_lmw_fraction=lmw0, include_lmw=True
                )
                preds.append(float(cf["hmw_fraction"][-1]))
            return (np.asarray(preds, dtype=np.float64) - y_f) / sig_f

        opt_l = least_squares(
            _lot_ols_res,
            x0=np.array([121.0, 38.0], dtype=np.float64),
            bounds=([dg_min, dh_min], [dg_max, dh_max]),
            method="trf",
            ftol=1e-6,
            xtol=1e-6,
        )
        emp_dg_train.append(float(opt_l.x[0]))
        emp_dh_train.append(float(opt_l.x[1]))

    emp_dg_arr = np.asarray(emp_dg_train, dtype=np.float64)
    emp_dh_arr = np.asarray(emp_dh_train, dtype=np.float64)

    # Baseline (a): Pooled training-mean parameters for every lot (no ANN)
    pooled_params = {
        "dG_eff_ref_kj_mol": float(np.mean(emp_dg_arr)),
        "dH_agg_ref_kj_mol": float(np.mean(emp_dh_arr)),
        "ku_ref_5c": 0.0010,
        "dH_u_kj_mol": 34.0,
        "dCp_agg_kj_mol_k": 0.0,
        "dG_frag_ref_kj_mol": 125.6,
        "ea_frag_kj_mol": 78.4,
    }

    # Baseline (b): Linear Ridge regression given the exact same 11 standardized inputs (including pH and pH^2)
    x_tr_11, lin_mean, lin_std = extract_feature_matrix(
        train_lot_df, feature_mean=feature_mean, feature_std=feature_std
    )
    x_tr_aug = np.hstack([np.ones((len(x_tr_11), 1)), x_tr_11])

    ridge_alpha = 5.0
    reg_mat = ridge_alpha * np.eye(x_tr_aug.shape[1])
    reg_mat[0, 0] = 0.0
    w_ridge_dg = np.linalg.solve(x_tr_aug.T @ x_tr_aug + reg_mat, x_tr_aug.T @ emp_dg_arr)
    w_ridge_dh = np.linalg.solve(x_tr_aug.T @ x_tr_aug + reg_mat, x_tr_aug.T @ emp_dh_arr)

    # Baseline (c): Tuned PyTorch Black-Box MLP (same pH and pH^2 inputs + T + t, early stopping on val lots)
    bb_tuned = _train_tuned_blackbox_mlp(df_train, df_val, seed=20261023)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "baseline_models.npz",
        pooled_dG_eff_ref=np.array(pooled_params["dG_eff_ref_kj_mol"]),
        pooled_dH_agg_ref=np.array(pooled_params["dH_agg_ref_kj_mol"]),
        lin_mean=lin_mean,
        lin_std=lin_std,
        w_ridge_dg=w_ridge_dg,
        w_ridge_dh=w_ridge_dh,
        bb_mean=bb_tuned["bb_mean"],
        bb_std=bb_tuned["bb_std"],
        w1_bb=bb_tuned["w1"],
        b1_bb=bb_tuned["b1"],
        w2_bb=bb_tuned["w2"],
        b2_bb=bb_tuned["b2"],
        w3_bb=bb_tuned["w3"],
        b3_bb=bb_tuned["b3"],
    )

    def _eval_param_baseline_on_val(mode: str) -> Dict[str, Any]:
        obs_res_all, obs_res_mat, obs_res_mis = [], [], []
        dg_err_all, dg_err_mat, dg_err_mis = [], [], []
        k5_err_all, k5_err_mat, k5_err_mis = [], [], []
        k25_err_all, k25_err_mat, k25_err_mis = [], [], []
        k40_err_all, k40_err_mat, k40_err_mis = [], [], []

        for _, vrow in val_lot_df.iterrows():
            lid = str(vrow["lot_id"])
            lt = lots_truth[lid]
            is_mat = lt["structural_group"] == "matched"
            if mode == "pooled_mean":
                p_lot = dict(pooled_params)
            else:
                x_v, _, _ = extract_feature_matrix(
                    vrow.to_dict(), feature_mean=lin_mean, feature_std=lin_std
                )
                x_v_aug = np.concatenate([[1.0], x_v[0]])
                p_lot = dict(pooled_params)
                p_lot["dG_eff_ref_kj_mol"] = float(np.clip(x_v_aug @ w_ridge_dg, dg_min, dg_max))
                p_lot["dH_agg_ref_kj_mol"] = float(np.clip(x_v_aug @ w_ridge_dh, dh_min, dh_max))

            ea_d = compute_lot_apparent_ea(p_lot)
            dg_e = abs(float(p_lot["dG_eff_ref_kj_mol"]) - float(lt["dG_eff_ref_kj_mol"]))
            k5_e = abs(float(ea_d["k_eff_5c_per_h"]) - float(lt["k_eff_5c_true_per_h"])) / float(lt["k_eff_5c_true_per_h"])
            k25_e = abs(float(ea_d["k_eff_25c_per_h"]) - float(lt["k_eff_25c_true_per_h"])) / float(lt["k_eff_25c_true_per_h"])
            k40_e = abs(float(ea_d["k_eff_40c_per_h"]) - float(lt["k_eff_40c_true_per_h"])) / float(lt["k_eff_40c_true_per_h"])

            dg_err_all.append(dg_e)
            k5_err_all.append(k5_e)
            k25_err_all.append(k25_e)
            k40_err_all.append(k40_e)
            if is_mat:
                dg_err_mat.append(dg_e)
                k5_err_mat.append(k5_e)
                k25_err_mat.append(k25_e)
                k40_err_mat.append(k40_e)
            else:
                dg_err_mis.append(dg_e)
                k5_err_mis.append(k5_e)
                k25_err_mis.append(k25_e)
                k40_err_mis.append(k40_e)

            hmw0 = float(vrow["initial_hmw_pct"]) / 100.0
            lmw0 = float(vrow["initial_lmw_pct"]) / 100.0
            sub_v = df_val[df_val["lot_id"] == lid]
            for cond_c, c_df in sub_v.groupby("temperature_c"):
                t_h = c_df["time_hours"].to_numpy(dtype=np.float64)
                y_obs = c_df["hmw_obs_pct"].to_numpy(dtype=np.float64)
                cf = closed_form_isothermal_trajectory(
                    t_h, float(cond_c), params=p_lot,
                    initial_hmw_fraction=hmw0, initial_lmw_fraction=lmw0, include_lmw=True
                )["hmw_fraction"] * 100.0
                res = (y_obs - cf).tolist()
                obs_res_all.extend(res)
                if is_mat:
                    obs_res_mat.extend(res)
                else:
                    obs_res_mis.extend(res)

        def _pack(res_l, dg_l, k5_l, k25_l, k40_l):
            return {
                "hmw_rmse_pp": round(float(np.sqrt(np.mean(np.square(res_l)))), 5),
                "dG_eff_ref_mae_kj_mol": round(float(np.mean(dg_l)), 4),
                "k_eff_5c_mape_pct": round(float(np.mean(k5_l)) * 100.0, 2),
                "k_eff_25c_mape_pct": round(float(np.mean(k25_l)) * 100.0, 2),
                "k_eff_40c_mape_pct": round(float(np.mean(k40_l)) * 100.0, 2),
            }

        return {
            "overall": _pack(obs_res_all, dg_err_all, k5_err_all, k25_err_all, k40_err_all),
            "matched": _pack(obs_res_mat, dg_err_mat, k5_err_mat, k25_err_mat, k40_err_mat),
            "mismatched": _pack(obs_res_mis, dg_err_mis, k5_err_mis, k25_err_mis, k40_err_mis),
        }

    pooled_val = _eval_param_baseline_on_val("pooled_mean")
    ridge_val = _eval_param_baseline_on_val("ridge")

    y_bb_pred = _predict_blackbox_mlp_numpy(df_val, bb_tuned)
    bb_res = df_val["hmw_obs_pct"].to_numpy(dtype=np.float64) - y_bb_pred
    val_is_mat = df_val["lot_id"].map(lambda lid: lots_truth[lid]["structural_group"] == "matched").to_numpy(dtype=bool)
    blackbox_val = {
        "search_log": bb_tuned["search_log"],
        "selected_config": {
            "hidden_dims": list(bb_tuned["config"]["hidden_dims"]),
            "lr": bb_tuned["config"]["lr"],
            "weight_decay": bb_tuned["config"]["weight_decay"],
            "best_epoch": bb_tuned["best_epoch"],
        },
        "overall": {"hmw_rmse_pp": round(float(np.sqrt(np.mean(bb_res ** 2))), 5)},
        "matched": {"hmw_rmse_pp": round(float(np.sqrt(np.mean(bb_res[val_is_mat] ** 2))), 5)},
        "mismatched": {"hmw_rmse_pp": round(float(np.sqrt(np.mean(bb_res[~val_is_mat] ** 2))), 5)},
    }

    return {
        "train_vs_val_rmse_analysis": train_vs_val_footing,
        "baseline_validation_comparison": {
            "pooled_mean_no_ann": pooled_val,
            "linear_ridge_on_descriptors": ridge_val,
            "blackbox_mlp_no_physics": blackbox_val,
        },
    }


def run_phase4_verification_and_save(
    output_dir: Path | str = DEFAULT_MODEL_DIR,
) -> Dict[str, Any]:
    """
    Runs the complete Phase 4 pipeline (including all 8 required Phase 4 follow-ups) and saves
    `phase4_uncertainty_report.json` and `calibration_bundle.json` to `output_dir`.
    """
    cfg = load_hybrid_config()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_meta = generate_calibration_dataset()
    weights_list, f_mean, f_std = load_exported_ensemble_npz(out_dir / "hybrid_ensemble_weights.npz")

    cal_bundle = calibrate_uncertainty_on_calibration_lots(
        weights_list, f_mean, f_std, cfg=cfg
    )
    (out_dir / "calibration_bundle.json").write_text(
        json.dumps(cal_bundle, indent=2), encoding="utf-8"
    )

    test_cov = evaluate_test_coverage_and_recovery(
        weights_list, f_mean, f_std, cal_bundle, cfg=cfg
    )
    exc_eval = evaluate_excursion_verdicts_with_ensemble(
        weights_list, f_mean, f_std, cal_bundle, cfg=cfg
    )
    followups = train_and_evaluate_baselines_and_phase3_followups(
        weights_list, f_mean, f_std, output_dir=out_dir, cfg=cfg
    )
    lot_adapt_eval = evaluate_early_pull_lot_adaptation_on_splits(
        weights_list, f_mean, f_std, cal_bundle, cfg=cfg
    )
    phys_validity = evaluate_physical_validity_extrapolation_and_15c(cfg=cfg)

    from .q1e import evaluate_shelf_life_vs_truth_on_test_lots
    shelf_life_vs_truth = evaluate_shelf_life_vs_truth_on_test_lots(model_dir=out_dir, cfg=cfg)

    model_card_path = out_dir / "model_card.json"
    artifact_hashes: Dict[str, str] = {}
    if model_card_path.exists():
        mc_data = json.loads(model_card_path.read_text(encoding="utf-8"))
        artifact_hashes.update(mc_data.get("artifacts", {}))
        artifact_hashes.update(mc_data.get("artifact_hashes_sha256", {}))
    for fname in ("hybrid_ensemble_weights.npz", "hybrid_ensemble_torch.pt", "calibration_bundle.json", "baseline_models.npz"):
        f_p = out_dir / fname
        if f_p.exists():
            artifact_hashes[fname] = hashlib.sha256(f_p.read_bytes()).hexdigest()

    report = {
        "classification": "Non-GMP Decision Support Prototype on Synthetic Data — Requires Human QA Approval",
        "config_sha256": get_config_sha256(),
        "retrained_model_artifacts_sha256": artifact_hashes,
        "calibration_dataset_meta": cal_meta,
        "calibration_bundle": cal_bundle,
        "phase3_followups_and_baselines": followups,
        "lot_adaptation_6m_9m_12m_evaluation": lot_adapt_eval,
        "test_set_coverage_and_recovery": test_cov,
        "excursion_verdict_evaluation": exc_eval,
        "physical_validity_and_15c_audit": phys_validity,
        "shelf_life_vs_truth_100_test_lots": shelf_life_vs_truth,
    }
    (out_dir / "phase4_uncertainty_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
