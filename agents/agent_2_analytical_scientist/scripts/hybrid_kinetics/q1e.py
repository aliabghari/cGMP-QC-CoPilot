"""
Phase 5: ICH Q1E Shelf-Life Estimation & Linear ANCOVA Batch-Poolability Comparison (`q1e.py`).

Implements:
1. **Hybrid Mechanistic + Ensemble Shelf-Life Estimator:**
   - Shelf life at 5 °C (or 25 °C) is defined as the time `t` at which the **one-sided 95% upper
     confidence bound on the mean degradation trajectory** (`mean_pct + 1.64485 * traj_sd_pp`)
     crosses the upper specification limit (`2.0%` HMW, or equivalently lower 95% bound crossing
     `97.5%` Monomer).
2. **Classic ICH Q1E Linear Regression + ANCOVA Batch-Poolability Test (`p = 0.25`):**
   - Tests equality of slopes (`lot x time` interaction F-test against `p = 0.25`) and equality of
     intercepts (`lot` main-effect F-test against `p = 0.25`) across multi-lot real-time stability
     data:
     - Case 1 (`p_slope >= 0.25` and `p_intercept >= 0.25`): Pool both slopes and intercepts into a
       single regression line and compute the one-sided 95% upper confidence limit on the mean line.
     - Case 2 (`p_slope >= 0.25` and `p_intercept < 0.25`): Common slope with separate lot intercepts;
       shelf life is set by the worst-case (minimum shelf-life) lot.
     - Case 3 (`p_slope < 0.25`): Separate slopes and separate intercepts per lot; shelf life is set
       by the worst-case lot.
3. **Extrapolation Guard & Regulatory Framing:**
   - Enforces the ICH Q1E extrapolation guard: any shelf-life claim beyond the observed real-time
     data window is capped at `min(max_observed_months + 12.0, 2 * max_observed_months)` (or `+6.0`
     months when accelerated data show an OOT change) with an explicit warning banner.
   - Notes explicitly that **ICH Q5C** (*Quality of Biotechnological Products: Stability Testing of
     Biotechnological/Biological Products*, requiring real-time/real-temperature data) is the primary
     regulatory standard for therapeutic mAbs, while **ICH Q1E** provides the statistical confidence-bound
     and ANCOVA batch-poolability methodology.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import f as scipy_f
from scipy.stats import norm as scipy_norm
from scipy.stats import t as scipy_t

from .config import load_hybrid_config
from .model import extract_feature_matrix, predict_params_from_npz_member
from .ode import closed_form_isothermal_trajectory
from .synth import (
    DEFAULT_SYNTH_DIR,
    integrate_ground_truth_3state_radau,
    load_ground_truth_for_verification_only,
    load_observations_for_training,
)
from .train import DEFAULT_MODEL_DIR, load_exported_ensemble_npz
from .uncertainty import clopper_pearson_interval

ICH_Q5C_Q1E_REGULATORY_NOTE = (
    "ICH Q5C is the primary regulatory standard for therapeutic biologics (requiring real-time, "
    "real-temperature stability data); ICH Q1E statistical confidence-bound and ANCOVA batch-poolability "
    "(p=0.25) methods are capped at observed_months + 6.0 months."
)


def check_late_pull_model_adequacy(
    observed_times_months: np.ndarray | Sequence[float],
    observed_hmw_pct: np.ndarray | Sequence[float],
    predicted_mean_hmw_pct: np.ndarray | Sequence[float],
    sigma_sec_pct: float = 0.035,
    late_pull_cutoff_months: float = 18.0,
    threshold_sd: float = 2.5,
) -> Dict[str, Any]:
    """
    Late-pull residual / model adequacy check:
    Flags when residuals (observed - predicted) at the latest pulls (>= 18M, or the latest observed pull
    if max_obs >= 12M) exceed threshold_sd * sigma_sec (e.g. 2.5 SEC-SD = 0.0875 pp) or trend upward
    with positive residual.
    When flagged, downgrades the shelf-life output status to 'model adequacy not demonstrated'.
    """
    t_arr = np.asarray(observed_times_months, dtype=np.float64)
    y_obs = np.asarray(observed_hmw_pct, dtype=np.float64)
    y_pred = np.asarray(predicted_mean_hmw_pct, dtype=np.float64)
    res = y_obs - y_pred

    if len(t_arr) == 0:
        return {
            "model_adequacy_demonstrated": True,
            "status": "model adequacy demonstrated",
            "reason": "No observations provided to assess adequacy.",
            "late_pull_residuals": [],
        }

    max_t = float(np.max(t_arr))
    if max_t >= late_pull_cutoff_months:
        late_mask = t_arr >= late_pull_cutoff_months
    elif max_t >= 12.0:
        late_mask = t_arr == max_t
    else:
        late_mask = np.zeros_like(t_arr, dtype=bool)

    late_indices = np.where(late_mask)[0]
    late_residuals_info = []
    threshold_pp = threshold_sd * sigma_sec_pct

    exceeded_count = 0
    reasons = []

    for idx in late_indices:
        tm = float(t_arr[idx])
        r_pp = float(res[idx])
        r_sd = r_pp / max(sigma_sec_pct, 1e-9)
        exceeds = bool(r_pp > threshold_pp)
        late_residuals_info.append(
            {
                "time_months": tm,
                "observed_pct": round(float(y_obs[idx]), 4),
                "predicted_pct": round(float(y_pred[idx]), 4),
                "residual_pp": round(r_pp, 4),
                "residual_sec_sd": round(r_sd, 2),
                "exceeds_threshold": exceeds,
            }
        )
        if exceeds:
            exceeded_count += 1
            reasons.append(
                f"Late pull at {tm:.0f}M residual ({r_pp:+.3f} pp = {r_sd:+.1f} SEC-SD) "
                f"exceeds +{threshold_sd:g} SEC-SD (+{threshold_pp:.4f} pp)"
            )

    upward_trend = False
    if len(late_indices) >= 2:
        last_res = [float(res[i]) for i in late_indices[-3:]]
        if all(last_res[i] < last_res[i + 1] for i in range(len(last_res) - 1)) and last_res[-1] > 2.0 * sigma_sec_pct:
            upward_trend = True
            reasons.append(
                f"Late-pull residuals exhibit consecutive upward acceleration (latest {last_res[-1] / sigma_sec_pct:+.1f} SEC-SD)"
            )

    adequacy_demonstrated = not (exceeded_count > 0 or upward_trend)
    status = "model adequacy demonstrated" if adequacy_demonstrated else "model adequacy not demonstrated"

    return {
        "model_adequacy_demonstrated": adequacy_demonstrated,
        "status": status,
        "reason": "; ".join(reasons) if reasons else "Late-pull residuals within acceptable assay repeatability envelope.",
        "late_pull_residuals": late_residuals_info,
    }


def _find_crossing_time_months(
    months_grid: np.ndarray,
    curve_pct: np.ndarray,
    spec_limit_pct: float = 2.00,
) -> float:
    """
    Finds the earliest time (in months) via linear interpolation where `curve_pct` crosses
    above `spec_limit_pct`. Returns `months_grid[-1]` if it never crosses within the grid.
    """
    idx_above = np.where(curve_pct >= spec_limit_pct)[0]
    if len(idx_above) == 0:
        return float(months_grid[-1])
    i = int(idx_above[0])
    if i == 0:
        return 0.0
    t0, t1 = float(months_grid[i - 1]), float(months_grid[i])
    y0, y1 = float(curve_pct[i - 1]), float(curve_pct[i])
    if abs(y1 - y0) < 1e-12:
        return t0
    frac = (spec_limit_pct - y0) / (y1 - y0)
    return float(t0 + np.clip(frac, 0.0, 1.0) * (t1 - t0))


def run_ich_q1e_linear_ancova_poolability(
    df_lots: pd.DataFrame,
    temperature_c: float = 5.0,
    spec_limit_hmw_pct: float = 2.00,
    alpha_pool: float = 0.25,
    months_grid: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Performs the classic ICH Q1E Section B.2.2 ANCOVA batch-poolability test across multiple lots
    at `temperature_c` using significance threshold `alpha_pool = 0.25` for both slope equality
    and intercept equality, and computes the one-sided 95% upper confidence bound on the mean
    regression line.
    """
    if months_grid is None:
        months_grid = np.linspace(0.0, 60.0, 241)

    sub = df_lots[df_lots["temperature_c"] == temperature_c].copy()
    lot_ids = sorted(sub["lot_id"].unique())
    k_lots = len(lot_ids)
    if k_lots == 0:
        raise ValueError(f"No observations found at temperature_c={temperature_c}.")

    t_all = sub["time_months"].to_numpy(dtype=np.float64)
    y_all = sub["hmw_obs_pct"].to_numpy(dtype=np.float64)
    n_total = len(y_all)

    # Model 1 (Pooled): common intercept + common slope (2 parameters)
    X_m1 = np.column_stack([np.ones(n_total), t_all])
    beta_m1, _, _, _ = np.linalg.lstsq(X_m1, y_all, rcond=None)
    rss_m1 = float(np.sum((y_all - X_m1 @ beta_m1) ** 2))
    df_m1 = n_total - 2

    if k_lots == 1:
        p_slope = 1.0
        p_intercept = 1.0
        f_slope = 0.0
        f_intercept = 0.0
        pooling_decision = "single_lot_only"
    else:
        # One-hot lot indicators (k_lots columns)
        lot_to_col = {lid: idx for idx, lid in enumerate(lot_ids)}
        lot_ind = np.zeros((n_total, k_lots), dtype=np.float64)
        for row_i, lid in enumerate(sub["lot_id"]):
            lot_ind[row_i, lot_to_col[str(lid)]] = 1.0

        # Model 2 (Common slope, separate intercepts): k_lots + 1 parameters
        X_m2 = np.column_stack([lot_ind, t_all])
        beta_m2, _, _, _ = np.linalg.lstsq(X_m2, y_all, rcond=None)
        rss_m2 = float(np.sum((y_all - X_m2 @ beta_m2) ** 2))
        df_m2 = n_total - (k_lots + 1)

        # Model 3 (Separate slopes, separate intercepts): 2 * k_lots parameters
        X_m3 = np.column_stack([lot_ind, lot_ind * t_all[:, None]])
        beta_m3, _, _, _ = np.linalg.lstsq(X_m3, y_all, rcond=None)
        rss_m3 = float(np.sum((y_all - X_m3 @ beta_m3) ** 2))
        df_m3 = n_total - 2 * k_lots

        # F-test for equality of slopes (Model 2 vs Model 3)
        ms_err_m3 = max(rss_m3 / max(df_m3, 1), 1e-12)
        f_slope = ((rss_m2 - rss_m3) / max(k_lots - 1, 1)) / ms_err_m3
        f_slope = max(0.0, float(f_slope))
        p_slope = float(1.0 - scipy_f.cdf(f_slope, k_lots - 1, max(df_m3, 1)))

        # F-test for equality of intercepts given common slope (Model 1 vs Model 2)
        ms_err_m2 = max(rss_m2 / max(df_m2, 1), 1e-12)
        f_intercept = ((rss_m1 - rss_m2) / max(k_lots - 1, 1)) / ms_err_m2
        f_intercept = max(0.0, float(f_intercept))
        p_intercept = float(1.0 - scipy_f.cdf(f_intercept, k_lots - 1, max(df_m2, 1)))

        if p_slope >= alpha_pool and p_intercept >= alpha_pool:
            pooling_decision = "pool_slopes_and_intercepts (p_slope >= 0.25, p_intercept >= 0.25)"
        elif p_slope >= alpha_pool and p_intercept < alpha_pool:
            pooling_decision = "common_slope_separate_intercepts (p_slope >= 0.25, p_intercept < 0.25)"
        else:
            pooling_decision = "separate_slopes_separate_intercepts_worst_case_lot (p_slope < 0.25)"

    # Compute one-sided 95% upper confidence curve for each lot and for the pooled/governing model
    per_lot_results: Dict[str, Dict[str, Any]] = {}
    for lid in lot_ids:
        l_sub = sub[sub["lot_id"] == lid]
        tl = l_sub["time_months"].to_numpy(dtype=np.float64)
        yl = l_sub["hmw_obs_pct"].to_numpy(dtype=np.float64)
        nl = len(yl)
        Xl = np.column_stack([np.ones(nl), tl])
        bl, _, _, _ = np.linalg.lstsq(Xl, yl, rcond=None)
        y_hat_l = bl[0] + bl[1] * months_grid
        s2_l = float(np.sum((yl - Xl @ bl) ** 2)) / max(nl - 2, 1)
        t_crit_l = float(scipy_t.ppf(0.95, df=max(nl - 2, 1)))
        t_mean_l = float(np.mean(tl))
        ss_t_l = max(float(np.sum((tl - t_mean_l) ** 2)), 1e-12)
        se_mean_l = np.sqrt(s2_l * (1.0 / nl + ((months_grid - t_mean_l) ** 2) / ss_t_l))
        u95_l = y_hat_l + t_crit_l * se_mean_l
        sl_l = _find_crossing_time_months(months_grid, u95_l, spec_limit_hmw_pct)
        per_lot_results[str(lid)] = {
            "intercept_pct": round(float(bl[0]), 4),
            "slope_pp_per_month": round(float(bl[1]), 5),
            "unclipped_shelf_life_months": round(sl_l, 2),
            "mean_curve_pct": y_hat_l,
            "upper_95_confidence_pct": u95_l,
        }

    # Governing linear Q1E curve & shelf life according to the ANCOVA pooling decision
    if pooling_decision.startswith("pool_slopes_and_intercepts") or pooling_decision == "single_lot_only":
        s2_p = rss_m1 / max(df_m1, 1)
        t_crit_p = float(scipy_t.ppf(0.95, df=max(df_m1, 1)))
        t_mean_p = float(np.mean(t_all))
        ss_t_p = max(float(np.sum((t_all - t_mean_p) ** 2)), 1e-12)
        gov_mean = beta_m1[0] + beta_m1[1] * months_grid
        se_gov = np.sqrt(s2_p * (1.0 / n_total + ((months_grid - t_mean_p) ** 2) / ss_t_p))
        gov_u95 = gov_mean + t_crit_p * se_gov
        gov_sl = _find_crossing_time_months(months_grid, gov_u95, spec_limit_hmw_pct)
        governing_lot_id = "POOLED_ALL_LOTS"
    elif pooling_decision.startswith("common_slope_separate_intercepts"):
        # Common slope beta_m2[-1], separate intercepts beta_m2[:k_lots]; worst-case lot has highest intercept
        s2_c = rss_m2 / max(df_m2, 1)
        t_crit_c = float(scipy_t.ppf(0.95, df=max(df_m2, 1)))
        cov_m2 = s2_c * np.linalg.pinv(X_m2.T @ X_m2)
        worst_idx = int(np.argmax(beta_m2[:k_lots]))
        governing_lot_id = str(lot_ids[worst_idx])
        X_grid_w = np.zeros((len(months_grid), k_lots + 1), dtype=np.float64)
        X_grid_w[:, worst_idx] = 1.0
        X_grid_w[:, -1] = months_grid
        gov_mean = X_grid_w @ beta_m2
        se_gov = np.sqrt(np.maximum(0.0, np.sum((X_grid_w @ cov_m2) * X_grid_w, axis=1)))
        gov_u95 = gov_mean + t_crit_c * se_gov
        gov_sl = _find_crossing_time_months(months_grid, gov_u95, spec_limit_hmw_pct)
    else:
        # Separate slopes and intercepts: pick lot with shortest shelf life
        worst_lid = min(per_lot_results.keys(), key=lambda k: per_lot_results[k]["unclipped_shelf_life_months"])
        governing_lot_id = worst_lid
        gov_mean = per_lot_results[worst_lid]["mean_curve_pct"]
        gov_u95 = per_lot_results[worst_lid]["upper_95_confidence_pct"]
        gov_sl = float(per_lot_results[worst_lid]["unclipped_shelf_life_months"])

    return {
        "k_lots": k_lots,
        "n_observations": n_total,
        "alpha_poolability_threshold": alpha_pool,
        "f_stat_slope_equality": round(f_slope, 4),
        "p_value_slope_equality": round(p_slope, 4),
        "f_stat_intercept_equality": round(f_intercept, 4),
        "p_value_intercept_equality": round(p_intercept, 4),
        "pooling_decision": pooling_decision,
        "governing_lot_id": governing_lot_id,
        "unclipped_linear_q1e_shelf_life_months": round(gov_sl, 2),
        "governing_mean_curve_pct": gov_mean,
        "governing_upper_95_confidence_pct": gov_u95,
        "per_lot_linear_summaries": {
            k: {
                "intercept_pct": v["intercept_pct"],
                "slope_pp_per_month": v["slope_pp_per_month"],
                "unclipped_shelf_life_months": v["unclipped_shelf_life_months"],
            }
            for k, v in per_lot_results.items()
        },
    }


def evaluate_q1e_shelf_life_comparison(
    lot_ids: List[str] | None = None,
    temperature_c: float = 5.0,
    spec_limit_hmw_pct: float = 2.00,
    max_extrapolation_beyond_observed_months: float | None = None,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Compares the **Hybrid Mechanistic + Ensemble one-sided 95% confidence bound** against
    **Classic ICH Q1E Linear Regression + ANCOVA Batch-Poolability (`p = 0.25`)** across a
    representative multi-lot stability cohort (defaulting to 3 held-out test lots), and enforces
    the ICH Q1E extrapolation guard.
    """
    if cfg is None:
        cfg = load_hybrid_config()
    if max_extrapolation_beyond_observed_months is None:
        max_extrapolation_beyond_observed_months = float(
            cfg["assay_and_decision_thresholds"].get("q1e_max_extrapolation_months_beyond_obs", 6.0)
        )
    reg_note = str(
        cfg["assay_and_decision_thresholds"].get(
            "q1e_biologics_regulatory_note", ICH_Q5C_Q1E_REGULATORY_NOTE
        )
    )

    obs_csv = DEFAULT_SYNTH_DIR / "synth_observations.csv"
    df_test = load_observations_for_training(obs_csv, allowed_splits=("test",))
    if lot_ids is None:
        all_test_lids = sorted(df_test["lot_id"].unique())
        # Pick 3 representative held-out test lots for the ICH Q1E 3-batch registration cohort
        lot_ids = [all_test_lids[0], all_test_lids[5], all_test_lids[12]]

    df_cohort = df_test[df_test["lot_id"].isin(lot_ids)].copy()
    max_obs_months = float(df_cohort[df_cohort["temperature_c"] == temperature_c]["time_months"].max())
    allowed_capped_max_months = max_obs_months + max_extrapolation_beyond_observed_months

    months_grid = np.linspace(0.0, 60.0, 241)
    hours_grid = months_grid * 730.5

    # 1. Run classic ICH Q1E linear regression + ANCOVA poolability (p = 0.25)
    ancova_res = run_ich_q1e_linear_ancova_poolability(
        df_cohort,
        temperature_c=temperature_c,
        spec_limit_hmw_pct=spec_limit_hmw_pct,
        alpha_pool=0.25,
        months_grid=months_grid,
    )

    # 2. Run Hybrid Mechanistic + 25-member Ensemble one-sided 95% confidence bound across the cohort
    mdir = Path(model_dir)
    weights_list, f_mean, f_std = load_exported_ensemble_npz(mdir / "hybrid_ensemble_weights.npz")
    cal_path = mdir / "calibration_bundle.json"
    if cal_path.exists():
        cal_bundle = json.loads(cal_path.read_text(encoding="utf-8"))
        gamma_lot = float(cal_bundle.get("gamma_lot_trajectory", 0.08))
    else:
        gamma_lot = 0.08

    base_sec_sd = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    z_95_one_sided = float(scipy_norm.ppf(0.95))  # 1.644854

    hybrid_lot_curves: Dict[str, Dict[str, Any]] = {}
    cohort_adequacy_checks: Dict[str, Any] = {}
    cohort_model_adequacy_demonstrated = True
    cohort_inadequacy_reasons: List[str] = []

    for lid in lot_ids:
        l_rows = df_cohort[df_cohort["lot_id"] == lid]
        cov = l_rows.iloc[0].to_dict()
        x_norm, _, _ = extract_feature_matrix(cov, feature_mean=f_mean, feature_std=f_std)
        hmw0_pct = float(cov["initial_hmw_pct"])
        hmw0_frac = hmw0_pct / 100.0
        lmw0_frac = float(cov["initial_lmw_pct"]) / 100.0

        member_curves = []
        for mw in weights_list:
            mp = predict_params_from_npz_member(x_norm[0], mw, cfg=cfg)
            single_p = {k: float(np.asarray(mp[k]).item()) for k in mp}
            cf = closed_form_isothermal_trajectory(
                time_grid_h=hours_grid,
                temp_c=temperature_c,
                params=single_p,
                initial_hmw_fraction=hmw0_frac,
                initial_lmw_fraction=lmw0_frac,
                include_lmw=True,
            )
            member_curves.append(cf["hmw_fraction"] * 100.0)

        mat = np.stack(member_curves, axis=0)
        mean_pct = np.mean(mat, axis=0)
        epistemic_sd_pct = np.std(mat, axis=0, ddof=1)
        delta_pp = np.maximum(0.0, mean_pct - hmw0_pct)
        traj_sd_pp = np.sqrt(epistemic_sd_pct ** 2 + (gamma_lot * delta_pp) ** 2)

        # One-sided 95% upper confidence bound on the mean degradation curve
        u95_one_sided_pct = mean_pct + z_95_one_sided * traj_sd_pp
        sl_unclipped = _find_crossing_time_months(months_grid, u95_one_sided_pct, spec_limit_hmw_pct)

        # Late-pull model adequacy check on observed data for this lot
        l_sub_obs = l_rows[l_rows["temperature_c"] == temperature_c].sort_values("time_months")
        t_obs_m = l_sub_obs["time_months"].to_numpy(dtype=np.float64)
        t_obs_h = t_obs_m * 730.5
        y_obs_val = l_sub_obs["hmw_obs_pct"].to_numpy(dtype=np.float64)

        obs_member_curves = []
        for mw in weights_list:
            mp = predict_params_from_npz_member(x_norm[0], mw, cfg=cfg)
            single_p = {k: float(np.asarray(mp[k]).item()) for k in mp}
            cf_o = closed_form_isothermal_trajectory(
                time_grid_h=t_obs_h,
                temp_c=temperature_c,
                params=single_p,
                initial_hmw_fraction=hmw0_frac,
                initial_lmw_fraction=lmw0_frac,
                include_lmw=True,
            )
            obs_member_curves.append(cf_o["hmw_fraction"] * 100.0)
        y_obs_pred_mean = np.mean(np.stack(obs_member_curves, axis=0), axis=0)

        ad_chk = check_late_pull_model_adequacy(
            observed_times_months=t_obs_m,
            observed_hmw_pct=y_obs_val,
            predicted_mean_hmw_pct=y_obs_pred_mean,
            sigma_sec_pct=base_sec_sd,
            threshold_sd=2.5,
        )
        cohort_adequacy_checks[str(lid)] = ad_chk
        if not ad_chk["model_adequacy_demonstrated"]:
            cohort_model_adequacy_demonstrated = False
            cohort_inadequacy_reasons.append(f"Lot {lid}: {ad_chk['reason']}")

        hybrid_lot_curves[str(lid)] = {
            "mean_pct": mean_pct,
            "u95_one_sided_pct": u95_one_sided_pct,
            "unclipped_shelf_life_months": round(sl_unclipped, 2),
            "capped_shelf_life_months": round(min(sl_unclipped, allowed_capped_max_months), 2),
            "model_adequacy_demonstrated": ad_chk["model_adequacy_demonstrated"],
            "model_adequacy_status": ad_chk["status"],
        }

    # Governing hybrid lot (shortest shelf life across the registration cohort, per ICH Q1E)
    worst_hybrid_lid = min(
        hybrid_lot_curves.keys(), key=lambda k: hybrid_lot_curves[k]["unclipped_shelf_life_months"]
    )
    gov_hyb = hybrid_lot_curves[worst_hybrid_lid]
    hyb_unclipped_m = float(gov_hyb["unclipped_shelf_life_months"])
    hyb_capped_m = min(hyb_unclipped_m, allowed_capped_max_months)

    lin_unclipped_m = float(ancova_res["unclipped_linear_q1e_shelf_life_months"])
    lin_capped_m = min(lin_unclipped_m, allowed_capped_max_months)

    extrapolation_guard_triggered = bool(
        hyb_unclipped_m > allowed_capped_max_months or lin_unclipped_m > allowed_capped_max_months
    )
    extrapolation_warning = (
        f"ICH Q1E Extrapolation Guard Active: Observed real-time data at {temperature_c:.0f} °C extends to "
        f"{max_obs_months:.0f} months. Unclipped kinetic projections ({hyb_unclipped_m:.1f}M hybrid / "
        f"{lin_unclipped_m:.1f}M linear ANCOVA) are capped at {allowed_capped_max_months:.1f} months "
        f"(+{max_extrapolation_beyond_observed_months:.0f}M past observed window) pending real-time ICH Q5C pulls."
    )

    df_curves = pd.DataFrame(
        {
            "Month": np.round(months_grid, 2),
            "Hybrid_Governing_Mean_Pct": np.round(gov_hyb["mean_pct"], 4),
            "Hybrid_Governing_OneSided_95UB_Pct": np.round(gov_hyb["u95_one_sided_pct"], 4),
            "Linear_Q1E_Governing_Mean_Pct": np.round(ancova_res["governing_mean_curve_pct"], 4),
            "Linear_Q1E_Governing_OneSided_95UB_Pct": np.round(ancova_res["governing_upper_95_confidence_pct"], 4),
            "Spec_Limit_HMW_Pct": float(spec_limit_hmw_pct),
            "Max_Observed_Data_Month": max_obs_months,
            "Max_Permitted_Extrapolation_Month": allowed_capped_max_months,
        }
    )

    return {
        "cohort_lot_ids": lot_ids,
        "temperature_c": temperature_c,
        "spec_limit_hmw_pct": spec_limit_hmw_pct,
        "max_observed_months": max_obs_months,
        "max_extrapolation_beyond_observed_months": max_extrapolation_beyond_observed_months,
        "allowed_capped_max_shelf_life_months": allowed_capped_max_months,
        "model_adequacy_demonstrated": cohort_model_adequacy_demonstrated,
        "shelf_life_status": "model adequacy demonstrated" if cohort_model_adequacy_demonstrated else "model adequacy not demonstrated",
        "model_adequacy_warning": "; ".join(cohort_inadequacy_reasons) if not cohort_model_adequacy_demonstrated else None,
        "cohort_adequacy_checks": cohort_adequacy_checks,
        "shelf_life_validity_statement": (
            "Extrapolation guard enforces strict capping at observed + 6 months per ICH Q1E; "
            "late-pull residual checks evaluate ongoing model adequacy."
        ),
        "extrapolation_guard_triggered": extrapolation_guard_triggered,
        "extrapolation_warning": extrapolation_warning,
        "ich_q5c_q1e_regulatory_note": reg_note,
        "hybrid_shelf_life_summary": {
            "governing_lot_id": worst_hybrid_lid,
            "unclipped_shelf_life_months": hyb_unclipped_m,
            "regulatory_capped_shelf_life_months": hyb_capped_m,
            "per_lot_shelf_life_months": {
                k: {
                    "unclipped_months": v["unclipped_shelf_life_months"],
                    "capped_months": v["capped_shelf_life_months"],
                }
                for k, v in hybrid_lot_curves.items()
            },
        },
        "linear_ancova_q1e_summary": {
            "f_stat_slope_equality": ancova_res["f_stat_slope_equality"],
            "p_value_slope_equality": ancova_res["p_value_slope_equality"],
            "f_stat_intercept_equality": ancova_res["f_stat_intercept_equality"],
            "p_value_intercept_equality": ancova_res["p_value_intercept_equality"],
            "pooling_decision": ancova_res["pooling_decision"],
            "governing_lot_id": ancova_res["governing_lot_id"],
            "unclipped_shelf_life_months": lin_unclipped_m,
            "regulatory_capped_shelf_life_months": lin_capped_m,
            "per_lot_linear_summaries": ancova_res["per_lot_linear_summaries"],
        },
        "df_curves": df_curves,
        "df_cohort_observations": df_cohort[df_cohort["temperature_c"] == temperature_c].copy(),
    }


def evaluate_shelf_life_vs_truth_on_test_lots(
    temperature_c: float = 5.0,
    spec_limit_hmw_pct: float = 2.00,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    obs_csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_observations.csv",
    truth_json_path: Path | str = DEFAULT_SYNTH_DIR / "_ground_truth_do_not_read_in_train.json",
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Check #4 Deliverable:
    For each of the 100 held-out test lots:
    1. Computes the true time to reach the `2.0%` HMW spec at `5 °C` from the generator's 3-state Radau
       ODE (`integrate_ground_truth_3state_radau` evaluated up to `120` months).
    2. Computes the per-lot classic ICH Q1E linear regression one-sided 95% upper confidence bound
       crossing time (`unclipped_linear_q1e_months`) on the `0..24M` 5 °C observations of that lot.
    3. Computes the per-lot Hybrid 25-member ensemble one-sided 95% upper confidence bound crossing
       time (`unclipped_hybrid_95ub_months`, before the 30-month cap).
    4. Reports how often each estimate exceeds the true time (`unsafe`: `est_months > true_months`),
       with exact Clopper-Pearson 95% intervals, MAE, and mean signed error, broken down by
       `overall`, `matched`, `mismatched`, `mismatched_ps80_only`, and `mismatched_full`, both before
       and after the `30.0`-month (`observed + 6M`) regulatory cap.
    """
    if cfg is None:
        cfg = load_hybrid_config()
    max_extrap_m = float(
        cfg["assay_and_decision_thresholds"].get("q1e_max_extrapolation_months_beyond_obs", 6.0)
    )
    reg_note = str(
        cfg["assay_and_decision_thresholds"].get(
            "q1e_biologics_regulatory_note", ICH_Q5C_Q1E_REGULATORY_NOTE
        )
    )

    df_test = load_observations_for_training(obs_csv_path, allowed_splits=("test",))
    lots_truth = load_ground_truth_for_verification_only(truth_json_path)["lots"]

    mdir = Path(model_dir)
    weights_list, f_mean, f_std = load_exported_ensemble_npz(mdir / "hybrid_ensemble_weights.npz")
    cal_path = mdir / "calibration_bundle.json"
    if cal_path.exists():
        cal_bundle = json.loads(cal_path.read_text(encoding="utf-8"))
        gamma_lot = float(cal_bundle.get("gamma_lot_trajectory", 0.08))
    else:
        gamma_lot = 0.08

    z_95_one_sided = float(scipy_norm.ppf(0.95))
    months_grid = np.linspace(0.0, 120.0, 481)
    hours_grid = months_grid * 730.5

    records = []
    for lid in sorted(df_test["lot_id"].unique()):
        lid_str = str(lid)
        lt = lots_truth[lid_str]
        l_sub_5c = df_test[(df_test["lot_id"] == lid_str) & (df_test["temperature_c"] == temperature_c)].sort_values("time_months")
        max_obs_m = float(l_sub_5c["time_months"].max())
        cap_m = max_obs_m + max_extrap_m

        # 1. Generator 3-state Radau truth at 5C up to 120M
        radau_out = integrate_ground_truth_3state_radau(
            hours_grid, [0.0, float(hours_grid[-1])], [temperature_c, temperature_c], lt
        )
        true_curve_pct = radau_out["hmw_fraction"] * 100.0
        true_sl_m = _find_crossing_time_months(months_grid, true_curve_pct, spec_limit_hmw_pct)

        # 2. Classic ICH Q1E linear regression one-sided 95% UB on this lot's 0..24M 5C pulls
        tl = l_sub_5c["time_months"].to_numpy(dtype=np.float64)
        yl = l_sub_5c["hmw_obs_pct"].to_numpy(dtype=np.float64)
        nl = len(yl)
        Xl = np.column_stack([np.ones(nl), tl])
        bl, _, _, _ = np.linalg.lstsq(Xl, yl, rcond=None)
        y_hat_l = bl[0] + bl[1] * months_grid
        s2_l = float(np.sum((yl - Xl @ bl) ** 2)) / max(nl - 2, 1)
        t_crit_l = float(scipy_t.ppf(0.95, df=max(nl - 2, 1)))
        t_mean_l = float(np.mean(tl))
        ss_t_l = max(float(np.sum((tl - t_mean_l) ** 2)), 1e-12)
        se_mean_l = np.sqrt(s2_l * (1.0 / nl + ((months_grid - t_mean_l) ** 2) / ss_t_l))
        u95_lin = y_hat_l + t_crit_l * se_mean_l
        lin_unclipped_m = _find_crossing_time_months(months_grid, u95_lin, spec_limit_hmw_pct)
        lin_capped_m = min(lin_unclipped_m, cap_m)

        # 3. Hybrid 25-member ensemble one-sided 95% UB on this lot
        cov = l_sub_5c.iloc[0].to_dict()
        x_norm, _, _ = extract_feature_matrix(cov, feature_mean=f_mean, feature_std=f_std)
        hmw0_pct = float(cov["initial_hmw_pct"])
        hmw0_frac = hmw0_pct / 100.0
        lmw0_frac = float(cov["initial_lmw_pct"]) / 100.0

        m_curves = []
        for mw in weights_list:
            mp = predict_params_from_npz_member(x_norm[0], mw, cfg=cfg)
            single_p = {k: float(np.asarray(mp[k]).item()) for k in mp}
            cf = closed_form_isothermal_trajectory(
                hours_grid, temperature_c, params=single_p,
                initial_hmw_fraction=hmw0_frac, initial_lmw_fraction=lmw0_frac, include_lmw=True
            )
            m_curves.append(cf["hmw_fraction"] * 100.0)
        mat = np.stack(m_curves, axis=0)
        mean_pct = np.mean(mat, axis=0)
        epistemic_sd_pct = np.std(mat, axis=0, ddof=1)
        delta_pp = np.maximum(0.0, mean_pct - hmw0_pct)
        traj_sd_pp = np.sqrt(epistemic_sd_pct ** 2 + (gamma_lot * delta_pp) ** 2)
        u95_hyb = mean_pct + z_95_one_sided * traj_sd_pp
        hyb_unclipped_m = _find_crossing_time_months(months_grid, u95_hyb, spec_limit_hmw_pct)
        hyb_capped_m = min(hyb_unclipped_m, cap_m)

        # 4. Late-pull residual / model-adequacy check on this lot's 5C observed pulls
        tl_h = tl * 730.5
        m_curves_obs = []
        for mw in weights_list:
            mp = predict_params_from_npz_member(x_norm[0], mw, cfg=cfg)
            single_p = {k: float(np.asarray(mp[k]).item()) for k in mp}
            cf_obs = closed_form_isothermal_trajectory(
                tl_h, temperature_c, params=single_p,
                initial_hmw_fraction=hmw0_frac, initial_lmw_fraction=lmw0_frac, include_lmw=True
            )
            m_curves_obs.append(cf_obs["hmw_fraction"] * 100.0)
        y_hat_hyb_obs = np.mean(np.stack(m_curves_obs, axis=0), axis=0)

        ad_chk = check_late_pull_model_adequacy(
            observed_times_months=tl,
            observed_hmw_pct=yl,
            predicted_mean_hmw_pct=y_hat_hyb_obs,
            sigma_sec_pct=float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"]),
            threshold_sd=2.5,
        )

        records.append(
            {
                "lot_id": lid_str,
                "structural_group": lt["structural_group"],
                "structural_subgroup": lt["structural_subgroup"],
                "true_shelf_life_months": true_sl_m,
                "linear_unclipped_months": lin_unclipped_m,
                "linear_capped_months": lin_capped_m,
                "hybrid_unclipped_months": hyb_unclipped_m,
                "hybrid_capped_months": hyb_capped_m,
                "linear_unclipped_unsafe": bool(lin_unclipped_m > true_sl_m),
                "linear_capped_unsafe": bool(lin_capped_m > true_sl_m),
                "hybrid_unclipped_unsafe": bool(hyb_unclipped_m > true_sl_m),
                "hybrid_capped_unsafe": bool(hyb_capped_m > true_sl_m),
                "model_adequacy_demonstrated": bool(ad_chk["model_adequacy_demonstrated"]),
                "model_adequacy_reason": ad_chk["reason"],
            }
        )

    df_sl = pd.DataFrame(records)
    summary_by_group: Dict[str, Any] = {}
    for g_label, sub in (
        ("overall", df_sl),
        ("matched", df_sl[df_sl["structural_group"] == "matched"]),
        ("mismatched", df_sl[df_sl["structural_group"] == "mismatched"]),
        ("mismatched_ps80_only", df_sl[df_sl["structural_subgroup"] == "mismatched_ps80_only"]),
        ("mismatched_full", df_sl[df_sl["structural_subgroup"] == "mismatched_full"]),
    ):
        n_g = len(sub)
        if n_g == 0:
            continue

        def _summarize_est(col_est: str, col_unsafe: str) -> Dict[str, Any]:
            k_u = int(sub[col_unsafe].sum())
            l95, u95 = clopper_pearson_interval(k_u, n_g)
            diff = sub[col_est] - sub["true_shelf_life_months"]
            ratio = (sub[col_est] / sub["true_shelf_life_months"]).to_numpy(dtype=np.float64)
            p25 = float(np.percentile(ratio, 25))
            p75 = float(np.percentile(ratio, 75))
            return {
                "n_lots": n_g,
                "unsafe_exceedance_count": k_u,
                "unsafe_k_over_n": f"{k_u}/{n_g}",
                "unsafe_rate_pct": round(100.0 * k_u / n_g, 2),
                "unsafe_clopper_pearson_95_ci_pct": [round(100.0 * l95, 2), round(100.0 * u95, 2)],
                "mean_estimated_months": round(float(sub[col_est].mean()), 2),
                "mean_true_months": round(float(sub["true_shelf_life_months"].mean()), 2),
                "mae_vs_truth_months": round(float(np.abs(diff).mean()), 2),
                "mean_signed_bias_months": round(float(diff.mean()), 2),
                "ratio_estimated_over_true_median": round(float(np.median(ratio)), 4),
                "ratio_estimated_over_true_p25": round(p25, 4),
                "ratio_estimated_over_true_p75": round(p75, 4),
                "ratio_estimated_over_true_iqr": round(p75 - p25, 4),
            }

        k_adeq = int(sub["model_adequacy_demonstrated"].sum())
        k_inadeq = n_g - k_adeq
        l_ad, u_ad = clopper_pearson_interval(k_inadeq, n_g)

        summary_by_group[g_label] = {
            "n_lots": n_g,
            "model_adequacy_demonstrated_count": k_adeq,
            "model_adequacy_not_demonstrated_count": k_inadeq,
            "model_adequacy_not_demonstrated_rate_pct": round(100.0 * k_inadeq / n_g, 2),
            "model_adequacy_not_demonstrated_clopper_pearson_95_ci_pct": [round(100.0 * l_ad, 2), round(100.0 * u_ad, 2)],
            "mean_true_shelf_life_months": round(float(sub["true_shelf_life_months"].mean()), 2),
            "min_true_shelf_life_months": round(float(sub["true_shelf_life_months"].min()), 2),
            "max_true_shelf_life_months": round(float(sub["true_shelf_life_months"].max()), 2),
            "classic_linear_q1e_before_30m_cap": _summarize_est("linear_unclipped_months", "linear_unclipped_unsafe"),
            "hybrid_95ub_before_30m_cap": _summarize_est("hybrid_unclipped_months", "hybrid_unclipped_unsafe"),
            "classic_linear_q1e_after_30m_cap": _summarize_est("linear_capped_months", "linear_capped_unsafe"),
            "hybrid_95ub_after_30m_cap": _summarize_est("hybrid_capped_months", "hybrid_capped_unsafe"),
        }

    return {
        "temperature_c": temperature_c,
        "spec_limit_hmw_pct": spec_limit_hmw_pct,
        "regulatory_extrapolation_cap_months_beyond_observed": max_extrap_m,
        "ich_q5c_q1e_regulatory_note": reg_note,
        "shelf_life_validity_statement": (
            "Extrapolation guard enforces strict capping at observed + 6 months per ICH Q1E; "
            "late-pull residual checks evaluate ongoing model adequacy."
        ),
        "by_group": summary_by_group,
    }
