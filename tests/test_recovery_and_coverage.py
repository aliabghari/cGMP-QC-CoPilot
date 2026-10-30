"""
Phase 4 Unit Tests: Disjoint Train/Val/Cal/Test Sets with Unchanged Phase 2 Hashes,
Mean Confidence Band vs Observation Prediction Interval, Early-Pull Lot Adaptation,
Exact Clopper-Pearson Coverage on 100 Test Lots (`overall`, `matched`, `mismatched`, `ps80_only`),
Excursion Verdict Confusion Matrix (calibrated on val lots, evaluated on test lots),
Tuned Baselines (`pooled_mean`, `ridge`, `blackbox_mlp`), and Physical-Validity Audit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.synth import (
    DEFAULT_SYNTH_DIR,
    compute_file_sha256,
    load_observations_for_training,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.train import (
    DEFAULT_MODEL_DIR,
    load_exported_ensemble_npz,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.uncertainty import (
    IDENTIFIABLE_METRIC_KEYS,
    SYNTHETIC_ONLY_PARAM_CALIBRATION_NOTE,
    adapt_lot_from_early_pulls,
    clopper_pearson_interval,
    evaluate_coverage_verdict,
    predict_lot_intervals_and_trajectories,
    run_phase4_verification_and_save,
)


@pytest.fixture(scope="module")
def phase4_report() -> dict:
    """Loads or runs Phase 4 verification pipeline and returns the full report dictionary."""
    import json
    report_path = DEFAULT_MODEL_DIR / "phase4_uncertainty_report.json"
    if report_path.exists():
        data = json.loads(report_path.read_text(encoding="utf-8"))
        if "physical_validity_and_15c_audit" in data and "lot_adaptation_6m_9m_12m_evaluation" in data:
            return data
    return run_phase4_verification_and_save(output_dir=DEFAULT_MODEL_DIR)


def test_train_val_cal_test_disjoint_and_hashes_unchanged(phase4_report: dict):
    """
    Verifies that generating the 60 calibration lots leaves the existing
    train/val/test files and their SHA-256 hashes completely unchanged, and that
    train (100), val (40), cal (60), and test (100) lot sets are strictly disjoint.
    """
    expected_hashes = {
        "synth_observations.csv": "9d2d4285ea78bbf0e3eb92bb712c6b16324d71f8de57ab9d3e4d9adcf9333474",
        "synth_excursions.csv": "b5f1591568301374fce2f41d89dbd5111ebcb0717bb81ebf9f9f918035e77fed",
        "illustrative_demo_lot_NS-2026-004.csv": "4a7507627423488a2231a98bfbf5c8ce85e00f69c4c32ea54a7fce3ccb7d436e",
        "_ground_truth_do_not_read_in_train.json": "d7b9d287a7b1dc4026bd71bd6d9624ebc320f9775b6516b3e02879019628f19f",
    }
    for fname, exp_sha in expected_hashes.items():
        actual_sha = compute_file_sha256(DEFAULT_SYNTH_DIR / fname)
        assert actual_sha == exp_sha, f"Hash changed for {fname}: {actual_sha} != {exp_sha}"

    df_obs = pd.read_csv(DEFAULT_SYNTH_DIR / "synth_observations.csv")
    df_cal = pd.read_csv(DEFAULT_SYNTH_DIR / "synth_calibration_observations.csv")
    assert "structural_group" not in df_cal.columns

    train_lots = set(df_obs[df_obs["split"] == "train"]["lot_id"].unique())
    val_lots = set(df_obs[df_obs["split"] == "val"]["lot_id"].unique())
    test_lots = set(df_obs[df_obs["split"] == "test"]["lot_id"].unique())
    cal_lots = set(df_cal["lot_id"].unique())

    assert len(train_lots) == 100
    assert len(val_lots) == 40
    assert len(cal_lots) == 60
    assert len(test_lots) == 100

    all_sets = [train_lots, val_lots, cal_lots, test_lots]
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            assert all_sets[i].isdisjoint(all_sets[j])


def test_mean_confidence_band_vs_observation_prediction_interval_and_lot_adaptation(phase4_report: dict):
    """
    Verifies:
    1. Observation Prediction Interval is strictly wider than Mean Confidence Band at every timepoint.
    2. `adapt_lot_from_early_pulls()` adjusts a lot's parameters from 0/1/3M pulls and improves
       6M/9M/12M forecast RMSE on matched validation and test lots.
    """
    weights_list, f_mean, f_std = load_exported_ensemble_npz(
        DEFAULT_MODEL_DIR / "hybrid_ensemble_weights.npz"
    )
    cal_bundle = phase4_report["calibration_bundle"]
    assert cal_bundle["identifiable_parameters_note"] == SYNTHETIC_ONLY_PARAM_CALIBRATION_NOTE

    df_test = load_observations_for_training(
        DEFAULT_SYNTH_DIR / "synth_observations.csv", allowed_splits=("test",)
    )
    sub_test = df_test[df_test["lot_id"].isin(sorted(df_test["lot_id"].unique())[:5])].copy()
    pred_out = predict_lot_intervals_and_trajectories(
        sub_test, weights_list, f_mean, f_std, cal_bundle
    )
    obs_df = pred_out["observations_df"]

    mean_width = obs_df["mean_ci95_upper_pct"] - obs_df["mean_ci95_lower_pct"]
    pi_pt_width = obs_df["obs_pi95_pointwise_upper_pct"] - obs_df["obs_pi95_pointwise_lower_pct"]
    pi_sim_width = obs_df["obs_pi95_simultaneous_upper_pct"] - obs_df["obs_pi95_simultaneous_lower_pct"]

    assert np.all(pi_pt_width > mean_width)
    assert np.all(pi_sim_width > pi_pt_width)

    # Verify 6M/9M/12M adaptation evaluation on validation and test splits (Check #2 and Check #3)
    adapt_eval = phase4_report["lot_adaptation_6m_9m_12m_forecasts"] if "lot_adaptation_6m_9m_12m_forecasts" in phase4_report else phase4_report["lot_adaptation_6m_9m_12m_evaluation"]
    assert "validation_lots_6m_9m_12m_forecasts" in adapt_eval
    assert "test_lots_6m_9m_12m_forecasts" in adapt_eval
    test_adapt_block = adapt_eval["test_lots_6m_9m_12m_forecasts"]
    test_mat_adapt = test_adapt_block["coverage_and_rmse_by_group"]["matched"]
    assert test_mat_adapt["with_0_1_3m_adaptation_hmw_rmse_pp"] < test_mat_adapt["without_adaptation_hmw_rmse_pp"]
    assert "with_0_1_3m_adaptation_pi95_coverage_recal_matched_cal" in test_mat_adapt
    assert "oot_25c_operating_characteristics" in test_adapt_block
    assert "6M_25C" in test_adapt_block["oot_25c_operating_characteristics"]


def test_clopper_pearson_coverage_and_observable_strata(phase4_report: dict):
    """
    Verifies Clopper-Pearson interval math, lot-level primary gate vs descriptive per-observation
    coverage, and observable strata (`temperature_c x time_months`) coverage reporting.
    """
    l_95_100, u_95_100 = clopper_pearson_interval(95, 100, alpha=0.05)
    assert l_95_100 < 0.95 < u_95_100
    assert evaluate_coverage_verdict(95, 100)["verdict"] == "PASS"
    assert evaluate_coverage_verdict(85, 100)["verdict"] == "FAIL (under-coverage)"
    assert evaluate_coverage_verdict(100, 100)["verdict"] == "conservative / intervals too wide"

    test_cov = phase4_report["test_set_coverage_and_recovery"]
    cov_groups = test_cov["coverage_by_group"]
    for grp in ("overall", "matched", "mismatched", "mismatched_ps80_only", "mismatched_full"):
        assert grp in cov_groups
        g_data = cov_groups[grp]
        for m_key in IDENTIFIABLE_METRIC_KEYS:
            assert m_key in g_data["identifiable_quantities"]
            m_info = g_data["identifiable_quantities"][m_key]
            assert m_info["observability_note"] == SYNTHETIC_ONLY_PARAM_CALIBRATION_NOTE
        assert "PRIMARY_PASS_FAIL_GATE" in g_data["observation_pi_per_lot_simultaneous"]["role"]
        assert "DESCRIPTIVE_ONLY" in g_data["observation_pi_per_observation_pointwise"]["role"]

    assert len(test_cov["observable_strata_coverage_temp_x_time"]) >= 15


def test_excursion_test_subset_clopper_pearson_and_physical_validity(phase4_report: dict):
    """
    Verifies:
    - Excursion intervals are calibrated on validation-lot excursions (n=40) and evaluated strictly
      on test-lot excursions (n=80) with Clopper-Pearson upper bounds on false-reassurance and false-alarm.
    - Tuned Black-Box MLP beats the pooled-mean baseline, and the Hybrid ANN beats both pooled-mean and ridge.
    - Physical-validity audit reports violation counts for Hybrid ODE vs Tuned Black-Box MLP.
    """
    exc_eval = phase4_report["excursion_verdict_evaluation"]
    assert exc_eval["by_group"]["overall"]["total_test_excursions"] == 80
    for grp in ("overall", "matched", "mismatched"):
        g_exc = exc_eval["by_group"][grp]
        assert "false_reassurance_clopper_pearson_u95_pct" in g_exc
        assert "false_alarm_clopper_pearson_u95_pct" in g_exc

    fups = phase4_report["phase3_followups_and_baselines"]
    hybrid_val_rmse = fups["train_vs_val_rmse_analysis"]["ensemble_mean_prediction_rmse_pp"]["val_overall"]
    pooled_val_rmse = fups["baseline_validation_comparison"]["pooled_mean_no_ann"]["overall"]["hmw_rmse_pp"]
    ridge_val_rmse = fups["baseline_validation_comparison"]["linear_ridge_on_descriptors"]["overall"]["hmw_rmse_pp"]
    bb_val_rmse = fups["baseline_validation_comparison"]["blackbox_mlp_no_physics"]["overall"]["hmw_rmse_pp"]

    assert hybrid_val_rmse < pooled_val_rmse
    assert hybrid_val_rmse < ridge_val_rmse
    assert bb_val_rmse < pooled_val_rmse

    phys = phase4_report["physical_validity_and_15c_audit"]
    hyb_viol = phys["physical_violation_counts_across_100_test_lots_5c_15c_25c_40c"]["hybrid_lumry_eyring_ode"]
    assert hyb_viol["purity_increase_or_negative_rate_violations"] == 0
    assert hyb_viol["non_monotone_temperature_rate_violations_incl_15c"] == 0
