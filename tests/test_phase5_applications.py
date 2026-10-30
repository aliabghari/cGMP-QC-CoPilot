"""
Phase 5 Unit Tests: Thermal Excursion Evaluator (`excursion.py`), Dynamic OOT Stability Band &
Early-Pull Lot Adaptation (`oot.py`), ICH Q1E Shelf-Life & Linear ANCOVA Poolability (`q1e.py`),
and Backward-Compatible Wrappers (`eyring_excursion.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.agent_2_analytical_scientist.scripts.eyring_excursion import (
    analyze_6m_oot_stability,
    calculate_eyring_rate,
    eyring_rate_constant_per_hour,
    simulate_cold_chain_excursion,
    simulate_excursion,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.excursion import (
    PRIOR_DRIVEN_FU_CAVEAT,
    SEC_REVERSIBLE_VS_IRREVERSIBLE_CAVEAT,
    evaluate_thermal_excursion,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.oot import (
    evaluate_lot_oot_with_adaptation,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.q1e import (
    ICH_Q5C_Q1E_REGULATORY_NOTE,
    evaluate_q1e_shelf_life_comparison,
    run_ich_q1e_linear_ancova_poolability,
)


def test_excursion_verdict_wording_caveats_and_backward_compatibility():
    """
    Verifies:
    - Verdict is strictly one of `"no detectable change"`, `"inconclusive"`, `"detectable change likely"`.
    - Never outputs `"RELEASE FOR USE"` or disposition language.
    - Includes both `SEC_REVERSIBLE_VS_IRREVERSIBLE_CAVEAT` and `PRIOR_DRIVEN_FU_CAVEAT`.
    - Backward-compatible wrapper `simulate_cold_chain_excursion` preserves all keys required
      by `app.py` and `comms_generator.py`.
    - `eyring_rate_constant_per_hour` uses `delta_s_j_mol_k` (fixing the legacy dead parameter bug).
    """
    k_s1 = eyring_rate_constant_per_hour(28.0, delta_h_kj_mol=38.0, delta_s_j_mol_k=-241.0)
    k_s2 = eyring_rate_constant_per_hour(28.0, delta_h_kj_mol=38.0, delta_s_j_mol_k=-230.0)
    assert k_s2 > k_s1
    assert calculate_eyring_rate(28.0, 38.0, -241.0) == pytest.approx(k_s1, rel=1e-12)

    exc_36h = simulate_cold_chain_excursion(baseline_temp_c=5.0, excursion_temp_c=28.0, excursion_duration_hours=36.0)
    s_36 = exc_36h["summary"]
    assert s_36["verdict"] in ("no detectable change", "inconclusive", "detectable change likely")
    assert s_36["verdict"] == "no detectable change"
    assert "RELEASE FOR USE" not in s_36["disposition_recommendation"]
    assert s_36["sec_reversible_vs_irreversible_caveat"] == SEC_REVERSIBLE_VS_IRREVERSIBLE_CAVEAT
    assert s_36["prior_driven_f_u_caveat"] == PRIOR_DRIVEN_FU_CAVEAT

    # Check backward-compatible keys
    for req_key in (
        "initial_hmw_pct",
        "post_excursion_hmw_hybrid_pct",
        "post_excursion_hmw_naive_pct",
        "delta_hmw_excursion_pct",
        "max_ann_tvp_factor",
        "equivalent_4c_days_consumed",
        "shelf_life_deduction_months",
        "projected_remaining_shelf_life_months",
        "disposition_recommendation",
    ):
        assert req_key in s_36

    # Check severe 14d @ 40C excursion triggers 'detectable change likely'
    exc_14d = evaluate_thermal_excursion(excursion_temp_c=40.0, excursion_duration_hours=336.0)
    assert exc_14d["summary"]["verdict"] == "detectable change likely"


def test_oot_early_pull_adaptation_on_demo_lot_honest_reporting():
    """
    Verifies:
    - `analyze_6m_oot_stability()` uses ODE adaptation from 0/1/3M pulls on `NS-2026-004`
      with no hard-coded forecast arrays.
    - Reports honestly whether `1.58%` at 6M 25°C exceeds the adapted 95% PI upper bound while
      remaining within the `2.0%` specification limit.
    - Preserves all keys expected by `app.py` and `comms_generator.py`.
    """
    oot_out = analyze_6m_oot_stability()
    df_proj = oot_out["df_projection"]
    diag = oot_out["oot_diagnostics"]
    adapt = oot_out["adaptation_summary"]

    assert adapt["lot_id"] == "NS-2026-004"
    assert adapt["prior_dG_eff_mean_kj_mol"] != adapt["posterior_dG_eff_mean_kj_mol"]

    # Ensure legacy hard-coded [0.45, 0.59, 0.86, 1.54, 2.28, 3.15] is NOT present
    forecast_vals = df_proj["Hybrid_Eyring_ANN_25C_Forecast_Pct"].to_numpy(dtype=np.float64)
    assert not np.allclose(forecast_vals, [0.45, 0.59, 0.86, 1.54, 2.28, 3.15])
    assert forecast_vals[0] == pytest.approx(0.62, abs=1e-6)

    # Verify honest flag logic and Demo-Lot inconclusive verdict
    assert diag["observed_6m_25c_hmw_pct"] == pytest.approx(1.58, abs=1e-6)
    expected_flag = bool(diag["observed_6m_25c_hmw_pct"] > diag["adapted_6m_25c_pi95_upper_pct"])
    assert diag["oot_flagged_by_adapted_pi95"] is expected_flag
    assert diag["within_specification_at_6m"] is True
    assert diag["oot_verdict"] == "inconclusive: model and data disagree; treat as potential OOT and investigate"
    assert diag["treat_as_potential_oot"] is True

    # Check #1: Verify early-pull residuals in SEC-SD units and prior-scale sensitivity table (x0.5, x1, x2, x4)
    assert len(oot_out["early_pull_residuals_0_1_3m"]) == 6
    sens_table = oot_out["prior_scale_sensitivity_table"]
    assert [r["prior_scale_label"] for r in sens_table] == ["x0.5", "x1", "x2", "x4"]
    assert "prior-dominated" in diag["prior_dominance_note"]
    assert "prior-dominated" in diag["finding"]


def test_q1e_shelf_life_ancova_poolability_and_extrapolation_guard():
    """
    Verifies:
    - One-sided 95% upper confidence bound shelf-life estimation alongside classic ICH Q1E
      linear regression with ANCOVA batch-poolability (`p = 0.25`).
    - Extrapolation guard caps shelf-life claims beyond `max_observed_months + 6.0` months.
    - Explicit regulatory note stating ICH Q5C is primary for biologics.
    """
    q1e_out = evaluate_q1e_shelf_life_comparison()
    assert q1e_out["ich_q5c_q1e_regulatory_note"] == ICH_Q5C_Q1E_REGULATORY_NOTE
    assert q1e_out["max_observed_months"] == 24.0
    assert q1e_out["allowed_capped_max_shelf_life_months"] == 30.0
    assert "model_adequacy_demonstrated" in q1e_out
    assert q1e_out["shelf_life_status"] in ("model adequacy demonstrated", "model adequacy not demonstrated")
    assert "shelf_life_validity_statement" in q1e_out

    hyb_s = q1e_out["hybrid_shelf_life_summary"]
    lin_s = q1e_out["linear_ancova_q1e_summary"]

    assert hyb_s["regulatory_capped_shelf_life_months"] <= 30.0
    assert lin_s["regulatory_capped_shelf_life_months"] <= 30.0
    assert "p_value_slope_equality" in lin_s
    assert "p_value_intercept_equality" in lin_s
    assert any(
        lin_s["pooling_decision"].startswith(prefix)
        for prefix in (
            "pool_slopes_and_intercepts",
            "common_slope_separate_intercepts",
            "separate_slopes_separate_intercepts",
        )
    )
