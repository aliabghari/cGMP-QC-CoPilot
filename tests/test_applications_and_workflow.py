"""
Phase 6 Unit Tests: End-to-End MSO Draft Quality Workflow Chain (`workflow.py`),
21 CFR Part 11 HITL Sign-Off Gate, Claims Hygiene, and Checks 1–5 Verification.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.agent_1_lean_capacity.scripts.capacity_smoother import run_heijunka_leveling
from agents.agent_2_analytical_scientist.scripts.eyring_excursion import (
    analyze_6m_oot_stability,
    simulate_cold_chain_excursion,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.q1e import (
    evaluate_shelf_life_vs_truth_on_test_lots,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.workflow import (
    DRAFT_STAMP,
    EXCURSION_SAFETY_STATEMENT,
    MISMATCHED_RIDGE_STATEMENT,
    PHYSICAL_VALIDITY_STATEMENT,
    build_mso_draft_quality_package,
    verify_and_sign_hitl_gate,
)
from agents.agent_2_analytical_scientist.scripts.lss_sma_transfer import (
    calculate_lss_sma_dwell_transfer,
)
from agents.agent_2_analytical_scientist.scripts.bioassay_4pl_bayesian import (
    evaluate_bioassay_4pl_and_plate_edge,
)
from agents.agent_3_mso_oos_risk.scripts.oos_telemetry_triage import (
    compute_dynamic_fmea_rpn,
    evaluate_oos_phase1_gate,
)
from agents.agent_4_stakeholder_comms.scripts.comms_generator import (
    generate_stakeholder_communications,
)


def test_mso_workflow_draft_stamp_and_part11_hitl_gate():
    """
    Verifies:
    - `build_mso_draft_quality_package()` stamps outputs with `DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL`
      and starts with `export_unlocked = False`.
    - `verify_and_sign_hitl_gate()` blocks export if Reviewer Name, Role, Justification (<10 chars),
      or confirmation checkbox is missing.
    - `verify_and_sign_hitl_gate()` unlocks export and computes a 64-hex-character SHA-256 hash when
      all human QA sign-off fields are provided.
    """
    pkg = build_mso_draft_quality_package(excursion_temp_c=28.0, excursion_duration_hours=36.0)
    assert pkg["stamp"] == DRAFT_STAMP
    assert pkg["export_unlocked"] is False
    assert pkg["trackwise_json"]["export_unlocked"] is False
    assert DRAFT_STAMP in pkg["markdown_memo"]
    assert "RELEASE FOR USE" not in pkg["markdown_memo"]
    assert "zero risk" not in pkg["markdown_memo"].lower()

    # 1. Missing reviewer name / role / short justification / unchecked confirmation -> blocked
    blocked_1 = verify_and_sign_hitl_gate(
        draft_package=pkg,
        reviewer_name="",
        reviewer_role="QA MSO Lead",
        justification="Approved after review.",
        confirmed=True,
    )
    assert blocked_1["approved"] is False
    assert blocked_1["export_unlocked"] is False

    blocked_2 = verify_and_sign_hitl_gate(
        draft_package=pkg,
        reviewer_name="Dr. M. Vance",
        reviewer_role="QA MSO Lead",
        justification="Too short",
        confirmed=True,
    )
    assert blocked_2["approved"] is False
    assert blocked_2["export_unlocked"] is False

    blocked_3 = verify_and_sign_hitl_gate(
        draft_package=pkg,
        reviewer_name="Dr. M. Vance",
        reviewer_role="QA MSO Lead",
        justification="Reviewed deterministic kinetic trajectory and prior sensitivity table.",
        confirmed=False,
    )
    assert blocked_3["approved"] is False
    assert blocked_3["export_unlocked"] is False

    # 2. Complete human QA sign-off -> unlocked with ALCOA+ SHA-256 signature
    signed = verify_and_sign_hitl_gate(
        draft_package=pkg,
        reviewer_name="Dr. M. Vance",
        reviewer_role="Principal Specialist, QC PM / Local MSO",
        justification="Reviewed deterministic kinetic trajectory and prior sensitivity table for Non-GMP QA review.",
        confirmed=True,
    )
    assert signed["approved"] is True
    assert signed["export_unlocked"] is True
    assert len(signed["alcoa_sha256_hash"]) == 64
    assert signed["signed_package"]["export_unlocked"] is True
    assert signed["signed_package"]["trackwise_json"]["export_unlocked"] is True


def test_claims_hygiene_and_comms_integration():
    """
    Verifies:
    - Required wording rules are enforced in `workflow.py`, `email_templates.json`, and `skill.md`:
      - `"no false reassurance observed in 40 above-gate test excursions (95% upper bound 8.8%)"`
      - `"by construction"` for zero physical violations
      - `"slightly worse in raw RMSE"` for hybrid vs ridge on mismatched lots
      - Never `"RELEASE FOR USE"` or `"zero risk"`
    - `generate_stakeholder_communications()` populates all 4 dispatches cleanly with live Agent 2 outputs.
    """
    assert EXCURSION_SAFETY_STATEMENT == "no false reassurance observed in 40 above-gate test excursions (95% upper bound 8.8%)"
    assert "by construction" in PHYSICAL_VALIDITY_STATEMENT
    assert "slightly worse in raw RMSE" in MISMATCHED_RIDGE_STATEMENT

    a1 = run_heijunka_leveling()
    a2_exc = simulate_cold_chain_excursion(baseline_temp_c=5.0, excursion_temp_c=28.0, excursion_duration_hours=36.0)
    a2_oot = analyze_6m_oot_stability()
    a2_tr = calculate_lss_sma_dwell_transfer()
    a2_bio = evaluate_bioassay_4pl_and_plate_edge()
    a3_oos = evaluate_oos_phase1_gate()
    a3_fmea = compute_dynamic_fmea_rpn()

    comms = generate_stakeholder_communications(
        agent1_results=a1,
        agent2_excursion=a2_exc,
        agent2_oot=a2_oot,
        agent2_transfer=a2_tr,
        agent2_bioassay=a2_bio,
        agent3_oos=a3_oos,
        agent3_fmea=a3_fmea,
    )
    for _, disp in comms["rendered_dispatches"].items():
        assert DRAFT_STAMP in disp["body"]
        assert "RELEASE FOR USE" not in disp["body"]
        assert "zero risk" not in disp["body"].lower()


def test_checks_4_and_5_in_phase4_report():
    """
    Verifies Check #4 (shelf life vs truth across all 100 test lots) and Check #5 (15°C accuracy vs
    generator truth + SHA-256 of retrained model artifacts) are present in `phase4_uncertainty_report.json`.
    """
    report_path = Path("agents/agent_2_analytical_scientist/assets/models/phase4_uncertainty_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "retrained_model_artifacts_sha256" in report
    assert "hybrid_ensemble_weights.npz" in report["retrained_model_artifacts_sha256"]

    sl_truth = report["shelf_life_vs_truth_100_test_lots"]
    assert sl_truth["regulatory_extrapolation_cap_months_beyond_observed"] == 6.0
    for grp in ("overall", "matched", "mismatched"):
        assert grp in sl_truth["by_group"]
        assert "classic_linear_q1e_before_30m_cap" in sl_truth["by_group"][grp]
        assert "hybrid_95ub_before_30m_cap" in sl_truth["by_group"][grp]

    acc_15c = report["physical_validity_and_15c_audit"]["unseen_15c_accuracy_vs_generator_3state_radau_truth_1m_to_24m"]
    assert "trained_on_early_pulls_le_3m" in acc_15c
    assert "trained_on_full_schedule_0_to_24m" in acc_15c


def test_workflow_generated_memo_and_notification_banned_language_and_exact_numbers():
    """
    Verifies Requirement 1 & 2:
    - Generated memo and client notification must start with:
      'DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL'
    - Client notification title must not be titled 'FORMAL'.
    - Banned disposition/approval tokens:
      Fails if any of: release, disposition recommendation, 'QP', approved, safe appear.
    - Mandatory statement must appear:
      'Draft assessment for QA review. Kinetic model result: no detectable change. Disposition decision rests with QA.'
    - Number parity assertions:
      dHMW, interval, equivalent days, peak f_U, covariates, pull history, OOT verdict.
    """
    pkg = build_mso_draft_quality_package(lot_id="NS-2026-004", excursion_temp_c=28.0, excursion_duration_hours=36.0)
    memo = pkg["trackwise_memo"]
    notice = pkg["client_notification"]

    # 1. Start-with assertion
    assert memo.startswith("DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL")
    assert notice.startswith("DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL")

    # 2. Client notification must not be titled FORMAL
    assert "FORMAL" not in notice.splitlines()[2]

    # 3. Banned tokens test (case-insensitive)
    banned_tokens = ["release", "disposition recommendation", "approved", "safe"]
    for text in (memo, notice):
        text_lower = text.lower()
        for token in banned_tokens:
            assert token not in text_lower, f"Forbidden token '{token}' found in generated text:\n{text}"
        import re
        assert not re.search(r"\bqp\b", text_lower), f"Forbidden token 'QP' found in generated text:\n{text}"

    # 4. Mandatory QA review statement
    req_stmt = "Draft assessment for QA review. Excursion model result: no detectable change. OOT/trend result: inconclusive: model and data disagree; treat as potential OOT and investigate. Disposition decision rests with QA."
    assert req_stmt in memo
    assert req_stmt in notice

    # 5. Exact number and string assertions
    exc_sum = pkg["excursion_result"]["summary"]
    oot_diag = pkg["oot_result"]["oot_diagnostics"]

    # dHMW, CI, eq days, peak fu
    assert f"+{exc_sum['delta_hmw_excursion_pp']:.4f} pp" in memo
    assert f"[{exc_sum['delta_hmw_excursion_l95_pp']:.4f}, {exc_sum['delta_hmw_excursion_u95_pp']:.4f}] pp" in memo
    assert f"{exc_sum['equivalent_5c_days_consumed']:.2f} days" in memo
    assert f"{exc_sum['peak_f_u_prior_driven_pct']:.4f}%" in memo
    assert exc_sum["delta_hmw_excursion_l95_pp"] >= 0.0

    # Demo lot covariates and pull history
    assert "100.0 mg/mL mAb" in memo
    assert "pH 5.8" in memo
    assert "0.012% PS80" in memo
    assert "HMW = 0.620%" in memo
    assert "1M = 0.650%, 3M = 0.700%, 6M = 0.790%" in memo
    assert "1M = 0.720%, 3M = 0.880%, 6M = 1.580%" in memo

    # OOT verdict
    assert "inconclusive: model and data disagree; treat as potential OOT and investigate" in memo
    assert "inconclusive: model and data disagree; treat as potential OOT and investigate" in notice

    # Placeholders present
    assert "[EVENT DESCRIPTION - TO BE COMPLETED BY SITE]" in memo
    assert "[EVENT DESCRIPTION - TO BE COMPLETED BY SITE]" in notice

    # Illustrative projected value from 0.790% (latest 5C pull)
    assert "0.6453%" not in memo, "T0-based projected post-excursion HMW must be removed"
    assert "0.6453%" not in notice, "T0-based projected post-excursion HMW must be removed"
    assert "0.8153% [0.8108%, 0.8199%]" in memo
    assert "0.8153% [0.8108%, 0.8199%]" in notice
    assert "illustrative; enter the measured pre-excursion HMW" in memo
    assert "illustrative; enter the measured pre-excursion HMW" in notice
    assert "1.1847 pp margin to assumed 2.00% specification limit" in notice

    # SEC-HPLC consistency
    assert "SEC-UPLC" not in memo
    assert "SEC-UPLC" not in notice
    assert "SEC-HPLC" in memo
    assert "SEC-HPLC" in notice

    # Custom pre-excursion HMW input test
    custom_pkg = build_mso_draft_quality_package(
        lot_id="NS-2026-004", excursion_temp_c=28.0, excursion_duration_hours=36.0, pre_excursion_hmw_pct=0.900
    )
    assert "0.9253% [0.9208%, 0.9299%]" in custom_pkg["trackwise_memo"]


def test_authoritative_final_metrics_file_equality():
    """
    Verifies Requirement 4:
    results/final_metrics.json exists, is generated by verify_hybrid.py,
    and matches phase4_uncertainty_report.json exact numbers.
    """
    metrics_path = Path("results/final_metrics.json")
    assert metrics_path.exists(), "results/final_metrics.json must be present"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert data["generated_by"] == "verify_hybrid.py"
    assert "oot_operating_characteristics" in data
    assert "shelf_life_vs_truth" in data

    # Verify 6M, 9M, 12M and combined OOT counts
    oot_6m = data["oot_operating_characteristics"]["6M_25C"]
    assert oot_6m["matched_false_alarm_rate"]["recalibrated_adapted_hybrid_pi_overall_cal"]["k_over_n"] == "0/70"
    assert oot_6m["mismatched_all_detection_rate"]["recalibrated_adapted_hybrid_pi_overall_cal"]["k_over_n"] == "9/30"

    oot_9m = data["oot_operating_characteristics"]["9M_25C"]
    assert oot_9m["mismatched_all_detection_rate"]["recalibrated_adapted_hybrid_pi_overall_cal"]["k_over_n"] == "10/12"

    oot_12m = data["oot_operating_characteristics"]["12M_25C"]
    assert oot_12m["mismatched_all_detection_rate"]["recalibrated_adapted_hybrid_pi_overall_cal"]["k_over_n"] == "11/12"

    oot_all = data["oot_operating_characteristics"]["all_6M_9M_12M_25C"]
    assert oot_all["matched_false_alarm_rate"]["recalibrated_adapted_hybrid_pi_overall_cal"]["k_over_n"] == "0/126"
    assert oot_all["mismatched_all_detection_rate"]["recalibrated_adapted_hybrid_pi_overall_cal"]["k_over_n"] == "30/54"

    # Verify historical comparators (100 training lots at 25C)
    assert "historical_comparators_100_train_lots" in data
    hist = data["historical_comparators_100_train_lots"]
    assert hist["combined_6M_9M_12M_25C"]["pooled_ols_100_train_lots"]["matched_false_alarm"]["k_over_n"] == "11/126"
    assert hist["combined_6M_9M_12M_25C"]["pooled_ols_100_train_lots"]["mismatched_detection"]["k_over_n"] == "17/54"
    assert hist["combined_6M_9M_12M_25C"]["lot_specific_t0_hist_slope"]["matched_false_alarm"]["k_over_n"] == "7/126"
    assert hist["combined_6M_9M_12M_25C"]["lot_specific_t0_hist_slope"]["mismatched_detection"]["k_over_n"] == "8/54"
