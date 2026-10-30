"""
tests/test_wp3_quality_intelligence.py

Comprehensive test suite for WP3 Quality Intelligence:
- Statistical Process Control (I-MR, X-bar/R, p-chart, c-chart, EWMA, CUSUM)
- Nelson / Western Electric Rules 1-4 with plain-language diagnostics
- Phase I vs Phase II limit freezing
- Process Capability (Cp/Cpk) with Chou et al. 95% CIs and normality warnings
- Historical Lessons-Learned Knowledge Base (seeding, TF-IDF search, audit logging)
- Proactive FMEA (RPN arithmetic, Action Priority, before/after, sensitivity, occurrence link)
- Reactive CAPA state machine (lifecycle transitions, closure gating, overdue radar)
- Regulatory claims hygiene and draft stamp enforcement across all outputs
"""

from __future__ import annotations

import sqlite3
import numpy as np
import pytest

from core.audit import AuditTrail
from core.db import init_db
from core.stamps import DRAFT_STAMP, assert_no_banned_phrases
from models.spc.spc_engine import (
    calculate_imr_chart,
    calculate_xbar_r_chart,
    calculate_p_chart,
    calculate_c_chart,
    calculate_ewma_chart,
    calculate_cusum_chart,
    calculate_process_capability,
    evaluate_nelson_rules,
)
from models.spc.synthetic_spc import (
    generate_purity_series,
    generate_rft_series,
    generate_doc_errors_series,
    generate_analyst_subgroups,
)
from pm.knowledge_base import (
    seed_knowledge_base,
    get_all_root_causes,
    get_all_deviation_types,
    get_lessons_learned,
    search_lessons_learned,
    add_lesson_learned,
    ADVISORY_DISCLAIMER,
)
from pm.fmea_capa import (
    seed_fmea_and_capa,
    sync_fmea_with_lessons_learned,
    get_all_fmea_items,
    get_all_capa_records,
    transition_capa_status,
    evaluate_fmea_sensitivity,
    compute_action_priority,
    CAPA_STATE_TRANSITIONS,
)


@pytest.fixture
def mem_db():
    conn = init_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def audit_trail(mem_db):
    return AuditTrail(mem_db)


# =========================================================================
# 1. SPC ENGINE TESTS
# =========================================================================

def test_imr_chart_calculation_and_nelson_rules():
    # In-control sequence
    vals = np.array([98.2, 98.22, 98.18, 98.25, 98.19, 98.21, 98.23, 98.20])
    res = calculate_imr_chart(vals)
    assert res["chart_type"] == "I-MR"
    assert res["mode"] == "Phase I (Baseline Establishment)"
    assert res["center_line_x"] == pytest.approx(float(np.mean(vals)), 0.01)
    assert res["ucl_x"] > res["center_line_x"]
    assert res["lcl_x"] < res["center_line_x"]
    assert res["in_control"] is True
    assert res["disclaimer"] == DRAFT_STAMP
    assert_no_banned_phrases(str(res))

    # Test Rule 1 (point > 3 sigma)
    excursion_vals = np.array([98.2, 98.2, 98.2, 98.2, 98.2, 98.2, 98.2, 98.2, 105.0])
    res_exc = calculate_imr_chart(excursion_vals)
    r1_found = any(v["rule"] == "Rule 1" for v in res_exc["violations_x"])
    assert r1_found is True
    assert res_exc["in_control"] is False


def test_imr_chart_phase2_limit_freezing():
    vals = np.array([98.5, 98.6, 98.4, 98.5])
    frozen = {
        "center_line_x": 98.20,
        "ucl_x": 98.60,
        "lcl_x": 97.80,
        "center_line_mr": 0.15,
        "ucl_mr": 0.49,
    }
    res = calculate_imr_chart(vals, frozen_limits=frozen)
    assert res["mode"] == "Phase II (Monitored with Frozen Limits)"
    assert res["center_line_x"] == 98.20
    assert res["ucl_x"] == 98.60


def test_xbar_r_chart():
    subgroups = [
        np.array([100.1, 99.8, 100.2, 99.9]),
        np.array([100.0, 100.3, 99.7, 100.1]),
        np.array([99.9, 100.2, 100.0, 99.8]),
    ]
    res = calculate_xbar_r_chart(subgroups)
    assert res["chart_type"] == "X-bar/R"
    assert res["subgroup_size"] == 4
    assert res["center_line_x"] == pytest.approx(100.0, 0.2)
    assert res["ucl_x"] > res["center_line_x"]
    assert res["disclaimer"] == DRAFT_STAMP


def test_p_chart_variable_sample_sizes():
    defects = np.array([2, 5, 1, 8, 3])
    sizes = np.array([50, 45, 60, 40, 55])
    res = calculate_p_chart(defects, sizes)
    assert res["chart_type"] == "p-chart"
    assert len(res["ucl_per_sample"]) == 5
    assert all(u > res["center_line_p"] for u in res["ucl_per_sample"])
    assert res["disclaimer"] == DRAFT_STAMP


def test_c_chart_counts():
    counts = np.array([1, 2, 0, 3, 2, 1, 4])
    res = calculate_c_chart(counts)
    assert res["chart_type"] == "c-chart"
    assert res["ucl"] > res["center_line_c"]
    assert res["lcl"] >= 0.0


def test_ewma_and_cusum_small_shift_detection():
    # Shift of +1.0 sigma
    rng = np.random.RandomState(42)
    clean = rng.normal(98.2, 0.15, 20)
    shifted = rng.normal(98.2 + 1.5 * 0.15, 0.15, 30)
    combined = np.concatenate([clean, shifted])

    # EWMA
    res_ewma = calculate_ewma_chart(combined, lambda_=0.20, L=3.0, target=98.2, sigma=0.15)
    assert res_ewma["chart_type"] == "EWMA"
    assert len(res_ewma["violations"]) > 0

    # CUSUM
    res_cusum = calculate_cusum_chart(combined, target=98.2, sigma=0.15, k_sig=0.5, h_sig=4.0)
    assert res_cusum["chart_type"] == "CUSUM"
    assert len(res_cusum["violations"]) > 0


# =========================================================================
# 2. PROCESS CAPABILITY (Cp / Cpk) TESTS
# =========================================================================

def test_process_capability_and_chou_confidence_intervals():
    rng = np.random.RandomState(42)
    sample = rng.normal(loc=98.2, scale=0.15, size=40)
    res = calculate_process_capability(sample, lsl=97.0, usl=100.0)

    assert res["cp"] > 1.33
    assert res["cpk"] > 1.33
    ci = res["cpk_confidence_interval_95"]
    assert ci[0] < res["cpk"] < ci[1]
    assert res["normality_test"]["test_name"] == "Shapiro-Wilk"
    assert bool(res["normality_test"]["is_normal"]) is True
    assert len(res["warnings"]) == 0
    assert_no_banned_phrases(str(res))


def test_process_capability_warnings_on_small_n_and_non_normal():
    # Small non-normal bimodal sample
    sample = np.array([90.0, 90.1, 90.2, 110.0, 110.1, 110.2])
    res = calculate_process_capability(sample, lsl=85.0, usl=115.0)
    assert len(res["warnings"]) >= 1  # Should warn about small sample and/or non-normality


# =========================================================================
# 3. KNOWLEDGE BASE & LESSONS LEARNED TESTS
# =========================================================================

def test_knowledge_base_seeding_and_retrieval(mem_db):
    counts = seed_knowledge_base(mem_db)
    assert counts["root_causes_count"] >= 15
    assert counts["deviation_types_count"] >= 5
    assert counts["lessons_learned_count"] >= 40

    rcs = get_all_root_causes(mem_db)
    assert len(rcs) >= 15
    phase1_rcs = [r for r in rcs if "Phase I" in r["investigation_tier"]]
    phase2_rcs = [r for r in rcs if "Phase II" in r["investigation_tier"]]
    assert len(phase1_rcs) > 0
    assert len(phase2_rcs) > 0

    lessons = get_lessons_learned(mem_db)
    assert len(lessons) >= 40


def test_knowledge_base_tfidf_search(mem_db):
    seed_knowledge_base(mem_db)
    res = search_lessons_learned(mem_db, query="degasser vacuum bubble cavitation", top_k=3)
    assert res["results_count"] > 0
    assert len(res["matches"]) <= 3
    assert "advisory" in res
    assert "decision support" in res["advisory"].lower()
    # Check that top match relates to degasser
    top = res["matches"][0]
    assert "bubble" in top["what_happened"].lower() or "degasser" in top["what_learned"].lower() or "degass" in top["root_cause_name"].lower()


def test_add_lesson_learned_with_audit_trail(mem_db, audit_trail):
    seed_knowledge_base(mem_db)
    new_lesson = {
        "project_id": "PRJ-MAB-001",
        "method_name": "SEC-HPLC Monomer Purity",
        "investigation_tier": "Phase I (Laboratory Investigation)",
        "what_happened": "Autosampler needle seal leak caused drop in peak area.",
        "what_learned": "Rotor seal exceeded 600 injection preventive maintenance threshold.",
        "preventive_action": "Tightened rotor seal PM frequency to 500 injections.",
    }
    lid = add_lesson_learned(
        conn=mem_db,
        lesson=new_lesson,
        audit_trail=audit_trail,
        actor_name="Lead QC Specialist",
        actor_role="Principal Specialist, QC PM",
        justification="Documenting completed laboratory investigation.",
    )
    assert lid.startswith("LL-")

    # Verify present in db
    lessons = get_lessons_learned(mem_db, project_id="PRJ-MAB-001")
    assert any(l["id"] == lid for l in lessons)

    # Verify audit event
    events = [e for e in audit_trail.records if e.entity_id == lid]
    assert len(events) == 1
    assert events[0].action == "CREATE_LESSON_LEARNED"


# =========================================================================
# 4. FMEA & CAPA TESTS
# =========================================================================

def test_fmea_seeding_and_rpn_arithmetic(mem_db):
    seed_knowledge_base(mem_db)
    counts = seed_fmea_and_capa(mem_db)
    assert counts["fmea_items_count"] >= 8
    assert counts["capa_records_count"] >= 6

    sync_fmea_with_lessons_learned(mem_db)
    items = get_all_fmea_items(mem_db)
    assert len(items) >= 8

    for it in items:
        expected_rpn = it["severity"] * it["occurrence"] * it["detection"]
        assert it["rpn"] == expected_rpn
        assert it["action_priority"] in ["HIGH", "MEDIUM", "LOW"]
        assert it["post_rpn"] <= it["rpn"]
        assert it["rpn_reduction_pct"] >= 0.0
        assert "sensitivity" in it
        assert_no_banned_phrases(str(it))


def test_fmea_sensitivity_analysis():
    sens = evaluate_fmea_sensitivity(severity=8, occurrence=5, detection=5)
    assert sens["nominal_rpn"] == 200
    assert sens["most_sensitive_parameter"] in ["severity", "occurrence", "detection"]
    p_sens = sens["parameter_sensitivities"]
    assert p_sens["severity"]["minus_1"] == 7 * 5 * 5
    assert p_sens["severity"]["plus_1"] == 9 * 5 * 5


def test_capa_state_machine_valid_transitions(mem_db, audit_trail):
    seed_fmea_and_capa(mem_db)
    capas = get_all_capa_records(mem_db)
    capa_open = next(c for c in capas if c["status"] == "OPEN")

    # OPEN -> IN_PROGRESS
    res = transition_capa_status(
        conn=mem_db,
        capa_id=capa_open["id"],
        target_status="IN_PROGRESS",
        audit_trail=audit_trail,
        actor_name="QC Metrology Lead",
        actor_role="Metrology Engineer",
        justification="Initiated equipment qualification protocol.",
    )
    assert res["new_status"] == "IN_PROGRESS"


def test_capa_state_machine_illegal_transition_rejection(mem_db, audit_trail):
    seed_fmea_and_capa(mem_db)
    capas = get_all_capa_records(mem_db)
    capa_open = next(c for c in capas if c["status"] == "OPEN")

    # Illegal: OPEN -> CLOSED (skipping IN_PROGRESS, IMPLEMENTED, EFFECTIVENESS_CHECK)
    with pytest.raises(ValueError, match="Illegal CAPA transition"):
        transition_capa_status(
            conn=mem_db,
            capa_id=capa_open["id"],
            target_status="CLOSED",
            audit_trail=audit_trail,
            actor_name="QC Metrology Lead",
            actor_role="Metrology Engineer",
            justification="Attempting premature closure.",
            effectiveness_result="Not verified",
        )


def test_capa_closure_requires_effectiveness_and_justification(mem_db, audit_trail):
    seed_fmea_and_capa(mem_db)
    capas = get_all_capa_records(mem_db)
    capa_eff = next(c for c in capas if c["status"] == "EFFECTIVENESS_CHECK")

    # Missing effectiveness check result should raise ValueError
    with pytest.raises(ValueError, match="verified effectiveness"):
        transition_capa_status(
            conn=mem_db,
            capa_id=capa_eff["id"],
            target_status="CLOSED",
            audit_trail=audit_trail,
            actor_name="QA Manager",
            actor_role="Quality Assurance Manager",
            justification="Closing without result.",
            effectiveness_result="",  # empty
        )

    # Valid closure with confirmed effectiveness
    res = transition_capa_status(
        conn=mem_db,
        capa_id=capa_eff["id"],
        target_status="CLOSED",
        audit_trail=audit_trail,
        actor_name="QA Manager",
        actor_role="Quality Assurance Manager",
        justification="Confirmed 30 consecutive runs met all acceptance criteria with zero backpressure deviations.",
        effectiveness_result="Effectiveness verified: 30/30 runs acceptable.",
    )
    assert res["new_status"] == "CLOSED"


def test_overdue_capa_radar_and_reminder_draft(mem_db):
    seed_fmea_and_capa(mem_db)
    capas = get_all_capa_records(mem_db)
    overdue = [c for c in capas if c["is_overdue"]]
    assert len(overdue) > 0
    for o in overdue:
        assert o["reminder_draft"] is not None
        assert "OVERDUE" in o["reminder_draft"]
        assert_no_banned_phrases(o["reminder_draft"])
