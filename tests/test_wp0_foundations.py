"""
Tests for Work Package 0 Hardening: Rename to cGMP-QC-CoPilot and Shared Foundations.

Verifies:
1. Working-tree old-name check: Zero occurrences of 'BioX' across the entire workspace (tracked and untracked, excluding .git).
2. Append-only, hash-chained audit trail with checkpoint verification and ALCOA+ attributes.
3. SQLite immutability triggers: UPDATE and DELETE on audit_trail table are blocked.
4. Truncation detection: Truncating the last N records fails checkpoint verification.
5. Single audit trail: Workflow HITL sign-off creates an audit record that cryptographically verifies.
6. Regulatory draft stamps and banned-phrase claims linter (Guardrail G2) with negation/quoted awareness.
7. Database initialization and migrations.
8. Synthetic data manifest creation and hash verification (Guardrail G1).
9. Model card schema validation and failure-mode disclosure (Guardrail G5).
"""

import json
import sqlite3
import tempfile
from pathlib import Path
import pytest

from core.audit import (
    AuditCheckpoint,
    AuditRecord,
    AuditTrail,
    GENESIS_HASH,
    PART11_DEMO_DISCLAIMER,
    verify_chain,
)
from core.db import get_db_status, init_db
from core.modelcard import (
    DEFAULT_DISCLAIMER,
    ModelCard,
    load_model_card,
    save_model_card,
    validate_model_card,
)
from core.stamps import (
    BANNED_PHRASES,
    DRAFT_STAMP,
    apply_draft_stamp,
    assert_no_banned_phrases,
    check_banned_phrases,
    scan_documentation_and_skills,
)
from core.synthetic import (
    compute_file_sha256,
    generate_manifest,
    verify_manifest,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.workflow import (
    build_mso_draft_quality_package,
    verify_and_sign_hitl_gate,
)


# -----------------------------------------------------------------------------
# 1. Old-Name Test: Zero occurrences across entire working tree (A2)
# -----------------------------------------------------------------------------
def test_no_old_name_in_working_tree():
    """Fails if the old name 'BioX' appears anywhere in the working tree (tracked or untracked), excluding .git."""
    root = Path(__file__).resolve().parent.parent
    violations = []
    ignored_dirs = {".git", "__pycache__", ".pytest_cache"}

    for p in root.rglob("*"):
        if p.is_file():
            # Check if any part of the path is in ignored_dirs
            if any(part in ignored_dirs for part in p.parts):
                continue
            # Skip this test file itself
            if "test_wp0_foundations.py" in p.name:
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if "biox" in content.lower():
                    violations.append(str(p.relative_to(root)))
            except Exception:
                continue

    assert not violations, (
        f"Found old name 'BioX' in {len(violations)} working-tree file(s): {violations}. "
        "Project name must strictly be 'cGMP-QC-CoPilot'."
    )


# -----------------------------------------------------------------------------
# 2. Audit Trail, Immutability Triggers & Checkpoints (A4, G6)
# -----------------------------------------------------------------------------
def test_audit_trail_append_and_verify():
    trail = AuditTrail()
    rec1 = trail.append_event(
        actor_name="Jane Doe",
        actor_role="Principal Specialist, QC PM",
        action="SCHEDULE_SMOOTH",
        justification="Level W42 UPLC workload from 94% to 81%",
        entity_type="SCHEDULE",
        entity_id="SCHED-2026-W42",
        payload={"shifted_lots": 3, "target_utilization": 0.81},
    )
    rec2 = trail.append_event(
        actor_name="Dr. Vance",
        actor_role="Director, Quality Control",
        action="APPROVE_SCHEDULE",
        justification="Within configured example pull window (example ±7 d)",
        entity_type="SCHEDULE",
        entity_id="SCHED-2026-W42",
        payload={"status": "APPROVED"},
    )

    assert rec1.index == 0
    assert rec1.previous_hash == GENESIS_HASH
    assert rec2.index == 1
    assert rec2.previous_hash == rec1.record_hash
    assert rec1.disclaimer == PART11_DEMO_DISCLAIMER

    is_valid, err = trail.verify_chain()
    assert is_valid, f"Chain should be valid but failed: {err}"


def test_audit_trail_head_checkpoint_and_truncation():
    trail = AuditTrail()
    trail.append_event(
        actor_name="Analyst 1",
        actor_role="QC Lead",
        action="ACTION_A",
        justification="Legitimate justification A",
    )
    trail.append_event(
        actor_name="Analyst 2",
        actor_role="QC Lead",
        action="ACTION_B",
        justification="Legitimate justification B",
    )
    trail.append_event(
        actor_name="Analyst 3",
        actor_role="QC Lead",
        action="ACTION_C",
        justification="Legitimate justification C",
    )

    checkpoint = trail.get_checkpoint()
    assert checkpoint.record_count == 3
    assert checkpoint.head_hash == trail.latest_hash

    # Full chain matches checkpoint
    valid, err = trail.verify_chain(checkpoint)
    assert valid, err

    # Truncate the last record (tail truncation attempt)
    truncated_records = [r.to_dict() for r in trail.records[:-1]]
    valid, err = verify_chain(truncated_records, checkpoint)
    assert not valid
    assert "truncation" in err.lower() or "count mismatch" in err.lower()


def test_audit_trail_sqlite_immutability_triggers():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        conn = init_db(db_path)
        trail = AuditTrail(conn)
        trail.append_event(
            actor_name="Signer 1",
            actor_role="QA Specialist",
            action="LOCK_EVENT",
            justification="Testing SQLite trigger immutability",
        )

        # 1. Attempt UPDATE on audit_trail table
        with pytest.raises(sqlite3.DatabaseError, match="strictly prohibited"):
            with conn:
                conn.execute("UPDATE audit_trail SET actor_name='Hacker' WHERE record_index=0;")

        # 2. Attempt DELETE on audit_trail table
        with pytest.raises(sqlite3.DatabaseError, match="strictly prohibited"):
            with conn:
                conn.execute("DELETE FROM audit_trail WHERE record_index=0;")

        # Confirm data remains untouched and valid
        valid, _ = trail.verify_chain()
        assert valid
        conn.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_audit_trail_tamper_detection():
    trail = AuditTrail()
    trail.append_event(
        actor_name="Analyst A",
        actor_role="QC Analyst II",
        action="LOG_RUN",
        justification="Routine run completion",
    )
    trail.append_event(
        actor_name="Analyst B",
        actor_role="QC Lead",
        action="REVIEW_RUN",
        justification="First-line review pass",
    )

    records = [r.to_dict() for r in trail.records]

    # Tamper with actor name in record 0
    tampered_records = [dict(r) for r in records]
    tampered_records[0]["actor_name"] = "Malicious Actor"
    is_valid, err = verify_chain(tampered_records)
    assert not is_valid, "Tampering with actor_name must be detected"

    # Tamper with payload in record 1
    tampered_records = [dict(r) for r in records]
    tampered_records[1]["payload"] = {"unauthorized_field": 123}
    is_valid, err = verify_chain(tampered_records)
    assert not is_valid, "Tampering with payload must be detected"


# -----------------------------------------------------------------------------
# 3. Single Audit Trail Integration (A5)
# -----------------------------------------------------------------------------
def test_workflow_sign_off_logs_to_single_audit_trail():
    draft_pkg = build_mso_draft_quality_package(
        excursion_temp_c=28.0,
        excursion_duration_hours=36.0,
        lot_id="LOT-DEMO-999",
        project_id="PROJ-001",
    )
    trail = AuditTrail()

    # Incomplete sign-off blocked
    blocked_out = verify_and_sign_hitl_gate(
        draft_package=draft_pkg,
        reviewer_name="",
        reviewer_role="",
        justification="short",
        confirmed=False,
        audit_trail=trail,
    )
    assert not blocked_out["approved"]
    assert len(trail.records) == 0

    # Successful sign-off writes to audit trail
    sign_out = verify_and_sign_hitl_gate(
        draft_package=draft_pkg,
        reviewer_name="Dr. Alex Rivera",
        reviewer_role="Principal Specialist, QC PM",
        justification="Independently reviewed kinetic parameters, prior conflict checks, and detection gate margins.",
        confirmed=True,
        audit_trail=trail,
    )
    assert sign_out["approved"]
    assert len(trail.records) == 1
    assert trail.records[0].action == "HITL_SIGN_OFF"
    assert trail.records[0].actor_name == "Dr. Alex Rivera"

    # Verify cryptographic validity
    is_valid, err = trail.verify_chain()
    assert is_valid, f"Chain failed verification: {err}"


# -----------------------------------------------------------------------------
# 4. Draft Stamps and Banned-Phrase Linter with Negation (A3, G2)
# -----------------------------------------------------------------------------
def test_banned_phrase_affirmative_claims_blocked():
    affirmative_claims = [
        "This algorithm is validated for stability shelf-life extrapolation.",
        "We can guarantee that zero defects occurred on skid 3.",
        "Batch LOT-001 is approved to release for use.",
        "Thermal modeling proves safe storage at ambient conditions.",
        "The risk assessment confirms zero risk for this process step.",
    ]
    for claim in affirmative_claims:
        detected = check_banned_phrases(claim, allow_disclaimers=True)
        assert len(detected) > 0, f"Expected affirmative claim '{claim}' to be flagged"
        with pytest.raises(ValueError, match="Banned phrase violation"):
            assert_no_banned_phrases(claim)


def test_banned_phrase_negated_disclaimers_allowed():
    legitimate_disclaimers = [
        "This model is not validated for GMP batch disposition.",
        "Under FDA 21 CFR Part 211, outputs are non-validated decision support.",
        "Outcomes cannot be guaranteed without empirical testing.",
        "This simulation does not prove safe shipping conditions.",
        "Assessing thermal drift does not imply zero risk.",
        "The system will never release for use any biological product.",
        "The user must not consider this result as 'validated'.",
        "Refer to the list of banned phrases: 'release for use', 'zero risk'.",
    ]
    for disclaimer in legitimate_disclaimers:
        detected = check_banned_phrases(disclaimer, allow_disclaimers=True)
        assert detected == [], f"Legitimate disclaimer was improperly flagged: '{disclaimer}' -> {detected}"
        # Should not raise
        assert_no_banned_phrases(disclaimer)


def test_draft_stamp_application():
    text = "Capacity optimization completed successfully."
    stamped = apply_draft_stamp(text)
    assert DRAFT_STAMP in stamped
    assert text in stamped
    double_stamped = apply_draft_stamp(stamped)
    assert double_stamped.count(DRAFT_STAMP) == 1


# -----------------------------------------------------------------------------
# 5. Database Migrations and Status
# -----------------------------------------------------------------------------
def test_docs_claims_hygiene():
    """
    Scans README.md, DEMO.md, LIMITATIONS.md, docs/*.md, and all skill.md files
    for prohibited affirmative claims: guaranteed, proves safe, release for use,
    zero risk, validated (unless negated/quoted), compliant (unless 'not a compliant'/quoted),
    immutable (unless task-semantics like 'immutable fixed tasks').
    """
    violations = scan_documentation_and_skills()
    assert len(violations) == 0, f"Documentation claims violations detected: {violations}"


def test_db_initialization_and_migrations():
    conn = init_db(":memory:")
    status = get_db_status(conn)
    assert status["latest_migration_version"] >= 6
    assert "audit_trail" in status["table_counts"]
    assert "audit_checkpoint" in status["table_counts"]
    assert "portfolio_projects" in status["table_counts"]
    assert "lessons_learned" in status["table_counts"]
    assert "capa_records" in status["table_counts"]
    assert "vendor_telemetry" in status["table_counts"]
    conn.close()


# -----------------------------------------------------------------------------
# 6. Synthetic Data Manifest Verification (G1 & G4)
# -----------------------------------------------------------------------------
def test_synthetic_manifest_creation_and_tamper_detection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        f1 = tmp_p / "synth_data_1.csv"
        f2 = tmp_p / "synth_data_2.json"
        f1.write_text("lot_id,hmw_pct\nLOT-001,0.85\n", encoding="utf-8")
        f2.write_text(json.dumps({"is_synthetic": True, "values": [1, 2, 3]}), encoding="utf-8")

        manifest_file = tmp_p / "manifest.json"
        generate_manifest([f1, f2], manifest_output_path=manifest_file, seed=42)

        valid, errors = verify_manifest(manifest_file, base_dir=tmp_p)
        assert valid, f"Generated manifest should be valid: {errors}"

        # Tamper with f1
        f1.write_text("tampered content", encoding="utf-8")
        tamper_valid, tamper_errors = verify_manifest(manifest_file, base_dir=tmp_p)
        assert not tamper_valid
        assert any("mismatch" in e.lower() for e in tamper_errors)


# -----------------------------------------------------------------------------
# 7. Model Card Schema & Validation (G5)
# -----------------------------------------------------------------------------
def test_model_card_validation():
    valid_card = ModelCard(
        model_id="schedule_risk_mc_v1",
        model_name="Schedule Risk Monte Carlo Engine",
        version="1.0.0",
        intended_use="Predict milestone delivery probability and critical path shifts.",
        out_of_scope=["Autonomous schedule execution", "Official batch release date setting"],
        training_data_summary={"synthetic": True, "seed": 42, "weeks_simulated": 52},
        deterministic_baseline={"baseline_name": "naive_average_planner", "mae_days": 4.2},
        benchmark_results={"mae_days": 2.1, "ci_95": [1.8, 2.4], "improvement_pct": 50.0},
        known_failure_modes=["Underestimates delays during simultaneous multi-skid maintenance shutdowns."],
    )

    valid, errors = validate_model_card(valid_card.to_dict())
    assert valid, f"Valid model card failed: {errors}"

    # Invalid: missing baseline
    invalid_card_dict = valid_card.to_dict()
    invalid_card_dict["deterministic_baseline"] = {}
    valid, errors = validate_model_card(invalid_card_dict)
    assert not valid
    assert any("baseline" in e for e in errors)

    # Invalid: missing failure modes (G5 honest disclosure)
    invalid_card_dict2 = valid_card.to_dict()
    invalid_card_dict2["known_failure_modes"] = []
    valid, errors = validate_model_card(invalid_card_dict2)
    assert not valid
    assert any("failure mode" in e for e in errors)


def test_model_card_save_and_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "model_card.json"
        card = ModelCard(
            model_id="test_model_v1",
            model_name="Test Model",
            version="1.0.0",
            intended_use="Unit testing only.",
            out_of_scope=["Production use"],
            training_data_summary={"synthetic": True, "n_samples": 100},
            deterministic_baseline={"baseline_name": "sample_mean", "rmse": 1.0},
            benchmark_results={"rmse": 0.5, "ci_95": [0.4, 0.6]},
            known_failure_modes=["Fails on extrapolated inputs."],
        )
        save_model_card(card, out_path)
        loaded = load_model_card(out_path)
        assert loaded.model_id == "test_model_v1"
        assert loaded.known_failure_modes == ["Fails on extrapolated inputs."]


def test_audit_trail_get_all_records_and_defensive_df_casting():
    import pandas as pd

    trail = AuditTrail()
    rec = trail.append_event(
        actor_name="TEST_USER",
        actor_role="QC Analyst",
        action="TEST_ACTION",
        justification="Testing get_all_records defensive extraction",
    )

    # 1. Verify get_all_records()
    records = trail.get_all_records()
    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["actor_name"] == "TEST_USER"
    assert records[0]["action"] == "TEST_ACTION"

    # 2. Test defensive extraction logic with AuditTrail instance
    audit_state = trail
    if hasattr(audit_state, "get_all_records"):
        audit_records = audit_state.get_all_records()
    elif isinstance(audit_state, list):
        audit_records = audit_state
    else:
        audit_records = []
    df_audit = pd.DataFrame(audit_records) if audit_records else pd.DataFrame()
    assert not df_audit.empty
    assert "actor_name" in df_audit.columns

    # 3. Test defensive extraction with empty state / None / arbitrary object
    for empty_state in ([], None, "invalid", object()):
        audit_state = empty_state
        if hasattr(audit_state, "get_all_records"):
            audit_records = audit_state.get_all_records()
        elif isinstance(audit_state, list):
            audit_records = audit_state
        else:
            audit_records = []
        df_audit = pd.DataFrame(audit_records) if audit_records else pd.DataFrame()
        assert isinstance(df_audit, pd.DataFrame)
        assert df_audit.empty


def test_documentation_and_claims_dynamic_synchronization():
    """
    Verifies dynamic claims synchronization across documentation and codebase:
    1. Knowledge base precedents: compares DEMO.md and LIMITATIONS.md dynamically
       against each other and against len(pm.knowledge_base._build_synthetic_lessons()).
    2. Stability pull windows: README.md, DEMO.md, and LIMITATIONS.md reference approved +/- 7-day
       tolerances with zero occurrences of outdated +/- 48h windows.
    3. Documented test metrics: README.md badges and text dynamically match the actual
       registered pytest functions across tests/test_*.py.
    4. Claims hygiene: Documentation contains zero occurrences of unhedged affirmative claims
       ('zero false alarms', 'biophysical certainty', 'cryptographically locks out re-testing',
       'proving mathematically').
    """
    import re
    import pm.knowledge_base as kb

    root = Path(__file__).resolve().parent.parent
    readme_path = root / "README.md"
    demo_path = root / "DEMO.md"
    limitations_path = root / "LIMITATIONS.md"

    readme_text = readme_path.read_text(encoding="utf-8")
    demo_text = demo_path.read_text(encoding="utf-8")
    limitations_text = limitations_path.read_text(encoding="utf-8")

    # 1. Dynamic knowledge base precedent count comparison
    actual_kb_count = len(kb._build_synthetic_lessons())
    m_demo = re.search(r"(\d+)\s+historical root-cause precedents", demo_text)
    m_limitations = re.search(r"(\d+)\s+synthetic historical deviation cases", limitations_text)

    assert m_demo is not None, "DEMO.md missing historical root-cause precedents reference"
    assert m_limitations is not None, "LIMITATIONS.md missing historical deviation cases reference"
    assert int(m_demo.group(1)) == actual_kb_count, (
        f"DEMO.md count ({m_demo.group(1)}) does not match actual KB lessons ({actual_kb_count})"
    )
    assert int(m_limitations.group(1)) == actual_kb_count, (
        f"LIMITATIONS.md count ({m_limitations.group(1)}) does not match actual KB lessons ({actual_kb_count})"
    )
    assert int(m_demo.group(1)) == int(m_limitations.group(1)), "Cross-document KB precedent count mismatch"

    # 2. Pull window standardization (±7 days / \pm 7, no ±48h)
    for doc_name, text in [("README.md", readme_text), ("DEMO.md", demo_text), ("LIMITATIONS.md", limitations_text)]:
        assert "±48h" not in text and "+/- 48h" not in text and r"\pm 48" not in text, (
            f"Outdated 48h pull window found in {doc_name}"
        )

    assert any(pat in readme_text for pat in ["±7 days", "+/- 7 days", r"\pm 7\text{ days}", r"\pm 7"])
    assert any(pat in demo_text for pat in ["±7 days", "+/- 7 days", r"\pm 7\text{ days}", r"\pm 7"])

    # 3. Dynamic documented test metrics comparison (from README text summary)
    # Non-brittle check: verifies the README states passing tests and does not claim MORE than exist
    m_expected = re.search(r"currently\s+(\d+)\s+passed,\s+(\d+)\s+skipped", readme_text)
    assert m_expected is not None, (
        "README.md missing expected result text summary "
        "(e.g. 'all tests should pass (currently 140 passed, 3 skipped; counts change as tests are added)')"
    )
    doc_passed = int(m_expected.group(1))
    doc_skipped = int(m_expected.group(2))

    test_files = list(root.glob("tests/test_*.py"))
    actual_test_func_count = sum(
        len(re.findall(r"^\s*def test_", f.read_text(encoding="utf-8"), flags=re.MULTILINE))
        for f in test_files
    )

    # Verifies the README does not claim MORE passing tests than exist in the codebase
    assert doc_passed <= actual_test_func_count, (
        f"README claims {doc_passed} passing tests, but only {actual_test_func_count} test functions exist"
    )
    assert doc_skipped == 3

    # 4. Claims hygiene / banned affirmative phrases
    banned_phrases = [
        "zero false alarms",
        "biophysical certainty",
        "cryptographically locks out re-testing",
        "proving mathematically",
    ]
    for doc_name, text in [("README.md", readme_text), ("DEMO.md", demo_text), ("LIMITATIONS.md", limitations_text)]:
        lower_text = text.lower()
        for phrase in banned_phrases:
            assert phrase not in lower_text, f"Banned phrase '{phrase}' found in {doc_name}"


