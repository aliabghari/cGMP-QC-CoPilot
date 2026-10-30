"""
tests/test_qc_pm_modules.py

Comprehensive Unit Tests for 3 Rule-Based QC Project Management Modules:
- Module 1: Contract Lab Data Review Tracker (pm/contract_lab_tracker.py)
- Module 2: Extended Roadblock Categories (pm/blockers.py)
- Module 3: Inspection-Readiness One-Pager (pm/inspection_readiness.py)

Covers all Part 6 Acceptance Criteria:
- Rule enforcement, accepted and rejected transitions
- Boundary dates (due today, due in exactly 3 days, overdue)
- Escalation threshold boundaries (3d Team, 7d QC Lead, 8d Quality Head)
- Empty registers handling
- DRAFT stamp presence in exports
- Audit-trail-unavailable refusal path (audit_trail is None)
- Audit writes and cryptographic hash chain verification (verify_chain)
- Zero banned affirmative phrases
"""

from __future__ import annotations

import sqlite3
import pytest

from core.audit import AuditTrail, verify_chain
from core.db import init_db
from core.stamps import assert_no_banned_phrases, check_banned_phrases
from pm.contract_lab_tracker import (
    compute_contract_lab_summary,
    compute_turnaround_and_flags,
    get_seed_contract_lab_packages,
    transition_package_to_accepted,
)
from pm.blockers import (
    ALL_BLOCKER_CATEGORIES,
    BLOCKER_CATEGORIES,
    EXTENDED_ROADBLOCK_CATEGORIES,
    assign_roadblock_escalation_tier,
    evaluate_blocker_escalation,
    generate_escalation_draft,
    get_seed_extended_roadblocks,
)
from pm.inspection_readiness import (
    generate_inspection_readiness_markdown,
    generate_readiness_flags,
    get_capa_readiness_data,
    get_consolidated_readiness_dataset,
    get_open_oos_investigations,
    get_open_qc_deviations,
    get_stability_pull_readiness_data,
)


# =============================================================================
# MODULE 1: CONTRACT LAB DATA REVIEW TRACKER TESTS
# =============================================================================
class TestModule1ContractLabTracker:
    """Verifies Module 1 business rules, flags, transitions, and audit trail integration."""

    def test_seed_contract_lab_packages_structure(self):
        """Verifies 8-10 packages seeded, covering all 4 statuses and required edge cases."""
        pkgs = get_seed_contract_lab_packages("2026-10-30")
        assert len(pkgs) == 10, f"Expected exactly 10 seeded packages, got {len(pkgs)}"

        statuses = {p["review_status"] for p in pkgs}
        assert statuses == {"Awaiting Data", "In Review", "Queries Open", "Accepted"}

        # Verify overdue package exists
        overdue_pkgs = [p for p in pkgs if compute_turnaround_and_flags(p, "2026-10-30")["is_overdue"]]
        assert len(overdue_pkgs) >= 1, "Expected at least one overdue package"

        # Verify at-risk packages (due within 3 days: PKG-002 at 2d, PKG-010 at 3d)
        at_risk_pkgs = [p for p in pkgs if compute_turnaround_and_flags(p, "2026-10-30")["is_at_risk"]]
        assert len(at_risk_pkgs) == 2, f"Expected exactly 2 at-risk packages, got {len(at_risk_pkgs)}"

        # Verify PKG-CRO-2026-010 is due in exactly 3 days and flagged accordingly
        pkg_10 = next(p for p in pkgs if p["package_id"] == "PKG-CRO-2026-010")
        f_10 = compute_turnaround_and_flags(pkg_10, "2026-10-30")
        assert f_10["days_until_due"] == 3
        assert f_10["is_at_risk"] is True
        assert f_10["flag_label"] == "DUE SOON (≤3d) ⏳"

    def test_turnaround_and_flag_computation_boundaries(self):
        """Verifies boundary cases: due today, due in exactly 3 days, overdue, and on-track."""
        today = "2026-10-30"

        # 1. Due in past -> Overdue
        pkg_overdue = {
            "package_id": "PKG-01",
            "sow_due_date": "2026-10-25",
            "review_status": "In Review",
        }
        res_overdue = compute_turnaround_and_flags(pkg_overdue, today)
        assert res_overdue["is_overdue"] is True
        assert res_overdue["is_at_risk"] is False
        assert "OVERDUE" in res_overdue["flag_label"]

        # 2. Due today (0 days) -> At-Risk
        pkg_today = {
            "package_id": "PKG-02",
            "sow_due_date": "2026-10-30",
            "review_status": "Awaiting Data",
        }
        res_today = compute_turnaround_and_flags(pkg_today, today)
        assert res_today["is_overdue"] is False
        assert res_today["is_at_risk"] is True
        assert res_today["days_until_due"] == 0

        # 3. Due in exactly 3 days (boundary) -> At-Risk
        pkg_3d = {
            "package_id": "PKG-03",
            "sow_due_date": "2026-11-02",
            "review_status": "Queries Open",
        }
        res_3d = compute_turnaround_and_flags(pkg_3d, today)
        assert res_3d["is_overdue"] is False
        assert res_3d["is_at_risk"] is True
        assert res_3d["days_until_due"] == 3
        assert res_3d["flag_label"] == "DUE SOON (≤3d) ⏳"

        # 4. Due in 4 days -> On Track (not at risk)
        pkg_4d = {
            "package_id": "PKG-04",
            "sow_due_date": "2026-11-03",
            "review_status": "Awaiting Data",
        }
        res_4d = compute_turnaround_and_flags(pkg_4d, today)
        assert res_4d["is_overdue"] is False
        assert res_4d["is_at_risk"] is False
        assert res_4d["flag_label"] == "ON TRACK"

        # 5. Accepted package is never overdue or at risk
        pkg_accepted = {
            "package_id": "PKG-05",
            "sow_due_date": "2026-10-20",
            "review_status": "Accepted",
        }
        res_acc = compute_turnaround_and_flags(pkg_accepted, today)
        assert res_acc["is_overdue"] is False
        assert res_acc["is_at_risk"] is False
        assert "ACCEPTED" in res_acc["flag_label"]

    def test_transition_rejected_when_open_queries_exist(self):
        """Rule: A package cannot move to Accepted while any query is open."""
        trail = AuditTrail()
        pkg = {
            "package_id": "PKG-TEST-01",
            "lab_name": "Northbay Analytical Labs (Fictional)",
            "method": "SEC-HPLC",
            "review_status": "Queries Open",
            "open_query_count": 2,
        }

        success, reason, updated = transition_package_to_accepted(
            pkg=pkg,
            reviewer_name="Dr. Sarah Lin",
            reviewer_role="QC Lead",
            justification="Chromatograms verified and integration acceptable.",
            audit_trail=trail,
        )

        assert success is False
        assert updated is None
        assert "2 unresolved analytical query" in reason
        assert len(trail.records) == 0, "Audit trail must not record rejected transition"

    def test_transition_rejected_when_justification_too_short(self):
        """Rule: A package cannot move to Accepted without >= 10-char justification."""
        trail = AuditTrail()
        pkg = {
            "package_id": "PKG-TEST-02",
            "lab_name": "Apex Biologics Testing (Fictional)",
            "method": "icIEF",
            "review_status": "In Review",
            "open_query_count": 0,
        }

        # Short justification (< 10 chars)
        success, reason, updated = transition_package_to_accepted(
            pkg=pkg,
            reviewer_name="Dr. Sarah Lin",
            reviewer_role="QC Lead",
            justification="Looks ok",  # 8 chars
            audit_trail=trail,
        )

        assert success is False
        assert updated is None
        assert "at least 10 characters" in reason
        assert len(trail.records) == 0

    def test_transition_refused_when_audit_trail_is_none(self):
        """Rule: If audit_trail is None, refuse transition and provide visible notice."""
        pkg = {
            "package_id": "PKG-TEST-03",
            "lab_name": "Cedar Ridge Bioanalytics (Fictional)",
            "method": "qPCR",
            "review_status": "In Review",
            "open_query_count": 0,
        }

        success, reason, updated = transition_package_to_accepted(
            pkg=pkg,
            reviewer_name="Dr. Sarah Lin",
            reviewer_role="QC Lead",
            justification="Valid data package with system suitability met.",
            audit_trail=None,  # Unavailable!
        )

        assert success is False
        assert updated is None
        assert "audit trail ledger is unavailable" in reason.lower()

    def test_successful_transition_writes_to_audit_trail_and_verifies_chain(self):
        """Rule: Valid transition mutates status to Accepted and writes to hash-chained audit ledger."""
        db_conn = init_db(":memory:")
        trail = AuditTrail(db_conn=db_conn)

        pkg = {
            "package_id": "PKG-TEST-04",
            "lab_name": "Cascadia Bioanalytical Labs (Fictional)",
            "method": "Bioassay Potency",
            "review_status": "In Review",
            "open_query_count": 0,
            "sow_due_date": "2026-11-05",
            "data_received_date": "2026-10-28",
        }

        justification_text = "System suitability and 4-PL parallelism verified against reference standard."
        success, msg, updated = transition_package_to_accepted(
            pkg=pkg,
            reviewer_name="Elena Gomez",
            reviewer_role="Senior QC Specialist",
            justification=justification_text,
            audit_trail=trail,
        )

        assert success is True
        assert updated is not None
        assert updated["review_status"] == "Accepted"
        assert updated["reviewer"] == "Elena Gomez"

        # Verify audit trail record
        assert len(trail.records) == 1
        rec = trail.records[0]
        assert rec.action == "ACCEPT_CONTRACT_LAB_PACKAGE"
        assert rec.entity_id == "PKG-TEST-04"
        assert rec.justification == justification_text
        assert rec.payload["new_status"] == "Accepted"

        # Cryptographic chain verification
        is_valid, err = trail.verify_chain()
        assert is_valid is True, f"Hash chain verification failed: {err}"

    def test_contract_lab_summary_panel_metrics(self):
        """Verifies summary metrics across packages: status distribution, on-time rate, oldest query."""
        pkgs = get_seed_contract_lab_packages("2026-10-30")
        summary = compute_contract_lab_summary(pkgs, "2026-10-30")

        assert summary["total_packages"] == 10
        assert summary["status_counts"]["Queries Open"] >= 1
        assert summary["status_counts"]["Accepted"] >= 1
        assert summary["overdue_count"] >= 1
        assert summary["at_risk_count"] == 2
        assert summary["oldest_open_query_days"] >= 3
        assert summary["oldest_open_query_pkg_id"] != "None"
        assert 0.0 <= summary["on_time_rate_pct"] <= 100.0


# =============================================================================
# MODULE 2: EXTENDED ROADBLOCK CATEGORIES TESTS
# =============================================================================
class TestModule2ExtendedRoadblocks:
    """Verifies Module 2 extended categories, configurable tier assignment, and draft generator."""

    def test_extended_categories_and_backward_compatibility(self):
        """Verifies the four new categories exist while preserving existing BLOCKER_CATEGORIES."""
        expected_new = {
            "Pending Approval",
            "Late Sample Arrival",
            "Client Dependency",
            "Vendor Dependency",
        }
        assert set(EXTENDED_ROADBLOCK_CATEGORIES) == expected_new
        # Existing categories must not be modified
        assert len(BLOCKER_CATEGORIES) == 5
        assert "Reagent / Reference Standard" in BLOCKER_CATEGORIES
        assert len(ALL_BLOCKER_CATEGORIES) == len(BLOCKER_CATEGORIES) + len(EXTENDED_ROADBLOCK_CATEGORIES)

    def test_escalation_tier_boundaries(self):
        """Verifies boundaries: <= 3d (Team), 3d boundary (Team), 7d boundary (QC Lead), 8d (Quality Head)."""
        # 0 days -> Team
        assert assign_roadblock_escalation_tier(0) == "Team"
        # 2 days -> Team
        assert assign_roadblock_escalation_tier(2) == "Team"
        # 3 days (exact boundary) -> Team
        assert assign_roadblock_escalation_tier(3) == "Team"
        # 4 days -> QC Lead
        assert assign_roadblock_escalation_tier(4) == "QC Lead"
        # 7 days (exact boundary) -> QC Lead
        assert assign_roadblock_escalation_tier(7) == "QC Lead"
        # 8 days -> Quality Head
        assert assign_roadblock_escalation_tier(8) == "Quality Head"
        # 14 days -> Quality Head
        assert assign_roadblock_escalation_tier(14) == "Quality Head"

    def test_escalation_tier_custom_configuration(self):
        """Verifies that configurable days-open thresholds change tier boundaries dynamically."""
        custom_cfg = {
            "team_tier_max_days": 2,      # Team: <= 2 days
            "qc_lead_tier_max_days": 5,   # QC Lead: 3-5 days
        }
        assert assign_roadblock_escalation_tier(2, custom_cfg) == "Team"
        assert assign_roadblock_escalation_tier(3, custom_cfg) == "QC Lead"
        assert assign_roadblock_escalation_tier(5, custom_cfg) == "QC Lead"
        assert assign_roadblock_escalation_tier(6, custom_cfg) == "Quality Head"

    def test_escalation_draft_generation_content_and_hygiene(self):
        """Verifies escalation draft includes DRAFT stamp, human review notice, and zero banned words."""
        blocker = {
            "id": "BLK-EXT-104",
            "category": "Late Sample Arrival",
            "owner": "Sample Coordination Group",
            "days_open": 7,
            "impacted_work_package": "WP-STAB-28C",
            "description": "Thermal shipper delayed at air transit customs clearance.",
            "escalation_tier": "QC Lead",
        }

        draft = generate_escalation_draft(blocker)

        # Mandatory elements
        assert "DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL" in draft
        assert "QC LEAD TIER" in draft
        assert "WP-STAB-28C" in draft
        assert "BLK-EXT-104" in draft
        assert "7 day(s) open" in draft
        assert "Human project manager review and manual transmission required. Do not auto-send." in draft

        # Claims hygiene
        assert_no_banned_phrases(draft, context="Test Escalation Draft")

    def test_seed_extended_roadblocks_exercise_all_categories_and_tiers(self):
        """Verifies seeded roadblocks exercise all 4 categories and all 3 tiers with boundary cases."""
        rbs = get_seed_extended_roadblocks()
        assert len(rbs) >= 8

        cats = {r["category"] for r in rbs}
        assert cats == set(EXTENDED_ROADBLOCK_CATEGORIES)

        tiers = {r["escalation_tier"] for r in rbs}
        assert tiers == {"Team", "QC Lead", "Quality Head"}

        # Check boundary cases: exactly 3 days and exactly 7 days
        days_set = {r["days_open"] for r in rbs}
        assert 3 in days_set, "Expected boundary case: exactly 3 days"
        assert 7 in days_set, "Expected boundary case: exactly 7 days"

    def test_existing_blocker_escalation_behavior_unbroken(self):
        """Verifies backward compatibility with existing evaluate_blocker_escalation logic."""
        res_t1_esc = evaluate_blocker_escalation(blocker_age_days=6, project_tier="Tier 1A", severity="HIGH")
        assert res_t1_esc["escalate"] is True
        assert res_t1_esc["urgency"] == "CRITICAL_ESCALATION"

        res_t1_ok = evaluate_blocker_escalation(blocker_age_days=3, project_tier="Tier 1A", severity="HIGH")
        assert res_t1_ok["escalate"] is False
        assert res_t1_ok["urgency"] == "MONITORED"


# =============================================================================
# MODULE 3: INSPECTION-READINESS ONE-PAGER TESTS
# =============================================================================
class TestModule3InspectionReadiness:
    """Verifies Module 3 empirical aggregation, plain language flags, and Markdown exports."""

    def test_open_oos_investigations_data(self):
        """Verifies active OOS investigations have age, method, preliminary cause, and zero scores."""
        oos_list = get_open_oos_investigations(reference_date="2026-10-30")
        assert len(oos_list) >= 2
        for o in oos_list:
            assert o["investigation_id"].startswith("OOS-")
            assert o["age_days"] > 0
            assert "reported_result" in o
            assert "specification" in o
            assert "preliminary_cause" in o

    def test_capa_readiness_and_overdue_effectiveness_checks(self):
        """Verifies detection of open CAPAs and overdue effectiveness checks past due date."""
        capa_data = get_capa_readiness_data(reference_date="2026-10-30")
        assert capa_data["open_capa_count"] >= 2
        assert capa_data["overdue_effectiveness_count"] >= 1

        # Check that overdue CAPA indeed has due date before reference date
        for c in capa_data["overdue_effectiveness_checks"]:
            assert c["effectiveness_due_date"] < "2026-10-30"

    def test_stability_pull_window_variance_identification(self):
        """Verifies stability pull calendar flags timepoints outside ±7-day ICH Q1A window."""
        stab_data = get_stability_pull_readiness_data(reference_date="2026-10-30")
        assert stab_data["pulls_next_30_days_count"] >= 4
        assert stab_data["pulls_outside_window_count"] >= 1

        for p in stab_data["pulls_outside_window"]:
            assert abs(p["days_variance_from_target"]) > 7

    def test_plain_language_readiness_flags_hygiene(self):
        """Verifies readiness flags contain plain language, counts, ages, and zero banned words."""
        dataset = get_consolidated_readiness_dataset(reference_date="2026-10-30")
        flags = generate_readiness_flags(dataset)

        assert len(flags) >= 4

        # Check empirical content
        assert any("Out-of-Specification" in f and "calendar days" in f for f in flags)
        assert any("CAPA" in f and "effectiveness-check" in f for f in flags)
        assert any("stability pull" in f and "±7-day" in f for f in flags)
        assert any("laboratory deviation" in f for f in flags)

        # Prohibited affirmative and subjective words check
        combined = " ".join(flags)
        assert_no_banned_phrases(combined, context="Readiness Flags")
        # Ensure no subjective scoring or pass/fail grades
        for forbidden in ["score", "grade", "pass/fail", "compliant", "inspection-ready"]:
            assert forbidden not in combined.lower()

    def test_inspection_readiness_markdown_export_format(self):
        """Verifies Markdown export includes DRAFT stamp, tables, disclaimers, and zero banned claims."""
        dataset = get_consolidated_readiness_dataset(reference_date="2026-10-30")
        md_text = generate_inspection_readiness_markdown(dataset)

        # DRAFT stamp check
        assert "DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL" in md_text
        assert "Part 11-style demo only" in md_text

        # Required sections
        assert "## 1. Executive Operational Readiness Flags" in md_text
        assert "## 2. Active Out-of-Specification (OOS) Investigations" in md_text
        assert "## 3. Active CAPAs & Effectiveness Check Status" in md_text
        assert "## 4. 30-Day Stability Pull Calendar & ICH Q1A Window Tracking" in md_text
        assert "## 5. Open QC Laboratory Deviations" in md_text
        assert "## 6. Regulatory Basis & Non-Affirmative Governance Notice" in md_text

        # Tables presence
        assert "| Investigation ID | Target Lot |" in md_text
        assert "| CAPA ID | Description |" in md_text
        assert "| Study ID | Lot ID |" in md_text

        # Claims hygiene
        assert_no_banned_phrases(md_text, context="Readiness Markdown Export")

    def test_empty_registers_handling(self):
        """Verifies all modules handle empty datasets gracefully without exceptions or ZeroDivisionError."""
        # 1. Contract Lab Tracker empty summary
        empty_lab_summary = compute_contract_lab_summary([])
        assert empty_lab_summary["total_packages"] == 0
        assert empty_lab_summary["on_time_rate_pct"] == 100.0

        # 2. Inspection Readiness empty dataset
        empty_dataset = {
            "reference_date": "2026-10-30",
            "open_oos": [],
            "capa_data": {"open_capas": [], "open_capa_count": 0, "overdue_effectiveness_checks": [], "overdue_effectiveness_count": 0},
            "stability_data": {"pulls_next_30_days": [], "pulls_next_30_days_count": 0, "pulls_outside_window": [], "pulls_outside_window_count": 0},
            "open_deviations": [],
        }
        empty_flags = generate_readiness_flags(empty_dataset)
        assert len(empty_flags) == 4
        assert "0 open Out-of-Specification" in empty_flags[0]
        assert "0 open CAPA" in empty_flags[1]

        empty_md = generate_inspection_readiness_markdown(empty_dataset)
        assert "DRAFT - NOT A GMP RECORD" in empty_md
        assert_no_banned_phrases(empty_md, context="Empty Readiness Markdown")
