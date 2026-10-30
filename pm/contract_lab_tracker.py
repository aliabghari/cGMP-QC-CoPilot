"""
pm/contract_lab_tracker.py

Contract Lab Data Review Tracker (QC PM Module 1).
Manages outsourced analytical testing packages across external contract testing laboratories (CDMO/CRO).
Enforces:
- SOW turnaround tracking, overdue alerts, and at-risk (< 3 days) flags.
- Gate rule: No package can transition to 'Accepted' with open queries or without >= 10-char justification.
- Gated state transitions writing strictly to the persistent, hash-chained AuditTrail ledger.
- Rejection of transitions if AuditTrail is unavailable (never silent failure).

Synthetic data only. All outputs carry the DRAFT disclaimer:
"DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL"
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from core.audit import AuditTrail
from core.stamps import DRAFT_STAMP, PART11_DISCLAIMER


# -----------------------------------------------------------------------------
# Synthetic Data Seeding
# -----------------------------------------------------------------------------
def get_seed_contract_lab_packages(reference_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns 10 representative synthetic contract laboratory packages.
    All laboratory facilities and lot numbers are fictional examples.
    """
    ref_dt = (
        datetime.strptime(reference_date[:10], "%Y-%m-%d").date()
        if reference_date
        else date(2026, 10, 30)
    )

    # Date offsets relative to reference_date
    d_m10 = (ref_dt - pd.Timedelta(days=10)).isoformat()
    d_m5 = (ref_dt - pd.Timedelta(days=5)).isoformat()
    d_m2 = (ref_dt - pd.Timedelta(days=2)).isoformat()
    d_p2 = (ref_dt + pd.Timedelta(days=2)).isoformat()  # Due in 2 days (At-Risk)
    d_p3 = (ref_dt + pd.Timedelta(days=3)).isoformat()  # Due in exactly 3 days (At-Risk boundary)
    d_p5 = (ref_dt + pd.Timedelta(days=5)).isoformat()
    d_p12 = (ref_dt + pd.Timedelta(days=12)).isoformat()
    d_m15 = (ref_dt - pd.Timedelta(days=15)).isoformat()  # Overdue package

    return [
        {
            "package_id": "PKG-CRO-2026-001",
            "lab_name": "Northbay Analytical Labs (Fictional)",
            "method": "SEC-HPLC Monomer Purity",
            "sow_due_date": d_m15,  # Overdue!
            "data_received_date": d_m10,
            "review_status": "Queries Open",
            "reviewer": "Dr. Sarah Lin (QC Specialist)",
            "open_query_count": 2,
            "oldest_query_age_days": 8,
            "notes": "Query regarding baseline integration drift across injections 12-16.",
        },
        {
            "package_id": "PKG-CRO-2026-002",
            "lab_name": "Cedar Ridge Bioanalytics (Fictional)",
            "method": "Residual Host Cell DNA qPCR",
            "sow_due_date": d_p2,  # Due in 2 days (At-Risk boundary)
            "data_received_date": None,
            "review_status": "Awaiting Data",
            "reviewer": "Alex Mercer (QC PM)",
            "open_query_count": 0,
            "oldest_query_age_days": 0,
            "notes": "Testing completed; certificate of analysis pending secondary review.",
        },
        {
            "package_id": "PKG-CRO-2026-003",
            "lab_name": "Apex Biologics Testing (Fictional)",
            "method": "icIEF Charge Variant Profile",
            "sow_due_date": d_m2,
            "data_received_date": d_m5,
            "review_status": "In Review",
            "reviewer": "Elena Gomez (Senior QC Analyst)",
            "open_query_count": 0,
            "oldest_query_age_days": 0,
            "notes": "System suitability verified; reviewing electropherogram peak identities.",
        },
        {
            "package_id": "PKG-CRO-2026-004",
            "lab_name": "Cascadia Bioanalytical Labs (Fictional)",
            "method": "Cell-Based Relative Potency Bioassay",
            "sow_due_date": d_m5,
            "data_received_date": d_m10,
            "review_status": "Accepted",
            "reviewer": "Dr. Marcus Vance (QC Lead)",
            "open_query_count": 0,
            "oldest_query_age_days": 0,
            "notes": "4-PL parallelism test met criteria; EC50 ratio within 0.85-1.15 specification.",
        },
        {
            "package_id": "PKG-CRO-2026-005",
            "lab_name": "Northbay Analytical Labs (Fictional)",
            "method": "HCP ELISA Impurity Assay",
            "sow_due_date": d_p5,
            "data_received_date": d_m2,
            "review_status": "In Review",
            "reviewer": "Dr. Sarah Lin (QC Specialist)",
            "open_query_count": 0,
            "oldest_query_age_days": 0,
            "notes": "Standard curve R2 > 0.995; confirming dilutional linearity recovery.",
        },
        {
            "package_id": "PKG-CRO-2026-006",
            "lab_name": "Highland Bioservices (Fictional)",
            "method": "N-Glycan Profiling by HILIC-FLR",
            "sow_due_date": d_p12,
            "data_received_date": None,
            "review_status": "Awaiting Data",
            "reviewer": "Rachel Thorne (QC Analyst)",
            "open_query_count": 0,
            "oldest_query_age_days": 0,
            "notes": "Samples received at CRO facility; digest step in progress.",
        },
        {
            "package_id": "PKG-CRO-2026-007",
            "lab_name": "Cedar Ridge Bioanalytics (Fictional)",
            "method": "Mycoplasma PCR Rapid Detection",
            "sow_due_date": d_m5,
            "data_received_date": d_m10,
            "review_status": "Accepted",
            "reviewer": "Elena Gomez (Senior QC Analyst)",
            "open_query_count": 0,
            "oldest_query_age_days": 0,
            "notes": "Negative across all test matrices; positive extraction controls passed.",
        },
        {
            "package_id": "PKG-CRO-2026-008",
            "lab_name": "Apex Biologics Testing (Fictional)",
            "method": "Intact Mass Determination (LC-MS)",
            "sow_due_date": d_m2,
            "data_received_date": d_m5,
            "review_status": "Queries Open",
            "reviewer": "Julian Hayes (QC Operations)",
            "open_query_count": 1,
            "oldest_query_age_days": 3,
            "notes": "Mass accuracy query issued regarding deconvoluted spectrum shoulder peak.",
        },
        {
            "package_id": "PKG-CRO-2026-009",
            "lab_name": "Olympic Analytical Testing (Fictional)",
            "method": "Sub-Visible Particle Analysis (HIAC)",
            "sow_due_date": d_p5,
            "data_received_date": d_m2,
            "review_status": "Queries Open",
            "reviewer": "Dr. Marcus Vance (QC Lead)",
            "open_query_count": 1,
            "oldest_query_age_days": 2,
            "notes": "Cumulative particle counts at >= 10um within range; clarification on tare volume.",
        },
        {
            "package_id": "PKG-CRO-2026-010",
            "lab_name": "Cedar Ridge Bioanalytics (Fictional)",
            "method": "Heavy Metal Impurities by ICP-MS",
            "sow_due_date": d_p3,  # Due in exactly 3 days (At-Risk boundary)
            "data_received_date": None,
            "review_status": "Awaiting Data",
            "reviewer": "Alex Mercer (QC PM)",
            "open_query_count": 0,
            "oldest_query_age_days": 0,
            "notes": "Digestion completed; ICP-MS autosampler run scheduled for tonight.",
        },
    ]


# -----------------------------------------------------------------------------
# Business Rules & Flag Computation
# -----------------------------------------------------------------------------
def compute_turnaround_and_flags(
    pkg: Dict[str, Any],
    today_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes SOW turnaround, overdue status, and near-term risk flags.
    - is_overdue: True if review_status != 'Accepted' and today > sow_due_date.
    - is_at_risk: True if review_status != 'Accepted' and 0 <= days_until_due <= 3.
    """
    today_dt = (
        datetime.strptime(today_iso[:10], "%Y-%m-%d").date()
        if today_iso
        else date(2026, 10, 30)
    )
    sow_due_str = pkg.get("sow_due_date", "")[:10]
    status = pkg.get("review_status", "")

    try:
        sow_dt = datetime.strptime(sow_due_str, "%Y-%m-%d").date()
        days_until_due = (sow_dt - today_dt).days
    except ValueError:
        days_until_due = 999

    is_accepted = status == "Accepted"
    is_overdue = not is_accepted and (days_until_due < 0)
    is_at_risk = not is_accepted and (0 <= days_until_due <= 3)

    received_str = pkg.get("data_received_date")
    turnaround_days = None
    if received_str:
        try:
            recv_dt = datetime.strptime(received_str[:10], "%Y-%m-%d").date()
            turnaround_days = (recv_dt - sow_dt).days
        except ValueError:
            pass

    return {
        "days_until_due": days_until_due,
        "is_overdue": is_overdue,
        "is_at_risk": is_at_risk,
        "turnaround_days": turnaround_days,
        "flag_label": (
            "OVERDUE ⚠️"
            if is_overdue
            else ("DUE SOON (≤3d) ⏳" if is_at_risk else ("ACCEPTED ✅" if is_accepted else "ON TRACK"))
        ),
    }


def compute_contract_lab_summary(
    packages: List[Dict[str, Any]],
    today_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes summary metrics across contract lab packages:
    - Distribution by review_status
    - On-time delivery rate
    - Oldest open query and age in days
    """
    if not packages:
        return {
            "total_packages": 0,
            "status_counts": {"Awaiting Data": 0, "In Review": 0, "Queries Open": 0, "Accepted": 0},
            "on_time_rate_pct": 100.0,
            "oldest_open_query_days": 0,
            "oldest_open_query_pkg_id": "None",
            "overdue_count": 0,
            "at_risk_count": 0,
        }

    status_counts: Dict[str, int] = {
        "Awaiting Data": 0,
        "In Review": 0,
        "Queries Open": 0,
        "Accepted": 0,
    }

    overdue_count = 0
    at_risk_count = 0
    oldest_query_age = 0
    oldest_query_pkg = "None"
    on_time_compliant_count = 0
    evaluated_for_ontime = 0

    for p in packages:
        st_val = p.get("review_status", "Awaiting Data")
        status_counts[st_val] = status_counts.get(st_val, 0) + 1

        flags = compute_turnaround_and_flags(p, today_iso=today_iso)
        if flags["is_overdue"]:
            overdue_count += 1
        if flags["is_at_risk"]:
            at_risk_count += 1

        q_count = p.get("open_query_count", 0)
        q_age = p.get("oldest_query_age_days", 0)
        if q_count > 0 and q_age > oldest_query_age:
            oldest_query_age = q_age
            oldest_query_pkg = p.get("package_id", "N/A")

        # On-time calculation for packages with received data
        if p.get("data_received_date"):
            evaluated_for_ontime += 1
            if not flags["is_overdue"]:
                on_time_compliant_count += 1

    on_time_pct = (
        round((on_time_compliant_count / evaluated_for_ontime) * 100.0, 1)
        if evaluated_for_ontime > 0
        else 100.0
    )

    return {
        "total_packages": len(packages),
        "status_counts": status_counts,
        "on_time_rate_pct": on_time_pct,
        "oldest_open_query_days": oldest_query_age,
        "oldest_open_query_pkg_id": oldest_query_pkg,
        "overdue_count": overdue_count,
        "at_risk_count": at_risk_count,
    }


def transition_package_to_accepted(
    pkg: Dict[str, Any],
    reviewer_name: str,
    reviewer_role: str,
    justification: str,
    audit_trail: Optional[AuditTrail] = None,
    timestamp_utc: Optional[str] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Attempts to transition a contract laboratory package to 'Accepted'.
    Strict Rules:
    1. If audit_trail is None -> REFUSE transition with visible notice. Never skip audit logging.
    2. If open_query_count > 0 -> REFUSE transition. Queries must be 0.
    3. If justification length < 10 characters -> REFUSE transition.
    4. On success -> Update status to 'Accepted', call audit_trail.log_action, return updated copy.
    """
    pkg_id = pkg.get("package_id", "UNKNOWN")

    # Rule 1: Audit trail mandatory
    if audit_trail is None:
        return (
            False,
            f"Transition Refused: Hash-chained audit trail ledger is unavailable for {pkg_id}. "
            "ALCOA+ principles strictly prohibit ungoverned data acceptance without cryptographic audit records.",
            None,
        )

    # Rule 2: Open queries check
    open_queries = pkg.get("open_query_count", 0)
    if open_queries > 0:
        return (
            False,
            f"Transition Refused: Package {pkg_id} has {open_queries} unresolved analytical query/queries. "
            "All vendor queries must be formally resolved prior to data package acceptance.",
            None,
        )

    # Rule 3: Justification length check
    clean_just = justification.strip() if justification else ""
    if len(clean_just) < 10:
        return (
            False,
            f"Transition Refused: Reviewer justification must be at least 10 characters (received {len(clean_just)}). "
            "Please provide a substantive review rationale.",
            None,
        )

    # Success path: update copy and log to audit trail
    updated = dict(pkg)
    updated["review_status"] = "Accepted"
    updated["reviewer"] = reviewer_name.strip()
    updated["notes"] = f"Accepted by {reviewer_name.strip()}: {clean_just}"

    rec = audit_trail.log_action(
        actor_name=reviewer_name.strip(),
        actor_role=reviewer_role.strip() or "QC Analytical Reviewer",
        action="ACCEPT_CONTRACT_LAB_PACKAGE",
        entity_type="contract_lab_package",
        entity_id=pkg_id,
        justification=clean_just,
        payload={
            "package_id": pkg_id,
            "lab_name": pkg.get("lab_name"),
            "method": pkg.get("method"),
            "previous_status": pkg.get("review_status"),
            "new_status": "Accepted",
            "open_query_count": 0,
            "sow_due_date": pkg.get("sow_due_date"),
            "data_received_date": pkg.get("data_received_date"),
        },
        timestamp_utc=timestamp_utc,
    )

    return (
        True,
        f"Package {pkg_id} successfully marked Accepted. Audit record #{rec.index} chained with hash {rec.record_hash[:16]}...",
        updated,
    )


# -----------------------------------------------------------------------------
# Streamlit UI Component
# -----------------------------------------------------------------------------
def render_contract_lab_tracker_tab(
    conn: Optional[sqlite3.Connection] = None,
    audit_trail: Optional[AuditTrail] = None,
) -> None:
    """Renders the interactive Contract Lab Data Review Tracker UI."""
    from pm.ui_view import render_feature_help_popover

    render_feature_help_popover(
        title="Contract Lab Data Review Tracker (CRO/CDMO)",
        summary="Tracks turnaround, SOW commitments, and analytical queries for outsourced testing packages.",
        formula_or_logic="Deterministic date delta vs SOW milestones + zero-query gate rule before data package acceptance.",
        regulatory_basis="21 CFR 211.22(a) (Quality Control Unit vendor data review), ICH Q10 Section 2.7",
        demo_action="Review package PKG-CRO-2026-001 (overdue) or attempt accepting PKG-CRO-2026-008 with open queries to inspect gate enforcement.",
        what_this_does_not_do="Does not connect directly to external CRO LIMS/portals; does not replace qualified human QA batch release disposition.",
    )

    st.markdown(
        f"""
        <div class="gate-box-pass">
        <strong>{DRAFT_STAMP}</strong><br>
        Outsourced testing governance cockpit. Tracks external CRO testing deliverables, monitors query aging,
        and enforces cryptographic ALCOA+ audit trails for analytical package acceptance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for packages
    if "_contract_lab_packages" not in st.session_state:
        st.session_state["_contract_lab_packages"] = get_seed_contract_lab_packages()

    packages: List[Dict[str, Any]] = st.session_state["_contract_lab_packages"]
    summary = compute_contract_lab_summary(packages)

    # KPI summary cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Packages", summary["total_packages"])
    c2.metric("Accepted", summary["status_counts"]["Accepted"])
    c3.metric("Queries Open", summary["status_counts"]["Queries Open"])
    c4.metric(
        "Overdue / At-Risk",
        f"{summary['overdue_count']} / {summary['at_risk_count']}",
        help="Overdue past SOW date / Due within 3 calendar days",
    )
    c5.metric(
        "Oldest Open Query",
        f"{summary['oldest_open_query_days']} days",
        f"Pkg: {summary['oldest_open_query_pkg_id']}",
    )

    st.markdown("#### 📋 Outsourced Analytical Packages Register")

    # Format table for display
    table_rows = []
    for p in packages:
        flags = compute_turnaround_and_flags(p)
        table_rows.append({
            "Package ID": p["package_id"],
            "Laboratory (Fictional)": p["lab_name"],
            "Analytical Method": p["method"],
            "SOW Due Date": p["sow_due_date"],
            "Data Received": p["data_received_date"] or "Pending",
            "Status": p["review_status"],
            "Open Queries": p["open_query_count"],
            "Query Age (d)": p["oldest_query_age_days"],
            "Schedule Flag": flags["flag_label"],
            "Assigned Reviewer": p["reviewer"],
        })

    df_display = pd.DataFrame(table_rows)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Acceptance Action Interface
    st.markdown("#### ✍️ Gated Package Acceptance & ALCOA+ Sign-Off")
    st.caption("Packages can only be accepted when all vendor queries are closed and substantive justification is provided.")

    selectable_pkgs = [p["package_id"] for p in packages if p["review_status"] != "Accepted"]
    if not selectable_pkgs:
        st.success("All outsourced packages are currently Accepted.")
        return

    sel_col1, sel_col2 = st.columns([1.5, 2.5])
    with sel_col1:
        selected_pkg_id = st.selectbox("Select Package to Review", options=selectable_pkgs)
        target_pkg = next(p for p in packages if p["package_id"] == selected_pkg_id)

        # Allow user to resolve queries interactively for demonstration
        if target_pkg["open_query_count"] > 0:
            if st.button(f"Resolve All Queries ({target_pkg['open_query_count']} open)", key="btn_resolve_queries"):
                target_pkg["open_query_count"] = 0
                target_pkg["oldest_query_age_days"] = 0
                target_pkg["review_status"] = "In Review"
                st.session_state["_contract_lab_packages"] = packages
                st.rerun()

    with sel_col2:
        rev_name = st.text_input("Reviewer Name", value="Dr. Sarah Lin", key="cro_rev_name")
        rev_role = st.text_input("Reviewer Role", value="Senior QC Specialist", key="cro_rev_role")
        justification = st.text_area(
            "Human Review Justification (Minimum 10 characters required)",
            placeholder="Confirm chromatograms, integration baselines, and system suitability meet specification...",
            key="cro_justification",
        )

        if st.button("Mark Package as Accepted", key="btn_cro_accept", type="primary"):
            success, msg, updated_pkg = transition_package_to_accepted(
                pkg=target_pkg,
                reviewer_name=rev_name,
                reviewer_role=rev_role,
                justification=justification,
                audit_trail=audit_trail,
            )
            if success and updated_pkg:
                # Update in list
                for idx, p in enumerate(packages):
                    if p["package_id"] == updated_pkg["package_id"]:
                        packages[idx] = updated_pkg
                st.session_state["_contract_lab_packages"] = packages
                st.success(f"✅ {msg}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")
