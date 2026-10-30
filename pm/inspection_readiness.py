"""
pm/inspection_readiness.py

Inspection-Readiness One-Pager (QC PM Module 3).
Consolidates cross-functional operational signals across OOX (OOS/OOT) investigations,
CAPAs, stability pull calendars, and open laboratory deviations.

Key Principles:
- Single holistic operational cockpit for OOS and stability processes.
- Plain language Readiness Flags based strictly on empirical counts and aging.
- Strictly prohibited: affirmative claims, inspection-readiness certifications, compliance scores, grades, or color-coded pass/fail judgements.
- Exportable Markdown summary stamped with DRAFT disclaimer.

Synthetic data only. All outputs carry:
"DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL"
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from core.stamps import DRAFT_STAMP, PART11_DISCLAIMER, assert_no_banned_phrases


# -----------------------------------------------------------------------------
# Data Retrieval & Synthetic Seed Generators
# -----------------------------------------------------------------------------
def get_open_oos_investigations(
    conn: Optional[sqlite3.Connection] = None,
    reference_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns active OOS investigations.
    In enterprise systems, pulled from eQMS (TrackWise/Veeva); here provided as synthetic records.
    """
    return [
        {
            "investigation_id": "OOS-2026-004",
            "batch_id": "LOT-NS-2026-004",
            "program_name": "Spesolimab (Commercial Stream)",
            "method": "SEC-HPLC Monomer Purity",
            "reported_result": "93.8%",
            "specification": "≥ 95.0%",
            "age_days": 48,
            "phase": "Phase I Laboratory Investigation",
            "assigned_lead": "Lead MSO / Dr. Sarah Lin",
            "preliminary_cause": "Assignable instrument hardware backpressure surge (310.4 bar) & chiller fault (17.8°C).",
        },
        {
            "investigation_id": "OOS-2026-007",
            "batch_id": "LOT-OVX-2026-02",
            "program_name": "Ovaleat-CDMO Client Program",
            "method": "icIEF Main Peak Distribution",
            "reported_result": "64.2%",
            "specification": "≥ 70.0%",
            "age_days": 14,
            "phase": "Phase II Cross-Functional Investigation",
            "assigned_lead": "Elena Gomez (Senior QC Specialist)",
            "preliminary_cause": "Under investigation: capillary cartridge ampholyte batch lot variance.",
        },
    ]


def get_capa_readiness_data(
    conn: Optional[sqlite3.Connection] = None,
    reference_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves open CAPAs and flags those past their scheduled effectiveness-check date.
    Reads from SQLite capa_records when connection is present, with synthetic fallback.
    """
    ref_dt = (
        datetime.strptime(reference_date[:10], "%Y-%m-%d").date()
        if reference_date
        else date(2026, 10, 30)
    )

    capas: List[Dict[str, Any]] = []

    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, owner, status, effectiveness_due_date, effectiveness_result, due_date
                FROM capa_records
                WHERE status != 'CLOSED'
                ORDER BY due_date ASC
                """
            )
            rows = cursor.fetchall()
            for r in rows:
                cid, title, owner, st_val, eff_due, eff_res, due = r
                capas.append({
                    "capa_id": cid,
                    "title": title,
                    "owner": owner,
                    "status": st_val,
                    "effectiveness_due_date": eff_due or "",
                    "effectiveness_result": eff_res or "PENDING",
                    "due_date": due,
                })
        except Exception:
            capas = []

    if not capas:
        # Fallback synthetic CAPA register
        d_past1 = (ref_dt - pd.Timedelta(days=12)).isoformat()
        d_past2 = (ref_dt - pd.Timedelta(days=3)).isoformat()
        d_future = (ref_dt + pd.Timedelta(days=25)).isoformat()

        capas = [
            {
                "capa_id": "CAPA-2026-011",
                "title": "Recalibration of Waters UPLC Autosampler Chiller & Pressure Sensors",
                "owner": "QC Metrology Lead",
                "status": "EFFECTIVENESS_CHECK",
                "effectiveness_due_date": d_past1,  # Past effectiveness check!
                "effectiveness_result": "Pending 30-day thermal stability verification run",
                "due_date": (ref_dt - pd.Timedelta(days=30)).isoformat(),
            },
            {
                "capa_id": "CAPA-2026-014",
                "title": "Revision of Mobile Phase pH Buffer Preparation SOP-QC-104",
                "owner": "Analytical Technical Specialist",
                "status": "IMPLEMENTED",
                "effectiveness_due_date": d_past2,  # Past effectiveness check!
                "effectiveness_result": "Pending second-analyst verification testing",
                "due_date": (ref_dt - pd.Timedelta(days=15)).isoformat(),
            },
            {
                "capa_id": "CAPA-2026-018",
                "title": "Dual-Operator Requalification on Maurice icIEF Capillary Cartridges",
                "owner": "QC Operations Supervisor",
                "status": "IN_PROGRESS",
                "effectiveness_due_date": d_future,
                "effectiveness_result": "Scheduled",
                "due_date": d_future,
            },
        ]

    # Evaluate overdue effectiveness checks
    overdue_checks = []
    for c in capas:
        eff_str = c.get("effectiveness_due_date", "")[:10]
        if eff_str:
            try:
                eff_dt = datetime.strptime(eff_str, "%Y-%m-%d").date()
                if eff_dt < ref_dt and c.get("status") != "CLOSED":
                    overdue_checks.append(c)
            except ValueError:
                pass

    return {
        "open_capas": capas,
        "open_capa_count": len(capas),
        "overdue_effectiveness_checks": overdue_checks,
        "overdue_effectiveness_count": len(overdue_checks),
    }


def get_stability_pull_readiness_data(
    conn: Optional[sqlite3.Connection] = None,
    reference_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates stability pulls scheduled over the next 30 calendar days
    and flags any timepoint falling outside the permitted ±7-day ICH Q1A pull window.
    """
    ref_dt = (
        datetime.strptime(reference_date[:10], "%Y-%m-%d").date()
        if reference_date
        else date(2026, 10, 30)
    )

    # 5 synthetic stability timepoints across the 30-day horizon
    pulls = [
        {
            "study_id": "STAB-SPES-5C-T12M",
            "lot_id": "LOT-SPES-2025-08",
            "condition": "5°C ± 3°C (Long Term)",
            "timepoint": "12 Month",
            "scheduled_pull_date": (ref_dt + pd.Timedelta(days=4)).isoformat(),
            "target_pull_date": (ref_dt + pd.Timedelta(days=4)).isoformat(),
            "days_variance_from_target": 0,
            "chamber_id": "Chamber-01 (Rees Verified)",
            "assigned_skid": "UPLC_SEC_CEX",
        },
        {
            "study_id": "STAB-SPES-25C-T6M",
            "lot_id": "LOT-SPES-2026-01",
            "condition": "25°C / 60% RH (Accelerated)",
            "timepoint": "6 Month",
            "scheduled_pull_date": (ref_dt + pd.Timedelta(days=11)).isoformat(),
            "target_pull_date": (ref_dt + pd.Timedelta(days=1)).isoformat(),
            "days_variance_from_target": 10,  # Outside ±7-day window (+10d delay)!
            "chamber_id": "Chamber-03 (Rees Verified)",
            "assigned_skid": "CE_icIEF",
        },
        {
            "study_id": "STAB-OVX-40C-T3M",
            "lot_id": "LOT-OVX-2026-03",
            "condition": "40°C / 75% RH (Stress)",
            "timepoint": "3 Month",
            "scheduled_pull_date": (ref_dt + pd.Timedelta(days=16)).isoformat(),
            "target_pull_date": (ref_dt + pd.Timedelta(days=18)).isoformat(),
            "days_variance_from_target": -2,
            "chamber_id": "Chamber-04 (Rees Verified)",
            "assigned_skid": "UPLC_SEC_CEX",
        },
        {
            "study_id": "STAB-CDMO-5C-T24M",
            "lot_id": "LOT-BI-CAP-103",
            "condition": "5°C ± 3°C (Long Term)",
            "timepoint": "24 Month",
            "scheduled_pull_date": (ref_dt + pd.Timedelta(days=22)).isoformat(),
            "target_pull_date": (ref_dt + pd.Timedelta(days=23)).isoformat(),
            "days_variance_from_target": -1,
            "chamber_id": "Chamber-02 (Rees Verified)",
            "assigned_skid": "Cell_Bioassay",
        },
        {
            "study_id": "STAB-OVX-25C-T9M",
            "lot_id": "LOT-OVX-2025-11",
            "condition": "25°C / 60% RH (Accelerated)",
            "timepoint": "9 Month",
            "scheduled_pull_date": (ref_dt + pd.Timedelta(days=28)).isoformat(),
            "target_pull_date": (ref_dt + pd.Timedelta(days=19)).isoformat(),
            "days_variance_from_target": 9,  # Outside ±7-day window (+9d delay)!
            "chamber_id": "Chamber-03 (Rees Verified)",
            "assigned_skid": "CE_icIEF",
        },
    ]

    pulls_next_30 = [p for p in pulls]
    pulls_outside_window = [p for p in pulls if abs(p["days_variance_from_target"]) > 7]

    return {
        "pulls_next_30_days": pulls_next_30,
        "pulls_next_30_days_count": len(pulls_next_30),
        "pulls_outside_window": pulls_outside_window,
        "pulls_outside_window_count": len(pulls_outside_window),
    }


def get_open_qc_deviations(
    conn: Optional[sqlite3.Connection] = None,
    reference_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns open QC laboratory deviations.
    """
    return [
        {
            "deviation_id": "DEV-QC-2026-11",
            "title": "Autosampler Chiller Temperature Excursion during Sequence Run #42",
            "area": "SEC-HPLC Laboratory Skid #2",
            "severity": "Major",
            "age_days": 18,
            "owner": "Shift B Operations Lead",
            "status": "Under Investigation",
        },
        {
            "deviation_id": "DEV-QC-2026-14",
            "title": "pH Drift in Mobile Phase Buffer Lot B-901 Observed Pre-Run",
            "area": "Buffer Preparation Suite",
            "severity": "Minor",
            "age_days": 6,
            "owner": "Reagent Specialist",
            "status": "Corrective Action Pending",
        },
    ]


# -----------------------------------------------------------------------------
# Plain Language Readiness Flags & Consolidated Dataset
# -----------------------------------------------------------------------------
def get_consolidated_readiness_dataset(
    conn: Optional[sqlite3.Connection] = None,
    reference_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aggregates all inspection-readiness signals into a single structured dictionary.
    """
    oos_list = get_open_oos_investigations(conn, reference_date)
    capa_data = get_capa_readiness_data(conn, reference_date)
    stab_data = get_stability_pull_readiness_data(conn, reference_date)
    dev_list = get_open_qc_deviations(conn, reference_date)

    return {
        "reference_date": reference_date or "2026-10-30",
        "open_oos": oos_list,
        "capa_data": capa_data,
        "stability_data": stab_data,
        "open_deviations": dev_list,
    }


def generate_readiness_flags(dataset: Dict[str, Any]) -> List[str]:
    """
    Generates plain-language Readiness Flags based strictly on observed counts and ages.
    Strictly avoids letter grades, scores, color-coded pass/fail badges,
    or affirmative compliance claims.
    """
    flags: List[str] = []

    # 1. OOS Flags
    open_oos = dataset.get("open_oos", [])
    if open_oos:
        oldest_oos_age = max(o.get("age_days", 0) for o in open_oos)
        flags.append(
            f"{len(open_oos)} open Out-of-Specification (OOS) investigation(s) currently active; "
            f"oldest has been open for {oldest_oos_age} calendar days."
        )
    else:
        flags.append("0 open Out-of-Specification (OOS) investigations currently active.")

    # 2. CAPA Effectiveness Flags
    capa_info = dataset.get("capa_data", {})
    open_capa_count = capa_info.get("open_capa_count", 0)
    overdue_eff_count = capa_info.get("overdue_effectiveness_count", 0)
    if overdue_eff_count > 0:
        flags.append(
            f"{overdue_eff_count} CAPA(s) currently past their scheduled effectiveness-check date "
            f"(out of {open_capa_count} total open CAPAs)."
        )
    else:
        flags.append(
            f"{open_capa_count} open CAPA(s) active; 0 overdue past their effectiveness-check date."
        )

    # 3. Stability Pull Horizon & Window Variance Flags
    stab_info = dataset.get("stability_data", {})
    pulls_30d = stab_info.get("pulls_next_30_days_count", 0)
    pulls_outside = stab_info.get("pulls_outside_window_count", 0)
    if pulls_outside > 0:
        flags.append(
            f"{pulls_outside} stability pull(s) projected outside the approved ±7-day ICH Q1A pull window "
            f"(out of {pulls_30d} pulls scheduled across the next 30 days)."
        )
    else:
        flags.append(
            f"{pulls_30d} stability pull(s) scheduled in the next 30 days; all projected within the ±7-day window."
        )

    # 4. Open Laboratory Deviations Flag
    devs = dataset.get("open_deviations", [])
    if devs:
        oldest_dev_age = max(d.get("age_days", 0) for d in devs)
        flags.append(
            f"{len(devs)} open laboratory deviation(s) pending root-cause investigation; "
            f"oldest age is {oldest_dev_age} days."
        )
    else:
        flags.append("0 open laboratory deviations currently active.")

    for f in flags:
        assert_no_banned_phrases(f, context="Readiness Flags")

    return flags


def generate_inspection_readiness_markdown(dataset: Dict[str, Any]) -> str:
    """
    Generates a formal, printable Markdown export of the Inspection-Readiness One-Pager.
    Includes explicit DRAFT disclaimer, structured empirical tables, and transparent limitation disclosures.
    """
    flags = generate_readiness_flags(dataset)
    ref_date = dataset.get("reference_date", "2026-10-30")

    oos_rows = dataset.get("open_oos", [])
    capa_rows = dataset.get("capa_data", {}).get("open_capas", [])
    stab_rows = dataset.get("stability_data", {}).get("pulls_next_30_days", [])
    dev_rows = dataset.get("open_deviations", [])

    lines = [
        "# Operational Inspection-Readiness One-Pager (OOX & Stability)",
        "",
        f"> **{DRAFT_STAMP}**  ",
        f"> *{PART11_DISCLAIMER}*  ",
        f"> Generated on: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}` | Evaluation Basis Date: `{ref_date}`",
        "",
        "---",
        "",
        "## 1. Executive Operational Readiness Flags (Empirical Observations)",
        "",
    ]

    for f in flags:
        lines.append(f"- 📌 **Observation:** {f}")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Active Out-of-Specification (OOS) Investigations",
        "",
        "| Investigation ID | Target Lot | Method | Result vs Spec | Age (Days) | Current Phase | Assigned Lead |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    if oos_rows:
        for o in oos_rows:
            lines.append(
                f"| `{o['investigation_id']}` | {o['batch_id']} | {o['method']} | {o['reported_result']} (Spec: {o['specification']}) | {o['age_days']} | {o['phase']} | {o['assigned_lead']} |"
            )
    else:
        lines.append("| None | N/A | N/A | N/A | 0 | No active OOS investigations | N/A |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Active CAPAs & Effectiveness Check Status",
        "",
        "| CAPA ID | Description | Owner | Status | Effectiveness Due Date | Effectiveness Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    if capa_rows:
        for c in capa_rows:
            lines.append(
                f"| `{c['capa_id']}` | {c['title']} | {c['owner']} | {c['status']} | {c['effectiveness_due_date'] or 'N/A'} | {c['effectiveness_result']} |"
            )
    else:
        lines.append("| None | N/A | N/A | No active CAPAs | N/A | N/A |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. 30-Day Stability Pull Calendar & ICH Q1A Window Tracking",
        "",
        "| Study ID | Lot ID | Storage Condition | Timepoint | Target Date | Sched Date | Variance (Days) | Window Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    if stab_rows:
        for s in stab_rows:
            v = s["days_variance_from_target"]
            w_status = "OUTSIDE ±7d WINDOW ⚠️" if abs(v) > 7 else "Within ±7d Window"
            lines.append(
                f"| `{s['study_id']}` | {s['lot_id']} | {s['condition']} | {s['timepoint']} | {s['target_pull_date']} | {s['scheduled_pull_date']} | {v:+d} d | {w_status} |"
            )
    else:
        lines.append("| None | N/A | N/A | N/A | N/A | N/A | 0 | No scheduled pulls |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Open QC Laboratory Deviations",
        "",
        "| Deviation ID | Title | Laboratory Area | Severity | Age (Days) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    if dev_rows:
        for d in dev_rows:
            lines.append(
                f"| `{d['deviation_id']}` | {d['title']} | {d['area']} | {d['severity']} | {d['age_days']} | {d['status']} |"
            )
    else:
        lines.append("| None | N/A | N/A | N/A | 0 | No active deviations |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Regulatory Basis & Non-Affirmative Governance Notice",
        "",
        "- **FDA 21 CFR 211.192:** Requires thorough investigation of any unexplained discrepancy or batch failure.",
        "- **ICH Q1A(R2) Section 2.1.7:** Governs stability testing frequency and permitted pull window tolerances.",
        "- **Governance Notice:** This one-pager provides decision support visibility into operational backlog and aging. "
        "It does NOT certify or guarantee inspection readiness or regulatory compliance. Formal inspection disposition requires authorized Quality Assurance review.",
    ])

    doc_str = "\n".join(lines)
    assert_no_banned_phrases(doc_str, context="Inspection-Readiness Markdown")
    return doc_str


# -----------------------------------------------------------------------------
# Streamlit UI Component
# -----------------------------------------------------------------------------
def render_inspection_readiness_tab(
    conn: Optional[sqlite3.Connection] = None,
    audit_trail: Optional[Any] = None,
) -> None:
    """Renders the interactive Inspection-Readiness One-Pager UI."""
    from pm.ui_view import render_feature_help_popover

    render_feature_help_popover(
        title="Inspection-Readiness One-Pager (OOX & Stability)",
        summary="Consolidates empirical aging signals across open OOS investigations, CAPA effectiveness checks, stability pulls, and deviations.",
        formula_or_logic="Deterministic empirical aggregation of open records, aging days, and window deviations without subjective scoring.",
        regulatory_basis="21 CFR 211.192 (Discrepancy Investigation), ICH Q1A(R2) Stability Frequency",
        demo_action="Review plain language Readiness Flags and export the printable Markdown one-pager.",
        what_this_does_not_do="Does not certify inspection readiness or compliance status; does not assign pass/fail grades; does not replace formal QA audit.",
    )

    st.markdown(
        f"""
        <div class="gate-box-pass">
        <strong>{DRAFT_STAMP}</strong><br>
        Operational inspection-readiness overview. Consolidates empirical aging signals across OOX investigations,
        CAPA effectiveness checks, ICH Q1A stability pull windows, and open lab deviations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset = get_consolidated_readiness_dataset(conn=conn)
    flags = generate_readiness_flags(dataset)

    # Top KPI summary cards
    oos_count = len(dataset["open_oos"])
    capa_info = dataset["capa_data"]
    stab_info = dataset["stability_data"]
    dev_count = len(dataset["open_deviations"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active OOS", oos_count, help="Open Out-of-Specification laboratory investigations")
    c2.metric(
        "Open CAPAs",
        capa_info["open_capa_count"],
        f"{capa_info['overdue_effectiveness_count']} Overdue Checks",
    )
    c3.metric(
        "30d Stability Pulls",
        stab_info["pulls_next_30_days_count"],
        f"{stab_info['pulls_outside_window_count']} Outside ±7d",
    )
    c4.metric("Open Lab Deviations", dev_count, help="Active laboratory deviations pending closure")

    # Plain Language Readiness Flags Card
    st.markdown("#### 📌 Operational Readiness Flags (Empirical Observations)")
    flags_html = "".join(
        f"<li style='margin-bottom: 6px; font-size: 12.5px; color: #202124;'><strong>{f}</strong></li>"
        for f in flags
    )
    st.markdown(
        f"""
        <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-left: 4px solid #1a73e8; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
            <ul style="margin: 0; padding-left: 18px;">
                {flags_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed Sub-Sections
    with st.expander("🔍 1. Active Out-of-Specification (OOS) Investigations", expanded=True):
        if dataset["open_oos"]:
            df_oos = pd.DataFrame(dataset["open_oos"])
            st.dataframe(df_oos, use_container_width=True, hide_index=True)
        else:
            st.info("No active OOS investigations.")

    with st.expander("🔍 2. Active CAPAs & Effectiveness Check Status", expanded=True):
        if dataset["capa_data"]["open_capas"]:
            df_capa = pd.DataFrame(dataset["capa_data"]["open_capas"])
            st.dataframe(df_capa, use_container_width=True, hide_index=True)
        else:
            st.info("No open CAPAs recorded.")

    with st.expander("🔍 3. 30-Day Stability Pull Calendar & ICH Q1A Window Tracking", expanded=True):
        if dataset["stability_data"]["pulls_next_30_days"]:
            df_stab = pd.DataFrame(dataset["stability_data"]["pulls_next_30_days"])
            st.dataframe(df_stab, use_container_width=True, hide_index=True)
        else:
            st.info("No stability pulls scheduled in next 30 days.")

    with st.expander("🔍 4. Open QC Laboratory Deviations", expanded=False):
        if dataset["open_deviations"]:
            df_dev = pd.DataFrame(dataset["open_deviations"])
            st.dataframe(df_dev, use_container_width=True, hide_index=True)
        else:
            st.info("No open laboratory deviations.")

    # Export Section
    st.markdown("#### 📥 Printable Markdown One-Pager Export")
    md_content = generate_inspection_readiness_markdown(dataset)

    exp_col1, exp_col2 = st.columns([3, 1], vertical_alignment="center")
    with exp_col1:
        st.caption("Generates a structured, self-contained Markdown briefing containing all empirical tables and the DRAFT stamp.")
    with exp_col2:
        st.download_button(
            label="Download Markdown Briefing",
            data=md_content,
            file_name=f"inspection_readiness_briefing_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            key="btn_download_readiness_md",
        )
