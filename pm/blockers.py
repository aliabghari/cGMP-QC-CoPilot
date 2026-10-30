"""
cGMP-QC-CoPilot: Blocker & Impediment Radar Engine.

Monitors operational roadblocks across 5 standardized laboratory categories:
1. Reagent / Reference Standard
2. Instrument Down / Maintenance
3. Analyst Qualification / Absence
4. Client QTA / Specification Alignment
5. Method Transfer / Co-validation

Provides automated escalation triggers based on project priority tier and aging thresholds:
- Tier 1 project: Age >= 5 days -> Escalate to Director of QA
- Tier 2/3 project: Age >= 10 days -> Escalate to QC PM Lead / Local MSO
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.audit import AuditTrail
from core.stamps import DRAFT_STAMP, assert_no_banned_phrases
from pm.config import BLOCKER_ESCALATION_POLICY, ROADBLOCK_TIER_CONFIG
import pandas as pd
try:
    import streamlit as st
except ImportError:
    st = None

BLOCKER_CATEGORIES = [
    "Reagent / Reference Standard",
    "Instrument Down / Maintenance",
    "Analyst Qualification / Absence",
    "Client QTA / Specification Alignment",
    "Method Transfer / Co-validation",
]

# Extended categories for Module 2
EXTENDED_ROADBLOCK_CATEGORIES = [
    "Pending Approval",
    "Late Sample Arrival",
    "Client Dependency",
    "Vendor Dependency",
]

ALL_BLOCKER_CATEGORIES = BLOCKER_CATEGORIES + EXTENDED_ROADBLOCK_CATEGORIES


# -----------------------------------------------------------------------------
# Module 2: Roadblock Tier Assignment & Escalation Message Drafts
# -----------------------------------------------------------------------------
def assign_roadblock_escalation_tier(
    days_open: int,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Assigns an escalation tier (Team / QC Lead / Quality Head) based on days-open thresholds.
    Thresholds are configurable in pm/config.py:
    - days_open <= team_tier_max_days (default: 3) -> 'Team'
    - days_open <= qc_lead_tier_max_days (default: 7) -> 'QC Lead'
    - days_open > qc_lead_tier_max_days -> 'Quality Head'
    """
    cfg = config or ROADBLOCK_TIER_CONFIG
    team_max = cfg.get("team_tier_max_days", 3)
    qc_lead_max = cfg.get("qc_lead_tier_max_days", 7)

    if days_open <= team_max:
        return "Team"
    elif days_open <= qc_lead_max:
        return "QC Lead"
    else:
        return "Quality Head"


def generate_escalation_draft(
    blocker: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generates an escalation message DRAFT for human project manager review and transmission.
    Adheres strictly to Guardrail G2 / G6:
    - Prominently displays DRAFT_STAMP.
    - Explicitly reminds the human user: NEVER auto-send.
    - Contains zero prohibited affirmative compliance claims.
    """
    cfg = config or ROADBLOCK_TIER_CONFIG
    days_open = blocker.get("days_open", blocker.get("age_days", 0))
    tier = blocker.get("escalation_tier") or assign_roadblock_escalation_tier(days_open, cfg)
    roles = cfg.get("roles", {})
    target_role = roles.get(tier, "Operational Supervisor (Example Policy, Configurable)")

    b_id = blocker.get("id", "BLK-UNKNOWN")
    category = blocker.get("category", "General")
    impacted_wp = blocker.get("impacted_work_package", blocker.get("impact", "General Work Package"))
    owner = blocker.get("owner", "Unassigned")
    desc = blocker.get("description", "No description provided.")

    draft_body = (
        f"--- {DRAFT_STAMP} ---\n"
        f"ESCALATION LEVEL: {tier.upper()} TIER\n"
        f"TARGET AUDIENCE: {target_role}\n"
        f"SUBJECT: [ESCALATION NOTICE] Operational Roadblock {b_id} on {impacted_wp}\n\n"
        f"Dear Operations Leadership,\n\n"
        f"This is an automated escalation draft regarding operational roadblock {b_id}.\n\n"
        f"DETAILS:\n"
        f"- Roadblock ID: {b_id}\n"
        f"- Category: {category}\n"
        f"- Impacted Work Package: {impacted_wp}\n"
        f"- Assigned Owner: {owner}\n"
        f"- Aging Duration: {days_open} day(s) open (Tier Threshold: {tier})\n"
        f"- Description: {desc}\n\n"
        f"RECOMMENDED NEXT ACTIONS:\n"
        f"1. Review roadblock root cause with assigned owner {owner}.\n"
        f"2. Coordinate cross-functional intervention to clear dependency.\n"
        f"3. Record meeting outcomes and formal decision rationale in the hash-chained audit trail.\n\n"
        f"NOTE: This message is a DRAFT prepared for human review. "
        f"Human project manager review and manual transmission required. Do not auto-send.\n"
        f"--- END DRAFT ---"
    )

    assert_no_banned_phrases(draft_body, context="Escalation Draft")
    return draft_body


def get_seed_extended_roadblocks(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Returns 8 synthetic roadblocks exercising all 4 extended categories and all 3 escalation tiers,
    including exact threshold boundary cases (3 days -> Team, 7 days -> QC Lead).
    """
    cfg = config or ROADBLOCK_TIER_CONFIG
    raw_items = [
        {
            "id": "BLK-EXT-101",
            "category": "Pending Approval",
            "owner": "Marcus Vance (QA Director)",
            "days_open": 2,
            "impacted_work_package": "WP-CEX-01 (Lot Spesolimab Release)",
            "description": "Pre-execution test protocol revision awaiting QA signature.",
            "status": "OPEN",
        },
        {
            "id": "BLK-EXT-102",
            "category": "Pending Approval",
            "owner": "Dr. Sarah Lin (QC Specialist)",
            "days_open": 3,  # Boundary case: exactly 3 days -> Team tier
            "impacted_work_package": "WP-icIEF-02 (Lot Faricimab-Stability)",
            "description": "Sample preparation laboratory investigation memo pending supervisory review.",
            "status": "OPEN",
        },
        {
            "id": "BLK-EXT-103",
            "category": "Late Sample Arrival",
            "owner": "Elena Rostova (MSAT Lead)",
            "days_open": 5,
            "impacted_work_package": "WP-HARVEST-W43 (Upstream Bioreactor)",
            "description": "Fed-batch fermentation cell density plateau delayed harvest release by 48 hours.",
            "status": "IN_ANALYSIS",
        },
        {
            "id": "BLK-EXT-104",
            "category": "Late Sample Arrival",
            "owner": "Sample Coordination Group",
            "days_open": 7,  # Boundary case: exactly 7 days -> QC Lead tier
            "impacted_work_package": "WP-STAB-28C (Lot NS-2026-004 Excursion)",
            "description": "Thermal shipper hold at port customs clearance awaiting air transit release certificate.",
            "status": "OPEN",
        },
        {
            "id": "BLK-EXT-105",
            "category": "Client Dependency",
            "owner": "CDMO Client PM Lead",
            "days_open": 8,  # > 7 days -> Quality Head tier
            "impacted_work_package": "WP-SPEC-QTA-01 (Client BI-CAP-101)",
            "description": "Client alignment required on updated CE-SDS impurity reporting threshold in QTA.",
            "status": "OPEN",
        },
        {
            "id": "BLK-EXT-106",
            "category": "Vendor Dependency",
            "owner": "QC Metrology Specialist",
            "days_open": 1,
            "impacted_work_package": "WP-UPLC-SKID-02 (Waters Acquity)",
            "description": "Replacement pressure transducer shipment tracking number pending vendor dispatch.",
            "status": "OPEN",
        },
        {
            "id": "BLK-EXT-107",
            "category": "Vendor Dependency",
            "owner": "Reference Standard Lead",
            "days_open": 10,  # Quality Head tier
            "impacted_work_package": "WP-REF-STD-RS09 (Primary Monomer RS)",
            "description": "Certified reference standard manufacturer batch requalification testing behind schedule.",
            "status": "OPEN",
        },
        {
            "id": "BLK-EXT-108",
            "category": "Client Dependency",
            "owner": "CDMO Technical Specialist",
            "days_open": 4,  # QC Lead tier
            "impacted_work_package": "WP-BIOASSAY-04 (Potency Assay)",
            "description": "Client QA clarification requested on cell line passage count documentation.",
            "status": "IN_ANALYSIS",
        },
    ]

    for item in raw_items:
        item["escalation_tier"] = assign_roadblock_escalation_tier(item["days_open"], cfg)

    return raw_items


def render_extended_roadblocks_subtab(
    conn: Optional[sqlite3.Connection] = None,
    audit_trail: Optional[AuditTrail] = None,
) -> None:
    """Renders the interactive Extended Roadblocks sub-tab UI."""
    from pm.ui_view import render_feature_help_popover

    render_feature_help_popover(
        title="Extended Operational Roadblocks & Escalation Radar",
        summary="Classifies laboratory roadblocks across 4 extended operational categories and computes tier-based escalation drafts.",
        formula_or_logic="Deterministic aging thresholds (<=3d Team, 4-7d QC Lead, >7d Quality Head) generating human-in-the-loop escalation drafts.",
        regulatory_basis="21 CFR 211.100(a) (Written procedures for production and process control deviations)",
        demo_action="Select any roadblock to review its details and generate a draft escalation notification for leadership review.",
        what_this_does_not_do="Does not transmit automated emails or notifications; does not bypass site-specific deviation governance or SOP workflows.",
    )

    st.markdown(
        f"""
        <div class="gate-box-pass">
        <strong>{DRAFT_STAMP}</strong><br>
        Roadblock radar tracking four operational bottleneck categories: Pending Approval, Late Sample Arrival,
        Client Dependency, and Vendor Dependency. Generates human-in-the-loop escalation drafts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "_extended_roadblocks" not in st.session_state:
        st.session_state["_extended_roadblocks"] = get_seed_extended_roadblocks()

    roadblocks: List[Dict[str, Any]] = st.session_state["_extended_roadblocks"]

    # Summary metrics by tier
    team_count = sum(1 for r in roadblocks if r["escalation_tier"] == "Team")
    qc_count = sum(1 for r in roadblocks if r["escalation_tier"] == "QC Lead")
    qa_count = sum(1 for r in roadblocks if r["escalation_tier"] == "Quality Head")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Roadblocks", len(roadblocks))
    m2.metric("Team Tier (≤3d)", team_count)
    m3.metric("QC Lead Tier (4-7d)", qc_count)
    m4.metric("Quality Head Tier (>7d)", qa_count)

    # Filter controls
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        tier_filter = st.selectbox("Filter by Escalation Tier", options=["All Tiers", "Team", "QC Lead", "Quality Head"])
    with f_c2:
        cat_filter = st.selectbox("Filter by Category", options=["All Categories"] + EXTENDED_ROADBLOCK_CATEGORIES)

    filtered_rbs = [
        r for r in roadblocks
        if (tier_filter == "All Tiers" or r["escalation_tier"] == tier_filter)
        and (cat_filter == "All Categories" or r["category"] == cat_filter)
    ]

    # Display table
    t_rows = []
    for r in filtered_rbs:
        t_rows.append({
            "Blocker ID": r["id"],
            "Category": r["category"],
            "Impacted Work Package": r["impacted_work_package"],
            "Days Open": r["days_open"],
            "Escalation Tier": r["escalation_tier"],
            "Assigned Owner": r["owner"],
            "Status": r["status"],
            "Description": r["description"],
        })

    st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

    # Draft Escalation Message Generator
    st.markdown("#### ✉️ Human-in-the-Loop Escalation Draft Generator")
    st.caption("Select an operational roadblock to generate a tailored escalation notification draft for leadership review.")

    rb_options = [r["id"] for r in filtered_rbs]
    if rb_options:
        selected_id = st.selectbox("Select Roadblock for Escalation Draft", options=rb_options, key="ext_rb_sel")
        selected_rb = next(r for r in filtered_rbs if r["id"] == selected_id)

        draft_text = generate_escalation_draft(selected_rb)
        st.text_area(
            "Generated Escalation Draft (Human PM Review Required Prior to Sending)",
            value=draft_text,
            height=260,
            key=f"draft_area_{selected_id}",
        )
        st.info("ℹ️ Draft is prepared for copy-paste and human review. No automated transmission is performed.")
    else:
        st.info("No roadblocks match the active filter criteria.")


def evaluate_blocker_escalation(
    blocker_age_days: int,
    project_tier: str,
    severity: str = "HIGH",
    status: str = "OPEN",
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates escalation policy against operational aging thresholds.
    NOTE: Aging thresholds (Tier 1 >= 5d, Tier 2/3 >= 10d) and escalation roles
    are an EXAMPLE POLICY (CONFIGURABLE) per site SOP.
    """
    if status == "RESOLVED":
        return {
            "escalate": False,
            "escalation_level": "None (Resolved)",
            "urgency": "Resolved",
            "threshold_days": None,
            "policy_note": "Example policy, configurable",
        }

    cfg = policy or BLOCKER_ESCALATION_POLICY
    t1_thresh = cfg.get("tier1_aging_threshold_days", 5)
    t2_thresh = cfg.get("tier2_tier3_aging_threshold_days", 10)
    crit_fast_track = cfg.get("critical_severity_fast_track_days", 2)
    t1_role = cfg.get("tier1_escalation_role", "Director of Quality Assurance (Example Policy, Configurable)")
    t2_role = cfg.get("tier2_tier3_escalation_role", "QC Project Management Lead / Local MSO (Example Policy, Configurable)")

    is_tier1 = "Tier 1" in project_tier or "Tier 1A" in project_tier or "Tier 1B" in project_tier
    threshold = t1_thresh if is_tier1 else t2_thresh

    if blocker_age_days >= threshold or (severity == "CRITICAL" and blocker_age_days >= crit_fast_track):
        level = t1_role if is_tier1 else t2_role
        urgency = "CRITICAL_ESCALATION" if is_tier1 else "STANDARD_ESCALATION"
        return {
            "escalate": True,
            "escalation_level": level,
            "urgency": urgency,
            "threshold_days": threshold,
            "days_over_threshold": blocker_age_days - threshold,
            "policy_note": "Example policy, configurable",
        }

    return {
        "escalate": False,
        "escalation_level": "Laboratory Supervisor (Monitoring; Example Policy, Configurable)",
        "urgency": "MONITORED",
        "threshold_days": threshold,
        "days_over_threshold": 0,
        "policy_note": "Example policy, configurable",
    }


def get_blocker_radar_data(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Retrieves all active and resolved blockers with automated escalation assessments.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            b.id, b.category, b.description, b.age_days,
            b.severity, b.impact, b.owner, b.status,
            b.prompt_text, b.created_at
        FROM blocker_items b
        ORDER BY
            CASE b.severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END,
            b.age_days DESC
        """
    )
    rows = cursor.fetchall()
    results = []

    for r in rows:
        bid, cat, desc, age, sev, imp, own, st, prompt, cr = r
        # Estimate project tier from context or default
        tier_guess = "Tier 1A" if ("BI-CAP-101" in imp or "PROJ-001" in imp or "PROJ-002" in imp) else "Tier 2"
        esc_info = evaluate_blocker_escalation(age, tier_guess, sev, st)

        # Estimated delay days from severity
        delay_map = {"CRITICAL": 5.0, "HIGH": 3.0, "MEDIUM": 1.5, "LOW": 0.5}
        estimated_delay = delay_map.get(sev, 1.0) if st != "RESOLVED" else 0.0

        results.append({
            "id": bid,
            "category": cat,
            "description": desc,
            "age_days": age,
            "severity": sev,
            "impact": imp,
            "estimated_delay_days": estimated_delay,
            "owner": own,
            "status": st,
            "prompt_text": prompt,
            "created_at": cr,
            "escalate": esc_info["escalate"],
            "escalation_level": esc_info["escalation_level"],
            "urgency": esc_info["urgency"],
        })

    return results


def resolve_blocker(
    conn: sqlite3.Connection,
    blocker_id: str,
    resolution_notes: str,
    actor_name: str = "QC PM Lead",
    actor_role: str = "Principal Specialist",
    audit_trail: Optional[AuditTrail] = None,
) -> Dict[str, Any]:
    """
    Marks a blocker as RESOLVED with resolution notes and audit logging.
    """
    assert_no_banned_phrases(resolution_notes)
    if not resolution_notes or len(resolution_notes.strip()) < 5:
        raise ValueError("Resolution notes must be provided (>= 5 chars).")

    cursor = conn.cursor()
    cursor.execute("SELECT category, description FROM blocker_items WHERE id = ?", (blocker_id,))
    row = cursor.fetchone()
    if not row:
        raise KeyError(f"Blocker {blocker_id} not found.")

    with conn:
        conn.execute(
            """
            UPDATE blocker_items
            SET status = 'RESOLVED', prompt_text = prompt_text || ' [RESOLVED: ' || ? || ']'
            WHERE id = ?
            """,
            (resolution_notes.strip(), blocker_id),
        )

    if audit_trail is not None:
        audit_trail.log_action(
            actor_name=actor_name,
            actor_role=actor_role,
            action="RESOLVE_BLOCKER",
            entity_type="blocker",
            entity_id=blocker_id,
            justification=resolution_notes,
            payload={"blocker_id": blocker_id, "resolution": resolution_notes},
        )

    return {"blocker_id": blocker_id, "status": "RESOLVED", "notes": resolution_notes}
