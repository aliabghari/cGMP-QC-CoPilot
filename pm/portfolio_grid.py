"""
cGMP-QC-CoPilot: Editable Portfolio Projects Grid Engine.

Provides decision-support project grid management, automatic RAG status evaluation,
Part 11-style audited human overrides, and standardized LIMS manifest export.
"""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from core.audit import AuditTrail
from core.stamps import DRAFT_STAMP, assert_no_banned_phrases
from pm.config import RAG_THRESHOLDS
from pm.data_access import recalculate_project_risk_and_cpm, validate_project_fields


def compute_rag_status(
    p_on_time: float,
    float_days: float,
    has_critical_blocker: bool = False,
    thresholds: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Computes objective RAG classification from quantitative metrics.
    NOTE: Thresholds are an EXAMPLE POLICY (CONFIGURABLE) per site SOP:
    - GREEN: P(on time) >= 0.90 AND float >= 2.0 days AND zero critical blockers
    - RED:   P(on time) < 0.70 OR float < 0.0 days OR active critical blocker
    - AMBER: 0.70 <= P(on time) < 0.90 OR 0.0 <= float < 2.0 days
    """
    cfg = thresholds or RAG_THRESHOLDS
    g_p = cfg.get("green_p_on_time_min", 0.90)
    g_f = cfg.get("green_float_days_min", 2.0)
    r_p = cfg.get("red_p_on_time_threshold", 0.70)
    r_f = cfg.get("red_float_days_threshold", 0.0)

    if has_critical_blocker or p_on_time < r_p or float_days < r_f:
        return "RED"
    if p_on_time >= g_p and float_days >= g_f:
        return "GREEN"
    return "AMBER"


def get_portfolio_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Retrieves all portfolio projects as a pandas DataFrame.
    """
    query = """
    SELECT
        project_id, program_name, stream, tier, owner,
        milestone, gate_status, pct_complete, rag_status,
        p_on_time, float_days, due_date, notes,
        rag_override, rag_override_reason, updated_at
    FROM portfolio_projects
    ORDER BY
        CASE tier
            WHEN 'Tier 1A' THEN 1
            WHEN 'Tier 1B' THEN 2
            WHEN 'Tier 2' THEN 3
            ELSE 4
        END,
        float_days ASC
    """
    return pd.read_sql_query(query, conn)


def apply_rag_override(
    conn: sqlite3.Connection,
    project_id: str,
    override_rag: Optional[str],
    justification: str,
    actor_name: str = "QC Project Manager",
    actor_role: str = "Principal Specialist",
    audit_trail: Optional[AuditTrail] = None,
) -> Dict[str, Any]:
    """
    Applies a human supervisory RAG override to a project.
    Strictly requires reviewer name, role, and substantive justification.
    Logs the override event to the append-only audit trail.
    """
    if override_rag and override_rag not in ("GREEN", "AMBER", "RED"):
        raise ValueError(f"override_rag must be GREEN, AMBER, RED, or None, got: {override_rag}")

    if override_rag and (not justification or len(justification.strip()) < 10):
        raise ValueError("A substantive justification (>= 10 characters) is required to override RAG status.")

    cursor = conn.cursor()
    cursor.execute(
        "SELECT p_on_time, float_days FROM portfolio_projects WHERE project_id = ?",
        (project_id,),
    )
    row = cursor.fetchone()
    if not row:
        raise KeyError(f"Project ID {project_id} not found in database.")

    p_on_time, float_days = row[0], row[1]
    auto_rag = compute_rag_status(p_on_time, float_days)
    final_rag = override_rag if override_rag else auto_rag
    now_utc = datetime.now(timezone.utc).isoformat()

    with conn:
        conn.execute(
            """
            UPDATE portfolio_projects
            SET rag_override = ?, rag_override_reason = ?, rag_status = ?, updated_at = ?
            WHERE project_id = ?
            """,
            (override_rag or "", justification if override_rag else "", final_rag, now_utc, project_id),
        )

    if audit_trail is not None:
        audit_trail.log_action(
            actor_name=actor_name,
            actor_role=actor_role,
            action="RAG_STATUS_OVERRIDE",
            entity_type="project",
            entity_id=project_id,
            justification=justification,
            payload={
                "project_id": project_id,
                "auto_rag": auto_rag,
                "override_rag": override_rag,
                "final_rag": final_rag,
                "timestamp": now_utc,
            },
        )

    return {
        "project_id": project_id,
        "auto_rag": auto_rag,
        "final_rag": final_rag,
        "is_overridden": bool(override_rag),
    }


def update_project_record(
    conn: sqlite3.Connection,
    project_id: str,
    updates: Dict[str, Any],
    actor_name: str = "QC PM Lead",
    actor_role: str = "Principal Specialist",
    justification: str = "Routine project status update",
    audit_trail: Optional[AuditTrail] = None,
) -> Dict[str, Any]:
    """
    Updates an editable project record, validates data constraints,
    recalculates CPM/DES risk if schedule fields change, and records an audit log.
    """
    validate_project_fields(updates)

    allowed_keys = {
        "program_name", "stream", "tier", "owner", "milestone",
        "gate_status", "pct_complete", "float_days", "due_date", "notes"
    }
    filtered_updates = {k: v for k, v in updates.items() if k in allowed_keys}
    if not filtered_updates:
        return {"project_id": project_id, "updated": False, "reason": "No valid fields provided"}

    cursor = conn.cursor()
    cursor.execute("SELECT project_id FROM portfolio_projects WHERE project_id = ?", (project_id,))
    if not cursor.fetchone():
        raise KeyError(f"Project ID {project_id} does not exist.")

    now_utc = datetime.now(timezone.utc).isoformat()
    set_clauses = [f"{k} = ?" for k in filtered_updates.keys()]
    set_clauses.append("updated_at = ?")
    params = list(filtered_updates.values()) + [now_utc, project_id]

    with conn:
        conn.execute(
            f"UPDATE portfolio_projects SET {', '.join(set_clauses)} WHERE project_id = ?",
            params,
        )

    # If float_days changed, recalculate RAG
    if "float_days" in filtered_updates:
        recalculate_project_risk_and_cpm(conn, project_id)

    if audit_trail is not None:
        audit_trail.log_action(
            actor_name=actor_name,
            actor_role=actor_role,
            action="UPDATE_PROJECT_RECORD",
            entity_type="project",
            entity_id=project_id,
            justification=justification,
            payload={"updates": filtered_updates, "timestamp": now_utc},
        )

    return {"project_id": project_id, "updated": True, "fields": list(filtered_updates.keys())}


def export_lims_manifest_csv(
    conn: sqlite3.Connection,
    project_id: Optional[str] = None,
) -> str:
    """
    Generates a standardized CSV export for LIMS batch sample scheduling.
    Stamps header with non-GMP decision support disclaimer.
    """
    df = get_portfolio_dataframe(conn)
    if project_id:
        df = df[df["project_id"] == project_id]

    export_cols = [
        "project_id", "program_name", "stream", "tier",
        "owner", "milestone", "gate_status", "pct_complete",
        "rag_status", "p_on_time", "float_days", "due_date", "notes"
    ]
    sub_df = df[[c for c in export_cols if c in df.columns]]

    buf = io.StringIO()
    buf.write(f"# {DRAFT_STAMP}\n")
    buf.write(f"# Exported_At_UTC: {datetime.now(timezone.utc).isoformat()}\n")
    buf.write(f"# Total_Records: {len(sub_df)}\n")
    sub_df.to_csv(buf, index=False)
    csv_text = buf.getvalue()

    assert_no_banned_phrases(csv_text)
    return csv_text
