"""
cGMP-QC-CoPilot: Operational Registers & KPI Engine.

Manages:
1. RAID Log (Risks, Assumptions, Issues, Dependencies) with 5x5 rating matrix
2. RACI Matrix with strictly ONE Accountable (A) role per activity enforcement
3. Stakeholder Register & Communication Plan
4. Team Charter & Operational KPI metrics (OTD, RFT, Cycle Time, Backlog, Skid Utilization)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.audit import AuditTrail
from core.stamps import assert_no_banned_phrases


# -----------------------------------------------------------------------------
# 1. RAID Log
# -----------------------------------------------------------------------------
def get_raid_log(conn: sqlite3.Connection, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves RAID entries ordered by risk score (Likelihood x Severity).
    """
    query = """
    SELECT
        id, project_id, category, description,
        probability, severity, rating_score,
        owner, mitigation, status, created_at
    FROM raid_items
    WHERE 1=1
    """
    params = []
    if project_id:
        query += " AND project_id = ?"
        params.append(project_id)
    query += " ORDER BY rating_score DESC"

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    return [
        {
            "id": r[0],
            "project_id": r[1],
            "category": r[2],
            "description": r[3],
            "probability": r[4],
            "severity": r[5],
            "rating_score": r[6],
            "owner": r[7],
            "mitigation": r[8],
            "status": r[9],
            "created_at": r[10],
        }
        for r in rows
    ]


def add_raid_item(
    conn: sqlite3.Connection,
    project_id: str,
    category: str,
    description: str,
    probability: int,
    severity: int,
    owner: str,
    mitigation: str,
    status: str = "OPEN",
    actor_name: str = "QC PM Lead",
    actor_role: str = "Principal Specialist",
    audit_trail: Optional[AuditTrail] = None,
) -> str:
    """
    Adds a new RAID item. Rating score is computed deterministically as Probability x Severity.
    """
    assert_no_banned_phrases(description)
    assert_no_banned_phrases(mitigation)
    if not (1 <= probability <= 5) or not (1 <= severity <= 5):
        raise ValueError("Probability and Severity must both be integers between 1 and 5.")

    valid_cats = ["Risk", "Assumption", "Issue", "Dependency"]
    if category not in valid_cats:
        raise ValueError(f"Invalid category: {category}. Allowed: {valid_cats}")

    score = probability * severity
    raid_id = f"RAID-{datetime.now(timezone.utc).strftime('%m%d%H%M%S')}"
    now_utc = datetime.now(timezone.utc).isoformat()

    with conn:
        conn.execute(
            """
            INSERT INTO raid_items (
                id, project_id, category, description,
                probability, severity, rating_score,
                owner, mitigation, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (raid_id, project_id, category, description, probability, severity, score, owner, mitigation, status, now_utc),
        )

    if audit_trail is not None:
        audit_trail.log_action(
            actor_name=actor_name,
            actor_role=actor_role,
            action="ADD_RAID_ITEM",
            entity_type="raid",
            entity_id=raid_id,
            justification=f"Logged new {category}: {description[:40]}",
            payload={"raid_id": raid_id, "score": score, "category": category},
        )

    return raid_id


# -----------------------------------------------------------------------------
# 2. RACI Matrix with Single Accountable Constraint
# -----------------------------------------------------------------------------
def validate_raci_accountable(accountable: str) -> None:
    """
    Strictly enforces that each RACI entry has EXACTLY ONE Accountable role.
    Raises ValueError if 0 or >1 Accountable roles are assigned.
    """
    if not accountable or not accountable.strip():
        raise ValueError("RACI validation failed: An activity must have an Accountable role (cannot be empty).")

    # If comma, semicolon, or slash is used to list multiple roles
    delimiters = [",", ";", "/", " and ", "&"]
    for d in delimiters:
        if d in accountable.lower():
            raise ValueError(
                f"RACI validation failed: Strictly ONE Accountable role permitted per activity. Found multiple in '{accountable}'."
            )


def get_raci_matrix(conn: sqlite3.Connection, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves RACI entries and validates the single Accountable constraint.
    """
    query = """
    SELECT id, project_id, activity, responsible, accountable, consulted, informed
    FROM raci_entries
    WHERE 1=1
    """
    params = []
    if project_id:
        query += " AND project_id = ?"
        params.append(project_id)

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()

    results = []
    for r in rows:
        validate_raci_accountable(r[4])
        results.append({
            "id": r[0],
            "project_id": r[1],
            "activity": r[2],
            "responsible": r[3],
            "accountable": r[4],
            "consulted": r[5],
            "informed": r[6],
        })
    return results


def add_raci_entry(
    conn: sqlite3.Connection,
    project_id: str,
    activity: str,
    responsible: str,
    accountable: str,
    consulted: str,
    informed: str,
) -> str:
    """
    Adds a RACI entry after verifying the single Accountable role constraint.
    """
    validate_raci_accountable(accountable)
    entry_id = f"RACI-{datetime.now(timezone.utc).strftime('%m%d%H%M%S')}"

    with conn:
        conn.execute(
            """
            INSERT INTO raci_entries (
                id, project_id, activity, responsible, accountable, consulted, informed
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entry_id, project_id, activity, responsible, accountable.strip(), consulted, informed),
        )
    return entry_id


# -----------------------------------------------------------------------------
# 3. Stakeholder Register & Communication Plan
# -----------------------------------------------------------------------------
def get_stakeholders_and_comms(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Retrieves the stakeholder register and associated communication plan.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, role, organization, influence, interest, engagement_strategy FROM stakeholders")
    stk_rows = cursor.fetchall()
    stakeholders = [
        {
            "id": r[0],
            "name": r[1],
            "role": r[2],
            "organization": r[3],
            "influence": r[4],
            "interest": r[5],
            "engagement_strategy": r[6],
        }
        for r in stk_rows
    ]

    cursor.execute(
        """
        SELECT c.id, c.stakeholder_id, s.name, s.role, c.channel, c.frequency, c.owner, c.objective
        FROM communication_plan c
        JOIN stakeholders s ON c.stakeholder_id = s.id
        """
    )
    comm_rows = cursor.fetchall()
    comm_plan = [
        {
            "id": r[0],
            "stakeholder_id": r[1],
            "stakeholder_name": r[2],
            "stakeholder_role": r[3],
            "channel": r[4],
            "frequency": r[5],
            "owner": r[6],
            "objective": r[7],
        }
        for r in comm_rows
    ]

    return {"stakeholders": stakeholders, "communication_plan": comm_plan}


# -----------------------------------------------------------------------------
# 4. Operational KPIs & Formulas
# -----------------------------------------------------------------------------
def compute_operational_kpi_metrics(
    total_lots: int,
    on_time_lots: int,
    rft_lots: int,
    total_cycle_days: float,
    backlog_lots: int,
    max_skid_hours: float,
    staffed_skid_capacity_hours: float = 120.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes standard QC operational KPIs using verified mathematical formulas.
    """
    if total_lots <= 0:
        raise ValueError("total_lots must be > 0 to compute percentage metrics.")

    otd_pct = round((on_time_lots / total_lots) * 100.0, 1)
    rft_pct = round((rft_lots / total_lots) * 100.0, 1)
    mean_cycle_time = round(total_cycle_days / total_lots, 1)
    skid_util_pct = round((max_skid_hours / staffed_skid_capacity_hours) * 100.0, 1)

    otd_rag = "GREEN" if otd_pct >= 95.0 else ("AMBER" if otd_pct >= 90.0 else "RED")
    rft_rag = "GREEN" if rft_pct >= 98.0 else ("AMBER" if rft_pct >= 95.0 else "RED")
    cycle_rag = "GREEN" if mean_cycle_time <= 10.0 else ("AMBER" if mean_cycle_time <= 14.0 else "RED")
    backlog_rag = "GREEN" if backlog_lots <= 5 else ("AMBER" if backlog_lots <= 8 else "RED")
    util_rag = "GREEN" if skid_util_pct <= 85.0 else ("AMBER" if skid_util_pct <= 100.0 else "RED")

    return {
        "OTD": {
            "name": "On-Time Delivery (OTD %)",
            "value": otd_pct,
            "target": 95.0,
            "unit": "%",
            "rag": otd_rag,
            "formula": "(on_time_lots / total_lots) * 100.0",
        },
        "RFT": {
            "name": "Right-First-Time (RFT %)",
            "value": rft_pct,
            "target": 98.0,
            "unit": "%",
            "rag": rft_rag,
            "formula": "(rft_lots / total_lots) * 100.0",
        },
        "CycleTime": {
            "name": "Mean QC Turnaround Cycle Time",
            "value": mean_cycle_time,
            "target": 10.0,
            "unit": "Days",
            "rag": cycle_rag,
            "formula": "total_cycle_days / total_lots",
        },
        "Backlog": {
            "name": "Active Testing Backlog",
            "value": backlog_lots,
            "target": 5,
            "unit": "Lots",
            "rag": backlog_rag,
            "formula": "count_lots_in_testing_queue",
        },
        "SkidUtilization": {
            "name": "Peak Skid Utilization",
            "value": skid_util_pct,
            "target": 85.0,
            "unit": "%",
            "rag": util_rag,
            "formula": "(max_skid_hours / staffed_capacity) * 100.0",
        },
    }


def get_team_charter_kpis(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Retrieves stored team charter KPIs from the database.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, kpi_name, target, actual, unit, period, rag_status, calculation_formula FROM team_charter_kpis")
    rows = cursor.fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "target": r[2],
            "actual": r[3],
            "unit": r[4],
            "period": r[5],
            "rag": r[6],
            "formula": r[7],
        }
        for r in rows
    ]
