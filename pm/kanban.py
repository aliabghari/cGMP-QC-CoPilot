"""
cGMP-QC-CoPilot: Action Item Kanban Board & Operational Tracker.

Manages QC actions across standard stages:
[Backlog] -> [In Progress] -> [Blocked] -> [Review] -> [Done]

Enforces:
- Mandatory completion notes before moving any item to 'Done'
- Aging tracking (days in status) & overdue date flagging (>14 days / past SLA)
- Part 11-style audit logging for state transitions
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from core.audit import AuditTrail
from core.stamps import assert_no_banned_phrases

VALID_KANBAN_STATUSES = ["Backlog", "In Progress", "Blocked", "Review", "Done"]


def get_kanban_actions(
    conn: sqlite3.Connection,
    project_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    owner_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves action items with computed overdue flags and aging diagnostics.
    """
    query = """
    SELECT
        id, title, owner, due_date, status, project_id,
        dependency_id, age_days, created_at, updated_at
    FROM action_items
    WHERE 1=1
    """
    params: List[Any] = []
    if project_id:
        query += " AND project_id = ?"
        params.append(project_id)
    if status_filter:
        query += " AND status = ?"
        params.append(status_filter)
    if owner_filter:
        query += " AND owner = ?"
        params.append(owner_filter)

    query += " ORDER BY age_days DESC, due_date ASC"

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results = []
    for r in rows:
        aid, title, owner, due, st, pid, dep, age, cr, up = r
        is_overdue = (due < today_str) and (st != "Done")
        is_stale = (age >= 14) and (st != "Done")
        results.append({
            "id": aid,
            "title": title,
            "owner": owner,
            "due_date": due,
            "status": st,
            "project_id": pid,
            "dependency_id": dep,
            "age_days": age,
            "is_overdue": is_overdue,
            "is_stale": is_stale,
            "created_at": cr,
            "updated_at": up,
        })
    return results


def move_kanban_action_status(
    conn: sqlite3.Connection,
    action_id: str,
    new_status: str,
    completion_note: Optional[str] = None,
    actor_name: str = "QC Specialist",
    actor_role: str = "Analyst",
    justification: str = "Action item workflow transition",
    audit_trail: Optional[AuditTrail] = None,
) -> Dict[str, Any]:
    """
    Transitions an action item to a new status.
    CRITICAL RULE: Moving to 'Done' strictly requires a non-empty completion note.
    """
    if new_status not in VALID_KANBAN_STATUSES:
        raise ValueError(f"Invalid status: {new_status}. Allowed: {VALID_KANBAN_STATUSES}")

    if new_status == "Done":
        if not completion_note or len(completion_note.strip()) < 5:
            raise ValueError(
                "Moving an action to 'Done' strictly requires a substantive completion note (>= 5 chars)."
            )

    cursor = conn.cursor()
    cursor.execute("SELECT status, project_id, title FROM action_items WHERE id = ?", (action_id,))
    row = cursor.fetchone()
    if not row:
        raise KeyError(f"Action item {action_id} not found.")

    old_status, project_id, title = row[0], row[1], row[2]
    now_utc = datetime.now(timezone.utc).isoformat()

    full_note = completion_note.strip() if completion_note else ""
    if full_note:
        assert_no_banned_phrases(full_note)

    with conn:
        conn.execute(
            """
            UPDATE action_items
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_status, now_utc, action_id),
        )

    if audit_trail is not None:
        audit_trail.log_action(
            actor_name=actor_name,
            actor_role=actor_role,
            action="KANBAN_STATUS_TRANSITION",
            entity_type="action_item",
            entity_id=action_id,
            justification=f"{justification} [Note: {full_note}]" if full_note else justification,
            payload={
                "action_id": action_id,
                "project_id": project_id,
                "title": title,
                "old_status": old_status,
                "new_status": new_status,
                "completion_note": full_note,
                "timestamp": now_utc,
            },
        )

    return {
        "action_id": action_id,
        "old_status": old_status,
        "new_status": new_status,
        "completion_note": full_note,
        "updated_at": now_utc,
    }


def create_kanban_action(
    conn: sqlite3.Connection,
    title: str,
    owner: str,
    due_date: str,
    project_id: str,
    status: str = "Backlog",
    dependency_id: Optional[str] = None,
    actor_name: str = "QC Specialist",
    actor_role: str = "Analyst",
    audit_trail: Optional[AuditTrail] = None,
) -> str:
    """
    Creates a new action item in the Kanban board.
    """
    assert_no_banned_phrases(title)
    if status not in VALID_KANBAN_STATUSES:
        raise ValueError(f"Invalid status: {status}")

    action_id = f"ACT-{datetime.now(timezone.utc).strftime('%m%d%H%M%S')}"
    now_utc = datetime.now(timezone.utc).isoformat()

    with conn:
        conn.execute(
            """
            INSERT INTO action_items (
                id, title, owner, due_date, status, project_id,
                dependency_id, age_days, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (action_id, title, owner, due_date, status, project_id, dependency_id, 0, now_utc, now_utc),
        )

    if audit_trail is not None:
        audit_trail.log_action(
            actor_name=actor_name,
            actor_role=actor_role,
            action="CREATE_ACTION_ITEM",
            entity_type="action_item",
            entity_id=action_id,
            justification="Added new QC action item to project tracker",
            payload={"action_id": action_id, "title": title, "project_id": project_id},
        )

    return action_id
