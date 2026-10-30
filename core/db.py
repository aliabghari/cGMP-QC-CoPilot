"""
cGMP-QC-CoPilot: SQLite Database Abstraction and Migration Engine.

Manages relational persistence for:
- Audit Trail (append-only hash chain)
- Action Tracker & Blocker Radar (WP2)
- Portfolio Projects Grid (WP2)
- Root Causes, Deviation Types & Lessons Learned Knowledge Base (WP4)
- Proactive FMEA & Reactive CAPA State Machine (WP5)
- Vendor & CRO Telemetry (WP8)
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "cgmp_qc.db"


MIGRATIONS: List[Tuple[int, str, List[str]]] = [
    (
        1,
        "create_audit_trail_table",
        [
            """
            CREATE TABLE IF NOT EXISTS audit_trail (
                record_index INTEGER PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                actor_name TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                justification TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT,
                previous_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL,
                disclaimer TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_checkpoint (
                checkpoint_id TEXT PRIMARY KEY,
                record_count INTEGER NOT NULL,
                head_hash TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL
            );
            """,
            """
            CREATE TRIGGER IF NOT EXISTS trg_audit_trail_no_update
            BEFORE UPDATE ON audit_trail
            BEGIN
                SELECT RAISE(ABORT, 'UPDATE operations are strictly prohibited on append-only audit_trail');
            END;
            """,
            """
            CREATE TRIGGER IF NOT EXISTS trg_audit_trail_no_delete
            BEFORE DELETE ON audit_trail
            BEGIN
                SELECT RAISE(ABORT, 'DELETE operations are strictly prohibited on append-only audit_trail');
            END;
            """
        ],
    ),
    (
        2,
        "create_actions_and_blockers_tables",
        [
            """
            CREATE TABLE IF NOT EXISTS action_items (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                owner TEXT NOT NULL,
                due_date TEXT NOT NULL,
                status TEXT NOT NULL,
                project_id TEXT NOT NULL,
                dependency_id TEXT,
                age_days INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS blocker_items (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                age_days INTEGER DEFAULT 0,
                severity TEXT NOT NULL,
                impact TEXT NOT NULL,
                owner TEXT NOT NULL,
                status TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """,
        ],
    ),
    (
        3,
        "create_portfolio_projects_table",
        [
            """
            CREATE TABLE IF NOT EXISTS portfolio_projects (
                project_id TEXT PRIMARY KEY,
                program_name TEXT NOT NULL,
                stream TEXT NOT NULL,
                tier TEXT NOT NULL,
                owner TEXT NOT NULL,
                milestone TEXT NOT NULL,
                gate_status TEXT NOT NULL,
                pct_complete REAL DEFAULT 0.0,
                rag_status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        ],
    ),
    (
        4,
        "create_knowledge_base_tables",
        [
            """
            CREATE TABLE IF NOT EXISTS root_causes (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                investigation_tier TEXT NOT NULL,
                description TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS deviation_types (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                area TEXT NOT NULL,
                description TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS lessons_learned (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                project_id TEXT NOT NULL,
                method_name TEXT NOT NULL,
                root_cause_id TEXT NOT NULL,
                deviation_type_id TEXT NOT NULL,
                what_happened TEXT NOT NULL,
                what_learned TEXT NOT NULL,
                preventive_action TEXT NOT NULL,
                owner TEXT NOT NULL,
                investigation_tier TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (root_cause_id) REFERENCES root_causes(id),
                FOREIGN KEY (deviation_type_id) REFERENCES deviation_types(id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS lesson_links (
                id TEXT PRIMARY KEY,
                lesson_id TEXT NOT NULL,
                linked_lesson_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                FOREIGN KEY (lesson_id) REFERENCES lessons_learned(id),
                FOREIGN KEY (linked_lesson_id) REFERENCES lessons_learned(id)
            );
            """,
        ],
    ),
    (
        5,
        "create_fmea_and_capa_tables",
        [
            """
            CREATE TABLE IF NOT EXISTS fmea_items (
                id TEXT PRIMARY KEY,
                process_step TEXT NOT NULL,
                failure_mode TEXT NOT NULL,
                failure_effect TEXT NOT NULL,
                severity INTEGER NOT NULL,
                potential_cause TEXT NOT NULL,
                occurrence INTEGER NOT NULL,
                current_controls TEXT NOT NULL,
                detection INTEGER NOT NULL,
                rpn INTEGER NOT NULL,
                action_priority TEXT NOT NULL,
                recommended_action TEXT,
                post_severity INTEGER,
                post_occurrence INTEGER,
                post_detection INTEGER,
                post_rpn INTEGER,
                lessons_learned_count INTEGER DEFAULT 0
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS capa_records (
                id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                root_cause_id TEXT,
                title TEXT NOT NULL,
                corrective_action TEXT NOT NULL,
                preventive_action TEXT NOT NULL,
                owner TEXT NOT NULL,
                due_date TEXT NOT NULL,
                status TEXT NOT NULL,
                effectiveness_criteria TEXT,
                effectiveness_due_date TEXT,
                effectiveness_result TEXT,
                closed_by TEXT,
                closed_at TEXT,
                justification TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (root_cause_id) REFERENCES root_causes(id)
            );
            """,
        ],
    ),
    (
        6,
        "create_vendor_and_cro_tables",
        [
            """
            CREATE TABLE IF NOT EXISTS vendors (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                service_type TEXT NOT NULL,
                contact TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS vendor_telemetry (
                id TEXT PRIMARY KEY,
                vendor_id TEXT NOT NULL,
                batch_or_shipment_id TEXT NOT NULL,
                date TEXT NOT NULL,
                on_time INTEGER NOT NULL,
                right_first_time INTEGER NOT NULL,
                deviation_count INTEGER DEFAULT 0,
                turnaround_days REAL NOT NULL,
                notes TEXT,
                FOREIGN KEY (vendor_id) REFERENCES vendors(id)
            );
            """,
        ],
    ),
    (
        7,
        "create_wp2_pm_tables",
        [
            # Extend portfolio_projects with PM fields
            """
            ALTER TABLE portfolio_projects ADD COLUMN p_on_time REAL DEFAULT 0.95;
            """,
            """
            ALTER TABLE portfolio_projects ADD COLUMN float_days REAL DEFAULT 0.0;
            """,
            """
            ALTER TABLE portfolio_projects ADD COLUMN due_date TEXT DEFAULT '';
            """,
            """
            ALTER TABLE portfolio_projects ADD COLUMN notes TEXT DEFAULT '';
            """,
            """
            ALTER TABLE portfolio_projects ADD COLUMN rag_override TEXT DEFAULT '';
            """,
            """
            ALTER TABLE portfolio_projects ADD COLUMN rag_override_reason TEXT DEFAULT '';
            """,
            # Milestones table
            """
            CREATE TABLE IF NOT EXISTS milestones (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                target_date TEXT NOT NULL,
                status TEXT NOT NULL,
                is_critical INTEGER DEFAULT 0,
                FOREIGN KEY (project_id) REFERENCES portfolio_projects(project_id)
            );
            """,
            # Decision Gates table
            """
            CREATE TABLE IF NOT EXISTS decision_gates (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                gate_name TEXT NOT NULL,
                gate_type TEXT NOT NULL,
                status TEXT NOT NULL,
                criteria TEXT NOT NULL,
                review_date TEXT,
                approved_by TEXT,
                FOREIGN KEY (project_id) REFERENCES portfolio_projects(project_id)
            );
            """,
            # WBS Tasks table
            """
            CREATE TABLE IF NOT EXISTS wbs_tasks (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                skid TEXT NOT NULL,
                duration_hours REAL NOT NULL,
                start_day REAL NOT NULL,
                end_day REAL NOT NULL,
                float_days REAL NOT NULL,
                is_critical INTEGER DEFAULT 0,
                predecessors TEXT DEFAULT '',
                status TEXT NOT NULL,
                flexibility_class TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES portfolio_projects(project_id)
            );
            """,
            # RAID Items table (Risks, Assumptions, Issues, Dependencies)
            """
            CREATE TABLE IF NOT EXISTS raid_items (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                probability INTEGER NOT NULL,
                severity INTEGER NOT NULL,
                rating_score INTEGER NOT NULL,
                owner TEXT NOT NULL,
                mitigation TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES portfolio_projects(project_id)
            );
            """,
            # RACI Entries table
            """
            CREATE TABLE IF NOT EXISTS raci_entries (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                activity TEXT NOT NULL,
                responsible TEXT NOT NULL,
                accountable TEXT NOT NULL,
                consulted TEXT NOT NULL,
                informed TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES portfolio_projects(project_id)
            );
            """,
            # Stakeholders Register
            """
            CREATE TABLE IF NOT EXISTS stakeholders (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                organization TEXT NOT NULL,
                influence TEXT NOT NULL,
                interest TEXT NOT NULL,
                engagement_strategy TEXT NOT NULL
            );
            """,
            # Communication Plan
            """
            CREATE TABLE IF NOT EXISTS communication_plan (
                id TEXT PRIMARY KEY,
                stakeholder_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                frequency TEXT NOT NULL,
                owner TEXT NOT NULL,
                objective TEXT NOT NULL,
                FOREIGN KEY (stakeholder_id) REFERENCES stakeholders(id)
            );
            """,
            # Team Charter & Synthetic Operational KPIs
            """
            CREATE TABLE IF NOT EXISTS team_charter_kpis (
                id TEXT PRIMARY KEY,
                kpi_name TEXT NOT NULL,
                target REAL NOT NULL,
                actual REAL NOT NULL,
                unit TEXT NOT NULL,
                period TEXT NOT NULL,
                rag_status TEXT NOT NULL,
                calculation_formula TEXT NOT NULL
            );
            """,
            # Modular Tech Transfer Sections
            """
            CREATE TABLE IF NOT EXISTS tech_transfer_sections (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                section_name TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                signed_off_by TEXT,
                signed_off_at TEXT,
                FOREIGN KEY (project_id) REFERENCES portfolio_projects(project_id)
            );
            """,
            # Sample Arrivals Forecast & Alerts
            """
            CREATE TABLE IF NOT EXISTS sample_arrivals (
                id TEXT PRIMARY KEY,
                lot_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                method TEXT NOT NULL,
                scheduled_day REAL NOT NULL,
                predicted_q10_day REAL NOT NULL,
                predicted_q50_day REAL NOT NULL,
                predicted_q90_day REAL NOT NULL,
                confidence_level REAL NOT NULL,
                alert_status TEXT NOT NULL,
                heads_up_draft TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES portfolio_projects(project_id)
            );
            """,
        ],
    ),
]


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Returns an open SQLite connection with Row factory enabled.
    If db_path is None, defaults to data/cgmp_qc.db.
    """
    if db_path is None:
        DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
        path = str(DEFAULT_DB_PATH)
    else:
        path = str(db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Initializes the database and runs all unapplied migrations in order.
    """
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );
            """
        )

        cursor = conn.cursor()
        cursor.execute("SELECT version FROM schema_migrations")
        applied_versions = {row[0] for row in cursor.fetchall()}

        now_utc = datetime.now(timezone.utc).isoformat()
        for version, name, sql_statements in MIGRATIONS:
            if version not in applied_versions:
                for stmt in sql_statements:
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (version, name, now_utc),
                )

    return conn


def get_db_status(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Returns database migration status and table row counts."""
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(version) FROM schema_migrations")
    latest_migration = cursor.fetchone()[0] or 0

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cursor.fetchall()]

    counts = {}
    for tbl in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
            counts[tbl] = cursor.fetchone()[0]
        except Exception:
            counts[tbl] = -1

    return {
        "latest_migration_version": latest_migration,
        "total_migrations_available": len(MIGRATIONS),
        "table_counts": counts,
    }
