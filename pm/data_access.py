"""
cGMP-QC-CoPilot: Project Management Data Access & Persistence Engine.

Manages relational database access, seeding, audit logging, and CPM/DES updates
for the Project-Management Operating Layer (WP2).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.audit import AuditTrail
from core.db import get_connection, init_db
from models.schedule_risk.des import Task, InstrumentDefinition
from models.schedule_risk.risk import run_monte_carlo_schedule_risk
from models.schedule_risk.optimize import FlexibilityClass

DEFAULT_PORTFOLIO_PATH = (
    Path(__file__).resolve().parent.parent
    / "agents"
    / "agent_1_lean_capacity"
    / "assets"
    / "example_portfolio_10_proj.json"
)


# -----------------------------------------------------------------------------
# Cycle Rejection / Topological Graph Verification
# -----------------------------------------------------------------------------
def check_cycle_in_dependencies(dependency_graph: Dict[str, List[str]]) -> bool:
    """
    Detects whether a directed cycle exists in task or milestone dependencies.
    Returns True if a cycle exists (invalid DAG), False otherwise.
    Uses Kahn's algorithm / 3-color DFS.
    """
    # 0 = unvisited, 1 = visiting (in current DFS recursion stack), 2 = visited
    state: Dict[str, int] = {}
    nodes = set(dependency_graph.keys())
    for deps in dependency_graph.values():
        nodes.update(deps)

    for n in nodes:
        state[n] = 0

    def dfs(node: str) -> bool:
        state[node] = 1  # visiting
        for neighbor in dependency_graph.get(node, []):
            if state.get(neighbor, 0) == 1:
                return True  # Back-edge detected -> cycle
            if state.get(neighbor, 0) == 0:
                if dfs(neighbor):
                    return True
        state[node] = 2  # visited
        return False

    for n in nodes:
        if state[n] == 0:
            if dfs(n):
                return True
    return False


def validate_project_fields(data: Dict[str, Any]) -> None:
    """
    Validates input fields for project edits or creation.
    Raises ValueError on negative durations, invalid percentages, or malformed data.
    """
    pct = data.get("pct_complete")
    if pct is not None:
        if not isinstance(pct, (int, float)) or pct < 0.0 or pct > 100.0:
            raise ValueError(f"pct_complete must be a number between 0.0 and 100.0, got: {pct}")

    float_d = data.get("float_days")
    if float_d is not None:
        if not isinstance(float_d, (int, float)) or float_d < -30.0:
            raise ValueError(f"float_days cannot be below -30 days, got: {float_d}")

    due_date = data.get("due_date")
    if due_date:
        try:
            datetime.strptime(str(due_date)[:10], "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"due_date must be in ISO YYYY-MM-DD format, got: {due_date}")


# -----------------------------------------------------------------------------
# Seeding Logic
# -----------------------------------------------------------------------------
def seed_pm_database(
    conn: sqlite3.Connection,
    portfolio_path: Optional[Path] = None,
    audit_trail: Optional[AuditTrail] = None,
    force: bool = False,
) -> int:
    """
    Populates SQLite with initial synthetic data from example_portfolio_10_proj.json
    and synthetic operational registers.
    Returns the number of projects seeded.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM portfolio_projects")
    count = cursor.fetchone()[0]
    if count > 0 and not force:
        return count

    path = portfolio_path or DEFAULT_PORTFOLIO_PATH
    if not path.exists():
        raise FileNotFoundError(f"Portfolio seed asset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        demo_data = json.load(f)

    now_utc = datetime.now(timezone.utc).isoformat()
    projects = demo_data.get("projects", [])

    with conn:
        for p in projects:
            pid = p["project_id"]
            stream = "Captive Pipeline" if p.get("engine_type") == "Captive Pipeline" else "CDMO Client"
            tier = p.get("priority_tier", "Tier 2")
            owner = "Lead MSO (Example Biologics Site)" if stream == "Captive Pipeline" else "CDMO Client Lead"
            milestone = "Lot Release Testing" if "Release" in p.get("sample_type", "") else "Stability Pull Analysis"
            gate = "Gate 3: Quality Release Review" if "Release" in p.get("sample_type", "") else "Gate 2: Testing In-Progress"
            float_days = float(p.get("float_days", 0.0))
            is_critical = 1 if p.get("is_critical_path", False) else 0

            # Default RAG based on float and tier
            if float_days == 0 and is_critical:
                rag = "AMBER"
            elif float_days < 0:
                rag = "RED"
            else:
                rag = "GREEN"

            due_date = p.get("sla_due_date", "2026-10-30")
            notes = p.get("notes", "")

            # 1. Insert or replace project
            conn.execute(
                """
                INSERT OR REPLACE INTO portfolio_projects (
                    project_id, program_name, stream, tier, owner,
                    milestone, gate_status, pct_complete, rag_status,
                    p_on_time, float_days, due_date, notes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    p.get("program_name", pid),
                    stream,
                    tier,
                    owner,
                    milestone,
                    gate,
                    25.0 if "Release" in milestone else 50.0,
                    rag,
                    0.95 if rag == "GREEN" else (0.82 if rag == "AMBER" else 0.65),
                    float_days,
                    due_date,
                    notes,
                    now_utc,
                ),
            )

            # 2. Insert Milestones
            conn.execute(
                """
                INSERT OR REPLACE INTO milestones (
                    id, project_id, name, target_date, status, is_critical
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"MS-{pid}-01",
                    pid,
                    f"{p.get('program_name')} Sample Delivery",
                    p.get("usp_dsp_delivery_date", "2026-10-19"),
                    "COMPLETED",
                    is_critical,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO milestones (
                    id, project_id, name, target_date, status, is_critical
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"MS-{pid}-02",
                    pid,
                    f"{p.get('program_name')} Final CoA Release",
                    due_date,
                    "IN_PROGRESS",
                    is_critical,
                ),
            )

            # 3. Insert Decision Gates
            conn.execute(
                """
                INSERT OR REPLACE INTO decision_gates (
                    id, project_id, gate_name, gate_type, status, criteria, review_date, approved_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"GATE-{pid}-01",
                    pid,
                    "Pre-Execution Sample Acceptance",
                    "Technical",
                    "PASSED",
                    "Integrity check, temperature log verification, chain of custody confirmation",
                    p.get("usp_dsp_delivery_date", "2026-10-19"),
                    "Sample Coordination Team",
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO decision_gates (
                    id, project_id, gate_name, gate_type, status, criteria, review_date, approved_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"GATE-{pid}-02",
                    pid,
                    "Phase I Laboratory Triage & Disposition",
                    "Quality",
                    "PENDING_REVIEW",
                    "100% test completion, system suitability verified, zero uninvestigated OOX/OOS",
                    due_date,
                    None,
                ),
            )

            # 4. Insert WBS Tasks for each testing panel
            skid_hours = p.get("skid_hours", {})
            skid_map = [
                ("SEC_HPLC", "UPLC_SEC_CEX", skid_hours.get("UPLC_SEC_CEX", 20.0)),
                ("icIEF_CE", "CE_icIEF", skid_hours.get("CE_icIEF", 14.0)),
                ("Potency_Bioassay", "Cell_Bioassay", skid_hours.get("Cell_Bioassay", 16.0)),
            ]
            prev_tid = ""
            for idx, (task_type, skid_code, dur_hours) in enumerate(skid_map):
                tid = f"WBS-{pid}-{task_type}"
                flex_cls = (
                    "FIXED_COMMERCIAL"
                    if "Tier 1" in tier and "Release" in milestone
                    else "FLEXIBLE_WINDOW"
                )
                start_d = 0.0 if idx == 0 else (1.0 if idx == 1 else 2.0)
                end_d = start_d + round(dur_hours / 8.0, 1)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO wbs_tasks (
                        id, project_id, name, skid, duration_hours,
                        start_day, end_day, float_days, is_critical,
                        predecessors, status, flexibility_class
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tid,
                        pid,
                        f"{task_type} Testing Panel",
                        skid_code,
                        dur_hours,
                        start_d,
                        end_d,
                        float_days,
                        is_critical,
                        prev_tid,
                        "IN_PROGRESS" if idx == 0 else "NOT_STARTED",
                        flex_cls,
                    ),
                )
                prev_tid = tid

        # 5. Insert Synthetic Action Items (Kanban)
        actions = [
            ("ACT-101", "Verify Waters Acquity Column Pressure Suitability", "Analyst Shift A", "2026-10-22", "In Progress", "BI-CAP-101", None, 1),
            ("ACT-102", "Calibrate Maurice icIEF Capillary Cartridge Lot #44", "QC Metrology", "2026-10-23", "Backlog", "PROJ-001", "ACT-101", 3),
            ("ACT-103", "Conduct 36h Thermal Excursion Kinetic Review", "Local MSO", "2026-10-21", "Review", "PROJ-002", None, 2),
            ("ACT-104", "Northbay Reference Standard Re-Bridging Protocol", "Stability Lead", "2026-10-18", "Done", "BI-CAP-103", None, 6),
            ("ACT-105", "Resolve Mobile Phase pH 5.8 Buffer Drift", "Shift B Analyst", "2026-10-24", "Blocked", "PROJ-004", None, 4),
        ]
        for aid, title, owner, due, st, pid, dep, age in actions:
            conn.execute(
                """
                INSERT OR REPLACE INTO action_items (
                    id, title, owner, due_date, status, project_id, dependency_id, age_days, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (aid, title, owner, due, st, pid, dep, age, now_utc, now_utc),
            )

        # 6. Insert Synthetic Blocker Items
        blockers = [
            ("BLK-201", "Reagent / Reference Standard", "Primary Reference Standard RS-MAb-09 expired; replacement vial in customs clearance.", 6, "HIGH", "Delays BI-CAP-101 Final CoA Release by up to 2 days.", "QC Reference Standard Lead", "OPEN", "Escalate to customs broker and QA for priority release clearance.", now_utc),
            ("BLK-202", "Instrument Down / Maintenance", "Waters UPLC Skid #2 pressure transducer fluctuating >15 bar.", 2, "MEDIUM", "Shifts flexible stability runs into W43.", "QC Metrology", "IN_ANALYSIS", "Perform emergency seal and transducer recalibration.", now_utc),
            ("BLK-203", "Client QTA / Specification Alignment", "Example CDMO Client specification alignment pending on CE-SDS minor impurity threshold.", 7, "HIGH", "BLA critical path hold for PROJ-001.", "Client Project Manager", "OPEN", "Issue Joint Steering Committee priority clarification draft.", now_utc),
            ("BLK-204", "Analyst Qualification / Absence", "Shift B lead bioassay analyst on emergency leave; backup analyst requires 4-PL requalification.", 3, "MEDIUM", "Cell-Bioassay throughput throttled to 70% nominal.", "QC Supervisor", "IN_ANALYSIS", "Expedite supervisory sign-off on requalification run.", now_utc),
        ]
        for bid, cat, desc, age, sev, imp, own, st, prompt, cr in blockers:
            conn.execute(
                """
                INSERT OR REPLACE INTO blocker_items (
                    id, category, description, age_days, severity, impact, owner, status, prompt_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (bid, cat, desc, age, sev, imp, own, st, prompt, cr),
            )

        # 7. Insert Synthetic RAID Items
        raid_data = [
            ("RAID-01", "BI-CAP-101", "Risk", "Bioreactor harvest delay in USP could shift sample arrival into weekend shift", 4, 4, 16, "Local MSO", "Monitor rolling-origin CQR prediction window daily; pre-alert swing shift", "OPEN"),
            ("RAID-02", "PROJ-001", "Dependency", "Critical viral clearance IPC sample depends on external CRO Northbay Analytical turnaround", 3, 5, 15, "CDMO Lead", "Establish 48h SLA notification clause in Quality Technical Agreement", "IN_MONITORING"),
            ("RAID-03", "PROJ-002", "Issue", "Thermal excursion during air transit (36h at 28C) required Eyring non-isothermal integration", 5, 4, 20, "Lead MSO", "Run 25-member hybrid ensemble to quantify consumed 5C shelf life", "CLOSED"),
            ("RAID-04", "BI-CAP-103", "Assumption", "Assumed site-defined pull window (example +/-7 d) acceptable for W43 leveling per site SOP", 2, 2, 4, "Stability PM", "Verify against registered regulatory commitment document", "VERIFIED"),
        ]
        for rid, pid, cat, desc, prob, sev, score, own, mit, st in raid_data:
            conn.execute(
                """
                INSERT OR REPLACE INTO raid_items (
                    id, project_id, category, description, probability, severity, rating_score, owner, mitigation, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rid, pid, cat, desc, prob, sev, score, own, mit, st, now_utc),
            )

        # 8. Insert Synthetic RACI Entries (Strictly 1 Accountable role!)
        raci_data = [
            ("RACI-01", "BI-CAP-101", "Sample Receipt & Chain of Custody", "Sample Coordinator", "QC Operations Supervisor", "Local MSO", "QA Release"),
            ("RACI-02", "BI-CAP-101", "Analytical Testing & Run Approval", "QC Analyst", "Lab Technical Specialist", "Local MSO", "Client PM"),
            ("RACI-03", "PROJ-001", "Phase I Laboratory Investigation", "Investigating Analyst", "Local MSO", "QC Metrology", "Director of QA"),
            ("RACI-04", "PROJ-002", "Thermal Excursion Assessment", "Lead Analytical Scientist", "Lead MSO", "MSAT Lead", "QA Head"),
            ("RACI-05", "PROJ-004", "Method Transfer Protocol Sign-Off", "Transfer Analyst", "QC PM Lead", "Analytical Development", "Client QA"),
        ]
        for r_id, p_id, act, resp, acc, cons, inf in raci_data:
            conn.execute(
                """
                INSERT OR REPLACE INTO raci_entries (
                    id, project_id, activity, responsible, accountable, consulted, informed
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (r_id, p_id, act, resp, acc, cons, inf),
            )

        # 9. Insert Stakeholders & Communication Plan
        stakeholders = [
            ("STK-01", "Dr. Priya Nair", "External CDMO Partner Lead", "Example CDMO Partner", "HIGH", "HIGH", "Weekly steering call & 24h milestone flash notifications"),
            ("STK-02", "Marcus Vance", "Director of Quality Assurance", "Example Biologics Site Manufacturing Site", "HIGH", "MEDIUM", "Bi-weekly executive summary & automated OOS escalation alerts"),
            ("STK-03", "Elena Rostova", "MSAT USP/DSP Technical Lead", "Example Biologics Site Biomanufacturing", "MEDIUM", "HIGH", "Daily morning standup & harvest arrival Slack integration"),
            ("STK-04", "Julian Hayes", "QC Operations Lead", "Example Biologics Site QC Laboratories", "HIGH", "HIGH", "Real-time skid capacity leveling & analyst shift scheduling"),
        ]
        for sid, name, role, org, inf, int_lvl, strat in stakeholders:
            conn.execute(
                """
                INSERT OR REPLACE INTO stakeholders (
                    id, name, role, organization, influence, interest, engagement_strategy
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, name, role, org, inf, int_lvl, strat),
            )

        comm_plan = [
            ("COMM-01", "STK-01", "Email / Secure Portal", "Weekly & Event-Driven", "CDMO Client PM", "Provide 24h QTA release flash reports and testing milestones"),
            ("COMM-02", "STK-02", "Executive Dashboard", "Bi-Weekly", "Principal Specialist PM", "Review lab capacity, OTD/RFT KPIs, and open Phase II CAPAs"),
            ("COMM-03", "STK-03", "Standup & Automated Ping", "Daily", "Local MSO", "Coordinate upstream harvest release timing with QC testing skids"),
            ("COMM-04", "STK-04", "Shop-Floor Meeting", "Daily (Shift Change)", "QC PM Lead", "Level skid work orders and resolve active instrument blockers"),
        ]
        for cid, sid, ch, freq, own, obj in comm_plan:
            conn.execute(
                """
                INSERT OR REPLACE INTO communication_plan (
                    id, stakeholder_id, channel, frequency, owner, objective
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cid, sid, ch, freq, own, obj),
            )

        # 10. Insert Team Charter & Operational KPIs
        kpis = [
            ("KPI-01", "On-Time Delivery (OTD %)", 95.0, 93.3, "%", "Trailing 30 Days", "AMBER", "(Lots Released On-Time / Total Lots Released) * 100%"),
            ("KPI-02", "Right-First-Time (RFT %)", 98.0, 96.7, "%", "Trailing 30 Days", "AMBER", "(Lots Tested Without Lab Deviation / Total Lots Tested) * 100%"),
            ("KPI-03", "Mean QC Cycle Time", 10.0, 8.4, "Days", "Trailing 30 Days", "GREEN", "Sum(Turnaround Days) / Count(Released Lots)"),
            ("KPI-04", "Testing Backlog", 5.0, 4.0, "Lots", "Current Week", "GREEN", "Count(Lots in Queue Awaiting Skid Allocation)"),
            ("KPI-05", "Peak Skid Utilization", 85.0, 120.4, "%", "Current Week (W42)", "RED", "Max(Weekly Scheduled Skid Hours / Weekly Staffed Capacity) * 100%"),
        ]
        for kid, kname, tgt, act, unit, per, rag, formula in kpis:
            conn.execute(
                """
                INSERT OR REPLACE INTO team_charter_kpis (
                    id, kpi_name, target, actual, unit, period, rag_status, calculation_formula
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (kid, kname, tgt, act, unit, per, rag, formula),
            )

        # 11. Insert Sample Arrivals Forecast
        arrivals = [
            ("ARR-01", "LOT-SPES-2026-14", "BI-CAP-101", "SEC_HPLC", 1.0, 0.8, 1.2, 2.4, 0.80, "ON_TRACK", "Lot LOT-SPES-2026-14 scheduled Day 1.0; 80% conformal arrival window: Day 0.8 - 2.4 (median Day 1.2)."),
            ("ARR-02", "LOT-OVX-2026-02", "PROJ-001", "icIEF_CE", 1.0, 1.0, 1.5, 3.1, 0.80, "POTENTIAL_SLIP", "Lot LOT-OVX-2026-02 scheduled Day 1.0; 80% conformal arrival window: Day 1.0 - 3.1 (median Day 1.5)."),
            ("ARR-03", "LOT-NS-2026-004", "PROJ-002", "SEC_HPLC", 2.0, 2.1, 2.8, 4.9, 0.80, "ALERT_DELAY", "Lot LOT-NS-2026-004 scheduled Day 2.0; 80% conformal arrival window: Day 2.1 - 4.9 (median Day 2.8)."),
        ]
        for aid, lid, pid, meth, sched, q10, q50, q90, conf, st, draft in arrivals:
            conn.execute(
                """
                INSERT OR REPLACE INTO sample_arrivals (
                    id, lot_id, project_id, method, scheduled_day,
                    predicted_q10_day, predicted_q50_day, predicted_q90_day,
                    confidence_level, alert_status, heads_up_draft
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (aid, lid, pid, meth, sched, q10, q50, q90, conf, st, draft),
            )

    if audit_trail is not None:
        audit_trail.log_action(
            actor_name="System PM Seeder",
            actor_role="Principal Specialist Automation",
            action="SEED_PM_DATABASE",
            entity_type="portfolio",
            entity_id="ALL_10_PROJECTS",
            justification="Seeded relational database from default synthetic Example Biologics Site portfolio and operational registers.",
            payload={"projects_seeded": len(projects), "timestamp": now_utc},
        )

    return len(projects)


def reset_to_seed_data(
    conn: sqlite3.Connection,
    audit_trail: Optional[AuditTrail] = None,
    actor_name: str = "QC PM Lead",
    actor_role: str = "Principal Specialist",
    justification: str = "Reset portfolio data to default synthetic baseline",
) -> int:
    """
    Clears all PM tables and re-seeds from the authoritative synthetic Example Biologics Site portfolio.
    Logs the reset action to the append-only audit trail.
    """
    tables = [
        "portfolio_projects",
        "milestones",
        "decision_gates",
        "wbs_tasks",
        "action_items",
        "blocker_items",
        "raid_items",
        "raci_entries",
        "stakeholders",
        "communication_plan",
        "team_charter_kpis",
        "tech_transfer_sections",
        "sample_arrivals",
    ]
    with conn:
        for tbl in tables:
            conn.execute(f"DELETE FROM {tbl}")

    count = seed_pm_database(conn=conn, force=True)

    if audit_trail is not None:
        audit_trail.log_action(
            actor_name=actor_name,
            actor_role=actor_role,
            action="RESET_TO_SEED_DATA",
            entity_type="portfolio",
            entity_id="FULL_PORTFOLIO_RESET",
            justification=justification,
            payload={"tables_cleared": tables, "projects_reseeded": count},
        )

    return count


# -----------------------------------------------------------------------------
# CPM & Reduced DES Recalculation Engine
# -----------------------------------------------------------------------------
def recalculate_project_risk_and_cpm(
    conn: sqlite3.Connection,
    project_id: str,
    n_simulations: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Recalculates Critical Path Method (CPM) float values and runs a reduced-replication
    DES risk simulation to update P(on time) for a specific project.
    Updates portfolio_projects and wbs_tasks tables in place.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, name, skid, duration_hours, start_day, end_day, float_days, flexibility_class, predecessors
        FROM wbs_tasks WHERE project_id = ?
        ORDER BY start_day ASC
        """,
        (project_id,),
    )
    task_rows = cursor.fetchall()
    if not task_rows:
        return {"project_id": project_id, "updated": False, "reason": "No WBS tasks found"}

    # Build CPM tasks
    tasks = []
    instruments = {
        "UPLC_SEC_CEX": InstrumentDefinition("UPLC_SEC_CEX", "Waters UPLC", 1),
        "CE_icIEF": InstrumentDefinition("CE_icIEF", "ProteinSimple Maurice", 1),
        "Cell_Bioassay": InstrumentDefinition("Cell_Bioassay", "96-Well Potency", 1),
    }

    for r in task_rows:
        tid, name, skid, dur, s_day, e_day, f_days, flex_cls, preds = r
        fc = FlexibilityClass.FIXED_COMMERCIAL if flex_cls == "FIXED_COMMERCIAL" else FlexibilityClass.FLEXIBLE_WINDOW
        pred_list = [p.strip() for p in preds.split(",") if p.strip()] if preds else []
        t = Task.create(
            task_id=tid,
            project_id=project_id,
            program_name=name,
            test_panel=name,
            instrument_id=skid,
            sample_arrival_day=s_day,
            due_day=e_day + max(f_days, 1.0),
            flexibility_class=fc,
            nominal_duration_hours=dur,
            predecessors=pred_list,
        )
        tasks.append(t)

    # Run DES simulation
    risk_results = run_monte_carlo_schedule_risk(
        tasks=tasks,
        instruments=instruments,
        n_simulations=n_simulations,
        seed=seed,
    )

    # Aggregate project metrics
    p_on_times = [res.p_on_time for res in risk_results.values()]
    floats = [res.total_float_days for res in risk_results.values()]

    min_p_on_time = round(float(min(p_on_times)), 3) if p_on_times else 0.95
    min_float = round(float(min(floats)), 1) if floats else 0.0

    # Determine automatic RAG
    if min_p_on_time >= 0.90 and min_float >= 2.0:
        auto_rag = "GREEN"
    elif min_p_on_time < 0.70 or min_float < 0.0:
        auto_rag = "RED"
    else:
        auto_rag = "AMBER"

    now_utc = datetime.now(timezone.utc).isoformat()
    with conn:
        # Check if project has manual override
        cursor.execute("SELECT rag_override FROM portfolio_projects WHERE project_id = ?", (project_id,))
        row = cursor.fetchone()
        override = row[0] if row else ""
        final_rag = override if override else auto_rag

        conn.execute(
            """
            UPDATE portfolio_projects
            SET p_on_time = ?, float_days = ?, rag_status = ?, updated_at = ?
            WHERE project_id = ?
            """,
            (min_p_on_time, min_float, final_rag, now_utc, project_id),
        )

        for tid, res in risk_results.items():
            is_crit = 1 if (res.total_float_days <= 0.0 or res.criticality_index >= 0.8) else 0
            conn.execute(
                """
                UPDATE wbs_tasks
                SET float_days = ?, is_critical = ?
                WHERE id = ?
                """,
                (res.total_float_days, is_crit, tid),
            )

    return {
        "project_id": project_id,
        "updated": True,
        "p_on_time": min_p_on_time,
        "float_days": min_float,
        "rag_status": final_rag,
        "task_count": len(tasks),
    }
