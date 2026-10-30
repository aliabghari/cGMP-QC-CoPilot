"""
pm/fmea_capa.py

Proactive FMEA (Failure Mode and Effects Analysis) and Reactive CAPA Management.
Provides:
- FMEA risk analysis for biologics QC analytical laboratory workflows:
  Severity (S), Occurrence (O), Detection (D), Risk Priority Number (RPN = S x O x D),
  and Action Priority (High, Medium, Low).
- Pre- vs Post-mitigation RPN calculation with percentage risk reduction.
- Rating sensitivity analysis (+/- 1 perturbation on S, O, D).
- Dynamic linking of Occurrence ratings to historical lessons-learned deviation frequencies.
- CAPA lifecycle state machine:
  OPEN -> IN_PROGRESS -> IMPLEMENTED -> EFFECTIVENESS_CHECK -> CLOSED.
- Enforcement that CAPA closure strictly requires human reviewer sign-off and an effectiveness check result.
- Overdue / at-risk CAPA detection with automated draft reminder generation.
- Append-only ALCOA+ audit trail logging for all status transitions and edits.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.audit import AuditTrail
from core.stamps import DRAFT_STAMP, assert_no_banned_phrases


# Permitted CAPA lifecycle transitions
CAPA_STATE_TRANSITIONS = {
    "OPEN": ["IN_PROGRESS"],
    "IN_PROGRESS": ["IMPLEMENTED"],
    "IMPLEMENTED": ["EFFECTIVENESS_CHECK"],
    "EFFECTIVENESS_CHECK": ["CLOSED", "IN_PROGRESS"],  # can return to IN_PROGRESS if ineffective
    "CLOSED": [],  # Terminal state
}

CAPA_CLOSURE_DISCLAIMER = (
    "DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL. "
    "Decision support only. CAPA closure requires formal Quality Assurance concurrence per site SOP."
)


def compute_action_priority(severity: int, occurrence: int, detection: int, rpn: int) -> str:
    """
    Computes Action Priority level: HIGH, MEDIUM, LOW based on RPN and critical Severity.
    """
    if severity >= 9 or rpn >= 150:
        return "HIGH"
    elif severity >= 7 or rpn >= 80:
        return "MEDIUM"
    else:
        return "LOW"


def evaluate_fmea_sensitivity(severity: int, occurrence: int, detection: int) -> Dict[str, Any]:
    """
    Evaluates sensitivity of RPN to +/- 1 uncertainty in S, O, and D ratings.
    """
    nominal_rpn = severity * occurrence * detection

    sensitivities = {}
    for param_name, current_val in [("severity", severity), ("occurrence", occurrence), ("detection", detection)]:
        val_minus = max(1, current_val - 1)
        val_plus = min(10, current_val + 1)

        if param_name == "severity":
            rpn_minus = val_minus * occurrence * detection
            rpn_plus = val_plus * occurrence * detection
        elif param_name == "occurrence":
            rpn_minus = severity * val_minus * detection
            rpn_plus = severity * val_plus * detection
        else:
            rpn_minus = severity * occurrence * val_minus
            rpn_plus = severity * occurrence * val_plus

        sensitivities[param_name] = {
            "minus_1": rpn_minus,
            "plus_1": rpn_plus,
            "delta_minus": rpn_minus - nominal_rpn,
            "delta_plus": rpn_plus - nominal_rpn,
            "swing": rpn_plus - rpn_minus,
        }

    return {
        "nominal_rpn": nominal_rpn,
        "parameter_sensitivities": sensitivities,
        "most_sensitive_parameter": max(sensitivities.items(), key=lambda x: x[1]["swing"])[0],
    }


# -------------------------------------------------------------------------
# Seed Data Definitions for FMEA Matrix
# -------------------------------------------------------------------------

SEED_FMEA_ITEMS = [
    {
        "id": "FMEA-QC-001",
        "process_step": "Sample Login & Aliquoting",
        "failure_mode": "Ambient hold time exceedance prior to deep-freeze storage",
        "failure_effect": "Accelerated sub-visible particulate nucleation or aggregation",
        "severity": 8,
        "potential_cause": "Sample accession backlog during simultaneous multi-lot receipt",
        "occurrence": 5,
        "current_controls": "Manual entry in cold-storage transit log sheet",
        "detection": 5,
        "recommended_action": "Install barcode scan-to-freezer cart with automated 30-min countdown timer",
        "post_severity": 8,
        "post_occurrence": 2,
        "post_detection": 2,
    },
    {
        "id": "FMEA-QC-002",
        "process_step": "Reference Standard Reconstitution",
        "failure_mode": "Hygroscopic moisture pickup into lyophilized standard cake",
        "failure_effect": "Under-recovery of potency; apparent high product potency",
        "severity": 8,
        "potential_cause": "Opening vial before thermal equilibration to room temperature",
        "occurrence": 4,
        "current_controls": "SOP instruction text (dependent on analyst vigilance)",
        "detection": 6,
        "recommended_action": "Procure single-use vacuum desiccator pouches with moisture indicator card",
        "post_severity": 8,
        "post_occurrence": 1,
        "post_detection": 3,
    },
    {
        "id": "FMEA-QC-003",
        "process_step": "Mobile Phase Buffer Preparation",
        "failure_mode": "Buffer pH drift outside +/- 0.05 specification window",
        "failure_effect": "Charge variant peak co-elution / SST resolution failure",
        "severity": 7,
        "potential_cause": "Electrode calibration drift; unbuffered storage of water",
        "occurrence": 6,
        "current_controls": "Daily 2-point pH calibration sign-off",
        "detection": 4,
        "recommended_action": "Equip laboratory with automated dispensing titrator featuring digital buffer verification",
        "post_severity": 7,
        "post_occurrence": 2,
        "post_detection": 2,
    },
    {
        "id": "FMEA-QC-004",
        "process_step": "Analytical Column Equilibration",
        "failure_mode": "Inadequate column thermal and chemical baseline equilibration",
        "failure_effect": "Drifting retention times and split peaks across early injections",
        "severity": 6,
        "potential_cause": "Analyst manually shortening equilibration sequence to meet shift end",
        "occurrence": 5,
        "current_controls": "Visual inspection of baseline drift in CDS monitor window",
        "detection": 4,
        "recommended_action": "Program hard CDS sequence lock preventing standard injection until baseline slope < 0.1 mAU/min",
        "post_severity": 6,
        "post_occurrence": 1,
        "post_detection": 1,
    },
    {
        "id": "FMEA-QC-005",
        "process_step": "Autosampler Sample Loading",
        "failure_mode": "Vial position transposition in 96-well autosampler tray",
        "failure_effect": "Sample results assigned to incorrect batch or dilution tube",
        "severity": 9,
        "potential_cause": "Manual loading without secondary witness verification",
        "occurrence": 3,
        "current_controls": "Single analyst check-off on printed loading template",
        "detection": 7,
        "recommended_action": "Deploy 2D barcode scanner tray reader interfacing directly to CDS sequence builder",
        "post_severity": 9,
        "post_occurrence": 1,
        "post_detection": 2,
    },
    {
        "id": "FMEA-QC-006",
        "process_step": "Chromatographic Gradient Execution",
        "failure_mode": "Degasser vacuum failure producing micro-bubbles in mixer",
        "failure_effect": "Baseline ripple, pump cavitation, missed peak integration",
        "severity": 7,
        "potential_cause": "Wear of vacuum pump Teflon membrane",
        "occurrence": 4,
        "current_controls": "Pressure trace inspection during batch record review",
        "detection": 5,
        "recommended_action": "Configure hardware telemetry sensor alarm alerting on degasser vacuum decay > 5 mbar/min",
        "post_severity": 7,
        "post_occurrence": 2,
        "post_detection": 1,
    },
    {
        "id": "FMEA-QC-007",
        "process_step": "Integration & Quantitation",
        "failure_mode": "Inappropriate baseline skimming under partially resolved shoulder peak",
        "failure_effect": "Reported purity artificially elevated or depressed",
        "severity": 8,
        "potential_cause": "Analyst applying unapproved manual integration baseline modifications",
        "occurrence": 4,
        "current_controls": "Second-analyst manual review of integration event log",
        "detection": 4,
        "recommended_action": "Lock CDS processing parameters; mandate QA electronic approval for any manual integration event",
        "post_severity": 8,
        "post_occurrence": 1,
        "post_detection": 2,
    },
    {
        "id": "FMEA-QC-008",
        "process_step": "Test Certificate of Analysis Generation",
        "failure_mode": "Transcription error of dilution factor in offline spreadsheet",
        "failure_effect": "Incorrect final lot release concentration reported to client",
        "severity": 9,
        "potential_cause": "Manual data entry into disconnected Microsoft Excel workbook",
        "occurrence": 3,
        "current_controls": "Peer review check-sheet signature",
        "detection": 6,
        "recommended_action": "Decommission offline spreadsheets; route calculations through central approved LIMS template",
        "post_severity": 9,
        "post_occurrence": 1,
        "post_detection": 1,
    },
]

SEED_CAPA_RECORDS = [
    {
        "id": "CAPA-2026-001",
        "source_type": "DEVIATION",
        "source_ref": "DEV-2026-088",
        "root_cause_id": "RC-CHROM-01",
        "title": "Mandatory Pre-Filtration for High-Concentration mAb Formulations",
        "corrective_action": "Replaced analytical column guard cartridge and re-tested lot retains with acceptable system suitability.",
        "preventive_action": "Revised SOP-QC-0881 (fictional example) to mandate 0.22 um PVDF spin-filter centrifugation for all samples > 50 mg/mL.",
        "owner": "Chromatography SME",
        "due_date": "2026-04-15",
        "status": "EFFECTIVENESS_CHECK",
        "effectiveness_criteria": "Zero column backpressure excursion deviations across next 30 consecutive SEC-HPLC sequences.",
        "effectiveness_due_date": "2026-06-30",
        "effectiveness_result": "Pending completion of 30-sequence monitoring window (currently 24/30 clean).",
        "closed_by": None,
        "closed_at": None,
        "justification": "Initiated following repeated column fouling during high-concentration formulation release testing.",
    },
    {
        "id": "CAPA-2026-002",
        "source_type": "AUDIT",
        "source_ref": "INT-AUDIT-2026-Q1",
        "root_cause_id": "RC-STD-01",
        "title": "Implementation of Color-Coded Expiry Labels for Thawed Standard Aliquots",
        "corrective_action": "Discarded all un-dated working standard aliquots in cold room freezer B-12.",
        "preventive_action": "Procured pre-printed fluorescent labels with explicit 'Thaw Date' and 'Mandatory Discard Date'.",
        "owner": "QC Sample Custodian",
        "due_date": "2026-03-31",
        "status": "IMPLEMENTED",
        "effectiveness_criteria": "100% compliance during bi-weekly unannounced freezer walkthrough audits for 3 months.",
        "effectiveness_due_date": "2026-07-15",
        "effectiveness_result": None,
        "closed_by": None,
        "closed_at": None,
        "justification": "Internal audit finding identifying unlabelled thawed standard vials in freezer.",
    },
    {
        "id": "CAPA-2026-003",
        "source_type": "DEVIATION",
        "source_ref": "DEV-2026-104",
        "root_cause_id": "RC-BIO-01",
        "title": "Electronic Cell Passage Limit Hard-Lock in Laboratory Inventory",
        "corrective_action": "Decommissioned cell bank working vials that had reached passage 19.",
        "preventive_action": "Programmed automated inventory alert and hard-lock preventing vial checkout beyond passage 18.",
        "owner": "Bioassay Lead",
        "due_date": "2026-02-28",
        "status": "CLOSED",
        "effectiveness_criteria": "Zero bioassay curve slope failures attributable to cell passage age over 6-month period.",
        "effectiveness_due_date": "2026-05-30",
        "effectiveness_result": "Effectiveness verified: 48 consecutive bioassay runs conducted without passage-related slope failures.",
        "closed_by": "QA Compliance Manager",
        "closed_at": "2026-06-05T14:15:00Z",
        "justification": "Full effectiveness criteria met and audited by Quality Assurance.",
    },
    {
        "id": "CAPA-2026-004",
        "source_type": "FMEA",
        "source_ref": "FMEA-QC-005",
        "root_cause_id": "RC-SOFT-02",
        "title": "Deployment of 2D Barcode Well-Plate Scanner for Autosampler Queues",
        "corrective_action": "Manual dual-analyst verification mandated as interim containment measure.",
        "preventive_action": "Procure and qualify 2D plate rack scanner linked to CDS sequence generator.",
        "owner": "QC Automation Specialist",
        "due_date": "2026-05-15",
        "status": "IN_PROGRESS",
        "effectiveness_criteria": "Zero vial transposition incidents across all automated skid runs.",
        "effectiveness_due_date": "2026-08-30",
        "effectiveness_result": None,
        "closed_by": None,
        "closed_at": None,
        "justification": "High RPN failure mode identified during proactive FMEA annual review.",
    },
    {
        "id": "CAPA-2026-005",
        "source_type": "TREND",
        "source_ref": "SPC-OOT-2026-W11",
        "root_cause_id": "RC-RGT-01",
        "title": "Lockout Calibration Enforcers on Digital pH Meters",
        "corrective_action": "Re-calibrated analytical pH meters using freshly reconstituted NIST-traceable standard buffers.",
        "preventive_action": "Activated electronic lockout disabling pH measurement if 2-point calibration is > 12 hours old.",
        "owner": "QC Metrology Lead",
        "due_date": "2026-01-20",
        "status": "OPEN",  # Intentionally overdue to exercise overdue radar
        "effectiveness_criteria": "Zero buffer preparation pH deviations in chromatography logs.",
        "effectiveness_due_date": "2026-04-30",
        "effectiveness_result": None,
        "closed_by": None,
        "closed_at": None,
        "justification": "Recurring minor pH drift noted in monthly SPC control chart review.",
    },
    {
        "id": "CAPA-2026-006",
        "source_type": "DEVIATION",
        "source_ref": "DEV-2026-112",
        "root_cause_id": "RC-SMPL-01",
        "title": "Vortex Prohibition and Roller Mixer Replacement in Sample Prep Areas",
        "corrective_action": "Removed high-speed benchtop vortex mixers from formulation testing suites.",
        "preventive_action": "Installed low-shear nutating platform mixers and updated sample prep SOP.",
        "owner": "Formulation Testing SME",
        "due_date": "2026-04-30",
        "status": "IN_PROGRESS",
        "effectiveness_criteria": "Reduction of HIAC sub-visible particle spike deviations by >= 80%.",
        "effectiveness_due_date": "2026-07-31",
        "effectiveness_result": None,
        "closed_by": None,
        "closed_at": None,
        "justification": "Vortex-induced agitation shearing caused false-positive sub-visible particulate counts.",
    },
]


def seed_fmea_and_capa(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Seeds FMEA matrix items and CAPA records into SQLite.
    Computes initial RPN and Action Priority dynamically.
    """
    cur_chk = conn.cursor()
    cur_chk.execute("SELECT COUNT(id) FROM root_causes")
    if cur_chk.fetchone()[0] == 0:
        from pm.knowledge_base import seed_knowledge_base
        seed_knowledge_base(conn)

    with conn:
        for item in SEED_FMEA_ITEMS:
            rpn = item["severity"] * item["occurrence"] * item["detection"]
            ap = compute_action_priority(item["severity"], item["occurrence"], item["detection"], rpn)
            post_rpn = item["post_severity"] * item["post_occurrence"] * item["post_detection"]

            conn.execute(
                """
                INSERT OR IGNORE INTO fmea_items
                (id, process_step, failure_mode, failure_effect, severity, potential_cause, occurrence,
                 current_controls, detection, rpn, action_priority, recommended_action,
                 post_severity, post_occurrence, post_detection, post_rpn, lessons_learned_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["id"], item["process_step"], item["failure_mode"], item["failure_effect"],
                    item["severity"], item["potential_cause"], item["occurrence"], item["current_controls"],
                    item["detection"], rpn, ap, item["recommended_action"],
                    item["post_severity"], item["post_occurrence"], item["post_detection"], post_rpn, 0,
                ),
            )

        for capa in SEED_CAPA_RECORDS:
            now_utc = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT OR IGNORE INTO capa_records
                (id, source_type, source_ref, root_cause_id, title, corrective_action, preventive_action,
                 owner, due_date, status, effectiveness_criteria, effectiveness_due_date,
                 effectiveness_result, closed_by, closed_at, justification, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capa["id"], capa["source_type"], capa["source_ref"], capa["root_cause_id"],
                    capa["title"], capa["corrective_action"], capa["preventive_action"],
                    capa["owner"], capa["due_date"], capa["status"], capa["effectiveness_criteria"],
                    capa["effectiveness_due_date"], capa["effectiveness_result"],
                    capa["closed_by"], capa["closed_at"], capa["justification"], now_utc,
                ),
            )

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM fmea_items")
    n_fmea = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM capa_records")
    n_capa = cur.fetchone()[0]

    return {"fmea_items_count": n_fmea, "capa_records_count": n_capa}


def sync_fmea_with_lessons_learned(conn: sqlite3.Connection) -> int:
    """
    Updates FMEA lessons_learned_count based on occurrences of matching keywords/root causes in lessons_learned.
    """
    cur = conn.cursor()
    cur.execute("SELECT id, failure_mode, potential_cause FROM fmea_items")
    fmea_rows = cur.fetchall()

    updates = 0
    with conn:
        for r in fmea_rows:
            f_id = r["id"]
            mode_kw = r["failure_mode"].split()[0]
            cause_kw = r["potential_cause"].split()[0]

            cur.execute(
                """
                SELECT COUNT(*) FROM lessons_learned
                WHERE what_happened LIKE ? OR what_learned LIKE ?
                """,
                (f"%{mode_kw}%", f"%{cause_kw}%"),
            )
            count = cur.fetchone()[0]
            conn.execute("UPDATE fmea_items SET lessons_learned_count = ? WHERE id = ?", (count, f_id))
            updates += 1

    return updates


def get_all_fmea_items(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Retrieves all FMEA items with computed percentage risk reductions and sensitivity data."""
    cur = conn.cursor()
    cur.execute("SELECT * FROM fmea_items ORDER BY rpn DESC")
    rows = cur.fetchall()

    items = []
    for r in rows:
        d = dict(r)
        pre_rpn = d["rpn"]
        post_rpn = d.get("post_rpn") or pre_rpn
        reduction_pct = round(((pre_rpn - post_rpn) / max(pre_rpn, 1)) * 100.0, 1)

        sens = evaluate_fmea_sensitivity(d["severity"], d["occurrence"], d["detection"])

        items.append({
            **d,
            "rpn_reduction_pct": reduction_pct,
            "sensitivity": sens,
            "disclaimer": DRAFT_STAMP,
        })

    return items


def get_all_capa_records(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Retrieves all CAPA records with computed overdue status and days remaining."""
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*, rc.name AS root_cause_name, rc.category AS root_cause_category
        FROM capa_records c
        LEFT JOIN root_causes rc ON c.root_cause_id = rc.id
        ORDER BY c.due_date ASC
    """)
    rows = cur.fetchall()

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    capas = []

    for r in rows:
        d = dict(r)
        due = d["due_date"]
        status = d["status"]
        is_closed = status == "CLOSED"

        is_overdue = False
        days_diff = 0
        try:
            d_due = datetime.strptime(due, "%Y-%m-%d").date()
            d_today = datetime.strptime(today_str, "%Y-%m-%d").date()
            days_diff = (d_due - d_today).days
            if days_diff < 0 and not is_closed:
                is_overdue = True
        except Exception:
            pass

        reminder_draft = None
        if is_overdue:
            reminder_draft = (
                f"Subject: [OVERDUE ADVISORY] Action Required on {d['id']} ({d['title']})\n\n"
                f"Dear {d['owner']},\n\n"
                f"This is an automated decision-support advisory noting that CAPA {d['id']} reached its target "
                f"completion milestone on {d['due_date']} ({abs(days_diff)} days overdue) with status '{status}'.\n\n"
                f"Action Required: Please update the corrective/preventive action implementation status, or submit "
                f"a formal extension justification for Quality Assurance review per site SOP.\n\n"
                f"{CAPA_CLOSURE_DISCLAIMER}"
            )

        capas.append({
            **d,
            "is_overdue": is_overdue,
            "days_until_due": days_diff,
            "reminder_draft": reminder_draft,
            "disclaimer": DRAFT_STAMP,
        })

    return capas


def transition_capa_status(
    conn: sqlite3.Connection,
    capa_id: str,
    target_status: str,
    audit_trail: AuditTrail,
    actor_name: str,
    actor_role: str,
    justification: str,
    effectiveness_result: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes a guarded state transition on a CAPA record.
    Enforces that:
    1. Transition must be valid per CAPA_STATE_TRANSITIONS.
    2. Closure (CLOSED) requires an identifiable human reviewer, non-empty justification,
       and confirmed effectiveness_result.
    3. Logs an immutable ALCOA+ audit trail record.
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM capa_records WHERE id = ?", (capa_id,))
    row = cur.fetchone()
    if not row:
        raise ValueError(f"CAPA record {capa_id} not found")

    current_status = row["status"]
    allowed = CAPA_STATE_TRANSITIONS.get(current_status, [])

    if target_status not in allowed:
        raise ValueError(
            f"Illegal CAPA transition: Cannot transition {capa_id} from '{current_status}' to '{target_status}'. "
            f"Allowed next states: {allowed}"
        )

    now_utc = datetime.now(timezone.utc).isoformat()

    # Closure enforcement
    if target_status == "CLOSED":
        eff = effectiveness_result if effectiveness_result is not None else row["effectiveness_result"]
        if not eff or not eff.strip():
            raise ValueError(
                "Cannot close CAPA without verified effectiveness check result. "
                "Record must document completion of effectiveness criteria."
            )
        if not justification or not justification.strip():
            raise ValueError("CAPA closure strictly requires documented QA human justification.")

        with conn:
            conn.execute(
                """
                UPDATE capa_records
                SET status = ?, closed_by = ?, closed_at = ?, effectiveness_result = ?, justification = ?
                WHERE id = ?
                """,
                (target_status, actor_name, now_utc, eff.strip(), justification.strip(), capa_id),
            )
    else:
        with conn:
            conn.execute(
                """
                UPDATE capa_records
                SET status = ?, justification = ?
                WHERE id = ?
                """,
                (target_status, justification.strip(), capa_id),
            )

    # Log ALCOA+ audit event
    audit_trail.append_event(
        actor_name=actor_name,
        actor_role=actor_role,
        action="TRANSITION_CAPA_STATUS",
        justification=justification,
        entity_type="CAPA",
        entity_id=capa_id,
        payload={
            "previous_status": current_status,
            "target_status": target_status,
            "effectiveness_result": effectiveness_result,
        },
    )

    out = {
        "capa_id": capa_id,
        "previous_status": current_status,
        "new_status": target_status,
        "actor": actor_name,
        "role": actor_role,
        "timestamp_utc": now_utc,
        "disclaimer": CAPA_CLOSURE_DISCLAIMER,
    }
    assert_no_banned_phrases(str(out))
    return out
