"""
cGMP-QC-CoPilot: Sample-Arrival Uncertainty Forecaster & Heads-Up Alert Engine.

Translates rolling-origin Conformalized Quantile Regression (CQR) arrival predictions
into calibrated uncertainty intervals (q10, q50, q90) and supervisory heads-up notifications.

Feature Role:
- Calibrated arrival window around a baseline-level estimate. Since the point forecaster
  ties baseline methods on MAE, the primary delivered value is providing valid conformal
  prediction intervals (85.7% overall held-out coverage; 95% Clopper-Pearson CI [77.5%, 91.8%])
  to bound upstream bioreactor harvest variability.

Key Guardrails:
- Leads with the uncertainty interval (e.g., 'expected between Day 1.8 and Day 4.2')
- Numbers strictly mirror the underlying statistical model (no language-model computation)
- Dispatches are locked behind human supervisory review sign-off
- Stamped 'DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL'
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.audit import AuditTrail
from core.stamps import DRAFT_STAMP, assert_no_banned_phrases


@dataclass
class SampleArrivalAlert:
    alert_id: str
    lot_id: str
    project_id: str
    program_name: str
    method_name: str
    scheduled_day: float
    q10_day: float
    q50_day: float
    q90_day: float
    confidence_level: float
    alert_status: str  # ON_TRACK, POTENTIAL_SLIP, ALERT_DELAY
    draft_message: str
    is_signed_off: bool = False
    signed_off_by: Optional[str] = None
    signed_off_at: Optional[str] = None
    disclaimer: str = DRAFT_STAMP


def generate_sample_arrival_alert(
    lot_id: str,
    project_id: str,
    program_name: str,
    method_name: str,
    scheduled_day: float,
    predicted_slip_q10: float,
    predicted_slip_q50: float,
    predicted_slip_q90: float,
    target_coverage: float = 0.80,
) -> SampleArrivalAlert:
    """
    Constructs an uncertainty-aware sample arrival prediction and draft heads-up alert.
    Leads with the uncertainty interval in accordance with B9 requirements.
    """
    arrival_q10 = round(scheduled_day + max(0.0, predicted_slip_q10), 1)
    arrival_q50 = round(scheduled_day + max(0.0, predicted_slip_q50), 1)
    arrival_q90 = round(scheduled_day + max(0.0, predicted_slip_q90), 1)

    # Classify slip risk
    if arrival_q90 <= scheduled_day + 0.5:
        status = "ON_TRACK"
        lead_sentence = f"Lot {lot_id} is on schedule: arrival expected between Day {arrival_q10:.1f} and Day {arrival_q90:.1f} (median Day {arrival_q50:.1f})."
    elif arrival_q50 <= scheduled_day + 1.0:
        status = "POTENTIAL_SLIP"
        lead_sentence = f"Lot {lot_id} has potential upstream slip: arrival expected between Day {arrival_q10:.1f} and Day {arrival_q90:.1f} (median Day {arrival_q50:.1f}, scheduled Day {scheduled_day:.1f})."
    else:
        status = "ALERT_DELAY"
        lead_sentence = f"CRITICAL SLIP ALERT: Lot {lot_id} arrival expected between Day {arrival_q10:.1f} and Day {arrival_q90:.1f} (median Day {arrival_q50:.1f}, scheduled Day {scheduled_day:.1f})."

    draft = f"""MEMORANDUM: QC Laboratory Sample Arrival Advisory
{DRAFT_STAMP}

{lead_sentence}

- Program: {program_name} ({project_id})
- Testing Method: {method_name}
- Scheduled Baseline Arrival: Day {scheduled_day:.1f}
- Conformal Prediction Window ({int(target_coverage * 100)}% Coverage): Day {arrival_q10:.1f} to Day {arrival_q90:.1f}
- Point Median Forecast: Day {arrival_q50:.1f}

Operational Action Plan:
1. Maintain allocated testing skid buffer on standby across the Day {arrival_q10:.1f}–{arrival_q90:.1f} window.
2. If arrival shifts beyond Day {arrival_q50:.1f}, evaluate shifting flexible stability pulls to prevent skid idle time.
3. Reviewer sign-off required prior to releasing formal dispatch to Shift Supervisors.
"""

    assert_no_banned_phrases(draft)

    alert_id = f"ALERT-{datetime.now(timezone.utc).strftime('%m%d%H%M%S')}"
    return SampleArrivalAlert(
        alert_id=alert_id,
        lot_id=lot_id,
        project_id=project_id,
        program_name=program_name,
        method_name=method_name,
        scheduled_day=scheduled_day,
        q10_day=arrival_q10,
        q50_day=arrival_q50,
        q90_day=arrival_q90,
        confidence_level=target_coverage,
        alert_status=status,
        draft_message=draft,
    )


def sign_off_sample_alert(
    alert: SampleArrivalAlert,
    reviewer_name: str,
    reviewer_role: str,
    justification: str,
    audit_trail: Optional[AuditTrail] = None,
) -> SampleArrivalAlert:
    """
    Applies the human sign-off gate to release a sample-arrival notification.
    """
    if not reviewer_name or len(reviewer_name.strip()) < 3:
        raise ValueError("Reviewer name must be at least 3 characters.")
    if not justification or len(justification.strip()) < 10:
        raise ValueError("A substantive justification (>= 10 characters) is required for sign-off.")

    now_utc = datetime.now(timezone.utc).isoformat()
    alert.is_signed_off = True
    alert.signed_off_by = f"{reviewer_name} ({reviewer_role})"
    alert.signed_off_at = now_utc

    if audit_trail is not None:
        audit_trail.log_action(
            actor_name=reviewer_name,
            actor_role=reviewer_role,
            action="SIGN_OFF_SAMPLE_ARRIVAL_ALERT",
            entity_type="sample_alert",
            entity_id=alert.alert_id,
            justification=justification,
            payload={
                "alert_id": alert.alert_id,
                "lot_id": alert.lot_id,
                "arrival_q10": alert.q10_day,
                "arrival_q50": alert.q50_day,
                "arrival_q90": alert.q90_day,
                "timestamp": now_utc,
            },
        )

    return alert


def get_stored_sample_arrivals(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Retrieves stored sample arrival predictions from the database.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            id, lot_id, project_id, method, scheduled_day,
            predicted_q10_day, predicted_q50_day, predicted_q90_day,
            confidence_level, alert_status, heads_up_draft
        FROM sample_arrivals
        ORDER BY scheduled_day ASC
        """
    )
    rows = cursor.fetchall()
    return [
        {
            "id": r[0],
            "lot_id": r[1],
            "project_id": r[2],
            "method": r[3],
            "scheduled_day": r[4],
            "q10_day": r[5],
            "q50_day": r[6],
            "q90_day": r[7],
            "confidence_level": r[8],
            "alert_status": r[9],
            "heads_up_draft": r[10],
        }
        for r in rows
    ]
