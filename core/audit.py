"""
cGMP-QC-CoPilot: Append-Only Hash-Chained Audit Trail with Checkpoint and Immutability Triggers.

Implements Guardrail G6 (Human-in-the-loop) and ALCOA+ Data Integrity Principles:
- Attributable: Reviewer Name, Role, User ID
- Legible: Clear timestamps, actions, plain text justifications
- Contemporaneous: UTC timestamp at signing
- Original: First recording of the decision event
- Accurate: Cryptographic SHA-256 hash chaining of entire record payload
- Complete & Consistent: Any modification, deletion, or reordering breaks verify_chain()
- Tamper-Proof: SQLite triggers strictly prevent UPDATE and DELETE
- Checkpoint Verification: An external head checkpoint (record count + head hash) prevents tail truncation

Labeled: "Part 11-style demo only, not a compliant e-signature system."
"""

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

GENESIS_HASH = "0" * 64
PART11_DEMO_DISCLAIMER = "Part 11-style demo only, not a compliant e-signature system."


@dataclass
class AuditRecord:
    index: int
    timestamp_utc: str
    actor_name: str
    actor_role: str
    action: str
    entity_type: str
    entity_id: str
    justification: str
    payload_hash: str
    payload: Optional[Dict[str, Any]]
    previous_hash: str
    record_hash: str
    disclaimer: str = PART11_DEMO_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditCheckpoint:
    record_count: int
    head_hash: str
    timestamp_utc: str
    checkpoint_id: str = "primary_audit_checkpoint"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_payload_hash(payload: Optional[Dict[str, Any]]) -> str:
    """Computes a deterministic SHA-256 hash of a JSON-serializable payload dictionary."""
    if payload is None:
        return "none"
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_record_hash(
    index: int,
    timestamp_utc: str,
    actor_name: str,
    actor_role: str,
    action: str,
    entity_type: str,
    entity_id: str,
    justification: str,
    payload_hash: str,
    previous_hash: str,
) -> str:
    """Computes the cryptographic block hash binding the record contents to its previous hash."""
    canonical_str = (
        f"{index}|{timestamp_utc}|{actor_name}|{actor_role}|"
        f"{action}|{entity_type}|{entity_id}|{justification}|"
        f"{payload_hash}|{previous_hash}"
    )
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class AuditTrail:
    """
    Append-only, hash-chained audit trail supporting in-memory operation
    and optional SQLite backing with immutability triggers and head checkpoints.
    """

    def __init__(self, db_conn: Optional[sqlite3.Connection] = None):
        self.db_conn = db_conn
        self._records: List[AuditRecord] = []
        if self.db_conn:
            self._ensure_table()
            self._load_from_db()

    def _ensure_table(self) -> None:
        if not self.db_conn:
            return
        with self.db_conn:
            # 1. Main audit_trail table
            self.db_conn.execute(
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
                """
            )

            # 2. External head checkpoint table
            self.db_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_checkpoint (
                    checkpoint_id TEXT PRIMARY KEY,
                    record_count INTEGER NOT NULL,
                    head_hash TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL
                );
                """
            )

            # 3. SQLite triggers preventing UPDATE and DELETE on audit_trail
            self.db_conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_audit_trail_no_update
                BEFORE UPDATE ON audit_trail
                BEGIN
                    SELECT RAISE(ABORT, 'UPDATE operations are strictly prohibited on append-only audit_trail');
                END;
                """
            )
            self.db_conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_audit_trail_no_delete
                BEFORE DELETE ON audit_trail
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE operations are strictly prohibited on append-only audit_trail');
                END;
                """
            )

    def _load_from_db(self) -> None:
        if not self.db_conn:
            return
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM audit_trail ORDER BY record_index ASC")
        rows = cursor.fetchall()
        self._records = []
        for row in rows:
            payload = json.loads(row[9]) if row[9] else None
            rec = AuditRecord(
                index=row[0],
                timestamp_utc=row[1],
                actor_name=row[2],
                actor_role=row[3],
                action=row[4],
                entity_type=row[5],
                entity_id=row[6],
                justification=row[7],
                payload_hash=row[8],
                payload=payload,
                previous_hash=row[10],
                record_hash=row[11],
                disclaimer=row[12],
            )
            self._records.append(rec)

    @property
    def records(self) -> List[AuditRecord]:
        return list(self._records)

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Returns all audit records as a list of dictionaries for tabular display and export."""
        return [r.to_dict() for r in self._records]

    @property
    def latest_hash(self) -> str:
        if not self._records:
            return GENESIS_HASH
        return self._records[-1].record_hash

    def get_checkpoint(self) -> AuditCheckpoint:
        """Returns the current head checkpoint for this audit trail."""
        return AuditCheckpoint(
            record_count=len(self._records),
            head_hash=self.latest_hash,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    def append_event(
        self,
        actor_name: str,
        actor_role: str,
        action: str,
        justification: str,
        entity_type: str = "GENERAL",
        entity_id: str = "SYSTEM",
        payload: Optional[Dict[str, Any]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> AuditRecord:
        """
        Contemporaneously appends an event to the hash chain and updates the external checkpoint.
        Requires actor name, role, action, and non-empty justification.
        """
        if not actor_name or not actor_name.strip():
            raise ValueError("Audit record must have an identifiable actor_name (ALCOA+ Attributable)")
        if not actor_role or not actor_role.strip():
            raise ValueError("Audit record must specify actor_role")
        if not justification or not justification.strip():
            raise ValueError("Audit record requires a non-empty human justification")

        index = len(self._records)
        previous_hash = self.latest_hash
        ts = timestamp_utc or datetime.now(timezone.utc).isoformat()
        p_hash = compute_payload_hash(payload)

        r_hash = compute_record_hash(
            index=index,
            timestamp_utc=ts,
            actor_name=actor_name.strip(),
            actor_role=actor_role.strip(),
            action=action.strip(),
            entity_type=entity_type.strip(),
            entity_id=entity_id.strip(),
            justification=justification.strip(),
            payload_hash=p_hash,
            previous_hash=previous_hash,
        )

        record = AuditRecord(
            index=index,
            timestamp_utc=ts,
            actor_name=actor_name.strip(),
            actor_role=actor_role.strip(),
            action=action.strip(),
            entity_type=entity_type.strip(),
            entity_id=entity_id.strip(),
            justification=justification.strip(),
            payload_hash=p_hash,
            payload=payload,
            previous_hash=previous_hash,
            record_hash=r_hash,
            disclaimer=PART11_DEMO_DISCLAIMER,
        )

        self._records.append(record)

        if self.db_conn:
            with self.db_conn:
                self.db_conn.execute(
                    """
                    INSERT INTO audit_trail VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.index,
                        record.timestamp_utc,
                        record.actor_name,
                        record.actor_role,
                        record.action,
                        record.entity_type,
                        record.entity_id,
                        record.justification,
                        record.payload_hash,
                        json.dumps(record.payload) if record.payload else None,
                        record.previous_hash,
                        record.record_hash,
                        record.disclaimer,
                    ),
                )
                # Update external head checkpoint
                self.db_conn.execute(
                    """
                    INSERT OR REPLACE INTO audit_checkpoint (checkpoint_id, record_count, head_hash, timestamp_utc)
                    VALUES (?, ?, ?, ?);
                    """,
                    ("primary_audit_checkpoint", len(self._records), record.record_hash, ts),
                )

        return record

    def log_action(
        self,
        actor_name: str,
        actor_role: str,
        action: str,
        justification: str,
        entity_type: str = "GENERAL",
        entity_id: str = "SYSTEM",
        payload: Optional[Dict[str, Any]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> AuditRecord:
        """Alias for append_event to maintain backward compatibility across all modules."""
        return self.append_event(
            actor_name=actor_name,
            actor_role=actor_role,
            action=action,
            justification=justification,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=payload,
            timestamp_utc=timestamp_utc,
        )

    def verify_chain(
        self, checkpoint: Optional[Union[AuditCheckpoint, Tuple[int, str]]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies the cryptographic integrity of the entire audit chain,
        optionally checking against a known head checkpoint.
        """
        cp = checkpoint
        if cp is None and self.db_conn:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT record_count, head_hash FROM audit_checkpoint WHERE checkpoint_id='primary_audit_checkpoint'")
            row = cursor.fetchone()
            if row:
                cp = (row[0], row[1])

        return verify_chain(self._records, checkpoint=cp)


def verify_chain(
    records: List[Union[AuditRecord, Dict[str, Any]]],
    checkpoint: Optional[Union[AuditCheckpoint, Tuple[int, str], Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Standalone verification function for any sequence of AuditRecords or record dicts.
    Validates:
    1. Sequential indexing from 0
    2. Genesis block references GENESIS_HASH
    3. Each block's previous_hash matches preceding block's record_hash
    4. Payload hash matches canonical serialization of payload
    5. Block record_hash recalculates identically
    6. If checkpoint is provided, validates that record count matches and final record_hash matches head_hash
    """
    # 1. Checkpoint validation (detects tail truncation)
    if checkpoint is not None:
        if isinstance(checkpoint, AuditCheckpoint):
            exp_count, exp_head = checkpoint.record_count, checkpoint.head_hash
        elif isinstance(checkpoint, dict):
            exp_count, exp_head = checkpoint["record_count"], checkpoint["head_hash"]
        else:
            exp_count, exp_head = checkpoint[0], checkpoint[1]

        if len(records) != exp_count:
            return (
                False,
                f"Checkpoint count mismatch: expected {exp_count} records, but found {len(records)} (tail truncation detected)",
            )
        if exp_count > 0:
            last_rec = records[-1]
            actual_head = last_rec.record_hash if isinstance(last_rec, AuditRecord) else last_rec.get("record_hash")
            if actual_head != exp_head:
                return (
                    False,
                    f"Checkpoint head mismatch: expected head hash '{exp_head}', but found '{actual_head}' (tail truncation/tamper detected)",
                )

    if not records:
        return True, None

    for i, rec in enumerate(records):
        if isinstance(rec, AuditRecord):
            r = rec.to_dict()
        else:
            r = rec

        expected_index = i
        if r.get("index") != expected_index:
            return (
                False,
                f"Index mismatch at block {i}: expected {expected_index}, got {r.get('index')}",
            )

        expected_prev_hash = GENESIS_HASH if i == 0 else (
            records[i - 1].record_hash if isinstance(records[i - 1], AuditRecord)
            else records[i - 1]["record_hash"]
        )

        if r.get("previous_hash") != expected_prev_hash:
            return (
                False,
                f"Broken chain at block {i}: previous_hash '{r.get('previous_hash')}' != expected '{expected_prev_hash}'",
            )

        # Check payload hash
        payload = r.get("payload")
        recomputed_payload_hash = compute_payload_hash(payload)
        if r.get("payload_hash") != recomputed_payload_hash:
            return (
                False,
                f"Payload tamper detected at block {i}: recorded '{r.get('payload_hash')}' != recomputed '{recomputed_payload_hash}'",
            )

        # Check block hash
        recomputed_block_hash = compute_record_hash(
            index=r["index"],
            timestamp_utc=r["timestamp_utc"],
            actor_name=r["actor_name"],
            actor_role=r["actor_role"],
            action=r["action"],
            entity_type=r["entity_type"],
            entity_id=r["entity_id"],
            justification=r["justification"],
            payload_hash=r["payload_hash"],
            previous_hash=r["previous_hash"],
        )

        if r.get("record_hash") != recomputed_block_hash:
            return (
                False,
                f"Record hash tamper detected at block {i}: recorded '{r.get('record_hash')}' != recomputed '{recomputed_block_hash}'",
            )

    return True, None
