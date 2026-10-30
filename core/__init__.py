"""
cGMP-QC-CoPilot: Core Shared Package.

Provides shared foundations across all specialist agents:
- audit: Append-only hash-chained audit trail (ALCOA+ and G6)
- db: SQLite migration and relational persistence layer
- stamps: Non-GMP draft stamping and banned-phrase claims linter (G2)
- synthetic: Synthetic data governance, labeling, and SHA-256 manifests (G1, G4)
- modelcard: Model card schema, validation, and benchmarking documentation (G5)
"""

from core.audit import (
    AuditRecord,
    AuditTrail,
    GENESIS_HASH,
    PART11_DEMO_DISCLAIMER,
    verify_chain,
)
from core.db import (
    get_connection,
    get_db_status,
    init_db,
)
from core.modelcard import (
    DEFAULT_DISCLAIMER,
    ModelCard,
    load_model_card,
    save_model_card,
    validate_model_card,
)
from core.stamps import (
    BANNED_PHRASES,
    DRAFT_STAMP,
    PART11_DISCLAIMER,
    SYNTHETIC_DATA_NOTICE,
    apply_draft_stamp,
    assert_no_banned_phrases,
    check_banned_phrases,
    scan_file_for_banned_phrases,
)
from core.synthetic import (
    SYNTHETIC_HEADER_TAG,
    check_synthetic_labeling,
    compute_file_sha256,
    generate_manifest,
    verify_manifest,
)

from core.session_recorder import (
    InteractionEvent,
    SessionRecorder,
    get_session_recorder,
    render_browser_screen_recorder_widget,
    render_session_recorder_modal_or_popover,
)

__all__ = [
    "AuditRecord",
    "AuditTrail",
    "GENESIS_HASH",
    "PART11_DEMO_DISCLAIMER",
    "verify_chain",
    "get_connection",
    "get_db_status",
    "init_db",
    "ModelCard",
    "save_model_card",
    "load_model_card",
    "validate_model_card",
    "DEFAULT_DISCLAIMER",
    "DRAFT_STAMP",
    "PART11_DISCLAIMER",
    "SYNTHETIC_DATA_NOTICE",
    "BANNED_PHRASES",
    "check_banned_phrases",
    "assert_no_banned_phrases",
    "apply_draft_stamp",
    "scan_file_for_banned_phrases",
    "SYNTHETIC_HEADER_TAG",
    "compute_file_sha256",
    "generate_manifest",
    "verify_manifest",
    "check_synthetic_labeling",
    "InteractionEvent",
    "SessionRecorder",
    "get_session_recorder",
    "render_browser_screen_recorder_widget",
    "render_session_recorder_modal_or_popover",
]
