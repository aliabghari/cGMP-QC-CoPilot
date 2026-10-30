"""
cGMP-QC-CoPilot: Client Request to GMP QC Work Package Generator.

A deterministic, rule-based engine converting unstructured CDMO client requests
into formal QC testing work packages.

Key Principles:
- Rule-based method mapping (deterministic lookup; no LLM computation)
- Mandatory specification placeholders ('[TBD - Client Specification Sign-Off Required; Never Auto-Populated]')
- Explicit reasoning trace for every test and resource allocation
- Untrusted text sanitization & prompt injection defense
- Human review sign-off gate before release of draft work packages
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.audit import AuditTrail
from core.stamps import DRAFT_STAMP, assert_no_banned_phrases

INJECTION_PATTERNS = [
    r"ignore\s+(?:\w+\s+){0,6}instructions?",
    r"release\s+(?:\w+\s+){0,6}without\s+testing",
    r"release\s+for\s+use",
    r"override\s+(?:\w+\s+){0,4}(?:spec|specification|limits?)",
    r"system\s*(?:prompt|directive)",
    r"(?:you\s+are\s+now\s+in\s+)?developer\s+mode",
    r"bypass\s+(?:\w+\s+){0,4}(?:quality|qa|gmp|controls?)",
]

SPEC_PLACEHOLDER = "[TBD - Client Specification Sign-Off Required; Never Auto-Populated]"


@dataclass
class TestItem:
    test_code: str
    method_name: str
    instrument_skid: str
    duration_hours: float
    sample_volume_required: str
    spec_limit: str
    reasoning: str


@dataclass
class WorkPackage:
    work_package_id: str
    client_program: str
    raw_request_text: str
    sanitized_text: str
    injection_flag: bool
    injection_warnings: List[str]
    tests: List[TestItem]
    total_skid_hours: float
    total_sample_volume: str
    deliverables: List[str]
    open_questions: List[str]
    reasoning_traces: List[str]
    is_signed_off: bool = False
    signed_off_by: Optional[str] = None
    signed_off_at: Optional[str] = None
    disclaimer: str = DRAFT_STAMP

    def to_markdown(self) -> str:
        gate_status = f"SIGNED-OFF by {self.signed_off_by} at {self.signed_off_at}" if self.is_signed_off else "PENDING HUMAN QA REVIEW (LOCKED)"
        md = f"""# GMP QC Work Package: {self.client_program}
> **{self.disclaimer}**  
> **Status:** {gate_status}  
> **Work Package ID:** `{self.work_package_id}`

---

## 1. Scope & Client Request
**Sanitized Request:**
> {self.sanitized_text}

"""
        if self.injection_flag:
            md += f"""\n> [!WARNING]
> **Adversarial Input Detected:** Request text is treated as untrusted data and never executed; suspicious directives are flagged.
> Flagged patterns: {', '.join(self.injection_warnings)}
\n"""

        md += """## 2. Analytical Testing Matrix & Skid Allocations
| Test Code | Method Name | Instrument Skid | Duration (h) | Sample Volume | Specification Limit |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for t in self.tests:
            md += f"| `{t.test_code}` | {t.method_name} | {t.instrument_skid} | {t.duration_hours:.1f}h | {t.sample_volume_required} | `{t.spec_limit}` |\n"

        md += f"""\n**Total Estimated Instrument Hours:** {self.total_skid_hours:.1f} hours  
**Total Sample Material Requirement:** {self.total_sample_volume}

---

## 3. Method Selection Reasoning Traces
"""
        for trace in self.reasoning_traces:
            md += f"- {trace}\n"

        md += """\n---

## 4. Work Package Deliverables
"""
        for d in self.deliverables:
            md += f"- [ ] {d}\n"

        md += """\n---

## 5. Clarifying Questions for Client Alignment
"""
        for q in self.open_questions:
            md += f"- **Q:** {q}\n"

        assert_no_banned_phrases(md)
        return md


def sanitize_client_input(raw_text: str) -> Tuple[str, bool, List[str]]:
    """
    Sanitizes untrusted client request text.
    Request text is treated as untrusted data and never executed; suspicious directives are flagged.
    """
    clean_text = unicodedata.normalize("NFKD", raw_text)
    warnings = []
    injection_detected = False

    for pat in INJECTION_PATTERNS:
        matches = re.findall(pat, clean_text, flags=re.IGNORECASE)
        if matches:
            injection_detected = True
            warnings.append(f"Suspicious directive flagged: '{pat}'")
            clean_text = re.sub(pat, "[SUSPICIOUS DIRECTIVE FLAGGED - UNTRUSTED DATA NOT EXECUTED]", clean_text, flags=re.IGNORECASE)

    return clean_text.strip(), injection_detected, warnings


def generate_work_package_from_client_request(
    client_program: str,
    raw_request_text: str,
    lot_type: str = "DS",  # DS or DP
) -> WorkPackage:
    """
    Deterministically parses client request text and constructs a formal QC Work Package.
    """
    sanitized, injection_detected, warnings = sanitize_client_input(raw_request_text)
    lower = sanitized.lower()

    tests: List[TestItem] = []
    traces: List[str] = []
    open_questions: List[str] = []

    # Rule 1: Monomer & Aggregate Purity (SEC-HPLC)
    if any(k in lower for k in ["sec", "size exclusion", "aggregate", "hmw", "monomer", "purity"]):
        tests.append(
            TestItem(
                test_code="QC-SEC-01",
                method_name="SEC-UPLC Monomer & High-Molecular Weight Aggregate Analysis",
                instrument_skid="UPLC_SEC_CEX",
                duration_hours=18.0,
                sample_volume_required="150 µL (>= 1.0 mg/mL)",
                spec_limit=SPEC_PLACEHOLDER,
                reasoning="Triggered by request for SEC/aggregate purity. Allocated to Waters Acquity UPLC skid.",
            )
        )
        traces.append("SEC-HPLC: Rule matched 'sec'/'aggregate'. Allocated 18.0h on UPLC_SEC_CEX for triplicate injection + system suitability.")
    else:
        open_questions.append("Is SEC-HPLC size-exclusion chromatography required for soluble aggregate quantification?")

    # Rule 2: Charge Variant Profile (icIEF / CEX)
    if any(k in lower for k in ["charge", "icief", "cex", "cief", "acidic", "basic", "isoelectric"]):
        tests.append(
            TestItem(
                test_code="QC-ICIEF-02",
                method_name="Imaged Capillary Isoelectric Focusing (icIEF) Charge Profiling",
                instrument_skid="CE_icIEF",
                duration_hours=14.0,
                sample_volume_required="100 µL (>= 0.5 mg/mL)",
                spec_limit=SPEC_PLACEHOLDER,
                reasoning="Triggered by request for charge variants/icIEF. Allocated to ProteinSimple Maurice skid.",
            )
        )
        traces.append("icIEF: Rule matched 'charge'/'icief'. Allocated 14.0h on CE_icIEF for focus/mobilization capillary cartridge run.")
    else:
        open_questions.append("Is charge heterogeneity (icIEF/CEX) required for stability or lot release profiling?")

    # Rule 3: Biological Activity & Potency (Bioassay)
    if any(k in lower for k in ["potency", "bioassay", "activity", "binding", "elisa", "cell-based"]):
        tests.append(
            TestItem(
                test_code="QC-BIO-03",
                method_name="Cell-Based Relative Potency Bioassay (4-PL Parallelism)",
                instrument_skid="Cell_Bioassay",
                duration_hours=20.0,
                sample_volume_required="500 µL (>= 2.0 mg/mL)",
                spec_limit=SPEC_PLACEHOLDER,
                reasoning="Triggered by request for potency/bioassay. Allocated to 96-Well Hamilton Star / BSL-2 Potency Suite.",
            )
        )
        traces.append("Bioassay: Rule matched 'potency'/'bioassay'. Allocated 20.0h on Cell_Bioassay for plate plating, incubation, and 4-PL statistical curve fitting.")

    # Rule 4: Purity & Truncation (CE-SDS)
    if any(k in lower for k in ["ce-sds", "sds", "fragment", "reduced", "non-reduced", "clip"]):
        tests.append(
            TestItem(
                test_code="QC-CESDS-04",
                method_name="Capillary Electrophoresis SDS (Reduced & Non-Reduced)",
                instrument_skid="CE_icIEF",
                duration_hours=12.0,
                sample_volume_required="120 µL (>= 1.0 mg/mL)",
                spec_limit=SPEC_PLACEHOLDER,
                reasoning="Triggered by request for CE-SDS purity/fragmentation. Allocated to ProteinSimple Maurice CE-SDS mode.",
            )
        )
        traces.append("CE-SDS: Rule matched 'ce-sds'/'fragment'. Allocated 12.0h on CE_icIEF for SDS separation.")

    # Rule 5: Safety Panel (Endotoxin & Bioburden)
    if any(k in lower for k in ["release", "commercial", "ppq", "safety", "endotoxin", "bioburden"]):
        tests.append(
            TestItem(
                test_code="QC-ENDO-05",
                method_name="Bacterial Endotoxin Testing (Kinetic Chromogenic LAL)",
                instrument_skid="Offline_Microbiology",
                duration_hours=6.0,
                sample_volume_required="1.0 mL undiluted",
                spec_limit=SPEC_PLACEHOLDER,
                reasoning="Triggered by lot release/safety context. Allocated to QC Microbiology suite.",
            )
        )
        traces.append("Endotoxin: Rule matched 'release'/'safety'. Allocated 6.0h for LAL chromogenic kinetic run.")

    # Default fallback if no rules matched
    if not tests:
        tests.append(
            TestItem(
                test_code="QC-GEN-00",
                method_name="Preliminary SEC-HPLC Screening Panel",
                instrument_skid="UPLC_SEC_CEX",
                duration_hours=10.0,
                sample_volume_required="100 µL",
                spec_limit=SPEC_PLACEHOLDER,
                reasoning="Default analytical baseline for unidentified biologics request.",
            )
        )
        traces.append("Default: No specific method keywords matched. Configured minimal SEC-HPLC screening panel.")
        open_questions.append("Please specify target analytical testing panel (e.g. SEC-HPLC, icIEF, Bioassay, CE-SDS).")

    # Always add general alignment questions
    if "temperature" not in lower and "storage" not in lower:
        open_questions.append("What are the registered storage temperatures (e.g., -80 °C, 5 °C, 25 °C/60% RH)?")
    if "stability" in lower and not any(k in lower for k in ["pull", "month", "timeline", "interval"]):
        open_questions.append("What are the target stability pull intervals (e.g., 0M, 1M, 3M, 6M, 12M)?")

    total_hours = sum(t.duration_hours for t in tests)
    wp_id = f"WP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"

    deliverables = [
        f"Gated Phase I/II Laboratory Testing Log & System Suitability Report for {client_program}",
        "Empower 3 / Compass Chromatographic & Electrophoretic Data Export",
        "Official Certificate of Analysis (CoA) Draft for Quality Assurance Review",
        "Summary Testing Dossier (DRAFT - NOT A GMP RECORD)",
    ]

    return WorkPackage(
        work_package_id=wp_id,
        client_program=client_program,
        raw_request_text=raw_request_text,
        sanitized_text=sanitized,
        injection_flag=injection_detected,
        injection_warnings=warnings,
        tests=tests,
        total_skid_hours=total_hours,
        total_sample_volume=f"~{len(tests) * 0.5:.1f} mL DS material",
        deliverables=deliverables,
        open_questions=open_questions,
        reasoning_traces=traces,
    )


def sign_off_work_package(
    work_package: WorkPackage,
    reviewer_name: str,
    reviewer_role: str,
    justification: str,
    audit_trail: Optional[AuditTrail] = None,
) -> WorkPackage:
    """
    Applies the human review sign-off gate to unlock export of a work package.
    """
    if not reviewer_name or len(reviewer_name.strip()) < 3:
        raise ValueError("Reviewer name must be at least 3 characters.")
    if not justification or len(justification.strip()) < 10:
        raise ValueError("A substantive justification (>= 10 characters) is required for sign-off.")

    now_utc = datetime.now(timezone.utc).isoformat()
    work_package.is_signed_off = True
    work_package.signed_off_by = f"{reviewer_name} ({reviewer_role})"
    work_package.signed_off_at = now_utc

    if audit_trail is not None:
        audit_trail.log_action(
            actor_name=reviewer_name,
            actor_role=reviewer_role,
            action="SIGN_OFF_WORK_PACKAGE",
            entity_type="work_package",
            entity_id=work_package.work_package_id,
            justification=justification,
            payload={
                "work_package_id": work_package.work_package_id,
                "client_program": work_package.client_program,
                "total_hours": work_package.total_skid_hours,
                "tests_count": len(work_package.tests),
                "timestamp": now_utc,
            },
        )

    return work_package
