"""
cGMP-QC-CoPilot: Modular Analytical Method Tech-Transfer Engine.

Generates structured, regulatory-aligned analytical method transfer protocols and reports
between donor (sending) and receiving QC laboratories (e.g., CDMO client to Example Biologics Site site).

Sections:
1. Scope & Objective
2. Analytical Target Profile (ATP)
3. Method Transfer List & Equipment Equivalency
4. Method Comparability & Statistical Acceptance Criteria (Strict placeholder rule: NO invented limits)
5. Roles & Responsibilities
6. Timeline & Deliverables
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.audit import AuditTrail
from core.stamps import DRAFT_STAMP, assert_no_banned_phrases

ACCEPTANCE_CRITERIA_PLACEHOLDER = (
    "[TBD - Human QA & Analytical Development Sign-Off Required; Limits Must Not Be Auto-Generated]"
)


def generate_tech_transfer_document(
    project_id: str,
    program_name: str,
    sending_lab: str = "Client Donor Analytical Development (Ext)",
    receiving_lab: str = "Example Biologics Site QC Laboratories (Commercial Mammalian)",
    methods_to_transfer: Optional[List[Dict[str, str]]] = None,
    signed_off_by: Optional[str] = None,
    signed_off_at: Optional[str] = None,
) -> str:
    """
    Generates a structured method transfer Markdown document.
    Enforces that statistical acceptance criteria are strictly placeholders requiring human sign-off.
    """
    methods = methods_to_transfer or [
        {
            "method_code": "TM-SEC-01",
            "name": "High-Molecular-Weight Aggregate Analysis by SEC-UPLC",
            "sending_instrument": "Agilent 1260 Infinity HPLC (V_D = 1.15 mL)",
            "receiving_instrument": "Waters Acquity H-Class Bio UPLC (V_D = 0.32 mL)",
            "cqa": "Purity (Monomer %, HMW %)",
            "dwell_volume_hold": "+1.04 min isocratic hold to match dwell volume",
        },
        {
            "method_code": "TM-CEX-02",
            "name": "Charge Heterogeneity by Imaged Capillary Isoelectric Focusing (icIEF)",
            "sending_instrument": "ProteinSimple iCE3",
            "receiving_instrument": "ProteinSimple Maurice (icIEF Mode)",
            "cqa": "Charge Variants (Acidic %, Main %, Basic %)",
            "dwell_volume_hold": "Direct cartridge transfer with matched ampholyte pH 3-10 range",
        },
    ]

    status_badge = (
        f"APPROVED FOR TRIAL EXECUTION by {signed_off_by} on {signed_off_at}"
        if signed_off_by
        else "DRAFT - PENDING METHOD TRANSFER PROTOCOL APPROVAL"
    )

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    md = f"""# Analytical Method Transfer Protocol: {program_name}
> **{DRAFT_STAMP}**  
> **Status:** `{status_badge}`  
> **Project ID:** `{project_id}` | **Date:** `{now_str}`  
> **Sending Lab:** `{sending_lab}` | **Receiving Lab:** `{receiving_lab}`

---

## 1. Scope & Technical Objective
This protocol governs the formal analytical method transfer for **{program_name}** (`{project_id}`) from `{sending_lab}` to `{receiving_lab}`.
The transfer establishes intermediate precision, repeatability, and comparability between the donor and receiving platforms in accordance with **ICH Q2(R2)** and **USP <1224> Transfer of Analytical Procedures**.

---

## 2. Analytical Target Profile (ATP)
- **Primary Analyte:** {program_name} (Recombinant Humanized Monoclonal Antibody / Biologic)
- **Critical Quality Attributes (CQAs):**
  - Soluble aggregation (HMW species) & fragment degradation
  - Charge variant distribution (Main isoform, acidic and basic variants)
  - Relative biological binding and cell-based potency
- **Intended Purpose:** In-process control (IPC), drug substance (DS) lot release, and long-term ICH stability testing.

---

## 3. Method Transfer List & Equipment Equivalency
| Method Code | Procedure Name | Sending Platform | Receiving Platform | Dwell Volume / Operational Adaptation |
| :--- | :--- | :--- | :--- | :--- |
"""
    for m in methods:
        md += f"| `{m['method_code']}` | {m['name']} | {m['sending_instrument']} | {m['receiving_instrument']} | {m['dwell_volume_hold']} |\n"

    md += f"""
---

## 4. Method Comparability & Statistical Acceptance Criteria
> [!IMPORTANT]
> **Regulatory Requirement:** In accordance with ICH Q2(R2) and site SOPs, numerical acceptance criteria for repeatability, intermediate precision, and two one-sided t-tests (TOST) equivalence must be set by Qualified Analytical Development & QA Leads based on historical process capability.
> **The decision-support tool strictly prohibits auto-generating or inventing numerical acceptance limits.**

| Test Parameter | Acceptance Criteria Specification | Statistical Evaluation Method | Status |
| :--- | :--- | :--- | :--- |
| **SEC-HPLC Repeatability (Receiving Lab)** | `{ACCEPTANCE_CRITERIA_PLACEHOLDER}` | %RSD of 6 replicate injections | PENDING QA SIGN-OFF |
| **SEC-HPLC Intermediate Precision** | `{ACCEPTANCE_CRITERIA_PLACEHOLDER}` | 2 Analysts, 2 Instruments, 2 Days | PENDING QA SIGN-OFF |
| **Platform Comparability (TOST Equivalence)** | `{ACCEPTANCE_CRITERIA_PLACEHOLDER}` | 90% Confidence Interval within Equivalence Margin (-delta, +delta) | PENDING QA SIGN-OFF |
| **icIEF Isoelectric Point & Profile Concordance** | `{ACCEPTANCE_CRITERIA_PLACEHOLDER}` | Visual pattern match & peak area % delta | PENDING QA SIGN-OFF |

---

## 5. Roles & Responsibilities (RACI Summary)
- **Sending Unit (Donor Lab):** Provides qualified reference standards, historical assay validation reports, and technical troubleshooting support.
- **Receiving Unit (Example Biologics Site QC):** Executes co-validation runs, prepares testing reagents, documents raw data in Empower 3, and authors the method transfer report.
- **Quality Assurance (QA):** Reviews and approves transfer protocol, oversees deviations, and dispositions the final Transfer Summary Report.
- **QC Project Management:** Coordinates skid availability, analyst scheduling, sample logistics, and SLA adherence.

---

## 6. Execution Milestones & Timeline
1. **Milestone 1 (Pre-Transfer Preparation):** Reagent acquisition & instrument suitability verification (Target: Day 1–3).
2. **Milestone 2 (Co-Validation Runs):** Parallel execution across Sending and Receiving units on 3 distinct production lots (Target: Day 4–7).
3. **Milestone 3 (Statistical Data Evaluation):** Evaluation of %RSD, retention time shifts, and TOST confidence intervals (Target: Day 8–9).
4. **Milestone 4 (Final Transfer Report Sign-Off):** QA approval and registration into LIMS test method catalog (Target: Day 10).
"""

    assert_no_banned_phrases(md)
    return md


def verify_tech_transfer_acceptance_placeholders(doc_text: str) -> bool:
    """
    Verifies that the generated tech-transfer document contains proper acceptance criteria placeholders
    and does not invent numerical limits without sign-off.
    """
    if ACCEPTANCE_CRITERIA_PLACEHOLDER not in doc_text:
        return False

    # Ensure no fabricated claims of approval or release
    assert_no_banned_phrases(doc_text)
    return True
