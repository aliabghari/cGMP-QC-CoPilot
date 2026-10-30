# GMP QC Work Package Generation: Rules & Governance Card
## cGMP-QC-CoPilot (WP2 Operating Layer)

> **DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL**

---

### 1. Architectural Purpose
The Work Package Generator (`pm/work_package_generator.py`) provides a deterministic, rule-based pipeline that translates unstructured client statements of work (SOW) or testing requests into standardized, structured draft QC work packages for commercial and clinical biologics testing.

In accordance with **Guardrail G3 (Deterministic Computing)**, language models **never compute hours, invent numbers, or set specifications**. All mappings are executed via deterministic rule engines. The architectural protection against prompt injection is structural: the deterministic rule engine never executes request text, parses it into executable code, or delegates free request text to an interpreter.

---

### 2. Method & Equipment Mapping Rules

| Client Request Keywords | Analytical Method | Instrument Skid | Standard Duration (h) | Sample Volume | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sec`, `size exclusion`, `aggregate`, `hmw`, `monomer`, `purity` | **SEC-UPLC** Monomer & High-Molecular Weight Aggregate Analysis | `UPLC_SEC_CEX` (Waters Acquity) | 18.0h | 150 µL (>= 1.0 mg/mL) | Triplicate injections, system suitability, standard curve |
| `charge`, `icief`, `cex`, `cief`, `acidic`, `basic`, `isoelectric` | **icIEF** Imaged Capillary Isoelectric Focusing | `CE_icIEF` (ProteinSimple Maurice) | 14.0h | 100 µL (>= 0.5 mg/mL) | Focus & mobilization cartridge sequence with internal pI markers |
| `potency`, `bioassay`, `activity`, `binding`, `elisa`, `cell-based` | **Cell-Based Potency Bioassay** (4-PL Parallelism) | `Cell_Bioassay` (Hamilton / 96-well) | 20.0h | 500 µL (>= 2.0 mg/mL) | Cell seeding, incubation, dosing, plate readout, 4-PL statistical fit |
| `ce-sds`, `sds`, `fragment`, `reduced`, `non-reduced`, `clip` | **Capillary Electrophoresis SDS** | `CE_icIEF` (Maurice CE-SDS Mode) | 12.0h | 120 µL (>= 1.0 mg/mL) | Sample denaturation, capillary electrokinetic injection |
| `release`, `commercial`, `ppq`, `safety`, `endotoxin`, `bioburden` | **Bacterial Endotoxin Testing** (Kinetic Chromogenic LAL) | `Offline_Microbiology` | 6.0h | 1.0 mL undiluted | 96-well microplate endotoxin kinetic reader run |

---

### 3. Specification Placeholder Rule (Strict Non-Invention)
To protect GMP data integrity:
- The system is **strictly prohibited from inventing numeric specification limits** (e.g. `>= 95.0% monomer`).
- Every test parameter is stamped with:
  `[TBD - Client Specification Sign-Off Required; Never Auto-Populated]`
- Numerical specification limits may only be populated through formally approved Quality Technical Agreements (QTAs) signed off by Quality Assurance.

---

### 4. Adversarial Text Sanitization & Suspicious Directive Defense
Unstructured client text is treated as **untrusted data and never executed; suspicious directives are flagged**:
- Directives such as `"ignore previous instructions"`, `"release lot without testing"`, `"override specification"`, or `"bypass QA"`.
- Malicious patterns are stripped and replaced with `[SUSPICIOUS DIRECTIVE FLAGGED - UNTRUSTED DATA NOT EXECUTED]`.
- The work package is flagged with `injection_flag = True` and surfaced to QA with an adversarial input warning.

---

### 5. Human-in-the-Loop Sign-Off Gate
Every generated work package is initially created in a **LOCKED / PENDING QA REVIEW** state.
To unlock the work package for export or scheduling:
1. Reviewer Name (minimum 3 characters)
2. Reviewer Role (e.g., Principal Specialist, QC Project Manager)
3. Substantive Justification (minimum 10 characters)
4. Cryptographic record appended to `audit_trail` via `core/audit.py`.
