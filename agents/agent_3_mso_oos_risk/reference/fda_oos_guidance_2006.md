# FDA Guidance for Industry (2006) & 21 CFR 211.192: Gated Phase I vs. Phase II OOS Investigation Rules

**Document ID:** REF-QC-MSO-003 (Effective Rev 6.1)  
**Applicable Regulations:** FDA OOS Guidance (Oct 2006), 21 CFR 211.192, EudraLex Vol 4 Chapter 6, ICH Q9(R1)  

---

## 1. Regulatory Mandate: Prohibition of "Testing Into Compliance"
Under **21 CFR 211.192** and the **FDA Guidance for Industry: Investigating Out-of-Specification (OOS) Test Results for Pharmaceutical Production (2006)**, every unexplained discrepancy or failure of a batch to meet any of its specifications must be thoroughly investigated, whether or not the batch has already been distributed.

> **CARDINAL cGMP RULE:** An initial OOS test result **cannot** be invalidated by simply re-testing the sample and obtaining a passing result ("testing into compliance"), nor can passing re-test values be averaged with an original OOS value to report a passing mean, unless a **documented, contemporaneous, assignable laboratory root cause** is proven during Phase I.

---

## 2. Gated Two-Phase OOS Architecture at Example Biologics Site

### Phase I: Laboratory Assessment (Bench & Instrument Telemetry Audit)
Before any vial is re-injected or any new sample preparation is authorized, the Analyst and QC Supervisor must execute the **Gated Phase I Checklist** (`assets/gated_phase1_checklist.json`):
1. **Phase Ia (Obvious / Contemporaneous Instrument or Utility Malfunction):**
   - Power outage, air/nitrogen drop, or **Waters Empower 3 hardware fault alarm** logged prior to or during peak integration (e.g., pump overpressure abort $>300\text{ bar}$, lamp failure, broken syringe, or autosampler Peltier cooling failure).
2. **Phase Ib (Detailed Bench & Electronic Audit Trail Forensic Review):**
   - Verification of raw integration parameters (prohibiting manual "inhibit integration" or baseline skimming without SOP authorization).
   - Verification of pipetting records, balance printouts, mobile phase pH/conductivity logs, and **sample residence time/temperature inside the autosampler tray**.
   - **Biochemical Artifact Check:** In CEX-UPLC (`TC-CEX-204`), monoclonal antibodies held in weakly buffered mobile phase or pierced vials at elevated autosampler temperatures ($>15^\circ\text{C}$) for $>24\text{ hours}$ undergo accelerated non-enzymatic **Asparagine (Asn) deamidation to Isoaspartate/Aspartate**, artificially elevating the Acidic Peak Group (%).

**Phase I Gate Decision:**
- **IF Assignable Lab Root Cause Proven:** Document root cause in TrackWise, invalidate the compromised run with QA concurrence, execute corrective action on the instrument, and perform the pre-defined re-injection protocol.
- **IF NO Assignable Lab Root Cause Found:** **STOP LAB RE-TESTING.** The OOS result is confirmed as representative of the sample. Immediately escalate to **Phase II**.

---

## 3. Phase II: Full-Scale MSAT Bioprocess & Manufacturing Investigation
When Phase I finds no laboratory error, the Local MSO Lead convenes a cross-functional Phase II Root Cause Analysis (RCA) with **MSAT (Manufacturing Science & Technology)**, **USP (Upstream Processing)**, and **DSP (Downstream Processing)**:

| Suspect CQA / OOS Mode | Upstream (USP - commercial-scale (fictional example) Bioreactor) Root Causes | Downstream (DSP - Purification) Root Causes |
| :--- | :--- | :--- |
| **Elevated Acidic Charge Variants (`TC-CEX-204`)** | Bioreactor pH control drift ($>7.15$), prolonged cell culture duration ($>14\text{ days}$), or elevated $37^\circ\text{C}$ harvest hold time inducing **Asn deamidation**. | Alkaline strip buffer carryover or extended room-temperature intermediate pool hold time prior to UF/DF formulation. |
| **Elevated HMW Aggregates (`TC-SEC-101`)** | High shear sparging / dissolved oxygen (DO) microbubble cavitation or glycosylation deficit in CHO production culture. | **Protein A affinity elution & Low-pH Viral Inactivation ($\text{pH } 3.5 \pm 0.1$) over-incubation** ($>60\text{ min}$) or local pH overshoot during Tris titration. |
| **Reduced Potency (`TC-BIO-309`)** | Methionine (`Met-252`/`Met-428`) oxidation from trace $\text{Cu}^{2+}$/dissolved oxygen or CDR Trp photo-oxidation. | Peroxide leachables from degraded Polysorbate-20/80 raw material or light exposure in clear glass vessels. |
