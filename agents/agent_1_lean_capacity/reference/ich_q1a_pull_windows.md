# ICH Q1A(R2) & Example Biologics Site Site Governance: Stability Pull Windows & Dual-Stream Prioritization

**Document ID:** REF-QC-LEAN-001 (Effective Rev 5.0)  
**Applicable Regulations:** 21 CFR 211.166, ICH Q1A(R2), EU GMP Annex 15, Example Biologics Site SOP-QC-0412 (fictional example)  

---

## 1. Purpose & Regulatory Context
In a high-utilization mammalian cell culture facility operating both **Captive Commercial/Clinical programs** and **Example CDMO Client partner campaigns**, analytical instrument skids (Waters Acquity H-Class Bio UPLC, ProteinSimple Maurice icIEF/CE-SDS, and BSL-2 Cell-Based Potency Suites) experience periodic wave-loading caused by Upstream (USP) commercial-scale (fictional example) bioreactor harvests and Downstream (DSP) chromatography cadence.

To prevent laboratory overload—which historically correlates with elevated Phase I Analyst Invalid rates and Waters Empower sequence setup errors—the QC Project Management Office (PMO) utilizes **Heijunka (LEAN Load Leveling)** within site-defined pull windows (example ±7 d) governed by site SOP-QC-0412 (fictional example).

---

## 2. Site-Defined Stability Pull Float Windows (Example Policy)
Stability samples stored in walk-in environmental chambers ($5^\circ\text{C} \pm 3^\circ\text{C}$, $25^\circ\text{C}/60\%\text{ RH}$, $40^\circ\text{C}/75\%\text{ RH}$) must be pulled and initiated within the following windows relative to the nominal T0 placement date:

| Stability Timepoint | Nominal Interval | Permitted Pull Window ($\pm$ Days) | Max Testing Initiation Window post-Pull | Regulatory / SOP Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Accelerated / Stress (2W, 1M)** | 14 – 30 Days | **$\pm 2$ to $\pm 3$ Days** | $\le 5$ Business Days | High kinetic sensitivity at $40^\circ\text{C}$; narrow window mandatory. |
| **Intermediate / Real-Time (3M)** | 90 Days | **$\pm 3$ Days** | $\le 7$ Business Days | Early trend verification window per SOP-QC-0412 (fictional example). |
| **Mid-Term Stability (6M, 9M, 12M)** | 180 – 365 Days | **$\pm 7$ Days** | $\le 10$ Business Days | ICH Q1A(R2) standard semi-annual/annual window ($\pm 7\text{d}$ used for LEAN smoothing). |
| **Long-Term Stability (18M, 24M, 36M+)** | 540 – 1825 Days | **$\pm 14$ Days** | $\le 14$ Business Days | Low kinetic slope at $5^\circ\text{C}$; maximum scheduling flexibility. |

> **CRITICAL RULE:** Shifting a stability pull date *within* the permitted window (e.g., shifting a 6M, 9M, or 12M pull by $+5$ to $+7$ days into the following workweek) is **NOT** a laboratory deviation, provided the schedule update is logged in LIMS *prior* to the close of the nominal pull window and approved by the QC PM.

---

## 3. Dual-Stream Priority Hierarchy (Captive Pipeline vs. Example CDMO Client)
When weekly analytical skid utilization exceeds the **85.0% LEAN Operational Ceiling**, work packages are prioritized strictly by **Critical-Path Tier**, regardless of internal vs. external sponsorship:

1. **Tier 1A — Zero-Float Regulatory & Commercial Critical Path (`float_days == 0`):**
   - PPQ (Process Performance Qualification) DS/DP Lot Release batches.
   - BLA / MAA registration lots and FDA Pre-License Inspection (PLI) commitment samples.
   - CDMO Commercial Launch lots with contractually locked fill-finish slots.
2. **Tier 1B — Active Biomanufacturing In-Process & Clinical Release (`float_days <= 2`):**
   - DSP UF/DF Bulk Drug Substance release required to clear cleanroom suite hold-time limits.
   - Phase I/II/III IND/IMPD clinical trial supply lots approaching drug depot ship dates.
3. **Tier 2 — Time-Critical Accelerated Stability & Excursion Assessments (`float_days <= 3`):**
   - $40^\circ\text{C}$ stress/accelerated pulls and cold-chain thermal excursion investigation samples.
4. **Tier 3 — Routine Long-Term Stability Pulls (`float_days >= 7`):**
   - Real-time $5^\circ\text{C}$ and intermediate $25^\circ\text{C}$ stability pulls at 6M, 9M, 12M, 24M, and 36M timepoints. **Eligible for automatic LEAN $\pm 7\text{d}$ smoothing.**

---

## 4. Skid Capacity Metrics & Analyst Shift Constraints
- **Standard Weekly Skid Capacity (100%):** 120 instrument-hours per skid family per week (3 parallel analytical systems $\times$ 40 productive hours/week net of system suitability and preventive maintenance).
- **LEAN Redline Threshold (85%):** 102 instrument-hours/week. Exceeding 85% eliminates buffer capacity for OOS Phase I re-injections or urgent DSP hold-time checks.
- **Hard Overtime / Burnout Risk (100%+):** >120 instrument-hours/week triggers mandatory Heijunka leveling and cross-trained shift balancing between Day Shift (Shift A) and Swing Shift (Shift B).
