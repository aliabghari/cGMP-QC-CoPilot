# QTA & SOW Governance: Communication Cadences, Chain-of-Custody (CoC) & Change Order Triggers

**Document ID:** REF-QC-COMMS-004 (Effective Rev 3.4)  
**Applicable Regulations:** FDA Guidance on Quality Agreements (2016), ICH Q10, EU GMP Chapter 7, Example Biologics Site SOP-PMO-0108 (fictional example)  

---

## 1. Quality Technical Agreement (QTA) Notification SLAs
Every Example CDMO Client campaign and Captive Pipeline commercial program operates under a binding Quality Technical Agreement (QTA) and Project Statement of Work (SOW). The QC Project Manager enforces the following mandatory notification clocks:

| Event Type | Target Audience | Mandatory Notification SLA | Required Content & Technical Evidence |
| :--- | :--- | :--- | :--- |
| **Initial OOS / OOT Alert (Phase I Initiated)** | External CDMO Client QA & Internal CMC Lead | **$\le 24\text{ Hours}$ (1 Business Day)** | Batch ID, LIMS Test Code, observed value vs. spec, and contemporaneous Phase I Empower telemetry status. |
| **Phase I Invalidation OR Phase II Escalation** | Director QC, QA MSO, Client QA, MSAT Lead | **$\le 48\text{ Hours}$ post-Phase I Gate** | Full Gated Phase I Checklist summary, root cause proof (e.g., Run #14 chiller/pressure telemetry), and re-injection ETA. |
| **Cold-Chain Excursion Impact Assessment** | Client CMC Lead, QA Disposition, Global Supply Chain | **$\le 3\text{ Business Days}$** | Hybrid Eyring+ANN-TVP ODE trajectory, net $\Delta\text{HMW}\%$, and exact shelf-life deduction recommendation. |
| **Routine ICH Q1A Stability Pull Leveling ($\pm 7\text{d}$)** | Peer QC PMs, LIMS Coordinator, Lab Supervisors | **Weekly (Prior to Pull Window Close)** | Heijunka leveled schedule table confirming zero critical-path SLA impact. |
| **CRO Outsourced Testing Follow-Up (Lot Release)** | External CRO PM (Bioburden/Mycoplasma/Viral) | **T-minus 5 & T-minus 2 Business Days** | Sample Chain-of-Custody (CoC) tracking ID, LIMS manifest verification, and CoA upload reminder. |

---

## 2. Sample Chain-of-Custody (CoC) & LIMS Governance
1. **Internal USP/DSP-to-QC Handoff:** All Drug Substance (DS) and In-Process Control (IPC) samples require dual barcode scan at the QC Sample Management Cryo-Vault (`2–8°C` or `-80°C`) with verified time-out-of-refrigeration (TOR) logs.
2. **External CRO Shipments:** Outsourced safety assays (e.g., Adventitious Agents, qPCR Mycoplasma, Extended characterization) require qualified dry-ice/phase-change shipper temp-tale verification within 4 hours of CRO receipt before testing is authorized.

---

## 3. Statement of Work (SOW) Change Order (CO) Triggers
While standard method execution and routine Phase Ia instrument trouble-shooting are absorbed under baseline SOW pricing, the QC PM must flag a **Proactive SOW Change Order** to Program Management & Business Development when:
- **Trigger CO-01 (Client Method Remediation / Dwell-Volume Re-Engineering):** Donor HPLC method fails initial system suitability due to uncompensated gradient dwell volume or non-robust peak resolution requiring Example Biologics Site Analytical Science LSS/SMA mechanistic re-development (`16–24 FTE hours`).
- **Trigger CO-02 (Client-Caused Cold-Chain Excursion Modeling & Extra Stability Pull):** Third-party courier thermal excursion requiring custom Hybrid Eyring+ANN-TVP kinetic modeling and unscheduled verification testing (`12–18 FTE hours + $4,800 skid fee`).
- **Trigger CO-03 (Expedited <5-Day Turnaround Time Request):** Client requests compressed release TAT requiring dedicated weekend/holiday Shift B overtime allocation.
