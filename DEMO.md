# Live Panel Demo Script: cGMP-QC-CoPilot
### 5-minute walkthrough for the Principal Specialist, QC Project Management role

> **Data status:** every lot, run, lab, blocker and trace in this demo is synthetic and every organization name is fictional. All outputs are stamped `DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL`. The sign-off gate is a Part 11-*style* demonstration, not a compliant e-signature system.

**Story in one line:** *"A QC project manager sees conflicts early, protects commitments, keeps investigations clean, and tells the right people in time. This prototype shows each of those on synthetic data."*

| Time | Stop | Role skill shown |
| :--- | :--- | :--- |
| 0:00-0:30 | Opening | Framing and honesty about how it was built |
| 0:30-2:00 | **Stop 1:** Plan vs. physical arrivals, leveling, escalation | Planning, resource conflict escalation, client communication |
| 2:00-3:30 | **Stop 2:** OOS Phase I telemetry gate | Local MSO for OOX, 21 CFR 211.192 discipline |
| 3:30-4:30 | **Stop 3:** Cold-chain excursion and stability decision support | Stability MSO, judgment under uncertainty |
| 4:30-5:00 | Close | Limits and next steps |

---

## Setup (T-minus 5 minutes)

```bash
python preflight_check.py
streamlit run app.py
```

Expected:
- Preflight ends with `PREFLIGHT CHECK PASSED` (demo scenario about 0.3 s, but timing varies run to run; do not quote it).
- `pytest tests/ -q` on Python 3.14 without PyTorch: **140 passed, 3 skipped, 143 total** (update if you add tests). The 3 skips are PyTorch-only and will run if PyTorch is installed. Python 3.13 is the declared demo interpreter; a warning on 3.14 is expected.
- Keep `results/screenshots/demo_stops/` open as a fallback walkthrough.

---

## Opening (0:00-0:30)

> *"This was built with AI coding tools under my direction, and every dataset and organization name in it is synthetic. I'm showing it to demonstrate how I think about QC project management. It is a prototype, not a validated system. It makes no LLM calls: scheduling, kinetics and triage are deterministic code, and every output needs human review."*

---

## Stop 1 (0:30-2:00): Plan vs. physical arrivals, leveling, escalation

**Show:** workspace *QC Operations & Project Management*, tab *1. LEAN Capacity & Critical Path* (the CP-SAT view). Use this tab, not the scripted tour, for the numbers.

**What you will see:** Unleveled UPLC demand of **172.0 h = 143.3%** of the 120 h staffed capacity, which exceeds the 168 h in a calendar week. After CP-SAT leveling: **168.0 h = 140.0%**, non-overlapping, with fixed commercial commitments protected.

**Say:**
> *"Here is a planning trap. The plan tags this skid's week 42 workload at 136 hours, 113% of staffed capacity: tight but arguably manageable. But two samples tagged for the following week physically arrive Thursday and Friday of this week. Count by arrival date and the real load in the first seven days is 172 hours, more than a calendar week holds. A manager who plans from the tags would miss it. The optimizer gives a feasible, non-overlapping schedule and protects the fixed releases, but 168 hours is the skid running 24/7 with no maintenance or changeover. Leveling alone doesn't solve it. The right output is an escalation now: weekend coverage, a cross-skid transfer, or a conversation with the client."*

Then show the stakeholder communication draft (tab *3. QTA/SOW Comms & Matrix*).
> *"This is a draft notification for me to review. It is never sent automatically."*

**If you use the 5-minute tour (Stop 2):** the chart there is titled *AFTER Weekly-Bucket Leveling (Heijunka Weekly Aggregates)* and shows 136 h to 102 h (113.3% to 85.0%). It is the plan-tag view. The caption underneath gives the CP-SAT result (172.0 h to 168.0 h). Say so out loud, and never describe the 85% bar as "solved."

**Benchmark (100 synthetic scenarios), only if asked:** CP-SAT mean weighted tardiness **85,854.5** vs **98,701.5** for FIFO, about 13% lower; feasibility 100%; fixed commitments protected 100%. CP-SAT peak utilization (123.6%) is slightly *higher* than FIFO (120.4%): its value is feasibility and lower tardiness, not a lower peak.

> *Verify on screen before the interview: the Agent 4 draft wording, and that the tour and main tab labels read consistently.*

---

## Stop 2 (2:00-3:30): OOS Phase I telemetry gate

**Show:** Quality group, tab *1. OOS Telemetry Gate*. Use the toggle in the right inspector.

**Case A, "Empower Run #14 Hardware Fault":** backpressure **310.4 bar** against a **300 bar** limit; autosampler/chiller **17.8 °C** against a **10 °C** limit. Result: assignable laboratory cause, run voided, no Phase II escalation.

**Case B, "Clean Bench & Telemetry":** backpressure **215.2 bar**, within limit. Result: true OOS, re-test locked, Phase II investigation checklist generated.

**Say:**
> *"This is the part I'd own as local MSO for OOX. Under 21 CFR 211.192 and the FDA 2006 OOS guidance, you don't retest your way to a passing result. A re-test needs an assignable laboratory cause. This gate checks instrument telemetry for one. With a documented cause the run is invalidated; without one the result stands and the investigation moves to Phase II. In real life the evidence also comes from the CDS audit trail and the analyst interview. This shows the decision logic, not the whole investigation."*

> *Verify on screen: the OOS result value displayed. Use what the app shows.*

---

## Stop 3 (3:30-4:30): Cold-chain excursion and stability decision support

**Show:** Science group, tab *1. Kinetics & Dynamic OOT*. Demo lot **NS-2026-004**, **28 °C for 36 h**.

**What you will see:** ΔHMW **+0.0138 pp**, monomer about **99.36%**, shelf-life budget consumed about **0.59 months (17.95 days)**, excursion verdict **"no detectable change"**, OOT verdict **"inconclusive: model and data disagree; treat as potential OOT and investigate."**

**Say:**
> *"On the excursion alone the model says the impact is negligible, and it shows the integration instead of a Q10 rule of thumb. But the early-pull check on this lot disagrees with the model prior. The system doesn't resolve that quietly. It says inconclusive, treat as potential OOT, investigate. That's what I want from decision support: it surfaces the disagreement and a person decides. Disposition stays with QA."*

**Know the limits (all in `LIMITATIONS.md`):** on synthetic benchmarks the hybrid OOT detector had lower sensitivity at 6M (30%) than a simple linear same-lot method (76.7%), with a lower false-alarm rate; some kinetic parameters are prior-driven, not identifiable from SEC data; ICH Q5C requires real-time data.

---

## Close (4:30-5:00)

> *"I think in terms of commitments, conflicts and compliance: spot the conflict early, protect what's fixed, keep the investigation clean, communicate in time. What's missing for production is real LIMS, CDS and QMS integration, an e-signature that meets Part 11 requirements, and computer system validation. I'd be glad to talk through how I'd approach that."*

---

## Backup stops (only if asked or time remains)

All three sit in *QC Operations & Project Management*, tab *2. Portfolio & Operations*, as sub-tabs 9 to 11. Rehearse the click path: the sub-tab ribbon is long.

**Backup A: Contract Lab Data Review Tracker (sub-tab 9).** Shows outsourced analytical packages against SOW due dates, relative to a fixed as-of date (**2026-10-30**, shown in the demo data; point it out so the flags don't look stale).
- `PKG-CRO-2026-001`: **OVERDUE**, 2 open queries.
- `PKG-CRO-2026-002`: **DUE SOON (≤3d)**, due in 2 days. `PKG-CRO-2026-010`: due in exactly 3 days, same flag. A package due in 4 days shows ON TRACK.
- Try to Accept `PKG-CRO-2026-008` with an open query: *"Transition Refused: ... unresolved analytical query/queries."* Try Accept with under 10 characters of justification: refused.
- Accepted transitions are written to the hash-chained audit trail.

> *"Outsourced data review is a project-management problem as much as a technical one: turnaround against the SOW, open queries, and a rule that nothing is accepted with a query open or without a documented reason."*

**Backup B: Extended Roadblocks (sub-tab 10).** Categories: pending approval, late sample arrival, client dependency, vendor dependency. Escalation tier by days open (configurable): Team up to 3 days, QC Lead 4 to 7, Quality Head over 7.
- `BLK-EXT-102` (3 days) is Team; `BLK-EXT-104` (7 days) is QC Lead; `BLK-EXT-105` (8 days) is Quality Head.
- The escalation message is a draft for human review: *"Human project manager review and manual transmission required. Do not auto-send."*

**Backup C: Inspection-Readiness One-Pager (sub-tab 11).** Counts and ages only, no score or grade.
- 2 active OOS investigations (oldest `OOS-2026-004`, 48 days).
- 2 CAPAs past their effectiveness-check date (`CAPA-2026-011`, `CAPA-2026-014`).
- 5 stability pulls in the next 30 days, 2 outside the ±7 days pull tolerance (+10 d and +9 d).
- Markdown export carries the DRAFT disclaimer.

> *"If an inspector asked tomorrow for the state of my OOX and stability processes, this is the single page I'd want ready. It lists counts and ages. It does not claim we are ready."*

**Other backups:** SPC and CAPA (select a planted drift or shift; the default is in-control with 0 violations), FMEA, and the Part 11-style sign-off with the hash-chained audit trail. The lessons-learned search draws on 40 historical root-cause precedents (synthetic) and never determines root cause.

---

## Anticipated questions

| Question | Suggested answer |
| :--- | :--- |
| **Did you write this yourself?** | "I directed AI coding tools, and I reviewed, tested and can explain what they produced. The scenarios and the compliance logic are mine." (Be ready to walk through the OOS gate rule in plain language.) |
| **Does it use an LLM?** | "No. Zero LLM calls. The work package parser is rules and regex; schedules and kinetics are deterministic, so every number is reproducible." |
| **Why do two screens show different utilization (113% vs 143%)?** | "They count different things. The weekly view sums hours by the week each project is tagged with: 136 h. The CP-SAT view uses actual sample arrival dates over the first seven days: 172 h, because two projects tagged for next week arrive early. The arrival-based number is the honest one; the gap between them is exactly the planning risk." |
| **If week 2 has spare capacity, why does CP-SAT fill week 1 to 168 h?** | "The optimizer minimizes weighted tardiness while keeping the schedule feasible and protecting fixed commitments. Peak-load reduction is not its objective, and I report that openly: its peak is slightly above FIFO's. If I wanted flatter loads I'd add a peak or buffer penalty to the objective." *(Verify against `models/schedule_risk/optimize.py` before the interview.)* |
| **Is this validated / how would you validate it?** | "It is not validated. Production use would need a risk-based computer system validation approach (GAMP 5 thinking: requirements, testing, traceability), integration testing with LIMS/CDS, and QA ownership of the configuration." |
| **Does the e-signature meet Part 11?** | "No. It shows attributable, timestamped, hash-chained records. Real Part 11 needs unique credentials, signature manifestation, and an identity system that has itself been qualified." |
| **Why not just retest?** | "Retesting without an assignable cause is testing into compliance. The gate makes the correct path the default." |
| **How do you know the model isn't overfit?** | "I don't claim it isn't. On synthetic data it was calibrated on held-out lots, and I report where it underperforms: lower 6M sensitivity than a simple method. On real products I'd rely on real-time data and a statistician's review." |
| **Why a single shared SQLite connection?** | "It's a single-user demo. A production system would use a managed database with proper transaction handling." |
| **How does this relate to the role?** | "The role runs QC deliverables across clients and internal programs, owns OOX and stability processes, and escalates resource conflicts. Each stop is a small version of one of those." |
| **First 90 days?** | "Learn the current work package and escalation practice, the OOX and stability metrics, and the client commitments. Then pick one recurring pain point and build a lightweight tracker around it." |

---

## Pre-interview checklist

- [ ] Run the app end to end once on the presentation machine.
- [ ] Confirm every on-screen number against this script, especially Stop 1 wording, the Stop 2 OOS value, and the lab names in Backup A.
- [ ] Confirm the main capacity tab and the tour tell the same story (overload still present after leveling).
- [ ] Read `LIMITATIONS.md` once more.
- [ ] Be ready to explain the OOS rule and the audit hash chain without notes.
- [ ] Have the screenshots folder open as a fallback.