# QC Capacity & Utilization Metric Definitions
## cGMP-QC-CoPilot Decision Support

> **DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL**  
> *Demonstration prototype for operational decision support; not a validated system and not a compliant system.*

---

### Standardized Utilization Basis & Scope

To prevent ambiguity between nominal staffed hours, continuous physical skid capacity, and analytical queue dynamics, every utilization metric in **cGMP-QC-CoPilot** is explicitly defined according to the following matrix:

| Metric Name | Operating Basis | Temporal Scope | Aggregation Scope | Formula / Operational Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Nominal Staffed Skid Utilization** | Nominal staffed capacity:<br>• Waters UPLC: 120 h/wk<br>• Maurice (CE): 100 h/wk<br>• Bioassay: 90 h/wk | Weekly | Single Scenario (Demo W42/W43) or 100-Scenario Mean | $\frac{\sum \text{Scheduled Hours on Skid in Week } w}{\text{Nominal Staffed Hours}} \times 100\%$.<br>Target: $\le 85.0\%$ (LEAN flow buffer). Values $>100\%$ represent scheduled overtime/weekend operation. |
| **Physical 24/7 Facility Capacity** | Continuous 24/7 physical limit: 168.0 h/week per instrument | Weekly | Single Scenario (Demo W42/W43) or Scenario Peak | $\frac{\text{Scheduled Hours on Skid}}{168.0\text{ h}} \times 100\%$.<br>Physical limit. At $140.0\%$ of 120 h UPLC staffed capacity ($168.0\text{ h}$), **every single hour of a 24/7 week is booked**, leaving zero room for maintenance, column equilibration, or changeover. |
| **Daily Peak Skid Occupancy** | Operating hours per calendar day (24.0 h) | Daily (0–24 h) | Discrete-Event Simulation (DES) Minute-by-Minute | Peak concurrent instrument run-time divided by available skid count. Identifies within-week wave-loading spikes. |
| **100-Scenario Benchmark Mean** | Nominal staffed hours across 100 varied scenarios | Multi-Week Planning Horizon | 100 Contended Scenarios (50–80% Fixed Work) | Monte Carlo mean across 100 stochastic shop-floor instances evaluating optimizer vs. fair dispatch baselines (FIFO, EDD, Pinned Heijunka). |

---

### Operational Caveats & Capacity Gap

1. **Unleveled Schedule Infeasibility:** An "unleveled" schedule places samples at arrival date without collision avoidance, creating unphysical double-booking on instruments. It is strictly marked **INFEASIBLE** and reported only as a baseline demand indicator.
2. **24/7 Capacity Gap Warning:** When the CP-SAT optimizer levels the 10-program demo portfolio, UPLC utilization reaches **140.0% of nominal staffed hours (168.0 h)**. This load fills every hour of a 24/7 week with no changeover, maintenance, or shift gaps; **it is not an operationally workable schedule without added instrument capacity, shift overtime, or contract outsourcing.**
