# LIMITATIONS & SCIENTIFIC BOUNDARIES
## cGMP QC Decision-Support Kinetic Modeling Architecture

This document records the explicit scientific boundaries, non-identifiable parameters, and regulatory limitations of the hybrid Lumry-Eyring + 25-member ensemble model and the decision-support multi-agent copilot.

---

### 1. Regulatory Status & Non-GMP Classification
- **Decision-Support Status:** All four agents operate strictly as **Non-GMP Decision-Support Systems**. They provide deterministic modeling, risk triage, and draft technical packages for Quality Review.
- **Human-in-the-Loop (HITL) Gate:** Outputs are stamped `DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL`. No system outputs may be imported into official GMP records (e.g., TrackWise Digital) without independent review and sign-off.
- **Part 11 E-Signature Disclaimer:** The electronic signature gate in the UI and workflow scripts is a **Part 11-style HITL sign-off (demonstration only; not a compliant e-signature system)**. It illustrates ALCOA+ contemporaneous metadata capture and SHA-256 cryptographic hashing but does not constitute a validated 21 CFR Part 11 electronic signature platform.

---

### 2. Prior-Driven vs. Identifiable Thermodynamic Parameters
From standard SEC-UPLC monomer/HMW percentage trajectories alone, only a subset of the Lumry-Eyring thermodynamic parameters are mathematically identifiable:

| Parameter | Identifiability Status | Scientific Rationale |
| :--- | :--- | :--- |
| **$\Delta G^\ddagger_{\text{eff,ref}}$ (at 5 °C)** | **Identifiable** | Directly determines the effective aggregation rate $k_{\text{eff}}(5^\circ\text{C})$ at baseline storage temperature. |
| **$k_{\text{eff}}(5^\circ\text{C}), k_{\text{eff}}(25^\circ\text{C}), k_{\text{eff}}(40^\circ\text{C})$** | **Identifiable** | Measured directly by slope progression across the 3 stability storage temperatures. |
| **Apparent $E_a$ ($5\text{--}40^\circ\text{C}$)** | **Identifiable** | Derived directly from Arrhenius/Eyring slope of observed $\ln k_{\text{eff}}$ vs $1/T$. |
| **$\Delta H^\ddagger_{\text{agg}}$** | **Prior-Driven** | Strongly confounded with unfolding enthalpy $\Delta H_u$; reported for diagnostic transparency and never gated. |
| **$K_u(5^\circ\text{C})$ and $f_U(5^\circ\text{C})$** | **Prior-Driven** | Reversible unfolding equilibrium cannot be isolated by SEC-UPLC (which measures irreversible soluble aggregates after rapid cooling). Peak $f_U(T)$ relaxes in $<1\text{ min}$ at $k_{-1} = 100\text{ h}^{-1}$. |
| **$\Delta H_u$** | **Prior-Driven** | Unfolding enthalpy cannot be uncoupled from aggregation activation enthalpy without orthogonal biophysical data (e.g., DSC, nanoDSF). |
| **$\Delta S^\ddagger_{\text{agg}} = -241\text{ J/mol/K}$** | **Fixed SME Reference** | Nominal aggregation activation entropy chosen for pre-exponential consistency; SME review noted. |

> [!NOTE]
> Latent parameter calibration (evaluating whether true parameters fall inside calibrated confidence intervals) is possible **only in synthetic benchmarks** where true physical values are known. On real production lots, latent parameters are unobservable; only observation prediction intervals are empirically verifiable.

---

### 3. Kinetic Model Assumptions vs. Stiff Physical Truth
To avoid an **inverse crime** in synthetic validation, the synthetic benchmark generator differs structurally from the fitted hybrid model:
1. **Pre-Equilibrium vs. Stiff Radau ODE:** The fitted hybrid model uses a 2-state pre-equilibrium approximation ($k_{-1} \gg k_2$) integrated via vectorized 4th-order Runge-Kutta (RK4) or closed-form analytical equations. The synthetic benchmark is generated using a full **3-state stiff Radau IIA integrator** at finite refolding rate $k_{-1}$.
2. **Unmodeled Polysorbate-80 (PS80) Depletion:** The benchmark generator incorporates late-time surfactant degradation on 35% of lots (`mismatched` group), where PS80 hydrolytic/oxidative loss accelerates aggregation after 3–6 months. The fitted hybrid model intentionally contains **no PS80 state or time-varying surfactant term**.
   - *Consequence:* On mismatched lots, the hybrid model cannot anticipate the late inflection from initial covariates alone.
   - *Benchmark Result:* On mismatched validation lots, hybrid RMSE is slightly higher than linear ridge (`0.415 pp` vs `0.406 pp`), reflecting this shared structural bias.

---

### 4. Early-Pull Lot Adaptation & Dynamic OOT Boundaries
- **Prior-Data Conflict & Demo Lot NS-2026-004:** When adapting a lot's parameters from early stability pulls (`0, 1, 3M`), if the lot has stressed formulation covariates (e.g., Demo Lot `NS-2026-004` at pH 5.8, 100 mg/mL, 0.012% PS80), the covariate prior mean ($\approx 2.82\%$ at 6M 25°C) pulls strongly upward.
- With the population-calibrated prior scale ($\times 1.0$), penalized least squares overpredicts the 3M pull ($\sim 1.07\%$ model vs $0.88\%$ observed, $-2.67\text{ SEC-SD}$), shifting $\Delta G^\ddagger_{\text{eff}}$ by $+1.626\text{ kJ/mol}$ ($>2.5\sigma$ calibrated lot-effect SD threshold of $0.198\text{ kJ/mol}$). Because the prior-data-conflict flag is true and early pull residuals exceed $2.5\text{ SEC-SD}$, the system classifies the lot as **`inconclusive: model and data disagree; treat as potential OOT and investigate`**, issuing a draft recommendation for QA review.
- **Flat Prior-Scale Grid on Calibration Lots:** Grid search of the prior-scale multiplier over $\times 0.5 \dots \times 4.0$ on the 60 calibration lots yields a nearly flat forecast RMSE profile ($0.345\text{--}0.348\text{ pp}$), demonstrating that held-out forecast error on typical lots cannot meaningfully discriminate between prior scales; scale $\times 1.0$ is maintained as the population benchmark.
- **Operating Characteristics (Lower False-Alarm Rate, Lower or Comparable Sensitivity):** In this synthetic setup, the hybrid model achieves a **lower false-alarm rate in this synthetic setup; lower or comparable sensitivity**. Hybrid OOT detection has **lower sensitivity at 6M than classic methods** ($9/30$ = 30.0% vs $23/30$ = 76.7% for classic linear $+ 3\times 0.035$, and $18/30$ = 60.0% for classic linear regression 95% PI), but classic methods suffer severe false alarms on matched lots ($14/70$ = 20.0% for linear $+ 3\times 0.035$; $11/70$ = 15.7% for linear regression PI95; $9/70$ = 12.9% for linear $+ 3\times 0.050$), whereas the recalibrated hybrid achieves **zero false-alarm alerts on matched lots ($0/70$ = 0.0%, 95% CI [0.0%, 5.1%]) in this evaluation cohort**.
  - *Comparator Scope:* Note that the classic comparator evaluated is a same-lot linear extrapolation from 0/1/3M pulls. The common commercial practice of historical multi-lot regression prediction intervals (e.g. JMP-style stability trending across prior release batches) was not tested as the primary baseline; we make no claim of superiority over JMP-style trending across historical lots.
- **Pull Schedule Changes (Phase 2):** Matched pulls at 6/9/12M total $126$ ($70$ at 6M + $28$ at 9M + $28$ at 12M) because realistic pull schedules increased the extended 25°C lot cohort from 20% ($14$ matched lots) to 40% ($28$ matched lots). Similarly, mismatched 9M/12M counts increased from $n=6$ to $n=12$.

---

### 5. Shelf-Life Regulatory Constraints & Extrapolation Guard
- **ICH Q5C Primary Authority:** For therapeutic biologics, real-time, real-temperature stability data is required. Accelerated data can provide supporting evidence but cannot replace real-time testing.
- **Regulatory Extrapolation Guard:** Enforces strict capping at $\text{Observed} + 6\text{M}$ ($30.0\text{ months}$ max for a 24-month study) per ICH Q1E.
- **Late-Pull Residual / Model-Adequacy Check:** To protect against unmodeled late-stage acceleration, the copilot includes an automated late-pull residual check (evaluating whether residuals at $\ge 18\text{M}$ exceed $2.5\times \sigma_{\text{SEC}}$ or trend upward). If triggered, the output is downgraded to **`model adequacy not demonstrated`**.

---

### 6. Schedule-Risk & Capacity Optimization Boundaries (WP1)
- **Synthetic Operational History & Temporal Partition:** The 104-week capacity log is synthetically generated (`data/synthetic_capacity/`). It incorporates latent upstream bioreactor harvest shocks, holiday staffing slowdowns, and multi-skid downtime bursts that are not exposed to downstream ML feature matrices (Guardrails G1, G4).
  - **Training Split:** Weeks 1 – 70 ($N = 427$ task records).
  - **Conformal Calibration / Validation Split:** Weeks 71 – 86 ($N = 96$ task records).
  - **Held-Out Test Split:** Weeks 87 – 104 ($N = 105$ task records, strictly held-out).
- **Rolling-Origin Conformalized Quantile Regression (CQR) & Empirical Coverage:**
  - Using finite-sample Conformalized Quantile Regression (Romano, Patterson, Candès, 2019) with a 16-week rolling window, nonconformity scores $E_i = \max(q_{0.1}(x_i) - y_i, y_i - q_{0.9}(x_i))$ adapt dynamically to shop-floor distribution drift:
  - **Sample-Arrival Slippage Forecaster:**
    - Rolling-origin calibrated test coverage: **85.7%** (mean width 4.096 days; mean rolling margin $Q = 0.629$d).
    - Point MAE: 1.152 days (95% CI [0.981, 1.307]) vs Naive Mean 1.110 days (95% CI [0.932, 1.295]), an error reduction of -3.8% (honest reporting: upstream harvest slip cannot be accurately predicted from historical lot attributes alone without live bioreactor telemetry).
    - Pinball Loss (calibrated quantiles): $q_{10} = 0.187$ (95% CI [0.159, 0.216]), $q_{50} = 0.576$ (95% CI [0.491, 0.654]), $q_{90} = 0.270$ (95% CI [0.240, 0.298]).
  - **Analyst Turnaround Duration Forecaster:**
    - Rolling-origin calibrated test coverage: **81.0%** (mean width 5.341 hours; mean rolling margin $Q = 0.614$h). Reconciles earlier 99.0% static test coverage caused by over-expansion from turbulent validation weeks.
    - Point MAE: 1.376 hours (95% CI [1.179, 1.580]) vs Naive Mean 1.996 hours (95% CI [1.754, 2.239]), an error reduction of +31.1%.
    - Pinball Loss (calibrated quantiles): $q_{10} = 0.322$ (95% CI [0.279, 0.373]), $q_{50} = 0.688$ (95% CI [0.589, 0.790]), $q_{90} = 0.368$ (95% CI [0.321, 0.419]).
- **Schedule-Level Calibration & Shared Assumptions (Independent Stress Simulator):**
  - Held-out actual outcomes ($Y_k \in \{0, 1\}$) were generated independently via an independent stress simulator across 2,066 tasks in 100 held-out test scenarios under Weibull wear-out machine breakdowns ($k=2.2, \lambda=90\text{h}$), preemptive scheduling, and analyst absence shocks.
  - **Shared Assumptions Between Risk Model and Stress Simulator:**
    - *Shared Failure Model Form:* Both systems model QC operations as discrete-event queues with single-instrument resource constraints, FIFO/priority queuing, and state-based machine interruption.
    - *Shared Parameter Families:* Both assume parametric or semi-parametric duration and delay distributions (lognormal durations, Weibull hazard processes, quantile GBDT representations); renewal occurs upon service/repair completion.
    - *Shared Feature Definitions:* Both rely on identical definitions for historical 4-week moving averages, priority tiers (Tier 1A/1B/2/3), and sample types.
  - **Scientific Boundary:** **Agreement with an independent simulator is necessary but not sufficient evidence of real-world validity.** Real-world analytical laboratories face complex sociotechnical failure modes (untracked reagent lot recalls, column degradation across repeated injections, sudden facility utility excursions, analyst qualification lapses) not modeled in either simulation.
  - Base late rate: 37.9% ($\text{Brier}_{\text{base}} = 0.2354$).
  - Naive deterministic planner: Brier score 0.3059 (95% CI [0.2858, 0.3265]), Brier skill score -0.2997.
  - Simple Logistic Regression baseline: Brier score 0.1308 (95% CI [0.1231, 0.1396]), Brier skill score 0.4443, ROC AUC 0.891.
  - Raw Risk Model: Brier score 0.1167 (95% CI [0.1046, 0.1290]), Brier skill score 0.5041, ROC AUC 0.910 (95% CI [0.894, 0.924]).
  - Calibrated Risk Model: Brier score **0.1112** (95% scenario-level cluster bootstrap CI [0.1006, 0.1218]).
  - Brier Skill Score: **+0.5277** (95% CI [0.4857, 0.5673], 52.8% skill vs base-rate).
  - Calibrated ROC AUC: **0.907** (95% CI [0.891, 0.921]).
  - Honest finding: Point Brier score improves from 0.1167 to 0.1112, but 95% CIs overlap with raw risk model (calibration gain is modest on held-out test data).
  - Populated 5-bin reliability table: $[0.0, 0.2): 942, [0.2, 0.4): 373, [0.4, 0.6): 194, [0.6, 0.8): 152, [0.8, 1.0]: 405$.
- **Fair Baseline Optimizer Comparison (100 Seeded Scenarios with 50–85% Fixed Contention):**
  - **Standardized Peak Weekly Skid Utilization:** Defined as $\max_{w \in \{\text{W1}, \text{W2}\}} (\text{hours on skid in week } w \div \text{weekly capacity}) \times 100\%$. Capacity basis: nominal staffed weekly hours (120h UPLC, 100h Maurice, 90h Bioassay; fictional example). Physical 24/7 calendar capacity is 168h/week. Utilization exceeding 100% in a feasible schedule represents scheduled overtime/weekend load beyond nominal staffed capacity.
  - **Unleveled Baseline:** Status: `INFEASIBLE (double-books instruments; concurrent tasks overlap on same skid; reported for demand reference only)`. Peak utilization: 120.4%, mean weighted tardiness: N/A (physically impossible schedule; concurrent overlapping runs artificially suppress apparent tardiness). Moves: 0.
  - **FIFO Pinned Baseline:** Status: `FEASIBLE`. Dispatches flexible tasks in pure arrival order at earliest non-overlapping slot. Feasibility 100%, mean weighted tardiness **98,701.5** $\$ \cdot \text{days}$, mean moves per scenario **8.6**, fixed commitments protected **100.0%**, mean peak weekly utilization **120.4%**.
  - **CP-SAT Optimizer:** Feasibility 100%, mean weighted tardiness **85,854.5** $\$ \cdot \text{days}$, mean moves per scenario **7.6**, fixed commitments protected **100.0%**, mean peak weekly utilization **123.6%**.
  - **Pinned Heijunka Baseline:** Feasibility 100%, mean weighted tardiness **98,701.5** $\$ \cdot \text{days}$, mean moves per scenario **8.6**, fixed commitments protected **100.0%**, mean peak weekly utilization **120.4%**.
  - **Earliest-Due-Date (EDD) Baseline:** Feasibility 100%, mean weighted tardiness **86,734.2** $\$ \cdot \text{days}$, mean moves per scenario **8.6**, fixed commitments protected **100.0%**, mean peak weekly utilization **120.4%**.
  - **Peak Utilization & Delivered Value Analysis:**
    - *Why peak utilization is identical (120.4%) for Unleveled/Heijunka/EDD and higher for CP-SAT (123.6%):* In a fixed 2-week planning horizon with heavy bottleneck workloads, Heijunka and EDD dispatch place tasks sequentially or push them past due dates. CP-SAT enforces non-overlapping intervals while optimizing a multi-objective cost that heavily penalizes tardiness on high-value lots, concentrating work into the bottleneck week when doing so prevents customer release delays.
    - *Delivered Value Proposition:* The optimizer's delivered value is **mathematical schedule feasibility (zero double-booked instrument hours)**, **explainable schedule moves with full audit rationales**, and **provable mathematical infeasibility diagnostics**, **NOT peak capacity reduction**.
  - **Demo Portfolio Physical Capacity Finding:**
    - For the 10-project demo portfolio (30 tasks), unleveled UPLC peak load is **143.3% of 120h nominal staffed capacity = 172.0 h/week**.
    - This **strictly exceeds the 168.0 h physical 24/7 calendar capacity of a single instrument skid**.
    - CP-SAT levels this load to **140.0% of 120h = 168.0 h/week**, packing the instrument continuously for all 168 hours of the 7-day week without overlap.
    - Because 140.0% equals a full 24/7 week, **capacity gap: this load fills every hour of a 24/7 week with no changeover, maintenance, or shift gaps; it is not an operationally workable schedule.** The demo portfolio cannot fit in one week on UPLC under the 85% LEAN buffer ceiling (102h) without added instrument capacity, weekend overtime, or rescheduling tasks across adjacent weeks.
- **Sample-Arrival CQR Coverage Subgroup Limitation:**
  - While overall held-out prediction interval coverage is 85.7% (95% CI: 77.5%–91.8%), **sample-arrival interval coverage by subgroup is unverified for small groups (Tier 1B: 7/12 = 58.3%) and tier and stream are confounded in the synthetic data.**
- **Project Management Feature Scope & Roadblock Categories:**
  - The Roadblock Radar evaluates batch slip and skid contention, as well as extended operational categories built under rule-based synthetic logic: `pending approval`, `late sample arrival`, `client dependency`, and `vendor dependency` with configurable escalation thresholds. Dynamic real-time staffing optimization remains out of scope.
  - In the Work Package Generator, the architectural protection against prompt injection is that **the protection is that the rule engine never executes request text**, parses it into executable code, or delegates free request text to an interpreter.
- **Multi-Criteria Prioritization Sensitivity (Multi-Concentration Dirichlet):**
  - Sensitivity evaluated across multiple Dirichlet concentrations $\alpha_0 \in [10, 30, 100]$ (500 draws each).
  - In the dominant scenario (`PRJ-DOM-01` strictly dominating all 5 criteria simultaneously), rank stability is trivially 100.0% across all concentration levels.
  - In contrasting conflicting scenarios (`PRJ-CONF-A/B/C/D` with trade-offs between penalty, tier, float, and critical path), rank stability scores are **0.720** (72.0%) at $\alpha_0 = 10$, **0.868** (86.8%) at $\alpha_0 = 30$, and **0.986** (98.6%) at $\alpha_0 = 100$.
- **Real-World Deployment Requirements:** Production deployment requires direct bidirectional integration with LIMS (e.g. LabWare/LabVantage), CDS (Waters Empower 3), ERP (SAP QM), an independent Part 11 e-signature system, formal computer system qualification under site CSV policy (not yet performed), and human QA electronic review.

---

### 7. Quality Intelligence, Statistical Process Control (SPC) & CAPA Boundaries (WP3)
- **SPC Distributional Assumptions:** Shewhart I-MR and X-bar/R control charts assume independent, identically distributed normal observations. In biological quality control, serial autocorrelation (e.g. from common column batches or reagent lots) can inflate the effective false alarm rate.
- **Shewhart vs. EWMA/CUSUM Sensitivity Trade-offs:** As demonstrated in the empirical Monte Carlo benchmark (
esults/spc_performance.json), Shewhart Rule 1 exhibits low false-alarm frequency under in-control conditions, but poor sensitivity to small sustained shifts (detection rate < 50% within 100 runs at $\delta = 0.5\sigma$, ARL $pprox 76.2$ steps), whereas EWMA and CUSUM detect small persistent shifts rapidly ( pprox 25.7$ to .5$ steps at .5\sigma$, and $<10$ steps at .0\sigma$). Conversely, Shewhart Rule 1 is superior for immediate gross single-point excursions ($\ge 2.0\sigma$) without memory lag.
- **Process Capability ( / C_{pk}$) Normal-Theory Bounds:** $ and {pk}$ metrics and their Chou et al. 95% confidence intervals assume normality. The copilot executes a Shapiro-Wilk test and flags explicit warnings when non-normality ( < 0.05$) or small sample size ( < 30$) is detected. Normal-theory capability estimates must not be used on skewed attribute data.
- **Historical Precedent Search:** The Lessons-Learned Knowledge Base search operates via TF-IDF token matching across 40 synthetic historical deviation cases. It surfaces relevant precedents for decision support and investigative hypotheses; it **never determines definitive root cause**, which legally and procedurally remains the responsibility of the human investigator and Quality Assurance.
- **CAPA Lifecycle Governance:** The CAPA state machine enforces that closure cannot occur without documented effectiveness verification and human QA justification behind a review gate.
