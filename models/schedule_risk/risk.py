"""
cGMP-QC-CoPilot: Critical Path Method (CPM) & Monte Carlo Schedule Risk Engine.

Calculates:
- Deterministic CPM (Early Start, Early Finish, Late Start, Late Finish, Total Float)
- Critical Path identification (tasks with Float == 0)
- Monte Carlo Milestone Delivery Risk: P(on time), Criticality Index, Expected Delay
- What-If Sensitivity: Upstream harvest slip (0-5d), instrument outage, analyst absence, rush requests
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from models.schedule_risk.tasks import Task


@dataclass
class CPMNode:
    task_id: str
    duration_days: float
    early_start: float
    early_finish: float
    late_start: float
    late_finish: float
    total_float: float
    is_critical: bool


def compute_deterministic_cpm(
    tasks: List[Task],
    duration_map: Optional[Dict[str, float]] = None,
    arrival_map: Optional[Dict[str, float]] = None,
) -> Dict[str, CPMNode]:
    """
    Computes Early Start/Finish, Late Start/Finish, and Total Float on the task precedence network.
    """
    task_dict = {t.task_id: t for t in tasks}

    # 1. Forward Pass (Early Start & Early Finish)
    es: Dict[str, float] = {}
    ef: Dict[str, float] = {}

    # Topological order evaluation
    unprocessed = set(task_dict.keys())
    while unprocessed:
        progress = False
        for tid in list(unprocessed):
            t = task_dict[tid]
            # All predecessors must have EF computed
            if all(p in ef for p in t.predecessors):
                arrival = arrival_map.get(tid, t.sample_arrival_day) if arrival_map else t.sample_arrival_day
                dur_h = duration_map.get(tid, t.nominal_duration_hours) if duration_map else t.nominal_duration_hours
                dur_d = dur_h / 24.0

                pred_ef = max([ef[p] for p in t.predecessors], default=0.0)
                early_start = max(arrival, pred_ef)
                early_finish = early_start + dur_d

                es[tid] = round(early_start, 3)
                ef[tid] = round(early_finish, 3)
                unprocessed.remove(tid)
                progress = True
        if not progress and unprocessed:
            # Cycle fallback break
            for tid in unprocessed:
                t = task_dict[tid]
                es[tid] = t.sample_arrival_day
                ef[tid] = t.sample_arrival_day + (t.nominal_duration_hours / 24.0)
            break

    # 2. Backward Pass (Late Start & Late Finish)
    # Project end is maximum early finish or specified due dates
    max_project_ef = max(ef.values(), default=14.0)

    # Successor map
    succ_map: Dict[str, List[str]] = {t_id: [] for t_id in task_dict}
    for t in tasks:
        for p in t.predecessors:
            if p in succ_map:
                succ_map[p].append(t.task_id)

    ls: Dict[str, float] = {}
    lf: Dict[str, float] = {}

    unprocessed_back = set(task_dict.keys())
    while unprocessed_back:
        progress = False
        for tid in list(unprocessed_back):
            t = task_dict[tid]
            successors = succ_map[tid]
            if all(s in ls for s in successors):
                dur_h = duration_map.get(tid, t.nominal_duration_hours) if duration_map else t.nominal_duration_hours
                dur_d = dur_h / 24.0

                if successors:
                    late_finish = min(ls[s] for s in successors)
                else:
                    late_finish = max(ef[tid], t.due_day)

                late_start = late_finish - dur_d

                lf[tid] = round(late_finish, 3)
                ls[tid] = round(late_start, 3)
                unprocessed_back.remove(tid)
                progress = True
        if not progress and unprocessed_back:
            for tid in unprocessed_back:
                lf[tid] = max_project_ef
                ls[tid] = max_project_ef - (task_dict[tid].nominal_duration_hours / 24.0)
            break

    cpm_nodes: Dict[str, CPMNode] = {}
    for tid, t in task_dict.items():
        dur_h = duration_map.get(tid, t.nominal_duration_hours) if duration_map else t.nominal_duration_hours
        dur_d = dur_h / 24.0
        float_d = max(0.0, round(ls[tid] - es[tid], 3))
        is_crit = float_d < 0.05  # within numerical tolerance

        cpm_nodes[tid] = CPMNode(
            task_id=tid,
            duration_days=round(dur_d, 3),
            early_start=es[tid],
            early_finish=ef[tid],
            late_start=ls[tid],
            late_finish=lf[tid],
            total_float=float_d,
            is_critical=is_crit,
        )

    return cpm_nodes


@dataclass
class MilestoneRiskRecord:
    task_id: str
    project_id: str
    program_name: str
    due_day: float
    mean_completion_day: float
    p_on_time: float
    rag_status: str
    criticality_index: float
    expected_delay_days: float
    total_float_days: float = 0.0
    evaluation_source: str = "DES (SimPy)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "program_name": self.program_name,
            "due_day": self.due_day,
            "mean_completion_day": round(self.mean_completion_day, 2),
            "p_on_time_pct": round(self.p_on_time * 100.0, 1),
            "rag_status": self.rag_status,
            "criticality_index_pct": round(self.criticality_index * 100.0, 1),
            "expected_delay_days": round(self.expected_delay_days, 2),
            "total_float_days": round(self.total_float_days, 2),
            "evaluation_source": self.evaluation_source,
        }


def run_monte_carlo_schedule_risk(
    tasks: List[Task],
    instruments: Optional[Dict[str, Any]] = None,
    n_simulations: int = 500,
    seed: int = 42,
    usp_harvest_delay_days: float = 0.0,
    instrument_outage_mult: float = 1.0,
    analyst_absence_mult: float = 1.0,
    inject_rush_task: bool = False,
    rush_tasks: Optional[List[Task]] = None,
    rag_green_thresh: float = 0.90,
    rag_amber_thresh: float = 0.70,
) -> Dict[str, MilestoneRiskRecord]:
    """
    Monte Carlo schedule risk evaluation combining SimPy Discrete-Event Simulation (DES)
    and Critical Path Method (CPM).
    - DES is the primary source of completion distributions, P(on time), and delay metrics,
      modeling shared instrument queueing and resource contention.
    - CPM provides topological total float and criticality index.
    Supports what-if sensitivity:
    - usp_harvest_delay_days: Upstream bioreactor/harvest slip (0-5d)
    - instrument_outage_mult: Skid breakdown / qualification delay multiplier (>= 1.0)
    - analyst_absence_mult: Unplanned analyst shortage multiplier (>= 1.0)
    - inject_rush_task / rush_tasks: Urgent commercial/clinical rush lot injection
    """
    import simpy
    from models.schedule_risk.tasks import FlexibilityClass, InstrumentDefinition

    rng = np.random.default_rng(seed)
    eval_tasks = list(tasks)

    if inject_rush_task:
        rush_t = Task.create(
            task_id="RUSH-REQ-001",
            project_id="BI-RUSH-EMERGENCY",
            program_name="Spesolimab-Urgent-Release",
            test_panel="SEC_HPLC",
            instrument_id="UPLC_SEC_CEX",
            sample_arrival_day=1.0,
            due_day=3.0,
            flexibility_class=FlexibilityClass.FIXED_COMMERCIAL,
            window_days=0.0,
            nominal_duration_hours=18.0,
            penalty_weight=100000.0,
        )
        eval_tasks.append(rush_t)

    if rush_tasks:
        eval_tasks.extend(rush_tasks)

    # Establish instrument dictionary for DES resources
    inst_dict: Dict[str, InstrumentDefinition] = {}
    if instruments:
        inst_dict = {
            k: (v if isinstance(v, InstrumentDefinition) else InstrumentDefinition(id=k, name=k))
            for k, v in instruments.items()
        }
    else:
        # Default standard facility skids
        known_skids = {
            "UPLC_SEC_CEX": InstrumentDefinition(id="UPLC_SEC_CEX", name="Waters Acquity UPLC", weekly_max_hours=120.0),
            "CE_icIEF": InstrumentDefinition(id="CE_icIEF", name="ProteinSimple Maurice", weekly_max_hours=100.0),
            "Cell_Bioassay": InstrumentDefinition(id="Cell_Bioassay", name="BSL-2 Hamilton Star", weekly_max_hours=90.0),
        }
        for t in eval_tasks:
            inst_id = t.instrument_id
            if inst_id in known_skids:
                inst_dict[inst_id] = known_skids[inst_id]
            elif inst_id not in inst_dict:
                inst_dict[inst_id] = InstrumentDefinition(id=inst_id, name=inst_id)

    # 1. Deterministic CPM for baseline topological total float
    cpm_baseline = compute_deterministic_cpm(eval_tasks)

    # 2. Monte Carlo Replications via DES and CPM
    des_completions: Dict[str, List[float]] = {t.task_id: [] for t in eval_tasks}
    critical_counts: Dict[str, int] = {t.task_id: 0 for t in eval_tasks}

    for sim_idx in range(n_simulations):
        # Sample stochastic parameters for this replication
        sim_arrivals_h: Dict[str, float] = {}
        sim_durations_h: Dict[str, float] = {}

        for t in eval_tasks:
            harvest_slip = usp_harvest_delay_days if t.flexibility_class.is_fixed() else (usp_harvest_delay_days * 0.5)
            stoch_slip = rng.normal(0.2, 0.4)
            arr_d = max(0.0, t.sample_arrival_day + harvest_slip + stoch_slip)
            sim_arrivals_h[t.task_id] = arr_d * 24.0

            stoch_dur = max(
                2.0,
                t.nominal_duration_hours * instrument_outage_mult * analyst_absence_mult + rng.normal(0.0, 1.2),
            )
            sim_durations_h[t.task_id] = stoch_dur

        # CPM evaluation for precedence criticality
        sim_arrivals_d = {k: v / 24.0 for k, v in sim_arrivals_h.items()}
        cpm_rep = compute_deterministic_cpm(eval_tasks, duration_map=sim_durations_h, arrival_map=sim_arrivals_d)
        for tid, node in cpm_rep.items():
            if node.is_critical:
                critical_counts[tid] += 1

        # SimPy DES evaluation with shared resource contention & precedence constraints
        env = simpy.Environment()
        resources = {k: simpy.Resource(env, capacity=1) for k in inst_dict}
        done_events = {t.task_id: env.event() for t in eval_tasks}

        def simpy_task(task_obj: Task):
            # Precedence dependencies: wait for predecessors to finish
            if task_obj.predecessors:
                pred_evs = [done_events[p] for p in task_obj.predecessors if p in done_events]
                if pred_evs:
                    yield simpy.AllOf(env, pred_evs)

            # Wait for sample arrival
            arrival_time = sim_arrivals_h[task_obj.task_id]
            if env.now < arrival_time:
                yield env.timeout(arrival_time - env.now)

            # Request instrument skid
            inst_key = task_obj.instrument_id if task_obj.instrument_id in resources else list(resources.keys())[0]
            with resources[inst_key].request() as req:
                yield req
                dur_h = sim_durations_h[task_obj.task_id]
                yield env.timeout(dur_h)

            des_completions[task_obj.task_id].append(env.now / 24.0)
            if not done_events[task_obj.task_id].triggered:
                done_events[task_obj.task_id].succeed()

        for t in eval_tasks:
            env.process(simpy_task(t))

        env.run(until=14.0 * 24.0 * 2.0)  # horizon 28 days

        # Fallback if uncompleted in horizon
        for t in eval_tasks:
            if len(des_completions[t.task_id]) <= sim_idx:
                des_completions[t.task_id].append(28.0)

    # 3. Assemble MilestoneRiskRecord with DES as primary P(on time) source
    records: Dict[str, MilestoneRiskRecord] = {}
    for t in eval_tasks:
        comp_arr = np.array(des_completions[t.task_id])
        on_time = float(np.mean(comp_arr <= t.due_day))
        crit_index = critical_counts[t.task_id] / max(1, n_simulations)
        delays = np.maximum(0.0, comp_arr - t.due_day)
        exp_delay = float(np.mean(delays))

        if on_time >= rag_green_thresh:
            rag = "GREEN (>=90%)"
        elif on_time >= rag_amber_thresh:
            rag = "AMBER (70-90%)"
        else:
            rag = "RED (<70%)"

        baseline_float = cpm_baseline[t.task_id].total_float if t.task_id in cpm_baseline else 0.0

        records[t.task_id] = MilestoneRiskRecord(
            task_id=t.task_id,
            project_id=t.project_id,
            program_name=t.program_name,
            due_day=t.due_day,
            mean_completion_day=float(np.mean(comp_arr)),
            p_on_time=on_time,
            rag_status=rag,
            criticality_index=float(crit_index),
            expected_delay_days=exp_delay,
            total_float_days=baseline_float,
            evaluation_source="DES (SimPy)",
        )

    return records
