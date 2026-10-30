"""
cGMP-QC-CoPilot: Independent Stress Simulator (Physical Stress Test Engine).

Implements Guardrails G4 (No inverse crime) and G5 (Honest Evaluation):
- Structurally disparate from the internal Monte Carlo / DES risk models.
- Uses Weibull hazard aging/wear-out breakdown processes (vs exponential).
- Simulates true shop-floor preemption: urgent fixed commercial lots preempt flexible runs.
- Includes unplanned analyst absence shocks, column fouling, and re-equilibration delays.
- Operates under independent seed spaces to generate unbiased test-period ground-truth outcomes.

Note on Limitations:
Agreement with an independent simulator is necessary but not sufficient evidence of real-world validity.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import simpy

from models.schedule_risk.tasks import FlexibilityClass, InstrumentDefinition, Task


@dataclass
class StressSimulationOutcome:
    task_id: str
    project_id: str
    scheduled_due_day: float
    actual_finish_day: float
    is_late: bool
    queue_hours: float
    was_preempted: bool
    breakdown_delays_hours: float


# Backwards-compatibility alias
RealitySimulationOutcome = StressSimulationOutcome


def simulate_stress_scenario(
    tasks: List[Task],
    instruments: Dict[str, InstrumentDefinition],
    seed: int = 9001,
    horizon_days: float = 21.0,
) -> Dict[str, StressSimulationOutcome]:
    """
    Executes a high-fidelity independent stress simulation of a commercial QC laboratory.
    Incorporates unmodeled effects:
    1. Weibull wear-out machine breakdowns (shape k=2.2, scale lambda=90h)
    2. Preemptive resource allocation for zero-float Tier 1A / fixed commercial lots
    3. Unplanned analyst call-out shocks
    4. Column fouling & re-equilibration penalties upon preemption
    """
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    horizon_hours = horizon_days * 24.0

    # SimPy PreemptiveResource for each instrument skid
    resources = {inst_id: simpy.PreemptiveResource(env, capacity=1) for inst_id in instruments}

    completion_times: Dict[str, float] = {}
    queue_records: Dict[str, float] = {}
    preempted_tasks: set = set()
    breakdown_delays: Dict[str, float] = {inst_id: 0.0 for inst_id in instruments}

    # 1. Independent Weibull Machine Failure Process per Skid
    def skid_weibull_failure_process(inst_id: str):
        res = resources[inst_id]
        while True:
            # Weibull wear-out time to failure (k=2.2, lambda=90.0h)
            ttf = rng.weibull(2.2) * 90.0
            yield env.timeout(ttf)
            if env.now >= horizon_hours:
                break
            # Breakdown repair time: log-normal (mu=1.8, sigma=0.5 -> ~6 to 12 hours)
            repair_h = float(rng.lognormal(mean=1.8, sigma=0.5))
            breakdown_delays[inst_id] += repair_h
            # Request instrument with highest possible priority (-100) to halt runs
            try:
                with res.request(priority=-100, preempt=True) as req:
                    yield req
                    yield env.timeout(repair_h)
            except simpy.Interrupt:
                pass

    for inst_id in instruments:
        env.process(skid_weibull_failure_process(inst_id))

    # 2. Analyst Shift Shock Process (random morning call-outs)
    analyst_shock_days = set(rng.choice(range(1, int(horizon_days)), size=max(1, int(horizon_days // 5)), replace=False))

    # 3. Task Execution Processes
    def task_process(task: Task):
        # Unmodeled upstream buffer slip (log-normal shock)
        if rng.random() < 0.25:
            latent_usp_slip = float(rng.lognormal(mean=0.3, sigma=0.6))
        else:
            latent_usp_slip = float(rng.normal(0.1, 0.2))
        effective_arr_h = max(0.0, (task.sample_arrival_day + max(0.0, latent_usp_slip)) * 24.0)

        # Wait for physical sample arrival
        if env.now < effective_arr_h:
            yield env.timeout(effective_arr_h - env.now)

        # Wait for predecessors to complete in reality
        for pred_id in task.predecessors:
            while pred_id not in completion_times:
                yield env.timeout(0.5)

        # Preemption priority: Tier 1A / Fixed lots get priority -10; Flexible gets 0
        is_urgent = task.flexibility_class.is_fixed() or task.priority_tier == "Tier 1A"
        req_priority = -10 if is_urgent else 0

        inst_res = resources.get(task.instrument_id)
        if inst_res is None:
            inst_res = list(resources.values())[0]

        start_req_time = env.now
        rem_dur = task.nominal_duration_hours

        # Unplanned analyst shortage duration penalty
        arr_day_int = int(env.now // 24)
        if arr_day_int in analyst_shock_days:
            rem_dur *= 1.25  # 25% duration stretch due to cross-coverage slowdown

        was_interrupted = False

        while rem_dur > 0:
            with inst_res.request(priority=req_priority, preempt=is_urgent) as req:
                yield req
                queue_records[task.task_id] = queue_records.get(task.task_id, 0.0) + (env.now - start_req_time)
                exec_start = env.now
                try:
                    yield env.timeout(rem_dur)
                    rem_dur = 0
                except simpy.Interrupt:
                    # Preempted by machine failure or urgent lot
                    spent = env.now - exec_start
                    rem_dur = max(0.5, rem_dur - spent + 1.5)  # 1.5h column re-equilibration penalty
                    was_interrupted = True
                    preempted_tasks.add(task.task_id)
                    start_req_time = env.now

        completion_times[task.task_id] = env.now / 24.0

    for t in tasks:
        env.process(task_process(t))

    env.run(until=horizon_hours)

    results: Dict[str, StressSimulationOutcome] = {}
    for t in tasks:
        act_finish = completion_times.get(t.task_id, horizon_days + 1.0)
        is_late = act_finish > t.due_day
        results[t.task_id] = StressSimulationOutcome(
            task_id=t.task_id,
            project_id=t.project_id,
            scheduled_due_day=t.due_day,
            actual_finish_day=round(act_finish, 2),
            is_late=is_late,
            queue_hours=round(queue_records.get(t.task_id, 0.0), 2),
            was_preempted=(t.task_id in preempted_tasks),
            breakdown_delays_hours=round(breakdown_delays.get(t.instrument_id, 0.0), 2),
        )

    return results


# Backwards-compatibility alias
simulate_reality_scenario = simulate_stress_scenario
