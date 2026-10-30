"""
cGMP-QC-CoPilot: Agent 1 Hybrid Capacity & Schedule-Risk Integration Interface.

Integrates:
1. Legacy Heijunka baseline (from capacity_smoother.py)
2. OR-Tools CP-SAT discrete optimization (from models/schedule_risk/optimize.py)
3. Deterministic Critical Path Method (CPM) and Monte Carlo schedule risk (from models/schedule_risk/risk.py)
4. Multi-criteria weighted prioritization matrix with Pareto frontier (from models/schedule_risk/prioritize.py)
5. SimPy discrete-event simulation (from models/schedule_risk/des.py)
6. Quantile GBDT forecasters (from models/schedule_risk/forecast.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from agents.agent_1_lean_capacity.scripts.capacity_smoother import (
    load_portfolio_data,
    run_heijunka_leveling,
)
from core.audit import AuditRecord, AuditTrail
from core.stamps import DRAFT_STAMP, apply_draft_stamp, assert_no_banned_phrases
from models.schedule_risk.des import run_des_simulation
from models.schedule_risk.optimize import OptimizationResult, solve_cpsat_leveling
from models.schedule_risk.prioritize import (
    DEFAULT_WEIGHTS,
    compute_prioritization_ranking,
    evaluate_weight_sensitivity,
)
from models.schedule_risk.risk import (
    compute_deterministic_cpm,
    run_monte_carlo_schedule_risk,
)
from models.schedule_risk.tasks import (
    FlexibilityClass,
    InstrumentDefinition,
    MethodParameters,
    Task,
)


def build_tasks_from_portfolio(portfolio_dict: Dict[str, Any]) -> Tuple[List[Task], Dict[str, InstrumentDefinition]]:
    """Converts the Example Biologics Site 10-project portfolio into Task and InstrumentDefinition objects."""
    skid_meta = portfolio_dict["facility_metadata"]["skid_definitions"]
    instruments = {
        skid_id: InstrumentDefinition(
            id=skid_id,
            name=meta["display_name"],
            weekly_max_hours=float(meta["weekly_max_hours"]),
            lean_threshold_pct=float(meta["lean_threshold_pct"]),
        )
        for skid_id, meta in skid_meta.items()
    }

    tasks: List[Task] = []
    skid_tracker = {skid_id: 0.0 for skid_id in instruments}

    for p in portfolio_dict["projects"]:
        is_crit = bool(p["is_critical_path"])
        window_d = float(p.get("ich_pull_window_days", 0))

        if is_crit:
            f_class = (
                FlexibilityClass.FIXED_COMMERCIAL
                if p["engine_type"] == "Captive Pipeline"
                else FlexibilityClass.FIXED_CONTRACTUAL
            )
        else:
            f_class = (
                FlexibilityClass.FLEXIBLE_WINDOW
                if window_d > 0
                else FlexibilityClass.FLEXIBLE_INTERNAL
            )

        # Create discrete tasks per instrument skid for each project
        for skid_id, skid_h in p["skid_hours"].items():
            if skid_h <= 0:
                continue

            base_w = 0.0 if p["scheduled_week"] == "W42" else 7.0
            # If fixed, assign sequential non-overlapping scheduled arrival slot
            if is_crit:
                arr_d = skid_tracker[skid_id]
                skid_tracker[skid_id] += round(float(skid_h) / 24.0, 2)
            else:
                arr_d = base_w + (1.0 if "CDMO" in p["engine_type"] else 0.0)

            due_d = base_w + 7.0

            t = Task.create(
                task_id=f"TASK-{p['project_id']}-{skid_id}",
                project_id=p["project_id"],
                program_name=p["program_name"],
                test_panel=f"{skid_id} Panel",
                instrument_id=skid_id,
                sample_arrival_day=round(arr_d, 2),
                due_day=round(due_d, 2),
                flexibility_class=f_class,
                window_days=window_d,
                penalty_weight=float(p.get("contract_penalty", 10000)) / 1000.0,
                patient_impact=is_crit,
                priority_tier=p["priority_tier"],
                nominal_duration_hours=float(skid_h),
            )
            tasks.append(t)

    return tasks, instruments


def run_hybrid_capacity_and_risk(
    portfolio: Optional[Dict[str, Any]] = None,
    usp_harvest_delay_days: int = 0,
    instrument_outage_mult: float = 1.0,
    analyst_absence_mult: float = 1.0,
    inject_rush_task: bool = False,
    prioritization_weights: Optional[Dict[str, float]] = None,
    n_mc_sims: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes end-to-end WP1 decision support pipeline:
    1. Legacy Heijunka baseline run
    2. CP-SAT workload leveling with hard-constraint enforcement
    3. Monte Carlo milestone schedule risk
    4. Multi-criteria prioritization matrix & sensitivity
    5. SimPy DES resource dynamics
    """
    if portfolio is None:
        portfolio = load_portfolio_data()

    # 1. Run Legacy Baseline
    legacy_res = run_heijunka_leveling(portfolio=portfolio, usp_harvest_delay_days=usp_harvest_delay_days)

    # 2. Build Task Network
    tasks, instruments = build_tasks_from_portfolio(portfolio)

    # 3. Solve CP-SAT Optimization
    cpsat_res = solve_cpsat_leveling(
        tasks=tasks,
        instruments=instruments,
        planning_horizon_days=14,
        buffer_ceiling_pct=85.0,
        time_limit_seconds=2.0,
        seed=seed,
    )

    # 4. Monte Carlo Schedule Risk
    risk_records = run_monte_carlo_schedule_risk(
        tasks=tasks,
        n_simulations=n_mc_sims,
        seed=seed,
        usp_harvest_delay_days=float(usp_harvest_delay_days),
        instrument_outage_mult=instrument_outage_mult,
        analyst_absence_mult=analyst_absence_mult,
        inject_rush_task=inject_rush_task,
    )
    df_risk = pd.DataFrame([r.to_dict() for r in risk_records.values()])

    # 5. Prioritization Matrix & Sensitivity
    weights = prioritization_weights or DEFAULT_WEIGHTS
    ranked_scores, df_prio = compute_prioritization_ranking(portfolio["projects"], weights=weights)
    sensitivity = evaluate_weight_sensitivity(portfolio["projects"], base_weights=weights, n_samples=200, seed=seed)

    # 6. SimPy DES Verification Run
    des_res = run_des_simulation(
        tasks=tasks,
        instruments=instruments,
        n_replications=100,
        planning_horizon_days=21.0,
        seed=seed,
    )

    # Format move comparison table
    moves_data = [m.to_dict() for m in cpsat_res.moves]
    df_moves = pd.DataFrame(moves_data) if moves_data else pd.DataFrame(columns=["task_id", "project_id", "delta_days", "reason"])

    # Output dictionary with draft stamp
    summary_text = (
        f"Hybrid Schedule Risk & Capacity Analysis:\n"
        f"- Legacy Baseline UPLC: {legacy_res['metrics']['w42_uplc_pct_before']}% -> {legacy_res['metrics']['w42_uplc_pct_after']}%\n"
        f"- CP-SAT Solver Status: {cpsat_res.status} | Peak Utilization: {max(cpsat_res.peak_utilization_after.values())}%\n"
        f"- Zero-Float SLAs Protected: {legacy_res['metrics']['critical_path_protected_count']}/6\n"
        f"- Monte Carlo Mean Delay: {df_risk['expected_delay_days'].mean():.2f} days\n"
        f"- Priority #1 Project: {ranked_scores[0].project_id} (Score: {ranked_scores[0].composite_score:.2f})\n"
    )
    assert_no_banned_phrases(summary_text, context="Capacity Summary")

    return {
        "regulatory_notice": DRAFT_STAMP,
        "legacy_baseline": legacy_res,
        "cpsat_result": cpsat_res,
        "df_moves": df_moves,
        "df_risk": df_risk,
        "df_prioritization": df_prio,
        "prioritization_scores": ranked_scores,
        "sensitivity": sensitivity,
        "des_summary": des_res.summary(),
        "summary_text": apply_draft_stamp(summary_text),
    }


def execute_schedule_sign_off(
    audit_trail: AuditTrail,
    reviewer_name: str,
    reviewer_role: str,
    justification: str,
    confirmed: bool,
    optimization_result: OptimizationResult,
) -> Tuple[bool, Optional[AuditRecord], List[str]]:
    """
    Enforces the Part 11-style review gate for applying a leveled schedule.
    Writes an immutable, hash-chained record to core.audit.AuditTrail.
    """
    errors: List[str] = []
    if not reviewer_name or not reviewer_name.strip():
        errors.append("Reviewer Name is required (Attributable).")
    if not reviewer_role or not reviewer_role.strip():
        errors.append("Reviewer Role is required.")
    if not justification or len(justification.strip()) < 10:
        errors.append("Sign-off justification must be at least 10 characters explaining schedule changes.")
    if not confirmed:
        errors.append("Human reviewer confirmation checkbox must be checked.")

    if errors:
        return False, None, errors

    payload = {
        "solver_status": optimization_result.status,
        "moves_count": len(optimization_result.moves),
        "peak_utilization_after": optimization_result.peak_utilization_after,
        "regulatory_notice": DRAFT_STAMP,
    }

    rec = audit_trail.append_event(
        actor_name=reviewer_name.strip(),
        actor_role=reviewer_role.strip(),
        action="COMMIT_LEVELED_SCHEDULE",
        justification=justification.strip(),
        entity_type="SCHEDULE",
        entity_id="SCHED-2026-W42-LEVELED",
        payload=payload,
    )

    return True, rec, []
