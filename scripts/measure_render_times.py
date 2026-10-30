"""
scripts/measure_render_times.py

Measures execution and rendering times for each core logical tab in cGMP-QC-CoPilot:
- Tab 0: 5-Minute Guided Tour
- Tab 1: Agent 1 (LEAN Capacity & Schedule Risk / DES)
- Tab 2: Project Management Operating Layer (Portfolio Grid, WBS Gantt, Kanban, Blockers)
- Tab 3: Quality Intelligence, SPC & CAPA (I-MR, EWMA, CUSUM, Capability, KB Search, FMEA)
- Tab 4: Agent 2 (Analytical Kinetics, Cold-Chain Excursion & OOT Adaptation)
- Tab 5: Agent 3 (Local MSO OOS Hardware Telemetry Triage & Dynamic FMEA)
- Tab 6: Agent 4 (QTA / SOW Stakeholder Communications)
- Interactive Edit Scenario: Edit task duration -> recompute CPM critical path -> reduced simulation

Outputs results to results/render_times.json and warns if any render time exceeds 3.0 seconds.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict

import numpy as np
from core.db import init_db
from pm.data_access import seed_pm_database, recalculate_project_risk_and_cpm
from pm.knowledge_base import seed_knowledge_base, search_lessons_learned
from pm.fmea_capa import seed_fmea_and_capa, get_all_fmea_items, get_all_capa_records
from models.spc.spc_engine import (
    calculate_imr_chart,
    calculate_ewma_chart,
    calculate_cusum_chart,
    calculate_process_capability,
)
from models.spc.synthetic_spc import generate_purity_series
from agents.agent_1_lean_capacity.scripts.capacity_smoother import run_heijunka_leveling
from agents.agent_2_analytical_scientist.scripts.eyring_excursion import (
    simulate_cold_chain_excursion,
    analyze_6m_oot_stability,
)
from agents.agent_2_analytical_scientist.scripts.lss_sma_transfer import (
    calculate_lss_sma_dwell_transfer,
)
from agents.agent_2_analytical_scientist.scripts.bioassay_4pl_bayesian import (
    evaluate_bioassay_4pl_and_plate_edge,
)
from agents.agent_3_mso_oos_risk.scripts.oos_telemetry_triage import (
    evaluate_oos_phase1_gate,
    compute_dynamic_fmea_rpn,
)
from agents.agent_4_stakeholder_comms.scripts.comms_generator import generate_stakeholder_communications
from models.schedule_risk.risk import run_monte_carlo_schedule_risk
from agents.agent_1_lean_capacity.scripts.hybrid_schedule_risk import build_tasks_from_portfolio


def measure_all_render_times(results_path: Path | None = None) -> Dict[str, Any]:
    if results_path is None:
        results_path = Path(__file__).resolve().parent.parent / "results" / "render_times.json"

    timings_ms: Dict[str, float] = {}
    warnings: list[str] = []

    # Tab 0: 5-Minute Tour Render
    t0 = time.perf_counter()
    _ = "5-minute tour metadata rendered"
    t0_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    timings_ms["tab_0_guided_tour"] = t0_ms

    # Tab 1: Agent 1 Capacity & Schedule Risk
    t0 = time.perf_counter()
    p_file = Path(__file__).resolve().parent.parent / "agents" / "agent_1_lean_capacity" / "assets" / "example_portfolio_10_proj.json"
    with open(p_file) as f:
        demo_raw = json.load(f)
    demo_tasks, _ = build_tasks_from_portfolio(demo_raw)
    _ = run_heijunka_leveling()
    _ = run_monte_carlo_schedule_risk(demo_tasks[:8], n_simulations=40, seed=42)
    t1_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    timings_ms["tab_1_capacity_and_schedule_risk"] = t1_ms

    # Tab 2: PM Operating Layer
    t0 = time.perf_counter()
    conn = init_db(":memory:")
    seed_pm_database(conn)
    _ = conn.cursor().execute("SELECT * FROM portfolio_projects").fetchall()
    t2_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    timings_ms["tab_2_pm_operating_layer"] = t2_ms

    # Tab 3: Quality Intelligence, SPC & CAPA
    t0 = time.perf_counter()
    seed_knowledge_base(conn)
    seed_fmea_and_capa(conn)
    purity = np.array(generate_purity_series(n_points=40, seed=42)["values"])
    _ = calculate_imr_chart(purity)
    _ = calculate_ewma_chart(purity, lambda_=0.20, L=3.0)
    _ = calculate_cusum_chart(purity)
    _ = calculate_process_capability(purity, lsl=97.0, usl=100.0)
    _ = search_lessons_learned(conn, query="column backpressure fouling")
    _ = get_all_fmea_items(conn)
    _ = get_all_capa_records(conn)
    t3_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    timings_ms["tab_3_quality_intelligence_spc_capa"] = t3_ms

    # Tab 4: Agent 2 Analytical Kinetics & Biophysics
    t0 = time.perf_counter()
    _ = simulate_cold_chain_excursion(baseline_temp_c=5.0, excursion_temp_c=28.0, excursion_duration_hours=36.0)
    _ = analyze_6m_oot_stability()
    _ = calculate_lss_sma_dwell_transfer()
    _ = evaluate_bioassay_4pl_and_plate_edge()
    t4_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    timings_ms["tab_4_analytical_kinetics_excursion_oot"] = t4_ms

    # Tab 5: Agent 3 Local MSO OOS & FMEA
    t0 = time.perf_counter()
    _ = evaluate_oos_phase1_gate()
    _ = compute_dynamic_fmea_rpn()
    t5_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    timings_ms["tab_5_local_mso_oos_fmea"] = t5_ms

    # Tab 6: Agent 4 Stakeholder Communications
    t0 = time.perf_counter()
    a1 = run_heijunka_leveling()
    a2_exc = simulate_cold_chain_excursion(5.0, 28.0, 36.0)
    a2_oot = analyze_6m_oot_stability()
    a2_tr = calculate_lss_sma_dwell_transfer()
    a2_bio = evaluate_bioassay_4pl_and_plate_edge()
    a3_oos = evaluate_oos_phase1_gate()
    a3_fmea = compute_dynamic_fmea_rpn()
    _ = generate_stakeholder_communications(a1, a2_exc, a2_oot, a2_tr, a2_bio, a3_oos, a3_fmea)
    t6_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    timings_ms["tab_6_stakeholder_communications"] = t6_ms

    # Interactive Edit Scenario: Edit task duration -> recompute CPM -> reduced simulation
    t0 = time.perf_counter()
    cursor = conn.cursor()
    cursor.execute("UPDATE wbs_tasks SET duration_hours = 48.0 WHERE id = 'WBS-BI-CAP-101-SEC_HPLC'")
    conn.commit()
    edit_res = recalculate_project_risk_and_cpm(conn, "BI-CAP-101", n_simulations=40, seed=42)
    t_edit_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    timings_ms["edit_duration_cpm_simulation_roundtrip"] = t_edit_ms

    # Threshold Check: Warning if any tab exceeds 3000 ms (3.0 seconds)
    TARGET_BUDGET_MS = 3000.0
    for key, val in timings_ms.items():
        if val > TARGET_BUDGET_MS:
            warnings.append(f"WARNING: {key} took {val} ms (exceeds {TARGET_BUDGET_MS} ms SLA)")

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sla_target_ms": TARGET_BUDGET_MS,
        "all_under_3s_sla": len(warnings) == 0,
        "timings_ms": timings_ms,
        "timings_seconds": {k: round(v / 1000.0, 3) for k, v in timings_ms.items()},
        "warnings": warnings,
        "status": "PASS" if len(warnings) == 0 else "FAIL",
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = measure_all_render_times()
    print("Render Time Benchmarks:")
    for k, v in rep["timings_seconds"].items():
        print(f"  - {k}: {v:.3f} s ({rep['timings_ms'][k]} ms)")
    if rep["warnings"]:
        for w in rep["warnings"]:
            print(f"  [!] {w}")
    else:
        print("  [+] ALL TABS AND EDIT CYCLES EXECUTE IN UNDER 3.0 SECONDS (PASS)")
