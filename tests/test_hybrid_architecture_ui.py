"""
tests/test_hybrid_architecture_ui.py

Unit tests for the Lumry-Eyring PINN & Hybrid Kinetics Architecture Schematic (`pm/hybrid_architecture_ui.py`).
Verifies:
1. 4-tier node hierarchy and dual-branch engine definitions.
2. Directed edge graph integrity (all sources and targets exist).
3. Plotly figure generation, traces, shapes, and annotations.
4. Mathematical governing equations and SciPy solver parameters.
5. Performance budget (< 0.35 seconds execution).
6. Strict Guardrail G2 compliance (zero banned affirmative claims).
"""

from __future__ import annotations

import math
import time
import pytest

from core.stamps import assert_no_banned_phrases, check_banned_phrases
from pm.hybrid_architecture_ui import (
    build_hybrid_architecture_figure,
    get_governing_equations_markdown,
    get_hybrid_architecture_edges,
    get_hybrid_architecture_nodes,
    get_solver_and_invariants_dict,
)


def test_hybrid_architecture_node_tiers_and_keys():
    """Verifies that all 4 tiers and required nodes are correctly populated."""
    ctx = {
        "excursion_temp": 28.0,
        "excursion_hours": 36.0,
        "pre_exc_input": 0.790,
        "summary": {"verdict": "no detectable change"},
        "oot_diag": {"oot_verdict": "inconclusive: model and data disagree"},
    }
    nodes = get_hybrid_architecture_nodes(ctx)

    # 1. Tier 1: Biophysical Inputs (X, t)
    assert "inp_temp" in nodes
    assert "inp_purity" in nodes
    assert "inp_dwell" in nodes
    assert "inp_plate" in nodes
    for k in ("inp_temp", "inp_purity", "inp_dwell", "inp_plate"):
        assert nodes[k]["tier"] == 1
    assert "28.0°C" in nodes["inp_temp"]["subtitle"]
    assert "36" in nodes["inp_temp"]["subtitle"]
    assert "115 µL" in nodes["inp_dwell"]["details"]["Dwell Volume V_D"]
    assert "Row A/H, Col 1/12" in nodes["inp_plate"]["short_label"]

    # 2. Tier 2: Dual-Branch Engine Core
    assert "eng_branch_a" in nodes
    assert "eng_branch_b" in nodes
    assert nodes["eng_branch_a"]["tier"] == 2
    assert nodes["eng_branch_b"]["tier"] == 2

    # Branch A: Deterministic Mechanistic ODE
    branch_a_details = nodes["eng_branch_a"]["details"]
    assert "74.37 kJ/mol" in branch_a_details["Apparent Ea"]
    assert "0 / 3,200" in branch_a_details["Enforced Invariants"]
    assert "0 / 2,400" in branch_a_details["Enforced Invariants"]
    assert "RK45" in branch_a_details["Numerical Solver"]
    assert "1.000000" in branch_a_details["Mass Conservation"]

    # Branch B: Data-Driven ML / Bayesian Inference
    branch_b_details = nodes["eng_branch_b"]["details"]
    assert "synth_observations.csv" in branch_b_details["Training Inputs"]
    assert "calibration_bundle.json" in branch_b_details["Calibration Artifact"]
    assert "25-member" in branch_b_details["Architecture"]

    # 3. Tier 3: PINN Loss & Governance Constraints (L_total)
    assert "loss_phys" in nodes
    assert "loss_data" in nodes
    assert "loss_ich" in nodes
    for k in ("loss_phys", "loss_data", "loss_ich"):
        assert nodes[k]["tier"] == 3

    assert "Mass-Action" in nodes["loss_phys"]["name"]
    assert "synth_excursions.csv" in nodes["loss_data"]["details"]["Empirical Source"]
    assert "0.035%" in nodes["loss_data"]["details"]["Analytical Repeatability"]
    assert "ICH Q1E" in nodes["loss_ich"]["details"]["Guideline Scope"]
    assert "±7-day" in nodes["loss_ich"]["details"]["Tolerance Windows"]
    assert "0.099 pp" in nodes["loss_ich"]["details"]["Detection Gate"]

    # 4. Tier 4: Output & Agent Orchestration Tier
    assert "out_traj" in nodes
    assert "out_agent3" in nodes
    assert "out_part11" in nodes
    for k in ("out_traj", "out_agent3", "out_part11"):
        assert nodes[k]["tier"] == 4

    assert "no detectable change" in nodes["out_traj"]["subtitle"]
    assert "Phase I" in nodes["out_agent3"]["short_label"]
    assert "SHA-256" in nodes["out_part11"]["short_label"]


def test_hybrid_architecture_edges_integrity():
    """Verifies that all directed edges reference valid existing node IDs."""
    nodes = get_hybrid_architecture_nodes()
    edges = get_hybrid_architecture_edges()

    assert len(edges) >= 12
    for edge in edges:
        assert edge["src"] in nodes, f"Source node {edge['src']} missing"
        assert edge["dst"] in nodes, f"Destination node {edge['dst']} missing"
        assert len(edge["label"]) > 0
        assert len(edge["desc"]) > 0


def test_hybrid_architecture_figure_generation_and_timing():
    """Verifies Plotly figure builds cleanly and meets performance budget (< 0.35s)."""
    ctx = {
        "excursion_temp": 28.0,
        "excursion_hours": 36.0,
        "pre_exc_input": 0.790,
    }

    # Warm-up call to avoid cold-import JIT latency on Windows
    _ = build_hybrid_architecture_figure(ctx)

    t0 = time.perf_counter()
    fig = build_hybrid_architecture_figure(ctx)
    t_elapsed = time.perf_counter() - t0

    # Timing check (< 0.35s)
    assert t_elapsed < 0.35, f"Figure generation took {t_elapsed:.3f}s (exceeded 0.35s budget)"

    # Figure structure check
    assert fig is not None
    assert len(fig.data) >= 5  # edge lines + 4 tier node traces
    assert len(fig.layout.shapes) >= 4  # 4 tier background cards
    assert len(fig.layout.annotations) >= 4  # tier title labels + edge arrows


def test_governing_equations_and_solver_parameters():
    """Verifies mathematical formulas and solver dictionary contents."""
    md = get_governing_equations_markdown()
    assert "Lumry-Eyring" in md
    assert "74.37" in md
    assert "121.07" in md
    assert "RK45" not in md or "Radau" in md or "solve_ivp" in md or "PINN" in md
    assert r"\mathcal{L}_{\text{total}}" in md
    assert r"\mathcal{L}_{\text{physics}}" in md
    assert r"\mathcal{L}_{\text{data}}" in md
    assert r"\mathcal{L}_{\text{ICH}}" in md

    solver_dict = get_solver_and_invariants_dict()
    assert "solver" in solver_dict
    assert "invariants" in solver_dict
    assert "regulatory_governance" in solver_dict
    assert "RK45" in solver_dict["solver"]["primary_method"]
    assert "0 / 3,200" in solver_dict["invariants"]["negative_rate_violations"]
    assert "0 / 2,400" in solver_dict["invariants"]["temperature_order_inversions"]
    assert "74.37 kJ/mol" in solver_dict["invariants"]["apparent_activation_energy"]


def test_guardrail_g2_claims_hygiene():
    """Verifies zero affirmative banned-phrase violations in node descriptions and texts."""
    nodes = get_hybrid_architecture_nodes()
    edges = get_hybrid_architecture_edges()
    eq_md = get_governing_equations_markdown()
    solver_dict = get_solver_and_invariants_dict()

    # Check all node attributes
    for node_id, node in nodes.items():
        assert_no_banned_phrases(node["name"], context=f"Node {node_id} name")
        assert_no_banned_phrases(node["short_label"], context=f"Node {node_id} short_label")
        assert_no_banned_phrases(node["subtitle"], context=f"Node {node_id} subtitle")
        assert_no_banned_phrases(node["desc"], context=f"Node {node_id} desc")
        for k, v in node["details"].items():
            assert_no_banned_phrases(f"{k}: {v}", context=f"Node {node_id} details {k}")

    # Check edges
    for i, edge in enumerate(edges):
        assert_no_banned_phrases(edge["label"], context=f"Edge {i} label")
        assert_no_banned_phrases(edge["desc"], context=f"Edge {i} desc")

    # Check equations and solver dict
    assert_no_banned_phrases(eq_md, context="Governing equations markdown")
    for section_k, section_v in solver_dict.items():
        for k, v in section_v.items():
            assert_no_banned_phrases(f"{k}: {v}", context=f"Solver dict {section_k}.{k}")


def test_reactive_parameter_pipeline_shifts_and_node_labels():
    """
    Verifies that varying input parameters produces corresponding numerical shifts
    in the output arrays and updates node label strings across Tier 1 to Tier 4.
    """
    from models.hybrid_pipeline import run_reactive_hybrid_pipeline

    # 1. Baseline Run
    p_base = {
        "excursion_temp_c": 28.0,
        "excursion_duration_hours": 36.0,
        "initial_monomer_pct": 99.38,
        "formulation_ph": 5.8,
        "surfactant_conc_pct": 0.040,
        "peroxide_level_ppm": 0.2,
        "dwell_volume_ul": 115.0,
        "flow_rate_ml_min": 0.50,
        "column_void_vol_ml": 1.50,
        "include_outer_wells": False,
        "cell_passage_num": 12,
    }
    t0 = time.perf_counter()
    res_base = run_reactive_hybrid_pipeline(p_base)
    t_elapsed = time.perf_counter() - t0
    assert t_elapsed < 0.35, f"Execution took {t_elapsed:.3f}s (budget < 0.35s)"

    # Base node labels
    nodes_base = get_hybrid_architecture_nodes({"pipeline_res": res_base})
    assert "[28.0°C / 36h]" in nodes_base["inp_temp"]["subtitle"]
    assert "[99.38% Mass Conserved]" in nodes_base["inp_purity"]["subtitle"]
    assert "[pH 5.8 | PS20: 0.040% | 0.2 ppm Ox]" in nodes_base["inp_matrix"]["subtitle"]
    assert "[115 µL | t_D = 0.23 min]" in nodes_base["inp_dwell"]["subtitle"]
    assert "[Inner-60 Moat | P12]" in nodes_base["inp_plate"]["subtitle"]
    assert nodes_base["out_agent3"]["subtitle"] == "State: NOMINAL"

    # 2. Elevated Excursion Temperature (40 °C) Shift
    p_high_temp = dict(p_base, excursion_temp_c=40.0, excursion_duration_hours=48.0)
    res_high_temp = run_reactive_hybrid_pipeline(p_high_temp)
    # Excursion degradation must increase monotonically
    assert res_high_temp["branch_a"]["delta_hmw_pp"] > res_base["branch_a"]["delta_hmw_pp"]
    assert res_high_temp["branch_a"]["k_eff_peak"] > res_base["branch_a"]["k_eff_peak"]

    nodes_high_temp = get_hybrid_architecture_nodes({"pipeline_res": res_high_temp})
    assert "[40.0°C / 48h]" in nodes_high_temp["inp_temp"]["subtitle"]

    # 3. Surfactant Depletion (PS20: 0.010% < 0.030% threshold)
    p_depleted_ps20 = dict(p_base, surfactant_conc_pct=0.010)
    res_ps20 = run_reactive_hybrid_pipeline(p_depleted_ps20)
    # Micelle depletion must reduce activation enthalpy barrier dH‡
    assert res_ps20["branch_a"]["dH_agg_kj_mol"] < res_base["branch_a"]["dH_agg_kj_mol"]
    assert math.isclose(res_ps20["branch_a"]["dH_agg_kj_mol"], 38.0 - (100.0 * 0.020) - (1.2 * 0.2), abs_tol=1e-2)

    # 4. Peroxide Oxidation (2.5 ppm > 1.0 ppm)
    p_perox = dict(p_base, peroxide_level_ppm=2.5)
    res_perox = run_reactive_hybrid_pipeline(p_perox)
    # Oxidation increases activation entropy dS‡ (greater disorder)
    assert res_perox["branch_a"]["dS_agg_j_mol_k"] > res_base["branch_a"]["dS_agg_j_mol_k"]
    assert res_perox["branch_b"]["total_acceleration_factor"] > res_base["branch_b"]["total_acceleration_factor"]

    # 5. Chromatography Hydrodynamics Shift (V_D = 500 µL)
    p_dwell = dict(p_base, dwell_volume_ul=500.0, flow_rate_ml_min=0.50)
    res_dwell = run_reactive_hybrid_pipeline(p_dwell)
    assert math.isclose(res_dwell["chromatography"]["t_dwell_min"], 1.00, abs_tol=1e-2)
    nodes_dwell = get_hybrid_architecture_nodes({"pipeline_res": res_dwell})
    assert "[500 µL | t_D = 1.00 min]" in nodes_dwell["inp_dwell"]["subtitle"]

    # 6. Bioassay Edge Bias & Phenotypic Passage Drift (P30 > 25, Outer Wells Active)
    p_bio = dict(p_base, include_outer_wells=True, cell_passage_num=30)
    res_bio = run_reactive_hybrid_pipeline(p_bio)
    assert res_bio["bioassay"]["slope_ratio"] > 1.20  # Edge evaporation distorts Hill slope
    nodes_bio = get_hybrid_architecture_nodes({"pipeline_res": res_bio})
    assert "[Outer-36 Edge | P30]" in nodes_bio["inp_plate"]["subtitle"]


def test_downstream_oos_triage_flip_and_audit_ledger():
    """
    Verifies that stress excursions trip the 0.099 pp detection gate or 2.00% spec ceiling,
    automatically flipping Agent 3 state to OOT DETECTED / CRITICAL ALARM and updating SHA-256 ledger.
    """
    from models.hybrid_pipeline import run_reactive_hybrid_pipeline

    # Nominal scenario (within 0.099 pp gate)
    p_nom = {"excursion_temp_c": 28.0, "excursion_duration_hours": 36.0}
    res_nom = run_reactive_hybrid_pipeline(p_nom)
    assert res_nom["triage"]["agent3_state"] == "NOMINAL"
    assert res_nom["triage"]["verdict"] == "no detectable change"

    # Severe stress scenario (42°C for 60h)
    p_stress = {
        "excursion_temp_c": 42.0,
        "excursion_duration_hours": 60.0,
        "surfactant_conc_pct": 0.010,
        "peroxide_level_ppm": 2.5,
    }
    res_stress = run_reactive_hybrid_pipeline(p_stress)
    assert res_stress["branch_a"]["delta_hmw_pp"] > 0.099
    assert res_stress["triage"]["agent3_state"] in ("OOT DETECTED", "CRITICAL ALARM")
    assert "no detectable change" not in res_stress["triage"]["verdict"]

    # Verify SHA-256 audit digest generation
    assert len(res_stress["triage"]["audit_hash"]) == 64
    assert res_stress["triage"]["audit_hash"] != res_nom["triage"]["audit_hash"]

    # Verify node reflection
    nodes_stress = get_hybrid_architecture_nodes({"pipeline_res": res_stress})
    assert nodes_stress["out_agent3"]["subtitle"] in ("State: OOT DETECTED", "State: CRITICAL ALARM")

