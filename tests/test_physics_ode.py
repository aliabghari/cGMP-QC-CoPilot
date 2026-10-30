"""
Phase 1 Unit Tests: Deterministic Physics, Amendments A & B Sanity Gates,
Differentiable RK4 ODE vs Closed-Form & 3-State Stiff Radau Oracle, and Torch-vs-NumPy Parity.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.config import (
    load_hybrid_config,
    get_config_sha256,
    get_nominal_parameters,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.physics import (
    HAS_TORCH,
    KB_SI,
    H_PLANCK_SI,
    R_GAS_SI,
    SECONDS_PER_HOUR,
    T_REF_K,
    celsius_to_kelvin,
    eyring_prefactor_per_hour,
    eyring_rate_from_dG,
    eyring_thermodynamics,
    dS_from_dG_dH,
    dG_from_dH_dS,
    eyring_to_arrhenius_ea,
    arrhenius_rate,
    apparent_activation_energy,
    lumry_eyring_rates,
    evaluate_pathway_rates,
    register_pathway,
    PATHWAY_REGISTRY,
    compute_excursion_detection_gate_pct,
    classify_excursion_verdict,
    compute_amendment_ab_summary,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.ode import (
    build_excursion_profile_knots,
    integrate_pathway_rk4,
    closed_form_isothermal_trajectory,
    integrate_3state_radau_oracle,
)

if HAS_TORCH:
    import torch


def test_config_and_amendment_ab_numbers():
    """Verifies Amendment A (fraction/hour units, ~121 kJ/mol dG_eff_ref) and Amendment B sanity gates."""
    cfg = load_hybrid_config()
    sha = get_config_sha256()
    assert len(sha) == 64

    summary = compute_amendment_ab_summary()
    # Amendment A: dG_eff_ref at 5C in fraction/hour is ~121.07 kJ/mol within [116, 126]
    b_dg = cfg["parameter_bounds"]["dG_eff_ref_kj_mol"]
    assert b_dg["min"] <= summary["dG_eff_ref_5c_kj_mol"] <= b_dg["max"]
    assert 120.0 <= summary["dG_eff_ref_5c_kj_mol"] <= 122.0

    # Intrinsic step barrier dG_agg_ref is linked via dG_eff_ref + R*T_ref*ln(f_U(5C))
    b_dg_agg = cfg["parameter_bounds"]["dG_agg_ref_kj_mol"]
    assert b_dg_agg["min"] <= summary["dG_agg_ref_5c_kj_mol"] <= b_dg_agg["max"]

    # Amendment B: Apparent Ea between 5C and 40C must fall in configured range [60, 200] kJ/mol
    gates = cfg["sanity_gates"]
    assert gates["apparent_ea_5_40c_min_kj_mol"] <= summary["apparent_ea_5_40c_kj_mol"] <= gates["apparent_ea_5_40c_max_kj_mol"]
    assert gates["apparent_ea_5_40c_min_kj_mol"] <= summary["apparent_ea_5_25c_kj_mol"] <= gates["apparent_ea_5_40c_max_kj_mol"]

    # Every parameter bound in hybrid_config.json is marked "ASSUMPTION - SME to confirm"
    for p_key, p_meta in cfg["parameter_bounds"].items():
        assert p_meta["status"] == "ASSUMPTION - SME to confirm", f"Missing status on {p_key}"

    # Verify nominal model excursion outcomes (36h@28C, 2h@28C, 48h@38C, 72h@40C, 7d@40C, 14d@40C) using M0 = 1 - HMW0 - LMW0 = 0.9938
    exc = summary["excursion_outcomes"]
    assert math.isclose(exc["36h_at_28C"]["initial_monomer_fraction"], 0.9938, rel_tol=1e-12)
    assert math.isclose(exc["36h_at_28C"]["delta_hmw_pp"], 0.016007, rel_tol=1e-3)
    assert math.isclose(exc["2h_at_28C"]["delta_hmw_pp"], 0.000889, rel_tol=1e-3)
    assert math.isclose(exc["48h_at_38C"]["delta_hmw_pp"], 0.055460, rel_tol=1e-3)
    assert math.isclose(exc["72h_at_40C"]["delta_hmw_pp"], 0.099947, rel_tol=1e-3)
    assert math.isclose(exc["7d_at_40C"]["delta_hmw_pp"], 0.233045, rel_tol=1e-3)
    assert math.isclose(exc["14d_at_40C"]["delta_hmw_pp"], 0.465559, rel_tol=1e-3)
    assert exc["36h_at_28C"]["verdict"] == "no detectable change"
    assert exc["2h_at_28C"]["verdict"] == "no detectable change"
    assert exc["72h_at_40C"]["verdict"] == "inconclusive"
    assert exc["14d_at_40C"]["verdict"] == "detectable change likely"
    assert "PLACEHOLDER_PRE_ENSEMBLE" in exc["36h_at_28C"]["verdict_interval_source"]


def test_excursion_verdict_rule_and_sqrt2_sd_gate():
    """
    Item 2: Verifies the pre/post difference gate uses 2 * sqrt(2) * SD (~0.099 pp for SD = 0.035 pp),
    that sec_hmw_repeatability_sd_pct is marked as an assumption from method-validation intermediate precision,
    and tests all three verdict outcomes ("no detectable change", "detectable change likely", "inconclusive"),
    including an ODE-integrated intermediate scenario at 40 °C whose 95% interval straddles the 0.099 pp gate.
    """
    cfg = load_hybrid_config()
    thresh = cfg["assay_and_decision_thresholds"]
    assert (
        thresh["sec_hmw_repeatability_sd_status"]
        == "ASSUMPTION - SME to confirm; should come from method-validation intermediate precision"
    )
    sd = float(thresh["sec_hmw_repeatability_sd_pct"])
    gate = compute_excursion_detection_gate_pct(sd, z_multiplier=2.0)
    assert math.isclose(gate, 2.0 * math.sqrt(2.0) * 0.035, rel_tol=1e-12)
    assert math.isclose(gate, 0.098994949, rel_tol=1e-6)

    # Outcome 1: "no detectable change" (36h @ 28C, ~0.0160 pp, upper 95% bound < gate)
    v_no = classify_excursion_verdict(0.012, 0.022, sec_repeatability_sd_pct=sd)
    assert v_no["verdict"] == "no detectable change"
    assert "SEC alone cannot separate reversible" in v_no["assessment_statement"]
    assert "RELEASE" not in v_no["assessment_statement"].upper()

    # Outcome 2: "detectable change likely" (14d @ 40C, ~0.466 pp, lower 95% bound > gate)
    v_yes = classify_excursion_verdict(0.380, 0.540, sec_repeatability_sd_pct=sd)
    assert v_yes["verdict"] == "detectable change likely"
    assert "SEC alone cannot separate reversible" in v_yes["assessment_statement"]
    assert "RELEASE" not in v_yes["assessment_statement"].upper()

    # Outcome 3: "inconclusive" via ODE integration of intermediate 40 °C scenario
    # (a) 72h @ 40 °C (~0.1000 pp point estimate with ±15% parameter uncertainty -> [0.0850, 0.1149] pp)
    params = get_nominal_parameters(cfg)
    tk_72h, tc_72h = build_excursion_profile_knots(
        baseline_temp_c=5.0,
        excursion_temp_c=40.0,
        excursion_duration_hours=72.0,
        ramp_hours=0.0,
        total_hours=72.0,
        start_hour=0.0,
    )
    ode_72h = integrate_pathway_rk4(
        time_grid_h=np.linspace(0.0, 72.0, 145),
        time_knots_h=tk_72h,
        temp_knots_c=tc_72h,
        params=params,
        initial_hmw_fraction=0.0062,
        initial_lmw_fraction=0.0,
        include_lmw=False,
    )
    dhmw_72h_pp = float(ode_72h["hmw_fraction"][-1] - ode_72h["hmw_fraction"][0]) * 100.0
    v_inc_72h = classify_excursion_verdict(
        delta_hmw_lower95_pct=dhmw_72h_pp * 0.85,
        delta_hmw_upper95_pct=dhmw_72h_pp * 1.15,
        sec_repeatability_sd_pct=sd,
    )
    assert v_inc_72h["verdict"] == "inconclusive"

    # (b) 7d (168h) @ 40 °C (~0.2330 pp point estimate) when lower 95% bound is below 0.099 pp
    # (e.g., wide prior/uncalibrated epistemic uncertainty spanning [0.080, 0.380] pp)
    tk_7d, tc_7d = build_excursion_profile_knots(
        baseline_temp_c=5.0,
        excursion_temp_c=40.0,
        excursion_duration_hours=168.0,
        ramp_hours=0.0,
        total_hours=168.0,
        start_hour=0.0,
    )
    ode_7d = integrate_pathway_rk4(
        time_grid_h=np.linspace(0.0, 168.0, 337),
        time_knots_h=tk_7d,
        temp_knots_c=tc_7d,
        params=params,
        initial_hmw_fraction=0.0062,
        initial_lmw_fraction=0.0,
        include_lmw=False,
    )
    dhmw_7d_pp = float(ode_7d["hmw_fraction"][-1] - ode_7d["hmw_fraction"][0]) * 100.0
    assert math.isclose(dhmw_7d_pp, 0.233045, rel_tol=1e-3)
    v_inc_7d = classify_excursion_verdict(
        delta_hmw_lower95_pct=0.085,
        delta_hmw_upper95_pct=dhmw_7d_pp * 1.25,
        sec_repeatability_sd_pct=sd,
    )
    assert v_inc_7d["verdict"] == "inconclusive"
    assert "SEC alone cannot separate reversible" in v_inc_7d["assessment_statement"]
    assert "RELEASE" not in v_inc_7d["assessment_statement"].upper()


def test_eyring_and_arrhenius_vs_hand_computed():
    """Verifies Eyring and Arrhenius equations against independent literal hand-calculated CODATA constants."""
    t_k = 298.15  # 25.0 °C
    dH_ref = 75.0  # kJ/mol
    dS_ref = -140.0  # J/(mol*K)

    # Independent literal hand-calculated values (not computed via the functions under test):
    # dG(298.15K) = 75.0 - 0.29815 * (-140.0) = 116.741 kJ/mol
    # nu(298.15K) = (1.380649e-23 * 298.15 / 6.62607015e-34) * 3600 = 2.2364866899574292e+16 h^-1
    # exp(-116741.0 / (8.314462618 * 298.15)) = 3.530692623719842e-21
    # k(298.15K)  = 2.2364866899574292e+16 * 3.530692623719842e-21 = 7.89634701850167e-05 h^-1
    # Ea(298.15K) = 75.0 + 0.008314462618 * 298.15 = 77.4789570295567 kJ/mol
    dG_literal = 116.741
    k_literal = 7.89634701850167e-05
    ea_literal = 77.4789570295567

    res = eyring_thermodynamics(temp_k=t_k, dH_ref_kj_mol=dH_ref, dS_ref_j_mol_k=dS_ref, dCp_kj_mol_k=0.0)
    assert math.isclose(float(res["dG_kj_mol"]), dG_literal, rel_tol=1e-12)
    assert math.isclose(float(res["k_fraction_per_h"]), k_literal, rel_tol=1e-12)
    assert math.isclose(float(res["ea_kj_mol"]), ea_literal, rel_tol=1e-12)

    # Non-zero heat capacity dCp‡ check against independent literal constants:
    # dH(298.15K) = 75.0 + 0.45 * (298.15 - 278.15) = 84.0 kJ/mol
    # dS(298.15K) = -140.0 + 450.0 * ln(298.15 / 278.15) = -108.7537194284843 J/(mol*K)
    dCp = 0.45  # kJ/(mol*K)
    dH_cp_literal = 84.0
    dS_cp_literal = -108.7537194284843
    res_cp = eyring_thermodynamics(temp_k=t_k, dH_ref_kj_mol=dH_ref, dS_ref_j_mol_k=dS_ref, dCp_kj_mol_k=dCp)
    assert math.isclose(float(res_cp["dH_kj_mol"]), dH_cp_literal, rel_tol=1e-12)
    assert math.isclose(float(res_cp["dS_j_mol_k"]), dS_cp_literal, rel_tol=1e-12)

    # Arrhenius check against independent literal constant:
    # k(298.15K) = 1e-6 * exp(-(80000.0 / 8.314462618) * (1/298.15 - 1/278.15)) = 1.018023833116948e-05 h^-1
    arr = arrhenius_rate(temp_k=t_k, ea_kj_mol=80.0, k_ref_per_h=1e-6, t_ref_k=T_REF_K)
    k_arr_literal = 1.018023833116948e-05
    assert math.isclose(float(arr["k_fraction_per_h"]), k_arr_literal, rel_tol=1e-12)


def test_delta_s_measurably_changes_rate():
    """
    Confirms that changing dS‡ measurably changes k(T) by the exact factor exp(Delta_dS / R),
    fixing the legacy placeholder where dS‡ cancelled out.
    """
    t_k = celsius_to_kelvin(25.0)
    delta_s_shift = 10.0  # +10 J/(mol*K)
    expected_ratio = math.exp(delta_s_shift / R_GAS_SI)

    r1 = eyring_thermodynamics(temp_k=t_k, dH_ref_kj_mol=60.0, dS_ref_j_mol_k=-180.0)
    r2 = eyring_thermodynamics(temp_k=t_k, dH_ref_kj_mol=60.0, dS_ref_j_mol_k=-180.0 + delta_s_shift)
    ratio = float(r2["k_fraction_per_h"]) / float(r1["k_fraction_per_h"])
    assert math.isclose(ratio, expected_ratio, rel_tol=1e-10)

    # Also verify in Lumry-Eyring when dS_agg_ref_j_mol_k is varied directly
    le1 = lumry_eyring_rates(t_k, dH_agg_ref_kj_mol=32.8, dS_agg_ref_j_mol_k=-260.0)
    le2 = lumry_eyring_rates(t_k, dH_agg_ref_kj_mol=32.8, dS_agg_ref_j_mol_k=-260.0 + delta_s_shift)
    le_ratio = float(le2["k_eff_fraction_per_h"]) / float(le1["k_eff_fraction_per_h"])
    assert math.isclose(le_ratio, expected_ratio, rel_tol=1e-10)


def test_rate_positivity_and_temperature_monotonicity_2_to_45c():
    """Verifies k(T) > 0 and strictly monotonically increasing with T over 2-45 °C."""
    temps_c = np.linspace(2.0, 45.0, 87)
    temps_k = celsius_to_kelvin(temps_c)
    params = get_nominal_parameters()

    pw_rates = evaluate_pathway_rates(temps_k, params, pathways=("SEC_HMW", "SEC_LMW", "CEX_ACIDIC"))
    for pw_name, pw_dict in pw_rates.items():
        k_vals = np.asarray(pw_dict["k_eff_fraction_per_h"], dtype=float)
        assert np.all(k_vals > 0.0), f"{pw_name} produced non-positive rate"
        diffs = np.diff(k_vals)
        assert np.all(diffs > 0.0), f"{pw_name} rate is not strictly increasing over 2-45 °C"


def test_ode_rk4_vs_closed_form_isothermal():
    """
    Verifies:
    1. RK4 isothermal integration matches closed-form solution within < 1e-4 relative error on nominal grid.
    2. Harder case (40 °C, 3 months = 2191.5 h, coarse dt = 730.5 h vs 365.25 h) produces a non-zero
       truncation error that scales at the theoretical 4th-order rate (~16x reduction per halving of dt).
    3. An intentionally over-coarse step (dt where k*dt ~ 0.65) exceeds the 1e-4 tolerance, proving the
       test tolerance can fail when RK4 step size is too coarse.
    """
    params = get_nominal_parameters()
    time_grid = np.linspace(0.0, 720.0, 1441)  # 30 days at 0.5h step
    temp_c = 25.0
    temp_profile = np.full_like(time_grid, temp_c)

    rk4_out = integrate_pathway_rk4(
        time_grid_h=time_grid,
        temp_profile_c=temp_profile,
        params=params,
        initial_hmw_fraction=0.0045,
        initial_lmw_fraction=0.0012,
        include_lmw=True,
    )
    cf_out = closed_form_isothermal_trajectory(
        time_grid_h=time_grid,
        temp_c=temp_c,
        params=params,
        initial_hmw_fraction=0.0045,
        initial_lmw_fraction=0.0012,
        include_lmw=True,
    )

    rel_err_hmw = np.max(np.abs(rk4_out["hmw_fraction"] - cf_out["hmw_fraction"]) / cf_out["hmw_fraction"])
    rel_err_mon = np.max(np.abs(rk4_out["monomer_fraction"] - cf_out["monomer_fraction"]) / cf_out["monomer_fraction"])
    assert rel_err_hmw < 1e-4
    assert rel_err_mon < 1e-4

    # Harder case: 40 °C for 3 months (2191.5 h) with fast-lot barrier (dG_eff_ref = 116.5 kJ/mol)
    fast_params = dict(params)
    fast_params["dG_eff_ref_kj_mol"] = 116.5
    fast_params["dH_agg_ref_kj_mol"] = 65.0
    fast_params["dH_u_kj_mol"] = 50.0

    # Coarse grid 1: 3 steps over 3 months (dt = 730.5 h)
    grid_coarse = np.linspace(0.0, 2191.5, 4)
    rk4_c1 = integrate_pathway_rk4(
        time_grid_h=grid_coarse,
        temp_profile_c=np.full_like(grid_coarse, 40.0),
        params=fast_params,
        initial_hmw_fraction=0.0062,
        include_lmw=False,
    )
    cf_c1 = closed_form_isothermal_trajectory(
        time_grid_h=grid_coarse,
        temp_c=40.0,
        params=fast_params,
        initial_hmw_fraction=0.0062,
        include_lmw=False,
    )
    err_c1 = float(np.max(np.abs(rk4_c1["hmw_fraction"] - cf_c1["hmw_fraction"]) / cf_c1["hmw_fraction"]))

    # Coarse grid 2: 6 steps over 3 months (dt = 365.25 h, half of grid_coarse)
    grid_half = np.linspace(0.0, 2191.5, 7)
    rk4_c2 = integrate_pathway_rk4(
        time_grid_h=grid_half,
        temp_profile_c=np.full_like(grid_half, 40.0),
        params=fast_params,
        initial_hmw_fraction=0.0062,
        include_lmw=False,
    )
    cf_c2 = closed_form_isothermal_trajectory(
        time_grid_h=grid_half,
        temp_c=40.0,
        params=fast_params,
        initial_hmw_fraction=0.0062,
        include_lmw=False,
    )
    err_c2 = float(np.max(np.abs(rk4_c2["hmw_fraction"] - cf_c2["hmw_fraction"]) / cf_c2["hmw_fraction"]))

    # Non-zero truncation error that improves by ~16x (4th-order RK4) when step size is halved
    assert err_c1 > 1e-9
    assert 12.0 <= (err_c1 / err_c2) <= 22.0

    # Single giant step over 2191.5 h with extreme barrier (dG_eff_ref = 113.0 kJ/mol, k*dt ~ 1.1)
    # must exceed the 1e-4 relative error gate, proving the test gate can fail on an under-resolved grid
    extreme_params = dict(fast_params)
    extreme_params["dG_eff_ref_kj_mol"] = 113.0
    grid_1step = np.array([0.0, 2191.5])
    rk4_bad = integrate_pathway_rk4(
        time_grid_h=grid_1step,
        temp_profile_c=np.full_like(grid_1step, 40.0),
        params=extreme_params,
        initial_hmw_fraction=0.0062,
        include_lmw=False,
    )
    cf_bad = closed_form_isothermal_trajectory(
        time_grid_h=grid_1step,
        temp_c=40.0,
        params=extreme_params,
        initial_hmw_fraction=0.0062,
        include_lmw=False,
    )
    err_bad = float(np.max(np.abs(rk4_bad["hmw_fraction"] - cf_bad["hmw_fraction"]) / cf_bad["hmw_fraction"]))
    assert err_bad > 1e-4, f"Expected single giant step to fail 1e-4 tolerance, got {err_bad}"


def test_preequilibrium_rk4_vs_3state_radau_oracle_and_mass_balance():
    """
    Verifies:
    1. Rapid pre-equilibrium RK4 agrees with the 3-state stiff Radau oracle during a 36h 4->28 °C excursion with ramps.
    2. Mass balance N + U + HMW + LMW = 1.0 (100%) within 1e-6 at all timepoints.
    3. HMW is non-decreasing and Monomer purity never increases.
    """
    params = get_nominal_parameters()
    tk, tc = build_excursion_profile_knots(
        baseline_temp_c=4.0,
        excursion_temp_c=28.0,
        excursion_duration_hours=36.0,
        ramp_hours=2.0,
        n_spikes=1,
        total_hours=120.0,
        start_hour=24.0,
    )
    time_grid = np.linspace(0.0, 120.0, 481)

    rk4_out = integrate_pathway_rk4(
        time_grid_h=time_grid,
        time_knots_h=tk,
        temp_knots_c=tc,
        params=params,
        initial_hmw_fraction=0.0062,
        initial_lmw_fraction=0.0010,
        include_lmw=True,
    )
    radau_out = integrate_3state_radau_oracle(
        time_grid_h=time_grid,
        time_knots_h=tk,
        temp_knots_c=tc,
        params=params,
        initial_hmw_fraction=0.0062,
        initial_lmw_fraction=0.0010,
        include_lmw=True,
        k_unfold_relax_per_hour=100.0,
    )

    # Mass balance within 1e-6 for both solvers
    assert np.max(np.abs(rk4_out["mass_balance_fraction"] - 1.0)) < 1e-6
    assert np.max(np.abs(radau_out["mass_balance_fraction"] - 1.0)) < 1e-6

    # Monotonicity: HMW non-decreasing, Monomer never increases
    assert np.all(np.diff(rk4_out["hmw_fraction"]) >= -1e-15)
    assert np.all(np.diff(rk4_out["monomer_fraction"]) <= 1e-15)
    assert np.all(np.diff(radau_out["hmw_fraction"]) >= -1e-12)
    assert np.all(np.diff(radau_out["monomer_fraction"]) <= 1e-12)

    # Agreement between rapid pre-equilibrium and 3-state stiff Radau oracle
    delta_hmw_rk4 = rk4_out["hmw_fraction"][-1] - rk4_out["hmw_fraction"][0]
    delta_hmw_radau = radau_out["hmw_fraction"][-1] - radau_out["hmw_fraction"][0]
    rel_diff_delta = abs(delta_hmw_rk4 - delta_hmw_radau) / max(delta_hmw_radau, 1e-12)
    assert rel_diff_delta < 5e-3


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in current interpreter")
def test_torch_vs_numpy_parity():
    """
    Amendment D: Verifies PyTorch and NumPy forward passes match within 1e-6 relative error,
    confirms every PyTorch output is a genuine Tensor with active autograd (`requires_grad=True`,
    `grad_fn is not None`), and verifies backpropagation through the full RK4 ODE + physics layer
    against central finite differences across all active parameters.
    """
    params_np = get_nominal_parameters()
    params_torch = {
        k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
        for k, v in params_np.items()
    }

    tk, tc = build_excursion_profile_knots(
        baseline_temp_c=4.0,
        excursion_temp_c=28.0,
        excursion_duration_hours=36.0,
        ramp_hours=2.0,
    )
    time_grid = np.linspace(0.0, 120.0, 241)

    out_np = integrate_pathway_rk4(
        time_grid_h=time_grid,
        time_knots_h=tk,
        temp_knots_c=tc,
        params=params_np,
        initial_hmw_fraction=0.0062,
        include_lmw=True,
        include_acidic=True,
    )
    out_pt = integrate_pathway_rk4(
        time_grid_h=time_grid,
        time_knots_h=tk,
        temp_knots_c=tc,
        params=params_torch,
        initial_hmw_fraction=0.0062,
        include_lmw=True,
        include_acidic=True,
    )

    for key in ("monomer_fraction", "hmw_fraction", "lmw_fraction", "acidic_fraction", "f_U", "k_eff_fraction_per_h"):
        assert isinstance(out_pt[key], torch.Tensor), f"{key} is not a torch.Tensor"
        assert out_pt[key].requires_grad, f"{key} lost requires_grad (converted to numpy?)"
        assert out_pt[key].grad_fn is not None, f"{key} has no grad_fn"
        arr_np = out_np[key]
        arr_pt = out_pt[key].detach().cpu().numpy()
        rel_err = np.max(np.abs(arr_np - arr_pt) / np.maximum(np.abs(arr_np), 1e-15))
        assert rel_err < 1e-6, f"Torch vs NumPy mismatch on {key}: {rel_err}"

    # Backpropagate composite loss through all 3 pathways (SEC_HMW, SEC_LMW, CEX_ACIDIC)
    loss = (
        out_pt["hmw_fraction"][-1] * 100.0
        + out_pt["lmw_fraction"][-1] * 100.0
        + out_pt["acidic_fraction"][-1] * 10.0
    )
    loss.backward()

    active_grad_keys = (
        "dG_eff_ref_kj_mol",
        "dH_agg_ref_kj_mol",
        "ku_ref_5c",
        "dH_u_kj_mol",
        "dG_frag_ref_kj_mol",
        "ea_frag_kj_mol",
        "dG_acidic_ref_kj_mol",
        "ea_acidic_kj_mol",
    )
    for k in active_grad_keys:
        g_tensor = params_torch[k].grad
        assert g_tensor is not None, f"Missing gradient for {k}"
        g_val = float(g_tensor.item())
        assert abs(g_val) > 0.0, f"Zero gradient for {k}"

        # Compare autograd against central finite difference through the NumPy ODE path
        eps = 1e-6 * max(1.0, abs(float(params_np[k])))
        p_plus = dict(params_np)
        p_minus = dict(params_np)
        p_plus[k] = float(params_np[k]) + eps
        p_minus[k] = float(params_np[k]) - eps
        o_plus = integrate_pathway_rk4(
            time_grid_h=time_grid, time_knots_h=tk, temp_knots_c=tc,
            params=p_plus, initial_hmw_fraction=0.0062, include_lmw=True, include_acidic=True,
        )
        o_minus = integrate_pathway_rk4(
            time_grid_h=time_grid, time_knots_h=tk, temp_knots_c=tc,
            params=p_minus, initial_hmw_fraction=0.0062, include_lmw=True, include_acidic=True,
        )
        loss_plus = o_plus["hmw_fraction"][-1] * 100.0 + o_plus["lmw_fraction"][-1] * 100.0 + o_plus["acidic_fraction"][-1] * 10.0
        loss_minus = o_minus["hmw_fraction"][-1] * 100.0 + o_minus["lmw_fraction"][-1] * 100.0 + o_minus["acidic_fraction"][-1] * 10.0
        fd_grad = float((loss_plus - loss_minus) / (2.0 * eps))
        rel_grad_err = abs(g_val - fd_grad) / max(abs(fd_grad), 1e-12)
        assert rel_grad_err < 1e-4, f"Autograd vs FD mismatch on {k}: autograd={g_val}, fd={fd_grad}"


def test_pathway_registry_extensibility():
    """Verifies adding a new pathway to PATHWAY_REGISTRY does not require touching the solver."""
    register_pathway(
        "TEST_OXIDATION",
        lambda temp_k, params: {"k_eff_fraction_per_h": 1e-7 * (temp_k / T_REF_K)},
    )
    try:
        res = evaluate_pathway_rates(T_REF_K, {}, pathways=("TEST_OXIDATION",))
        assert math.isclose(float(res["TEST_OXIDATION"]["k_eff_fraction_per_h"]), 1e-7, rel_tol=1e-12)
    finally:
        PATHWAY_REGISTRY.pop("TEST_OXIDATION", None)
