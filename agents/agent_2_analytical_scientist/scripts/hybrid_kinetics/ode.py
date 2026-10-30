"""
Differentiable RK4 ODE Integrator (PyTorch + NumPy parity) and 3-State Stiff Radau Validation Oracle.

All internal state variables (M, N, U, HMW, LMW, Acidic) are integrated in FRACTION [0.0, 1.0],
where N(t) + U(t) + HMW(t) + LMW(t) = 1.000000 (100% mass balance).
All rate constants are in fraction / hour (h^-1).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    HAS_TORCH = False

from .physics import (
    _exp,
    _is_tensor,
    celsius_to_kelvin,
    evaluate_pathway_rates,
    lumry_eyring_rates,
)


def interpolate_temperature_profile(
    t_query_h: Any,
    time_knots_h: Sequence[float] | np.ndarray,
    temp_knots_c: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Piecewise-linear interpolation of arbitrary logger temperature series T(t) (including ramps).
    Returns temperature in Celsius at `t_query_h`.
    """
    t_arr = np.asarray(t_query_h, dtype=float)
    tk = np.asarray(time_knots_h, dtype=float)
    tc = np.asarray(temp_knots_c, dtype=float)
    return np.interp(t_arr, tk, tc)


def build_excursion_profile_knots(
    baseline_temp_c: float = 4.0,
    excursion_temp_c: float = 28.0,
    excursion_duration_hours: float = 36.0,
    ramp_hours: float = 0.0,
    n_spikes: int = 1,
    total_hours: float = 120.0,
    start_hour: float = 24.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs piecewise-linear (time_knots_h, temp_knots_c) for single or repeated thermal excursions
    with configurable ramp duration.
    """
    if excursion_duration_hours <= 0.0 or n_spikes <= 0:
        return np.array([0.0, total_hours], dtype=float), np.array([baseline_temp_c, baseline_temp_c], dtype=float)

    knots_t = [0.0]
    knots_c = [baseline_temp_c]

    spike_dur = excursion_duration_hours / float(n_spikes)
    gap_dur = 6.0 if n_spikes > 1 else 0.0
    curr_t = start_hour

    for _ in range(n_spikes):
        t0 = curr_t
        t_up = t0 + max(0.0, min(ramp_hours, spike_dur * 0.45))
        t_plateau_end = t0 + spike_dur
        t_down = t_plateau_end + max(0.0, min(ramp_hours, spike_dur * 0.45))

        if ramp_hours <= 1e-9:
            knots_t.extend([t0 - 1e-7, t0, t_plateau_end - 1e-7, t_plateau_end])
            knots_c.extend([baseline_temp_c, excursion_temp_c, excursion_temp_c, baseline_temp_c])
            curr_t = t_plateau_end + gap_dur
        else:
            knots_t.extend([t0, t_up, t_plateau_end, t_down])
            knots_c.extend([baseline_temp_c, excursion_temp_c, excursion_temp_c, baseline_temp_c])
            curr_t = t_down + gap_dur

    if knots_t[-1] < total_hours:
        knots_t.append(total_hours)
        knots_c.append(baseline_temp_c)

    return np.asarray(knots_t, dtype=float), np.asarray(knots_c, dtype=float)


def _any_tensor_in_params(params: Dict[str, Any], init_vals: Sequence[Any]) -> bool:
    if any(_is_tensor(v) for v in init_vals):
        return True
    return any(_is_tensor(v) for v in params.values())


def integrate_pathway_rk4(
    time_grid_h: np.ndarray | Sequence[float],
    temp_profile_c: np.ndarray | Sequence[float] | None = None,
    time_knots_h: Sequence[float] | np.ndarray | None = None,
    temp_knots_c: Sequence[float] | np.ndarray | None = None,
    params: Dict[str, Any] | None = None,
    initial_hmw_fraction: Any = 0.0045,
    initial_lmw_fraction: Any = 0.0010,
    initial_acidic_fraction: Any = 0.1420,
    include_lmw: bool = False,
    include_acidic: bool = False,
) -> Dict[str, Any]:
    """
    Differentiable 4th-order Runge-Kutta (RK4) solver for the rapid pre-equilibrium Lumry-Eyring
    aggregation pathway (`SEC_HMW`), plus optional `SEC_LMW` and `CEX_ACIDIC` pathways:

        f_U(T)   = K_u(T) / (1 + K_u(T))
        dM/dt    = -(k_agg(T) * f_U(T) + k_frag(T)) * M
        dHMW/dt  =  k_agg(T) * f_U(T) * M
        dLMW/dt  =  k_frag(T) * M
        N(t)     = (1 - f_U(T(t))) * M(t)
        U(t)     =  f_U(T(t)) * M(t)

    Operates seamlessly on either PyTorch Tensors (preserving full autograd graph for training)
    or NumPy floats/arrays (for fast deterministic inference without PyTorch).
    Achieves relative parity < 1e-6 between PyTorch and NumPy (Amendment D).
    """
    if params is None:
        params = {}

    t_grid = np.asarray(time_grid_h, dtype=np.float64)
    n_steps = len(t_grid)
    if n_steps < 2:
        raise ValueError("time_grid_h must contain at least 2 timepoints.")

    if time_knots_h is not None and temp_knots_c is not None:
        t_mids = 0.5 * (t_grid[:-1] + t_grid[1:])
        temp_start_c = interpolate_temperature_profile(t_grid[:-1], time_knots_h, temp_knots_c)
        temp_mid_c = interpolate_temperature_profile(t_mids, time_knots_h, temp_knots_c)
        temp_end_c = interpolate_temperature_profile(t_grid[1:], time_knots_h, temp_knots_c)
        temp_grid_c = interpolate_temperature_profile(t_grid, time_knots_h, temp_knots_c)
    elif temp_profile_c is not None:
        temp_grid_c = np.asarray(temp_profile_c, dtype=np.float64)
        if len(temp_grid_c) != n_steps:
            raise ValueError("temp_profile_c must have the same length as time_grid_h.")
        temp_start_c = temp_grid_c[:-1]
        temp_end_c = temp_grid_c[1:]
        temp_mid_c = 0.5 * (temp_start_c + temp_end_c)
    else:
        raise ValueError("Provide either temp_profile_c or (time_knots_h, temp_knots_c).")

    use_torch = _any_tensor_in_params(
        params, [initial_hmw_fraction, initial_lmw_fraction, initial_acidic_fraction]
    )

    active_pathways = ["SEC_HMW"]
    if include_lmw:
        active_pathways.append("SEC_LMW")
    if include_acidic:
        active_pathways.append("CEX_ACIDIC")
    pw_tuple = tuple(active_pathways)

    if use_torch:
        device = next(
            (v.device for v in params.values() if _is_tensor(v)),
            torch.device("cpu"),
        )
        dtype = torch.float64
        t_start_k = torch.tensor(celsius_to_kelvin(temp_start_c), dtype=dtype, device=device)
        t_mid_k = torch.tensor(celsius_to_kelvin(temp_mid_c), dtype=dtype, device=device)
        t_end_k = torch.tensor(celsius_to_kelvin(temp_end_c), dtype=dtype, device=device)
        t_grid_k = torch.tensor(celsius_to_kelvin(temp_grid_c), dtype=dtype, device=device)
        dt_vec = torch.tensor(np.diff(t_grid), dtype=dtype, device=device)
    else:
        t_start_k = celsius_to_kelvin(temp_start_c)
        t_mid_k = celsius_to_kelvin(temp_mid_c)
        t_end_k = celsius_to_kelvin(temp_end_c)
        t_grid_k = celsius_to_kelvin(temp_grid_c)
        dt_vec = np.diff(t_grid)

    rates_start = evaluate_pathway_rates(t_start_k, params, pathways=pw_tuple)
    rates_mid = evaluate_pathway_rates(t_mid_k, params, pathways=pw_tuple)
    rates_end = evaluate_pathway_rates(t_end_k, params, pathways=pw_tuple)
    rates_grid = evaluate_pathway_rates(t_grid_k, params, pathways=pw_tuple)

    k_hmw_s = rates_start["SEC_HMW"]["k_eff_fraction_per_h"]
    k_hmw_m = rates_mid["SEC_HMW"]["k_eff_fraction_per_h"]
    k_hmw_e = rates_end["SEC_HMW"]["k_eff_fraction_per_h"]

    if include_lmw:
        k_lmw_s = rates_start["SEC_LMW"]["k_eff_fraction_per_h"]
        k_lmw_m = rates_mid["SEC_LMW"]["k_eff_fraction_per_h"]
        k_lmw_e = rates_end["SEC_LMW"]["k_eff_fraction_per_h"]
    else:
        k_lmw_s = 0.0 * k_hmw_s
        k_lmw_m = 0.0 * k_hmw_m
        k_lmw_e = 0.0 * k_hmw_e

    if include_acidic:
        k_ac_s = rates_start["CEX_ACIDIC"]["k_eff_fraction_per_h"]
        k_ac_m = rates_mid["CEX_ACIDIC"]["k_eff_fraction_per_h"]
        k_ac_e = rates_end["CEX_ACIDIC"]["k_eff_fraction_per_h"]
    else:
        k_ac_s = 0.0 * k_hmw_s
        k_ac_m = 0.0 * k_hmw_m
        k_ac_e = 0.0 * k_hmw_e

    if use_torch:
        hmw0 = (
            initial_hmw_fraction.to(dtype=torch.float64, device=device)
            if _is_tensor(initial_hmw_fraction)
            else torch.tensor(float(initial_hmw_fraction), dtype=torch.float64, device=device)
        )
        lmw0 = (
            initial_lmw_fraction.to(dtype=torch.float64, device=device)
            if _is_tensor(initial_lmw_fraction)
            else torch.tensor(float(initial_lmw_fraction) if include_lmw else 0.0, dtype=torch.float64, device=device)
        )
        ac0 = (
            initial_acidic_fraction.to(dtype=torch.float64, device=device)
            if _is_tensor(initial_acidic_fraction)
            else torch.tensor(float(initial_acidic_fraction), dtype=torch.float64, device=device)
        )
    else:
        hmw0 = float(initial_hmw_fraction)
        lmw0 = float(initial_lmw_fraction) if include_lmw else 0.0
        ac0 = float(initial_acidic_fraction)

    m0 = 1.0 - hmw0 - lmw0

    m_list = [m0]
    hmw_list = [hmw0]
    lmw_list = [lmw0]
    ac_list = [ac0]

    curr_m = m0
    curr_hmw = hmw0
    curr_lmw = lmw0
    curr_ac = ac0

    for i in range(n_steps - 1):
        dt = dt_vec[i]
        kh_1, kl_1, ka_1 = k_hmw_s[i], k_lmw_s[i], k_ac_s[i]
        kh_2, kl_2, ka_2 = k_hmw_m[i], k_lmw_m[i], k_ac_m[i]
        kh_4, kl_4, ka_4 = k_hmw_e[i], k_lmw_e[i], k_ac_e[i]

        ktot_1 = kh_1 + kl_1
        ktot_2 = kh_2 + kl_2
        ktot_4 = kh_4 + kl_4

        # RK4 Stage 1
        dm_1 = -ktot_1 * curr_m
        dhmw_1 = kh_1 * curr_m
        dlmw_1 = kl_1 * curr_m

        # RK4 Stage 2
        m_k2 = curr_m + 0.5 * dt * dm_1
        dm_2 = -ktot_2 * m_k2
        dhmw_2 = kh_2 * m_k2
        dlmw_2 = kl_2 * m_k2

        # RK4 Stage 3
        m_k3 = curr_m + 0.5 * dt * dm_2
        dm_3 = -ktot_2 * m_k3
        dhmw_3 = kh_2 * m_k3
        dlmw_3 = kl_2 * m_k3

        # RK4 Stage 4
        m_k4 = curr_m + dt * dm_3
        dm_4 = -ktot_4 * m_k4
        dhmw_4 = kh_4 * m_k4
        dlmw_4 = kl_4 * m_k4

        curr_m = curr_m + (dt / 6.0) * (dm_1 + 2.0 * dm_2 + 2.0 * dm_3 + dm_4)
        curr_hmw = curr_hmw + (dt / 6.0) * (dhmw_1 + 2.0 * dhmw_2 + 2.0 * dhmw_3 + dhmw_4)
        curr_lmw = curr_lmw + (dt / 6.0) * (dlmw_1 + 2.0 * dlmw_2 + 2.0 * dlmw_3 + dlmw_4)

        if include_acidic:
            dac_1 = ka_1 * (1.0 - curr_ac)
            ac_k2 = curr_ac + 0.5 * dt * dac_1
            dac_2 = ka_2 * (1.0 - ac_k2)
            ac_k3 = curr_ac + 0.5 * dt * dac_2
            dac_3 = ka_2 * (1.0 - ac_k3)
            ac_k4 = curr_ac + dt * dac_3
            dac_4 = ka_4 * (1.0 - ac_k4)
            curr_ac = curr_ac + (dt / 6.0) * (dac_1 + 2.0 * dac_2 + 2.0 * dac_3 + dac_4)

        m_list.append(curr_m)
        hmw_list.append(curr_hmw)
        lmw_list.append(curr_lmw)
        ac_list.append(curr_ac)

    if use_torch:
        monomer_frac = torch.stack(m_list)
        hmw_frac = torch.stack(hmw_list)
        lmw_frac = torch.stack(lmw_list)
        acidic_frac = torch.stack(ac_list)
    else:
        monomer_frac = np.asarray(m_list, dtype=np.float64)
        hmw_frac = np.asarray(hmw_list, dtype=np.float64)
        lmw_frac = np.asarray(lmw_list, dtype=np.float64)
        acidic_frac = np.asarray(ac_list, dtype=np.float64)

    fu_grid = rates_grid["SEC_HMW"]["f_U"]
    k_eff_grid = rates_grid["SEC_HMW"]["k_eff_fraction_per_h"]
    k_agg_grid = rates_grid["SEC_HMW"]["k_agg_fraction_per_h"]

    native_frac = (1.0 - fu_grid) * monomer_frac
    unfolded_frac = fu_grid * monomer_frac
    mass_balance_sum = native_frac + unfolded_frac + hmw_frac + lmw_frac

    return {
        "time_hours": t_grid,
        "temperature_c": temp_grid_c,
        "monomer_fraction": monomer_frac,
        "native_fraction": native_frac,
        "unfolded_fraction": unfolded_frac,
        "hmw_fraction": hmw_frac,
        "lmw_fraction": lmw_frac,
        "acidic_fraction": acidic_frac,
        "f_U": fu_grid,
        "k_eff_fraction_per_h": k_eff_grid,
        "k_agg_fraction_per_h": k_agg_grid,
        "mass_balance_fraction": mass_balance_sum,
        "rates_grid": rates_grid,
    }


def closed_form_isothermal_trajectory(
    time_grid_h: np.ndarray | Sequence[float],
    temp_c: float,
    params: Dict[str, Any] | None = None,
    initial_hmw_fraction: float = 0.0045,
    initial_lmw_fraction: float = 0.0010,
    include_lmw: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Exact closed-form analytical solution for constant temperature `temp_c`:
        M(t)   = M0 * exp(-(k_eff + k_frag) * t)
        HMW(t) = HMW0 + M0 * (k_eff / (k_eff + k_frag)) * (1 - exp(-(k_eff + k_frag) * t))
        LMW(t) = LMW0 + M0 * (k_frag / (k_eff + k_frag)) * (1 - exp(-(k_eff + k_frag) * t))
    Used to verify the RK4 integrator (< 1e-4 relative error).
    """
    if params is None:
        params = {}
    t_arr = np.asarray(time_grid_h, dtype=np.float64)
    temp_k = celsius_to_kelvin(temp_c)

    pws = ("SEC_HMW", "SEC_LMW") if include_lmw else ("SEC_HMW",)
    rates = evaluate_pathway_rates(temp_k, params, pathways=pws)
    k_eff = float(rates["SEC_HMW"]["k_eff_fraction_per_h"])
    fu = float(rates["SEC_HMW"]["f_U"])
    k_frag = float(rates["SEC_LMW"]["k_eff_fraction_per_h"]) if include_lmw else 0.0
    k_tot = k_eff + k_frag

    lmw0 = float(initial_lmw_fraction) if include_lmw else 0.0
    hmw0 = float(initial_hmw_fraction)
    m0 = 1.0 - hmw0 - lmw0

    exp_factor = _exp(-k_tot * t_arr)
    m_t = m0 * exp_factor
    if k_tot > 0.0:
        hmw_t = hmw0 + m0 * (k_eff / k_tot) * (1.0 - exp_factor)
        lmw_t = lmw0 + m0 * (k_frag / k_tot) * (1.0 - exp_factor)
    else:
        hmw_t = np.full_like(t_arr, hmw0)
        lmw_t = np.full_like(t_arr, lmw0)

    n_t = (1.0 - fu) * m_t
    u_t = fu * m_t

    return {
        "time_hours": t_arr,
        "monomer_fraction": m_t,
        "native_fraction": n_t,
        "unfolded_fraction": u_t,
        "hmw_fraction": hmw_t,
        "lmw_fraction": lmw_t,
        "mass_balance_fraction": n_t + u_t + hmw_t + lmw_t,
    }


def integrate_3state_radau_oracle(
    time_grid_h: np.ndarray | Sequence[float],
    time_knots_h: Sequence[float] | np.ndarray,
    temp_knots_c: Sequence[float] | np.ndarray,
    params: Dict[str, Any] | None = None,
    initial_hmw_fraction: float = 0.0045,
    initial_lmw_fraction: float = 0.0010,
    include_lmw: bool = False,
    k_unfold_relax_per_hour: float = 100.0,
) -> Dict[str, np.ndarray]:
    """
    Full 3-state stiff ODE validation oracle using `scipy.integrate.solve_ivp` with `Radau`:
        dN/dt   = -k_1(T) * N + k_-1 * U - k_frag(T) * N
        dU/dt   =  k_1(T) * N - k_-1 * U - (k_agg(T) + k_frag(T)) * U
        dHMW/dt =  k_agg(T) * U
        dLMW/dt =  k_frag(T) * (N + U)
    where k_1(T) = K_u(T) * k_-1.
    Used to verify the rapid pre-equilibrium approximation and 100% mass balance.
    """
    if params is None:
        params = {}

    t_arr = np.asarray(time_grid_h, dtype=np.float64)
    tk = np.asarray(time_knots_h, dtype=np.float64)
    tc = np.asarray(temp_knots_c, dtype=np.float64)
    k_rev = float(params.get("k_unfold_relax_per_hour", k_unfold_relax_per_hour))

    t0_c = float(np.interp(t_arr[0], tk, tc))
    r0 = lumry_eyring_rates(
        celsius_to_kelvin(t0_c),
        dG_eff_ref_kj_mol=params.get("dG_eff_ref_kj_mol", 121.066),
        dH_agg_ref_kj_mol=params.get("dH_agg_ref_kj_mol", 38.0),
        ku_ref_5c=params.get("ku_ref_5c", 0.0010),
        dH_u_kj_mol=params.get("dH_u_kj_mol", 34.0),
        dCp_agg_kj_mol_k=params.get("dCp_agg_kj_mol_k", 0.0),
        dCp_u_kj_mol_k=params.get("dCp_u_kj_mol_k", 0.0),
    )
    fu0 = float(r0["f_U"])
    hmw0 = float(initial_hmw_fraction)
    lmw0 = float(initial_lmw_fraction) if include_lmw else 0.0
    m0 = 1.0 - hmw0 - lmw0
    n0 = (1.0 - fu0) * m0
    u0 = fu0 * m0

    y0 = np.array([n0, u0, hmw0, lmw0], dtype=np.float64)
    pws = ("SEC_HMW", "SEC_LMW") if include_lmw else ("SEC_HMW",)

    def rhs(t_val: float, y: np.ndarray) -> np.ndarray:
        n_val, u_val, _, _ = y
        curr_c = float(np.interp(t_val, tk, tc))
        curr_k = celsius_to_kelvin(curr_c)
        pw_out = evaluate_pathway_rates(curr_k, params, pathways=pws)
        hmw_rates = pw_out["SEC_HMW"]
        ku_val = float(hmw_rates["K_u"])
        k_agg_val = float(hmw_rates["k_agg_fraction_per_h"])
        k_frag_val = float(pw_out["SEC_LMW"]["k_eff_fraction_per_h"]) if include_lmw else 0.0

        k_fwd = ku_val * k_rev
        dn = -k_fwd * n_val + k_rev * u_val - k_frag_val * n_val
        du = k_fwd * n_val - k_rev * u_val - (k_agg_val + k_frag_val) * u_val
        dhmw = k_agg_val * u_val
        dlmw = k_frag_val * (n_val + u_val)
        return np.array([dn, du, dhmw, dlmw], dtype=np.float64)

    sol = solve_ivp(
        rhs,
        t_span=(float(t_arr[0]), float(t_arr[-1])),
        y0=y0,
        t_eval=t_arr,
        method="Radau",
        rtol=1e-9,
        atol=1e-11,
        max_step=0.5,
    )
    if not sol.success:
        raise RuntimeError(f"Radau 3-state solver failed: {sol.message}")

    n_traj = sol.y[0]
    u_traj = sol.y[1]
    hmw_traj = sol.y[2]
    lmw_traj = sol.y[3]
    m_traj = n_traj + u_traj

    return {
        "time_hours": t_arr,
        "monomer_fraction": m_traj,
        "native_fraction": n_traj,
        "unfolded_fraction": u_traj,
        "hmw_fraction": hmw_traj,
        "lmw_fraction": lmw_traj,
        "f_U": u_traj / np.maximum(m_traj, 1e-15),
        "mass_balance_fraction": n_traj + u_traj + hmw_traj + lmw_traj,
    }
