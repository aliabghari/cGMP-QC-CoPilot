"""
Backward-Compatible Excursion & OOT Wrappers (`eyring_excursion.py`).
Agent 2: Principal Analytical Scientist & Hybrid Physics+ML Modeling Specialist.

Delegates all calculations to the real trained 25-member hybrid Lumry-Eyring + ANN ensemble
and split-conformal calibration modules (`hybrid_kinetics/physics.py`, `excursion.py`, `oot.py`):
- Fixes the legacy dead `delta_s_j_mol_k` bug by routing `eyring_rate_constant_per_hour()`
  directly to `physics.eyring_rate_constant()`.
- Preserves all dictionary keys and DataFrame column names expected by `app.py` and
  `comms_generator.py`.
- Replaces automatic disposition language (`"RELEASE FOR USE"`) with the strictly bounded
  statistical detection verdicts (`"no detectable change"`, `"inconclusive"`,
  `"detectable change likely"`) and includes the SEC reversible-vs-irreversible and
  prior-driven `f_U` caveats.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .hybrid_kinetics.excursion import evaluate_thermal_excursion
from .hybrid_kinetics.oot import evaluate_lot_oot_with_adaptation
from .hybrid_kinetics.physics import celsius_to_kelvin, eyring_thermodynamics, lumry_eyring_rates


def eyring_rate_constant_per_hour(
    temp_celsius: float,
    delta_h_kj_mol: float = 38.0,
    delta_s_j_mol_k: float = -241.1869,
) -> float:
    """
    Computes the first-principles Eyring-Polanyi rate constant k(T) in [1/hour] using both
    `delta_h_kj_mol` and `delta_s_j_mol_k` via `hybrid_kinetics.physics.eyring_thermodynamics`.
    (Fixes the legacy bug where `delta_s_j_mol_k` was unused.)
    """
    t_k = celsius_to_kelvin(float(temp_celsius))
    k_val = eyring_thermodynamics(
        temp_k=t_k,
        dH_ref_kj_mol=float(delta_h_kj_mol),
        dS_ref_j_mol_k=float(delta_s_j_mol_k),
    )["k_fraction_per_h"]
    return float(np.asarray(k_val).item())


def calculate_eyring_rate(
    temp_celsius: float,
    delta_h_kj_mol: float = 38.0,
    delta_s_j_mol_k: float = -241.1869,
) -> float:
    """Alias for `eyring_rate_constant_per_hour`."""
    return eyring_rate_constant_per_hour(
        temp_celsius=temp_celsius,
        delta_h_kj_mol=delta_h_kj_mol,
        delta_s_j_mol_k=delta_s_j_mol_k,
    )


def ann_tvp_acceleration_factor(temp_celsius: float, hours_at_elevated_temp: float = 0.0) -> float:
    """
    Returns the Lumry-Eyring unfolded-state equilibrium ratio f_U(T) / f_U(5°C) at `temp_celsius`
    (replacing the legacy heuristic curve with the physical pre-equilibrium unfolding factor).
    """
    del hours_at_elevated_temp
    if temp_celsius <= 5.0:
        return 1.0
    r_ref = lumry_eyring_rates(celsius_to_kelvin(5.0))
    r_cur = lumry_eyring_rates(celsius_to_kelvin(float(temp_celsius)))
    fu_ref = float(np.asarray(r_ref["f_U"]).item())
    fu_cur = float(np.asarray(r_cur["f_U"]).item())
    return float(fu_cur / max(fu_ref, 1e-15))


def simulate_cold_chain_excursion(
    baseline_temp_c: float = 5.0,
    excursion_temp_c: float = 28.0,
    excursion_duration_hours: float = 36.0,
    initial_hmw_pct: float = 0.62,
    spec_limit_hmw_pct: float = 2.00,
    ramp_hours: float = 0.0,
    n_spikes: int = 1,
    lot_covariates: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper around `hybrid_kinetics.excursion.evaluate_thermal_excursion()`.
    Returns `df_trajectory` and `summary` with all legacy and new Phase 5 keys.
    """
    cov = dict(lot_covariates or {})
    cov["initial_hmw_pct"] = float(initial_hmw_pct)
    cov.setdefault("initial_lmw_pct", 0.0)
    return evaluate_thermal_excursion(
        baseline_temp_c=baseline_temp_c,
        excursion_temp_c=excursion_temp_c,
        excursion_duration_hours=excursion_duration_hours,
        ramp_hours=ramp_hours,
        n_spikes=n_spikes,
        lot_covariates=cov,
        spec_limit_hmw_pct=spec_limit_hmw_pct,
    )


def simulate_excursion(
    baseline_temp_c: float = 5.0,
    excursion_temp_c: float = 28.0,
    excursion_duration_hours: float = 36.0,
    initial_hmw_pct: float = 0.62,
    spec_limit_hmw_pct: float = 2.00,
) -> Dict[str, Any]:
    """Alias for `simulate_cold_chain_excursion`."""
    return simulate_cold_chain_excursion(
        baseline_temp_c=baseline_temp_c,
        excursion_temp_c=excursion_temp_c,
        excursion_duration_hours=excursion_duration_hours,
        initial_hmw_pct=initial_hmw_pct,
        spec_limit_hmw_pct=spec_limit_hmw_pct,
    )


def analyze_6m_oot_stability(
    early_cutoff_months: float = 3.0,
    spec_limit_hmw_pct: float = 2.00,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper around `hybrid_kinetics.oot.evaluate_lot_oot_with_adaptation()`.
    Uses early-pull (0, 1, 3M) adaptation through the ODE on `NS-2026-004` without hard-coded forecasts.
    """
    return evaluate_lot_oot_with_adaptation(
        lot_df=None,
        early_cutoff_months=early_cutoff_months,
        spec_limit_hmw_pct=spec_limit_hmw_pct,
    )


if __name__ == "__main__":
    exc = simulate_cold_chain_excursion()
    print("Excursion Summary:", exc["summary"])
    oot = analyze_6m_oot_stability()
    print("OOT Diagnostics:", oot["oot_diagnostics"])
