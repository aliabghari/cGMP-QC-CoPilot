"""
Hybrid Mechanistic + Neural-Network Kinetic Modeling Package (Agent 2).
Classification: Non-GMP Decision Support Prototype on Synthetic Data.
"""

from .config import load_hybrid_config, get_config_sha256, DEFAULT_CONFIG_PATH
from .physics import (
    KB_SI,
    H_PLANCK_SI,
    R_GAS_SI,
    SECONDS_PER_HOUR,
    HOURS_PER_MONTH,
    T_REF_K,
    celsius_to_kelvin,
    kelvin_to_celsius,
    eyring_rate_from_dG,
    eyring_thermodynamics,
    dS_from_dG_dH,
    dG_from_dH_dS,
    eyring_to_arrhenius_ea,
    arrhenius_rate,
    apparent_activation_energy,
    vant_hoff_unfolding,
    lumry_eyring_rates,
    evaluate_pathway_rates,
    PATHWAY_REGISTRY,
    register_pathway,
    compute_excursion_detection_gate_pct,
    classify_excursion_verdict,
    compute_amendment_ab_summary,
)
from .ode import (
    interpolate_temperature_profile,
    integrate_pathway_rk4,
    integrate_3state_radau_oracle,
    closed_form_isothermal_trajectory,
)
from .excursion import (
    evaluate_thermal_excursion,
    SEC_REVERSIBLE_VS_IRREVERSIBLE_CAVEAT,
    PRIOR_DRIVEN_FU_CAVEAT,
)
from .oot import evaluate_lot_oot_with_adaptation
from .q1e import (
    evaluate_q1e_shelf_life_comparison,
    run_ich_q1e_linear_ancova_poolability,
    ICH_Q5C_Q1E_REGULATORY_NOTE,
)

__all__ = [
    "load_hybrid_config",
    "get_config_sha256",
    "DEFAULT_CONFIG_PATH",
    "KB_SI",
    "H_PLANCK_SI",
    "R_GAS_SI",
    "SECONDS_PER_HOUR",
    "HOURS_PER_MONTH",
    "T_REF_K",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "eyring_rate_from_dG",
    "eyring_thermodynamics",
    "dS_from_dG_dH",
    "dG_from_dH_dS",
    "eyring_to_arrhenius_ea",
    "arrhenius_rate",
    "apparent_activation_energy",
    "vant_hoff_unfolding",
    "lumry_eyring_rates",
    "evaluate_pathway_rates",
    "PATHWAY_REGISTRY",
    "register_pathway",
    "compute_excursion_detection_gate_pct",
    "classify_excursion_verdict",
    "compute_amendment_ab_summary",
    "interpolate_temperature_profile",
    "integrate_pathway_rk4",
    "integrate_3state_radau_oracle",
    "closed_form_isothermal_trajectory",
    "evaluate_thermal_excursion",
    "SEC_REVERSIBLE_VS_IRREVERSIBLE_CAVEAT",
    "PRIOR_DRIVEN_FU_CAVEAT",
    "evaluate_lot_oot_with_adaptation",
    "evaluate_q1e_shelf_life_comparison",
    "run_ich_q1e_linear_ancova_poolability",
    "ICH_Q5C_Q1E_REGULATORY_NOTE",
]
