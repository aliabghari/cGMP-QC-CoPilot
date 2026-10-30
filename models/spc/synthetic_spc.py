"""
models/spc/synthetic_spc.py

Synthetic Quality Data Generators for cGMP-QC-CoPilot SPC Module.
Provides deterministic, seeded generators for 4 representative biologics QC series:
1. Assay Purity (% monomer by SEC-HPLC, continuous individual observations)
2. Right-First-Time rate (% RFT attribute proportion, variable sample sizes)
3. Documentation error counts (c-chart Poisson defect counts per batch record)
4. Analyst variability (subgrouped X-bar/R, e.g. n=4 replicates across runs)

Supports plantable signals:
- Mean shift (+1.0 sigma)
- Linear drift (+0.05 sigma / step)
- Variance inflation (1.8x sigma)
- In-control baseline
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.stamps import DRAFT_STAMP, assert_no_banned_phrases


def generate_purity_series(
    n_points: int = 40,
    seed: int = 42,
    nominal_mean: float = 98.2,
    sigma: float = 0.15,
    signal_type: str = "none",  # "none", "mean_shift", "drift", "variance_inflation"
    signal_start_idx: int = 25,
) -> Dict[str, Any]:
    """
    Generates synthetic SEC-HPLC % Monomer purity series.
    """
    rng = np.random.RandomState(seed)
    base_noise = rng.normal(0.0, sigma, n_points)
    values = nominal_mean + base_noise

    applied_description = "In-control process (common cause variation only)"

    if signal_type == "mean_shift":
        shift_magnitude = 1.0 * sigma
        values[signal_start_idx:] += shift_magnitude
        applied_description = f"Planted +1.0 sigma ({shift_magnitude:.3f}%) mean shift starting at point {signal_start_idx+1}"

    elif signal_type == "drift":
        drift_rate = 0.05 * sigma
        steps = np.arange(n_points - signal_start_idx)
        values[signal_start_idx:] += steps * drift_rate
        applied_description = f"Planted systematic drift (+0.05 sigma/step = +{drift_rate:.4f}%/step) starting at point {signal_start_idx+1}"

    elif signal_type == "variance_inflation":
        inflated_noise = rng.normal(0.0, sigma * 1.8, n_points - signal_start_idx)
        values[signal_start_idx:] = nominal_mean + inflated_noise
        applied_description = f"Planted 1.8x variance inflation starting at point {signal_start_idx+1}"

    # Round to realistic chromatographic precision (2 decimals)
    values = np.round(values, 3)

    out = {
        "series_id": "SEC-HPLC-MONOMER",
        "name": "SEC-HPLC % Monomer Purity (synthetic example)",
        "unit": "%",
        "n_points": n_points,
        "nominal_mean": nominal_mean,
        "nominal_sigma": sigma,
        "signal_type": signal_type,
        "signal_start_idx": signal_start_idx if signal_type != "none" else None,
        "signal_description": applied_description,
        "values": values.tolist(),
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out


def generate_rft_series(
    n_periods: int = 30,
    seed: int = 43,
    nominal_p: float = 0.08,  # 8% non-RFT rate (i.e. 92% RFT)
    min_n: int = 25,
    max_n: int = 60,
    signal_type: str = "none",  # "none", "mean_shift", "drift"
    signal_start_idx: int = 20,
) -> Dict[str, Any]:
    """
    Generates synthetic Right-First-Time (RFT) defect proportions with variable batch inspection sizes.
    """
    rng = np.random.RandomState(seed)
    sample_sizes = rng.randint(min_n, max_n + 1, size=n_periods)
    p_targets = np.full(n_periods, nominal_p)

    applied_description = "In-control RFT process (common cause variation)"

    if signal_type == "mean_shift":
        p_targets[signal_start_idx:] = min(0.35, nominal_p + 0.10)  # shift to 18% non-RFT
        applied_description = f"Planted non-RFT rate increase (+10% absolute) starting at period {signal_start_idx+1}"

    elif signal_type == "drift":
        drift = np.linspace(0.0, 0.12, n_periods - signal_start_idx)
        p_targets[signal_start_idx:] = np.minimum(0.35, nominal_p + drift)
        applied_description = f"Planted deteriorating RFT trend starting at period {signal_start_idx+1}"

    defect_counts = np.array([
        rng.binomial(n_i, p_i) for n_i, p_i in zip(sample_sizes, p_targets)
    ])

    out = {
        "series_id": "RFT-DEFECTS",
        "name": "Right-First-Time (RFT) Non-Conformance Proportion (synthetic example)",
        "n_periods": n_periods,
        "nominal_p": nominal_p,
        "sample_sizes": sample_sizes.tolist(),
        "defect_counts": defect_counts.tolist(),
        "signal_type": signal_type,
        "signal_start_idx": signal_start_idx if signal_type != "none" else None,
        "signal_description": applied_description,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out


def generate_doc_errors_series(
    n_batches: int = 35,
    seed: int = 44,
    nominal_lambda: float = 1.8,
    signal_type: str = "none",  # "none", "mean_shift", "drift"
    signal_start_idx: int = 22,
) -> Dict[str, Any]:
    """
    Generates synthetic batch record documentation error counts (c-chart Poisson process).
    """
    rng = np.random.RandomState(seed)
    lambdas = np.full(n_batches, nominal_lambda)

    applied_description = "In-control documentation review errors (Poisson lambda = 1.8)"

    if signal_type == "mean_shift":
        lambdas[signal_start_idx:] = nominal_lambda + 2.5
        applied_description = f"Planted documentation error spike (+2.5 errors/batch) starting at batch {signal_start_idx+1}"

    elif signal_type == "drift":
        drift = np.linspace(0.0, 3.0, n_batches - signal_start_idx)
        lambdas[signal_start_idx:] = nominal_lambda + drift
        applied_description = f"Planted creeping documentation error trend (+3.0 errors max) starting at batch {signal_start_idx+1}"

    counts = rng.poisson(lambdas)

    out = {
        "series_id": "DOC-ERRORS",
        "name": "Batch Record Documentation Errors per Unit (synthetic example)",
        "n_batches": n_batches,
        "nominal_lambda": nominal_lambda,
        "counts": counts.tolist(),
        "signal_type": signal_type,
        "signal_start_idx": signal_start_idx if signal_type != "none" else None,
        "signal_description": applied_description,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out


def generate_analyst_subgroups(
    n_subgroups: int = 25,
    subgroup_size: int = 4,
    seed: int = 45,
    nominal_mean: float = 100.0,  # e.g. % Relative Potency
    sigma_within: float = 2.0,
    signal_type: str = "none",  # "none", "mean_shift", "variance_inflation"
    signal_start_idx: int = 15,
) -> Dict[str, Any]:
    """
    Generates rational subgroups of bioassay relative potency for X-bar/R chart.
    """
    rng = np.random.RandomState(seed)
    subgroups: List[List[float]] = []

    applied_description = "In-control bioassay relative potency subgroups"

    for i in range(n_subgroups):
        mu = nominal_mean
        sig = sigma_within

        if i >= signal_start_idx:
            if signal_type == "mean_shift":
                mu += 1.2 * sigma_within
            elif signal_type == "variance_inflation":
                sig *= 2.0

        sample = rng.normal(mu, sig, subgroup_size)
        subgroups.append(np.round(sample, 2).tolist())

    if signal_type == "mean_shift":
        applied_description = f"Planted subgroup mean shift (+1.2 sigma) starting at subgroup {signal_start_idx+1}"
    elif signal_type == "variance_inflation":
        applied_description = f"Planted within-subgroup variance inflation (2.0x sigma) starting at subgroup {signal_start_idx+1}"

    out = {
        "series_id": "BIOASSAY-POTENCY",
        "name": "Bioassay Relative Potency Subgroups (synthetic example)",
        "n_subgroups": n_subgroups,
        "subgroup_size": subgroup_size,
        "nominal_mean": nominal_mean,
        "nominal_sigma_within": sigma_within,
        "subgroups": subgroups,
        "signal_type": signal_type,
        "signal_start_idx": signal_start_idx if signal_type != "none" else None,
        "signal_description": applied_description,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out
