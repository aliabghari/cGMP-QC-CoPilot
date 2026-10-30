"""
models/spc/spc_engine.py

Core Statistical Process Control (SPC) calculation engine for cGMP-QC-CoPilot.
Provides:
- I-MR (Individual & Moving Range)
- X-bar / R (Subgrouped Mean & Range)
- p-chart (Variable sample size attribute proportion)
- c-chart (Count of defect items)
- EWMA (Exponentially Weighted Moving Average)
- CUSUM (Cumulative Sum)
- Nelson / Western Electric rules engine with plain-language diagnostic names
- Phase I (establish limits) vs Phase II (monitor with frozen limits)
- Process capability Cp / Cpk with analytical confidence intervals and normality warnings
- Single source of truth with DRAFT stamps and ALCOA+ audit logging
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

from core.stamps import DRAFT_STAMP, assert_no_banned_phrases


# Constant control chart factors
D2_N2 = 1.128  # d2 for n=2
D4_N2 = 3.267  # D4 for n=2
D3_N2 = 0.0    # D3 for n=2

A2_TABLE = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483}
D3_TABLE = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
D4_TABLE = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004}
D2_TABLE = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534}


def evaluate_nelson_rules(
    values: np.ndarray,
    center_line: float,
    sigma: float,
) -> List[Dict[str, Any]]:
    """
    Evaluates Nelson / Western Electric rules across a numeric sequence.
    Returns list of violations with index, rule number, and plain language name.
    """
    if sigma <= 0 or len(values) == 0:
        return []

    violations: List[Dict[str, Any]] = []
    n = len(values)
    z_scores = (values - center_line) / sigma

    for i in range(n):
        point_violations = []

        # Rule 1: Point beyond Zone A (> 3 sigma)
        if abs(z_scores[i]) > 3.0:
            point_violations.append({
                "rule": "Rule 1",
                "name": "Point beyond 3-sigma action limit (gross process excursion)",
                "severity": "CRITICAL",
            })

        # Rule 2: 9 points in a row on same side of center line
        if i >= 8:
            window = z_scores[i - 8 : i + 1]
            if np.all(window > 0) or np.all(window < 0):
                point_violations.append({
                    "rule": "Rule 2",
                    "name": "9 consecutive points on same side of center line (process level shift)",
                    "severity": "WARNING",
                })

        # Rule 3: 6 points in a row steadily increasing or decreasing
        if i >= 5:
            w = values[i - 5 : i + 1]
            diffs = np.diff(w)
            if np.all(diffs > 0) or np.all(diffs < 0):
                point_violations.append({
                    "rule": "Rule 3",
                    "name": "6 consecutive points steadily increasing or decreasing (systematic drift)",
                    "severity": "WARNING",
                })

        # Rule 4: 14 points in a row alternating up and down
        if i >= 13:
            w = values[i - 13 : i + 1]
            diffs = np.diff(w)
            signs = np.sign(diffs)
            if len(signs) == 13 and np.all(signs[:-1] * signs[1:] == -1):
                point_violations.append({
                    "rule": "Rule 4",
                    "name": "14 consecutive points alternating direction (systematic oscillation / two alternating skids)",
                    "severity": "INFO",
                })

        if point_violations:
            for v in point_violations:
                violations.append({
                    "index": i,
                    "value": float(values[i]),
                    "z_score": float(z_scores[i]),
                    **v,
                })

    return violations


def calculate_imr_chart(
    data: np.ndarray,
    frozen_limits: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Computes Individual and Moving Range (I-MR) control chart.
    """
    x = np.asarray(data, dtype=float)
    if len(x) < 2:
        raise ValueError("I-MR chart requires at least 2 data points")

    mr = np.abs(np.diff(x))

    if frozen_limits is not None:
        cl_x = frozen_limits["center_line_x"]
        ucl_x = frozen_limits["ucl_x"]
        lcl_x = frozen_limits["lcl_x"]
        cl_mr = frozen_limits["center_line_mr"]
        ucl_mr = frozen_limits["ucl_mr"]
        lcl_mr = frozen_limits.get("lcl_mr", 0.0)
        sigma = (ucl_x - cl_x) / 3.0
        mode = "Phase II (Monitored with Frozen Limits)"
    else:
        cl_x = float(np.mean(x))
        cl_mr = float(np.mean(mr))
        sigma = cl_mr / D2_N2
        ucl_x = cl_x + 3.0 * sigma
        lcl_x = cl_x - 3.0 * sigma
        ucl_mr = D4_N2 * cl_mr
        lcl_mr = 0.0
        mode = "Phase I (Baseline Establishment)"

    rules_x = evaluate_nelson_rules(x, center_line=cl_x, sigma=sigma)
    mr_padded = np.insert(mr, 0, np.nan)

    out = {
        "chart_type": "I-MR",
        "mode": mode,
        "n_points": len(x),
        "center_line_x": round(cl_x, 4),
        "ucl_x": round(ucl_x, 4),
        "lcl_x": round(lcl_x, 4),
        "center_line_mr": round(cl_mr, 4),
        "ucl_mr": round(ucl_mr, 4),
        "lcl_mr": round(lcl_mr, 4),
        "estimated_sigma": round(sigma, 4),
        "x_values": x.tolist(),
        "mr_values": [None if np.isnan(v) else round(float(v), 4) for v in mr_padded],
        "violations_x": rules_x,
        "in_control": len(rules_x) == 0,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out


def calculate_xbar_r_chart(
    subgroups: List[np.ndarray],
    frozen_limits: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Computes X-bar and Range (X-bar/R) control chart for rational subgroups.
    """
    if len(subgroups) < 2:
        raise ValueError("X-bar/R chart requires at least 2 subgroups")

    n_sub = len(subgroups[0])
    if n_sub not in A2_TABLE:
        n_sub = min(max(n_sub, 2), 6)

    means = np.array([float(np.mean(s)) for s in subgroups])
    ranges = np.array([float(np.ptp(s)) for s in subgroups])

    a2 = A2_TABLE[n_sub]
    d3 = D3_TABLE[n_sub]
    d4 = D4_TABLE[n_sub]
    d2 = D2_TABLE[n_sub]

    if frozen_limits is not None:
        cl_x = frozen_limits["center_line_x"]
        ucl_x = frozen_limits["ucl_x"]
        lcl_x = frozen_limits["lcl_x"]
        cl_r = frozen_limits["center_line_r"]
        ucl_r = frozen_limits["ucl_r"]
        lcl_r = frozen_limits.get("lcl_r", 0.0)
        sigma = cl_r / d2
        mode = "Phase II (Monitored with Frozen Limits)"
    else:
        cl_x = float(np.mean(means))
        cl_r = float(np.mean(ranges))
        ucl_x = cl_x + a2 * cl_r
        lcl_x = cl_x - a2 * cl_r
        ucl_r = d4 * cl_r
        lcl_r = d3 * cl_r
        sigma = cl_r / d2
        mode = "Phase I (Baseline Establishment)"

    subgroup_sigma = sigma / math.sqrt(n_sub)
    rules_x = evaluate_nelson_rules(means, center_line=cl_x, sigma=subgroup_sigma)

    out = {
        "chart_type": "X-bar/R",
        "mode": mode,
        "subgroup_size": n_sub,
        "n_subgroups": len(subgroups),
        "center_line_x": round(cl_x, 4),
        "ucl_x": round(ucl_x, 4),
        "lcl_x": round(lcl_x, 4),
        "center_line_r": round(cl_r, 4),
        "ucl_r": round(ucl_r, 4),
        "lcl_r": round(lcl_r, 4),
        "estimated_process_sigma": round(sigma, 4),
        "means": [round(float(m), 4) for m in means],
        "ranges": [round(float(r), 4) for r in ranges],
        "violations_x": rules_x,
        "in_control": len(rules_x) == 0,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out


def calculate_p_chart(
    defect_counts: np.ndarray,
    sample_sizes: np.ndarray,
    frozen_p: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes p-chart for attribute proportions with variable sample sizes.
    """
    k = len(defect_counts)
    if k != len(sample_sizes):
        raise ValueError("Defect counts and sample sizes must have matching length")

    p_i = defect_counts / np.maximum(sample_sizes, 1)

    if frozen_p is not None:
        p_bar = float(frozen_p)
        mode = "Phase II (Monitored with Frozen Limits)"
    else:
        p_bar = float(np.sum(defect_counts) / np.sum(sample_sizes))
        mode = "Phase I (Baseline Establishment)"

    ucl_i = []
    lcl_i = []
    violations = []

    for idx, (p_val, n_i) in enumerate(zip(p_i, sample_sizes)):
        se_i = math.sqrt((p_bar * (1.0 - p_bar)) / max(n_i, 1))
        ucl = min(1.0, p_bar + 3.0 * se_i)
        lcl = max(0.0, p_bar - 3.0 * se_i)
        ucl_i.append(ucl)
        lcl_i.append(lcl)

        if p_val > ucl:
            violations.append({
                "index": idx,
                "value": round(float(p_val), 4),
                "ucl": round(float(ucl), 4),
                "rule": "Upper Exceedance",
                "name": "Defect proportion exceeds upper 3-sigma control limit",
            })
        elif p_val < lcl and lcl > 0.0:
            violations.append({
                "index": idx,
                "value": round(float(p_val), 4),
                "lcl": round(float(lcl), 4),
                "rule": "Lower Exceedance",
                "name": "Defect proportion dropped below lower 3-sigma limit (significant process improvement)",
            })

    out = {
        "chart_type": "p-chart",
        "mode": mode,
        "n_samples": k,
        "center_line_p": round(p_bar, 4),
        "proportions": [round(float(p), 4) for p in p_i],
        "ucl_per_sample": [round(float(u), 4) for u in ucl_i],
        "lcl_per_sample": [round(float(l), 4) for l in lcl_i],
        "violations": violations,
        "in_control": len(violations) == 0,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out


def calculate_c_chart(
    counts: np.ndarray,
    frozen_c: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes c-chart for count of defects per unit.
    """
    c_arr = np.asarray(counts, dtype=float)
    if len(c_arr) < 2:
        raise ValueError("c-chart requires at least 2 inspection units")

    if frozen_c is not None:
        c_bar = float(frozen_c)
        mode = "Phase II (Monitored with Frozen Limits)"
    else:
        c_bar = float(np.mean(c_arr))
        mode = "Phase I (Baseline Establishment)"

    ucl = c_bar + 3.0 * math.sqrt(c_bar)
    lcl = max(0.0, c_bar - 3.0 * math.sqrt(c_bar))

    violations = []
    for idx, c_val in enumerate(c_arr):
        if c_val > ucl:
            violations.append({
                "index": idx,
                "value": float(c_val),
                "ucl": round(ucl, 2),
                "rule": "Upper Exceedance",
                "name": "Defect count exceeds upper 3-sigma limit",
            })
        elif c_val < lcl and lcl > 0.0:
            violations.append({
                "index": idx,
                "value": float(c_val),
                "lcl": round(lcl, 2),
                "rule": "Lower Exceedance",
                "name": "Defect count below lower limit (inspection improvement or under-reporting)",
            })

    out = {
        "chart_type": "c-chart",
        "mode": mode,
        "n_units": len(c_arr),
        "center_line_c": round(c_bar, 2),
        "ucl": round(ucl, 2),
        "lcl": round(lcl, 2),
        "counts": [float(c) for c in c_arr],
        "violations": violations,
        "in_control": len(violations) == 0,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out


def calculate_ewma_chart(
    data: np.ndarray,
    lambda_: float = 0.2,
    L: float = 3.0,
    target: Optional[float] = None,
    sigma: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes Exponentially Weighted Moving Average (EWMA) control chart.
    Optimized for small persistent shifts (0.5 to 1.5 sigma).
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("EWMA chart requires at least 2 data points")

    mu_0 = float(np.mean(x)) if target is None else float(target)
    sig_0 = float(np.mean(np.abs(np.diff(x))) / D2_N2) if sigma is None else float(sigma)

    z = np.zeros(n)
    ucl = np.zeros(n)
    lcl = np.zeros(n)

    prev_z = mu_0
    violations = []

    for i in range(n):
        curr_z = lambda_ * x[i] + (1.0 - lambda_) * prev_z
        z[i] = curr_z
        prev_z = curr_z

        se_i = sig_0 * math.sqrt((lambda_ / (2.0 - lambda_)) * (1.0 - math.pow(1.0 - lambda_, 2 * (i + 1))))
        ucl[i] = mu_0 + L * se_i
        lcl[i] = mu_0 - L * se_i

        if curr_z > ucl[i]:
            violations.append({
                "index": i,
                "z_value": round(float(curr_z), 4),
                "ucl": round(float(ucl[i]), 4),
                "rule": "EWMA Upper Shift",
                "name": f"EWMA statistic breached upper limit at point {i+1} (small sustained shift)",
            })
        elif curr_z < lcl[i]:
            violations.append({
                "index": i,
                "z_value": round(float(curr_z), 4),
                "lcl": round(float(lcl[i]), 4),
                "rule": "EWMA Lower Shift",
                "name": f"EWMA statistic breached lower limit at point {i+1} (small sustained shift)",
            })

    out = {
        "chart_type": "EWMA",
        "lambda": lambda_,
        "L": L,
        "target": round(mu_0, 4),
        "sigma": round(sig_0, 4),
        "ewma_values": [round(float(v), 4) for v in z],
        "ucl": [round(float(u), 4) for u in ucl],
        "lcl": [round(float(l), 4) for l in lcl],
        "violations": violations,
        "in_control": len(violations) == 0,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out


def calculate_cusum_chart(
    data: np.ndarray,
    target: Optional[float] = None,
    sigma: Optional[float] = None,
    k_sig: float = 0.5,
    h_sig: float = 4.0,
) -> Dict[str, Any]:
    """
    Computes Tabular Cumulative Sum (CUSUM) control chart.
    Sensitive to persistent shifts of size delta = 2 * k_sig * sigma.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("CUSUM chart requires at least 2 data points")

    mu_0 = float(np.mean(x)) if target is None else float(target)
    sig_0 = float(np.mean(np.abs(np.diff(x))) / D2_N2) if sigma is None else float(sigma)

    k = k_sig * sig_0
    h = h_sig * sig_0

    c_plus = np.zeros(n)
    c_minus = np.zeros(n)
    violations = []

    for i in range(n):
        prev_plus = c_plus[i - 1] if i > 0 else 0.0
        prev_minus = c_minus[i - 1] if i > 0 else 0.0

        c_plus[i] = max(0.0, x[i] - (mu_0 + k) + prev_plus)
        c_minus[i] = max(0.0, (mu_0 - k) - x[i] + prev_minus)

        if c_plus[i] > h:
            violations.append({
                "index": i,
                "c_plus": round(float(c_plus[i]), 4),
                "decision_interval_h": round(h, 4),
                "rule": "CUSUM Upper Alarm",
                "name": f"Upper CUSUM breached decision threshold h={h:.2f} at point {i+1}",
            })
        if c_minus[i] > h:
            violations.append({
                "index": i,
                "c_minus": round(float(c_minus[i]), 4),
                "decision_interval_h": round(h, 4),
                "rule": "CUSUM Lower Alarm",
                "name": f"Lower CUSUM breached decision threshold h={h:.2f} at point {i+1}",
            })

    out = {
        "chart_type": "CUSUM",
        "target": round(mu_0, 4),
        "sigma": round(sig_0, 4),
        "k_reference": round(k, 4),
        "h_threshold": round(h, 4),
        "c_plus": [round(float(v), 4) for v in c_plus],
        "c_minus": [round(float(v), 4) for v in c_minus],
        "violations": violations,
        "in_control": len(violations) == 0,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out


def calculate_process_capability(
    data: np.ndarray,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Computes Cp and Cpk capability indices with Chou et al. confidence bounds.
    Includes explicit normality test (Shapiro-Wilk) and sample size warnings.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n < 5:
        raise ValueError("Capability analysis requires at least 5 data points")

    if lsl is None and usl is None:
        raise ValueError("Must specify at least LSL or USL")

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))

    # Normality test
    warnings: List[str] = []
    shapiro_stat, shapiro_p = stats.shapiro(x) if n <= 5000 else (1.0, 1.0)
    is_normal = shapiro_p >= 0.05
    if not is_normal:
        warnings.append(
            f"Non-normal data detected (Shapiro-Wilk p = {shapiro_p:.4f} < 0.05). "
            "Normal-theory Cp/Cpk indices and confidence intervals may be misleading."
        )
    if n < 30:
        warnings.append(
            f"Small sample size (n = {n} < 30). Capability estimates exhibit wide estimation variance."
        )

    cp = None
    cpk = None
    cpk_ci = None

    if lsl is not None and usl is not None:
        cp = (usl - lsl) / (6.0 * sigma)

    cpu = (usl - mu) / (3.0 * sigma) if usl is not None else float("inf")
    cpl = (mu - lsl) / (3.0 * sigma) if lsl is not None else float("inf")
    cpk = min(cpu, cpl)

    # Chou et al. approximation for Cpk confidence bounds
    z_alpha = stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    se_cpk = math.sqrt((1.0 / (9.0 * n)) + (cpk * cpk) / (2.0 * (n - 1)))
    cpk_lower = max(0.0, cpk - z_alpha * se_cpk)
    cpk_upper = cpk + z_alpha * se_cpk
    cpk_ci = (round(cpk_lower, 3), round(cpk_upper, 3))

    out = {
        "sample_size": n,
        "mean": round(mu, 4),
        "std_dev": round(sigma, 4),
        "lsl": lsl,
        "usl": usl,
        "cp": round(cp, 3) if cp is not None else None,
        "cpk": round(cpk, 3),
        "cpk_confidence_interval_95": cpk_ci,
        "normality_test": {
            "test_name": "Shapiro-Wilk",
            "statistic": round(float(shapiro_stat), 4),
            "p_value": round(float(shapiro_p), 4),
            "is_normal": is_normal,
        },
        "warnings": warnings,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out
