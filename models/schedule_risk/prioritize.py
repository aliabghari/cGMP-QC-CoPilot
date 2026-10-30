"""
cGMP-QC-CoPilot: Weighted Prioritization Matrix & Sensitivity Analysis.

Evaluates multi-criteria project priority:
1. Contract Penalty ($)
2. Project Tier (Tier 1A/1B, Tier 2, Tier 3)
3. Remaining Float (Inverse: lower float -> higher priority)
4. Critical-Path Dependencies (Downstream impact count)
5. Patient-Impact Flag (Direct clinical/commercial supply)

Sensitivity & Robustness:
- Dirichlet weight perturbation sampling
- Probability each project remains top-ranked
- Kendall's Tau rank correlation vs baseline
- Pareto frontier identification for conflicting criteria
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


DEFAULT_WEIGHTS = {
    "contract_penalty": 0.25,
    "project_tier": 0.25,
    "remaining_float": 0.20,
    "critical_path_dep": 0.15,
    "patient_impact": 0.15,
}

TIER_SCORES = {
    "Tier 1A": 1.0,
    "Tier 1B": 0.85,
    "Tier 2": 0.50,
    "Tier 3": 0.25,
}

DOMINANT_SCENARIO_PROJECTS: List[Dict[str, Any]] = [
    {"project_id": "PRJ-DOM-01", "name": "Spesolimab Lot Release", "contract_penalty": 100000.0, "priority_tier": "Tier 1A", "remaining_float": 0.0, "critical_path_dep": 5, "patient_impact": True},
    {"project_id": "PRJ-DOM-02", "name": "Bevacizumab CDMO Cont.", "contract_penalty": 50000.0, "priority_tier": "Tier 1B", "remaining_float": 1.0, "critical_path_dep": 3, "patient_impact": True},
    {"project_id": "PRJ-DOM-03", "name": "Adalimumab Biosimilar", "contract_penalty": 25000.0, "priority_tier": "Tier 2", "remaining_float": 3.0, "critical_path_dep": 2, "patient_impact": False},
    {"project_id": "PRJ-DOM-04", "name": "Internal Platform Study", "contract_penalty": 5000.0, "priority_tier": "Tier 3", "remaining_float": 7.0, "critical_path_dep": 1, "patient_impact": False},
]

CONFLICTING_SCENARIO_PROJECTS: List[Dict[str, Any]] = [
    # Project A: High financial contract penalty, but generous float and no direct patient impact
    {"project_id": "PRJ-CONF-A", "name": "CDMO High-Penalty Monomer", "contract_penalty": 100000.0, "priority_tier": "Tier 2", "remaining_float": 6.0, "critical_path_dep": 1, "patient_impact": False},
    # Project B: Critical clinical lot release, direct patient impact and zero float, but low financial contract penalty
    {"project_id": "PRJ-CONF-B", "name": "Commercial Pediatric Rescue", "contract_penalty": 10000.0, "priority_tier": "Tier 1A", "remaining_float": 0.0, "critical_path_dep": 4, "patient_impact": True},
    # Project C: Moderate on all criteria
    {"project_id": "PRJ-CONF-C", "name": "Phase III Registration Lot", "contract_penalty": 50000.0, "priority_tier": "Tier 1B", "remaining_float": 2.0, "critical_path_dep": 5, "patient_impact": False},
    # Project D: Routine stability
    {"project_id": "PRJ-CONF-D", "name": "Annual 25C Pull Study", "contract_penalty": 5000.0, "priority_tier": "Tier 3", "remaining_float": 10.0, "critical_path_dep": 1, "patient_impact": False},
]


def evaluate_single_criterion_orderings(
    projects_data: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Sanity check: computes project rankings when each single criterion's weight is 1.0 (others 0.0).
    In a non-degenerate/conflicting dataset, setting single weights must produce criterion-specific orderings
    that differ from each other and from the composite baseline.
    """
    orderings: Dict[str, List[str]] = {}
    criteria = ["contract_penalty", "project_tier", "remaining_float", "critical_path_dep", "patient_impact"]
    for crit in criteria:
        single_w = {k: (1.0 if k == crit else 0.0) for k in criteria}
        ranked_res, _ = compute_prioritization_ranking(projects_data, weights=single_w)
        orderings[crit] = [r.project_id for r in ranked_res]
    return orderings


@dataclass
class PrioritizationScore:
    project_id: str
    program_name: str
    composite_score: float
    rank: int
    score_breakdown: Dict[str, float]
    raw_values: Dict[str, Any]
    is_pareto_optimal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "project_id": self.project_id,
            "program_name": self.program_name,
            "composite_score": round(self.composite_score, 3),
            "is_pareto_optimal": self.is_pareto_optimal,
            **{f"norm_{k}": round(v, 3) for k, v in self.score_breakdown.items()},
        }


def compute_prioritization_ranking(
    projects_data: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[PrioritizationScore], pd.DataFrame]:
    """
    Computes normalized multi-criteria prioritization ranking.
    """
    w = weights or DEFAULT_WEIGHTS
    # Normalize weights so sum == 1.0
    w_sum = sum(w.values())
    norm_w = {k: v / w_sum for k, v in w.items()}

    # Extract raw arrays for normalization
    penalties = np.array([float(p.get("contract_penalty", p.get("penalty", 10000))) for p in projects_data])
    tiers = np.array([TIER_SCORES.get(p.get("priority_tier", p.get("tier", "Tier 2")), 0.5) for p in projects_data])
    floats = np.array([float(p.get("remaining_float", p.get("float_days", 5.0))) for p in projects_data])
    deps = np.array([float(p.get("critical_path_dep", 1.0)) for p in projects_data])
    patients = np.array([1.0 if p.get("patient_impact", False) else 0.0 for p in projects_data])

    # Normalization (min-max scaling into [0, 1])
    def norm_scale(arr: np.ndarray, invert: bool = False) -> np.ndarray:
        min_v, max_v = np.min(arr), np.max(arr)
        if max_v - min_v < 1e-9:
            return np.ones_like(arr)
        scaled = (arr - min_v) / (max_v - min_v)
        return 1.0 - scaled if invert else scaled

    norm_penalty = norm_scale(penalties)
    norm_tier = norm_scale(tiers)
    norm_float = norm_scale(floats, invert=True)  # Lower float -> higher priority
    norm_dep = norm_scale(deps)
    norm_patient = patients  # already binary 0/1

    scores: List[float] = []
    breakdowns: List[Dict[str, float]] = []

    for i in range(len(projects_data)):
        b = {
            "contract_penalty": float(norm_penalty[i]),
            "project_tier": float(norm_tier[i]),
            "remaining_float": float(norm_float[i]),
            "critical_path_dep": float(norm_dep[i]),
            "patient_impact": float(norm_patient[i]),
        }
        total = sum(b[k] * norm_w[k] for k in norm_w)
        scores.append(total)
        breakdowns.append(b)

    # Pareto optimality check: Penalty vs Patient Impact
    pareto_flags = [False] * len(projects_data)
    for i in range(len(projects_data)):
        dominated = False
        for j in range(len(projects_data)):
            if i != j:
                # Does j dominate i? (strictly better in at least one, and no worse in both)
                if (penalties[j] >= penalties[i] and patients[j] >= patients[i]) and \
                   (penalties[j] > penalties[i] or patients[j] > patients[i]):
                    dominated = True
                    break
        pareto_flags[i] = not dominated

    ranked_indices = np.argsort(scores)[::-1]
    results: List[PrioritizationScore] = []
    for rank_idx, idx in enumerate(ranked_indices, start=1):
        p = projects_data[idx]
        item = PrioritizationScore(
            project_id=p.get("project_id", f"PROJ-{idx:03d}"),
            program_name=p.get("program_name", p.get("name", "Biologics Program")),
            composite_score=float(scores[idx]),
            rank=rank_idx,
            score_breakdown=breakdowns[idx],
            raw_values=p,
            is_pareto_optimal=pareto_flags[idx],
        )
        results.append(item)

    df_summary = pd.DataFrame([r.to_dict() for r in results])
    return results, df_summary


def evaluate_weight_sensitivity(
    projects_data: List[Dict[str, Any]],
    base_weights: Optional[Dict[str, float]] = None,
    n_samples: int = 500,
    concentration: float = 30.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates ranking robustness by Dirichlet sampling around base weights.
    """
    rng = np.random.default_rng(seed)
    w = base_weights or DEFAULT_WEIGHTS
    keys = list(w.keys())
    w_arr = np.array([w[k] for k in keys])
    w_norm = w_arr / np.sum(w_arr)

    # Baseline ranking
    base_res, _ = compute_prioritization_ranking(projects_data, weights=w)
    base_ranks = [r.project_id for r in base_res]
    top_base_proj = base_ranks[0]

    alpha = w_norm * concentration
    dirichlet_weights = rng.dirichlet(alpha, size=n_samples)

    top_counts: Dict[str, int] = {p.get("project_id", ""): 0 for p in projects_data}
    kendall_taus: List[float] = []

    for sample in dirichlet_weights:
        sample_w = {k: float(sample[i]) for i, k in enumerate(keys)}
        res, _ = compute_prioritization_ranking(projects_data, weights=sample_w)
        top_proj = res[0].project_id
        top_counts[top_proj] = top_counts.get(top_proj, 0) + 1

        sample_ranks = [r.project_id for r in res]
        # Calculate Kendall Tau correlation with baseline
        tau, _ = kendalltau(
            [base_ranks.index(pid) for pid in base_ranks],
            [sample_ranks.index(pid) for pid in base_ranks],
        )
        if not np.isnan(tau):
            kendall_taus.append(float(tau))

    prob_top = {pid: round(cnt / n_samples, 3) for pid, cnt in top_counts.items() if cnt > 0}
    mean_tau = round(float(np.mean(kendall_taus)), 3) if kendall_taus else 1.0

    return {
        "n_samples": n_samples,
        "baseline_top_project": top_base_proj,
        "prob_top_ranked": prob_top,
        "mean_kendall_tau": mean_tau,
        "rank_stability_score": round(prob_top.get(top_base_proj, 0.0), 3),
    }


def evaluate_multi_concentration_sensitivity(
    projects_data: List[Dict[str, Any]],
    concentrations: Optional[List[float]] = None,
    base_weights: Optional[Dict[str, float]] = None,
    n_samples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates Dirichlet rank-stability across multiple concentration parameter values alpha_0
    (e.g. 10.0 for high dispersion, 30.0 for moderate dispersion, 100.0 for tight consensus).
    """
    conc_list = concentrations or [10.0, 30.0, 100.0]
    results: Dict[str, Any] = {}
    for c in conc_list:
        key = f"alpha_{int(c)}"
        res = evaluate_weight_sensitivity(
            projects_data=projects_data,
            base_weights=base_weights,
            n_samples=n_samples,
            concentration=float(c),
            seed=seed,
        )
        results[key] = {
            "concentration": float(c),
            "rank_stability_score": res["rank_stability_score"],
            "mean_kendall_tau": res["mean_kendall_tau"],
            "prob_top_ranked": res["prob_top_ranked"],
        }
    return results
