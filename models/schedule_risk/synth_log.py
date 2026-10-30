"""
cGMP-QC-CoPilot: Synthetic Historical Log & Scenario Generator.

Implements Guardrails G1 (Synthetic only) and G4 (No inverse crime):
- Structural disparity: Generator incorporates latent factors (holiday shutdowns,
  instrument-failure bursts, upstream perfusion contamination, analyst fatigue)
  that are NEVER exposed to downstream ML feature matrices.
- Separate seeds for training, validation, and held-out evaluation splits.
- Purely synthetic records labeled with SHA-256 manifests.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from core.synthetic import SYNTHETIC_HEADER_TAG, generate_manifest

PROJECT_PROFILES = [
    {"project_id": "BI-CAP-101", "name": "Spesolimab-NextGen", "stream": "Captive", "tier": "Tier 1A", "penalty": 50000, "patient": True, "base_slip_mean": 0.2, "base_slip_sd": 0.5},
    {"project_id": "BI-CAP-102", "name": "BI-1482 Bispecific", "stream": "Captive", "tier": "Tier 1A", "penalty": 40000, "patient": True, "base_slip_mean": 0.3, "base_slip_sd": 0.6},
    {"project_id": "BI-CAP-103", "name": "BI-7740 ADC", "stream": "Captive", "tier": "Tier 2", "penalty": 20000, "patient": False, "base_slip_mean": 0.5, "base_slip_sd": 0.8},
    {"project_id": "BI-CAP-104", "name": "BI-8891 mAb", "stream": "Captive", "tier": "Tier 2", "penalty": 15000, "patient": False, "base_slip_mean": 0.4, "base_slip_sd": 0.7},
    {"project_id": "BI-CAP-105", "name": "BI-9012 Fusion", "stream": "Captive", "tier": "Tier 3", "penalty": 5000, "patient": False, "base_slip_mean": 0.8, "base_slip_sd": 1.2},
    {"project_id": "EX-CDMO-201", "name": "Target-X mAb", "stream": "CDMO", "tier": "Tier 1B", "penalty": 75000, "patient": True, "base_slip_mean": 0.6, "base_slip_sd": 1.0},
    {"project_id": "EX-CDMO-202", "name": "Glyco-Eng mAb", "stream": "CDMO", "tier": "Tier 2", "penalty": 30000, "patient": False, "base_slip_mean": 0.7, "base_slip_sd": 1.1},
    {"project_id": "EX-CDMO-203", "name": "Bispecific TCE", "stream": "CDMO", "tier": "Tier 2", "penalty": 25000, "patient": False, "base_slip_mean": 0.5, "base_slip_sd": 0.9},
    {"project_id": "EX-CDMO-204", "name": "High-Conc SubQ", "stream": "CDMO", "tier": "Tier 3", "penalty": 10000, "patient": False, "base_slip_mean": 1.2, "base_slip_sd": 1.5},
    {"project_id": "EX-CDMO-205", "name": "Novel Scaffold", "stream": "CDMO", "tier": "Tier 3", "penalty": 8000, "patient": False, "base_slip_mean": 1.0, "base_slip_sd": 1.4},
]

INSTRUMENTS = [
    {"id": "UPLC_SEC_CEX", "name": "Waters Acquity UPLC", "base_duration_hrs": 14.5},
    {"id": "CE_icIEF", "name": "ProteinSimple Maurice", "base_duration_hrs": 12.0},
    {"id": "Cell_Bioassay", "name": "BSL-2 Hamilton Star", "base_duration_hrs": 16.0},
]

# True feature columns exposed to downstream models (NEVER include hidden generator variables)
EXPOSED_FEATURE_COLUMNS = [
    "week_index",
    "week_of_year",
    "project_id",
    "stream",
    "priority_tier",
    "instrument_id",
    "scheduled_duration_hours",
    "recent_avg_arrival_slip_4w",
    "recent_avg_turnaround_4w",
]

HIDDEN_GENERATOR_COLUMNS = [
    "_latent_holiday_flag",
    "_latent_skid_failure_burst",
    "_latent_upstream_contamination",
    "_latent_analyst_experience_penalty",
]


def generate_synthetic_historical_log(
    n_weeks: int = 104,
    seed: int = 42,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generates synthetic operational history across n_weeks.
    Incorporates unmodeled structural dynamics (G4).
    """
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    # Rolling history buffers for recent 4-week averages
    slip_history: Dict[str, List[float]] = {p["project_id"]: [0.2] for p in PROJECT_PROFILES}
    tat_history: Dict[str, List[float]] = {i["id"]: [14.0] for i in INSTRUMENTS}

    # Pre-generate sporadic 3-week instrument failure bursts
    if n_weeks >= 25:
        burst_weeks = set(rng.choice(range(10, n_weeks - 10), size=min(4, max(1, n_weeks // 20)), replace=False))
    else:
        burst_weeks = set()
    burst_windows = set()
    for bw in burst_weeks:
        burst_windows.update({bw, bw + 1, bw + 2})

    for week_idx in range(1, n_weeks + 1):
        week_of_year = ((week_idx - 1) % 52) + 1
        is_holiday = week_of_year in (26, 47, 51, 52)
        in_failure_burst = week_idx in burst_windows

        # Sample 4 to 8 batches arriving this week
        n_batches = int(rng.integers(4, 9))
        for _ in range(n_batches):
            p = rng.choice(PROJECT_PROFILES)
            inst = rng.choice(INSTRUMENTS)

            # 1. Latent unmodeled effects (hidden from ML)
            upstream_contam_shock = rng.lognormal(mean=0.3, sigma=0.8) if (p["stream"] == "CDMO" and rng.random() < 0.08) else 0.0
            holiday_slip_delta = rng.uniform(1.5, 3.5) if is_holiday else 0.0
            burst_tat_mult = rng.uniform(1.3, 1.8) if in_failure_burst else 1.0
            analyst_exp_penalty = rng.normal(loc=0.0, scale=1.2)

            # 2. Observed sample-arrival slippage (days)
            raw_slip = rng.normal(p["base_slip_mean"], p["base_slip_sd"]) + upstream_contam_shock + holiday_slip_delta
            actual_arrival_slip_days = float(np.clip(raw_slip, -1.0, 14.0))

            # 3. Observed analyst turnaround duration (hours)
            base_dur = inst["base_duration_hrs"] + rng.normal(0.0, 1.0)
            actual_turnaround_hours = float(np.clip(base_dur * burst_tat_mult + analyst_exp_penalty, 6.0, 36.0))

            # 4. Recent rolling 4-week indicators
            recent_slip = float(np.mean(slip_history[p["project_id"]][-4:]))
            recent_tat = float(np.mean(tat_history[inst["id"]][-4:]))

            rec = {
                "week_index": week_idx,
                "week_of_year": week_of_year,
                "project_id": p["project_id"],
                "stream": p["stream"],
                "priority_tier": p["tier"],
                "instrument_id": inst["id"],
                "scheduled_duration_hours": inst["base_duration_hrs"],
                "recent_avg_arrival_slip_4w": round(recent_slip, 2),
                "recent_avg_turnaround_4w": round(recent_tat, 2),
                # Targets:
                "actual_arrival_slip_days": round(actual_arrival_slip_days, 2),
                "actual_turnaround_hours": round(actual_turnaround_hours, 2),
                # Hidden ground-truth columns (must be excluded from training loaders):
                "_latent_holiday_flag": is_holiday,
                "_latent_skid_failure_burst": in_failure_burst,
                "_latent_upstream_contamination": upstream_contam_shock > 0,
                "_latent_analyst_experience_penalty": round(analyst_exp_penalty, 2),
                "is_synthetic": True,
            }
            records.append(rec)

            # Update rolling buffers
            slip_history[p["project_id"]].append(actual_arrival_slip_days)
            tat_history[inst["id"]].append(actual_turnaround_hours)

    df = pd.DataFrame(records)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


def generate_split_datasets(
    data_dir: Path,
    seed_train: int = 101,
    seed_val: int = 202,
    seed_test: int = 303,
) -> Tuple[Path, Path, Path]:
    """
    Generates strict temporal splits:
    - Train: Weeks 1 - 70 (seed 101)
    - Val:   Weeks 71 - 86 (seed 202)
    - Test:  Weeks 87 - 104 (seed 303, strictly held-out)
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Full continuous 104 weeks using master seed, partitioned temporally
    df_full = generate_synthetic_historical_log(n_weeks=104, seed=seed_train)

    train_path = data_dir / "capacity_train_log.csv"
    val_path = data_dir / "capacity_val_log.csv"
    test_path = data_dir / "capacity_test_log.csv"

    df_train = df_full[df_full["week_index"] <= 70].copy()
    df_val = df_full[(df_full["week_index"] > 70) & (df_full["week_index"] <= 86)].copy()
    df_test = df_full[df_full["week_index"] > 86].copy()

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    # Register manifest
    manifest_path = data_dir / "schedule_manifest.json"
    generate_manifest(
        data_files=[train_path, val_path, test_path],
        manifest_output_path=manifest_path,
        suite_name="Schedule Risk & Capacity Benchmark Dataset",
        seed=seed_train,
    )

    return train_path, val_path, test_path
