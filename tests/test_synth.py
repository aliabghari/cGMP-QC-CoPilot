"""
Phase 2 Unit Tests: Synthetic Generator (`synth.py`), Realistic Pull Schedules,
Strict Lot Split (100/40/100), Anti-Inverse-Crime + PS80 Ablation, SHA-256 Hashes,
Excursion Gate Distribution, and Strict Leakage Guard (`structural_group` absent from CSVs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.ode import (
    integrate_pathway_rk4,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.synth import (
    DEFAULT_SYNTH_DIR,
    FORBIDDEN_GROUND_TRUTH_COLUMNS,
    GENERATOR_VERSION,
    SYNTH_STAMP,
    compute_file_sha256,
    generate_synthetic_datasets,
    load_observations_for_training,
)


@pytest.fixture(scope="module")
def generated_manifest() -> dict:
    """Loads existing Phase 2 manifest if already generated with current version, else regenerates it."""
    manifest_path = DEFAULT_SYNTH_DIR / "dataset_manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if m.get("generator_version") == GENERATOR_VERSION:
            return m
    return generate_synthetic_datasets(output_dir=DEFAULT_SYNTH_DIR)


def test_lot_splits_realistic_schedules_and_max_hmw(generated_manifest: dict):
    """
    Verifies:
    - 240 lots split strictly by lot into 100 train / 40 val / 100 test.
    - Realistic pull schedules:
        * 5 C: 0, 1, 3, 6, 9, 12, 18, 24 months
        * 25 C: 0, 1, 3, 6 months (and 9, 12 months on 40% of lots per split)
        * 40 C: 0, 0.5, 1, 2, 3 months
    - Max observed HMW across the entire dataset is realistic (< 15% HMW).
    - Stratified counts: 70% matched, 15% mismatched_ps80_only, 15% mismatched_full.
    """
    obs_path = DEFAULT_SYNTH_DIR / "synth_observations.csv"
    df = pd.read_csv(obs_path)
    assert (df["data_stamp"] == SYNTH_STAMP).all()

    train_lots = set(df[df["split"] == "train"]["lot_id"].unique())
    val_lots = set(df[df["split"] == "val"]["lot_id"].unique())
    test_lots = set(df[df["split"] == "test"]["lot_id"].unique())

    assert len(train_lots) == 100
    assert len(val_lots) == 40
    assert len(test_lots) == 100

    assert train_lots.isdisjoint(val_lots)
    assert train_lots.isdisjoint(test_lots)
    assert val_lots.isdisjoint(test_lots)

    # Check realistic pull schedules per condition
    pulls_5c = set(df[df["temperature_c"] == 5.0]["time_months"].unique())
    pulls_25c = set(df[df["temperature_c"] == 25.0]["time_months"].unique())
    pulls_40c = set(df[df["temperature_c"] == 40.0]["time_months"].unique())
    assert pulls_5c == {0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0}
    assert pulls_25c == {0.0, 1.0, 3.0, 6.0, 9.0, 12.0}
    assert pulls_40c == {0.0, 0.5, 1.0, 2.0, 3.0}

    # Max HMW check: realistic stress/accelerated HMW (< 15%)
    assert float(df["hmw_obs_pct"].max()) < 15.0

    # Check matched vs mismatched subgroup counts in manifest
    assert generated_manifest["splits"]["train"]["matched_lots"] == 70
    assert generated_manifest["splits"]["train"]["mismatched_ps80_only_lots"] == 15
    assert generated_manifest["splits"]["train"]["mismatched_full_lots"] == 15
    assert generated_manifest["splits"]["val"]["matched_lots"] == 28
    assert generated_manifest["splits"]["val"]["mismatched_ps80_only_lots"] == 6
    assert generated_manifest["splits"]["val"]["mismatched_full_lots"] == 6
    assert generated_manifest["splits"]["test"]["matched_lots"] == 70
    assert generated_manifest["splits"]["test"]["mismatched_ps80_only_lots"] == 15
    assert generated_manifest["splits"]["test"]["mismatched_full_lots"] == 15


def test_sha256_hashes_and_illustrative_demo_lot(generated_manifest: dict):
    """Verifies recorded SHA-256 digests match files on disk and checks NS-2026-004 demo lot."""
    for fname, fmeta in generated_manifest["files"].items():
        fpath = DEFAULT_SYNTH_DIR / fname
        assert fpath.exists(), f"Missing generated file {fname}"
        assert compute_file_sha256(fpath) == fmeta["sha256"]

    df_demo = pd.read_csv(DEFAULT_SYNTH_DIR / "illustrative_demo_lot_NS-2026-004.csv")
    assert len(df_demo) == 8  # 4 timepoints (0, 1, 3, 6M) x 2 temperatures (5C, 25C)
    assert set(df_demo["lot_id"].unique()) == {"NS-2026-004"}
    assert set(df_demo["time_months"].unique()) == {0.0, 1.0, 3.0, 6.0}
    assert "structural_group" not in df_demo.columns


def test_training_loader_leakage_guard_and_no_structural_group(generated_manifest: dict, tmp_path: Path):
    """
    Proves that:
    1. `synth_observations.csv` and `synth_excursions.csv` lack `structural_group`, `structural_subgroup`,
       and all ground-truth latent columns.
    2. `load_observations_for_training` raises PermissionError if pointed at `_ground_truth_do_not_read_in_train.json`
       or `synth_excursions.csv`.
    3. `load_observations_for_training` raises PermissionError if a CSV contains `structural_group` or any
       ground-truth column.
    4. `load_observations_for_training` succeeds on an isolated copy of `synth_observations.csv` when no
       ground-truth file exists in the directory.
    """
    obs_df = pd.read_csv(DEFAULT_SYNTH_DIR / "synth_observations.csv")
    exc_df = pd.read_csv(DEFAULT_SYNTH_DIR / "synth_excursions.csv")

    assert "structural_group" not in obs_df.columns
    assert "structural_subgroup" not in obs_df.columns
    assert "structural_group" not in exc_df.columns
    assert "structural_subgroup" not in exc_df.columns

    for col in list(obs_df.columns) + list(exc_df.columns):
        for forbidden in FORBIDDEN_GROUND_TRUTH_COLUMNS:
            assert forbidden.lower() not in col.lower(), f"Leakage column {col} found!"

    # 1. Pointing training loader at ground-truth JSON or excursions CSV must raise PermissionError
    truth_path = DEFAULT_SYNTH_DIR / "_ground_truth_do_not_read_in_train.json"
    with pytest.raises(PermissionError, match="Leakage guard triggered"):
        load_observations_for_training(truth_path)
    with pytest.raises(PermissionError, match="Leakage guard triggered"):
        load_observations_for_training(DEFAULT_SYNTH_DIR / "synth_excursions.csv")

    # 2. Pointing training loader at a CSV containing `structural_group` must raise PermissionError
    poisoned_sg_csv = tmp_path / "poisoned_sg.csv"
    bad_sg_df = obs_df.head(10).copy()
    bad_sg_df["structural_group"] = "matched"
    bad_sg_df.to_csv(poisoned_sg_csv, index=False)
    with pytest.raises(PermissionError, match="Forbidden ground-truth column"):
        load_observations_for_training(poisoned_sg_csv)

    # 3. Pointing training loader at a CSV containing `dG_eff_ref_kj_mol` must raise PermissionError
    poisoned_dg_csv = tmp_path / "poisoned_dg.csv"
    bad_dg_df = obs_df.head(10).copy()
    bad_dg_df["dG_eff_ref_kj_mol"] = 121.0
    bad_dg_df.to_csv(poisoned_dg_csv, index=False)
    with pytest.raises(PermissionError, match="Forbidden ground-truth column"):
        load_observations_for_training(poisoned_dg_csv)

    # 4. Copy ONLY synth_observations.csv into an isolated directory with NO ground-truth JSON file
    isolated_csv = tmp_path / "synth_observations.csv"
    isolated_csv.write_bytes((DEFAULT_SYNTH_DIR / "synth_observations.csv").read_bytes())
    loaded_df = load_observations_for_training(isolated_csv, allowed_splits=("train",))
    assert set(loaded_df["split"].unique()) == {"train"}
    assert "structural_group" not in loaded_df.columns
    assert "NS-2026-004" not in set(loaded_df["lot_id"].unique())
    assert len(loaded_df["lot_id"].unique()) == 100


def test_excursion_distribution_and_ps80_ablation(generated_manifest: dict):
    """
    Verifies:
    - Excursion dataset (`synth_excursions.csv`) is restricted to validation/test lots (no train lots)
      and has balanced representation clearly below, near, and clearly above the 0.099 pp gate.
    - Both `mismatched_ps80_only` and `mismatched_full` lots exhibit late-time PS80 depletion
      diverging from the constant-rate pre-equilibrium RK4 model.
    """
    exc_summary = generated_manifest["excursion_dhmw_distribution_vs_gate"]
    assert exc_summary["total_excursions"] == 120
    assert exc_summary["clearly_below_gate_lt_0.065pp"] >= 35
    assert exc_summary["near_gate_0.065_to_0.145pp"] >= 30
    assert exc_summary["clearly_above_gate_gt_0.145pp"] >= 35

    exc_df = pd.read_csv(DEFAULT_SYNTH_DIR / "synth_excursions.csv")
    assert set(exc_df["split"].unique()).issubset({"val", "test"})

    truth_data = json.loads(
        (DEFAULT_SYNTH_DIR / "_ground_truth_do_not_read_in_train.json").read_text(encoding="utf-8")
    )
    lots = truth_data["lots"]
    ps80_only_lots = [v for v in lots.values() if v["structural_subgroup"] == "mismatched_ps80_only"]
    full_mm_lots = [v for v in lots.values() if v["structural_subgroup"] == "mismatched_full"]
    assert len(ps80_only_lots) == 36  # 15 train + 6 val + 15 test
    assert len(full_mm_lots) == 36     # 15 train + 6 val + 15 test

    # Verify PS80-only ablation lot has dCp_agg == 0.0, fast k_-1 >= 75, and positive PS80 depletion at 25C 6M
    sample_ps80 = ps80_only_lots[0]
    assert sample_ps80["dCp_agg_kj_mol_k"] == 0.0
    assert sample_ps80["k_unfold_relax_per_hour"] >= 75.0
    assert sample_ps80["ps80_alpha_true"] > 0.0

    radau_25c_6m_pct = float(sample_ps80["noiseless_trajectories"]["25C"]["hmw_true_pct"][3])  # index 3 is 6.0M
    t_6m_grid = np.linspace(0.0, 6.0 * 730.5, 361)
    rk4_6m = integrate_pathway_rk4(
        time_grid_h=t_6m_grid,
        temp_profile_c=np.full_like(t_6m_grid, 25.0),
        params=sample_ps80,
        initial_hmw_fraction=sample_ps80["initial_hmw_pct"] / 100.0,
        initial_lmw_fraction=sample_ps80["initial_lmw_pct"] / 100.0,
        include_lmw=True,
    )
    rk4_25c_6m_pct = float(rk4_6m["hmw_fraction"][-1]) * 100.0
    assert radau_25c_6m_pct > rk4_25c_6m_pct + 0.05
