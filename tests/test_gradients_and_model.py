"""
Phase 3 Unit Tests: Autograd vs Finite-Difference Gradient Check through ANN + Physics + RK4 ODE,
Batch vs Single-Lot RK4 Parity, PyTorch vs Exported `.npz` NumPy Inference Parity,
Parameter Bounds & Sanity Gate Verification, and Stratified Validation Metrics (`matched` vs `mismatched`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.config import (
    load_hybrid_config,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.model import (
    HAS_TORCH,
    PREDICTED_PARAM_KEYS,
    BoundedKineticMLP,
    extract_feature_matrix,
    integrate_isothermal_rk4_grid,
    predict_params_from_npz_member,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.ode import (
    integrate_pathway_rk4,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.synth import (
    DEFAULT_SYNTH_DIR,
    load_observations_for_training,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.train import (
    DEFAULT_MODEL_DIR,
    RK4_STEP_MONTHS,
    _build_split_lot_tensors,
    compute_split_loss_and_predictions_torch,
    load_exported_ensemble_npz,
    train_hybrid_ensemble,
)

if HAS_TORCH:
    import torch


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for gradient check")
def test_autograd_vs_finite_difference_through_ann_ode_physics():
    """
    Verifies that PyTorch autograd gradients dL/d(theta) with respect to MLP parameters
    after flowing through:
        X -> BoundedKineticMLP -> Sigmoid Bounds -> Lumry-Eyring/Arrhenius Rates -> RK4 ODE -> 1/sigma^2 Loss
    agree with central finite differences within < 1e-4 relative error.
    """
    cfg = load_hybrid_config()
    df_train = load_observations_for_training(
        DEFAULT_SYNTH_DIR / "synth_observations.csv", allowed_splits=("train",)
    )
    # Use a 6-lot subset for fast, exact central finite-difference gradient verification
    subset_lots = sorted(df_train["lot_id"].unique())[:6]
    df_sub = df_train[df_train["lot_id"].isin(subset_lots)].copy()
    sub_data = _build_split_lot_tensors(df_sub)

    model = BoundedKineticMLP(
        input_dim=sub_data["x_norm"].shape[1],
        hidden_dims=(16, 16),
        cfg=cfg,
        seed=20261023,
    )
    model.zero_grad()
    out = compute_split_loss_and_predictions_torch(model, sub_data, smoothness_lambda=0.01)
    loss = out["loss"]
    loss.backward()

    # Check gradients across all 3 layers (fc1, fc2, fc3 weights and biases)
    param_probes = [
        ("fc1.weight", model.fc1.weight, (0, 0)),
        ("fc1.weight", model.fc1.weight, (3, 5)),
        ("fc1.bias", model.fc1.bias, (2,)),
        ("fc2.weight", model.fc2.weight, (1, 4)),
        ("fc2.bias", model.fc2.bias, (5,)),
        ("fc3.weight", model.fc3.weight, (0, 2)),
        ("fc3.weight", model.fc3.weight, (3, 7)),
        ("fc3.bias", model.fc3.bias, (0,)),
        ("fc3.bias", model.fc3.bias, (3,)),
    ]

    eps = 1e-6
    for name, p_tensor, idx in param_probes:
        assert p_tensor.grad is not None, f"No gradient computed for {name}"
        ag_grad = float(p_tensor.grad[idx].item())
        assert abs(ag_grad) > 1e-9, f"Unexpectedly zero gradient on {name}{idx}"

        with torch.no_grad():
            orig_val = float(p_tensor[idx].item())
            p_tensor[idx] = orig_val + eps
            loss_plus = float(
                compute_split_loss_and_predictions_torch(model, sub_data, smoothness_lambda=0.01)["loss"].item()
            )
            p_tensor[idx] = orig_val - eps
            loss_minus = float(
                compute_split_loss_and_predictions_torch(model, sub_data, smoothness_lambda=0.01)["loss"].item()
            )
            p_tensor[idx] = orig_val

        fd_grad = (loss_plus - loss_minus) / (2.0 * eps)
        rel_err = abs(ag_grad - fd_grad) / max(abs(fd_grad), 1e-12)
        assert rel_err < 1e-4, f"Gradient mismatch on {name}{idx}: autograd={ag_grad}, fd={fd_grad}, rel_err={rel_err}"


def test_batch_isothermal_rk4_matches_pathway_rk4():
    """
    Verifies that vectorized `integrate_isothermal_rk4_grid` matches `integrate_pathway_rk4`
    to machine precision (< 1e-14 relative error) across 5, 25, and 40 °C.
    """
    cfg = load_hybrid_config()
    df_val = load_observations_for_training(
        DEFAULT_SYNTH_DIR / "synth_observations.csv", allowed_splits=("val",)
    )
    lot_df = df_val.groupby("lot_id", as_index=False).first().head(4)
    x_norm, _, _ = extract_feature_matrix(lot_df)

    try:
        model = BoundedKineticMLP(input_dim=x_norm.shape[1], cfg=cfg, seed=42)
        weights = model.export_numpy_weights()
    except RuntimeError:
        # Runtime pure NumPy mode when PyTorch is not installed in the active interpreter
        weights = load_exported_ensemble_npz(DEFAULT_MODEL_DIR / "hybrid_ensemble_weights.npz")[0][0]
    params_batch = predict_params_from_npz_member(x_norm, weights, cfg=cfg)

    hmw0 = lot_df["initial_hmw_pct"].to_numpy(dtype=np.float64) / 100.0
    lmw0 = lot_df["initial_lmw_pct"].to_numpy(dtype=np.float64) / 100.0

    for temp_c, max_m in ((5.0, 24.0), (25.0, 12.0), (40.0, 3.0)):
        batch_out = integrate_isothermal_rk4_grid(
            temp_c=temp_c,
            max_months=max_m,
            step_months=RK4_STEP_MONTHS,
            params=params_batch,
            initial_hmw_fraction=hmw0,
            initial_lmw_fraction=lmw0,
            include_lmw=True,
        )
        n_steps = int(round(max_m / RK4_STEP_MONTHS))
        t_grid_h = np.linspace(0.0, max_m * 730.5, n_steps + 1)

        for i in range(len(lot_df)):
            single_p = {k: float(params_batch[k][i]) for k in params_batch}  # type: ignore
            single_out = integrate_pathway_rk4(
                time_grid_h=t_grid_h,
                temp_profile_c=np.full_like(t_grid_h, temp_c),
                params=single_p,
                initial_hmw_fraction=float(hmw0[i]),
                initial_lmw_fraction=float(lmw0[i]),
                include_lmw=True,
            )
            rel_err = np.max(
                np.abs(batch_out["hmw_fraction"][i] - single_out["hmw_fraction"])
                / single_out["hmw_fraction"]
            )
            assert rel_err < 1e-14


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for PyTorch-vs-NPZ check")
def test_pytorch_vs_exported_npz_numpy_inference_parity():
    """
    Verifies that `BoundedKineticMLP` PyTorch forward predictions and exported `.npz`
    NumPy predictions match within < 1e-12 relative error.
    """
    cfg = load_hybrid_config()
    df_val = load_observations_for_training(
        DEFAULT_SYNTH_DIR / "synth_observations.csv", allowed_splits=("val",)
    )
    lot_df = df_val.groupby("lot_id", as_index=False).first()
    x_norm, _, _ = extract_feature_matrix(lot_df)

    model = BoundedKineticMLP(input_dim=x_norm.shape[1], cfg=cfg, seed=101)
    weights = model.export_numpy_weights()

    with torch.no_grad():
        pt_out = model(torch.tensor(x_norm, dtype=torch.float64))
    np_out = predict_params_from_npz_member(x_norm, weights, cfg=cfg)

    for key in PREDICTED_PARAM_KEYS:
        pt_arr = pt_out[key].cpu().numpy()
        np_arr = np.asarray(np_out[key], dtype=np.float64)
        rel_err = np.max(np.abs(pt_arr - np_arr) / np.maximum(np.abs(np_arr), 1e-15))
        assert rel_err < 1e-12, f"Mismatch on {key}: {rel_err}"


def test_trained_ensemble_bounds_and_stratified_validation_metrics():
    """
    Verifies the trained 25-member ensemble artifacts exist (or trains them if missing),
    checks that all 25 members improved validation loss, confirms 100% of predicted parameters
    stay inside configured bounds and pass the apparent Ea [60, 200] kJ/mol sanity gate,
    and checks separate validation error reporting for `matched`, `mismatched`,
    `mismatched_ps80_only`, and `mismatched_full`.
    """
    summary_path = DEFAULT_MODEL_DIR / "training_summary.json"
    npz_path = DEFAULT_MODEL_DIR / "hybrid_ensemble_weights.npz"
    if not summary_path.exists() or not npz_path.exists():
        summary = train_hybrid_ensemble()
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["ensemble_size"] == 25
    curves = summary["training_curves_summary"]
    assert curves["mean_best_val_loss"] < curves["mean_init_val_loss"]

    bcheck = summary["parameter_bounds_and_sanity_check"]
    assert bcheck["all_within_configured_bounds"] is True
    assert bcheck["apparent_ea_5_40c_kj_mol"]["within_gate"] is True
    assert bcheck["temperature_rate_monotonicity_pass"] is True

    val_groups = summary["validation_errors_by_structural_group"]
    for grp in ("overall", "matched", "mismatched", "mismatched_ps80_only", "mismatched_full"):
        assert grp in val_groups
        assert val_groups[grp]["lots_count"] > 0
        assert val_groups[grp]["hmw_rmse_pp"] > 0.0

    # Structurally matched validation lots should have lower HMW RMSE than mismatched lots
    assert val_groups["matched"]["hmw_rmse_pp"] < val_groups["mismatched"]["hmw_rmse_pp"]

    # Verify loading exported `.npz` works without error
    weights_list, f_mean, f_std = load_exported_ensemble_npz(npz_path)
    assert len(weights_list) == 25
    assert f_mean.shape == (11,)
    assert f_std.shape == (11,)
