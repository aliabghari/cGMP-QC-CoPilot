"""
Phase 3: Offline Ensemble Training Pipeline (`train.py`).

Trains a 25-member deep ensemble of `BoundedKineticMLP` models coupled to the deterministic
Lumry-Eyring + RK4 ODE solver:
- Fits isothermal stability observations (`synth_observations.csv`, `split="train"`) using
  1/sigma^2 heteroscedastic Gaussian negative log-likelihood on HMW fraction (plus secondary
  LMW and Acidic pathways).
- Never uses `structural_group`, ground-truth parameters, excursion data, or `NS-2026-004`
  during training.
- Uses lot-level bootstrap resampling across the 100 training lots for each ensemble member.
- Applies early stopping on the 40 held-out validation lots.
- Exports both `.pt` (PyTorch checkpoint) and `.npz` (NumPy weights for zero-PyTorch inference)
  plus `training_summary.json` and `model_card.json` to `assets/models/`.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    HAS_TORCH = False

from .config import get_config_sha256, load_hybrid_config
from .model import (
     PREDICTED_PARAM_KEYS,
    BoundedKineticMLP,
    compute_lot_apparent_ea,
    extract_feature_matrix,
    integrate_isothermal_rk4_grid,
    predict_params_from_npz_member,
)
from .physics import HOURS_PER_MONTH, celsius_to_kelvin, evaluate_pathway_rates
from .synth import (
    DEFAULT_SYNTH_DIR,
    LOT_COVARIATE_COLUMNS,
    compute_file_sha256,
    heteroscedastic_sec_sd_fraction,
    load_ground_truth_for_verification_only,
    load_observations_for_training,
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "assets" / "models"

# Step size in months for isothermal RK4 integration (0.1 months = 73.05 h).
# Every pull month in {0, 0.5, 1, 2, 3, 6, 9, 12, 18, 24} is an exact integer multiple of 0.1 months.
RK4_STEP_MONTHS = 0.1


def _build_split_lot_tensors(
    df_split: pd.DataFrame,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
    base_sec_sd_pct: float = 0.035,
    cex_sd_pct: float = 0.35,
) -> Dict[str, Any]:
    """
    Organizes observable stability rows for a split (`train` or `val`) by unique `lot_id`
    into structured arrays mapped onto the uniform `RK4_STEP_MONTHS = 0.1` month grid for
    5 °C (max 24M = 240 steps), 25 °C (max 12M = 120 steps), and 40 °C (max 3M = 30 steps).
    """
    lot_df = (
        df_split.groupby("lot_id", as_index=False)
        .first()[["lot_id", "split"] + list(LOT_COVARIATE_COLUMNS)]
        .sort_values("lot_id")
        .reset_index(drop=True)
    )
    lot_ids = lot_df["lot_id"].tolist()
    lot_to_idx = {lid: i for i, lid in enumerate(lot_ids)}
    n_lots = len(lot_ids)

    x_norm, f_mean, f_std = extract_feature_matrix(
        lot_df, feature_mean=feature_mean, feature_std=feature_std
    )
    hmw0_frac = lot_df["initial_hmw_pct"].to_numpy(dtype=np.float64) / 100.0
    lmw0_frac = lot_df["initial_lmw_pct"].to_numpy(dtype=np.float64) / 100.0
    ac0_frac = lot_df["initial_acidic_pct"].to_numpy(dtype=np.float64) / 100.0

    cond_specs = {
        5.0: 24.0,
        25.0: 12.0,
        40.0: 3.0,
    }
    obs_by_cond: Dict[float, Dict[str, Any]] = {}
    for cond_c, max_m in cond_specs.items():
        sub = df_split[df_split["temperature_c"] == cond_c].copy()
        lot_idx_arr = np.array([lot_to_idx[lid] for lid in sub["lot_id"]], dtype=np.int64)
        step_idx_arr = np.round(sub["time_months"].to_numpy(dtype=np.float64) / RK4_STEP_MONTHS).astype(np.int64)
        t_hours_arr = sub["time_hours"].to_numpy(dtype=np.float64)
        hmw_obs_frac = sub["hmw_obs_pct"].to_numpy(dtype=np.float64) / 100.0
        lmw_obs_frac = sub["lmw_obs_pct"].to_numpy(dtype=np.float64) / 100.0
        ac_obs_frac = sub["acidic_obs_pct"].to_numpy(dtype=np.float64) / 100.0
        sigma_hmw_frac = heteroscedastic_sec_sd_fraction(
            hmw_obs_frac, base_sd_pct=base_sec_sd_pct, rel_rsd_slope=0.03
        )
        sigma_lmw_frac = np.full_like(hmw_obs_frac, (base_sec_sd_pct * 0.65) / 100.0)
        sigma_ac_frac = np.full_like(hmw_obs_frac, cex_sd_pct / 100.0)

        obs_by_cond[cond_c] = {
            "max_months": max_m,
            "lot_idx": lot_idx_arr,
            "step_idx": step_idx_arr,
            "time_hours": t_hours_arr,
            "hmw_obs_frac": hmw_obs_frac,
            "lmw_obs_frac": lmw_obs_frac,
            "ac_obs_frac": ac_obs_frac,
            "sigma_hmw_frac": sigma_hmw_frac,
            "sigma_lmw_frac": sigma_lmw_frac,
            "sigma_ac_frac": sigma_ac_frac,
        }

    return {
        "lot_ids": lot_ids,
        "n_lots": n_lots,
        "x_norm": x_norm,
        "feature_mean": f_mean,
        "feature_std": f_std,
        "hmw0_frac": hmw0_frac,
        "lmw0_frac": lmw0_frac,
        "ac0_frac": ac0_frac,
        "obs_by_cond": obs_by_cond,
    }


def compute_split_loss_and_predictions_torch(
    model: BoundedKineticMLP,
    split_data: Dict[str, Any],
    lot_weights: torch.Tensor | None = None,
    smoothness_lambda: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes the 1/sigma^2 heteroscedastic Gaussian NLL loss and unweighted RMSE (in pp HMW)
    by predicting bounded parameters from lot covariates and integrating the RK4 ODE trajectory
    across 5 °C, 25 °C, and 40 °C.
    """
    x_t = torch.tensor(split_data["x_norm"], dtype=torch.float64)
    hmw0_t = torch.tensor(split_data["hmw0_frac"], dtype=torch.float64)
    lmw0_t = torch.tensor(split_data["lmw0_frac"], dtype=torch.float64)
    ac0_t = torch.tensor(split_data["ac0_frac"], dtype=torch.float64)

    if lot_weights is None:
        lot_weights = torch.ones(split_data["n_lots"], dtype=torch.float64)

    params = model(x_t)
    log_sigma_adj = params["log_sigma_adj"]  # shape (n_lots,)

    total_weighted_nll = torch.tensor(0.0, dtype=torch.float64)
    total_weight_sum = torch.tensor(0.0, dtype=torch.float64)
    sq_err_hmw_pp_list: List[torch.Tensor] = []

    for cond_c, cdata in split_data["obs_by_cond"].items():
        traj = integrate_isothermal_rk4_grid(
            temp_c=cond_c,
            max_months=cdata["max_months"],
            step_months=RK4_STEP_MONTHS,
            params=params,
            initial_hmw_fraction=hmw0_t,
            initial_lmw_fraction=lmw0_t,
            include_lmw=True,
        )
        lot_idx = torch.tensor(cdata["lot_idx"], dtype=torch.long)
        step_idx = torch.tensor(cdata["step_idx"], dtype=torch.long)
        t_hours_t = torch.tensor(cdata["time_hours"], dtype=torch.float64)

        hmw_pred = traj["hmw_fraction"][lot_idx, step_idx]
        lmw_pred = traj["lmw_fraction"][lot_idx, step_idx]

        # Secondary acidic pathway exact / RK4 evolution
        pw_ac = evaluate_pathway_rates(celsius_to_kelvin(cond_c), params, pathways=("CEX_ACIDIC",))
        k_ac_lot = pw_ac["CEX_ACIDIC"]["k_eff_fraction_per_h"][lot_idx]
        ac_pred = 1.0 - (1.0 - ac0_t[lot_idx]) * torch.exp(-k_ac_lot * t_hours_t)

        hmw_obs = torch.tensor(cdata["hmw_obs_frac"], dtype=torch.float64)
        lmw_obs = torch.tensor(cdata["lmw_obs_frac"], dtype=torch.float64)
        ac_obs = torch.tensor(cdata["ac_obs_frac"], dtype=torch.float64)

        sigma_hmw_base = torch.tensor(cdata["sigma_hmw_frac"], dtype=torch.float64)
        sigma_lmw = torch.tensor(cdata["sigma_lmw_frac"], dtype=torch.float64)
        sigma_ac = torch.tensor(cdata["sigma_ac_frac"], dtype=torch.float64)

        # Learned heteroscedastic variance adjustment per lot: sigma^2 = sigma_base^2 * exp(2 * log_sigma_adj)
        log_var_adj = 2.0 * log_sigma_adj[lot_idx]
        var_hmw = (sigma_hmw_base ** 2) * torch.exp(log_var_adj)

        w_obs = lot_weights[lot_idx]

        # Primary 1/sigma^2 weighted HMW loss + log-variance normalizer
        nll_hmw = 0.5 * (((hmw_pred - hmw_obs) ** 2) / var_hmw + log_var_adj)
        # Secondary 1/sigma^2 weighted LMW & Acidic losses (down-weighted so HMW is primary)
        nll_lmw = 0.08 * 0.5 * (((lmw_pred - lmw_obs) ** 2) / (sigma_lmw ** 2))
        nll_ac = 0.05 * 0.5 * (((ac_pred - ac_obs) ** 2) / (sigma_ac ** 2))

        total_weighted_nll = total_weighted_nll + torch.sum(w_obs * (nll_hmw + nll_lmw + nll_ac))
        total_weight_sum = total_weight_sum + torch.sum(w_obs)

        sq_err_hmw_pp_list.append(((hmw_pred - hmw_obs) * 100.0) ** 2)

    mean_nll = total_weighted_nll / torch.clamp(total_weight_sum, min=1.0)

    # Physical sanity gate penalty on apparent Ea(5-40C) in [60, 200] kJ/mol
    ea_info = compute_lot_apparent_ea(params)
    ea_5_40 = ea_info["apparent_ea_5_40c_kj_mol"]
    ea_low_pen = torch.relu(60.0 - ea_5_40) ** 2
    ea_high_pen = torch.relu(ea_5_40 - 200.0) ** 2
    ea_penalty = torch.mean(ea_low_pen + ea_high_pen)

    # Smoothness penalty across small input perturbations
    if smoothness_lambda > 0.0:
        delta_x = 0.05 * torch.sin(torch.arange(x_t.numel(), dtype=torch.float64).reshape_as(x_t))
        params_pert = model(x_t + delta_x)
        smooth_pen = (
            torch.mean((params["dG_eff_ref_kj_mol"] - params_pert["dG_eff_ref_kj_mol"]) ** 2)
            + 0.01 * torch.mean((params["dH_agg_ref_kj_mol"] - params_pert["dH_agg_ref_kj_mol"]) ** 2)
        )
    else:
        smooth_pen = torch.tensor(0.0, dtype=torch.float64)

    total_loss = mean_nll + 0.5 * ea_penalty + smoothness_lambda * smooth_pen
    all_sq_err_pp = torch.cat(sq_err_hmw_pp_list)
    rmse_hmw_pp = torch.sqrt(torch.mean(all_sq_err_pp))

    return {
        "loss": total_loss,
        "mean_nll": mean_nll,
        "rmse_hmw_pp": rmse_hmw_pp,
        "params": params,
        "ea_info": ea_info,
    }


def evaluate_member_or_ensemble_numpy(
    ensemble_weights: List[Dict[str, np.ndarray]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    df_eval: pd.DataFrame,
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Pure NumPy evaluation of an ensemble (or single member) on any stability DataFrame.
    Returns per-lot predicted parameters (ensemble mean & std) and per-observation predicted HMW (%).
    """
    if cfg is None:
        cfg = load_hybrid_config()
    base_sec_sd_pct = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    cex_sd_pct = float(cfg["assay_and_decision_thresholds"]["cex_acidic_repeatability_sd_pct"])

    split_data = _build_split_lot_tensors(
        df_eval,
        feature_mean=feature_mean,
        feature_std=feature_std,
        base_sec_sd_pct=base_sec_sd_pct,
        cex_sd_pct=cex_sd_pct,
    )
    x_norm = split_data["x_norm"]
    hmw0_frac = split_data["hmw0_frac"]
    lmw0_frac = split_data["lmw0_frac"]

    n_members = len(ensemble_weights)
    member_params_list: List[Dict[str, np.ndarray]] = []
    for mw in ensemble_weights:
        p_dict = predict_params_from_npz_member(x_norm, mw, cfg=cfg)
        member_params_list.append(p_dict)  # type: ignore

    # Aggregate predicted parameters and rates across ensemble members
    param_summary_by_lot: Dict[str, Dict[str, float]] = {}
    lot_ids = split_data["lot_ids"]
    for l_idx, lid in enumerate(lot_ids):
        lot_summary: Dict[str, float] = {}
        for p_key in list(PREDICTED_PARAM_KEYS) + ["dG_agg_ref_kj_mol", "dS_agg_ref_j_mol_k", "log_sigma_adj"]:
            vals = np.array([mp[p_key][l_idx] for mp in member_params_list], dtype=np.float64)
            lot_summary[f"{p_key}_mean"] = float(np.mean(vals))
            lot_summary[f"{p_key}_std"] = float(np.std(vals, ddof=1) if n_members > 1 else 0.0)
            lot_summary[f"{p_key}_l95"] = float(np.quantile(vals, 0.025))
            lot_summary[f"{p_key}_u95"] = float(np.quantile(vals, 0.975))

        # Also compute derived k_eff(5/25/40C) and apparent Ea(5-40C) per member
        k5_m, k25_m, k40_m, ea_m = [], [], [], []
        for mp in member_params_list:
            single_p = {k: float(mp[k][l_idx]) for k in mp}
            ea_d = compute_lot_apparent_ea(single_p)
            k5_m.append(float(ea_d["k_eff_5c_per_h"]))
            k25_m.append(float(ea_d["k_eff_25c_per_h"]))
            k40_m.append(float(ea_d["k_eff_40c_per_h"]))
            ea_m.append(float(ea_d["apparent_ea_5_40c_kj_mol"]))

        for name_k, arr_k in (
            ("k_eff_5c_per_h", k5_m),
            ("k_eff_25c_per_h", k25_m),
            ("k_eff_40c_per_h", k40_m),
            ("apparent_ea_5_40c_kj_mol", ea_m),
        ):
            v_arr = np.asarray(arr_k, dtype=np.float64)
            lot_summary[f"{name_k}_mean"] = float(np.mean(v_arr))
            lot_summary[f"{name_k}_std"] = float(np.std(v_arr, ddof=1) if n_members > 1 else 0.0)
            lot_summary[f"{name_k}_l95"] = float(np.quantile(v_arr, 0.025))
            lot_summary[f"{name_k}_u95"] = float(np.quantile(v_arr, 0.975))

        param_summary_by_lot[lid] = lot_summary

    # Compute predicted HMW trajectories per condition across members
    obs_records: List[Dict[str, Any]] = []
    for cond_c, cdata in split_data["obs_by_cond"].items():
        lot_idx = cdata["lot_idx"]
        step_idx = cdata["step_idx"]
        hmw_obs_pct = cdata["hmw_obs_frac"] * 100.0

        member_hmw_preds_pct = []
        for mp in member_params_list:
            traj = integrate_isothermal_rk4_grid(
                temp_c=cond_c,
                max_months=cdata["max_months"],
                step_months=RK4_STEP_MONTHS,
                params=mp,
                initial_hmw_fraction=hmw0_frac,
                initial_lmw_fraction=lmw0_frac,
                include_lmw=True,
            )
            pred_pct = traj["hmw_fraction"][lot_idx, step_idx] * 100.0
            member_hmw_preds_pct.append(pred_pct)

        pred_matrix_pct = np.stack(member_hmw_preds_pct, axis=0)  # (n_members, n_obs_cond)
        pred_mean_pct = np.mean(pred_matrix_pct, axis=0)
        pred_epistemic_sd_pct = np.std(pred_matrix_pct, axis=0, ddof=1) if n_members > 1 else np.zeros_like(pred_mean_pct)

        for o_i in range(len(lot_idx)):
            lid = lot_ids[int(lot_idx[o_i])]
            obs_records.append(
                {
                    "lot_id": lid,
                    "temperature_c": float(cond_c),
                    "time_months": float(step_idx[o_i] * RK4_STEP_MONTHS),
                    "hmw_obs_pct": float(hmw_obs_pct[o_i]),
                    "hmw_pred_mean_pct": float(pred_mean_pct[o_i]),
                    "hmw_pred_epistemic_sd_pct": float(pred_epistemic_sd_pct[o_i]),
                    "residual_pp": float(hmw_obs_pct[o_i] - pred_mean_pct[o_i]),
                }
            )

    df_preds = pd.DataFrame(obs_records)
    return {
        "lot_param_summaries": param_summary_by_lot,
        "observations_df": df_preds,
    }


def train_hybrid_ensemble(
    obs_csv_path: Path | str = DEFAULT_SYNTH_DIR / "synth_observations.csv",
    output_dir: Path | str = DEFAULT_MODEL_DIR,
    ensemble_size: int | None = None,
    max_epochs: int | None = None,
) -> Dict[str, Any]:
    """
    Trains the 25-member deep ensemble with lot-level bootstrap resampling on the 100 training
    lots and early stopping on the 40 held-out validation lots.
    Saves `.pt` and `.npz` artifacts plus `training_summary.json` and `model_card.json`.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to run offline ensemble training.")

    t_start = time.time()
    cfg = load_hybrid_config()
    tcfg = cfg["training_and_uncertainty"]
    base_seed = int(tcfg["random_seed"])
    if ensemble_size is None:
        ensemble_size = int(tcfg["ensemble_size"])
    if max_epochs is None:
        max_epochs = int(tcfg["max_epochs_adam"])

    lr = float(tcfg["learning_rate"])
    weight_decay = float(tcfg["weight_decay_l2"])
    smooth_lambda = float(tcfg["smoothness_lambda"])
    grad_clip = float(tcfg["grad_clip_norm"])
    hidden_dims = tuple(tcfg["mlp_hidden_dims"])

    base_sec_sd_pct = float(cfg["assay_and_decision_thresholds"]["sec_hmw_repeatability_sd_pct"])
    cex_sd_pct = float(cfg["assay_and_decision_thresholds"]["cex_acidic_repeatability_sd_pct"])

    # Load ONLY observable isothermal stability data via the leakage-guarded loader
    df_train = load_observations_for_training(obs_csv_path, allowed_splits=("train",))
    df_val = load_observations_for_training(obs_csv_path, allowed_splits=("val",))

    train_data = _build_split_lot_tensors(
        df_train, base_sec_sd_pct=base_sec_sd_pct, cex_sd_pct=cex_sd_pct
    )
    val_data = _build_split_lot_tensors(
        df_val,
        feature_mean=train_data["feature_mean"],
        feature_std=train_data["feature_std"],
        base_sec_sd_pct=base_sec_sd_pct,
        cex_sd_pct=cex_sd_pct,
    )

    n_train_lots = train_data["n_lots"]
    rng_boot = np.random.default_rng(base_seed + 1000)

    ensemble_npz_weights: List[Dict[str, np.ndarray]] = []
    ensemble_torch_states: List[Dict[str, Any]] = []
    member_curves: List[Dict[str, Any]] = []

    for m_idx in range(ensemble_size):
        member_seed = base_seed + m_idx
        # Lot-level bootstrap: sample 100 training lots with replacement
        boot_indices = rng_boot.choice(n_train_lots, size=n_train_lots, replace=True)
        boot_counts = np.bincount(boot_indices, minlength=n_train_lots).astype(np.float64)
        lot_weights_t = torch.tensor(boot_counts, dtype=torch.float64)

        model = BoundedKineticMLP(
            input_dim=train_data["x_norm"].shape[1],
            hidden_dims=hidden_dims,
            cfg=cfg,
            seed=member_seed,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.1)

        best_val_loss = float("inf")
        best_val_rmse_pp = float("inf")
        best_epoch = 0
        best_weights = model.export_numpy_weights()
        best_state = copy.deepcopy(model.state_dict())

        init_train_loss = None
        init_val_loss = None
        patience = 40
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            tr_out = compute_split_loss_and_predictions_torch(
                model,
                train_data,
                lot_weights=lot_weights_t,
                smoothness_lambda=smooth_lambda,
            )
            loss = tr_out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_out = compute_split_loss_and_predictions_torch(
                    model, val_data, lot_weights=None, smoothness_lambda=0.0
                )
                val_loss = float(val_out["loss"].item())
                val_rmse_pp = float(val_out["rmse_hmw_pp"].item())
                tr_loss_val = float(loss.item())

            if epoch == 1:
                init_train_loss = tr_loss_val
                init_val_loss = val_loss

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_val_rmse_pp = val_rmse_pp
                best_epoch = epoch
                best_weights = model.export_numpy_weights()
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience and epoch >= 60:
                    break

        ensemble_npz_weights.append(best_weights)
        ensemble_torch_states.append(best_state)
        member_curves.append(
            {
                "member_index": m_idx,
                "seed": member_seed,
                "unique_bootstrapped_lots": int(np.sum(boot_counts > 0)),
                "init_train_loss": round(float(init_train_loss or 0.0), 4),
                "init_val_loss": round(float(init_val_loss or 0.0), 4),
                "best_epoch": best_epoch,
                "best_val_loss": round(best_val_loss, 4),
                "best_val_rmse_pp": round(best_val_rmse_pp, 5),
            }
        )

    # Post-training evaluation on train and validation splits, stratified by matched vs mismatched
    # (joining structural_group from ground-truth file ONLY here in verification summary)
    truth_data = load_ground_truth_for_verification_only()
    lots_truth = truth_data["lots"]

    val_eval = evaluate_member_or_ensemble_numpy(
        ensemble_npz_weights,
        train_data["feature_mean"],
        train_data["feature_std"],
        df_val,
        cfg=cfg,
    )
    train_eval = evaluate_member_or_ensemble_numpy(
        ensemble_npz_weights,
        train_data["feature_mean"],
        train_data["feature_std"],
        df_train,
        cfg=cfg,
    )

    val_obs_df = val_eval["observations_df"].copy()
    val_obs_df["structural_group"] = val_obs_df["lot_id"].map(
        lambda lid: lots_truth[lid]["structural_group"]
    )
    val_obs_df["structural_subgroup"] = val_obs_df["lot_id"].map(
        lambda lid: lots_truth[lid]["structural_subgroup"]
    )

    def _compute_group_metrics(sub_obs: pd.DataFrame, lot_ids_group: List[str], param_summaries: Dict[str, Any]) -> Dict[str, Any]:
        res_pp = sub_obs["residual_pp"].to_numpy(dtype=np.float64)
        rmse_pp = float(np.sqrt(np.mean(res_pp ** 2)))
        mae_pp = float(np.mean(np.abs(res_pp)))

        # Parameter recovery errors on identifiable quantities
        dg_errs = []
        ea_errs = []
        k5_rel_errs = []
        k25_rel_errs = []
        k40_rel_errs = []
        for lid in lot_ids_group:
            ps = param_summaries[lid]
            lt = lots_truth[lid]
            dg_errs.append(abs(ps["dG_eff_ref_kj_mol_mean"] - lt["dG_eff_ref_kj_mol"]))
            ea_errs.append(abs(ps["apparent_ea_5_40c_kj_mol_mean"] - lt["apparent_ea_5_40c_true_kj_mol"]))
            k5_rel_errs.append(abs(ps["k_eff_5c_per_h_mean"] - lt["k_eff_5c_true_per_h"]) / lt["k_eff_5c_true_per_h"])
            k25_rel_errs.append(abs(ps["k_eff_25c_per_h_mean"] - lt["k_eff_25c_true_per_h"]) / lt["k_eff_25c_true_per_h"])
            k40_rel_errs.append(abs(ps["k_eff_40c_per_h_mean"] - lt["k_eff_40c_true_per_h"]) / lt["k_eff_40c_true_per_h"])

        return {
            "lots_count": len(lot_ids_group),
            "observations_count": len(sub_obs),
            "hmw_rmse_pp": round(rmse_pp, 5),
            "hmw_mae_pp": round(mae_pp, 5),
            "dG_eff_ref_mae_kj_mol": round(float(np.mean(dg_errs)), 4),
            "apparent_ea_5_40c_mae_kj_mol": round(float(np.mean(ea_errs)), 4),
            "k_eff_5c_mape_pct": round(float(np.mean(k5_rel_errs)) * 100.0, 2),
            "k_eff_25c_mape_pct": round(float(np.mean(k25_rel_errs)) * 100.0, 2),
            "k_eff_40c_mape_pct": round(float(np.mean(k40_rel_errs)) * 100.0, 2),
        }

    val_lot_ids = val_data["lot_ids"]
    val_by_group = {
        "overall": _compute_group_metrics(val_obs_df, val_lot_ids, val_eval["lot_param_summaries"]),
        "matched": _compute_group_metrics(
            val_obs_df[val_obs_df["structural_group"] == "matched"],
            [lid for lid in val_lot_ids if lots_truth[lid]["structural_group"] == "matched"],
            val_eval["lot_param_summaries"],
        ),
        "mismatched": _compute_group_metrics(
            val_obs_df[val_obs_df["structural_group"] == "mismatched"],
            [lid for lid in val_lot_ids if lots_truth[lid]["structural_group"] == "mismatched"],
            val_eval["lot_param_summaries"],
        ),
        "mismatched_ps80_only": _compute_group_metrics(
            val_obs_df[val_obs_df["structural_subgroup"] == "mismatched_ps80_only"],
            [lid for lid in val_lot_ids if lots_truth[lid]["structural_subgroup"] == "mismatched_ps80_only"],
            val_eval["lot_param_summaries"],
        ),
        "mismatched_full": _compute_group_metrics(
            val_obs_df[val_obs_df["structural_subgroup"] == "mismatched_full"],
            [lid for lid in val_lot_ids if lots_truth[lid]["structural_subgroup"] == "mismatched_full"],
            val_eval["lot_param_summaries"],
        ),
    }

    # Check parameter bounds and physical sanity across all 240 lots and all 25 ensemble members
    df_all = load_observations_for_training(obs_csv_path, allowed_splits=("train", "val", "test"))
    all_lot_df = df_all.groupby("lot_id", as_index=False).first()
    x_all, _, _ = extract_feature_matrix(
        all_lot_df, feature_mean=train_data["feature_mean"], feature_std=train_data["feature_std"]
    )
    bounds_cfg = cfg["parameter_bounds"]
    bounds_check: Dict[str, Any] = {"all_within_configured_bounds": True, "parameters": {}}

    for p_key in PREDICTED_PARAM_KEYS:
        vals_all = np.concatenate(
            [predict_params_from_npz_member(x_all, mw, cfg=cfg)[p_key] for mw in ensemble_npz_weights]  # type: ignore
        )
        b_min = float(bounds_cfg[p_key]["min"])
        b_max = float(bounds_cfg[p_key]["max"])
        in_bounds = bool(np.all((vals_all >= b_min) & (vals_all <= b_max)))
        if not in_bounds:
            bounds_check["all_within_configured_bounds"] = False
        bounds_check["parameters"][p_key] = {
            "configured_min": b_min,
            "configured_max": b_max,
            "predicted_min": round(float(np.min(vals_all)), 5),
            "predicted_median": round(float(np.median(vals_all)), 5),
            "predicted_max": round(float(np.max(vals_all)), 5),
            "within_bounds": in_bounds,
        }

    # Check apparent Ea(5-40C) sanity gate [60, 200] kJ/mol across all 240 lots x 25 members
    ea_all_list = []
    mono_ok = True
    for mw in ensemble_npz_weights:
        mp = predict_params_from_npz_member(x_all, mw, cfg=cfg)
        ea_d = compute_lot_apparent_ea(mp)
        ea_all_list.append(np.asarray(ea_d["apparent_ea_5_40c_kj_mol"], dtype=np.float64))
        k5 = np.asarray(ea_d["k_eff_5c_per_h"], dtype=np.float64)
        k25 = np.asarray(ea_d["k_eff_25c_per_h"], dtype=np.float64)
        k40 = np.asarray(ea_d["k_eff_40c_per_h"], dtype=np.float64)
        if not bool(np.all((k5 > 0.0) & (k25 > k5) & (k40 > k25))):
            mono_ok = False
    ea_all = np.concatenate(ea_all_list)
    ea_gate_pass = bool(np.all((ea_all >= 60.0) & (ea_all <= 200.0)))
    bounds_check["apparent_ea_5_40c_kj_mol"] = {
        "gate_min": 60.0,
        "gate_max": 200.0,
        "predicted_min": round(float(np.min(ea_all)), 4),
        "predicted_median": round(float(np.median(ea_all)), 4),
        "predicted_max": round(float(np.max(ea_all)), 4),
        "within_gate": ea_gate_pass,
    }
    bounds_check["temperature_rate_monotonicity_pass"] = mono_ok

    # Export `.npz` and `.pt` artifacts
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "hybrid_ensemble_weights.npz"
    pt_path = out_dir / "hybrid_ensemble_checkpoint.pt"
    summary_path = out_dir / "training_summary.json"
    card_path = out_dir / "model_card.json"

    npz_dict: Dict[str, np.ndarray] = {
        "n_members": np.array(ensemble_size, dtype=np.int64),
        "feature_mean": train_data["feature_mean"],
        "feature_std": train_data["feature_std"],
    }
    for m_idx, mw in enumerate(ensemble_npz_weights):
        for k_w, arr_w in mw.items():
            npz_dict[f"m{m_idx}_{k_w}"] = arr_w
    np.savez(npz_path, **npz_dict)

    torch.save(
        {
            "ensemble_size": ensemble_size,
            "hidden_dims": hidden_dims,
            "feature_mean": train_data["feature_mean"],
            "feature_std": train_data["feature_std"],
            "member_state_dicts": ensemble_torch_states,
            "config_sha256": get_config_sha256(),
        },
        pt_path,
    )

    elapsed_sec = round(time.time() - t_start, 2)
    train_rmse_pp = float(
        np.sqrt(np.mean(train_eval["observations_df"]["residual_pp"].to_numpy(dtype=np.float64) ** 2))
    )

    training_summary = {
        "classification": "Non-GMP Decision Support Prototype on Synthetic Data — Requires Human QA Approval",
        "config_sha256": get_config_sha256(),
        "synth_observations_sha256": compute_file_sha256(Path(obs_csv_path)),
        "ensemble_size": ensemble_size,
        "training_time_seconds": elapsed_sec,
        "training_curves_summary": {
            "mean_init_train_loss": round(float(np.mean([c["init_train_loss"] for c in member_curves])), 4),
            "mean_init_val_loss": round(float(np.mean([c["init_val_loss"] for c in member_curves])), 4),
            "mean_best_val_loss": round(float(np.mean([c["best_val_loss"] for c in member_curves])), 4),
            "min_best_val_loss": round(float(np.min([c["best_val_loss"] for c in member_curves])), 4),
            "max_best_val_loss": round(float(np.max([c["best_val_loss"] for c in member_curves])), 4),
            "mean_best_epoch": round(float(np.mean([c["best_epoch"] for c in member_curves])), 1),
            "ensemble_train_hmw_rmse_pp": round(train_rmse_pp, 5),
            "ensemble_val_hmw_rmse_pp": val_by_group["overall"]["hmw_rmse_pp"],
        },
        "validation_errors_by_structural_group": val_by_group,
        "parameter_bounds_and_sanity_check": bounds_check,
        "member_summaries": member_curves,
    }
    summary_path.write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    model_card = {
        "model_name": "Hybrid Lumry-Eyring + BoundedKineticMLP Deep Ensemble",
        "version": "1.0.0-non-gmp",
        "intended_use": (
            "Non-GMP technical decision-support prototype for biologics thermal excursion evaluation, "
            "dynamic Out-of-Trend (OOT) stability band construction, and ICH Q1E shelf-life comparison "
            "on synthetic stability data. Requires human QA review and approval."
        ),
        "out_of_scope_and_limitations": [
            "Never makes automatic lot release or disposition decisions.",
            "Trained exclusively on synthetic data (SYNTHETIC - NOT GMP DATA).",
            "SEC alone cannot separate reversible conformational stress from sub-detectable irreversible nucleation.",
            "Latent unfolded fraction f_U(t) is prior-driven from configured K_u(5C) and dH_u bounds unless constrained by orthogonal DSC/DSF/HDX-MS data.",
            "Constant-rate Lumry-Eyring fitted model does not include a dynamic polysorbate-80 (PS80) depletion state; structurally mismatched lots with late-time PS80 depletion exhibit higher late-time errors at 25/40C.",
        ],
        "architecture": {
            "mlp_input_features": list(LOT_COVARIATE_COLUMNS) + ["(ph - 6.0)^2"],
            "mlp_hidden_dims": list(hidden_dims),
            "activation": "SiLU",
            "predicted_bounded_parameters": list(PREDICTED_PARAM_KEYS) + ["log_sigma_adj"],
            "ode_integrator": "4th-order Runge-Kutta (RK4) on rapid pre-equilibrium Lumry-Eyring + secondary LMW/Acidic pathways",
            "ensemble_size": ensemble_size,
            "resampling": "Lot-level bootstrap across 100 training lots per ensemble member",
        },
        "artifacts": {
            "npz_weights_sha256": compute_file_sha256(npz_path),
            "pt_checkpoint_sha256": compute_file_sha256(pt_path),
            "config_sha256": get_config_sha256(),
            "synth_observations_sha256": compute_file_sha256(Path(obs_csv_path)),
        },
        "validation_performance_summary": val_by_group,
    }
    card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    return training_summary


def load_exported_ensemble_npz(
    npz_path: Path | str = DEFAULT_MODEL_DIR / "hybrid_ensemble_weights.npz",
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Loads the exported `.npz` ensemble weights and feature standardization vectors
    for fast deterministic NumPy inference (no PyTorch dependency required at runtime).
    """
    data = np.load(Path(npz_path))
    n_members = int(data["n_members"])
    f_mean = data["feature_mean"].astype(np.float64)
    f_std = data["feature_std"].astype(np.float64)
    weights_list: List[Dict[str, np.ndarray]] = []
    for m_idx in range(n_members):
        weights_list.append(
            {
                "w1": data[f"m{m_idx}_w1"].astype(np.float64),
                "b1": data[f"m{m_idx}_b1"].astype(np.float64),
                "w2": data[f"m{m_idx}_w2"].astype(np.float64),
                "b2": data[f"m{m_idx}_b2"].astype(np.float64),
                "w3": data[f"m{m_idx}_w3"].astype(np.float64),
                "b3": data[f"m{m_idx}_b3"].astype(np.float64),
            }
        )
    return weights_list, f_mean, f_std
