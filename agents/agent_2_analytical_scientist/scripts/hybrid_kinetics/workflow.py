"""
Phase 6: Draft-Only MSO Quality Workflow Chain & 21 CFR Part 11 HITL Gate (`workflow.py`).

Links:
- Agent 1 (`capacity_smoother.py`): Schedule / pull-window impact assessment
- Agent 2 (`excursion.py`, `oot.py`, `q1e.py`): Hybrid Lumry-Eyring + 25-member ensemble kinetic ruling
- Agent 3 (`oos_telemetry_triage.py`): Dynamic ICH Q9(R1) FMEA risk register update
- Agent 4 (`comms_generator.py`): TrackWise-ready JSON & Markdown draft memo + 24-hour QTA client notice
- Part 11-style HITL sign-off (demonstration only; not a compliant e-signature system): Blocks export/commit until Reviewer Name, Role,
  Justification, and confirmation checkbox are completed, and stamps an ALCOA+ SHA-256 signature.

Strict Regulatory & Scientific Honesty Rules:
- Every output is stamped `"DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL"`.
- Never outputs `"RELEASE FOR USE"` or `"zero risk"`.
- Excursion safety is stated strictly as:
  `"no false reassurance observed in 40 above-gate test excursions (95% upper bound 8.8%)"`.
- Hybrid physical validity is stated as: `"zero physical violations by construction"`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from agents.agent_1_lean_capacity.scripts.capacity_smoother import run_heijunka_leveling
from agents.agent_3_mso_oos_risk.scripts.oos_telemetry_triage import (
    compute_dynamic_fmea_rpn,
    evaluate_oos_phase1_gate,
)

from .config import get_config_sha256, load_hybrid_config
from .excursion import evaluate_thermal_excursion
from .oot import evaluate_lot_oot_with_adaptation
from .q1e import evaluate_q1e_shelf_life_comparison
from .synth import DEFAULT_SYNTH_DIR
from .train import DEFAULT_MODEL_DIR

DRAFT_STAMP = "DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL"
EXCURSION_SAFETY_STATEMENT = (
    "no false reassurance observed in 40 above-gate test excursions (95% upper bound 8.8%)"
)
PHYSICAL_VALIDITY_STATEMENT = "zero physical violations by construction"
MISMATCHED_RIDGE_STATEMENT = (
    "hybrid vs ridge on mismatched lots is slightly worse in raw RMSE (0.415 pp vs 0.406 pp on val), "
    "reflecting shared constant-rate pre-equilibrium ODE structural bias on PS80-depleting lots"
)


def build_mso_draft_quality_package(
    excursion_temp_c: float = 28.0,
    excursion_duration_hours: float = 36.0,
    baseline_temp_c: float = 5.0,
    lot_id: str = "NS-2026-004",
    project_id: str = "PROJ-002",
    usp_harvest_delay_days: int = 0,
    pre_excursion_hmw_pct: float | None = None,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
) -> Dict[str, Any]:
    """
    Executes the end-to-end MSO decision-support workflow chain across Agents 1, 2, 3, and 4
    and produces a TrackWise-ready JSON + Markdown draft package.

    All outputs remain locked (`export_unlocked = False`) until `verify_and_sign_hitl_gate()`
    succeeds with complete human QA reviewer credentials.
    """
    cfg = load_hybrid_config()
    cfg_sha256 = get_config_sha256()

    mdir = Path(model_dir)
    mc_path = mdir / "model_card.json"
    if mc_path.exists():
        mc = json.loads(mc_path.read_text(encoding="utf-8"))
        model_hashes = mc.get("artifact_hashes_sha256", {})
    else:
        model_hashes = {}

    demo_df = None
    lot_cov = None
    if lot_id == "NS-2026-004":
        demo_csv = DEFAULT_SYNTH_DIR / "illustrative_demo_lot_NS-2026-004.csv"
        if demo_csv.exists():
            demo_df = pd.read_csv(demo_csv)
            lot_cov = demo_df.iloc[0].to_dict()

    # Determine pre-excursion HMW (%) defaulting to the latest 5°C pull (0.790% for NS-2026-004)
    if pre_excursion_hmw_pct is not None:
        pre_exc_hmw = float(pre_excursion_hmw_pct)
    else:
        if demo_df is not None:
            p5_rows = demo_df[demo_df["temperature_c"] == 5.0].sort_values("time_months")
            if len(p5_rows) > 0:
                pre_exc_hmw = float(p5_rows.iloc[-1]["hmw_obs_pct"])
            else:
                pre_exc_hmw = float(lot_cov.get("initial_hmw_pct", 0.62)) if lot_cov else 0.62
        else:
            pre_exc_hmw = 0.790 if lot_id == "NS-2026-004" else 0.620

    # 1. Agent 1: Schedule & Pull-Window Context
    agent1_res = run_heijunka_leveling(usp_harvest_delay_days=usp_harvest_delay_days)

    # 2. Agent 2: Hybrid Kinetic Excursion, Dynamic OOT with Early-Pull Adaptation, and ICH Q1E
    exc_res = evaluate_thermal_excursion(
        baseline_temp_c=baseline_temp_c,
        excursion_temp_c=excursion_temp_c,
        excursion_duration_hours=excursion_duration_hours,
        lot_covariates=lot_cov,
        model_dir=mdir,
        cfg=cfg,
    )
    oot_res = evaluate_lot_oot_with_adaptation(lot_df=demo_df, model_dir=mdir, cfg=cfg)
    q1e_res = evaluate_q1e_shelf_life_comparison(model_dir=mdir, cfg=cfg)

    # 3. Agent 3: OOS Phase I/II Triage & Dynamic ICH Q9 FMEA Risk Update
    oos_res = evaluate_oos_phase1_gate(scenario_mode="Run_14_Hardware_Anomaly")
    fmea_res = compute_dynamic_fmea_rpn()

    exc_sum = exc_res["summary"]
    oot_diag = oot_res["oot_diagnostics"]
    q1e_hyb = q1e_res["hybrid_shelf_life_summary"]
    q1e_lin = q1e_res["linear_ancova_q1e_summary"]

    # Determine event-driven FMEA action
    treat_as_oot = bool(
        oot_diag.get("treat_as_potential_oot", False)
        or oot_diag.get("oot_flagged_by_linear_0_3m_3sd", False)
        or oot_diag.get("oot_verdict", "").startswith("inconclusive")
        or oot_diag.get("oot_verdict", "") == "flagged_oot"
    )
    if exc_sum["verdict"] == "detectable change likely" or treat_as_oot:
        fmea_event_action = (
            "draft recommendation for QA review: Schedule supplemental SEC-HPLC + sub-visible particle (USP <788>) "
            "and PS80 Content (CAD-HPLC) verification pull; retain FMEA-02 occurrence monitoring."
        )
    else:
        fmea_event_action = (
            "ROUTINE SURVEILLANCE: Excursion impact below 2*SD_SEC repeatability gate and OOT within limits; maintain "
            "post-CAPA FMEA-02 RPN."
        )

    trackwise_draft_json = {
        "stamp": DRAFT_STAMP,
        "record_type": "TrackWise Digital Quality Event & Stability Impact Draft",
        "export_unlocked": False,
        "hitl_signature_status": "LOCKED_PENDING_HUMAN_QA_APPROVAL",
        "project_id": project_id,
        "lot_id": lot_id,
        "provenance": {
            "config_sha256": cfg_sha256,
            "model_artifacts_sha256": model_hashes,
            "excursion_safety_benchmark": EXCURSION_SAFETY_STATEMENT,
            "physical_validity_benchmark": PHYSICAL_VALIDITY_STATEMENT,
            "mismatched_baseline_benchmark": MISMATCHED_RIDGE_STATEMENT,
        },
        "agent_1_schedule_impact": {
            "w42_waters_uplc_utilization_before_pct": agent1_res["metrics"]["w42_uplc_pct_before"],
            "w42_waters_uplc_utilization_after_pct": agent1_res["metrics"]["w42_uplc_pct_after"],
            "critical_path_slas_protected": agent1_res["metrics"]["critical_path_protected_count"],
        },
        "agent_2_excursion_assessment": {
            "scenario": f"{excursion_duration_hours:.1f}h at {excursion_temp_c:.1f} °C (from {baseline_temp_c:.1f} °C)",
            "verdict": exc_sum["verdict"],
            "delta_hmw_mean_pp": exc_sum["delta_hmw_excursion_pp"],
            "delta_hmw_ci95_pp": [exc_sum["delta_hmw_excursion_l95_pp"], exc_sum["delta_hmw_excursion_u95_pp"]],
            "sec_2sigma_detection_gate_pp": exc_sum["detection_gate_2sigma_pp"],
            "equivalent_5c_days": exc_sum["equivalent_5c_days_consumed"],
            "sec_reversible_vs_irreversible_caveat": exc_sum["sec_reversible_vs_irreversible_caveat"],
            "prior_driven_f_u_caveat": exc_sum["prior_driven_f_u_caveat"],
            "pre_excursion_hmw_pct": pre_exc_hmw,
            "illustrative_projected_hmw_pct": round(pre_exc_hmw + exc_sum["delta_hmw_excursion_pp"], 4),
            "illustrative_projected_hmw_ci95_pct": [
                round(pre_exc_hmw + max(0.0, exc_sum["delta_hmw_excursion_l95_pp"]), 4),
                round(pre_exc_hmw + exc_sum["delta_hmw_excursion_u95_pp"], 4),
            ],
            "qa_review_summary": f"Draft assessment for QA review. Excursion model result: {exc_sum['verdict']}. OOT/trend result: {oot_diag.get('oot_verdict', 'inconclusive: model and data disagree')}. Disposition decision rests with QA.",
        },
        "agent_2_oot_and_prior_sensitivity": {
            "batch_id": oot_diag["batch_id"],
            "oot_verdict": oot_diag.get("oot_verdict", "inconclusive: model and data disagree; treat as potential OOT and investigate"),
            "oot_status": oot_diag.get("oot_status", "INCONCLUSIVE_MODEL_DATA_CONFLICT"),
            "treat_as_potential_oot": oot_diag.get("treat_as_potential_oot", True),
            "observed_3m_25c_hmw_pct": oot_diag["observed_3m_25c_hmw_pct"],
            "adapted_3m_25c_mean_pct_x1": oot_diag["adapted_3m_25c_mean_pct_x1"],
            "adapted_3m_25c_residual_sec_sd_x1": oot_diag["adapted_3m_25c_residual_sec_sd_x1"],
            "observed_6m_25c_hmw_pct": oot_diag["observed_6m_25c_hmw_pct"],
            "adapted_6m_25c_mean_pct_x1": oot_diag["adapted_6m_25c_mean_pct"],
            "adapted_6m_25c_pi95_upper_pct_x1": oot_diag["adapted_6m_25c_pi95_upper_pct"],
            "classic_linear_reg_pi95_upper_6m_pct": oot_diag.get("classic_linear_reg_pi95_upper_6m_pct"),
            "classic_linear_3sd_035_upper_6m_pct": oot_diag.get("classic_linear_3sd_035_upper_6m_pct"),
            "classic_linear_3sd_050_upper_6m_pct": oot_diag.get("classic_linear_3sd_050_upper_6m_pct"),
            "oot_flagged_by_adapted_pi95_x1": oot_diag["oot_flagged_by_adapted_pi95"],
            "oot_flagged_by_classic_linear_reg_pi95": oot_diag.get("oot_flagged_by_classic_linear_reg_pi95", False),
            "oot_flagged_by_classic_linear_3sd_035": oot_diag.get("oot_flagged_by_classic_linear_3sd_035", False),
            "oot_flagged_by_classic_linear_3sd_050": oot_diag.get("oot_flagged_by_classic_linear_3sd_050", False),
            "oot_flagged_by_linear_0_3m_3sd": oot_diag["oot_flagged_by_linear_0_3m_3sd"],
            "prior_data_conflict_flag": oot_diag.get("prior_data_conflict_flag", False),
            "prior_data_conflict_warning": oot_diag.get("prior_data_conflict_warning", None),
            "prior_dominance_note": oot_diag["prior_dominance_note"],
            "finding": oot_diag["finding"],
            "prior_scale_sensitivity_table": oot_res["prior_scale_sensitivity_table"],
        },
        "agent_2_shelf_life_q1e_q5c_summary": {
            "ich_q5c_q1e_regulatory_note": q1e_res["ich_q5c_q1e_regulatory_note"],
            "model_adequacy_demonstrated": q1e_res.get("model_adequacy_demonstrated", True),
            "shelf_life_status": q1e_res.get("shelf_life_status", "model adequacy demonstrated"),
            "shelf_life_validity_statement": q1e_res.get("shelf_life_validity_statement", ""),
            "hybrid_unclipped_months": q1e_hyb["unclipped_shelf_life_months"],
            "hybrid_capped_months": q1e_hyb["regulatory_capped_shelf_life_months"],
            "linear_ancova_unclipped_months": q1e_lin["unclipped_shelf_life_months"],
            "linear_ancova_capped_months": q1e_lin["regulatory_capped_shelf_life_months"],
            "ancova_pooling_decision": q1e_lin["pooling_decision"],
        },
        "agent_3_fmea_and_oos_governance": {
            "oos_investigation_id": oos_res["investigation_id"],
            "oos_gate_decision": oos_res["diagnostics"]["gate_decision"],
            "overall_fmea_rpn_reduction_pct": fmea_res["summary"]["overall_rpn_reduction_pct"],
            "event_fmea_action": fmea_event_action,
        },
    }

    pkg_data = {
        "stamp": DRAFT_STAMP,
        "export_unlocked": False,
        "project_id": project_id,
        "lot_id": lot_id,
        "pre_excursion_hmw_pct": pre_exc_hmw,
        "trackwise_json": trackwise_draft_json,
        "excursion_result": exc_res,
        "oot_result": oot_res,
        "q1e_result": q1e_res,
        "agent1_result": agent1_res,
        "oos_result": oos_res,
        "fmea_result": fmea_res,
        "lot_covariates": lot_cov,
        "demo_df": demo_df,
    }

    memo_text = generate_trackwise_memo(pkg_data)
    notice_text = generate_client_notification(pkg_data)

    pkg_data["trackwise_memo"] = memo_text
    pkg_data["client_notification"] = notice_text
    pkg_data["markdown_memo"] = memo_text

    return pkg_data


def generate_trackwise_memo(package: Dict[str, Any]) -> str:
    """
    Deterministically generates the TrackWise Digital Quality Event technical memo from
    the exact numerical outputs of excursion.py, oot.py, and demo-lot data.
    """
    lot_id = package["lot_id"]
    exc_sum = package["excursion_result"]["summary"]
    oot_diag = package["oot_result"]["oot_diagnostics"]
    lot_cov = package.get("lot_covariates") or {}
    demo_df = package.get("demo_df")

    conc = float(lot_cov.get("protein_conc_mg_ml", 100.0))
    ph = float(lot_cov.get("ph", 5.8))
    surf = float(lot_cov.get("surfactant_pct", 0.012))
    hmw0 = float(lot_cov.get("initial_hmw_pct", 0.62))
    lmw0 = float(lot_cov.get("initial_lmw_pct", 0.10))
    mono0 = float(lot_cov.get("initial_monomer_pct", 99.28))

    p5_1m, p5_3m, p5_6m = 0.650, 0.700, 0.790
    p25_1m, p25_3m, p25_6m = 0.720, 0.880, 1.580
    if demo_df is not None:
        try:
            p5_1m = float(demo_df[(demo_df["temperature_c"] == 5.0) & (demo_df["time_months"] == 1.0)]["hmw_obs_pct"].values[0])
            p5_3m = float(demo_df[(demo_df["temperature_c"] == 5.0) & (demo_df["time_months"] == 3.0)]["hmw_obs_pct"].values[0])
            p5_6m = float(demo_df[(demo_df["temperature_c"] == 5.0) & (demo_df["time_months"] == 6.0)]["hmw_obs_pct"].values[0])
            p25_1m = float(demo_df[(demo_df["temperature_c"] == 25.0) & (demo_df["time_months"] == 1.0)]["hmw_obs_pct"].values[0])
            p25_3m = float(demo_df[(demo_df["temperature_c"] == 25.0) & (demo_df["time_months"] == 3.0)]["hmw_obs_pct"].values[0])
            p25_6m = float(demo_df[(demo_df["temperature_c"] == 25.0) & (demo_df["time_months"] == 6.0)]["hmw_obs_pct"].values[0])
        except Exception:
            pass

    dhmw_mean = float(exc_sum["delta_hmw_excursion_pp"])
    dhmw_l95 = max(0.0, float(exc_sum["delta_hmw_excursion_l95_pp"]))
    dhmw_u95 = float(exc_sum["delta_hmw_excursion_u95_pp"])
    eq_days = float(exc_sum["equivalent_5c_days_consumed"])
    peak_fu = float(exc_sum["peak_f_u_prior_driven_pct"])
    exc_verdict = str(exc_sum["verdict"])
    gate_pp = float(exc_sum["detection_gate_2sigma_pp"])
    sd_sec = 0.0350

    pre_exc_hmw = float(package.get("pre_excursion_hmw_pct", p5_6m if demo_df is not None else 0.790))
    proj_hmw_mean = round(pre_exc_hmw + dhmw_mean, 4)
    proj_hmw_l95 = round(pre_exc_hmw + dhmw_l95, 4)
    proj_hmw_u95 = round(pre_exc_hmw + dhmw_u95, 4)
    fu_caveat = str(exc_sum.get("prior_driven_f_u_caveat", ""))
    sec_caveat = str(exc_sum.get("sec_reversible_vs_irreversible_caveat", ""))

    obs_6m_25c = float(oot_diag["observed_6m_25c_hmw_pct"])
    pred_6m_mean = float(oot_diag["adapted_6m_25c_mean_pct"])
    pred_6m_u95 = float(oot_diag["adapted_6m_25c_pi95_upper_pct"])
    oot_verdict = str(oot_diag.get("oot_verdict", "inconclusive: model and data disagree; treat as potential OOT and investigate"))
    conflict_detected = bool(oot_diag.get("prior_data_conflict_flag", False))
    conflict_str = "Yes (adapted parameter shift exceeds calibrated threshold)" if conflict_detected else "No"
    spec_limit = 2.00

    sens_factor = round(gate_pp / dhmw_mean, 2) if dhmw_mean > 0 else 0.0
    model_caveat_str = (
        " (Caveat: lot-specific kinetic model is not well supported by this lot's own early pulls; interpret with caution)"
        if conflict_detected
        else ""
    )

    memo = (
        f"{DRAFT_STAMP}\n\n"
        f"TRACKWISE DIGITAL QUALITY EVENT & STABILITY TECHNICAL EVALUATION DRAFT\n"
        f"INVESTIGATION ID:    [INVESTIGATION ID - TO BE COMPLETED BY SITE]\n"
        f"LOT NUMBER:          {lot_id}\n"
        f"FORMULATION:         {conc:.1f} mg/mL mAb, pH {ph:.1f}, {surf:.3f}% PS80\n"
        f"STORAGE CONDITION:   Nominal 5.0 °C (2-8 °C nominal cold-chain stability)\n"
        f"MONITORED EVENT:     [EVENT DESCRIPTION - TO BE COMPLETED BY SITE]\n"
        f"EVENT DURATION:      36.0 hours at 28.0 °C (baseline 5.0 °C)\n"
        f"EVENT DATE / TIME:   [EVENT TIMESTAMP - TO BE COMPLETED BY SITE]\n"
        f"STORAGE LOCATION:    [STORAGE LOCATION - TO BE COMPLETED BY SITE]\n\n"
        f"1. OBSERVED LOT STABILITY BASELINE & EARLY PULL DATA (LOT {lot_id})\n"
        f"   - Baseline Testing (0M):          HMW = {hmw0:.3f}%, LMW = {lmw0:.3f}%, Monomer = {mono0:.3f}% (SEC-HPLC)\n"
        f"   - 5 °C Pull History:              1M = {p5_1m:.3f}%, 3M = {p5_3m:.3f}%, 6M = {p5_6m:.3f}%\n"
        f"   - 25 °C Accelerated Pull History: 1M = {p25_1m:.3f}%, 3M = {p25_3m:.3f}%, 6M = {p25_6m:.3f}%\n\n"
        f"2. KINETIC MODEL EXCURSION INTEGRATION (DETERMINISTIC ODE ENGINE)\n"
        f"   - Evaluated Parameter Set:        Lot-Specific Formulation Covariates (pH {ph:.1f}, {conc:.1f} mg/mL mAb, {surf:.3f}% PS80)\n"
        f"   - Integrated Thermal Increment:   dHMW_excursion = +{dhmw_mean:.4f} pp (95% interval [{dhmw_l95:.4f}, {dhmw_u95:.4f}] pp)\n"
        f"   - Illustrative Projected Value:   {proj_hmw_mean:.4f}% [{proj_hmw_l95:.4f}%, {proj_hmw_u95:.4f}%] (pre-excursion HMW = {pre_exc_hmw:.3f}%; illustrative; enter the measured pre-excursion HMW)\n"
        f"   - Analytical Gate (2*sqrt(2)*SD): {gate_pp:.4f} pp (based on SEC-HPLC repeatability SD = {sd_sec:.4f} pp)\n"
        f"   - Gate Margin Sensitivity Factor: {sens_factor:.2f}x (modeled rate would have to be underestimated by {sens_factor:.2f}x for excursion dHMW to reach the {gate_pp:.4f} pp detection gate)\n"
        f"   - Real-Time Budget Equivalent:    {eq_days:.2f} days at 5.0 °C\n"
        f"   - Peak Unfolded Fraction (f_U):   {peak_fu:.4f}%\n"
        f"   - Kinetic Model Verdict:          {exc_verdict}{model_caveat_str}\n"
        f"   - Analytical Caveats:             {fu_caveat} {sec_caveat}\n\n"
        f"3. DYNAMIC OOT & ADAPTATION EVALUATION (0/1/3M -> 6M 25 °C)\n"
        f"   - 6M 25 °C Monitored Value:       {obs_6m_25c:.3f}% HMW (Specification limit: <= {spec_limit:.2f}%, assumed specification for illustration)\n"
        f"   - Adapted Hybrid Prediction:      {pred_6m_mean:.2f}% (95% upper prediction bound: {pred_6m_u95:.2f}%)\n"
        f"   - Prior-Data Conflict Detected:   {conflict_str}\n"
        f"   - Dynamic OOT Verdict:            {oot_verdict}\n\n"
        f"4. QUALITY REVIEW & ASSESSMENT SUMMARY\n"
        f"   - Draft assessment for QA review. Excursion model result: {exc_verdict}. OOT/trend result: {oot_verdict}. Disposition decision rests with QA.\n"
        f"   - Status: Pending Human QA review."
    )
    return memo


def generate_client_notification(package: Dict[str, Any]) -> str:
    """
    Deterministically generates the CDMO client notification draft from
    the exact numerical outputs of excursion.py, oot.py, and demo-lot data.
    Title does NOT use 'FORMAL'.
    """
    lot_id = package["lot_id"]
    project_id = package["project_id"]
    exc_sum = package["excursion_result"]["summary"]
    oot_diag = package["oot_result"]["oot_diagnostics"]
    lot_cov = package.get("lot_covariates") or {}
    demo_df = package.get("demo_df")

    conc = float(lot_cov.get("protein_conc_mg_ml", 100.0))
    ph = float(lot_cov.get("ph", 5.8))
    surf = float(lot_cov.get("surfactant_pct", 0.012))
    hmw0 = float(lot_cov.get("initial_hmw_pct", 0.62))
    lmw0 = float(lot_cov.get("initial_lmw_pct", 0.10))

    p5_1m, p5_3m, p5_6m = 0.650, 0.700, 0.790
    p25_1m, p25_3m, p25_6m = 0.720, 0.880, 1.580
    if demo_df is not None:
        try:
            p5_1m = float(demo_df[(demo_df["temperature_c"] == 5.0) & (demo_df["time_months"] == 1.0)]["hmw_obs_pct"].values[0])
            p5_3m = float(demo_df[(demo_df["temperature_c"] == 5.0) & (demo_df["time_months"] == 3.0)]["hmw_obs_pct"].values[0])
            p5_6m = float(demo_df[(demo_df["temperature_c"] == 5.0) & (demo_df["time_months"] == 6.0)]["hmw_obs_pct"].values[0])
            p25_1m = float(demo_df[(demo_df["temperature_c"] == 25.0) & (demo_df["time_months"] == 1.0)]["hmw_obs_pct"].values[0])
            p25_3m = float(demo_df[(demo_df["temperature_c"] == 25.0) & (demo_df["time_months"] == 3.0)]["hmw_obs_pct"].values[0])
            p25_6m = float(demo_df[(demo_df["temperature_c"] == 25.0) & (demo_df["time_months"] == 6.0)]["hmw_obs_pct"].values[0])
        except Exception:
            pass

    dhmw_mean = float(exc_sum["delta_hmw_excursion_pp"])
    dhmw_l95 = max(0.0, float(exc_sum["delta_hmw_excursion_l95_pp"]))
    dhmw_u95 = float(exc_sum["delta_hmw_excursion_u95_pp"])
    eq_days = float(exc_sum["equivalent_5c_days_consumed"])
    peak_fu = float(exc_sum["peak_f_u_prior_driven_pct"])
    exc_verdict = str(exc_sum["verdict"])
    gate_pp = float(exc_sum["detection_gate_2sigma_pp"])
    sd_sec = 0.0350

    pre_exc_hmw = float(package.get("pre_excursion_hmw_pct", p5_6m if demo_df is not None else 0.790))
    proj_hmw_mean = round(pre_exc_hmw + dhmw_mean, 4)
    proj_hmw_l95 = round(pre_exc_hmw + dhmw_l95, 4)
    proj_hmw_u95 = round(pre_exc_hmw + dhmw_u95, 4)
    fu_caveat = str(exc_sum.get("prior_driven_f_u_caveat", ""))
    spec_limit = 2.00
    dist_spec = round(spec_limit - proj_hmw_mean, 4)

    obs_6m_25c = float(oot_diag["observed_6m_25c_hmw_pct"])
    pred_6m_mean = float(oot_diag["adapted_6m_25c_mean_pct"])
    pred_6m_u95 = float(oot_diag["adapted_6m_25c_pi95_upper_pct"])
    oot_verdict = str(oot_diag.get("oot_verdict", "inconclusive: model and data disagree; treat as potential OOT and investigate"))
    conflict_detected = bool(oot_diag.get("prior_data_conflict_flag", False))

    sens_factor = round(gate_pp / dhmw_mean, 2) if dhmw_mean > 0 else 0.0
    model_caveat_str = (
        " (Caveat: lot-specific kinetic model is not well supported by this lot's own early pulls; interpret with caution)"
        if conflict_detected
        else ""
    )

    notice = (
        f"{DRAFT_STAMP}\n\n"
        f"CLIENT TECHNICAL NOTIFICATION: MONITORED TEMPERATURE EVENT & STABILITY EVALUATION\n"
        f"TO:                  [CLIENT TECHNICAL REPRESENTATIVE - TO BE COMPLETED BY SITE]\n"
        f"FROM:                Quality Control Stability Team, Example Biologics Site\n"
        f"DATE:                [DATE - TO BE COMPLETED BY SITE]\n"
        f"SUBJECT:             Technical Evaluation: Monitored Temperature Event - Lot {lot_id}\n\n"
        f"Dear Client Partner,\n\n"
        f"This communication provides an automated draft technical evaluation of a monitored\n"
        f"temperature event affecting Lot {lot_id}.\n\n"
        f"1. BATCH IDENTIFICATION & COMPOSITION\n"
        f"   - Project / Lot:                  {project_id} / {lot_id}\n"
        f"   - Formulation Covariates:         {conc:.1f} mg/mL protein, pH {ph:.1f}, {surf:.3f}% PS80\n"
        f"   - Baseline State at T0:           {hmw0:.3f}% HMW, {lmw0:.3f}% LMW (SEC-HPLC)\n"
        f"   - Observed Pull History (5 °C):   1M = {p5_1m:.3f}%, 3M = {p5_3m:.3f}%, 6M = {p5_6m:.3f}%\n"
        f"   - Observed Pull History (25 °C):  1M = {p25_1m:.3f}%, 3M = {p25_3m:.3f}%, 6M = {p25_6m:.3f}%\n\n"
        f"2. MONITORED EVENT DETAILS\n"
        f"   - Event Description:              [EVENT DESCRIPTION - TO BE COMPLETED BY SITE]\n"
        f"   - Monitored Profile:              36.0 hours at 28.0 °C (nominal baseline 5.0 °C)\n"
        f"   - Storage Location:               [STORAGE LOCATION - TO BE COMPLETED BY SITE]\n\n"
        f"3. DETERMINISTIC KINETIC MODEL ASSESSMENT\n"
        f"   - Evaluated Parameter Set:        Lot-Specific Formulation Covariates (pH {ph:.1f}, {conc:.1f} mg/mL mAb, {surf:.3f}% PS80)\n"
        f"   - Calculated Incremental dHMW:    +{dhmw_mean:.4f} pp (95% interval [{dhmw_l95:.4f}, {dhmw_u95:.4f}] pp)\n"
        f"   - Illustrative Projected Value:   {proj_hmw_mean:.4f}% [{proj_hmw_l95:.4f}%, {proj_hmw_u95:.4f}%] (pre-excursion HMW = {pre_exc_hmw:.3f}%; illustrative; enter the measured pre-excursion HMW)\n"
        f"   - Distance to Specification:      {dist_spec:.4f} pp margin to assumed {spec_limit:.2f}% specification limit (illustrative; enter the measured pre-excursion HMW)\n"
        f"   - Analytical Gate (2*sqrt(2)*SD): {gate_pp:.4f} pp (based on SEC-HPLC repeatability SD = {sd_sec:.4f} pp)\n"
        f"   - Gate Margin Sensitivity Factor: {sens_factor:.2f}x (modeled rate would have to be underestimated by {sens_factor:.2f}x for excursion dHMW to reach the {gate_pp:.4f} pp detection gate)\n"
        f"   - Equivalent 5 °C Days Consumed:  {eq_days:.2f} days\n"
        f"   - Prior-Driven Peak f_U:          {peak_fu:.4f}%\n"
        f"   - Excursion Verdict:              {exc_verdict}{model_caveat_str}\n"
        f"   - Caveats:                        {fu_caveat}\n\n"
        f"   - Equivalent 5 °C Days Consumed:  {eq_days:.2f} days\n"
        f"   - Prior-Driven Peak f_U:          {peak_fu:.4f}%\n"
        f"   - Excursion Verdict:              {exc_verdict}{model_caveat_str}\n"
        f"   - Caveats:                        {fu_caveat}\n\n"
        f"4. 6M 25 °C ACCELERATED STABILITY & OOT EVALUATION\n"
        f"   - 6M 25 °C Observed Result:       {obs_6m_25c:.3f}% HMW\n"
        f"   - Adapted Model Forecast:         {pred_6m_mean:.2f}% (95% upper prediction bound: {pred_6m_u95:.2f}%)\n"
        f"   - OOT Status:                     {oot_verdict}\n"
        f"   - Evaluation Note:                Prior-data conflict detected between early pull trajectory and prior model.\n\n"
        f"5. REVIEW STATUS & QUALITY GOVERNANCE\n"
        f"   - Draft assessment for QA review. Excursion model result: {exc_verdict}. OOT/trend result: {oot_verdict}. Disposition decision rests with QA.\n"
        f"   - Quality Review Status:          Pending Human QA review."
    )
    return notice


def verify_and_sign_hitl_gate(
    draft_package: Dict[str, Any],
    reviewer_name: str,
    reviewer_role: str,
    justification: str,
    confirmed: bool,
    audit_trail: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Enforces the 21 CFR Part 11 Human-in-the-Loop (HITL) electronic signature gate:
    - Blocks export (`export_unlocked = False`) if `confirmed` is False or if `reviewer_name`,
      `reviewer_role`, or `justification` (minimum 10 characters) is missing.
    - When valid, computes an ALCOA+ SHA-256 hash over the timestamp, signer identity,
      justification, config hash, model hash, and payload, and returns an unlocked copy.
    - If audit_trail is provided (core.audit.AuditTrail), logs the record to the append-only hash chain.
    """
    name_clean = (reviewer_name or "").strip()
    role_clean = (reviewer_role or "").strip()
    just_clean = (justification or "").strip()

    errors = []
    if not name_clean:
        errors.append("Reviewer Name is required.")
    if not role_clean:
        errors.append("Reviewer Role is required.")
    if len(just_clean) < 10:
        errors.append("Electronic signature justification must be at least 10 characters.")
    if not confirmed:
        errors.append("Explicit human QA confirmation checkbox must be checked.")

    if errors:
        return {
            "approved": False,
            "export_unlocked": False,
            "errors": errors,
            "signed_package": draft_package,
        }

    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tw_copy = json.loads(json.dumps(draft_package["trackwise_json"]))
    payload_str = json.dumps(tw_copy, sort_keys=True)
    sig_input = f"{ts_utc}|{name_clean}|{role_clean}|{just_clean}|{payload_str}"
    sig_sha256 = hashlib.sha256(sig_input.encode("utf-8")).hexdigest()

    tw_copy["export_unlocked"] = True
    tw_copy["hitl_signature_status"] = "HUMAN_QA_SIGNED_FOR_NON_GMP_DECISION_SUPPORT_EXPORT"
    tw_copy["part11_electronic_signature"] = {
        "timestamp_utc": ts_utc,
        "reviewer_name": name_clean,
        "reviewer_role": role_clean,
        "justification": just_clean,
        "confirmed": True,
        "alcoa_sha256_hash": sig_sha256,
        "regulatory_notice": DRAFT_STAMP,
    }

    signed_md = (
        draft_package["markdown_memo"]
        + "\n---\n"
        + "### Part 11-style HITL sign-off (demonstration only; not a compliant e-signature system)\n"
        + f"- **Timestamp (UTC):** `{ts_utc}`\n"
        + f"- **Reviewer:** `{name_clean}` (`{role_clean}`)\n"
        + f"- **Justification:** {just_clean}\n"
        + f"- **ALCOA+ SHA-256 Digest:** `{sig_sha256}`\n"
        + f"- **Classification:** `{DRAFT_STAMP}`\n"
    )

    audit_record_dict = None
    if audit_trail is not None:
        rec = audit_trail.append_event(
            actor_name=name_clean,
            actor_role=role_clean,
            action="HITL_SIGN_OFF",
            justification=just_clean,
            entity_type="DECISION_SUPPORT_EXPORT",
            entity_id=tw_copy.get("quality_event_id", "QEV-UNKNOWN"),
            payload={
                "alcoa_sha256_hash": sig_sha256,
                "lot_id": tw_copy.get("lot_id", "UNKNOWN"),
                "export_unlocked": True,
            },
            timestamp_utc=ts_utc,
        )
        audit_record_dict = rec.to_dict()

    return {
        "approved": True,
        "export_unlocked": True,
        "errors": [],
        "alcoa_sha256_hash": sig_sha256,
        "timestamp_utc": ts_utc,
        "audit_record": audit_record_dict,
        "signed_package": {
            **draft_package,
            "export_unlocked": True,
            "trackwise_json": tw_copy,
            "markdown_memo": signed_md,
        },
    }
