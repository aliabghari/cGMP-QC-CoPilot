"""
SOW-Aligned External Client & Cross-Functional Matrix Communication Generator
Agent 4: Stakeholder Communications & QTA/SOW Governance Specialist
Regulatory Basis: ICH Q10, FDA Quality Agreements Guidance (2016), Example Biologics Site SOP-PMO-0108 (fictional example)
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def load_stakeholder_register(asset_path: str | Path | None = None) -> Dict[str, Any]:
    """Loads the 6-persona Example Biologics Site QC Stakeholder Register."""
    if asset_path is None:
        asset_path = Path(__file__).resolve().parent.parent / "assets" / "stakeholder_register.json"
    with open(asset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_email_templates(asset_path: str | Path | None = None) -> Dict[str, Any]:
    """Loads the dynamic SOW/QTA email templates."""
    if asset_path is None:
        asset_path = Path(__file__).resolve().parent.parent / "assets" / "email_templates.json"
    with open(asset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_stakeholder_communications(
    agent1_results: Dict[str, Any],
    agent2_excursion: Dict[str, Any],
    agent2_oot: Dict[str, Any],
    agent2_transfer: Dict[str, Any],
    agent2_bioassay: Dict[str, Any],
    agent3_oos: Dict[str, Any],
    agent3_fmea: Dict[str, Any],
    selected_project_id: str = "PROJ-001",
    harvest_slip_days: int = 0
) -> Dict[str, Any]:
    """
    Populates all 4 primary dispatch templates (External Client QTA Update, Internal MSAT Ping,
    External CRO Follow-Up, and Director QC Executive Brief) using live deterministic outputs
    from Agents 1, 2, and 3.
    """
    reg_data = load_stakeholder_register()
    tpl_data = load_email_templates()["templates"]

    df_stakeholders = pd.DataFrame([
        {
            "ID": s["stakeholder_id"],
            "Role": s["role_title"],
            "Contact_Persona": s["persona_name"],
            "Organization": s["organization"],
            "Channel": s["communication_channel"],
            "Cadence": s["cadence"],
            "Primary_KPI_Focus": " | ".join(s["primary_kpis"]),
        }
        for s in reg_data["stakeholders"]
    ])

    # Extract project info from Agent 1 portfolio
    df_port = agent1_results["df_portfolio"]
    proj_row = df_port[df_port["Project_ID"] == selected_project_id]
    if proj_row.empty:
        proj_row = df_port.iloc[[1]]
    program_name = str(proj_row["Program_Name"].iloc[0])
    sla_due_date = "2026-10-27"

    # Extract key metrics across agents
    w42_before = agent1_results["metrics"]["w42_uplc_pct_before"]
    w42_after = agent1_results["metrics"]["w42_uplc_pct_after"]
    shifted_count = agent1_results["metrics"]["projects_leveled_count"]

    exc_sum = agent2_excursion["summary"]
    oot_diag = agent2_oot["oot_diagnostics"]
    tr_params = agent2_transfer["parameters"]
    bio_metrics = agent2_bioassay["metrics"]

    oos_diag = agent3_oos["diagnostics"]
    fmea_sum = agent3_fmea["summary"]

    context_vars = {
        "project_id": selected_project_id,
        "program_name": program_name,
        "sla_due_date": sla_due_date,
        "w42_uplc_before": w42_before,
        "w42_uplc_after": w42_after,
        "shifted_count": shifted_count,
        "delta_hmw_pct": exc_sum["delta_hmw_excursion_pct"],
        "post_hmw_pct": exc_sum["post_excursion_hmw_hybrid_pct"],
        "shelf_life_deduction_months": exc_sum["shelf_life_deduction_months"],
        "excursion_verdict": exc_sum.get("verdict", "no detectable change"),
        "oot_obs_6m_pct": oot_diag.get("observed_6m_25c_hmw_pct", 1.58),
        "oot_alert_limit_pct": oot_diag.get("oot_alert_limit_6m_pct", 1.72),
        "oot_z_score": oot_diag["z_score_deviation"],
        "recommended_hold_min": tr_params["recommended_receiver_hold_min"],
        "Recommended_Hold_Delta": round(
            tr_params["recommended_receiver_hold_min"] - tr_params["donor_initial_hold_min"], 2
        ),
        "max_pressure_bar": oos_diag["max_pump_pressure_bar"],
        "max_chiller_temp_c": oos_diag["max_autosampler_temp_c"],
        "deamidation_artifact_pct": oos_diag["calculated_on_tray_deamidation_artifact_pct"],
        "gate_decision": oos_diag["gate_decision"],
        "harvest_slip_days": harvest_slip_days,
        "hill_slope_ratio": bio_metrics["hill_slope_ratio"],
        "rpn_reduction_pct": fmea_sum["overall_rpn_reduction_pct"],
        "max_pre_rpn": fmea_sum["max_pre_rpn"],
        "max_post_rpn": fmea_sum["max_post_rpn"],
    }

    recipient_lookup = {
        "EXTERNAL_CLIENT_QTA_UPDATE": "Dr. Priya Nair (VP Technical Operations, Example CDMO Client Partner)",
        "INTERNAL_MSAT_USP_DSP_PING": "Dr. Kenji Takahashi (Principal MSAT Engineer, Example Biologics Site USP/DSP)",
        "EXTERNAL_CRO_LOT_RELEASE_PING": "Sarah Jenkins (Associate Director, Contract Testing Lab)",
        "DIRECTOR_QC_EXECUTIVE_BRIEF": "Dr. Marcus Vance (Director of Quality Control, Example Biologics Site)",
    }

    rendered_dispatches: Dict[str, Dict[str, str]] = {}
    for key, tpl in tpl_data.items():
        vars_with_recip = {**context_vars, "recipient_name": recipient_lookup.get(key, "Stakeholder")}
        rendered_dispatches[key] = {
            "template_key": key,
            "recipient_role": tpl["recipient_role"],
            "recipient_name": vars_with_recip["recipient_name"],
            "qta_clause": tpl["qta_clause_reference"],
            "subject": tpl["subject_template"].format(**vars_with_recip),
            "body": tpl["body_template"].format(**vars_with_recip),
        }

    # SOW Change Order Assessment Table
    df_change_orders = pd.DataFrame([
        {
            "Change_Order_ID": "CO-2026-PMO-09A",
            "Project_ID": "PROJ-004",
            "Trigger_Category": "CO-01: Mechanistic LSS/SMA Dwell-Volume Method Remediation",
            "Technical_Driver": f"Agilent 1260 (1.15 mL) -> Waters UPLC (0.32 mL) ΔV_D compensation (+{context_vars['Recommended_Hold_Delta']} min hold)",
            "FTE_Hours": 16.0,
            "Billable_Fee_USD": "$6,400",
            "QTA_SOW_Status": "STAGED FOR CLIENT JPT APPROVAL",
        },
        {
            "Change_Order_ID": "CO-2026-PMO-09B",
            "Project_ID": "PROJ-002",
            "Trigger_Category": "CO-02: Courier Cold-Chain Excursion Modeling (36h @ 28°C)",
            "Technical_Driver": f"Hybrid Eyring+ANN-TVP ODE simulation (+{exc_sum['delta_hmw_excursion_pct']}% HMW impact & shelf-life memo)",
            "FTE_Hours": 14.0,
            "Billable_Fee_USD": "$5,600",
            "QTA_SOW_Status": "STAGED FOR CLIENT JPT APPROVAL",
        },
        {
            "Change_Order_ID": "CO-2026-PMO-09C",
            "Project_ID": "PROJ-001",
            "Trigger_Category": "Internal Lab Absorb (No Client Charge)",
            "Technical_Driver": "Empower Run #14 Phase I In-Line Filter & Peltier Chiller Hardware Remediation",
            "FTE_Hours": 12.0,
            "Billable_Fee_USD": "$0 (Example Biologics Site QC CAPA Absorbed)",
            "QTA_SOW_Status": "INTERNAL QC OVERHEAD (PER QTA SEC 9.4)",
        },
    ])

    return {
        "df_stakeholders": df_stakeholders,
        "rendered_dispatches": rendered_dispatches,
        "df_change_orders": df_change_orders,
    }


if __name__ == "__main__":
    print("Agent 4 Comms Generator module loaded successfully.")
