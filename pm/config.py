"""
cGMP-QC-CoPilot: Operational Policies & Thresholds Configuration.

NOTE: All thresholds, RAG parameters, escalation rules, instrument capacities,
and role designations defined here are an EXAMPLE POLICY (CONFIGURABLE) for demonstration.
In an enterprise deployment, these parameters are configured per site SOPs.
"""

from typing import Any, Dict

# Example Policy, Configurable: RAG Status Thresholds
RAG_THRESHOLDS: Dict[str, Any] = {
    "policy_name": "Standard Site QC RAG Policy (Example Policy, Configurable)",
    "green_p_on_time_min": 0.90,
    "green_float_days_min": 2.0,
    "amber_p_on_time_min": 0.70,
    "red_p_on_time_threshold": 0.70,
    "red_float_days_threshold": 0.0,
    "disclaimer": "Example policy, configurable per site SOP",
}

# Example Policy, Configurable: Blocker Escalation Thresholds & Roles
BLOCKER_ESCALATION_POLICY: Dict[str, Any] = {
    "policy_name": "Standard Site Blocker Escalation Matrix (Example Policy, Configurable)",
    "tier1_aging_threshold_days": 5,
    "tier1_escalation_role": "Director of Quality Assurance (Example Policy, Configurable)",
    "tier2_tier3_aging_threshold_days": 10,
    "tier2_tier3_escalation_role": "QC Project Management Lead / Local MSO (Example Policy, Configurable)",
    "critical_severity_fast_track_days": 2,
    "disclaimer": "Example policy, configurable per site SOP",
}

# Example Policy, Configurable: Roadblock Escalation Tiers by Days Open (Module 2)
ROADBLOCK_TIER_CONFIG: Dict[str, Any] = {
    "policy_name": "Standard Site Roadblock Tier Escalation Policy (Example Policy, Configurable)",
    "team_tier_max_days": 3,        # 0 to 3 days: Team
    "qc_lead_tier_max_days": 7,     # 4 to 7 days: QC Lead
    # > 7 days: Quality Head
    "roles": {
        "Team": "Lab Operations Team / Shift Analysts (Example Policy, Configurable)",
        "QC Lead": "QC Project Management Lead / Local MSO (Example Policy, Configurable)",
        "Quality Head": "Director of Quality Assurance / Site Quality Head (Example Policy, Configurable)",
    },
    "disclaimer": "Example policy, configurable per site SOP",
}

# Example Policy, Configurable: Invented Physical Instrument Capacity Basis (Fictional Example)
INVENTED_INSTRUMENT_CAPACITY: Dict[str, Any] = {
    "policy_name": "Staffed Weekly Skid Capacity (Fictional Example)",
    "UPLC_SEC_CEX": {
        "display_name": "Waters Acquity H-Class Bio UPLC (SEC/CEX/RP) (fictional example)",
        "weekly_staffed_hours": 120.0,
        "weekly_physical_max_hours": 168.0,
        "lean_buffer_ceiling_pct": 85.0,
        "disclaimer": "Fictional example capacity",
    },
    "CE_icIEF": {
        "display_name": "ProteinSimple Maurice (icIEF & CE-SDS) (fictional example)",
        "weekly_staffed_hours": 100.0,
        "weekly_physical_max_hours": 168.0,
        "lean_buffer_ceiling_pct": 85.0,
        "disclaimer": "Fictional example capacity",
    },
    "Cell_Bioassay": {
        "display_name": "BSL-2 Hamilton Star / 96-Well 4-PL Potency Suite (fictional example)",
        "weekly_staffed_hours": 90.0,
        "weekly_physical_max_hours": 168.0,
        "lean_buffer_ceiling_pct": 85.0,
        "disclaimer": "Fictional example capacity",
    },
}
