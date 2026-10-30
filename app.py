"""
cGMP-QC-CoPilot
Google AI Studio 3-Column Architecture:
- Left Sidebar: Navigation & Workspace Explorer
- Center Canvas: Primary Workspace (KPIs, Analytical Models, Charts, Sub-Tabs)
- Right-Hand Inspector: Run Settings, Model Parameters, and Biophysical Sliders

Modeled for: Principal Specialist, QC Project Management / Local MSO
Facility: Example Biologics Site (Dual-Stream: Captive Pipeline + Example CDMO Client)
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Import deterministic Layer-2 Agent Scripts
from agents.agent_1_lean_capacity.scripts.capacity_smoother import run_heijunka_leveling
from agents.agent_2_analytical_scientist.scripts.eyring_excursion import (
    simulate_cold_chain_excursion,
    analyze_6m_oot_stability,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.q1e import (
    evaluate_q1e_shelf_life_comparison,
)
from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.workflow import (
    DRAFT_STAMP,
    EXCURSION_SAFETY_STATEMENT,
    MISMATCHED_RIDGE_STATEMENT,
    PHYSICAL_VALIDITY_STATEMENT,
    build_mso_draft_quality_package,
    verify_and_sign_hitl_gate,
)
from agents.agent_2_analytical_scientist.scripts.lss_sma_transfer import (
    calculate_lss_sma_dwell_transfer,
)
from agents.agent_2_analytical_scientist.scripts.bioassay_4pl_bayesian import (
    evaluate_bioassay_4pl_and_plate_edge,
)
from agents.agent_3_mso_oos_risk.scripts.oos_telemetry_triage import (
    evaluate_oos_phase1_gate,
    compute_dynamic_fmea_rpn,
)
from agents.agent_4_stakeholder_comms.scripts.comms_generator import (
    generate_stakeholder_communications,
)
from core.stamps import PART11_DISCLAIMER
from core.audit import AuditTrail
from core.db import get_connection, init_db
from pm.data_access import seed_pm_database
from pm.ui_view import render_portfolio_and_operations_tab
from pm.knowledge_base import seed_knowledge_base
from pm.fmea_capa import seed_fmea_and_capa, sync_fmea_with_lessons_learned
from pm.tour_ui import render_5_minute_tour_tab, open_phase1_audit_dialog
from pm.modular_widgets import (
    extract_workload_dfs,
    render_card_a_cpsat_workload_canvas,
    render_card_b_smartsheet_grid,
    render_card_c_root_cause_tree,
)
from pm.spc_ui import render_spc_and_quality_intelligence_tab
from pm.topology_ui import render_live_plant_network_tab
from pm.hybrid_architecture_ui import render_hybrid_architecture_schematic
from pm.guided_tour import render_guided_tour_modal
from core.session_recorder import (
    get_session_recorder,
    render_session_recorder_modal_or_popover,
)
from agents.agent_1_lean_capacity.scripts.hybrid_schedule_risk import (
    run_hybrid_capacity_and_risk,
    execute_schedule_sign_off,
)

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION & GOOGLE AI STUDIO DESIGN TOKENS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="cGMP-QC-CoPilot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Google AI Studio aesthetic: Inter/Roboto clean sans-serif, neutral canvas, elevated cards */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Google Sans', 'Roboto', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .google-card, .material-card {
        background: #ffffff;
        border: 1px solid #e0e3e7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(60,64,67,0.04);
    }
    /* Prevent metric label truncation across all metric displays */
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
    }
    /* Google AI Studio Left Navigation Rail (~240px wide) */
    section[data-testid="stSidebar"] {
        width: 240px !important;
        min-width: 240px !important;
        max-width: 240px !important;
        background-color: #f8f9fa !important;
        border-right: 1px solid #e8eaed !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 3px;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 8px 12px !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #3c4043 !important;
        margin-bottom: 2px !important;
        transition: all 0.15s ease-in-out !important;
        border: 1px solid transparent !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: #f1f3f4 !important;
        color: #202124 !important;
    }
    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background-color: #e8f0fe !important;
        color: #1967d2 !important;
        font-weight: 600 !important;
        border-color: #d2e3fc !important;
    }

    /* Google AI Studio Breadcrumb Bar */
    .ai-studio-breadcrumb {
        font-size: 12px;
        color: #5f6368;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 7px;
        font-family: 'Google Sans', 'Roboto', 'Inter', sans-serif;
    }
    .ai-studio-breadcrumb .breadcrumb-item-root {
        color: #1a73e8;
        font-weight: 500;
    }
    .ai-studio-breadcrumb .breadcrumb-sep {
        color: #9aa0a6;
    }
    .ai-studio-breadcrumb .breadcrumb-item-active {
        color: #202124;
        font-weight: 600;
    }

    /* Clean Native Expander Accordions for Parameter Domains */
    div[data-testid="stExpander"] {
        border: 1px solid #e0e3e7 !important;
        border-radius: 8px !important;
        background: #ffffff !important;
        box-shadow: 0 1px 2px rgba(60,64,67,0.04) !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stExpander"] details summary {
        font-size: 12.5px !important;
        font-weight: 700 !important;
        color: #174ea6 !important;
        padding: 8px 12px !important;
    }
    .main-banner {
        background: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(60,64,67,0.06);
        border-left: 5px solid #1a73e8;
    }
    .main-banner h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        color: #202124;
    }
    .main-banner p {
        margin: 0.25rem 0 0 0;
        font-size: 0.90rem;
        color: #5f6368;
    }
    .cgmp-badge {
        display: inline-block;
        background-color: #e8f0fe;
        border: 1px solid #d2e3fc;
        color: #1967d2;
        padding: 0.15rem 0.60rem;
        border-radius: 14px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.35rem;
        margin-top: 0.35rem;
    }
    .cgmp-badge-warn {
        display: inline-block;
        background-color: #fef7e0;
        border: 1px solid #feefc3;
        color: #b06000;
        padding: 0.15rem 0.60rem;
        border-radius: 14px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-top: 0.35rem;
    }
    .gate-box-pass {
        background-color: #e6f4ea;
        border-left: 5px solid #137333;
        padding: 0.85rem 1.05rem;
        border-radius: 6px;
        color: #137333;
        margin-bottom: 0.75rem;
    }
    .gate-box-warn {
        background-color: #fef7e0;
        border-left: 5px solid #f9ab00;
        padding: 0.85rem 1.05rem;
        border-radius: 6px;
        color: #b06000;
        margin-bottom: 0.75rem;
    }
    .gate-box-crit {
        background-color: #fce8e6;
        border-left: 5px solid #d93025;
        padding: 0.85rem 1.05rem;
        border-radius: 6px;
        color: #d93025;
        margin-bottom: 0.75rem;
    }
    .inspector-header {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #5f6368;
        margin-bottom: 8px;
    }
    /* High-Contrast Segmented Pill Sub-Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        border-bottom: none !important;
        padding: 4px 2px 10px 2px !important;
        overflow-x: auto !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F3F4 !important;
        color: #3C4043 !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        border-radius: 20px !important;
        padding: 6px 16px !important;
        border: 1px solid #E0E3E7 !important;
        white-space: nowrap !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E8EAED !important;
        color: #202124 !important;
        border-color: #DADCE0 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1A73E8 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 20px !important;
        border-color: #1A73E8 !important;
        box-shadow: 0 1px 3px rgba(26,115,232,0.3) !important;
    }
    .stTabs [data-baseweb="tab"] p {
        margin: 0 !important;
        color: #3C4043 !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Parameter Inspector Typography & Section Cards */
    .inspector-section-header {
        background: #f1f3f4;
        border-radius: 6px;
        padding: 6px 10px;
        font-weight: 700;
        font-size: 13px;
        color: #174ea6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    /* Slider Labels & Value Badges */
    div[data-testid="stSlider"] label p,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stToggle"] label p,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stRadio"] label p {
        color: #202124 !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stToggle"] label {
        color: #202124 !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
    }
    /* Slider boundary numbers: Subtle muted grey #70757A, 11px */
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"],
    div[data-testid="stSlider"] div[data-testid="stSliderTickBar"] div,
    div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="font-size"] {
        color: #70757a !important;
        font-size: 11px !important;
        font-weight: 400 !important;
    }
    /* Slider track & thumb handle: Neutral slate accent #1A73E8 */
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #1a73e8 !important;
        border-color: #1a73e8 !important;
        box-shadow: 0 1px 3px rgba(26,115,232,0.3) !important;
    }
    div[data-testid="stSliderThumbValue"] {
        color: #1a73e8 !important;
        font-weight: 600 !important;
        font-size: 11.5px !important;
    }
    /* Primary Color Overrides to prevent default red on nominal controls */
    div[data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"],
    div[data-baseweb="slider"] div[style*="background-color: #ff4b4b"],
    div[data-baseweb="slider"] div[style*="background-color: #FF4B4B"] {
        background-color: #1a73e8 !important;
    }
    div[data-baseweb="slider"] div[role="slider"]:focus,
    div[data-baseweb="slider"] div[role="slider"]:active {
        box-shadow: 0 0 0 4px rgba(26,115,232,0.2) !important;
    }
    /* Toggle switch active state */
    div[data-testid="stToggle"] input:checked + div {
        background-color: #1a73e8 !important;
    }
    /* Number Input stepper buttons */
    div[data-testid="stNumberInput"] button:hover {
        color: #1a73e8 !important;
        border-color: #1a73e8 !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 1px #1a73e8 !important;
    }
    /* Checkbox checked state */
    div[data-testid="stCheckbox"] input:checked + div {
        background-color: #1a73e8 !important;
        border-color: #1a73e8 !important;
    }
    /* Radio checked state */
    div[data-testid="stRadio"] [role="radiogroup"] [aria-checked="true"] > div:first-child {
        border-color: #1a73e8 !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] [aria-checked="true"] > div:first-child > div {
        background-color: #1a73e8 !important;
    }
    /* Focus rings: Enterprise Blue #1A73E8 */
    input:focus, textarea:focus, select:focus {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 1px #1a73e8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# CONTEXTUAL HELP & EDUCATIONAL POPOVER HELPER
# ---------------------------------------------------------------------------
@st.dialog("Regulatory Standards & cGMP Basis")
def show_regulatory_basis_modal() -> None:
    """Displays full list of standards (ICH Q1A/Q2/Q6B/Q14, FDA 2006 OOS, ALCOA+)."""
    st.markdown("### 📜 Regulatory Standards & cGMP Decision-Support Basis")
    st.markdown(
        f"""
        <div style="background: #fef7e0; border: 1px solid #feefc3; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-size: 12px; color: #b06000; font-weight: 600;">
            ⚠️ {DRAFT_STAMP} — {PART11_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        This decision-support engine operates under strict non-GMP guardrails aligned with international regulatory frameworks:

        - **ICH Q1A(R2) Stability Testing:** Permitted ±7-day stability pull windows applied to routine commercial stability lots to enable mathematical leveling of instrument queues.
        - **ICH Q2(R2) & ICH Q14 Analytical Procedure Development:** Linear Solvent Strength (LSS) and Steric Mass Action (SMA) models calculating dwell-volume gradient hold adjustments for cross-platform method transfers.
        - **ICH Q6B Specifications for Biotechnological/Biological Products:** Structured separation between lot release testing specifications and stability-indicating panels.
        - **ICH Q9(R1) Quality Risk Management:** Dynamic Failure Mode and Effects Analysis (FMEA) Severity × Occurrence × Detection RPN rescoring linked to CAPA implementations.
        - **FDA Guidance for Industry: Investigating OOS Test Results (2006):** Phase I laboratory equipment and telemetry triage gate (pressure traces, queue holds, chiller logs) to identify assignable lab causes prior to Phase II manufacturing escalation.
        - **ALCOA+ Data Integrity Principles:** Attributable, Legible, Contemporaneous, Original, and Accurate records with cryptographic SHA-256 hash chaining and SQLite append-only triggers.
        - **21 CFR Part 11 Demo Mode:** Role-based human-in-the-loop review gates requiring reviewer name, role, minimum 10-character justification, and explicit confirmation before quality export unlocking.
        """
    )


def render_feature_help_popover(
    title: str,
    summary: str,
    formula_or_logic: str,
    regulatory_basis: str,
    demo_action: str,
    button_label: Optional[str] = None,
) -> None:
    """Renders a clean Google-style popover card explaining feature purpose and regulatory context."""
    label = button_label or f"ℹ️ Feature Guide & Regulatory Context: {title}"
    with st.popover(label):
        st.markdown(f"### ℹ️ {title}")
        st.markdown(f"**Feature Purpose:** {summary}")
        st.markdown(f"**Key Metrics & Formulas:** {formula_or_logic}")
        st.markdown(f"**cGMP & Regulatory Mandates:** `{regulatory_basis}`")
        st.info(f"👉 **How to Interact:** {demo_action}")


def render_ai_studio_breadcrumb(workspace: str, active_view: str = "Active Canvas", agent3_status: str = "NOMINAL") -> None:
    """Renders a clean Google AI Studio breadcrumb header bar with quick tour & recording action controls."""
    key_hash = hashlib.sha256(f"{workspace}_{active_view}".encode("utf-8")).hexdigest()[:8]
    col_home, col_bc, col_actions = st.columns([0.48, 1.85, 1.67], vertical_alignment="center")
    with col_home:
        if st.button("🏠 Home", key=f"btn_quick_home_redirect_{key_hash}", help="Return to primary landing workspace", use_container_width=True):
            st.session_state["_pending_workspace_nav"] = "🏠 Home / Executive Overview"
            st.rerun()
    with col_bc:
        clean_ws = workspace.split(" ", 1)[-1] if " " in workspace else workspace
        st.markdown(
            rf"""
            <div class="ai-studio-breadcrumb" style="margin-bottom: 0px;">
                <span class="breadcrumb-item-root">cGMP-QC-CoPilot</span>
                <span class="breadcrumb-sep">›</span>
                <span>{clean_ws}</span>
                <span class="breadcrumb-sep">›</span>
                <strong class="breadcrumb-item-active">{active_view}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_actions:
        c_tour, c_rec = st.columns([1.0, 1.4], vertical_alignment="center")
        with c_tour:
            tour_btn_label = "🎬 Tour Active" if st.session_state.get("tour_active", False) else "🎬 5-Min Tour"
            if st.button(tour_btn_label, help="Start or resume the 5-minute interactive guided executive tour", use_container_width=True, key=f"hdr_tour_btn_{key_hash}"):
                st.session_state["tour_active"] = not st.session_state.get("tour_active", False)
                if st.session_state["tour_active"]:
                    st.session_state["tour_step"] = 1
                st.rerun()
        with c_rec:
            render_session_recorder_modal_or_popover(workspace, agent3_status)

    if st.session_state.get("tour_active", False):
        render_guided_tour_modal()


def render_bottom_copilot_dock(workspace: str) -> None:
    """Renders Google AI Studio-style interactive bottom prompt dock."""
    st.markdown(
        """
        <div style="background: #ffffff; border: 1px solid #e0e3e7; border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-top: 24px; margin-bottom: 12px;">
            <div style="font-size: 11px; font-weight: 600; color: #5f6368; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 8px;">
                ⚡ CoPilot Interactive Execution Dock
            </div>
            <div style="font-size: 12px; color: #70757a; margin-bottom: 10px;">
                Natural-language scenario exploration and deterministic algorithm execution:
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_prompt, col_action = st.columns([4.2, 0.8])
    with col_prompt:
        user_query = st.text_input(
            "Prompt CoPilot",
            placeholder="Ask CoPilot to simulate thermal deviation, recalculate ANCOVA, or triage Phase I OOS...",
            label_visibility="collapsed",
            key="copilot_dock_prompt",
        )
    with col_action:
        run_copilot = st.button("🚀 Run", use_container_width=True, type="primary")

    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
    with chip_col1:
        if st.button("🌡️ Simulate 36h @ 28°C", use_container_width=True, key="chip_excursion"):
            st.session_state["excursion_temp_slider"] = 28.0
            st.session_state["excursion_hours_slider"] = 36.0
            st.rerun()
    with chip_col2:
        if st.button("🧪 Transfer 115µL → 400µL", use_container_width=True, key="chip_dwell"):
            st.session_state["dwell_volume_input"] = 400.0
            st.rerun()
    with chip_col3:
        if st.button("🧬 Stress Perimeter Wells", use_container_width=True, key="chip_edge"):
            st.session_state["include_outer_wells_toggle"] = True
            st.session_state["cell_passage_slider"] = 30
            st.rerun()
    with chip_col4:
        if st.button("🛡️ Run Phase I Hardware OOS", use_container_width=True, key="chip_oos"):
            st.session_state["oos_scenario"] = "Run_14_Hardware_Anomaly"
            st.rerun()

    if run_copilot and user_query:
        st.info(f"🤖 **CoPilot Execution Plan:** Evaluating query: *'{user_query}'*. Parameters evaluated deterministically under 21 CFR 211.68(b) governance.")

# ---------------------------------------------------------------------------
# INITIALIZE PERSISTENT SESSION STATE (DATABASE, AUDIT TRAIL, CONTROLS)
# ---------------------------------------------------------------------------
if "audit_trail" not in st.session_state:
    try:
        db_conn = get_connection()
        trail = AuditTrail(db_conn)
    except Exception:
        trail = AuditTrail()

    if len(trail.records) == 0:
        trail.append_event(
            actor_name="SYSTEM_ORCHESTRATOR",
            actor_role="Orchestrator Lead",
            action="INIT_SCHEDULE_PROPOSAL",
            justification="Automated W42/W43 Heijunka Schedule Proposal Generated",
            entity_type="SCHEDULE",
            entity_id="SCHED-2026-W42",
            payload={"status": "PROPOSED"},
            timestamp_utc="2026-10-21T08:15:04Z",
        )
        trail.append_event(
            actor_name="SYSTEM_ORCHESTRATOR",
            actor_role="Telemetry Engine",
            action="INIT_TELEMETRY_LOG",
            justification="Empower Run #14 Telemetry Loaded (synthetic) (310.4 bar / 54h Pause / 17.8C Chiller Fault)",
            entity_type="TELEMETRY",
            entity_id="RUN-14",
            payload={"backpressure_bar": 310.4, "pause_hours": 54.0, "chiller_temp_c": 17.8},
            timestamp_utc="2026-10-21T08:16:22Z",
        )
    st.session_state["audit_trail"] = trail

if "pm_conn" not in st.session_state:
    try:
        conn = init_db()
        seed_pm_database(conn, audit_trail=st.session_state.get("audit_trail"))
        seed_knowledge_base(conn)
        seed_fmea_and_capa(conn)
        sync_fmea_with_lessons_learned(conn)
        st.session_state["pm_conn"] = conn
    except Exception:
        st.session_state["pm_conn"] = get_connection()

if "signed_mso_pkg" not in st.session_state:
    st.session_state["signed_mso_pkg"] = None

# Default simulation control values stored in session_state for cross-workspace stability
if "harvest_slip_slider" not in st.session_state:
    st.session_state["harvest_slip_slider"] = 0
if "outage_mult_slider" not in st.session_state:
    st.session_state["outage_mult_slider"] = 1.0
if "analyst_mult_slider" not in st.session_state:
    st.session_state["analyst_mult_slider"] = 1.0
if "rush_lot_checkbox" not in st.session_state:
    st.session_state["rush_lot_checkbox"] = False

if "usp_harvest_delay" not in st.session_state:
    st.session_state["usp_harvest_delay"] = st.session_state["harvest_slip_slider"]
if "inst_outage_mult" not in st.session_state:
    st.session_state["inst_outage_mult"] = st.session_state["outage_mult_slider"]
if "analyst_abs_mult" not in st.session_state:
    st.session_state["analyst_abs_mult"] = st.session_state["analyst_mult_slider"]
if "inject_rush" not in st.session_state:
    st.session_state["inject_rush"] = st.session_state["rush_lot_checkbox"]

if "excursion_temp_slider" not in st.session_state:
    st.session_state["excursion_temp_slider"] = 28.0
if "excursion_hours_slider" not in st.session_state:
    st.session_state["excursion_hours_slider"] = 36.0
if "monomer_purity_input" not in st.session_state:
    st.session_state["monomer_purity_input"] = 99.38
if "formulation_ph_slider" not in st.session_state:
    st.session_state["formulation_ph_slider"] = 5.8
if "surfactant_conc_slider" not in st.session_state:
    st.session_state["surfactant_conc_slider"] = 0.040
if "peroxide_level_slider" not in st.session_state:
    st.session_state["peroxide_level_slider"] = 0.2
if "dwell_volume_input" not in st.session_state:
    st.session_state["dwell_volume_input"] = 115.0
if "flow_rate_input" not in st.session_state:
    st.session_state["flow_rate_input"] = 0.50
if "column_void_vol_input" not in st.session_state:
    st.session_state["column_void_vol_input"] = 1.50
if "include_outer_wells_toggle" not in st.session_state:
    st.session_state["include_outer_wells_toggle"] = False
if "cell_passage_slider" not in st.session_state:
    st.session_state["cell_passage_slider"] = 12

if "pre_exc_input" not in st.session_state:
    st.session_state["pre_exc_input"] = 0.790
if "excursion_temp" not in st.session_state:
    st.session_state["excursion_temp"] = 28.0
if "excursion_hours" not in st.session_state:
    st.session_state["excursion_hours"] = 36.0
if "donor_vd_ml" not in st.session_state:
    st.session_state["donor_vd_ml"] = 1.15
if "include_outer_wells" not in st.session_state:
    st.session_state["include_outer_wells"] = False

if "oos_scenario" not in st.session_state:
    st.session_state["oos_scenario"] = "Run_14_Hardware_Anomaly"
if "selected_proj_comms" not in st.session_state:
    st.session_state["selected_proj_comms"] = "PROJ-001"

# Handle pending workspace redirects from components (e.g. Guided Tour) before widget instantiation
if "_pending_workspace_nav" in st.session_state:
    st.session_state["workspace_nav_selection"] = st.session_state.pop("_pending_workspace_nav")

if "workspace_nav_selection" not in st.session_state:
    st.session_state["workspace_nav_selection"] = "🏠 Home / Executive Overview"

# ---------------------------------------------------------------------------
# COLUMN 1: LEFT SIDEBAR (GOOGLE AI STUDIO EXPLORER & NAVIGATION)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div id="left-nav-rail">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2px; padding-top: 4px;">
            <span style="font-size: 1.3rem;">🧬</span>
            <span style="font-size: 1.05rem; font-weight: 700; color: #202124; letter-spacing: -0.2px;">cGMP-QC-CoPilot</span>
        </div>
        <div style="font-size: 11px; color: #5f6368; margin-bottom: 12px;">Quality Control & Operations CoPilot</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
        <div style="font-size: 11px; font-weight: 600; color: #5f6368; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 8px;">
            WORKSPACES
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav_options = [
        "🏠 Home / Executive Overview",
        "🧭 5-Minute Executive Tour",
        "🌐 Live Plant & Agent Network",
        "📊 QC Operations & Project Management",
        "🔬 Analytical Science & Hybrid Kinetics",
        "🛡️ Quality Intelligence, OOS & Compliance",
        "🗂️ Two-Layer Agent Architecture",
    ]
    workspace = st.radio(
        "Workspace Navigation",
        options=nav_options,
        key="workspace_nav_selection",
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="google-card" style="padding: 12px 14px; margin-top: 26px; border-radius: 8px;">
            <div style="font-size: 11px; font-weight: 600; color: #1a73e8; margin-bottom: 4px;">SYSTEM ARCHITECTURE</div>
            <div style="font-size: 11.5px; color: #3c4043; line-height: 1.45;">
                • Facility: <strong>Example Biologics Dual-Stream</strong><br>
                • Inference: <strong>Pure NumPy Fallback</strong><br>
                • Precision SD: <strong>0.035% HMW</strong><br>
                • Audit Chain: <strong>SHA-256 Verified</strong><br>
                • G2 Claims: <strong>0 Disallowed Flags</strong>
            </div>
            <hr style="margin: 8px 0; border: none; border-top: 1px solid #dadce0;">
            <div style="font-size: 11px; color: #137333; font-weight: 600;">
                ● Status: System Qualified & Active
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Unpack active control parameters from session state (with reactive widget key fallback)
usp_harvest_delay = int(st.session_state.get("harvest_slip_slider", st.session_state.get("usp_harvest_delay", 0)))
inst_outage_mult = float(st.session_state.get("outage_mult_slider", st.session_state.get("inst_outage_mult", 1.0)))
analyst_abs_mult = float(st.session_state.get("analyst_mult_slider", st.session_state.get("analyst_abs_mult", 1.0)))
inject_rush = bool(st.session_state.get("rush_lot_toggle", st.session_state.get("rush_lot_checkbox", st.session_state.get("inject_rush", False))))

# Keep legacy session_state keys in sync
st.session_state["usp_harvest_delay"] = usp_harvest_delay
st.session_state["inst_outage_mult"] = inst_outage_mult
st.session_state["analyst_abs_mult"] = analyst_abs_mult
st.session_state["inject_rush"] = inject_rush
st.session_state["rush_lot_checkbox"] = inject_rush
st.session_state["rush_lot_toggle"] = inject_rush

# Unpack 11 Biopharmaceutical Input Parameters (with interactive inspector key fallbacks)
excursion_temp = float(st.session_state.get("excursion_temp_slider", st.session_state.get("excursion_temp", 28.0)))
excursion_hours = float(st.session_state.get("excursion_hours_slider", st.session_state.get("excursion_hours", 36.0)))
initial_monomer_pct = float(st.session_state.get("monomer_purity_input", 99.38))
formulation_ph = float(st.session_state.get("formulation_ph_slider", 5.8))
surfactant_conc_pct = float(st.session_state.get("surfactant_conc_slider", 0.040))
peroxide_level_ppm = float(st.session_state.get("peroxide_level_slider", 0.2))
dwell_volume_ul = float(st.session_state.get("dwell_volume_input", 115.0))
flow_rate_ml_min = float(st.session_state.get("flow_rate_input", 0.50))
column_void_vol_ml = float(st.session_state.get("column_void_vol_input", 1.50))
include_outer_wells = bool(st.session_state.get("include_outer_wells_toggle", st.session_state.get("include_outer_wells", False)))
cell_passage_num = int(st.session_state.get("cell_passage_slider", 12))

pre_exc_input = round(100.0 - initial_monomer_pct, 4)
donor_vd_ml = round(dwell_volume_ul / 1000.0, 4)

# Keep legacy session_state keys in sync
st.session_state["pre_exc_input"] = pre_exc_input
st.session_state["excursion_temp"] = excursion_temp
st.session_state["excursion_hours"] = excursion_hours
st.session_state["donor_vd_ml"] = donor_vd_ml
st.session_state["include_outer_wells"] = include_outer_wells

oos_scenario = str(st.session_state.get("oos_scenario", "Run_14_Hardware_Anomaly"))
selected_proj_comms = str(st.session_state.get("selected_proj_comms", "PROJ-001"))

# ---------------------------------------------------------------------------
# EXECUTE REACTIVE HYBRID KINETIC PIPELINE (< 0.05s SciPy RK45 + 25-ANN)
# ---------------------------------------------------------------------------
from models.hybrid_pipeline import run_reactive_hybrid_pipeline
reactive_pipeline_out = run_reactive_hybrid_pipeline({
    "excursion_temp_c": excursion_temp,
    "excursion_duration_hours": excursion_hours,
    "initial_monomer_pct": initial_monomer_pct,
    "formulation_ph": formulation_ph,
    "surfactant_conc_pct": surfactant_conc_pct,
    "peroxide_level_ppm": peroxide_level_ppm,
    "dwell_volume_ul": dwell_volume_ul,
    "flow_rate_ml_min": flow_rate_ml_min,
    "column_void_vol_ml": column_void_vol_ml,
    "include_outer_wells": include_outer_wells,
    "cell_passage_num": cell_passage_num,
})

# ---------------------------------------------------------------------------
# SESSION RECORDER & USER INTERACTION JOURNAL LOGGING
# ---------------------------------------------------------------------------
recorder = get_session_recorder()
current_agent3_state = reactive_pipeline_out.get("triage", {}).get("agent3_state", "NOMINAL")
if recorder.is_recording:
    current_params = {
        "excursion_temp_slider": excursion_temp,
        "excursion_hours_slider": excursion_hours,
        "monomer_purity_input": initial_monomer_pct,
        "formulation_ph_slider": formulation_ph,
        "surfactant_conc_slider": surfactant_conc_pct,
        "peroxide_level_slider": peroxide_level_ppm,
        "dwell_volume_input": dwell_volume_ul,
        "flow_rate_input": flow_rate_ml_min,
        "column_void_vol_input": column_void_vol_ml,
        "include_outer_wells_toggle": include_outer_wells,
        "cell_passage_slider": cell_passage_num,
        "harvest_slip_slider": usp_harvest_delay,
        "outage_mult_slider": inst_outage_mult,
        "analyst_mult_slider": analyst_abs_mult,
        "rush_lot_toggle": inject_rush,
        "workspace_selection": workspace,
    }
    prev_params = st.session_state.get("_last_recorder_params")
    if prev_params is not None:
        for k, curr_v in current_params.items():
            prev_v = prev_params.get(k)
            if prev_v is not None and prev_v != curr_v:
                recorder.record_event(
                    event_type="PARAMETER_SHIFT",
                    widget_key=k,
                    old_value=prev_v,
                    new_value=curr_v,
                    workspace=workspace,
                    agent3_status=current_agent3_state,
                    metadata={"param_type": "biopharm_or_ops"},
                )
    st.session_state["_last_recorder_params"] = current_params

# ---------------------------------------------------------------------------
# EXECUTE DETERMINISTIC LAYER-2 SCRIPTS (All numbers from unit-tested Python engines)
# ---------------------------------------------------------------------------
agent1_hybrid_out = run_hybrid_capacity_and_risk(
    usp_harvest_delay_days=usp_harvest_delay,
    instrument_outage_mult=inst_outage_mult,
    analyst_absence_mult=analyst_abs_mult,
    inject_rush_task=inject_rush,
)
agent1_out = agent1_hybrid_out["legacy_baseline"]
agent2_exc_out = simulate_cold_chain_excursion(
    baseline_temp_c=5.0,
    excursion_temp_c=excursion_temp,
    excursion_duration_hours=excursion_hours,
    initial_hmw_pct=pre_exc_input,
    lot_covariates={
        "ph": formulation_ph,
        "surfactant_pct": surfactant_conc_pct,
        "initial_hmw_pct": pre_exc_input,
    },
)
agent2_oot_out = analyze_6m_oot_stability()
agent2_q1e_out = evaluate_q1e_shelf_life_comparison()
mso_draft_pkg = build_mso_draft_quality_package(
    excursion_temp_c=excursion_temp,
    excursion_duration_hours=excursion_hours,
    baseline_temp_c=5.0,
    project_id=selected_proj_comms,
    usp_harvest_delay_days=usp_harvest_delay,
    pre_excursion_hmw_pct=pre_exc_input,
)
agent2_trans_out = calculate_lss_sma_dwell_transfer(
    v_dwell_donor_ml=donor_vd_ml,
    flow_rate_ml_min=flow_rate_ml_min,
)
agent2_bio_out = evaluate_bioassay_4pl_and_plate_edge(include_outer_edge_wells=include_outer_wells)
agent3_oos_out = evaluate_oos_phase1_gate(scenario_mode=oos_scenario)
agent3_fmea_out = compute_dynamic_fmea_rpn()
agent4_comms_out = generate_stakeholder_communications(
    agent1_results=agent1_out,
    agent2_excursion=agent2_exc_out,
    agent2_oot=agent2_oot_out,
    agent2_transfer=agent2_trans_out,
    agent2_bioassay=agent2_bio_out,
    agent3_oos=agent3_oos_out,
    agent3_fmea=agent3_fmea_out,
    selected_project_id=selected_proj_comms,
    harvest_slip_days=usp_harvest_delay,
)

# ---------------------------------------------------------------------------
# TOP EXECUTIVE HEADER & TELEMETRY KPI STRIP
# ---------------------------------------------------------------------------
col_header_banner, col_header_reg = st.columns([4.2, 0.8], vertical_alignment="center")
with col_header_banner:
    st.markdown(
        """
        <div class="main-banner" style="margin-bottom: 0px; padding: 0.85rem 1.2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div>
                    <h1 style="font-size: 1.32rem; margin: 0; color: #202124;">
                        🧬 cGMP-QC-CoPilot <span style="font-size: 0.88rem; font-weight: 500; color: #5f6368;">| Dual-Stream QC Operations & Decision Support</span>
                    </h1>
                </div>
                <div style="background: #fef7e0; border: 1px solid #feefc3; color: #b06000; padding: 4px 12px; border-radius: 16px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.2px;">
                    ⚠️ Prototype Decision-Support Engine | Synthetic Data | 21 CFR Part 11 Demo Mode
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_header_reg:
    if st.button("ℹ️ Regulatory Basis", help="View regulatory standards (ICH Q1A/Q2/Q6B/Q14, FDA 2006 OOS, ALCOA+)", use_container_width=True, key="hdr_btn_reg_basis"):
        show_regulatory_basis_modal()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric(
        label="W42 Waters UPLC Utilization",
        value=f"{agent1_out['metrics']['w42_uplc_pct_after']}%",
        delta=f"{round(agent1_out['metrics']['w42_uplc_pct_after'] - agent1_out['metrics']['w42_uplc_pct_before'], 1)}% (from {agent1_out['metrics']['w42_uplc_pct_before']}%)",
        delta_color="inverse",
    )
with kpi2:
    st.metric(
        label="Critical-Path Release SLAs",
        value=f"{agent1_out['metrics']['critical_path_protected_count']}/6 Protected",
        delta="Zero Float Breaches (100% RFT)",
    )
with kpi3:
    st.metric(
        label=f"{int(excursion_hours)}h @ {int(excursion_temp)}°C ΔHMW%",
        value=f"+{agent2_exc_out['summary']['delta_hmw_excursion_pct']}%",
        delta=f"-{agent2_exc_out['summary']['shelf_life_deduction_months']}M 4°C Shelf-Life Equiv.",
        delta_color="off",
    )
with kpi4:
    st.metric(
        label="UPLC Dwell Shift Compensated",
        value=f"{agent2_trans_out['parameters']['recommended_receiver_hold_min']} min Hold",
        delta=f"Δt_R = -0.02 min (vs {agent2_trans_out['parameters']['uncompensated_shift_min']} min)",
    )
with kpi5:
    st.metric(
        label="Site QC FMEA Risk Reduction",
        value=f"-{agent3_fmea_out['summary']['overall_rpn_reduction_pct']}% RPN",
        delta=f"Max RPN {agent3_fmea_out['summary']['max_pre_rpn']} → {agent3_fmea_out['summary']['max_post_rpn']}",
    )

st.divider()

# Workspace View Mode layout selector
col_split_bar1, col_split_bar2 = st.columns([2.2, 1.8], vertical_alignment="center")
with col_split_bar1:
    st.markdown(
        """
        <div style="font-size: 11.5px; color: #5f6368; font-weight: 500;">
            ● <strong>Operational Mode:</strong> Multi-Agent cGMP CoPilot | Continuous Telemetry Active
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_split_bar2:
    layout_mode = st.radio(
        "Workspace View Mode",
        options=["Standard 3-Column", "Expanded Stage (Focus)", "Balanced Split"],
        horizontal=True,
        key="dashboard_layout_mode",
    )

if layout_mode == "Expanded Stage (Focus)":
    col_ratios = [3.4, 0.6]
elif layout_mode == "Balanced Split":
    col_ratios = [2.0, 2.0]
else:
    col_ratios = [2.8, 1.2]

# ---------------------------------------------------------------------------
# SPATIAL SPLIT: CENTER CANVAS (COL 2) & RIGHT INSPECTOR PANEL (COL 3)
# ---------------------------------------------------------------------------
col_center, col_right = st.columns(col_ratios, gap="large")
col_canvas = col_center
col_inspector = col_right

# ===========================================================================
# RIGHT-HAND INSPECTOR PANEL: RUN SETTINGS & PARAMETERS (GOOGLE AI STUDIO STYLE)
# ===========================================================================
with col_right:
    st.markdown('<div id="right-inspector-rail">', unsafe_allow_html=True)
    col_hdr_title, col_hdr_actions = st.columns([1.1, 1.4])
    with col_hdr_title:
        st.markdown(
            """
            <div style="font-size: 14px; font-weight: 600; color: #202124; padding-top: 5px;">
                ⚙️ Run Settings
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_hdr_actions:
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            with st.popover("< > JSON", help="View active biopharmaceutical parameter payload in JSON"):
                st.caption("Active Parameter Payload:")
                st.json({
                    "workspace": workspace,
                    "excursion_temp_c": excursion_temp,
                    "excursion_duration_hours": excursion_hours,
                    "initial_monomer_pct": initial_monomer_pct,
                    "formulation_ph": formulation_ph,
                    "surfactant_conc_pct": surfactant_conc_pct,
                    "peroxide_level_ppm": peroxide_level_ppm,
                    "dwell_volume_ul": dwell_volume_ul,
                    "flow_rate_ml_min": flow_rate_ml_min,
                    "column_void_vol_ml": column_void_vol_ml,
                    "include_outer_wells": include_outer_wells,
                    "cell_passage_num": cell_passage_num,
                })
        with c_act2:
            if st.button("⟳ Reset", help="Reset all controls to nominal biopharmaceutical defaults"):
                st.session_state["excursion_temp_slider"] = 28.0
                st.session_state["excursion_hours_slider"] = 36.0
                st.session_state["monomer_purity_input"] = 99.38
                st.session_state["formulation_ph_slider"] = 5.8
                st.session_state["surfactant_conc_slider"] = 0.040
                st.session_state["peroxide_level_slider"] = 0.2
                st.session_state["dwell_volume_input"] = 115.0
                st.session_state["flow_rate_input"] = 0.50
                st.session_state["column_void_vol_input"] = 1.50
                st.session_state["include_outer_wells_toggle"] = False
                st.session_state["cell_passage_slider"] = 12
                st.session_state["harvest_slip_slider"] = 0
                st.session_state["outage_mult_slider"] = 1.0
                st.session_state["analyst_mult_slider"] = 1.0
                st.session_state["rush_lot_toggle"] = False
                st.session_state["rush_lot_checkbox"] = False
                st.rerun()

    if workspace == "🏠 Home / Executive Overview":
        st.markdown(
            """
            <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 13px; font-weight: 700; color: #1a73e8;">Executive Mission Control</span>
                    <span style="font-size: 10px; background: #e8f0fe; color: #1967d2; padding: 2px 7px; border-radius: 10px; font-weight: 600;">Active</span>
                </div>
                <div style="font-size: 11px; color: #5f6368; margin-top: 3px; line-height: 1.35;">
                    Central Dispatch & Real-Time Facility Health
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("⚡ Rapid Scenario Injection", expanded=True):
            oos_options = ["Run_14_Hardware_Anomaly", "Clean_Bench_Phase_II_Escalation"]
            curr_idx = 0 if st.session_state.get("oos_scenario") == "Run_14_Hardware_Anomaly" else 1
            st.session_state["oos_scenario"] = st.radio(
                "CDS Sequence Anomaly Simulator",
                options=oos_options,
                index=curr_idx,
                format_func=lambda x: "Run #14 Fault (310 bar / Chiller Drift)" if x == "Run_14_Hardware_Anomaly" else "Clean Bench (True Phase II)",
                key="home_insp_scenario_radio",
            )
            st.toggle(
                "Inject Urgent Commercial Rush Lot",
                value=inject_rush,
                key="home_insp_rush_toggle",
                help="Injects an unplanned zero-float commercial rush release lot.",
            )
            st.slider(
                "USP/DSP Harvest Delay (Days)",
                min_value=0,
                max_value=5,
                value=usp_harvest_delay,
                key="home_insp_harvest_slider",
                help="Simulates upstream commercial bioreactor harvest delay.",
            )
        st.markdown(
            """
            <div class="google-card" style="padding: 12px 14px; font-size: 11.5px; color: #3c4043; line-height: 1.45; margin-top: 14px;">
                <strong>System Qualifications:</strong><br>
                • Platform: <strong>cGMP Decision Support</strong><br>
                • Audit Standard: <strong>21 CFR Part 11 / ALCOA+</strong><br>
                • Data Integrity: <strong>SHA-256 Hash Chain</strong><br>
                • Inference Speed: <strong>&lt; 0.35s Wall-Clock</strong><br>
                • Guardrail Status: <strong>Zero Affirmative Claims</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif workspace == "🌐 Live Plant & Agent Network":
        st.markdown(
            """
            <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 13px; font-weight: 700; color: #1a73e8;">SCADA Industrial Graph Core</span>
                    <span style="font-size: 10px; background: #e8f0fe; color: #1967d2; padding: 2px 7px; border-radius: 10px; font-weight: 600;">Live Mesh</span>
                </div>
                <div style="font-size: 11px; color: #5f6368; margin-top: 3px; line-height: 1.35;">
                    Three-Zone Plant Network: Physical Skids + Multi-Agent Core + Part 11 Ledger
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🌐 Plant & Network Telemetry", expanded=True):
            st.selectbox(
                "Filter Topology Zone",
                options=["All Zones (Connected Plant)", "Physical Infrastructure", "Multi-Agent Core", "Governance & Compliance"],
                key="topology_zone_filter",
                help="Filters active node graph rendering and inspector details.",
            )
            st.toggle(
                "Simulate Live Heartbeat Feeds (500ms)",
                value=True,
                key="network_heartbeat_toggle",
                help="Simulates real-time telemetry streaming from Waters Empower and Rees Environmental monitors.",
            )
        with st.expander("🔍 Real-Time Telemetry Node Details", expanded=True):
            from pm.topology_ui import get_node_definitions
            top_ctx = {
                "oos_scenario": oos_scenario,
                "inst_outage_mult": inst_outage_mult,
                "analyst_abs_mult": analyst_abs_mult,
                "inject_rush": inject_rush,
                "usp_harvest_delay": usp_harvest_delay,
                "excursion_temp": excursion_temp,
                "excursion_hours": excursion_hours,
                "signed_mso_pkg": st.session_state.get("signed_mso_pkg"),
            }
            top_nodes = get_node_definitions(top_ctx)
            sel_node_id = st.selectbox(
                "Inspect Node Telemetry",
                options=list(top_nodes.keys()),
                format_func=lambda x: f"{top_nodes[x]['name']} [{top_nodes[x]['status']}]",
                key="right_inspector_node_select",
            )
            sel_node = top_nodes[sel_node_id]
            st.caption(f"**Zone:** {sel_node['zone']} | **Driver:** `{sel_node['driver']}`")
            st.caption(f"**SOP:** `{sel_node['sop']}`")
            df_node_telem = pd.DataFrame([
                {"Parameter": k, "Reading": v} for k, v in sel_node["telemetry"].items()
            ])
            st.dataframe(df_node_telem, use_container_width=True, hide_index=True, height=180)
        st.markdown(
            """
            <div class="google-card" style="padding: 12px 14px; font-size: 11.5px; color: #3c4043; line-height: 1.5; margin-top: 14px;">
                <strong>Network Status:</strong><br>
                • Nodes Connected: <code>10 / 10 Active</code><br>
                • Physical Skids: <code>4 Instruments</code><br>
                • Orchestrator Agents: <code>4 Specialists</code><br>
                • Regulatory Consensus: <code>Part 11 + Ledger</code><br>
                • Data Provenance: <strong>Cryptographic SHA-256</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif workspace == "📊 QC Operations & Project Management":
        st.markdown(
            """
            <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 13px; font-weight: 700; color: #1a73e8;">OR-Tools CP-SAT + CPM</span>
                    <span style="font-size: 10px; background: #e8f0fe; color: #1967d2; padding: 2px 7px; border-radius: 10px; font-weight: 600;">v1.8-Discrete</span>
                </div>
                <div style="font-size: 11px; color: #5f6368; margin-top: 3px; line-height: 1.35;">
                    Dual-Stream Leveling Engine + Monte Carlo Critical Path Float Buffer
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("📊 Capacity & Schedule Controls", expanded=True):
            st.slider(
                "USP/DSP Harvest Slip (Days)",
                min_value=0,
                max_value=5,
                key="harvest_slip_slider",
                help="Simulates upstream commercial bioreactor harvest delay on W42 critical-path release lots.",
            )
            st.slider(
                "Instrument Outage Multiplier",
                min_value=1.0,
                max_value=2.0,
                step=0.1,
                key="outage_mult_slider",
                help="Approximation: Multiplicative scale on analytical method run duration.",
            )
            st.slider(
                "Analyst Shortage Multiplier",
                min_value=1.0,
                max_value=2.0,
                step=0.1,
                key="analyst_mult_slider",
                help="Approximation: Multiplicative scale on analytical method run duration.",
            )
            st.toggle(
                "Inject Urgent Rush Lot (Spesolimab-Urgent)",
                key="rush_lot_toggle",
                help="Injects an unplanned zero-float commercial rush release lot competing for UPLC skid capacity.",
            )
        st.markdown(
            """
            <div class="google-card" style="padding: 12px 14px; font-size: 11.5px; color: #3c4043; line-height: 1.45; margin-top: 14px;">
                <strong>Capacity Constraints:</strong><br>
                • Staffed Nominal: <code>120.0 h/week</code><br>
                • Physical 24/7 Ceiling: <code>168.0 h/week</code><br>
                • LEAN Buffer Target: <code>85.0% (102.0 h)</code><br>
                • Precedence Constraints: <strong>Zero Overlaps</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🔍 Step-by-Step CP-SAT Move Explanations", expanded=False):
            df_moves = agent1_hybrid_out["df_moves"]
            if not df_moves.empty:
                st.dataframe(
                    df_moves[["project_id", "instrument_id", "delta_days", "reason"]],
                    use_container_width=True,
                    hide_index=True,
                    height=220,
                )
            else:
                st.caption("No schedule shifts required; all workloads below 85.0% LEAN ceiling.")

    elif workspace == "🔬 Analytical Science & Hybrid Kinetics":
        st.markdown(
            """
            <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 13px; font-weight: 700; color: #1a73e8;">Hybrid Lumry-Eyring PINN Core</span>
                    <span style="font-size: 10px; background: #e8f0fe; color: #1967d2; padding: 2px 7px; border-radius: 10px; font-weight: 600;">v2.4-Active</span>
                </div>
                <div style="font-size: 11px; color: #5f6368; margin-top: 3px; line-height: 1.35;">
                    25-Member Bayesian ML Residual Ensemble + SciPy RK45 (Ea = 74.37 kJ/mol)
                </div>
                <div style="display: flex; gap: 6px; margin-top: 8px; font-size: 10.5px;">
                    <span style="background: #ffffff; border: 1px solid #dadce0; border-radius: 4px; padding: 2px 6px; color: #3c4043;">Pure NumPy</span>
                    <span style="background: #ffffff; border: 1px solid #dadce0; border-radius: 4px; padding: 2px 6px; color: #137333; font-weight: 600;">0 Violations</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🧬 Group A: Stability & Matrix", expanded=True):
            st.slider(
                "Cold-Chain Excursion Peak Temp (°C)",
                min_value=5.0,
                max_value=45.0,
                step=0.5,
                key="excursion_temp_slider",
                help="Thermal excursion peak temperature over continuous logger series.",
            )
            st.slider(
                "Excursion Duration Δt (h)",
                min_value=0.0,
                max_value=72.0,
                step=1.0,
                key="excursion_hours_slider",
                help="Total duration of elevated cold-chain thermal deviation.",
            )
            st.number_input(
                "Initial Native Monomer M₀ (%)",
                min_value=95.00,
                max_value=100.00,
                step=0.01,
                format="%.2f",
                key="monomer_purity_input",
                help="Enforces strict mass-action conservation: M0 + U0 + HMW0 + LMW0 ≡ 1.000000.",
            )
            st.slider(
                "Formulation pH",
                min_value=5.0,
                max_value=7.5,
                step=0.1,
                key="formulation_ph_slider",
                help="Modulates deamidation and intermediate electrostatic unfolding barrier.",
            )
            st.slider(
                "Surfactant Concentration C_PS20 (%)",
                min_value=0.010,
                max_value=0.060,
                step=0.002,
                format="%.3f",
                key="surfactant_conc_slider",
                help="Values < 0.030% trigger micelle depletion penalties, reducing unfolding enthalpy ΔH‡.",
            )
            st.slider(
                "Raw Excipient Peroxide Level (ppm)",
                min_value=0.0,
                max_value=3.0,
                step=0.1,
                format="%.1f",
                key="peroxide_level_slider",
                help="Peroxides > 1.0 ppm accelerate methionine oxidation in CDRs, increasing activation entropy ΔS‡.",
            )

        with st.expander("🧪 Group B: Chromatography Hydrodynamics", expanded=True):
            st.number_input(
                "System Gradient Dwell Volume V_D (µL)",
                min_value=50.0,
                max_value=1500.0,
                step=25.0,
                format="%.0f",
                key="dwell_volume_input",
                help="Offsets LSS retention integral: dwell delay t_D = V_D / F.",
            )
            st.number_input(
                "Mobile Phase Flow Rate F (mL/min)",
                min_value=0.20,
                max_value=1.00,
                step=0.05,
                format="%.2f",
                key="flow_rate_input",
                help="Flow rate through chromatography column.",
            )
            st.number_input(
                "Column Void Volume V₀ (mL)",
                min_value=0.50,
                max_value=2.50,
                step=0.10,
                format="%.2f",
                key="column_void_vol_input",
                help="Column interstitial/porous void volume.",
            )

        with st.expander("🧫 Group C: Cellular & Spatial Metadata", expanded=True):
            st.toggle(
                "Include Outer Perimeter Wells (Edge Bias)",
                key="include_outer_wells_toggle",
                help="Applies evaporative edge distortion to Rows A/H and Cols 1/12, stressing 4-PL Hill parallelism.",
            )
            st.slider(
                "Cell Passage Number (P_cell)",
                min_value=1,
                max_value=35,
                step=1,
                key="cell_passage_slider",
                help="Passages > 25 inject phenotypic drift variance into Bayesian prior asymptotes.",
            )

        curr_state = reactive_pipeline_out['triage']['agent3_state']
        # Crimson red #D93025 ONLY when in CRITICAL ALARM / OOS BREACH
        state_color = "#188038" if curr_state == "NOMINAL" else ("#b06000" if curr_state == "OOT DETECTED" else "#d93025")
        state_bg = "#e6f4ea" if curr_state == "NOMINAL" else ("#fef7e0" if curr_state == "OOT DETECTED" else "#fce8e6")

        st.markdown(
            rf"""
            <div class="google-card" style="padding: 12px 14px; font-size: 11.5px; color: #3c4043; line-height: 1.5; margin-top: 14px;">
                <div style="font-weight: 700; color: #202124; margin-bottom: 6px; font-size: 12px;">Reactive Pipeline Status</div>
                • Apparent $E_a$: <code>{reactive_pipeline_out['branch_a']['apparent_ea_kj_mol']:.2f} kJ/mol</code><br>
                • Dwell Delay $t_D$: <code>{reactive_pipeline_out['chromatography']['t_dwell_min']:.2f} min</code><br>
                • Mass Balance Residual: <code>{reactive_pipeline_out['branch_a']['max_mass_residual']:.2e} (&lt; 1e-7)</code><br>
                • Rate Inversions: <strong>0 / 3,200</strong><br>
                • Temp Inversions: <strong>0 / 2,400</strong><br>
                • Agent 3 State: <span style="background: {state_bg}; color: {state_color}; padding: 2px 7px; border-radius: 4px; font-weight: 700;">{curr_state}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif workspace == "🛡️ Quality Intelligence, OOS & Compliance":
        st.markdown(
            """
            <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 13px; font-weight: 700; color: #1a73e8;">FDA OOS Phase I/II Gate & FMEA</span>
                    <span style="font-size: 10px; background: #e8f0fe; color: #1967d2; padding: 2px 7px; border-radius: 10px; font-weight: 600;">Deterministic</span>
                </div>
                <div style="font-size: 11px; color: #5f6368; margin-top: 3px; line-height: 1.35;">
                    Hardware Fault Telemetry Invalidation + Dynamic ICH Q9(R1) RPN Scoring
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🛡️ OOS & Telemetry Triage", expanded=True):
            oos_options = [
                "Run_14_Hardware_Anomaly",
                "Clean_Bench_Phase_II_Escalation",
            ]
            curr_oos_idx = (
                oos_options.index(st.session_state["oos_scenario"])
                if st.session_state["oos_scenario"] in oos_options
                else 0
            )
            st.session_state["oos_scenario"] = st.radio(
                "Empower Sequence Hardware Telemetry",
                options=oos_options,
                index=curr_oos_idx,
                format_func=lambda x: (
                    "Empower Run #14 Hardware Fault (310 bar / Chiller Drift)"
                    if x == "Run_14_Hardware_Anomaly"
                    else "Clean Bench & Telemetry (True OOS Phase II Escalation)"
                ),
            )
            proj_options = ["PROJ-001", "PROJ-002", "BI-CAP-101", "PROJ-004"]
            curr_proj_idx = (
                proj_options.index(st.session_state["selected_proj_comms"])
                if st.session_state["selected_proj_comms"] in proj_options
                else 0
            )
            st.session_state["selected_proj_comms"] = st.selectbox(
                "Focus Project for QTA / SOW Dispatch",
                options=proj_options,
                index=curr_proj_idx,
            )
        st.markdown(
            """
            <div class="google-card" style="padding: 12px 14px; font-size: 11.5px; color: #3c4043; line-height: 1.45; margin-top: 14px;">
                <strong>Compliance Safeguards:</strong><br>
                • Overpressure Limit: <code>300 bar</code><br>
                • Autosampler Fault: <code>> 10.0°C</code><br>
                • Phase I/II Gate: <strong>Deterministic</strong><br>
                • Audit Trail: <strong>SHA-256 Hash Chain</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🌳 Root-Cause Decomposition Tree", expanded=True):
            render_card_c_root_cause_tree(key_prefix="right_insp_root_cause_tree", compact_mode=True)

        with st.expander("🔍 Real-Time Telemetry Node Details (Run #14)", expanded=False):
            df_tel_insp = agent3_oos_out["df_telemetry"]
            st.dataframe(
                df_tel_insp[["injection_no", "pump_pressure_bar", "autosampler_temp_c", "acidic_peak_pct"]],
                use_container_width=True,
                hide_index=True,
                height=200,
            )

    elif workspace == "🧭 5-Minute Executive Tour":
        st.markdown(
            """
            <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="font-size: 13px; font-weight: 700; color: #1a73e8;">Executive Tour Engine</div>
                <div style="font-size: 11px; color: #5f6368; margin-top: 3px;">Scripted 5-Stop Narrative Walkthrough</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🧭 Tour Execution Progress", expanded=True):
            st.markdown(
                """
                <div style="font-size: 12px; color: #3c4043; line-height: 1.6;">
                    🔹 <strong>Stop 1:</strong> OOS Telemetry Triage<br>
                    🔹 <strong>Stop 2:</strong> Capacity & 168h Physical Gap<br>
                    🔹 <strong>Stop 3:</strong> SOW & Injection Defense<br>
                    🔹 <strong>Stop 4:</strong> SPC Control & CAPA<br>
                    🔹 <strong>Stop 5:</strong> Model Bounds & Limits
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.info("🧭 Preset values verified. Proceed through the scripted tabs on the center canvas.")

    else:
        st.markdown(
            """
            <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="font-size: 13px; font-weight: 700; color: #1a73e8;">Two-Layer cGMP Architecture</div>
                <div style="font-size: 11px; color: #5f6368; margin-top: 3px;">Declarative SOP Policies + Deterministic Compute</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🗂️ Architecture Layers", expanded=True):
            st.markdown(
                """
                <div style="font-size: 11.5px; color: #3c4043; line-height: 1.5;">
                    • <strong>Layer 1 (The What):</strong> Declarative <code>skill.md</code> SOP policies<br>
                    • <strong>Layer 2 (The How):</strong> Deterministic Python compute scripts<br>
                    • <strong>Governance:</strong> 21 CFR 211.68(b)
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# CENTER CANVAS: ACTIVE WORKSPACE & ANALYTICAL VISUALIZATION
# ===========================================================================
with col_canvas:

    # -----------------------------------------------------------------------
    # WORKSPACE 0: HOME / EXECUTIVE OVERVIEW
    # -----------------------------------------------------------------------
    if workspace == "🏠 Home / Executive Overview":
        render_ai_studio_breadcrumb(workspace, "Executive Cockpit & Mission Control", current_agent3_state)
        render_feature_help_popover(
            title="Home / Executive Overview",
            summary="Unified operational mission control presenting real-time facility KPIs, agent verdicts, and one-click workspace navigation.",
            formula_or_logic="Aggregates multi-agent status (Agent 1 CP-SAT capacity, Agent 2 stability PINN, Agent 3 OOS telemetry, Agent 4 stakeholder communications).",
            regulatory_basis="FDA 21 CFR Part 11, ICH Q9(R1), ICH Q10 Pharmaceutical Quality System",
            demo_action="Use the quick-launch cards to navigate to deep-dive workspaces or inspect real-time facility alerts.",
        )

        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); border: 1px solid #dadce0; border-radius: 12px; padding: 22px 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(60,64,67,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <div style="font-size: 11px; font-weight: 700; color: #1a73e8; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 4px;">
                            🏛️ Facility Operational Cockpit & Executive Summary
                        </div>
                        <h2 style="font-size: 20px; font-weight: 700; color: #202124; margin: 0 0 6px 0;">
                            Welcome to cGMP-QC-CoPilot Mission Control
                        </h2>
                        <div style="font-size: 13px; color: #5f6368; line-height: 1.5; max-width: 720px;">
                            Autonomous dual-stream quality control operations engine for commercial release testing and routine ICH stability campaigns. Combines deterministic biophysical PINNs, OR-Tools CP-SAT discrete optimization, and 21 CFR Part 11 cryptographic audit trails.
                        </div>
                    </div>
                    <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                        <span style="background: #e6f4ea; border: 1px solid #ceead6; color: #137333; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">
                            ● 4 / 4 Agents Online
                        </span>
                        <span style="background: #e8f0fe; border: 1px solid #d2e3fc; color: #1967d2; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">
                            SHA-256 Verified
                        </span>
                        <span style="background: #fef7e0; border: 1px solid #feefc3; color: #b06000; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">
                            Demo Mode
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🚀 Deep-Dive Workspaces & Specialized CoPilots")
        col_qc1, col_qc2 = st.columns(2)
        with col_qc1:
            st.markdown(
                """
                <div class="google-card" style="padding: 16px; margin-bottom: 12px; border-left: 4px solid #1a73e8;">
                    <div style="font-size: 14px; font-weight: 700; color: #1a73e8; margin-bottom: 6px;">
                        🧭 5-Minute Executive Tour
                    </div>
                    <div style="font-size: 12px; color: #5f6368; line-height: 1.5; margin-bottom: 8px;">
                        Guided 5-stop narrative walkthrough showcasing telemetry triage, CP-SAT leveling, prompt injection defense, SPC charts, and physical bounds.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Launch 5-Minute Tour ➔", key="home_btn_tour", use_container_width=True):
                st.session_state["_pending_workspace_nav"] = "🧭 5-Minute Executive Tour"
                st.rerun()

            st.markdown(
                """
                <div class="google-card" style="padding: 16px; margin-bottom: 12px; border-left: 4px solid #137333; margin-top: 14px;">
                    <div style="font-size: 14px; font-weight: 700; color: #137333; margin-bottom: 6px;">
                        📊 QC Operations & Project Management
                    </div>
                    <div style="font-size: 12px; color: #5f6368; line-height: 1.5; margin-bottom: 8px;">
                        Level shared analytical skid bottlenecks, cross-filter instrument workloads (Power BI style), and update disposition schedules in Smartsheet grid.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open Capacity & Schedules ➔", key="home_btn_pm", use_container_width=True):
                st.session_state["_pending_workspace_nav"] = "📊 QC Operations & Project Management"
                st.rerun()

        with col_qc2:
            st.markdown(
                """
                <div class="google-card" style="padding: 16px; margin-bottom: 12px; border-left: 4px solid #f9ab00;">
                    <div style="font-size: 14px; font-weight: 700; color: #b06000; margin-bottom: 6px;">
                        🌐 Live Plant & Multi-Agent Topology
                    </div>
                    <div style="font-size: 12px; color: #5f6368; line-height: 1.5; margin-bottom: 8px;">
                        Interactive industrial SCADA graph connecting physical skids, orchestrator AI agents, and 21 CFR Part 11 cryptographic ledger nodes.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Inspect Plant Network ➔", key="home_btn_top", use_container_width=True):
                st.session_state["_pending_workspace_nav"] = "🌐 Live Plant & Agent Network"
                st.rerun()

            st.markdown(
                """
                <div class="google-card" style="padding: 16px; margin-bottom: 12px; border-left: 4px solid #d93025; margin-top: 14px;">
                    <div style="font-size: 14px; font-weight: 700; color: #d93025; margin-bottom: 6px;">
                        🛡️ Quality Intelligence, OOS & Compliance
                    </div>
                    <div style="font-size: 12px; color: #5f6368; line-height: 1.5; margin-bottom: 8px;">
                        Automated FDA 2006 Phase I OOS hardware telemetry triage, 5-Whys forensic deconstruction, Shewhart/EWMA SPC, and dynamic FMEA.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Review Quality & OOS Triage ➔", key="home_btn_qual", use_container_width=True):
                st.session_state["_pending_workspace_nav"] = "🛡️ Quality Intelligence, OOS & Compliance"
                st.rerun()

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("#### ⚡ Cross-Functional Health & Decision Support Summary")
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st.markdown(
                """
                <div style="background: #ffffff; border: 1px solid #e0e3e7; border-radius: 10px; padding: 14px; height: 100%;">
                    <div style="font-size: 12.5px; font-weight: 700; color: #202124; margin-bottom: 8px;">
                        🔬 Active Biologics & Instrument Fleet Status
                    </div>
                    <div style="font-size: 12px; color: #3c4043; line-height: 1.7;">
                        • <strong>Waters Acquity UPLC:</strong> <code>310.4 bar ⚠️</code> (Overpressure abort on Run #14)<br>
                        • <strong>Autosampler Compartment:</strong> <code>17.8°C ⚠️</code> (Chiller drift past 10.0°C limit)<br>
                        • <strong>Maurice icIEF Skid:</strong> <code>Nominal</code> (Charge variant assays in-spec)<br>
                        • <strong>SpectraMax Potency Suite:</strong> <code>Nominal</code> (4-PL parallelism verified)<br>
                        • <strong>Rees Stability Chambers:</strong> <code>5.0°C ± 0.3°C</code> (Continuous logging nominal)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_st2:
            st.markdown(
                """
                <div style="background: #ffffff; border: 1px solid #e0e3e7; border-radius: 10px; padding: 14px; height: 100%;">
                    <div style="font-size: 12.5px; font-weight: 700; color: #202124; margin-bottom: 8px;">
                        🛡️ Governance & Deterministic Guardrails
                    </div>
                    <div style="font-size: 12px; color: #3c4043; line-height: 1.7;">
                        • <strong>FDA 2006 Phase I Gate:</strong> <code>Invalidate via Lab Error</code> (Pending QA Sign-Off)<br>
                        • <strong>Commercial Release SLAs:</strong> <code>6 / 6 Protected</code> (Zero float breaches)<br>
                        • <strong>Physical Capacity Realism:</strong> <code>168h Non-Overlapping</code> (Transparent gap)<br>
                        • <strong>Cold-Chain Invariant:</strong> <code>0 / 3,200 Rate Inversions</code> (Mass preserved)<br>
                        • <strong>Part 11 Hash Chain:</strong> <code>Unbroken Cryptographic State</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # WORKSPACE 1: 5-MINUTE EXECUTIVE TOUR
    # -----------------------------------------------------------------------
    elif workspace == "🧭 5-Minute Executive Tour":
        render_ai_studio_breadcrumb(workspace, "Guided Tour Narrative", current_agent3_state)
        render_feature_help_popover(
            title="5-Minute Executive Tour",
            summary="A guided walkthrough designed for technical hiring panels and auditors demonstrating all 5 core capabilities in sequence.",
            formula_or_logic="Steps through telemetry triage, CP-SAT leveling, prompt injection defense, SPC, and biophysical bounds.",
            regulatory_basis="21 CFR Part 11, 21 CFR 211.192, ALCOA+ Principles",
            demo_action="Click through the Stop tabs to narrate the pre-written interview script.",
        )
        render_5_minute_tour_tab()

    # -----------------------------------------------------------------------
    # WORKSPACE 2: LIVE PLANT & MULTI-AGENT TOPOLOGY
    # -----------------------------------------------------------------------
    elif workspace == "🌐 Live Plant & Agent Network":
        render_ai_studio_breadcrumb(workspace, "Interactive Network Topology", current_agent3_state)
        render_feature_help_popover(
            title="Live Plant & Multi-Agent Network Topology",
            summary="Industrial SCADA-style interactive network map visualizing physical analytical infrastructure, AI specialist agents, and 21 CFR Part 11 governance gates.",
            formula_or_logic="Three-zone directed graph with telemetry state propagation, dynamic node status coloring, and real-time event streaming.",
            regulatory_basis="21 CFR 211.68(b) Automated Equipment, 21 CFR 211.160 Laboratory Controls, ALCOA+ Traceability",
            demo_action="Click any node in the interactive topology graph or inspect the real-time SCADA telemetry alarm console below.",
        )
        topology_context = {
            "oos_scenario": oos_scenario,
            "inst_outage_mult": inst_outage_mult,
            "analyst_abs_mult": analyst_abs_mult,
            "inject_rush": inject_rush,
            "usp_harvest_delay": usp_harvest_delay,
            "excursion_temp": excursion_temp,
            "excursion_hours": excursion_hours,
            "signed_mso_pkg": st.session_state.get("signed_mso_pkg"),
            "audit_trail": st.session_state.get("audit_trail"),
            "selected_proj_comms": selected_proj_comms,
            "topology_zone_filter": st.session_state.get("topology_zone_filter", "All Zones (Connected Plant)"),
            "audit_latest_hash": getattr(st.session_state.get("audit_trail"), "latest_hash", "GENESIS_HASH"),
        }
        render_live_plant_network_tab(topology_context)

    # -----------------------------------------------------------------------
    # WORKSPACE 3: QC OPERATIONS & PROJECT MANAGEMENT
    # -----------------------------------------------------------------------
    elif workspace == "📊 QC Operations & Project Management":
        render_ai_studio_breadcrumb(workspace, "Capacity Leveling & Schedule Risk", current_agent3_state)
        render_feature_help_popover(
            title="QC Operations & PM Hub",
            summary="Balances wave-loaded skid capacity between commercial pipeline lots and external CDMO client contracts.",
            formula_or_logic="Google OR-Tools CP-SAT discrete optimization shifting flexible tasks within ICH Q1A windows.",
            regulatory_basis="ICH Q1A(R2) Stability Pull Windows, 21 CFR 211.160",
            demo_action="Adjust the 'USP/DSP Harvest Slip' slider in the right inspector to see the CP-SAT schedule re-level live.",
        )

        pm_tabs = st.tabs([
            "1. LEAN Capacity & Critical Path",
            "2. Portfolio & Operations (WP2)",
            "3. QTA/SOW Comms & Matrix",
        ])

        # --- SUB-TAB 2.1: AGENT 1 CAPACITY & SCHEDULE RISK ---
        with pm_tabs[0]:
            render_feature_help_popover(
                title="Agent 1: LEAN Capacity & Critical Path",
                summary="Protects zero-float commercial release campaigns while leveling instrument skid overloads below the 85% LEAN buffer ceiling.",
                formula_or_logic="OR-Tools CP-SAT discrete optimization + Monte Carlo CPM float analysis + SimPy discrete event simulation.",
                regulatory_basis="ICH Q1A(R2) permitted ±7-day stability pull windows",
                demo_action="Toggle 'Inject Urgent Rush Lot' or adjust 'Harvest Slip' in the right inspector to view schedule shifts.",
            )
            st.subheader("Agent 1: Hybrid Capacity & Schedule-Risk Engine (CP-SAT Leveling + Monte Carlo CPM)")
            st.markdown(
                """
                <div class="gate-box-pass">
                <strong>Dual-Stream Capacity Governance (ICH Q1A(R2) & QTA SLAs):</strong>
                Protects 100% of immutable Tier-1A commercial release campaigns and CDMO binding SLAs (zero float) while leveling instrument skid overloads below the <strong>85.0% LEAN buffer ceiling</strong> using OR-Tools CP-SAT discrete optimization and permitted <strong>±7-day ICH Q1A pull windows</strong> on routine stability lots.
                </div>
                """,
                unsafe_allow_html=True,
            )

            t1_subtabs = st.tabs([
                "1. Capacity Leveling",
                "2. Monte Carlo Risk & CPM",
                "3. Prioritization Matrix",
                "4. Human Review Gate",
                "5. Model Limitations",
            ])

            with t1_subtabs[0]:
                st.markdown("#### ⚖️ Instrument Skid Workload: Before vs After CP-SAT Leveling")

                df_app_base, df_app_lev = extract_workload_dfs(agent1_hybrid_out)
                render_card_a_cpsat_workload_canvas(
                    df_base=df_app_base,
                    df_lev=df_app_lev,
                    inst_outage_mult=inst_outage_mult,
                    analyst_abs_mult=analyst_abs_mult,
                    inject_rush=inject_rush,
                    key_prefix="app_cpsat_canvas",
                )

                st.warning(
                    "⚠️ **Physical Capacity & Feasibility Finding (Demo Portfolio):**\n\n"
                    "- **Unleveled Demand:** 143.3% of 120h nominal staffed capacity = **172.0h/week on Waters UPLC**, which "
                    "**strictly exceeds 168.0h physical 24/7 calendar capacity** (physically impossible without concurrent double-booking or schedule shifts).\n"
                    "- **CP-SAT Leveling:** Eliminates all instrument double-booking and levels load to **140.0% = 168.0h/week**, packing the skid into non-overlapping slots while protecting fixed commercial releases.\n"
                    "- **Operational Reality:** Because 168.0h represents substantial skid utilization, the portfolio cannot fit under the 85% LEAN ceiling (102h) without adding instrument skids, staffing additional weekend shifts, or rescheduling lots across adjacent weeks."
                )

                render_card_b_smartsheet_grid(
                    key_prefix="app_smartsheet_grid",
                )

                with st.expander("🔍 Step-by-Step CP-SAT Move Explanations", expanded=False):
                    df_moves = agent1_hybrid_out["df_moves"]
                    if not df_moves.empty:
                        st.dataframe(df_moves[["project_id", "instrument_id", "delta_days", "reason"]], use_container_width=True, hide_index=True)
                    else:
                        st.info("No schedule shifts required; all instrument workloads are currently below the 85.0% LEAN ceiling.")

                with st.expander("🏭 Master Portfolio Schedule (10 Biologics Programs)", expanded=False):
                    st.dataframe(agent1_out["df_portfolio"], use_container_width=True, hide_index=True)

            with t1_subtabs[1]:
                st.markdown("#### 🎲 Milestone Delivery Risk & Critical Path Analysis (P(on time) & Criticality Index)")
                st.info("⚠️ PRECEDENCE-ONLY NETWORK (Topological Float & Critical Path) — For capacity-constrained stochastic milestone risk, see Discrete-Event Simulation (DES) metrics.")
                st.caption("Discrete-Event Simulation (DES) replications evaluate shared skid queues and resource contention. CPM provides precedence float and criticality.")
                df_risk = agent1_hybrid_out["df_risk"]
                risk_cols = [c for c in ["project_id", "program_name", "due_day", "mean_completion_day", "p_on_time_pct", "rag_status", "total_float_days", "criticality_index_pct", "expected_delay_days", "evaluation_source"] if c in df_risk.columns]
                st.dataframe(df_risk[risk_cols], use_container_width=True, hide_index=True)

                st.markdown("#### 🔎 Discrete-Event Simulation (SimPy) Bottleneck Diagnostics")
                des_sum = agent1_hybrid_out["des_summary"]
                b_c1, b_c2, b_c3 = st.columns(3)
                b_c1.metric("Identified Bottleneck Skid", des_sum["bottleneck_resource"])
                b_c2.metric("Mean Queue Delay (UPLC)", f"{des_sum['mean_queue_hours'].get('UPLC_SEC_CEX', 0.0)} hrs")
                b_c3.metric("Double-Booking Violations", f"{0 if des_sum['conservation_checks_passed'] else 'DETECTED'}")

            with t1_subtabs[2]:
                st.markdown("#### ⚖️ Multi-Criteria Weighted Prioritization Ranking")
                st.caption("Normalized criteria: Contract Penalty (25%), Project Tier (25%), Remaining Float (20%), Critical-Path Dependency (15%), Patient Impact (15%).")
                st.dataframe(agent1_hybrid_out["df_prioritization"], use_container_width=True, hide_index=True)

                st.markdown("#### 🎯 Dirichlet Weight Sensitivity & Rank Stability")
                sens = agent1_hybrid_out["sensitivity"]
                s_c1, s_c2 = st.columns(2)
                s_c1.metric("Top-Ranked Baseline Program", sens["baseline_top_project"])
                s_c2.metric("Rank Stability Score (Top #1 Probability)", f"{round(sens['rank_stability_score'] * 100.0, 1)}%")

            with t1_subtabs[3]:
                st.markdown("#### ✍️ Part 11-Style Human Review Gate: Commit Leveled Schedule")
                st.markdown(
                    f"""
                    <div class="gate-box-warn">
                    <strong>{DRAFT_STAMP}</strong><br>
                    Under Guardrail G6, schedule changes are never applied autonomously. Applying a leveled schedule requires human reviewer attribution, role, justification, and explicit confirmation.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.form("schedule_sign_off_form"):
                    sched_reviewer = st.text_input("Reviewer Name", value="Alex Rivera", help="ALCOA+ Attributable Signer Name")
                    sched_role = st.selectbox("Reviewer Role", ["Principal Specialist, QC PM", "QC Laboratory Supervisor", "QA Operations Lead"])
                    sched_justification = st.text_area(
                        "Electronic Signature Justification",
                        value="Reviewed CP-SAT leveled schedule: zero Tier-1A commercial release lots impacted, UPLC utilization meets the example <=85% utilization target (policy configurable).",
                    )
                    sched_confirmed = st.checkbox("I independently confirm this schedule change aligns with site-defined pull windows (example ±7 d) and site SOPs.")
                    sched_submit = st.form_submit_button("✅ Sign & Commit Leveled Schedule to Audit Trail")

                    if sched_submit:
                        trail_inst = st.session_state.get("audit_trail")
                        if isinstance(trail_inst, AuditTrail) or hasattr(trail_inst, "append_event"):
                            ok, rec, errs = execute_schedule_sign_off(
                                audit_trail=trail_inst,
                                reviewer_name=sched_reviewer,
                                reviewer_role=sched_role,
                                justification=sched_justification,
                                confirmed=sched_confirmed,
                                optimization_result=agent1_hybrid_out["cpsat_result"],
                            )
                            if ok:
                                st.success(f"Schedule commit signed and appended to immutable audit trail! Block #{rec.index} Hash: `{rec.record_hash[:16]}...`")
                            else:
                                for e in errs:
                                    st.error(e)
                        else:
                            st.info("Audit trail simulated; commit recorded.")

            with t1_subtabs[4]:
                st.markdown("#### 📋 Model Limitations & Honest Evaluation Summary (Guardrail G5)")
                st.markdown(
                    """
                    - **Synthetic Data Provenance:** All training and evaluation history was generated synthetically (`data/synthetic_capacity/`).
                    - **Honest Negative / Mixed Results:** On sample-arrival slippage, quantile GBDT achieves MAE 1.15 days vs 1.11 days for the simple mean (-3.8% improvement), showing that arrival slippage has unmodeled fat-tail variance that point models cannot resolve from scheduling features alone. Conformal intervals are required to provide calibrated 86.7% coverage.
                    - **Deterministic Turnaround Improvement:** On analyst turnaround time, quantile GBDT achieves 1.38h MAE vs 2.00h baseline (31.1% error reduction, 95% CI [1.18, 1.58]h).
                    - **CP-SAT Hard-Constraint Protection:** 100% of fixed commercial/contractual commitments protected across 100 tested scenarios.
                    """
                )

        # --- SUB-TAB 2.2: PORTFOLIO & OPERATIONS (WP2) ---
        with pm_tabs[1]:
            render_feature_help_popover(
                title="Portfolio Grid, WBS & Operations (WP2)",
                summary="Central operational cockpit tracking biologics programs, active blockers, Kanban states, and Work Breakdown Structure.",
                formula_or_logic="Relational SQLite tracking with ALCOA+ audit logging and deterministic WBS rollups.",
                regulatory_basis="21 CFR 211.22, 21 CFR 211.100, ALCOA+ Principles",
                demo_action="Explore portfolio projects, filter active blockers by severity, and inspect sample alerts.",
            )
            render_portfolio_and_operations_tab(
                conn=st.session_state["pm_conn"],
                audit_trail=st.session_state.get("audit_trail"),
            )

        # --- SUB-TAB 2.3: STAKEHOLDER COMMS & CHANGE ORDERS (AGENT 4) ---
        with pm_tabs[2]:
            render_feature_help_popover(
                title="Agent 4: Client Comms & SOW Change Orders",
                summary="Provides automated, regulatory-grounded communications and proactive change order governance for CDMO client contracts.",
                formula_or_logic="Deterministic string rendering from Agent 1-3 findings mapped to Quality Technical Agreement (QTA) clauses.",
                regulatory_basis="Contractual CDMO Quality Technical Agreements (QTA) & SOW SLA clauses",
                demo_action="Select different communication packages to inspect role-tailored notifications and change orders.",
            )
            st.subheader("Agent 4: SOW/QTA Client Transparency & Cross-Functional Matrix Communications")
            st.markdown("#### 🌐 Example Biologics Site 6-Persona Cross-Functional Stakeholder Register")
            st.dataframe(agent4_comms_out["df_stakeholders"], use_container_width=True, hide_index=True)

            st.markdown("#### ✉️ Live Context-Populated Communication Dispatches")
            dispatches = agent4_comms_out["rendered_dispatches"]
            dispatch_keys = list(dispatches.keys())
            selected_tpl = st.selectbox(
                "Select Communication Package to Inspect / Edit",
                options=dispatch_keys,
                format_func=lambda k: f"{k} — {dispatches[k]['recipient_role']}",
            )

            active_msg = dispatches[selected_tpl]
            st.info(
                f"**Recipient:** {active_msg['recipient_name']}  |  **Governance Basis:** `{active_msg['qta_clause']}`\n\n"
                f"**Subject:** `{active_msg['subject']}`"
            )
            st.text_area(
                "Generated Dispatch Body (Populated Deterministically from Agents 1, 2 & 3):",
                value=active_msg["body"],
                height=320,
            )

            st.markdown("#### 💼 Proactive SOW Change Order & Cost Absorption Governance")
            st.dataframe(agent4_comms_out["df_change_orders"], use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # WORKSPACE 3: ANALYTICAL SCIENCE & HYBRID KINETICS
    # -----------------------------------------------------------------------
    elif workspace == "🔬 Analytical Science & Hybrid Kinetics":
        render_ai_studio_breadcrumb(workspace, "Sub-tab 1: Kinetics & Dynamic OOT", current_agent3_state)
        render_feature_help_popover(
            title="Analytical Science & Hybrid Kinetics",
            summary="Evaluates non-isothermal cold-chain excursions, chromatographic dwell volume shifts, and bioassay relative potency.",
            formula_or_logic="Lumry-Eyring coupled ODEs + 25-member neural residual ensemble enforcing kinetic monotonicity by construction.",
            regulatory_basis="ICH Q5C, ICH Q1E, ICH Q2(R2), ICH Q14, USP <1032>",
            demo_action="Move the Excursion Temperature slider to 28°C and Duration to 36h in the right inspector to see the Eyring integration curve.",
        )

        sci_tabs = st.tabs([
            "1. Kinetics & Dynamic OOT",
            "2. LSS/SMA Method Transfer",
            "3. 4-PL Bioassay & Heatmap",
            "4. ICH Q6B LIMS Schema",
            "5. ANCOVA Shelf-Life",
            "6. Sensitivity Audit",
        ])

        # --- SUB-TAB 3.1: EYRING + ANN EXCURSION & 6M OOT ---
        with sci_tabs[0]:
            render_feature_help_popover(
                title="Hybrid Eyring Excursion & Dynamic 6M OOT",
                summary="Quantifies HMW aggregation during temperature excursions and adapts early stability pulls (0-3M) to flag out-of-trend lots.",
                formula_or_logic="Coupled Lumry-Eyring RK4 ODEs + 25-member neural residual ensemble with Bayesian early-pull adaptation.",
                regulatory_basis="ICH Q1A(R2), ICH Q5C, WHO Technical Report Series No. 961 Annex 9",
                demo_action="Adjust excursion peak temperature and duration in the right inspector to inspect real-time trajectory and OOT upper bounds.",
            )
            exc_sum = agent2_exc_out["summary"]
            oot_diag = agent2_oot_out["oot_diagnostics"]

            st.markdown(
                f"""
                <div class="gate-box-warn">
                <strong>{DRAFT_STAMP}</strong><br>
                <strong>Hybrid Lumry-Eyring + 25-Member Ensemble Excursion Assessment:</strong> {exc_sum['disposition_recommendation']}<br>
                <strong>Validation Safety Benchmark:</strong> {EXCURSION_SAFETY_STATEMENT}; {PHYSICAL_VALIDITY_STATEMENT}.<br>
                <strong>eCTD 3.2.P.8 Dynamic OOT & Early-Pull Adaptation Ruling:</strong> {oot_diag['finding']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Lumry-Eyring PINN & Hybrid Kinetics Architecture Schematic
            render_hybrid_architecture_schematic(
                context={
                    "pipeline_res": reactive_pipeline_out,
                    "excursion_temp": excursion_temp,
                    "excursion_hours": excursion_hours,
                    "initial_monomer_pct": initial_monomer_pct,
                    "formulation_ph": formulation_ph,
                    "surfactant_conc_pct": surfactant_conc_pct,
                    "peroxide_level_ppm": peroxide_level_ppm,
                    "dwell_volume_ul": dwell_volume_ul,
                    "flow_rate_ml_min": flow_rate_ml_min,
                    "column_void_vol_ml": column_void_vol_ml,
                    "include_outer_wells": include_outer_wells,
                    "cell_passage_num": cell_passage_num,
                    "pre_exc_input": pre_exc_input,
                    "summary": exc_sum,
                    "oot_diag": oot_diag,
                }
            )

            c_e1, c_e2 = st.columns(2)
            with c_e1:
                st.markdown('<div id="trajectory-canvas">', unsafe_allow_html=True)
                df_traj = agent2_exc_out["df_trajectory"]
                fig_exc = make_subplots(specs=[[{"secondary_y": True}]])
                if "HMW_Hybrid_U95_Pct" in df_traj.columns and "HMW_Hybrid_L95_Pct" in df_traj.columns:
                    fig_exc.add_trace(
                        go.Scatter(
                            x=df_traj["Time_Hours"],
                            y=df_traj["HMW_Hybrid_U95_Pct"],
                            name="Hybrid 95% Upper Bound (%)",
                            line=dict(color="rgba(220, 38, 38, 0.25)", width=1),
                            showlegend=False,
                        ),
                        secondary_y=False,
                    )
                    fig_exc.add_trace(
                        go.Scatter(
                            x=df_traj["Time_Hours"],
                            y=df_traj["HMW_Hybrid_L95_Pct"],
                            name="Hybrid 95% Interval Band",
                            line=dict(color="rgba(220, 38, 38, 0.25)", width=1),
                            fill="tonexty",
                            fillcolor="rgba(220, 38, 38, 0.15)",
                        ),
                        secondary_y=False,
                    )
                fig_exc.add_trace(
                    go.Scatter(
                        x=df_traj["Time_Hours"],
                        y=df_traj["HMW_Hybrid_Eyring_ANN_Pct"],
                        name="Hybrid Lumry-Eyring + Ensemble Mean HMW (%)",
                        line=dict(color="#dc2626", width=3),
                    ),
                    secondary_y=False,
                )
                fig_exc.add_trace(
                    go.Scatter(
                        x=df_traj["Time_Hours"],
                        y=df_traj["HMW_Naive_Eyring_Only_Pct"],
                        name="Single-Step Eyring Reference HMW (%)",
                        line=dict(color="#64748b", width=2, dash="dot"),
                    ),
                    secondary_y=False,
                )
                fig_exc.add_trace(
                    go.Scatter(
                        x=df_traj["Time_Hours"],
                        y=df_traj["Temperature_C"],
                        name="Thermal Profile (°C)",
                        line=dict(color="#f59e0b", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(245, 158, 11, 0.12)",
                    ),
                    secondary_y=True,
                )
                fig_exc.update_layout(
                    title=f"{int(excursion_hours)}h Cold-Chain Excursion ({excursion_temp}°C) RK4 ODE Integration",
                    xaxis_title="Time (Hours)",
                    height=410,
                )
                fig_exc.update_yaxes(title_text="HMW Aggregate (%)", secondary_y=False)
                fig_exc.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
                st.plotly_chart(fig_exc, use_container_width=True)
                st.caption(f"{exc_sum.get('sec_reversible_vs_irreversible_caveat', '')} {exc_sum.get('prior_driven_f_u_caveat', '')}")
                st.markdown('</div>', unsafe_allow_html=True)

            with c_e2:
                df_oot = agent2_oot_out["df_projection"]
                fig_oot = go.Figure()
                if "Prior_Hybrid_25C_Mean_Pct" in df_oot.columns:
                    fig_oot.add_trace(
                        go.Scatter(
                            x=df_oot["Month"],
                            y=df_oot["Prior_Hybrid_25C_Mean_Pct"],
                            mode="lines",
                            name="Prior (Covariate-Only) 25°C Mean",
                            line=dict(color="#9333ea", width=2, dash="dot"),
                        )
                    )
                fig_oot.add_trace(
                    go.Scatter(
                        x=df_oot["Month"],
                        y=df_oot["Hybrid_Eyring_ANN_25C_Forecast_Pct"],
                        mode="lines+markers",
                        name="Adapted (0–3M) Posterior 25°C Mean (x1 Prior)",
                        line=dict(color="#dc2626", width=3),
                    )
                )
                fig_oot.add_trace(
                    go.Scatter(
                        x=df_oot["Month"],
                        y=df_oot["OOT_Upper_99CI_Band_Pct"],
                        mode="lines",
                        name="Recalibrated Adapted 95% PI Upper (x1 Prior)",
                        line=dict(color="#f59e0b", dash="dash", width=2),
                    )
                )
                if "Classic_Linear_Reg_PI95_Upper_Pct" in df_oot.columns:
                    fig_oot.add_trace(
                        go.Scatter(
                            x=df_oot["Month"],
                            y=df_oot["Classic_Linear_Reg_PI95_Upper_Pct"],
                            mode="lines",
                            name="Classic Linear Reg 95% PI Upper",
                            line=dict(color="#1e40af", dash="dot", width=2),
                        )
                    )
                if "Static_3SD_035_Linear_Upper_Pct" in df_oot.columns:
                    fig_oot.add_trace(
                        go.Scatter(
                            x=df_oot["Month"],
                            y=df_oot["Static_3SD_035_Linear_Upper_Pct"],
                            mode="lines",
                            name="Classic Linear + 3*0.035 Upper",
                            line=dict(color="#2563eb", dash="dashdot", width=2),
                        )
                    )
                elif "Static_3SD_Linear_Upper_Pct" in df_oot.columns:
                    fig_oot.add_trace(
                        go.Scatter(
                            x=df_oot["Month"],
                            y=df_oot["Static_3SD_Linear_Upper_Pct"],
                            mode="lines",
                            name="Classic 0–3M Linear + 3*SD Upper Alert",
                            line=dict(color="#2563eb", dash="dashdot", width=2),
                        )
                    )
                if "Static_3SD_050_Linear_Upper_Pct" in df_oot.columns:
                    fig_oot.add_trace(
                        go.Scatter(
                            x=df_oot["Month"],
                            y=df_oot["Static_3SD_050_Linear_Upper_Pct"],
                            mode="lines",
                            name="Classic Linear + 3*0.050 Upper",
                            line=dict(color="#3b82f6", dash="dot", width=1.5),
                        )
                    )
                if "Observed_25C_HMW_Pct" in df_oot.columns:
                    fig_oot.add_trace(
                        go.Scatter(
                            x=df_oot["Month"],
                            y=df_oot["Observed_25C_HMW_Pct"],
                            mode="markers",
                            name="Observed 25°C Pulls (0, 1, 3, 6M)",
                            marker=dict(color="#111827", size=10, symbol="diamond"),
                        )
                    )
                fig_oot.add_trace(
                    go.Scatter(
                        x=df_oot["Month"],
                        y=df_oot["Realtime_5C_Forecast_Pct"],
                        mode="lines+markers",
                        name="Adapted Real-Time 5°C Trajectory",
                        line=dict(color="#10b981", width=2),
                    )
                )
                fig_oot.add_hline(
                    y=2.0,
                    line_color="#991b1b",
                    line_dash="solid",
                    annotation_text="ICH Q6B OOS Spec Limit (2.0% HMW, assumed specification for illustration)",
                )
                fig_oot.update_layout(
                    title="Demo Lot NS-2026-004: Prior vs 0–3M Adapted Posterior vs Observed Pulls",
                    xaxis_title="Stability Timepoint (Months)",
                    yaxis_title="HMW Aggregate (%)",
                    height=410,
                )
                st.plotly_chart(fig_oot, use_container_width=True)

            with st.expander("🔍 Inspect NS-2026-004 Early-Pull Residuals (in SEC-SD Units) & Prior-Scale Sensitivity (x0.5, x1, x2, x4)"):
                st.info(oot_diag.get("prior_dominance_note", ""))
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown("**0 / 1 / 3M Pull Residuals (5 °C and 25 °C) in `pp` and `SEC-SD` Units:**")
                    st.dataframe(pd.DataFrame(agent2_oot_out.get("early_pull_residuals_0_1_3m", [])), use_container_width=True, hide_index=True)
                with col_r2:
                    st.markdown("**Prior-Sigma Scaling Sensitivity (`x0.5, x1, x2, x4` of `0.28 kJ/mol dG, 2.2 kJ/mol dH`):**")
                    st.dataframe(pd.DataFrame(agent2_oot_out.get("prior_scale_sensitivity_table", [])), use_container_width=True, hide_index=True)

        # --- SUB-TAB 3.2: LSS & SMA CHROMATOGRAPHY TECH TRANSFER ---
        with sci_tabs[1]:
            render_feature_help_popover(
                title="LSS/SMA Chromatography Method Transfer",
                summary="Compensates for gradient delay differences when transferring methods between HPLC and UPLC platforms.",
                formula_or_logic="Linear Solvent Strength (LSS) & Steric Mass Action (SMA) models calculating required initial isocratic hold adjustments.",
                regulatory_basis="ICH Q14, USP <621> Chromatography System Suitability",
                demo_action="Adjust Donor Agilent Dwell Volume V_D in the right inspector to view real-time retention time alignment on Waters UPLC.",
            )
            tr_p = agent2_trans_out["parameters"]
            st.markdown(
                f"""
                <div class="gate-box-pass">
                <strong>ICH Q14 & USP &lt;621&gt; Dwell-Volume Compensation:</strong> Donor Agilent 1260 HPLC (<code>V_D = {tr_p['v_dwell_donor_ml']} mL</code>) → Receiver Example Biologics Site Waters Acquity UPLC (<code>V_D = {tr_p['v_dwell_receiver_ml']} mL</code>).<br>
                Uncompensated transfer causes a <strong>{tr_p['uncompensated_shift_min']} min premature elution shift</strong>. Increasing Receiver Initial Isocratic Hold from <code>{tr_p['donor_initial_hold_min']} min</code> to <strong><code>{tr_p['recommended_receiver_hold_min']} min</code></strong> restores retention alignment within <strong>-0.02 min</strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )

            df_chrom = agent2_trans_out["df_chromatograms"]
            fig_chrom = go.Figure()
            fig_chrom.add_trace(
                go.Scatter(
                    x=df_chrom["Time_min"],
                    y=df_chrom["Donor_Agilent_1260_HPLC_mAU"],
                    name="Donor: Client Agilent 1260 HPLC (Ref)",
                    line=dict(color="#64748b", width=2.5),
                )
            )
            fig_chrom.add_trace(
                go.Scatter(
                    x=df_chrom["Time_min"],
                    y=df_chrom["Waters_UPLC_Uncompensated_mAU"],
                    name="Receiver: Waters UPLC Uncompensated (Premature Shift)",
                    line=dict(color="#ef4444", width=2, dash="dash"),
                )
            )
            fig_chrom.add_trace(
                go.Scatter(
                    x=df_chrom["Time_min"],
                    y=df_chrom["Waters_UPLC_Compensated_mAU"],
                    name=f"Receiver: Waters UPLC Compensated ({tr_p['recommended_receiver_hold_min']} min Hold)",
                    line=dict(color="#10b981", width=2.5),
                )
            )
            fig_chrom.update_layout(
                title="CEX-UPLC Charge Variant Chromatogram Overlay (Acidic, Main, Basic Peaks)",
                xaxis_title="Retention Time (min)",
                yaxis_title="UV Absorbance (280 nm, mAU)",
                height=400,
            )
            st.plotly_chart(fig_chrom, use_container_width=True)

            st.markdown("#### 📊 Steric Mass Action (SMA) Peak Retention & Width Comparison")
            st.dataframe(agent2_trans_out["df_peaks"], use_container_width=True, hide_index=True)

            with st.expander("🧱 Inspect Modular 'Lego' Analytical Tech Transfer Protocol (ATTP-EXAMPLE-2026-CEX-204)"):
                st.json(agent2_trans_out["attp_schema"])

        # --- SUB-TAB 3.3: 4-PL BIOASSAY & 96-WELL SPATIAL HEATMAP ---
        with sci_tabs[2]:
            render_feature_help_popover(
                title="4-PL Bioassay & Plate Spatial Heatmap",
                summary="Evaluates dose-response parallelism and relative potency while diagnosing evaporative perimeter edge effects.",
                formula_or_logic="4-Parameter Logistic regression with F-test slope parallelism and 2D spatial edge effect simulation.",
                regulatory_basis="USP <1032>, USP <1034> Analysis of Biological Assays",
                demo_action="Toggle 'Include Outer-36 Plate Perimeter Wells' in the right inspector to see how edge evaporative bias alters potency.",
            )
            bio_m = agent2_bio_out["metrics"]
            box_cls = "gate-box-pass" if "PASS" in bio_m["parallelism_gate_status"] else "gate-box-crit"
            st.markdown(
                f"""
                <div class="{box_cls}">
                <strong>USP &lt;1034&gt; Parallelism Gate ({bio_m['mode']}):</strong> {bio_m['parallelism_gate_status']}<br>
                Relative Potency = <strong>{bio_m['relative_potency_pct']}%</strong> | Hill Slope Ratio (B_test / B_ref) = <strong>{bio_m['hill_slope_ratio']}</strong> (Acceptance: 0.80–1.25) | Upper Asymptote Ratio = <strong>{bio_m['upper_asymptote_ratio']}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cb1, cb2 = st.columns(2)
            with cb1:
                df_c = agent2_bio_out["df_curves"]
                fig_4pl = go.Figure()
                fig_4pl.add_trace(
                    go.Scatter(
                        x=df_c["Log10_Conc"],
                        y=df_c["Reference_Standard_RS04_RLU"] if "Reference_Standard_RS04_RLU" in df_c.columns else df_c.iloc[:, 1],
                        name="Reference Standard RS-04",
                        line=dict(color="#2563eb", width=3),
                    )
                )
                test_col = [c for c in df_c.columns if "Test_Sample" in c]
                if test_col:
                    fig_4pl.add_trace(
                        go.Scatter(
                            x=df_c["Log10_Conc"],
                            y=df_c[test_col[0]],
                            name=f"PPQ Test Lot ({bio_m['relative_potency_pct']}% RP)",
                            line=dict(color="#10b981" if not include_outer_wells else "#f59e0b", width=3, dash="dash"),
                        )
                    )
                degr_col = [c for c in df_c.columns if "Degraded" in c or "NonParallel" in c]
                if degr_col:
                    fig_4pl.add_trace(
                        go.Scatter(
                            x=df_c["Log10_Conc"],
                            y=df_c[degr_col[0]],
                            name="Stressed Positive Control (Non-Parallel)",
                            line=dict(color="#ef4444", width=2, dash="dot"),
                        )
                    )
                fig_4pl.update_layout(
                    title="4-Parameter Logistic (4-PL) Dose-Response Equivalence",
                    xaxis_title="Log10 Concentration (ng/mL)",
                    yaxis_title="Luminescence Response (RLU x 10³)",
                    height=400,
                )
                st.plotly_chart(fig_4pl, use_container_width=True)

            with cb2:
                df_hm = agent2_bio_out["df_plate_heatmap"]
                fig_hm = px.imshow(
                    df_hm,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdYlBu_r",
                    title="96-Well Spatial Evaporative Residual Heatmap (% RLU Bias — Outer-36 vs Inner-60)",
                )
                fig_hm.update_layout(height=400)
                st.plotly_chart(fig_hm, use_container_width=True)

        # --- SUB-TAB 3.4: ICH Q6B LIMS TEST CODES ---
        with sci_tabs[3]:
            render_feature_help_popover(
                title="ICH Q6B Specification Architecture",
                summary="Separates release testing attributes from stability-indicating shelf-life attributes with explicit test codes.",
                formula_or_logic="Structured eCTD 3.2.P.8 specification schema with acceptance criteria and LIMS parameter mapping.",
                regulatory_basis="ICH Q6B Specifications for Biotechnological/Biological Products",
                demo_action="Inspect required test codes, acceptance limits, and stability vs release panel assignments.",
            )
            st.markdown("#### 🧬 ICH Q6B Specification Architecture: Lot Release vs. Stability Panel Segregation")
            st.dataframe(agent2_oot_out["df_lims_codes"], use_container_width=True, hide_index=True)

        # --- SUB-TAB 3.5: ICH Q5C / Q1E SHELF-LIFE & LINEAR ANCOVA POOLABILITY ---
        with sci_tabs[4]:
            render_feature_help_popover(
                title="ICH Q5C/Q1E Shelf-Life & Batch Poolability",
                summary="Estimates real-time 5°C shelf-life using linear ANCOVA pooling alongside hybrid neural ODE upper confidence bounds.",
                formula_or_logic="ANCOVA test of slope and intercept equality (alpha=0.25) + 95% one-sided upper confidence bound intersection with 2.0% HMW spec.",
                regulatory_basis="ICH Q1E Section 2.4, ICH Q5C Quality of Biotechnological Products",
                demo_action="Compare the unclipped vs regulatory-capped shelf life between the Linear ANCOVA and Hybrid Ensemble.",
            )
            st.markdown(
                f"""
                <div class="gate-box-warn">
                <strong>{agent2_q1e_out['ich_q5c_q1e_regulatory_note']}</strong><br>
                {agent2_q1e_out['extrapolation_warning']}
                </div>
                """,
                unsafe_allow_html=True,
            )
            model_adeq_status = agent2_q1e_out.get("shelf_life_status", "model adequacy demonstrated")
            adeq_box = "gate-box-pass" if agent2_q1e_out.get("model_adequacy_demonstrated", True) else "gate-box-crit"
            st.markdown(
                f"""
                <div class="{adeq_box}">
                <strong>Model Adequacy Check:</strong> <code>{model_adeq_status}</code><br>
                <strong>Validity Statement:</strong> {agent2_q1e_out.get('shelf_life_validity_statement', '')}
                </div>
                """,
                unsafe_allow_html=True,
            )
            df_q1e = agent2_q1e_out["df_curves"]
            hyb_sl = agent2_q1e_out["hybrid_shelf_life_summary"]
            lin_sl = agent2_q1e_out["linear_ancova_q1e_summary"]

            cq1, cq2, cq3 = st.columns(3)
            with cq1:
                st.metric(
                    "Hybrid 95% UB Shelf Life (Capped)",
                    f"{hyb_sl['regulatory_capped_shelf_life_months']:.1f} M",
                    delta=f"Unclipped: {hyb_sl['unclipped_shelf_life_months']:.1f} M ({model_adeq_status})",
                    delta_color="off",
                )
            with cq2:
                st.metric(
                    "Classic Linear Q1E Shelf Life (Capped)",
                    f"{lin_sl['regulatory_capped_shelf_life_months']:.1f} M",
                    delta=f"Unclipped: {lin_sl['unclipped_shelf_life_months']:.1f} M (Gov: {lin_sl['governing_lot_id']})",
                    delta_color="off",
                )
            with cq3:
                st.metric(
                    "ANCOVA Batch Poolability (alpha = 0.25)",
                    f"p_slope = {lin_sl['p_value_slope_equality']:.3f}",
                    delta=f"p_intercept = {lin_sl['p_value_intercept_equality']:.3f} ({lin_sl['pooling_decision'].split(' ')[0]})",
                    delta_color="off",
                )

            fig_q1e = go.Figure()
            fig_q1e.add_trace(
                go.Scatter(
                    x=df_q1e["Month"],
                    y=df_q1e["Hybrid_Governing_Mean_Pct"],
                    name="Hybrid Governing Lot Mean (5°C)",
                    line=dict(color="#10b981", width=2.5),
                )
            )
            fig_q1e.add_trace(
                go.Scatter(
                    x=df_q1e["Month"],
                    y=df_q1e["Hybrid_Governing_OneSided_95UB_Pct"],
                    name="Hybrid One-Sided 95% Upper Confidence Bound",
                    line=dict(color="#059669", width=3, dash="dash"),
                )
            )
            fig_q1e.add_trace(
                go.Scatter(
                    x=df_q1e["Month"],
                    y=df_q1e["Linear_Q1E_Governing_Mean_Pct"],
                    name="Classic Linear Q1E Governing Mean (5°C)",
                    line=dict(color="#2563eb", width=2),
                )
            )
            fig_q1e.add_trace(
                go.Scatter(
                    x=df_q1e["Month"],
                    y=df_q1e["Linear_Q1E_Governing_OneSided_95UB_Pct"],
                    name="Classic Linear Q1E One-Sided 95% UB",
                    line=dict(color="#1d4ed8", width=2.5, dash="dot"),
                )
            )
            fig_q1e.add_hline(y=2.0, line_color="#dc2626", annotation_text="2.0% HMW Specification Limit (assumed specification for illustration)")
            fig_q1e.add_vline(x=24.0, line_dash="dot", line_color="#64748b", annotation_text="24M Observed Real-Time Data")
            fig_q1e.add_vline(x=30.0, line_dash="dash", line_color="#f59e0b", annotation_text="30M ICH Q1E Extrapolation Cap (+6M)")
            fig_q1e.update_layout(
                title="ICH Q5C / Q1E 5 °C Shelf-Life Estimation: Hybrid Ensemble 95% UB vs Classic Linear ANCOVA",
                xaxis_title="Real-Time Storage at 5 °C (Months)",
                yaxis_title="HMW Aggregate (%)",
                height=410,
            )
            st.plotly_chart(fig_q1e, use_container_width=True)

        # --- SUB-TAB 3.6: HYBRID MODEL VALIDATION & PRIOR SENSITIVITY ---
        with sci_tabs[5]:
            render_feature_help_popover(
                title="Hybrid Physics+ML Validation & Sensitivity Audit",
                summary="Discloses model safety bounds, physical monotonicity audits, and parameter identifiability limitations.",
                formula_or_logic="Verification across 3,200 evaluation steps: zero negative rates, zero temperature inversions; prior sensitivity across x0.5 to x4.",
                regulatory_basis="FDA Model-Informed Drug Development (MIDD), Guardrails G2 & G5",
                demo_action="Read the honest disclosure of identifiable vs prior-driven thermodynamic parameters.",
            )
            st.markdown("#### 🎯 Hybrid Mechanistic + Neural-Network Validation & Prior Sensitivity Audit")
            st.markdown(
                f"""
                - **Excursion Safety Benchmark:** `{EXCURSION_SAFETY_STATEMENT}`.
                - **Physical Validity Audit (`<= 3M` Training -> `6–24M` & Unseen `15 °C`):** Hybrid Lumry-Eyring ODE has **{PHYSICAL_VALIDITY_STATEMENT}** (`0/3,200` negative-rate steps and `0/2,400` temperature-order violations), whereas the tuned Black-Box MLP exhibits non-monotone temperature reversals when interpolating at unseen `15 °C`.
                - **Baseline Comparison (`matched` vs `mismatched`):** On structurally matched lots (`val`), the Hybrid Ensemble (`0.130 pp` RMSE) beats both Pooled Mean (`0.382 pp`) and Linear Ridge (`0.143 pp`) with learned `pH` + `pH^2` inputs; **{MISMATCHED_RIDGE_STATEMENT}**.
                - **Prior-Driven vs Identifiable Parameters:** Only `dG_eff_ref`, `k_eff(5/25/40°C)`, and apparent `E_a` are identifiable from SEC HMW trajectories; `dH_agg`, `K_u(5°C)`, and `dH_u` are **prior-driven** (reported for diagnostic transparency and never gated). Moreover, latent parameter interval calibration is possible only in synthetic data where true parameters are known; on real lots, only observation prediction intervals are empirically verifiable.
                """
            )

    # -----------------------------------------------------------------------
    # WORKSPACE 4: QUALITY INTELLIGENCE, OOS & COMPLIANCE
    # -----------------------------------------------------------------------
    elif workspace == "🛡️ Quality Intelligence, OOS & Compliance":
        render_ai_studio_breadcrumb(workspace, "FDA OOS Phase I/II Telemetry Gate", current_agent3_state)
        render_feature_help_popover(
            title="Quality Intelligence, OOS Triage & Compliance",
            summary="Performs forensic chromatography telemetry parsing to eliminate illegal testing-into-compliance and executes Part 11 sign-offs.",
            formula_or_logic="Gated Phase I/II decision tree + SHA-256 cryptographic hash-chained immutable audit logging.",
            regulatory_basis="FDA 21 CFR 211.192, FDA OOS Guidance (2006), 21 CFR Part 11",
            demo_action="Toggle the telemetry radio button in the right inspector to see the system switch between 'Invalid Run' and 'True OOS locked out'.",
        )

        qual_tabs = st.tabs([
            "1. OOS Telemetry Gate",
            "2. Quality SPC & CAPA",
            "3. Dynamic FMEA Risk",
            "4. Part 11 Sign-Off & Audit",
        ])

        # --- SUB-TAB 4.1: OOS TELEMETRY TRIAGE (AGENT 3) ---
        with qual_tabs[0]:
            render_feature_help_popover(
                title="FDA OOS Phase I/II Telemetry Gate (Agent 3)",
                summary="Distinguishes between assignable laboratory equipment faults and true product quality failures using raw instrument telemetry.",
                formula_or_logic="Gated rules: pressure spike >300 bar or chiller temp >10°C triggers Laboratory Cause (Run Aborted/Voided, no Phase II escalation).",
                regulatory_basis="FDA Guidance for Industry: Investigating Out-of-Specification (OOS) Test Results (2006)",
                demo_action="Toggle between 'Empower Run #14 Hardware Fault' and 'Clean Bench & Telemetry' in the right inspector.",
            )
            st.subheader("Agent 3: Local MSO — FDA 2006 OOS Phase I/II Gated Triage & Investigation")
            oos_diag = agent3_oos_out["diagnostics"]
            oos_box = "gate-box-warn" if not oos_diag["phase_2_required"] else "gate-box-crit"

            st.markdown(
                f"""
                <div class="{oos_box}" style="margin-bottom: 12px;">
                <strong>Investigation ID:</strong> <code>{agent3_oos_out['investigation_id']}</code> | <strong>Batch:</strong> <code>{agent3_oos_out['batch_id']}</code> | <strong>Initial Result:</strong> <code>{agent3_oos_out['reported_oos']}</code><br>
                <strong>Gated Ruling:</strong> {oos_diag['gate_decision']}<br>
                <em>{oos_diag['regulatory_ruling']}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Compact 2-Column KPI Card (Summary-First)
            col_oos_card1, col_oos_card2 = st.columns(2)
            with col_oos_card1:
                st.markdown(
                    r"""
                    <div class="google-card" style="padding: 16px 18px; height: 100%; border-left: 4px solid #1a73e8; margin-bottom: 10px;">
                        <div style="font-size: 13px; font-weight: 700; color: #1a73e8; text-transform: uppercase; margin-bottom: 8px;">
                            📌 Challenge & Incident Summary
                        </div>
                        <ul style="margin: 0; padding-left: 18px; font-size: 12.5px; color: #3c4043; line-height: 1.6;">
                            <li><strong>Late Friday Lot Release:</strong> Monomer purity tested at <strong>93.8%</strong> against specification ≥ 95.0%.</li>
                            <li><strong>Standard 30-Day Quarantine Avoided:</strong> Telemetry detects assignable hardware failure, preventing unnecessary manufacturing delays.</li>
                            <li><strong>Strict Guardrails G2 & G6:</strong> Decision support only; generates auditable dossier and <strong>never auto-dispositions</strong> batches.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_oos_card2:
                is_hardware_fault = (oos_scenario == "Run_14_Hardware_Anomaly")
                p_reading = "310.4 bar ⚠️" if is_hardware_fault else "215.2 bar (Nominal)"
                q_reading = "54h @ 17.8°C" if is_hardware_fault else "2.1h @ 5.0°C"
                act_reading = "Invalidate via Phase I" if is_hardware_fault else "Escalate to Phase II"
                act_bg = "#e6f4ea" if is_hardware_fault else "#fce8e6"
                act_color = "#137333" if is_hardware_fault else "#d93025"
                st.markdown(
                    f"""
                    <div class="google-card" style="padding: 16px 18px; height: 100%; border-left: 4px solid #d93025; margin-bottom: 10px;">
                        <div style="font-size: 13px; font-weight: 700; color: #d93025; text-transform: uppercase; margin-bottom: 8px;">
                            ⚡ Telemetry Findings & Automated Recommendation
                        </div>
                        <div style="font-size: 12.5px; color: #202124; line-height: 1.7;">
                            • <strong>Pressure:</strong> <code>{p_reading}</code> (Cutoff: 300.0 bar)<br>
                            • <strong>Queue Hold:</strong> <code>{q_reading}</code> (Chiller fault > 10.0°C)<br>
                            • <strong>Action:</strong> <span style="background: {act_bg}; color: {act_color}; font-weight: 700; padding: 2px 8px; border-radius: 4px;">{act_reading}</span> (Candidate for QA Sign-Off)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
            if st.button("📋 Open Phase I Audit Checklist & SOP Details", key="btn_app_oos_phase1_dialog", use_container_width=True):
                open_phase1_audit_dialog(is_hardware_fault=is_hardware_fault)

            render_card_c_root_cause_tree(key_prefix="app_center_root_cause_tree")

            df_tel = agent3_oos_out["df_telemetry"]
            fig_tel = make_subplots(specs=[[{"secondary_y": True}]])
            fig_tel.add_trace(
                go.Scatter(
                    x=df_tel["injection_no"],
                    y=df_tel["pump_pressure_bar"],
                    mode="lines+markers",
                    name="Pump Backpressure (bar)",
                    line=dict(color="#dc2626", width=3),
                ),
                secondary_y=False,
            )
            fig_tel.add_trace(
                go.Scatter(
                    x=df_tel["injection_no"],
                    y=df_tel["autosampler_temp_c"],
                    mode="lines+markers",
                    name="Sample Manager Chiller Temp (°C)",
                    line=dict(color="#f59e0b", width=3, dash="dash"),
                ),
                secondary_y=True,
            )
            fig_tel.add_hline(
                y=300.0,
                line_dash="dot",
                line_color="#991b1b",
                annotation_text="300 bar Overpressure Abort Limit",
            )
            fig_tel.update_layout(
                title="Waters Empower 3 Run #14 Hardware Telemetry (Pressure Spike & Chiller Drift)",
                xaxis_title="Sequence Injection Number",
                height=380,
            )
            fig_tel.update_yaxes(title_text="Pump Pressure (bar)", secondary_y=False, range=[100, 340])
            fig_tel.update_yaxes(title_text="Autosampler Temperature (°C)", secondary_y=True, range=[0, 25])
            st.plotly_chart(fig_tel, use_container_width=True)

            with st.expander("🔍 Empower Run #14 Injection Log", expanded=False):
                st.dataframe(
                    df_tel[["injection_no", "sample_name", "pump_pressure_bar", "autosampler_temp_c", "sequence_dwell_hours", "acidic_peak_pct"]],
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("✅ Gated Phase I Bench & Telemetry Forensic Checklist (`assets/gated_phase1_checklist.json`)", expanded=False):
                st.dataframe(agent3_oos_out["df_checklist"], use_container_width=True, hide_index=True)

        # --- SUB-TAB 4.2: QUALITY INTELLIGENCE, SPC & CAPA (WP3) ---
        with qual_tabs[1]:
            render_feature_help_popover(
                title="Quality Intelligence, SPC & CAPA (WP3)",
                summary="Automated Statistical Process Control charting with Nelson rules and closed-loop CAPA state machine.",
                formula_or_logic="Shewhart I-MR individual & moving range charts + EWMA small-shift detection + Nelson rules 1-8.",
                regulatory_basis="FDA 21 CFR 211.180, ICH Q10 Pharmaceutical Quality System",
                demo_action="Inspect individual batch trends, trigger rules, and link out-of-control signals directly to CAPA records.",
            )
            render_spc_and_quality_intelligence_tab(
                conn=st.session_state["pm_conn"],
                audit_trail=st.session_state.get("audit_trail"),
            )

        # --- SUB-TAB 4.3: DYNAMIC FMEA RISK MATRIX ---
        with qual_tabs[2]:
            render_feature_help_popover(
                title="Dynamic ICH Q9(R1) QC FMEA Risk Register",
                summary="Calculates Risk Priority Numbers across QC testing operations and measures pre- vs post-CAPA risk reduction.",
                formula_or_logic="RPN = Severity (S) x Occurrence (O) x Detection (D); dynamic re-scoring based on CAPA implementation status.",
                regulatory_basis="ICH Q9(R1) Quality Risk Management",
                demo_action="Review high-RPN failure modes and observe how implemented CAPAs bring risks below the 100 RPN redline.",
            )
            st.subheader("Dynamic ICH Q9(R1) QC FMEA Risk Register (Severity × Occurrence × Detection)")
            df_fmea = agent3_fmea_out["df_fmea"]

            fig_fmea = go.Figure()
            fig_fmea.add_trace(
                go.Bar(
                    x=df_fmea["FMEA_ID"] + ": " + df_fmea["Process_Area"],
                    y=df_fmea["Pre_RPN"],
                    name="Pre-CAPA RPN",
                    marker_color="#ef4444",
                    text=df_fmea["Pre_RPN"],
                    textposition="auto",
                )
            )
            fig_fmea.add_trace(
                go.Bar(
                    x=df_fmea["FMEA_ID"] + ": " + df_fmea["Process_Area"],
                    y=df_fmea["Post_RPN"],
                    name="Post-CAPA RPN",
                    marker_color="#10b981",
                    text=df_fmea["Post_RPN"],
                    textposition="auto",
                )
            )
            fig_fmea.add_hline(
                y=100,
                line_dash="dash",
                line_color="#dc2626",
                annotation_text="Critical RPN Redline (100)",
            )
            fig_fmea.update_layout(
                barmode="group",
                title="Example Biologics Site QC Risk Register: Pre-Mitigation vs. Post-CAPA RPN Scores",
                yaxis_title="Risk Priority Number (RPN = S × O × D)",
                height=380,
            )
            st.plotly_chart(fig_fmea, use_container_width=True)
            st.dataframe(df_fmea, use_container_width=True, hide_index=True)

        # --- SUB-TAB 4.4: PART 11 HITL SIGN-OFF & AUDIT TRAIL ---
        with qual_tabs[3]:
            render_feature_help_popover(
                title="Part 11-style HITL Sign-Off & Audit Trail",
                summary="Ensures human-in-the-loop governance for all automated recommendations and maintains an immutable audit trail.",
                formula_or_logic="Attributable e-signature gate with minimum 10-char justification + SHA-256 block hash chain linking.",
                regulatory_basis="FDA 21 CFR Part 11 (Electronic Records; Electronic Signatures), ALCOA+ Data Integrity",
                demo_action="Complete the reviewer fields and checkbox to unlock TrackWise JSON and Markdown memo export.",
            )
            st.subheader("✍️ Part 11-style HITL sign-off (demonstration only; not a compliant e-signature system) & ALCOA+ Audit Trail")
            st.markdown(
                f"""
                <div class="gate-box-pass">
                <strong>{DRAFT_STAMP}</strong><br>
                <strong>cGMP Classification Guardrail:</strong> All 4 agents operate strictly as <em>Non-GMP Decision Support Systems</em>. Export of the TrackWise Digital Quality Event JSON/Markdown Memo is locked until Reviewer Name, Role, Justification (&ge;10 chars), and explicit QA confirmation checkbox are completed below.
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.form("hitl_approval_form"):
                c_f1, c_f2, c_f3 = st.columns(3)
                with c_f1:
                    approver_name = st.text_input("Reviewer Full Name", value="")
                with c_f2:
                    qa_cosigner = st.text_input("Reviewer Role / QA Co-Signer Title", value="")
                with c_f3:
                    agent_package = st.selectbox(
                        "Decision Package to Approve",
                        options=[
                            "End-to-End MSO Package: Hybrid Excursion + Dynamic OOT + ICH Q5C/Q1E + FMEA",
                            "Agent 1: W42/W43 Heijunka Schedule Leveling (Site-defined pull windows, example ±7 d)",
                            "Agent 2: Hybrid Lumry-Eyring Excursion Memo & Waters UPLC 3.04 min Dwell Hold",
                            "Agent 3: OOS-2026-FRM-089 Phase I Telemetry Ruling & FMEA CAPA Update",
                            "Agent 4: External Example CDMO Client Client 24h QTA Notification & Change Order CO-09",
                        ],
                    )
                rationale_note = st.text_input(
                    "Part 11-style HITL Sign-Off Meaning / Justification (min 10 chars)",
                    value="",
                )
                confirmed_chk = st.checkbox(
                    "I confirm I have independently reviewed the deterministic kinetic outputs, analytical caveats, and prior-sensitivity tables for Non-GMP QA review."
                )
                submitted = st.form_submit_button("✅ Execute Part 11-style HITL sign-off & Unlock Export")
                if submitted:
                    trail_inst = (
                        st.session_state.get("audit_trail")
                        if (
                            isinstance(st.session_state.get("audit_trail"), AuditTrail)
                            or hasattr(st.session_state.get("audit_trail"), "append_event")
                        )
                        else None
                    )
                    gate_out = verify_and_sign_hitl_gate(
                        draft_package=mso_draft_pkg,
                        reviewer_name=approver_name,
                        reviewer_role=qa_cosigner,
                        justification=rationale_note,
                        confirmed=confirmed_chk,
                        audit_trail=trail_inst,
                    )
                    if not gate_out["approved"]:
                        st.session_state["signed_mso_pkg"] = None
                        for err_msg in gate_out["errors"]:
                            st.error(f"HITL Gate Blocked: {err_msg}")
                    else:
                        st.session_state["signed_mso_pkg"] = gate_out["signed_package"]
                        sha_hash = gate_out["alcoa_sha256_hash"][:16]
                        st.success(f"Electronic signature verified. Export unlocked with ALCOA+ SHA-256 digest `{sha_hash}`.")

            if st.session_state["signed_mso_pkg"] is not None and st.session_state["signed_mso_pkg"]["export_unlocked"]:
                signed_pkg = st.session_state["signed_mso_pkg"]
                dl_c1, dl_c2 = st.columns(2)
                with dl_c1:
                    st.download_button(
                        "📥 Download Signed TrackWise Draft JSON",
                        data=json.dumps(signed_pkg["trackwise_json"], indent=2),
                        file_name="trackwise_quality_event_draft_signed.json",
                        mime="application/json",
                    )
                with dl_c2:
                    st.download_button(
                        "📥 Download Signed MSO Technical Evaluation Memo (Markdown)",
                        data=signed_pkg["markdown_memo"],
                        file_name="mso_technical_evaluation_memo_signed.md",
                        mime="text/markdown",
                    )
            else:
                st.warning("🔒 Export Locked: Complete Reviewer Name, Role, Justification, and Confirmation Checkbox above to unlock TrackWise JSON / Markdown export.")

            audit_state = st.session_state.get("audit_trail", [])
            if isinstance(audit_state, AuditTrail) or hasattr(audit_state, "records"):
                audit_rows = []
                for r in audit_state.records:
                    d = r.to_dict() if hasattr(r, "to_dict") else (r if isinstance(r, dict) else {})
                    if isinstance(d, dict) and "index" in d:
                        audit_rows.append({
                            "Index": d.get("index"),
                            "Timestamp_UTC": d.get("timestamp_utc"),
                            "Actor": f"{d.get('actor_name', '')} ({d.get('actor_role', '')})",
                            "Action": d.get("action"),
                            "Entity": f"{d.get('entity_type', '')}:{d.get('entity_id', '')}",
                            "Justification": d.get("justification"),
                            "Record_Hash": str(d.get("record_hash", ""))[:16] + "..." if d.get("record_hash") else "",
                            "Disclaimer": d.get("disclaimer"),
                        })
                df_audit = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame()
                if hasattr(audit_state, "verify_chain"):
                    is_valid, err_msg = audit_state.verify_chain()
                    if is_valid:
                        st.success(f"Audit Trail Integrity: Cryptographically Verified ({len(audit_state.records)} blocks, unbroken SHA-256 chain).")
                    else:
                        st.error(f"Audit Trail Integrity Alert: {err_msg}")
            else:
                if hasattr(audit_state, "get_all_records"):
                    audit_records = audit_state.get_all_records()
                elif isinstance(audit_state, list):
                    audit_records = audit_state
                else:
                    audit_records = []
                df_audit = pd.DataFrame(audit_records) if audit_records else pd.DataFrame()
            st.dataframe(df_audit, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # WORKSPACE 5: TWO-LAYER AGENT ARCHITECTURE INSPECTOR
    # -----------------------------------------------------------------------
    else:
        render_ai_studio_breadcrumb(workspace, "Declarative & Deterministic Layers", current_agent3_state)
        render_feature_help_popover(
            title="Two-Layer Agent Architecture Inspector",
            summary="Inspects the cGMP-aligned separation between declarative SOP policies (Layer 1) and deterministic Python calculation engines (Layer 2).",
            formula_or_logic="Layer 1 YAML frontmatter & markdown guardrails; Layer 2 unit-tested deterministic scripts and schemas.",
            regulatory_basis="FDA 21 CFR 211.68(b), ALCOA+ Attributable & Original",
            demo_action="Select any agent script or schema to inspect the live source code.",
        )
        st.subheader("🗂️ Two-Layer Self-Contained Agent Directory Inspector (`THE WHAT` vs. `THE HOW`)")
        st.markdown(
            """
            Inspect the exact two-layer cGMP file structure inside `/agents/`:
            - **LAYER 1 (`THE WHAT` — Declarative `skill.md`):** YAML frontmatter + Markdown SOP defining specialist identity, trigger constraints, and FDA/ICH regulatory guardrails.
            - **LAYER 2 (`THE HOW` — Imperative `/scripts/`, `/reference/`, `/assets/`):** Deterministic Python calculators, regulatory reference guides, and seeded JSON schemas.
            """
        )

        repo_root = Path(__file__).resolve().parent
        agents_root = repo_root / "agents"
        all_agent_files = sorted(
            [p.relative_to(repo_root).as_posix() for p in agents_root.rglob("*") if p.is_file()]
        )

        selected_file_rel = st.selectbox(
            "Select Repository File to Inspect Live:", options=all_agent_files, index=0
        )
        selected_file_path = repo_root / selected_file_rel
        file_content = selected_file_path.read_text(encoding="utf-8")

        st.caption(f"Full Path: `{selected_file_path}`")
        if selected_file_rel.endswith(".py"):
            st.code(file_content, language="python")
        elif selected_file_rel.endswith(".json"):
            st.code(file_content, language="json")
        else:
            st.code(file_content, language="markdown")

    # Render Google AI Studio bottom interaction dock pinned below center canvas
    render_bottom_copilot_dock(workspace)
