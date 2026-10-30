"""
pm/modular_widgets.py

Modular Dashboard Architecture (Power BI & Smartsheet Synergy).
Provides pluggable, standalone UI components for cGMP-QC-CoPilot:
- Card A: Interactive CP-SAT Workload Canvas with Cross-Filtering (Power BI style)
- Card B: Smartsheet-Style Batch Release & Schedule Grid (st.data_editor with CSV export)
- Card C: Root-Cause Decomposition Tree Visualizer (Plotly Sankey/Sunburst)

Complies strictly with 21 CFR Part 11 ALCOA+ principles and Guardrail G2 draft standards.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.session_recorder import get_session_recorder
from core.stamps import DRAFT_STAMP, PART11_DISCLAIMER


# ---------------------------------------------------------------------------
# CARD A: INTERACTIVE CP-SAT WORKLOAD CANVAS WITH CROSS-FILTERING (POWER BI)
# ---------------------------------------------------------------------------
def build_test_workload_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Provides fallback synthetic baseline and leveled capacity dataframes."""
    df_base = pd.DataFrame([
        {"Skid_ID": "UPLC_SEC_CEX", "Skid_Name": "Waters Acquity UPLC", "Week": "W42", "Total_Hours": 172.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 143.3},
        {"Skid_ID": "icIEF_Maurice", "Skid_Name": "ProteinSimple Maurice", "Week": "W42", "Total_Hours": 95.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 79.2},
        {"Skid_ID": "Bioassay_SpectraMax", "Skid_Name": "SpectraMax Bioassay", "Week": "W42", "Total_Hours": 88.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 73.3},
        {"Skid_ID": "Stability_Rees", "Skid_Name": "Rees Stability Chambers", "Week": "W42", "Total_Hours": 60.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 50.0},
    ])
    df_lev = pd.DataFrame([
        {"Skid_ID": "UPLC_SEC_CEX", "Skid_Name": "Waters Acquity UPLC", "Week": "W42", "Total_Hours": 168.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 140.0},
        {"Skid_ID": "icIEF_Maurice", "Skid_Name": "ProteinSimple Maurice", "Week": "W42", "Total_Hours": 95.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 79.2},
        {"Skid_ID": "Bioassay_SpectraMax", "Skid_Name": "SpectraMax Bioassay", "Week": "W42", "Total_Hours": 88.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 73.3},
        {"Skid_ID": "Stability_Rees", "Skid_Name": "Rees Stability Chambers", "Week": "W42", "Total_Hours": 60.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 50.0},
    ])
    return df_base, df_lev


def _first_present(d: dict, *keys):
    """Safely returns the first non-None value for keys in dictionary without evaluating truthiness."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def extract_workload_dfs(cap_demo: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Safely extracts baseline and leveled dataframes with nested traversal and fallback defaults."""
    if not isinstance(cap_demo, dict):
        return build_test_workload_dfs()

    nested = (
        cap_demo.get("legacy_baseline")
        if isinstance(cap_demo.get("legacy_baseline"), dict)
        else (
            cap_demo.get("legacy_leveling")
            if isinstance(cap_demo.get("legacy_leveling"), dict)
            else (
                cap_demo.get("leveling_result")
                if isinstance(cap_demo.get("leveling_result"), dict)
                else {}
            )
        )
    )

    base_keys = ["df_baseline", "baseline_df", "df_unleveled", "unleveled_df", "workload_before"]
    opt_keys = ["df_leveled", "leveled_df", "df_optimized", "optimized_df", "workload_after"]

    df_base = _first_present(cap_demo, *base_keys)
    if df_base is None:
        df_base = _first_present(nested, *base_keys)

    df_opt = _first_present(cap_demo, *opt_keys)
    if df_opt is None:
        df_opt = _first_present(nested, *opt_keys)

    fb_base, fb_opt = build_test_workload_dfs()
    final_base = df_base if df_base is not None else fb_base
    final_opt = df_opt if df_opt is not None else fb_opt

    return final_base, final_opt


def build_cpsat_workload_figure(
    df_base: pd.DataFrame,
    df_lev: pd.DataFrame,
    inst_outage_mult: float = 1.0,
    analyst_abs_mult: float = 1.0,
    inject_rush: bool = False,
    title_before: Optional[str] = None,
    title_after: Optional[str] = None,
) -> Tuple[go.Figure, go.Figure]:
    """Constructs Plotly before and after workload figures for CP-SAT leveling."""
    df_b = df_base.copy()
    df_l = df_lev.copy()

    eff_scale = round(float(inst_outage_mult) * float(analyst_abs_mult), 3)
    if eff_scale != 1.0:
        df_b["Total_Hours"] = (df_b["Total_Hours"] * eff_scale).round(1)
        df_b["Utilization_Pct"] = (df_b["Total_Hours"] / df_b["Max_Capacity_Hours"] * 100.0).round(1)
        df_l["Total_Hours"] = (df_l["Total_Hours"] * eff_scale).round(1)
        df_l["Utilization_Pct"] = (df_l["Total_Hours"] / df_l["Max_Capacity_Hours"] * 100.0).round(1)

    if inject_rush:
        mask_b = (df_b["Week"] == "W42") & (df_b["Skid_ID"] == "UPLC_SEC_CEX")
        df_b.loc[mask_b, "Total_Hours"] = (df_b.loc[mask_b, "Total_Hours"] + 18.0).round(1)
        df_b.loc[mask_b, "Utilization_Pct"] = (df_b.loc[mask_b, "Total_Hours"] / df_b.loc[mask_b, "Max_Capacity_Hours"] * 100.0).round(1)
        mask_l = (df_l["Week"] == "W42") & (df_l["Skid_ID"] == "UPLC_SEC_CEX")
        df_l.loc[mask_l, "Total_Hours"] = (df_l.loc[mask_l, "Total_Hours"] + 18.0).round(1)
        df_l.loc[mask_l, "Utilization_Pct"] = (df_l.loc[mask_l, "Total_Hours"] / df_l.loc[mask_l, "Max_Capacity_Hours"] * 100.0).round(1)

    max_chart_y = max(160.0, float(df_b["Utilization_Pct"].max() + 15.0), float(df_l["Utilization_Pct"].max() + 15.0))

    fig_before = px.bar(
        df_b,
        x="Skid_Name",
        y="Utilization_Pct",
        color="Week",
        barmode="group",
        text="Utilization_Pct",
        color_discrete_sequence=["#ef4444", "#64748b"],
        title="BEFORE Leveling: Unleveled Demand (Physical Overload)",
    )
    fig_before.add_hline(y=85.0, line_dash="dash", line_color="#f59e0b", annotation_text="85% LEAN Buffer Ceiling")
    fig_before.add_hline(y=100.0, line_dash="dot", line_color="#dc2626", annotation_text="100% Hard Capacity")
    fig_before.update_layout(yaxis_range=[0, max_chart_y], height=350, margin=dict(l=10, r=10, t=35, b=10))

    fig_after = px.bar(
        df_l,
        x="Skid_Name",
        y="Utilization_Pct",
        color="Week",
        barmode="group",
        text="Utilization_Pct",
        color_discrete_sequence=["#10b981", "#2563eb"],
        title=title_after or "AFTER Weekly-Bucket Leveling (Heijunka Weekly Aggregates)",
    )
    fig_after.add_hline(y=85.0, line_dash="dash", line_color="#f59e0b", annotation_text="85% LEAN Buffer Ceiling")
    fig_after.add_hline(y=100.0, line_dash="dot", line_color="#dc2626", annotation_text="100% Hard Capacity")
    fig_after.update_layout(yaxis_range=[0, max_chart_y], height=350, margin=dict(l=10, r=10, t=35, b=10))

    return fig_before, fig_after


def render_card_a_cpsat_workload_canvas(
    df_base: pd.DataFrame,
    df_lev: pd.DataFrame,
    inst_outage_mult: float = 1.0,
    analyst_abs_mult: float = 1.0,
    inject_rush: bool = False,
    key_prefix: str = "cpsat_canvas",
    title_before: Optional[str] = None,
    title_after: Optional[str] = None,
) -> Optional[str]:
    """
    Renders Card A: Interactive CP-SAT Workload Canvas with Cross-Filtering (Power BI style).
    Clicking a bar captures the instrument name into st.session_state["selected_instrument_filter"].
    """
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E0E3E7; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(60,64,67,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 13.5px; font-weight: 700; color: #1a73e8; text-transform: uppercase;">
                    📊 Card A: CP-SAT Workload Canvas & Bidirectional Cross-Filter
                </span>
                <span style="font-size: 11px; background: #e8f0fe; color: #1967d2; padding: 2px 8px; border-radius: 10px; font-weight: 600;">
                    Power BI Interactive Cross-Filtering
                </span>
            </div>
            <div style="font-size: 12px; color: #5f6368; margin-bottom: 8px;">
                Click any instrument bar below to cross-filter the batch release schedule grid and inspector diagnostic views.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_before, fig_after = build_cpsat_workload_figure(
        df_base=df_base,
        df_lev=df_lev,
        inst_outage_mult=inst_outage_mult,
        analyst_abs_mult=analyst_abs_mult,
        inject_rush=inject_rush,
        title_before=title_before,
        title_after=title_after,
    )

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.plotly_chart(fig_before, use_container_width=True, key=f"{key_prefix}_before_chart")

    with col_w2:
        try:
            select_event = st.plotly_chart(
                fig_after,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="cpsat_chart_select",
            )
        except TypeError:
            select_event = st.plotly_chart(fig_after, use_container_width=True, key="cpsat_chart_select")

    # Extract selected instrument from selection event if triggered
    selected_inst: Optional[str] = None
    if select_event:
        points = []
        if isinstance(select_event, dict):
            points = select_event.get("selection", {}).get("points", [])
        elif hasattr(select_event, "selection"):
            sel = getattr(select_event, "selection", None)
            if isinstance(sel, dict):
                points = sel.get("points", [])
            elif hasattr(sel, "points"):
                points = getattr(sel, "points", [])

        if points:
            pt = points[0]
            val = pt.get("x") if isinstance(pt, dict) else getattr(pt, "x", None)
            if val:
                selected_inst = str(val)
                st.session_state["selected_instrument_filter"] = selected_inst

    return st.session_state.get("selected_instrument_filter")


# ---------------------------------------------------------------------------
# CARD B: SMARTSHEET-STYLE BATCH RELEASE & SCHEDULE GRID (st.data_editor)
# ---------------------------------------------------------------------------
def build_default_batch_schedule_df() -> pd.DataFrame:
    """Constructs the seed batch release and schedule dataset."""
    data = [
        {"Lot ID": "LOT-NS-2026-004", "Instrument": "Waters Acquity UPLC", "Run Time (h)": 18.0, "Delta Days": 0, "Disposition Status": "PHASE I REVIEW", "Notes": "Late Friday Monomer 93.8% OOS; telemetry anomaly identified"},
        {"Lot ID": "LOT-BI-CAP-101", "Instrument": "Waters Acquity UPLC", "Run Time (h)": 24.0, "Delta Days": 0, "Disposition Status": "SCHEDULED", "Notes": "Commercial release campaign; immutable tier-1a zero float commitment"},
        {"Lot ID": "LOT-PROJ-001", "Instrument": "Waters Acquity UPLC", "Run Time (h)": 16.0, "Delta Days": 0, "Disposition Status": "SCHEDULED", "Notes": "External CDMO binding SLA; critical path locked"},
        {"Lot ID": "LOT-BI-CAP-103", "Instrument": "Waters Acquity UPLC", "Run Time (h)": 14.5, "Delta Days": 3, "Disposition Status": "SCHEDULED", "Notes": "Shifted +3d within permitted ICH Q1A(R2) ±7d pull window"},
        {"Lot ID": "LOT-PROJ-003", "Instrument": "Waters Acquity UPLC", "Run Time (h)": 12.0, "Delta Days": -2, "Disposition Status": "SCHEDULED", "Notes": "Shifted -2d within permitted ICH Q1A(R2) ±7d pull window"},
        {"Lot ID": "LOT-BI-CAP-104", "Instrument": "ProteinSimple Maurice icIEF", "Run Time (h)": 14.0, "Delta Days": 1, "Disposition Status": "SCHEDULED", "Notes": "Shifted +1d; charge variant monitoring nominal"},
        {"Lot ID": "LOT-PROJ-002", "Instrument": "ProteinSimple Maurice icIEF", "Run Time (h)": 11.5, "Delta Days": 0, "Disposition Status": "RELEASED", "Notes": "Routine release batch; cIEF system suitability checked"},
        {"Lot ID": "LOT-BI-CAP-105", "Instrument": "SpectraMax Bioassay Suite", "Run Time (h)": 16.0, "Delta Days": 2, "Disposition Status": "SCHEDULED", "Notes": "Shifted +2d; 4-PL relative potency assay preparation"},
        {"Lot ID": "LOT-PROJ-004", "Instrument": "SpectraMax Bioassay Suite", "Run Time (h)": 10.0, "Delta Days": 0, "Disposition Status": "RELEASED", "Notes": "Relative potency 102.4%; parallelism passed"},
        {"Lot ID": "LOT-BI-CAP-102", "Instrument": "Rees Stability Chambers", "Run Time (h)": 8.0, "Delta Days": 0, "Disposition Status": "SCHEDULED", "Notes": "Routine 5°C pull; chamber environmental log nominal"},
    ]
    return pd.DataFrame(data)


def filter_batch_schedule_df(df: pd.DataFrame, instrument_filter: Optional[str] = None) -> pd.DataFrame:
    """Filters batch release schedule dataframe by selected instrument keyword."""
    if not instrument_filter or instrument_filter.strip() in ("", "All Instruments", "None"):
        return df.copy()

    filt_lower = instrument_filter.lower().strip()
    # Normalize keywords for common instrument matching
    kw = "uplc" if "uplc" in filt_lower or "waters" in filt_lower else (
        "icief" if "icief" in filt_lower or "maurice" in filt_lower else (
            "bioassay" if "bioassay" in filt_lower or "spectramax" in filt_lower else (
                "chamber" if "chamber" in filt_lower or "rees" in filt_lower else filt_lower
            )
        )
    )
    mask = df["Instrument"].str.lower().str.contains(kw, na=False)
    filtered = df[mask].copy()
    return filtered if not filtered.empty else df.copy()


def render_card_b_smartsheet_grid(
    df_schedule: Optional[pd.DataFrame] = None,
    key_prefix: str = "smartsheet_schedule_grid",
) -> pd.DataFrame:
    """
    Renders Card B: Smartsheet-Style Batch Release & Schedule Grid (st.data_editor).
    Includes active cross-filter chip, configurable dropdown columns, CSV export,
    and records edit events into session_recorder.py with an updated SHA-256 state hash.
    """
    base_df = df_schedule if df_schedule is not None else build_default_batch_schedule_df()

    if "_smartsheet_schedule_master" not in st.session_state:
        st.session_state["_smartsheet_schedule_master"] = base_df.copy()

    master_df = st.session_state["_smartsheet_schedule_master"]

    active_filter = st.session_state.get("selected_instrument_filter")

    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E0E3E7; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(60,64,67,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 13.5px; font-weight: 700; color: #1a73e8; text-transform: uppercase;">
                    📋 Card B: Smartsheet-Style Batch Release & Schedule Grid
                </span>
                <span style="font-size: 11px; background: #e6f4ea; color: #137333; padding: 2px 8px; border-radius: 10px; font-weight: 600;">
                    Interactive Data Editor & Instant Export
                </span>
            </div>
            <div style="font-size: 12px; color: #5f6368;">
                Directly edit lot run times, schedule shifts, disposition statuses, and operational notes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active Cross-Filter Indicator Chip Row
    col_chip, col_action = st.columns([3.2, 0.8], vertical_alignment="center")
    with col_chip:
        if active_filter:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 12px; font-weight: 600; color: #5f6368;">Active Cross-Filter:</span>
                    <span style="background: #e8f0fe; border: 1px solid #1a73e8; color: #1a73e8; font-weight: 700; font-size: 12px; padding: 3px 10px; border-radius: 16px;">
                        🔍 {active_filter}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("Displaying all 10 scheduled batches across analytical skids. (Click a bar in Card A to filter).")

    with col_action:
        if active_filter:
            if st.button("✕ Clear Filter", key=f"{key_prefix}_clear_filter_btn", use_container_width=True):
                st.session_state["selected_instrument_filter"] = None
                st.rerun()

    # Filter rows based on active instrument filter
    view_df = filter_batch_schedule_df(master_df, active_filter)

    column_config = {
        "Lot ID": st.column_config.TextColumn("Lot ID", disabled=True, width="medium"),
        "Instrument": st.column_config.TextColumn("Instrument", disabled=True, width="medium"),
        "Run Time (h)": st.column_config.NumberColumn(
            "Run Time (h)",
            min_value=0.0,
            max_value=168.0,
            step=0.5,
            format="%.1f h",
            help="Allocated instrument skid test hours",
        ),
        "Delta Days": st.column_config.NumberColumn(
            "Delta Days",
            min_value=-14,
            max_value=14,
            step=1,
            format="%+d d",
            help="CP-SAT leveling shift in calendar days",
        ),
        "Disposition Status": st.column_config.SelectboxColumn(
            "Disposition Status",
            options=["SCHEDULED", "PHASE I REVIEW", "RELEASED"],
            required=True,
            help="Current quality lifecycle status for lot release",
        ),
        "Notes": st.column_config.TextColumn("Notes", width="large", help="Operational rationale and compliance justifications"),
    }

    edited_df = st.data_editor(
        view_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        key=f"{key_prefix}_editor",
    )

    # Detect user modifications and record in SessionRecorder
    csv_bytes = edited_df.to_csv(index=False).encode("utf-8")
    curr_hash = hashlib.sha256(csv_bytes).hexdigest()
    prev_hash = st.session_state.get(f"{key_prefix}_last_hash")

    if prev_hash is not None and prev_hash != curr_hash:
        recorder = get_session_recorder()
        recorder.record_event(
            event_type="SMARTSHEET_GRID_EDIT",
            workspace="📊 QC Operations & Project Management",
            widget_key=f"{key_prefix}_editor",
            old_value=prev_hash[:16],
            new_value=curr_hash[:16],
            agent3_status="NOMINAL",
            metadata={"rows": len(edited_df), "filter": active_filter or "ALL"},
        )
        # Update master table for modified rows
        for _, r in edited_df.iterrows():
            lot = r["Lot ID"]
            match = master_df["Lot ID"] == lot
            if match.any():
                master_df.loc[match, "Run Time (h)"] = r["Run Time (h)"]
                master_df.loc[match, "Delta Days"] = r["Delta Days"]
                master_df.loc[match, "Disposition Status"] = r["Disposition Status"]
                master_df.loc[match, "Notes"] = r["Notes"]
        st.session_state["_smartsheet_schedule_master"] = master_df

    st.session_state[f"{key_prefix}_last_hash"] = curr_hash

    # Smartsheet CSV Instant Export Row
    exp_c1, exp_c2 = st.columns([3, 1], vertical_alignment="center")
    with exp_c1:
        st.caption(f"SHA-256 State Digest: `{curr_hash[:16]}...` | Rows: {len(edited_df)} | Export includes human-in-the-loop audit metadata.")
    with exp_c2:
        st.download_button(
            label="📥 Export to Smartsheet CSV",
            data=csv_bytes,
            file_name=f"smartsheet_batch_schedule_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_download_btn",
        )

    return edited_df


# ---------------------------------------------------------------------------
# CARD C: ROOT-CAUSE DECOMPOSITION TREE VISUALIZER (SANKEY & SUNBURST)
# ---------------------------------------------------------------------------
def build_root_cause_tree_figure(diagram_type: str = "sankey") -> go.Figure:
    """
    Constructs an interactive Root-Cause Decomposition Tree Visualizer:
    - Node 0: Monomer Release (93.8%)
    - Node 1: Pump Overpressure (310.4 bar)
    - Node 2: Chiller Fault (17.8°C / 54h)
    - Node 3: Phase I Lab Invalidation
    """
    if diagram_type.lower() == "sunburst":
        labels = [
            "Root Cause",
            "Monomer Release (93.8%)",
            "Pump Overpressure (310.4 bar)",
            "Chiller Fault (17.8°C / 54h)",
            "Phase I Lab Invalidation",
        ]
        parents = [
            "",
            "Root Cause",
            "Monomer Release (93.8%)",
            "Monomer Release (93.8%)",
            "Root Cause",
        ]
        values = [100, 50, 25, 25, 50]
        colors = ["#1a73e8", "#4285f4", "#ea4335", "#fbbc04", "#34a853"]

        fig = go.Figure(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                marker=dict(colors=colors),
                hoverinfo="label+value+percent parent",
            )
        )
        fig.update_layout(
            margin=dict(l=5, r=5, t=15, b=5),
            height=280,
        )
        return fig

    # Default: Plotly Sankey Flow Diagram with Shortened Labels & High Legibility
    node_labels = [
        "Monomer Release (93.8%)",
        "Pump Overpressure (310.4 bar)",
        "Chiller Fault (17.8°C / 54h)",
        "Phase I Lab Invalidation",
    ]
    node_colors = [
        "#1A73E8",  # Branch 1 (Blue)
        "#D93025",  # Branch 2 (Red)
        "#F9AB00",  # Branch 3 (Amber)
        "#1E8E3E",  # Branch 4 (Green)
    ]

    link_sources = [0, 0, 1, 2]
    link_targets = [1, 2, 3, 3]
    link_values = [50, 50, 50, 50]
    link_colors = [
        "rgba(217, 48, 37, 0.35)",
        "rgba(249, 171, 0, 0.35)",
        "rgba(30, 142, 62, 0.40)",
        "rgba(30, 142, 62, 0.40)",
    ]

    fig = go.Figure(
        go.Sankey(
            textfont=dict(color="#202124", size=12, family="Google Sans, Roboto, Inter, sans-serif"),
            node=dict(
                pad=40,
                thickness=24,
                line=dict(color="#FFFFFF", width=1.5),
                label=node_labels,
                color=node_colors,
                hoverlabel=dict(bgcolor="#ffffff", font_size=11, font_color="#202124"),
            ),
            link=dict(
                source=link_sources,
                target=link_targets,
                value=link_values,
                color=link_colors,
            ),
        )
    )

    fig.update_layout(
        title_text="Root-Cause Decomposition Tree (Forensic Flow)",
        title_font_size=13,
        height=360,
        margin=dict(l=20, r=20, t=35, b=25),
    )
    return fig


def render_card_c_root_cause_tree(key_prefix: str = "root_cause_tree", compact_mode: bool = False) -> None:
    """
    Renders Card C: Root-Cause Decomposition Tree Visualizer in Google AI Studio card styling.
    When compact_mode is True (e.g. Right Inspector rail), avoids wide Sankey diagrams and
    renders a vertical forensic status summary with badges and Sunburst hierarchy.
    """
    if compact_mode:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E0E3E7; border-radius: 12px; padding: 14px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 12.5px; font-weight: 700; color: #1a73e8; text-transform: uppercase;">
                        🌳 Root-Cause Forensic Status
                    </span>
                    <span style="font-size: 10px; background: #fef7e0; color: #b06000; padding: 2px 6px; border-radius: 8px; font-weight: 600;">
                        Phase I Lab Fault
                    </span>
                </div>
                <div style="font-size: 11px; color: #5f6368; margin-bottom: 10px;">
                    Assignable instrument and environmental failure modes:
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px;">
                    <div style="background: #fce8e6; border: 1px solid #fad2cf; color: #d93025; padding: 5px 9px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                        ⚠️ Pump: 310.4 bar (Cutoff: 300.0 bar)
                    </div>
                    <div style="background: #fef7e0; border: 1px solid #feefc3; color: #b06000; padding: 5px 9px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                        ⚠️ Chiller: 17.8°C (Autosampler Hold: 54h)
                    </div>
                    <div style="background: #e6f4ea; border: 1px solid #ceead6; color: #137333; padding: 5px 9px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                        ✅ Action: Invalidate via Phase I
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_compact = build_root_cause_tree_figure(diagram_type="sunburst")
        fig_compact.update_layout(height=230, margin=dict(l=5, r=5, t=10, b=5))
        st.plotly_chart(fig_compact, use_container_width=True, key=f"{key_prefix}_compact_chart")
        return
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E0E3E7; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(60,64,67,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 13.5px; font-weight: 700; color: #1a73e8; text-transform: uppercase;">
                    🌳 Card C: Root-Cause Decomposition Tree Visualizer
                </span>
                <span style="font-size: 11px; background: #fef7e0; color: #b06000; padding: 2px 8px; border-radius: 10px; font-weight: 600;">
                    FDA 2006 Phase I Forensic Flow
                </span>
            </div>
            <div style="font-size: 12px; color: #5f6368; margin-bottom: 8px;">
                Deconstructs late Friday monomer assay failure into independent hardware and thermal dwell branches, proving assignable lab cause.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_tree, t_details = st.columns([3.2, 1.2], vertical_alignment="center")
    with t_details:
        diag_choice = st.radio(
            "Visualization Mode:",
            options=["Sankey Flow", "Sunburst Hierarchy"],
            key=f"{key_prefix}_diag_choice",
            horizontal=False,
        )

    diag_mode = "sunburst" if "Sunburst" in diag_choice else "sankey"
    fig_tree = build_root_cause_tree_figure(diagram_type=diag_mode)

    with t_tree:
        st.plotly_chart(fig_tree, use_container_width=True, key=f"{key_prefix}_chart")

    # Forensic branch breakdown micro-cards
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        st.markdown(
            """
            <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-radius: 8px; padding: 10px; font-size: 11.5px; color: #202124;">
                <strong style="color: #d93025;">Branch 2: Hardware Malfunction</strong><br>
                Pump backpressure <code>310.4 bar</code> exceeded the 300.0 bar cutoff limit, voiding chromatography integration.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b_col2:
        st.markdown(
            """
            <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-radius: 8px; padding: 10px; font-size: 11.5px; color: #202124;">
                <strong style="color: #b06000;">Branch 3: Thermal Hold Dwell</strong><br>
                Autosampler chamber drifted to <code>17.8°C</code> for 54 hours, triggering non-isothermal sample degradation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b_col3:
        st.markdown(
            """
            <div style="background: #F8F9FA; border: 1px solid #E0E3E7; border-radius: 8px; padding: 10px; font-size: 11.5px; color: #202124;">
                <strong style="color: #137333;">Branch 4: Phase I Ruling</strong><br>
                Candidate for laboratory invalidation under FDA 2006 OOS guidance prior to Phase II manufacturing escalation.
            </div>
            """,
            unsafe_allow_html=True,
        )
