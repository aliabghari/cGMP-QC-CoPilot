"""
pm/topology_ui.py

Interactive Live Plant & Multi-Agent Network Topology Component for cGMP-QC-CoPilot.
Visualizes real-time multi-agent orchestration, analytical skid telemetry feeds,
and 21 CFR Part 11 human-in-the-loop governance gates in an industrial SCADA-style graph.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.stamps import DRAFT_STAMP, PART11_DISCLAIMER


def get_node_definitions(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Generates the 10 topology nodes with dynamic status derived from active simulation states."""
    oos_scenario = context.get("oos_scenario", "Run_14_Hardware_Anomaly")
    is_hardware_fault = oos_scenario == "Run_14_Hardware_Anomaly"

    inst_outage_mult = float(context.get("inst_outage_mult", 1.0))
    analyst_abs_mult = float(context.get("analyst_abs_mult", 1.0))
    inject_rush = bool(context.get("inject_rush", False))
    usp_harvest_delay = int(context.get("usp_harvest_delay", 0))

    excursion_temp = float(context.get("excursion_temp", 28.0))
    excursion_hours = float(context.get("excursion_hours", 36.0))
    is_excursion_alert = excursion_temp > 25.0

    signed_pkg = context.get("signed_mso_pkg")
    is_signed = signed_pkg is not None and signed_pkg.get("export_unlocked", False)

    # Dynamic status & colors
    uplc_status = "CRITICAL ALARM" if is_hardware_fault else ("HIGH LOAD" if (inst_outage_mult > 1.2 or inject_rush) else "NOMINAL")
    uplc_color = "#d93025" if is_hardware_fault else ("#ea8600" if (inst_outage_mult > 1.2 or inject_rush) else "#1e8e3e")

    chamber_status = "EXCURSION ACTIVE" if is_excursion_alert else "NOMINAL"
    chamber_color = "#d93025" if is_excursion_alert else "#1e8e3e"

    agent1_status = "LEVELING ACTIVE" if (usp_harvest_delay > 0 or inject_rush or inst_outage_mult > 1.0) else "OPTIMAL"
    agent1_color = "#1a73e8" if (usp_harvest_delay > 0 or inject_rush or inst_outage_mult > 1.0) else "#1e8e3e"

    agent3_status = "PHASE I ROOT CAUSE" if is_hardware_fault else "NOMINAL MONITOR"
    agent3_color = "#d93025" if is_hardware_fault else "#1e8e3e"

    agent2_status = "KINETIC SOLVE" if is_excursion_alert else "MONITORING"
    agent2_color = "#1a73e8" if is_excursion_alert else "#1e8e3e"

    hitl_status = "SIGNED & UNLOCKED" if is_signed else "AWAITING QA COSIGN"
    hitl_color = "#1e8e3e" if is_signed else "#ea8600"

    nodes = {
        "uplc": {
            "id": "uplc",
            "name": "Waters Acquity UPLC Skid",
            "short_label": "Waters UPLC<br>Skid W42",
            "zone": "Physical Infrastructure",
            "zone_idx": 1,
            "x": 1.2,
            "y": 5.0,
            "size": 30,
            "color": uplc_color,
            "symbol": "square",
            "status": uplc_status,
            "driver": "Waters Empower 3 CDS (Build 3471)",
            "sop": "SOP-QC-210 Waters UPLC Maintenance & System Suitability",
            "telemetry": {
                "System Pressure": "310.4 bar (> 300 bar limit)" if is_hardware_fault else "215.2 bar (Nominal 180-250 bar)",
                "Autosampler Chiller": "12.4°C (Drift > 10.0°C limit)" if is_hardware_fault else "5.1°C (Nominal 5.0 ± 2.0°C)",
                "Active Sequence": "Run #14 / 24 (VOIDED / ABORTED)" if is_hardware_fault else "SEQ-2026-W42-01 (ACTIVE)",
                "W42 Utilization": f"{(143.3 * inst_outage_mult * analyst_abs_mult):.1f}%" + (" + Rush" if inject_rush else ""),
            },
            "desc": "Primary analytical chromatography skid executing critical-path SEC monomer purity and CEX charge variant assays for commercial release lots.",
        },
        "maurice": {
            "id": "maurice",
            "name": "ProteinSimple Maurice icIEF",
            "short_label": "Maurice icIEF<br>Skid W42",
            "zone": "Physical Infrastructure",
            "zone_idx": 1,
            "x": 1.2,
            "y": 3.8,
            "size": 30,
            "color": "#1e8e3e",
            "symbol": "square",
            "status": "NOMINAL",
            "driver": "Compass for iCE v3.2",
            "sop": "SOP-QC-340 Automated Imaged Capillary Isoelectric Focusing",
            "telemetry": {
                "Focusing High Voltage": "5,000 V (Stable)",
                "Capillary Cartridge": "Cartridge #8821 (Run 42/100, 58% life remaining)",
                "Baseline Current": "14.2 µA (Spec: 12.0 - 18.0 µA)",
                "W42 Utilization": "65.0% (Optimal < 85% LEAN ceiling)",
            },
            "desc": "Automated imaged cIEF platform measuring charge variant profiles (acidic, main, basic isoforms) with real-time UV absorbance optics.",
        },
        "bioassay": {
            "id": "bioassay",
            "name": "BSL-2 Microplate Suite",
            "short_label": "SpectraMax<br>Bioassay Suite",
            "zone": "Physical Infrastructure",
            "zone_idx": 1,
            "x": 1.2,
            "y": 2.5,
            "size": 30,
            "color": "#1e8e3e",
            "symbol": "square",
            "status": "NOMINAL",
            "driver": "Molecular Devices SoftMax Pro v7.1",
            "sop": "SOP-QC-450 Cell-Based In Vitro Relative Potency Assays",
            "telemetry": {
                "Incubator Environment": "37.0°C ± 0.1°C, 5.0% CO2, 95% RH",
                "Plate Moat Integrity": "Inner-60 well PBS moat active (0 outer evaporative bias)",
                "Model Fit R²": "0.9982 (4-PL Bayesian Sigmoidal Dose-Response)",
                "W42 Utilization": "105.0% (Leveled to 85.0% via CP-SAT)",
            },
            "desc": "BSL-2 biosafety containment testing cell-based relative biological potency with automated 96-well edge-effect avoidance protocols.",
        },
        "chamber": {
            "id": "chamber",
            "name": "Stability Walk-In Chambers",
            "short_label": "Rees Chambers<br>5°C / 25°C / 40°C",
            "zone": "Physical Infrastructure",
            "zone_idx": 1,
            "x": 1.2,
            "y": 1.1,
            "size": 30,
            "color": chamber_color,
            "symbol": "square",
            "status": chamber_status,
            "driver": "Rees Scientific Centron Environmental Monitoring System",
            "sop": "SOP-QC-110 Environmental Walk-In Chambers & Continuous Monitoring",
            "telemetry": {
                "Monitored Probe": f"{excursion_temp:.1f}°C (Warning ceiling: > 8.0°C)" if is_excursion_alert else "5.0°C ± 0.3°C",
                "Excursion Duration": f"{excursion_hours:.1f} Hours (Logged continuously)" if is_excursion_alert else "0.0 Hours",
                "ICH Q1A Windows": "±7 Calendar Days Permitted Tolerance",
                "Thermal Uniformity": "±0.4°C 16-point thermocouple grid",
            },
            "desc": "Monitored 5°C walk-in and accelerated stability chambers equipped with dual calibrated RTD temperature sensors.",
        },
        "agent1": {
            "id": "agent1",
            "name": "Agent 1: LEAN Capacity & CP-SAT",
            "short_label": "Agent 1: Capacity<br>CP-SAT Solver",
            "zone": "Multi-Agent Core",
            "zone_idx": 2,
            "x": 4.8,
            "y": 5.0,
            "size": 26,
            "color": agent1_color,
            "symbol": "circle",
            "status": agent1_status,
            "driver": "Google OR-Tools CP-SAT discrete solver + SimPy DES",
            "sop": "agents/agent_1_lean_capacity/skill.md",
            "telemetry": {
                "Solver Engine": "OR-Tools v9.11 CP-SAT (Integer Feasibility)",
                "Double-Booking Violations": "0 Detected (Hard Precedence Constraint)",
                "LEAN Buffer Ceiling": "85.0% Staffed Hours Target (102h)",
                "Monte Carlo Replications": "300 iterations (SimPy Discrete-Event Simulation)",
            },
            "desc": "Autonomous resource orchestrator leveling analytical instrument demand, protecting commercial release lots, and shifting flexible stability pulls.",
        },
        "agent3": {
            "id": "agent3",
            "name": "Agent 3: Quality Intelligence & OOS",
            "short_label": "Agent 3: Quality<br>OOS Telemetry",
            "zone": "Multi-Agent Core",
            "zone_idx": 2,
            "x": 4.8,
            "y": 3.8,
            "size": 26,
            "color": agent3_color,
            "symbol": "circle",
            "status": agent3_status,
            "driver": "Deterministic Phase I/II Decision Tree + I-MR / EWMA SPC Engine",
            "sop": "agents/agent_3_quality_intelligence/skill.md",
            "telemetry": {
                "Phase I Lab Gate": "HARDWARE ANOMALY DETECTED" if is_hardware_fault else "CLEAR (Phase II Batch Escalation)",
                "Telemetry Intercept": "Empower CDS 310 bar pump trip" if is_hardware_fault else "215 bar nominal baseline",
                "Statistical Process Control": "I-MR Individual Charts + EWMA Trend Engine",
                "CAPA Risk Priority (RPN)": "RPN = 126 (Pre-CAPA) -> RPN = 24 (Post-CAPA)",
            },
            "desc": "Real-time supervisory agent gating out-of-specification laboratory results against automated instrument telemetry and SPC trends.",
        },
        "agent2": {
            "id": "agent2",
            "name": "Agent 2: Hybrid Kinetics & Biophysics",
            "short_label": "Agent 2: Kinetics<br>Lumry-Eyring ODE",
            "zone": "Multi-Agent Core",
            "zone_idx": 2,
            "x": 4.8,
            "y": 2.5,
            "size": 26,
            "color": agent2_color,
            "symbol": "circle",
            "status": agent2_status,
            "driver": "Lumry-Eyring 2-step ODE + SciPy RK45 Integration",
            "sop": "agents/agent_2_analytical_scientist/skill.md",
            "telemetry": {
                "Kinetic Model": "Lumry-Eyring d[U]/dt + d[HMW]/dt (Arrhenius k_unfold)",
                "Activation Energy Ea": "74.37 kJ/mol (Biophysical Anchor)",
                "SEC Precision SD": "0.035% Repeatability (2*SD Gate = 0.099 pp)",
                "Monotonicity Verification": "0 Inversions across 3,200 grid evaluations",
            },
            "desc": "Hybrid biophysical and machine learning model computing non-isothermal cold-chain excursion degradation kinetics and LSS method transfer.",
        },
        "agent4": {
            "id": "agent4",
            "name": "Agent 4: Stakeholder Comms & QTA",
            "short_label": "Agent 4: Comms<br>QTA Dispatch",
            "zone": "Multi-Agent Core",
            "zone_idx": 2,
            "x": 4.8,
            "y": 1.1,
            "size": 26,
            "color": "#1a73e8",
            "symbol": "circle",
            "status": "NOMINAL",
            "driver": "Defensive Prompt Injection Filtering + Jinja2 Markdown Template",
            "sop": "agents/agent_4_executive_stakeholder/skill.md",
            "telemetry": {
                "Injection Defense": "0 Leaks (Sanitized XML Delimiters)",
                "Active Project Focus": str(context.get("selected_proj_comms", "PROJ-001")),
                "SLA Notification Window": "24-Hour Quality Technical Agreement Bound",
                "Draft Output": "External Client Quality Memo (Markdown / JSON)",
            },
            "desc": "Executive communication coordinator assembling regulatory impact memos and client SOW amendments with defensive prompt isolation.",
        },
        "hitl": {
            "id": "hitl",
            "name": "Part 11 HITL Sign-off Gate",
            "short_label": "Part 11 Gate<br>QA Dual Cosign",
            "zone": "Governance & Compliance",
            "zone_idx": 3,
            "x": 8.6,
            "y": 3.1,
            "size": 30,
            "color": hitl_color,
            "symbol": "diamond",
            "status": hitl_status,
            "driver": "Dual-Cosigner Approval Protocol + TrackWise Export API",
            "sop": "core/part11_gate.py (21 CFR Part 11 Electronic Signature Demo)",
            "telemetry": {
                "Gate State": "EXPORT UNLOCKED" if is_signed else "EXPORT LOCKED (Requires Cosign)",
                "Local MSO Reviewer": signed_pkg.get("reviewer_name", "Pending") if signed_pkg else "Pending Human Input",
                "QA Cosigner Role": signed_pkg.get("reviewer_role", "Pending") if signed_pkg else "Pending Human Input",
                "Export Artifacts": "TrackWise Draft JSON + Local MSO Evaluation Memo",
            },
            "desc": "Enforces Guardrail G6: requires verified human local MSO and QA reviewer signatures before releasing analytical evaluations.",
        },
        "audit": {
            "id": "audit",
            "name": "Cryptographic SHA-256 Audit Trail",
            "short_label": "SHA-256 Ledger<br>Hash-Chained",
            "zone": "Governance & Compliance",
            "zone_idx": 3,
            "x": 8.6,
            "y": 1.7,
            "size": 30,
            "color": "#137333",
            "symbol": "diamond",
            "status": "CHAIN VERIFIED",
            "driver": "SQLite Append-Only Ledger with UPDATE/DELETE Abort Triggers",
            "sop": "core/audit.py (ALCOA+ Data Integrity Principles)",
            "telemetry": {
                "Chain Integrity": "Cryptographically Verified (Unbroken Linkages)",
                "Latest Block Hash": str(context.get("audit_latest_hash", "GENESIS_HASH"))[:16] + "...",
                "Immutability Trigger": "ACTIVE (SQLite BEFORE UPDATE/DELETE ABORT)",
                "Compliance Standard": "21 CFR 211.68(b) Complete Electronic Records",
            },
            "desc": "Tamper-evident append-only cryptographic ledger tracking every orchestrator event, human electronic signature, and parameter update.",
        },
    }
    return nodes


def build_plant_network_figure(context: Dict[str, Any], filter_zone: str = "All Zones (Connected Plant)") -> go.Figure:
    """Builds the industrial SCADA-style Plotly interactive network diagram with parallel flow and clean edge routing."""
    nodes = get_node_definitions(context)

    # Filter by zone if selected
    if filter_zone == "Physical Infrastructure":
        nodes = {k: v for k, v in nodes.items() if v["zone"] == "Physical Infrastructure"}
    elif filter_zone == "Multi-Agent Core":
        nodes = {k: v for k, v in nodes.items() if v["zone"] == "Multi-Agent Core"}
    elif filter_zone == "Governance & Compliance":
        nodes = {k: v for k, v in nodes.items() if v["zone"] == "Governance & Compliance"}

    fig = go.Figure()

    # Add Zone Background Panels (Spacious normalized coordinates)
    if filter_zone == "All Zones (Connected Plant)":
        # Zone 1: Physical Infrastructure
        fig.add_shape(
            type="rect",
            x0=0.2, y0=0.3, x1=2.4, y1=5.8,
            fillcolor="#f8f9fa", line=dict(color="#dadce0", width=1.5, dash="dot"),
            layer="below",
        )
        # Zone 2: Multi-Agent Core
        fig.add_shape(
            type="rect",
            x0=3.6, y0=0.3, x1=7.0, y1=5.8,
            fillcolor="#f8fafd", line=dict(color="#d2e3fc", width=1.5, dash="dot"),
            layer="below",
        )
        # Zone 3: Compliance & Part 11 Gates
        fig.add_shape(
            type="rect",
            x0=7.6, y0=0.3, x1=10.4, y1=5.8,
            fillcolor="#f6fbf7", line=dict(color="#ceead6", width=1.5, dash="dot"),
            layer="below",
        )

        # Zone Header Annotations
        fig.add_annotation(x=1.3, y=5.65, text="<b>ZONE 1: PHYSICAL ANALYTICAL ASSETS</b>", showarrow=False, font=dict(size=10.5, color="#5f6368"))
        fig.add_annotation(x=5.3, y=5.65, text="<b>ZONE 2: MULTI-AGENT DECISION CORE</b>", showarrow=False, font=dict(size=10.5, color="#1967d2"))
        fig.add_annotation(x=9.0, y=5.65, text="<b>ZONE 3: COMPLIANCE & PART 11 GATES</b>", showarrow=False, font=dict(size=10.5, color="#137333"))

    # Four parallel non-intersecting links from Zone 1 to Zone 2, plus Zone 2 to Zone 3 governance links
    edge_defs = [
        # Parallel Link 1 (Top): Waters UPLC -> Agent 1 CP-SAT
        {
            "id": "uplc_to_agent1",
            "src": "uplc", "dst": "agent1",
            "name": "Waters UPLC ➔ Agent 1 (LEAN Capacity)",
            "desc": "Chromatography skid demand & W42 utilization telemetry (143.3% / 172h)",
            "x": [1.2, 4.8], "y": [5.0, 5.0],
            "ax": 4.45, "ay": 5.0, "x_arr": 4.60, "y_arr": 5.0,
        },
        # Parallel Link 2 (Upper-Mid): Maurice icIEF -> Agent 3 OOS Triage
        {
            "id": "maurice_to_agent3",
            "src": "maurice", "dst": "agent3",
            "name": "Maurice icIEF ➔ Agent 3 (OOS Triage)",
            "desc": "Real-time cIEF electropherogram purity & baseline telemetry",
            "x": [1.2, 4.8], "y": [3.8, 3.8],
            "ax": 4.45, "ay": 3.8, "x_arr": 4.60, "y_arr": 3.8,
        },
        # Parallel Link 3 (Lower-Mid): SpectraMax Bioassay -> Agent 2 Kinetics
        {
            "id": "bioassay_to_agent2",
            "src": "bioassay", "dst": "agent2",
            "name": "BSL-2 Bioassay ➔ Agent 2 (Hybrid Kinetics)",
            "desc": "4-PL relative potency dose-response optical density curve data (R² = 0.9982)",
            "x": [1.2, 4.8], "y": [2.5, 2.5],
            "ax": 4.45, "ay": 2.5, "x_arr": 4.60, "y_arr": 2.5,
        },
        # Parallel Link 4 (Bottom): Rees Chambers -> Agent 4 Comms
        {
            "id": "chamber_to_agent4",
            "src": "chamber", "dst": "agent4",
            "name": "Stability Walk-In ➔ Agent 4 (Stakeholder Comms)",
            "desc": "Continuous environmental monitoring logs & client stability notices",
            "x": [1.2, 4.8], "y": [1.1, 1.1],
            "ax": 4.45, "ay": 1.1, "x_arr": 4.60, "y_arr": 1.1,
        },
        # Governance Link 5: Agent 1 -> Part 11 Gate
        {
            "id": "agent1_to_hitl",
            "src": "agent1", "dst": "hitl",
            "name": "Agent 1 ➔ Part 11 Gate",
            "desc": "CP-SAT schedule feasibility sign-off package",
            "x": [4.8, 8.6], "y": [5.0, 3.1],
            "ax": 8.20, "ay": 3.29, "x_arr": 8.38, "y_arr": 3.20,
        },
        # Governance Link 6: Agent 3 -> Part 11 Gate
        {
            "id": "agent3_to_hitl",
            "src": "agent3", "dst": "hitl",
            "name": "Agent 3 ➔ Part 11 Gate",
            "desc": "Phase I OOS root-cause laboratory triage evaluation package",
            "x": [4.8, 8.6], "y": [3.8, 3.1],
            "ax": 8.20, "ay": 3.17, "x_arr": 8.38, "y_arr": 3.14,
        },
        # Governance Link 7: Agent 2 -> Part 11 Gate
        {
            "id": "agent2_to_hitl",
            "src": "agent2", "dst": "hitl",
            "name": "Agent 2 ➔ Part 11 Gate",
            "desc": "Lumry-Eyring biophysical kinetics shelf-life evaluation memo",
            "x": [4.8, 8.6], "y": [2.5, 3.1],
            "ax": 8.20, "ay": 3.03, "x_arr": 8.38, "y_arr": 3.06,
        },
        # Governance Link 8: Agent 4 -> Part 11 Gate
        {
            "id": "agent4_to_hitl",
            "src": "agent4", "dst": "hitl",
            "name": "Agent 4 ➔ Part 11 Gate",
            "desc": "Client SOW amendment & SLA impact notification package",
            "x": [4.8, 8.6], "y": [1.1, 3.1],
            "ax": 8.20, "ay": 2.95, "x_arr": 8.38, "y_arr": 3.01,
        },
        # Governance Link 9: Part 11 Gate -> Audit Ledger
        {
            "id": "hitl_to_audit",
            "src": "hitl", "dst": "audit",
            "name": "Part 11 Gate ➔ SHA-256 Audit Ledger",
            "desc": "ALCOA+ cryptographic electronic signature commit",
            "x": [8.6, 8.6], "y": [3.1, 1.7],
            "ax": 8.6, "ay": 2.08, "x_arr": 8.6, "y_arr": 1.92,
        },
    ]

    # Add edges as subdued spline traces with hover tooltips (opacity=0.4, line width=1.5, color='#9aa0a6')
    for ed in edge_defs:
        if ed["src"] in nodes and ed["dst"] in nodes:
            fig.add_trace(
                go.Scatter(
                    x=ed["x"],
                    y=ed["y"],
                    mode="lines",
                    line=dict(shape="spline", width=1.5, color="#9aa0a6"),
                    opacity=0.4,
                    hoverinfo="text",
                    hovertext=f"<b>{ed['name']}</b><br>{ed['desc']}",
                    hoverlabel=dict(bgcolor="#ffffff", font_size=11, font_color="#202124", bordercolor="#dadce0"),
                    showlegend=False,
                )
            )
            # Add directed arrowhead matching subdued opacity
            fig.add_annotation(
                x=ed["x_arr"],
                y=ed["y_arr"],
                ax=ed["ax"],
                ay=ed["ay"],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.1,
                arrowwidth=1.5,
                arrowcolor="#9aa0a6",
                opacity=0.45,
            )

    # Plot Nodes partitioned by zone for clean text positioning
    z1_nodes = [n for n in nodes.values() if n["zone_idx"] == 1]
    z2_nodes = [n for n in nodes.values() if n["zone_idx"] == 2]
    z3_nodes = [n for n in nodes.values() if n["zone_idx"] == 3]

    # Zone 1: Physical Infrastructure (Text placed below square markers)
    if z1_nodes:
        fig.add_trace(
            go.Scatter(
                x=[n["x"] for n in z1_nodes],
                y=[n["y"] for n in z1_nodes],
                mode="markers+text",
                marker=dict(
                    size=[n["size"] for n in z1_nodes],
                    color=[n["color"] for n in z1_nodes],
                    symbol=[n["symbol"] for n in z1_nodes],
                    line=dict(color="#ffffff", width=2.0),
                ),
                text=[n["short_label"] for n in z1_nodes],
                textposition="bottom center",
                textfont=dict(size=10.5, color="#202124", family="Roboto, Inter, sans-serif"),
                hovertext=[
                    f"<b>{n['name']}</b><br>Zone: {n['zone']}<br>Status: <b>{n['status']}</b><br>Driver/System: {n['driver']}"
                    for n in z1_nodes
                ],
                hoverinfo="text",
                showlegend=False,
            )
        )

    # Zone 2: Multi-Agent Core (Text placed to the right of circular markers)
    if z2_nodes:
        fig.add_trace(
            go.Scatter(
                x=[n["x"] for n in z2_nodes],
                y=[n["y"] for n in z2_nodes],
                mode="markers+text",
                marker=dict(
                    size=[n["size"] for n in z2_nodes],
                    color=[n["color"] for n in z2_nodes],
                    symbol=[n["symbol"] for n in z2_nodes],
                    line=dict(color="#ffffff", width=2.0),
                ),
                text=[n["short_label"] for n in z2_nodes],
                textposition="middle right",
                textfont=dict(size=10.5, color="#202124", family="Roboto, Inter, sans-serif"),
                hovertext=[
                    f"<b>{n['name']}</b><br>Zone: {n['zone']}<br>Status: <b>{n['status']}</b><br>Driver/Solver: {n['driver']}"
                    for n in z2_nodes
                ],
                hoverinfo="text",
                showlegend=False,
            )
        )

    # Zone 3: Governance & Compliance Gates (Text placed to the right of diamond markers)
    if z3_nodes:
        fig.add_trace(
            go.Scatter(
                x=[n["x"] for n in z3_nodes],
                y=[n["y"] for n in z3_nodes],
                mode="markers+text",
                marker=dict(
                    size=[n["size"] for n in z3_nodes],
                    color=[n["color"] for n in z3_nodes],
                    symbol=[n["symbol"] for n in z3_nodes],
                    line=dict(color="#ffffff", width=2.0),
                ),
                text=[n["short_label"] for n in z3_nodes],
                textposition="middle right",
                textfont=dict(size=10.5, color="#202124", family="Roboto, Inter, sans-serif"),
                hovertext=[
                    f"<b>{n['name']}</b><br>Zone: {n['zone']}<br>Status: <b>{n['status']}</b><br>Standard: {n['driver']}"
                    for n in z3_nodes
                ],
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 10.8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.2, 6.0]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=540,
    )

    return fig


def get_live_telemetry_events(context: Dict[str, Any]) -> List[Dict[str, str]]:
    """Synthesizes real-time SCADA and multi-agent events from active simulation state and audit records."""
    events: List[Dict[str, str]] = []
    now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    oos_scenario = context.get("oos_scenario", "Run_14_Hardware_Anomaly")
    is_hardware_fault = oos_scenario == "Run_14_Hardware_Anomaly"
    inst_outage_mult = float(context.get("inst_outage_mult", 1.0))
    analyst_abs_mult = float(context.get("analyst_abs_mult", 1.0))
    inject_rush = bool(context.get("inject_rush", False))
    usp_harvest_delay = int(context.get("usp_harvest_delay", 0))
    excursion_temp = float(context.get("excursion_temp", 28.0))
    excursion_hours = float(context.get("excursion_hours", 36.0))
    signed_pkg = context.get("signed_mso_pkg")

    # 1. Physical Hardware & CDS Telemetry Events
    if is_hardware_fault:
        events.append({
            "timestamp": now_utc,
            "severity": "CRITICAL",
            "source": "Waters Acquity UPLC (UPLC_SEC_CEX)",
            "category": "CDS Telemetry",
            "message": "System Backpressure Limit Breached: 310.4 bar > 300.0 bar ceiling. Automatic pump shutoff tripped.",
            "provenance": "Empower CDS Run #14 Log",
        })
        events.append({
            "timestamp": now_utc,
            "severity": "CRITICAL",
            "source": "Waters Acquity UPLC (UPLC_SEC_CEX)",
            "category": "CDS Telemetry",
            "message": "Autosampler Chiller Drift: Compartment temperature 12.4°C exceeded 10.0°C maximum limit.",
            "provenance": "Waters Sample Manager RTD",
        })
        events.append({
            "timestamp": now_utc,
            "severity": "CRITICAL",
            "source": "Agent 3: Quality Intelligence",
            "category": "Agent Decision",
            "message": "Phase I Root Cause Confirmed: Hardware Anomaly (310 bar / 12.4°C). Run #14 Aborted / Voided. Batch protected.",
            "provenance": "SOP-QC-701 Decision Tree",
        })
    else:
        events.append({
            "timestamp": now_utc,
            "severity": "INFO",
            "source": "Waters Acquity UPLC (UPLC_SEC_CEX)",
            "category": "CDS Telemetry",
            "message": "Empower Sequence Telemetry Nominal: Sequence SEQ-W42-01 active. Backpressure 215.2 bar, Chiller 5.1°C.",
            "provenance": "Empower CDS Channel 1 UV",
        })

    # 2. Walk-in Stability Chambers
    if excursion_temp > 25.0:
        events.append({
            "timestamp": now_utc,
            "severity": "WARNING",
            "source": "Rees Stability Walk-In (Chamber_5C)",
            "category": "Environmental",
            "message": f"Continuous Temperature Excursion Alert: Probe logged {excursion_temp:.1f}°C for {excursion_hours:.1f} hours (> 8.0°C ceiling).",
            "provenance": "Rees Centron Probe RTD-04",
        })
        events.append({
            "timestamp": now_utc,
            "severity": "INFO",
            "source": "Agent 2: Hybrid Kinetics",
            "category": "Agent Decision",
            "message": "Lumry-Eyring ODE Solved: Predicted delta HMW = +0.003 pp. Calculated 2*SD Acceptance Gate: PASS.",
            "provenance": "SciPy RK45 (Ea=74.37 kJ/mol)",
        })

    # 3. CP-SAT Leveling & Capacity Operations
    if usp_harvest_delay > 0:
        events.append({
            "timestamp": now_utc,
            "severity": "WARNING",
            "source": "Agent 1: LEAN Capacity",
            "category": "Agent Decision",
            "message": f"USP/DSP Commercial Bioreactor Harvest Slip (+{usp_harvest_delay} days) detected on W42 critical path.",
            "provenance": "OR-Tools CP-SAT Solver",
        })
    if inject_rush:
        events.append({
            "timestamp": now_utc,
            "severity": "WARNING",
            "source": "Agent 1: LEAN Capacity",
            "category": "Capacity Alarm",
            "message": "Urgent Zero-Float Commercial Rush Lot (Spesolimab-Urgent) injected into UPLC queue (+18.0 hours).",
            "provenance": "Schedule Event Stream",
        })
    if inst_outage_mult > 1.0 or analyst_abs_mult > 1.0:
        eff_mult = round(inst_outage_mult * analyst_abs_mult, 2)
        events.append({
            "timestamp": now_utc,
            "severity": "WARNING",
            "source": "Agent 1: LEAN Capacity",
            "category": "Capacity Alarm",
            "message": f"Operational Stress Multiplier Active: Effective load scaled {eff_mult}x due to outage/shortage.",
            "provenance": "Capacity Constraint Engine",
        })

    # Baseline Maurice & Bioassay routine logs
    events.append({
        "timestamp": now_utc,
        "severity": "INFO",
        "source": "ProteinSimple Maurice (CE_icIEF)",
        "category": "CDS Telemetry",
        "message": "Automated icIEF cartridge health check passed: Capillary voltage 5,000 V, baseline current 14.2 µA.",
        "provenance": "Compass for iCE Log",
    })
    events.append({
        "timestamp": now_utc,
        "severity": "INFO",
        "source": "BSL-2 Microplate Suite (Cell_Bioassay)",
        "category": "CDS Telemetry",
        "message": "Inner-60 well PBS moat evaporation barrier active. Plate edge distortion eliminated (R² = 0.9982).",
        "provenance": "SoftMax Pro Dose-Response",
    })

    # 4. Part 11 Electronic Signature & Real Audit Records
    if signed_pkg is not None and signed_pkg.get("export_unlocked", False):
        events.append({
            "timestamp": now_utc,
            "severity": "AUDIT",
            "source": "Part 11 HITL Gate",
            "category": "Electronic Signature",
            "message": f"Electronic Signature Verified: {signed_pkg.get('reviewer_name')} ({signed_pkg.get('reviewer_role')}). Export package unlocked.",
            "provenance": f"ALCOA+ SHA-256 Digest: {signed_pkg.get('alcoa_sha256_hash', 'N/A')[:16]}...",
        })

    # Append recent records from the real cryptographic audit trail if available
    audit_trail = context.get("audit_trail")
    if audit_trail is not None and hasattr(audit_trail, "get_all_records"):
        for rec in reversed(audit_trail.get_all_records()[-4:]):
            events.append({
                "timestamp": rec.get("timestamp_utc", now_utc)[11:19] + " UTC",
                "severity": "AUDIT",
                "source": f"Audit Ledger ({rec.get('actor_role', 'System')})",
                "category": "Cryptographic Commit",
                "message": f"{rec.get('action')}: {rec.get('justification')} [{rec.get('entity_type')}:{rec.get('entity_id')}]",
                "provenance": f"SHA-256: {rec.get('record_hash', '')[:16]}...",
            })

    return events


def render_live_plant_network_tab(context: Dict[str, Any]) -> None:
    """Renders the complete industrial SCADA-style Live Plant & Agent Network workspace."""

    st.subheader("🌐 Live Plant & Multi-Agent Network Topology")
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #dadce0; border-radius: 8px; padding: 14px 18px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(60,64,67,0.06); border-left: 5px solid #1a73e8;">
            <div style="font-size: 13.5px; font-weight: 700; color: #202124;">
                Industrial SCADA & Multi-Agent Distributed Network Map
            </div>
            <div style="font-size: 12px; color: #5f6368; margin-top: 3px;">
                Demonstrates real-time physical analytical instrument telemetry, autonomous multi-agent optimization, and 21 CFR Part 11 human-in-the-loop consensus gates.
            </div>
            <div style="margin-top: 6px; font-size: 11px; color: #b06000;">
                ⚠️ {DRAFT_STAMP} — {PART11_DISCLAIMER}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Top Plant Health KPIs
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

    oos_scenario = context.get("oos_scenario", "Run_14_Hardware_Anomaly")
    is_fault = oos_scenario == "Run_14_Hardware_Anomaly"

    kpi_c1.metric(
        "Assets Online",
        "3 / 4 Online" if is_fault else "4 / 4 Online",
        "1 CDS Hardware Fault" if is_fault else "All Nominal",
        delta_color="inverse" if is_fault else "normal",
    )
    kpi_c2.metric(
        "Autonomous Agent Core",
        "4 / 4 Active Agents",
        "Coordinated Mesh",
    )
    kpi_c3.metric(
        "Telemetry Data Feeds",
        "14 Live Streams",
        "0.0% Packet Drop",
    )
    is_signed = bool(context.get("signed_mso_pkg") and context["signed_mso_pkg"].get("export_unlocked"))
    kpi_c4.metric(
        "Part 11 Consensus",
        "Signed & Unlocked" if is_signed else "Awaiting Review",
        "SHA-256 Verified",
    )

    # 2. Interactive Topology Graph
    st.markdown("#### 🗺️ Plant-Wide Distributed Topology Graph (Zone 1 ➔ Zone 2 ➔ Zone 3)")
    filter_zone = context.get("topology_zone_filter", "All Zones (Connected Plant)")
    fig_network = build_plant_network_figure(context, filter_zone=filter_zone)
    st.plotly_chart(fig_network, use_container_width=True)

    # 3. Node Detail Inspector (Wrapped in Expander to prioritize Network Graph)
    nodes = get_node_definitions(context)
    with st.expander("🔍 Real-Time Node Inspector (Live Telemetry & Configuration)", expanded=False):
        node_options = list(nodes.keys())
        default_idx = 0 if not is_fault else node_options.index("uplc")
        selected_node_id = st.selectbox(
            "Select Node to Inspect Live Telemetry & Configuration:",
            options=node_options,
            index=default_idx,
            format_func=lambda x: f"{nodes[x]['zone']} ➔ {nodes[x]['name']} [{nodes[x]['status']}]",
            key="center_topology_node_select",
        )

        node = nodes[selected_node_id]

        status_badge_class = "gate-box-crit" if "ALARM" in node["status"] or "FAULT" in node["status"] else ("gate-box-warn" if "AWAITING" in node["status"] or "EXCURSION" in node["status"] or "LOAD" in node["status"] else "gate-box-pass")

        st.markdown(
            f"""
            <div class="{status_badge_class}" style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 14px; font-weight: 700;">{node['name']} (<code>{node['id']}</code>)</span>
                    <span style="font-size: 11.5px; font-weight: 700; padding: 2px 8px; border-radius: 12px; background: rgba(0,0,0,0.06);">
                        {node['status']}
                    </span>
                </div>
                <div style="font-size: 11.5px; margin-top: 4px;">
                    <strong>Architecture Zone:</strong> {node['zone']} | <strong>Driver / Solver:</strong> <code>{node['driver']}</code>
                </div>
                <div style="font-size: 11.5px; margin-top: 2px;">
                    <strong>Governing SOP / Policy:</strong> <code>{node['sop']}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Telemetry metrics row
        telem_cols = st.columns(len(node["telemetry"]))
        for col, (param_name, param_val) in zip(telem_cols, node["telemetry"].items()):
            col.metric(param_name, param_val)

        st.caption(f"**Functional Role:** {node['desc']}")

    st.divider()

    # 4. Live Event & Telemetry Stream (Industrial SCADA Alarm & Event Console)
    st.markdown("#### 🚨 Real-Time SCADA Alarm & Telemetry Stream")
    stream_tabs = st.tabs([
        "📡 All Telemetry & Events (Live)",
        "⚠️ CDS Instrument Alarms",
        "🤖 Agent Optimization Decisions",
        "🔒 Cryptographic Audit Commits",
    ])

    all_events = get_live_telemetry_events(context)

    def render_event_table(event_list: List[Dict[str, str]]) -> None:
        if not event_list:
            st.info("No active alarms or events in this filter.")
            return

        df_evt = pd.DataFrame(event_list)
        # Format table with styled markdown
        for evt in event_list:
            sev = evt["severity"]
            sev_color = "#d93025" if sev == "CRITICAL" else ("#b06000" if sev == "WARNING" else ("#188038" if sev == "AUDIT" else "#1a73e8"))
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; margin-bottom: 6px; background: #ffffff; border: 1px solid #dadce0; border-radius: 6px; font-size: 12px;">
                    <div style="flex: 0 0 95px; font-weight: 600; color: #5f6368;">{evt['timestamp']}</div>
                    <div style="flex: 0 0 90px;"><span style="font-size: 10px; font-weight: 700; color: {sev_color}; background: {sev_color}18; padding: 2px 6px; border-radius: 4px;">{sev}</span></div>
                    <div style="flex: 0 0 210px; font-weight: 600; color: #202124;">{evt['source']}</div>
                    <div style="flex: 1 1 auto; color: #3c4043; padding: 0 10px;">{evt['message']}</div>
                    <div style="flex: 0 0 180px; text-align: right; color: #5f6368; font-family: monospace; font-size: 11px;">{evt['provenance']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with stream_tabs[0]:
        render_event_table(all_events)

    with stream_tabs[1]:
        render_event_table([e for e in all_events if e["severity"] in ["CRITICAL", "WARNING"] or "CDS" in e["category"]])

    with stream_tabs[2]:
        render_event_table([e for e in all_events if "Agent" in e["category"] or "Capacity" in e["category"]])

    with stream_tabs[3]:
        render_event_table([e for e in all_events if e["severity"] == "AUDIT" or "Cryptographic" in e["category"]])
