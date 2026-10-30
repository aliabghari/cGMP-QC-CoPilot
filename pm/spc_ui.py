"""
pm/spc_ui.py

Streamlit UI view for Quality Intelligence, Statistical Process Control (SPC),
Root-Cause Knowledge Base, Proactive FMEA, and Reactive CAPA Management (WP3).
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.audit import AuditTrail
from core.stamps import DRAFT_STAMP
from models.spc.spc_engine import (
    calculate_imr_chart,
    calculate_xbar_r_chart,
    calculate_p_chart,
    calculate_c_chart,
    calculate_ewma_chart,
    calculate_cusum_chart,
    calculate_process_capability,
)
from models.spc.synthetic_spc import (
    generate_purity_series,
    generate_rft_series,
    generate_doc_errors_series,
    generate_analyst_subgroups,
)
from pm.knowledge_base import (
    get_all_root_causes,
    get_all_deviation_types,
    get_lessons_learned,
    search_lessons_learned,
    add_lesson_learned,
    ADVISORY_DISCLAIMER,
)
from pm.fmea_capa import (
    get_all_fmea_items,
    get_all_capa_records,
    transition_capa_status,
    sync_fmea_with_lessons_learned,
    CAPA_STATE_TRANSITIONS,
    CAPA_CLOSURE_DISCLAIMER,
)


def render_spc_and_quality_intelligence_tab(
    conn: sqlite3.Connection,
    audit_trail: Optional[AuditTrail] = None,
    current_user_name: str = "Lead QC Analyst",
    current_user_role: str = "Principal Specialist, QC PM",
) -> None:
    """Renders the Quality Intelligence, SPC & CAPA tab."""

    st.subheader("📈 Quality Intelligence, Statistical Process Control (SPC) & CAPA Management")
    st.markdown(
        f"""
        <div style="background-color: #f8fafc; border-left: 5px solid #2563eb; padding: 0.8rem 1rem; border-radius: 4px; margin-bottom: 1rem;">
            <strong>Quality Intelligence Decision Support:</strong>
            Statistical trending (Shewhart, EWMA, CUSUM), process capability metrics ($C_p, C_{{pk}}$),
            historical deviation precedents from the Lessons-Learned Knowledge Base, proactive FMEA failure mode mitigation,
            and guarded CAPA lifecycle tracking with ALCOA+ audit logging.<br>
            <span style="color: #b91c1c; font-weight: 600; font-size: 0.85rem;">{DRAFT_STAMP}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    qi_subtabs = st.tabs([
        "📊 1. Statistical Process Control (SPC)",
        "📐 2. Process Capability (Cp/Cpk)",
        "🔬 3. Detection Benchmark (Monte Carlo)",
        "📚 4. Lessons Learned Knowledge Base",
        "🛡️ 5. Proactive FMEA Risk Register",
        "🔄 6. Reactive CAPA State Machine",
    ])

    # =========================================================================
    # SUB-TAB 1: STATISTICAL PROCESS CONTROL
    # =========================================================================
    with qi_subtabs[0]:
        st.markdown("#### 📊 Quality Control Statistical Monitoring Charts")
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])

        with col_ctrl1:
            series_choice = st.selectbox(
                "Select Quality Control Series",
                [
                    "SEC-HPLC % Monomer Purity (Individual I-MR)",
                    "Right-First-Time Defect Proportion (p-chart)",
                    "Batch Record Documentation Errors (c-chart)",
                    "Bioassay Relative Potency Subgroups (X-bar/R)",
                    "SEC-HPLC Monomer EWMA (Small Shift Tracking)",
                    "SEC-HPLC Monomer CUSUM (Persistent Shift)",
                ],
            )

        with col_ctrl2:
            signal_choice = st.selectbox(
                "Planted Process Condition (Synthetic)",
                [
                    "In-Control Baseline (Common Cause Only)",
                    "Mean Shift (+1.0 sigma)",
                    "Systematic Drift (+0.05 sigma/step)",
                    "Variance Inflation (1.8x sigma)",
                ],
            )
            sig_map = {
                "In-Control Baseline (Common Cause Only)": "none",
                "Mean Shift (+1.0 sigma)": "mean_shift",
                "Systematic Drift (+0.05 sigma/step)": "drift",
                "Variance Inflation (1.8x sigma)": "variance_inflation",
            }
            signal_key = sig_map[signal_choice]

        with col_ctrl3:
            control_mode = st.radio(
                "Control Phase",
                ["Phase I (Baseline Calculation)", "Phase II (Frozen Historical Limits)"],
                horizontal=True,
            )
            is_phase2 = "Phase II" in control_mode

        st.markdown("---")

        # 1. I-MR
        if "Individual I-MR" in series_choice:
            data_dict = generate_purity_series(signal_type=signal_key, seed=42)
            vals = np.array(data_dict["values"])

            frozen = None
            if is_phase2:
                frozen = {
                    "center_line_x": 98.20,
                    "ucl_x": 98.60,
                    "lcl_x": 97.80,
                    "center_line_mr": 0.169,
                    "ucl_mr": 0.552,
                    "lcl_mr": 0.0,
                }

            res = calculate_imr_chart(vals, frozen_limits=frozen)

            fig_imr = go.Figure()
            x_pts = list(range(1, len(vals) + 1))
            fig_imr.add_trace(go.Scatter(x=x_pts, y=vals, mode="lines+markers", name="Observed % Purity", line=dict(color="#2563eb", width=2)))
            fig_imr.add_hline(y=res["center_line_x"], line_dash="dash", line_color="#10b981", annotation_text=f"CL: {res['center_line_x']:.2f}%")
            fig_imr.add_hline(y=res["ucl_x"], line_dash="dot", line_color="#ef4444", annotation_text=f"UCL: {res['ucl_x']:.2f}%")
            fig_imr.add_hline(y=res["lcl_x"], line_dash="dot", line_color="#ef4444", annotation_text=f"LCL: {res['lcl_x']:.2f}%")

            # Highlight violations
            v_x = [v["index"] + 1 for v in res["violations_x"]]
            v_y = [v["value"] for v in res["violations_x"]]
            if v_x:
                fig_imr.add_trace(go.Scatter(x=v_x, y=v_y, mode="markers", marker=dict(size=11, color="#dc2626", symbol="circle-open", line=dict(width=3)), name="Nelson Violation"))

            fig_imr.update_layout(
                title=f"Individuals (I-Chart) — {res['mode']}",
                xaxis_title="Sequential Injection Number",
                yaxis_title="% Monomer Purity",
                height=420,
            )
            st.plotly_chart(fig_imr, use_container_width=True)

            if res["violations_x"]:
                st.warning(f"⚠️ {len(res['violations_x'])} Rule Violation(s) Detected:")
                df_v = pd.DataFrame(res["violations_x"])[["index", "rule", "name", "value"]]
                df_v["index"] += 1
                df_v.columns = ["Injection #", "Rule", "Diagnostic Description", "Observed Value"]
                st.dataframe(df_v, use_container_width=True)
            else:
                st.success("✅ Process in statistical control. No Nelson rule violations detected.")

        # 2. p-chart
        elif "p-chart" in series_choice:
            data_dict = generate_rft_series(signal_type=signal_key, seed=43)
            defects = np.array(data_dict["defect_counts"])
            sizes = np.array(data_dict["sample_sizes"])

            frozen_p = 0.08 if is_phase2 else None
            res = calculate_p_chart(defects, sizes, frozen_p=frozen_p)

            fig_p = go.Figure()
            p_pts = list(range(1, len(defects) + 1))
            fig_p.add_trace(go.Scatter(x=p_pts, y=res["proportions"], mode="lines+markers", name="Observed Non-RFT Rate", line=dict(color="#8b5cf6", width=2)))
            fig_p.add_hline(y=res["center_line_p"], line_dash="dash", line_color="#10b981", annotation_text=f"p_bar: {res['center_line_p']:.3f}")
            fig_p.add_trace(go.Scatter(x=p_pts, y=res["ucl_per_sample"], mode="lines", name="UCL (Variable n)", line=dict(color="#ef4444", dash="dot")))
            fig_p.add_trace(go.Scatter(x=p_pts, y=res["lcl_per_sample"], mode="lines", name="LCL (Variable n)", line=dict(color="#ef4444", dash="dot")))

            fig_p.update_layout(
                title=f"p-Chart (Non-RFT Proportion with Variable Inspection Sizes) — {res['mode']}",
                xaxis_title="Inspection Batch Index",
                yaxis_title="Non-RFT Proportion",
                height=420,
            )
            st.plotly_chart(fig_p, use_container_width=True)

            if res["violations"]:
                st.warning(f"⚠️ {len(res['violations'])} Out-of-Control Points Detected on p-chart:")
                st.dataframe(pd.DataFrame(res["violations"]), use_container_width=True)

        # 3. c-chart
        elif "c-chart" in series_choice:
            data_dict = generate_doc_errors_series(signal_type=signal_key, seed=44)
            counts = np.array(data_dict["counts"])

            frozen_c = 1.8 if is_phase2 else None
            res = calculate_c_chart(counts, frozen_c=frozen_c)

            fig_c = go.Figure()
            c_pts = list(range(1, len(counts) + 1))
            fig_c.add_trace(go.Bar(x=c_pts, y=counts, name="Error Count", marker_color="#f59e0b"))
            fig_c.add_hline(y=res["center_line_c"], line_dash="dash", line_color="#10b981", annotation_text=f"c_bar: {res['center_line_c']:.2f}")
            fig_c.add_hline(y=res["ucl"], line_dash="dot", line_color="#ef4444", annotation_text=f"UCL: {res['ucl']:.2f}")
            fig_c.add_hline(y=res["lcl"], line_dash="dot", line_color="#ef4444", annotation_text=f"LCL: {res['lcl']:.2f}")

            fig_c.update_layout(
                title=f"c-Chart (Documentation Errors per Batch Record) — {res['mode']}",
                xaxis_title="Batch Record Review Index",
                yaxis_title="Count of Errors",
                height=420,
            )
            st.plotly_chart(fig_c, use_container_width=True)

            if res["violations"]:
                st.warning(f"⚠️ {len(res['violations'])} Points Breaching 3-Sigma Limits on c-chart:")
                st.dataframe(pd.DataFrame(res["violations"]), use_container_width=True)

        # 4. X-bar / R
        elif "X-bar/R" in series_choice:
            data_dict = generate_analyst_subgroups(signal_type=signal_key, seed=45)
            subs = [np.array(s) for s in data_dict["subgroups"]]

            frozen_xbar = None
            if is_phase2:
                frozen_xbar = {
                    "center_line_x": 100.0,
                    "ucl_x": 102.5,
                    "lcl_x": 97.5,
                    "center_line_r": 3.5,
                    "ucl_r": 8.0,
                    "lcl_r": 0.0,
                }

            res = calculate_xbar_r_chart(subs, frozen_limits=frozen_xbar)

            fig_xbar = go.Figure()
            s_pts = list(range(1, len(subs) + 1))
            fig_xbar.add_trace(go.Scatter(x=s_pts, y=res["means"], mode="lines+markers", name="Subgroup Mean", line=dict(color="#0284c7", width=2)))
            fig_xbar.add_hline(y=res["center_line_x"], line_dash="dash", line_color="#10b981", annotation_text=f"X-bar-bar: {res['center_line_x']:.2f}%")
            fig_xbar.add_hline(y=res["ucl_x"], line_dash="dot", line_color="#ef4444", annotation_text=f"UCL: {res['ucl_x']:.2f}%")
            fig_xbar.add_hline(y=res["lcl_x"], line_dash="dot", line_color="#ef4444", annotation_text=f"LCL: {res['lcl_x']:.2f}%")

            fig_xbar.update_layout(
                title=f"X-bar Chart (Bioassay Potency Subgroups n={res['subgroup_size']}) — {res['mode']}",
                xaxis_title="Subgroup Run Index",
                yaxis_title="Mean % Potency",
                height=420,
            )
            st.plotly_chart(fig_xbar, use_container_width=True)

        # 5. EWMA
        elif "EWMA" in series_choice:
            data_dict = generate_purity_series(signal_type=signal_key, seed=42)
            vals = np.array(data_dict["values"])
            res = calculate_ewma_chart(vals, lambda_=0.20, L=3.0, target=98.20, sigma=0.15)

            fig_ewma = go.Figure()
            pts = list(range(1, len(vals) + 1))
            fig_ewma.add_trace(go.Scatter(x=pts, y=vals, mode="markers", name="Raw Observations", marker=dict(color="#94a3b8", size=5)))
            fig_ewma.add_trace(go.Scatter(x=pts, y=res["ewma_values"], mode="lines+markers", name="EWMA Statistic (λ=0.2)", line=dict(color="#7c3aed", width=2)))
            fig_ewma.add_hline(y=res["target"], line_dash="dash", line_color="#10b981", annotation_text=f"Target: {res['target']:.2f}%")
            fig_ewma.add_trace(go.Scatter(x=pts, y=res["ucl"], mode="lines", name="UCL (Dynamic)", line=dict(color="#ef4444", dash="dot")))
            fig_ewma.add_trace(go.Scatter(x=pts, y=res["lcl"], mode="lines", name="LCL (Dynamic)", line=dict(color="#ef4444", dash="dot")))

            fig_ewma.update_layout(
                title="EWMA Control Chart (Optimized for Small 0.5-1.5σ Shifts)",
                xaxis_title="Sequential Observation",
                yaxis_title="% Monomer Purity",
                height=420,
            )
            st.plotly_chart(fig_ewma, use_container_width=True)

            if res["violations"]:
                st.warning(f"⚠️ {len(res['violations'])} EWMA Shift Alarms Detected:")
                st.dataframe(pd.DataFrame(res["violations"]), use_container_width=True)

        # 6. CUSUM
        else:
            data_dict = generate_purity_series(signal_type=signal_key, seed=42)
            vals = np.array(data_dict["values"])
            res = calculate_cusum_chart(vals, target=98.20, sigma=0.15, k_sig=0.5, h_sig=4.0)

            fig_cusum = go.Figure()
            pts = list(range(1, len(vals) + 1))
            fig_cusum.add_trace(go.Scatter(x=pts, y=res["c_plus"], mode="lines+markers", name="C+ (Upper Cumulative Sum)", line=dict(color="#dc2626", width=2)))
            fig_cusum.add_trace(go.Scatter(x=pts, y=res["c_minus"], mode="lines+markers", name="C- (Lower Cumulative Sum)", line=dict(color="#2563eb", width=2)))
            fig_cusum.add_hline(y=res["h_threshold"], line_dash="dot", line_color="#b91c1c", annotation_text=f"Decision Threshold h: {res['h_threshold']:.3f}")

            fig_cusum.update_layout(
                title="Tabular CUSUM Control Chart (Decision Threshold h = 4.0σ, Reference k = 0.5σ)",
                xaxis_title="Sequential Observation",
                yaxis_title="Cumulative Excursion Statistic",
                height=420,
            )
            st.plotly_chart(fig_cusum, use_container_width=True)

            if res["violations"]:
                st.warning(f"⚠️ {len(res['violations'])} CUSUM Threshold Alarms Triggered:")
                st.dataframe(pd.DataFrame(res["violations"]), use_container_width=True)

        # Re-baseline form
        with st.expander("✍️ ALCOA+ Limit Freeze & Re-Baselining Action"):
            st.markdown("Propose updating frozen Phase II control limits based on approved process changes or annual product quality review (APQR).")
            with st.form("form_rebaseline"):
                rb_target = st.number_input("Proposed Center Line", value=98.20, format="%.3f")
                rb_ucl = st.number_input("Proposed Upper Control Limit (UCL)", value=98.65, format="%.3f")
                rb_lcl = st.number_input("Proposed Lower Control Limit (LCL)", value=97.75, format="%.3f")
                rb_just = st.text_input("Engineering / QA Justification (Mandatory)", value="Annual re-qualification after column stationary phase lot transition.")
                rb_sub = st.form_submit_button("Record Re-Baselining Event in Audit Trail")
                if rb_sub:
                    if audit_trail and rb_just.strip():
                        audit_trail.append_event(
                            actor_name=current_user_name,
                            actor_role=current_user_role,
                            action="REBASELINE_SPC_LIMITS",
                            justification=rb_just.strip(),
                            entity_type="SPC_CONTROL_CHART",
                            entity_id=series_choice[:15],
                            payload={"target": rb_target, "ucl": rb_ucl, "lcl": rb_lcl},
                        )
                        st.success("✅ Re-baselining event committed to append-only audit trail.")
                    else:
                        st.error("Please supply a valid justification.")

    # =========================================================================
    # SUB-TAB 2: PROCESS CAPABILITY
    # =========================================================================
    with qi_subtabs[1]:
        st.markdown("#### 📐 Process Capability Analysis ($C_p / C_{pk}$)")
        st.markdown("Evaluates whether analytical method or manufacturing batch process meets specification tolerances.")

        col_cap1, col_cap2, col_cap3 = st.columns(3)
        with col_cap1:
            lsl_val = st.number_input("Lower Specification Limit (LSL)", value=97.00, format="%.2f")
        with col_cap2:
            usl_val = st.number_input("Upper Specification Limit (USL)", value=100.00, format="%.2f")
        with col_cap3:
            cap_sample_size = st.slider("Evaluation Sample Size (n)", min_value=15, max_value=80, value=35)

        # Generate sample
        purity_sample = generate_purity_series(n_points=cap_sample_size, seed=42)["values"]
        cap_res = calculate_process_capability(purity_sample, lsl=lsl_val, usl=usl_val)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sample Mean", f"{cap_res['mean']:.3f}%")
        m2.metric("Sample Std Dev", f"{cap_res['std_dev']:.4f}%")
        m3.metric("Cp Potential", f"{cap_res['cp']:.2f}" if cap_res['cp'] else "N/A")
        m4.metric("Cpk Index", f"{cap_res['cpk']:.2f}", help=f"95% CI: {cap_res['cpk_confidence_interval_95']}")

        st.info(f"**Cpk 95% Confidence Interval (Chou et al.):** [{cap_res['cpk_confidence_interval_95'][0]:.2f}, {cap_res['cpk_confidence_interval_95'][1]:.2f}]")

        if cap_res["warnings"]:
            for w in cap_res["warnings"]:
                st.warning(f"⚠️ {w}")

        # Histogram with specs
        fig_hist = px.histogram(
            x=purity_sample,
            nbins=12,
            title=f"Sample Distribution (n={cap_sample_size}) vs Specifications",
            labels={"x": "% Monomer Purity", "y": "Frequency"},
            color_discrete_sequence=["#3b82f6"],
        )
        fig_hist.add_vline(x=lsl_val, line_dash="dash", line_color="#ef4444", annotation_text=f"LSL: {lsl_val}")
        fig_hist.add_vline(x=usl_val, line_dash="dash", line_color="#ef4444", annotation_text=f"USL: {usl_val}")
        fig_hist.add_vline(x=cap_res['mean'], line_dash="solid", line_color="#10b981", annotation_text=f"Mean: {cap_res['mean']:.2f}")
        fig_hist.update_layout(height=380)
        st.plotly_chart(fig_hist, use_container_width=True)

    # =========================================================================
    # SUB-TAB 3: DETECTION BENCHMARK (MONTE CARLO)
    # =========================================================================
    with qi_subtabs[2]:
        st.markdown("#### 🔬 SPC Detection Performance Comparison (Monte Carlo Benchmark)")
        st.markdown("Empirical comparison of Shewhart (Rule 1), EWMA (λ=0.20, L=3.00), and Tabular CUSUM (k=0.5σ, h=4.0σ) across 400 replications.")

        bench_path = Path("results/spc_performance.json")
        if bench_path.exists():
            with open(bench_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)

            rows_bench = []
            for k, v in bench_data["results_by_shift"].items():
                d = v["shift_delta"]
                sh = v["shewhart"]
                ew = v["ewma"]
                cu = v["cusum"]
                rows_bench.append({
                    "Shift Delta (σ)": f"+{d:.1f}σ" if d > 0 else "0.0σ (In-Control)",
                    "Shewhart Detection Rate": f"{sh['detection_rate']*100:.1f}%",
                    "Shewhart Mean RL": f"{sh['mean_run_length']:.1f}",
                    "EWMA Detection Rate": f"{ew['detection_rate']*100:.1f}%",
                    "EWMA Mean RL": f"{ew['mean_run_length']:.1f}",
                    "CUSUM Detection Rate": f"{cu['detection_rate']*100:.1f}%",
                    "CUSUM Mean RL": f"{cu['mean_run_length']:.1f}",
                })

            st.dataframe(pd.DataFrame(rows_bench), use_container_width=True)

            st.markdown("##### 🔍 Methodological Insights & Limitations")
            for obs in bench_data.get("honest_methodological_observations", []):
                st.markdown(f"- {obs}")
        else:
            st.warning("Performance benchmark results file not found. Run `python -m models.spc.performance` to generate.")

    # =========================================================================
    # SUB-TAB 4: LESSONS LEARNED KNOWLEDGE BASE
    # =========================================================================
    with qi_subtabs[3]:
        st.markdown("#### 📚 Historical Deviation Precedents & Lessons-Learned Knowledge Base")
        st.markdown("Search past laboratory (Phase I) and manufacturing (Phase II) investigations for cited precedent.")

        search_q = st.text_input("🔍 Search Historical Deviation Knowledge Base", value="column backpressure fouling bubble")
        search_res = search_lessons_learned(conn, query=search_q, top_k=5)

        st.caption(f"Found {search_res['results_count']} precedent match(es) for '{search_q}'. *{ADVISORY_DISCLAIMER}*")

        if search_res["matches"]:
            for m in search_res["matches"]:
                with st.expander(f"📁 [{m['lesson_id']}] {m['method_name']} — {m['root_cause_name']} (Similarity: {m['similarity_score']:.2f})"):
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.markdown(f"**Date:** {m['date']} | **Project:** `{m['project_id']}`")
                        st.markdown(f"**Tier:** {m['investigation_tier']}")
                        st.markdown(f"**Root Cause Category:** {m['root_cause_category']}")
                    with col_m2:
                        st.markdown(f"**Investigator / Owner:** {m['owner']}")
                        st.markdown(f"*{m['citation_note']}*")

                    st.markdown(f"**What Happened:** {m['what_happened']}")
                    st.markdown(f"**What Learned:** {m['what_learned']}")
                    st.markdown(f"**Preventive Action (CAPA):** `{m['preventive_action']}`")

        # Category Breakdown
        st.markdown("---")
        st.markdown("##### 📊 Historical Knowledge Base Distribution")
        all_lessons = get_lessons_learned(conn)
        if all_lessons:
            df_ll = pd.DataFrame(all_lessons)
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fig_cat = px.pie(df_ll, names="root_cause_category", title="Deviations by Root Cause Category", hole=0.4)
                fig_cat.update_layout(height=320)
                st.plotly_chart(fig_cat, use_container_width=True)
            with col_p2:
                fig_tier = px.bar(df_ll, x="investigation_tier", color="investigation_tier", title="Phase I (Lab) vs Phase II (Mfg) Investigations")
                fig_tier.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_tier, use_container_width=True)

        # Form to add new lesson
        with st.expander("➕ Record New Lesson Learned (ALCOA+ Audit Logged)"):
            with st.form("form_add_lesson"):
                nl_proj = st.selectbox("Project ID", ["PRJ-MAB-001", "PRJ-MAB-002", "PRJ-FUS-003", "PRJ-BSPEC-004", "PRJ-ADC-005"])
                nl_method = st.text_input("Analytical Method", value="SEC-HPLC Monomer Purity")
                nl_tier = st.selectbox("Investigation Tier", ["Phase I (Laboratory Investigation)", "Phase II (Cross-Functional / Manufacturing)"])
                nl_what = st.text_area("What Happened (Observation)", value="Autosampler needle blocked during sequence injection.")
                nl_learned = st.text_area("What Was Learned", value="Sample buffer salt crystallized on unwashed needle tip.")
                nl_capa = st.text_area("Preventive Action (CAPA)", value="Instituted needle wash flush step before and after each vial entry.")
                nl_just = st.text_input("Audit Justification", value="Documenting resolved investigation from QC run #44.")
                nl_submit = st.form_submit_button("Save Lesson to Knowledge Base")
                if nl_submit:
                    if nl_what.strip() and nl_just.strip():
                        lid = add_lesson_learned(
                            conn=conn,
                            lesson={
                                "project_id": nl_proj,
                                "method_name": nl_method,
                                "investigation_tier": nl_tier,
                                "what_happened": nl_what.strip(),
                                "what_learned": nl_learned.strip(),
                                "preventive_action": nl_capa.strip(),
                            },
                            audit_trail=audit_trail,
                            actor_name=current_user_name,
                            actor_role=current_user_role,
                            justification=nl_just.strip(),
                        )
                        st.success(f"✅ Successfully saved lesson {lid} to database and audit trail.")
                    else:
                        st.error("Please fill in required fields.")

    # =========================================================================
    # SUB-TAB 5: PROACTIVE FMEA RISK REGISTER
    # =========================================================================
    with qi_subtabs[4]:
        st.markdown("#### 🛡️ Failure Mode and Effects Analysis (FMEA) — Biologics QC Workflow")
        st.markdown("Proactive risk evaluation of analytical operations: Severity (S), Occurrence (O), Detection (D), Risk Priority Number (RPN).")

        fmea_items = get_all_fmea_items(conn)
        if fmea_items:
            df_fmea = pd.DataFrame(fmea_items)[
                [
                    "id", "process_step", "failure_mode", "severity", "occurrence", "detection",
                    "rpn", "action_priority", "post_rpn", "rpn_reduction_pct", "lessons_learned_count"
                ]
            ]
            df_fmea.columns = [
                "ID", "Process Step", "Failure Mode", "S", "O", "D",
                "Pre-RPN", "Priority", "Post-RPN", "Risk Reduction %", "Linked Lessons"
            ]
            st.dataframe(df_fmea, use_container_width=True)

            with st.expander("🔍 FMEA Rating Sensitivity Analysis (±1 Rating Perturbation)"):
                st.markdown("Evaluates which rating parameter has the largest swing on RPN for each failure mode.")
                sens_data = []
                for it in fmea_items:
                    sens = it["sensitivity"]
                    sens_data.append({
                        "ID": it["id"],
                        "Failure Mode": it["failure_mode"][:40] + "...",
                        "Nominal RPN": sens["nominal_rpn"],
                        "Most Sensitive Parameter": sens["most_sensitive_parameter"].upper(),
                        "Severity Swing": sens["parameter_sensitivities"]["severity"]["swing"],
                        "Occurrence Swing": sens["parameter_sensitivities"]["occurrence"]["swing"],
                        "Detection Swing": sens["parameter_sensitivities"]["detection"]["swing"],
                    })
                st.dataframe(pd.DataFrame(sens_data), use_container_width=True)

    # =========================================================================
    # SUB-TAB 6: REACTIVE CAPA STATE MACHINE
    # =========================================================================
    with qi_subtabs[5]:
        st.markdown("#### 🔄 Corrective and Preventive Action (CAPA) Lifecycle Tracker")
        st.markdown("State machine lifecycle governance: `OPEN` ➔ `IN_PROGRESS` ➔ `IMPLEMENTED` ➔ `EFFECTIVENESS_CHECK` ➔ `CLOSED`.")

        capas = get_all_capa_records(conn)

        # Overdue Radar
        overdue_capas = [c for c in capas if c.get("is_overdue")]
        if overdue_capas:
            st.error(f"🚨 **Overdue CAPA Radar:** {len(overdue_capas)} action(s) past target milestone!")
            for ov in overdue_capas:
                with st.expander(f"⚠️ [{ov['id']}] {ov['title']} (Due: {ov['due_date']} — {abs(ov['days_until_due'])} days overdue)"):
                    st.markdown(f"**Owner:** `{ov['owner']}` | **Current Status:** `{ov['status']}`")
                    st.markdown(f"**Corrective Action:** {ov['corrective_action']}")
                    st.markdown(f"**Preventive Action:** {ov['preventive_action']}")
                    if ov.get("reminder_draft"):
                        st.text_area("Draft Automated Escalation Reminder", value=ov["reminder_draft"], height=130)

        # CAPA Table
        if capas:
            df_c = pd.DataFrame(capas)[
                ["id", "source_type", "title", "owner", "due_date", "status", "effectiveness_due_date", "closed_by"]
            ]
            df_c.columns = ["CAPA ID", "Source", "Title", "Owner", "Due Date", "Status", "Effectiveness Due", "Closed By"]
            st.dataframe(df_c, use_container_width=True)

        # Transition Form
        with st.expander("✍️ Execute Guarded CAPA State Transition (HITL Sign-Off)"):
            st.markdown("Enforces permitted state machine transitions and requires verified effectiveness criteria before closure.")
            with st.form("form_transition_capa"):
                sel_capa = st.selectbox("Select CAPA ID", [c["id"] for c in capas])
                curr_capa = next((c for c in capas if c["id"] == sel_capa), None)
                curr_status = curr_capa["status"] if curr_capa else "OPEN"
                allowed_next = CAPA_STATE_TRANSITIONS.get(curr_status, [])

                st.info(f"Current Status: **{curr_status}** | Permitted Next States: **{allowed_next}**")

                if allowed_next:
                    target_st = st.selectbox("Target Next Status", allowed_next)
                else:
                    target_st = curr_status
                    st.caption("CAPA is in a terminal state (CLOSED). No further transitions permitted.")

                eff_input = st.text_area("Effectiveness Check Result (Mandatory for Closure)", value=curr_capa.get("effectiveness_result") or "" if curr_capa else "")
                trans_just = st.text_input("Reviewer Sign-Off Justification (Mandatory)", value="Action verified and audited against acceptance criteria.")
                trans_submit = st.form_submit_button("Commit CAPA Transition")

                if trans_submit:
                    if target_st in allowed_next:
                        try:
                            res = transition_capa_status(
                                conn=conn,
                                capa_id=sel_capa,
                                target_status=target_st,
                                audit_trail=audit_trail,
                                actor_name=current_user_name,
                                actor_role=current_user_role,
                                justification=trans_just.strip(),
                                effectiveness_result=eff_input.strip() if eff_input else None,
                            )
                            st.success(f"✅ CAPA {sel_capa} transitioned to {target_st}. ALCOA+ audit record generated.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Transition rejected: {e}")
                    else:
                        st.error(f"Cannot transition from {curr_status} to {target_st}.")
