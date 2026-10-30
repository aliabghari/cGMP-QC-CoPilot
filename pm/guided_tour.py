"""
pm/guided_tour.py

Interactive 5-Minute Guided Tour Engine for cGMP-QC-CoPilot.
Provides a step-by-step product walkthrough visually highlighting the 3-column
Google AI Studio layout (Left Nav Rail, Center PINN Canvas, Right Inspector, and Bottom Dock)
with synchronized audio narration cues and modal navigation controls.

Strictly adheres to Guardrail G2 draft standards and 21 CFR Part 11 guidance.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components

from core.stamps import DRAFT_STAMP


TOUR_STEPS: List[Dict[str, Any]] = [
    {
        "step": 1,
        "title": "1. Executive Control & Industrial SCADA Rail",
        "target_id": "left-nav-rail",
        "zone_name": "Left Navigation Rail (~240px)",
        "summary": "Industrial SCADA-style multi-agent orchestration balancing commercial pipeline releases against binding CDMO SLAs.",
        "details": [
            "4 Specialized Agents: Agent 1 (LEAN Capacity), Agent 2 (Analytical Scientist), Agent 3 (OOS Risk/MSO), and Agent 4 (QTA Comms).",
            "Dual-Stream Facility: Captive pipeline lots + CDMO client contract campaigns.",
            "Governance: 21 CFR 211.68(b) automated equipment validation and ALCOA+ traceability.",
        ],
        "narration": "Welcome to cGMP-QC-CoPilot. On the left navigation rail, our four autonomous specialist agents coordinate site capacity, chromatography transfer, and regulatory triage across the facility network.",
        "badge": "Agent Orchestration Core",
    },
    {
        "step": 2,
        "title": "2. Physics-Informed Kinetics & Mass Conservation",
        "target_id": "pinn-schematic-container",
        "zone_name": "Center Stage PINN Schematic",
        "summary": "Coupled deterministic Lumry-Eyring 2-step reversible/irreversible thermal degradation paired with a 25-member Bayesian ML ensemble.",
        "details": [
            "Branch A: Deterministic mass-action ODEs (N ⇌ U → Agg) solved via SciPy RK45 in < 0.01s.",
            "Physical Invariants: 0/3,200 negative rate violations; 0/2,400 temperature inversions; exact mass conservation (< 1e-15).",
            "Branch B: 25-member neural residual ensemble providing 95% conformal prediction intervals.",
        ],
        "narration": "In the center stage, the hybrid kinetics architecture pairs deterministic first-principles Lumry-Eyring ODEs with a 25-member Bayesian residual ensemble, ensuring exact mass conservation and physical validity.",
        "badge": "Ea = 74.37 kJ/mol Anchor",
    },
    {
        "step": 3,
        "title": "3. Multi-Parametric Biophysical Inspector",
        "target_id": "right-inspector-rail",
        "zone_name": "Right Inspector Rail (Run Settings)",
        "summary": "Reactive biopharmaceutical parameter controls modulating thermodynamic activation barriers and hydrodynamic delays.",
        "details": [
            "Group A (Stability): Excursion peak temp (5–45°C), duration (0–72h), surfactant PS20 enthalpy depletion, and peroxide oxidation entropy.",
            "Group B (Chromatography): Acquity UPLC system gradient dwell volume (V_D), computing dwell delay t_D = V_D / F.",
            "Group C (Bioassay): 96-well perimeter edge evaporation bias and cell passage phenotypic drift.",
        ],
        "narration": "On the right inspector rail, eleven biopharmaceutical sliders dynamically modulate activation enthalpy and entropy barriers, recalculating trajectories in under twenty milliseconds.",
        "badge": "11 Coupled Variables",
    },
    {
        "step": 4,
        "title": "4. Automated Phase I OOS Telemetry Triage",
        "target_id": "trajectory-canvas",
        "zone_name": "Center Stage Trajectory & Alarm Gate",
        "summary": "Autonomous evaluation against the 0.099 pp analytical repeatability gate and 2.00% specification ceiling.",
        "details": [
            "Analytical Precision Gate: 2√2·σ_SEC ≈ 0.099 pp for intermediate precision SD = 0.035%.",
            "Automated Invalidation: Empirically links CDS hardware anomalies (310 bar pressure spike, 17.8°C chiller drift) to abort lab errors.",
            "Defensive cGMP: Prevents illegal testing-into-compliance under FDA 2006 OOS Phase I/II standards.",
        ],
        "narration": "When thermal excursion aggregates breach the zero point zero nine nine percentage point analytical gate, Agent 3 initiates Phase I laboratory triage and parses CDS hardware logs to prevent illegal testing into compliance.",
        "badge": "FDA 2006 Phase I Gate",
    },
    {
        "step": 5,
        "title": "5. 21 CFR Part 11 Audit Ledger & Cosign Gate",
        "target_id": "audit-ledger-container",
        "zone_name": "Center Stage Audit Commitment Footer",
        "summary": "Immutable cryptographic SHA-256 tamper-evident hash chaining across every parameter shift.",
        "details": [
            "Cryptographic Provenance: Every state change generates a 64-character SHA-256 digest bound to previous blocks.",
            "Human-in-the-Loop Gate: Dual QA sign-off required prior to production disposition or eCTD 3.2.P.8 filing.",
            "Draft Protection: DRAFT_STAMP enforces zero unverified regulatory release claims.",
        ],
        "narration": "Finally, the Part 11 audit ledger generates immutable SHA-256 digests for every run, requiring dual Human-in-the-Loop quality assurance approval before any regulatory commitment.",
        "badge": "SHA-256 Immutable Ledger",
    },
]


def render_tour_speech_synthesizer(narration_text: str, step_num: int) -> None:
    """Renders a browser Web Speech API button that speaks the narration aloud."""
    escaped_text = json.dumps(narration_text)
    comp_html = f"""
    <div style="margin-top: 8px;">
        <button id="btn-speak-{step_num}" style="
            background: #f1f3f4;
            color: #1a73e8;
            border: 1px solid #dadce0;
            border-radius: 6px;
            padding: 5px 12px;
            font-size: 11.5px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s ease;
        ">
            🔊 Play Audio Narration (Web Speech)
        </button>
        <span id="speech-indicator-{step_num}" style="font-size: 11px; color: #5f6368; margin-left: 8px;"></span>
    </div>
    <script>
    (function() {{
        const btn = document.getElementById('btn-speak-{step_num}');
        const indicator = document.getElementById('speech-indicator-{step_num}');
        btn.addEventListener('click', () => {{
            if (!('speechSynthesis' in window)) {{
                indicator.textContent = 'Web Speech not supported in this browser.';
                return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance({escaped_text});
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.onstart = () => {{
                indicator.textContent = '▶ Speaking...';
                btn.style.background = '#e8f0fe';
            }};
            utterance.onend = () => {{
                indicator.textContent = '✓ Finished';
                btn.style.background = '#f1f3f4';
            }};
            utterance.onerror = () => {{
                indicator.textContent = 'Audio error';
            }};
            window.speechSynthesis.speak(utterance);
        }});
    }})();
    </script>
    """
    components.html(comp_html, height=44)


def render_guided_tour_modal() -> None:
    """
    Renders the active step modal card and highlights the corresponding target element
    with a pulsing Google Blue (#1A73E8) glowing indicator.
    """
    if not st.session_state.get("tour_active", False):
        return

    curr_step = st.session_state.get("tour_step", 1)
    if curr_step < 1:
        curr_step = 1
    elif curr_step > len(TOUR_STEPS):
        curr_step = len(TOUR_STEPS)

    step_info = TOUR_STEPS[curr_step - 1]
    target_id = step_info["target_id"]

    # Injected CSS for target pulsing glow and modal styling
    st.markdown(
        rf"""
        <style>
        /* Pulse highlight on active tour element */
        #{target_id} {{
            box-shadow: 0 0 0 4px #1a73e8, 0 0 20px rgba(26, 115, 232, 0.45) !important;
            border-color: #1a73e8 !important;
            transition: all 0.3s ease-in-out !important;
        }}
        @keyframes pulse-tour {{
            0% {{ box-shadow: 0 0 0 0 rgba(26, 115, 232, 0.6); }}
            70% {{ box-shadow: 0 0 0 10px rgba(26, 115, 232, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(26, 115, 232, 0); }}
        }}
        .tour-modal-card {{
            background: #ffffff;
            border: 2px solid #1a73e8;
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
            margin-bottom: 20px;
            border-left: 6px solid #1a73e8;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Tour Modal Card Container
    st.markdown(
        rf"""
        <div class="tour-modal-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: #1a73e8; color: #ffffff; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px;">
                        STOP {curr_step} OF {len(TOUR_STEPS)}
                    </span>
                    <span style="font-size: 11.5px; font-weight: 600; color: #5f6368;">
                        Target: {step_info['zone_name']}
                    </span>
                </div>
                <span style="background: #e8f0fe; color: #1967d2; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 6px;">
                    {step_info['badge']}
                </span>
            </div>
            <h3 style="margin: 0 0 8px 0; color: #202124; font-size: 1.15rem; font-weight: 700;">
                {step_info['title']}
            </h3>
            <p style="margin: 0 0 10px 0; color: #3c4043; font-size: 0.90rem; line-height: 1.45;">
                {step_info['summary']}
            </p>
            <div style="background: #f8f9fa; border: 1px solid #e0e3e7; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.85rem; color: #3c4043;">
                <ul style="margin: 0; padding-left: 18px;">
                    {''.join(f'<li>{item}</li>' for item in step_info['details'])}
                </ul>
            </div>
            <div style="font-size: 0.80rem; color: #5f6368; font-style: italic; margin-bottom: 4px;">
                💬 Narration Script: "{step_info['narration']}"
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Audio Narration Trigger
    render_tour_speech_synthesizer(step_info["narration"], curr_step)

    # Tour Navigation Controls
    col_prev, col_dots, col_next, col_exit = st.columns([1.2, 2.0, 1.4, 1.0])
    with col_prev:
        if st.button("⬅️ Previous", disabled=(curr_step == 1), use_container_width=True, key="btn_tour_prev"):
            prev_step = curr_step - 1
            st.session_state["tour_step"] = prev_step
            if prev_step == 1:
                st.session_state["_pending_workspace_nav"] = "🧭 5-Minute Executive Tour"
            else:
                st.session_state["_pending_workspace_nav"] = "🔬 Analytical Science & Hybrid Kinetics"
            st.rerun()

    with col_dots:
        # Step progress indicator dots
        dots_html = "".join(
            f"<span style='color: {'#1a73e8' if i == curr_step else '#dadce0'}; font-size: 18px; margin: 0 4px;'>●</span>"
            for i in range(1, len(TOUR_STEPS) + 1)
        )
        st.markdown(
            f"<div style='text-align: center; padding-top: 5px;'>{dots_html}</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        if curr_step < len(TOUR_STEPS):
            if st.button("Next Stop ➡️", type="primary", use_container_width=True, key="btn_tour_next"):
                next_step = curr_step + 1
                st.session_state["tour_step"] = next_step
                if next_step in [2, 3, 4, 5]:
                    st.session_state["_pending_workspace_nav"] = "🔬 Analytical Science & Hybrid Kinetics"
                st.rerun()
        else:
            if st.button("🏁 Finish Tour", type="primary", use_container_width=True, key="btn_tour_finish"):
                st.session_state["tour_active"] = False
                st.session_state["tour_step"] = 1
                st.rerun()

    with col_exit:
        if st.button("✕ Exit", use_container_width=True, key="btn_tour_exit"):
            st.session_state["tour_active"] = False
            st.session_state["tour_step"] = 1
            st.rerun()
