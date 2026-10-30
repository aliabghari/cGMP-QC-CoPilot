"""
tests/test_session_recorder_and_tour.py

Unit tests for:
1. Interactive 5-Minute Guided Tour Engine (`pm/guided_tour.py`).
2. User Interaction & Session Recording Engine (`core/session_recorder.py`).

Verifies:
- 5 guided stops sequence, matching target IDs, and narration contents.
- Session recorder event capture, immutable SHA-256 hash chaining, export to JSON/CSV.
- Replay state retrieval and timeline fidelity.
- Strict Guardrail G2 compliance (zero affirmative regulatory release claims).
"""

from __future__ import annotations

import json
import time
import pytest

from core.session_recorder import (
    InteractionEvent,
    SessionRecorder,
    get_session_recorder,
)
from core.stamps import assert_no_banned_phrases, check_banned_phrases
from pm.guided_tour import TOUR_STEPS


def test_tour_steps_structure_and_target_ids():
    """Verifies the 5 guided tour stops and their target element anchors."""
    assert len(TOUR_STEPS) == 5, f"Expected 5 tour stops, got {len(TOUR_STEPS)}"

    expected_targets = [
        ("left-nav-rail", "Left Navigation Rail"),
        ("pinn-schematic-container", "Center Stage PINN Schematic"),
        ("right-inspector-rail", "Right Inspector Rail"),
        ("trajectory-canvas", "Center Stage Trajectory"),
        ("audit-ledger-container", "Center Stage Audit Commitment"),
    ]

    for idx, (expected_target, zone_substr) in enumerate(expected_targets):
        step = TOUR_STEPS[idx]
        assert step["step"] == idx + 1
        assert step["target_id"] == expected_target
        assert zone_substr in step["zone_name"]
        assert len(step["title"]) > 5
        assert len(step["summary"]) > 20
        assert len(step["details"]) >= 2
        assert len(step["narration"]) > 30
        assert step["badge"]


def test_session_recorder_lifecycle_and_hashing():
    """Verifies starting, stopping, recording events, and SHA-256 hash chaining."""
    recorder = SessionRecorder()
    assert not recorder.is_recording
    assert len(recorder.events) == 0

    recorder.start()
    assert recorder.is_recording

    # Record first parameter shift
    evt1 = recorder.record_event(
        event_type="PARAMETER_SHIFT",
        widget_key="excursion_temp_slider",
        old_value=28.0,
        new_value=32.5,
        workspace="🔬 Analytical Science & Hybrid Kinetics",
        agent3_status="NOMINAL",
        metadata={"param_type": "biopharm_stability"},
    )

    assert evt1 is not None
    assert evt1.event_index == 1
    assert evt1.widget_key == "excursion_temp_slider"
    assert evt1.old_value == 28.0
    assert evt1.new_value == 32.5
    assert evt1.prev_hash == "GENESIS_BLOCK_0000000000000000000000000000000000000000000000000000"
    assert len(evt1.sha256_hash) == 64

    # Record second event
    evt2 = recorder.record_event(
        event_type="PARAMETER_SHIFT",
        widget_key="excursion_hours_slider",
        old_value=36.0,
        new_value=48.0,
        workspace="🔬 Analytical Science & Hybrid Kinetics",
        agent3_status="OOT DETECTED",
        metadata={"param_type": "biopharm_stability"},
    )

    assert evt2 is not None
    assert evt2.event_index == 2
    assert evt2.prev_hash == evt1.sha256_hash
    assert len(evt2.sha256_hash) == 64
    assert evt2.sha256_hash != evt1.sha256_hash

    # Stop recording
    recorder.stop()
    assert not recorder.is_recording

    # Attempting to record when stopped should return None
    evt_noop = recorder.record_event(
        event_type="NOOP",
        widget_key="dummy",
        old_value=0,
        new_value=1,
    )
    assert evt_noop is None
    assert len(recorder.events) == 2


def test_session_recorder_json_and_csv_exports():
    """Verifies that interaction journal exports valid JSON and CSV logs."""
    recorder = SessionRecorder()
    recorder.start()
    recorder.record_event(
        event_type="PARAMETER_SHIFT",
        widget_key="formulation_ph_slider",
        old_value=5.8,
        new_value=6.2,
        workspace="🔬 Analytical Science & Hybrid Kinetics",
        agent3_status="NOMINAL",
    )
    recorder.stop()

    # JSON Export
    json_str = recorder.export_json()
    parsed = json.loads(json_str)
    assert parsed["session_summary"]["total_events"] == 1
    assert len(parsed["interaction_journal"]) == 1
    assert parsed["interaction_journal"][0]["widget_key"] == "formulation_ph_slider"

    # CSV Export
    csv_str = recorder.export_csv()
    assert "event_index,timestamp_utc,event_type,widget_key" in csv_str
    assert "formulation_ph_slider" in csv_str
    assert "5.8" in csv_str
    assert "6.2" in csv_str


def test_tour_pending_redirect_mechanism():
    """Verifies that guided tour avoids StreamlitAPIException by using _pending_workspace_nav."""
    with open("pm/guided_tour.py", "r", encoding="utf-8") as f:
        tour_code = f.read()

    # Must NOT directly mutate workspace_nav_selection after widget instantiation
    assert 'st.session_state["workspace_nav_selection"] =' not in tour_code
    assert 'st.session_state[\'workspace_nav_selection\'] =' not in tour_code

    # Must use _pending_workspace_nav redirect pattern
    assert 'st.session_state["_pending_workspace_nav"]' in tour_code

    # app.py must consume and pop _pending_workspace_nav prior to widget instantiation
    with open("app.py", "r", encoding="utf-8") as f:
        app_code = f.read()
    assert '_pending_workspace_nav' in app_code
    assert 'st.session_state.pop("_pending_workspace_nav")' in app_code


def test_g2_claims_compliance():
    """Enforces Guardrail G2 claims hygiene across guided tour and session recorder."""
    with open("pm/guided_tour.py", "r", encoding="utf-8") as f:
        tour_code = f.read()
    assert_no_banned_phrases(tour_code, "pm/guided_tour.py")

    with open("core/session_recorder.py", "r", encoding="utf-8") as f:
        rec_code = f.read()
    assert_no_banned_phrases(rec_code, "core/session_recorder.py")


def test_tour_render_path_regression_no_keyerror(monkeypatch):
    """
    Regression test for KeyError: 'df_baseline' in pm/tour_ui.py.
    Exercises Stop 2 render path with missing keys and ensures 'Showing fallback sample data' is displayed.
    """
    from unittest.mock import MagicMock
    import streamlit as st
    from pm.tour_ui import render_5_minute_tour_tab

    # Mock streamlit components to capture calls
    info_calls = []
    def mock_info(msg, *args, **kwargs):
        info_calls.append(str(msg))

    mock_tab = MagicMock()
    mock_tab.__enter__.return_value = mock_tab
    mock_tab.__exit__.return_value = None

    monkeypatch.setattr(st, "tabs", lambda *args, **kwargs: [mock_tab] * 5)
    monkeypatch.setattr(st, "columns", lambda spec, *args, **kwargs: [mock_tab] * (spec if isinstance(spec, int) else len(spec)))
    monkeypatch.setattr(st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(st, "info", mock_info)
    monkeypatch.setattr(st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "radio", lambda *args, **kwargs: "Sankey Flow")
    monkeypatch.setattr(st, "plotly_chart", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "download_button", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "data_editor", lambda df, *args, **kwargs: df)
    monkeypatch.setattr(st, "dialog", lambda *args, **kwargs: (lambda fn: fn))

    # Force run_hybrid_capacity_and_risk to return empty dict or legacy-only dict
    import agents.agent_1_lean_capacity.scripts.hybrid_schedule_risk as hsr
    monkeypatch.setattr(hsr, "run_hybrid_capacity_and_risk", lambda **kwargs: {})

    # Execute render path: must not raise KeyError: 'df_baseline'
    render_5_minute_tour_tab()

    # Confirm fallback notification was displayed
    assert any("Showing fallback sample data" in c for c in info_calls), (
        f"Expected 'Showing fallback sample data' notice, got: {info_calls}"
    )


def test_tour_render_path_real_run_hybrid_heijunka_aggregates():
    """
    Feeds real output of run_hybrid_capacity_and_risk(usp_harvest_delay_days=0)
    through tour Stop 2 extraction logic.
    Verifies:
    - used_fallback is False
    - Baseline and leveled DataFrames are correctly extracted from legacy_baseline
    - Week-42 Waters UPLC hours match the Heijunka weekly aggregate schedule:
        - Before leveling: 136.0 hours (113.3% of 120h nominal capacity)
        - After leveling: 102.0 hours (85.0% of 120h nominal capacity)
    """
    from agents.agent_1_lean_capacity.scripts.hybrid_schedule_risk import run_hybrid_capacity_and_risk
    from pm.modular_widgets import _first_present, extract_workload_dfs

    cap_demo = run_hybrid_capacity_and_risk(usp_harvest_delay_days=0)

    legacy = cap_demo.get("legacy_baseline", {}) if isinstance(cap_demo.get("legacy_baseline"), dict) else {}
    df_base = _first_present(cap_demo, "df_baseline", "baseline_df")
    if df_base is None:
        df_base = _first_present(legacy, "df_baseline", "baseline_df")

    df_opt = _first_present(cap_demo, "df_leveled", "df_optimized", "leveled_df")
    if df_opt is None:
        df_opt = _first_present(legacy, "df_leveled", "df_optimized", "leveled_df")

    used_fallback = False
    if df_base is None or df_opt is None:
        used_fallback = True
        fb_base, fb_opt = extract_workload_dfs(cap_demo)
        if df_base is None:
            df_base = fb_base
        if df_opt is None:
            df_opt = fb_opt

    # 1. Assert fallback was not used
    assert used_fallback is False, "Expected real data extraction, but fallback was triggered"

    # 2. Assert DataFrames were extracted from legacy_baseline
    assert df_base is not None and df_opt is not None
    assert df_base is legacy.get("df_baseline")
    assert df_opt is legacy.get("df_leveled")

    # 3. Locate Week 42 Waters UPLC row and extract hours
    row_base = df_base[(df_base["Week"] == "W42") & (df_base["Skid_ID"].str.contains("UPLC", case=False, na=False))]
    assert not row_base.empty, "Could not find W42 UPLC row in df_base"
    hours_before = float(row_base.iloc[0]["Total_Hours"])

    row_opt = df_opt[(df_opt["Week"] == "W42") & (df_opt["Skid_ID"].str.contains("UPLC", case=False, na=False))]
    assert not row_opt.empty, "Could not find W42 UPLC row in df_opt"
    hours_after = float(row_opt.iloc[0]["Total_Hours"])

    # 4. Assert against exact observed Heijunka weekly aggregates (136.0h before, 102.0h after)
    assert hours_before == pytest.approx(136.0, abs=0.05), (
        f"Week 42 Waters UPLC before leveling hours mismatch: observed {hours_before}, expected 136.0 +- 0.05"
    )
    assert hours_after == pytest.approx(102.0, abs=0.05), (
        f"Week 42 Waters UPLC after leveling hours mismatch: observed {hours_after}, expected 102.0 +- 0.05"
    )


def test_cpsat_solver_peak_utilization_and_nominal_hours():
    """
    Verifies CP-SAT solver peak utilization percentages and corresponding nominal workload hours:
    - peak_utilization_before['UPLC_SEC_CEX'] == 143.3% -> 171.96 h (~172.0 h on 120 h nominal capacity)
    - peak_utilization_after['UPLC_SEC_CEX'] == 140.0% -> 168.0 h on 120 h nominal capacity
    """
    from agents.agent_1_lean_capacity.scripts.hybrid_schedule_risk import run_hybrid_capacity_and_risk

    cap_demo = run_hybrid_capacity_and_risk(usp_harvest_delay_days=0)
    cpsat = cap_demo.get("cpsat_result")
    assert cpsat is not None, "cpsat_result missing from run_hybrid_capacity_and_risk output"

    peak_before = cpsat.peak_utilization_before.get("UPLC_SEC_CEX")
    peak_after = cpsat.peak_utilization_after.get("UPLC_SEC_CEX")

    assert peak_before is not None and peak_after is not None

    # Assert peak utilization percentages
    assert peak_before == pytest.approx(143.3, abs=0.05), (
        f"UPLC_SEC_CEX peak utilization before mismatch: observed {peak_before}, expected 143.3 +- 0.05"
    )
    assert peak_after == pytest.approx(140.0, abs=0.05), (
        f"UPLC_SEC_CEX peak utilization after mismatch: observed {peak_after}, expected 140.0 +- 0.05"
    )

    # Assert that 172.0 h and 168.0 h figures follow from the 120 h nominal capacity
    nominal_capacity_hours = 120.0
    calc_hours_before = (peak_before / 100.0) * nominal_capacity_hours
    calc_hours_after = (peak_after / 100.0) * nominal_capacity_hours

    assert calc_hours_before == pytest.approx(172.0, abs=0.05), (
        f"Calculated peak hours before mismatch: observed {calc_hours_before}, expected 172.0 +- 0.05"
    )
    assert calc_hours_after == pytest.approx(168.0, abs=0.05), (
        f"Calculated peak hours after mismatch: observed {calc_hours_after}, expected 168.0 +- 0.05"
    )
