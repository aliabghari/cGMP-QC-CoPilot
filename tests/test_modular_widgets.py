"""
tests/test_modular_widgets.py

Unit tests for Modular Dashboard Architecture (Power BI & Smartsheet Synergy):
- Card A: CP-SAT Workload Canvas with Cross-Filtering
- Card B: Smartsheet-Style Batch Release & Schedule Grid
- Card C: Root-Cause Decomposition Tree Visualizer
- Data integrity, SHA-256 state tracking, and claims hygiene.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from core.session_recorder import SessionRecorder, get_session_recorder
from core.stamps import check_doc_claims_hygiene
from pm.modular_widgets import (
    _first_present,
    build_cpsat_workload_figure,
    build_default_batch_schedule_df,
    build_root_cause_tree_figure,
    build_test_workload_dfs,
    extract_workload_dfs,
    filter_batch_schedule_df,
)
from pm.tour_ui import _first_present as tour_first_present


@pytest.fixture
def sample_workload_dfs():
    df_base = pd.DataFrame([
        {"Skid_ID": "UPLC_SEC_CEX", "Skid_Name": "Waters Acquity UPLC", "Week": "W42", "Total_Hours": 172.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 143.3},
        {"Skid_ID": "icIEF_Maurice", "Skid_Name": "ProteinSimple Maurice", "Week": "W42", "Total_Hours": 95.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 79.2},
    ])
    df_lev = pd.DataFrame([
        {"Skid_ID": "UPLC_SEC_CEX", "Skid_Name": "Waters Acquity UPLC", "Week": "W42", "Total_Hours": 168.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 140.0},
        {"Skid_ID": "icIEF_Maurice", "Skid_Name": "ProteinSimple Maurice", "Week": "W42", "Total_Hours": 95.0, "Max_Capacity_Hours": 120.0, "Utilization_Pct": 79.2},
    ])
    return df_base, df_lev


class TestCardACPSATWorkloadCanvas:
    """Verifies Card A before/after figure generation and scaling logic."""

    def test_build_cpsat_workload_figure_structure(self, sample_workload_dfs):
        df_base, df_lev = sample_workload_dfs
        fig_before, fig_after = build_cpsat_workload_figure(df_base, df_lev)

        assert isinstance(fig_before, go.Figure)
        assert isinstance(fig_after, go.Figure)

        # Check that reference lines for 85% LEAN ceiling and 100% capacity are present
        shapes_before = [s.y0 for s in fig_before.layout.shapes if hasattr(s, "y0")]
        assert 85.0 in shapes_before
        assert 100.0 in shapes_before

        shapes_after = [s.y0 for s in fig_after.layout.shapes if hasattr(s, "y0")]
        assert 85.0 in shapes_after
        assert 100.0 in shapes_after

    def test_workload_figure_outage_scaling_and_rush_injection(self, sample_workload_dfs):
        df_base, df_lev = sample_workload_dfs
        fig_before, fig_after = build_cpsat_workload_figure(
            df_base, df_lev,
            inst_outage_mult=1.2,
            analyst_abs_mult=1.1,
            inject_rush=True,
        )

        assert isinstance(fig_before, go.Figure)
        assert isinstance(fig_after, go.Figure)
        assert len(fig_before.data) > 0
        assert len(fig_after.data) > 0

    def test_build_test_workload_dfs(self):
        df_base, df_lev = build_test_workload_dfs()
        assert isinstance(df_base, pd.DataFrame)
        assert isinstance(df_lev, pd.DataFrame)
        assert not df_base.empty
        assert not df_lev.empty
        assert "Skid_ID" in df_base.columns
        assert "Skid_Name" in df_lev.columns

    def test_extract_workload_dfs_variations(self, sample_workload_dfs):
        df_base, df_lev = sample_workload_dfs
        # 1. Flat dictionary
        flat_dict = {"df_baseline": df_base, "df_leveled": df_lev}
        b1, l1 = extract_workload_dfs(flat_dict)
        assert len(b1) == len(df_base)
        assert len(l1) == len(df_lev)

        # 2. Nested legacy_baseline dictionary
        nested_dict = {"legacy_baseline": {"df_baseline": df_base, "df_leveled": df_lev}}
        b2, l2 = extract_workload_dfs(nested_dict)
        assert len(b2) == len(df_base)
        assert len(l2) == len(df_lev)

        # 3. None / Empty / Malformed input
        b3, l3 = extract_workload_dfs(None)
        assert isinstance(b3, pd.DataFrame) and not b3.empty
        assert isinstance(l3, pd.DataFrame) and not l3.empty

        b4, l4 = extract_workload_dfs({})
        assert isinstance(b4, pd.DataFrame) and not b4.empty
        assert isinstance(l4, pd.DataFrame) and not l4.empty

    def test_first_present_dataframe_safe_extraction(self, sample_workload_dfs):
        """Verifies _first_present avoids truthiness evaluation on DataFrames and KeyError."""
        df_base, _ = sample_workload_dfs
        df_empty = pd.DataFrame()

        for fn in (_first_present, tour_first_present):
            # Safe extraction with present DataFrame
            res = fn({"df_baseline": df_base}, "missing_key", "df_baseline")
            assert isinstance(res, pd.DataFrame)
            assert len(res) == len(df_base)

            # Safe extraction with empty DataFrame without ValueError: The truth value of a DataFrame is ambiguous
            res_empty = fn({"empty_df": df_empty}, "empty_df")
            assert isinstance(res_empty, pd.DataFrame)
            assert res_empty.empty

            # Missing key returns None without KeyError
            res_none = fn({"other": 123}, "df_baseline", "baseline_df")
            assert res_none is None

            # None dict returns None
            assert fn(None, "df_baseline") is None
            assert fn({}, "df_baseline") is None


class TestCardBSmartsheetScheduleGrid:
    """Verifies Card B grid dataset generation, cross-filtering, and state tracking."""

    def test_build_default_batch_schedule_df(self):
        df = build_default_batch_schedule_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

        expected_cols = [
            "Lot ID",
            "Instrument",
            "Run Time (h)",
            "Delta Days",
            "Disposition Status",
            "Notes",
        ]
        for col in expected_cols:
            assert col in df.columns

        # Verify lot IDs
        assert "LOT-NS-2026-004" in df["Lot ID"].values
        assert "LOT-BI-CAP-101" in df["Lot ID"].values

    def test_filter_batch_schedule_df(self):
        df = build_default_batch_schedule_df()

        # Unfiltered cases
        assert len(filter_batch_schedule_df(df, None)) == 10
        assert len(filter_batch_schedule_df(df, "")) == 10
        assert len(filter_batch_schedule_df(df, "All Instruments")) == 10

        # Filter by Waters UPLC
        df_uplc = filter_batch_schedule_df(df, "Waters Acquity UPLC")
        assert len(df_uplc) == 5
        assert all("Waters" in inst for inst in df_uplc["Instrument"])

        # Filter by Maurice icIEF
        df_icief = filter_batch_schedule_df(df, "ProteinSimple Maurice")
        assert len(df_icief) == 2
        assert all("Maurice" in inst for inst in df_icief["Instrument"])

        # Filter by Bioassay Suite
        df_bio = filter_batch_schedule_df(df, "SpectraMax Bioassay Suite")
        assert len(df_bio) == 2
        assert all("SpectraMax" in inst for inst in df_bio["Instrument"])

        # Filter by Stability Chambers
        df_rees = filter_batch_schedule_df(df, "Rees Stability Chambers")
        assert len(df_rees) == 1
        assert "Rees" in df_rees["Instrument"].iloc[0]

    def test_audit_hash_tracking_on_grid_modification(self):
        """Verifies ALCOA+ SHA-256 state tracking when grid data changes."""
        df = build_default_batch_schedule_df()
        csv_before = df.to_csv(index=False).encode("utf-8")
        hash_before = hashlib.sha256(csv_before).hexdigest()

        # Modify a row
        df_edited = df.copy()
        df_edited.loc[df_edited["Lot ID"] == "LOT-NS-2026-004", "Disposition Status"] = "RELEASED"
        csv_after = df_edited.to_csv(index=False).encode("utf-8")
        hash_after = hashlib.sha256(csv_after).hexdigest()

        assert hash_before != hash_after

        # Record in SessionRecorder
        recorder = SessionRecorder()
        recorder.start()
        event = recorder.record_event(
            event_type="SMARTSHEET_GRID_EDIT",
            workspace="📊 QC Operations & Project Management",
            widget_key="smartsheet_schedule_grid_editor",
            old_value=hash_before[:16],
            new_value=hash_after[:16],
            agent3_status="NOMINAL",
            metadata={"lot_modified": "LOT-NS-2026-004", "rows": len(df_edited)},
        )
        assert event is not None
        assert event.event_type == "SMARTSHEET_GRID_EDIT"
        assert event.new_value == hash_after[:16]
        assert event.sha256_hash is not None


class TestCardCRootCauseTreeVisualizer:
    """Verifies Card C root-cause decomposition tree figures."""

    def test_sankey_root_cause_tree_branches(self):
        fig = build_root_cause_tree_figure(diagram_type="sankey")
        assert isinstance(fig, go.Figure)

        sankey_trace = fig.data[0]
        labels = list(sankey_trace.node.label)

        # Verify all 4 required branches
        assert "Monomer Release (93.8%)" in labels
        assert "Pump Overpressure (310.4 bar)" in labels
        assert "Chiller Fault (17.8°C / 54h)" in labels
        assert "Phase I Lab Invalidation" in labels

        # Verify link connectivity
        assert len(sankey_trace.link.source) == 4
        assert len(sankey_trace.link.target) == 4

        # Verify enhanced node styling and contrast
        assert sankey_trace.node.pad == 40
        assert sankey_trace.node.thickness == 24
        assert sankey_trace.node.line.color == "#FFFFFF"
        assert sankey_trace.node.line.width == 1.5

    def test_sunburst_root_cause_tree_branches(self):
        fig = build_root_cause_tree_figure(diagram_type="sunburst")
        assert isinstance(fig, go.Figure)

        sunburst_trace = fig.data[0]
        labels = list(sunburst_trace.labels)

        assert "Root Cause" in labels
        assert "Monomer Release (93.8%)" in labels
        assert "Pump Overpressure (310.4 bar)" in labels
        assert "Chiller Fault (17.8°C / 54h)" in labels
        assert "Phase I Lab Invalidation" in labels


class TestClaimsHygieneModularWidgets:
    """Verifies Guardrail G2 claims hygiene on pm/modular_widgets.py."""

    def test_modular_widgets_no_banned_phrases(self):
        mod_file = Path("pm/modular_widgets.py")
        assert mod_file.exists()
        content = mod_file.read_text(encoding="utf-8")
        violations = check_doc_claims_hygiene(content)
        assert len(violations) == 0, f"Found banned affirmative claims in pm/modular_widgets.py: {violations}"
