"""
cGMP-QC-CoPilot: WBS Gantt Chart & Critical Path Visualizer.

Generates interactive Plotly timelines showing:
- Project-level milestone horizons & decision gates
- Task-level test panel drill-down (UPLC, Maurice icIEF, Cell Potency)
- Color-coded critical path highlights (zero-float tasks)
- Site-defined pull-window ranges (example +/-7 d) for flexible stability pulls
- Commercial fixed release deadline markers
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from core.stamps import DRAFT_STAMP


def build_portfolio_gantt_figure(
    conn: sqlite3.Connection,
    selected_project_id: Optional[str] = None,
    view_mode: str = "task_drilldown",  # "milestones" or "task_drilldown"
    base_calendar_date: str = "2026-10-19",
) -> Any:
    """
    Constructs an interactive Plotly timeline representing QC WBS scheduling.
    Falls back gracefully if plotly is not installed in the execution environment.
    """
    if go is None:
        class FallbackLayoutTitle:
            text = f"WBS Laboratory Execution Timeline & Critical Path [{DRAFT_STAMP}]"
        class FallbackLayout:
            title = FallbackLayoutTitle()
        class FallbackFigure:
            def __init__(self):
                self.data = ["mock_trace_fallback"]
                self.layout = FallbackLayout()
            def update_layout(self, *args, **kwargs):
                pass
            def add_trace(self, *args, **kwargs):
                self.data.append("trace")
        return FallbackFigure()

    base_dt = datetime.strptime(base_calendar_date, "%Y-%m-%d")

    fig = go.Figure()

    if view_mode == "milestones":
        query = """
        SELECT
            p.project_id, p.program_name, p.tier, p.stream,
            m.name as milestone_name, m.target_date, m.status, m.is_critical
        FROM portfolio_projects p
        JOIN milestones m ON p.project_id = m.project_id
        """
        params = []
        if selected_project_id:
            query += " WHERE p.project_id = ?"
            params.append(selected_project_id)
        query += " ORDER BY p.tier, m.target_date"

        df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            fig.update_layout(title="No Milestones Found")
            return fig

        for _, row in df.iterrows():
            m_date = str(row["target_date"])[:10]
            is_crit = bool(row["is_critical"])
            color = "#E63946" if is_crit else ("#2A9D8F" if row["status"] == "COMPLETED" else "#457B9D")

            fig.add_trace(
                go.Scatter(
                    x=[m_date],
                    y=[f"{row['project_id']} - {row['program_name'][:20]}"],
                    mode="markers+text",
                    marker=dict(
                        symbol="diamond" if is_crit else "circle",
                        size=14,
                        color=color,
                        line=dict(width=2, color="black"),
                    ),
                    text=[f"{row['milestone_name']} ({row['status']})"],
                    textposition="top right",
                    name=f"{row['project_id']} Milestone",
                    hoverinfo="text",
                    hovertext=(
                        f"<b>{row['project_id']} - {row['milestone_name']}</b><br>"
                        f"Date: {m_date}<br>"
                        f"Status: {row['status']}<br>"
                        f"Critical Path: {'YES' if is_crit else 'NO'}"
                    ),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=f"QC Portfolio Milestones Horizon [{DRAFT_STAMP}]",
            xaxis_title="Calendar Date",
            yaxis_title="Project",
            height=max(400, len(df["project_id"].unique()) * 45),
            template="plotly_white",
        )
        return fig

    # Default: task_drilldown view
    query = """
    SELECT
        w.id as task_id, w.project_id, p.program_name, w.name as task_name,
        w.skid, w.duration_hours, w.start_day, w.end_day,
        w.float_days, w.is_critical, w.flexibility_class, w.status
    FROM wbs_tasks w
    JOIN portfolio_projects p ON w.project_id = p.project_id
    """
    params = []
    if selected_project_id:
        query += " WHERE w.project_id = ?"
        params.append(selected_project_id)
    query += " ORDER BY w.project_id, w.start_day"

    tasks_df = pd.read_sql_query(query, conn, params=params)
    if tasks_df.empty:
        fig.update_layout(title="No WBS Tasks Found")
        return fig

    # Add task bars
    for _, task in tasks_df.iterrows():
        s_date = base_dt + timedelta(days=float(task["start_day"]))
        e_date = base_dt + timedelta(days=float(task["end_day"]))
        is_crit = bool(task["is_critical"])
        is_fixed = task["flexibility_class"] == "FIXED_COMMERCIAL"

        # Color: Critical = Crimson, Fixed non-critical = Navy, Flexible = Teal
        if is_crit:
            bar_color = "#E63946"  # Crimson
        elif is_fixed:
            bar_color = "#1D3557"  # Deep Navy
        else:
            bar_color = "#2A9D8F"  # Teal

        label = f"{task['project_id']}: {task['task_name']}"

        fig.add_trace(
            go.Bar(
                x=[(e_date - s_date).total_seconds() / 86400.0],
                y=[label],
                base=[s_date.strftime("%Y-%m-%d")],
                orientation="h",
                marker=dict(
                    color=bar_color,
                    line=dict(color="#000000", width=1.5 if is_crit else 1),
                ),
                name="Critical Path" if is_crit else "Flexible / Non-Critical",
                hoverinfo="text",
                hovertext=(
                    f"<b>{task['task_id']} ({task['task_name']})</b><br>"
                    f"Project: {task['project_id']} ({task['program_name']})<br>"
                    f"Skid: {task['skid']}<br>"
                    f"Duration: {task['duration_hours']}h<br>"
                    f"Start: {s_date.strftime('%Y-%m-%d')} | End: {e_date.strftime('%Y-%m-%d')}<br>"
                    f"Total Float: {task['float_days']} days<br>"
                    f"Critical Path: {'YES (Zero Float)' if is_crit else 'NO'}<br>"
                    f"Class: {task['flexibility_class']}"
                ),
                showlegend=False,
            )
        )

        # If flexible stability pull, add site-defined pull-window range (example +/-7 d) indicator
        if task["flexibility_class"] == "FLEXIBLE_WINDOW":
            win_start = (s_date - timedelta(days=7)).strftime("%Y-%m-%d")
            win_end = (e_date + timedelta(days=7)).strftime("%Y-%m-%d")
            fig.add_trace(
                go.Scatter(
                    x=[win_start, win_end],
                    y=[label, label],
                    mode="lines",
                    line=dict(color="#E76F51", width=2, dash="dash"),
                    hoverinfo="text",
                    hovertext=f"Site-Defined Pull Window (example +/-7 d): {win_start} to {win_end}",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"WBS Laboratory Execution Timeline & Critical Path [{DRAFT_STAMP}]",
        xaxis=dict(
            title="Calendar Timeline (Day / Week)",
            type="date",
        ),
        yaxis=dict(title="WBS Testing Task", autorange="reversed"),
        height=max(450, len(tasks_df) * 28),
        template="plotly_white",
        margin=dict(l=180, r=40, t=60, b=40),
    )

    return fig
