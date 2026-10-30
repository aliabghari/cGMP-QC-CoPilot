"""
cGMP-QC-CoPilot: Project-Management Operating Layer (WP2).
Provides decision support for QC Project Management & Local MSO operations:
- Relational SQLite data access & Part 11-style audit trail integration
- Editable Portfolio Projects Grid with CPM/DES risk recalculation and RAG overrides
- Plotly WBS Gantt chart with critical-path analysis and ICH Q1A pull windows
- Kanban action tracker with mandatory completion notes and aging diagnostics
- Blocker radar with delay impact and automated escalation triggers
- Rule-based client request to GMP work package generator with reasoning traces and prompt injection defense
- Modular tech-transfer templates with placeholder validation
- RAID, RACI (strictly 1 Accountable), Stakeholder, and Operational KPI registers
- Conformal sample arrival forecasts and draft heads-up notifications
"""

__all__ = [
    "data_access",
    "portfolio_grid",
    "gantt",
    "kanban",
    "blockers",
    "work_package_generator",
    "tech_transfer",
    "registers",
    "sample_alerts",
    "hybrid_architecture_ui",
    "guided_tour",
]
