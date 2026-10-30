"""
cGMP-QC-CoPilot: Schedule Risk & Capacity Data Model.

Defines the core operational primitives:
- Flexibility classes: fixed_commercial, fixed_contractual, flexible_window, flexible_internal
- Instruments with changeovers, calibration, and planned downtime
- Analysts with shift calendars, method qualifications, and cross-training
- Tasks with method-derived durations and precedence constraints
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FlexibilityClass(str, Enum):
    FIXED_COMMERCIAL = "fixed_commercial"
    FIXED_CONTRACTUAL = "fixed_contractual"
    FLEXIBLE_WINDOW = "flexible_window"
    FLEXIBLE_INTERNAL = "flexible_internal"

    def is_fixed(self) -> bool:
        return self in (FlexibilityClass.FIXED_COMMERCIAL, FlexibilityClass.FIXED_CONTRACTUAL)


@dataclass
class InstrumentDefinition:
    id: str
    name: str
    weekly_max_hours: float = 120.0
    lean_threshold_pct: float = 85.0
    changeover_hours: float = 1.5
    planned_downtime_hours: float = 4.0

    @property
    def lean_target_hours(self) -> float:
        return self.weekly_max_hours * (self.lean_threshold_pct / 100.0)


@dataclass
class Analyst:
    id: str
    name: str
    shift: str = "Shift A (Day)"
    weekly_capacity_hours: float = 40.0
    qualifications: List[str] = field(default_factory=lambda: ["UPLC", "icIEF", "CE_SDS", "Bioassay"])
    cross_trained: bool = True


@dataclass
class MethodParameters:
    equilibration_hours: float = 1.0
    system_suitability_hours: float = 1.5
    injection_count: int = 24
    run_time_minutes: float = 25.0
    wash_time_minutes: float = 5.0
    plate_run_time_hours: Optional[float] = None

    def compute_duration_hours(self) -> float:
        if self.plate_run_time_hours is not None:
            return self.system_suitability_hours + self.plate_run_time_hours
        cycle_minutes = self.run_time_minutes + self.wash_time_minutes
        total_inj_hours = (self.injection_count * cycle_minutes) / 60.0
        return round(self.equilibration_hours + self.system_suitability_hours + total_inj_hours, 2)


@dataclass
class Task:
    task_id: str
    project_id: str
    program_name: str
    test_panel: str
    instrument_id: str
    required_qualification: str
    method_params: MethodParameters
    nominal_duration_hours: float
    sample_arrival_day: float
    due_day: float
    flexibility_class: FlexibilityClass
    window_days: float = 0.0
    penalty_weight: float = 1.0
    patient_impact: bool = False
    priority_tier: str = "Tier 1A"
    predecessors: List[str] = field(default_factory=list)
    scheduled_start_day: Optional[float] = None
    scheduled_instrument_id: Optional[str] = None
    scheduled_analyst_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        task_id: str,
        project_id: str,
        program_name: str,
        test_panel: str,
        instrument_id: str,
        sample_arrival_day: float,
        due_day: float,
        flexibility_class: FlexibilityClass = FlexibilityClass.FIXED_COMMERCIAL,
        window_days: float = 0.0,
        penalty_weight: float = 1.0,
        patient_impact: bool = False,
        priority_tier: str = "Tier 1A",
        predecessors: Optional[List[str]] = None,
        method_params: Optional[MethodParameters] = None,
        nominal_duration_hours: Optional[float] = None,
    ) -> "Task":
        params = method_params or MethodParameters()
        dur = nominal_duration_hours if nominal_duration_hours is not None else params.compute_duration_hours()
        qual = "Bioassay" if "Bioassay" in test_panel else ("icIEF" if "icIEF" in test_panel else "UPLC")
        return cls(
            task_id=task_id,
            project_id=project_id,
            program_name=program_name,
            test_panel=test_panel,
            instrument_id=instrument_id,
            required_qualification=qual,
            method_params=params,
            nominal_duration_hours=dur,
            sample_arrival_day=sample_arrival_day,
            due_day=due_day,
            flexibility_class=flexibility_class,
            window_days=window_days,
            penalty_weight=penalty_weight,
            patient_impact=patient_impact,
            priority_tier=priority_tier,
            predecessors=predecessors or [],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "program_name": self.program_name,
            "test_panel": self.test_panel,
            "instrument_id": self.instrument_id,
            "required_qualification": self.required_qualification,
            "nominal_duration_hours": self.nominal_duration_hours,
            "sample_arrival_day": self.sample_arrival_day,
            "due_day": self.due_day,
            "flexibility_class": self.flexibility_class.value,
            "window_days": self.window_days,
            "penalty_weight": self.penalty_weight,
            "patient_impact": self.patient_impact,
            "priority_tier": self.priority_tier,
            "predecessors": self.predecessors,
        }
