"""
Pydantic DTOs for the adaptive control engine HTTP boundary.

Internal computation uses dataclasses (see application/*.py); these schemas
exist only to validate and serialize requests/responses.
"""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PhaseFlow(BaseModel):
    phase_id: str = Field(..., min_length=1)
    flow: float = Field(..., ge=0, description="Demand flow in veh/h.")
    saturation_flow: float = Field(..., gt=0, description="Saturation flow in veh/h.")
    queue: int = Field(0, ge=0, description="Current queue length in vehicles.")
    has_pedestrian: bool = Field(False, description="Whether this phase serves a pedestrian movement.")


class PredictedDemand(BaseModel):
    horizon_minutes: int = Field(..., gt=0)
    vehicles_by_phase: dict[str, int]


class IntersectionState(BaseModel):
    intersection_id: str = Field(..., min_length=1)
    timestamp: str = Field(..., description="ISO-8601 timestamp of the snapshot.")
    phases: list[PhaseFlow] = Field(..., min_length=1)
    lost_time: float = Field(8.0, ge=0, description="Seconds lost per cycle (yellow + all-red, etc.).")
    predicted_demand: Optional[PredictedDemand] = None


class PhaseTimings(BaseModel):
    phase_id: str
    green: float = Field(..., ge=0)
    yellow: float = Field(..., ge=0)
    all_red: float = Field(..., ge=0)


class ControlRecommendation(BaseModel):
    intersection_id: str
    mode: Literal["webster", "max_pressure"]
    cycle_seconds: float
    phase_timings: list[PhaseTimings]
    next_phase: Optional[str] = None
    reasoning: str
    adjustments: list[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    data: ControlRecommendation


class ActiveStateResponse(BaseModel):
    """Read-only view of the currently active strategy for a node (HU-05).

    Exposes ``strategy_mode`` (the raw mode key — ``webster`` /
    ``max_pressure``) and the parameters of the active decision (cycle and
    green times per phase). Excludes ``flow_total`` and ``y_load_factor``
    deliberately: they belong to the engine's internal calculation and are
    not part of the Operator's pasive view (DHU-021 #21).

    The human-readable label is mapped in the frontend (DHU-006) from
    ``strategy_mode``; this DTO does NOT include the label so the backend
    stays agnostic to operator-facing naming.
    """
    node_id: str
    strategy_mode: Literal["webster", "max_pressure"]
    cycle_seconds: float
    phase_timings: list[PhaseTimings]
    decided_at: datetime
    activated_at: datetime
    activated_by: Optional[str] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
