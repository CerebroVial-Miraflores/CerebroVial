"""
Domain entities for the Control module.
"""
from dataclasses import dataclass
from enum import Enum

class TrafficLightState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

@dataclass
class TrafficLightPhase:
    """
    Represents a phase in the traffic light cycle.
    """
    phase_id: str
    state: TrafficLightState
    duration_seconds: int

@dataclass
class IntersectionControlPlan:
    """
    A plan for controlling traffic lights at an intersection.
    """
    intersection_id: str
    generated_at: float
    phases: list[TrafficLightPhase]
