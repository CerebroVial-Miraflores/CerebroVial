"""
Domain entities for the Computer Vision module.
"""
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DetectedVehicle:
    """
    Represents a vehicle detected by the vision system.
    """
    id: str
    type: str  # car, bus, truck, motorcycle
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float

@dataclass
class FrameAnalysis:
    """
    Result of analyzing a single video frame.
    """
    frame_id: int
    timestamp: float
    vehicles: List[DetectedVehicle]
    total_count: int

class VehicleDetector:
    """
    Abstract base class for vehicle detection.
    """
    def detect(self, frame) -> FrameAnalysis:
        raise NotImplementedError
