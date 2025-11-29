"""
Domain protocols for the Computer Vision module.
"""
from typing import List, Protocol, Iterator
from .entities import FrameAnalysis, DetectedVehicle, Frame

class VehicleDetector(Protocol):
    """
    Protocol for vehicle detection.
    """
    def detect(self, frame: object, frame_id: int) -> FrameAnalysis:
        ...

class VehicleTracker(Protocol):
    """
    Protocol for vehicle tracking.
    """
    def track(self, detections: List[DetectedVehicle]) -> List[DetectedVehicle]:
        ...

class SpeedEstimator(Protocol):
    """
    Protocol for speed estimation.
    """
    def estimate(self, vehicles: List[DetectedVehicle]) -> List[DetectedVehicle]:
        ...

class FrameProducer(Protocol):
    """
    Abstract base class for frame production (video source).
    """
    def __iter__(self) -> Iterator[Frame]:
        ...
    
    def release(self):
        ...
