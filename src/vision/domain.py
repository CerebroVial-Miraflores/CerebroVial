"""
Domain entities for the Computer Vision module.
"""
from dataclasses import dataclass
from typing import List, Tuple, Iterator

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
    speed: float = None # km/h

@dataclass
class ZoneVehicleCount:
    """
    Status of a specific zone.
    """
    zone_id: str
    vehicle_count: int
    timestamp: float = 0.0
    vehicles: List[str] = None # List of vehicle IDs

@dataclass
class FrameAnalysis:
    """
    Result of analyzing a single video frame.
    """
    frame_id: int
    timestamp: float
    vehicles: List[DetectedVehicle]
    total_count: int
    zones: List[ZoneVehicleCount] = None # Optional for backward compatibility

class VehicleDetector:
    """
    Abstract base class for vehicle detection.
    """
    def detect(self, frame) -> FrameAnalysis:
        raise NotImplementedError

class VehicleTracker:
    """
    Abstract base class for vehicle tracking.
    """
    def track(self, detections: List[DetectedVehicle]) -> List[DetectedVehicle]:
        raise NotImplementedError

class SpeedEstimator:
    """
    Abstract base class for speed estimation.
    """
    def estimate(self, vehicles: List[DetectedVehicle]) -> List[DetectedVehicle]:
        raise NotImplementedError

@dataclass
class Frame:
    """
    Represents a single video frame.
    """
    id: int
    timestamp: float
    image: object # numpy array

class FrameProducer:
    """
    Abstract base class for frame production (video source).
    """
    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError
    
    def release(self):
        raise NotImplementedError

@dataclass
class TrafficData:
    """
    Aggregated traffic data for a specific zone and time window.
    """
    timestamp: float # Unix timestamp of the end of the window
    zone_id: str
    duration_seconds: float
    avg_density: float # Average number of vehicles in zone
    avg_speed: float # Average speed in km/h (if available)
    vehicle_types: dict # Count of unique vehicles by type (approximate)

class TrafficRepository:
    """
    Abstract base class for saving traffic data.
    """
    def save(self, data: TrafficData):
        raise NotImplementedError
