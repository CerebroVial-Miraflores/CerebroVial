"""
Domain entities for the Computer Vision module.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

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
    speed: Optional[float] = None # km/h

@dataclass
class ZoneVehicleCount:
    """
    Status of a specific zone.
    """
    zone_id: str
    vehicle_count: int
    timestamp: float = 0.0
    vehicles: List[str] = None # List of vehicle IDs
    vehicle_details: Dict[str, str] = field(default_factory=dict) # Map ID -> Type
    avg_speed: float = 0.0
    occupancy: float = 0.0 # Percentage of zone area occupied (0.0 - 1.0)
    vehicle_types: Dict[str, int] = field(default_factory=dict)
    camera_id: str = "unknown"
    street_monitored: str = "unknown"

@dataclass
class FrameAnalysis:
    """
    Result of analyzing a single video frame.
    """
    frame_id: int
    timestamp: float
    vehicles: List[DetectedVehicle]
    total_count: int
    raw_detection_count: int = 0 # Debug: Count of raw detections before tracking
    zones: List[ZoneVehicleCount] = None # Optional for backward compatibility

@dataclass
class Frame:
    """
    Represents a single video frame.
    """
    id: int
    timestamp: float
    image: object # numpy array

@dataclass
class TrafficData:
    """
    Aggregated traffic data for a specific zone and time window.
    Standardized to match CameraTrafficData schema.
    """
    timestamp: float # Unix timestamp of the end of the window
    zone_id: str
    camera_id: str
    street_monitored: str
    duration_seconds: float
    
    # Metrics
    total_vehicles: int 
    avg_density: float # Average number of vehicles in zone (float precision)
    avg_speed: float # Average speed in km/h
    avg_occupancy: float # Average percentage of zone area occupied
    flow_rate_per_min: int # Number of unique vehicles seen in the window
    
    # Vehicle Counts (Breakdown)
    car_count: int
    bus_count: int
    truck_count: int
    motorcycle_count: int
    
    vehicle_types: dict # Keep original dict for debug/flexibility
