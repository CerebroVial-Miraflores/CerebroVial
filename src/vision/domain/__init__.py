"""
Domain module initialization.
"""
from .entities import (
    DetectedVehicle, 
    ZoneVehicleCount, 
    FrameAnalysis, 
    Frame, 
    TrafficData
)
from .protocols import (
    VehicleDetector, 
    VehicleTracker, 
    SpeedEstimator, 
    FrameProducer
)
from .repositories import TrafficRepository
