"""
Domain layer of the Computer Vision module.

Barrel export ordered by layer (value objects -> entities -> service
protocols -> repository) so it reads as the map of the domain.
"""
from .value_objects import VehicleId, ZoneId, CameraId
from .entities import (
    DetectedVehicle,
    ZoneVehicleCount,
    FrameAnalysis,
    Frame,
    TrafficData,
)
from .protocols import (
    VehicleDetector,
    VehicleTracker,
    SpeedEstimator,
    FrameProducer,
    ZoneCounter,
    AsyncAggregator,
    Broadcaster,
    FrameRenderer,
)
from .repositories import TrafficRepository

__all__ = [
    # Value objects
    "VehicleId",
    "ZoneId",
    "CameraId",
    # Entities
    "DetectedVehicle",
    "ZoneVehicleCount",
    "FrameAnalysis",
    "Frame",
    "TrafficData",
    # Service protocols
    "VehicleDetector",
    "VehicleTracker",
    "SpeedEstimator",
    "FrameProducer",
    "ZoneCounter",
    "AsyncAggregator",
    "Broadcaster",
    "FrameRenderer",
    # Repository
    "TrafficRepository",
]
