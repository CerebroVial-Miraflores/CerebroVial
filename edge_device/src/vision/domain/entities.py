"""
Domain entities for the Computer Vision module.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from .value_objects import CameraId, VehicleId, ZoneId

_ALLOWED_VEHICLE_TYPES = frozenset({"car", "bus", "truck", "motorcycle"})


@dataclass(frozen=True)
class DetectedVehicle:
    """A vehicle detected (and tracked) within a single frame."""
    id: VehicleId
    type: str  # car, bus, truck, motorcycle
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float
    speed: Optional[float] = None  # km/h; set by the speed estimator (Fase 4) via dataclasses.replace


@dataclass(frozen=True)
class ZoneVehicleCount:
    """Vehicles counted within a single configured zone (polygon) for a frame."""
    zone_id: ZoneId
    count: int
    vehicle_ids: list[VehicleId]


@dataclass(frozen=True)
class FrameAnalysis:
    """Result of analyzing a single video frame."""
    frame_id: int
    timestamp: float
    vehicles: list[DetectedVehicle]
    unique_vehicles: int
    zones: dict[ZoneId, ZoneVehicleCount]


@dataclass(frozen=True)
class Frame:
    """A single video frame."""
    id: int
    timestamp: float
    image: np.ndarray


@dataclass(frozen=True)
class TrafficData:
    """Aggregated traffic metrics for a single zone over a closed time window.

    This is the canonical output shape of the vision module's aggregation
    layer. It is the value object that the application use case persists
    via `TrafficRepository` and publishes via `Broadcaster`.

    All fields are required unless explicitly typed Optional.
    """

    # Identifiers
    camera_id: CameraId
    zone_id: ZoneId

    # Temporal window (closed interval)
    window_start: datetime           # tz-aware UTC
    window_end: datetime             # tz-aware UTC; window_end > window_start
    window_duration_seconds: float   # equal to (window_end - window_start).total_seconds(); > 0

    # Counts
    unique_vehicles: int                   # cardinality of distinct VehicleIds in window; >= 0
    vehicles_by_type: dict[str, int]       # keys subset of {"car", "bus", "truck", "motorcycle"};
                                           # sum of values equals unique_vehicles

    # Temporal aggregates
    mean_speed_kmh: Optional[float]        # count-weighted average; None if no frames have count > 0
                                           # or if camera has no pixel-to-meter calibration
    flow_vehicles_per_hour: float          # unique_vehicles / window_duration_seconds * 3600; >= 0

    # Local derived
    mean_occupancy: float                  # mean over frames of (bbox_area ∩ polygon_area) / polygon_area;
                                           # in [0.0, 1.0]
    density_vehicles_per_km: Optional[float]  # unique_vehicles / zone.segment_length_meters * 1000;
                                              # None if zone has no segment_length_meters configured;
                                              # >= 0 when not None

    def __post_init__(self) -> None:
        if self.window_end <= self.window_start:
            raise ValueError("window_end debe ser posterior a window_start")
        if self.window_duration_seconds <= 0:
            raise ValueError("window_duration_seconds debe ser > 0")
        if self.unique_vehicles < 0:
            raise ValueError("unique_vehicles no puede ser negativo")
        invalid_types = set(self.vehicles_by_type) - _ALLOWED_VEHICLE_TYPES
        if invalid_types:
            raise ValueError(
                f"vehicles_by_type contiene tipos no permitidos: {sorted(invalid_types)}"
            )
        if sum(self.vehicles_by_type.values()) != self.unique_vehicles:
            raise ValueError("sum(vehicles_by_type) debe igualar unique_vehicles")
        if self.mean_speed_kmh is not None and self.mean_speed_kmh < 0:
            raise ValueError("mean_speed_kmh no puede ser negativo")
        if self.flow_vehicles_per_hour < 0:
            raise ValueError("flow_vehicles_per_hour no puede ser negativo")
        if not 0.0 <= self.mean_occupancy <= 1.0:
            raise ValueError("mean_occupancy debe estar en [0.0, 1.0]")
        if self.density_vehicles_per_km is not None and self.density_vehicles_per_km < 0:
            raise ValueError("density_vehicles_per_km no puede ser negativo")
