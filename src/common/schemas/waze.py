from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class WazeJam(BaseModel):
    """
    Represents a traffic jam event from Waze.
    Corresponds to WAZE_JAMS_RAW table.
    """
    event_uuid: str = Field(..., description="Unique identifier for the traffic event")
    snapshot_timestamp: int = Field(..., description="Publication timestamp (pubMillis)")
    edge_id: Optional[str] = Field(None, description="Graph edge ID after Map Matching")
    waze_line_geometry: List[dict] = Field(..., description="List of {x: lon, y: lat} coordinates")
    speed_mps: float = Field(..., description="Current speed in meters per second")
    delay_seconds: int = Field(..., description="Delay in seconds compared to free flow")
    congestion_level: int = Field(..., ge=0, le=5, description="Congestion severity level (0-5)")
    jam_length_m: int = Field(..., description="Length of the jam in meters")
    road_type: int = Field(..., description="Waze road type ID")
    turn_type: Optional[str] = Field(None, description="Turn context (Left, Right, etc.)")

    @field_validator('speed_mps')
    def speed_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('speed_mps must be non-negative')
        return v

class WazeAlert(BaseModel):
    """
    Represents a user-reported alert from Waze.
    Corresponds to WAZE_ALERTS_RAW table.
    """
    alert_uuid: str = Field(..., description="Unique identifier for the alert")
    timestamp: int = Field(..., description="Publication timestamp (pubMillis)")
    edge_id: Optional[str] = Field(None, description="Graph edge ID after Map Matching")
    alert_type: str = Field(..., description="Major category (ACCIDENT, HAZARD, etc.)")
    alert_subtype: Optional[str] = Field(None, description="Detailed category")
    reliability: int = Field(..., ge=1, le=10, description="Reliability score (1-10)")
    confidence: int = Field(..., ge=0, le=5, description="Confidence score (0-5)")
    magvar: int = Field(..., ge=0, le=359, description="Heading in degrees")
    report_location: dict = Field(..., description="{x: lon, y: lat} coordinate")

class WazeIrregularity(BaseModel):
    """
    Represents a traffic irregularity from Waze.
    """
    irregularity_uuid: str = Field(..., description="Unique identifier")
    timestamp: int = Field(..., description="Publication timestamp")
    regular_speed_mps: float = Field(..., description="Historical average speed")
    current_speed_mps: float = Field(..., description="Current speed")
    trend: float = Field(..., description="Trend indicator")
    line_geometry: List[dict] = Field(..., description="Geometry of the irregularity")

class WazeTrafficData(BaseModel):
    """
    Represents the flattened traffic data structure for ML training.
    """
    uuid: str = Field(..., description="Unique identifier for the record")
    timestamp: float = Field(..., description="Timestamp of the record")
    city: str = Field(..., description="City name")
    street: str = Field(..., description="Street name")
    road_type: int = Field(..., description="Waze road type ID")
    location_lat: float = Field(..., description="Latitude of the center")
    location_lon: float = Field(..., description="Longitude of the center")
    length_meters: int = Field(..., description="Length of the affected segment")
    speed_kmh: float = Field(..., description="Average speed in km/h")
    delay_seconds: int = Field(..., description="Delay in seconds")
    level: int = Field(..., ge=0, le=5, description="Traffic level (0-5)")

class WazeTrafficFeatures(WazeTrafficData):
    """
    Represents Waze traffic data with additional engineered features.
    """
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)")
    is_weekend: bool = Field(..., description="True if it is a weekend")
    lag_delay_1h: Optional[int] = Field(None, description="Delay 1 hour ago (if available)")
