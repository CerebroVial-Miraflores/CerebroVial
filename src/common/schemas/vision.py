from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

class VisionTrack(BaseModel):
    """
    Represents a tracked vehicle from computer vision.
    Corresponds to VISION_TRACKS_RAW table.
    """
    track_uuid: str = Field(..., description="Unique vehicle ID in the session")
    camera_id: str = Field(..., description="ID of the camera sensor")
    class_id: int = Field(..., description="COCO class ID (2=car, 3=motorcycle, 5=bus, 7=truck)")
    entry_timestamp: datetime = Field(..., description="Time of first detection")
    exit_timestamp: datetime = Field(..., description="Time of last detection")
    trajectory_wkt: str = Field(..., description="Projected trajectory in WKT format")
    avg_speed_px: float = Field(..., description="Average visual speed in pixels/frame")
    direction_vector: Optional[str] = Field(None, description="Movement vector description")

class VisionFlow(BaseModel):
    """
    Represents aggregated traffic flow (counts) for a specific movement.
    """
    flow_id: str = Field(..., description="Unique identifier for this flow record")
    camera_id: str = Field(..., description="Camera capturing this flow")
    timestamp_bin: datetime = Field(..., description="Start time of the aggregation bin")
    period_seconds: int = Field(..., description="Duration of the bin in seconds")
    from_edge_id: Optional[str] = Field(None, description="Origin edge (if mapped)")
    to_edge_id: Optional[str] = Field(None, description="Destination edge (if mapped)")
    turn_direction: Optional[str] = Field(None, description="Turn direction (Left, Right, Straight)")
    vehicle_count: int = Field(..., ge=0, description="Number of vehicles counted")
    avg_speed_mps: Optional[float] = Field(None, description="Average speed in m/s")
