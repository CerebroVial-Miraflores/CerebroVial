from pydantic import BaseModel, Field
from typing import Optional

class Camera(BaseModel):
    """
    Represents a physical camera sensor in the network.
    """
    camera_id: str = Field(..., description="Unique identifier for the camera")
    node_id: Optional[str] = Field(None, description="ID of the intersection/node this camera monitors")
    lat: float = Field(..., description="Latitude of the camera")
    lon: float = Field(..., description="Longitude of the camera")
    heading: float = Field(..., ge=0, le=360, description="Direction the camera is facing (degrees, 0=North)")
    fov: float = Field(..., gt=0, description="Field of view in degrees")
