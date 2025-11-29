from pydantic import BaseModel, Field

class CameraTrafficData(BaseModel):
    """
    Represents traffic data derived from camera footage (YOLO output).
    """
    timestamp: float = Field(..., description="Timestamp of the record")
    camera_id: str = Field(..., description="Unique identifier for the camera")
    street_monitored: str = Field(..., description="Name of the street being monitored")
    car_count: int = Field(..., ge=0, description="Number of cars detected")
    bus_count: int = Field(..., ge=0, description="Number of buses detected")
    truck_count: int = Field(..., ge=0, description="Number of trucks detected")
    motorcycle_count: int = Field(..., ge=0, description="Number of motorcycles detected")
    total_vehicles: int = Field(..., ge=0, description="Total number of vehicles")
    occupancy_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of street area occupied (0.0 - 1.0)")
    flow_rate_per_min: int = Field(..., ge=0, description="Vehicles per minute passing a line")

class Camera(BaseModel):
    """
    Represents a camera sensor's metadata.
    """
    camera_id: str = Field(..., description="Unique identifier for the camera")
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    heading: float = Field(..., ge=0, le=360, description="Heading in degrees (0-360)")
    fov: float = Field(..., gt=0, lt=180, description="Field of View in degrees")
