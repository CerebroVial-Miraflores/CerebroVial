"""
API for managing multiple cameras.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
from ....application.services.multi_camera import MultiCameraManager
from ....infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster

app = FastAPI()

# Singleton
_manager: Optional[MultiCameraManager] = None

def init_manager(broadcaster: RealtimeBroadcaster):
    global _manager
    _manager = MultiCameraManager(broadcaster)

def get_manager() -> MultiCameraManager:
    if _manager is None:
        raise HTTPException(500, "Manager not initialized")
    return _manager

@app.post("/cameras/{camera_id}/start")
async def start_camera(camera_id: str, background_tasks: BackgroundTasks):
    """Starts a camera in background."""
    manager = get_manager()
    background_tasks.add_task(manager.start_camera, camera_id)
    return {"status": "starting", "camera_id": camera_id}

@app.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    """Stops a camera."""
    manager = get_manager()
    await manager.stop_camera(camera_id)
    return {"status": "stopped", "camera_id": camera_id}

@app.get("/cameras/status")
async def get_cameras_status():
    """Status of all cameras."""
    manager = get_manager()
    return manager.get_status()

class CameraConfig(BaseModel):
    source: str
    source_type: str
    zones: Dict = {}

@app.post("/cameras/{camera_id}")
async def add_camera(camera_id: str, config: CameraConfig):
    """
    Adds a new camera dynamically.
    
    Body example:
    {
        "source": "https://youtube.com/...",
        "source_type": "youtube",
        "zones": {
            "zone1": {
                "polygon": [[0,0], [100,0], [100,100], [0,100]],
                "street": "Main St"
            }
        }
    }
    """
    manager = get_manager()
    
    # Create config
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "vision": {
            "source": config.source,
            "source_type": config.source_type,
            "zones": config.zones,
            "model": {"path": "yolo11n.pt", "conf_threshold": 0.5},
            "performance": {
                "detect_every_n_frames": 3,
                "opencv_buffer_size": 2,
                "target_width": 1280,
                "target_height": 720
            },
            "speed_estimation": {"enabled": True, "pixels_per_meter": 10.0},
            "persistence": {"enabled": True, "type": "csv", "interval_seconds": 60}
        }
    })
    
    manager.add_camera(camera_id, cfg)
    return {"status": "added", "camera_id": camera_id}
