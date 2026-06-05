"""
API for managing multiple cameras.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
from ....application.services.multi_camera import MultiCameraManager
from ....infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster
from ...visualization import build_visualizer_from_vision_cfg

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
    Alta on-demand de una cámara (C1, D3): registra Y arranca el pipeline YOLO.

    Single-slot: garantiza un solo YOLO vivo a la vez (libera cualquier otra
    cámara activa). Idempotente sobre el mismo `source`. El frontend la llama al
    entrar al detalle, pasando el id real (`cam_<intersection>`) y la URL de
    Claro como `source` (`source_type: "hls"`).

    Body example:
    {
        "source": "https://.../claro/....m3u8",
        "source_type": "hls",
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
            # build_persistence() exige vision.camera_id cuando persistence está
            # habilitada (pipeline_builder); lo seteamos con el id real entrante.
            "camera_id": camera_id,
            "zones": config.zones,
            "model": {"path": "yolo11n.pt", "conf_threshold": 0.5},
            "performance": {
                # D4: 3 en el path on-demand (HLS sobre CPU). Palanca a 5 si lag.
                "detect_every_n_frames": 3,
                "opencv_buffer_size": 2,
                "target_width": 1280,
                "target_height": 720
            },
            "speed_estimation": {"enabled": True, "pixels_per_meter": 10.0},
            "persistence": {"enabled": True, "type": "csv", "interval_seconds": 60}
        }
    })

    renderer = build_visualizer_from_vision_cfg(cfg.vision)
    await manager.activate_camera(camera_id, cfg, renderer=renderer)
    return {"status": "started", "camera_id": camera_id}

@app.delete("/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    """Baja on-demand (C1, D3): para la cámara y libera su modelo YOLO."""
    manager = get_manager()
    await manager.remove_camera(camera_id)
    return {"status": "removed", "camera_id": camera_id}
