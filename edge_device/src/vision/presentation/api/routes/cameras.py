"""
API for managing multiple cameras.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from typing import Optional, Dict
from ....application.services.multi_camera import MultiCameraManager
from ....infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster
from ...visualization import build_visualizer_from_vision_cfg

app = FastAPI()


def _build_camera_config(
    camera_id: str, source: str, source_type: str, zones: Dict
) -> DictConfig:
    """Arma el cfg del alta on-demand (C1).

    Extraído del handler para que la config sea inspeccionable por tests (el
    bug del `persistence.type` viejo vivía justo acá, en config a mano que
    ningún test miraba).

    - persistence.type = "postgres": CSV fue eliminado en Fase 5b (decisión #8);
      build_persistence() solo acepta postgres.
    - persistence.interval_seconds = 5: ventana corta (igual que default.yaml)
      para que el conteo/velocidad de las tarjetas (SSE por-ventana) aparezca en
      segundos, no al cerrar una ventana de 60s.
    - camera_id seteado: build_persistence() lo exige con persistence.enabled.
    """
    # Cada campo está anotado con su intención vs conf/vision/default.yaml
    # (match-default o override-live) para que una divergencia accidental salte
    # en code review — fue así como se colaron csv/interval/conf/n.
    return OmegaConf.create({
        "vision": {
            "source": source,
            "source_type": source_type,
            "camera_id": camera_id,
            "zones": zones,
            "model": {
                "path": "yolo11n.pt",  # match default.yaml
                # OVERRIDE-LIVE: 0.2 (default.yaml usa 0.3). yolo11n sobre cámara
                # de tráfico de ángulo alto (vehículos chicos/lejanos) capturaba
                # casi nada a 0.5/0.3; 0.2 detecta más para la demo, asumiendo
                # algún falso positivo.
                "conf_threshold": 0.2,
            },
            "performance": {
                # OVERRIDE-LIVE: 1 — elegido PARA LA DEMO. Con n=1 YOLO infiere en
                # cada frame, así que los boxes se repintan constantemente: máxima
                # evidencia visual de que la detección corre. Trade-off conocido y
                # asumido: yolo11n en CPU (Docker en Mac no accede a MPS) infiere
                # más lento que el stream HLS 720p, así que el reloj de la cámara
                # SALTA y hay más lag (DEUDA-YOLO-CPU). No resuelve el freeze del
                # <img> MJPEG en el navegador (DEUDA-MJPEG-BROWSER: el src del <img>
                # ya es estable, la causa es browser-side y no hay reconexión); n=1
                # solo lo atenúa al repintar más seguido. Palanca: volver a n>=2 si
                # se prioriza fluidez sobre evidencia — ahí revive la persistencia
                # visual de boxes en skip frames (visual-only, no agrega inferencias).
                "detect_every_n_frames": 1,
                # OVERRIDE-LIVE: buffer chico = menos lag en stream en vivo
                # (default.yaml usa 30, tuneado para batch/archivo).
                "opencv_buffer_size": 2,
                "target_width": 1280,   # match default.yaml
                "target_height": 720,   # match default.yaml
                # frame_buffer_size se omite a propósito → toma el default de
                # código (10), también favorable a live (menos lag que los 30 de
                # default.yaml). result_buffer_size/target_fps/youtube_format
                # toman sus defaults de código, que ya coinciden con default.yaml.
            },
            "speed_estimation": {"enabled": True, "pixels_per_meter": 10.0},  # match default.yaml
            # match default.yaml (alineado en el fix anterior; CSV eliminado, decisión #8).
            "persistence": {"enabled": True, "type": "postgres", "interval_seconds": 5},
        }
    })

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

    cfg = _build_camera_config(camera_id, config.source, config.source_type, config.zones)
    renderer = build_visualizer_from_vision_cfg(cfg.vision)
    await manager.activate_camera(camera_id, cfg, renderer=renderer)
    return {"status": "started", "camera_id": camera_id}

@app.delete("/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    """Baja on-demand (C1, D3): para la cámara y libera su modelo YOLO."""
    manager = get_manager()
    await manager.remove_camera(camera_id)
    return {"status": "removed", "camera_id": camera_id}
