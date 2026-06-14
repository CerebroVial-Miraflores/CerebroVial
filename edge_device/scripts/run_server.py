import os
import sys
import hydra
import uvicorn
from omegaconf import DictConfig

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.presentation.api import app
from src.vision.presentation.api.routes import cameras
from src.vision.infrastructure.detection.device import select_device

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("Configuration loaded.")

    vision_cfg = cfg.vision

    # Knobs de config a nivel de instancia: cfg con override por env. Defaults
    # (auto / 15 / 640) = comportamiento actual.
    device_req = os.environ.get("VISION_DEVICE", str(vision_cfg.get("device", "auto")))
    analyze_fps = int(os.environ.get("VISION_ANALYZE_FPS", vision_cfg.get("analyze_fps", 15)))
    imgsz = int(os.environ.get("VISION_IMGSZ", vision_cfg.get("imgsz", 640)))

    # Probe de hardware UNA vez al levantar: imprime el banner (verde si hay GPU o
    # device forzado; ROJO si AUTO cae a CPU) y deja el device resuelto. NO carga el
    # modelo (solo `is_available()`), así que respeta el on-demand C1/D1: el YOLO
    # sigue cargando lazy en la 1ª cámara. `device` forzado no disponible → error
    # claro al boot (sin fallback silencioso).
    inference_device = select_device(requested=device_req)

    # Initialize Manager
    manager = cameras.get_manager()
    manager.inference_device = inference_device
    manager.analyze_fps = analyze_fps
    manager.imgsz = imgsz

    # C1 (D1): arranque on-demand. El server NO registra ni arranca cámaras al
    # iniciar — cero modelos YOLO en memoria. El frontend da de alta cada cámara
    # con `POST /cameras/{id}` (id real + URL de Claro) al entrar al detalle, y
    # la baja con `DELETE /cameras/{id}` al salir. El watchdog (E4) libera por
    # timeout si el front no cierra limpio. (Las ex-cámaras hardcodeadas
    # CAM_001..004 sobre el .mp4 local quedaron obsoletas con este modelo.)

    # 4. Start Server
    server_cfg = vision_cfg.get('server', {'host': '0.0.0.0', 'port': 8000})
    print(f"Starting server at http://{server_cfg.host}:{server_cfg.port}")

    @app.on_event("startup")
    async def startup_event():
        # Solo arranca el watchdog de auto-liberación; las cámaras son on-demand.
        manager.start_watchdog()

    uvicorn.run(app, host=server_cfg.host, port=server_cfg.port)

if __name__ == "__main__":
    main()
