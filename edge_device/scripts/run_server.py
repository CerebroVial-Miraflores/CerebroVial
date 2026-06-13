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

    # Probe de hardware UNA vez al levantar: imprime el banner (verde si hay
    # GPU; ROJO si cae a CPU) y deja el device resuelto. NO carga el modelo
    # (solo `is_available()`), así que respeta el on-demand C1/D1: el YOLO sigue
    # cargando lazy en la 1ª cámara, con este device inyectado.
    inference_device = select_device()

    # Initialize Manager
    manager = cameras.get_manager()
    manager.inference_device = inference_device

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
