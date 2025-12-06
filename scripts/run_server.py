import os
import sys
import hydra
import uvicorn
from omegaconf import DictConfig

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.presentation.api import app
from src.vision.presentation.api.routes import cameras

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Configuration loaded.")
    
    vision_cfg = cfg.vision
    
    # Initialize Manager
    manager = cameras.get_manager()
    
    # Define cameras
    CAMERAS = [
        {"id": "CAM_001", "source": "https://www.youtube.com/watch?v=6dp-bvQ7RWo"},
        {"id": "CAM_002", "source": "https://www.youtube.com/watch?v=ByED80IKdIU"},
        {"id": "CAM_003", "source": "https://www.youtube.com/watch?v=0IgonpX1jMg"},
        {"id": "CAM_004", "source": "https://www.youtube.com/watch?v=rPxWUFTKgds"},
    ]
    
    # Add cameras
    for cam_info in CAMERAS:
        try:
            # Create a copy of the config for this camera
            cam_cfg = cfg.copy()
            cam_cfg.vision.source = cam_info["source"]
            
            manager.add_camera(cam_info["id"], cam_cfg)
        except Exception as e:
            print(f"Error adding camera {cam_info['id']}: {e}")

    # 4. Start Server
    server_cfg = vision_cfg.get('server', {'host': '0.0.0.0', 'port': 8000})
    print(f"Starting server at http://{server_cfg.host}:{server_cfg.port}")
    
    # Add startup event to start all cameras
    @app.on_event("startup")
    async def startup_event():
        try:
            print("Starting all cameras...")
            await manager.start_all()
        except Exception as e:
            print(f"Failed to start cameras: {e}")

    uvicorn.run(app, host=server_cfg.host, port=server_cfg.port)

if __name__ == "__main__":
    main()
