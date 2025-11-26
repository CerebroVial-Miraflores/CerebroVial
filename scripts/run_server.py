import os
import sys
import hydra
import uvicorn
from omegaconf import DictConfig

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.infrastructure.visualization import OpenCVVisualizer
from src.vision.application.builder import VisionApplicationBuilder
from src.vision.presentation.api import app, set_pipeline

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Configuration:\n{cfg}")
    
    vision_cfg = cfg.vision
    
    # Use Builder to construct application
    builder = VisionApplicationBuilder(cfg)
    pipeline = (
        builder
        .build_detector()
        .build_tracker()
        .build_speed_estimator()
        .build_zones()
        .build_persistence()
        .build_source()
        .build_pipeline()
    )
    
    components = builder.get_components()
    zone_counter = components['zone_counter']
    
    # Setup Visualizer
    zones_config = {k: list(v) for k, v in vision_cfg.zones.items()} if vision_cfg.zones else {}
    visualizer = OpenCVVisualizer(zones_config=zones_config)
    
    # 3. Setup API
    set_pipeline(pipeline, visualizer)
    
    # 4. Start Server
    server_cfg = vision_cfg.get('server', {'host': '0.0.0.0', 'port': 8000})
    print(f"Starting server at http://{server_cfg.host}:{server_cfg.port}")
    uvicorn.run(app, host=server_cfg.host, port=server_cfg.port)

if __name__ == "__main__":
    main()
