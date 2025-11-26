import os
import sys
import hydra
import uvicorn
from omegaconf import DictConfig

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.infrastructure.yolo_detector import YoloDetector
from src.vision.infrastructure.sources import create_source
from src.vision.infrastructure.visualization import OpenCVVisualizer
from src.vision.infrastructure.zones import ZoneManager
from src.vision.infrastructure.tracking import SupervisionTracker, SimpleSpeedEstimator
from src.vision.application.pipeline import VisionPipeline
from src.vision.presentation.api import app, set_pipeline

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Configuration:\n{cfg}")
    
    vision_cfg = cfg.vision
    
    # 1. Setup Infrastructure
    print(f"Loading model: {vision_cfg.model.path}...")
    detector = YoloDetector(
        model_path=vision_cfg.model.path, 
        conf_threshold=vision_cfg.model.conf_threshold
    )
    
    # Get performance settings
    perf_cfg = vision_cfg.get('performance', {})
    target_width = perf_cfg.get('target_width', None)
    target_height = perf_cfg.get('target_height', None)
    buffer_size = perf_cfg.get('opencv_buffer_size', 3)
    detect_every_n = perf_cfg.get('detect_every_n_frames', 3)
    youtube_format = perf_cfg.get('youtube_format', 'best')
    
    # Setup Zones
    zone_manager = None
    zones_config = None
    if 'zones' in vision_cfg and vision_cfg.zones:
        print("Initializing zones...")
        # Convert OmegaConf to dict
        zones_config = {k: list(v) for k, v in vision_cfg.zones.items()}
        zone_manager = ZoneManager(zones_config, resolution=(target_width or 1280, target_height or 720))

    visualizer = OpenCVVisualizer(zones_config=zones_config)
    
    # Setup Tracking & Speed
    tracker = None
    speed_estimator = None
    if vision_cfg.get('speed_estimation', {}).get('enabled', False):
        print("Initializing tracking and speed estimation...")
        tracker = SupervisionTracker()
        pixels_per_meter = vision_cfg.speed_estimation.pixels_per_meter
        speed_estimator = SimpleSpeedEstimator(pixels_per_meter=pixels_per_meter)

    print(f"\nOpening source: {vision_cfg.source} (Type: {vision_cfg.source_type})...")
    try:
        source = create_source(
            vision_cfg.source,
            source_type=vision_cfg.source_type,
            target_width=target_width,
            target_height=target_height,
            buffer_size=buffer_size,
            format=youtube_format # For YouTube
        )
    except Exception as e:
        print(f"Failed to open source: {e}")
        return

    # 2. Setup Application
    pipeline = VisionPipeline(
        source=source,
        detector=detector,
        tracker=tracker,
        speed_estimator=speed_estimator,
        zone_manager=zone_manager,
        detect_every_n_frames=detect_every_n
    )
    
    # 3. Setup API
    set_pipeline(pipeline, visualizer)
    
    # 4. Start Server
    server_cfg = vision_cfg.get('server', {'host': '0.0.0.0', 'port': 8000})
    print(f"Starting server at http://{server_cfg.host}:{server_cfg.port}")
    uvicorn.run(app, host=server_cfg.host, port=server_cfg.port)

if __name__ == "__main__":
    main()
