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
from src.vision.infrastructure.zones import ZoneCounter
from src.vision.infrastructure.tracking import SupervisionTracker, SimpleSpeedEstimator
from src.vision.infrastructure.repositories import CSVTrafficRepository
from src.vision.application.aggregator import TrafficDataAggregator
from src.vision.application.pipeline import VisionPipeline
from src.common.metrics import MetricsCollector
from src.vision.presentation.api import app, set_pipeline
from src.vision.application.processors import (
    DetectionProcessor, TrackingProcessor, SpeedEstimationProcessor, 
    ZoneProcessor, AggregationProcessor
)

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
    print("Initializing zones...")
    zones_config = {k: list(v) for k, v in vision_cfg.zones.items()} if vision_cfg.zones else {}
    
    zone_counter = ZoneCounter(zones_config, resolution=(target_width or 1280, target_height or 720))
    visualizer = OpenCVVisualizer(zones_config=zones_config)

    # Setup Tracking & Speed (Optional)
    print("Initializing tracking and speed estimation...")
    tracker = SupervisionTracker()
    
    speed_estimator = None
    if vision_cfg.speed_estimation.enabled:
        pixels_per_meter = vision_cfg.speed_estimation.pixels_per_meter
        speed_estimator = SimpleSpeedEstimator(pixels_per_meter=pixels_per_meter)

    # Setup Persistence
    aggregator = None
    if vision_cfg.get('persistence', {}).get('enabled', False):
        print("Initializing data persistence...")
        repo_type = vision_cfg.persistence.type
        output_dir = vision_cfg.persistence.output_dir
        interval = vision_cfg.persistence.interval_seconds
        
        if repo_type == 'csv':
            repository = CSVTrafficRepository(output_dir=output_dir)
            aggregator = TrafficDataAggregator(repository=repository, window_duration=interval)

    print(f"\nOpening source: {vision_cfg.source} (Type: {vision_cfg.source_type})...")
    try:
        source = create_source(
            source_config=vision_cfg.source,
            source_type=vision_cfg.source_type,
            buffer_size=vision_cfg.performance.opencv_buffer_size,
            target_width=vision_cfg.performance.target_width,
            target_height=vision_cfg.performance.target_height,
            format=vision_cfg.performance.youtube_format
        )
    except Exception as e:
        print(f"Failed to open source: {e}")
        return

    # Setup Metrics
    metrics_collector = MetricsCollector()

    # 2. Setup Application (Chain of Responsibility)
    # Base processor: Detection
    processor_chain = DetectionProcessor(detector, detect_every_n=detect_every_n, metrics_collector=metrics_collector)
    current_link = processor_chain
    
    # Add Tracking
    if tracker:
        tracking_processor = TrackingProcessor(tracker, metrics_collector=metrics_collector)
        current_link.set_next(tracking_processor)
        current_link = tracking_processor
        
    # Add Speed Estimation
    if speed_estimator:
        speed_processor = SpeedEstimationProcessor(speed_estimator)
        current_link.set_next(speed_processor)
        current_link = speed_processor
        
    # Add Zones
    if zone_counter:
        zone_processor = ZoneProcessor(zone_counter)
        current_link.set_next(zone_processor)
        current_link = zone_processor
        
    # Add Aggregation
    if aggregator:
        agg_processor = AggregationProcessor(aggregator)
        current_link.set_next(agg_processor)
        current_link = agg_processor

    pipeline = VisionPipeline(
        source=source,
        processor_chain=processor_chain,
        metrics_collector=metrics_collector
    )
    
    # 3. Setup API
    set_pipeline(pipeline, visualizer)
    
    # 4. Start Server
    server_cfg = vision_cfg.get('server', {'host': '0.0.0.0', 'port': 8000})
    print(f"Starting server at http://{server_cfg.host}:{server_cfg.port}")
    uvicorn.run(app, host=server_cfg.host, port=server_cfg.port)

if __name__ == "__main__":
    main()
