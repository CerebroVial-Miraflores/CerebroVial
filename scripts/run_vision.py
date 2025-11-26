import os
import cv2
import sys
import hydra
from omegaconf import DictConfig

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.infrastructure.yolo_detector import YoloDetector
from src.vision.infrastructure.sources import create_source
from src.vision.infrastructure.visualization import OpenCVVisualizer
from src.vision.infrastructure.zones import ZoneManager
from src.vision.infrastructure.interaction import ZoneSelector
from src.vision.infrastructure.tracking import SupervisionTracker, SimpleSpeedEstimator
from src.vision.infrastructure.repositories import CSVTrafficRepository
from src.vision.application.aggregator import TrafficAggregator
from src.vision.application.pipeline import VisionPipeline

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
    
    print(f"\nPerformance Settings:")
    print(f"  Resolution: {target_width}x{target_height}" if target_width else "  Resolution: Native")
    print(f"  Buffer size: {buffer_size}")
    print(f"  Detection frequency: every {detect_every_n} frames")
    print(f"  YouTube format: {youtube_format}")

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

    # Setup Persistence
    aggregator = None
    if vision_cfg.get('persistence', {}).get('enabled', False):
        print("Initializing data persistence...")
        repo_type = vision_cfg.persistence.type
        output_dir = vision_cfg.persistence.output_dir
        interval = vision_cfg.persistence.interval_seconds
        
        if repo_type == 'csv':
            repository = CSVTrafficRepository(output_dir=output_dir)
            aggregator = TrafficAggregator(repository=repository, window_duration=interval)

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
        aggregator=aggregator,
        detect_every_n_frames=detect_every_n
    )

    print("\nStarting video processing. Press 'q' to exit.")
    
    try:
        for frame, analysis in pipeline.run():
            # Visualization
            if analysis:
                frame.image = visualizer.draw(frame.image, analysis)
            
            if vision_cfg.display:
                cv2.imshow("CerebroVial Vision", frame.image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Interactive ROI selection
                    print("Pausing for ROI selection...")
                    selector = ZoneSelector("CerebroVial Vision")
                    points = selector.select_zone(frame.image)
                    
                    if points:
                        print(f"New zone points: {points}")
                        if not zone_manager:
                            # Initialize if it didn't exist
                            zone_manager = ZoneManager({}, resolution=(frame.image.shape[1], frame.image.shape[0]))
                            pipeline.zone_manager = zone_manager
                        
                        # Update zone (defaulting to 'zone1' for single zone interaction)
                        zone_manager.update_zone("zone1", points)
                        
                        # Update visualizer config
                        if visualizer.zones_config is None:
                            visualizer.zones_config = {}
                        visualizer.zones_config["zone1"] = points
                        
                        print("Zone updated. You can copy the points above to your config file.")
            else:
                # Print progress if no display
                if frame.id % 30 == 0 and analysis:
                    print(f"Frame {frame.id}: Detected {analysis.total_count} vehicles")
                    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
