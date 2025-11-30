import os
import cv2
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.presentation.visualization.opencv_visualizer import OpenCVVisualizer
from src.vision.infrastructure.interaction import InteractiveZoneSelector
from src.vision.application.builders.pipeline_builder import VisionApplicationBuilder

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
    zones_config = {}
    if vision_cfg.zones:
        for k, v in vision_cfg.zones.items():
            if isinstance(v, list) or OmegaConf.is_list(v):
                zones_config[k] = list(v)
            else:
                zones_config[k] = list(v.polygon)
    visualizer = OpenCVVisualizer(zones_config=zones_config)

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
                    # Enter ROI selection mode
                    print("\nSelect ROI...")
                    selector = InteractiveZoneSelector(window_name="CerebroVial Vision")
                    points = selector.select_zone(frame.image)
                    if points:
                        print(f"Zone selected: {points}")
                        # Update zone manager
                        if zone_counter:
                            zone_counter.update_zone("zone1", points)
                        # Update visualizer
                        if visualizer.zones_config is None:
                            visualizer.zones_config = {}
                        visualizer.zones_config["zone1"] = points
                        if zone_counter:
                            visualizer.zones = zone_counter.zones
                        
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
