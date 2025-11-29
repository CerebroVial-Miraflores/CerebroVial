import argparse
import sys
import os

# Add project root to sys.path to allow imports from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """
    Main entry point for the Modular Monolith application.
    """
    parser = argparse.ArgumentParser(description="CerebroVial - Modular Monolith Entry Point")
    parser.add_argument('module', choices=['vision', 'prediction', 'control'], help="Module to run")
    
    args, unknown = parser.parse_known_args()
    
    print(f"Starting module: {args.module}")
    
    if args.module == 'vision':
        print("Initializing Computer Vision Module...")
        try:
            import cv2
            import hydra
            from omegaconf import DictConfig, OmegaConf
            from src.vision.application.builders.pipeline_builder import VisionApplicationBuilder
            from src.common.logging import setup_logger
            from src.vision.presentation.visualization.opencv_visualizer import OpenCVVisualizer
            from src.vision.infrastructure.interaction import InteractiveZoneSelector
            
            # Load configuration
            vision_raw_cfg = OmegaConf.load("conf/vision/default.yaml")
            base_cfg = OmegaConf.create({"vision": vision_raw_cfg})
            
            # Merge with CLI overrides
            cli_cfg = OmegaConf.from_dotlist(unknown)
            cfg = OmegaConf.merge(base_cfg, cli_cfg)
            
            vision_cfg = cfg.vision
            
            # Build application
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
            
            # Setup Visualizer
            # Handle new zone structure (dict with polygon) or old list format
            zones_config = {}
            if vision_cfg.zones:
                for k, v in vision_cfg.zones.items():
                    if isinstance(v, list) or OmegaConf.is_list(v):
                        zones_config[k] = list(v)
                    else:
                        # It's a dict/config object with 'polygon' key
                        zones_config[k] = list(v.polygon)
            visualizer = OpenCVVisualizer(zones_config=zones_config)
            
            # Get zone counter for updates
            components = builder.get_components()
            zone_counter = components.get('zone_counter')
            
            print("Starting Vision Pipeline... Press 'q' to exit, 'r' to select ROI.")
            
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
                                
                                print("Zone updated. You can copy the points above to your config file.")
                    else:
                        # Log progress if no display
                        if frame.id % 30 == 0 and analysis:
                            print(f"Frame {frame.id}: Detected {analysis.total_count} vehicles")
                            
            except KeyboardInterrupt:
                print("\nStopping pipeline...")
            finally:
                pipeline.stop()
                cv2.destroyAllWindows()
                print("Pipeline stopped.")
                
        except Exception as e:
            print(f"Error running vision module: {e}")
            raise
    elif args.module == 'prediction':
        print("Initializing Congestion Prediction Module...")
        # TODO: Import and run prediction pipeline
    elif args.module == 'control':
        print("Initializing Control Module...")
        # TODO: Import and run control service

if __name__ == "__main__":
    main()
