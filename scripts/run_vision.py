import os
import cv2
import sys
import os
import hydra
from omegaconf import DictConfig

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.infrastructure.yolo_detector import YoloDetector
from src.vision.infrastructure.video_source import VideoSource

def draw_results(frame, analysis):
    """
    Draw bounding boxes and labels on the frame.
    """
    for vehicle in analysis.vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        label = f"{vehicle.type} {vehicle.confidence:.2f}"
        
        # Color based on type
        color = (0, 255, 0) # Green default
        if vehicle.type == 'car': color = (0, 255, 0)
        elif vehicle.type == 'bus': color = (0, 165, 255) # Orange
        elif vehicle.type == 'truck': color = (0, 0, 255) # Red
        elif vehicle.type == 'motorcycle': color = (255, 255, 0) # Cyan
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw total count
    cv2.putText(frame, f"Vehicles: {analysis.total_count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Configuration:\n{cfg}")
    
    vision_cfg = cfg.vision
    
    print(f"Loading model: {vision_cfg.model.path}...")
    detector = YoloDetector(
        model_path=vision_cfg.model.path, 
        conf_threshold=vision_cfg.model.conf_threshold
    )
    
    print(f"Opening source: {vision_cfg.source} (Type: {vision_cfg.source_type})...")
    try:
        source = VideoSource(str(vision_cfg.source))
    except Exception as e:
        print(f"Failed to open source: {e}")
        return

    print("Starting detection. Press 'q' to exit.")
    
    try:
        for frame_id, frame in source:
            # Detect
            analysis = detector.detect(frame, frame_id)
            
            # Draw
            frame = draw_results(frame, analysis)
            
            if vision_cfg.display:
                cv2.imshow("CerebroVial Vision", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Just print progress if no display
                if frame_id % 30 == 0:
                    print(f"Frame {frame_id}: Detected {analysis.total_count} vehicles")
                    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        source.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
