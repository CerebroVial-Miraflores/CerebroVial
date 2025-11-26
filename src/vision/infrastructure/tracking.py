import numpy as np
import supervision as sv
from typing import List, Dict
from ..domain import VehicleTracker, SpeedEstimator, DetectedVehicle

class SupervisionTracker(VehicleTracker):
    """
    Wrapper around supervision's ByteTrack.
    """
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def track(self, detections: List[DetectedVehicle]) -> List[DetectedVehicle]:
        if not detections:
            return []

        # Convert to supervision Detections
        xyxy = np.array([d.bbox for d in detections])
        conf = np.array([d.confidence for d in detections])
        class_ids = np.zeros(len(detections), dtype=int) # We don't need class for tracking usually
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_ids
        )
        
        # Update tracker
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        # Map back to domain objects
        # Note: ByteTrack might return fewer detections than input if not matched
        # We need to preserve the original vehicle type if possible.
        # Since supervision doesn't store our custom data, we might need a heuristic or just assume 'car' if lost
        # Or better, we can try to match by IoU or just use the tracker's output.
        
        # Ideally, we should pass the class_id to the tracker.
        # Let's map types to IDs for the tracker
        type_map = {'car': 0, 'bus': 1, 'truck': 2, 'motorcycle': 3}
        inv_type_map = {v: k for k, v in type_map.items()}
        
        # Re-create detections with class IDs
        class_ids = np.array([type_map.get(d.type, 0) for d in detections])
        sv_detections.class_id = class_ids
        
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        results = []
        for i in range(len(tracked_detections)):
            # tracked_detections is a Detections object
            bbox = tracked_detections.xyxy[i]
            tracker_id = tracked_detections.tracker_id[i]
            class_id = tracked_detections.class_id[i]
            confidence = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 0.0
            
            vehicle = DetectedVehicle(
                id=str(tracker_id),
                type=inv_type_map.get(class_id, 'car'),
                confidence=float(confidence),
                bbox=tuple(map(int, bbox)),
                timestamp=detections[0].timestamp if detections else 0 # Approx timestamp
            )
            results.append(vehicle)
            
        return results

class SimpleSpeedEstimator(SpeedEstimator):
    """
    Estimates speed based on pixel distance traveled over time.
    """
    def __init__(self, pixels_per_meter: float = 10.0, fps: float = 30.0):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.history: Dict[str, List[tuple]] = {} # id -> [(timestamp, center_y)]
        
    def estimate(self, vehicles: List[DetectedVehicle]) -> List[DetectedVehicle]:
        current_time = vehicles[0].timestamp if vehicles else 0
        
        for vehicle in vehicles:
            if not vehicle.id:
                continue
                
            # Calculate center Y (assuming movement is primarily vertical for now, or use Euclidean)
            # Using bottom center is usually better for ground plane
            _, _, _, y2 = vehicle.bbox
            center_y = y2 
            
            if vehicle.id not in self.history:
                self.history[vehicle.id] = []
            
            self.history[vehicle.id].append((current_time, center_y))
            
            # Keep only recent history (e.g., last 1 second)
            self.history[vehicle.id] = [
                (t, y) for t, y in self.history[vehicle.id] 
                if current_time - t < 1.0
            ]
            
            if len(self.history[vehicle.id]) >= 2:
                # Calculate speed
                # Get oldest and newest point in window
                t1, y1 = self.history[vehicle.id][0]
                t2, y2 = self.history[vehicle.id][-1]
                
                time_diff = t2 - t1
                if time_diff > 0.1: # Avoid division by zero or noise
                    dist_pixels = abs(y2 - y1)
                    dist_meters = dist_pixels / self.pixels_per_meter
                    speed_mps = dist_meters / time_diff
                    speed_kmh = speed_mps * 3.6
                    
                    vehicle.speed = speed_kmh
                    
        return vehicles
