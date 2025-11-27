import numpy as np
import supervision as sv
from typing import List, Dict
from ..domain import VehicleTracker, SpeedEstimator, DetectedVehicle

class SupervisionTracker(VehicleTracker):
    """
    Wrapper around supervision's ByteTrack.
    """
    def __init__(self, vehicle_classes: Dict[str, int]):
        self.tracker = sv.ByteTrack()
        # Map class ID (int) to class name (str)
        self.id_to_name = {v: k for k, v in vehicle_classes.items()}
        # Map class name (str) to class ID (int)
        self.name_to_id = vehicle_classes

    def track(self, detections: List[DetectedVehicle]) -> List[DetectedVehicle]:
        if not detections:
            return []

        # Convert to supervision Detections
        xyxy = np.array([d.bbox for d in detections])
        conf = np.array([d.confidence for d in detections])
        
        # Map types to IDs for the tracker
        class_ids = np.array([self.name_to_id.get(d.type, 0) for d in detections])
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_ids
        )
        
        # Update tracker
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
                type=self.id_to_name.get(class_id, 'car'),
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
