"""
Simple speed estimator.
"""
from typing import List, Dict
from ...domain.entities import DetectedVehicle
from ...domain.protocols import SpeedEstimator

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
