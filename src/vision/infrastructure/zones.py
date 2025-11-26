import numpy as np
import supervision as sv
from typing import List, Dict, Any
from ..domain import ZoneVehicleCount, DetectedVehicle

class ZoneCounter:
    """
    Manages detection zones and counts vehicles within them.
    Uses supervision library for polygon operations.
    """
    def __init__(self, zones_config: Dict[str, List[List[int]]], resolution: tuple = (1280, 720)):
        self.zones: Dict[str, sv.PolygonZone] = {}
        self.resolution = resolution
        
        for zone_id, polygon_points in zones_config.items():
            polygon = np.array(polygon_points)
            self.zones[zone_id] = sv.PolygonZone(
                polygon=polygon
            )

    def update_zone(self, zone_id: str, points: List[List[int]], resolution: tuple = None):
        """
        Updates or adds a zone dynamically.
        """
        if resolution:
            self.resolution = resolution
            
        polygon = np.array(points)
        self.zones[zone_id] = sv.PolygonZone(
            polygon=polygon
        )

    def count_vehicles_in_zones(self, detections: List[DetectedVehicle]) -> List[ZoneVehicleCount]:
        """
        Updates zone counts based on current detections.
        """
        if not detections:
            return [ZoneVehicleCount(zone_id=zid, vehicle_count=0) for zid in self.zones]

        # Convert domain detections to supervision Detections
        # sv.Detections(xyxy=..., confidence=..., class_id=...)
        xyxy = np.array([d.bbox for d in detections])
        conf = np.array([d.confidence for d in detections])
        # We don't strictly need class_id for counting if we already filtered by vehicle type in detector
        # But let's map types back to IDs if needed, or just use 0
        class_ids = np.zeros(len(detections), dtype=int) 
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_ids
        )

        zone_counts = []
        import time
        current_time = time.time()
        
        for zone_id, zone in self.zones.items():
            # trigger returns a boolean mask of detections inside the zone
            mask = zone.trigger(detections=sv_detections)
            
            # Get indices of detections in this zone
            indices = np.where(mask)[0]
            count = len(indices)
            
            vehicles_in_zone = [detections[i] for i in indices]
            
            # Calculate metrics
            avg_speed = 0.0
            vehicle_types = {}
            vehicle_ids = []
            
            if count > 0:
                # Speed
                speeds = [v.speed for v in vehicles_in_zone if v.speed is not None]
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                
                # Types
                for v in vehicles_in_zone:
                    vehicle_types[v.type] = vehicle_types.get(v.type, 0) + 1
                    vehicle_ids.append(v.id)

            zone_counts.append(ZoneVehicleCount(
                zone_id=zone_id, 
                vehicle_count=count,
                timestamp=current_time,
                vehicles=vehicle_ids,
                avg_speed=avg_speed,
                vehicle_types=vehicle_types
            ))
            
        return zone_counts
