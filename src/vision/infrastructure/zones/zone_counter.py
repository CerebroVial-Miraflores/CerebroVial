import cv2
import time
import numpy as np
import supervision as sv
from typing import List, Dict, Tuple, Any
from shapely.geometry import Polygon, Point
from ...domain.entities import DetectedVehicle, ZoneVehicleCount, FrameAnalysis

class ZoneCounter:
    """
    Manages detection zones and counts vehicles within them.
    Uses supervision library for polygon operations.
    """
    def __init__(self, zones_config: Dict[str, Any], resolution: tuple = (1280, 720)):
        self.zones: Dict[str, sv.PolygonZone] = {}
        self.zone_metadata: Dict[str, dict] = {}
        self.zone_areas: Dict[str, float] = {} # Cache for zone areas
        self.resolution = resolution
        
        for zone_id, config in zones_config.items():
            # Support both old list format and new dict format
            if isinstance(config, list):
                polygon = np.array(config)
                metadata = {"camera_id": "unknown", "street": "unknown"}
            else:
                polygon = np.array(config['polygon'])
                metadata = {
                    "camera_id": config.get('camera_id', 'unknown'),
                    "street": config.get('street', 'unknown')
                }
            
            self.zones[zone_id] = sv.PolygonZone(
                polygon=polygon
            )
            self.zone_metadata[zone_id] = metadata
            
            # Pre-calculate zone area
            self.zone_areas[zone_id] = cv2.contourArea(polygon)

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
        # Preserve existing metadata if updating, or set default
        if zone_id not in self.zone_metadata:
             self.zone_metadata[zone_id] = {"camera_id": "unknown", "street": "unknown"}
             
        # Update cached area
        self.zone_areas[zone_id] = cv2.contourArea(polygon)

    def count_vehicles_in_zones(self, detections: List[DetectedVehicle]) -> List[ZoneVehicleCount]:
        """
        Updates zone counts based on current detections.
        """
        if not detections:
            return [
                ZoneVehicleCount(
                    zone_id=zid, 
                    vehicle_count=0,
                    camera_id=self.zone_metadata[zid]['camera_id'],
                    street_monitored=self.zone_metadata[zid]['street']
                ) for zid in self.zones
            ]

        # Convert domain detections to supervision Detections
        xyxy = np.array([d.bbox for d in detections])
        conf = np.array([d.confidence for d in detections])
        class_ids = np.zeros(len(detections), dtype=int) 
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_ids
        )

        zone_counts = []
        current_time = time.time()
        
        for zone_id, zone in self.zones.items():
            # trigger returns a boolean mask of detections inside the zone
            mask = zone.trigger(detections=sv_detections)
            
            # Get indices of detections in this zone
            indices = np.where(mask)[0]
            count = len(indices)
            
            # Calculate metrics
            avg_speed = 0.0
            vehicle_types = {}
            vehicle_ids = []
            vehicle_details = {}
            occupancy = 0.0
            
            if count > 0:
                vehicles_in_zone = [detections[i] for i in indices]
                
                # Speed
                speeds = [v.speed for v in vehicles_in_zone if v.speed is not None]
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                
                # Types and IDs
                # Optimize: Single pass for types and IDs
                vehicle_details = {}
                for v in vehicles_in_zone:
                    vehicle_types[v.type] = vehicle_types.get(v.type, 0) + 1
                    vehicle_ids.append(v.id)
                    vehicle_details[v.id] = v.type
                
                # Occupancy (Geometric Estimation)
                # Use cached zone area
                zone_area = self.zone_areas.get(zone_id, 0.0)
                
                if zone_area > 0:
                    # Calculate total vehicle area
                    # Vectorized calculation if possible, but list comprehension is fast enough for small N
                    total_vehicle_area = sum(
                        (v.bbox[2] - v.bbox[0]) * (v.bbox[3] - v.bbox[1]) 
                        for v in vehicles_in_zone
                    )
                    occupancy = min(total_vehicle_area / zone_area, 1.0)

            metadata = self.zone_metadata[zone_id]
            zone_counts.append(ZoneVehicleCount(
                zone_id=zone_id, 
                vehicle_count=count,
                timestamp=current_time,
                vehicles=vehicle_ids,
                vehicle_details=vehicle_details,
                avg_speed=avg_speed,
                occupancy=occupancy,
                vehicle_types=vehicle_types,
                camera_id=metadata['camera_id'],
                street_monitored=metadata['street']
            ))
            
        return zone_counts
