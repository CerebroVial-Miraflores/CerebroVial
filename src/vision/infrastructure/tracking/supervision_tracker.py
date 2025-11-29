"""
Supervision-based vehicle tracker.
"""
import numpy as np
import supervision as sv
from typing import List, Dict
from ...domain.entities import DetectedVehicle
from ...domain.protocols import VehicleTracker

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
