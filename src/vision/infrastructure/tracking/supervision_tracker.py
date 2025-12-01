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
        # Tune ByteTrack for lower confidence detections
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.15,  # Lower threshold to keep tracks alive
            minimum_matching_threshold=0.8,   # IoU threshold for matching
            lost_track_buffer=60,    # Keep lost tracks for 2 seconds (30fps * 2)
            frame_rate=30
        )
        # Map class ID (int) to class name (str)
        self.id_to_name = {v: k for k, v in vehicle_classes.items()}
        # Map class name (str) to class ID (int)
        self.name_to_id = vehicle_classes
        # History of class IDs for each tracker ID: {tracker_id: [class_id1, class_id2, ...]}
        self.class_history: Dict[int, List[int]] = {}

    def track(self, detections: List[DetectedVehicle]) -> List[DetectedVehicle]:
        # Convert to supervision Detections
        if detections:
            xyxy = np.array([d.bbox for d in detections])
            conf = np.array([d.confidence for d in detections])
            # Map types to IDs for the tracker
            class_ids = np.array([self.name_to_id.get(d.type, 0) for d in detections])
        else:
            xyxy = np.empty((0, 4))
            conf = np.array([])
            class_ids = np.array([])
        
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
            current_class_id = tracked_detections.class_id[i]
            confidence = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 0.0
            
            # Update class history
            if tracker_id not in self.class_history:
                self.class_history[tracker_id] = []
            self.class_history[tracker_id].append(current_class_id)
            
            # Keep history size limited (e.g., last 30 frames)
            if len(self.class_history[tracker_id]) > 30:
                self.class_history[tracker_id].pop(0)
            
            # Determine stable class using majority vote
            history = self.class_history[tracker_id]
            # Find most frequent class_id
            stable_class_id = max(set(history), key=history.count)
            
            vehicle = DetectedVehicle(
                id=str(tracker_id),
                type=self.id_to_name.get(stable_class_id, 'car'),
                confidence=float(confidence),
                bbox=tuple(map(int, bbox)),
                timestamp=detections[0].timestamp if detections else 0 # Approx timestamp
            )
            results.append(vehicle)
            
        return results
