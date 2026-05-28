"""
Supervision-based vehicle tracker.
"""
from typing import Dict, List

import numpy as np

from ...domain.entities import DetectedVehicle
from ...domain.protocols import VehicleTracker

_HISTORY_LIMIT = 30


class SupervisionTracker(VehicleTracker):
    """Wrapper around supervision's ByteTrack.

    `tracker` and `detections_factory` are injectable so the module collects
    and tests run without `supervision` installed. When `tracker` is None the
    real ByteTrack (and `sv.Detections` as the factory) are imported lazily.
    """

    def __init__(self, vehicle_classes: Dict[str, int], tracker=None, detections_factory=None):
        if tracker is None:
            import supervision as sv

            tracker = sv.ByteTrack(
                track_activation_threshold=0.15,  # keep lower-confidence tracks alive
                minimum_matching_threshold=0.8,   # IoU threshold for matching
                lost_track_buffer=60,             # ~2s at 30fps
                frame_rate=30,
            )
            if detections_factory is None:
                detections_factory = sv.Detections

        self.tracker = tracker
        self._detections_factory = detections_factory
        # class id (int) <-> class name (str)
        self.id_to_name = {v: k for k, v in vehicle_classes.items()}
        self.name_to_id = vehicle_classes
        # tracker_id -> recent class ids, for majority-vote stabilization
        self.class_history: Dict[int, List[int]] = {}

    def update(self, detections: List[DetectedVehicle]) -> List[DetectedVehicle]:
        if detections:
            xyxy = np.array([d.bbox for d in detections])
            conf = np.array([d.confidence for d in detections])
            class_ids = np.array([self.name_to_id.get(d.type, 0) for d in detections])
        else:
            xyxy = np.empty((0, 4))
            conf = np.array([])
            class_ids = np.array([])

        sv_detections = self._detections_factory(xyxy=xyxy, confidence=conf, class_id=class_ids)
        tracked = self.tracker.update_with_detections(sv_detections)

        results: List[DetectedVehicle] = []
        for i in range(len(tracked)):
            bbox = tracked.xyxy[i]
            tracker_id = tracked.tracker_id[i]
            current_class_id = tracked.class_id[i]
            confidence = tracked.confidence[i] if tracked.confidence is not None else 0.0

            history = self.class_history.setdefault(tracker_id, [])
            history.append(current_class_id)
            if len(history) > _HISTORY_LIMIT:
                history.pop(0)

            # Stable class via majority vote over the recent window.
            stable_class_id = max(set(history), key=history.count)

            results.append(
                DetectedVehicle(
                    id=str(tracker_id),
                    type=self.id_to_name.get(stable_class_id, "car"),
                    confidence=float(confidence),
                    bbox=tuple(map(int, bbox)),
                    timestamp=detections[0].timestamp if detections else 0,
                )
            )

        return results
