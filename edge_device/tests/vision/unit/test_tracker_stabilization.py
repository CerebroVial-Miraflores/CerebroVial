"""Unit tests for SupervisionTracker (CT-08.2 — identidad).

A fake ByteTrack and a fake Detections factory are injected so the suite
runs without `supervision` installed. The fake tracker echoes detections
back with a deterministic tracker_id, isolating the wrapper's own logic
(majority-vote class stabilization + history window) from ByteTrack's
IoU matching.
"""
import numpy as np

from src.vision.domain.entities import DetectedVehicle
from src.vision.infrastructure.tracking.supervision_tracker import SupervisionTracker


class FakeDetections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id

    def __len__(self):
        return len(self.xyxy)


class FakeByteTrack:
    """Echoes detections back, assigning tracker_id 1..n per call.

    A single detection always gets tracker_id=1, so it is stable across
    frames — the deterministic stand-in for ByteTrack's IoU matching.
    """

    def update_with_detections(self, dets):
        n = len(dets)
        return _Tracked(dets, np.arange(1, n + 1))


class _Tracked:
    def __init__(self, dets, tracker_id):
        self.xyxy = dets.xyxy
        self.confidence = dets.confidence
        self.class_id = dets.class_id
        self.tracker_id = tracker_id

    def __len__(self):
        return len(self.tracker_id)


def _tracker(classes):
    return SupervisionTracker(classes, tracker=FakeByteTrack(), detections_factory=FakeDetections)


def test_frame_rate_default_is_25():
    """Default = fps operativo (nativo del HLS de Claro)."""
    assert SupervisionTracker({"car": 1}, tracker=FakeByteTrack(),
                              detections_factory=FakeDetections).frame_rate == 25


def test_frame_rate_is_configurable():
    t = SupervisionTracker({"car": 1}, tracker=FakeByteTrack(),
                           detections_factory=FakeDetections, frame_rate=10)
    assert t.frame_rate == 10


def test_class_stabilization_majority_vote():
    tracker = _tracker({"car": 1, "truck": 2})
    bbox = (100, 100, 200, 200)

    res1 = tracker.update([DetectedVehicle("0", "car", 0.9, bbox, 1.0)])
    assert res1[0].type == "car"
    assert tracker.class_history[1] == [1]

    tracker.update([DetectedVehicle("0", "car", 0.9, bbox, 1.1)])
    assert tracker.class_history[1] == [1, 1]

    # Truck noise: history [1,1,2] -> majority still car.
    res3 = tracker.update([DetectedVehicle("0", "truck", 0.8, bbox, 1.2)])
    assert res3[0].type == "car"
    assert tracker.class_history[1] == [1, 1, 2]

    tracker.update([DetectedVehicle("0", "truck", 0.8, bbox, 1.3)])  # [1,1,2,2]
    res5 = tracker.update([DetectedVehicle("0", "car", 0.9, bbox, 1.4)])  # [1,1,2,2,1]
    assert res5[0].type == "car"


def test_output_is_domain_entity_with_stable_id():
    tracker = _tracker({"car": 1})
    out = tracker.update([DetectedVehicle("temp", "car", 0.9, (10, 10, 50, 50), 5.0)])

    assert len(out) == 1
    v = out[0]
    assert isinstance(v, DetectedVehicle)
    assert v.id == "1"  # tracker-assigned, replaces the temporary id
    assert v.bbox == (10, 10, 50, 50)
    assert v.timestamp == 5.0


def test_history_window_is_capped():
    tracker = _tracker({"car": 1})
    bbox = (100, 100, 200, 200)
    for i in range(35):
        tracker.update([DetectedVehicle("0", "car", 0.9, bbox, float(i))])

    tracker_id = next(iter(tracker.class_history))
    assert len(tracker.class_history[tracker_id]) == 30


def test_empty_detections_returns_empty():
    tracker = _tracker({"car": 1})
    assert tracker.update([]) == []
