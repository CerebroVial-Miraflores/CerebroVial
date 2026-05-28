"""Unit tests for YoloDetector (CT-08.1 / CT-08.11a).

The detector is exercised with an injected fake model that mimics the
ultralytics call interface, so the suite runs without ultralytics/torch
installed and without weights on disk.
"""
from types import SimpleNamespace

import numpy as np

from src.vision.domain.entities import DetectedVehicle
from src.vision.infrastructure.detection.yolo_detector import YoloDetector


def _box(class_id, xyxy, conf):
    return SimpleNamespace(cls=[float(class_id)], xyxy=[np.array(xyxy)], conf=[float(conf)])


class FakeModel:
    """Mimics ultralytics: model(frame, ...) -> [result_with_boxes]."""

    def __init__(self, boxes):
        self._boxes = boxes

    def __call__(self, frame, verbose=False, conf=0.5):
        return [SimpleNamespace(boxes=self._boxes)]


def _detector(boxes, conf_threshold=0.5):
    return YoloDetector(model=FakeModel(boxes), conf_threshold=conf_threshold)


def test_detect_returns_list_of_domain_vehicles():
    det = _detector([_box(2, [100, 100, 200, 200], 0.9)])
    result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), frame_id=7)

    assert isinstance(result, list)
    assert len(result) == 1
    v = result[0]
    assert isinstance(v, DetectedVehicle)
    assert v.type == "car"
    assert v.confidence == 0.9
    assert v.bbox == (100, 100, 200, 200)
    assert v.id == "7_0"  # temporary id derived from frame_id + index


def test_detect_filters_non_vehicle_classes():
    det = _detector([_box(0, [10, 10, 20, 20], 0.99)])  # COCO class 0 = person
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_detect_filters_below_confidence_threshold():
    det = _detector(
        [_box(2, [0, 0, 10, 10], 0.4), _box(7, [0, 0, 10, 10], 0.8)],
        conf_threshold=0.5,
    )
    result = det.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(result) == 1
    assert result[0].type == "truck"
    assert result[0].confidence == 0.8


def test_detect_supports_all_native_vehicle_classes():
    det = _detector(
        [
            _box(2, [0, 0, 1, 1], 0.9),
            _box(3, [0, 0, 1, 1], 0.9),
            _box(5, [0, 0, 1, 1], 0.9),
            _box(7, [0, 0, 1, 1], 0.9),
        ]
    )
    result = det.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert [v.type for v in result] == ["car", "motorcycle", "bus", "truck"]


def test_injected_model_skips_weight_loading():
    # Constructs without touching ultralytics/torch or asserting weights on disk.
    det = YoloDetector(model=FakeModel([]))
    assert det.conf_threshold == 0.5
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []
