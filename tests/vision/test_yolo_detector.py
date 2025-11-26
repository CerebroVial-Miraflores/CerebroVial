import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.vision.infrastructure.yolo_detector import YoloDetector
from src.vision.domain import DetectedVehicle

@pytest.fixture
def mock_yolo():
    with patch("src.vision.infrastructure.yolo_detector.YOLO") as mock:
        yield mock

def test_yolo_detector_initialization(mock_yolo):
    detector = YoloDetector(model_path="test.pt")
    mock_yolo.assert_called_once_with("test.pt")
    assert detector.conf_threshold == 0.5

def test_yolo_detector_detect(mock_yolo):
    # Mock YOLO results
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = [2.0] # Car
    mock_box.xyxy = [np.array([100, 100, 200, 200])]
    mock_box.conf = [0.9]
    
    mock_result.boxes = [mock_box]
    mock_yolo.return_value.return_value = [mock_result]
    
    detector = YoloDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    analysis = detector.detect(frame)
    
    assert analysis.total_count == 1
    assert len(analysis.vehicles) == 1
    vehicle = analysis.vehicles[0]
    assert vehicle.type == "car"
    assert vehicle.confidence == 0.9
    assert vehicle.bbox == (100, 100, 200, 200)

def test_yolo_detector_filter_classes(mock_yolo):
    # Mock YOLO results with a person (class 0, not in target)
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = [0.0] # Person
    mock_box.xyxy = [np.array([100, 100, 200, 200])]
    mock_box.conf = [0.9]
    
    mock_result.boxes = [mock_box]
    mock_yolo.return_value.return_value = [mock_result]
    
    detector = YoloDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    analysis = detector.detect(frame)
    
    assert analysis.total_count == 0
    assert len(analysis.vehicles) == 0
