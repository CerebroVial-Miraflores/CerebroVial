import pytest
import time
from unittest.mock import MagicMock, patch
from src.vision.application.processors.smart_detection import SmartDetectionProcessor
from src.vision.domain.entities import Frame, FrameAnalysis, DetectedVehicle

@pytest.fixture
def mock_detector():
    detector = MagicMock()
    return detector

@pytest.fixture
def mock_metrics():
    return MagicMock()

def test_detection_frequency(mock_detector, mock_metrics):
    processor = SmartDetectionProcessor(
        detector=mock_detector,
        detect_every_n=3,
        metrics_collector=mock_metrics,
        interpolate=False
    )
    
    # Frame 0: Should detect
    frame0 = Frame(id=0, timestamp=1.0, image=None)
    mock_detector.detect.return_value = FrameAnalysis(0, 1.0, [], 0)
    processor._process(frame0, None)
    mock_detector.detect.assert_called_once()
    mock_metrics.record_detection.assert_called_once()
    
    # Frame 1: Should skip
    frame1 = Frame(id=1, timestamp=1.1, image=None)
    processor._process(frame1, None)
    mock_detector.detect.assert_called_once() # Count shouldn't increase
    
    # Frame 2: Should skip
    frame2 = Frame(id=2, timestamp=1.2, image=None)
    processor._process(frame2, None)
    mock_detector.detect.assert_called_once()
    
    # Frame 3: Should detect
    frame3 = Frame(id=3, timestamp=1.3, image=None)
    processor._process(frame3, None)
    assert mock_detector.detect.call_count == 2

def test_interpolation_logic(mock_detector):
    processor = SmartDetectionProcessor(
        detector=mock_detector,
        detect_every_n=2,
        interpolate=True
    )
    
    # Frame 0: Detect vehicle at [0, 0, 10, 10]
    v0 = DetectedVehicle("1", "car", 0.9, (0, 0, 10, 10), 1.0)
    mock_detector.detect.return_value = FrameAnalysis(0, 1.0, [v0], 1)
    
    frame0 = Frame(id=0, timestamp=1.0, image=None)
    processor._process(frame0, None)
    
    # Frame 1: Interpolate. 
    # Since we only have 1 point, it should return the same position (fallback)
    frame1 = Frame(id=1, timestamp=1.1, image=None)
    analysis1 = processor._process(frame1, None)
    assert analysis1.vehicles[0].bbox == (0, 0, 10, 10)
    
    # Frame 2: Detect vehicle at [10, 10, 20, 20]
    v2 = DetectedVehicle("1", "car", 0.9, (10, 10, 20, 20), 1.2)
    mock_detector.detect.return_value = FrameAnalysis(2, 1.2, [v2], 1)
    
    frame2 = Frame(id=2, timestamp=1.2, image=None)
    processor._process(frame2, None)
    
    # Frame 3: Interpolate/Extrapolate
    # Trajectory has (0, [0,0,10,10]) and (2, [10,10,20,20])
    # Frame 3 is next.
    # t = (3 - 2) / (2 - 0) = 1 / 2 = 0.5
    # bbox = bbox2 + t * (bbox2 - bbox1)
    # x1 = 10 + 0.5 * (10 - 0) = 15
    # y1 = 10 + 0.5 * (10 - 0) = 15
    # x2 = 20 + 0.5 * (20 - 10) = 25
    # y2 = 20 + 0.5 * (20 - 10) = 25
    
    frame3 = Frame(id=3, timestamp=1.3, image=None)
    analysis3 = processor._process(frame3, None)
    
    assert len(analysis3.vehicles) == 1
    bbox = analysis3.vehicles[0].bbox
    assert bbox == (15, 15, 25, 25)
    assert analysis3.vehicles[0].confidence < 0.9 # Should be reduced

def test_trajectory_update(mock_detector):
    processor = SmartDetectionProcessor(mock_detector, detect_every_n=1)
    
    # Process 6 frames to fill buffer (max 5)
    for i in range(6):
        v = DetectedVehicle("1", "car", 0.9, (i, i, i+10, i+10), float(i))
        mock_detector.detect.return_value = FrameAnalysis(i, float(i), [v], 1)
        processor._process(Frame(i, float(i), None), None)
        
    assert len(processor._vehicle_trajectories["1"]) == 5
    assert processor._vehicle_trajectories["1"][-1][0] == 5 # Last frame id
    assert processor._vehicle_trajectories["1"][0][0] == 1 # First frame id (0 popped)
