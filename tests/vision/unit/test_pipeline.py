import pytest
from unittest.mock import Mock, MagicMock
from src.vision.application.pipeline import VisionPipeline
from src.vision.domain import FrameProducer, VehicleDetector, Frame

def test_pipeline_initialization():
    source = Mock(spec=FrameProducer)
    detector = Mock(spec=VehicleDetector)
    pipeline = VisionPipeline(source, detector)
    
    assert pipeline.source == source
    assert pipeline.detector == detector
    assert pipeline.detect_every_n_frames == 1

def test_pipeline_run(mock_frame, mock_analysis):
    source = Mock(spec=FrameProducer)
    source.__iter__ = Mock(return_value=iter([mock_frame]))
    
    detector = Mock(spec=VehicleDetector)
    detector.detect.return_value = mock_analysis
    
    pipeline = VisionPipeline(source, detector)
    
    results = list(pipeline.run())
    
    assert len(results) == 1
    frame, analysis = results[0]
    assert frame == mock_frame
    assert analysis == mock_analysis
    detector.detect.assert_called_once()

def test_pipeline_frame_skipping(mock_frame, mock_analysis):
    # Create 3 frames
    frames = [
        Frame(id=0, timestamp=1.0, image=mock_frame.image),
        Frame(id=1, timestamp=2.0, image=mock_frame.image),
        Frame(id=2, timestamp=3.0, image=mock_frame.image)
    ]
    
    source = Mock(spec=FrameProducer)
    source.__iter__ = Mock(return_value=iter(frames))
    
    detector = Mock(spec=VehicleDetector)
    detector.detect.return_value = mock_analysis
    
    # Detect every 2 frames
    pipeline = VisionPipeline(source, detector, detect_every_n_frames=2)
    
    results = list(pipeline.run())
    
    assert len(results) == 3
    
    # Frame 0: Detect called
    # Frame 1: Detect NOT called (skipped)
    # Frame 2: Detect called
    
    assert detector.detect.call_count == 2
    
    # Check that analysis is carried over for skipped frames
    assert results[0][1] == mock_analysis # Frame 0
    assert results[1][1] == mock_analysis # Frame 1 (reused)
    assert results[2][1] == mock_analysis # Frame 2
