import pytest
from unittest.mock import Mock, MagicMock
from src.vision.application.pipelines.sync_pipeline import VisionPipeline
from src.vision.domain.protocols import FrameProducer
from src.vision.domain.entities import Frame
from src.vision.application.processors import FrameProcessor

def test_pipeline_initialization():
    source = Mock(spec=FrameProducer)
    processor = Mock(spec=FrameProcessor)
    pipeline = VisionPipeline(source, processor)
    
    assert pipeline.source == source
    assert pipeline.processor_chain == processor

def test_pipeline_run(mock_frame, mock_analysis):
    source = Mock(spec=FrameProducer)
    source.__iter__ = Mock(return_value=iter([mock_frame]))
    
    processor = Mock(spec=FrameProcessor)
    processor.process.return_value = mock_analysis
    
    pipeline = VisionPipeline(source, processor)
    
    results = list(pipeline.run())
    
    assert len(results) == 1
    frame, analysis = results[0]
    assert frame == mock_frame
    assert analysis == mock_analysis
    processor.process.assert_called_once_with(mock_frame, None)

def test_pipeline_chain_delegation(mock_frame, mock_analysis):
    # Create 3 frames
    frames = [
        Frame(id=0, timestamp=1.0, image=mock_frame.image),
        Frame(id=1, timestamp=2.0, image=mock_frame.image),
        Frame(id=2, timestamp=3.0, image=mock_frame.image)
    ]
    
    source = Mock(spec=FrameProducer)
    source.__iter__ = Mock(return_value=iter(frames))
    
    processor = Mock(spec=FrameProcessor)
    # Return analysis for each call
    processor.process.side_effect = [mock_analysis, mock_analysis, mock_analysis]
    
    pipeline = VisionPipeline(source, processor)
    
    results = list(pipeline.run())
    
    assert len(results) == 3
    assert processor.process.call_count == 3
    
    # Check that previous analysis is passed to process (though in this mock it's None for first call)
    # The pipeline passes the result of previous process call to the next one
    # But here we are iterating frames.
    # In run(): analysis = self.processor_chain.process(frame, analysis)
    # So for frame 0: process(frame0, None) -> analysis0
    # For frame 1: process(frame1, analysis0) -> analysis1
    
    # Let's verify calls arguments
    calls = processor.process.call_args_list
    assert calls[0][0][0] == frames[0]
    assert calls[0][0][1] is None
    
    assert calls[1][0][0] == frames[1]
    assert calls[1][0][1] == mock_analysis
    
    assert calls[2][0][0] == frames[2]
    assert calls[2][0][1] == mock_analysis
