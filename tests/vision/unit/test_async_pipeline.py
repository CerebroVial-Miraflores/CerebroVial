import pytest
import threading
import queue
import time
from unittest.mock import MagicMock, patch
from src.vision.application.pipelines.async_pipeline import AsyncVisionPipeline
from src.vision.domain.entities import Frame, FrameAnalysis

class MockSource:
    def __init__(self, frames):
        self.frames = frames
        self.released = False

    def __iter__(self):
        for frame in self.frames:
            yield frame
            time.sleep(0.01) # Simulate delay

    def release(self):
        self.released = True

class MockProcessor:
    def process(self, frame, analysis):
        return FrameAnalysis(
            frame_id=frame.id,
            timestamp=frame.timestamp,
            vehicles=[],
            total_count=0
        )

def test_pipeline_initialization():
    source = MagicMock()
    processor = MagicMock()
    pipeline = AsyncVisionPipeline(source, processor)
    assert pipeline.frame_queue.maxsize == 10
    assert pipeline.result_queue.maxsize == 30

def test_pipeline_start_stop():
    source = MagicMock()
    # Make source infinite so thread stays alive
    source.__iter__.return_value = iter(lambda: time.sleep(0.1) or Frame(1, 1.0, None), 1)
    
    processor = MagicMock()
    pipeline = AsyncVisionPipeline(source, processor)
    
    pipeline.start()
    # Give threads time to start
    time.sleep(0.1)
    assert pipeline._capture_thread.is_alive()
    assert pipeline._processing_thread.is_alive()
    
    pipeline.stop()
    # Give threads time to stop
    time.sleep(0.2)
    assert not pipeline._capture_thread.is_alive()
    assert not pipeline._processing_thread.is_alive()

def test_pipeline_processing_flow():
    # Setup
    frames = [
        Frame(id=1, timestamp=1.0, image=None),
        Frame(id=2, timestamp=2.0, image=None)
    ]
    source = MockSource(frames)
    processor = MockProcessor()
    pipeline = AsyncVisionPipeline(source, processor)
    
    # Run
    results = []
    
    # We run the generator in a way that we can break out
    gen = pipeline.run()
    try:
        for _ in range(2):
            result = next(gen)
            results.append(result)
    except StopIteration:
        pass
    finally:
        pipeline.stop()

    # Verify
    assert len(results) == 2
    assert results[0][0].id == 1
    assert results[1][0].id == 2
    assert source.released

def test_pipeline_stop_event_propagation():
    """Verify that stopping the pipeline stops the capture loop even if source is infinite."""
    source = MagicMock()
    source.__iter__.return_value = iter([Frame(id=i, timestamp=i, image=None) for i in range(1000)])
    processor = MagicMock()
    
    pipeline = AsyncVisionPipeline(source, processor)
    pipeline.start()
    
    time.sleep(0.1)
    pipeline.stop()
    
    assert not pipeline._capture_thread.is_alive()
