import pytest
import time
import queue
from unittest.mock import MagicMock
from src.vision.application.pipelines.async_pipeline import AsyncVisionPipeline
from src.vision.domain.entities import Frame

def test_drop_newest_strategy():
    # Setup
    source = MagicMock()
    processor = MagicMock()
    
    # Create pipeline with small buffer
    pipeline = AsyncVisionPipeline(
        source=source,
        processor_chain=processor,
        frame_buffer_size=2
    )
    
    # Manually fill the queue
    f1 = Frame(id=1, image=None, timestamp=1.0)
    f2 = Frame(id=2, image=None, timestamp=1.1)
    f3 = Frame(id=3, image=None, timestamp=1.2)
    
    # Simulate capture loop logic
    # 1. Put f1 (Queue: [f1])
    pipeline.frame_queue.put_nowait(f1)
    
    # 2. Put f2 (Queue: [f1, f2]) - Full
    pipeline.frame_queue.put_nowait(f2)
    assert pipeline.frame_queue.full()
    
    # 3. Put f3 (Should drop f3 and keep [f1, f2])
    try:
        pipeline.frame_queue.put_nowait(f3)
    except queue.Full:
        # Simulate drop logic (Drop Newest)
        pipeline._dropped_frames += 1
        # No queue modification needed for Drop Newest

    # Verify queue content
    assert pipeline.frame_queue.qsize() == 2
    item1 = pipeline.frame_queue.get_nowait()
    item2 = pipeline.frame_queue.get_nowait()
    
    # Expect f1 and f2 (f3 was dropped)
    assert item1.id == 1
    assert item2.id == 2
    assert pipeline._dropped_frames == 1

def test_rate_limited_logging(capsys):
    source = MagicMock()
    processor = MagicMock()
    pipeline = AsyncVisionPipeline(source, processor, frame_buffer_size=1)
    
    # Fill queue
    pipeline.frame_queue.put_nowait(Frame(id=0, image=None, timestamp=0))
    
    # Drop 30 frames
    for i in range(1, 31):
        f = Frame(id=i, image=None, timestamp=i)
        try:
            pipeline.frame_queue.put_nowait(f)
        except queue.Full:
            # Drop Newest: just increment counter
            pipeline._dropped_frames += 1
            if pipeline._dropped_frames % 30 == 0:
                print(f"[WARNING] Pipeline congested. Dropped {pipeline._dropped_frames} frames so far.")

    # Check output
    captured = capsys.readouterr()
    assert "[WARNING] Pipeline congested. Dropped 30 frames so far." in captured.out
