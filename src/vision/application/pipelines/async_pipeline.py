"""
Asynchronous pipeline with threading to process frames without blocking.
Uses queues to decouple production, processing, and persistence.
"""
import queue
import threading
import time
from typing import Iterator, Tuple, Optional
from ...domain.protocols import FrameProducer
from ...domain.entities import FrameAnalysis, Frame
from ..processors import FrameProcessor
from ....common.metrics import MetricsCollector

class AsyncVisionPipeline:
    """
    Asynchronous pipeline that decouples:
    1. Frame capture (source thread)
    2. Processing (processing thread)
    """
    
    def __init__(
        self, 
        source: FrameProducer, 
        processor_chain: FrameProcessor,
        metrics_collector: Optional[MetricsCollector] = None,
        frame_buffer_size: int = 10,  # Avoid memory overflow
        result_buffer_size: int = 30
    ):
        self.source = source
        self.processor_chain = processor_chain
        self.metrics_collector = metrics_collector
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=frame_buffer_size)
        self.result_queue = queue.Queue(maxsize=result_buffer_size)
        
        # Thread control
        self._stop_event = threading.Event()
        self._processing_thread = None
        self._capture_thread = None
        
        # Shared state (thread-safe)
        self._latest_analysis = None
        self._analysis_lock = threading.Lock()

    def start(self):
        """Starts capture and processing threads."""
        self._stop_event.clear()
        
        # Thread 1: Frame Capture
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CaptureThread",
            daemon=True
        )
        self._capture_thread.start()
        
        # Thread 2: Processing
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="ProcessingThread",
            daemon=True
        )
        self._processing_thread.start()

    def _capture_loop(self):
        """Thread dedicated to frame capture (I/O bound)."""
        try:
            for frame in self.source:
                if self._stop_event.is_set():
                    break
                    
                try:
                    # Non-blocking put with timeout
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    # If queue is full, skip frame (avoid memory leak)
                    print(f"[WARNING] Frame {frame.id} dropped - queue full")
                    continue
        except Exception as e:
            print(f"[ERROR] Capture thread failed: {e}")
        finally:
            print("[INFO] Capture thread stopped")
            # Signal other threads to stop when capture is done
            self._stop_event.set()

    def _processing_loop(self):
        """Thread dedicated to processing (CPU bound)."""
        analysis = None
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Get frame with timeout to check stop_event
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process
                analysis = self.processor_chain.process(frame, analysis)
                
                # Update shared state
                with self._analysis_lock:
                    self._latest_analysis = analysis
                
                if self.metrics_collector:
                    self.metrics_collector.increment_frames()
                
                # Send result
                try:
                    self.result_queue.put((frame, analysis), block=False)
                except queue.Full:
                    # If full, remove oldest and add new
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put((frame, analysis), block=False)
                    except:
                        pass
                        
        except Exception as e:
            print(f"[ERROR] Processing thread failed: {e}")
        finally:
            print("[INFO] Processing thread stopped")

    def run(self) -> Iterator[Tuple[Frame, FrameAnalysis]]:
        """
        Generator that yields processed results.
        Consumes from result_queue without blocking.
        """
        self.start()
        
        try:
            while not self._stop_event.is_set():
                try:
                    frame, analysis = self.result_queue.get(timeout=0.05)
                    yield frame, analysis
                except queue.Empty:
                    # No results ready, yield None or continue to avoid blocking UI
                    # Here we continue to simulate non-blocking generator behavior
                    continue
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            self.stop()

    def get_latest(self) -> Optional[Tuple[Frame, FrameAnalysis]]:
        """
        Gets the latest available analysis without blocking.
        Useful for APIs that need fast polling.
        """
        with self._analysis_lock:
            return self._latest_analysis

    def stop(self):
        """Stops all threads safely."""
        print("[INFO] Stopping pipeline...")
        self._stop_event.set()
        
        # Wait for threads (with timeout to avoid hang)
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        
        # Release source
        self.source.release()
        print("[INFO] Pipeline stopped")
