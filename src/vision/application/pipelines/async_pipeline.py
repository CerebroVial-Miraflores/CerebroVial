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
        result_buffer_size: int = 30,
        target_fps: int = 30
    ):
        self.source = source
        self.processor_chain = processor_chain
        self.metrics_collector = metrics_collector
        self.target_fps = target_fps
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=frame_buffer_size)
        self.result_queue = queue.Queue(maxsize=result_buffer_size)
        self.display_queue = queue.Queue(maxsize=60)  # High FPS display queue
        
        # Thread control
        self._stop_event = threading.Event()
        self._processing_thread = None
        self._capture_thread = None
        
        # Shared state (thread-safe)
        self._latest_analysis = None
        self._analysis_lock = threading.Lock()
        self._dropped_frames = 0

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
                    
                # 1. Feed Display Queue (High Priority - Drop Oldest)
                try:
                    self.display_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put_nowait(frame)
                    except:
                        pass

                # 2. Feed Processing Queue (Low Priority - Drop Newest)
                try:
                    # Non-blocking put
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop Newest strategy (better for fluidity)
                    # If queue is full, we simply discard the new frame for processing.
                    # This ensures that the frames already in the queue (which form a smooth sequence)
                    # are processed.
                    self._dropped_frames += 1
                    if self._dropped_frames % 30 == 0:
                        # Log only occasionally to avoid spam
                        pass
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
                
                # We don't strictly need result_queue for display anymore, 
                # but we keep it if we want to consume processed results elsewhere.
                try:
                    self.result_queue.put((frame, analysis), block=False)
                except queue.Full:
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
        Generator that yields frames for display immediately.
        Attaches the latest available analysis.
        """
        self.start()
        
        # Pre-buffering: Wait for queue to fill to avoid immediate stutter
        print("[INFO] Pre-buffering video stream...")
        while self.display_queue.qsize() < 50 and not self._stop_event.is_set():
            time.sleep(0.1)
        print("[INFO] buffer full, starting playback.")
        
        frame_duration = 1.0 / self.target_fps
        
        try:
            while not self._stop_event.is_set():
                start_time = time.time()
                
                try:
                    # Get from Display Queue (Fast)
                    frame = self.display_queue.get(timeout=0.05)
                    
                    # Get latest analysis (Thread-safe)
                    with self._analysis_lock:
                        analysis = self._latest_analysis
                    
                    yield frame, analysis
                    
                    # Pacing: Sleep to maintain target FPS
                    elapsed = time.time() - start_time
                    if elapsed < frame_duration:
                        time.sleep(frame_duration - elapsed)
                    
                except queue.Empty:
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
