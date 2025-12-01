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
        target_fps: int = 30,
        source_fps: int = 30
    ):
        self.source = source
        self.processor_chain = processor_chain
        self.metrics_collector = metrics_collector
        self.target_fps = target_fps
        self.source_fps = source_fps
        
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
        self._latest_capture_ts = 0.0 # Track latest network frame time

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
        # Synchronized Pipeline Strategy:
        # Capture -> Frame Queue (Blocking) -> Processing -> Result Queue -> Display
        # We block on frame_queue to ensure we don't capture faster than we can process/display.
        
        try:
            for frame in self.source:
                if self._stop_event.is_set():
                    break
                
                # Update latest capture timestamp for lag calculation
                self._latest_capture_ts = frame.timestamp
                    
                # Feed Processing Queue (Blocking - Backpressure)
                # We use a large buffer (3s) to absorb network jitter.
                # If buffer fills, we block to prevent memory overflow, effectively pausing capture.
                try:
                    while not self._stop_event.is_set():
                        try:
                            self.frame_queue.put(frame, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                except Exception:
                    pass

        except Exception as e:
            print(f"[ERROR] Capture thread failed: {e}")
        finally:
            print("[INFO] Capture thread stopped")
            self._stop_event.set()

    def _processing_loop(self):
        """Thread dedicated to processing (CPU bound)."""
        analysis = None
        skipped_counter = 0
        
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Smart Catch-Up Logic
                # Calculate lag between the latest captured frame (Network) and this frame (Processing)
                # We use the shared _latest_capture_ts updated by the capture thread
                current_lag = self._latest_capture_ts - frame.timestamp
                
                should_process = True
                
                if current_lag > 1.5:
                    skipped_counter += 1
                    # If lag > 1.5s, process only 50% of frames (2x speed)
                    if current_lag <= 2.5:
                        if skipped_counter % 2 != 0:
                            should_process = False
                    # If lag > 2.5s, process only 33% of frames (3x speed)
                    else:
                        if skipped_counter % 3 != 0:
                            should_process = False
                            
                    if not should_process:
                        # print(f"[DEBUG] Catch-up: Dropping frame. Lag: {current_lag:.2f}s")
                        continue
                
                # Process
                analysis = self.processor_chain.process(frame, analysis)
                
                # Update shared state
                with self._analysis_lock:
                    self._latest_analysis = analysis
                
                if self.metrics_collector:
                    self.metrics_collector.increment_frames()
                
                # Feed Result Queue (Blocking)
                # This ensures the display loop gets every processed frame in order.
                try:
                    while not self._stop_event.is_set():
                        try:
                            self.result_queue.put((frame, analysis), timeout=0.5)
                            break
                        except queue.Full:
                            continue
                except Exception:
                    pass
                        
        except Exception as e:
            print(f"[ERROR] Processing thread failed: {e}")
        finally:
            print("[INFO] Processing thread stopped")

    def run(self) -> Iterator[Tuple[Frame, FrameAnalysis]]:
        """
        Generator that yields frames for display AFTER processing.
        Guarantees perfect synchronization (Frame N + Analysis N).
        """
        self.start()
        
        # Pre-buffering: Wait for result queue to fill
        print("[INFO] Pre-buffering processed frames...")
        while self.result_queue.qsize() < 5 and not self._stop_event.is_set():
            time.sleep(0.1)
        print("[INFO] buffer ready, starting synchronized playback.")
        
        frame_duration = 1.0 / self.target_fps
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Get from Result Queue (Synchronized)
                    # This contains the frame AND its exact analysis
                    frame, analysis = self.result_queue.get(timeout=0.1)
                    
                    # Start timer AFTER getting frame to decouple network lag from pacing
                    start_time = time.time()
                    
                    yield frame, analysis
                    
                    # Pacing: Sleep to maintain target FPS (if processing is faster than target)
                    elapsed = time.time() - start_time
                    sleep_time = 0.0
                    if elapsed < frame_duration:
                        sleep_time = frame_duration - elapsed
                        time.sleep(sleep_time)

                    
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
