from typing import Optional, List, Dict, Tuple
import time
import threading
from ...domain.entities import Frame, FrameAnalysis, DetectedVehicle
from ...domain.protocols import VehicleDetector
from . import FrameProcessor

class SmartDetectionProcessor(FrameProcessor):
    """
    Smart detector that:
    1. Detects every N frames
    2. For intermediate frames, interpolates/extrapolates positions using tracking history
    3. Reduces latency without significantly sacrificing precision
    """
    
    def __init__(
        self, 
        detector: VehicleDetector, 
        detect_every_n: int = 3,
        interpolate: bool = True,
        metrics_collector = None
    ):
        super().__init__()
        self.detector = detector
        self.detect_every_n = detect_every_n
        self.interpolate = interpolate
        self.metrics_collector = metrics_collector
        
        self._last_detection_frame = -1
        self._last_analysis: Optional[FrameAnalysis] = None
        self._vehicle_trajectories: Dict[str, List[Tuple[int, Tuple[int, int, int, int]]]] = {}  # {id: [(frame_id, bbox)]}
        self._lock = threading.Lock()  # Protect shared state

    def _process(self, frame: Frame, analysis: Optional[FrameAnalysis]) -> Optional[FrameAnalysis]:
        # Use relative difference to handle frame drops/gaps robustly
        should_detect = (
            self._last_detection_frame == -1 or 
            (frame.id - self._last_detection_frame) >= self.detect_every_n
        )
        
        if should_detect:
            # Real detection
            start = time.time()
            new_analysis = self.detector.detect(frame.image, frame.id)
            if new_analysis:
                new_analysis.raw_detection_count = len(new_analysis.vehicles)
            duration_ms = (time.time() - start) * 1000
            
            if self.metrics_collector:
                count = new_analysis.total_count if new_analysis else 0
                self.metrics_collector.record_detection(duration_ms, count)

            # Update shared state safely
            with self._lock:
                self._last_analysis = new_analysis
                self._last_detection_frame = frame.id
            
            return new_analysis
        
        else:
            # Return analysis with empty vehicles but cached raw count
            # This signals "no new detections" to the tracker, but preserves debug info
            raw_count = 0
            with self._lock:
                if self._last_analysis:
                    raw_count = self._last_analysis.raw_detection_count
            
            return FrameAnalysis(
                frame_id=frame.id,
                timestamp=frame.timestamp,
                vehicles=[],
                total_count=0,
                raw_detection_count=raw_count
            )

    def get_analysis_for_frame(self, frame_id: int) -> Optional[FrameAnalysis]:
        """
        Deprecated: Interpolation moved to Tracker.
        Returns None to force pipeline to use latest chain result.
        """
        return None
