from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    fps: float
    avg_detection_time_ms: float
    avg_tracking_time_ms: float
    frames_processed: int
    vehicles_detected: int
    
    def to_dict(self) -> Dict:
        return {
            'fps': self.fps,
            'avg_detection_time_ms': self.avg_detection_time_ms,
            'avg_tracking_time_ms': self.avg_tracking_time_ms,
            'frames_processed': self.frames_processed,
            'vehicles_detected': self.vehicles_detected
        }


class MetricsCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self):
        self.detection_times: List[float] = []
        self.tracking_times: List[float] = []
        self.frames_processed = 0
        self.vehicles_detected = 0
        self.start_time = time.time()
    
    def record_detection(self, duration_ms: float, vehicle_count: int):
        self.detection_times.append(duration_ms)
        self.vehicles_detected += vehicle_count
        # Keep buffer size manageable
        if len(self.detection_times) > 1000:
            self.detection_times.pop(0)
    
    def record_tracking(self, duration_ms: float):
        self.tracking_times.append(duration_ms)
        if len(self.tracking_times) > 1000:
            self.tracking_times.pop(0)
    
    def increment_frames(self):
        self.frames_processed += 1
    
    def get_metrics(self) -> PerformanceMetrics:
        elapsed = time.time() - self.start_time
        fps = self.frames_processed / elapsed if elapsed > 0 else 0.0
        
        avg_det = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0.0
        avg_track = sum(self.tracking_times) / len(self.tracking_times) if self.tracking_times else 0.0
        
        return PerformanceMetrics(
            fps=fps,
            avg_detection_time_ms=avg_det,
            avg_tracking_time_ms=avg_track,
            frames_processed=self.frames_processed,
            vehicles_detected=self.vehicles_detected
        )
