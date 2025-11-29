from typing import Optional, List, Dict, Tuple
import time
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

    def _process(self, frame: Frame, analysis: Optional[FrameAnalysis]) -> Optional[FrameAnalysis]:
        should_detect = (frame.id % self.detect_every_n == 0)
        
        if should_detect:
            # Real detection
            start = time.time()
            self._last_analysis = self.detector.detect(frame.image, frame.id)
            duration_ms = (time.time() - start) * 1000
            
            if self.metrics_collector:
                count = self._last_analysis.total_count if self._last_analysis else 0
                self.metrics_collector.record_detection(duration_ms, count)

            self._last_detection_frame = frame.id
            
            # Update trajectories
            if self._last_analysis:
                self._update_trajectories(frame.id, self._last_analysis)
            
            return self._last_analysis
        
        elif self.interpolate and self._last_analysis:
            # Interpolate positions
            return self._interpolate_positions(frame.id)
        
        else:
            # Return last analysis without changes
            return self._last_analysis

    def _update_trajectories(self, frame_id: int, analysis: FrameAnalysis):
        """Saves positions for interpolation."""
        if not analysis or not analysis.vehicles:
            return
        
        for vehicle in analysis.vehicles:
            if vehicle.id not in self._vehicle_trajectories:
                self._vehicle_trajectories[vehicle.id] = []
            
            self._vehicle_trajectories[vehicle.id].append((frame_id, vehicle.bbox))
            
            # Keep only last 5 points
            if len(self._vehicle_trajectories[vehicle.id]) > 5:
                self._vehicle_trajectories[vehicle.id].pop(0)

    def _interpolate_positions(self, frame_id: int) -> FrameAnalysis:
        """
        Interpolates vehicle positions based on trajectories.
        Uses simple linear extrapolation.
        """
        if not self._last_analysis:
            return None
        
        interpolated_vehicles = []
        
        for vehicle in self._last_analysis.vehicles:
            if vehicle.id not in self._vehicle_trajectories:
                # No history, use last position
                interpolated_vehicles.append(vehicle)
                continue
            
            trajectory = self._vehicle_trajectories[vehicle.id]
            if len(trajectory) < 2:
                interpolated_vehicles.append(vehicle)
                continue
            
            # Get last two points
            (f1, bbox1), (f2, bbox2) = trajectory[-2], trajectory[-1]
            
            # Linear extrapolation
            # If f2 == f1, avoid division by zero
            if f2 == f1:
                t = 0
            else:
                # We are extrapolating based on the last two known points (f1, f2)
                # to the current frame (frame_id).
                # t represents the normalized time step from f2.
                t = (frame_id - f2) / (f2 - f1)
            
            # Limit extrapolation to avoid objects shooting off to infinity if there's a large gap
            if t > 5.0: 
                t = 0 # Fallback to last position
            
            x1_new = int(bbox2[0] + t * (bbox2[0] - bbox1[0]))
            y1_new = int(bbox2[1] + t * (bbox2[1] - bbox1[1]))
            x2_new = int(bbox2[2] + t * (bbox2[2] - bbox1[2]))
            y2_new = int(bbox2[3] + t * (bbox2[3] - bbox1[3]))
            
            interpolated_vehicle = DetectedVehicle(
                id=vehicle.id,
                type=vehicle.type,
                confidence=vehicle.confidence * 0.8,  # Reduce confidence (interpolated)
                bbox=(x1_new, y1_new, x2_new, y2_new),
                timestamp=time.time(),
                speed=vehicle.speed
            )
            interpolated_vehicles.append(interpolated_vehicle)
        
        return FrameAnalysis(
            frame_id=frame_id,
            timestamp=time.time(),
            vehicles=interpolated_vehicles,
            total_count=len(interpolated_vehicles),
            zones=self._last_analysis.zones if self._last_analysis else []
        )
