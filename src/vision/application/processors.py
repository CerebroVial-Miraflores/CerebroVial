from abc import ABC, abstractmethod
from typing import Optional, List
import time
from ..domain.entities import Frame, FrameAnalysis, DetectedVehicle
from ..domain.protocols import VehicleDetector, VehicleTracker, SpeedEstimator
from ..infrastructure.zones.zone_counter import ZoneCounter
from .aggregators.sync_aggregator import TrafficDataAggregator
from ...common.metrics import MetricsCollector

class FrameProcessor(ABC):
    """
    Abstract handler in the Chain of Responsibility pattern for frame processing.
    """
    
    def __init__(self):
        self._next_processor: Optional['FrameProcessor'] = None
    
    def set_next(self, processor: 'FrameProcessor') -> 'FrameProcessor':
        """
        Sets the next processor in the chain.
        Returns the passed processor to allow chaining (fluent interface).
        """
        self._next_processor = processor
        return processor
    
    def process(self, frame: Frame, analysis: Optional[FrameAnalysis]) -> Optional[FrameAnalysis]:
        """
        Template method: executes current processing logic and then delegates to the next processor.
        """
        result = self._process(frame, analysis)
        
        if self._next_processor:
            return self._next_processor.process(frame, result)
        return result
    
    @abstractmethod
    def _process(self, frame: Frame, analysis: Optional[FrameAnalysis]) -> Optional[FrameAnalysis]:
        """
        Specific processing logic to be implemented by concrete classes.
        """
        pass


class DetectionProcessor(FrameProcessor):
    """
    Handles vehicle detection using a detector model.
    This is usually the first link in the chain as it creates the FrameAnalysis object.
    """
    
    def __init__(self, detector: VehicleDetector, detect_every_n: int = 1, metrics_collector: Optional[MetricsCollector] = None):
        super().__init__()
        self.detector = detector
        self.detect_every_n = detect_every_n
        self.metrics_collector = metrics_collector
        self._last_analysis: Optional[FrameAnalysis] = None
    
    def _process(self, frame: Frame, analysis: Optional[FrameAnalysis]) -> Optional[FrameAnalysis]:
        # If analysis is already provided (e.g. from a previous step), we could use it, 
        # but usually detection starts the analysis.
        
        if frame.id % self.detect_every_n == 0:
            start = time.time()
            self._last_analysis = self.detector.detect(frame.image, frame.id)
            duration_ms = (time.time() - start) * 1000
            
            if self.metrics_collector:
                count = self._last_analysis.total_count if self._last_analysis else 0
                self.metrics_collector.record_detection(duration_ms, count)
                
            return self._last_analysis
        
        # Return cached analysis for skipped frames
        return self._last_analysis or analysis


class TrackingProcessor(FrameProcessor):
    """
    Handles vehicle tracking (assigning IDs).
    """
    
    def __init__(self, tracker: VehicleTracker, metrics_collector: Optional[MetricsCollector] = None):
        super().__init__()
        self.tracker = tracker
        self.metrics_collector = metrics_collector
    
    def _process(self, frame: Frame, analysis: Optional[FrameAnalysis]) -> Optional[FrameAnalysis]:
        if analysis and analysis.vehicles:
            start = time.time()
            analysis.vehicles = self.tracker.track(analysis.vehicles)
            duration_ms = (time.time() - start) * 1000
            
            if self.metrics_collector:
                self.metrics_collector.record_tracking(duration_ms)
                
        return analysis


class SpeedEstimationProcessor(FrameProcessor):
    """
    Handles vehicle speed estimation.
    """
    
    def __init__(self, speed_estimator: SpeedEstimator):
        super().__init__()
        self.speed_estimator = speed_estimator
    
    def _process(self, frame: Frame, analysis: Optional[FrameAnalysis]) -> Optional[FrameAnalysis]:
        if analysis and analysis.vehicles:
            analysis.vehicles = self.speed_estimator.estimate(analysis.vehicles)
        return analysis


class ZoneProcessor(FrameProcessor):
    """
    Handles zone counting and status updates.
    """
    
    def __init__(self, zone_counter):
        super().__init__()
        self.zone_counter = zone_counter
    
    def _process(self, frame: Frame, analysis: Optional[FrameAnalysis]) -> Optional[FrameAnalysis]:
        if analysis and analysis.vehicles:
            analysis.zones = self.zone_counter.count_vehicles_in_zones(analysis.vehicles)
        return analysis


class AggregationProcessor(FrameProcessor):
    """
    Handles data aggregation and persistence.
    """
    
    def __init__(self, aggregator):
        super().__init__()
        self.aggregator = aggregator
    
    def _process(self, frame: Frame, analysis: Optional[FrameAnalysis]) -> Optional[FrameAnalysis]:
        if analysis:
            self.aggregator.aggregate_and_persist(analysis)
        return analysis
