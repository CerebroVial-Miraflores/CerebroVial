from typing import Iterator, Tuple
from ..domain import FrameProducer, VehicleDetector, FrameAnalysis, Frame, VehicleTracker, SpeedEstimator

class VisionPipeline:
    """
    Orchestrates the computer vision pipeline:
    Source -> Detection -> Result
    """
    def __init__(
        self, 
        source: FrameProducer, 
        detector: VehicleDetector,
        tracker: VehicleTracker = None,
        speed_estimator: SpeedEstimator = None,
        zone_manager: object = None, # Optional ZoneManager
        aggregator: object = None, # Optional TrafficAggregator
        detect_every_n_frames: int = 1
    ):
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.speed_estimator = speed_estimator
        self.zone_manager = zone_manager
        self.aggregator = aggregator
        self.detect_every_n_frames = detect_every_n_frames

    def run(self) -> Iterator[Tuple[Frame, FrameAnalysis]]:
        """
        Runs the pipeline, yielding the current frame and the latest analysis.
        """
        last_analysis = None
        
        for frame in self.source:
            # Run detection periodically
            if frame.id % self.detect_every_n_frames == 0:
                last_analysis = self.detector.detect(frame.image, frame.id)
                
                # Tracking
                if self.tracker and last_analysis:
                    last_analysis.vehicles = self.tracker.track(last_analysis.vehicles)
                
                # Speed Estimation
                if self.speed_estimator and last_analysis:
                    last_analysis.vehicles = self.speed_estimator.estimate(last_analysis.vehicles)
                
                # Update zones if manager exists
                if self.zone_manager and last_analysis:
                    last_analysis.zones = self.zone_manager.update(last_analysis.vehicles)
                
                # Aggregate data
                if self.aggregator and last_analysis:
                    self.aggregator.process(last_analysis)
            
            yield frame, last_analysis

    def stop(self):
        """
        Stops the pipeline and releases resources.
        """
        self.source.release()
