from omegaconf import DictConfig
from typing import Optional, Dict, List

from ..domain import FrameProducer, VehicleDetector, VehicleTracker, SpeedEstimator
from ..infrastructure.yolo_detector import YoloDetector
from ..infrastructure.sources import create_source
from ..infrastructure.tracking import SupervisionTracker, SimpleSpeedEstimator
from ..infrastructure.zones import ZoneCounter
from ..infrastructure.repositories import CSVTrafficRepository
from ..application.aggregator import TrafficDataAggregator
from ..application.processors import (
    FrameProcessor, DetectionProcessor, TrackingProcessor, 
    SpeedEstimationProcessor, ZoneProcessor, AggregationProcessor
)
from ..application.pipeline import VisionPipeline
from ...common.metrics import MetricsCollector

class VisionApplicationBuilder:
    """
    Builder pattern for constructing the Vision Pipeline application.
    Centralizes component instantiation and wiring.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.vision_cfg = config.vision
        self.metrics_collector = MetricsCollector()
        
        # Components
        self.detector: Optional[VehicleDetector] = None
        self.tracker: Optional[VehicleTracker] = None
        self.speed_estimator: Optional[SpeedEstimator] = None
        self.zone_counter: Optional[ZoneCounter] = None
        self.aggregator: Optional[TrafficDataAggregator] = None
        self.source: Optional[FrameProducer] = None
        self.pipeline: Optional[VisionPipeline] = None

    def build_detector(self) -> 'VisionApplicationBuilder':
        print(f"Loading model: {self.vision_cfg.model.path}...")
        self.detector = YoloDetector(
            model_path=self.vision_cfg.model.path, 
            conf_threshold=self.vision_cfg.model.conf_threshold
        )
        return self

    def build_source(self) -> 'VisionApplicationBuilder':
        print(f"Opening source: {self.vision_cfg.source} (Type: {self.vision_cfg.source_type})...")
        perf_cfg = self.vision_cfg.get('performance', {})
        self.source = create_source(
            source_config=self.vision_cfg.source,
            source_type=self.vision_cfg.source_type,
            buffer_size=perf_cfg.get('opencv_buffer_size', 3),
            target_width=perf_cfg.get('target_width', None),
            target_height=perf_cfg.get('target_height', None),
            format=perf_cfg.get('youtube_format', 'best')
        )
        return self

    def build_tracker(self) -> 'VisionApplicationBuilder':
        print("Initializing tracking...")
        try:
            # Load vehicle classes from external config
            from omegaconf import OmegaConf
            import os
            
            config_path = "conf/vision/vehicle_classes.yaml"
            if os.path.exists(config_path):
                vc_cfg = OmegaConf.load(config_path)
                vehicle_classes = dict(vc_cfg.vehicle_classes)
            else:
                print(f"Warning: {config_path} not found, using defaults.")
                vehicle_classes = {'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7}
        except Exception as e:
            print(f"Warning: Failed to load vehicle classes: {e}. Using defaults.")
            vehicle_classes = {'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7}
            
        self.tracker = SupervisionTracker(vehicle_classes)
        return self

    def build_speed_estimator(self) -> 'VisionApplicationBuilder':
        if self.vision_cfg.speed_estimation.enabled:
            print("Initializing speed estimation...")
            pixels_per_meter = self.vision_cfg.speed_estimation.pixels_per_meter
            self.speed_estimator = SimpleSpeedEstimator(pixels_per_meter=pixels_per_meter)
        return self

    def build_zones(self) -> 'VisionApplicationBuilder':
        if 'zones' in self.vision_cfg and self.vision_cfg.zones:
            print("Initializing zones...")
            # Convert OmegaConf to dict/list to preserve structure (dict with metadata or list of points)
            from omegaconf import OmegaConf
            zones_config = OmegaConf.to_container(self.vision_cfg.zones, resolve=True)
            perf_cfg = self.vision_cfg.get('performance', {})
            resolution = (
                perf_cfg.get('target_width', 1280), 
                perf_cfg.get('target_height', 720)
            )
            self.zone_counter = ZoneCounter(zones_config, resolution=resolution)
        return self

    def build_persistence(self) -> 'VisionApplicationBuilder':
        if self.vision_cfg.get('persistence', {}).get('enabled', False):
            print("Initializing data persistence...")
            repo_type = self.vision_cfg.persistence.type
            output_dir = self.vision_cfg.persistence.output_dir
            interval = self.vision_cfg.persistence.interval_seconds
            
            if repo_type == 'csv':
                repository = CSVTrafficRepository(output_dir=output_dir)
                self.aggregator = TrafficDataAggregator(repository=repository, window_duration=interval)
        return self

    def build_pipeline(self) -> VisionPipeline:
        if not self.detector:
            self.build_detector()
        if not self.source:
            self.build_source()
            
        # Build Chain
        detect_every_n = self.vision_cfg.get('performance', {}).get('detect_every_n_frames', 3)
        
        processor_chain = DetectionProcessor(
            self.detector, 
            detect_every_n=detect_every_n, 
            metrics_collector=self.metrics_collector
        )
        current_link = processor_chain
        
        if self.tracker:
            tracking_processor = TrackingProcessor(self.tracker, metrics_collector=self.metrics_collector)
            current_link.set_next(tracking_processor)
            current_link = tracking_processor
            
        if self.speed_estimator:
            speed_processor = SpeedEstimationProcessor(self.speed_estimator)
            current_link.set_next(speed_processor)
            current_link = speed_processor
            
        if self.zone_counter:
            zone_processor = ZoneProcessor(self.zone_counter)
            current_link.set_next(zone_processor)
            current_link = zone_processor
            
        if self.aggregator:
            agg_processor = AggregationProcessor(self.aggregator)
            current_link.set_next(agg_processor)
            current_link = agg_processor

        self.pipeline = VisionPipeline(
            source=self.source,
            processor_chain=processor_chain,
            metrics_collector=self.metrics_collector
        )
        return self.pipeline

    def get_components(self) -> Dict:
        """Returns built components for external use (e.g. visualization)"""
        return {
            'detector': self.detector,
            'tracker': self.tracker,
            'speed_estimator': self.speed_estimator,
            'zone_counter': self.zone_counter,
            'aggregator': self.aggregator,
            'metrics_collector': self.metrics_collector
        }
