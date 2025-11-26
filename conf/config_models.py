from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

@dataclass
class PerformanceConfig:
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    opencv_buffer_size: int = 3
    detect_every_n_frames: int = 3
    youtube_format: str = "best"

@dataclass
class ModelConfig:
    path: str = "yolo11n.pt"
    conf_threshold: float = 0.5

@dataclass
class ZoneConfig:
    zones: Dict[str, List[List[int]]] = field(default_factory=dict)

@dataclass
class PersistenceConfig:
    enabled: bool = False
    type: str = "csv"
    interval_seconds: int = 60
    output_dir: str = "data/traffic_logs"

@dataclass
class SpeedEstimationConfig:
    enabled: bool = False
    pixels_per_meter: float = 10.0

@dataclass
class VisionConfig:
    source: str
    source_type: str = "youtube"
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    zones: Dict[str, List[List[int]]] = field(default_factory=dict) # Simplified to match hydra dict structure for now
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    speed_estimation: SpeedEstimationConfig = field(default_factory=SpeedEstimationConfig)
    display: bool = True
