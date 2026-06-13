"""Post-chain headless (topología B / 15Hz) — split del head de detección.

`VisionApplicationBuilder.build_post_chain()` arranca en `TrackingProcessor` y
recibe detecciones ya calculadas (el batch worker central las inyecta como un
`FrameAnalysis` con `vehicles` poblado). Se verifica que fluyen por
Tracking → Speed → Zone → Aggregation con componentes fake (sin modelo real,
sin supervision/ultralytics), arrancando en Tracking y NO en SmartDetection.
"""
from omegaconf import OmegaConf

from src.vision.application.builders.pipeline_builder import VisionApplicationBuilder
from src.vision.application.processors import TrackingProcessor
from src.vision.domain.entities import DetectedVehicle, Frame, FrameAnalysis
import numpy as np


def _base_cfg() -> OmegaConf:
    return OmegaConf.create({
        'vision': {
            'source': 'test_video.mp4',
            'source_type': 'file',
            'model': {'path': 'yolo11n.pt', 'conf_threshold': 0.5},
            'performance': {'detect_every_n_frames': 1},
            'zones': {},
            'speed_estimation': {'enabled': False},
            'persistence': {'enabled': False},
        }
    })


class _FakeTracker:
    """Asigna ids estables `t{N}` y registra la entrada (prueba que recibe las
    detecciones inyectadas, no las re-detecta)."""

    def __init__(self):
        self.seen = None

    def update(self, detections):
        self.seen = detections
        return [
            DetectedVehicle(
                id=f"t{i}",
                type=d.type,
                confidence=d.confidence,
                bbox=d.bbox,
                timestamp=d.timestamp,
            )
            for i, d in enumerate(detections)
        ]


class _FakeSpeed:
    def __init__(self):
        self.seen = None

    def estimate(self, vehicles):
        self.seen = vehicles
        return vehicles


class _FakeZone:
    def __init__(self):
        self.seen = None

    def count(self, detections, frame_id):
        self.seen = (detections, frame_id)
        return {"z1": object()}


class _FakeAggregator:
    def __init__(self):
        self.added = []

    def add(self, analysis):
        self.added.append(analysis)


def _frame(fid=5):
    return Frame(id=fid, timestamp=100.0, image=np.zeros((10, 10, 3), dtype=np.uint8))


def _injected_analysis(frame):
    """Lo que el demux del batch worker entrega: detecciones ya calculadas."""
    return FrameAnalysis(
        frame_id=frame.id,
        timestamp=frame.timestamp,
        vehicles=[
            DetectedVehicle(id="raw0", type="car", confidence=0.9, bbox=(0, 0, 10, 10), timestamp=100.0),
            DetectedVehicle(id="raw1", type="truck", confidence=0.8, bbox=(5, 5, 20, 20), timestamp=100.0),
        ],
        unique_vehicles=2,
        zones={},
        detection_ran=True,
    )


def test_build_post_chain_head_is_tracking():
    """La post-chain arranca en Tracking (no en SmartDetection)."""
    builder = VisionApplicationBuilder(_base_cfg())
    builder.tracker = _FakeTracker()  # evita construir SupervisionTracker real

    head = builder.build_post_chain()
    assert isinstance(head, TrackingProcessor)


def test_build_post_chain_flows_injected_detections_to_aggregator():
    """Las detecciones inyectadas fluyen Tracking → Speed → Zone → Aggregation."""
    builder = VisionApplicationBuilder(_base_cfg())
    tracker = _FakeTracker()
    speed = _FakeSpeed()
    zone = _FakeZone()
    agg = _FakeAggregator()
    builder.tracker = tracker
    builder.speed_estimator = speed
    builder.zone_counter = zone
    builder.aggregator = agg

    head = builder.build_post_chain()

    frame = _frame()
    head.process(frame, _injected_analysis(frame))

    # Tracking recibió las detecciones inyectadas (no re-detectó).
    assert [d.id for d in tracker.seen] == ["raw0", "raw1"]
    # Speed recibió los vehículos ya trackeados (ids estables del tracker).
    assert [v.id for v in speed.seen] == ["t0", "t1"]
    # Zone contó sobre el frame correcto y sobre los vehículos trackeados.
    assert zone.seen[1] == frame.id
    assert [v.id for v in zone.seen[0]] == ["t0", "t1"]
    # Aggregation recibió el análisis final con los vehículos trackeados y las zonas.
    assert len(agg.added) == 1
    final = agg.added[0]
    assert [v.id for v in final.vehicles] == ["t0", "t1"]
    assert "z1" in final.zones


def test_build_post_chain_tracker_only_when_others_disabled():
    """Sin speed/zone/persistence: la post-chain es solo Tracking y devuelve el
    análisis trackeado sin reventar."""
    builder = VisionApplicationBuilder(_base_cfg())
    tracker = _FakeTracker()
    builder.tracker = tracker

    head = builder.build_post_chain()
    frame = _frame()
    result = head.process(frame, _injected_analysis(frame))

    assert [v.id for v in result.vehicles] == ["t0", "t1"]
    assert result.unique_vehicles == 2
