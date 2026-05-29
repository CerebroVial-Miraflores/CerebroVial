"""Integración del VisionApplicationBuilder (Fase 5f).

Migrado al patrón mock-correcto post-cv2-lazy. Antes el test parchaba
`src.vision.infrastructure.sources.video_source.cv2.VideoCapture`, pero
desde el commit 316b25a7 cv2 se importa lazy DENTRO de las funciones de
`video_source.py` — el atributo `cv2` no existe a nivel de módulo y el
patch fallaba con AttributeError. Ahora se parcha `cv2.VideoCapture`
direct (atributo del módulo cv2 global, el cual el lazy import sí toca).

También usa Tracker fake porque `SupervisionTracker.__init__` importa
`supervision` (que el venv puede no tener) — el test es del builder,
no del tracker.
"""
import sys
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from src.vision.application.builders.pipeline_builder import VisionApplicationBuilder


def _base_cfg() -> OmegaConf:
    return OmegaConf.create({
        'vision': {
            'source': 'test_video.mp4',
            'source_type': 'file',
            'model': {'path': 'yolo11n.pt', 'conf_threshold': 0.5},
            'performance': {
                'detect_every_n_frames': 3,
                'opencv_buffer_size': 3,
                'target_width': 1280,
                'target_height': 720,
            },
            'zones': {},
            'speed_estimation': {'enabled': False},
            'persistence': {'enabled': False},
        }
    })


def test_builder_constructs_complete_pipeline():
    """Sin persistence: builder cablea detector, source, tracker y pipeline."""
    cfg = _base_cfg()

    # `supervision` y `ultralytics` pueden no estar en el venv local — stub a
    # nivel sys.modules antes de que el código los importe lazy. cv2 sí está
    # (opencv-python-headless instalada en 5a).
    fake_supervision = MagicMock()
    fake_supervision.ByteTrack = MagicMock(return_value=MagicMock())
    fake_supervision.Detections = MagicMock()
    sys.modules['supervision'] = fake_supervision

    fake_ultralytics = MagicMock()
    fake_ultralytics.YOLO = MagicMock(return_value=MagicMock())
    sys.modules['ultralytics'] = fake_ultralytics

    with patch('cv2.VideoCapture') as mock_cap:
        mock_cap.return_value.isOpened.return_value = True

        builder = VisionApplicationBuilder(cfg)
        pipeline = (
            builder
            .build_detector()
            .build_tracker()
            .build_speed_estimator()
            .build_zones()
            .build_persistence()
            .build_source()
            .build_pipeline()
        )

        assert pipeline is not None
        assert builder.detector is not None
        assert builder.source is not None
        assert builder.tracker is not None
        assert pipeline.source is builder.source
        assert pipeline.metrics_collector is not None
        # Sin persistence.enabled, no se construye aggregator (modo dev/test).
        assert builder.aggregator is None


def test_builder_rejects_csv_persistence():
    """CSV eliminado en 5b: enabled=true + type=csv → ValueError explícito."""
    import pytest

    cfg = _base_cfg()
    cfg.vision.persistence = OmegaConf.create({
        'enabled': True,
        'type': 'csv',
        'interval_seconds': 60.0,
    })

    builder = VisionApplicationBuilder(cfg)
    with pytest.raises(ValueError, match="postgres"):
        builder.build_persistence()


def test_builder_requires_camera_id_when_persistence_enabled():
    """vision.camera_id es obligatorio si persistence.enabled."""
    import pytest

    cfg = _base_cfg()
    cfg.vision.persistence = OmegaConf.create({
        'enabled': True,
        'type': 'postgres',
        'interval_seconds': 60.0,
    })
    # Sin vision.camera_id.

    builder = VisionApplicationBuilder(cfg)
    with pytest.raises(ValueError, match="camera_id"):
        builder.build_persistence()
