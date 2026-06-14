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

from src.vision.application.builders.pipeline_builder import (
    VisionApplicationBuilder,
    create_detector,
)


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


def test_builder_builds_postgres_aggregator():
    """Camino feliz (el que ejercita el alta on-demand de C1): postgres +
    camera_id → build_persistence construye el aggregator sin reventar.

    PostgresTrafficRepository se parcha para no tocar la DB (su __init__ es lazy
    igual, pero lo aislamos del entorno)."""
    cfg = _base_cfg()
    cfg.vision.camera_id = 'cam_larco_benavides'
    cfg.vision.persistence = OmegaConf.create({
        'enabled': True,
        'type': 'postgres',
        'interval_seconds': 5,
    })

    with patch(
        'src.vision.application.builders.pipeline_builder.PostgresTrafficRepository'
    ) as MockRepo:
        builder = VisionApplicationBuilder(cfg)
        builder.build_persistence()

        assert builder.aggregator is not None
        MockRepo.assert_called_once()


# ---- knob analyze_fps: build_source → FullDecodeSource(fps=...) --------------


def test_build_source_passes_analyze_fps_to_fulldecode():
    from src.vision.infrastructure.sources.full_decode_source import FullDecodeSource

    cfg = OmegaConf.create({'vision': {
        'source': 'https://x/index.m3u8', 'source_type': 'hls_fulldecode',
        'analyze_fps': 10,
        'performance': {'target_width': 1280, 'target_height': 720},
    }})
    builder = VisionApplicationBuilder(cfg)
    builder.build_source()
    assert isinstance(builder.source, FullDecodeSource)
    assert builder.source._fps == 10  # knob propagado a la fuente


def test_build_source_fulldecode_default_fps_15():
    from src.vision.infrastructure.sources.full_decode_source import FullDecodeSource

    cfg = OmegaConf.create({'vision': {
        'source': 'https://x/index.m3u8', 'source_type': 'hls_fulldecode',
        'performance': {'target_width': 1280, 'target_height': 720},
    }})
    builder = VisionApplicationBuilder(cfg)
    builder.build_source()
    assert isinstance(builder.source, FullDecodeSource)
    assert builder.source._fps == 15  # default = comportamiento actual


# ---- B1 Paso 1a: costura de inyección del detector (la que 1b consume) -------


def test_create_detector_standalone():
    """`create_detector` construye el detector SIN un builder (factory compartible).

    Es la mitad de la costura que 1b usa para construir UN modelo una vez."""
    cfg = _base_cfg()
    with patch(
        'src.vision.application.builders.pipeline_builder.YoloDetector'
    ) as MockYolo:
        sentinel = MagicMock()
        MockYolo.return_value = sentinel

        detector = create_detector(cfg.vision)

        assert detector is sentinel
        # device por defecto None: el detector aplica su fallback cuda→mps→cpu
        # (el device real se inyecta desde el arranque vía `select_device()`).
        MockYolo.assert_called_once_with(
            model_path='yolo11n.pt', conf_threshold=0.5, device=None
        )


def test_build_pipeline_accepts_injected_detector():
    """`build_pipeline(detector=X)` usa X y NO construye uno propio.

    Otra mitad de la costura: el chain RECIBE el detector inyectado (camino que
    el scheduler de 1b usa para compartir un modelo entre cámaras)."""
    cfg = _base_cfg()
    fake_supervision = MagicMock()
    fake_supervision.ByteTrack = MagicMock(return_value=MagicMock())
    fake_supervision.Detections = MagicMock()
    sys.modules['supervision'] = fake_supervision

    injected = MagicMock()
    with patch('cv2.VideoCapture') as mock_cap, patch(
        'src.vision.application.builders.pipeline_builder.create_detector'
    ) as mock_factory:
        mock_cap.return_value.isOpened.return_value = True

        builder = VisionApplicationBuilder(cfg)
        pipeline = builder.build_pipeline(detector=injected)

        assert pipeline is not None
        assert builder.detector is injected
        assert builder.get_components()['detector'] is injected
        # No se cayó al fallback: el detector inyectado evitó construir uno propio.
        mock_factory.assert_not_called()


def test_build_pipeline_builds_own_detector_when_none():
    """Sin inyección, `build_pipeline()` cae al fallback y construye su propio
    detector — preserva el comportamiento de los llamadores viejos (scripts,
    integración)."""
    cfg = _base_cfg()
    fake_supervision = MagicMock()
    fake_supervision.ByteTrack = MagicMock(return_value=MagicMock())
    fake_supervision.Detections = MagicMock()
    sys.modules['supervision'] = fake_supervision

    own = MagicMock()
    with patch('cv2.VideoCapture') as mock_cap, patch(
        'src.vision.application.builders.pipeline_builder.create_detector',
        return_value=own,
    ) as mock_factory:
        mock_cap.return_value.isOpened.return_value = True

        builder = VisionApplicationBuilder(cfg)
        pipeline = builder.build_pipeline()

        assert pipeline is not None
        mock_factory.assert_called_once()
        assert builder.detector is own
