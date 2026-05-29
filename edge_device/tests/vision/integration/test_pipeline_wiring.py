"""Integration test del wiring config↔código del `VisionApplicationBuilder`.

Reusa el builder **REAL** (sin mock) sobre una config representativa de producción
y verifica que la cadena completa de constructores se ensambla sin levantar
TypeError, ValueError ni ConfigKeyError.

Lo que atrapa este test (los cuatro bit-rots destapados al validar Fase 6):

1. `persistence.type` distinto de `postgres` (e.g. CSV legacy en YAML) — el
   builder lo rechaza con ValueError (*"persistence.type debe ser 'postgres'"*).
2. `camera_id` faltante con `persistence.enabled=True` — el builder lo rechaza
   con ValueError (*"vision.camera_id es obligatorio cuando persistence.enabled"*).
3. Desalineación de la firma de `ZoneCounter.__init__` vs el callsite del builder
   — el smoke completo cae con TypeError al instanciar.
4. Desalineación de la firma de `AsyncTrafficAggregator.__init__` vs el callsite
   — idem.

Mockeo selectivo: solo los componentes que requieren recursos externos (YOLO
weights, cv2.VideoCapture con URL real, tracker que carga torch). El builder y
todos los componentes Protocol-conformes corren **REALES** — esa es la diferencia
entre un wiring test útil y un smoke test que solo demuestra que MagicMock funciona.

NO valida persistencia contra Postgres real (CT-08.11(e) e2e queda para Fases 7-9
según el handoff). Foco: alineación config↔código que ningún test cubría hasta hoy.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from src.vision.application.builders.pipeline_builder import VisionApplicationBuilder


def _realistic_config() -> dict:
    """Config representativa basada en `conf/vision/default.yaml` post-Fase 5/6.

    Postgres + camera_id seteado + zonas con polígonos válidos + persistence
    habilitada. Patrón real del bootstrap (`run_server.py` setea camera_id por
    cámara).
    """
    return {
        "vision": {
            "source": "https://example.com/fake.mp4",
            "source_type": "auto",
            "camera_id": "cam_test_01",
            "zones": {
                "zone1": {
                    "camera_id": "cam_test_01",
                    "street": "Av Test",
                    "polygon": [[100, 100], [800, 100], [800, 600], [100, 600]],
                    "segment_length_meters": 50.0,
                },
            },
            "speed_estimation": {"enabled": True, "pixels_per_meter": 10.0},
            "persistence": {
                "enabled": True,
                "type": "postgres",
                "interval_seconds": 5,
            },
            "performance": {
                "target_width": 1280,
                "target_height": 720,
                "opencv_buffer_size": 30,
                "detect_every_n_frames": 1,
                "youtube_format": "best",
                "frame_buffer_size": 30,
                "result_buffer_size": 30,
                "target_fps": 30,
            },
            "model": {"path": "yolo11n.pt", "conf_threshold": 0.3},
            "display": False,
        }
    }


@pytest.fixture
def external_mocks():
    """Mockea solo recursos externos: YOLO weights, source, tracker.

    Yolo: requeriría descargar/cargar el .pt (~50MB). cv2.VideoCapture sobre URL:
    intentaría abrirla. Tracker (supervision): carga heavy. El resto del wiring
    (zone counter, speed estimator, aggregator, repo, pipeline) corre real para
    que cualquier desalineación de firma se manifieste.
    """
    with patch(
        "src.vision.application.builders.pipeline_builder.YoloDetector"
    ) as yolo, patch(
        "src.vision.application.builders.pipeline_builder.create_source"
    ) as source, patch(
        "src.vision.application.builders.pipeline_builder.SupervisionTracker"
    ) as tracker:
        yolo.return_value = MagicMock()
        source.return_value = MagicMock()
        tracker.return_value = MagicMock()
        yield {"yolo": yolo, "source": source, "tracker": tracker}


@pytest.fixture
def builder_cleanup():
    """Cierra el worker thread del aggregator construido por el test."""
    builders: list[VisionApplicationBuilder] = []
    yield builders
    for b in builders:
        agg = b.get_components().get("aggregator")
        if agg is not None:
            agg.stop()


# ---- Smoke: wiring completo (atrapa bit-rots 3 y 4 por firmas) ---------


def test_builder_wires_full_pipeline_with_realistic_config(
    external_mocks, builder_cleanup
):
    """Construir el pipeline completo no levanta excepciones de wiring.

    Atrapa cualquier desalineación de firma en los callsites del builder hacia
    `ZoneCounter`, `SimpleSpeedEstimator`, `AsyncTrafficAggregator`,
    `PostgresTrafficRepository` o `AsyncVisionPipeline` — todos se instancian
    reales con la firma actual.
    """
    cfg = OmegaConf.create(_realistic_config())
    builder = VisionApplicationBuilder(cfg)
    pipeline = builder.build_pipeline()
    builder_cleanup.append(builder)

    components = builder.get_components()
    assert pipeline is not None
    assert components["aggregator"] is not None, "AsyncTrafficAggregator no se ensambló"
    assert components["zone_counter"] is not None, "ZoneCounter no se ensambló"
    assert components["speed_estimator"] is not None, "SpeedEstimator no se ensambló"
    assert components["detector"] is not None, "Detector no se ensambló"
    assert components["tracker"] is not None, "Tracker no se ensambló"


# ---- Bit-rot 1: persistence.type legacy (CSV) --------------------------


def test_persistence_csv_in_config_is_rejected(external_mocks):
    """CSV eliminado en Fase 5b — el builder lo rechaza con ValueError."""
    cfg_dict = _realistic_config()
    cfg_dict["vision"]["persistence"]["type"] = "csv"
    cfg = OmegaConf.create(cfg_dict)
    builder = VisionApplicationBuilder(cfg)
    with pytest.raises(ValueError, match="postgres"):
        builder.build_pipeline()


# ---- Bit-rot 2: camera_id ausente con persistencia habilitada ----------


def test_camera_id_missing_with_persistence_enabled_is_rejected(external_mocks):
    """Aggregator necesita camera_id; el builder lo exige con persistence on."""
    cfg_dict = _realistic_config()
    cfg_dict["vision"]["camera_id"] = None
    cfg = OmegaConf.create(cfg_dict)
    builder = VisionApplicationBuilder(cfg)
    with pytest.raises(ValueError, match="camera_id"):
        builder.build_pipeline()


# ---- Cobertura adicional: persistence disabled (modo dev sin BD) -------


def test_builder_skips_aggregator_with_persistence_disabled(
    external_mocks, builder_cleanup
):
    """Cobertura del path enabled=False: no se construye aggregator."""
    cfg_dict = _realistic_config()
    cfg_dict["vision"]["persistence"]["enabled"] = False
    cfg = OmegaConf.create(cfg_dict)
    builder = VisionApplicationBuilder(cfg)
    pipeline = builder.build_pipeline()
    builder_cleanup.append(builder)

    assert pipeline is not None
    assert builder.get_components()["aggregator"] is None
