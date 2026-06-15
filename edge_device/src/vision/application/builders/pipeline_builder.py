"""Builder de la aplicación de visión (cierre de Fase 5f).

Cablea los componentes Protocol-conformes producidos en 5a-5e:
- `AsyncTrafficAggregator` (5b) con su `TrafficRepository` inyectado.
- `AsyncVisionPipeline` (5c) sobre `FrameProducer.read()`.
- Processors nuevos (5d) con firmas Protocol.
- `FrameRenderer` opcional (5e) — pasado al MultiCameraManager por afuera.

Persistencia: **solo Postgres** en MVP1. CSV eliminado en 5b (decisión #8 del
plan de Fase 5; reasonado en DHU-026 contexto). Si `persistence.enabled` está
en `True` pero `type` no es `'postgres'`, se levanta `ValueError` en boot
(sin fallback silencioso).

Logging por archivo (§6.3). El builder NO importa de `presentation/`.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from cerebrovial_shared.metrics import MetricsCollector

from ...domain.protocols import (
    FrameProducer,
    SpeedEstimator,
    VehicleDetector,
    VehicleTracker,
)
from ...domain.repositories import TrafficRepository
from ...domain.value_objects import CameraId, ZoneId
from ...infrastructure.detection.yolo_detector import YoloDetector
from ...infrastructure.persistence.postgres_repository import (
    PostgresTrafficRepository,
)
from ...infrastructure.sources import create_source
from ...infrastructure.tracking.speed_estimator import SimpleSpeedEstimator
from ...infrastructure.tracking.supervision_tracker import SupervisionTracker
from ...infrastructure.zones.zone_counter import ZoneCounter
from ..aggregators.async_aggregator import AsyncTrafficAggregator
from ..pipelines.async_pipeline import AsyncVisionPipeline
from ..processors import (
    AggregationProcessor,
    FrameProcessor,
    SpeedEstimationProcessor,
    TrackingProcessor,
    ZoneProcessor,
)
from ..processors.smart_detection import SmartDetectionProcessor

logger = logging.getLogger(__name__)

_DEFAULT_VEHICLE_CLASSES = {'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7}


def create_detector(
    vision_cfg: DictConfig, device: str | None = None
) -> VehicleDetector:
    """Construye el detector/modelo a partir de la config de visión.

    Factory standalone (independiente de un builder por-cámara): permite
    construir el modelo UNA vez y compartirlo. B1 Paso 1a deja esta costura
    para que el scheduler de 1b inyecte el mismo detector a N cámaras vía
    `VisionApplicationBuilder.build_pipeline(detector=...)`.

    `device` (resuelto en el arranque por `select_device()`): se pasa al
    `YoloDetector`. `None` → el detector aplica su fallback cuda→mps→cpu sin
    banner (path legacy / construcción directa).
    """
    logger.info("Loading model: %s", vision_cfg.model.path)
    return YoloDetector(
        model_path=vision_cfg.model.path,
        conf_threshold=vision_cfg.model.conf_threshold,
        device=device,
    )


class VisionApplicationBuilder:
    """Builder pattern del pipeline de visión.

    Construye, en orden o todo de una vez vía `build_pipeline()`:
    detector → source → tracker → speed_estimator → zone_counter →
    persistence (aggregator+repo) → pipeline.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.vision_cfg = config.vision
        self.metrics_collector = MetricsCollector()

        # Componentes (poblados por build_*).
        self.detector: Optional[VehicleDetector] = None
        self.tracker: Optional[VehicleTracker] = None
        self.speed_estimator: Optional[SpeedEstimator] = None
        self.zone_counter: Optional[ZoneCounter] = None
        self.aggregator: Optional[AsyncTrafficAggregator] = None
        self.source: Optional[FrameProducer] = None
        self.pipeline: Optional[AsyncVisionPipeline] = None

        # Mapeo zone_id → segment_length_meters (None si no calibrada).
        # Lo llena `build_zones()` y lo consume `build_persistence()`.
        self._zone_segment_lengths: dict[ZoneId, float | None] = {}

    # ---- Component builders --------------------------------------------

    def build_detector(self) -> 'VisionApplicationBuilder':
        """Construye el detector dentro del builder (vía legacy / fallback).

        Lo usan los llamadores viejos con cadena fluida (scripts, integración) y
        el fallback de `build_pipeline`. El código nuevo NO llama acá: inyecta el
        detector con `build_pipeline(detector=...)` (ver `create_detector`).
        """
        self.detector = create_detector(self.vision_cfg)
        return self

    def build_source(self) -> 'VisionApplicationBuilder':
        logger.info(
            "Opening source: %s (Type: %s)",
            self.vision_cfg.source,
            self.vision_cfg.source_type,
        )
        perf_cfg = self.vision_cfg.get('performance', {})
        # Knob analyze_fps: solo aplica a la fuente full-decode (topología B). Para
        # otros source_type, pasar `fps` rompería SourceConfig (no lo conoce), así
        # que se incluye condicionalmente. Default 25 = nativo del HLS de Claro.
        extra = {}
        if self.vision_cfg.source_type == "hls_fulldecode":
            extra["fps"] = int(self.vision_cfg.get('analyze_fps', 25))
        self.source = create_source(
            source_config=self.vision_cfg.source,
            source_type=self.vision_cfg.source_type,
            buffer_size=perf_cfg.get('opencv_buffer_size', 3),
            target_width=perf_cfg.get('target_width', None),
            target_height=perf_cfg.get('target_height', None),
            format=perf_cfg.get('youtube_format', 'best'),
            loop=self.vision_cfg.get('loop', True),
            **extra,
        )
        return self

    def build_tracker(self) -> 'VisionApplicationBuilder':
        logger.info("Initializing tracking")
        vehicle_classes = _load_vehicle_classes()
        # frame_rate alineado al fps operativo (analyze_fps): el Kalman de ByteTrack y
        # el buffer de oclusión asumen el frame_rate real. Default 25 (nativo HLS Claro).
        self.tracker = SupervisionTracker(
            vehicle_classes, frame_rate=int(self.vision_cfg.get('analyze_fps', 25))
        )
        return self

    def build_speed_estimator(self) -> 'VisionApplicationBuilder':
        if self.vision_cfg.speed_estimation.enabled:
            logger.info("Initializing speed estimation")
            self.speed_estimator = SimpleSpeedEstimator(
                pixels_per_meter=self.vision_cfg.speed_estimation.pixels_per_meter
            )
        return self

    def build_zones(self) -> 'VisionApplicationBuilder':
        zones_cfg = self.vision_cfg.get('zones', {})
        if not zones_cfg:
            return self
        logger.info("Initializing zones")
        raw = OmegaConf.to_container(zones_cfg, resolve=True)
        zones_for_counter: dict[ZoneId, list[tuple[int, int]]] = {}
        for zid, zcfg in raw.items():
            if not isinstance(zcfg, dict) or 'polygon' not in zcfg:
                continue
            zone_id = ZoneId(zid)
            zones_for_counter[zone_id] = [tuple(pt) for pt in zcfg['polygon']]
            self._zone_segment_lengths[zone_id] = zcfg.get(
                'segment_length_meters', None
            )
        if zones_for_counter:
            self.zone_counter = ZoneCounter(zones_for_counter)
        return self

    def build_persistence(self) -> 'VisionApplicationBuilder':
        """Construye el AsyncTrafficAggregator con su `TrafficRepository`.

        Solo soporta Postgres (CSV eliminado en 5b). `type != 'postgres'` o
        ausente con `enabled=True` levanta `ValueError` (sin fallback
        silencioso). `enabled=False` → no se construye aggregator (modo
        dev/test sin persistencia).

        `camera_id` se lee de `vision.camera_id`. Required si
        `persistence.enabled`. Sin él, `ValueError` explícito.
        """
        pcfg = self.vision_cfg.get('persistence', {})
        if not pcfg.get('enabled', False):
            return self

        ptype = pcfg.get('type', None)
        if ptype != 'postgres':
            raise ValueError(
                "persistence.type debe ser 'postgres' "
                f"(recibido: {ptype!r}). CSV eliminado en Fase 5 (decisión #8)."
            )

        cam = self.vision_cfg.get('camera_id', None)
        if cam is None:
            raise ValueError(
                "vision.camera_id es obligatorio cuando persistence.enabled "
                "(el aggregator lo requiere para construir TrafficData)."
            )
        camera_id = CameraId(str(cam))

        window_duration_s = float(pcfg.get('interval_seconds', 60.0))
        repository: TrafficRepository = PostgresTrafficRepository()

        self.aggregator = AsyncTrafficAggregator(
            repository=repository,
            camera_id=camera_id,
            window_duration_s=window_duration_s,
            zone_segment_lengths=self._zone_segment_lengths,
        )
        logger.info(
            "Persistence wired: postgres, camera_id=%s, window=%.1fs",
            camera_id.value,
            window_duration_s,
        )
        return self

    # ---- Pipeline assembly --------------------------------------------

    def build_pipeline(
        self, detector: Optional[VehicleDetector] = None
    ) -> AsyncVisionPipeline:
        """Orquesta la construcción del chain por-cámara.

        `detector` (B1 Paso 1a): costura ÚNICA de inyección. Si se pasa, el chain
        usa ESE detector (lo que el scheduler de 1b aprovecha para compartir UN
        modelo entre cámaras). Si es None, se cae al fallback `build_detector()`
        — pensado SOLO para los llamadores viejos que no inyectan (scripts,
        integración). Construye los demás componentes que falten.
        """
        if detector is not None:
            self.detector = detector
        if not self.detector:
            self.build_detector()
        if not self.source:
            self.build_source()
        if not self.tracker:
            self.build_tracker()
        if not self.speed_estimator:
            self.build_speed_estimator()
        if not self.zone_counter:
            self.build_zones()
        if not self.aggregator:
            self.build_persistence()

        perf_cfg = self.vision_cfg.get('performance', {})
        detect_every_n = perf_cfg.get('detect_every_n_frames', 3)

        # Chain of Responsibility — el primer link es SmartDetection.
        head = SmartDetectionProcessor(
            self.detector,
            detect_every_n=detect_every_n,
            metrics_collector=self.metrics_collector,
        )
        current = head
        if self.tracker:
            tp = TrackingProcessor(self.tracker, metrics_collector=self.metrics_collector)
            current.set_next(tp)
            current = tp
        if self.speed_estimator:
            sp = SpeedEstimationProcessor(self.speed_estimator)
            current.set_next(sp)
            current = sp
        if self.zone_counter:
            zp = ZoneProcessor(self.zone_counter)
            current.set_next(zp)
            current = zp
        if self.aggregator:
            ap = AggregationProcessor(self.aggregator)
            current.set_next(ap)
            current = ap

        self.pipeline = AsyncVisionPipeline(
            source=self.source,
            processor_chain=head,
            metrics_collector=self.metrics_collector,
            frame_buffer_size=perf_cfg.get('frame_buffer_size', 10),
            result_buffer_size=perf_cfg.get('result_buffer_size', 30),
            target_fps=perf_cfg.get('target_fps', 30),
        )
        return self.pipeline

    def build_post_chain(self) -> FrameProcessor:
        """Construye la POST-CHAIN headless (topología B, 15Hz).

        Arranca en `TrackingProcessor` y recibe detecciones ya calculadas por el
        batch worker central — la detección sale del head de la chain y se
        centraliza. Orden: Tracking → Speed → Zone → Aggregation, reusando
        `build_tracker/speed/zones/persistence`. Devuelve el head
        (`TrackingProcessor`), un `FrameProcessor` que se invoca con un
        `FrameAnalysis` que ya trae `vehicles` poblado (`detection_ran=True`).

        NO toca `build_pipeline` (el path actual) ni construye source/detector: la
        fuente la maneja el producer-push y el detector es el singleton compartido
        del batch worker.
        """
        if not self.tracker:
            self.build_tracker()
        if not self.speed_estimator:
            self.build_speed_estimator()
        if not self.zone_counter:
            self.build_zones()
        if not self.aggregator:
            self.build_persistence()

        if not self.tracker:
            raise ValueError("post-chain requiere un tracker")

        head = TrackingProcessor(self.tracker, metrics_collector=self.metrics_collector)
        current = head
        if self.speed_estimator:
            sp = SpeedEstimationProcessor(self.speed_estimator)
            current.set_next(sp)
            current = sp
        if self.zone_counter:
            zp = ZoneProcessor(self.zone_counter)
            current.set_next(zp)
            current = zp
        if self.aggregator:
            ap = AggregationProcessor(self.aggregator)
            current.set_next(ap)
            current = ap
        return head

    def get_components(self) -> dict:
        """Componentes construidos, para consumo externo (e.g., Fase 6 cablea el renderer)."""
        return {
            'detector': self.detector,
            'tracker': self.tracker,
            'speed_estimator': self.speed_estimator,
            'zone_counter': self.zone_counter,
            'aggregator': self.aggregator,
            'metrics_collector': self.metrics_collector,
        }


def _load_vehicle_classes() -> dict[str, int]:
    """Carga vehicle_classes.yaml si existe; fallback a defaults."""
    config_path = "conf/vision/vehicle_classes.yaml"
    if os.path.exists(config_path):
        try:
            vc_cfg = OmegaConf.load(config_path)
            return dict(vc_cfg.vehicle_classes)
        except Exception:
            logger.warning(
                "Falló al cargar %s; usando defaults", config_path, exc_info=True
            )
    return dict(_DEFAULT_VEHICLE_CLASSES)
