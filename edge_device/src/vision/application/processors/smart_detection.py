"""SmartDetectionProcessor — skip-cada-N puro (Fase 5d / decisión #5d).

Detección cada N frames; en frames intermedios el processor emite un
`FrameAnalysis` con `vehicles=[]`, sin intentar interpolar bboxes. La
continuidad temporal de IDs se delega al `VehicleTracker` downstream
(ByteTrack con `lost_track_buffer=60`).

La versión Sesión 1 mantenía `_vehicle_trajectories` + `get_analysis_for_frame`
para "interpolar" detecciones en frames skip. Esa funcionalidad fue identificada
como muerta tras el refactor que movió tracking a una capa propia (xfails
C1.7). El paso 0 de 5d confirmó que el tracker cubre la continuidad de ID
que importa para el cómputo de TrafficData §5.4 (window-level), por lo que
la interpolación per-frame es redundante: `unique_vehicles`, `vehicles_by_type`,
`mean_speed_kmh`, `flow_vehicles_per_hour` y `density_vehicles_per_km` se
computan sobre la ventana entera; `mean_occupancy` toma el promedio sobre
frames con frames-skip aportando occupancy=0 (trade-off aceptado del skip).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from cerebrovial_shared.metrics import MetricsCollector

from ...domain.entities import Frame, FrameAnalysis
from ...domain.protocols import VehicleDetector
from . import FrameProcessor

logger = logging.getLogger(__name__)


class SmartDetectionProcessor(FrameProcessor):
    """Llama al detector cada `detect_every_n` frames; el resto, vacío."""

    def __init__(
        self,
        detector: VehicleDetector,
        detect_every_n: int = 3,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> None:
        super().__init__()
        if detect_every_n < 1:
            raise ValueError("detect_every_n debe ser >= 1")
        self.detector = detector
        self.detect_every_n = detect_every_n
        self.metrics_collector = metrics_collector
        # Frame ID del último frame en el que se ejecutó la detección.
        # Sentinel -1: aún no se detectó nada → el primer frame detecta.
        self._last_detection_frame: int = -1

    def _process(
        self, frame: Frame, analysis: Optional[FrameAnalysis]
    ) -> Optional[FrameAnalysis]:
        # Detectar si: nunca se detectó, o si pasaron al menos N frames desde
        # la última detección. Usamos diferencia relativa para tolerar gaps
        # de frames (drop o skip upstream).
        should_detect = (
            self._last_detection_frame == -1
            or (frame.id - self._last_detection_frame) >= self.detect_every_n
        )

        if should_detect:
            start = time.time()
            detections = self.detector.detect(frame.image, frame.id)
            duration_ms = (time.time() - start) * 1000.0

            if self.metrics_collector:
                self.metrics_collector.record_detection(duration_ms, len(detections))

            self._last_detection_frame = frame.id
            return FrameAnalysis(
                frame_id=frame.id,
                timestamp=frame.timestamp,
                vehicles=detections,
                unique_vehicles=len({d.id for d in detections}),
                zones={},
            )

        # Frame skip: emitir analysis vacío. El tracker downstream mantiene
        # los IDs vivos hasta `lost_track_buffer` frames.
        return FrameAnalysis(
            frame_id=frame.id,
            timestamp=frame.timestamp,
            vehicles=[],
            unique_vehicles=0,
            zones={},
        )
