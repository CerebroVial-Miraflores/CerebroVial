"""Chain of Responsibility de processors del pipeline de visión (Fase 5d).

Migrado al schema canónico §5.4 y a los Protocols del dominio:
- `VehicleTracker.update(detections) -> list[DetectedVehicle]`.
- `ZoneCounter.count(detections, frame_id) -> dict[ZoneId, ZoneVehicleCount]`.
- `AsyncAggregator.add(analysis) -> None` (worker-persiste, DHU-026).

Las entidades del dominio (`FrameAnalysis`, `ZoneVehicleCount`) son frozen
dataclasses: cada processor que "muta" en realidad construye un nuevo
`FrameAnalysis` vía `dataclasses.replace()`.

Logging por archivo (§6.3 / §12).
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Optional

from cerebrovial_shared.metrics import MetricsCollector

from ...domain.entities import Frame, FrameAnalysis
from ...domain.protocols import SpeedEstimator, VehicleTracker, ZoneCounter

logger = logging.getLogger(__name__)


class FrameProcessor(ABC):
    """Handler abstracto del Chain of Responsibility."""

    def __init__(self) -> None:
        self._next_processor: Optional["FrameProcessor"] = None

    def set_next(self, processor: "FrameProcessor") -> "FrameProcessor":
        """Encadena el siguiente processor. Retorna `processor` para chaining fluido."""
        self._next_processor = processor
        return processor

    def process(
        self, frame: Frame, analysis: Optional[FrameAnalysis]
    ) -> Optional[FrameAnalysis]:
        """Template: corre lógica propia y delega al siguiente eslabón."""
        result = self._process(frame, analysis)
        if self._next_processor:
            return self._next_processor.process(frame, result)
        return result

    @abstractmethod
    def _process(
        self, frame: Frame, analysis: Optional[FrameAnalysis]
    ) -> Optional[FrameAnalysis]:
        """Lógica específica que implementa cada processor concreto."""


class TrackingProcessor(FrameProcessor):
    """Asigna IDs persistentes a las detecciones vía `VehicleTracker.update`.

    El tracker es responsable de la continuidad temporal del ID a través
    de frames con detecciones intermitentes (skip-cada-N en upstream).
    """

    def __init__(
        self,
        tracker: VehicleTracker,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> None:
        super().__init__()
        self.tracker = tracker
        self.metrics_collector = metrics_collector

    def _process(
        self, frame: Frame, analysis: Optional[FrameAnalysis]
    ) -> Optional[FrameAnalysis]:
        # Defensiva: si llega sin analysis upstream (no debería en producción),
        # construimos uno vacío para que el chain siga.
        if analysis is None:
            analysis = FrameAnalysis(
                frame_id=frame.id,
                timestamp=frame.timestamp,
                vehicles=[],
                unique_vehicles=0,
                zones={},
            )

        start = time.time()
        tracked = self.tracker.update(analysis.vehicles)
        duration_ms = (time.time() - start) * 1000.0

        if self.metrics_collector:
            self.metrics_collector.record_tracking(duration_ms)

        return replace(
            analysis,
            vehicles=tracked,
            unique_vehicles=len({v.id for v in tracked}),
        )


class SpeedEstimationProcessor(FrameProcessor):
    """Estima velocidad de vehículos tracked vía `SpeedEstimator.estimate`."""

    def __init__(self, speed_estimator: SpeedEstimator) -> None:
        super().__init__()
        self.speed_estimator = speed_estimator

    def _process(
        self, frame: Frame, analysis: Optional[FrameAnalysis]
    ) -> Optional[FrameAnalysis]:
        if analysis is None or not analysis.vehicles:
            return analysis
        vehicles_with_speed = self.speed_estimator.estimate(analysis.vehicles)
        return replace(analysis, vehicles=vehicles_with_speed)


class ZoneProcessor(FrameProcessor):
    """Cuenta vehículos por zona vía `ZoneCounter.count(detections, frame_id)`.

    Llena `analysis.zones` con `dict[ZoneId, ZoneVehicleCount]`. La entrada
    `count` puede recibir lista vacía: `ZoneCounter.count([], frame_id)`
    igual emite un `ZoneVehicleCount(count=0, occupancy=0.0)` por zona, lo
    que el aggregator necesita para mean_occupancy de la ventana.
    """

    def __init__(self, zone_counter: ZoneCounter) -> None:
        super().__init__()
        self.zone_counter = zone_counter

    def _process(
        self, frame: Frame, analysis: Optional[FrameAnalysis]
    ) -> Optional[FrameAnalysis]:
        if analysis is None:
            analysis = FrameAnalysis(
                frame_id=frame.id,
                timestamp=frame.timestamp,
                vehicles=[],
                unique_vehicles=0,
                zones={},
            )
        zones = self.zone_counter.count(analysis.vehicles, frame.id)
        return replace(analysis, zones=zones)


class AggregationProcessor(FrameProcessor):
    """Empuja el `FrameAnalysis` al aggregator (worker-persiste, DHU-026).

    `aggregator.add(analysis)` es non-blocking (§Protocol). NO persiste acá;
    el worker del aggregator hace save best-effort + push best-effort
    independientes según §11.1/§11.2.
    """

    def __init__(self, aggregator) -> None:
        super().__init__()
        self.aggregator = aggregator

    def _process(
        self, frame: Frame, analysis: Optional[FrameAnalysis]
    ) -> Optional[FrameAnalysis]:
        if analysis is not None:
            self.aggregator.add(analysis)
        return analysis
