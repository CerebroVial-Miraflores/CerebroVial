"""Asynchronous traffic aggregator: Protocol-conforme + worker §11.

Implementa el contrato del Protocol `domain.protocols.AsyncAggregator`. El
worker thread es dueño de la persistencia (DHU-026, supersede de §4.4 Cambio
2 en modo async-only MVP1).

Save y push a output queue son **paths independientes best-effort** para el
mismo `TrafficData` computado:

- §11.1: `repository.save` falla → `logger.exception` + counter
  `aggregation_errors` + el loop continúa. Sin retry, sin DLQ (§11.4).
- §11.2: output queue llena → drop-newest (la ventana entrante se descarta) +
  counter `data_dropped`.
- Un fallo de save **NO** saca el item de la output queue (la operación en
  vivo sobrevive, exactamente lo que §11.1 promete).

`flush()` drena la output queue = lo computado-y-no-dropeado (alineado con el
docstring del Protocol). NO retorna lo persistido.

Tamaño default de la output queue: 256 (§11.2 "sugerido ~256"),
parametrizable por la variable de entorno `VISION_AGGREGATOR_QUEUE_SIZE`.

Logging vía `logging.getLogger(__name__)` por archivo (§6.3 / §12).
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from datetime import datetime, timezone

from ...domain.entities import FrameAnalysis, TrafficData
from ...domain.repositories import TrafficRepository
from ...domain.value_objects import CameraId, ZoneId
from ._compute import ZONELESS_ZONE_ID, compute_traffic_data

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_QUEUE_MAXSIZE = int(
    os.environ.get("VISION_AGGREGATOR_QUEUE_SIZE", "256")
)
_FORCE_FLUSH_TIMEOUT_S = 10.0
_STOP_JOIN_TIMEOUT_S = 5.0


class AsyncTrafficAggregator:
    """Aggregator basado en worker thread que es dueño de la persistencia.

    Cumple `domain.protocols.AsyncAggregator`: `add` / `flush` / `force_flush`
    / `stop`. Ver el docstring del Protocol para el contrato público.
    """

    def __init__(
        self,
        repository: TrafficRepository,
        camera_id: CameraId,
        window_duration_s: float,
        zone_segment_lengths: dict[ZoneId, float | None] | None = None,
        output_queue_maxsize: int = _DEFAULT_OUTPUT_QUEUE_MAXSIZE,
    ) -> None:
        if window_duration_s <= 0:
            raise ValueError("window_duration_s debe ser > 0")
        self._repository = repository
        self._camera_id = camera_id
        self._window_duration_s = window_duration_s
        self._zone_segment_lengths: dict[ZoneId, float | None] = (
            dict(zone_segment_lengths) if zone_segment_lengths else {}
        )

        # Buffer de FrameAnalysis pendientes de procesar. El worker lo vacía
        # cada window_duration_s o ante un force_flush.
        self._buffer: list[FrameAnalysis] = []
        self._buffer_lock = threading.Lock()
        self._window_start: datetime | None = None

        # Output queue: TrafficData computados, esperando flush() del caller.
        self._output_queue: queue.Queue[TrafficData] = queue.Queue(
            maxsize=output_queue_maxsize
        )

        # Worker control.
        self._stop_event = threading.Event()
        self._force_flush_event = threading.Event()
        self._force_flush_done = threading.Event()

        # Counters §11.3 (expuestos como properties).
        self._aggregation_errors = 0
        self._data_dropped = 0
        self._counters_lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="AsyncTrafficAggregator-worker",
            daemon=True,
        )
        self._worker.start()

    # ---- Public Protocol contract --------------------------------------

    def add(self, analysis: FrameAnalysis) -> None:
        """Non-blocking: acumula `analysis` en el buffer de la ventana actual."""
        with self._buffer_lock:
            if self._window_start is None:
                self._window_start = datetime.now(timezone.utc)
            self._buffer.append(analysis)

    def flush(self) -> list[TrafficData]:
        """Non-blocking: drena la output queue.

        Retorna los `TrafficData` ya computados y no dropeados al momento de
        la llamada. NO incluye los que están siendo procesados por el worker.
        """
        out: list[TrafficData] = []
        while True:
            try:
                out.append(self._output_queue.get_nowait())
            except queue.Empty:
                break
        return out

    def force_flush(self) -> list[TrafficData]:
        """Blocking: fuerza al worker a cerrar la ventana actual y drena.

        Útil en shutdown limpio (antes de `stop`) y para sincronización de
        tests.
        """
        if self._stop_event.is_set():
            return self.flush()
        self._force_flush_done.clear()
        self._force_flush_event.set()
        self._force_flush_done.wait(timeout=_FORCE_FLUSH_TIMEOUT_S)
        return self.flush()

    def stop(self) -> None:
        """Señaliza al worker que termine. NO retorna data (usar `force_flush` antes)."""
        self._stop_event.set()
        # Despierta al worker si estaba esperando el timeout de la ventana.
        self._force_flush_event.set()
        self._worker.join(timeout=_STOP_JOIN_TIMEOUT_S)

    # ---- Counters (telemetría §11.3) -----------------------------------

    @property
    def aggregation_errors(self) -> int:
        """Cantidad de saves fallidos desde el arranque (§11.1)."""
        with self._counters_lock:
            return self._aggregation_errors

    @property
    def data_dropped(self) -> int:
        """Cantidad de TrafficData dropeados por output queue llena (§11.2)."""
        with self._counters_lock:
            return self._data_dropped

    # ---- Worker loop + window processing -------------------------------

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            triggered = self._force_flush_event.wait(
                timeout=self._window_duration_s
            )

            # Snapshot atómico del buffer + reset.
            with self._buffer_lock:
                snapshot = list(self._buffer)
                self._buffer.clear()
                window_start = self._window_start
                window_end = datetime.now(timezone.utc)
                self._window_start = None

            if snapshot and window_start is not None and window_end > window_start:
                self._process_window(snapshot, window_start, window_end)

            if triggered:
                self._force_flush_event.clear()
                self._force_flush_done.set()

            if self._stop_event.is_set():
                break

    def _process_window(
        self,
        analyses: list[FrameAnalysis],
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        try:
            traffic_data = compute_traffic_data(
                analyses,
                self._camera_id,
                window_start,
                window_end,
                self._zone_segment_lengths,
            )
        except Exception:
            # Error de cómputo (no de save): la ventana no se emite. No
            # incrementamos aggregation_errors (ese contador es de save).
            logger.exception(
                "compute_traffic_data falló; ventana [%s, %s] descartada",
                window_start.isoformat(),
                window_end.isoformat(),
            )
            return

        for td in traffic_data:
            # §11.1 — save best-effort. INDEPENDIENTE del push a output queue
            # (DHU-026): un fallo de save NO impide ni revierte el push.
            #
            # Fix Fase 3: el sentinel zone-less (ZONELESS_ZONE_ID) NO se persiste.
            # Es presencia extrapolada no calibrada (DemoBadge en el panel),
            # apropiada para mostrar en vivo pero NO para guardar como agregado
            # medido en vision_aggregates (perdería el disclaimer y se mezclaría
            # con agregados calibrados por-zona). Se EXCLUYE solo del save; sigue
            # yendo al output queue → broadcaster → panel (abajo, sin condición).
            if td.zone_id.value != ZONELESS_ZONE_ID:
                try:
                    self._repository.save(td)
                except Exception:
                    logger.exception(
                        "TrafficRepository.save falló para zone_id=%s; continuando (§11.1)",
                        td.zone_id.value,
                    )
                    with self._counters_lock:
                        self._aggregation_errors += 1

            # §11.2 — push best-effort a output queue. INDEPENDIENTE del
            # resultado del save: si save falló, el item igualmente entra a
            # la output queue (telemetría en vivo sobrevive).
            try:
                self._output_queue.put_nowait(td)
            except queue.Full:
                logger.warning(
                    "Output queue llena; drop-newest TrafficData zone_id=%s (§11.2)",
                    td.zone_id.value,
                )
                with self._counters_lock:
                    self._data_dropped += 1
