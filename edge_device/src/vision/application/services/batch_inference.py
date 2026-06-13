"""Batch inference worker + cola compartida acotada (topología B, 15Hz).

Reemplaza el modelo "tick 1Hz → snapshot-latest → inferencia serial por cámara"
por: los producers per-cámara EMPUJAN frames a una cola FIFO compartida acotada
(drop-oldest), y UN solo worker central junta hasta `max_batch` / espera hasta
`max_wait`, infiere en lote (`detect_batch`, una pasada GPU) y demuxea por
`camera_id` a la post-chain registrada (`Tracking → Speed → Zone → Aggregation`,
ver `pipeline_builder.build_post_chain`).

Claves (cerradas en el plan):
- **UN worker (una GPU)** + **cola FIFO** → el orden per-cámara se preserva sin
  sincronizar las cámaras (el detector es stateless por imagen).
- **Cola acotada drop-oldest**: ni drenaje infinito (OOM) ni snapshot-latest. Si
  dropea bajo carga, la cadencia baja parejo y el tracker lo banca (timestamps
  reales + `lost_track_buffer`).
- **Concurrencia**: el worker corre como task asyncio; la inferencia GPU se
  off-loadea al executor compartido (igual que el scheduler viejo) y el demux corre
  en el loop. El `ts` del item es el **frame-clock** (`frame_index/15`), no el
  wall-clock de lectura, para blindar la velocidad contra el jitter de entrega de
  ffmpeg; se propaga como `FrameAnalysis.timestamp`.

NO cableado a las cámaras vivas todavía (sub-fase 4): acá se construye y testea con
stubs. El drain del aggregator → broadcaster y el write de `latest_detections` se
cablean en la sub-fase 4 (necesitan el `CameraState`).
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ...domain.entities import Frame, FrameAnalysis

if TYPE_CHECKING:  # evita dependencia en runtime; el head es duck-typed (.process)
    from ..processors import FrameProcessor

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BATCH = 16
_DEFAULT_MAX_WAIT_S = 0.05      # 50 ms (rango sugerido 50-100 ms)
_DEFAULT_QUEUE_MAXSIZE = 64
_POLL_INTERVAL_S = 0.005        # granularidad del puente thread→asyncio


@dataclass(frozen=True)
class QueueItem:
    camera_id: str
    frame: Frame
    ts: float  # frame-clock (frame_index/15), NO wall-clock de lectura


class BoundedFrameQueue:
    """Cola FIFO acotada con drop-oldest, thread-safe.

    La alimentan los producer threads per-cámara (`put`); el worker la drena
    (`collect_batch`). Llena → se descarta el ítem más viejo y se cuenta en
    `dropped`.
    """

    def __init__(self, maxsize: int = _DEFAULT_QUEUE_MAXSIZE) -> None:
        if maxsize < 1:
            raise ValueError("maxsize debe ser >= 1")
        self._maxsize = maxsize
        self._dq: deque[QueueItem] = deque()
        self._lock = threading.Lock()
        self._dropped = 0

    def put(self, item: QueueItem) -> None:
        with self._lock:
            if len(self._dq) >= self._maxsize:
                self._dq.popleft()  # drop-oldest
                self._dropped += 1
            self._dq.append(item)

    def collect_batch(self, max_batch: int) -> list[QueueItem]:
        """Saca hasta `max_batch` ítems en orden FIFO (no-bloqueante)."""
        with self._lock:
            n = min(max_batch, len(self._dq))
            return [self._dq.popleft() for _ in range(n)]

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped


class BatchInferenceWorker:
    """Worker central: junta frames de N cámaras, infiere en lote y demuxea."""

    def __init__(
        self,
        detector,
        executor,
        *,
        max_batch: int = _DEFAULT_MAX_BATCH,
        max_wait_s: float = _DEFAULT_MAX_WAIT_S,
        queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
    ) -> None:
        self._detector = detector
        self._executor = executor
        self._max_batch = max_batch
        self._max_wait_s = max_wait_s
        self.queue = BoundedFrameQueue(queue_maxsize)
        self._post_chains: dict[str, "FrameProcessor"] = {}
        self._registry_lock = threading.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ---- Registro per-cámara (alta/baja en sub-fase 4) -----------------

    def register(self, camera_id: str, post_chain: "FrameProcessor") -> None:
        with self._registry_lock:
            self._post_chains[camera_id] = post_chain

    def unregister(self, camera_id: str) -> None:
        with self._registry_lock:
            self._post_chains.pop(camera_id, None)

    # ---- Ingreso (lo llama el producer thread per-cámara) --------------

    def submit(self, camera_id: str, frame: Frame, ts: float) -> None:
        """Encola un frame (thread-safe vía la cola). `ts` = frame-clock."""
        self.queue.put(QueueItem(camera_id, frame, ts))

    # ---- Lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def run(self) -> None:
        self._running = True
        try:
            while self._running:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("BatchInferenceWorker.run() falló")

    # ---- Batching + demux ----------------------------------------------

    async def _collect_batch(self) -> list[QueueItem]:
        """Espera el primer ítem; luego junta hasta `max_batch`, esperando hasta
        `max_wait_s` por más si el lote viene parcial."""
        while self._running and len(self.queue) == 0:
            await asyncio.sleep(_POLL_INTERVAL_S)
        batch = self.queue.collect_batch(self._max_batch)
        if len(batch) >= self._max_batch or not self._running:
            return batch
        # Lote parcial: completar hasta max_batch o hasta agotar max_wait.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_wait_s
        while self._running and len(batch) < self._max_batch and loop.time() < deadline:
            await asyncio.sleep(_POLL_INTERVAL_S)
            batch.extend(self.queue.collect_batch(self._max_batch - len(batch)))
        return batch

    async def _process_batch(self, batch: list[QueueItem]) -> None:
        """Infiere el lote en GPU (off-loop) y demuxea por camera_id, en orden."""
        if not batch:
            return
        frames = [it.frame.image for it in batch]
        frame_ids = [it.frame.id for it in batch]
        loop = asyncio.get_running_loop()
        detections_per_frame = await loop.run_in_executor(
            self._executor, self._detector.detect_batch, frames, frame_ids
        )
        for it, dets in zip(batch, detections_per_frame):
            with self._registry_lock:
                post_chain = self._post_chains.get(it.camera_id)
            if post_chain is None:
                # Cámara dada de baja entre encolar y procesar: se descarta sin ruido.
                continue
            analysis = FrameAnalysis(
                frame_id=it.frame.id,
                timestamp=it.ts,  # frame-clock (refinamiento #2)
                vehicles=dets,
                unique_vehicles=len({d.id for d in dets}),
                zones={},
                detection_ran=True,
            )
            try:
                post_chain.process(it.frame, analysis)
            except Exception:
                logger.exception("post-chain falló para cámara %s", it.camera_id)
