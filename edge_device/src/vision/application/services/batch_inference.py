"""Batch inference worker + cola keep-latest-por-cámara (topología B, 15Hz).

Reemplaza el modelo "tick 1Hz → snapshot-latest → inferencia serial por cámara"
por: los producers per-cámara EMPUJAN frames a una cola compartida que guarda SOLO
el frame más reciente de cada cámara (drop-to-latest), y UN solo worker central
junta el latest de hasta `max_batch` cámaras / espera hasta `max_wait`, infiere en
lote (`detect_batch`, una pasada GPU) y demuxea por `camera_id` a la post-chain
registrada (`Tracking → Speed → Zone → Aggregation`, ver
`pipeline_builder.build_post_chain`).

Claves (cerradas en el plan):
- **UN worker (una GPU)** + **un frame por cámara por lote** → el orden per-cámara
  se preserva sin sincronizar las cámaras (el detector es stateless por imagen).
- **Cola keep-latest-por-cámara (A2)**: cuando el decode supera a la inferencia, el
  worker infiere el frame MÁS RECIENTE de cada cámara y descarta los intermedios.
  El throughput de inferencia no cambia (mismo nº de inferencias/s, solo CUÁLES
  frames); las cajas quedan a ~1 inferencia de lag (alineadas) en vez de arrastrar
  un backlog FIFO de varios segundos. El tracker banca el gap (timestamps reales +
  `lost_track_buffer`).
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ...domain.entities import Frame, FrameAnalysis

if TYPE_CHECKING:  # evita dependencia en runtime; el head es duck-typed (.process)
    from ..processors import FrameProcessor

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BATCH = 16
# 5 ms: con 1 cámara el lote nunca se llena, así que la ventana de gather es latencia
# pura por ciclo; a 50ms capeaba la inferencia a ~12/s, a 5ms llega a ~25/s (el techo
# del stream). DEUDA N-cámaras: con varias cámaras la ventana batchea (eficiencia GPU)
# → conviene un gather adaptativo (cortar al tener 1 frame por cámara activa). Ver
# documentation/docs/.
_DEFAULT_MAX_WAIT_S = 0.005
_POLL_INTERVAL_S = 0.005        # granularidad del puente thread→asyncio (await sleep, no busy-spin)


@dataclass(frozen=True)
class QueueItem:
    camera_id: str
    frame: Frame
    ts: float  # frame-clock (frame_index/15), NO wall-clock de lectura


@dataclass
class _Sink:
    """Destino per-cámara del demux: la post-chain + lo necesario para el drain.

    `aggregator`/`state` son opcionales (los tests de batching pasan solo la
    post-chain). Con `aggregator` + broadcaster, el demux drena y publica; con
    `state`, escribe `latest_detections` para el overlay del front (UNGATED).
    """

    post_chain: object  # FrameProcessor (duck-typed: .process(frame, analysis))
    aggregator: object = None
    state: object = None


class LatestPerCameraQueue:
    """Cola keep-latest POR CÁMARA, thread-safe (drop-to-latest).

    Reemplaza el FIFO drop-oldest: en vez de acumular un backlog y procesar los
    frames más viejos, guarda SOLO el frame más reciente de cada cámara. El worker
    infiere ese frame fresco y descarta los intermedios → las cajas quedan a ~1
    inferencia de lag (alineadas), sin que el lag crezca con el tiempo.

    La alimentan los producer threads per-cámara (`put`, sobrescribe el slot de la
    cámara); el worker la drena (`collect_batch`, un frame por cámara). El número de
    slots está acotado por el nº de cámaras vivas (no por un maxsize). `discard`
    limpia el slot de una cámara dada de baja para no inferir frames muertos.
    """

    def __init__(self) -> None:
        # dict ordenado por inserción → fairness entre cámaras en el drain.
        self._latest: dict[str, QueueItem] = {}
        self._lock = threading.Lock()
        self._dropped = 0

    def put(self, item: QueueItem) -> None:
        with self._lock:
            if item.camera_id in self._latest:
                # Había un frame de esta cámara sin inferir → se descarta (drop-to-latest).
                self._dropped += 1
            self._latest[item.camera_id] = item

    def collect_batch(self, max_batch: int) -> list[QueueItem]:
        """Saca el frame más reciente de hasta `max_batch` cámaras (uno por cámara),
        por orden de inserción (fairness). No-bloqueante."""
        with self._lock:
            if max_batch >= len(self._latest):
                items = list(self._latest.values())
                self._latest.clear()
                return items
            cids = list(self._latest.keys())[:max_batch]
            return [self._latest.pop(cid) for cid in cids]

    def discard(self, camera_id: str) -> None:
        """Limpia el slot de una cámara dada de baja (idempotente)."""
        with self._lock:
            self._latest.pop(camera_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._latest)

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
        broadcaster=None,
        *,
        max_batch: int = _DEFAULT_MAX_BATCH,
        max_wait_s: float = _DEFAULT_MAX_WAIT_S,
        imgsz: Optional[int] = None,
    ) -> None:
        self._detector = detector
        self._executor = executor
        self._broadcaster = broadcaster
        # Resolución de inferencia (knob de config). None → ultralytics nativo (640),
        # idéntico al comportamiento previo.
        self._imgsz = imgsz
        self._max_batch = max_batch
        self._max_wait_s = max_wait_s
        # Cola keep-latest-por-cámara (A2): sin maxsize, acotada por nº de cámaras.
        self.queue = LatestPerCameraQueue()
        self._sinks: dict[str, _Sink] = {}
        self._registry_lock = threading.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ---- Registro per-cámara (alta/baja desde el manager) --------------

    def register(
        self,
        camera_id: str,
        post_chain: "FrameProcessor",
        aggregator=None,
        state=None,
    ) -> None:
        with self._registry_lock:
            self._sinks[camera_id] = _Sink(post_chain, aggregator, state)

    def unregister(self, camera_id: str) -> None:
        with self._registry_lock:
            self._sinks.pop(camera_id, None)
        # Limpia el slot encolado de la cámara dada de baja: el demux ya tolera
        # sink=None (descarta sin romper), pero así no se gasta una inferencia en un
        # frame de una cámara que ya no infiere (A2, confirmación #4).
        self.queue.discard(camera_id)

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
        """Espera el primer frame; abre una ventana de gather de `max_wait_s` para que
        varias cámaras (y frames más nuevos que sobrescriben su slot) entren al mismo
        lote GPU; luego saca el frame MÁS RECIENTE de cada cámara de una sola vez.

        wait-then-collect-ONCE (A2): un único `collect_batch` al final garantiza UN
        frame por cámara por lote. El viejo `batch.extend` en loop podía meter 2 frames
        de la misma cámara (frame que llega durante la ventana) → rompía keep-latest.
        Si ya hay `max_batch` cámaras encoladas, no espera. Usa `await asyncio.sleep`
        (cede el loop, no es busy-spin; el CPU es el cuello, confirmación #2)."""
        while self._running and len(self.queue) == 0:
            await asyncio.sleep(_POLL_INTERVAL_S)
        if not self._running:
            return self.queue.collect_batch(self._max_batch)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_wait_s
        while (
            self._running
            and len(self.queue) < self._max_batch
            and loop.time() < deadline
        ):
            await asyncio.sleep(_POLL_INTERVAL_S)
        return self.queue.collect_batch(self._max_batch)

    async def _process_batch(self, batch: list[QueueItem]) -> None:
        """Infiere el lote en GPU (off-loop) y demuxea por camera_id, en orden."""
        if not batch:
            return
        frames = [it.frame.image for it in batch]
        frame_ids = [it.frame.id for it in batch]
        loop = asyncio.get_running_loop()
        detections_per_frame = await loop.run_in_executor(
            self._executor, self._detector.detect_batch, frames, frame_ids, self._imgsz
        )
        for it, dets in zip(batch, detections_per_frame):
            with self._registry_lock:
                sink = self._sinks.get(it.camera_id)
            if sink is None:
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
                result = sink.post_chain.process(it.frame, analysis)
            except Exception:
                logger.exception("post-chain falló para cámara %s", it.camera_id)
                continue
            await self._drain_and_route(it, sink, result)

    async def _drain_and_route(self, it: QueueItem, sink: _Sink, result) -> None:
        """Tras la post-chain: drena el aggregator → broadcaster y guarda las últimas
        cajas en `latest_detections` (UNGATED). El render del MJPEG
        (`latest_frame_processed`) ya NO se hace acá: lo escribe el PRODUCTOR a tasa de
        DECODE sobre el frame fresco (desacople de fluidez, A1), leyendo estas cajas.
        El crudo `latest_frame_raw` también lo escribe el productor."""
        if sink.aggregator is not None and self._broadcaster is not None:
            try:
                for td in sink.aggregator.flush():
                    await self._broadcaster.publish(td)
            except Exception:
                logger.exception("drain→publish falló para cámara %s", it.camera_id)
        state = sink.state
        if state is not None and getattr(it.frame, "image", None) is not None and result is not None:
            h, w = it.frame.image.shape[:2]
            # Cajas para el render del productor y el overlay HLS (UNGATED): la
            # inferencia corre permanente; el productor las dibuja sobre el frame
            # fresco si hay consumidor MJPEG (render_enabled). El swap de la tupla es
            # atómico (GIL) y el FrameAnalysis es frozen → lectura cross-thread segura.
            state.latest_detections = (result, w, h)
