"""Batch inference worker + cola keep-latest-por-cámara (topología B / 15Hz).

Tests sintéticos (detector stub, post-chains stub) — sin GPU, sin cámaras vivas:
- cola keep-latest-por-cámara + contador `dropped` (drop-to-latest, A2);
- formación de lote por tamaño y por espera (`max_wait`, un frame por cámara);
- demux por camera_id con **orden per-cámara preservado** (un worker);
- baja de cámara: descarta el frame al procesar Y limpia el slot encolado;
- el `ts` frame-clock se propaga como `FrameAnalysis.timestamp`.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.vision.application.services.batch_inference import (
    BatchInferenceWorker,
    LatestPerCameraQueue,
    QueueItem,
)
from src.vision.domain.entities import DetectedVehicle, Frame


def _frame(fid):
    return Frame(id=fid, timestamp=float(fid), image=np.zeros((4, 4, 3), dtype=np.uint8))


class _StubDetector:
    """detect_batch que registra los lotes y devuelve 1 detección por frame
    (el id del vehículo codifica el frame_id, para verificar el demux)."""

    def __init__(self):
        self.batches = []

    def detect_batch(self, frames, frame_ids=None, imgsz=None):
        self.batches.append(list(frame_ids) if frame_ids is not None else len(frames))
        ids = frame_ids if frame_ids is not None else range(len(frames))
        return [
            [DetectedVehicle(id=f"v{fid}", type="car", confidence=0.9, bbox=(0, 0, 1, 1), timestamp=0.0)]
            for fid in ids
        ]


class _ImgszRecordingDetector:
    """Registra el imgsz con que se llama detect_batch (knob de config)."""

    def __init__(self):
        self.imgsz_seen = []

    def detect_batch(self, frames, frame_ids=None, imgsz=None):
        self.imgsz_seen.append(imgsz)
        return [[] for _ in frames]


class _RecordingPostChain:
    """Post-chain stub: registra (frame_id, analysis) de cada process()."""

    def __init__(self):
        self.received = []

    def process(self, frame, analysis):
        self.received.append((frame.id, analysis))
        return analysis


class _FakeAggregator:
    """flush() devuelve TrafficData pendientes (stubs) una sola vez."""

    def __init__(self, pending=None):
        self._pending = list(pending or [])

    def flush(self):
        out, self._pending = self._pending, []
        return out


class _FakeBroadcaster:
    def __init__(self):
        self.published = []

    async def publish(self, td):
        self.published.append(td)


class _FakeState:
    latest_detections = None


# ---- LatestPerCameraQueue (keep-latest-por-cámara, A2) ------------------


def test_queue_keeps_latest_per_camera():
    q = LatestPerCameraQueue()
    for i in range(5):  # mismo cam → cada put sobrescribe el slot
        q.put(QueueItem("a", _frame(i), float(i)))
    assert len(q) == 1
    assert q.dropped == 4  # 4 frames descartados sin inferir (drop-to-latest)
    assert [it.frame.id for it in q.collect_batch(10)] == [4]  # solo el más reciente


def test_queue_one_slot_per_camera_and_max_batch_cap():
    q = LatestPerCameraQueue()
    for cid in ("a", "b", "c"):
        q.put(QueueItem(cid, _frame(0), 0.0))
        q.put(QueueItem(cid, _frame(1), 1.0))  # sobrescribe → queda el frame 1
    assert len(q) == 3  # un slot por cámara
    batch = q.collect_batch(2)  # tope max_batch
    assert [it.camera_id for it in batch] == ["a", "b"]  # 2 cámaras, orden de inserción
    assert all(it.frame.id == 1 for it in batch)  # el más reciente de cada una
    assert len(q) == 1  # queda "c"


def test_queue_discard_removes_camera_slot():
    q = LatestPerCameraQueue()
    q.put(QueueItem("a", _frame(0), 0.0))
    q.put(QueueItem("b", _frame(0), 0.0))
    q.discard("a")  # cámara dada de baja
    assert len(q) == 1
    assert [it.camera_id for it in q.collect_batch(10)] == ["b"]
    q.discard("nope")  # idempotente, no rompe


# ---- Demux / orden per-cámara ------------------------------------------


@pytest.mark.asyncio
async def test_process_batch_demux_preserves_per_camera_order():
    detector = _StubDetector()
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(detector, ex)
        pc_a, pc_b = _RecordingPostChain(), _RecordingPostChain()
        worker.register("a", pc_a)
        worker.register("b", pc_b)

        # Interleave: a:1, b:10, a:2, a:3, b:11
        batch = [
            QueueItem("a", _frame(1), 1.0),
            QueueItem("b", _frame(10), 10.0),
            QueueItem("a", _frame(2), 2.0),
            QueueItem("a", _frame(3), 3.0),
            QueueItem("b", _frame(11), 11.0),
        ]
        await worker._process_batch(batch)

    # Una sola pasada de inferencia, con TODOS los frames del lote en orden.
    assert detector.batches == [[1, 10, 2, 3, 11]]
    # Cada cámara recibe SUS frames, en orden FIFO.
    assert [fid for fid, _ in pc_a.received] == [1, 2, 3]
    assert [fid for fid, _ in pc_b.received] == [10, 11]


@pytest.mark.asyncio
async def test_process_batch_propagates_frame_clock_ts():
    detector = _StubDetector()
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(detector, ex)
        pc = _RecordingPostChain()
        worker.register("a", pc)
        # ts (frame-clock) distinto del frame.timestamp para distinguirlos.
        await worker._process_batch([QueueItem("a", _frame(7), ts=99.5)])

    _, analysis = pc.received[0]
    assert analysis.timestamp == 99.5  # frame-clock, no el frame.timestamp (7.0)
    assert analysis.detection_ran is True
    assert [v.id for v in analysis.vehicles] == ["v7"]


@pytest.mark.asyncio
async def test_worker_passes_imgsz_to_detect_batch():
    det = _ImgszRecordingDetector()
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(det, ex, imgsz=320)
        worker.register("a", _RecordingPostChain())
        await worker._process_batch([QueueItem("a", _frame(1), 1.0)])
    assert det.imgsz_seen == [320]


@pytest.mark.asyncio
async def test_worker_default_imgsz_is_none():
    """Default None = ultralytics nativo (comportamiento previo preservado)."""
    det = _ImgszRecordingDetector()
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(det, ex)
        worker.register("a", _RecordingPostChain())
        await worker._process_batch([QueueItem("a", _frame(1), 1.0)])
    assert det.imgsz_seen == [None]


@pytest.mark.asyncio
async def test_demux_drains_aggregator_and_publishes():
    detector = _StubDetector()
    broadcaster = _FakeBroadcaster()
    agg = _FakeAggregator(pending=["td1", "td2"])
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(detector, ex, broadcaster)
        worker.register("a", _RecordingPostChain(), aggregator=agg)
        await worker._process_batch([QueueItem("a", _frame(1), 1.0)])

    assert broadcaster.published == ["td1", "td2"]  # drenado y publicado en orden


@pytest.mark.asyncio
async def test_demux_writes_latest_detections():
    detector = _StubDetector()
    state = _FakeState()
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(detector, ex)
        worker.register("a", _RecordingPostChain(), state=state)
        await worker._process_batch([QueueItem("a", _frame(3), 3.0)])

    assert state.latest_detections is not None
    analysis, w, h = state.latest_detections
    assert (w, h) == (4, 4)  # dims del frame (4x4)
    assert [v.id for v in analysis.vehicles] == ["v3"]  # análisis final (post-chain)


@pytest.mark.asyncio
async def test_process_batch_skips_unregistered_camera():
    detector = _StubDetector()
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(detector, ex)
        pc = _RecordingPostChain()
        worker.register("a", pc)
        # 'b' nunca se registró (o se dio de baja): su frame se descarta sin romper.
        batch = [QueueItem("a", _frame(1), 1.0), QueueItem("b", _frame(2), 2.0)]
        await worker._process_batch(batch)

    assert [fid for fid, _ in pc.received] == [1]


def test_unregister_discards_queued_slot():
    # Dar de baja una cámara limpia su slot encolado: no se gasta una inferencia en
    # un frame de una cámara que ya no infiere (A2, confirmación #4).
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(_StubDetector(), ex)
        worker.register("a", _RecordingPostChain())
        worker.submit("a", _frame(0), 0.0)
        assert len(worker.queue) == 1
        worker.unregister("a")
    assert len(worker.queue) == 0


# ---- Formación de lote (tamaño / espera) -------------------------------


@pytest.mark.asyncio
async def test_collect_batch_by_size_returns_immediately():
    # Con max_batch cámaras ya encoladas (un slot c/u), no abre la ventana de espera.
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(detector=_StubDetector(), executor=ex, max_batch=3)
        worker._running = True
        for i in range(5):  # 5 cámaras distintas → 5 slots, len >= max_batch
            worker.submit(f"cam{i}", _frame(i), float(i))
        batch = await worker._collect_batch()
    # Tope max_batch: 3 cámaras por orden de inserción, sin esperar.
    assert [it.camera_id for it in batch] == ["cam0", "cam1", "cam2"]


@pytest.mark.asyncio
async def test_collect_batch_waits_then_collects_latest_per_camera():
    # Una sola cámara con frames sucesivos: keep-latest deja el más reciente; el
    # worker abre la ventana de gather (~max_wait) y luego saca un frame por cámara.
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(
            detector=_StubDetector(), executor=ex, max_batch=10, max_wait_s=0.02
        )
        worker._running = True
        worker.submit("a", _frame(1), 1.0)
        worker.submit("a", _frame(2), 2.0)  # sobrescribe → queda el 2
        loop = asyncio.get_running_loop()
        start = loop.time()
        batch = await worker._collect_batch()
        elapsed = loop.time() - start
    assert [it.frame.id for it in batch] == [2]  # solo el más reciente de "a"
    assert elapsed >= 0.02  # abrió la ventana de gather antes de devolver


# ---- Lifecycle (run/stop sobre la cola viva) ---------------------------


@pytest.mark.asyncio
async def test_run_processes_submitted_frames_then_stops():
    # Cámaras distintas: keep-latest guarda un frame por cámara → run() las demuxea
    # todas (con frames de la MISMA cámara solo sobreviviría el más reciente).
    detector = _StubDetector()
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(detector, ex, max_batch=4, max_wait_s=0.01)
        pcs = {cid: _RecordingPostChain() for cid in ("a", "b", "c")}
        for cid, pc in pcs.items():
            worker.register(cid, pc)
        worker.start()
        for cid in ("a", "b", "c"):
            worker.submit(cid, _frame(0), 0.0)
        # Dar tiempo a que el worker drene la cola.
        for _ in range(50):
            if all(pc.received for pc in pcs.values()):
                break
            await asyncio.sleep(0.005)
        await worker.stop()

    # run() drenó la cola y demuxeó: cada cámara recibió su frame.
    assert all(len(pc.received) == 1 for pc in pcs.values())
