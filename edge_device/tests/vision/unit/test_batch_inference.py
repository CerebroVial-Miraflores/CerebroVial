"""Batch inference worker + cola acotada (topología B / 15Hz).

Tests sintéticos (detector stub, post-chains stub) — sin GPU, sin cámaras vivas:
- cola FIFO acotada drop-oldest + contador `dropped` (backpressure);
- formación de lote por tamaño y por espera (`max_wait`);
- demux por camera_id con **orden per-cámara preservado** (FIFO + un worker);
- baja de cámara entre encolar y procesar (se descarta sin romper);
- el `ts` frame-clock se propaga como `FrameAnalysis.timestamp`.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.vision.application.services.batch_inference import (
    BatchInferenceWorker,
    BoundedFrameQueue,
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


class _RecordingPostChain:
    """Post-chain stub: registra (frame_id, analysis) de cada process()."""

    def __init__(self):
        self.received = []

    def process(self, frame, analysis):
        self.received.append((frame.id, analysis))
        return analysis


# ---- BoundedFrameQueue --------------------------------------------------


def test_queue_fifo_and_max_batch_cap():
    q = BoundedFrameQueue(maxsize=10)
    for i in range(5):
        q.put(QueueItem("a", _frame(i), float(i)))
    batch = q.collect_batch(3)
    assert [it.frame.id for it in batch] == [0, 1, 2]  # FIFO + tope max_batch
    assert len(q) == 2


def test_queue_drop_oldest_on_overflow():
    q = BoundedFrameQueue(maxsize=3)
    for i in range(5):  # 0,1 se dropean; quedan 2,3,4
        q.put(QueueItem("a", _frame(i), float(i)))
    assert q.dropped == 2
    assert [it.frame.id for it in q.collect_batch(10)] == [2, 3, 4]


def test_queue_rejects_bad_maxsize():
    with pytest.raises(ValueError):
        BoundedFrameQueue(maxsize=0)


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


# ---- Formación de lote (tamaño / espera) -------------------------------


@pytest.mark.asyncio
async def test_collect_batch_by_size_returns_immediately():
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(detector=_StubDetector(), executor=ex, max_batch=3)
        worker._running = True
        for i in range(5):
            worker.submit("a", _frame(i), float(i))
        batch = await worker._collect_batch()
    assert [it.frame.id for it in batch] == [0, 1, 2]  # tope max_batch, sin esperar


@pytest.mark.asyncio
async def test_collect_batch_by_wait_returns_partial():
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(
            detector=_StubDetector(), executor=ex, max_batch=10, max_wait_s=0.02
        )
        worker._running = True
        worker.submit("a", _frame(1), 1.0)
        worker.submit("a", _frame(2), 2.0)
        loop = asyncio.get_running_loop()
        start = loop.time()
        batch = await worker._collect_batch()  # parcial: 2 < max_batch=10
        elapsed = loop.time() - start
    assert [it.frame.id for it in batch] == [1, 2]
    assert elapsed >= 0.02  # esperó ~max_wait por más antes de devolver el parcial


# ---- Lifecycle (run/stop sobre la cola viva) ---------------------------


@pytest.mark.asyncio
async def test_run_processes_submitted_frames_then_stops():
    detector = _StubDetector()
    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = BatchInferenceWorker(detector, ex, max_batch=4, max_wait_s=0.01)
        pc = _RecordingPostChain()
        worker.register("a", pc)
        worker.start()
        for i in range(3):
            worker.submit("a", _frame(i), float(i))
        # Dar tiempo a que el worker drene la cola.
        for _ in range(50):
            if len(pc.received) == 3:
                break
            await asyncio.sleep(0.005)
        await worker.stop()

    assert [fid for fid, _ in pc.received] == [0, 1, 2]
