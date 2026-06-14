"""QueuePushProducer — captura push per-cámara (topología B / 15Hz).

Cubre la conducta per-cámara que antes vivía en el scheduler: empuje de frames a
la cola del worker con el frame-clock como ts, `latest_frame_raw` gated por
render_enabled, y liveness (`read()→None` → sensor_status FUENTE_NO_DISPONIBLE +
force_flush de la ventana). El daemon thread real corre sobre fakes.
"""
import time
import types

import numpy as np

from src.vision.application.services.queue_push_producer import QueuePushProducer
from src.vision.domain.entities import Frame
from src.vision.domain.value_objects import SensorStatus


class _ScriptedSource:
    """read() devuelve frames de una lista y luego None (fuente muerta)."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def read(self):
        if self._frames:
            return self._frames.pop(0)
        return None

    def release(self):
        self.released = True


class _LiveSource:
    """read() devuelve un frame nuevo en cada llamada (la fuente NO muere) — para
    testear el estado OK sostenido sin la carrera de la muerte inmediata."""

    def __init__(self):
        self._i = 0
        self.released = False

    def read(self):
        f = _frame(self._i)
        self._i += 1
        time.sleep(0.001)  # cede para que stop() pueda cortar el loop
        return f

    def release(self):
        self.released = True


class _RecordingWorker:
    def __init__(self):
        self.submitted = []

    def submit(self, camera_id, frame, ts):
        self.submitted.append((camera_id, frame.id, ts))


class _FakeAggregator:
    def __init__(self):
        self.force_flushed = False

    def force_flush(self):
        self.force_flushed = True
        return []


def _state(render_enabled=True, aggregator=None):
    return types.SimpleNamespace(
        sensor_status=None,
        last_frame_age_seconds=None,
        render_enabled=render_enabled,
        latest_frame_raw=None,
        aggregator=aggregator,
    )


def _frame(fid):
    # ts = frame-clock (lo estamparía FullDecodeSource = fid/fps).
    return Frame(id=fid, timestamp=fid / 15, image=np.zeros((4, 4, 3), dtype=np.uint8))


def _wait_until(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_pushes_frames_with_frame_clock_ts():
    worker = _RecordingWorker()
    state = _state()
    src = _ScriptedSource([_frame(0), _frame(1), _frame(2)])
    prod = QueuePushProducer("cam1", src, worker, state)
    prod.start()

    assert _wait_until(lambda: len(worker.submitted) == 3)
    prod.stop()

    # Empujó cada frame con el frame-clock (frame.timestamp = id/15), no wall-clock.
    assert worker.submitted == [("cam1", 0, 0 / 15), ("cam1", 1, 1 / 15), ("cam1", 2, 2 / 15)]


def test_writes_latest_frame_raw_when_render_enabled():
    worker = _RecordingWorker()
    state = _state(render_enabled=True)
    # Fuente VIVA (no muere) → el estado OK se sostiene sin la carrera de la muerte.
    prod = QueuePushProducer("cam1", _LiveSource(), worker, state)
    prod.start()
    assert _wait_until(lambda: state.latest_frame_raw is not None)
    assert state.sensor_status == SensorStatus.OK.value
    prod.stop()
    assert state.latest_frame_raw.shape == (4, 4, 3)


def test_skips_latest_frame_raw_when_render_disabled():
    worker = _RecordingWorker()
    state = _state(render_enabled=False)
    prod = QueuePushProducer("cam1", _ScriptedSource([_frame(0)]), worker, state)
    prod.start()
    assert _wait_until(lambda: len(worker.submitted) == 1)
    prod.stop()
    # Empujó el frame (muestreo permanente) pero NO escribió el crudo (watchdog off).
    assert state.latest_frame_raw is None


def test_source_death_sets_status_and_force_flushes():
    worker = _RecordingWorker()
    agg = _FakeAggregator()
    state = _state(aggregator=agg)
    # read() devuelve 1 frame y luego None (muerte).
    prod = QueuePushProducer("cam1", _ScriptedSource([_frame(0)]), worker, state)
    prod.start()
    # Al morir la fuente: sensor_status FUENTE_NO_DISPONIBLE + force_flush de la ventana.
    assert _wait_until(lambda: state.sensor_status == SensorStatus.FUENTE_NO_DISPONIBLE.value)
    assert _wait_until(lambda: agg.force_flushed)
    prod.stop()


def test_stop_releases_source():
    worker = _RecordingWorker()
    src = _ScriptedSource([_frame(0)])
    prod = QueuePushProducer("cam1", src, worker, _state())
    prod.start()
    assert _wait_until(lambda: src.released or len(worker.submitted) >= 1)
    prod.stop()
    assert src.released  # stop() libera el source (idempotente)
