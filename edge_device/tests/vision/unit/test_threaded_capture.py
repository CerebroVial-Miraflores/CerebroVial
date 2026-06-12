"""Unit tests de ThreadedCapture (B1 1b).

Cubre el contrato de frescura: snapshot no-bloqueante, edad por reloj monotónico
(inyectable), transición de liveness (frame→live True, EOF→live False), y la
robustez clave: snapshot() NO se cuelga aunque el read() del source esté atascado.
"""
import threading
import time

import numpy as np
import pytest

from src.vision.domain.entities import Frame
from src.vision.infrastructure.sources.threaded_capture import ThreadedCapture


class _FakeClock:
    """Reloj controlable (monotónico simulado)."""

    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class _ScriptedSource:
    """FrameProducer fake: entrega `frames`, luego None (EOF) o bloquea.

    `then`: 'none' → read() devuelve None tras agotar frames (EOF/fuente muerta);
            'block' → read() bloquea hasta release() (simula stream colgado).
    """

    def __init__(self, frames, then: str = "none") -> None:
        self._frames = list(frames)
        self._then = then
        self._gate = threading.Event()
        self.released = False

    def read(self):
        if self._frames:
            return self._frames.pop(0)
        if self._then == "none":
            return None
        self._gate.wait()  # 'block': cuelga hasta release()
        return None

    def release(self) -> None:
        self.released = True
        self._gate.set()


def _frame(fid: int = 0) -> Frame:
    return Frame(id=fid, timestamp=0.0, image=np.zeros((2, 2, 3), dtype=np.uint8))


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_snapshot_before_any_frame_is_empty_not_live():
    cap = ThreadedCapture(_ScriptedSource([], then="block"))
    snap = cap.snapshot()  # sin start(): nunca leyó
    assert snap.frame is None
    assert snap.age_seconds is None
    assert snap.live is False
    cap.stop()


def test_fresh_frame_reports_live_and_age_by_monotonic_clock():
    clock = _FakeClock(t=100.0)
    src = _ScriptedSource([_frame(7)], then="block")  # un frame, luego cuelga
    cap = ThreadedCapture(src, clock=clock)
    cap.start()
    try:
        assert _wait_until(lambda: cap.snapshot().frame is not None)
        # ts capturado a clock=100; avanzamos el reloj a 102 → age == 2.0.
        clock.t = 102.0
        snap = cap.snapshot()
        assert snap.frame.id == 7
        assert snap.live is True
        assert snap.age_seconds == pytest.approx(2.0)
    finally:
        cap.stop()


def test_eof_marks_not_live_keeping_last_frame():
    """Fuente que entrega un frame y luego muere (read→None): live=False, frame queda."""
    cap = ThreadedCapture(_ScriptedSource([_frame(1)], then="none"))
    cap.start()
    try:
        assert _wait_until(lambda: cap.snapshot().live is False)
        snap = cap.snapshot()
        assert snap.frame is not None and snap.frame.id == 1  # último frame retenido
        assert snap.live is False                              # fuente agotada
    finally:
        cap.stop()


def test_never_a_frame_eof_is_fuente_no_disponible_shape():
    """Fuente vacía que devuelve None de una: frame None + live False."""
    cap = ThreadedCapture(_ScriptedSource([], then="none"))
    cap.start()
    try:
        assert _wait_until(lambda: cap.snapshot().live is False)
        snap = cap.snapshot()
        assert snap.frame is None
        assert snap.live is False
    finally:
        cap.stop()


def test_snapshot_does_not_block_when_read_is_hung():
    """LO CENTRAL: read() colgado NO cuelga snapshot() (solo lee el slot)."""
    clock = _FakeClock(t=0.0)
    src = _ScriptedSource([_frame(3)], then="block")  # 1 frame, luego read() cuelga
    cap = ThreadedCapture(src, clock=clock)
    cap.start()
    try:
        assert _wait_until(lambda: cap.snapshot().frame is not None)
        # El thread está atascado en read(); snapshot debe volver al instante.
        for _ in range(50):
            start = time.monotonic()
            snap = cap.snapshot()
            assert time.monotonic() - start < 0.05  # no espera al read colgado
            assert snap.frame.id == 3
        # La edad crece con el reloj aunque el read siga colgado (→ no-fresco).
        clock.t = 999.0
        assert cap.snapshot().age_seconds == pytest.approx(999.0)
    finally:
        cap.stop()  # release() desbloquea el read colgado


def test_stop_releases_source():
    src = _ScriptedSource([_frame(0)], then="block")
    cap = ThreadedCapture(src)
    cap.start()
    cap.stop()
    assert src.released is True


def test_start_is_idempotent():
    src = _ScriptedSource([_frame(0)], then="block")
    cap = ThreadedCapture(src)
    cap.start()
    t1 = cap._thread
    cap.start()  # segunda vez: no crea otro thread
    assert cap._thread is t1
    cap.stop()
