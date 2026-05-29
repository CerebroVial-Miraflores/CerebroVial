"""Tests del AsyncVisionPipeline migrado a `FrameProducer.read()` (Fase 5c).

Cubre lifecycle (start / stop limpios), C1.5 (fix de race en stop, antes
xfail), terminación natural por EOF (source.read() → None), recovery ante
SourceError, y get_latest().

Los Mocks viejos usaban `source.__iter__.return_value = iter([...])` —
incompatible con el contrato nuevo del Protocol `FrameProducer` (read +
release). Acá `MockSource` implementa `read()` devolviendo frames y `None`
al final.
"""
import time
from unittest.mock import MagicMock

import numpy as np

from src.vision.application.pipelines.async_pipeline import AsyncVisionPipeline
from src.vision.domain.entities import Frame, FrameAnalysis


class MockSource:
    """FrameProducer fake: devuelve los frames en orden y luego None (EOF)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self._idx = 0
        self.released = False

    def read(self):
        if self.released:
            return None
        if self._idx >= len(self.frames):
            time.sleep(0.01)
            return None
        frame = self.frames[self._idx]
        self._idx += 1
        return frame

    def release(self):
        self.released = True


class _PassThroughProcessor:
    """Implementa FrameProcessor: process(frame, analysis) → FrameAnalysis nuevo."""

    def __init__(self):
        self.call_count = 0

    def process(self, frame, analysis):
        self.call_count += 1
        return FrameAnalysis(
            frame_id=frame.id,
            timestamp=frame.timestamp,
            vehicles=[],
            unique_vehicles=0,
            zones={},
        )


def _img():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_pipeline_initialization():
    source = MagicMock()
    processor = MagicMock()
    pipeline = AsyncVisionPipeline(source, processor)
    assert pipeline.frame_queue.maxsize == 10
    assert pipeline.result_queue.maxsize == 30


def test_pipeline_start_stop_lifecycle():
    """start() arranca threads; stop() los baja limpio y libera el source."""

    class InfiniteSource:
        def __init__(self):
            self.released = False

        def read(self):
            time.sleep(0.05)
            return Frame(id=1, timestamp=time.time(), image=_img())

        def release(self):
            self.released = True

    source = InfiniteSource()
    processor = _PassThroughProcessor()
    pipeline = AsyncVisionPipeline(source, processor, target_fps=1000)

    pipeline.start()
    time.sleep(0.1)
    assert pipeline._capture_thread.is_alive()
    assert pipeline._processing_thread.is_alive()

    pipeline.stop()
    assert not pipeline._capture_thread.is_alive()
    assert not pipeline._processing_thread.is_alive()
    assert source.released


def test_pipeline_processing_flow_drains_all_frames_C1_5():
    """C1.5 fix: dados N frames finitos, run() yields exactamente N.

    Era xfail por la race entre capture seteando `_stop_event` en su
    finally y processing perdiendo el último item. El fix (separación
    `_stop_event` / `_capture_done` / `_processing_done`) hace que `run()`
    drene todo lo procesado antes de salir.
    """
    # Timestamps cercanos (gap << 1.5s) para que catch-up §10.5 no dispare
    # skip — el catch-up apunta a streams reales rezagados, no a tests con
    # timestamps artificialmente espaciados.
    t0 = time.time()
    frames = [
        Frame(id=1, timestamp=t0 + 0.00, image=_img()),
        Frame(id=2, timestamp=t0 + 0.01, image=_img()),
        Frame(id=3, timestamp=t0 + 0.02, image=_img()),
    ]
    source = MockSource(frames)
    processor = _PassThroughProcessor()
    pipeline = AsyncVisionPipeline(source, processor, target_fps=1000)

    results = list(pipeline.run())

    assert len(results) == 3
    assert [r[0].id for r in results] == [1, 2, 3]
    # FrameAnalysis trae `zones` como dict (schema §5.4).
    assert all(isinstance(r[1].zones, dict) for r in results)
    # source.release() invocado por run()'s finally → stop().
    assert source.released


def test_pipeline_terminates_naturally_on_eof():
    """source.read() retorna None → capture termina → processing drena → run() sale."""
    t0 = time.time()
    frames = [Frame(id=i, timestamp=t0 + i * 0.01, image=_img()) for i in range(5)]
    source = MockSource(frames)
    processor = _PassThroughProcessor()
    pipeline = AsyncVisionPipeline(source, processor, target_fps=1000)

    results = list(pipeline.run())

    assert len(results) == 5
    assert processor.call_count == 5
    assert source.released


def test_pipeline_handles_source_error_and_recovers():
    """SourceError no propaga: se loguea, el loop sigue leyendo el siguiente frame."""
    from cerebrovial_shared.exceptions import SourceError

    class FlakySource:
        def __init__(self):
            self.calls = 0
            self.released = False

        def read(self):
            self.calls += 1
            if self.calls == 1:
                raise SourceError("transitorio simulado")
            if self.calls == 2:
                return Frame(id=1, timestamp=time.time(), image=_img())
            return None  # EOF

        def release(self):
            self.released = True

    source = FlakySource()
    processor = _PassThroughProcessor()
    pipeline = AsyncVisionPipeline(source, processor, target_fps=1000)

    results = list(pipeline.run())

    assert len(results) == 1
    assert results[0][0].id == 1
    assert source.released


def test_get_latest_returns_most_recent_after_processing():
    """get_latest() refleja el último (frame, analysis) procesado."""
    t0 = time.time()
    frames = [Frame(id=i, timestamp=t0 + i * 0.01, image=_img()) for i in range(3)]
    source = MockSource(frames)
    processor = _PassThroughProcessor()
    pipeline = AsyncVisionPipeline(source, processor, target_fps=1000)

    list(pipeline.run())

    latest = pipeline.get_latest()
    assert latest is not None
    frame, analysis = latest
    assert frame.id == 2
    assert isinstance(analysis, FrameAnalysis)
