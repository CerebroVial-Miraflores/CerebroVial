"""Integración in-process del path batched (topología B / 15Hz).

Cablea las piezas REALES sin red ni GPU ni DB:
  FullDecodeSource (spawner stub → frames sintéticos, frame-clock)
    → QueuePushProducer (daemon thread, push)
    → BatchInferenceWorker (real: cola + batch + demux)
    → post-chain real (ByteTrack de supervision, vía build_post_chain)
    → latest_detections

El detector se stubea para "mover" una caja a cadencia fija (15fps). Verifica lo
que el baseline 0.5Hz NO lograba: a 15fps la asociación de tracks SOBREVIVE — el
móvil mantiene un id de track ESTABLE frame a frame. Es la verificación de
correctitud del checkpoint de la sub-fase 4 (el throughput 11×15 real necesita
CUDA homogéneo → cloud).
"""
import asyncio
import time

from concurrent.futures import ThreadPoolExecutor

import pytest
from omegaconf import OmegaConf

from src.vision.application.builders.pipeline_builder import VisionApplicationBuilder
from src.vision.application.services.batch_inference import BatchInferenceWorker
from src.vision.application.services.queue_push_producer import QueuePushProducer
from src.vision.domain.entities import DetectedVehicle
from src.vision.infrastructure.sources.full_decode_source import FullDecodeSource


# ---- Fakes de ffmpeg (mismo patrón que test_full_decode_source) -------

_CLARO = "https://live.smartechlatam.online/claro/x/index.m3u8"


class _FakeStdout:
    # Pace los frames a una cadencia de decode (~real): el ffmpeg real entrega a
    # 15fps, no instantáneo. Sin esto el productor empuja los 24 frames de golpe y la
    # cola keep-latest-por-cámara (A2) descarta casi todos antes de que el worker
    # infiera → no quedaría secuencia para verificar el tracking.
    def __init__(self, chunks, read_delay_s=0.02):
        self._chunks = list(chunks)
        self._read_delay_s = read_delay_s

    def read(self, n):
        if not self._chunks:
            return b""
        time.sleep(self._read_delay_s)
        return self._chunks.pop(0)


class _FakeProc:
    def __init__(self, chunks):
        self.stdout = _FakeStdout(chunks)

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


class _Spawner:
    def __init__(self, proc):
        self._proc = proc

    def __call__(self, cmd):
        return self._proc


# ---- Detector stub: una caja que se MUEVE a cadencia fija -------------


class _MovingBoxDetector:
    """detect_batch devuelve, por frame, UNA caja que se desplaza ~5px/frame
    (un vehículo cruzando). ByteTrack debe asociarla en UN solo track."""

    def detect_batch(self, frames, frame_ids=None, imgsz=None):
        ids = frame_ids if frame_ids is not None else range(len(frames))
        out = []
        for fid in ids:
            x = 100 + 5 * fid
            out.append([
                DetectedVehicle(
                    id=f"{fid}_0", type="car", confidence=0.9,
                    bbox=(x, 100, x + 40, 160), timestamp=0.0,
                )
            ])
        return out


class _RecordingState:
    """CameraState mínimo: registra el historial de latest_detections."""

    render_enabled = False
    renderer = None
    aggregator = None

    def __init__(self):
        self.history = []

    @property
    def latest_detections(self):
        return self.history[-1] if self.history else None

    @latest_detections.setter
    def latest_detections(self, value):
        self.history.append(value)


def _post_chain_builder():
    """Builder con post-chain real (ByteTrack), sin persistencia/zonas/speed."""
    cfg = OmegaConf.create({
        "vision": {
            "source": _CLARO,
            "source_type": "hls_fulldecode",
            "model": {"path": "yolo11n.pt", "conf_threshold": 0.5},
            "zones": {},
            "speed_estimation": {"enabled": False},
            "persistence": {"enabled": False},
        }
    })
    return VisionApplicationBuilder(cfg)


@pytest.mark.asyncio
async def test_batched_path_tracks_moving_object_at_15fps():
    n_frames = 24
    proc = _FakeProc([bytes(4 * 2 * 3) for _ in range(n_frames)])
    source = FullDecodeSource(_CLARO, _cfg(), capture=_Spawner(proc), fps=15)

    post_chain = _post_chain_builder().build_post_chain()
    state = _RecordingState()

    with ThreadPoolExecutor(max_workers=1) as ex:
        # max_wait < cadencia del fake stdout (20ms) → el worker colecta cada frame
        # fresco sin que dos caigan en la misma ventana → procesa la secuencia entera.
        worker = BatchInferenceWorker(_MovingBoxDetector(), ex, max_batch=4, max_wait_s=0.005)
        worker.register("cam1", post_chain, aggregator=None, state=state)
        worker.start()

        producer = QueuePushProducer("cam1", source, worker, state)
        producer.start()

        # Espera ASYNC (cede al loop para que el task del worker corra; una espera
        # síncrona con time.sleep bloquearía el event loop y el worker nunca demuxearía).
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while loop.time() < deadline and len(state.history) < 12:
            await asyncio.sleep(0.02)
        got = len(state.history) >= 12

        producer.stop()
        await worker.stop()

    assert got, f"esperaba >=12 frames demuxeados, hubo {len(state.history)}"

    # Cada entrada es (FrameAnalysis, w, h); las cajas vienen trackeadas (ByteTrack).
    tracked_ids = []
    for analysis, _w, _h in state.history:
        for v in analysis.vehicles:
            tracked_ids.append(v.id)

    # Hubo tracking real (ByteTrack emitió vehículos con id estable).
    assert tracked_ids, "el tracker no emitió ningún vehículo trackeado"
    # Asociación SOBREVIVE a 15fps: un único id de track domina (no un id nuevo por
    # frame, que es lo que pasaría a 0.5Hz con IoU=0 entre frames espaciados).
    dominant = max(set(tracked_ids), key=tracked_ids.count)
    assert tracked_ids.count(dominant) >= 8, (
        f"track inestable: id dominante {dominant} solo en "
        f"{tracked_ids.count(dominant)}/{len(tracked_ids)} detecciones"
    )


# ---- helpers ----------------------------------------------------------


def _cfg():
    from src.vision.infrastructure.sources.base import SourceConfig
    return SourceConfig(target_width=4, target_height=2)
