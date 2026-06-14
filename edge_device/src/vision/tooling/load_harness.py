"""Harness de carga + métricas N-cámaras (cierre de throughput, topología B).

Es el **run combinado instrumentado** que A2 no midió: N decoders/productores
co-residentes con el inferencer batched (mismo proceso/cgroup), barriendo N para
encontrar el máximo donde el sistema sostiene ~15fps/cámara con drops≈0. Ese N da
cámaras-por-contenedor a 15Hz (cierra el gap de A2).

ADITIVO y fuera del hot path: NO toca `run_server`/`multi_camera`/cola/worker.
Reusa las piezas REALES (`FullDecodeSource`, `BatchInferenceWorker`,
`build_post_chain`, `YoloDetector` con auto-device de FASE 1). La observabilidad es
del lado del harness, costo cero en producción:
- `_InstrumentedWorker` subclasea el worker y solo registra `len(batch)` antes de
  delegar en `super()._process_batch`.
- `MetricsPostChain` envuelve la post-chain real y cuenta/timea por cámara.
- `queue.dropped` ya está expuesto por `BoundedFrameQueue`.

Honestidad del resultado: el **N fiel** sale de los modos `file`/`hls` en **CUDA**
(decode co-residente real). El modo `synthetic` da el **techo de inferencia** (sin
decode/red) — útil de referencia, NO es el N de deploy. La salida lo rotula.

Claude Code NO corre la medición: construye + smoke-testea local; el sweep CUDA lo
lanza el operador en la caja fiel.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Optional

import numpy as np
from omegaconf import OmegaConf

from ..application.builders.pipeline_builder import VisionApplicationBuilder
from ..application.services.batch_inference import BatchInferenceWorker
from ..domain.entities import Frame
from ..domain.protocols import FrameProducer
from ..infrastructure.sources.base import SourceConfig
from ..infrastructure.sources.full_decode_source import FullDecodeSource


# ---- Fuente sintética (techo de inferencia: sin decode/red) -----------


class SyntheticSource(FrameProducer):
    """`FrameProducer` in-memory: emite frames a `fps` controlado por su propio
    pacing (sleep). Aísla el costo infer+pipeline del decode/red (el factor
    dominante de A2). Repetible, sin dependencias externas."""

    def __init__(
        self,
        fps: float,
        width: int = 640,
        height: int = 480,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._period = 1.0 / fps
        self._clock = clock
        self._sleep = sleep
        self._i = 0
        self._next_t: Optional[float] = None
        # Una imagen de ruido reusada (el contenido no cambia el costo de inferir a
        # imgsz fijo; evita pagar un numpy nuevo por frame).
        rng = np.random.default_rng(0)
        self._img = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    def read(self) -> Optional[Frame]:
        now = self._clock()
        if self._next_t is None:
            self._next_t = now
        if now < self._next_t:
            self._sleep(self._next_t - now)
        self._next_t += self._period
        frame = Frame(id=self._i, timestamp=time.time(), image=self._img)
        self._i += 1
        return frame

    def release(self) -> None:
        pass


# ---- Productor del harness (push con wall-clock de submit) -------------


class HarnessProducer:
    """Daemon thread que lee de un `FrameProducer` y empuja a la cola del worker,
    estampando el **wall-clock de submit** como ts (para medir latencia e2e
    submit→demux). Variante de medición de `QueuePushProducer` (no escribe
    latest_frame/sensor_status; no es producción)."""

    def __init__(self, camera_id: str, source: FrameProducer, worker) -> None:
        self._camera_id = camera_id
        self._source = source
        self._worker = worker
        import threading

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"harness-prod-{camera_id}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._source.release()
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self._source.read()
            if frame is None:
                break
            self._worker.submit(self._camera_id, frame, time.time())  # ts = wall-clock submit


# ---- Métricas (agregación pura, testeable) ----------------------------


@dataclass
class HarnessMetrics:
    """Colector de eventos crudos; `summary()` agrega sobre la ventana de medición
    (post-warmup). Eventos con timestamp monotónico inyectable (testeable)."""

    demux: list = field(default_factory=list)   # (t_mono, camera_id, latency_ms)
    batches: list = field(default_factory=list)  # (t_mono, size)

    def record_demux(self, camera_id: str, latency_ms: float, t: float) -> None:
        self.demux.append((t, camera_id, latency_ms))

    def record_batch(self, size: int, t: float) -> None:
        self.batches.append((t, size))

    def summary(
        self,
        camera_ids: list,
        device: str,
        window_start: float,
        window_end: float,
        dropped: int,
    ) -> dict:
        dur = max(1e-9, window_end - window_start)
        in_win = [(t, c, lat) for (t, c, lat) in self.demux if window_start <= t <= window_end]

        per_cam: dict = {}
        for _t, c, _lat in in_win:
            per_cam[c] = per_cam.get(c, 0) + 1
        # fps por cámara sobre TODAS las N esperadas (una starved cuenta como 0).
        fps_list = [per_cam.get(cid, 0) / dur for cid in camera_ids]

        lat = [lat for _t, _c, lat in in_win]
        batch_sizes = [s for (t, s) in self.batches if window_start <= t <= window_end]
        processed = len(in_win)

        return {
            "device": device,
            "N": len(camera_ids),
            "fps_mean": mean(fps_list) if fps_list else 0.0,
            "fps_min": min(fps_list) if fps_list else 0.0,
            "dropped": dropped,
            "drop_rate": dropped / (processed + dropped) if (processed + dropped) else 0.0,
            "e2e_p50_ms": _percentile(lat, 0.50),
            "e2e_p95_ms": _percentile(lat, 0.95),
            "batch_mean": mean(batch_sizes) if batch_sizes else 0.0,
            "infers_s": processed / dur,
        }


def _percentile(values: list, q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# ---- Instrumentación (costo cero en producción) -----------------------


class MetricsPostChain:
    """Envuelve la post-chain real: registra latencia e2e (now - ts de submit, que
    viaja en `analysis.timestamp`) y cuenta una salida por cámara, luego delega."""

    def __init__(self, camera_id, inner, metrics: HarnessMetrics, clock=time.monotonic):
        self._camera_id = camera_id
        self._inner = inner
        self._metrics = metrics
        self._clock = clock

    def process(self, frame, analysis):
        latency_ms = (time.time() - analysis.timestamp) * 1000.0
        self._metrics.record_demux(self._camera_id, latency_ms, self._clock())
        return self._inner.process(frame, analysis)


class _InstrumentedWorker(BatchInferenceWorker):
    """Subclase del worker que solo registra el tamaño de lote y delega. No altera
    el hot path de producción (es del harness)."""

    def __init__(self, *args, metrics: HarnessMetrics, clock=time.monotonic, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = metrics
        self._clock = clock

    async def _process_batch(self, batch):
        if batch:
            self._metrics.record_batch(len(batch), self._clock())
        await super()._process_batch(batch)


# ---- Detectores ------------------------------------------------------


class StubDetector:
    """detect_batch instantáneo (smoke/CI sin pesos): 0 detecciones por frame."""

    _device = "stub"

    def detect_batch(self, frames, frame_ids=None, imgsz=None):
        return [[] for _ in frames]


def _make_detector(stub: bool, imgsz: int):
    if stub:
        return StubDetector()
    # Detector real (auto-device de FASE 1: cuda/mps/cpu). Necesita yolo11n.pt.
    from ..infrastructure.detection.yolo_detector import YoloDetector

    return YoloDetector(model_path="yolo11n.pt", conf_threshold=0.25)


# ---- Fuentes / post-chains -------------------------------------------


def _post_chain_config() -> "OmegaConf":
    return OmegaConf.create({
        "vision": {
            "source": "harness",
            "source_type": "synthetic",
            "model": {"path": "yolo11n.pt", "conf_threshold": 0.25},
            "zones": {},
            "speed_estimation": {"enabled": True, "pixels_per_meter": 10.0},
            "persistence": {"enabled": False},
        }
    })


def _make_source(mode: str, index: int, fps: float, width: int, height: int) -> FrameProducer:
    """Construye la fuente per-productor. `synthetic` = N independientes; `file:`/`hls:`
    ciclan las rutas/URLs dadas (un FullDecodeSource cada uno)."""
    if mode == "synthetic":
        return SyntheticSource(fps, width=width, height=height)
    cfg = SourceConfig(target_width=width, target_height=height)
    if mode.startswith("file:"):
        paths = mode[len("file:"):].split(",")
        return FullDecodeSource(paths[index % len(paths)], cfg, fps=int(fps))
    if mode.startswith("hls:"):
        urls = mode[len("hls:"):].split(",")
        return FullDecodeSource(urls[index % len(urls)], cfg, fps=int(fps))
    raise ValueError(f"--source desconocido: {mode!r} (usar synthetic | file:<p> | hls:<u>)")


# ---- Corrida de un N --------------------------------------------------


async def run_one(
    n: int,
    *,
    fps: float,
    duration: float,
    warmup: float,
    source: str,
    detector,
    max_batch: int,
    max_wait: float,
    width: int,
    height: int,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Una corrida con N cámaras: warmup → ventana de medición → fila de métricas."""
    metrics = HarnessMetrics()
    camera_ids = [f"cam_{i}" for i in range(n)]
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="harness-infer")
    worker = _InstrumentedWorker(
        detector, executor, None,
        max_batch=max_batch, max_wait_s=max_wait, queue_maxsize=max(2 * n, 16),
        metrics=metrics, clock=clock,
    )
    worker.start()

    producers: list[HarnessProducer] = []
    for i, cid in enumerate(camera_ids):
        post_chain = VisionApplicationBuilder(_post_chain_config()).build_post_chain()
        worker.register(cid, MetricsPostChain(cid, post_chain, metrics, clock=clock))
        prod = HarnessProducer(cid, _make_source(source, i, fps, width, height), worker)
        producers.append(prod)
        prod.start()

    try:
        await asyncio.sleep(warmup)
        win_start = clock()
        dropped_start = worker.queue.dropped
        await asyncio.sleep(duration)
        win_end = clock()
        dropped_window = worker.queue.dropped - dropped_start
    finally:
        for p in producers:
            p.stop()
        await worker.stop()
        executor.shutdown(wait=False)

    device = getattr(detector, "_device", None) or "unknown"
    return metrics.summary(camera_ids, device, win_start, win_end, dropped_window)


# ---- Tabla ------------------------------------------------------------


_COLS = [
    ("device", "{:<8}"), ("N", "{:>3}"), ("fps_mean", "{:>8.1f}"), ("fps_min", "{:>8.1f}"),
    ("dropped", "{:>7}"), ("drop_rate", "{:>9.3f}"), ("e2e_p50_ms", "{:>10.1f}"),
    ("e2e_p95_ms", "{:>10.1f}"), ("batch_mean", "{:>10.2f}"), ("infers_s", "{:>9.1f}"),
]


def format_table(rows: list, source: str) -> str:
    header = " | ".join(name for name, _ in _COLS)
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(" | ".join(fmt.format(r[name]) for name, fmt in _COLS))
    note = (
        f"\nfuente={source} · "
        + ("TECHO DE INFERENCIA (sin decode/red); el N de deploy sale de file:/hls: en CUDA"
           if source == "synthetic"
           else "decode co-residente real")
        + "\nknee = mayor N con fps_mean≈15 y dropped≈0  →  cámaras por contenedor a 15Hz"
    )
    return "\n".join(lines) + "\n" + note


# ---- Entrypoint -------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harness de carga N-cámaras (topología B / 15Hz)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sweep", type=str, help="lista de N, ej: 1,2,4,8,11")
    g.add_argument("--cameras", type=int, help="un solo N")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--duration", type=float, default=45.0, help="ventana de medición (s)")
    p.add_argument("--warmup", type=float, default=8.0, help="descarte de arranque (s)")
    p.add_argument("--source", type=str, default="synthetic",
                   help="synthetic | file:<path[,path]> | hls:<url[,url]>")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--max-wait", type=float, default=0.05)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--stub-detector", action="store_true",
                   help="detector instantáneo sin pesos (smoke/CI; NO mide infer real)")
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace) -> str:
    sweep = (
        [int(x) for x in args.sweep.split(",")] if args.sweep
        else [args.cameras] if args.cameras
        else [2]
    )
    detector = _make_detector(args.stub_detector, args.imgsz)
    rows = []
    for n in sweep:
        row = await run_one(
            n, fps=args.fps, duration=args.duration, warmup=args.warmup,
            source=args.source, detector=detector, max_batch=args.max_batch,
            max_wait=args.max_wait, width=args.width, height=args.height,
        )
        rows.append(row)
        print(f"[harness] N={n} listo: fps_mean={row['fps_mean']:.1f} dropped={row['dropped']}")
    return format_table(rows, args.source)


def main(argv=None) -> None:
    args = _parse_args(argv)
    table = asyncio.run(_amain(args))
    print("\n" + table)


if __name__ == "__main__":
    main()
