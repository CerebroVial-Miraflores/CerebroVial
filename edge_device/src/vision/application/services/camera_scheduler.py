"""CameraScheduler — captura threaded + inferencia secuencial a 1 Hz (B1 1b, D-018).

Cubre el rol triple del `AsyncVisionPipeline.run()` viejo para UNA cámara, pero
cambiando el modelo de ejecución:

- **Arranque de captura**: arranca un `ThreadedCapture` (envuelve el `FrameProducer`
  del pipeline ya construido). Reemplaza el capture thread del pipeline viejo.
- **Recorrido/drenaje a 1 Hz**: cada tick consulta `snapshot()` (no-bloqueante) y
  decide inferir-o-marcar-sin-señal; drena el aggregator → broadcaster y escribe
  `latest_frame_*` (render gateado por `render_enabled`, respeta el watchdog de Paso 0).
- **Inferencia secuencial con modelo compartido**: corre la processor chain (con el
  detector inyectado en 1a) en un `ThreadPoolExecutor(max_workers=1)` — el event loop
  queda libre para `await publish`, y la inferencia se serializa (modelo compartido
  seguro entre N cámaras en Paso 2). En 1b, una sola cámara.

Reemplaza (no mueve) el pacing/backpressure viejos: NO hay colas, ni catch-up/skip,
ni `target_fps`. El `ThreadedCapture` guarda solo el último frame; el tick a 1 Hz
es el pacing; el slot se sobrescribe (sin backlog). El `AsyncVisionPipeline` NO se
arranca: solo se le reusan `source` y `processor_chain` (su `__init__` no lanza threads).

Honestidad de datos (D-018):
- No-fresco → NO se infiere; se marca `SensorStatus` con motivo. Sin `add()` al
  aggregator → "no medido" por ausencia de filas (la persistencia NULL-con-motivo es
  Fase 1).
- En la **transición** fresco→sin-señal se fuerza `aggregator.force_flush()` UNA vez:
  cierra la ventana en el instante real de última señal (denominadores honestos), en
  vez de dejar que el worker la etiquete con el grid completo incluyendo tiempo muerto.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ...domain.entities import FrameAnalysis, FrameSnapshot
from ...domain.protocols import LiveFrameSource
from ...domain.value_objects import SensorStatus
from ...infrastructure.sources.threaded_capture import ThreadedCapture
from ..processors.smart_detection import DEFAULT_IMGSZ

logger = logging.getLogger(__name__)

_DEFAULT_TICK_INTERVAL_S = 1.0   # 1 Hz (D-018); no el pacing de 30fps del run() viejo
_DEFAULT_FRESH_THRESHOLD_S = 2.5  # respaldo si la fuente no declara su umbral (B-fase1)


class CameraScheduler:
    """Scheduler de UNA cámara: tick a 1 Hz sobre una captura threaded."""

    def __init__(
        self,
        camera,  # CameraInstance (acceso a state: render/renderer/latest_frame_*/sensor_status)
        broadcaster,
        *,
        capture: Optional[LiveFrameSource] = None,
        infer_executor: Optional[ThreadPoolExecutor] = None,
        tick_interval_s: float = _DEFAULT_TICK_INTERVAL_S,
        fresh_threshold_s: Optional[float] = None,
    ) -> None:
        self._camera = camera
        self._broadcaster = broadcaster
        self._tick_interval_s = tick_interval_s

        pipeline = camera.state.pipeline
        # B-fase1: el umbral de frescura es propiedad de la FUENTE (una keyframe-only
        # 0.5fps tolera ~4.5s; cv2 full-decode, 2.5s), no una constante del scheduler.
        # El param explícito gana (tests); si no se deriva de la fuente, con
        # _DEFAULT_FRESH_THRESHOLD_S de respaldo para fuentes que no lo declaran.
        self._fresh_threshold_s = (
            fresh_threshold_s
            if fresh_threshold_s is not None
            else getattr(pipeline.source, "fresh_threshold_s", _DEFAULT_FRESH_THRESHOLD_S)
        )
        # Reusa source + chain del pipeline ya construido (su __init__ no arranca
        # threads). El scheduler NUNCA llama pipeline.start()/run().
        self._capture: LiveFrameSource = capture or ThreadedCapture(pipeline.source)
        self._chain = pipeline.processor_chain
        self._aggregator = camera.state.aggregator

        # Executor de inferencia: max_workers=1 → event loop libre + inferencia
        # serializada (modelo compartido seguro). Compartible entre cámaras (Paso 2).
        # B1 Paso 4: en producción el manager SIEMPRE inyecta el executor compartido
        # (`_owns_executor=False`); el self-create de abajo NO es un path de
        # producción — existe solo para construir el scheduler aislado en tests.
        self._infer_executor = infer_executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vision-infer"
        )
        self._owns_executor = infer_executor is None

        self._prev_analysis: Optional[FrameAnalysis] = None
        self._last_render_analysis: Optional[FrameAnalysis] = None
        # Fase 4 Mitad A: análogo a `_last_render_analysis` pero para el tap de
        # detecciones (overlay del front), que NO está gateado por render_enabled.
        # Persiste las cajas del último detection frame en los skip frames.
        self._last_detections_analysis: Optional[FrameAnalysis] = None
        self._status: SensorStatus = SensorStatus.FUENTE_NO_DISPONIBLE

    # ---- Lifecycle -----------------------------------------------------

    async def run(self) -> None:
        """Loop principal: arranca la captura y tickea a `tick_interval_s` mientras
        `state.is_running`. El pacing es el sleep del tick (no `target_fps`)."""
        self._capture.start()
        loop = asyncio.get_running_loop()
        try:
            while self._camera.state.is_running:
                tick_start = loop.time()
                await self._tick(loop)
                elapsed = loop.time() - tick_start
                await asyncio.sleep(max(0.0, self._tick_interval_s - elapsed))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Camera %s scheduler falló", self._camera.state.camera_id)
            self._camera.state.is_running = False

    def stop(self) -> None:
        """Teardown limpio (cierra B2 para ESTA cámara): para la captura, fuerza el
        flush final del aggregator, para su worker y apaga el executor propio."""
        try:
            self._capture.stop()
        except Exception:
            logger.exception("ThreadedCapture.stop() falló")
        if self._aggregator is not None:
            try:
                self._aggregator.force_flush()  # drena ventana en curso antes de cerrar
                self._aggregator.stop()         # B2: el worker thread no queda colgado
            except Exception:
                logger.exception("aggregator teardown falló")
        if self._owns_executor:
            self._infer_executor.shutdown(wait=False)

    # ---- Tick ----------------------------------------------------------

    async def _tick(self, loop: asyncio.AbstractEventLoop) -> None:
        snap = self._capture.snapshot()  # no-bloqueante: nunca espera al read()
        fresh, status = self._decide(snap)
        prev_status = self._status
        self._status = status
        self._publish_status(snap)

        if fresh:
            # Hot-reload (B1 1c): releemos imgsz del cfg per-tick y lo aplicamos al
            # head de la chain (SmartDetectionProcessor). Si el valor cambió entre
            # ticks, el próximo tick lo usa sin reconstruir nada (imgsz va
            # por-llamada a detect()). El punto de relectura es `self._camera`.
            self._chain.imgsz = self._read_imgsz()
            # Inferencia fuera del event loop (executor serializado).
            analysis = await loop.run_in_executor(
                self._infer_executor, self._chain.process, snap.frame, self._prev_analysis
            )
            self._prev_analysis = analysis
            await self._route(snap.frame, analysis)
            return

        # No-fresco: NO se infiere.
        if self._aggregator is None:
            return
        if prev_status == SensorStatus.OK:
            # Transición fresco→sin-señal: cerrar la ventana en la última señal real
            # (denominador honesto, sin tiempo muerto). force_flush bloquea → executor.
            tds = await loop.run_in_executor(self._infer_executor, self._aggregator.force_flush)
            for td in tds:
                await self._broadcaster.publish(td)
        else:
            # Sin-señal sostenido: solo entregar lo que el worker ya computó (inocente).
            for td in self._aggregator.flush():
                await self._broadcaster.publish(td)

    def _read_imgsz(self) -> int:
        """Política de imgsz para este tick: cfg.vision.model.imgsz, fallback a la
        ÚNICA constante de política DEFAULT_IMGSZ (320). Leído per-tick → hot-reload."""
        model_cfg = self._camera.state.config.vision.get("model", {})
        return model_cfg.get("imgsz", DEFAULT_IMGSZ)

    def _decide(self, snap: FrameSnapshot) -> tuple[bool, SensorStatus]:
        """Decide fresco-vs-motivo desde el snapshot (sin distinguir archivo/stream)."""
        if (
            snap.frame is not None
            and snap.live
            and snap.age_seconds is not None
            and snap.age_seconds <= self._fresh_threshold_s
        ):
            return True, SensorStatus.OK
        if snap.frame is None or not snap.live:
            # Nunca hubo frame, o la captura detectó fin/muerte de la fuente.
            return False, SensorStatus.FUENTE_NO_DISPONIBLE
        # Hay frame pero viejo (stream lento/caído/colgado).
        return False, SensorStatus.SIN_FRAME_FRESCO

    async def _route(self, frame, analysis: Optional[FrameAnalysis]) -> None:
        """Drena aggregator→broadcaster y escribe latest_frame_* (render gateado).

        Muestreo permanente; render solo con `render_enabled` y persistencia visual
        de boxes en skip frames (`detection_ran=False`)."""
        if self._aggregator is not None:
            for td in self._aggregator.flush():
                await self._broadcaster.publish(td)

        state = self._camera.state

        # Tap de detecciones para el overlay del front (Fase 4 Mitad A, D-019).
        # UNGATED por `render_enabled`: a diferencia del render MJPEG, esto corre
        # siempre que haya frame — con el swap a HLS directo nadie consume el MJPEG
        # y el watchdog deja `render_enabled=False`, pero el overlay igual necesita
        # las cajas. Persiste las del último detection frame en los skip frames
        # (detection_ran=False), igual que el render, sin tocar métricas. Guarda las
        # dims del frame de inferencia para que el serializador normalice a [0,1].
        if frame is not None and getattr(frame, "image", None) is not None:
            det_analysis = analysis
            if analysis is not None and analysis.detection_ran:
                self._last_detections_analysis = analysis
            elif self._last_detections_analysis is not None:
                det_analysis = self._last_detections_analysis
            if det_analysis is not None:
                h, w = frame.image.shape[:2]
                state.latest_detections = (det_analysis, w, h)

        if state.render_enabled and frame is not None and getattr(frame, "image", None) is not None:
            render_analysis = analysis
            if analysis is not None and analysis.detection_ran:
                self._last_render_analysis = analysis
            elif self._last_render_analysis is not None:
                render_analysis = self._last_render_analysis

            state.latest_frame_raw = frame.image.copy()
            processed_frame = frame.image.copy()
            if render_analysis and state.renderer is not None:
                processed_frame = state.renderer.render(frame, render_analysis)
            state.latest_frame_processed = processed_frame

    def _publish_status(self, snap: FrameSnapshot) -> None:
        """Expone el estado del sensor en `CameraState` para health (seam §3, opción ii)."""
        self._camera.state.sensor_status = self._status.value
        self._camera.state.last_frame_age_seconds = snap.age_seconds
