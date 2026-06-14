"""QueuePushProducer — captura push per-cámara (topología B, 15Hz).

Reemplaza el modelo `ThreadedCapture` (snapshot-latest, drop-all-but-newest) por
PUSH: corre `source.read()` en un daemon thread y EMPUJA cada frame a la cola
compartida del `BatchInferenceWorker` (a ~15fps). El ts que carga es el frame-clock
(`frame.timestamp`, estampado por `FullDecodeSource` = `frame_index/fps`), no el
wall-clock de lectura.

Dueño de lo per-cámara (refinamiento #1):
- **Liveness**: `read()→None` = fuente muerta → `sensor_status=FUENTE_NO_DISPONIBLE`
  y `force_flush` de la ventana de ESTA cámara (cierra el denominador en la última
  señal real, honestidad D-018), publicando el resultado vía el event loop.
- **`latest_frame_raw`** (crudo): se escribe acá (gated por `render_enabled`, igual
  que el scheduler viejo, para que el watchdog pueda liberar memoria). El overlay
  (`latest_detections`) y `latest_frame_processed` (render) los escribe el demux,
  que tiene frame + detecciones.
- **`sensor_status`/`last_frame_age_seconds`**: se publican acá para health.

Nota (deferida): la detección fina de "fuente colgada" (frames que dejan de fluir
sin `read()→None`) es best-effort en push — se marca OK por frame y muerto en
`read()→None`; el caso colgado se afina si hace falta (no es foco de 15Hz).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

from ...domain.value_objects import SensorStatus

logger = logging.getLogger(__name__)

_STOP_JOIN_TIMEOUT_S = 2.0


class QueuePushProducer:
    """Daemon thread que lee de un `FrameProducer` y empuja a la cola del worker."""

    def __init__(
        self,
        camera_id: str,
        source,
        worker,
        state,
        *,
        broadcaster=None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        clock=time.monotonic,
    ) -> None:
        self._camera_id = camera_id
        self._source = source
        self._worker = worker
        self._state = state
        self._broadcaster = broadcaster
        self._loop = loop
        self._clock = clock

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    # ---- Lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, name=f"queue-push-{self._camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Señaliza el fin, drena el thread (timeout) y libera el source."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
        try:
            self._source.release()
        except Exception:
            logger.exception("source.release() falló en QueuePushProducer.stop()")

    # ---- Capture thread ------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self._source.read()
                if frame is None:
                    # Fuente muerta/agotada (EOF o max-retries): cerrar honesto.
                    self._on_source_dead()
                    break
                # Empuja a la cola compartida con el frame-clock como ts.
                self._worker.submit(self._camera_id, frame, frame.timestamp)
                self._mark_ok(frame)
        except Exception:
            logger.exception("QueuePushProducer %s: loop de captura falló", self._camera_id)
            self._on_source_dead()
        finally:
            logger.info("QueuePushProducer %s: loop terminado", self._camera_id)

    def _mark_ok(self, frame) -> None:
        self._state.sensor_status = SensorStatus.OK.value
        self._state.last_frame_age_seconds = 0.0
        # Desacople de fluidez (A1): el productor es el ÚNICO escritor de
        # latest_frame_raw Y latest_frame_processed, ambos sobre el frame FRESCO y a
        # tasa de DECODE. El batch worker dejó de escribir processed (solo guarda las
        # cajas en latest_detections). Gated por render_enabled (solo con consumidor
        # MJPEG; el watchdog lo apaga para liberar memoria). El render es copy+draws
        # cv2 (~ms) ≪ 66ms/frame a 15fps → no bloquea el loop de decode.
        if not (self._state.render_enabled and getattr(frame, "image", None) is not None):
            return
        self._state.latest_frame_raw = frame.image.copy()
        self._state.latest_frame_processed = self._render_processed(frame)

    def _render_processed(self, frame):
        """Dibuja las últimas cajas conocidas (`latest_detections`, escritas por el
        batch worker a tasa de inferencia) sobre el frame FRESCO. El drop-to-latest de
        la cola (A2) mantiene esas cajas a ~1 inferencia de lag → alineadas con los
        autos. Antes de la 1ª inferencia (`latest_detections` None) o sin renderer
        inyectado devuelve el frame crudo: video fluido desde el arranque, las cajas
        aparecen al entrar la 1ª detección."""
        renderer = getattr(self._state, "renderer", None)
        detections = self._state.latest_detections
        if renderer is None or detections is None:
            return frame.image.copy()
        analysis = detections[0]  # (result, w, h) — result es FrameAnalysis-compatible
        try:
            return renderer.render(frame, analysis)
        except Exception:
            logger.exception(
                "render falló en el productor para %s; processed = frame crudo",
                self._camera_id,
            )
            return frame.image.copy()

    def _on_source_dead(self) -> None:
        self._state.sensor_status = SensorStatus.FUENTE_NO_DISPONIBLE.value
        aggregator = getattr(self._state, "aggregator", None)
        if aggregator is None:
            return
        # Cierra la ventana en la última señal real (denominador honesto). La
        # persistencia la hace el worker del aggregator; el publish en vivo es
        # best-effort vía el event loop (el producer es un thread).
        try:
            tds = aggregator.force_flush()
        except Exception:
            logger.exception("force_flush falló al morir la fuente %s", self._camera_id)
            return
        if tds and self._broadcaster is not None and self._loop is not None:
            for td in tds:
                try:
                    asyncio.run_coroutine_threadsafe(self._broadcaster.publish(td), self._loop)
                except Exception:
                    logger.exception("publish on-death falló para %s", self._camera_id)
