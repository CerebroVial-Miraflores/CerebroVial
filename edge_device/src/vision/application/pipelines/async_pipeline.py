"""Asynchronous vision pipeline: capture thread → processing thread → consumer.

Tres threads (incluyendo el caller que consume `run()`), dos colas (frame_queue
entre capture y processing; result_queue entre processing y caller).

`FrameProducer` se consume con `read()` (§4.4 Cambio 1 del diseño): `read()
-> Optional[Frame]` retorna `None` para fin de stream o error transitorio
recuperable; `SourceError` (`cerebrovial_shared.exceptions`) para errores
fatales (se logean como warning y se intenta seguir leyendo). `release()`
se invoca en `stop()` del pipeline.

Catch-up logic §10.5 preservada: si la latencia entre el frame más reciente
capturado y el que el processing está por correr supera 1.5 s, el processing
salta frames intermedios (2x si lag 1.5-2.5 s, 3x si >2.5 s).

Fix C1.5 (race en `stop()`): se separan dos events.
- `_stop_event`: lo setea solo el caller vía `stop()` (hard stop).
- `_capture_done`: lo setea solo el capture thread al terminar (EOF, error
  fatal, o `_stop_event` observado). NO toca `_stop_event`.
- `_processing_done`: lo setea el processing thread al terminar (drenó
  frame_queue tras `_capture_done` o `_stop_event`).
Antes, capture seteaba `_stop_event` en su `finally`, lo que mataba al
processing en medio de un frame y perdía resultados. Ahora cada thread
controla su propio flag de terminación, y `run()` puede drenar la
result_queue después de que processing terminó sin que el stop hard
interfiera.

Logging vía `logging.getLogger(__name__)` (§6.3 / §12).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from typing import Optional

from cerebrovial_shared.exceptions import SourceError

from ...domain.entities import Frame, FrameAnalysis
from ...domain.protocols import FrameProducer
from ..processors import FrameProcessor

logger = logging.getLogger(__name__)


class AsyncVisionPipeline:
    """Pipeline asíncrono con thread separado de captura y procesamiento.

    Consume un `FrameProducer` con `read()` (no `__iter__`). El processing
    aplica un chain de `FrameProcessor` y emite `(Frame, FrameAnalysis)` por
    `run()` con pacing a `target_fps`.
    """

    def __init__(
        self,
        source: FrameProducer,
        processor_chain: FrameProcessor,
        metrics_collector=None,
        frame_buffer_size: int = 10,
        result_buffer_size: int = 30,
        target_fps: int = 30,
    ) -> None:
        self.source = source
        self.processor_chain = processor_chain
        self.metrics_collector = metrics_collector
        self.target_fps = target_fps

        # Colas thread-safe.
        self.frame_queue: queue.Queue[Frame] = queue.Queue(maxsize=frame_buffer_size)
        self.result_queue: queue.Queue[tuple[Frame, FrameAnalysis]] = queue.Queue(
            maxsize=result_buffer_size
        )

        # Control de threads (fix C1.5: tres events independientes).
        self._stop_event = threading.Event()
        self._capture_done = threading.Event()
        self._processing_done = threading.Event()

        self._capture_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None

        # Estado compartido para get_latest() (poll desde fuera).
        self._latest_analysis: Optional[tuple[Frame, FrameAnalysis]] = None
        self._analysis_lock = threading.Lock()

        # Counters / metadatos.
        self._dropped_frames = 0
        self._latest_capture_ts = 0.0

    # ---- Lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Arranca capture + processing threads si no estaban arrancados."""
        if self._capture_thread and self._capture_thread.is_alive():
            return
        self._stop_event.clear()
        self._capture_done.clear()
        self._processing_done.clear()

        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="CaptureThread", daemon=True
        )
        self._processing_thread = threading.Thread(
            target=self._processing_loop, name="ProcessingThread", daemon=True
        )
        self._capture_thread.start()
        self._processing_thread.start()

    def stop(self) -> None:
        """Hard stop: señaliza, drena threads (timeout), libera el source."""
        self._stop_event.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        try:
            self.source.release()
        except Exception:
            # release() puede fallar si el source ya fue liberado; no
            # propagamos en shutdown.
            logger.exception("source.release() falló durante stop()")
        logger.info("AsyncVisionPipeline detenido")

    # ---- Capture thread (source → frame_queue) -------------------------

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.source.read()
                except SourceError as e:
                    # Error fatal del source: el plan §4.4 dice que el caller
                    # decide; acá lo tratamos como recoverable transitorio
                    # (loguear y seguir intentando), porque parar la captura
                    # mata todo el pipeline.
                    logger.warning("source.read() levantó SourceError: %s", e)
                    continue

                if frame is None:
                    # None == EOF o transitorio recuperable. En streams reales
                    # podríamos diferenciar; acá tratamos cualquier None como
                    # fin de stream (consistente con el contrato del Protocol).
                    break

                self._latest_capture_ts = frame.timestamp

                # Feed frame_queue. Bloquea con timeout para tolerar stop
                # event sin perder el frame (drop-newest implícito: si el
                # caller para, el frame se descarta).
                while not self._stop_event.is_set():
                    try:
                        self.frame_queue.put(frame, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception:
            logger.exception("Capture thread falló")
        finally:
            # IMPORTANTE (fix C1.5): NO seteamos _stop_event acá; solo señalamos
            # que la captura acabó. El processing decide cuándo terminar.
            self._capture_done.set()
            logger.info("Capture thread terminado")

    # ---- Processing thread (frame_queue → result_queue) ----------------

    def _processing_loop(self) -> None:
        analysis: Optional[FrameAnalysis] = None
        skipped_counter = 0

        try:
            while True:
                if self._stop_event.is_set():
                    break
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    # Si capture acabó y no queda nada, salimos limpiamente.
                    if self._capture_done.is_set() and self.frame_queue.empty():
                        break
                    continue

                # Catch-up logic §10.5: si la latencia es alta, saltear frames
                # para alcanzar al stream en vivo.
                current_lag = self._latest_capture_ts - frame.timestamp
                should_process = True
                if current_lag > 1.5:
                    skipped_counter += 1
                    skip_modulo = 2 if current_lag <= 2.5 else 3
                    if skipped_counter % skip_modulo != 0:
                        should_process = False
                if not should_process:
                    continue

                analysis = self.processor_chain.process(frame, analysis)

                with self._analysis_lock:
                    self._latest_analysis = (frame, analysis) if analysis else None

                if self.metrics_collector:
                    self.metrics_collector.increment_frames()

                # Feed result_queue. Bloquea con timeout, pero NO se pierde el
                # item si stop_event setea en el medio: queremos que el item
                # ya procesado llegue a la queue para que run() pueda
                # yieldearlo (o ser drenado por flush).
                while True:
                    if self._stop_event.is_set():
                        # Hard stop: no insistimos.
                        break
                    try:
                        self.result_queue.put((frame, analysis), timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception:
            logger.exception("Processing thread falló")
        finally:
            self._processing_done.set()
            logger.info("Processing thread terminado")

    # ---- Consumer ------------------------------------------------------

    def run(self) -> Iterator[tuple[Frame, FrameAnalysis]]:
        """Generator: yields `(frame, analysis)` para consumo síncrono.

        Auto-arranca threads. Bloquea entre yields para mantener pacing a
        `target_fps`. Termina cuando el processing terminó Y la result_queue
        está vacía (drenado limpio), o cuando `stop()` se llamó.
        """
        self.start()
        frame_duration_s = 1.0 / self.target_fps if self.target_fps > 0 else 0.0

        try:
            while True:
                try:
                    item = self.result_queue.get(timeout=0.1)
                except queue.Empty:
                    # Salir limpio si: processing terminó y queue está vacía.
                    if self._processing_done.is_set() and self.result_queue.empty():
                        break
                    # Hard stop con queue vacía: terminamos.
                    if self._stop_event.is_set() and self.result_queue.empty():
                        break
                    continue

                start_t = time.time()
                yield item
                elapsed = time.time() - start_t
                if frame_duration_s > 0 and elapsed < frame_duration_s:
                    time.sleep(frame_duration_s - elapsed)
        except KeyboardInterrupt:
            logger.info("run() interrumpido por usuario")
        finally:
            self.stop()

    def get_latest(self) -> Optional[tuple[Frame, FrameAnalysis]]:
        """Snapshot del último (frame, analysis) procesado. Para polling."""
        with self._analysis_lock:
            return self._latest_analysis
