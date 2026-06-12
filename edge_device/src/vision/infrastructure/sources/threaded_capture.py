"""ThreadedCapture — captura threaded sobre un `FrameProducer` pull (B1 1b, D-018).

Envuelve un `FrameProducer` (p.ej. `OpenCVSource`, sin tocarlo) y corre su
`read()` bloqueante en un daemon thread, guardando SOLO el último frame en un
slot protegido por lock (drop-all-but-newest, sin cola ni catch-up). Expone ese
frame + su frescura por `snapshot()` de forma **no-bloqueante**, conforme al
Protocol `LiveFrameSource`.

Por qué existe (la pieza que unifica 1b):
- "No inferir sobre stream vacío": el scheduler consulta `snapshot()` y decide.
- Read-colgado defensivo: si el `read()` del stream se cuelga, el thread queda
  atascado pero `snapshot()` devuelve al instante (solo lee el slot); la edad del
  último frame crece → el scheduler lo lee como no-fresco y NO espera. El bug del
  `video_source.py` (reopen-sin-frame que nunca corta) queda **contenido**, no
  arreglado (deuda con trigger).
- Costura agnóstica: la frescura vive en el contrato, no en lógica HLS. Un archivo
  (re-loopea o EOF) y un stream (frames continuos o caída) responden según su
  naturaleza sin que el scheduler los distinga.

Reloj **monotónico** (`time.monotonic`, inyectable para tests): un salto de
wall-clock (NTP) no debe falsear la edad de un frame que decide inferir-o-no.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from ...domain.entities import Frame, FrameSnapshot
from ...domain.protocols import FrameProducer

logger = logging.getLogger(__name__)

_STOP_JOIN_TIMEOUT_S = 2.0


class ThreadedCapture:
    """Captura threaded `LiveFrameSource` que envuelve un `FrameProducer` pull."""

    def __init__(
        self,
        source: FrameProducer,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._clock = clock

        self._lock = threading.Lock()
        self._latest_frame: Optional[Frame] = None
        self._latest_ts: Optional[float] = None  # monotónico, instante de captura
        # live: True tras el primer frame; False ante EOF/fuente muerta detectada.
        # Un read() colgado NO lo cambia (queda True con age creciendo) — a propósito.
        self._live = False

        self._started = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- LiveFrameSource ----------------------------------------------

    def start(self) -> None:
        """Arranca el thread de captura (idempotente)."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="ThreadedCapture", daemon=True
        )
        self._thread.start()

    def snapshot(self) -> FrameSnapshot:
        """No-bloqueante: último frame + edad monotónica + liveness."""
        with self._lock:
            frame = self._latest_frame
            ts = self._latest_ts
            live = self._live
        age = None if ts is None else max(0.0, self._clock() - ts)
        return FrameSnapshot(frame=frame, age_seconds=age, live=live)

    def stop(self) -> None:
        """Señaliza el fin, drena el thread (timeout) y libera el source."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
        try:
            self._source.release()
        except Exception:
            logger.exception("source.release() falló en ThreadedCapture.stop()")

    # ---- Capture thread -----------------------------------------------

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                # read() bloquea / reintenta / reconecta a su antojo: lo absorbe
                # ESTE thread, no el scheduler. None == EOF o max-retries → fuente
                # agotada/muerta (contrato del FrameProducer).
                frame = self._source.read()
                if frame is None:
                    with self._lock:
                        self._live = False
                    break
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = self._clock()
                    self._live = True
        except Exception:
            logger.exception("ThreadedCapture: capture loop falló")
            with self._lock:
                self._live = False
        finally:
            logger.info("ThreadedCapture: capture loop terminado")
