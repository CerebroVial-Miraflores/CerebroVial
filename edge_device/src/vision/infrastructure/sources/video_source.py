"""
OpenCV-based video source implementation.
"""
import logging
import time
from typing import Optional

from ...domain.entities import Frame
from ...domain.protocols import FrameProducer
from cerebrovial_shared.exceptions import SourceError
from .base import SourceConfig

logger = logging.getLogger(__name__)

_MAX_STREAM_RETRIES = 500
_MAX_REWINDS = 3
_STREAM_PREFIXES = ("http", "https", "rtsp", "udp")
_CV2_CAP_PROP_POS_FRAMES = 1  # cv2.CAP_PROP_POS_FRAMES (valor estable de OpenCV)


class OpenCVSource(FrameProducer):
    """OpenCV-backed video source exposing the pull-based FrameProducer API.

    `cv2` is imported lazily (only on the real path) so the module collects
    without OpenCV installed. A `capture` (cv2.VideoCapture-like) can be
    injected for tests, in which case cv2 is never imported.
    """

    def __init__(self, source, config: SourceConfig, capture=None):
        self.source = source
        self.config = config
        self.cap = capture
        self._injected = capture is not None
        self._cv2 = None
        self._frame_id = 0
        self._retry_count = 0

        if not self._injected and isinstance(source, str) and source.startswith(("http", "https")):
            self._resolve_stream_url()

        self._initialize()

    def _resolve_stream_url(self):
        """Resolve an http(s) URL to a low-latency stream URL via Streamlink."""
        try:
            import streamlink

            session = streamlink.Streamlink()
            session.set_option("hls-live-edge", 2)
            session.set_option("hls-segment-threads", 3)
            session.set_option("stream-timeout", 15)
            streams = session.streams(self.source)
            if "best" in streams:
                self.source = streams["best"].url
        except Exception as e:  # noqa: BLE001 - fall back to the original URL
            logger.warning(f"Streamlink resolution failed: {e}. Using original URL.")

    def _initialize(self):
        if not self._injected:
            import cv2

            self._cv2 = cv2
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise SourceError(
                f"Could not open video source: {self.source}. "
                f"Check if the file exists or the camera is connected."
            )

        if self._cv2 is not None and self.config.buffer_size:
            self.cap.set(self._cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

    def _is_stream(self) -> bool:
        return isinstance(self.source, str) and self.source.startswith(_STREAM_PREFIXES)

    def _reconnect(self) -> bool:
        """Release and reopen a live stream; returns True if reopened."""
        self.cap.release()
        time.sleep(1.0)
        try:
            import cv2

            self._cv2 = cv2
            self.cap = cv2.VideoCapture(self.source)
            if self.config.buffer_size:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            if self.cap.isOpened():
                logger.info("Stream reconnected.")
                self._retry_count = 0
                return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Reconnection failed: {e}")
        return False

    def _rewind(self) -> bool:
        """Reposiciona un archivo al frame 0 para re-loop.

        Intenta primero el seek (`CAP_PROP_POS_FRAMES`); si el backend lo
        ignora o lanza, degrada a reabrir la captura. Devuelve True si se pudo
        intentar el rewind; la guarda de `read()` (`_MAX_REWINDS`) evita girar
        sin frames cuando el seek miente.
        """
        pos = (
            self._cv2.CAP_PROP_POS_FRAMES
            if self._cv2 is not None
            else _CV2_CAP_PROP_POS_FRAMES
        )
        try:
            if self.cap.set(pos, 0):
                logger.info("File source EOF: rewinding to start (loop).")
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Seek to frame 0 failed: {e}; reopening file.")
        # Degradar: reabrir la captura desde cero.
        return self._reopen_file()

    def _reopen_file(self) -> bool:
        """Reabre la captura de archivo (fallback cuando el seek no se soporta)."""
        if self._injected:
            return False
        try:
            import cv2

            self._cv2 = cv2
            self.cap.release()
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                logger.info("File source reopened for loop.")
                return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"File reopen failed: {e}")
        return False

    def read(self) -> Optional[Frame]:
        if self.cap is None:
            return None

        rewinds = 0
        while True:
            ret, img = self.cap.read()
            if ret:
                self._retry_count = 0
                if self.config.target_width and self.config.target_height:
                    if self._cv2 is None:
                        import cv2

                        self._cv2 = cv2
                    img = self._cv2.resize(
                        img, (self.config.target_width, self.config.target_height)
                    )
                frame = Frame(id=self._frame_id, timestamp=time.time(), image=img)
                self._frame_id += 1
                return frame

            # Read failed.
            if not self._is_stream():
                # EOF de archivo. Si el loop está habilitado y la fuente es un
                # path de archivo, rebobinar al inicio y reintentar. El contador
                # `rewinds` acota el caso degenerado en que el backend dice haber
                # rebobinado (set() -> True) pero no entrega frames: tras
                # _MAX_REWINDS sin progreso degradamos a EOF -> None.
                if (
                    self.config.loop
                    and isinstance(self.source, str)
                    and rewinds < _MAX_REWINDS
                    and self._rewind()
                ):
                    rewinds += 1
                    continue
                return None  # end of file (loop off, webcam, o rewind sin progreso)

            logger.warning(f"Stream disconnected. Reconnecting... (attempt {self._retry_count + 1})")
            if self._reconnect():
                continue
            self._retry_count += 1
            if self._retry_count > _MAX_STREAM_RETRIES:
                logger.error("Max retries reached. Stopping.")
                return None

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None


class VideoFileSource(OpenCVSource):
    """Reads from a local video file."""
    pass
