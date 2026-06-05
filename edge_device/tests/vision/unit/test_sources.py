"""Unit tests for video sources (CT-08.7).

A cv2.VideoCapture-like fake is injected through the factory, so the suite
runs without opencv/yt_dlp installed and exercises the pull-based read() API.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cerebrovial_shared.exceptions import SourceError
from src.vision.infrastructure.sources import (
    VideoFileSource,
    WebcamSource,
    YouTubeSource,
    create_source,
)
from src.vision.infrastructure.sources.base import SourceConfig
from src.vision.infrastructure.sources.video_source import (
    OpenCVSource,
    _host_needs_claro_referer,
)


class FakeCapture:
    """Yields n_frames successful reads, then signals end-of-stream.

    `set()` devuelve True pero NO rebobina (emula un backend cuyo seek miente):
    sirve para ejercitar la guarda anti-loop-infinito (_MAX_REWINDS).
    """

    def __init__(self, n_frames=3, opened=True):
        self._frames_left = n_frames
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._frames_left > 0:
            self._frames_left -= 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        return False, None

    def set(self, *args, **kwargs):
        return True

    def release(self):
        self.released = True


class RewindableFakeCapture(FakeCapture):
    """Como FakeCapture pero `set(POS_FRAMES, 0)` realmente rebobina los frames.

    Emula un backend de archivo cuyo seek funciona, así read() loopea de verdad.
    """

    def __init__(self, n_frames=3, opened=True):
        super().__init__(n_frames=n_frames, opened=opened)
        self._n_frames = n_frames

    def set(self, prop, value):
        if value == 0:
            self._frames_left = self._n_frames
        return True


def test_create_source_auto_file():
    src = create_source("video.mp4", capture=FakeCapture())
    assert isinstance(src, VideoFileSource)


def test_create_source_auto_webcam_int():
    src = create_source(0, capture=FakeCapture())
    assert isinstance(src, WebcamSource)


def test_create_source_auto_webcam_str_digit():
    src = create_source("0", capture=FakeCapture())
    assert isinstance(src, WebcamSource)


def test_create_source_auto_youtube():
    src = create_source("https://youtube.com/watch?v=123", capture=FakeCapture())
    assert isinstance(src, YouTubeSource)


def test_create_source_explicit_type():
    src = create_source("0", source_type="webcam", capture=FakeCapture())
    assert isinstance(src, WebcamSource)


def test_read_returns_frames_then_none():
    # loop=False => semántica EOF -> None (con loop por default ahora rebobinaría).
    src = create_source("video.mp4", loop=False, capture=FakeCapture(n_frames=2))

    f0 = src.read()
    f1 = src.read()
    assert f0.id == 0
    assert f1.id == 1
    assert f0.image.shape == (4, 4, 3)
    assert src.read() is None  # end of file


def test_read_loops_file_at_eof_when_enabled():
    # loop=True (default) + backend que rebobina => read() sigue dando frames al EOF.
    src = create_source("video.mp4", capture=RewindableFakeCapture(n_frames=2))

    # Consumimos más frames que n_frames: sin re-loop, la 3ra lectura sería None.
    frames = [src.read() for _ in range(5)]
    assert all(f is not None for f in frames)
    assert [f.id for f in frames] == [0, 1, 2, 3, 4]


def test_read_degrades_to_none_when_rewind_unsupported():
    # loop=True pero el backend no rebobina (set() miente): la guarda _MAX_REWINDS
    # evita el loop infinito y degrada a EOF -> None.
    src = create_source("video.mp4", capture=FakeCapture(n_frames=1))

    assert src.read() is not None  # frame 0
    assert src.read() is None      # EOF: rewind sin progreso -> None (no cuelga)


def test_release_delegates_to_capture():
    cap = FakeCapture()
    src = create_source("video.mp4", capture=cap)
    src.release()
    assert cap.released is True


def test_unopened_capture_raises_source_error():
    with pytest.raises(SourceError):
        create_source("video.mp4", capture=FakeCapture(opened=False))


# ---- C1/E3: Referer server-side para el HLS de Claro -------------------

_CLARO_URL = "https://video.claro.com.pe/live/cam.m3u8"
_NON_CLARO_URL = "https://cdn.example.com/live/cam.m3u8"
_REFERER = {"Referer": "https://claro.com.pe/"}
_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


def test_host_needs_claro_referer_detects_host():
    assert _host_needs_claro_referer(_CLARO_URL) is True
    assert _host_needs_claro_referer(_NON_CLARO_URL) is False
    assert _host_needs_claro_referer("/app/videos/trafico.mp4") is False
    assert _host_needs_claro_referer(0) is False  # webcam int, no es URL


def _fake_streamlink(recorded):
    class FakeSession:
        def set_option(self, key, value):
            recorded.append((key, value))

        def streams(self, url):
            return {}  # sin 'best' → no muta self.source

    return SimpleNamespace(Streamlink=FakeSession)


def test_resolve_stream_url_sends_referer_for_claro(monkeypatch):
    recorded = []
    monkeypatch.setitem(sys.modules, "streamlink", _fake_streamlink(recorded))
    # capture inyectado → _initialize no toca cv2 y resolve no se auto-llama.
    src = OpenCVSource(_CLARO_URL, SourceConfig(), capture=FakeCapture())
    assert src._needs_referer is True

    src._resolve_stream_url()
    assert ("http-headers", _REFERER) in recorded


def test_resolve_stream_url_omits_referer_for_non_claro(monkeypatch):
    recorded = []
    monkeypatch.setitem(sys.modules, "streamlink", _fake_streamlink(recorded))
    src = OpenCVSource(_NON_CLARO_URL, SourceConfig(), capture=FakeCapture())
    assert src._needs_referer is False

    src._resolve_stream_url()
    assert "http-headers" not in [k for k, _ in recorded]


def test_open_capture_injects_referer_env_for_claro(monkeypatch):
    """E3/SPIKE-D5: el Referer se inyecta a ffmpeg vía OPENCV_FFMPEG_CAPTURE_OPTIONS
    durante la construcción del VideoCapture, y se restaura después."""
    monkeypatch.delenv(_ENV, raising=False)
    seen = {}

    class FakeCv2:
        def VideoCapture(self, source):
            seen["env"] = os.environ.get(_ENV)
            return FakeCapture()

    src = OpenCVSource(_CLARO_URL, SourceConfig(), capture=FakeCapture())
    src._open_capture(FakeCv2())

    assert seen["env"] == "referer;https://claro.com.pe/"  # presente al abrir
    assert os.environ.get(_ENV) is None                    # restaurado después


def test_open_capture_no_referer_env_for_non_claro(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    seen = {}

    class FakeCv2:
        def VideoCapture(self, source):
            seen["env"] = os.environ.get(_ENV)
            return FakeCapture()

    src = OpenCVSource("video.mp4", SourceConfig(), capture=FakeCapture())
    src._open_capture(FakeCv2())

    assert seen["env"] is None  # archivo local: sin Referer
