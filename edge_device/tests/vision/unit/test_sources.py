"""Unit tests for video sources (CT-08.7).

A cv2.VideoCapture-like fake is injected through the factory, so the suite
runs without opencv/yt_dlp installed and exercises the pull-based read() API.
"""
import numpy as np
import pytest

from cerebrovial_shared.exceptions import SourceError
from src.vision.infrastructure.sources import (
    VideoFileSource,
    WebcamSource,
    YouTubeSource,
    create_source,
)


class FakeCapture:
    """Yields n_frames successful reads, then signals end-of-stream."""

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
    src = create_source("video.mp4", capture=FakeCapture(n_frames=2))

    f0 = src.read()
    f1 = src.read()
    assert f0.id == 0
    assert f1.id == 1
    assert f0.image.shape == (4, 4, 3)
    assert src.read() is None  # end of file


def test_release_delegates_to_capture():
    cap = FakeCapture()
    src = create_source("video.mp4", capture=cap)
    src.release()
    assert cap.released is True


def test_unopened_capture_raises_source_error():
    with pytest.raises(SourceError):
        create_source("video.mp4", capture=FakeCapture(opened=False))
