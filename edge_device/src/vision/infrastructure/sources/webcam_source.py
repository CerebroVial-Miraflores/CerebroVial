"""
Webcam source implementation.
"""
from .base import SourceConfig
from .video_source import OpenCVSource


class WebcamSource(OpenCVSource):
    """Reads from a webcam."""

    def __init__(self, device_id: int, config: SourceConfig, capture=None):
        super().__init__(device_id, config, capture=capture)
