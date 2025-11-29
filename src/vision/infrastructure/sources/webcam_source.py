"""
Webcam source implementation.
"""
from .video_source import OpenCVSource
from .base import SourceConfig

class WebcamSource(OpenCVSource):
    """
    Reads from a webcam.
    """
    def __init__(
        self, 
        device_id: int, 
        config: SourceConfig
    ):
        super().__init__(device_id, config)
