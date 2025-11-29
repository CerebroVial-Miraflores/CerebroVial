"""
OpenCV-based video source implementation.
"""
import cv2
import time
from typing import Iterator, Union
from ...domain.entities import Frame
from ...domain.protocols import FrameProducer
from ....common.exceptions import SourceError
from .base import SourceConfig

class OpenCVSource(FrameProducer):
    """
    Base class for OpenCV-based video sources.
    """
    def __init__(
        self, 
        source: Union[str, int], 
        config: SourceConfig
    ):
        self.source = source
        self.config = config
        self.cap = None
        self._initialize()

    def _initialize(self):
        try:
            print(f"Opening video source: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise SourceError(
                    f"Could not open video source: {self.source}. "
                    f"Check if the file exists or the camera is connected."
                )
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            
            if self.config.target_width and self.config.target_height:
                 print(f"Will resize frames to {self.config.target_width}x{self.config.target_height}")
        except cv2.error as e:
            raise SourceError(f"OpenCV error initializing source: {e}") from e

    def __iter__(self) -> Iterator[Frame]:
        frame_id = 0
        while True:
            if not self.cap:
                break
                
            ret, img = self.cap.read()
            if not ret:
                break
            
            if self.config.target_width and self.config.target_height:
                img = cv2.resize(img, (self.config.target_width, self.config.target_height))
                
            yield Frame(
                id=frame_id,
                timestamp=time.time(),
                image=img
            )
            frame_id += 1

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

class VideoFileSource(OpenCVSource):
    """
    Reads from a local video file.
    """
    pass
