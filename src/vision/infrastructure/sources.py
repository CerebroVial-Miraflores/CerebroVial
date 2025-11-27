import cv2
import yt_dlp
import time
import numpy as np
from typing import Iterator, Optional, Dict, Type, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, validator, Field
from ..domain import FrameProducer, Frame
from ...common.exceptions import SourceError

class SourceConfig(BaseModel):
    """Configuración validada para fuentes de video"""
    buffer_size: int = Field(3, ge=1, le=10, description="OpenCV buffer size")
    target_width: Optional[int] = Field(None, gt=0, description="Target width in pixels")
    target_height: Optional[int] = Field(None, gt=0, description="Target height in pixels")
    format: str = Field("best", description="YouTube format")

    @validator('target_width', 'target_height')
    def validate_resolution(cls, v, values):
        if v is not None and v % 2 != 0:
            raise ValueError('Resolution must be even number for video encoding')
        return v

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

class YouTubeSource(OpenCVSource):
    """
    Reads from a YouTube URL.
    """
    def __init__(
        self, 
        url: str, 
        config: SourceConfig
    ):
        self.original_url = url
        # We don't call super().__init__ immediately because we need to extract the URL first
        self.config = config
        self.cap = None
        
        self._initialize_youtube()

    def _initialize_youtube(self):
        print(f"Attempting to load YouTube video: {self.original_url}")
        
        ydl_opts = {
            'format': self.config.format,
            'noplaylist': True,
            'quiet': True
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.original_url, download=False)
                stream_url = info['url']
                print(f"Stream URL extracted.")
                
                # Now initialize the OpenCV source with the stream URL
                self.source = stream_url
                self._initialize()
        except Exception as e:
            print(f"Error loading YouTube video: {e}")
            raise SourceError(f"Failed to load YouTube video: {e}") from e

class SourceFactory(ABC):
    """
    Abstract factory for creating video sources.
    """
    
    @abstractmethod
    def create(self, config: str, **kwargs) -> FrameProducer:
        pass
    
    @abstractmethod
    def can_handle(self, config: str, source_type: str) -> bool:
        pass

    def _create_config(self, **kwargs) -> SourceConfig:
        return SourceConfig(**kwargs)


class YouTubeFactory(SourceFactory):
    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "youtube" or \
               (isinstance(config, str) and ("youtube.com" in config or "youtu.be" in config))
    
    def create(self, config: str, **kwargs) -> FrameProducer:
        source_config = self._create_config(**kwargs)
        return YouTubeSource(config, source_config)


class WebcamFactory(SourceFactory):
    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "webcam" or \
               (isinstance(config, (int, str)) and str(config).isdigit())
    
    def create(self, config: str, **kwargs) -> FrameProducer:
        device_id = int(config)
        source_config = self._create_config(**kwargs)
        return WebcamSource(device_id, source_config)


class VideoFileFactory(SourceFactory):
    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "file" or source_type == "auto"
    
    def create(self, config: str, **kwargs) -> FrameProducer:
        source_config = self._create_config(**kwargs)
        return VideoFileSource(config, source_config)


class SourceRegistry:
    """
    Centralized registry for source factories.
    """
    
    def __init__(self):
        self._factories: Dict[str, SourceFactory] = {}
    
    def register(self, name: str, factory: SourceFactory):
        self._factories[name] = factory
    
    def create_source(self, config: str, source_type: str = "auto", **kwargs) -> FrameProducer:
        for factory in self._factories.values():
            if factory.can_handle(config, source_type):
                return factory.create(config, **kwargs)
        
        raise ValueError(f"No factory found for source: {config}")


# Setup global registry
_registry = SourceRegistry()
_registry.register("youtube", YouTubeFactory())
_registry.register("webcam", WebcamFactory())
_registry.register("file", VideoFileFactory())


def create_source(source_config: str, source_type: str = "auto", **kwargs) -> FrameProducer:
    """
    Factory function to create the appropriate FrameProducer using the registry.
    """
    return _registry.create_source(source_config, source_type, **kwargs)
