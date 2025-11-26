import cv2
import yt_dlp
import time
import numpy as np
from typing import Iterator, Optional, Dict, Type
from abc import ABC, abstractmethod
from ..domain import FrameProducer, Frame

class OpenCVSource(FrameProducer):
    """
    Base class for OpenCV-based video sources.
    """
    def __init__(
        self, 
        source: str | int, 
        buffer_size: int = 3,
        target_width: int = None,
        target_height: int = None
    ):
        self.source = source
        self.buffer_size = buffer_size
        self.target_width = target_width
        self.target_height = target_height
        self.cap = None
        self._initialize()

    def _initialize(self):
        print(f"Opening video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.source}")
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        if self.target_width and self.target_height:
             print(f"Will resize frames to {self.target_width}x{self.target_height}")

    def __iter__(self) -> Iterator[Frame]:
        frame_id = 0
        while True:
            if not self.cap:
                break
                
            ret, img = self.cap.read()
            if not ret:
                break
            
            if self.target_width and self.target_height:
                img = cv2.resize(img, (self.target_width, self.target_height))
                
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
        buffer_size: int = 1, # Minimal buffer for live feed
        target_width: int = None,
        target_height: int = None
    ):
        super().__init__(device_id, buffer_size, target_width, target_height)

class YouTubeSource(OpenCVSource):
    """
    Reads from a YouTube URL.
    """
    def __init__(
        self, 
        url: str, 
        format: str = "best",
        buffer_size: int = 3,
        target_width: int = None,
        target_height: int = None
    ):
        self.original_url = url
        self.format = format
        # We don't call super().__init__ immediately because we need to extract the URL first
        self.buffer_size = buffer_size
        self.target_width = target_width
        self.target_height = target_height
        self.cap = None
        
        self._initialize_youtube()

    def _initialize_youtube(self):
        print(f"Attempting to load YouTube video: {self.original_url}")
        
        ydl_opts = {
            'format': self.format,
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
            raise

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


class YouTubeFactory(SourceFactory):
    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "youtube" or \
               (isinstance(config, str) and ("youtube.com" in config or "youtu.be" in config))
    
    def create(self, config: str, **kwargs) -> FrameProducer:
        return YouTubeSource(config, **kwargs)


class WebcamFactory(SourceFactory):
    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "webcam" or \
               (isinstance(config, (int, str)) and str(config).isdigit())
    
    def create(self, config: str, **kwargs) -> FrameProducer:
        device_id = int(config)
        return WebcamSource(device_id, **kwargs)


class VideoFileFactory(SourceFactory):
    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "file" or source_type == "auto"
    
    def create(self, config: str, **kwargs) -> FrameProducer:
        return VideoFileSource(config, **kwargs)


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
