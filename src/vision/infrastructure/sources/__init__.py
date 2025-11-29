"""
Source module initialization and factory registry.
"""
from typing import Dict
from ...domain.protocols import FrameProducer
from .base import SourceFactory, SourceConfig
from .youtube_source import YouTubeSource
from .webcam_source import WebcamSource
from .video_source import VideoFileSource

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
