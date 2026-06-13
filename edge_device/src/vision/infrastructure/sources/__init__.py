"""
Source module initialization and factory registry.
"""
from typing import Dict
from ...domain.protocols import FrameProducer
from .base import SourceFactory, SourceConfig
from .youtube_source import YouTubeSource
from .webcam_source import WebcamSource
from .video_source import VideoFileSource
from .hls_keyframe_source import HlsKeyframeSource
from .full_decode_source import FullDecodeSource

class YouTubeFactory(SourceFactory):
    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "youtube" or \
               (isinstance(config, str) and ("youtube.com" in config or "youtu.be" in config))

    def create(self, config: str, **kwargs) -> FrameProducer:
        capture = kwargs.pop("capture", None)
        resolver = kwargs.pop("resolver", None)
        source_config = self._create_config(**kwargs)
        return YouTubeSource(config, source_config, capture=capture, resolver=resolver)


class WebcamFactory(SourceFactory):
    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "webcam" or \
               (isinstance(config, (int, str)) and str(config).isdigit())

    def create(self, config: str, **kwargs) -> FrameProducer:
        capture = kwargs.pop("capture", None)
        device_id = int(config)
        source_config = self._create_config(**kwargs)
        return WebcamSource(device_id, source_config, capture=capture)


class VideoFileFactory(SourceFactory):
    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "file" or source_type == "auto"

    def create(self, config: str, **kwargs) -> FrameProducer:
        capture = kwargs.pop("capture", None)
        source_config = self._create_config(**kwargs)
        return VideoFileSource(config, source_config, capture=capture)


class HlsKeyframeFactory(SourceFactory):
    """Productor HLS por defecto (B-fase1): keyframe-only sobre ffmpeg CLI.

    Matchea `source_type` "hls"/"stream" (lo que manda el front) y el alias explícito
    "hls_keyframe". Reemplaza el decode completo para HLS: bajo CPU + frescura-a-vivo con
    aislamiento de fallas por subproceso (benchmark Eje 1). El productor OpenCV queda como
    escape hatch explícito en `HlsFactory` ("hls_opencv").
    """

    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type in ("hls", "stream", "hls_keyframe")

    def create(self, config: str, **kwargs) -> FrameProducer:
        capture = kwargs.pop("capture", None)
        source_config = self._create_config(**kwargs)
        return HlsKeyframeSource(config, source_config, capture=capture)


class HlsFactory(SourceFactory):
    """Escape hatch OpenCV para HLS (B-fase1): `source_type` EXPLÍCITO "hls_opencv".

    Sigue registrado (no se borra) para fuentes que requieran la ruta cv2 + Streamlink +
    Referer — p. ej. una URL HLS no-directa que necesite resolución Streamlink (las de
    Claro son `.m3u8` directas y van por keyframe). El default de HLS pasó a
    `HlsKeyframeFactory`; "hls"/"stream" ya no rutean acá. Crea un `VideoFileSource`
    (concreto `OpenCVSource`): `_resolve_stream_url()` (Streamlink) + `_open_capture()`
    (env ffmpeg con Referer), y `read()` se comporta como stream por el prefijo de la URL.
    """

    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "hls_opencv"

    def create(self, config: str, **kwargs) -> FrameProducer:
        capture = kwargs.pop("capture", None)
        source_config = self._create_config(**kwargs)
        return VideoFileSource(config, source_config, capture=capture)


class HlsFullDecodeFactory(SourceFactory):
    """Productor full-decode 15fps (topología B): `source_type` EXPLÍCITO "hls_fulldecode".

    Decodifica el segmento entero y muestrea a fps fijo (vs keyframe-only ~0.5fps).
    Disjunto de "hls"/"stream" (que siguen ruteando al keyframe por defecto) → subir
    a 15Hz es opt-in explícito, no cambia el default.
    """

    def can_handle(self, config: str, source_type: str) -> bool:
        return source_type == "hls_fulldecode"

    def create(self, config: str, **kwargs) -> FrameProducer:
        capture = kwargs.pop("capture", None)
        fps = kwargs.pop("fps", 15)
        source_config = self._create_config(**kwargs)
        return FullDecodeSource(config, source_config, capture=capture, fps=fps)


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
# B-fase1: keyframe-only es el productor HLS por defecto ("hls"/"stream"); el OpenCV
# queda como escape hatch explícito ("hls_opencv"). Disjuntos en can_handle; se registra
# keyframe antes por intención (default gana si algún día se solapan).
_registry.register("hls_keyframe", HlsKeyframeFactory())
_registry.register("hls_opencv", HlsFactory())
# Topología B (15Hz): full-decode opt-in explícito ("hls_fulldecode"); NO toca el
# default keyframe de "hls"/"stream".
_registry.register("hls_fulldecode", HlsFullDecodeFactory())
_registry.register("file", VideoFileFactory())


def create_source(source_config: str, source_type: str = "auto", **kwargs) -> FrameProducer:
    """
    Factory function to create the appropriate FrameProducer using the registry.
    """
    return _registry.create_source(source_config, source_type, **kwargs)
