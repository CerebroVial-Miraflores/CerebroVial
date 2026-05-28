"""
YouTube source implementation.
"""
import logging

from cerebrovial_shared.exceptions import SourceError
from .base import SourceConfig
from .video_source import OpenCVSource

logger = logging.getLogger(__name__)


def _default_youtube_resolver(url: str, config: SourceConfig) -> str:
    """Resolve a YouTube URL to a direct stream URL via yt_dlp (imported lazily)."""
    import yt_dlp

    ydl_opts = {
        "format": getattr(config, "format", "best"),
        "noplaylist": True,
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["default"]}},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]


class YouTubeSource(OpenCVSource):
    """Reads from a YouTube URL.

    Resolution path: Streamlink first (via OpenCVSource), then a yt_dlp
    fallback. `yt_dlp` is imported lazily inside the resolver, and both the
    capture and the resolver are injectable so tests run without yt_dlp/cv2.
    """

    def __init__(self, url: str, config: SourceConfig, capture=None, resolver=None):
        self.original_url = url
        self._resolver = resolver or _default_youtube_resolver

        if capture is not None:
            super().__init__(url, config, capture=capture)
            return

        try:
            super().__init__(url, config)
        except SourceError as e:
            logger.warning(f"Streamlink failed for YouTube ({e}); trying yt_dlp fallback.")
            self.source = self._resolver(url, config)
            self._initialize()
