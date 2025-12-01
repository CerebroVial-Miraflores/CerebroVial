"""
YouTube source implementation.
"""
import yt_dlp
from ....common.exceptions import SourceError
from .video_source import OpenCVSource
from .base import SourceConfig

class YouTubeSource(OpenCVSource):
    """
    Reads from a YouTube URL.
    """
    def __init__(self, url: str, config: SourceConfig):
        # Note: Factory passes (config, url) but OpenCVSource expects (source, config)
        # We need to align with how the factory calls it.
        # Checking YouTubeFactory in src/vision/infrastructure/sources/__init__.py:
        # return YouTubeSource(config, source_config) -> wait, factory passes (config, source_config)
        # Let's check the factory again.
        
        # Actually, let's look at the file content of __init__.py from previous turns.
        # YouTubeFactory.create: return YouTubeSource(config, source_config)
        # config is the URL string, source_config is the SourceConfig object.
        # So arguments are (url, config).
        
        # But OpenCVSource expects (source, config).
        
        try:
            # Attempt to use base class initialization (Streamlink)
            super().__init__(source=url, config=config)
        except SourceError as e:
            print(f"[WARNING] Streamlink failed for YouTube ({e}), trying yt_dlp fallback...")
            self.config = config
            self.original_url = url
            self._initialize_youtube_fallback()

    def _initialize_youtube_fallback(self):
        print(f"Attempting to load YouTube video with yt_dlp: {self.original_url}")
        
        ydl_opts = {
            'format': self.config.format if hasattr(self.config, 'format') else 'best',
            'noplaylist': True,
            'quiet': True,
            'extractor_args': {'youtube': {'player_client': ['default']}}
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.original_url, download=False)
                stream_url = info['url']
                print(f"Stream URL extracted via yt_dlp.")
                
                # Now initialize the OpenCV source with the stream URL
                self.source = stream_url
                self._initialize()
        except Exception as e:
            print(f"Error loading YouTube video: {e}")
            raise SourceError(f"Failed to load YouTube video: {e}") from e
