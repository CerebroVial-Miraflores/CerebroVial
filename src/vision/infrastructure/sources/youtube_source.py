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
            'quiet': True,
            'extractor_args': {'youtube': {'player_client': ['default']}}
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
