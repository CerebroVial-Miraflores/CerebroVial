import cv2
import yt_dlp
import numpy as np
from typing import Iterator, Tuple


import time

class VideoSource:
    """
    Handles video capture from local files, YouTube URLs, IP cameras, and webcams.
    Uses cv2.VideoCapture directly for all sources for simplicity and reliability.
    Supports performance tuning for low-latency processing.
    """
    def __init__(
        self, 
        source: str,
        target_width: int = None,
        target_height: int = None,
        buffer_size: int = 3,
        youtube_format: str = "best"
    ):
        self.source = source
        self.cap = None
        self.target_width = target_width
        self.target_height = target_height
        self.buffer_size = buffer_size
        self.youtube_format = youtube_format
        self._initialize_capture()

    def _initialize_capture(self):
        """Initialize video capture based on source type."""
        # Handle YouTube URLs
        if self.source.startswith("http") and ("youtube.com" in self.source or "youtu.be" in self.source):
            try:
                print(f"Attempting to load YouTube video: {self.source}")
                print(f"Requesting format: {self.youtube_format}")
                
                ydl_opts = {
                    'format': self.youtube_format,
                    'noplaylist': True,
                    'quiet': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source, download=False)
                    url = info['url']
                    native_width = info.get('width', 'unknown')
                    native_height = info.get('height', 'unknown')
                    print(f"Stream URL extracted. Native resolution: {native_width}x{native_height}")
                    
                    # Use OpenCV to open the extracted URL
                    self.cap = cv2.VideoCapture(url)
                    
            except Exception as e:
                print(f"Error loading YouTube video: {e}")
                raise
        else:
            # Handle numeric strings as webcam indices
            if self.source.isdigit():
                source = int(self.source)
                print(f"Opening webcam: {source}")
            else:
                source = self.source
                print(f"Opening video source: {source}")
            
            self.cap = cv2.VideoCapture(source)
        
        if not self.cap or not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.source}")
        
        # Set buffer size to reduce latency
        print(f"Setting buffer size to {self.buffer_size}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        # Log target resolution if specified
        if self.target_width and self.target_height:
            print(f"Will resize frames to {self.target_width}x{self.target_height}")

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yields (frame_id, frame) tuples.
        """
        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Resize if target resolution is specified
            if self.target_width and self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            yield frame_id, frame
            frame_id += 1

    def release(self):
        """Release the video capture."""
        if self.cap:
            self.cap.release()
