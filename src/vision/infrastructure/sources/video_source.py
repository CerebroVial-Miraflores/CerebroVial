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
    def __init__(self, source: str, config: SourceConfig):
        self.source = source
        self.config = config
        self.cap = None
        
        # Try to resolve URL with Streamlink for better stability
        if isinstance(source, str) and (source.startswith("http") or source.startswith("https")):
            try:
                import streamlink
                streams = streamlink.streams(source)
                if "best" in streams:
                    resolved_url = streams["best"].url
                    print(f"[INFO] Streamlink resolved URL: {resolved_url[:50]}...")
                    self.source = resolved_url
            except Exception as e:
                print(f"[WARNING] Streamlink resolution failed: {e}. Using original URL.")

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
            if self.config.buffer_size:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                
            if self.config.target_width and self.config.target_height:
                 print(f"Will resize frames to {self.config.target_width}x{self.config.target_height}")
        except cv2.error as e:
            raise SourceError(f"OpenCV error initializing source: {e}") from e

    def __iter__(self) -> Iterator[Frame]:
        frame_id = 0
        retry_count = 0
        max_retries = 500  # Infinite-ish retries for live stream
        
        while True:
            if not self.cap:
                break
                
            ret, img = self.cap.read()
            if not ret:
                # Check if it's a network stream to attempt reconnection
                is_stream = isinstance(self.source, str) and (
                    self.source.startswith("http") or 
                    self.source.startswith("rtsp") or
                    self.source.startswith("udp")
                )
                
                if is_stream:
                    print(f"[WARNING] Stream disconnected. Reconnecting... (Attempt {retry_count+1})")
                    self.cap.release()
                    time.sleep(1.0) # Wait before reconnecting
                    
                    try:
                        self.cap = cv2.VideoCapture(self.source)
                        if self.config.buffer_size:
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                        
                        if self.cap.isOpened():
                            print("[INFO] Stream reconnected.")
                            retry_count = 0
                            continue
                    except Exception as e:
                        print(f"[ERROR] Reconnection failed: {e}")
                    
                    retry_count += 1
                    if retry_count > max_retries:
                        print("[ERROR] Max retries reached. Stopping.")
                        break
                    continue
                else:
                    # End of file
                    print("[DEBUG] Stream ended (not identified as stream or ret=False).")
                    break
            
            # Reset retry count on successful read
            retry_count = 0
            
            if self.config.target_width and self.config.target_height:
                img = cv2.resize(img, (self.config.target_width, self.config.target_height))
                
            yield Frame(
                id=frame_id,
                timestamp=time.time(),
                image=img
            )
            frame_id += 1
        print("[DEBUG] OpenCVSource iterator finished.")

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

class VideoFileSource(OpenCVSource):
    """
    Reads from a local video file.
    """
    pass
