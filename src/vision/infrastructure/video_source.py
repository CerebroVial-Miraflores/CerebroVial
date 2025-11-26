import cv2
import yt_dlp
import numpy as np
from typing import Iterator, Tuple
import subprocess
import imageio_ffmpeg


class VideoSource:
    """
    Handles video capture from local files or YouTube URLs using FFmpeg for streams
    and OpenCV for local files/webcams.
    """
    def __init__(self, source: str):
        self.source = source
        self.cap = None
        self.process = None
        self.width = 1280
        self.height = 720
        self.use_ffmpeg = False
        self._initialize_capture()

    def _initialize_capture(self):
        if self.source.startswith("http") and ("youtube.com" in self.source or "youtu.be" in self.source):
            try:
                print(f"Attempting to load YouTube video: {self.source}")
                ydl_opts = {
                    'format': 'best[ext=mp4]/best',
                    'noplaylist': True,
                    'quiet': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source, download=False)
                    url = info['url']
                    # Try to get resolution from info, default to 1280x720
                    self.width = info.get('width', 1280)
                    self.height = info.get('height', 720)
                    print(f"Stream URL extracted. Resolution: {self.width}x{self.height}")
                    
                    self.use_ffmpeg = True
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    
                    command = [
                        ffmpeg_exe,
                        '-reconnect', '1',
                        '-reconnect_streamed', '1',
                        '-reconnect_delay_max', '5',
                        '-i', url,
                        '-f', 'image2pipe',
                        '-pix_fmt', 'bgr24',
                        '-vcodec', 'rawvideo',
                        '-'
                    ]
                    
                    # Suppress ffmpeg logs
                    self.process = subprocess.Popen(
                        command, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.DEVNULL,
                        bufsize=10**8
                    )
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
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source: {self.source}")

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yields (frame_id, frame) tuples.
        """
        frame_id = 0
        while True:
            if self.use_ffmpeg:
                # Read raw video frame from stdout
                raw_frame = self.process.stdout.read(self.width * self.height * 3)
                if len(raw_frame) != self.width * self.height * 3:
                    print("End of stream or error reading frame.")
                    break
                
                # Convert to numpy array and make a copy to ensure it's writable
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    break
            
            yield frame_id, frame
            frame_id += 1

    def release(self):
        if self.process:
            self.process.kill()
        if self.cap:
            self.cap.release()
