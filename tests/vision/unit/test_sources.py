import pytest
from unittest.mock import Mock, patch, MagicMock
from src.vision.infrastructure.sources import create_source, VideoFileSource, WebcamSource, YouTubeSource

def test_create_source_auto_file():
    with patch('src.vision.infrastructure.sources.video_source.cv2.VideoCapture') as mock_cap:
        mock_cap.return_value.isOpened.return_value = True
        source = create_source("video.mp4")
        assert isinstance(source, VideoFileSource)

def test_create_source_auto_webcam_int():
    with patch('src.vision.infrastructure.sources.video_source.cv2.VideoCapture') as mock_cap:
        mock_cap.return_value.isOpened.return_value = True
        source = create_source(0)
        assert isinstance(source, WebcamSource)

def test_create_source_auto_webcam_str_digit():
    with patch('src.vision.infrastructure.sources.video_source.cv2.VideoCapture') as mock_cap:
        mock_cap.return_value.isOpened.return_value = True
        source = create_source("0")
        assert isinstance(source, WebcamSource)

def test_create_source_auto_youtube():
    with patch('src.vision.infrastructure.sources.youtube_source.yt_dlp.YoutubeDL') as mock_ydl:
        mock_ydl_instance = mock_ydl.return_value.__enter__.return_value
        mock_ydl_instance.extract_info.return_value = {'url': 'http://stream.url'}
        
        with patch('src.vision.infrastructure.sources.video_source.cv2.VideoCapture') as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            source = create_source("https://youtube.com/watch?v=123")
            assert isinstance(source, YouTubeSource)

def test_create_source_explicit_type():
    with patch('src.vision.infrastructure.sources.video_source.cv2.VideoCapture') as mock_cap:
        mock_cap.return_value.isOpened.return_value = True
        # Force webcam type
        source = create_source("0", source_type="webcam")
        assert isinstance(source, WebcamSource)
