import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from src.vision.application.builders.pipeline_builder import VisionApplicationBuilder

def test_builder_constructs_complete_pipeline():
    """Test that the builder constructs a complete functional pipeline."""
    cfg = OmegaConf.create({
        'vision': {
            'source': 'test_video.mp4',
            'source_type': 'file',
            'model': {'path': 'yolo11n.pt', 'conf_threshold': 0.5},
            'performance': {
                'detect_every_n_frames': 3,
                'opencv_buffer_size': 3,
                'target_width': 1280,
                'target_height': 720
            },
            'zones': {},
            'speed_estimation': {'enabled': False},
            'persistence': {'enabled': False}
        }
    })
    
    # Mock cv2.VideoCapture and YOLO to avoid external dependencies
    with patch('src.vision.infrastructure.sources.video_source.cv2.VideoCapture') as mock_cap, \
         patch('src.vision.infrastructure.detection.yolo_detector.YOLO') as mock_yolo:
        
        mock_cap.return_value.isOpened.return_value = True
        
        builder = VisionApplicationBuilder(cfg)
        pipeline = (
            builder
            .build_detector()
            .build_tracker()
            .build_speed_estimator()
            .build_zones()
            .build_persistence()
            .build_source()
            .build_pipeline()
        )
        
        assert pipeline is not None
        assert builder.detector is not None
        assert builder.source is not None
        assert builder.tracker is not None
        
        # Verify components are linked in pipeline
        assert pipeline.source == builder.source
        assert pipeline.metrics_collector is not None
