import pytest
import os
import shutil
import time
import numpy as np
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from src.vision.application.builder import VisionApplicationBuilder
from src.vision.domain import Frame, FrameAnalysis, DetectedVehicle, ZoneVehicleCount

@pytest.fixture
def temp_output_dir():
    """Create and cleanup a temporary directory for test outputs."""
    dir_path = "tests/data/temp_traffic_logs"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def test_persistence_integration(temp_output_dir):
    """
    Test that the pipeline correctly persists data to CSV when enabled.
    """
    # 1. Configure Builder with persistence enabled
    cfg = OmegaConf.create({
        'vision': {
            'source': 'test_video.mp4',
            'source_type': 'file',
            'model': {'path': 'yolo11n.pt', 'conf_threshold': 0.5},
            'performance': {
                'detect_every_n_frames': 1,
                'opencv_buffer_size': 1,
                'target_width': 1280,
                'target_height': 720
            },
            'zones': {
                'zone1': [[0,0], [100,0], [100,100], [0,100]]
            },
            'speed_estimation': {'enabled': False},
            'persistence': {
                'enabled': True,
                'type': 'csv',
                'interval_seconds': 0.1, # Short interval for testing
                'output_dir': temp_output_dir
            }
        }
    })

    # 2. Mock Dependencies
    # Mock VideoCapture to return a few dummy frames
    with patch('src.vision.infrastructure.sources.cv2.VideoCapture') as mock_cap:
        mock_cap.return_value.isOpened.return_value = True
        
        # Create a dummy frame (black image)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Return True for read() 5 times, then False
        mock_cap.return_value.read.side_effect = [(True, dummy_frame)] * 5 + [(False, None)]
        
        # Mock YOLO to return dummy detections
        with patch('src.vision.infrastructure.yolo_detector.YOLO') as mock_yolo:
            # Mock detector to return a vehicle in the zone
            mock_detector = MagicMock()
            mock_detector.detect.return_value = FrameAnalysis(
                frame_id=1,
                timestamp=time.time(),
                vehicles=[
                    DetectedVehicle(id="1", type="car", confidence=0.9, bbox=(10, 10, 50, 50), timestamp=time.time())
                ],
                total_count=1,
                zones=[
                    ZoneVehicleCount(zone_id="zone1", vehicle_count=1, timestamp=time.time(), vehicles=["1"], vehicle_types={"car": 1})
                ]
            )
            
            # We need to patch the YoloDetector class to return our mock instance
            # But wait, YoloDetector is instantiated inside build_detector.
            # Let's patch the class itself.
            with patch('src.vision.application.builder.YoloDetector', return_value=mock_detector):
                
                # 3. Build Pipeline
                builder = VisionApplicationBuilder(cfg)
                pipeline = (
                    builder
                    .build_detector()
                    .build_zones() # Needed for aggregation
                    .build_persistence()
                    .build_source()
                    .build_pipeline()
                )
                
                # 4. Run Pipeline
                # Run for enough time to trigger at least one flush (interval is 0.1s)
                start_time = time.time()
                for _ in pipeline.run():
                    if time.time() - start_time > 0.5:
                        break
                
                # Force flush to ensure data is written
                if builder.aggregator:
                    builder.aggregator.flush()
                
                # 5. Verify Output
                files = os.listdir(temp_output_dir)
                csv_files = [f for f in files if f.endswith('.csv')]
                
                assert len(csv_files) > 0, "No CSV file created"
                
                # Read CSV content
                csv_path = os.path.join(temp_output_dir, csv_files[0])
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                    
                assert len(lines) > 1, "CSV file is empty or only has header"
                header = lines[0].strip().split(',')
                assert "zone_id" in header
                assert "avg_density" in header
                
                # Check data row
                data_row = lines[1].strip().split(',')
                # Find index of zone_id
                zone_idx = header.index("zone_id")
                assert data_row[zone_idx] == "zone1"
