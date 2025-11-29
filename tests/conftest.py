import pytest
import numpy as np
from src.vision.domain.entities import Frame, FrameAnalysis, DetectedVehicle

@pytest.fixture
def mock_frame():
    return Frame(
        id=0,
        timestamp=1234567890.0,
        image=np.zeros((100, 100, 3), dtype=np.uint8)
    )

@pytest.fixture
def mock_analysis():
    return FrameAnalysis(
        frame_id=0,
        timestamp=1234567890.0,
        vehicles=[
            DetectedVehicle(
                id="1",
                type="car",
                confidence=0.9,
                bbox=(10, 10, 50, 50),
                timestamp=1234567890.0
            )
        ],
        total_count=1
    )
