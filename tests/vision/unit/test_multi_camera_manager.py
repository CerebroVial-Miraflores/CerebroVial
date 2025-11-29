import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from omegaconf import DictConfig
from src.vision.application.services.multi_camera import MultiCameraManager, CameraInstance

@pytest.fixture
def mock_broadcaster():
    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    broadcaster.serialize_analysis = MagicMock(return_value="data")
    return broadcaster

@pytest.fixture
def mock_builder():
    builder = MagicMock()
    pipeline = MagicMock()
    pipeline.run.return_value = [] # Empty generator by default
    builder.build_pipeline.return_value = pipeline
    return builder

@pytest.fixture
def manager(mock_broadcaster):
    return MultiCameraManager(mock_broadcaster)

def test_add_camera(manager):
    config = DictConfig({'vision': {'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        MockBuilder.return_value.build_pipeline.return_value = MagicMock()
        
        camera = manager.add_camera("cam1", config)
        
        assert "cam1" in manager.cameras
        assert isinstance(camera, CameraInstance)
        assert camera.camera_id == "cam1"

def test_add_duplicate_camera(manager):
    config = DictConfig({'vision': {'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder'):
        manager.add_camera("cam1", config)
        with pytest.raises(ValueError):
            manager.add_camera("cam1", config)

@pytest.mark.asyncio
async def test_start_stop_camera(manager):
    config = DictConfig({'vision': {'zones': {}}})
    
    # Mock pipeline run to yield one item then sleep
    async def mock_run_gen():
        yield (MagicMock(), MagicMock())
        await asyncio.sleep(0.1)
        
    pipeline_mock = MagicMock()
    # We need to mock the run method to be an iterator
    pipeline_mock.run.return_value = iter([(MagicMock(), MagicMock())])
    
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        MockBuilder.return_value.build_pipeline.return_value = pipeline_mock
        
        manager.add_camera("cam1", config)
        
        # Start
        await manager.start_camera("cam1")
        assert manager.cameras["cam1"].is_running
        assert "cam1" in manager._tasks
        
        # Stop
        await manager.stop_camera("cam1")
        assert not manager.cameras["cam1"].is_running
        assert "cam1" not in manager._tasks
        pipeline_mock.stop.assert_called_once()

@pytest.mark.asyncio
async def test_start_camera_not_found(manager):
    with pytest.raises(ValueError):
        await manager.start_camera("non_existent")

def test_get_status(manager):
    config = DictConfig({'vision': {'source': 'test', 'zones': {'z1': {}}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder'):
        manager.add_camera("cam1", config)
        
        status = manager.get_status()
        assert "cam1" in status
        assert status["cam1"]["source"] == "test"
        assert "z1" in status["cam1"]["zones"]
