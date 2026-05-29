import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from omegaconf import DictConfig
from src.vision.application.services.multi_camera import MultiCameraManager, CameraInstance

@pytest.fixture
def mock_broadcaster():
    """Mock del broadcaster Protocol-conforme (§6.10/§6.11): publish/subscriber_count/is_subscribed."""
    broadcaster = MagicMock()
    broadcaster.publish = AsyncMock()
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
    """C1.6 migrado: CameraInstance expone su id vía state.camera_id (no atributo directo)."""
    config = DictConfig({'vision': {'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        MockBuilder.return_value.build_pipeline.return_value = MagicMock()

        camera = manager.add_camera("cam1", config)

        assert "cam1" in manager.cameras
        assert isinstance(camera, CameraInstance)
        assert camera.state.camera_id == "cam1"
        # Caso B roto: renderer no se instancia adentro; default None.
        assert camera.state.renderer is None

def test_add_duplicate_camera(manager):
    config = DictConfig({'vision': {'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder'):
        manager.add_camera("cam1", config)
        with pytest.raises(ValueError):
            manager.add_camera("cam1", config)

@pytest.mark.asyncio
async def test_start_stop_camera(manager):
    """C1.6 migrado: is_running vive en state.is_running (no atributo directo).

    Sin OpenCVVisualizer hardcoded (Caso B roto): el pipeline_mock yield
    MagicMocks sin tocar cv2.putText, porque el render condicional solo se
    ejecuta si `state.renderer is not None`.
    """
    config = DictConfig({'vision': {'zones': {}}})

    pipeline_mock = MagicMock()
    pipeline_mock.run.return_value = iter([(MagicMock(), MagicMock())])

    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        MockBuilder.return_value.build_pipeline.return_value = pipeline_mock

        manager.add_camera("cam1", config)

        # Start.
        await manager.start_camera("cam1")
        assert manager.cameras["cam1"].state.is_running
        assert "cam1" in manager._tasks

        # Stop.
        await manager.stop_camera("cam1")
        assert not manager.cameras["cam1"].state.is_running
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
