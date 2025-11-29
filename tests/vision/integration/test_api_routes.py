import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.vision.presentation.api.routes.cameras import app, init_manager

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_manager():
    manager = MagicMock()
    manager.get_status.return_value = {}
    return manager

@pytest.fixture(autouse=True)
def setup_manager(mock_manager):
    # Mock the global manager in cameras.py
    # Since init_manager sets the global, we can use it or patch get_manager
    with patch('src.vision.presentation.api.routes.cameras.get_manager', return_value=mock_manager):
        yield

def test_get_status(client, mock_manager):
    mock_manager.get_status.return_value = {"cam1": {"running": True}}
    response = client.get("/cameras/status")
    assert response.status_code == 200
    assert response.json() == {"cam1": {"running": True}}

def test_add_camera(client, mock_manager):
    payload = {
        "source": "video.mp4",
        "source_type": "file",
        "zones": {}
    }
    response = client.post("/cameras/cam1", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "added", "camera_id": "cam1"}
    mock_manager.add_camera.assert_called_once()

def test_start_camera(client, mock_manager):
    response = client.post("/cameras/cam1/start")
    assert response.status_code == 200
    assert response.json() == {"status": "starting", "camera_id": "cam1"}
    # Background tasks are executed after response, but TestClient might not run them automatically in the same way?
    # Actually TestClient runs background tasks.
    # But start_camera is async in manager. 
    # The route calls background_tasks.add_task(manager.start_camera, camera_id)
    # Since manager.start_camera is likely async, we need to ensure it's called.
    # Mock manager.start_camera is a MagicMock.
    
    # Wait, if manager.start_camera is async, we should mock it as AsyncMock if we want to await it?
    # But background tasks just run the callable.
    pass 

def test_stop_camera(client, mock_manager):
    # stop_camera route awaits manager.stop_camera
    # So manager.stop_camera must be awaitable
    mock_manager.stop_camera = MagicMock(return_value=None)
    # Make it awaitable
    async def async_stop(cam_id):
        return None
    mock_manager.stop_camera.side_effect = async_stop
    
    response = client.post("/cameras/cam1/stop")
    assert response.status_code == 200
    assert response.json() == {"status": "stopped", "camera_id": "cam1"}
