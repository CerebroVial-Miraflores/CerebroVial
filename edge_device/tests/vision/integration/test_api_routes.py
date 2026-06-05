import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from src.vision.presentation.api.routes.cameras import app, _build_camera_config

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
    # C1/D3: el alta on-demand registra Y arranca vía activate_camera (single-slot).
    mock_manager.activate_camera = AsyncMock()
    payload = {
        "source": "video.mp4",
        "source_type": "file",
        "zones": {}
    }
    response = client.post("/cameras/cam1", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "started", "camera_id": "cam1"}
    mock_manager.activate_camera.assert_called_once()


def test_remove_camera(client, mock_manager):
    # C1/D3: la baja on-demand para + libera el modelo vía remove_camera.
    mock_manager.remove_camera = AsyncMock()
    response = client.delete("/cameras/cam1")
    assert response.status_code == 200
    assert response.json() == {"status": "removed", "camera_id": "cam1"}
    mock_manager.remove_camera.assert_called_once()


def test_build_camera_config_uses_postgres_persistence():
    """Regresión C1: el cfg del alta on-demand persiste en postgres (no csv, que
    build_persistence rechaza → 500), con ventana corta y el camera_id real."""
    cfg = _build_camera_config(
        "cam_larco_benavides",
        "https://live.smartechlatam.online/claro/escuela_pnp/index.m3u8",
        "hls",
        {},
    )
    persistence = cfg.vision.persistence
    assert persistence.type == "postgres"   # NO csv (decisión #8)
    assert persistence.enabled is True
    assert persistence.interval_seconds == 5  # ventana corta → conteo pronto
    # camera_id obligatorio para build_persistence con persistence.enabled.
    assert cfg.vision.camera_id == "cam_larco_benavides"
    # El source/source_type llegan tal cual (ruteo hls).
    assert cfg.vision.source_type == "hls"


def test_build_camera_config_detection_tuning():
    """Regresión C1: umbral on-demand bajo (detecta más) y detección en cada
    frame (sin parpadeo de boxes)."""
    cfg = _build_camera_config(
        "cam_larco_benavides",
        "https://live.smartechlatam.online/claro/escuela_pnp/index.m3u8",
        "hls",
        {},
    )
    # OVERRIDE-LIVE: 0.2 para capturar más con yolo11n en ángulo alto (no 0.5/0.3).
    assert cfg.vision.model.conf_threshold == 0.2
    # OVERRIDE-LIVE: 2 → YOLO sigue el ritmo del stream en CPU (n=1 saturaba); el
    # parpadeo se resuelve persistiendo boxes en skip frames (visual-only).
    assert cfg.vision.performance.detect_every_n_frames == 2

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
