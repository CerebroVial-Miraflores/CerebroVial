"""Integration tests del endpoint `GET /vision/health` (6f, CT-08.10 + CT-08.11f).

Endpoint **separado** de `/vision/state` (6e). Reporta estado discreto
OK/Degradado/Fuera de servicio como payload estructurado consumible por TTH-04.
"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.vision.presentation.api.routes.health import app


def _make_camera(
    *,
    running: bool = True,
    frame_age_seconds: float | None = 1.0,
    aggregation_errors: int = 0,
    data_dropped: int = 0,
) -> MagicMock:
    """Construye un mock de `CameraInstance` con `state.pipeline` + `state.aggregator`."""
    camera = MagicMock()
    camera.state.is_running = running

    aggregator = MagicMock()
    aggregator.aggregation_errors = aggregation_errors
    aggregator.data_dropped = data_dropped
    camera.state.aggregator = aggregator

    if frame_age_seconds is None:
        camera.state.pipeline.get_latest.return_value = None
    else:
        frame = SimpleNamespace(timestamp=time.time() - frame_age_seconds)
        camera.state.pipeline.get_latest.return_value = (frame, MagicMock())

    return camera


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_manager():
    manager = MagicMock()
    manager.cameras = {}
    return manager


@pytest.fixture(autouse=True)
def patch_manager(mock_manager):
    with patch(
        "src.vision.presentation.api.routes.health.get_manager",
        return_value=mock_manager,
    ):
        yield


def test_returns_503_when_no_cameras_registered(client, mock_manager):
    mock_manager.cameras = {}
    response = client.get("/vision/health")
    assert response.status_code == 503
    assert response.json()["status"] == "Fuera de servicio"
    assert response.json()["cameras"] == {}


def test_returns_503_when_all_cameras_not_running(client, mock_manager):
    mock_manager.cameras = {
        "cam1": _make_camera(running=False),
        "cam2": _make_camera(running=False),
    }
    response = client.get("/vision/health")
    assert response.status_code == 503
    assert response.json()["status"] == "Fuera de servicio"


def test_returns_503_when_running_but_no_recent_frames(client, mock_manager):
    """Running pero ningún frame en los últimos 5s → no procesa → Fuera de servicio."""
    mock_manager.cameras = {
        "cam1": _make_camera(running=True, frame_age_seconds=10.0),
    }
    response = client.get("/vision/health")
    assert response.status_code == 503
    assert response.json()["status"] == "Fuera de servicio"


def test_returns_503_when_running_but_no_frame_yet(client, mock_manager):
    """Running pero pipeline.get_latest() es None (cámara recién arrancada)."""
    mock_manager.cameras = {
        "cam1": _make_camera(running=True, frame_age_seconds=None),
    }
    response = client.get("/vision/health")
    assert response.status_code == 503
    assert response.json()["status"] == "Fuera de servicio"


def test_returns_200_degraded_when_aggregation_errors_present(client, mock_manager):
    mock_manager.cameras = {
        "cam1": _make_camera(running=True, frame_age_seconds=1.0, aggregation_errors=3),
    }
    response = client.get("/vision/health")
    assert response.status_code == 200
    assert response.json()["status"] == "Degradado"


def test_returns_200_degraded_when_data_dropped_present(client, mock_manager):
    mock_manager.cameras = {
        "cam1": _make_camera(running=True, frame_age_seconds=1.0, data_dropped=2),
    }
    response = client.get("/vision/health")
    assert response.status_code == 200
    assert response.json()["status"] == "Degradado"


def test_returns_200_degraded_when_partial_fleet_unhealthy(client, mock_manager):
    """Una cámara saludable + otra con frames viejos → degradado, no caído."""
    mock_manager.cameras = {
        "cam1": _make_camera(running=True, frame_age_seconds=1.0),
        "cam2": _make_camera(running=True, frame_age_seconds=10.0),
    }
    response = client.get("/vision/health")
    assert response.status_code == 200
    assert response.json()["status"] == "Degradado"


def test_returns_200_ok_when_all_healthy_and_no_errors(client, mock_manager):
    mock_manager.cameras = {
        "cam1": _make_camera(running=True, frame_age_seconds=0.5),
        "cam2": _make_camera(running=True, frame_age_seconds=2.0),
    }
    response = client.get("/vision/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


def test_payload_includes_per_camera_telemetry(client, mock_manager):
    """CT-04.1 mandata payload con cuerpo, no solo HTTP code."""
    mock_manager.cameras = {
        "cam_javier_prado_01": _make_camera(
            running=True,
            frame_age_seconds=1.0,
            aggregation_errors=5,
            data_dropped=2,
        ),
    }
    response = client.get("/vision/health")
    body = response.json()
    assert "checked_at" in body
    cam_data = body["cameras"]["cam_javier_prado_01"]
    assert cam_data["running"] is True
    assert cam_data["aggregation_errors"] == 5
    assert cam_data["data_dropped"] == 2
    assert cam_data["last_frame_age_seconds"] is not None
    assert 0.0 <= cam_data["last_frame_age_seconds"] < 5.0


def test_camera_without_aggregator_does_not_crash(client, mock_manager):
    """`persistence.enabled=False` → aggregator None; el health check no debe crashear."""
    camera = MagicMock()
    camera.state.is_running = True
    camera.state.aggregator = None
    frame = SimpleNamespace(timestamp=time.time() - 1.0)
    camera.state.pipeline.get_latest.return_value = (frame, MagicMock())
    mock_manager.cameras = {"cam1": camera}

    response = client.get("/vision/health")
    assert response.status_code == 200
    cam = response.json()["cameras"]["cam1"]
    assert cam["aggregation_errors"] == 0
    assert cam["data_dropped"] == 0
