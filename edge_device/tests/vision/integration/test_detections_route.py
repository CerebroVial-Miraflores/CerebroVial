"""Tests de `GET /detections/{id}/latest` (Fase 4 Mitad A, D-019).

Manager mockeado (sin stack vivo): cubre 404 (cámara no registrada), payload vacío
(registrada sin detecciones) y payload serializado (con detecciones), espejando el
patrón de `test_api_routes.py`.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.vision.domain.entities import DetectedVehicle, FrameAnalysis
from src.vision.domain.value_objects import VehicleId
from src.vision.presentation.api.routes.detections import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_manager():
    manager = MagicMock()
    manager.cameras = {"cam1": object()}  # dict real → soporta `in`
    return manager


@pytest.fixture(autouse=True)
def setup_manager(mock_manager):
    with patch(
        "src.vision.presentation.api.routes.detections.get_manager",
        return_value=mock_manager,
    ):
        yield


def test_404_si_camara_no_registrada(client, mock_manager):
    response = client.get("/detections/desconocida/latest")
    assert response.status_code == 404


def test_payload_vacio_si_sin_detecciones(client, mock_manager):
    mock_manager.get_latest_detections.return_value = None
    response = client.get("/detections/cam1/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["camera_id"] == "cam1"
    assert body["frame"] is None
    assert body["detections"] == []
    assert body["server_timestamp"] is not None


def test_payload_serializado_con_detecciones(client, mock_manager):
    analysis = FrameAnalysis(
        frame_id=1,
        timestamp=10.0,
        vehicles=[
            DetectedVehicle(
                id=VehicleId("v1"),
                type="car",
                confidence=0.9,
                bbox=(640, 360, 1280, 720),
                timestamp=10.0,
            )
        ],
        unique_vehicles=1,
        zones={},
        detection_ran=True,
    )
    mock_manager.get_latest_detections.return_value = (analysis, 1280, 720)

    response = client.get("/detections/cam1/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["camera_id"] == "cam1"
    assert body["frame"] == {"width": 1280, "height": 720}
    assert body["frame_timestamp"] == 10.0
    assert len(body["detections"]) == 1
    assert body["detections"][0]["bbox"] == [0.5, 0.5, 1.0, 1.0]
    assert body["detections"][0]["type"] == "car"
