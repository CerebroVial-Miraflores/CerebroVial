"""Integration tests del endpoint `GET /vision/state/{intersection_id}` (6e).

Cubre CT-08.11(d): shape §6.5 (`intersection_id`, `timestamp`, `directions[]`),
serialización `camera_id → intersection_id`, agrupado por cámara.

Cubre la branch CT-08.10 del state endpoint: 503 cuando el módulo no procesa
frames (cámara registrada pero sin TrafficData + no running); 200 con
`directions=[]` para warm-up (running pero ventana actual no cerró todavía);
404 cuando `intersection_id` no está registrada.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.vision.domain.entities import TrafficData
from src.vision.domain.value_objects import CameraId, ZoneId
from src.vision.infrastructure.broadcast.realtime_broadcaster import (
    RealtimeBroadcaster,
)
from src.vision.presentation.api.routes.state import app


def _make_td(
    camera_id: str = "cam_javier_prado_01",
    zone_id: str = "north",
    *,
    unique_vehicles: int = 24,
    flow: float = 2880.0,
    density: float | None = 48.0,
    window_end_minutes: int = 1,
) -> TrafficData:
    start = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 28, 12, window_end_minutes, 0, tzinfo=timezone.utc)
    duration = (end - start).total_seconds()
    return TrafficData(
        camera_id=CameraId(camera_id),
        zone_id=ZoneId(zone_id),
        window_start=start,
        window_end=end,
        window_duration_seconds=duration,
        unique_vehicles=unique_vehicles,
        vehicles_by_type={"car": unique_vehicles, "bus": 0, "truck": 0, "motorcycle": 0},
        mean_speed_kmh=32.5,
        flow_vehicles_per_hour=flow,
        mean_occupancy=0.47,
        density_vehicles_per_km=density,
    )


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def broadcaster() -> RealtimeBroadcaster:
    return RealtimeBroadcaster()


@pytest.fixture
def mock_manager():
    manager = MagicMock()
    manager.cameras = {}
    manager.get_status.return_value = {}
    return manager


@pytest.fixture(autouse=True)
def patch_singletons(mock_manager, broadcaster):
    with patch(
        "src.vision.presentation.api.routes.state.get_manager",
        return_value=mock_manager,
    ), patch(
        "src.vision.presentation.api.routes.state.get_broadcaster",
        return_value=broadcaster,
    ):
        yield


def test_returns_404_when_intersection_id_unknown(client, mock_manager):
    mock_manager.cameras = {}
    response = client.get("/vision/state/unknown_cam")
    assert response.status_code == 404
    assert "unknown_cam" in response.json()["detail"]


def test_returns_503_when_camera_not_running_and_no_data(client, mock_manager):
    """CT-08.10: módulo no procesa frames → 5xx en /vision/state."""
    mock_manager.cameras = {"cam_javier_prado_01": MagicMock()}
    mock_manager.get_status.return_value = {
        "cam_javier_prado_01": {"running": False, "source": "test", "zones": []}
    }
    response = client.get("/vision/state/cam_javier_prado_01")
    assert response.status_code == 503
    assert "no procesa" in response.json()["detail"]


def test_returns_200_warm_up_with_empty_directions(client, mock_manager):
    """Running pero sin TrafficData (primera ventana aún no cerró) → 200 con directions=[]."""
    mock_manager.cameras = {"cam_javier_prado_01": MagicMock()}
    mock_manager.get_status.return_value = {
        "cam_javier_prado_01": {"running": True, "source": "test", "zones": []}
    }
    response = client.get("/vision/state/cam_javier_prado_01")
    assert response.status_code == 200
    body = response.json()
    assert body["intersection_id"] == "cam_javier_prado_01"
    assert body["directions"] == []
    assert "timestamp" in body


@pytest.mark.asyncio
async def test_returns_200_with_shape_v6_5(client, mock_manager, broadcaster):
    """Golden path: serializa shape §6.5 con todas las direcciones agrupadas."""
    mock_manager.cameras = {"cam_javier_prado_01": MagicMock()}
    mock_manager.get_status.return_value = {
        "cam_javier_prado_01": {"running": True, "source": "test", "zones": []}
    }
    await broadcaster.publish(_make_td(zone_id="north", unique_vehicles=10, flow=1200.0))
    await broadcaster.publish(
        _make_td(zone_id="south", unique_vehicles=5, flow=600.0, density=None)
    )

    response = client.get("/vision/state/cam_javier_prado_01")
    assert response.status_code == 200
    body = response.json()

    assert body["intersection_id"] == "cam_javier_prado_01"
    assert "timestamp" in body
    directions = body["directions"]
    assert len(directions) == 2

    by_dir = {d["direction"]: d for d in directions}
    assert by_dir["north"] == {
        "direction": "north",
        "count": 10,
        "queue": None,
        "flow": 1200.0,
        "density": 48.0,
    }
    assert by_dir["south"] == {
        "direction": "south",
        "count": 5,
        "queue": None,
        "flow": 600.0,
        "density": None,
    }


@pytest.mark.asyncio
async def test_intersection_id_is_camera_id_serialization(client, mock_manager, broadcaster):
    """§6.5: `camera_id` del dominio se serializa como `intersection_id`."""
    mock_manager.cameras = {"larco_schell": MagicMock()}
    mock_manager.get_status.return_value = {
        "larco_schell": {"running": True, "source": "test", "zones": []}
    }
    await broadcaster.publish(_make_td(camera_id="larco_schell", zone_id="east"))

    response = client.get("/vision/state/larco_schell")
    assert response.status_code == 200
    body = response.json()
    assert body["intersection_id"] == "larco_schell"
    assert "camera_id" not in body  # solo intersection_id; camera_id es nomenclatura interna


@pytest.mark.asyncio
async def test_timestamp_is_latest_window_end(client, mock_manager, broadcaster):
    """`timestamp` top-level = max(`window_end`) de las direcciones (§6.5)."""
    mock_manager.cameras = {"cam_javier_prado_01": MagicMock()}
    mock_manager.get_status.return_value = {
        "cam_javier_prado_01": {"running": True, "source": "test", "zones": []}
    }
    await broadcaster.publish(_make_td(zone_id="north", window_end_minutes=1))
    await broadcaster.publish(_make_td(zone_id="south", window_end_minutes=2))

    response = client.get("/vision/state/cam_javier_prado_01")
    assert response.status_code == 200
    body = response.json()
    assert body["timestamp"].startswith("2026-05-28T12:02:00")
