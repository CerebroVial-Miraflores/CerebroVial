"""Tests del `RealtimeBroadcaster` contra el Protocol §6.10/§6.11 + payload §6.2.

Reescritura 6d: cero acceso a `_subscribers`/`_latest_state`. Toda la verificación
pasa por la API pública (`publish`, `subscriber_count`, `is_subscribed`,
`subscribe`, `unsubscribe`, `subscribed_cameras`, `latest_state`).
"""
import asyncio
from datetime import datetime, timezone

import pytest

from src.vision.domain.entities import TrafficData
from src.vision.domain.protocols import Broadcaster
from src.vision.domain.value_objects import CameraId, ZoneId
from src.vision.infrastructure.broadcast.realtime_broadcaster import (
    RealtimeBroadcaster,
)


def _make_traffic_data(
    camera_id: str = "cam1",
    zone_id: str = "north",
    *,
    mean_speed_kmh: float | None = 32.5,
    density_vehicles_per_km: float | None = 48.0,
) -> TrafficData:
    start = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 28, 12, 0, 30, tzinfo=timezone.utc)
    return TrafficData(
        camera_id=CameraId(camera_id),
        zone_id=ZoneId(zone_id),
        window_start=start,
        window_end=end,
        window_duration_seconds=30.0,
        unique_vehicles=24,
        vehicles_by_type={"car": 18, "bus": 2, "truck": 1, "motorcycle": 3},
        mean_speed_kmh=mean_speed_kmh,
        flow_vehicles_per_hour=2880.0,
        mean_occupancy=0.47,
        density_vehicles_per_km=density_vehicles_per_km,
    )


@pytest.fixture
def broadcaster() -> RealtimeBroadcaster:
    return RealtimeBroadcaster()


def test_satisfies_broadcaster_protocol(broadcaster):
    """Structural compliance con `domain.protocols.Broadcaster` (§6.10/§6.11)."""
    b: Broadcaster = broadcaster
    assert callable(b.publish)
    assert callable(b.subscriber_count)
    assert callable(b.is_subscribed)


@pytest.mark.asyncio
async def test_publish_emits_payload_v6_2_shape(broadcaster):
    subscriber_id, queue = await broadcaster.subscribe("cam1")
    await broadcaster.publish(_make_traffic_data(camera_id="cam1"))

    payload = await queue.get()

    assert payload["schema_version"] == "1.0"
    assert payload["event_type"] == "traffic_update"
    assert "server_timestamp" in payload
    assert payload["camera"] == {"id": "cam1", "street_monitored": None}
    assert payload["zone"] == {"id": "north"}
    assert payload["window"]["duration_seconds"] == 30.0
    assert payload["window"]["start"].startswith("2026-05-28T12:00:00")
    assert payload["window"]["end"].startswith("2026-05-28T12:00:30")
    assert payload["metrics"] == {
        "unique_vehicles": 24,
        "vehicles_by_type": {"car": 18, "bus": 2, "truck": 1, "motorcycle": 3},
        "mean_speed_kmh": 32.5,
        "flow_vehicles_per_hour": 2880.0,
        "mean_occupancy": 0.47,
        "density_vehicles_per_km": 48.0,
    }


@pytest.mark.asyncio
async def test_publish_emits_null_for_optional_metrics(broadcaster):
    """`mean_speed_kmh` y `density_vehicles_per_km` como `null` cuando son None (§6.2)."""
    _, queue = await broadcaster.subscribe("cam1")
    await broadcaster.publish(
        _make_traffic_data(mean_speed_kmh=None, density_vehicles_per_km=None)
    )
    payload = await queue.get()
    assert payload["metrics"]["mean_speed_kmh"] is None
    assert payload["metrics"]["density_vehicles_per_km"] is None


@pytest.mark.asyncio
async def test_payload_does_not_contain_localized_or_thresholded_fields(broadcaster):
    """§9.2 — no strings localizadas, no `%`, no umbrales en transporte."""
    _, queue = await broadcaster.subscribe("cam1")
    await broadcaster.publish(_make_traffic_data())
    payload = await queue.get()

    metrics = payload["metrics"]
    # Sin strings ES.
    assert "congestion_level" not in payload
    assert "congestion_level" not in metrics
    # Sin sufijos / formateos.
    for value in metrics.values():
        if isinstance(value, str):
            assert "%" not in value
    # Sin `density` string.
    assert "density" not in metrics  # solo `density_vehicles_per_km` numérico
    # Sin placeholders viejos.
    assert "incidents" not in payload
    assert "pedestrians" not in payload


@pytest.mark.asyncio
async def test_subscriber_count_reflects_lifecycle(broadcaster):
    assert broadcaster.subscriber_count() == 0
    sid_a, _ = await broadcaster.subscribe("cam1")
    sid_b, _ = await broadcaster.subscribe("cam2")
    assert broadcaster.subscriber_count() == 2
    await broadcaster.unsubscribe(sid_a)
    assert broadcaster.subscriber_count() == 1
    await broadcaster.unsubscribe(sid_b)
    assert broadcaster.subscriber_count() == 0


@pytest.mark.asyncio
async def test_is_subscribed_reflects_lifecycle(broadcaster):
    sid, _ = await broadcaster.subscribe("cam1")
    assert broadcaster.is_subscribed(sid) is True
    await broadcaster.unsubscribe(sid)
    assert broadcaster.is_subscribed(sid) is False


@pytest.mark.asyncio
async def test_publish_only_to_matching_camera_subscribers(broadcaster):
    _, q_cam1_a = await broadcaster.subscribe("cam1")
    _, q_cam1_b = await broadcaster.subscribe("cam1")
    _, q_cam2 = await broadcaster.subscribe("cam2")

    await broadcaster.publish(_make_traffic_data(camera_id="cam1"))

    assert (await q_cam1_a.get())["camera"]["id"] == "cam1"
    assert (await q_cam1_b.get())["camera"]["id"] == "cam1"
    assert q_cam2.empty()


@pytest.mark.asyncio
async def test_new_subscriber_receives_latest_state(broadcaster):
    await broadcaster.publish(_make_traffic_data(camera_id="cam1"))
    # Subscriber nuevo entra después del primer publish.
    _, queue = await broadcaster.subscribe("cam1")
    payload = await queue.get()
    assert payload["camera"]["id"] == "cam1"
    assert payload["metrics"]["unique_vehicles"] == 24


@pytest.mark.asyncio
async def test_slow_subscriber_does_not_break_publish(broadcaster):
    _, slow_q = await broadcaster.subscribe("cam1", queue_size=1)
    # Llenar la queue del slow subscriber.
    await broadcaster.publish(_make_traffic_data())
    # Segundo publish encuentra la queue llena; debe dropear sin romper.
    await broadcaster.publish(_make_traffic_data())
    # El primer payload está; el segundo se dropeó.
    first = await slow_q.get()
    assert first["camera"]["id"] == "cam1"
    assert slow_q.empty()


@pytest.mark.asyncio
async def test_subscribed_cameras_helper_returns_unique_sorted(broadcaster):
    await broadcaster.subscribe("cam_b")
    await broadcaster.subscribe("cam_a")
    await broadcaster.subscribe("cam_a")
    assert broadcaster.subscribed_cameras() == ["cam_a", "cam_b"]


@pytest.mark.asyncio
async def test_latest_state_helper_returns_payload_after_publish(broadcaster):
    assert broadcaster.latest_state("cam1") is None
    await broadcaster.publish(_make_traffic_data(camera_id="cam1"))
    cached = broadcaster.latest_state("cam1")
    assert cached is not None
    assert cached["camera"]["id"] == "cam1"
    assert broadcaster.latest_state("cam_unknown") is None


@pytest.mark.asyncio
async def test_publish_under_concurrency_does_not_corrupt_subscribers(broadcaster):
    """Concurrencia: múltiples publish + subscribe en paralelo no rompen la estructura."""
    await broadcaster.subscribe("cam1")
    await broadcaster.subscribe("cam1")

    await asyncio.gather(
        broadcaster.publish(_make_traffic_data()),
        broadcaster.publish(_make_traffic_data()),
        broadcaster.publish(_make_traffic_data()),
    )

    assert broadcaster.subscriber_count() == 2
    assert broadcaster.subscribed_cameras() == ["cam1"]
