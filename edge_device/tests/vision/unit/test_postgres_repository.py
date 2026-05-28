"""Tests de PostgresTrafficRepository sin BD viva (TTH-08 Fase 4c, CT-08.11).

Dos técnicas separadas:
- Mapeo (CT-08.5): test directo de `_to_row`, lógica pura sobre el contrato §6.5.
- Idempotencia (CT-08.11): aserción sobre la sentencia que construye `save()`,
  compilada contra el dialecto Postgres (no se ejecuta — `on_conflict_do_nothing`
  es dialecto Postgres y no corre en SQLite).
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from sqlalchemy.dialects import postgresql

from src.vision.domain.entities import TrafficData
from src.vision.domain.value_objects import CameraId, ZoneId
from src.vision.infrastructure.persistence.postgres_repository import (
    PostgresTrafficRepository,
)

_WINDOW_START = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
_WINDOW_END = _WINDOW_START + timedelta(seconds=60)


def _make_traffic_data(
    *,
    camera_id="CAM_01",
    zone_id="zone1",
    vehicles_by_type=None,
    unique_vehicles=4,
    mean_speed_kmh=42.5,
    flow_vehicles_per_hour=240.0,
    mean_occupancy=0.35,
    density_vehicles_per_km=120.0,
) -> TrafficData:
    if vehicles_by_type is None:
        # car + bus only: truck/motorcycle ausentes a propósito (caso .get -> 0).
        vehicles_by_type = {"car": 3, "bus": 1}
    return TrafficData(
        camera_id=CameraId(camera_id),
        zone_id=ZoneId(zone_id),
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        window_duration_seconds=60.0,
        unique_vehicles=unique_vehicles,
        vehicles_by_type=vehicles_by_type,
        mean_speed_kmh=mean_speed_kmh,
        flow_vehicles_per_hour=flow_vehicles_per_hour,
        mean_occupancy=mean_occupancy,
        density_vehicles_per_km=density_vehicles_per_km,
    )


def test_to_row_maps_all_columns_ct_08_5():
    """CT-08.5: _to_row produce las 14 columnas §6.5 con los valores correctos,
    desempaquetando vehicles_by_type (tipos ausentes -> 0) y sin incluir queue."""
    data = _make_traffic_data()
    repo = PostgresTrafficRepository(session_factory=lambda: None)

    row = repo._to_row(data)

    assert row == {
        "camera_id": "CAM_01",
        "zone_id": "zone1",
        "window_start": _WINDOW_START,
        "window_end": _WINDOW_END,
        "window_duration_seconds": 60.0,
        "unique_vehicles": 4,
        "car_count": 3,
        "bus_count": 1,
        "truck_count": 0,
        "motorcycle_count": 0,
        "mean_speed_kmh": 42.5,
        "flow_vehicles_per_hour": 240.0,
        "mean_occupancy": 0.35,
        "density_vehicles_per_km": 120.0,
    }
    # queue queda fuera del dict -> NULL en la columna nullable (MVP1; F41).
    assert "queue" not in row


def test_to_row_maps_optional_nones_ct_08_5():
    """CT-08.5: los Optional del dominio en None se mapean a None (NULL)."""
    data = _make_traffic_data(mean_speed_kmh=None, density_vehicles_per_km=None)
    repo = PostgresTrafficRepository(session_factory=lambda: None)

    row = repo._to_row(data)

    assert row["mean_speed_kmh"] is None
    assert row["density_vehicles_per_km"] is None


def test_save_builds_on_conflict_do_nothing_and_commits_ct_08_11():
    """CT-08.11: save() construye un INSERT ... ON CONFLICT DO NOTHING sobre la
    PK compuesta (idempotencia) y commitea/cierra su propia sesión."""
    captured = {}
    session = MagicMock()
    session.execute.side_effect = lambda stmt: captured.__setitem__("stmt", stmt)

    repo = PostgresTrafficRepository(session_factory=lambda: session)
    repo.save(_make_traffic_data())

    compiled = str(
        captured["stmt"].compile(dialect=postgresql.dialect())
    ).upper()
    assert "INSERT INTO VISION_AGGREGATES" in compiled
    assert "ON CONFLICT" in compiled
    assert "DO NOTHING" in compiled

    session.execute.assert_called_once()
    session.commit.assert_called_once()
    session.close.assert_called_once()
