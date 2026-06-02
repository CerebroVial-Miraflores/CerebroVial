"""Tests del repositorio waze_jams (TTH-12 Fase 2, CT-12.5 / CT-12.9 e parcial)."""
from datetime import datetime, timedelta

from src.congestion.infrastructure.repositories import WazeJamRow, WazeJamsRepo

_T0 = datetime(2025, 1, 6, 0, 0, 0)


def _row(edge_id: str, ts: datetime, level: int) -> WazeJamRow:
    return WazeJamRow(
        event_uuid=f"{edge_id}@{ts.isoformat()}",
        snapshot_timestamp=ts,
        edge_id=edge_id,
        speed_mps=10.0,
        delay_seconds=0,
        congestion_level=level,
        jam_length_m=-1,
        road_type=0,
    )


def test_insert_one_and_count(congestion_session):
    repo = WazeJamsRepo(congestion_session)
    assert repo.count() == 0
    repo.insert_one(_row("e1", _T0, 3))
    assert repo.count() == 1


def test_batch_insert_is_idempotent(congestion_session):
    repo = WazeJamsRepo(congestion_session)
    rows = [_row("e1", _T0 + timedelta(seconds=60 * i), i % 6) for i in range(50)]
    repo.batch_insert(rows)
    assert repo.count() == 50
    # re-correr con event_uuid determinista NO duplica (ON CONFLICT DO NOTHING)
    repo.batch_insert(rows)
    assert repo.count() == 50


def test_latest_per_edge_returns_newest_snapshot(congestion_session):
    repo = WazeJamsRepo(congestion_session)
    repo.batch_insert([
        _row("e1", _T0, 1),
        _row("e1", _T0 + timedelta(seconds=60), 4),   # más nuevo para e1
        _row("e2", _T0, 2),
    ])
    latest = {ec.edge_id: ec for ec in repo.latest_per_edge()}
    assert set(latest) == {"e1", "e2"}
    assert latest["e1"].congestion_level == 4
    assert latest["e1"].snapshot_timestamp == _T0 + timedelta(seconds=60)
    assert latest["e2"].congestion_level == 2


def test_populate_geom_is_noop_on_sqlite(congestion_session):
    # En SQLite (sin geom ni UPDATE-FROM) el método es no-op seguro.
    repo = WazeJamsRepo(congestion_session)
    repo.insert_one(_row("e1", _T0, 0))
    assert repo.populate_geom_from_edges() == 0
