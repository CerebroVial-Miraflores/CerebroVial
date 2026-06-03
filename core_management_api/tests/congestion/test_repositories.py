"""Tests del repositorio waze_jams (TTH-12 Fase 2, CT-12.5 / CT-12.9 e parcial;
TTH-13 Fase 2, CT-13.1)."""
from datetime import datetime, timedelta

from src.congestion.infrastructure.repositories import WazeJamRow, WazeJamsRepo

_T0 = datetime(2025, 1, 6, 0, 0, 0)
_DAY = _T0.date()


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


# --- series_for_day (TTH-13 Fase 2, CT-13.1) ---


def test_series_for_day_groups_levels_per_edge_in_order(congestion_session):
    # Grilla conocida: 3 aristas × 4 instantes a 60s. Niveles deterministas por arista.
    repo = WazeJamsRepo(congestion_session)
    rows = []
    expected = {}
    for e in range(3):
        eid = f"e{e}"
        levels = [(e + i) % 6 for i in range(4)]
        expected[eid] = levels
        for i, lvl in enumerate(levels):
            rows.append(_row(eid, _T0 + timedelta(seconds=60 * i), lvl))
    repo.batch_insert(rows)

    series = repo.series_for_day(_DAY)

    assert series.day == _DAY
    assert series.t0 == _T0
    assert series.step_s == 60
    assert series.coverage_end == _T0 + timedelta(seconds=60 * 3)
    # Orden por edge_id y niveles correctos por arista.
    assert [e.edge_id for e in series.edges] == ["e0", "e1", "e2"]
    assert {e.edge_id: e.levels for e in series.edges} == expected


def test_series_for_day_empty_day_returns_empty_series(congestion_session):
    repo = WazeJamsRepo(congestion_session)
    repo.batch_insert([_row("e1", _T0, 3)])  # datos de OTRO día

    series = repo.series_for_day(_DAY + timedelta(days=1))

    assert series.edges == []
    assert series.t0 is None
    assert series.step_s is None
    assert series.coverage_end is None


def test_series_for_day_day_boundaries_inclusive_start_exclusive_next(congestion_session):
    # 00:00 del día incluido; 00:00 del día siguiente excluido (>= start, < end).
    repo = WazeJamsRepo(congestion_session)
    next_midnight = _T0 + timedelta(days=1)
    repo.batch_insert([
        _row("e1", _T0, 1),                                    # 00:00 → incluido
        _row("e1", _T0 + timedelta(hours=23, minutes=59), 2),  # mismo día → incluido
        _row("e1", next_midnight, 5),                          # 00:00 día sig. → excluido
    ])

    series = repo.series_for_day(_DAY)

    assert series.t0 == _T0
    assert series.coverage_end == _T0 + timedelta(hours=23, minutes=59)
    assert len(series.edges) == 1
    assert series.edges[0].levels == [1, 2]  # el de medianoche siguiente NO entra


def test_series_for_day_seeded_grid_is_contiguous(congestion_session):
    # Respalda el armado por append secuencial: la grilla sembrada NO tiene huecos,
    # así len(levels) == nº de instantes distintos para cada arista.
    repo = WazeJamsRepo(congestion_session)
    n = 10
    rows = []
    for e in range(2):
        for i in range(n):
            rows.append(_row(f"e{e}", _T0 + timedelta(seconds=60 * i), i % 6))
    repo.batch_insert(rows)

    series = repo.series_for_day(_DAY)

    n_instants = round((series.coverage_end - series.t0).total_seconds() / series.step_s) + 1
    assert n_instants == n
    for edge in series.edges:
        assert len(edge.levels) == n_instants
