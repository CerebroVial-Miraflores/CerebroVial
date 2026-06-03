"""Tests del endpoint de la serie del día por arista (TTH-13 Fase 3, CT-13.2/13.4/13.5).

El ``congestion_client`` arma un FastAPI bare con sólo el router (sin el
GZipMiddleware de ``main.py``), por eso acá NO se ejercita gzip: la compresión se
verifica en smoke manual. La forma de la respuesta (Formato B) y la semántica sí.
"""
from datetime import datetime, timedelta

from src.congestion.infrastructure.repositories import WazeJamRow, WazeJamsRepo

_T0 = datetime(2025, 1, 6, 0, 0, 0)
_DAY = _T0.date().isoformat()  # "2025-01-06"


def _row(edge_id: str, ts: datetime, level: int) -> WazeJamRow:
    return WazeJamRow(
        event_uuid=f"{edge_id}@{ts.isoformat()}", snapshot_timestamp=ts, edge_id=edge_id,
        speed_mps=10.0, delay_seconds=0, congestion_level=level, jam_length_m=-1, road_type=0,
    )


def _seed_grid(session, n_edges: int = 3, n_instants: int = 4) -> None:
    """Grilla a 60s: n_edges aristas × n_instants instantes del día _T0."""
    repo = WazeJamsRepo(session)
    rows = [
        _row(f"-100{e}#0", _T0 + timedelta(seconds=60 * i), (e + i) % 6)
        for e in range(n_edges)
        for i in range(n_instants)
    ]
    repo.batch_insert(rows)
    session.commit()


def test_series_requires_role(congestion_client):
    congestion_client.as_role("manager")
    assert congestion_client.get(f"/congestion/series?day={_DAY}").status_code == 403


def test_series_returns_format_b_with_coverage_end(congestion_client):
    _seed_grid(congestion_client.session, n_edges=3, n_instants=4)
    congestion_client.as_role("operator")
    resp = congestion_client.get(f"/congestion/series?day={_DAY}")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"day", "t0", "step_s", "coverage_end", "count", "edges"}
    assert body["day"] == _DAY
    assert body["t0"] == _T0.isoformat()
    assert body["step_s"] == 60
    assert body["coverage_end"] == (_T0 + timedelta(seconds=60 * 3)).isoformat()
    assert body["count"] == 3 == len(body["edges"])
    for e in body["edges"]:
        assert set(e) == {"edge_id", "levels"}
        assert len(e["levels"]) == 4
        assert all(0 <= lvl <= 5 for lvl in e["levels"])


def test_series_empty_day_is_not_an_error(congestion_client):
    """CA-23.7 / CT-13.5: día sin datos → 200 con edges vacío y campos temporales null."""
    congestion_client.as_role("operator")
    resp = congestion_client.get(f"/congestion/series?day={_DAY}")  # waze_jams vacío
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["edges"] == []
    assert body["t0"] is None
    assert body["step_s"] is None
    assert body["coverage_end"] is None


def test_series_day_param_is_required(congestion_client):
    congestion_client.as_role("operator")
    assert congestion_client.get("/congestion/series").status_code == 422


def test_state_endpoint_unchanged_by_series_addition(congestion_client):
    """CT-13.4: la adición de /series no altera /congestion/state (HU-22).

    /state sigue devolviendo el último snapshot por arista, sin los campos de la
    serie. La serie y el estado conviven sobre la misma tabla sin interferirse.
    """
    _seed_grid(congestion_client.session, n_edges=2, n_instants=3)
    congestion_client.as_role("operator")
    resp = congestion_client.get("/congestion/state")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"edges", "count"}
    assert body["count"] == 2 == len(body["edges"])
    for e in body["edges"]:
        assert set(e) == {"edge_id", "congestion_level", "snapshot_timestamp"}
        # el más reciente del día sembrado: instante i=2 (00:02)
        assert e["snapshot_timestamp"] == (_T0 + timedelta(seconds=60 * 2)).isoformat()
