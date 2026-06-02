"""Tests del endpoint de feed de estado por arista (TTH-12 Fase 3, CT-12.6/12.7 / CT-12.9 f)."""
from datetime import datetime, timedelta

from src.congestion.infrastructure.repositories import WazeJamRow, WazeJamsRepo

_T0 = datetime(2025, 1, 6, 0, 0, 0)


def _row(edge_id: str, ts: datetime, level: int) -> WazeJamRow:
    return WazeJamRow(
        event_uuid=f"{edge_id}@{ts.isoformat()}", snapshot_timestamp=ts, edge_id=edge_id,
        speed_mps=10.0, delay_seconds=0, congestion_level=level, jam_length_m=-1, road_type=0,
    )


def _seed(session, n_edges: int = 3) -> None:
    repo = WazeJamsRepo(session)
    rows = []
    for e in range(n_edges):
        eid = f"-100{e}#0"
        rows.append(_row(eid, _T0, 0))
        rows.append(_row(eid, _T0 + timedelta(seconds=60), (e % 5) + 1))  # más nuevo
    repo.batch_insert(rows)
    session.commit()


def test_feed_requires_role(congestion_client):
    congestion_client.as_role("manager")
    assert congestion_client.get("/congestion/state").status_code == 403


def test_feed_returns_latest_state_per_edge_with_timestamp(congestion_client):
    _seed(congestion_client.session, n_edges=3)
    congestion_client.as_role("operator")
    resp = congestion_client.get("/congestion/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3 == len(body["edges"])
    for e in body["edges"]:
        assert set(e) == {"edge_id", "congestion_level", "snapshot_timestamp"}
        assert 0 <= e["congestion_level"] <= 5
        # CT-12.7: expone la marca de tiempo del último snapshot conocido
        assert e["snapshot_timestamp"] == (_T0 + timedelta(seconds=60)).isoformat()


def test_feed_robust_when_no_source_running(congestion_client):
    """CT-12.7: aunque el adaptador no corra, el endpoint devuelve lo pre-sembrado."""
    _seed(congestion_client.session, n_edges=2)
    congestion_client.as_role("operator")
    resp = congestion_client.get("/congestion/state")  # sin adaptador activo
    assert resp.status_code == 200 and resp.json()["count"] == 2


def test_feed_empty_is_not_an_error(congestion_client):
    congestion_client.as_role("operator")
    resp = congestion_client.get("/congestion/state")  # waze_jams vacío
    assert resp.status_code == 200 and resp.json()["count"] == 0
