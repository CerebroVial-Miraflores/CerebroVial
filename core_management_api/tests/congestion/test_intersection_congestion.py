"""Tests de la congestión real por intersección (B2).

Cubre las dos costuras que componen ``GET /api/intersections``:
- ``congestion_status_from_level`` (mapeo puro 0-5 → status, B2 D4).
- ``WazeJamsRepo.max_level_by_intersection`` (agregación MAX por intersección sobre el
  último snapshot de sus aristas, reusando ``latest_per_edge``).

Fixture con esquema STRIPPED (sin geom) de las 4 tablas que cruza la agregación
(``graph_edges`` + ``intersections`` + ``intersection_edges`` + ``waze_jams``), mismo
patrón portable SQLite que ``tests/congestion/conftest.py`` y
``tests/intersections/conftest.py``. NO importa ``main.py`` (arranca el wiring pesado de
modelos/predictores): testea las unidades, no el endpoint montado.
"""
from datetime import datetime
from typing import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.congestion.application.intersection_congestion import (
    congestion_status_from_level,
)
from src.congestion.infrastructure.repositories import WazeJamRow, WazeJamsRepo

_GRAPH_EDGES_DDL = """
CREATE TABLE graph_edges (
    edge_id TEXT PRIMARY KEY,
    source_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    distance_m REAL NOT NULL,
    lanes INTEGER NOT NULL
)
"""
_INTERSECTIONS_DDL = """
CREATE TABLE intersections (
    intersection_id TEXT PRIMARY KEY,
    junction_id TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    los_pmu TEXT,
    tls_id TEXT
)
"""
_INTERSECTION_EDGES_DDL = """
CREATE TABLE intersection_edges (
    intersection_id TEXT NOT NULL REFERENCES intersections(intersection_id),
    edge_id TEXT NOT NULL REFERENCES graph_edges(edge_id),
    direction TEXT NOT NULL CHECK (direction IN ('incoming','outgoing')),
    PRIMARY KEY (intersection_id, edge_id)
)
"""
_WAZE_JAMS_DDL = """
CREATE TABLE waze_jams (
    event_uuid TEXT NOT NULL,
    snapshot_timestamp TIMESTAMP NOT NULL,
    edge_id TEXT,
    speed_mps REAL NOT NULL,
    delay_seconds INTEGER NOT NULL,
    congestion_level INTEGER NOT NULL,
    jam_length_m INTEGER NOT NULL,
    road_type INTEGER NOT NULL,
    PRIMARY KEY (event_uuid, snapshot_timestamp)
)
"""


@pytest.fixture
def bridge_session() -> Iterator[Session]:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with eng.begin() as conn:
        conn.execute(text(_GRAPH_EDGES_DDL))
        conn.execute(text(_INTERSECTIONS_DDL))
        conn.execute(text(_INTERSECTION_EDGES_DDL))
        conn.execute(text(_WAZE_JAMS_DDL))
    SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        eng.dispose()


def _add_edge(session: Session, edge_id: str) -> None:
    session.execute(
        text(
            "INSERT INTO graph_edges (edge_id, source_node, target_node, distance_m, lanes) "
            "VALUES (:e, 'n1', 'n2', 100.0, 2)"
        ),
        {"e": edge_id},
    )


def _add_intersection(session: Session, iid: str) -> None:
    session.execute(
        text(
            "INSERT INTO intersections (intersection_id, junction_id, lat, lon) "
            "VALUES (:i, 'j', -12.1, -77.0)"
        ),
        {"i": iid},
    )


def _link(session: Session, iid: str, edge_id: str) -> None:
    session.execute(
        text(
            "INSERT INTO intersection_edges (intersection_id, edge_id, direction) "
            "VALUES (:i, :e, 'incoming')"
        ),
        {"i": iid, "e": edge_id},
    )


def _jam(edge_id: str, level: int, ts: datetime) -> WazeJamRow:
    return WazeJamRow(
        event_uuid=f"{edge_id}@{ts.isoformat()}",
        snapshot_timestamp=ts,
        edge_id=edge_id,
        speed_mps=5.0,
        delay_seconds=10,
        congestion_level=level,
        jam_length_m=50,
        road_type=2,
    )


# --- mapeo puro nivel → status (B2 D4) ---

@pytest.mark.parametrize(
    "level,expected",
    [
        (0, "fluid"),
        (1, "fluid"),
        (2, "moderate"),
        (3, "moderate"),
        (4, "critical"),
        (5, "critical"),
        (None, "fluid"),
    ],
)
def test_status_from_level(level, expected):
    assert congestion_status_from_level(level) == expected


# --- agregación MAX por intersección ---

def test_max_level_es_el_peor_edge_y_ultimo_snapshot(bridge_session):
    """MAX sobre las aristas de cada intersección, tomando el último snapshot por arista."""
    t0 = datetime(2026, 6, 5, 8, 0, 0)
    t1 = datetime(2026, 6, 5, 8, 1, 0)

    # arequipa_angamos: dos aristas; el peor último-snapshot es 4.
    _add_intersection(bridge_session, "arequipa_angamos")
    for e in ("e_aa_1", "e_aa_2"):
        _add_edge(bridge_session, e)
        _link(bridge_session, "arequipa_angamos", e)
    # 28julio_reducto: una arista en flujo libre (0) → aparece con 0.
    _add_intersection(bridge_session, "28julio_reducto")
    _add_edge(bridge_session, "e_jr_1")
    _link(bridge_session, "28julio_reducto", "e_jr_1")
    # pardo_espinar: tiene aristas pero SIN dato Waze → ausente del dict.
    _add_intersection(bridge_session, "pardo_espinar")
    _add_edge(bridge_session, "e_pe_1")
    _link(bridge_session, "pardo_espinar", "e_pe_1")

    repo = WazeJamsRepo(bridge_session)
    repo.batch_insert([
        # e_aa_1: snapshot viejo nivel 5, snapshot nuevo nivel 2 → cuenta el 2 (último).
        _jam("e_aa_1", 5, t0),
        _jam("e_aa_1", 2, t1),
        # e_aa_2: último snapshot nivel 4 → es el MAX de la intersección.
        _jam("e_aa_2", 4, t1),
        # e_jr_1: nivel 0 (flujo libre).
        _jam("e_jr_1", 0, t1),
    ])
    bridge_session.commit()

    result = repo.max_level_by_intersection()

    assert result["arequipa_angamos"] == 4
    assert result["28julio_reducto"] == 0
    assert "pardo_espinar" not in result  # sin dato Waze → ausente (endpoint → fluid)


def test_interseccion_sin_aristas_no_aparece(bridge_session):
    _add_intersection(bridge_session, "huerfana")  # sin intersection_edges
    bridge_session.commit()

    assert WazeJamsRepo(bridge_session).max_level_by_intersection() == {}
