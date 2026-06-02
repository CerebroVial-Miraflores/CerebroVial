"""Fixtures de los tests de congestión (TTH-12 Fase 2).

SQLite en memoria con esquema STRIPPED (sin la columna PostGIS ``geom``, que SQLite
no entiende) para ``graph_edges`` y ``waze_jams`` — mismo patrón que
``tests/control/conftest.py``. Añade ``simulation/src`` al path para que el adaptador
pueda reusar ``cerebrovial_simulation.jam_level`` (la derivación 0-5).
"""
import sys
from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

_REPO = Path(__file__).resolve().parents[3]
_SIM_SRC = str(_REPO / "simulation/src")
if _SIM_SRC not in sys.path:
    sys.path.insert(0, _SIM_SRC)

# Esquema stripped (sin geom): refleja las columnas que el repo escribe/lee.
_GRAPH_EDGES_DDL = """
CREATE TABLE graph_edges (
    edge_id TEXT PRIMARY KEY,
    source_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    distance_m REAL NOT NULL,
    lanes INTEGER NOT NULL
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
def congestion_session() -> Iterator[Session]:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with eng.begin() as conn:
        conn.execute(text(_GRAPH_EDGES_DDL))
        conn.execute(text(_WAZE_JAMS_DDL))
    SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        eng.dispose()
