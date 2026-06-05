"""Fixtures de los tests de intersecciones (Fase A).

- Pone ``scripts/`` en el path para importar el seed (``seed_intersections.build_rows``),
  igual que el e2e del builder de geometría (TTH-12).
- SQLite en memoria con esquema STRIPPED (sin la columna PostGIS ``geom``) para
  ``intersections`` / ``intersection_edges`` / ``graph_edges``, mismo patrón que
  ``tests/congestion/conftest.py``, para ejercitar el JOIN del puente.
"""
import sys
from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = str(_REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Esquema stripped (sin geom): refleja las columnas que el puente JOIN-ea.
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
    SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        eng.dispose()
