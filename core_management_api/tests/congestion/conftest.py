"""Fixtures de los tests de congestión (TTH-12 Fase 2).

SQLite en memoria con esquema STRIPPED (sin la columna PostGIS ``geom``, que SQLite
no entiende) para ``graph_edges`` y ``waze_jams`` — mismo patrón que
``tests/control/conftest.py``. Añade ``simulation/src`` al path para que el adaptador
pueda reusar ``cerebrovial_simulation.jam_level`` (la derivación 0-5).
"""
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

_REPO = Path(__file__).resolve().parents[3]
_SIM_SRC = str(_REPO / "simulation/src")
if _SIM_SRC not in sys.path:
    sys.path.insert(0, _SIM_SRC)

# Fidelidad SQLite ↔ PostgreSQL en filtros de rango por timestamp (TTH-13).
# pysqlite, al adaptar un ``datetime`` con ``microsecond == 0``, omite los
# microsegundos (almacena ``"... 00:00:00"``), pero SQLAlchemy bindea los
# parámetros de filtro tipados como DateTime con ``".000000"``. En la comparación
# TEXT de SQLite ``"... 00:00:00" < "... 00:00:00.000000"``, así que el borde exacto
# (medianoche) se evalúa mal. PostgreSQL compara por valor y no tiene el artefacto.
# Alineamos el almacenamiento de SQLite al formato con microsegundos para que el
# test ejercite la query real (``series_for_day``) con la misma semántica de bordes.
sqlite3.register_adapter(
    datetime, lambda dt: dt.isoformat(sep=" ", timespec="microseconds")
)

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


class _CongestionClient(TestClient):
    """TestClient con helpers para simular el rol del usuario autenticado."""

    def as_role(self, role: str) -> None:
        from cerebrovial_shared.database.models import UserDB
        from src.auth.presentation.api.dependencies import get_current_user
        self.app.dependency_overrides[get_current_user] = (
            lambda: UserDB(id="u", email="u@x", password_hash="h", role=role)
        )

    def no_auth(self) -> None:
        from src.auth.presentation.api.dependencies import get_current_user
        self.app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def congestion_client(congestion_session) -> Iterator[_CongestionClient]:
    """App FastAPI con solo el router de congestión, get_db → SQLite de la fixture."""
    from cerebrovial_shared.database import get_db
    from src.congestion.presentation.api.routes import router as congestion_router

    app = FastAPI()
    app.include_router(congestion_router)
    app.dependency_overrides[get_db] = lambda: congestion_session
    client = _CongestionClient(app)
    client.session = congestion_session  # acceso directo para sembrar
    yield client
