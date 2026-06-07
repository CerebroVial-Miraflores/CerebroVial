"""Fixtures del suite de corredores (Fase B-2).

Molde: ``tests/control/conftest.py`` (app mínima + SQLite ``:memory:`` con StaticPool +
``dependency_overrides[get_db]``) y ``tests/bdd/conftest.py`` (``seed_user`` + login real para
ejercer ``require_role``). Se evita ``src.main`` (stack pesado torch/geoalchemy).

``graph_nodes``/``graph_edges`` tienen columnas PostGIS (Geometry) que SQLite no entiende; se
crean versiones *stripped* sólo con las columnas que el read-path del matching usa (sin
``geom``). El matching geométrico (que sí necesita PostGIS) se valida aparte en el e2e.
"""
import uuid
from collections.abc import Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from cerebrovial_shared.database import get_db
from cerebrovial_shared.database.models import CorridorDB, CorridorEdgeDB, UserDB

from src.auth.application.password import hash_password
from src.auth.presentation.api.routes import auth_router
from src.corridors.presentation.api.routes import router as corridors_router

GRAPH_NODES_SQLITE_DDL = """
CREATE TABLE graph_nodes (
    node_id TEXT PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    has_camera BOOLEAN
)
"""
GRAPH_EDGES_SQLITE_DDL = """
CREATE TABLE graph_edges (
    edge_id TEXT PRIMARY KEY,
    source_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    distance_m REAL NOT NULL,
    lanes INTEGER NOT NULL
)
"""

# Cadena de prueba: 3 nodos en línea → 2 aristas CONTINUAS e1(n1→n2), e2(n2→n3).
# ``nx`` es un nodo suelto para fabricar una cadena ROTA (e_broken arranca en nx, no en n2).
_NODES = [
    ("n1", -12.1200, -77.0300),
    ("n2", -12.1210, -77.0300),
    ("n3", -12.1220, -77.0300),
    ("nx", -12.2000, -77.1000),
]
_EDGES = [
    ("e1", "n1", "n2", 110.0, 2),
    ("e2", "n2", "n3", 110.0, 2),
    ("e_broken", "nx", "n3", 110.0, 2),
]


@pytest.fixture
def db_engine() -> Iterator[Engine]:
    # StaticPool: una sola conexión compartida → lo sembrado es visible entre sesiones.
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with eng.begin() as conn:
        conn.execute(sa_text(GRAPH_NODES_SQLITE_DDL))
        conn.execute(sa_text(GRAPH_EDGES_SQLITE_DDL))
    UserDB.__table__.create(bind=eng)
    CorridorDB.__table__.create(bind=eng)
    CorridorEdgeDB.__table__.create(bind=eng)
    yield eng
    eng.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    for node_id, lat, lon in _NODES:
        session.execute(
            sa_text(
                "INSERT INTO graph_nodes (node_id, lat, lon, has_camera) "
                "VALUES (:n, :lat, :lon, 1)"
            ),
            {"n": node_id, "lat": lat, "lon": lon},
        )
    for edge_id, src, tgt, dist, lanes in _EDGES:
        session.execute(
            sa_text(
                "INSERT INTO graph_edges (edge_id, source_node, target_node, distance_m, lanes) "
                "VALUES (:e, :s, :t, :d, :l)"
            ),
            {"e": edge_id, "s": src, "t": tgt, "d": dist, "l": lanes},
        )
    session.commit()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def app(db_session: Session) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(auth_router)
    test_app.include_router(corridors_router)

    def _override_get_db() -> Iterator[Session]:
        yield db_session

    test_app.dependency_overrides[get_db] = _override_get_db
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def seed_user(db_session: Session) -> Callable[..., UserDB]:
    def _seed(*, email: str, password: str, role: str = "operator") -> UserDB:
        user = UserDB(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(password),
            role=role,
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return user

    return _seed


@pytest.fixture
def auth_headers(client: TestClient, seed_user: Callable[..., UserDB]) -> Callable[..., dict]:
    """Siembra un usuario del rol pedido, hace login real y devuelve el header Bearer."""

    def _headers(role: str = "operator") -> dict[str, str]:
        email = f"{role}@cv.pe"
        password = "passw0rd-corr"
        seed_user(email=email, password=password, role=role)
        resp = client.post("/auth/login", data={"username": email, "password": password})
        assert resp.status_code == 200, resp.text
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    return _headers
