"""
Test fixtures shared by the control module test suite.

The fixtures intentionally avoid importing ``src.main`` so the suite runs
without the heavy prediction/vision dependencies (torch, geoalchemy2, etc.).
A minimal FastAPI app is assembled with only the control router.

Two parallel fixture families coexist:

- ``engine`` / ``app`` / ``client``: legacy, no DB. Used by tests that only
  exercise the engine's pure logic (no HTTP write-path).
- ``motor_db_engine`` / ``motor_db_session`` / ``app_with_db`` /
  ``client_with_db``: TTH-10 write-path. SQLite ``:memory:`` with the three
  tables involved (``graph_nodes``, ``motor_decisions``,
  ``engine_active_state``) and a single seeded node ``larco_schell`` so
  ``resolve_node_id`` resolves and the FK is satisfied.
"""
from collections.abc import Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from cerebrovial_shared.database import get_db
from cerebrovial_shared.database.models import (
    EngineActiveStateDB,
    MotorDecisionDB,
)

from src.control.application.adaptive_engine import (
    AdaptiveEngine,
    IntersectionStateDC,
    PhaseFlowDC,
)
from src.control.application.max_pressure import MaxPressureController
from src.control.application.mtc_constraints import MTCRestrictionApplier
from src.control.application.webster import WebsterCalculator
from src.control.presentation.api.routes import init_engine, router as control_router


# --- Legacy fixtures (no DB) ---

@pytest.fixture
def engine() -> AdaptiveEngine:
    return AdaptiveEngine(
        webster=WebsterCalculator(),
        max_pressure=MaxPressureController(),
        mtc=MTCRestrictionApplier(),
    )


@pytest.fixture
def app(engine: AdaptiveEngine) -> FastAPI:
    test_app = FastAPI()
    init_engine(engine)
    test_app.include_router(control_router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# --- TTH-10 write-path fixtures (SQLite in-memory) ---

# graph_nodes has a PostGIS Geometry column. SQLite does not understand it,
# so we build only the three columns we need for the FK + seed: node_id,
# lat, lon. We do this by creating a stripped table at the SQL level.
GRAPH_NODES_SQLITE_DDL = """
CREATE TABLE graph_nodes (
    node_id TEXT PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    has_camera BOOLEAN
)
"""


@pytest.fixture
def motor_db_engine() -> Iterator[Engine]:
    # StaticPool forces a single shared connection so the seeded data is
    # visible across the multiple sessions FastAPI may open per request.
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # graph_nodes stripped (no Geometry column under SQLite).
    with eng.begin() as conn:
        from sqlalchemy import text as sa_text
        conn.execute(sa_text(GRAPH_NODES_SQLITE_DDL))
    MotorDecisionDB.__table__.create(bind=eng)
    EngineActiveStateDB.__table__.create(bind=eng)
    yield eng
    eng.dispose()


@pytest.fixture
def motor_db_session(motor_db_engine: Engine) -> Iterator[Session]:
    SessionLocal = sessionmaker(
        bind=motor_db_engine, autoflush=False, autocommit=False
    )
    session = SessionLocal()
    # Seed via raw SQL: GraphNodeDB has a GeoAlchemy2 Geometry column that
    # SQLite cannot represent. The write-path only queries by PK, so the
    # stripped table (defined in GRAPH_NODES_SQLITE_DDL) is sufficient.
    from sqlalchemy import text as sa_text
    session.execute(
        sa_text(
            "INSERT INTO graph_nodes (node_id, lat, lon, has_camera) "
            "VALUES ('larco_schell', -12.1195, -77.0298, 1)"
        )
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def app_with_db(
    engine: AdaptiveEngine, motor_db_session: Session
) -> FastAPI:
    test_app = FastAPI()
    init_engine(engine)
    test_app.include_router(control_router)

    def _override_get_db() -> Iterator[Session]:
        yield motor_db_session

    test_app.dependency_overrides[get_db] = _override_get_db
    return test_app


@pytest.fixture
def client_with_db(app_with_db: FastAPI) -> TestClient:
    return TestClient(app_with_db)


# --- Builders ---

@pytest.fixture
def build_state() -> Callable[..., IntersectionStateDC]:
    """Helper to assemble an IntersectionStateDC with two phases (NS, EW) and a target flow_total."""

    def _build(
        flow_total: float,
        *,
        intersection_id: str = "I-001",
        timestamp: str = "2026-05-09T10:00:00Z",
        saturation: float = 1800.0,
        queues: tuple[int, int] = (3, 2),
        lost_time: float = 8.0,
        ns_share: float = 0.55,
    ) -> IntersectionStateDC:
        ns_flow = flow_total * ns_share
        ew_flow = flow_total * (1.0 - ns_share)
        return IntersectionStateDC(
            intersection_id=intersection_id,
            timestamp=timestamp,
            lost_time=lost_time,
            phases=[
                PhaseFlowDC(
                    phase_id="NS",
                    flow=ns_flow,
                    saturation_flow=saturation,
                    queue=queues[0],
                ),
                PhaseFlowDC(
                    phase_id="EW",
                    flow=ew_flow,
                    saturation_flow=saturation,
                    queue=queues[1],
                ),
            ],
        )

    return _build


@pytest.fixture
def http_payload() -> Callable[..., dict]:
    """Helper to assemble the JSON payload accepted by POST /control/recommend.

    Default ``intersection_id`` is ``"larco_schell"`` so the write-path
    (resolve_node_id) succeeds against the seeded ``motor_db_session``.
    """

    def _payload(
        flow_total: float,
        *,
        intersection_id: str = "larco_schell",
        saturation: float = 1800.0,
        queues: tuple[int, int] = (3, 2),
    ) -> dict:
        ns = flow_total * 0.55
        ew = flow_total - ns
        return {
            "intersection_id": intersection_id,
            "timestamp": "2026-05-09T10:00:00Z",
            "lost_time": 8.0,
            "phases": [
                {
                    "phase_id": "NS",
                    "flow": ns,
                    "saturation_flow": saturation,
                    "queue": queues[0],
                },
                {
                    "phase_id": "EW",
                    "flow": ew,
                    "saturation_flow": saturation,
                    "queue": queues[1],
                },
            ],
        }

    return _payload
