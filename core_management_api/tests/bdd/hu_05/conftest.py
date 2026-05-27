"""Fixtures BDD para HU-05 (vista pasiva del estado vigente).

Extiende el conftest BDD de hu-01 montando una FastAPI mínima con:
- auth_router (para /auth/login y el flujo de JWT)
- control_router (para GET /control/active-state/{node_id} y su stream)
- POST /control/__internal/activate (dev-only, gated por ENABLE_TEST_ACTIVATOR)

La BD es SQLite ``:memory:`` con las cuatro tablas relevantes:
- ``users`` (UserDB) — necesaria para seed_user/login
- ``graph_nodes`` (versión stripped: la columna Geometry se descarta por SQLite)
- ``motor_decisions`` (registro de decisiones del motor)
- ``engine_active_state`` (puntero a la decisión vigente por nodo)

Las fixtures ``test_db`` y ``client`` de este archivo sobreescriben las del
conftest padre para que ``seed_user`` y los handlers del control router usen
la misma sesión (la auth + el control comparten BD).
"""
import os

# Debe ejecutarse ANTES de importar src.control.presentation.api.routes para
# que el ``setup_test_activator()`` a nivel de módulo encuentre el gate abierto.
# Si routes.py ya fue importado por otro conftest, la llamada explícita más
# abajo cubre el caso idempotentemente.
os.environ.setdefault("ENABLE_TEST_ACTIVATOR", "true")

from collections.abc import Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from cerebrovial_shared.database import get_db
from cerebrovial_shared.database.models import (
    EngineActiveStateDB,
    MotorDecisionDB,
    UserDB,
)

from src.auth.presentation.api.routes import auth_router
from src.control.application.adaptive_engine import AdaptiveEngine
from src.control.application.max_pressure import MaxPressureController
from src.control.application.mtc_constraints import MTCRestrictionApplier
from src.control.application.webster import WebsterCalculator
from src.control.infrastructure import (
    ActiveStateBroadcaster,
    EngineActiveStateRepo,
    MotorDecisionsRepo,
    init_broadcaster,
)
from src.control.presentation.api.routes import (
    init_engine,
    router as control_router,
    setup_test_activator,
)

# Registro idempotente: si routes.py se importó antes de que esta conftest
# seteara la env var, la llamada al import de arriba ya hizo setup interno
# pero el gate puede haber estado off. Reintentamos con la env var ya seteada.
setup_test_activator()


GRAPH_NODES_SQLITE_DDL = """
CREATE TABLE graph_nodes (
    node_id TEXT PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    has_camera BOOLEAN
)
"""


@pytest.fixture
def hu05_engine() -> Iterator[Engine]:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with eng.begin() as conn:
        conn.execute(sa_text(GRAPH_NODES_SQLITE_DDL))
    UserDB.__table__.create(bind=eng)
    MotorDecisionDB.__table__.create(bind=eng)
    EngineActiveStateDB.__table__.create(bind=eng)
    yield eng
    eng.dispose()


@pytest.fixture
def hu05_session(hu05_engine: Engine) -> Iterator[Session]:
    SessionLocal = sessionmaker(
        bind=hu05_engine, autoflush=False, autocommit=False
    )
    session = SessionLocal()
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
def test_db(hu05_session: Session) -> Session:
    """Override del fixture homónimo del conftest raíz: aquí ``seed_user`` (del
    conftest BDD padre) usa la misma sesión que el control router."""
    return hu05_session


@pytest.fixture
def app_with_rbac(hu05_session: Session) -> FastAPI:
    """Override del fixture del conftest padre.

    Monta auth_router + control_router contra la sesión combinada para que el
    login y la lectura de estado vigente operen sobre la misma BD."""
    test_app = FastAPI()
    test_app.include_router(auth_router)
    test_app.include_router(control_router)

    engine = AdaptiveEngine(
        webster=WebsterCalculator(),
        max_pressure=MaxPressureController(),
        mtc=MTCRestrictionApplier(),
    )
    init_engine(engine)

    def _override_get_db() -> Iterator[Session]:
        yield hu05_session

    test_app.dependency_overrides[get_db] = _override_get_db
    return test_app


@pytest.fixture
def client(app_with_rbac: FastAPI) -> TestClient:
    return TestClient(app_with_rbac)


@pytest.fixture
def seed_motor_decision(hu05_session: Session) -> Callable[..., str]:
    """Inserta una fila en motor_decisions y devuelve su decision_id."""

    def _seed(
        *,
        node_id: str = "larco_schell",
        mode: str = "webster",
        cycle_seconds: float = 60.0,
    ) -> str:
        repo = MotorDecisionsRepo(hu05_session)
        decision_id = repo.insert(
            node_id=node_id,
            mode=mode,
            cycle_seconds=cycle_seconds,
            flow_total=1500.0,
            y_load_factor=0.65,
            next_phase=None,
            reasoning="seed-hu05",
            phase_timings=[
                {"phase_id": "NS", "green": 25, "yellow": 3, "all_red": 2},
                {"phase_id": "EW", "green": 20, "yellow": 3, "all_red": 2},
            ],
            adjustments=[],
            inputs_snapshot={"lost_time": 8.0, "phases": []},
        )
        hu05_session.commit()
        return decision_id

    return _seed


@pytest.fixture
def broadcaster_instance() -> ActiveStateBroadcaster:
    """Fresh broadcaster per test, wired into the process-global singleton so
    that ``Depends(get_broadcaster)`` in the SSE stream endpoint and the
    ``activator`` fixture below refer to the SAME instance."""
    instance = ActiveStateBroadcaster()
    init_broadcaster(instance)
    return instance


@pytest.fixture
def activator(
    hu05_session: Session, broadcaster_instance: ActiveStateBroadcaster
) -> Callable[..., None]:
    """Sync activator for non-SSE scenarios (CA-05.1/2/5 + contrato_404).

    Only writes to the DB + commits. Does NOT touch the broadcaster — the
    sync scenarios verify HTTP contract, not the live SSE wake-up. The async
    counterpart used by CA-05.3 lives in ``test_hu05_sse.py`` and exercises
    ``commit → publish`` end-to-end.

    The ``broadcaster_instance`` fixture is depended-upon to guarantee the
    singleton is initialized before ``Depends(get_broadcaster)`` is resolved
    (the stream endpoint depends on it even if a sync test doesn't hit it).
    """
    del broadcaster_instance  # only here to anchor the dependency

    def _activate(
        node_id: str, decision_id: str, activated_by: str = "test"
    ) -> None:
        repo = EngineActiveStateRepo(hu05_session)
        repo.activate(
            node_id=node_id,
            decision_id=decision_id,
            activated_by=activated_by,
        )
        hu05_session.commit()

    return _activate
