"""
Tests for ``GET /control/health`` (TTH-10, CT-10.13 partial — only the
health endpoint; the TTH-04 cascade is R2).
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from cerebrovial_shared.database import get_db
from src.control.presentation.api import routes as control_routes
from src.control.presentation.api.routes import router as control_router


def test_health_returns_200_when_engine_and_db_are_healthy_ct_10_13_1(client_with_db):
    """CT-10.13.1: engine initialised + DB reachable → 200 with explicit
    status payload."""
    response = client_with_db.get("/control/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "engine": "ready", "db": "ok"}


def test_health_returns_503_when_db_is_unreachable_ct_10_13_2(engine):
    """CT-10.13.2: engine initialised but DB raises → 503 with
    detail.db='unreachable'."""

    class _BrokenSession:
        def execute(self, *_args, **_kwargs):
            raise SQLAlchemyError("boom")

    app = FastAPI()
    control_routes.init_engine(engine)
    app.include_router(control_router)

    def _broken_db():
        yield _BrokenSession()

    app.dependency_overrides[get_db] = _broken_db

    with TestClient(app) as broken_client:
        response = broken_client.get("/control/health")

    assert response.status_code == 503
    assert response.json()["detail"] == {"engine": "ready", "db": "unreachable"}


def test_health_returns_503_when_engine_not_initialized():
    """Additional safety net: with no engine, /control/health is 503
    irrespective of DB state."""
    saved = control_routes._engine_instance
    control_routes._engine_instance = None
    try:
        app = FastAPI()
        app.include_router(control_router)

        def _dummy_db():
            yield None  # never reached: engine check fires first

        app.dependency_overrides[get_db] = _dummy_db

        with TestClient(app) as ephemeral:
            response = ephemeral.get("/control/health")

        assert response.status_code == 503
        assert response.json()["detail"] == {"engine": "not_ready", "db": "unknown"}
    finally:
        control_routes._engine_instance = saved
