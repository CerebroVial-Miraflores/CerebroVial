"""Fixtures para los escenarios BDD de HU-01 (RBAC).

Monta una FastAPI mínima con el auth_router + el endpoint de muestra
GET /api/health protegido con require_role(Role.ADMIN), evitando el stack
de prediction/vision (torch, joblib, geoalchemy) que es ortogonal al test.
"""
import uuid
from collections.abc import Callable

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from cerebrovial_shared.database.database import get_db
from cerebrovial_shared.database.models import UserDB

from src.auth.application.password import hash_password
from src.auth.domain import Role
from src.auth.presentation.api.dependencies import require_role
from src.auth.presentation.api.routes import auth_router


@pytest.fixture
def app_with_rbac(test_db) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(auth_router)

    @test_app.get(
        "/api/health",
        dependencies=[Depends(require_role(Role.ADMIN))],
    )
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    def _override_get_db():
        yield test_db

    test_app.dependency_overrides[get_db] = _override_get_db
    return test_app


@pytest.fixture
def client(app_with_rbac: FastAPI) -> TestClient:
    return TestClient(app_with_rbac)


@pytest.fixture
def seed_user(test_db) -> Callable[..., UserDB]:
    def _seed(
        *,
        email: str,
        password: str,
        role: str = "operator",
        user_id: str | None = None,
    ) -> UserDB:
        user = UserDB(
            id=user_id or str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(password),
            role=role,
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)
        return user

    return _seed
