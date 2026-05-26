"""Test fixtures for the auth suite.

Mirrors the pattern used by tests/control/conftest.py: assemble a minimal
FastAPI app that mounts only the auth router, avoiding the prediction/vision
stack (torch, geoalchemy2). The shared get_db dependency is overridden to
use the in-memory SQLite session from the root conftest.
"""
import uuid
from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cerebrovial_shared.database.database import get_db
from cerebrovial_shared.database.models import UserDB

from src.auth.application.password import hash_password
from src.auth.presentation.api.routes import auth_router


@pytest.fixture
def app(test_db) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(auth_router)

    def _override_get_db():
        yield test_db

    test_app.dependency_overrides[get_db] = _override_get_db
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


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
