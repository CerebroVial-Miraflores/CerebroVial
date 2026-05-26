"""Root pytest fixtures for core_management_api.

Sets JWT env vars at import time so any auth module imported during test
collection finds a valid configuration. Provides a SQLite in-memory engine
and session fixtures scoped per-test, used by suites that need persistence.
"""
import os

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production")
os.environ.setdefault("JWT_EXPIRATION_HOURS", "8")
# Avoid touching the real Postgres engine at import time of cerebrovial_shared.
# Tests use their own SQLite engine via the test_engine fixture and override get_db.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cerebrovial_shared.database.models import UserDB


@pytest.fixture
def test_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    UserDB.__table__.create(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def test_db(test_engine):
    TestingSession = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
