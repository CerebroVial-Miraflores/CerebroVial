"""CT-09.9(e) — persistencia durable de las predicciones del GRU (CT-09.5).

Verifica que tras un ``POST /predictions/predict`` válido las 120 filas
(4 direcciones × 30 pasos) quedan persistidas en la tabla ``predictions`` con los
campos del contrato §8 (intersection_id, direction, step, level, probs, model_version,
generated_at), y que la persistencia es **best-effort**: si el repo/DB falla, el
endpoint devuelve igual 200 con la predicción (la persistencia no bloquea la respuesta).

Molde SQLite in-memory + StaticPool del suite de control (``tests/control/conftest.py``),
sin graph_nodes: ``predictions.intersection_id`` no tiene FK, no hay que seedear nada.
``JsonType`` (JSONB→JSON en sqlite) ya está probado en SQLite por el suite de control.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# torch es dependencia CPU-only del core (D-010); skip limpio si no está.
pytest.importorskip("torch")

from cerebrovial_shared.database import get_db  # noqa: E402
from cerebrovial_shared.database.models import PredictionDB  # noqa: E402

from src.prediction.application.gru_predictor import (  # noqa: E402
    GRU_MODEL_VERSION,
    GruInferenceService,
)
from src.prediction.infrastructure.gru_engine import GRUModelEngine  # noqa: E402
from src.prediction.infrastructure.gru_model import GRUMultiOutput  # noqa: E402
from src.prediction.presentation.api import routes as routes_module  # noqa: E402
from src.prediction.presentation.api.routes import get_gru_service, router  # noqa: E402

DIRECTIONS = ("N", "S", "E", "W")


def _valid_payload() -> dict:
    return {
        "intersection_id": "larco_schell",
        "timestamp": "2026-05-31T10:00:00Z",
        "series": [
            {"direction": d, "jam_levels": [(i % 6) for i in range(30)]}
            for d in DIRECTIONS
        ],
    }


def _engine_with_models() -> GRUModelEngine:
    """Engine con 4 GRUMultiOutput construidos en memoria (sin IO de disco, sin .pt)."""
    engine = GRUModelEngine(model_dir="/nonexistent")  # __init__ no hace IO
    engine.models = {
        d: GRUMultiOutput(hidden=8, horizonte=30, n_classes=6).eval()
        for d in DIRECTIONS
    }
    return engine


@pytest.fixture
def db_session() -> Iterator[Session]:
    # StaticPool fuerza una única conexión compartida, de modo que las filas
    # escritas dentro del request sean visibles a las queries del test.
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    PredictionDB.__table__.create(bind=eng)
    SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        eng.dispose()


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(router)

    # Override de get_db con la forma generadora idéntica a control/conftest.py
    # (no `lambda: iter([...])`): replica la semántica generator-dependency con
    # cleanup que FastAPI espera y evita fragilidad si se itera más de una vez.
    def _override_get_db() -> Iterator[Session]:
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_gru_service] = lambda: GruInferenceService(
        _engine_with_models()
    )
    yield TestClient(app)
    app.dependency_overrides = {}


def test_predict_persists_120_rows_ct_09_9e(client: TestClient, db_session: Session):
    """CT-09.9(e): un POST válido persiste 120 filas con los campos de CT-09.5/§8."""
    resp = client.post("/predictions/predict", json=_valid_payload())
    assert resp.status_code == 200, resp.text

    rows = db_session.query(PredictionDB).all()
    assert len(rows) == 120  # 4 direcciones × 30 pasos

    # Una entrada por dirección, 30 pasos t+1..t+30 cada una.
    assert {r.direction for r in rows} == set(DIRECTIONS)
    for d in DIRECTIONS:
        steps = sorted(r.step for r in rows if r.direction == d)
        assert steps == list(range(1, 31))

    # Campos del contrato §8 en cada fila.
    for r in rows:
        assert r.intersection_id == "larco_schell"
        assert r.direction in DIRECTIONS
        assert 1 <= r.step <= 30
        assert 0 <= r.level <= 5
        assert isinstance(r.probs, list) and len(r.probs) == 6
        assert r.model_version == GRU_MODEL_VERSION
        assert r.generated_at is not None
        assert r.prediction_id  # uuid auto


def test_persistence_failure_is_best_effort(
    client: TestClient, db_session: Session, monkeypatch
):
    """Si la persistencia falla (repo que lanza), el endpoint devuelve igual 200.

    La predicción es el producto principal (contrato §8): un fallo de DB se loguea
    pero NO se propaga ni se convierte en 5xx.
    """

    class _RaisingRepo:
        def __init__(self, session):
            pass

        def insert_batch(self, response):
            raise RuntimeError("DB caída / repo que lanza")

    monkeypatch.setattr(routes_module, "PredictionsRepo", _RaisingRepo)

    resp = client.post("/predictions/predict", json=_valid_payload())
    assert resp.status_code == 200, resp.text

    # La predicción se devuelve completa pese al fallo de persistencia.
    body = resp.json()
    assert body["intersection_id"] == "larco_schell"
    assert [p["direction"] for p in body["predictions"]] == list(DIRECTIONS)
    for pred in body["predictions"]:
        assert len(pred["horizon"]) == 30

    # Nada quedó persistido (la transacción se hizo rollback).
    assert db_session.query(PredictionDB).count() == 0
