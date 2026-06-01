"""Tests del endpoint GRU de producción (TTH-09 Fase 3b, contrato §2-§3).

Cubren el shape nuevo de ``POST /predictions/predict`` (per-intersección, 4 direcciones,
30 pasos, ``level`` + ``probs``), la validación de input (422) y las rutas 5xx de CT-09.8
(503 modelo no disponible / servicio no inicializado; 500 falló la inferencia).

No dependen de los ``.pt`` reales: el happy path usa modelos ``GRUMultiOutput`` recién
construidos (pesos random) en ``eval`` — verifica el CONTRATO/shape de la response, no la
corrección numérica (eso lo cubre el test de paridad de Fase 3a, ``test_gru_engine.py``).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# torch es dependencia CPU-only del core (D-010); skip limpio si no está. No se usa
# torch directamente acá (los modelos se construyen vía GRUMultiOutput), solo se exige.
pytest.importorskip("torch")

from src.prediction.application.gru_predictor import (  # noqa: E402
    GRU_MODEL_VERSION,
    GruInferenceService,
)
from src.prediction.infrastructure.gru_engine import GRUModelEngine  # noqa: E402
from src.prediction.infrastructure.gru_model import GRUMultiOutput  # noqa: E402
from src.prediction.presentation.api import routes as routes_module  # noqa: E402
from src.prediction.presentation.api.routes import get_gru_service, router  # noqa: E402

DIRECTIONS = ("N", "S", "E", "W")

test_app = FastAPI()
test_app.include_router(router)
client = TestClient(test_app)


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


class _BoomModel:
    """Modelo 'cargado' cuyo forward revienta — ejercita el 500 de inferencia."""

    def __call__(self, x):
        raise RuntimeError("boom en forward")


@pytest.fixture(autouse=True)
def _reset_di():
    """Aísla el singleton global del módulo entre tests (y de otros archivos)."""
    routes_module._gru_service_instance = None
    test_app.dependency_overrides = {}
    yield
    test_app.dependency_overrides = {}
    routes_module._gru_service_instance = None


def _override(service: GruInferenceService):
    test_app.dependency_overrides[get_gru_service] = lambda: service


def test_predict_happy_path_shape():
    """Reemplaza la cobertura del viejo test_routes.py::test_predict_traffic_success."""
    _override(GruInferenceService(_engine_with_models()))
    resp = client.post("/predictions/predict", json=_valid_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["intersection_id"] == "larco_schell"
    assert body["model_version"] == GRU_MODEL_VERSION
    assert "generated_at" in body
    assert [p["direction"] for p in body["predictions"]] == list(DIRECTIONS)
    for pred in body["predictions"]:
        assert len(pred["horizon"]) == 30
        assert [s["step"] for s in pred["horizon"]] == list(range(1, 31))
        for step in pred["horizon"]:
            assert 0 <= step["level"] <= 5
            assert len(step["probs"]) == 6
            assert sum(step["probs"]) == pytest.approx(1.0, abs=1e-4)
            # level = argmax(probs), §3
            assert step["level"] == max(range(6), key=lambda i: step["probs"][i])


def test_validation_missing_direction_422():
    payload = _valid_payload()
    payload["series"] = payload["series"][:3]  # solo N/S/E
    _override(GruInferenceService(_engine_with_models()))
    resp = client.post("/predictions/predict", json=payload)
    assert resp.status_code == 422


def test_validation_wrong_length_422():
    payload = _valid_payload()
    payload["series"][0]["jam_levels"] = [0] * 29
    _override(GruInferenceService(_engine_with_models()))
    resp = client.post("/predictions/predict", json=payload)
    assert resp.status_code == 422


def test_validation_out_of_range_422():
    payload = _valid_payload()
    payload["series"][0]["jam_levels"][0] = 7
    _override(GruInferenceService(_engine_with_models()))
    resp = client.post("/predictions/predict", json=payload)
    assert resp.status_code == 422


def test_model_unavailable_503():
    """CT-09.8: engine inicializado pero SIN modelos -> 503 con body descriptivo."""
    engine = GRUModelEngine(model_dir="/nonexistent")  # models vacío
    _override(GruInferenceService(engine))
    resp = client.post("/predictions/predict", json=_valid_payload())
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


def test_inference_failure_500():
    """CT-09.8: modelo cargado pero el forward falla -> 500 con body descriptivo (≠ 503)."""
    engine = GRUModelEngine(model_dir="/nonexistent")
    engine.models = {d: _BoomModel() for d in DIRECTIONS}
    _override(GruInferenceService(engine))
    resp = client.post("/predictions/predict", json=_valid_payload())
    assert resp.status_code == 500
    assert "inference failed" in resp.json()["detail"].lower()


def test_service_not_initialized_503():
    """DI no cableado (init_gru_service nunca llamado) -> 503.

    Reemplaza al viejo test_routes.py::test_predict_service_unavailable, adaptado al GRU.
    """
    routes_module._gru_service_instance = None  # sin override, global None
    resp = client.post("/predictions/predict", json=_valid_payload())
    assert resp.status_code == 503
    assert "not initialized" in resp.json()["detail"].lower()
