"""Tests del RandomForest baseline preservado en el router de predicción.

TTH-09 Fase 3b reemplazó ``POST /predictions/predict`` por el contrato GRU (reemplazo
total, §1). Los tests del shape RF viejo en ``/predict`` se RETIRARON; su cobertura se
trasladó —sin pérdida— a ``tests/prediction/test_gru_routes.py``. Aquí queda lo que NO
cambió: ``GET /predictions/history``, servido por ``CongestionPredictor`` (RF), intacto.

Retiros (Fase 3b), registrados para que no se pierda cobertura silenciosamente:
- ``test_predict_traffic_success``    -> ``test_gru_routes.py::test_predict_happy_path_shape``
- ``test_predict_service_unavailable`` -> ``test_gru_routes.py::test_service_not_initialized_503``
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.prediction.application.predictor import CongestionPredictor
from src.prediction.presentation.api.routes import get_predictor, router
from src.prediction.presentation.api.schemas import HistoricalDataPoint

test_app = FastAPI()
test_app.include_router(router)

client = TestClient(test_app)


@pytest.fixture
def mock_predictor():
    return MagicMock(spec=CongestionPredictor)


@pytest.fixture
def override_dependency(mock_predictor):
    test_app.dependency_overrides[get_predictor] = lambda: mock_predictor
    yield
    test_app.dependency_overrides = {}


def test_get_history_success(mock_predictor, override_dependency):
    mock_predictor.get_traffic_history.return_value = [
        HistoricalDataPoint(
            timestamp="10:00",
            total_vehicles=20,
            congestion_level="Normal",
            is_prediction=False,
        )
    ]
    mock_predictor.predict_future_from_last_log.return_value = {
        "current_congestion_level": "Normal",
        "predicted_congestion_15min": "Normal",
        "predicted_vehicles_15min": 20,
        "predicted_congestion_30min": "Normal",
        "predicted_vehicles_30min": 20,
        "predicted_congestion_45min": "Normal",
        "predicted_vehicles_45min": 20,
        "confidence_score": 0.9,
        "meta_info": {},
    }

    response = client.get("/predictions/history/CAM_001?interval=5")

    assert response.status_code == 200
    data = response.json()
    assert data["camera_id"] == "CAM_001"
    assert len(data["history"]) == 1
    assert data["prediction"]["predicted_congestion_15min"] == "Normal"
