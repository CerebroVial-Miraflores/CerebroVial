import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from cerebrovial_shared.database import get_db

from ...application.gru_predictor import GruInferenceService, GruModelUnavailableError
from ...application.predictor import CongestionPredictor
from ...infrastructure.repositories import PredictionsRepo
from .gru_schemas import GruPredictRequest, GruPredictResponse
from .schemas import HistoryResponse

logger = logging.getLogger(__name__)

# Router definition
router = APIRouter(prefix="/predictions", tags=["predictions"])

# --- DI: RandomForest baseline (PRESERVADO) ----------------------------------------
# Sirve GET /predictions/history y queda invocable por TTH-04 como respaldo Nivel 2.
# Su shape viejo dejó de exponerse en POST /predict (reemplazo total, contrato §1).
_predictor_instance: Optional[CongestionPredictor] = None


def get_predictor() -> CongestionPredictor:
    if _predictor_instance is None:
        raise HTTPException(status_code=503, detail="Predictor service not initialized")
    return _predictor_instance


def init_predictor(predictor: CongestionPredictor):
    global _predictor_instance
    _predictor_instance = predictor


# --- DI: GRU (producción, TTH-09 Fase 3b) ------------------------------------------
# Espejo del patrón del RF: singleton in-process inicializado desde main.py.
_gru_service_instance: Optional[GruInferenceService] = None


def get_gru_service() -> GruInferenceService:
    if _gru_service_instance is None:
        raise HTTPException(
            status_code=503, detail="GRU prediction service not initialized"
        )
    return _gru_service_instance


def init_gru_service(service: GruInferenceService):
    global _gru_service_instance
    _gru_service_instance = service


@router.post("/predict", response_model=GruPredictResponse)
async def predict_traffic(
    request: GruPredictRequest,
    gru_service: GruInferenceService = Depends(get_gru_service),
    db: Session = Depends(get_db),
):
    """Predicción GRU por intersección (contrato TTH-09 §2-§3, Fase 3b).

    Reemplazo TOTAL del shape RF viejo en este path (contrato §1): per-intersección,
    4 direcciones (N/S/E/W), 30 pasos de horizonte, ``level`` + ``probs`` por paso.
    Expone ÚNICAMENTE el GRU principal; ante caída emite 5xx (CT-09.8) y delega la
    cascada al RandomForest (Nivel 2) a TTH-04 — el fallback NO se cablea en este módulo.

    Tras inferir, persiste las 120 filas (4 dirs × 30 pasos) de forma best-effort
    (CT-09.5, contrato §8): si la DB falla, se loguea y la predicción se devuelve igual.

    DEPENDENCIA HU-03 / Delta-01 (ruptura conocida, NO silenciosa): el frontend
    (``frontend_ui/src/services/predictionService.ts`` -> ``CameraDetailView.tsx``)
    todavía consume el shape RF viejo de este path y queda ROTO hasta que HU-03 migre el
    cliente al shape de §2-§3. Registrado en
    ``documentation/handoffs/tth-09/tth-09-fase3b-handoff.md``.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    try:
        response = gru_service.predict(request, generated_at=generated_at)
    except GruModelUnavailableError as e:
        # 503: GRU no disponible (modelo no cargado). Body descriptivo para que TTH-04
        # distinga "no disponible" de "falló inferencia" en la cascada (CT-09.8).
        raise HTTPException(status_code=503, detail=f"GRU model unavailable: {e}")
    except Exception as e:  # noqa: BLE001
        # 500: modelo cargado pero la inferencia falló. Body descriptivo (≠ 503).
        raise HTTPException(status_code=500, detail=f"GRU inference failed: {e}")

    # CT-09.5 — persistencia best-effort, NO bloqueante (contrato §8): la predicción es
    # el producto principal; si la DB falla se loguea pero la respuesta se devuelve igual.
    # NO se re-lanza, de modo que un fallo de persistencia nunca se convierte en 5xx.
    try:
        n = PredictionsRepo(db).insert_batch(response)
        db.commit()
        logger.debug(
            "Persisted %d GRU prediction rows (%s)", n, response.intersection_id
        )
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.warning(
            "Best-effort persistence of GRU predictions failed for %s; "
            "returning prediction anyway",
            request.intersection_id,
            exc_info=True,
        )

    return response


@router.get("/history/{camera_id}", response_model=HistoryResponse)
async def get_history(
    camera_id: str,
    interval: int = 5,
    predictor: CongestionPredictor = Depends(get_predictor),
):
    """Histórico + predicción del último log (RF baseline preservado, sin cambios)."""
    try:
        history = predictor.get_traffic_history(camera_id, interval=interval)
        prediction = predictor.predict_future_from_last_log(camera_id)
        return HistoryResponse(
            camera_id=camera_id,
            history=history,
            prediction=prediction,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
