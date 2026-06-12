import logging
import os
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from cerebrovial_shared.database.database import get_db
from cerebrovial_shared.database.models import CameraDB
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from src.prediction.presentation.api.routes import router as prediction_router, init_predictor, init_gru_service
from src.prediction.application.predictor import CongestionPredictor
from src.prediction.application.gru_predictor import GruInferenceService
from src.prediction.infrastructure.gru_engine import GRUModelEngine
from src.control.presentation.api.routes import router as control_router, init_engine
from src.control.application.adaptive_engine import AdaptiveEngine
from src.control.application.webster import WebsterCalculator
from src.control.application.max_pressure import MaxPressureController
from src.control.application.mtc_constraints import MTCConstants, MTCRestrictionApplier
from src.control.config import ControlSettings
from src.control.infrastructure import init_broadcaster
from src.congestion.presentation.api.routes import router as congestion_router, init_prediction_service
from src.congestion.application.congestion_prediction_service import CongestionPredictionService
from src.congestion.application.intersection_congestion import congestion_status_from_level
from src.congestion.infrastructure import init_congestion_broadcaster
from src.congestion.infrastructure.repositories import WazeJamsRepo
from src.congestion.infrastructure.gru_inference_adapter import load_gru_adapter
from src.congestion.infrastructure.prediction_day_source import PredictionDaySource
from src.corridors.presentation.api.routes import router as corridors_router
from src.auth.domain import Role
from src.auth.presentation.api.dependencies import require_role
from src.auth.presentation.api.routes import auth_router

logger = logging.getLogger(__name__)

_PREDICTION_CKPT = "models/miraflores_gru_baseline.pt"

app = FastAPI(title="CerebroVial Core API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TTH-13: compresión de transporte. La serie del día por arista
# (GET /congestion/series) ronda ~1.58 MB crudos y comprime a ~55 KB gzip.
# Aditivo y transparente para el resto de endpoints (sólo comprime sobre el umbral).
app.add_middleware(GZipMiddleware, minimum_size=1024)

os.makedirs("models", exist_ok=True)
os.makedirs("data/traffic_logs", exist_ok=True)
_predictor = CongestionPredictor(model_dir="models", data_dir="data/traffic_logs")
init_predictor(_predictor)

# TTH-09 Fase 3b: wiring del GRU (producción) AL LADO del RF baseline, sin tocarlo.
# Carga eager los 4 gru_<dir>.pt (tolerante a faltantes, Fase 3a). El RF queda como
# respaldo Nivel 2 invocable por TTH-04. Si el GRU no puede servir, POST /predictions/predict
# emite 5xx (CT-09.8); el core bootea igual y GET /predictions/history (RF) sigue operativo.
_gru_engine = GRUModelEngine()
_gru_engine.load_models()
init_gru_service(GruInferenceService(_gru_engine))

# Fase 3: predicción de congestión por arista servida in-process (GRU baseline vendorizado,
# GRU-only sin tsl). Carga eager el slice del día seed051 (~12 MB RAM) + el checkpoint. Es
# OTRO predictor que el GRU 4-way de TTH-09 (por intersección): coexisten. Tolerante a
# faltantes — si el slice o el .pt no están, el core bootea igual y GET /congestion/prediction
# responde 503 (no se cae el arranque ni el resto del dominio congestion/).
try:
    _prediction_day_source = PredictionDaySource()
    _prediction_day_source.load()
    _prediction_adapter = load_gru_adapter(_PREDICTION_CKPT)
    init_prediction_service(
        CongestionPredictionService(_prediction_day_source, _prediction_adapter)
    )
    logger.info(
        "Predicción de congestión lista (slice seed051 day_idx=%s + %s)",
        _prediction_day_source.day_idx, _PREDICTION_CKPT,
    )
except Exception as e:  # noqa: BLE001 - boot tolerante, igual que el GRU 4-way
    logger.error("Predicción de congestión NO disponible (GET /congestion/prediction → 503): %s", e)

_control_settings = ControlSettings()
_control_engine = AdaptiveEngine(
    webster=WebsterCalculator(
        min_cycle=_control_settings.webster.min_cycle,
        max_cycle=_control_settings.webster.max_cycle,
    ),
    max_pressure=MaxPressureController(
        default_cycle=_control_settings.max_pressure.default_cycle
    ),
    mtc=MTCRestrictionApplier(
        constants=MTCConstants(
            min_green=_control_settings.mtc.min_green,
            max_green=_control_settings.mtc.max_green,
            min_yellow=_control_settings.mtc.min_yellow,
            all_red=_control_settings.mtc.all_red,
            min_pedestrian=_control_settings.mtc.min_pedestrian,
        ),
        max_cycle=_control_settings.mtc.max_cycle,
    ),
    peak_threshold=_control_settings.adaptive.peak_threshold,
)
init_engine(_control_engine)
# HU-05: inicializa el singleton del broadcaster SSE. Sin esto,
# Depends(get_broadcaster) en GET /control/active-state/{node_id}/stream y
# en POST /control/__internal/activate lanza RuntimeError → 500. Es un
# singleton in-memory per-proceso (DHU-021); single-worker uvicorn cubre el
# alcance del MVP.
init_broadcaster()
# TTH-12: broadcaster propio del dominio de congestión (canal de RED, in-memory
# per-proceso). Sin esto, Depends(get_congestion_broadcaster) en
# GET /congestion/state/stream lanza RuntimeError → 500.
init_congestion_broadcaster()

app.include_router(prediction_router)
app.include_router(control_router)
app.include_router(auth_router)
app.include_router(congestion_router)
app.include_router(corridors_router)


def _camera_display_name(cam: CameraDB) -> str:
    """Nombre legible de la cámara desde ``intersection_id`` (snake_case → Title Case).

    Cámara sin intersección mapeada (DEUDA-CAM-GEO) → "Desconocida".
    """
    if not cam.intersection_id:
        return "Desconocida"
    return " ".join(word.capitalize() for word in cam.intersection_id.split("_"))


@app.get("/api/cameras")
def get_cameras(db: Session = Depends(get_db)):
    """Lista pura de cámaras (CameraDB) — NO toca Waze.

    Carril liviano: id/name/stream_url + lat/lng, sin congestión. Lo consume el detalle
    de cámara (``CameraDetailView``), que no usa ``status`` y no debe pagar el agregado
    Waze. Para el color de congestión del dashboard está ``/api/intersections``.
    """
    return [
        {
            "id": cam.camera_id,
            "name": _camera_display_name(cam),
            "lat": cam.lat,
            "lng": cam.lon,
            "stream_url": cam.stream_url,  # B0: HLS de Claro (nominal, DEUDA-CAM-GEO)
        }
        for cam in db.query(CameraDB).all()
    ]


@app.get("/api/intersections")
def get_intersections(db: Session = Depends(get_db)):
    """Lista de cámaras/intersecciones con su congestión REAL por intersección (B2).

    ``status`` deja de ser placeholder: sale del nivel Waze agregado por intersección
    (MAX de las aristas de la intersección, último snapshot —
    ``WazeJamsRepo.max_level_by_intersection``) mapeado a fluid|moderate|critical. Cámara
    sin intersección mapeada (DEUDA-CAM-GEO) o intersección sin dato Waze → fluid.
    ``speed``/``flow`` siguen en 0: son señal de VISIÓN (edge), el SSE del front los
    completa cuando corre YOLO — la congestión de red NO los deriva.
    """
    cameras = db.query(CameraDB).all()
    level_by_intersection = WazeJamsRepo(db).max_level_by_intersection()
    results = []
    for cam in cameras:
        level = level_by_intersection.get(cam.intersection_id)
        results.append({
            "id": cam.camera_id,
            "name": _camera_display_name(cam),
            "lat": cam.lat,
            "lng": cam.lon,
            "stream_url": cam.stream_url,  # B0: HLS de Claro (nominal, DEUDA-CAM-GEO)
            "speed": 0,
            "flow": 0,
            "status": congestion_status_from_level(level),
        })
    return results


@app.get("/api/health", dependencies=[Depends(require_role(Role.ADMIN))])
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"name": "CerebroVial Core API", "version": "0.1.0"}
