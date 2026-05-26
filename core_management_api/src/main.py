import os
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from cerebrovial_shared.database.database import get_db
from cerebrovial_shared.database.models import CameraDB
from fastapi.middleware.cors import CORSMiddleware

from src.prediction.presentation.api.routes import router as prediction_router, init_predictor
from src.prediction.application.predictor import CongestionPredictor
from src.control.presentation.api.routes import router as control_router, init_engine
from src.control.application.adaptive_engine import AdaptiveEngine
from src.control.application.webster import WebsterCalculator
from src.control.application.max_pressure import MaxPressureController
from src.control.application.mtc_constraints import MTCConstants, MTCRestrictionApplier
from src.control.config import ControlSettings
from src.auth.domain import Role
from src.auth.presentation.api.dependencies import require_role
from src.auth.presentation.api.routes import auth_router

app = FastAPI(title="CerebroVial Core API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("models", exist_ok=True)
os.makedirs("data/traffic_logs", exist_ok=True)
_predictor = CongestionPredictor(model_dir="models", data_dir="data/traffic_logs")
init_predictor(_predictor)

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

app.include_router(prediction_router)
app.include_router(control_router)
app.include_router(auth_router)


@app.get("/api/intersections")
def get_intersections(db: Session = Depends(get_db)):
    """Obtiene la lista de cámaras/intersecciones activas desde la Base de Datos"""
    cameras = db.query(CameraDB).all()
    results = []
    for cam in cameras:
        name = " ".join([word.capitalize() for word in cam.node_id.split("_")]) if cam.node_id else "Desconocida"
        results.append({
            "id": cam.camera_id,
            "name": name,
            "lat": cam.lat,
            "lng": cam.lon,
            "speed": 0,
            "flow": 0,
            "status": "fluid"
        })
    return results


@app.get("/api/health", dependencies=[Depends(require_role(Role.ADMIN))])
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"name": "CerebroVial Core API", "version": "0.1.0"}
