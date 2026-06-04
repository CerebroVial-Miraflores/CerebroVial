"""Boundary HTTP del dominio de congestión por arista (TTH-12 Fase 3, cierre).

Tres rutas, todas protegidas con ``require_role(OPERATOR, ADMIN)`` (patrón TTH-01,
igual que ``control/``):

- ``GET /congestion/geometry``      — geometría de la red (GeoJSON, CT-12.2). Estática.
- ``GET /congestion/state``         — estado de congestión actual por arista, último
                                      snapshot conocido desde ``waze_jams`` (CT-12.6/12.7).
- ``GET /congestion/state/stream``  — canal SSE de RED: wake-up "hay nueva congestión"
                                      SIN payload; el cliente re-lee ``/congestion/state``
                                      (patrón SSE-wake/REST-read, DHU-021 #15).

CT-12.7 (robustez): ``/congestion/state`` NUNCA falla si la fuente (adaptador) está
detenida — devuelve el último estado conocido con su ``snapshot_timestamp``. La lógica
de "desactualizado" es del consumidor (HU-22, CA-22.4), no de aquí.
"""
import asyncio
import json
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from cerebrovial_shared.database import get_db

from src.auth.domain import Role
from src.auth.presentation.api.dependencies import require_role

from ...application.congestion_prediction_service import (
    CongestionPredictionService,
    PredictionUnavailableError,
)
from ...infrastructure import (
    NetworkCongestionBroadcaster,
    NetworkGeometryRepo,
    WazeJamsRepo,
    get_congestion_broadcaster,
)
from .schemas import (
    CongestionPredictionResponse,
    CongestionSeriesResponse,
    CongestionStateResponse,
    EdgeCongestionSeries,
    EdgeCongestionState,
    GeometryFeatureCollection,
    PredictedEdgeCongestion,
)

router = APIRouter(prefix="/congestion", tags=["congestion"])

# Horizonte de la predicción servida (pasos de 60 s); = horizonte del modelo baseline.
_PREDICTION_HORIZON = 30
# Corte mínimo: la ventana lookback=30 termina en t y empieza en t-29 ⇒ t >= 29.
_MIN_T = 29
_MAX_T = 1439  # último timestep del día (minuto del día 0..1439)
_PREDICTION_SOURCE = "seed051 (day_idx=9)"
# DEUDA HERMANA del ``_PREDICTION_SOURCE`` hardcodeado: NO es la fecha resuelta del día real
# del slice. El slice (day_seed051_tensor.npz) NO lleva fecha; esta constante es coherente
# POR CONVENCIÓN con el mapeo seed051 ↔ 2026-06-08 del calendario de siembra
# (scripts/seed_congestion_calendar.py), NO derivada de él programáticamente. No leer como
# "fecha resuelta correctamente": se paga junto con la deuda del source hardcodeado en el
# cierre del STGNN servido / endpoint de predicción real.
_PREDICTION_SOURCE_DATE = date(2026, 6, 8)

# --- DI: servicio de predicción de congestión (Fase 3) -----------------------------
# Singleton in-process, espejo del patrón de TTH-09 (init/get), pero en congestion/ con su
# infra vendorizada (GRU-only, sin tsl). Inicializado desde main.py al startup.
_prediction_service_instance: Optional[CongestionPredictionService] = None


def get_prediction_service() -> CongestionPredictionService:
    if _prediction_service_instance is None:
        raise HTTPException(
            status_code=503, detail="Congestion prediction service not initialized"
        )
    return _prediction_service_instance


def init_prediction_service(service: CongestionPredictionService) -> None:
    global _prediction_service_instance
    _prediction_service_instance = service


def _resolve_t(t_param: Optional[int], db: Session) -> int:
    """Resuelve el corte ``t`` (minuto del día 0..1439).

    Si viene ``t`` explícito (demo controlada) → se valida y se usa. Si no → se deriva del
    step actual del feed vivo: ``max(snapshot_timestamp)`` de ``waze_jams`` (la misma fuente
    que pinta el mapa), NO del reloj de pared. ``t = hora*60 + minuto`` (minuto del día),
    robusto al ``DAY_EPOCH`` con el que se haya fechado el replay.
    """
    if t_param is not None:
        if not (_MIN_T <= t_param <= _MAX_T):
            raise HTTPException(
                status_code=422,
                detail=f"t={t_param} fuera de rango [{_MIN_T}, {_MAX_T}] (minuto del día).",
            )
        return t_param

    ts = WazeJamsRepo(db).latest_timestamp()
    if ts is None:
        raise HTTPException(
            status_code=503,
            detail="No hay feed de congestión para anclar t (waze_jams vacío). "
            "Sembrá/replayá el día seed051 o pasá t explícito.",
        )
    t = ts.hour * 60 + ts.minute
    if not (_MIN_T <= t <= _MAX_T):
        raise HTTPException(
            status_code=409,
            detail=f"el feed está en t={t}; se requieren >= {_MIN_T} pasos de historia "
            "para la ventana (lookback=30). Esperá a que el replay avance.",
        )
    return t


@router.get(
    "/geometry",
    response_model=GeometryFeatureCollection,
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
def get_network_geometry(db: Session = Depends(get_db)) -> GeometryFeatureCollection:
    """Geometría de la red completa como GeoJSON FeatureCollection (CT-12.2)."""
    features = NetworkGeometryRepo(db).network_features()
    return GeometryFeatureCollection(features=features, count=len(features))


@router.get(
    "/state",
    response_model=CongestionStateResponse,
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
def get_congestion_state(db: Session = Depends(get_db)) -> CongestionStateResponse:
    """Estado de congestión actual por arista — último snapshot conocido (CT-12.6/12.7).

    Lee ``waze_jams`` (DISTINCT por arista, el más reciente). No depende de que el
    adaptador esté corriendo: si la fuente se detuvo, devuelve lo último persistido
    con su marca de tiempo (robustez CT-12.7).
    """
    states = WazeJamsRepo(db).latest_per_edge()
    edges = [
        EdgeCongestionState(
            edge_id=s.edge_id,
            congestion_level=s.congestion_level,
            snapshot_timestamp=s.snapshot_timestamp,
        )
        for s in states
    ]
    return CongestionStateResponse(edges=edges, count=len(edges))


@router.get(
    "/series",
    response_model=CongestionSeriesResponse,
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
def get_congestion_series(
    day: date = Query(..., description="Día a recorrer (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
) -> CongestionSeriesResponse:
    """Serie de congestión del día por arista — Formato B compacto (CT-13.2).

    Para cada arista, un array ``levels`` de niveles 0-5 muestreados a paso
    ``step_s`` desde ``t0``; el consumidor (HU-23) indexa ``levels[i]`` en O(1) al
    mover el slider. ``coverage_end`` acota el control temporal (CA-23.2). Día sin
    datos → ``count: 0``, ``edges: []`` y campos temporales en ``null`` (CA-23.7),
    sin error. La respuesta se sirve comprimida (GZipMiddleware global).
    """
    series = WazeJamsRepo(db).series_for_day(day)
    return CongestionSeriesResponse(
        day=series.day,
        t0=series.t0,
        step_s=series.step_s,
        coverage_end=series.coverage_end,
        count=len(series.edges),
        edges=[
            EdgeCongestionSeries(edge_id=e.edge_id, levels=e.levels)
            for e in series.edges
        ],
    )


@router.get(
    "/state/stream",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def stream_congestion_updates(
    broadcaster: NetworkCongestionBroadcaster = Depends(get_congestion_broadcaster),
) -> EventSourceResponse:
    """SSE de RED: emite ``congestion-updated`` (wake-up sin payload).

    El cliente RE-LEE ``GET /congestion/state`` para el estado autoritativo — fuente
    única de verdad en la BD (DHU-021 #15). El stream es solo la señal de despertar.
    Lifecycle: subscribe → loop en la cola → unsubscribe en ``finally`` (también en
    desconexión del cliente vía CancelledError).
    """
    queue = await broadcaster.subscribe()

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                yield {"event": "congestion-updated", "data": json.dumps(event)}
        except asyncio.CancelledError:
            raise
        finally:
            await broadcaster.unsubscribe(queue)

    return EventSourceResponse(event_generator())


@router.get(
    "/prediction",
    response_model=CongestionPredictionResponse,
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
def get_congestion_prediction(
    t: Optional[int] = Query(
        None,
        ge=_MIN_T,
        le=_MAX_T,
        description="Corte (minuto del día 0..1439). Si se omite, deriva del step actual "
        "del feed vivo (max snapshot_timestamp de waze_jams), no del reloj de pared.",
    ),
    service: CongestionPredictionService = Depends(get_prediction_service),
    db: Session = Depends(get_db),
) -> CongestionPredictionResponse:
    """Predicción GRU baseline por arista a 30 pasos (Fase 3) — alimenta el mapa.

    Corta la ventana ``t-29..t`` del día seed051 (= lunes 8 jun, día-fuente del mapa vivo),
    corre el GRU baseline vendorizado (``timeLoss`` continuo) y presenta niveles 0-4 por
    arista para los pasos ``t+1..t+30``. El frontend (slider HU-23) elige cuál mostrar.

    Coherencia: cuando ``t`` no se pasa, sale del MISMO step que ``GET /congestion/state``
    muestra (``max(snapshot_timestamp)`` de ``waze_jams``), garantizando que predicción y
    mapa vivo hablan del mismo instante del mismo día-fuente.
    """
    resolved_t = _resolve_t(t, db)
    try:
        preds = service.predict(resolved_t)
    except PredictionUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return CongestionPredictionResponse(
        base_timestep=resolved_t,
        horizon=_PREDICTION_HORIZON,
        source=_PREDICTION_SOURCE,
        source_date=_PREDICTION_SOURCE_DATE,
        count=len(preds),
        edges=[
            PredictedEdgeCongestion(edge_id=p.edge_id, levels=p.levels) for p in preds
        ],
    )
