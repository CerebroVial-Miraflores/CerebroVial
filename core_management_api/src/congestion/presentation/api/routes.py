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

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from cerebrovial_shared.database import get_db

from src.auth.domain import Role
from src.auth.presentation.api.dependencies import require_role

from ...infrastructure import (
    NetworkCongestionBroadcaster,
    NetworkGeometryRepo,
    WazeJamsRepo,
    get_congestion_broadcaster,
)
from .schemas import (
    CongestionSeriesResponse,
    CongestionStateResponse,
    EdgeCongestionSeries,
    EdgeCongestionState,
    GeometryFeatureCollection,
)

router = APIRouter(prefix="/congestion", tags=["congestion"])


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
