"""
HTTP boundary for the adaptive control engine.

The Pydantic schemas live in this folder; conversion to/from the engine's
internal dataclasses happens here so that ``application/`` stays free of
FastAPI imports.

Write-path (TTH-10, CT-10.9): every successful recommendation persists a row
in ``motor_decisions`` within the request transaction. The opaque
``intersection_id`` from the payload is resolved to a real
``graph_nodes.node_id`` before the engine runs (DHU-021 V1); unknown
intersections fail-fast with HTTP 422 ``unknown_intersection``.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cerebrovial_shared.database import get_db

from ...application.adaptive_engine import (
    AdaptiveEngine,
    IntersectionStateDC,
    PhaseFlowDC,
    RecommendationDC,
)
from ...application.webster import WebsterInfeasible
from ...infrastructure import MotorDecisionsRepo, resolve_node_id
from .schemas import (
    ControlRecommendation,
    ErrorDetail,
    IntersectionState,
    PhaseTimings,
    RecommendResponse,
)

router = APIRouter(prefix="/control", tags=["control"])

_engine_instance: Optional[AdaptiveEngine] = None


def get_engine() -> AdaptiveEngine:
    if _engine_instance is None:
        raise HTTPException(status_code=503, detail="Control engine not initialized")
    return _engine_instance


def init_engine(engine: AdaptiveEngine) -> None:
    global _engine_instance
    _engine_instance = engine


def _to_dataclass(state: IntersectionState) -> IntersectionStateDC:
    return IntersectionStateDC(
        intersection_id=state.intersection_id,
        timestamp=state.timestamp,
        lost_time=state.lost_time,
        phases=[
            PhaseFlowDC(
                phase_id=p.phase_id,
                flow=p.flow,
                saturation_flow=p.saturation_flow,
                queue=p.queue,
                has_pedestrian=p.has_pedestrian,
            )
            for p in state.phases
        ],
    )


def _to_pydantic(rec: RecommendationDC) -> ControlRecommendation:
    return ControlRecommendation(
        intersection_id=rec.intersection_id,
        mode=rec.mode,
        cycle_seconds=rec.cycle_seconds,
        phase_timings=[
            PhaseTimings(
                phase_id=t.phase_id,
                green=t.green,
                yellow=t.yellow,
                all_red=t.all_red,
            )
            for t in rec.phase_timings
        ],
        next_phase=rec.next_phase,
        reasoning=rec.reasoning,
        adjustments=rec.adjustments,
    )


def _build_inputs_snapshot(state: IntersectionState) -> dict:
    return {
        "lost_time": state.lost_time,
        "phases": [
            {
                "phase_id": p.phase_id,
                "flow": p.flow,
                "saturation_flow": p.saturation_flow,
                "queue": p.queue,
                "has_pedestrian": p.has_pedestrian,
            }
            for p in state.phases
        ],
    }


@router.post("/recommend", response_model=RecommendResponse)
def recommend(
    state: IntersectionState,
    engine: AdaptiveEngine = Depends(get_engine),
    db: Session = Depends(get_db),
) -> RecommendResponse:
    # Resolve intersection_id -> node_id BEFORE running the engine (DHU-021 V1).
    node_id = resolve_node_id(db, state.intersection_id)
    if node_id is None:
        raise HTTPException(
            status_code=422,
            detail=ErrorDetail(
                code="unknown_intersection",
                message=f"intersection_id {state.intersection_id!r} not found in graph_nodes",
            ).model_dump(),
        )

    try:
        recommendation = engine.recommend(_to_dataclass(state))
    except WebsterInfeasible as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorDetail(code="webster_infeasible", message=str(exc)).model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorDetail(code="invalid_state", message=str(exc)).model_dump(),
        )

    # Persist the decision within the request transaction. Any failure here
    # rolls back and propagates a 500; the calculated recommendation is NOT
    # returned partially.
    try:
        MotorDecisionsRepo(db).insert(
            node_id=node_id,
            mode=recommendation.mode,
            cycle_seconds=recommendation.cycle_seconds,
            flow_total=recommendation.flow_total,
            y_load_factor=recommendation.y_load_factor,
            next_phase=recommendation.next_phase,
            reasoning=recommendation.reasoning,
            phase_timings=[
                {
                    "phase_id": t.phase_id,
                    "green": t.green,
                    "yellow": t.yellow,
                    "all_red": t.all_red,
                }
                for t in recommendation.phase_timings
            ],
            adjustments=list(recommendation.adjustments),
            inputs_snapshot=_build_inputs_snapshot(state),
        )
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise

    return RecommendResponse(data=_to_pydantic(recommendation))


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    """Liveness + readiness for the adaptive engine module (CT-10.13 partial,
    only the health endpoint — the TTH-04 cascade is R2).

    No auth: this endpoint is for orchestrator probes. The admin-protected
    ``/api/health`` is a different surface and stays untouched."""
    if _engine_instance is None:
        raise HTTPException(
            status_code=503,
            detail={"engine": "not_ready", "db": "unknown"},
        )
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail={"engine": "ready", "db": "unreachable"},
        )
    return {"status": "ok", "engine": "ready", "db": "ok"}
