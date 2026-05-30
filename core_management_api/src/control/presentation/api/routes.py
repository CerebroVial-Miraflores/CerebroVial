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
import asyncio
import json
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from cerebrovial_shared.database import get_db

from src.auth.domain import Role
from src.auth.presentation.api.dependencies import require_role

from ...application.adaptive_engine import (
    AdaptiveEngine,
    IntersectionStateDC,
    PhaseFlowDC,
    RecommendationDC,
)
from ...application.max_pressure import DownstreamLinkDC
from ...application.webster import WebsterInfeasible
from ...infrastructure import (
    ActiveStateBroadcaster,
    EngineActiveStateRepo,
    MotorDecisionsRepo,
    get_broadcaster,
    resolve_node_id,
)
from .schemas import (
    ActiveStateResponse,
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
                downstream=(
                    [DownstreamLinkDC(queue=d.queue, turn_ratio=d.turn_ratio) for d in p.downstream]
                    if p.downstream
                    else None
                ),
            )
            for p in state.phases
        ],
    )


def _to_pydantic(
    rec: RecommendationDC, decision_id: Optional[str] = None
) -> ControlRecommendation:
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
        decision_id=decision_id,
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
                "downstream": (
                    [{"queue": d.queue, "turn_ratio": d.turn_ratio} for d in p.downstream]
                    if p.downstream
                    else None
                ),
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
        decision_id = MotorDecisionsRepo(db).insert(
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

    return RecommendResponse(data=_to_pydantic(recommendation, decision_id))


@router.get(
    "/active-state/{node_id}",
    response_model=ActiveStateResponse,
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
def get_active_state(
    node_id: str,
    db: Session = Depends(get_db),
) -> ActiveStateResponse:
    """Read-only view of the currently active strategy for ``node_id`` (HU-05).

    Returns ``404 no_active_state`` if the node has no row in
    ``engine_active_state``. This is the contract that the frontend
    discriminates from a network error or a generic 404 to decide whether
    to show "no hay estrategia activa todavía" vs trigger DHU-005 Caso B
    (CA-05.4 — stateful resilience handled in the UI).
    """
    row = EngineActiveStateRepo(db).get_active_state_for_node(node_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code="no_active_state",
                message=f"No active strategy for node {node_id!r}.",
            ).model_dump(),
        )
    return ActiveStateResponse(
        node_id=row.node_id,
        strategy_mode=row.strategy_mode,
        cycle_seconds=row.cycle_seconds,
        phase_timings=[PhaseTimings(**pt) for pt in row.phase_timings],
        decided_at=row.decided_at,
        activated_at=row.activated_at,
        activated_by=row.activated_by,
    )


@router.get(
    "/active-state/{node_id}/stream",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def stream_active_state(
    node_id: str,
    broadcaster: ActiveStateBroadcaster = Depends(get_broadcaster),
) -> EventSourceResponse:
    """Server-Sent Events stream of ``active-state-changed`` for ``node_id``.

    Emits a minimal change signal: ``{type, node_id, decision_id, activated_at}``.
    The client is expected to RE-READ ``GET /control/active-state/{node_id}``
    for authoritative state — single source of truth lives in the database
    (DHU-021 #15). The stream is the wake-up signal; REST is the read path.

    Lifecycle: subscribe → loop on the queue → unsubscribe in ``finally``.
    The ``finally`` runs both on normal generator close and on
    ``asyncio.CancelledError`` (client disconnect), so subscriber queues are
    never leaked into the broadcaster's dict.
    """
    queue = await broadcaster.subscribe(node_id)

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                yield {
                    "event": "active-state-changed",
                    "data": json.dumps(event),
                }
        except asyncio.CancelledError:
            # Client disconnected — re-raise so the EventSourceResponse machinery
            # tears down cleanly. The ``finally`` below still runs.
            raise
        finally:
            await broadcaster.unsubscribe(node_id, queue)

    return EventSourceResponse(event_generator())


# --- Dev-only test activator (HU-05 verification, Fase 3) ----------------
#
# This endpoint is the intra-process trigger that lets manual e2e and BDD tests
# observe the SSE stream notifying a real change. It is registered IFF
# ``ENABLE_TEST_ACTIVATOR == "true"`` is set when ``setup_test_activator()``
# is invoked — typically once at app startup from main.py, and once from the
# HU-05 BDD conftest.
#
# Without the env var the path is NOT in the routing table: a GET or POST to
# it returns 404 (not 405) — confirming the frontier between productive
# surface and test scaffolding. The productive activation endpoint (operator-
# or motor-driven) is HU-07/futuro.


class _TestActivateBody(BaseModel):
    node_id: str
    decision_id: str
    activated_by: Optional[str] = None


def _is_test_activator_enabled() -> bool:
    """Gate for the dev-only activator endpoint. Read each call so tests can
    flip the env var via monkeypatch without reloading modules."""
    return os.environ.get("ENABLE_TEST_ACTIVATOR") == "true"


def setup_test_activator() -> bool:
    """Idempotently register ``POST /control/__internal/activate`` IF the env
    gate is on. Returns True if the route is now present on the router (either
    newly added or already there), False if the gate is off.

    Safe to call multiple times: subsequent calls are no-ops when the route is
    already registered."""
    if not _is_test_activator_enabled():
        return False
    already = any(
        getattr(r, "path", "") == "/control/__internal/activate"
        for r in router.routes
    )
    if already:
        return True

    @router.post(
        "/__internal/activate",
        dependencies=[Depends(require_role(Role.ADMIN))],
    )
    async def _internal_activate(
        body: _TestActivateBody,
        db: Session = Depends(get_db),
        broadcaster: ActiveStateBroadcaster = Depends(get_broadcaster),
    ) -> dict:
        """Activate ``decision_id`` for ``node_id`` and notify SSE subscribers.

        Order is load-bearing: ``session.commit()`` BEFORE ``broadcaster.publish``.
        Otherwise the SSE event would tell clients "go re-read" while the row
        is still uncommitted — they'd read stale state. If commit fails, the
        publish never happens, so no phantom events leak.
        """
        repo = EngineActiveStateRepo(db)
        repo.activate(
            node_id=body.node_id,
            decision_id=body.decision_id,
            activated_by=body.activated_by,
        )
        db.commit()  # COMMIT first.
        await broadcaster.publish(  # PUBLISH after commit.
            body.node_id,
            {
                "type": "active-state-changed",
                "node_id": body.node_id,
                "decision_id": body.decision_id,
                "activated_at": datetime.utcnow().isoformat() + "Z",
            },
        )
        return {"ok": True}

    return True


# Register at module import based on env var snapshot. main.py + BDD conftest
# may call setup_test_activator() again (idempotent) to cover the case where
# the env var is set AFTER this module was first imported.
setup_test_activator()


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
