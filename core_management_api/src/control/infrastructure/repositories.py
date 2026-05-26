"""
Persistence repositories for the adaptive control engine (TTH-10, CT-10.9).

`MotorDecisionsRepo` is append-only by construction: it exposes only
``insert(...)``. The absence of ``update`` and ``delete`` is enforced as an
invariant of the application layer (data-model.md §2.1), not as a DB-side
constraint.

`EngineActiveStateRepo` is mutable (one row per node_id). ``activate(...)``
is an upsert: insert if the node has no active strategy yet, otherwise
update the pointer and the ``activated_at`` timestamp.

``resolve_node_id`` is the entry-point for DHU-021 V1: the opaque
``intersection_id`` from the request payload is mapped to a real
``graph_nodes.node_id`` here, before the engine runs.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from cerebrovial_shared.database.models import (
    EngineActiveStateDB,
    GraphNodeDB,
    MotorDecisionDB,
)


def resolve_node_id(session: Session, intersection_id: str) -> Optional[str]:
    """Return the node_id if intersection_id matches a row in graph_nodes,
    otherwise None. The write-path uses this to fail-fast with HTTP 422 before
    the engine is even invoked (DHU-021 V1).

    Selects only ``node_id`` (not the full row) to avoid GeoAlchemy2 loading
    the PostGIS ``geom`` column — relevant in tests that use a stripped
    SQLite schema and irrelevant to existence-check semantics.
    """
    return session.scalar(
        select(GraphNodeDB.node_id).where(GraphNodeDB.node_id == intersection_id)
    )


class MotorDecisionsRepo:
    """Append-only repository for motor_decisions.

    Intentionally exposes no `update` or `delete` method (CT-10.9.7,
    data-model.md §2.1: append-only invariant).
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def insert(
        self,
        *,
        node_id: str,
        mode: str,
        cycle_seconds: float,
        flow_total: float,
        y_load_factor: Optional[float],
        next_phase: Optional[str],
        reasoning: str,
        phase_timings: list[dict],
        adjustments: list[str],
        inputs_snapshot: dict,
    ) -> str:
        row = MotorDecisionDB(
            node_id=node_id,
            mode=mode,
            cycle_seconds=cycle_seconds,
            flow_total=flow_total,
            y_load_factor=y_load_factor,
            next_phase=next_phase,
            reasoning=reasoning,
            phase_timings=phase_timings,
            adjustments=adjustments,
            inputs_snapshot=inputs_snapshot,
        )
        self.session.add(row)
        self.session.flush()
        return row.decision_id


class EngineActiveStateRepo:
    """Mutable repository for engine_active_state (one row per node_id).

    Not wired to any endpoint in TTH-10. The activation event belongs to
    HU-05/HU-07 (operator promotes a decision to active).
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def activate(
        self,
        *,
        node_id: str,
        decision_id: str,
        activated_by: Optional[str] = None,
    ) -> None:
        existing = self.session.get(EngineActiveStateDB, node_id)
        now = datetime.utcnow()
        if existing is None:
            row = EngineActiveStateDB(
                node_id=node_id,
                active_decision_id=decision_id,
                activated_at=now,
                activated_by=activated_by,
            )
            self.session.add(row)
        else:
            existing.active_decision_id = decision_id
            existing.activated_at = now
            existing.activated_by = activated_by
        self.session.flush()
