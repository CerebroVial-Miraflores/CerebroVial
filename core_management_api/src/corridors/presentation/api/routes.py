"""Boundary HTTP del dominio de corredores TomTom (Fase B-2).

Un endpoint: ``POST /corridors``. Recibe la cadena de aristas + las respuestas de
``flowSegmentData`` del front, valida la cadena, corre el matching geométrico
(edge → OpenLR), persiste el corredor y devuelve el mapping. Protegido con
``require_role(OPERATOR, ADMIN)`` (molde TTH-01, igual que ``congestion/`` y ``control/``).

El backend NUNCA consulta TomTom; la geometría TomTom es input efímero (ToS 11.4 / 11.6.1).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cerebrovial_shared.database import get_db
from cerebrovial_shared.database.models import UserDB

from src.auth.domain import Role
from src.auth.presentation.api.dependencies import require_role

from ...application.matching import (
    CorridorEdgeInput,
    TomTomSegment,
    match_corridor,
    validate_continuity,
    validate_sequences,
)
from ...infrastructure.repositories import CorridorMatchingRepo
from .schemas import (
    CorridorEdgeMapping,
    CreateCorridorRequest,
    CreateCorridorResponse,
)

router = APIRouter(prefix="/corridors", tags=["corridors"])


def _created_by(user: UserDB) -> str | None:
    """Identidad del operador para ``corridors.created_by`` (sin FK, patrón engine_active_state)."""
    return getattr(user, "email", None) or getattr(user, "id", None)


@router.post("", response_model=CreateCorridorResponse, status_code=201)
def create_corridor(
    payload: CreateCorridorRequest,
    user: UserDB = Depends(require_role(Role.OPERATOR, Role.ADMIN)),
    db: Session = Depends(get_db),
) -> CreateCorridorResponse:
    """Crea un corredor y matchea sus aristas contra los segmentos TomTom recibidos.

    Valida (4xx, sin persistir cadena rota): aristas existentes en ``graph_edges``, ``sequence``
    contigua y sin duplicados, y continuidad de la cadena (``target_node`` == ``source_node``
    consecutivos). Luego matchea (buffer + overlap + sentido) y persiste transaccionalmente.
    """
    # 1. sequence contigua / sin duplicados (puro, antes de tocar la DB).
    err = validate_sequences([e.sequence for e in payload.edges])
    if err is not None:
        raise HTTPException(status_code=422, detail=err)

    # 2. existencia de las aristas en graph_edges (con sus extremos resueltos).
    repo = CorridorMatchingRepo(db)
    by_id = {row.edge_id: row for row in repo.fetch_edges([e.edge_id for e in payload.edges])}
    missing = [e.edge_id for e in payload.edges if e.edge_id not in by_id]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"aristas inexistentes en graph_edges: {missing}"
        )

    # 3. orden por sequence + continuidad de la cadena (puro).
    ordered = sorted(payload.edges, key=lambda e: e.sequence)
    ordered_rows = [by_id[e.edge_id] for e in ordered]
    err = validate_continuity(ordered_rows)
    if err is not None:
        raise HTTPException(status_code=422, detail=err)

    # 4. matching geométrico edge → OpenLR.
    edge_inputs = [
        CorridorEdgeInput(
            edge_id=e.edge_id,
            sequence=e.sequence,
            source_coord=(by_id[e.edge_id].src_lon, by_id[e.edge_id].src_lat),
            target_coord=(by_id[e.edge_id].tgt_lon, by_id[e.edge_id].tgt_lat),
        )
        for e in ordered
    ]
    segments = [
        TomTomSegment(openlr=s.openlr, coordinates=[(c[0], c[1]) for c in s.coordinates])
        for s in payload.segments
    ]
    mapping = match_corridor(repo, edge_inputs, segments)

    # 5. persistencia transaccional (o todo el corredor o nada).
    ordered_mapping = [(e.edge_id, e.sequence, mapping.get(e.edge_id)) for e in ordered]
    try:
        corridor_id = repo.persist(payload.name, _created_by(user), ordered_mapping)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise

    return CreateCorridorResponse(
        corridor_id=corridor_id,
        count=len(ordered_mapping),
        edges=[
            CorridorEdgeMapping(edge_id=eid, sequence=seq, tomtom_openlr=olr)
            for eid, seq, olr in ordered_mapping
        ],
    )
