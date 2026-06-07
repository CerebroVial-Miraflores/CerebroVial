"""Schemas Pydantic del boundary HTTP de corredores (Fase B-2).

El request trae la cadena de aristas + las respuestas de ``flowSegmentData`` que el FRONT ya
obtuvo de TomTom. ``coordinates`` y ``openlr`` de cada segmento son INPUT EFÍMERO (ToS 11.6.1):
se usan para el matching y se descartan; sólo el ``openlr`` ganador se persiste, NUNCA la
geometría. El response devuelve el mapping (edge → openlr|null) para que el front sepa qué
aristas quedaron sin cobertura.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class EdgeRef(BaseModel):
    """Una arista del corredor con su posición en la cadena."""
    edge_id: str
    sequence: int = Field(ge=0)


class TomTomSegmentInput(BaseModel):
    """Segmento ``flowSegmentData`` obtenido por el front. EFÍMERO (no se persiste su geometría)."""
    openlr: str = Field(min_length=1)
    coordinates: list[tuple[float, float]] = Field(
        min_length=2, description="Polilínea [[lon, lat], ...] en orden de flujo"
    )


class CreateCorridorRequest(BaseModel):
    """Crea un corredor y matchea cada arista contra los segmentos TomTom recibidos."""
    name: str = Field(min_length=1)
    edges: list[EdgeRef] = Field(min_length=1)
    segments: list[TomTomSegmentInput] = Field(
        default_factory=list,
        description="Respuestas de flowSegmentData del front (geometría efímera + openlr)",
    )


class CorridorEdgeMapping(BaseModel):
    """Resultado del matching para una arista. ``tomtom_openlr=None`` = sin cobertura TomTom."""
    edge_id: str
    sequence: int
    tomtom_openlr: str | None


class CreateCorridorResponse(BaseModel):
    """Corredor persistido + el mapping resultante (edge → openlr|null)."""
    corridor_id: str
    count: int
    edges: list[CorridorEdgeMapping]
