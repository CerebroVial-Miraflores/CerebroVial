"""Schemas Pydantic del boundary HTTP de congestión (TTH-12 Fase 3).

Mantiene la presentación libre de ORM/PostGIS: la geometría sale como GeoJSON
(geometry LineString, 4326, orden [lon, lat]); el feed como nivel 0-5 + timestamp.
"""
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class GeometryFeature(BaseModel):
    """Feature GeoJSON de una arista (CT-12.2)."""
    type: str = "Feature"
    geometry: dict   # GeoJSON LineString {type, coordinates:[[lon,lat],...]}
    properties: dict  # {edge_id, source_node, target_node, distance_m, lanes}


class GeometryFeatureCollection(BaseModel):
    """FeatureCollection GeoJSON con la red completa (universo LCC de aristas)."""
    type: str = "FeatureCollection"
    features: list[GeometryFeature]
    count: int


class EdgeCongestionState(BaseModel):
    """Estado de congestión actual de una arista (CT-12.6 / CT-12.7)."""
    edge_id: str
    congestion_level: int = Field(ge=0, le=5)  # 0-5 (D-009)
    snapshot_timestamp: datetime               # marca de tiempo (robustez CT-12.7)


class CongestionStateResponse(BaseModel):
    """Estado de congestión por arista de toda la red (último snapshot conocido)."""
    edges: list[EdgeCongestionState]
    count: int


class EdgeCongestionSeries(BaseModel):
    """Serie de niveles 0-5 de UNA arista a lo largo del día (Formato B, TTH-13).

    ``levels[i]`` es el nivel en el instante ``t0 + i * step_s``; el consumidor
    (HU-23) indexa en O(1). No se valida cada entero por costo (universo LCC × timesteps); el
    repositorio sólo lee niveles 0-5 ya persistidos (D-009).
    """
    edge_id: str
    levels: list[int]


class CongestionSeriesResponse(BaseModel):
    """Serie de congestión del día por arista de toda la red (TTH-13 / CT-13.2).

    Día sin datos → ``count: 0``, ``edges: []`` y campos temporales en ``null``
    (señal de ausencia, CA-23.7): el consumidor deshabilita el control temporal.
    Por eso ``t0``/``step_s``/``coverage_end`` son ``Optional`` — la forma del
    contrato no cambia según haya datos o no.
    """
    day: date
    t0: datetime | None
    step_s: int | None
    coverage_end: datetime | None
    count: int
    edges: list[EdgeCongestionSeries]


class PredictedEdgeCongestion(BaseModel):
    """Predicción de congestión de UNA arista: niveles 0-4 para los pasos t+1..t+horizon.

    ``levels[i]`` es el nivel predicho en el paso ``base_timestep + 1 + i`` (futuro); el
    frontend (slider HU-23) elige cuál mostrar. Niveles 0-4: el modelo predice ``timeLoss``
    continuo y se presenta vía cortes; el nivel 5 ("cerrado") es observable, no predicho.
    """
    edge_id: str
    levels: list[int]


class CongestionPredictionResponse(BaseModel):
    """Predicción GRU baseline por arista de toda la red, a ``horizon`` pasos (Fase 3).

    ``base_timestep`` es el corte ``t`` (minuto del día, 0..1439) sobre el que se armó la
    ventana ``t-29..t``; la predicción cubre ``t+1..t+horizon``. ``edges`` viene en orden
    ``node_index`` (el mismo del training). ``source`` documenta el día-fuente (seed051);
    ``source_date`` es su fecha-calendario, para que el consumidor pida la base observada
    coherente vía ``getSeries(source_date)`` (HU-23, modo ``pinned``).
    """
    base_timestep: int
    horizon: int
    source: str
    source_date: date  # fecha-calendario del día-fuente (seed051); deuda hermana de ``source`` — ver routes.py
    count: int
    edges: list[PredictedEdgeCongestion]
