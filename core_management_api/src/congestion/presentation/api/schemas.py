"""Schemas Pydantic del boundary HTTP de congestión (TTH-12 Fase 3).

Mantiene la presentación libre de ORM/PostGIS: la geometría sale como GeoJSON
(geometry LineString, 4326, orden [lon, lat]); el feed como nivel 0-5 + timestamp.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class GeometryFeature(BaseModel):
    """Feature GeoJSON de una arista (CT-12.2)."""
    type: str = "Feature"
    geometry: dict   # GeoJSON LineString {type, coordinates:[[lon,lat],...]}
    properties: dict  # {edge_id, source_node, target_node, distance_m, lanes}


class GeometryFeatureCollection(BaseModel):
    """FeatureCollection GeoJSON con la red completa (375 aristas)."""
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
