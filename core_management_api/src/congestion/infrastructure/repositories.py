"""Repositorio de ``waze_jams`` (TTH-12 Fase 2, CT-12.5).

Encapsula la escritura (batch para la pre-siembra; single para el replay en vivo)
y la lectura (último estado por arista — el read-path del feed) de la hypertable
``waze_jams``. Hasta esta fase la tabla estaba HUÉRFANA (sin repo/endpoint/ingestor).

Decisiones del esquema (verificadas en Fase 2 Paso 1):
- ``congestion_level`` es INTEGER sin CHECK: el rango efectivo es **0-5** (D-009;
  0 = flujo libre). El "1-5" del DATA_MODEL heredado es obsoleto — deuda de
  saneamiento documental, no de esta TTH.
- ``geom`` es nullable: se inserta SIN geom (8 columnas) y luego se puebla con un
  UPDATE-join contra ``graph_edges`` (``populate_geom_from_edges``), porque la
  geometría es constante por arista — no viaja fila por fila desde Python.
- ``event_uuid`` es determinista (uuid5 en el adaptador), así que la pre-siembra es
  idempotente vía ``ON CONFLICT DO NOTHING`` sobre la PK (event_uuid, snapshot_timestamp).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import json

from sqlalchemy import and_, func, select, text
from sqlalchemy.orm import Session

from cerebrovial_shared.database.models import WazeJamDB

from ..application.feed import EdgeCongestion

# Columnas escritas explícitamente (geom queda NULL y se puebla por UPDATE-join).
_COLS = (
    "event_uuid",
    "snapshot_timestamp",
    "edge_id",
    "speed_mps",
    "delay_seconds",
    "congestion_level",
    "jam_length_m",
    "road_type",
)


@dataclass(frozen=True)
class WazeJamRow:
    """Fila lista para persistir en ``waze_jams`` (sin geom — se puebla aparte)."""
    event_uuid: str
    snapshot_timestamp: datetime
    edge_id: str
    speed_mps: float
    delay_seconds: int
    congestion_level: int
    jam_length_m: int
    road_type: int

    def as_tuple(self) -> tuple:
        return (
            self.event_uuid,
            self.snapshot_timestamp,
            self.edge_id,
            self.speed_mps,
            self.delay_seconds,
            self.congestion_level,
            self.jam_length_m,
            self.road_type,
        )


class WazeJamsRepo:
    """Lectura/escritura de ``waze_jams``."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # --- escritura ---

    def insert_one(self, row: WazeJamRow) -> None:
        """Inserta una fila (replay en vivo). Idempotente por la PK."""
        cols = ", ".join(_COLS)
        binds = ", ".join(f":{c}" for c in _COLS)
        self.session.execute(
            text(f"INSERT INTO waze_jams ({cols}) VALUES ({binds}) "
                 f"ON CONFLICT (event_uuid, snapshot_timestamp) DO NOTHING"),
            dict(zip(_COLS, row.as_tuple())),
        )

    def batch_insert(self, rows: Iterable[WazeJamRow], chunk: int = 10_000) -> int:
        """Inserta en lotes (pre-siembra). Idempotente (ON CONFLICT DO NOTHING).

        En PostgreSQL usa ``psycopg2.extras.execute_values`` (rápido); en otros
        dialectos (SQLite de los tests) cae a executemany por chunk. Devuelve el
        número de filas PROCESADAS (no necesariamente insertadas, por el conflicto).
        """
        rows = list(rows)
        if not rows:
            return 0
        cols = ", ".join(_COLS)
        dialect = self.session.get_bind().dialect.name

        if dialect == "postgresql":
            from psycopg2.extras import execute_values
            raw = self.session.connection().connection  # DBAPI psycopg2 conn
            sql = (f"INSERT INTO waze_jams ({cols}) VALUES %s "
                   f"ON CONFLICT (event_uuid, snapshot_timestamp) DO NOTHING")
            with raw.cursor() as cur:
                execute_values(cur, sql, [r.as_tuple() for r in rows], page_size=chunk)
        else:
            binds = ", ".join(f":{c}" for c in _COLS)
            sql = text(f"INSERT INTO waze_jams ({cols}) VALUES ({binds}) "
                       f"ON CONFLICT (event_uuid, snapshot_timestamp) DO NOTHING")
            conn = self.session.connection()
            for i in range(0, len(rows), chunk):
                conn.execute(sql, [dict(zip(_COLS, r.as_tuple())) for r in rows[i:i + chunk]])
        return len(rows)

    def populate_geom_from_edges(self) -> int:
        """Puebla ``geom`` de cada fila con la geometría de su arista (JOIN por edge_id).

        Set-based: la geometría es constante por arista, así que un solo UPDATE-join
        cubre las 540k filas vía las 375 de ``graph_edges``. Solo PostGIS (UPDATE FROM).
        Devuelve el nº de filas actualizadas.
        """
        if self.session.get_bind().dialect.name != "postgresql":
            return 0  # SQLite de tests no tiene geom ni UPDATE-FROM
        res = self.session.execute(text(
            "UPDATE waze_jams w SET geom = ge.geom "
            "FROM graph_edges ge WHERE w.edge_id = ge.edge_id AND w.geom IS NULL"
        ))
        return res.rowcount

    # --- lectura (read-path del feed) ---

    def count(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(WazeJamDB)))

    def latest_per_edge(self) -> list[EdgeCongestion]:
        """Último estado de congestión conocido por arista (CT-12.6 read-path).

        Portátil (subconsulta max por arista + join), no usa ``DISTINCT ON`` para
        correr igual en PostgreSQL y en el SQLite de los tests. Selecciona solo
        columnas escalares (no toca ``geom``).
        """
        latest = (
            select(
                WazeJamDB.edge_id.label("edge_id"),
                func.max(WazeJamDB.snapshot_timestamp).label("mx"),
            )
            .group_by(WazeJamDB.edge_id)
            .subquery()
        )
        q = (
            select(
                WazeJamDB.edge_id,
                WazeJamDB.congestion_level,
                WazeJamDB.snapshot_timestamp,
            )
            .join(
                latest,
                and_(
                    WazeJamDB.edge_id == latest.c.edge_id,
                    WazeJamDB.snapshot_timestamp == latest.c.mx,
                ),
            )
        )
        return [
            EdgeCongestion(edge_id=r.edge_id, congestion_level=r.congestion_level,
                           snapshot_timestamp=r.snapshot_timestamp)
            for r in self.session.execute(q)
        ]


class NetworkGeometryRepo:
    """Lectura de la geometría de la red desde ``graph_edges`` (CT-12.2).

    Sirve las 375 aristas como features GeoJSON (geometry LineString en 4326,
    orden [lon, lat]). Usa ``ST_AsGeoJSON`` de PostGIS — solo PostgreSQL/PostGIS.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def network_features(self) -> list[dict]:
        """Devuelve una lista de Features GeoJSON, una por arista.

        ``geometry`` ya viene como dict GeoJSON (ST_AsGeoJSON → json). Las
        propiedades llevan ``edge_id`` + metadatos de la arista.
        """
        rows = self.session.execute(text(
            "SELECT edge_id, source_node, target_node, distance_m, lanes, "
            "ST_AsGeoJSON(geom) AS geojson "
            "FROM graph_edges WHERE geom IS NOT NULL ORDER BY edge_id"
        ))
        features = []
        for r in rows:
            features.append({
                "type": "Feature",
                "geometry": json.loads(r.geojson),
                "properties": {
                    "edge_id": r.edge_id,
                    "source_node": r.source_node,
                    "target_node": r.target_node,
                    "distance_m": r.distance_m,
                    "lanes": r.lanes,
                },
            })
        return features
