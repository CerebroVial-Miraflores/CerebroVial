"""Persistencia y cálculo geométrico de corredores (Fase B-2).

Separa lo que corre en cualquier dialecto (lectura de la cadena por sus columnas
``source_node``/``target_node`` y los extremos vía ``graph_nodes``; persistencia de IDs) de
lo que SÓLO corre en PostGIS (el overlap geométrico, con guard de dialecto — molde
``WazeJamsRepo.populate_geom_from_edges``).

ToS TomTom 11.6.1: la geometría de TomTom entra como WKT en un BIND PARAM efímero del query
de overlap y se descarta. NUNCA aparece en un INSERT/UPDATE ni en columna persistida. Lo
único que se escribe es ``corridors`` + ``corridor_edges(edge_id, sequence, tomtom_openlr)``.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from cerebrovial_shared.database.models import CorridorDB, CorridorEdgeDB


@dataclass(frozen=True)
class EdgeRow:
    """Arista resuelta: sus nodos extremo y las coordenadas de cada extremo."""
    edge_id: str
    source_node: str
    target_node: str
    src_lon: float
    src_lat: float
    tgt_lon: float
    tgt_lat: float


class CorridorMatchingRepo:
    """Lectura de la cadena + overlap geométrico + persistencia del corredor."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # --- lectura de la cadena (cualquier dialecto: no usa geom) ---

    def fetch_edges(self, edge_ids: list[str]) -> list[EdgeRow]:
        """Trae las aristas existentes con los extremos de cada nodo (vía ``graph_nodes``).

        No toca la columna ``geom`` (PostGIS): usa ``source_node``/``target_node`` y
        ``graph_nodes.lon/lat``, así que funciona también en el SQLite de los tests. Las
        aristas inexistentes simplemente no vienen en el resultado (el caller detecta faltantes).
        """
        if not edge_ids:
            return []
        stmt = text(
            "SELECT ge.edge_id, ge.source_node, ge.target_node, "
            "       sn.lon AS src_lon, sn.lat AS src_lat, "
            "       tn.lon AS tgt_lon, tn.lat AS tgt_lat "
            "FROM graph_edges ge "
            "JOIN graph_nodes sn ON sn.node_id = ge.source_node "
            "JOIN graph_nodes tn ON tn.node_id = ge.target_node "
            "WHERE ge.edge_id IN :ids"
        ).bindparams(bindparam("ids", expanding=True))
        rows = self.session.execute(stmt, {"ids": edge_ids}).all()
        return [
            EdgeRow(
                edge_id=r.edge_id,
                source_node=r.source_node,
                target_node=r.target_node,
                src_lon=r.src_lon,
                src_lat=r.src_lat,
                tgt_lon=r.tgt_lon,
                tgt_lat=r.tgt_lat,
            )
            for r in rows
        ]

    # --- overlap geométrico (SÓLO PostGIS) ---

    def edge_overlaps(
        self, edge_id: str, segment_wkts: list[str], buffer_m: float, min_overlap: float
    ) -> dict[int, float]:
        """``{índice_de_segmento: overlap_ratio}`` para los segmentos que superan el umbral.

        overlap_ratio = longitud de la arista dentro del buffer / longitud de la arista. El
        buffer es métrico (``::geography``, convención del repo) sobre la polilínea TomTom, que
        entra como WKT EFÍMERO (``ST_GeomFromText(:wkt, 4326)``) — nunca se persiste. Sin PostGIS
        (SQLite de tests) devuelve ``{}`` (el matching geométrico se valida en el e2e).
        """
        if self.session.get_bind().dialect.name != "postgresql":
            return {}
        out: dict[int, float] = {}
        sql = text(
            "SELECT ST_Length("
            "  ST_Intersection("
            "    ge.geom, "
            "    ST_Buffer(ST_GeomFromText(:wkt, 4326)::geography, :buf)::geometry"
            "  )::geography"
            ") / NULLIF(ST_Length(ge.geom::geography), 0) AS ratio "
            "FROM graph_edges ge WHERE ge.edge_id = :eid"
        )
        for idx, wkt in enumerate(segment_wkts):
            ratio = self.session.execute(
                sql, {"wkt": wkt, "buf": buffer_m, "eid": edge_id}
            ).scalar()
            if ratio is not None and ratio >= min_overlap:
                out[idx] = float(ratio)
        return out

    # --- persistencia (transaccional; el caller hace commit/rollback) ---

    def persist(
        self,
        name: str,
        created_by: str | None,
        ordered_mapping: list[tuple[str, int, str | None]],
    ) -> str:
        """Crea el ``corridor`` y sus ``corridor_edges`` (sólo IDs). Devuelve ``corridor_id``.

        ``ordered_mapping`` = ``[(edge_id, sequence, openlr|None), ...]`` ya ordenado. Hace
        ``flush`` (genera el ``corridor_id``); el caller envuelve en transacción y hace commit.
        NADA de geometría/tráfico de TomTom — sólo el ID OpenLR (nullable).
        """
        corridor = CorridorDB(name=name, created_by=created_by)
        self.session.add(corridor)
        self.session.flush()  # materializa corridor_id (default uuid en el modelo)
        for edge_id, sequence, openlr in ordered_mapping:
            self.session.add(
                CorridorEdgeDB(
                    corridor_id=corridor.corridor_id,
                    edge_id=edge_id,
                    sequence=sequence,
                    tomtom_openlr=openlr,
                )
            )
        return corridor.corridor_id
