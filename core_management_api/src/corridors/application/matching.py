"""Orquestación del matching geométrico edge→OpenLR (Fase B-2).

Une las piezas: las funciones PURAS de ``geometry`` (bearing/sentido) + el cálculo
geométrico de overlap que vive en el repositorio (PostGIS). Aquí también viven las
constantes NOMBRADAS y ajustables del matching y las validaciones puras de la cadena.

La geometría de TomTom (``coordinates``) entra como WKT efímero al repositorio y se
descarta; este módulo nunca la persiste (ToS 11.6.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .geometry import Coord, bearing, local_bearing, midpoint, same_direction

# --- Constantes de matching (NOMBRADAS y ajustables; Cesar las calibra con datos reales) ---

# Buffer (metros) alrededor de la polilínea TomTom. Absorbe la desalineación entre fuentes
# (TomTom vs el grafo SUMO) sin invadir calles paralelas. Sobre este valor se decide qué
# fracción de la arista "cae sobre" el segmento.
BUFFER_METERS = 15.0

# Fracción mínima de la longitud de la arista que debe quedar dentro del buffer para contar
# como pertenencia. 0.70 tolera extremos desalineados sin aceptar cruces tangenciales.
MIN_OVERLAP_RATIO = 0.70

# Umbral angular (grados) de la desambiguación por sentido: diferencia de rumbo < 90° = mismo
# sentido. Descarta el segmento de la calzada opuesta (~180° de diferencia).
MAX_BEARING_DIFF_DEG = 90.0

# DEUDA-MATCHING-CALIBRACION (ampliada): además de calibrar estos 3 umbrales con corredores
# reales, vigilar el caso de calzadas paralelas cercanas en óvalos/cruces (las 11
# intersecciones): puede no resolverse con buffer y requerir lógica de desempate por MEJOR
# alineación de bearing (no sólo el umbral binario MAX_BEARING_DIFF_DEG). Probar primero sobre
# un óvalo (caso difícil) en Fase B-front. Hoy, ante empate de overlap entre candidatos del
# mismo sentido, match_corridor toma el de mayor overlap; el desempate fino por bearing no está.


# --- Estructuras de entrada del matching ---

@dataclass(frozen=True)
class TomTomSegment:
    """Segmento ``flowSegmentData`` que el FRONT obtuvo de TomTom. EFÍMERO.

    ``coordinates`` (la geometría) se usa para el matching y se descarta; sólo ``openlr``
    (el ID OpenLR) puede llegar a persistirse.
    """
    openlr: str
    coordinates: list[Coord]  # [(lon, lat), ...] en orden de flujo


@dataclass(frozen=True)
class CorridorEdgeInput:
    """Arista del corredor con sus extremos resueltos (para bearing y matching)."""
    edge_id: str
    sequence: int
    source_coord: Coord  # (lon, lat) del nodo origen
    target_coord: Coord  # (lon, lat) del nodo destino


class OverlapSource(Protocol):
    """Lo que el matching necesita del repositorio: el overlap geométrico por arista.

    Desacopla la orquestación del SQL/PostGIS para poder testear el flujo sin DB.
    """

    def edge_overlaps(
        self, edge_id: str, segment_wkts: list[str], buffer_m: float, min_overlap: float
    ) -> dict[int, float]:
        """``{índice_de_segmento: overlap_ratio}`` para los segmentos que pasan buffer+overlap."""
        ...


def polyline_wkt(coords: list[Coord]) -> str:
    """WKT ``LINESTRING`` a partir de coordenadas ``(lon, lat)``.

    Va como BIND PARAM efímero a ``ST_GeomFromText`` en el repositorio; nunca se persiste.
    """
    pts = ", ".join(f"{lon} {lat}" for lon, lat in coords)
    return f"LINESTRING({pts})"


def match_corridor(
    repo: OverlapSource,
    edges: list[CorridorEdgeInput],
    segments: list[TomTomSegment],
) -> dict[str, str | None]:
    """Devuelve ``{edge_id: openlr | None}`` para cada arista del corredor.

    Por cada arista: (1) el repo filtra los segmentos cuyo overlap >= ``MIN_OVERLAP_RATIO``
    dentro del buffer de ``BUFFER_METERS``; (2) se descartan los de sentido opuesto comparando
    el rumbo de la arista contra el rumbo LOCAL del segmento cerca de la arista; (3) de los que
    pasan buffer Y sentido, gana el de mayor overlap; si ninguno pasa → ``None`` (sin cobertura).
    """
    segment_wkts = [polyline_wkt(s.coordinates) for s in segments]
    mapping: dict[str, str | None] = {}
    for edge in edges:
        overlaps = repo.edge_overlaps(
            edge.edge_id, segment_wkts, BUFFER_METERS, MIN_OVERLAP_RATIO
        )
        edge_bearing = bearing(edge.source_coord, edge.target_coord)
        ref = midpoint(edge.source_coord, edge.target_coord)
        best: tuple[float, str] | None = None  # (overlap_ratio, openlr)
        for idx, ratio in overlaps.items():
            seg = segments[idx]
            seg_bearing = local_bearing(seg.coordinates, ref)
            if not same_direction(edge_bearing, seg_bearing, MAX_BEARING_DIFF_DEG):
                continue  # sentido opuesto: es la otra calzada
            if best is None or ratio > best[0]:
                best = (ratio, seg.openlr)
        mapping[edge.edge_id] = best[1] if best is not None else None
    return mapping


# --- Validaciones PURAS de la cadena (testeables en SQLite) ---

def validate_sequences(sequences: list[int]) -> str | None:
    """``None`` si las ``sequence`` forman una cadena contigua sin huecos ni duplicados.

    Devuelve un mensaje de error (para un 4xx) en caso contrario. No fuerza el valor inicial:
    sólo exige que sean enteros consecutivos (el front numera desde donde quiera).
    """
    if not sequences:
        return "el corredor no tiene aristas"
    if len(set(sequences)) != len(sequences):
        return "hay valores de sequence duplicados"
    ordered = sorted(sequences)
    if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
        return "la sequence tiene huecos (debe ser una cadena contigua)"
    return None


class _HasEndpoints(Protocol):
    edge_id: str
    source_node: str
    target_node: str


def validate_continuity(ordered_edges: list[_HasEndpoints]) -> str | None:
    """``None`` si la cadena (ya ordenada por ``sequence``) es continua.

    Continua = el ``target_node`` de cada arista es el ``source_node`` de la siguiente. El
    grafo es dirigido (par ``-id``/``id`` por calzada), así que esto también valida el sentido
    de la cadena. Devuelve un mensaje de error si se rompe.
    """
    for a, b in zip(ordered_edges, ordered_edges[1:]):
        if a.target_node != b.source_node:
            return (
                f"cadena rota entre {a.edge_id} (target={a.target_node}) y "
                f"{b.edge_id} (source={b.source_node})"
            )
    return None
