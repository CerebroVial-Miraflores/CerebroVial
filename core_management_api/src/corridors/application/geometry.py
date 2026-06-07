"""Geometría PURA del matching de corredores (Fase B-2).

Sin SQL, sin DB: funciones determinísticas sobre coordenadas ``(lon, lat)`` en grados.
Corren SIEMPRE en los tests unit (SQLite), independientes de PostGIS. Aquí vive la
desambiguación por SENTIDO (bearing), que es lo que distingue la calzada de ida de la de
vuelta cuando dos segmentos TomTom de sentidos opuestos pasan el mismo buffer.
"""
from __future__ import annotations

import math

Coord = tuple[float, float]  # (lon, lat) en grados


def bearing(p1: Coord, p2: Coord) -> float:
    """Rumbo inicial (azimut, 0-360°) de ``p1`` a ``p2``, cada uno ``(lon, lat)``.

    0° = Norte, 90° = Este (fórmula estándar de forward azimuth sobre la esfera). Es el
    SENTIDO de viaje: ``bearing(a, b)`` y ``bearing(b, a)`` difieren ~180°.
    """
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angular_diff(b1: float, b2: float) -> float:
    """Diferencia angular entre dos rumbos, normalizada a ``[0, 180]``."""
    d = abs(b1 - b2) % 360.0
    return d if d <= 180.0 else 360.0 - d


def same_direction(b1: float, b2: float, threshold: float) -> bool:
    """``True`` si la diferencia angular ``< threshold`` (mismo sentido).

    ``threshold`` se pasa explícito (la constante ``MAX_BEARING_DIFF_DEG`` vive en
    ``matching.py`` junto a las demás tunables, para no enterrar parámetros acá).
    """
    return angular_diff(b1, b2) < threshold


def midpoint(p1: Coord, p2: Coord) -> Coord:
    """Punto medio (promedio de coordenadas) de dos puntos ``(lon, lat)``."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def local_bearing(coords: list[Coord], ref_point: Coord) -> float:
    """Rumbo LOCAL de la polilínea ``coords`` en el tramo más cercano a ``ref_point``.

    Elige el par de vértices consecutivos cuyo punto medio está más cerca de ``ref_point``
    (el centro de la arista a matchear) y devuelve el ``bearing`` de ESE par. NO el rumbo
    extremo→extremo.

    Por qué local y no extremo→extremo (ajuste de diseño, Fase B-2): el segmento TomTom es
    largo (~4 km, 138-172 vértices) y curvo. Un rumbo extremo→extremo "miente" donde la
    avenida dobla (empieza al N, termina al E → "NE" global), lo que daría un falso descarte
    o falso match en avenidas curvas. El rumbo del tramo local es robusto a la curvatura.
    (Para calzadas opuestas el umbral de 90° absorbía el caso aun con bearing global —~180°
    sigue siendo >90°—, pero local es correcto en general, no sólo para ese caso.)

    La cercanía se mide con distancia euclídea planar en grados: es suficiente para ELEGIR el
    tramo más cercano a la escala de Miraflores (la anisotropía lon/lat no cambia cuál tramo
    queda más cerca); el rumbo devuelto sí usa la fórmula esférica de ``bearing``.
    """
    if len(coords) < 2:
        raise ValueError("local_bearing requiere una polilínea de al menos 2 puntos")
    best_i = 0
    best_d = math.inf
    for i in range(len(coords) - 1):
        mid = midpoint(coords[i], coords[i + 1])
        d = (mid[0] - ref_point[0]) ** 2 + (mid[1] - ref_point[1]) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return bearing(coords[best_i], coords[best_i + 1])
