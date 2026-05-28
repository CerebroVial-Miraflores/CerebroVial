"""Conteo de vehículos por zona (polígono) vía centroide del bbox.

Los polígonos se reciben en coordenadas de pixel del frame. No hay escalado
implícito de resolución: el rescale 1280x720 del módulo anterior era una de
las causas del bug C1.8 (vehículos dentro del polígono contaban 0).
"""
from collections.abc import Sequence

from ...domain.entities import DetectedVehicle, ZoneVehicleCount
from ...domain.value_objects import VehicleId, ZoneId

Polygon = Sequence[tuple[int, int]]


def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[int, int]]) -> bool:
    """Ray casting (regla even-odd). El centroide es el punto de contención."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class ZoneCounter:
    """Cuenta vehículos por zona configurada usando el centroide del bbox."""

    def __init__(self, zones: dict[ZoneId, Polygon]) -> None:
        self._zones: dict[ZoneId, list[tuple[int, int]]] = {
            zid: [tuple(pt) for pt in polygon] for zid, polygon in zones.items()
        }

    def count(
        self,
        detections: list[DetectedVehicle],
        frame_id: int,
    ) -> dict[ZoneId, ZoneVehicleCount]:
        result: dict[ZoneId, ZoneVehicleCount] = {}
        for zid, polygon in self._zones.items():
            ids: list[VehicleId] = [
                d.id for d in detections
                if _point_in_polygon(_centroid(d.bbox), polygon)
            ]
            result[zid] = ZoneVehicleCount(zone_id=zid, count=len(ids), vehicle_ids=ids)
        return result
