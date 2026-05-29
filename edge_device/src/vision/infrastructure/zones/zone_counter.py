"""Conteo de vehículos y occupancy por zona (polígono).

Conteo (`count`, `vehicle_ids`): por contención del centroide del bbox en el
polígono (criterio §6.1 / regresión C1.8). Polígonos en pixel del frame; sin
rescaling implícito.

Occupancy: fracción del área del polígono cubierta por la UNIÓN de los bboxes
de detecciones que intersectan la zona. Ver DHU-025 para la interpretación de
§5.4 (`Σ` como unión, no suma aritmética con double-counting). El cómputo se
hace en el canvas local del bounding-box del polígono — no se necesita
`frame_shape` (el approach es robusto a cambios de resolución del feed).

`cv2` se importa de forma lazy (mismo patrón que `presentation/opencv_visualizer.py`
y `infrastructure/sources/video_source.py`) para permitir colectar el módulo en
entornos sin opencv instalado. `opencv-python` está declarado en
`edge_device/requirements.txt`.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...domain.entities import DetectedVehicle, ZoneVehicleCount
from ...domain.value_objects import VehicleId, ZoneId

Polygon = Sequence[tuple[int, int]]


@dataclass(frozen=True)
class _ZoneState:
    """Estado precomputado por zona (una sola vez en `__init__`)."""
    polygon: list[tuple[int, int]]         # coords del frame, para contención por centroide
    origin: tuple[int, int]                # (xmin, ymin) del bbox del polígono
    canvas_shape: tuple[int, int]          # (h_local, w_local)
    zone_mask: Any                         # np.ndarray uint8, 1 dentro del polígono, 0 fuera
    polygon_area: int                      # countNonZero(zone_mask)


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


def _build_zone_state(polygon: Polygon) -> _ZoneState:
    import cv2  # lazy: ver docstring del módulo

    pts = [tuple(pt) for pt in polygon]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, ymin = min(xs), min(ys)
    xmax, ymax = max(xs), max(ys)
    h_local = max(ymax - ymin, 0)
    w_local = max(xmax - xmin, 0)
    zone_mask = np.zeros((h_local, w_local), dtype=np.uint8)
    if h_local > 0 and w_local > 0:
        local_poly = np.array(
            [[x - xmin, y - ymin] for (x, y) in pts], dtype=np.int32
        )
        cv2.fillPoly(zone_mask, [local_poly], color=1)
    polygon_area = int(cv2.countNonZero(zone_mask))
    return _ZoneState(
        polygon=pts,
        origin=(xmin, ymin),
        canvas_shape=(h_local, w_local),
        zone_mask=zone_mask,
        polygon_area=polygon_area,
    )


class ZoneCounter:
    """Cuenta vehículos por zona (centroide) y computa occupancy por unión de bboxes."""

    def __init__(self, zones: dict[ZoneId, Polygon]) -> None:
        self._zones: dict[ZoneId, _ZoneState] = {
            zid: _build_zone_state(polygon) for zid, polygon in zones.items()
        }

    def count(
        self,
        detections: list[DetectedVehicle],
        frame_id: int,
    ) -> dict[ZoneId, ZoneVehicleCount]:
        result: dict[ZoneId, ZoneVehicleCount] = {}
        for zid, state in self._zones.items():
            ids: list[VehicleId] = [
                d.id for d in detections
                if _point_in_polygon(_centroid(d.bbox), state.polygon)
            ]
            occupancy = self._compute_occupancy(detections, state)
            result[zid] = ZoneVehicleCount(
                zone_id=zid,
                count=len(ids),
                vehicle_ids=ids,
                occupancy=occupancy,
            )
        return result

    @staticmethod
    def _compute_occupancy(
        detections: list[DetectedVehicle],
        state: _ZoneState,
    ) -> float:
        import cv2  # lazy: ver docstring del módulo

        if state.polygon_area == 0:
            return 0.0
        h_local, w_local = state.canvas_shape
        if h_local == 0 or w_local == 0:
            return 0.0
        xmin, ymin = state.origin
        xmax = xmin + w_local
        ymax = ymin + h_local
        union_mask = np.zeros((h_local, w_local), dtype=np.uint8)
        for d in detections:
            bx1, by1, bx2, by2 = d.bbox
            # Clip del bbox al bbox del polígono (intersección de rectángulos).
            cx1 = max(bx1, xmin)
            cy1 = max(by1, ymin)
            cx2 = min(bx2, xmax)
            cy2 = min(by2, ymax)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            # Coords locales del canvas (origen en (xmin, ymin)).
            lx1, ly1 = cx1 - xmin, cy1 - ymin
            lx2, ly2 = cx2 - xmin, cy2 - ymin
            union_mask[ly1:ly2, lx1:lx2] = 1
        intersection = cv2.bitwise_and(union_mask, state.zone_mask)
        intersection_area = int(cv2.countNonZero(intersection))
        return intersection_area / state.polygon_area
