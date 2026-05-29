"""OpenCV `FrameRenderer` para anotar frames con zonas, bboxes y labels.

Implementa el Protocol `FrameRenderer` (§7.2): `render(Frame, FrameAnalysis) -> np.ndarray`.
Recibe `Frame` (entity) y devuelve un array nuevo — no muta la entidad.
"""
import numpy as np

from ...domain.entities import Frame, FrameAnalysis


def _id_value(obj):
    return obj.value if hasattr(obj, "value") else obj


class OpenCVVisualizer:
    """Renderer concreto basado en OpenCV. Inyectado al `MultiCameraManager`."""

    def __init__(self, zones_config: dict | None = None):
        self.zones_config = zones_config or {}

    def render(self, frame: Frame, analysis: FrameAnalysis) -> np.ndarray:
        import cv2

        img = frame.image.copy()

        if self.zones_config:
            for zone_id_str, points in self.zones_config.items():
                if not points:
                    continue
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (255, 0, 0), 2)

                count = 0
                for zvc in analysis.zones.values():
                    if _id_value(zvc.zone_id) == zone_id_str:
                        count = zvc.count
                        break

                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(
                        img,
                        f"{zone_id_str}: {count}",
                        (cX - 20, cY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

        for vehicle in analysis.vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            label = f"{vehicle.type} {_id_value(vehicle.id)}"
            if vehicle.speed is not None:
                label += f" {vehicle.speed:.0f}km/h"
            else:
                label += f" {vehicle.confidence:.2f}"

            color = (0, 255, 0)
            if vehicle.type == "bus":
                color = (0, 165, 255)
            elif vehicle.type == "truck":
                color = (0, 0, 255)
            elif vehicle.type == "motorcycle":
                color = (255, 255, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        cv2.putText(
            img,
            f"Vehicles: {len(analysis.vehicles)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        return img
