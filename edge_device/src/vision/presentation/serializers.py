"""Serializador del tap de detecciones por-frame (Fase 4 Mitad A, D-019).

Transforma un `FrameAnalysis` del dominio en el payload JSON que consume el
overlay del front sobre el video HLS directo. Función PURA y determinística (sin
red ni stack vivo): toma las cajas en píxeles del frame de inferencia y las
**normaliza a [0, 1]** dividiéndolas por las dimensiones del frame, de modo que el
overlay las escale al tamaño del `<video>` (dependiente del viewport, mobile-first)
sin conocer la resolución de inferencia (hoy 1280×720 forzada por
`HlsKeyframeSource`).

Coordenadas normalizadas (no píxeles): desacoplan el front de la resolución del
edge y del viewport. Se incluyen DOS timestamps, ambos en reloj del edge (sin
sesgo por skew con el reloj del browser):

- `frame_timestamp`  — cuándo se observó la detección (instante del frame).
- `server_timestamp` — cuándo se serializó la respuesta (el "ahora" del edge).

El overlay oculta las cajas cuando `server_timestamp - frame_timestamp` supera su
umbral de frescura (~3 s) — antigüedad de la CAJA, calculada en tiempo del edge.
Ese umbral del overlay es independiente del umbral de frescura del scheduler
(Fase 1): uno gobierna si el overlay dibuja, el otro si el scheduler infiere.
"""
from __future__ import annotations

from ..domain.entities import FrameAnalysis


def _id_value(obj):
    """Valor crudo de un id que puede venir como value object (`VehicleId.value`)
    o como str/int plano (el detector asigna str antes del tracking). Espeja
    `opencv_visualizer._id_value`."""
    return obj.value if hasattr(obj, "value") else obj


def _clamp01(value: float) -> float:
    """Acota a [0, 1]: una bbox puede asomar fuera del frame (caja parcial en el
    borde) y el overlay no debe pintar fuera del video."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def serialize_detections(
    analysis: FrameAnalysis,
    frame_width: int,
    frame_height: int,
    server_timestamp: float,
    camera_id: str,
) -> dict:
    """`FrameAnalysis` → payload del overlay, con bboxes normalizados a [0, 1].

    `frame_width`/`frame_height` son las dimensiones del frame de inferencia (en
    cuyo espacio de píxeles vive `DetectedVehicle.bbox`). Si llegan no-positivas
    (defensivo), se cae a 1.0 para no dividir por cero — el overlay clampearía.
    """
    w = float(frame_width) if frame_width and frame_width > 0 else 1.0
    h = float(frame_height) if frame_height and frame_height > 0 else 1.0

    detections = []
    for vehicle in analysis.vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        detections.append(
            {
                "id": str(_id_value(vehicle.id)),
                "type": vehicle.type,
                "confidence": round(float(vehicle.confidence), 4),
                "bbox": [
                    _clamp01(x1 / w),
                    _clamp01(y1 / h),
                    _clamp01(x2 / w),
                    _clamp01(y2 / h),
                ],
            }
        )

    return {
        "camera_id": camera_id,
        "frame": {"width": int(frame_width), "height": int(frame_height)},
        "frame_timestamp": analysis.timestamp,
        "server_timestamp": server_timestamp,
        "detection_ran": analysis.detection_ran,
        "detections": detections,
    }


def empty_detections_payload(camera_id: str, server_timestamp: float) -> dict:
    """Payload para una cámara registrada que aún no produjo detecciones (recién
    activada, o sin frame fresco). El front lo lee como "sin cajas que dibujar"."""
    return {
        "camera_id": camera_id,
        "frame": None,
        "frame_timestamp": None,
        "server_timestamp": server_timestamp,
        "detection_ran": False,
        "detections": [],
    }
