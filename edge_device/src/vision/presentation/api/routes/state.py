"""Endpoint `GET /vision/state/{intersection_id}` (CT-08.6 + CT-08.11d).

Shape §6.5: `{intersection_id, timestamp, directions: [...]}`. `camera_id` del
dominio se serializa como `intersection_id` para alinear con HU-02 (§6.5,
"DHU-024 §5 usa intersection_id desde la perspectiva del consumidor; el módulo
vision usa camera_id porque cada cámara monitorea una intersección").

`directions` agrupa las zonas de la cámara con sus métricas §5.4:
`{direction, count, queue, flow, density}`. `queue=null` en MVP1 (F41 lo
poblará); `density` puede ser `null` si la zona carece de `segment_length_meters`.

Branch 5xx (CT-08.10): retorna **503 Service Unavailable** cuando la cámara
está registrada pero el módulo no procesa frames (sin TrafficData publicado,
cámara no started). 404 cuando la `intersection_id` no está registrada en el
manager — semánticamente distinto de "módulo caído".

El estado discreto OK/Degradado/Fuera de servicio vive en el endpoint de 6f,
no en el cuerpo de éste (§ plan-file 6e/6f, CT-08.10 mandata dos cosas
separadas).
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from ....domain.entities import TrafficData
from .cameras import get_manager
from .streaming import get_broadcaster

app = FastAPI()


def _direction_from(td: TrafficData) -> dict:
    return {
        "direction": td.zone_id.value,
        "count": td.unique_vehicles,
        "queue": None,
        "flow": td.flow_vehicles_per_hour,
        "density": td.density_vehicles_per_km,
    }


@app.get("/vision/state/{intersection_id}")
async def get_vision_state(intersection_id: str):
    manager = get_manager()
    if intersection_id not in manager.cameras:
        raise HTTPException(404, f"intersection_id {intersection_id!r} no registrada")

    broadcaster = get_broadcaster()
    traffic_by_zone = broadcaster.traffic_data_for(intersection_id)

    camera_status = manager.get_status().get(intersection_id, {})
    is_running = camera_status.get("running", False)

    if not traffic_by_zone:
        # Cámara existe pero el aggregator no produjo TrafficData todavía
        # (cámara no started, o ventana actual aún no cerró, o módulo caído).
        # CT-08.10: 5xx cuando el módulo no procesa frames.
        if not is_running:
            raise HTTPException(
                503,
                f"Módulo de visión no procesa frames para {intersection_id!r}",
            )
        # Cámara running pero sin datos: warm-up de la primera ventana.
        return {
            "intersection_id": intersection_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "directions": [],
        }

    directions = [_direction_from(td) for td in traffic_by_zone.values()]
    latest_window_end = max(
        td.window_end for td in traffic_by_zone.values()
    )

    return {
        "intersection_id": intersection_id,
        "timestamp": latest_window_end.isoformat(),
        "directions": directions,
    }
