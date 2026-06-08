"""Endpoint `GET /vision/health` — health check del módulo (CT-08.10 + CT-08.11f).

Endpoint **separado** de `GET /vision/state` (6e). Reporta estado discreto
OK / Degradado / Fuera de servicio como payload estructurado, consumible por
TTH-04 conforme a CT-04.1 ("el mecanismo determina para cada componente uno
de tres estados").

Umbrales de implementación (no fijados por el diseño):
- `LAST_FRAME_FRESHNESS_SECONDS = 5.0`: frames con `time.time() - frame.timestamp`
  mayor a este threshold cuentan como "sin frames recientes".
- Estado global = worst-of-fleet sobre las cámaras registradas:
  * Sin cámaras registradas → **Fuera de servicio** (503).
  * Ninguna cámara saludable → **Fuera de servicio** (503).
  * Alguna cámara saludable + `aggregation_errors`/`data_dropped` > 0 o
    fleet parcialmente sano → **Degradado** (200).
  * Todas saludables + counters en 0 → **OK** (200).

Contadores consumidos del aggregator (Fase 5b, §11.3): `aggregation_errors`
(saves fallidos, §11.1) y `data_dropped` (output queue llena, §11.2). Son
properties thread-safe expuestas por `AsyncTrafficAggregator`.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ....domain.value_objects import SensorStatus
from .cameras import get_manager

app = FastAPI()

_LAST_FRAME_FRESHNESS_SECONDS = 5.0


def _camera_telemetry(camera) -> dict:
    aggregator = camera.state.aggregator
    aggregation_errors = aggregator.aggregation_errors if aggregator else 0
    data_dropped = aggregator.data_dropped if aggregator else 0

    # B1 Paso 1b (seam §3, opción ii): la cámara por scheduler no corre pipeline,
    # así que `get_latest()` no aplica. Reporta el SensorStatus REAL del scheduler
    # (más preciso que "edad del último frame procesado": distingue
    # FUENTE_NO_DISPONIBLE de SIN_FRAME_FRESCO). Sin fachada get_latest()-compatible.
    sensor_status = getattr(camera.state, "sensor_status", None)
    if isinstance(sensor_status, str):
        return {
            "running": bool(camera.state.is_running),
            "sensor_status": sensor_status,
            "last_frame_age_seconds": getattr(
                camera.state, "last_frame_age_seconds", None
            ),
            "aggregation_errors": int(aggregation_errors),
            "data_dropped": int(data_dropped),
        }

    # Path viejo (pipeline.run()): frescura por edad del último frame procesado.
    latest = camera.state.pipeline.get_latest() if camera.state.pipeline else None
    last_frame_age: float | None = None
    if latest is not None:
        frame, _ = latest
        ts = getattr(frame, "timestamp", None)
        if ts is not None:
            last_frame_age = max(0.0, time.time() - float(ts))

    return {
        "running": bool(camera.state.is_running),
        "last_frame_age_seconds": last_frame_age,
        "aggregation_errors": int(aggregation_errors),
        "data_dropped": int(data_dropped),
    }


def _is_camera_healthy(telemetry: dict) -> bool:
    if not telemetry["running"]:
        return False
    # Cámara scheduled (1b): sana sii el sensor está en OK (frame fresco → infiere).
    status = telemetry.get("sensor_status")
    if status is not None:
        return status == SensorStatus.OK.value
    age = telemetry["last_frame_age_seconds"]
    return age is not None and age < _LAST_FRAME_FRESHNESS_SECONDS


def _has_degradation(telemetry: dict) -> bool:
    return telemetry["aggregation_errors"] > 0 or telemetry["data_dropped"] > 0


@app.get("/vision/health")
async def get_vision_health():
    manager = get_manager()
    cameras_telemetry = {
        cam_id: _camera_telemetry(cam) for cam_id, cam in manager.cameras.items()
    }

    if not cameras_telemetry:
        status = "Fuera de servicio"
    else:
        healthy = [
            cid for cid, t in cameras_telemetry.items() if _is_camera_healthy(t)
        ]
        if not healthy:
            status = "Fuera de servicio"
        elif len(healthy) < len(cameras_telemetry) or any(
            _has_degradation(t) for t in cameras_telemetry.values()
        ):
            status = "Degradado"
        else:
            status = "OK"

    body = {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "cameras": cameras_telemetry,
    }
    http_status = 503 if status == "Fuera de servicio" else 200
    return JSONResponse(content=body, status_code=http_status)
