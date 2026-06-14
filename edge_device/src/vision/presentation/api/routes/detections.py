"""API del tap de detecciones por-frame (Fase 4 Mitad A, D-019).

`GET /detections/{camera_id}/latest` — sirve las cajas del último frame con
detección de la cámara, **normalizadas a [0, 1]**, para el overlay del front sobre
el video HLS directo. Pensado para polling ~1 Hz desde el front (la cadencia del
scheduler keyframe-only); espeja el patrón polling-fallback de `/snapshot/{id}` de
`streaming.py`, NO una conexión persistente.

Decoupling deliberado (criterio HU-22): canal SEPARADO del SSE de métricas
(`/stream/{id}`). El SSE lleva agregados por-ventana (~5 s) con contrato tipado que
alimenta las tarjetas; las cajas tienen cadencia, payload y consumidor distintos.
No se toca ese SSE.

Este endpoint solo LEE el último análisis cacheado en `CameraState.latest_detections`
(que el scheduler escribe ungated por `render_enabled`): no dispara inferencia ni
abre conexiones. Auth: el edge no exige JWT (deuda backend documentada), igual que
el resto de sus rutas.
"""
import time

from fastapi import FastAPI, HTTPException

from .cameras import get_manager
from ...serializers import empty_detections_payload, serialize_detections

app = FastAPI()


@app.get("/detections/{camera_id}/latest")
async def latest_detections(camera_id: str):
    """Cajas normalizadas [0,1] + timestamps del último frame con detección."""
    manager = get_manager()
    if camera_id not in manager.cameras:
        raise HTTPException(404, "Camera not found")

    server_ts = time.time()
    stored = manager.get_latest_detections(camera_id)
    if stored is None:
        return empty_detections_payload(camera_id, server_ts)

    analysis, width, height = stored
    return serialize_detections(analysis, width, height, server_ts, camera_id)
