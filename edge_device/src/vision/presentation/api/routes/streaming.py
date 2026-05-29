"""
Endpoints for realtime streaming.
"""
from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
from ....infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster

app = FastAPI()

# Singleton broadcaster
_broadcaster = None

def init_broadcaster(broadcaster: RealtimeBroadcaster):
    global _broadcaster
    _broadcaster = broadcaster

def get_broadcaster() -> RealtimeBroadcaster:
    if _broadcaster is None:
        raise HTTPException(500, "Broadcaster not initialized")
    return _broadcaster

@app.get("/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """Server-Sent Events endpoint para una cámara.

    Emite el payload §6.2 (event_type `traffic_update`) por cada `TrafficData`
    agregado de la ventana. Frontend consume `data.metrics.unique_vehicles`,
    `data.metrics.flow_vehicles_per_hour`, etc.; la discretización a labels
    ES (`Bajo`/`Moderado`/`Alto`) y el `density %` se computan client-side
    (§9.2 prohíbe localización en transporte).
    """
    broadcaster = get_broadcaster()
    subscriber_id, queue = await broadcaster.subscribe(camera_id)

    async def event_generator():
        try:
            while True:
                data = await queue.get()
                yield {
                    "event": data.get("event_type", "traffic_update"),
                    "data": json.dumps(data),
                }
        except asyncio.CancelledError:
            await broadcaster.unsubscribe(subscriber_id)
            raise

    return EventSourceResponse(event_generator())

@app.get("/cameras")
async def list_cameras():
    """Lists active cameras."""
    broadcaster = get_broadcaster()
    return {
        "cameras": broadcaster.subscribed_cameras(),
        "latest_states": broadcaster.latest_states(),
    }

@app.get("/snapshot/{camera_id}")
async def get_snapshot(camera_id: str):
    """Gets latest state of a camera (polling fallback)."""
    broadcaster = get_broadcaster()
    state = broadcaster.latest_state(camera_id)
    if state is None:
        raise HTTPException(404, "Camera not found")
    return state
