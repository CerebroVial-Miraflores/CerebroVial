"""
Endpoints for realtime streaming.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
from ....infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster

app = FastAPI()

# Singleton broadcaster
_broadcaster = RealtimeBroadcaster()

def get_broadcaster() -> RealtimeBroadcaster:
    return _broadcaster

@app.get("/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """
    Server-Sent Events endpoint for streaming.
    
    Frontend usage:
    ```javascript
    const eventSource = new EventSource('/stream/CAM_001');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Vehicles:', data.total_vehicles);
    };
    ```
    """
    broadcaster = get_broadcaster()
    queue = await broadcaster.subscribe(camera_id)
    
    async def event_generator():
        try:
            while True:
                data = await queue.get()
                yield {
                    "event": "analysis",
                    "data": json.dumps(data)
                }
        except asyncio.CancelledError:
            await broadcaster.unsubscribe(camera_id, queue)
            raise
    
    return EventSourceResponse(event_generator())

@app.get("/cameras")
async def list_cameras():
    """Lists active cameras."""
    broadcaster = get_broadcaster()
    return {
        "cameras": list(broadcaster._subscribers.keys()),
        "latest_states": broadcaster._latest_state
    }

@app.get("/snapshot/{camera_id}")
async def get_snapshot(camera_id: str):
    """Gets latest state of a camera (polling fallback)."""
    broadcaster = get_broadcaster()
    if camera_id not in broadcaster._latest_state:
        raise HTTPException(404, "Camera not found")
    return broadcaster._latest_state[camera_id]
