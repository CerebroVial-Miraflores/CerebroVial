import asyncio
import json
from typing import Dict, List, Set, Optional
from dataclasses import asdict
from datetime import datetime
# No domain imports here, but let's check if it uses any.
# It uses FrameAnalysis in serialize_analysis but as a parameter type hint (implicit or explicit).
# It doesn't import it. Wait, let's check the content.

class RealtimeBroadcaster:
    """
    Pub/sub system to transmit analysis to connected clients.
    Thread-safe and asynchronous.
    """
    
    def __init__(self):
        # Subscribers per camera
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
        
        # Cache latest state per camera (for new subscribers)
        self._latest_state: Dict[str, dict] = {}

    async def subscribe(self, camera_id: str, queue_size: int = 50) -> asyncio.Queue:
        """
        Subscribes a client to updates from a specific camera.
        Returns an async queue that will receive the data.
        """
        queue = asyncio.Queue(maxsize=queue_size)
        
        async with self._lock:
            if camera_id not in self._subscribers:
                self._subscribers[camera_id] = set()
            self._subscribers[camera_id].add(queue)
        
        # Send latest known state immediately
        if camera_id in self._latest_state:
            try:
                await queue.put(self._latest_state[camera_id])
            except asyncio.QueueFull:
                pass
        
        return queue

    async def unsubscribe(self, camera_id: str, queue: asyncio.Queue):
        """Removes a subscriber."""
        async with self._lock:
            if camera_id in self._subscribers:
                self._subscribers[camera_id].discard(queue)
                if not self._subscribers[camera_id]:
                    del self._subscribers[camera_id]

    async def broadcast(self, camera_id: str, analysis_data: dict):
        """
        Transmits analysis to all subscribers of a camera.
        Non-blocking: if a client is slow, it is skipped.
        """
        # Update cache
        self._latest_state[camera_id] = analysis_data
        
        async with self._lock:
            subscribers = self._subscribers.get(camera_id, set()).copy()
        
        # Send to each subscriber (non-blocking)
        for queue in subscribers:
            try:
                queue.put_nowait(analysis_data)
            except asyncio.QueueFull:
                # Slow client - skip
                print(f"[WARNING] Skipping slow client for {camera_id}")

    def serialize_analysis(self, frame_analysis, camera_id: str) -> dict:
        """
        Converts FrameAnalysis to JSON-serializable dict.
        """
        return {
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            "frame_id": frame_analysis.frame_id,
            "total_vehicles": frame_analysis.total_count,
            "vehicles": [
                {
                    "id": v.id,
                    "type": v.type,
                    "confidence": round(v.confidence, 2),
                    "bbox": v.bbox,
                    "speed": round(v.speed, 1) if v.speed else None
                }
                for v in frame_analysis.vehicles
            ] if frame_analysis.vehicles else [],
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "vehicle_count": z.vehicle_count,
                    "avg_speed": round(z.avg_speed, 1),
                    "occupancy": round(z.occupancy, 2),
                    "vehicle_types": z.vehicle_types
                }
                for z in frame_analysis.zones
            ] if frame_analysis.zones else []
        }
