"""
In-memory pub/sub broadcaster for the active-strategy stream (HU-05, DHU-021).

Per-process by construction: subscribers live in a Python dict on the uvicorn
worker. Horizontal scaling beyond one worker requires moving this to Redis
pub/sub or equivalent — out of scope for HU-05. Single-worker uvicorn is the
assumption (default of the core_management_api compose service).

The broadcaster only carries the *change signal*; the authoritative state is
served by ``GET /control/active-state/{node_id}`` against the database. Clients
re-read upon receiving an event — single source of truth (DHU-021 #15).
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ActiveStateBroadcaster:
    """Pub/sub for ``active-state-changed`` events keyed by node_id."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, node_id: str, queue_size: int = 32) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        async with self._lock:
            self._subscribers.setdefault(node_id, set()).add(queue)
            count = len(self._subscribers[node_id])
        logger.debug("subscribe: node=%s subscribers=%d", node_id, count)
        return queue

    async def unsubscribe(
        self, node_id: str, queue: asyncio.Queue
    ) -> None:
        async with self._lock:
            if node_id in self._subscribers:
                self._subscribers[node_id].discard(queue)
                if not self._subscribers[node_id]:
                    del self._subscribers[node_id]
                remaining = len(self._subscribers.get(node_id, ()))
            else:
                remaining = 0
        logger.debug("unsubscribe: node=%s subscribers=%d", node_id, remaining)

    async def publish(self, node_id: str, event: dict) -> int:
        """Deliver ``event`` to every subscriber of ``node_id``.

        Non-blocking per subscriber: a slow client whose queue is full is
        skipped (the event is dropped for them) — its next reconnect will
        re-fetch authoritative state from GET /control/active-state. Returns
        the number of subscribers the event was actually delivered to.
        """
        async with self._lock:
            queues = list(self._subscribers.get(node_id, set()))
        delivered = 0
        for q in queues:
            try:
                q.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning(
                    "publish drop: node=%s queue_full subscribers_total=%d",
                    node_id,
                    len(queues),
                )
        logger.debug(
            "publish: node=%s delivered=%d/%d", node_id, delivered, len(queues)
        )
        return delivered


# Singleton wiring (mirrors the pattern of init_engine/get_engine in routes.py)
_broadcaster_instance: Optional[ActiveStateBroadcaster] = None


def init_broadcaster(
    broadcaster: Optional[ActiveStateBroadcaster] = None,
) -> ActiveStateBroadcaster:
    """Set the process-global broadcaster. If ``broadcaster`` is None, creates
    a fresh one. Returns the instance now in use."""
    global _broadcaster_instance
    _broadcaster_instance = broadcaster or ActiveStateBroadcaster()
    return _broadcaster_instance


def get_broadcaster() -> ActiveStateBroadcaster:
    """FastAPI dependency. Raises if init_broadcaster was not called yet
    (matches the pattern used by ``get_engine`` in routes.py)."""
    if _broadcaster_instance is None:
        raise RuntimeError("ActiveStateBroadcaster not initialized")
    return _broadcaster_instance


def reset_broadcaster() -> None:
    """Test helper. Clears the singleton so the next test gets a fresh
    instance via init_broadcaster()."""
    global _broadcaster_instance
    _broadcaster_instance = None
