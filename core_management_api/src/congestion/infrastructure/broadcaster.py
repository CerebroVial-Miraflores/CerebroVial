"""Broadcaster pub/sub para el wake-up de congestión de RED (TTH-12, CT-12.6).

Patrón SSE-wake/REST-read (DHU-021 #15), idéntico en espíritu al
``ActiveStateBroadcaster`` de ``control/`` pero **instancia e implementación
propias** — congestion/ no se acopla a control/ (decisión de Fase 3). A diferencia
del de control (keyed por ``node_id``), este es un **único canal de RED**: emite
"hay nueva información de congestión" SIN payload; el cliente, al despertar, re-lee
``GET /congestion/state`` para el estado autoritativo desde ``waze_jams``.

Per-proceso por construcción (dict de colas en el worker uvicorn). Escalar más allá
de un worker exige Redis pub/sub — fuera de alcance del MVP (single-worker).
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NetworkCongestionBroadcaster:
    """Pub/sub de un único canal de red para ``congestion-updated`` (sin payload)."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, queue_size: int = 32) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        async with self._lock:
            self._subscribers.add(queue)
            count = len(self._subscribers)
        logger.debug("congestion subscribe: subscribers=%d", count)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue)
            remaining = len(self._subscribers)
        logger.debug("congestion unsubscribe: subscribers=%d", remaining)

    async def publish(self, event: Optional[dict] = None) -> int:
        """Despierta a todos los suscriptores. Devuelve a cuántos se entregó.

        No bloqueante por suscriptor: si una cola está llena se descarta el wake
        para ese cliente (su próxima re-lectura del REST trae el estado vigente).
        """
        payload = event or {"type": "congestion-updated"}
        async with self._lock:
            queues = list(self._subscribers)
        delivered = 0
        for q in queues:
            try:
                q.put_nowait(payload)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning("congestion publish drop: queue_full total=%d", len(queues))
        logger.debug("congestion publish: delivered=%d/%d", delivered, len(queues))
        return delivered


# Singleton per-proceso (mismo patrón que init_/get_broadcaster de control).
_instance: Optional[NetworkCongestionBroadcaster] = None


def init_congestion_broadcaster(
    broadcaster: Optional[NetworkCongestionBroadcaster] = None,
) -> NetworkCongestionBroadcaster:
    global _instance
    _instance = broadcaster or NetworkCongestionBroadcaster()
    return _instance


def get_congestion_broadcaster() -> NetworkCongestionBroadcaster:
    """Dependency FastAPI. Lanza si no se inicializó (igual que get_broadcaster)."""
    if _instance is None:
        raise RuntimeError("NetworkCongestionBroadcaster not initialized")
    return _instance


def reset_congestion_broadcaster() -> None:
    """Helper de tests: limpia el singleton."""
    global _instance
    _instance = None
