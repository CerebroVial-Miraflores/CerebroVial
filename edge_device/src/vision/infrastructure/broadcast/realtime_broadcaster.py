"""Broadcaster Protocol-conforme para emitir `TrafficData` al transporte SSE.

Cumple `domain.protocols.Broadcaster` (§6.10/§6.11): `publish(TrafficData)`,
`subscriber_count()`, `is_subscribed(subscriber_id)`. Cero acceso externo a
los atributos privados (los consumidores de `presentation/` usan los helpers
públicos `subscribed_cameras()`, `latest_state()`, `latest_states()`).

Payload §6.2 — agrupado en `schema_version`, `event_type`, `server_timestamp`,
`camera{id, street_monitored}`, `zone{id}`, `window{start, end, duration_seconds}`,
`metrics{...}`. Valores Optional del dominio (`mean_speed_kmh`,
`density_vehicles_per_km`) se serializan como `null`. **Prohibido por §9.2**:
strings localizadas ("Bajo"/"Moderado"/"Alto"), umbrales hardcoded de
congestión, formateos con `%` o sufijos de unidad — todo eso vive en el
frontend (§9.3: ningún consumidor MVP1 requiere strings localizadas en el
SSE).

`camera.street_monitored` se emite como `null`: el broadcaster es transporte
canónico y no tiene acceso al registry de cámaras. El frontend resuelve el
nombre humano vía su propio mapping camera_id → street. F41 puede formalizar
un `CameraMetadataProvider` inyectable si aparece un consumidor non-frontend
que lo requiera (YAGNI en MVP1).

Cruce thread→async (topología B / 15Hz): el broadcaster se llama desde el event
loop main. El `AsyncTrafficAggregator-worker` deposita `TrafficData` en su
`_output_queue` (thread-safe); el demux del `BatchInferenceWorker` (task asyncio,
en el event loop) lo consume con `aggregator.flush()` (sync, no bloquea) y hace
`await broadcaster.publish(td)`. Única excepción: el `QueuePushProducer` (daemon
thread), al morir la fuente, cierra la ventana con `force_flush()` y publica el
resultado vía `asyncio.run_coroutine_threadsafe(...)` sobre el loop main.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from ...domain.entities import TrafficData

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"
_EVENT_TYPE = "traffic_update"


class RealtimeBroadcaster:
    """Broadcaster SSE in-memory Protocol-conforme."""

    def __init__(self) -> None:
        # subscriber_id -> (camera_id, queue) — único índice de suscripciones.
        self._subscribers: dict[str, tuple[str, asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
        # Último payload §6.2 publicado por camera_id (cache para nuevos
        # subscribers).
        self._latest_state: dict[str, dict] = {}
        # Último TrafficData por (camera_id, zone_id). 6e lo necesita para
        # armar `GET /vision/state/{intersection_id}` con TODAS las direcciones
        # de la cámara — el cache `_latest_state` por sí solo solo guardaría
        # la zona del último publish.
        self._latest_traffic_data: dict[str, dict[str, TrafficData]] = {}

    # ---- Protocol §6.10/§6.11 ------------------------------------------

    async def publish(self, data: TrafficData) -> None:
        """Emite el payload §6.2 a todos los subscribers de la `camera_id` del dato."""
        camera_id = data.camera_id.value
        zone_id = data.zone_id.value
        payload = self._build_payload(data)
        self._latest_state[camera_id] = payload
        self._latest_traffic_data.setdefault(camera_id, {})[zone_id] = data

        async with self._lock:
            targets = [
                (sid, queue)
                for sid, (cid, queue) in self._subscribers.items()
                if cid == camera_id
            ]

        for sid, queue in targets:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning(
                    "Subscriber %s lento para camera %s; payload dropeado",
                    sid,
                    camera_id,
                )

    def subscriber_count(self) -> int:
        """Cantidad total de subscribers activos (todas las cámaras)."""
        return len(self._subscribers)

    def is_subscribed(self, subscriber_id: str) -> bool:
        return subscriber_id in self._subscribers

    # ---- Gestión de subscribers (API concreta, no Protocol) -----------

    async def subscribe(
        self, camera_id: str, queue_size: int = 50
    ) -> tuple[str, asyncio.Queue]:
        """Registra un subscriber para una `camera_id`. Retorna (subscriber_id, queue).

        Si hay `latest_state` cacheado para la cámara, se envía inmediatamente al
        nuevo subscriber.
        """
        subscriber_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        async with self._lock:
            self._subscribers[subscriber_id] = (camera_id, queue)

        latest = self._latest_state.get(camera_id)
        if latest is not None:
            try:
                queue.put_nowait(latest)
            except asyncio.QueueFull:
                pass

        return subscriber_id, queue

    async def unsubscribe(self, subscriber_id: str) -> None:
        async with self._lock:
            self._subscribers.pop(subscriber_id, None)

    # ---- Helpers públicos (sin acceso a privados desde presentation) --

    def subscribed_cameras(self) -> list[str]:
        """Camera IDs con al menos un subscriber activo (orden estable por sort)."""
        return sorted({cid for cid, _ in self._subscribers.values()})

    def latest_state(self, camera_id: str) -> dict | None:
        return self._latest_state.get(camera_id)

    def latest_states(self) -> dict[str, dict]:
        return dict(self._latest_state)

    def traffic_data_for(self, camera_id: str) -> dict[str, TrafficData]:
        """Últimos `TrafficData` por zona para una `camera_id`.

        Devuelve `{zone_id: TrafficData}`; vacío si nunca se publicó nada para
        esa cámara. Consumido por `GET /vision/state/{intersection_id}` (6e)
        para armar el shape §6.5 con todas las direcciones.
        """
        return dict(self._latest_traffic_data.get(camera_id, {}))

    # ---- Payload §6.2 --------------------------------------------------

    def _build_payload(self, data: TrafficData) -> dict:
        return {
            "schema_version": _SCHEMA_VERSION,
            "event_type": _EVENT_TYPE,
            "server_timestamp": datetime.now(timezone.utc).isoformat(),
            "camera": {
                "id": data.camera_id.value,
                "street_monitored": None,
            },
            "zone": {"id": data.zone_id.value},
            "window": {
                "start": data.window_start.isoformat(),
                "end": data.window_end.isoformat(),
                "duration_seconds": data.window_duration_seconds,
            },
            "metrics": {
                "unique_vehicles": data.unique_vehicles,
                "vehicles_by_type": dict(data.vehicles_by_type),
                "mean_speed_kmh": data.mean_speed_kmh,
                "flow_vehicles_per_hour": data.flow_vehicles_per_hour,
                "mean_occupancy": data.mean_occupancy,
                "density_vehicles_per_km": data.density_vehicles_per_km,
            },
        }
