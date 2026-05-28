"""Persistencia de agregados de tráfico a PostgreSQL/TimescaleDB.

`PostgresTrafficRepository` escribe `TrafficData` a la hypertable
`vision_aggregates` (TTH-08 Fase 4c, CT-08.5), siguiendo el schema rico de
`tth-08-fase1-diseno.md` §6.5: desempaqueta `vehicles_by_type` a las cuatro
columnas de conteo y mapea los `datetime` tz-aware UTC del dominio a TIMESTAMPTZ.

`save()` es síncrono y bloqueante. El no-bloqueo de I/O es responsabilidad del
worker del aggregator (Fase 5), no de este repo. El repo es dueño de su propia
sesión y **ya commitea** la transacción; el `_flush_worker` de Fase 5 que lo
invoque no debe gestionar la transacción él mismo.
"""
from typing import Callable

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from cerebrovial_shared.database import SessionLocal
from cerebrovial_shared.database.models import VisionAggregateDB

from ...domain.entities import TrafficData
from ...domain.repositories import TrafficRepository

# PK compuesta de §6.5: la idempotencia ante reintentos del aggregator se delega
# a ON CONFLICT DO NOTHING sobre estas columnas.
_CONFLICT_KEYS = ["camera_id", "zone_id", "window_start"]


class PostgresTrafficRepository(TrafficRepository):
    """Repo write-only de agregados de tráfico contra `vision_aggregates`."""

    def __init__(
        self, session_factory: Callable[[], Session] = SessionLocal
    ) -> None:
        self._session_factory = session_factory

    def _to_row(self, data: TrafficData) -> dict:
        """Mapea TrafficData a las columnas §6.5. Puro: sin BD, sesión ni dialecto.

        `queue` se omite a propósito (NULL en MVP1; F41 lo poblará). Los Optional
        del dominio (`mean_speed_kmh`, `density_vehicles_per_km`) pasan tal cual:
        `None` se materializa como NULL en la columna nullable.
        """
        by_type = data.vehicles_by_type
        return {
            "camera_id": data.camera_id.value,
            "zone_id": data.zone_id.value,
            "window_start": data.window_start,
            "window_end": data.window_end,
            "window_duration_seconds": data.window_duration_seconds,
            "unique_vehicles": data.unique_vehicles,
            "car_count": by_type.get("car", 0),
            "bus_count": by_type.get("bus", 0),
            "truck_count": by_type.get("truck", 0),
            "motorcycle_count": by_type.get("motorcycle", 0),
            "mean_speed_kmh": data.mean_speed_kmh,
            "flow_vehicles_per_hour": data.flow_vehicles_per_hour,
            "mean_occupancy": data.mean_occupancy,
            "density_vehicles_per_km": data.density_vehicles_per_km,
        }

    def save(self, data: TrafficData) -> None:
        stmt = (
            pg_insert(VisionAggregateDB)
            .values(**self._to_row(data))
            .on_conflict_do_nothing(index_elements=_CONFLICT_KEYS)
        )
        session = self._session_factory()
        try:
            session.execute(stmt)
            session.commit()
        finally:
            session.close()
