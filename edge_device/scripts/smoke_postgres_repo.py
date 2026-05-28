"""Smoke e2e de PostgresTrafficRepository contra la Timescale viva.
Valida que save() inserta una fila real en vision_aggregates e idempota.
NO es parte del stage-gate de 4c (requiere Docker/BD); es validación manual.

Correr con la BD del docker compose levantada (invoke up) y el .venv activo:
  cd edge_device && PYTHONPATH=. python scripts/smoke_postgres_repo.py
"""
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from src.vision.domain.entities import TrafficData
from src.vision.domain.value_objects import CameraId, ZoneId
from src.vision.infrastructure.persistence.postgres_repository import (
    PostgresTrafficRepository,
)
from cerebrovial_shared.database import SessionLocal

# --- 1. TrafficData sintético que respeta las invariantes de __post_init__ ---
start = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
end = start + timedelta(seconds=60)
td = TrafficData(
    camera_id=CameraId(value="smoke_cam"),
    zone_id=ZoneId(value="smoke_zone"),
    window_start=start,
    window_end=end,
    window_duration_seconds=(end - start).total_seconds(),   # = 60.0
    unique_vehicles=5,
    vehicles_by_type={"car": 3, "bus": 1, "truck": 1},        # suma = 5 = unique_vehicles
    mean_speed_kmh=42.5,
    flow_vehicles_per_hour=5 / 60.0 * 3600,                   # = 300.0
    mean_occupancy=0.37,
    density_vehicles_per_km=None,
)

# --- 2. Persistir ---
repo = PostgresTrafficRepository()   # default SessionLocal → la BD del .env
repo.save(td)
print("save() OK — sin excepción")

# --- 3. Leer de vuelta y verificar ---
session = SessionLocal()
try:
    row = session.execute(
        text(
            "SELECT camera_id, zone_id, unique_vehicles, car_count, bus_count, "
            "truck_count, motorcycle_count, queue, density_vehicles_per_km "
            "FROM vision_aggregates "
            "WHERE camera_id = :cid AND zone_id = :zid AND window_start = :ws"
        ),
        {"cid": "smoke_cam", "zid": "smoke_zone", "ws": start},
    ).fetchone()
finally:
    session.close()

assert row is not None, "FALLO: no se insertó la fila"
assert row.unique_vehicles == 5
assert (row.car_count, row.bus_count, row.truck_count, row.motorcycle_count) == (3, 1, 1, 0)
assert row.queue is None, "queue debería ser NULL en MVP1"
assert row.density_vehicles_per_km is None
print(f"Fila verificada: {dict(row._mapping)}")

# --- 4. Idempotencia: segundo save() de la misma PK no duplica ni revienta ---
repo.save(td)
session = SessionLocal()
try:
    count = session.execute(
        text(
            "SELECT COUNT(*) FROM vision_aggregates "
            "WHERE camera_id = :cid AND zone_id = :zid AND window_start = :ws"
        ),
        {"cid": "smoke_cam", "zid": "smoke_zone", "ws": start},
    ).scalar()
finally:
    session.close()
assert count == 1, f"FALLO idempotencia: esperaba 1 fila, hay {count}"
print("Idempotencia OK — ON CONFLICT DO NOTHING honrado por la BD real")

# --- 5. Limpieza del dato de prueba ---
session = SessionLocal()
try:
    session.execute(
        text("DELETE FROM vision_aggregates WHERE camera_id = :cid"),
        {"cid": "smoke_cam"},
    )
    session.commit()
finally:
    session.close()
print("Limpieza OK — fila de prueba borrada")

print("\n✅ SMOKE PASÓ — el repo persiste e idempota contra Timescale viva")
