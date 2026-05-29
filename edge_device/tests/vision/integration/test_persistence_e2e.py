"""e2e CT-08.11(e): PostgresTrafficRepository contra Postgres/Timescale vivo.

Cierre del sub-test (e) de CT-08.11 nominado como deuda en el handoff de Fase 6
(`documentation/docs/tth-08-fase6-handoff.md` §1 y §4.3). Los otros cinco
sub-tests (a, b, c, d, f) ya están verdes desde Fases 4-6.

ALCANCE HONESTO (D-005):

- Lo que valida: repo ↔ modelo SQLAlchemy ↔ Postgres vivo. En concreto, sobre
  un Postgres/Timescale efímero levantado por `testcontainers`:
    * mapping de columnas del `_to_row()` (desempaque `vehicles_by_type` →
      `car/bus/truck/motorcycle_count`, preservación TIMESTAMPTZ tz-aware UTC,
      `queue IS NULL` por §5.7, Optionals → NULL);
    * idempotencia ON CONFLICT DO NOTHING sobre la PK compuesta;
    * CheckConstraints del modelo activos en la BD real.

- Lo que NO valida (declarado en el handoff de Fase 7):
    * paridad migración Alembic ↔ modelo SQLAlchemy — el test crea la tabla
      con `VisionAggregateDB.__table__.create()` desde el modelo, no corriendo
      la migración. Si modelo y migración divergen el test pasa y producción
      rompe. Deuda chica nominada para F9.x (test con
      `alembic.autogenerate.api.compare_metadata`);
    * pipeline e2e video → detección → zonas → agregación → TrafficData. Esa
      producción ya está verde por los sub-tests (a) detección, (b) asignación
      direccional, (c) derivación de métricas. F7 cierra el último gap
      repo↔BD-real, no el pipeline entero.

Skip-graceful si Docker no está disponible: el test no rompe la suite — queda
como `skipped` con mensaje claro. Marker `e2e` registrado en `pyproject.toml`
para evitar warnings de marker desconocido.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from cerebrovial_shared.database.models import VisionAggregateDB
from src.vision.domain.entities import TrafficData
from src.vision.domain.value_objects import CameraId, ZoneId
from src.vision.infrastructure.persistence.postgres_repository import (
    PostgresTrafficRepository,
)

pytestmark = pytest.mark.e2e

# Prefijo único para los datos sintéticos: facilita el DELETE de limpieza por test
# sin riesgo de tocar datos de otros tests si el container se reusara entre sesiones.
_TEST_CAMERA_PREFIX = "cam_e2e_test"

# Ventana temporal sintética tz-aware UTC (lo que el dominio exige).
_WINDOW_START = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
_WINDOW_END = _WINDOW_START + timedelta(seconds=60)


@pytest.fixture(scope="session")
def timescale_engine():
    """Levanta un Postgres/Timescale efímero y crea la tabla del modelo.

    Usa la misma imagen que docker-compose (`timescale/timescaledb-ha:pg15`)
    para no introducir divergencia silenciosa de versión, aunque el repo NO
    depende de hypertable-behavior (INSERT/SELECT ORM es transparente sobre
    hypertables). Crea solo `vision_aggregates` desde el modelo SQLAlchemy
    saltando la hipertabilización: la perf-optimization no cambia el contrato
    que CT-08.11(e) verifica.
    """
    try:
        import docker

        docker.from_env().ping()
    except Exception as exc:
        pytest.skip(f"Docker no disponible — e2e CT-08.11(e) requiere Docker ({exc!r})")

    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        image="timescale/timescaledb-ha:pg15",
        username="test",
        password="test",
        dbname="cerebrovial_test",
    )
    container.start()
    try:
        engine = create_engine(container.get_connection_url(), pool_pre_ping=True)
        VisionAggregateDB.__table__.create(engine)
        yield engine
        engine.dispose()
    finally:
        container.stop()


@pytest.fixture
def session_factory(timescale_engine):
    """Session factory atada al container; sessionmaker fresh por test."""
    return sessionmaker(bind=timescale_engine, autocommit=False, autoflush=False)


@pytest.fixture
def repo(session_factory) -> PostgresTrafficRepository:
    """Repo bajo test con el session_factory del container inyectado."""
    return PostgresTrafficRepository(session_factory=session_factory)


@pytest.fixture(autouse=True)
def cleanup_test_rows(timescale_engine):
    """Borra todas las filas sintéticas insertadas por el test al terminar.

    Function-scope autouse: corre alrededor de CADA test. El DELETE filtra por
    el prefijo del test (`cam_e2e_test*`) para que el cleanup sea preciso si
    el container quedara vivo entre sesiones por algún motivo.
    """
    yield
    with timescale_engine.begin() as conn:
        conn.execute(
            text("DELETE FROM vision_aggregates WHERE camera_id LIKE :prefix"),
            {"prefix": f"{_TEST_CAMERA_PREFIX}%"},
        )


def _make_traffic_data(
    *,
    camera_id: str = f"{_TEST_CAMERA_PREFIX}_default",
    zone_id: str = "zone_a",
    vehicles_by_type: dict[str, int] | None = None,
    unique_vehicles: int = 4,
    mean_speed_kmh: float | None = 42.5,
    flow_vehicles_per_hour: float = 240.0,
    mean_occupancy: float = 0.35,
    density_vehicles_per_km: float | None = 120.0,
) -> TrafficData:
    """Factory de TrafficData sintético respetando las invariantes del dominio."""
    if vehicles_by_type is None:
        # 4 unique: 3 cars + 1 bus; motorcycle/truck AUSENTES a propósito para
        # ejercitar el caso `.get("motorcycle", 0)` del _to_row().
        vehicles_by_type = {"car": 3, "bus": 1}
    return TrafficData(
        camera_id=CameraId(camera_id),
        zone_id=ZoneId(zone_id),
        window_start=_WINDOW_START,
        window_end=_WINDOW_END,
        window_duration_seconds=(_WINDOW_END - _WINDOW_START).total_seconds(),
        unique_vehicles=unique_vehicles,
        vehicles_by_type=vehicles_by_type,
        mean_speed_kmh=mean_speed_kmh,
        flow_vehicles_per_hour=flow_vehicles_per_hour,
        mean_occupancy=mean_occupancy,
        density_vehicles_per_km=density_vehicles_per_km,
    )


def _select_row(timescale_engine, *, camera_id: str, zone_id: str):
    """Lee la fila persistida; retorna mapping o None."""
    with timescale_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT camera_id, zone_id, window_start, window_end, "
                "window_duration_seconds, unique_vehicles, "
                "car_count, bus_count, truck_count, motorcycle_count, "
                "mean_speed_kmh, flow_vehicles_per_hour, mean_occupancy, "
                "density_vehicles_per_km, queue "
                "FROM vision_aggregates "
                "WHERE camera_id = :cid AND zone_id = :zid AND window_start = :ws"
            ),
            {"cid": camera_id, "zid": zone_id, "ws": _WINDOW_START},
        ).mappings().first()
    return row


def test_save_persiste_traffic_data_con_shape_canonico(repo, timescale_engine):
    """Happy-path: el TrafficData aterriza con el shape §6.5 del contrato.

    Verifica el mapping completo de _to_row(): desempaque de vehicles_by_type
    a las cuatro columnas (incluyendo `motorcycle_count = 0` por `.get()`),
    preservación TIMESTAMPTZ tz-aware UTC, y `queue IS NULL` por §5.7.
    """
    camera_id = f"{_TEST_CAMERA_PREFIX}_canonical"
    data = _make_traffic_data(camera_id=camera_id)

    repo.save(data)

    row = _select_row(timescale_engine, camera_id=camera_id, zone_id="zone_a")
    assert row is not None, "save() no insertó la fila"

    assert row["camera_id"] == camera_id
    assert row["zone_id"] == "zone_a"

    # TIMESTAMPTZ: la zona UTC se preserva. El driver psycopg2 hidrata como
    # datetime tz-aware.
    assert row["window_start"] == _WINDOW_START
    assert row["window_end"] == _WINDOW_END
    assert row["window_start"].tzinfo is not None
    assert row["window_duration_seconds"] == pytest.approx(60.0)

    # Counts: desempaque vehicles_by_type con .get(default=0) para tipos ausentes.
    assert row["unique_vehicles"] == 4
    assert row["car_count"] == 3
    assert row["bus_count"] == 1
    assert row["truck_count"] == 0
    assert row["motorcycle_count"] == 0

    # Numéricos no-nullable.
    assert row["mean_speed_kmh"] == pytest.approx(42.5)
    assert row["flow_vehicles_per_hour"] == pytest.approx(240.0)
    assert row["mean_occupancy"] == pytest.approx(0.35)
    assert row["density_vehicles_per_km"] == pytest.approx(120.0)

    # Invariante §5.7: queue se omite en _to_row() y la columna queda NULL.
    assert row["queue"] is None


def test_save_es_idempotente_sobre_pk_compuesta(repo, timescale_engine):
    """Segundo save() del mismo TrafficData no duplica ni levanta excepción.

    El unit test (test_postgres_repository.py) valida ON CONFLICT DO NOTHING a
    nivel de SQL compilado; este e2e lo ejecuta contra una BD real.
    """
    camera_id = f"{_TEST_CAMERA_PREFIX}_idempotent"
    data = _make_traffic_data(camera_id=camera_id)

    repo.save(data)
    repo.save(data)  # misma PK = (camera_id, zone_id, window_start)

    with timescale_engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM vision_aggregates "
                "WHERE camera_id = :cid AND zone_id = :zid AND window_start = :ws"
            ),
            {"cid": camera_id, "zid": "zone_a", "ws": _WINDOW_START},
        ).scalar()

    assert count == 1, f"ON CONFLICT DO NOTHING no honrado: hay {count} filas"


def test_save_con_optionals_none_persiste_null(repo, timescale_engine):
    """Optionals None del dominio → NULL en columnas nullable de la BD."""
    camera_id = f"{_TEST_CAMERA_PREFIX}_nulls"
    data = _make_traffic_data(
        camera_id=camera_id,
        mean_speed_kmh=None,
        density_vehicles_per_km=None,
    )

    repo.save(data)

    row = _select_row(timescale_engine, camera_id=camera_id, zone_id="zone_a")
    assert row is not None
    assert row["mean_speed_kmh"] is None
    assert row["density_vehicles_per_km"] is None
    # queue sigue NULL — el repo no lo emite ni cuando los otros Optionals son None.
    assert row["queue"] is None


def test_check_constraint_mean_occupancy_activo_en_bd(timescale_engine):
    """`CheckConstraint('mean_occupancy BETWEEN 0 AND 1')` declarado en el modelo
    está activo en Postgres.

    El dominio TrafficData ya valida en `__post_init__`; este test bypasea el
    dominio insertando vía Table.insert() directo para verificar que el
    constraint del SCHEMA también atrapa la violación. **Claim acotado**:
    protege contra "modelo dice X, Postgres no lo enforza". NO protege contra
    "migración Alembic diverge del modelo" (deuda F9.x).
    """
    SessionMaker = sessionmaker(bind=timescale_engine, autocommit=False, autoflush=False)
    session = SessionMaker()
    try:
        with pytest.raises(IntegrityError):
            session.execute(
                VisionAggregateDB.__table__.insert().values(
                    camera_id=f"{_TEST_CAMERA_PREFIX}_constraint",
                    zone_id="zone_a",
                    window_start=_WINDOW_START,
                    window_end=_WINDOW_END,
                    window_duration_seconds=60.0,
                    unique_vehicles=1,
                    car_count=1,
                    bus_count=0,
                    truck_count=0,
                    motorcycle_count=0,
                    flow_vehicles_per_hour=60.0,
                    mean_occupancy=1.5,  # fuera de rango — el constraint debe rechazar
                )
            )
            session.commit()
    finally:
        session.close()
