"""e2e Fase A: la columna geom de intersections soporta una query espacial ST_DWithin.

Valida que el modelo geom (POINT 4326) corre una búsqueda por proximidad real en PostGIS,
aunque NO haya índice GIST (spatial_index=False — la fase no crea índices GIST). Mismo
patrón que el e2e del builder de geometría (TTH-12): Postgres/Timescale efímero con PostGIS
vía testcontainers, skip-graceful si no hay Docker. Marker `e2e`.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text


@pytest.fixture
def pg_engine():
    try:
        import docker

        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker no disponible — e2e Fase A requiere Docker ({exc!r})")

    from testcontainers.postgres import PostgresContainer

    from cerebrovial_shared.database.models import IntersectionDB

    container = PostgresContainer(
        image="timescale/timescaledb-ha:pg15",
        username="test",
        password="test",
        dbname="cerebrovial_test",
    )
    container.start()
    try:
        engine = create_engine(container.get_connection_url(), pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        IntersectionDB.__table__.create(engine)
        # Dos intersecciones reales del mapeo, separadas ~750 m.
        seed = [
            ("larco_benavides", "cluster_x", -12.12454, -77.029332),
            ("arequipa_angamos", "cluster_y", -12.113838, -77.030074),
        ]
        with engine.begin() as conn:
            for iid, jid, lat, lon in seed:
                conn.execute(
                    text(
                        "INSERT INTO intersections "
                        "(intersection_id, junction_id, lat, lon, geom) "
                        "VALUES (:iid, :jid, :lat, :lon, "
                        "ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))"
                    ),
                    {"iid": iid, "jid": jid, "lat": lat, "lon": lon},
                )
        yield engine
        engine.dispose()
    finally:
        container.stop()


@pytest.mark.e2e
def test_st_dwithin_alrededor_de_una_interseccion(pg_engine):
    # Punto a ~20 m de larco_benavides; radio 100 m (geography → metros).
    with pg_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT intersection_id FROM intersections "
                "WHERE ST_DWithin("
                "  geom::geography, "
                "  ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, "
                "  100)"
            ),
            {"lat": -12.12452, "lon": -77.029350},
        ).scalars().all()
    # Solo larco_benavides cae en el radio; arequipa_angamos (~750 m) queda fuera.
    assert rows == ["larco_benavides"]
