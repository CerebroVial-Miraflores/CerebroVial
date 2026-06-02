"""e2e CT-12.9(a): el builder de geometría carga las 375 aristas con geom válida.

Cierra el único sub-ítem de CT-12.9 que estaba verificado por ejecución pero sin
test de regresión: la extracción/carga desde el `.net.xml` produce el número esperado
de aristas con geometría válida (y sin tocar los nodos de control).

El builder (`scripts/build_graph_geometry.py`) depende de PostGIS (ST_Transform,
columnas Geometry) — no corre en SQLite. Igual que el e2e de TTH-08
(`edge_device/.../test_persistence_e2e.py`), se levanta un Postgres/Timescale efímero
con `testcontainers` (misma imagen que docker-compose, trae PostGIS) y se CORRE el
builder real contra él, sin re-implementar su lógica. Skip-graceful si no hay Docker.
Marker `e2e` (registrado en pyproject.toml).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

_REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def builder_engine(monkeypatch):
    """Postgres/Timescale efímero con PostGIS + esquema mínimo, y el builder apuntado a él."""
    try:
        import docker

        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker no disponible — e2e CT-12.9(a) requiere Docker ({exc!r})")

    from testcontainers.postgres import PostgresContainer

    from cerebrovial_shared.database.models import (
        GraphEdgeDB,
        GraphNodeDB,
        WazeAlertDB,
        WazeJamDB,
    )

    container = PostgresContainer(
        image="timescale/timescaledb-ha:pg15",
        username="test",
        password="test",
        dbname="cerebrovial_test",
    )
    container.start()
    try:
        url = container.get_connection_url()
        engine = create_engine(url, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        # Esquema en orden por FK: nodes ← edges ← (waze_jams, waze_alerts).
        GraphNodeDB.__table__.create(engine)
        GraphEdgeDB.__table__.create(engine)
        WazeJamDB.__table__.create(engine)
        WazeAlertDB.__table__.create(engine)
        # Sembrar 5 nodos de CONTROL (los que el builder NUNCA debe tocar).
        controls = [
            ("larco_schell", -12.1195, -77.0298),
            ("larco_benavides", -12.1227, -77.0301),
            ("benavides_miraflores", -12.1229, -77.0283),
            ("arequipa_angamos", -12.1158, -77.0239),
            ("ejercito_sucre", -12.1099, -77.0578),
        ]
        with engine.begin() as conn:
            for nid, lat, lon in controls:
                conn.execute(
                    text("INSERT INTO graph_nodes (node_id, lat, lon, has_camera, geom) "
                         "VALUES (:nid, :lat, :lon, false, "
                         "ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))"),
                    {"nid": nid, "lat": lat, "lon": lon},
                )
        # Apuntar el builder a este contenedor.
        monkeypatch.setenv("DATABASE_URL", url)
        yield engine
        engine.dispose()
    finally:
        container.stop()


def _run_builder() -> None:
    scripts_dir = str(_REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import build_graph_geometry  # noqa: WPS433

    build_graph_geometry.main()


@pytest.mark.e2e
def test_builder_loads_375_edges_with_valid_geometry(builder_engine):
    _run_builder()

    with builder_engine.connect() as conn:
        # 375 aristas, todas formato SUMO (no prefijo sumo_ ni nombre sintético).
        assert conn.execute(text("SELECT count(*) FROM graph_edges")).scalar_one() == 375
        assert conn.execute(text(
            "SELECT count(*) FROM graph_edges WHERE edge_id !~ '^sumo_'"
        )).scalar_one() == 375
        # 0 geom NULL; geometría LINESTRING en SRID 4326.
        assert conn.execute(text(
            "SELECT count(*) FROM graph_edges WHERE geom IS NULL"
        )).scalar_one() == 0
        gtypes = conn.execute(text(
            "SELECT DISTINCT ST_GeometryType(geom), ST_SRID(geom) FROM graph_edges"
        )).all()
        assert gtypes == [("ST_LineString", 4326)]
        # graph_nodes: 5 control intactos + 307 sumo_.
        assert conn.execute(text(
            "SELECT count(*) FROM graph_nodes WHERE node_id !~ '^sumo_'"
        )).scalar_one() == 5
        assert conn.execute(text(
            "SELECT count(*) FROM graph_nodes WHERE node_id ~ '^sumo_'"
        )).scalar_one() == 307
        # 0 FK huérfanas: source/target de cada arista resuelve a un nodo existente.
        assert conn.execute(text(
            "SELECT count(*) FROM graph_edges e "
            "WHERE NOT EXISTS (SELECT 1 FROM graph_nodes n WHERE n.node_id = e.source_node) "
            "   OR NOT EXISTS (SELECT 1 FROM graph_nodes n WHERE n.node_id = e.target_node)"
        )).scalar_one() == 0


@pytest.mark.e2e
def test_builder_is_idempotent_and_keeps_control_intact(builder_engine):
    _run_builder()
    _run_builder()  # segunda corrida: no debe duplicar ni tocar los nodos de control
    with builder_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM graph_edges")).scalar_one() == 375
        assert conn.execute(text(
            "SELECT count(*) FROM graph_nodes WHERE node_id ~ '^sumo_'"
        )).scalar_one() == 307
        assert conn.execute(text(
            "SELECT count(*) FROM graph_nodes WHERE node_id !~ '^sumo_'"
        )).scalar_one() == 5
