"""e2e Fase B-2: matching geométrico edge→OpenLR contra PostGIS real.

Prueba la pieza que SQLite no puede: el overlap geométrico (``ST_Buffer``/``ST_Intersection``
en geography) + la desambiguación por SENTIDO. Molde: ``tests/intersections/test_spatial_e2e.py``
(Postgres/Timescale efímero con PostGIS vía testcontainers, skip-graceful sin Docker, marker
``e2e``).

Escenario (stand-in sintético y controlado de las dos calzadas opuestas de 406008845#1 — ver
nota al pie): una avenida E-O con dos calzadas de sentidos OPUESTOS separadas ~11 m. El buffer
de 15 m de CADA segmento TomTom cubre AMBAS calzadas, así que el overlap por sí solo es
ambiguo: SÓLO el sentido resuelve cuál es cuál. Se verifica que cada calzada matchea el OpenLR
de SU sentido y NO el del opuesto, y que una arista fuera de cobertura → ``None``.

Nota: se usa geometría sintética (coordenadas explícitas) en vez de las calzadas reales de
406008845#1 para que el test sea autocontenido (no depende de cargar el net SUMO). Cubre la
MISMA propiedad lógica —distinguir ida de vuelta por sentido— que el caso real sugerido en el
plan. Un e2e contra la geometría real sería un follow-up de calibración si se quisiera.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.corridors.application.matching import (
    CorridorEdgeInput,
    TomTomSegment,
    match_corridor,
)
from src.corridors.infrastructure.repositories import CorridorMatchingRepo

# Calzada EASTBOUND (sur, lat -12.12005) y WESTBOUND (norte, lat -12.11995): ~11 m de
# separación; ambos dentro del buffer de 15 m de cualquiera de los dos segmentos.
_EB_LAT = -12.12005
_WB_LAT = -12.11995
_LON_W = -77.0400
_LON_E = -77.0300

_NODES = [
    ("n_w_eb", _EB_LAT, _LON_W),
    ("n_e_eb", _EB_LAT, _LON_E),
    ("n_e_wb", _WB_LAT, _LON_E),
    ("n_w_wb", _WB_LAT, _LON_W),
    ("n_far1", -12.13000, -77.0500),
    ("n_far2", -12.13000, -77.0490),
]
# (edge_id, source, target, wkt) — wkt en orden source→target (sentido del grafo dirigido).
_EDGES = [
    ("av_eb", "n_w_eb", "n_e_eb", f"LINESTRING({_LON_W} {_EB_LAT}, {_LON_E} {_EB_LAT})"),
    ("av_wb", "n_e_wb", "n_w_wb", f"LINESTRING({_LON_E} {_WB_LAT}, {_LON_W} {_WB_LAT})"),
    ("far", "n_far1", "n_far2", "LINESTRING(-77.0500 -12.13000, -77.0490 -12.13000)"),
]


@pytest.fixture
def pg_session():
    try:
        import docker

        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker no disponible — e2e Fase B-2 requiere Docker ({exc!r})")

    from testcontainers.postgres import PostgresContainer

    from cerebrovial_shared.database.models import (
        CorridorDB,
        CorridorEdgeDB,
        GraphEdgeDB,
        GraphNodeDB,
    )

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
        # Orden por FKs: nodos → aristas → corredores → puente.
        GraphNodeDB.__table__.create(engine)
        GraphEdgeDB.__table__.create(engine)
        CorridorDB.__table__.create(engine)
        CorridorEdgeDB.__table__.create(engine)
        with engine.begin() as conn:
            for node_id, lat, lon in _NODES:
                conn.execute(
                    text(
                        "INSERT INTO graph_nodes (node_id, lat, lon, has_camera) "
                        "VALUES (:n, :lat, :lon, false)"
                    ),
                    {"n": node_id, "lat": lat, "lon": lon},
                )
            for edge_id, src, tgt, wkt in _EDGES:
                conn.execute(
                    text(
                        "INSERT INTO graph_edges "
                        "(edge_id, source_node, target_node, distance_m, lanes, geom) "
                        "VALUES (:e, :s, :t, 1000.0, 2, ST_GeomFromText(:wkt, 4326))"
                    ),
                    {"e": edge_id, "s": src, "t": tgt, "wkt": wkt},
                )
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()
            engine.dispose()
    finally:
        container.stop()


def _edge_inputs(session):
    repo = CorridorMatchingRepo(session)
    rows = {r.edge_id: r for r in repo.fetch_edges(["av_eb", "av_wb", "far"])}
    return repo, [
        CorridorEdgeInput(
            edge_id=eid,
            sequence=i,
            source_coord=(rows[eid].src_lon, rows[eid].src_lat),
            target_coord=(rows[eid].tgt_lon, rows[eid].tgt_lat),
        )
        for i, eid in enumerate(["av_eb", "av_wb", "far"])
    ]


# Segmentos TomTom: EB (oeste→este, sobre la calzada sur) y WB (este→oeste, calzada norte).
_SEG_EB = TomTomSegment(
    openlr="OLR_EB",
    coordinates=[(_LON_W, _EB_LAT), (-77.0350, -12.120049), (_LON_E, _EB_LAT)],
)
_SEG_WB = TomTomSegment(
    openlr="OLR_WB",
    coordinates=[(_LON_E, _WB_LAT), (-77.0350, -12.119951), (_LON_W, _WB_LAT)],
)


@pytest.mark.e2e
def test_cada_calzada_matchea_su_sentido_no_el_opuesto(pg_session):
    """El overlap es ambiguo (ambos buffers cubren ambas calzadas); el SENTIDO decide."""
    repo, edges = _edge_inputs(pg_session)
    mapping = match_corridor(repo, edges, [_SEG_EB, _SEG_WB])

    assert mapping["av_eb"] == "OLR_EB"   # eastbound → su OpenLR, NO el opuesto
    assert mapping["av_wb"] == "OLR_WB"   # westbound → su OpenLR, NO el opuesto
    assert mapping["far"] is None         # fuera de cobertura → sin OpenLR


@pytest.mark.e2e
def test_persistencia_transaccional_solo_ids(pg_session):
    """Persiste el corredor + sus aristas con el openlr correcto; NADA de geometría TomTom."""
    repo, edges = _edge_inputs(pg_session)
    mapping = match_corridor(repo, edges, [_SEG_EB, _SEG_WB])
    ordered = [(e.edge_id, e.sequence, mapping[e.edge_id]) for e in edges]

    corridor_id = repo.persist("Av. E-O test", "tester@cv.pe", ordered)
    pg_session.commit()

    rows = pg_session.execute(
        text(
            "SELECT edge_id, sequence, tomtom_openlr FROM corridor_edges "
            "WHERE corridor_id = :cid ORDER BY sequence"
        ),
        {"cid": corridor_id},
    ).all()
    assert [(r.edge_id, r.tomtom_openlr) for r in rows] == [
        ("av_eb", "OLR_EB"),
        ("av_wb", "OLR_WB"),
        ("far", None),
    ]
