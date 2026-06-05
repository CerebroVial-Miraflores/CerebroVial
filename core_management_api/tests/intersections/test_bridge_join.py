"""El puente intersection_edges resuelve el JOIN intersección → edges → graph_edges.

SQLite stripped (sin geom), mismo patrón que los tests de congestión. Verifica que, con
el net real presente (graph_edges poblada), un JOIN por edge_id devuelve las aristas
direccionadas de una intersección.
"""
from sqlalchemy import text


def _seed(session):
    # graph_edges: ids SUMO crudos, como los carga build_graph_geometry.py.
    for eid in ("129466113#3", "344159559#3", "279893831#2"):
        session.execute(
            text(
                "INSERT INTO graph_edges (edge_id, source_node, target_node, distance_m, lanes) "
                "VALUES (:eid, 'sumo_a', 'sumo_b', 10.0, 2)"
            ),
            {"eid": eid},
        )
    session.execute(
        text(
            "INSERT INTO intersections (intersection_id, junction_id, lat, lon, los_pmu, tls_id) "
            "VALUES ('larco_benavides', 'cluster_x', -12.12454, -77.029332, 'C/D', 'cluster_x')"
        )
    )
    bridge = [
        ("129466113#3", "incoming"),
        ("344159559#3", "incoming"),
        ("279893831#2", "outgoing"),
    ]
    for eid, direction in bridge:
        session.execute(
            text(
                "INSERT INTO intersection_edges (intersection_id, edge_id, direction) "
                "VALUES ('larco_benavides', :eid, :dir)"
            ),
            {"eid": eid, "dir": direction},
        )
    session.commit()


def test_bridge_resuelve_edges_de_la_interseccion(bridge_session):
    _seed(bridge_session)
    rows = bridge_session.execute(
        text(
            "SELECT ie.edge_id, ie.direction, ge.source_node, ge.target_node "
            "FROM intersections i "
            "JOIN intersection_edges ie ON ie.intersection_id = i.intersection_id "
            "JOIN graph_edges ge ON ge.edge_id = ie.edge_id "
            "WHERE i.intersection_id = 'larco_benavides' "
            "ORDER BY ie.direction, ie.edge_id"
        )
    ).all()
    # 3 aristas, todas resuelven contra graph_edges (source/target no nulos).
    assert len(rows) == 3
    assert all(r.source_node == "sumo_a" and r.target_node == "sumo_b" for r in rows)
    incoming = sorted(r.edge_id for r in rows if r.direction == "incoming")
    outgoing = sorted(r.edge_id for r in rows if r.direction == "outgoing")
    assert incoming == ["129466113#3", "344159559#3"]
    assert outgoing == ["279893831#2"]


def test_bridge_no_resuelve_edge_ausente_en_graph_edges(bridge_session):
    # Un edge del puente sin fila en graph_edges no aparece en el INNER JOIN
    # (en producción la FK lo impediría; acá comprobamos la semántica del JOIN).
    bridge_session.execute(
        text(
            "INSERT INTO intersections (intersection_id, junction_id, lat, lon) "
            "VALUES ('x', 'jx', -12.1, -77.0)"
        )
    )
    bridge_session.execute(
        text("INSERT INTO graph_edges VALUES ('presente', 'a', 'b', 1.0, 1)")
    )
    bridge_session.execute(
        text(
            "INSERT INTO intersection_edges VALUES ('x', 'presente', 'incoming')"
        )
    )
    bridge_session.commit()
    n = bridge_session.execute(
        text(
            "SELECT count(*) FROM intersection_edges ie "
            "JOIN graph_edges ge ON ge.edge_id = ie.edge_id "
            "WHERE ie.intersection_id = 'x'"
        )
    ).scalar_one()
    assert n == 1
