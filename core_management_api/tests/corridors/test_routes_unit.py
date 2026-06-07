"""Tests del endpoint ``POST /corridors`` (Fase B-2), SQLite.

Cubren validación (4xx, sin persistir cadena rota), rol (require_role OPERATOR/ADMIN) y
persistencia. El matching GEOMÉTRICO se omite acá (el guard de dialecto devuelve overlaps
vacíos en SQLite → ``tomtom_openlr`` queda ``None``); su correctitud se valida en el e2e
PostGIS. Acá se prueba que el armado, las validaciones, el rol y el guardado funcionan.
"""
from sqlalchemy import text as sa_text


def _segments():
    return [{"openlr": "OLR_X", "coordinates": [[-77.0300, -12.1205], [-77.0300, -12.1215]]}]


def _payload(edges, name="Corredor Larco test", segments=None):
    return {
        "name": name,
        "edges": edges,
        "segments": _segments() if segments is None else segments,
    }


def _count(db_session, table):
    return db_session.execute(sa_text(f"SELECT COUNT(*) FROM {table}")).scalar()


def test_crea_corredor_operator_201(client, db_session, auth_headers):
    payload = _payload([{"edge_id": "e1", "sequence": 0}, {"edge_id": "e2", "sequence": 1}])
    resp = client.post("/corridors", json=payload, headers=auth_headers("operator"))

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["count"] == 2
    assert {e["edge_id"] for e in body["edges"]} == {"e1", "e2"}
    # Sin PostGIS no hay matching geométrico → todas las aristas quedan sin cobertura.
    assert all(e["tomtom_openlr"] is None for e in body["edges"])
    # Persistido: 1 corredor + 2 corridor_edges.
    assert _count(db_session, "corridors") == 1
    assert _count(db_session, "corridor_edges") == 2


def test_crea_corredor_admin_201(client, auth_headers):
    payload = _payload([{"edge_id": "e1", "sequence": 0}])
    resp = client.post("/corridors", json=payload, headers=auth_headers("admin"))
    assert resp.status_code == 201, resp.text


def test_rol_no_autorizado_403(client, auth_headers):
    payload = _payload([{"edge_id": "e1", "sequence": 0}])
    resp = client.post("/corridors", json=payload, headers=auth_headers("manager"))
    assert resp.status_code == 403


def test_sin_token_401(client):
    payload = _payload([{"edge_id": "e1", "sequence": 0}])
    resp = client.post("/corridors", json=payload)
    assert resp.status_code == 401


def test_cadena_rota_422_no_persiste(client, db_session, auth_headers):
    # e_broken arranca en nx, no en n2 → cadena discontinua.
    payload = _payload(
        [{"edge_id": "e1", "sequence": 0}, {"edge_id": "e_broken", "sequence": 1}]
    )
    resp = client.post("/corridors", json=payload, headers=auth_headers("operator"))
    assert resp.status_code == 422
    assert "cadena rota" in resp.json()["detail"]
    # NADA persistido: ni corredor ni aristas.
    assert _count(db_session, "corridors") == 0
    assert _count(db_session, "corridor_edges") == 0


def test_sequence_con_huecos_422(client, auth_headers):
    payload = _payload([{"edge_id": "e1", "sequence": 0}, {"edge_id": "e2", "sequence": 2}])
    resp = client.post("/corridors", json=payload, headers=auth_headers("operator"))
    assert resp.status_code == 422
    assert "hueco" in resp.json()["detail"]


def test_arista_inexistente_422(client, auth_headers):
    payload = _payload([{"edge_id": "no_existe", "sequence": 0}])
    resp = client.post("/corridors", json=payload, headers=auth_headers("operator"))
    assert resp.status_code == 422
    assert "inexistentes" in resp.json()["detail"]
