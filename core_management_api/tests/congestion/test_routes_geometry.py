"""Tests del endpoint de geometría de red (TTH-12 Fase 3, CT-12.2 / CT-12.9 b).

La geometría real usa ST_AsGeoJSON (PostGIS), no disponible en SQLite: el shape se
verifica monkeypatcheando el repo con features canónicos; el conteo real (375) se
verifica contra PostGIS en la VERIFICACIÓN de la fase. El rol se prueba de verdad.
"""
import src.congestion.presentation.api.routes as routes_mod

_FAKE_FEATURES = [
    {"type": "Feature",
     "geometry": {"type": "LineString", "coordinates": [[-77.033, -12.118], [-77.033, -12.116]]},
     "properties": {"edge_id": "-129822384#0", "source_node": "sumo_138854736",
                    "target_node": "sumo_262576671", "distance_m": 241.2, "lanes": 1}},
    {"type": "Feature",
     "geometry": {"type": "LineString", "coordinates": [[-77.034, -12.119], [-77.033, -12.118]]},
     "properties": {"edge_id": "-129822384#1", "source_node": "sumo_6122459600",
                    "target_node": "sumo_138854736", "distance_m": 44.0, "lanes": 1}},
]


def _patch_geometry(monkeypatch):
    monkeypatch.setattr(
        routes_mod.NetworkGeometryRepo, "network_features",
        lambda self: _FAKE_FEATURES,
    )


def test_geometry_requires_role(congestion_client):
    congestion_client.as_role("manager")  # no está en (operator, admin)
    assert congestion_client.get("/congestion/geometry").status_code == 403


def test_geometry_ok_shape_geojson(congestion_client, monkeypatch):
    _patch_geometry(monkeypatch)
    congestion_client.as_role("operator")
    resp = congestion_client.get("/congestion/geometry")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert body["count"] == 2 == len(body["features"])
    f0 = body["features"][0]
    assert f0["type"] == "Feature"
    assert f0["geometry"]["type"] == "LineString"
    assert "edge_id" in f0["properties"]
    # GeoJSON: orden [lon, lat]
    lon, lat = f0["geometry"]["coordinates"][0]
    assert -77.05 < lon < -77.02 and -12.13 < lat < -12.10


def test_geometry_admin_also_allowed(congestion_client, monkeypatch):
    _patch_geometry(monkeypatch)
    congestion_client.as_role("admin")
    assert congestion_client.get("/congestion/geometry").status_code == 200
