"""Unit del seed de intersecciones (Fase A): build_rows() sobre el mapeo canónico.

Función pura, sin BD: valida exclusión de ovalo_gutierrez, las 11 semaforizadas, tls_id
solo para larco_benavides (DEUDA-CTRL-TLS), mapeo de los_pmu (incl. nulls), conteo de
edges por dirección y la asignación 1:1 de cámaras Claro (DEUDA-CAM-GEO).
"""
import seed_intersections as si


def _rows():
    return si.build_rows(si.load_mapeo())


def test_11_intersecciones_semaforizadas_sin_ovalo_gutierrez():
    rows = _rows()
    ids = [it["intersection_id"] for it in rows["intersections"]]
    assert len(ids) == 11
    assert "ovalo_gutierrez" not in ids
    assert set(ids) == {
        "larco_benavides", "ovalo_miraflores", "pardo_espinar", "arequipa_angamos",
        "espinar_angamos", "28julio_reducto", "28julio_lapaz", "ricardopalma_paseo",
        "paseo_angamos", "paseo_benavides", "benavides_panama",
    }


def test_tls_id_solo_larco_benavides():
    by_id = {it["intersection_id"]: it for it in _rows()["intersections"]}
    # larco_benavides: tls_id == su junction_id (verificado contra corredor_adaptive.py).
    lb = by_id["larco_benavides"]
    assert lb["tls_id"] == lb["junction_id"]
    assert lb["tls_id"] == "cluster_108178119_263630444_2673400749_3245705958_#6more"
    # arequipa_angamos y las otras 9: NULL (no metemos tls_id sin verificar).
    assert by_id["arequipa_angamos"]["tls_id"] is None
    n_con_tls = sum(1 for it in _rows()["intersections"] if it["tls_id"] is not None)
    assert n_con_tls == 1


def test_los_pmu_mapeado_incluye_nulls():
    by_id = {it["intersection_id"]: it for it in _rows()["intersections"]}
    assert by_id["larco_benavides"]["los_pmu"] == "C/D"
    assert by_id["arequipa_angamos"]["los_pmu"] == "E/H"
    # nivel_los null en el mapeo → los_pmu None (no se excluye la intersección).
    assert by_id["ovalo_miraflores"]["los_pmu"] is None
    assert by_id["benavides_panama"]["los_pmu"] is None


def test_edges_por_direccion_coinciden_con_n_edges_del_mapeo():
    mapeo = si.load_mapeo()
    n_edges_yaml = {
        it["nombre"]: it["n_edges"]
        for it in mapeo["intersecciones"]
        if it["nombre"] != "ovalo_gutierrez"
    }
    edges = _rows()["edges"]
    # toda fila tiene dirección válida
    assert all(e["direction"] in ("incoming", "outgoing") for e in edges)
    # por intersección, incoming+outgoing == n_edges del mapeo
    por_int: dict[str, int] = {}
    for e in edges:
        por_int[e["intersection_id"]] = por_int.get(e["intersection_id"], 0) + 1
    assert por_int == n_edges_yaml


def test_camaras_1a1_con_stream_url_claro():
    rows = _rows()
    cams = rows["cameras"]
    inter_by_id = {it["intersection_id"]: it for it in rows["intersections"]}
    assert len(cams) == 11
    for c in cams:
        assert c["camera_id"] == f"cam_{c['intersection_id']}"
        assert c["stream_url"].startswith("https://live.smartechlatam.online/claro/")
        assert c["stream_url"].endswith("/index.m3u8")
        # lat/lon de la cámara == coord de su intersección (nominal, DEUDA-CAM-GEO)
        inter = inter_by_id[c["intersection_id"]]
        assert (c["lat"], c["lon"]) == (inter["lat"], inter["lon"])
    # asignación 1:1: 11 stream_url distintos
    assert len({c["stream_url"] for c in cams}) == 11
