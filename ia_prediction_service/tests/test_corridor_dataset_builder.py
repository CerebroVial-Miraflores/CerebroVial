"""Tests del dataset builder por-arista del corredor (corridor_dataset_builder.py).

Verifica: free-flow aplicado POR ARISTA (nodo0 vs nodo4, no global), cap del vType,
ratio/jam_level correctos para velocidades conocidas, y bucket vacío → jam 0.
Corre con el venv de training: `cd ia_prediction_service && .venv/bin/python -m pytest tests/...`.
"""
from __future__ import annotations

import math

import numpy as np

from src.data.corridor_dataset_builder import build_corridor_series, parse_freeflow


# Los 6 edges instrumentados (orden graph_index del corridor_graph_mapping.json real).
NODE0 = "129466113#0"   # Larco Sur — carril 16.67 → free-flow capado a 13.89
NODE4 = "430180649"     # Schell Este — carril 11.11 → free-flow 11.11
EDGES = [NODE0, "344159559#2", "279893875#1", "279893875#2", NODE4, "279893875#4"]


def _write_net(tmp_path):
    """net.xml mínimo: los 6 edges con speeds de carril + un edge extra a filtrar."""
    speeds = {
        NODE0: "16.67",        # > maxSpeed → debe capar a 13.89
        "344159559#2": "13.89",
        "279893875#1": "13.89",
        "279893875#2": "13.89",
        NODE4: "11.11",
        "279893875#4": "13.89",
        "X_extra": "20.0",     # no instrumentado → ignorado por parse_freeflow
    }
    edges_xml = "\n".join(
        f'  <edge id="{eid}">\n'
        f'    <lane id="{eid}_0" index="0" speed="{spd}" length="50.0"/>\n'
        f'    <lane id="{eid}_1" index="1" speed="{spd}" length="50.0"/>\n'
        f'  </edge>'
        for eid, spd in speeds.items()
    )
    net = f'<?xml version="1.0" encoding="UTF-8"?>\n<net>\n{edges_xml}\n</net>\n'
    p = tmp_path / "net.xml"
    p.write_text(net)
    return p


def _write_edgedata(tmp_path):
    """edgedata.xml: 2 buckets, 6 edges (+ extra), velocidades conocidas y casos de vacío."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<meandata>
  <interval begin="0.00" end="60.00" id="aggE">
    <edge id="129466113#0" sampledSeconds="100.0" speed="8.0"/>
    <edge id="344159559#2" sampledSeconds="100.0" speed="13.0" density="20.0"/>
    <edge id="279893875#1" sampledSeconds="100.0" speed="10.0"/>
    <edge id="279893875#2" sampledSeconds="100.0" speed="2.0"/>
    <edge id="430180649" sampledSeconds="100.0" speed="8.0"/>
    <edge id="279893875#4" sampledSeconds="100.0" speed="0.0"/>
    <edge id="X_extra" sampledSeconds="100.0" speed="5.0"/>
  </interval>
  <interval begin="60.00" end="120.00" id="aggE">
    <edge id="129466113#0" sampledSeconds="0.0"/>
    <edge id="344159559#2" sampledSeconds="120.0" speed="13.89"/>
    <edge id="279893875#1" sampledSeconds="0.0" speed="5.0"/>
    <edge id="279893875#2" sampledSeconds="60.0" speed="7.0"/>
    <edge id="430180649" sampledSeconds="120.0" speed="11.11"/>
  </interval>
</meandata>
"""
    p = tmp_path / "edgedata.xml"
    p.write_text(xml)
    return p


def test_free_flow_por_arista_con_cap(tmp_path):
    net = _write_net(tmp_path)
    ff = parse_freeflow(net, set(EDGES))
    # Nodo Larco: carril 16.67 capado a maxSpeed del vType (13.89), NO 16.67 ni global.
    assert ff[NODE0] == 13.89
    # Nodo Schell Este: 11.11 (< cap) → se respeta el límite real del edge.
    assert ff[NODE4] == 11.11
    # El resto: 13.89.
    assert ff["344159559#2"] == 13.89
    # free-flow es POR ARISTA: nodo0 y nodo4 difieren.
    assert ff[NODE0] != ff[NODE4]


def test_builder_shape_y_canales(tmp_path):
    net = _write_net(tmp_path)
    ed = _write_edgedata(tmp_path)
    res = build_corridor_series(ed, net_xml=net)

    assert res["node_order"] == EDGES               # 6 nodos, orden graph_index
    assert res["data"].shape == (2, 6, len(res["channels"]))
    assert res["channels"][0] == "jam_level"        # canal 0 = target
    assert "ratio" in res["channels"]
    # canales raw del edgeData presentes (para la ablación de Fase 4).
    for c in ("speed", "sampledSeconds", "density"):
        assert c in res["channels"]
    # X_extra no instrumentado → no entra.
    assert "X_extra" not in res["node_order"]


def test_jam_level_por_velocidades_conocidas(tmp_path):
    net = _write_net(tmp_path)
    ed = _write_edgedata(tmp_path)
    res = build_corridor_series(ed, net_xml=net)
    jam = res["channels"].index("jam_level")
    ratio = res["channels"].index("ratio")
    n = {e: i for i, e in enumerate(res["node_order"])}

    # ── Bucket t=60 ──
    # Misma velocidad (8.0) da jam distinto por free-flow distinto → prueba per-arista:
    #   nodo0 8/13.89=0.576 → jam 2 ; nodo4 8/11.11=0.720 → jam 1
    assert res["data"][0, n[NODE0], jam] == 2
    assert res["data"][0, n[NODE4], jam] == 1
    assert math.isclose(res["data"][0, n[NODE0], ratio], 8.0 / 13.89, rel_tol=1e-9)
    assert math.isclose(res["data"][0, n[NODE4], ratio], 8.0 / 11.11, rel_tol=1e-9)
    # 13/13.89=0.936 → jam 0 ; 10/13.89=0.720 → jam 1 ; 2/13.89=0.144 → jam 4
    assert res["data"][0, n["344159559#2"], jam] == 0
    assert res["data"][0, n["279893875#1"], jam] == 1
    assert res["data"][0, n["279893875#2"], jam] == 4
    # speed=0 con presencia real → ratio 0 → jam 5 (atasco real, no vacío)
    assert res["data"][0, n["279893875#4"], jam] == 5

    # ── Bucket t=120: convenciones de vacío → jam 0 ──
    # sampledSeconds=0 sin speed → vacío → jam 0
    assert res["data"][1, n[NODE0], jam] == 0
    assert res["data"][1, n[NODE0], ratio] == 1.0
    # sampledSeconds=0 CON speed → igual vacío (sampled<=0) → jam 0
    assert res["data"][1, n["279893875#1"], jam] == 0
    # edge ausente del bucket → vacío → jam 0
    assert res["data"][1, n["279893875#4"], jam] == 0
    # presencia real con free-flow → jam 0 legítimo
    assert res["data"][1, n["344159559#2"], jam] == 0
    assert res["data"][1, n["279893875#2"], jam] == 2


def test_canal_raw_density_solo_donde_se_emitio(tmp_path):
    net = _write_net(tmp_path)
    ed = _write_edgedata(tmp_path)
    res = build_corridor_series(ed, net_xml=net)
    dens = res["channels"].index("density")
    n = {e: i for i, e in enumerate(res["node_order"])}
    # density se emitió sólo en 344159559#2 @ t=60.
    assert res["data"][0, n["344159559#2"], dens] == 20.0
    # donde no se emitió → NaN (la ablación decide cómo imputar).
    assert np.isnan(res["data"][0, n[NODE0], dens])


def test_empty_policy_blocked(tmp_path):
    net = _write_net(tmp_path)
    ed = _write_edgedata(tmp_path)
    res = build_corridor_series(ed, net_xml=net, empty_policy="blocked")
    jam = res["channels"].index("jam_level")
    n = {e: i for i, e in enumerate(res["node_order"])}
    # con la hipótesis de bloqueo, el bucket vacío sube a jam 5.
    assert res["data"][1, n[NODE0], jam] == 5
