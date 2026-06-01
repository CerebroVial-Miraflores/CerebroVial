"""Tests del constructor del grafo edge-as-node de Miraflores (Fase 1, track STGNN).

Verifican la topología derivada de la red SUMO canónica de Miraflores
(``simulation/conf/network/miraflores.net.xml``, sin red, sin DB): descubrimiento de
los 381 nodos vehiculares, la regla ``<connection>`` que elimina las 12 aristas-
fantasma, el orden determinista y el gate de subconjunto. Las aserciones son sobre
comportamiento (lanza / no lanza, conteos del mapping), no sobre logs.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
import torch

from src.data.miraflores_graph_builder import (
    DEFAULT_NET_XML,
    EXPECTED_GHOST_EDGES,
    build_miraflores_graph,
)

EXPECTED_NODES = 381

# Pares fantasma conocidos (sumo_edge → sumo_edge): comparten junction pero NO tienen
# <connection> real. Deben quedar fuera del edge_index del grafo completo.
KNOWN_GHOSTS = [
    ("-129822384#2", "129822384#2"),
    ("1152311680#1", "1152311679#0"),
]


def _full_graph():
    # No escribe el artefacto JSON durante el test.
    return build_miraflores_graph(mapping_out=None)


def test_node_count_gate():
    # Canónico: expected_nodes=381 no lanza y el mapping reporta 381 nodos.
    _ei, _ew, mapping = build_miraflores_graph(expected_nodes=EXPECTED_NODES, mapping_out=None)
    assert mapping["counts"]["nodes"] == EXPECTED_NODES
    assert len(mapping["nodes"]) == EXPECTED_NODES

    # Gate fail-fast: un esperado distinto del real lanza ValueError.
    with pytest.raises(ValueError):
        build_miraflores_graph(expected_nodes=EXPECTED_NODES - 1, mapping_out=None)


def test_every_edge_has_a_real_connection():
    edge_index, _ew, mapping = _full_graph()
    sumo_of = {n["node_index"]: n["sumo_edge"] for n in mapping["nodes"]}

    # Re-parseo independiente de las <connection> a nivel edge.
    root = ET.parse(DEFAULT_NET_XML).getroot()
    conn = set()
    for c in root.findall("connection"):
        a, b = c.get("from"), c.get("to")
        if a is None or b is None or a.startswith(":") or b.startswith(":"):
            continue
        conn.add((a, b))

    for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        pair = (sumo_of[s], sumo_of[t])
        assert pair in conn, f"arista {pair} sin <connection> real en la red"


def test_full_graph_filters_exactly_twelve_ghosts():
    edge_index, _ew, mapping = _full_graph()
    assert mapping["counts"]["filtered"] == EXPECTED_GHOST_EDGES

    # Los pares fantasma conocidos NO están en el edge_index.
    idx_of = {n["sumo_edge"]: n["node_index"] for n in mapping["nodes"]}
    present = {(int(s), int(t)) for s, t in zip(edge_index[0], edge_index[1])}
    for a, b in KNOWN_GHOSTS:
        assert a in idx_of and b in idx_of, f"par fantasma {a}->{b} fuera de los nodos"
        assert (idx_of[a], idx_of[b]) not in present, f"fantasma {a}->{b} presente"


def test_determinism():
    ei1, _w1, m1 = _full_graph()
    ei2, _w2, m2 = _full_graph()
    assert [n["sumo_edge"] for n in m1["nodes"]] == [n["sumo_edge"] for n in m2["nodes"]]
    assert torch.equal(ei1, ei2)


def test_output_shapes_and_placeholder_weight():
    edge_index, edge_weight, _m = _full_graph()
    assert edge_index.dtype == torch.long
    assert edge_index.shape[0] == 2
    e = edge_index.shape[1]
    assert edge_weight.shape == (e,)
    assert edge_weight.dtype == torch.float
    assert bool((edge_weight == 1.0).all())  # placeholder uniforme


def test_subset_param():
    # Subconjunto válido construye y respeta el orden lexicográfico.
    _ei, _ew, full = build_miraflores_graph(mapping_out=None)
    some = [n["sumo_edge"] for n in full["nodes"][:20]]
    _ei2, _ew2, sub = build_miraflores_graph(edge_ids=some, mapping_out=None)
    assert sub["counts"]["nodes"] == len(some)
    assert [n["sumo_edge"] for n in sub["nodes"]] == sorted(some)

    # edge_id inexistente entre los 381 lanza ValueError.
    with pytest.raises(ValueError):
        build_miraflores_graph(edge_ids=["__no_existe__"], mapping_out=None)
