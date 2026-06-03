"""Tests del constructor del grafo edge-as-node de Miraflores (Fase 1, track STGNN).

Verifican la topología derivada de la red SUMO canónica de Miraflores
(``simulation/conf/network/miraflores.net.xml``, sin red, sin DB): descubrimiento de
los 1664 nodos vehiculares, la regla ``<connection>`` que elimina las 510 aristas-
fantasma, el orden determinista y el gate de subconjunto. Las aserciones son sobre
comportamiento (lanza / no lanza, conteos del mapping), no sobre logs.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque

import pytest
import torch

from src.data import miraflores_graph_builder as mgb
from src.data.miraflores_graph_builder import (
    DEFAULT_NET_XML,
    EXPECTED_GHOST_EDGES,
    build_miraflores_graph,
)

EXPECTED_NODES = 1664

# Componente conexa principal (LCC): la red tiene 2 componentes de tamaños [1660, 4].
EXPECTED_LCC_NODES = 1660
EXPECTED_COMPONENT_SIZES = (1660, 4)

# Los 4 sumo_edge de la islita desconectada (componente menor). NO deben aparecer en
# el grafo de 1660.
ISLET = {
    "-24332322#0",
    "24332322#0",
    "24332322#1",
    "319671627",
}


def _full_graph():
    # No escribe el artefacto JSON durante el test.
    return build_miraflores_graph(mapping_out=None)


def test_node_count_gate():
    # Canónico: expected_nodes=1664 no lanza y el mapping reporta 1664 nodos.
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


def test_full_graph_filters_exactly_510_ghosts():
    _ei, _ew, mapping = _full_graph()
    assert mapping["counts"]["filtered"] == EXPECTED_GHOST_EDGES


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

    # edge_id inexistente entre los 1664 lanza ValueError.
    with pytest.raises(ValueError):
        build_miraflores_graph(edge_ids=["__no_existe__"], mapping_out=None)


# --------------------------------------------------------------------------- #
# Modo componente-grande (LCC): recorte por conectividad topológica.
# --------------------------------------------------------------------------- #
def _lcc():
    # No escribe artefacto JSON durante el test.
    return build_miraflores_graph(largest_component_only=True, mapping_out=None)


def test_largest_component_1660():
    # El grafo de la componente principal tiene 1660 nodos y reporta procedencia.
    _ei, _ew, mapping = _lcc()
    assert mapping["counts"]["nodes"] == EXPECTED_LCC_NODES
    assert len(mapping["nodes"]) == EXPECTED_LCC_NODES
    assert mapping["derived_from"] == "largest connected component of full 1664-node graph"
    assert mapping["full_graph_nodes"] == EXPECTED_NODES
    assert mapping["dropped_component_nodes"] == len(ISLET)
    assert set(mapping["dropped_edges"]) == ISLET


def test_component_gate():
    # Tamaños correctos no lanzan; tamaños incorrectos lanzan ValueError.
    build_miraflores_graph(
        largest_component_only=True,
        expected_component_sizes=EXPECTED_COMPONENT_SIZES,
        mapping_out=None,
    )
    with pytest.raises(ValueError):
        build_miraflores_graph(
            largest_component_only=True,
            expected_component_sizes=(1659, 1),
            mapping_out=None,
        )


def test_mutual_exclusion():
    # edge_ids + largest_component_only son mutuamente excluyentes.
    _ei, _ew, full = build_miraflores_graph(mapping_out=None)
    some = [n["sumo_edge"] for n in full["nodes"][:5]]
    with pytest.raises(ValueError):
        build_miraflores_graph(
            edge_ids=some, largest_component_only=True, mapping_out=None
        )


def test_lcc_is_connected():
    # El grafo de 1660 es una sola componente conexa (no quedó ninguna islita).
    edge_index, _ew, mapping = _lcc()
    n = len(mapping["nodes"])
    adj = {i: set() for i in range(n)}
    for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[s].add(t)
        adj[t].add(s)  # no-dirigido: conectividad
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    assert len(seen) == n == EXPECTED_LCC_NODES


def test_islet_excluded():
    # Los 4 sumo_edge de la islita NO están entre los nodos del grafo de 1660.
    _ei, _ew, mapping = _lcc()
    present = {n["sumo_edge"] for n in mapping["nodes"]}
    assert ISLET.isdisjoint(present)


def test_lcc_determinism():
    # Dos construcciones LCC → mismo mapping (orden de nodos) y mismo edge_index.
    ei1, _w1, m1 = _lcc()
    ei2, _w2, m2 = _lcc()
    assert [n["sumo_edge"] for n in m1["nodes"]] == [n["sumo_edge"] for n in m2["nodes"]]
    assert m1["dropped_edges"] == m2["dropped_edges"]
    assert torch.equal(ei1, ei2)


def test_ghost_gate_runs_under_lcc(monkeypatch):
    # El gate de 510 fantasmas del grafo COMPLETO sigue corriendo en modo LCC: si el
    # invariante se rompe, lanza antes de recortar (no se saltea).
    monkeypatch.setattr(mgb, "EXPECTED_GHOST_EDGES", EXPECTED_GHOST_EDGES + 1)
    with pytest.raises(ValueError):
        build_miraflores_graph(largest_component_only=True, mapping_out=None)
