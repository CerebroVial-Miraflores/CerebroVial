"""Tests del constructor de grafo dirigido del corredor Larco (Fase 1, D-011).

Verifican la topología derivada de los XML de SUMO: 6 nodos, 5 aristas en el
sentido S→N correcto, fuentes/sumidero esperados, cobertura de detectores y los
4 esquemas de peso. Lee los XML canónicos de
``simulation/conf/corredor_larco/`` (sin red, sin DB).
"""

from __future__ import annotations

import math

import torch

from src.data.corridor_graph_builder import (
    NODE_ORDER,
    WEIGHT_SCHEMES,
    build_corridor_graph,
    load_corridor_nodes,
)

# Aristas dirigidas esperadas (índices de grafo), flujo S→N.
EXPECTED_EDGES = {(0, 2), (1, 2), (2, 3), (3, 5), (4, 5)}


def _graph():
    # No escribe el artefacto JSON durante el test.
    return build_corridor_graph(weight_scheme="inverse", mapping_out=None)


def test_six_nodes_five_directed_edges():
    edge_index, _ew, mapping = _graph()
    assert len(NODE_ORDER) == 6
    assert len(mapping["nodes"]) == 6
    assert edge_index.shape == (2, 5)
    assert edge_index.dtype == torch.long


def test_edges_are_the_expected_five_in_correct_direction():
    edge_index, _ew, _m = _graph()
    got = {(int(s), int(t)) for s, t in zip(edge_index[0], edge_index[1])}
    assert got == EXPECTED_EDGES


def test_sources_and_sink():
    edge_index, _ew, _m = _graph()
    sources = {int(s) for s in edge_index[0]}  # nodos con sucesor
    targets = {int(t) for t in edge_index[1]}  # nodos con predecesor
    all_nodes = set(range(6))

    # Nodo 5 (279893875#4, salida Diez Canseco ciega) es sumidero: sin sucesores.
    assert 5 not in sources
    assert NODE_ORDER[5] == "279893875#4"

    # Nodos 0, 1, 4 son fuentes: sin predecesores.
    assert (all_nodes - targets) == {0, 1, 4}


def test_mapping_covers_six_edges_each_with_detector():
    nodes = load_corridor_nodes()
    assert {n.sumo_id for n in nodes} == set(NODE_ORDER)
    assert len(nodes) == 6
    for n in nodes:
        assert len(n.detector_ids) >= 1, f"{n.sumo_id} sin detector"
        assert n.length > 0


def test_all_weight_schemes_finite_positive_len5():
    for scheme in WEIGHT_SCHEMES:
        _ei, ew, _m = build_corridor_graph(weight_scheme=scheme, mapping_out=None)
        assert ew.shape == (5,)
        assert ew.dtype == torch.float
        vals = ew.tolist()
        assert all(math.isfinite(v) and v > 0 for v in vals), f"{scheme}: {vals}"


def test_mapping_includes_all_schemes_per_edge():
    _ei, _ew, mapping = _graph()
    assert len(mapping["edges"]) == 5
    for e in mapping["edges"]:
        assert set(e["weights"]) == set(WEIGHT_SCHEMES)
        assert (e["source_index"], e["target_index"]) in EXPECTED_EDGES
