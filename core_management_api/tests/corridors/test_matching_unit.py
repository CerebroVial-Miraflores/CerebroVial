"""Tests unitarios de la orquestación del matching (Fase B-2), sin DB.

Validaciones puras de la cadena + ``match_corridor`` contra un repo FALSO (la geometría no
necesita PostGIS acá: se inyectan overlaps controlados). Lo clave: que el SENTIDO mande sobre
el overlap — un segmento de mayor overlap pero sentido opuesto NO debe ganar.
"""
from dataclasses import dataclass

import pytest

from src.corridors.application.matching import (
    CorridorEdgeInput,
    TomTomSegment,
    match_corridor,
    validate_continuity,
    validate_sequences,
)


@dataclass(frozen=True)
class _Edge:  # objeto mínimo con extremos para validate_continuity
    edge_id: str
    source_node: str
    target_node: str


class _FakeRepo:
    """``OverlapSource`` falso: devuelve overlaps predefinidos por edge_id."""

    def __init__(self, overlaps_by_edge):
        self._by_edge = overlaps_by_edge

    def edge_overlaps(self, edge_id, segment_wkts, buffer_m, min_overlap):
        return self._by_edge.get(edge_id, {})


# --- validate_sequences ---

@pytest.mark.parametrize(
    "seqs,ok",
    [
        ([0, 1, 2], True),
        ([5, 6, 7], True),   # no fuerza inicio en 0
        ([0, 2], False),     # hueco
        ([0, 1, 1], False),  # duplicado
        ([], False),         # vacío
    ],
)
def test_validate_sequences(seqs, ok):
    assert (validate_sequences(seqs) is None) is ok


# --- validate_continuity ---

def test_validate_continuity_cadena_continua():
    edges = [_Edge("e1", "n1", "n2"), _Edge("e2", "n2", "n3")]
    assert validate_continuity(edges) is None


def test_validate_continuity_cadena_rota():
    edges = [_Edge("e1", "n1", "n2"), _Edge("e_broken", "nx", "n3")]
    msg = validate_continuity(edges)
    assert msg is not None and "cadena rota" in msg


# --- match_corridor: el sentido manda ---

# Arista hacia el ESTE (bearing ~90°).
_EB_EDGE = CorridorEdgeInput(
    edge_id="eb", sequence=0,
    source_coord=(-77.040, -12.120), target_coord=(-77.030, -12.120),
)
_SEG_EB = TomTomSegment(openlr="OLR_EB", coordinates=[(-77.040, -12.120), (-77.030, -12.120)])
_SEG_WB = TomTomSegment(openlr="OLR_WB", coordinates=[(-77.030, -12.120), (-77.040, -12.120)])


def test_match_descarta_sentido_opuesto_aunque_tenga_mas_overlap():
    # seg WB (opuesto) tiene MÁS overlap, pero debe descartarse por sentido.
    repo = _FakeRepo({"eb": {0: 0.80, 1: 0.99}})  # idx0=EB, idx1=WB
    mapping = match_corridor(repo, [_EB_EDGE], [_SEG_EB, _SEG_WB])
    assert mapping == {"eb": "OLR_EB"}


def test_match_elige_mayor_overlap_entre_mismos_sentidos():
    seg_eb2 = TomTomSegment(openlr="OLR_EB2", coordinates=[(-77.040, -12.120), (-77.030, -12.120)])
    repo = _FakeRepo({"eb": {0: 0.75, 1: 0.92}})  # ambos EB
    mapping = match_corridor(repo, [_EB_EDGE], [_SEG_EB, seg_eb2])
    assert mapping == {"eb": "OLR_EB2"}


def test_match_sin_candidatos_devuelve_none():
    repo = _FakeRepo({"eb": {}})  # ningún segmento pasa el buffer
    mapping = match_corridor(repo, [_EB_EDGE], [_SEG_EB, _SEG_WB])
    assert mapping == {"eb": None}


def test_match_todos_opuestos_devuelve_none():
    repo = _FakeRepo({"eb": {1: 0.99}})  # sólo el WB pasa el buffer → filtrado por sentido
    mapping = match_corridor(repo, [_EB_EDGE], [_SEG_EB, _SEG_WB])
    assert mapping == {"eb": None}
