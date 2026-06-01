"""Constructor de grafo dirigido del corredor Larco (Fase 1, track STGNN — D-011).

Spike de INVESTIGACIÓN, fuera de producción. Este módulo NO toca
``core_management_api``, ni ``graph_edges/scripts/seed.py``, ni el predictor de
producción (D-006/D-010). Solo lee los XML de SUMO del corredor y produce la
topología que la Fase 2 usará para alinear cada serie de detector a su nodo.

Modelo: *edge-as-node*. Cada uno de los 6 edges SUMO instrumentados (los que
tienen ``<laneAreaDetector>`` en ``corredor_larco.det.add.xml``) es un nodo del
grafo. Hay una arista de grafo dirigida A→B sii ``net[A].to == net[B].from``,
es decir, si el flujo vehicular sale de A y entra a B en la red real. El sentido
resultante es S→N (sur → Benavides → Schell → Diez Canseco), el eje troncal del
corredor.

Salida (formato PyG, sin dependencia de ``torch_geometric``):
- ``edge_index``  : ``torch.long``  de shape ``[2, num_edges]`` (fila 0 = source
  idx, fila 1 = target idx).
- ``edge_weight`` : ``torch.float`` de shape ``[num_edges]``, según ``weight_scheme``.
- ``mapping``     : dict serializable (puente nodo↔edge↔detectores para Fase 2).

torch_geometric NO está instalado en el repo; no se necesita: el formato
``[2, N]`` se construye con torch puro. Probado con el venv raíz
(``.venv/bin/python``, torch 2.9.1).

Dependencias: ``xml.etree`` (stdlib) + ``torch``. Sin red, sin DB, sin escritura
a producción.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Rutas por defecto (relativas a la raíz del repo CerebroVial).
# corridor_graph_builder.py vive en ia_prediction_service/src/data/ ⇒ la raíz
# del repo es parents[3].
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NET_XML = _REPO_ROOT / "simulation" / "conf" / "corredor_larco" / "corredor_larco.net.xml"
DEFAULT_DET_XML = _REPO_ROOT / "simulation" / "conf" / "corredor_larco" / "corredor_larco.det.add.xml"
DEFAULT_MAPPING_OUT = Path(__file__).resolve().parent / "artifacts" / "corridor_graph_mapping.json"

# Orden de nodos estable y documentado (índice de grafo 0–5). NO reordenar:
# la Fase 2 alinea series por este índice.
NODE_ORDER: tuple[str, ...] = (
    "129466113#0",   # 0 — aproximación sur Larco (fuente S)
    "344159559#2",   # 1 — aproximación transversal Benavides Este (fuente)
    "279893875#1",   # 2 — link interno Benavides→Schell
    "279893875#2",   # 3 — aproximación a Schell
    "430180649",     # 4 — aproximación transversal Schell Este (fuente)
    "279893875#4",   # 5 — link interno Schell→Diez Canseco (sumidero, salida ciega)
)

# Esquemas de peso soportados. Todos se computan sobre la ``length`` del edge
# SOURCE de cada arista de grafo.
WEIGHT_SCHEMES: tuple[str, ...] = ("uniform", "length", "inverse", "inv_sqrt")
DEFAULT_WEIGHT_SCHEME = "inverse"


@dataclass(frozen=True)
class EdgeInfo:
    """Atributos de un edge SUMO instrumentado."""

    sumo_id: str
    node_index: int
    from_node: str
    to_node: str
    length: float
    detector_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Parsing de XML (IDs leídos verbatim; nunca se reconstruyen a mano)
# --------------------------------------------------------------------------- #
def _parse_detector_edges(det_xml: Path) -> dict[str, list[str]]:
    """Devuelve {edge_id: [detector_ids]} derivando el edge del campo ``lane``.

    El edge se obtiene quitando el sufijo de carril ``_N`` del ``lane`` (p. ej.
    ``279893875#1_0`` → ``279893875#1``). El ``#N`` del propio edge se conserva
    porque pertenece al ID del edge, no al carril.
    """
    tree = ET.parse(det_xml)
    out: dict[str, list[str]] = {}
    for det in tree.getroot().findall("laneAreaDetector"):
        lane = det.get("lane")
        det_id = det.get("id")
        if lane is None or det_id is None:
            continue
        edge_id = lane.rsplit("_", 1)[0]  # quita SOLO el sufijo de carril
        out.setdefault(edge_id, []).append(det_id)
    return out


def _parse_net_edges(net_xml: Path, wanted: set[str]) -> dict[str, dict]:
    """Devuelve {edge_id: {from, to, length}} para los edges en ``wanted``.

    ``length`` se toma de los ``<lane>``. Si los carriles difieren, se usa el
    promedio (en este corredor los 3 carriles coinciden por edge — verificado).
    """
    tree = ET.parse(net_xml)
    out: dict[str, dict] = {}
    for edge in tree.getroot().findall("edge"):
        eid = edge.get("id")
        if eid not in wanted:
            continue
        lanes = edge.findall("lane")
        lengths = [float(la.get("length")) for la in lanes if la.get("length")]
        out[eid] = {
            "from": edge.get("from"),
            "to": edge.get("to"),
            "length": sum(lengths) / len(lengths) if lengths else float("nan"),
        }
    return out


# --------------------------------------------------------------------------- #
# Pesos
# --------------------------------------------------------------------------- #
def _raw_weight(scheme: str, length: float) -> float:
    if scheme == "uniform":
        return 1.0
    if scheme == "length":
        return length
    if scheme == "inverse":
        return 1.0 / length
    if scheme == "inv_sqrt":
        return 1.0 / math.sqrt(length)
    raise ValueError(f"weight_scheme desconocido: {scheme!r}. Válidos: {WEIGHT_SCHEMES}")


def _normalized_weights(scheme: str, source_lengths: list[float]) -> list[float]:
    """Pesos por arista, normalizados a media = 1.0 (convención del track).

    ``uniform`` se deja tal cual (ya tiene media 1). Para los demás esquemas se
    divide cada peso crudo por la media del vector, de modo que la escala
    absoluta no afecte la magnitud de la convolución de grafo y los esquemas
    sean comparables entre sí. Todos los pesos resultan finitos y positivos.
    """
    raw = [_raw_weight(scheme, L) for L in source_lengths]
    if scheme == "uniform":
        return raw
    mean = sum(raw) / len(raw)
    return [w / mean for w in raw]


# --------------------------------------------------------------------------- #
# Construcción del grafo
# --------------------------------------------------------------------------- #
def _build_edges(nodes: list[EdgeInfo]) -> list[tuple[int, int]]:
    """Aristas dirigidas A→B sii ``net[A].to == net[B].from`` (flujo S→N)."""
    graph_edges: list[tuple[int, int]] = []
    for a in nodes:
        for b in nodes:
            if a.sumo_id == b.sumo_id:
                continue
            if a.to_node == b.from_node:
                graph_edges.append((a.node_index, b.node_index))
    # Orden estable: por (source, target).
    graph_edges.sort()
    return graph_edges


def load_corridor_nodes(
    net_xml: Path | str = DEFAULT_NET_XML,
    det_xml: Path | str = DEFAULT_DET_XML,
) -> list[EdgeInfo]:
    """Carga los 6 nodos (edges instrumentados) con sus atributos y detectores.

    Falla ruidosamente si algún edge esperado no aparece en el net o no tiene
    detector — no se silencia un desalineamiento de datos.
    """
    net_xml = Path(net_xml)
    det_xml = Path(det_xml)

    det_map = _parse_detector_edges(det_xml)
    instrumented = set(det_map)
    expected = set(NODE_ORDER)

    missing_det = expected - instrumented
    if missing_det:
        raise ValueError(
            f"Edges esperados sin detector en {det_xml.name}: {sorted(missing_det)}"
        )

    net_map = _parse_net_edges(net_xml, expected)
    missing_net = expected - set(net_map)
    if missing_net:
        raise ValueError(
            f"Edges esperados no hallados en {net_xml.name}: {sorted(missing_net)}"
        )

    nodes: list[EdgeInfo] = []
    for idx, eid in enumerate(NODE_ORDER):
        info = net_map[eid]
        nodes.append(
            EdgeInfo(
                sumo_id=eid,
                node_index=idx,
                from_node=info["from"],
                to_node=info["to"],
                length=info["length"],
                detector_ids=sorted(det_map[eid]),
            )
        )
    return nodes


def build_mapping(nodes: list[EdgeInfo], graph_edges: list[tuple[int, int]]) -> dict:
    """Artefacto de mapeo persistente (puente nodo↔edge↔detectores para Fase 2).

    Incluye, por nodo: índice de grafo, edge SUMO, detectores, from, to, length.
    Y la lista de aristas del grafo con su peso en CADA esquema.
    """
    source_lengths = [nodes[s].length for s, _t in graph_edges]
    weights_by_scheme = {
        scheme: _normalized_weights(scheme, source_lengths) for scheme in WEIGHT_SCHEMES
    }

    edges_payload = []
    for i, (s, t) in enumerate(graph_edges):
        edges_payload.append(
            {
                "source_index": s,
                "target_index": t,
                "source_edge": nodes[s].sumo_id,
                "target_edge": nodes[t].sumo_id,
                "weights": {scheme: weights_by_scheme[scheme][i] for scheme in WEIGHT_SCHEMES},
            }
        )

    return {
        "model": "edge-as-node",
        "flow_direction": "S->N",
        "weight_normalization": "media=1.0 (uniform sin normalizar)",
        "default_weight_scheme": DEFAULT_WEIGHT_SCHEME,
        "nodes": [
            {
                "graph_index": n.node_index,
                "sumo_edge": n.sumo_id,
                "detector_ids": n.detector_ids,
                "from": n.from_node,
                "to": n.to_node,
                "length": n.length,
            }
            for n in nodes
        ],
        "edges": edges_payload,
    }


def build_corridor_graph(
    net_xml: Path | str = DEFAULT_NET_XML,
    det_xml: Path | str = DEFAULT_DET_XML,
    weight_scheme: str = DEFAULT_WEIGHT_SCHEME,
    mapping_out: Path | str | None = DEFAULT_MAPPING_OUT,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Construye el grafo dirigido del corredor Larco (función pública principal).

    Args:
        net_xml: ruta a ``corredor_larco.net.xml``.
        det_xml: ruta a ``corredor_larco.det.add.xml``.
        weight_scheme: uno de ``WEIGHT_SCHEMES`` (default ``inverse``).
        mapping_out: si no es None, escribe el JSON de mapeo a esa ruta.

    Returns:
        ``(edge_index, edge_weight, mapping)`` donde ``edge_index`` es
        ``torch.long [2, num_edges]``, ``edge_weight`` es ``torch.float
        [num_edges]`` para el ``weight_scheme`` pedido, y ``mapping`` el dict
        con los 4 esquemas computados.
    """
    if weight_scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"weight_scheme desconocido: {weight_scheme!r}. Válidos: {WEIGHT_SCHEMES}")

    nodes = load_corridor_nodes(net_xml, det_xml)
    graph_edges = _build_edges(nodes)
    mapping = build_mapping(nodes, graph_edges)

    edge_index = torch.tensor(
        [[s for s, _t in graph_edges], [t for _s, t in graph_edges]],
        dtype=torch.long,
    )
    source_lengths = [nodes[s].length for s, _t in graph_edges]
    edge_weight = torch.tensor(
        _normalized_weights(weight_scheme, source_lengths), dtype=torch.float
    )

    if mapping_out is not None:
        mapping_out = Path(mapping_out)
        mapping_out.parent.mkdir(parents=True, exist_ok=True)
        mapping_out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))

    return edge_index, edge_weight, mapping


if __name__ == "__main__":
    ei, ew, m = build_corridor_graph()
    print("edge_index:\n", ei)
    print("edge_weight (", DEFAULT_WEIGHT_SCHEME, "):\n", ew)
    print("mapping escrito en:", DEFAULT_MAPPING_OUT)
