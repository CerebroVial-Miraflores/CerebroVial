"""Constructor del grafo edge-as-node de toda la red Miraflores (Fase 1, track STGNN).

Spike de INVESTIGACIÓN, fuera de producción. Este módulo NO toca
``core_management_api``, ni la tabla ``graph_edges``, ni ``scripts/seed.py``, ni el
predictor de producción. Tampoco lee la BD ni el Parquet. Solo lee la red SUMO
canónica de Miraflores y escribe un JSON de mapeo en ``artifacts/``. Es el análogo
de ``corridor_graph_builder.py`` (Larco, 6 nodos hardcodeados) pero a escala de toda
la red: descubre los nodos dinámicamente y construye las aristas con la regla real
de ``<connection>``.

Modelo: *edge-as-node*. Cada edge SUMO **vehicular** (alguna ``<lane>`` permite la
clase ``passenger``) es un nodo del grafo; se excluyen los internos (``id`` que
empieza con ``:`` o ``function="internal"``). Hay una arista de grafo dirigida A→B
sii (1) ``net[A].to == net[B].from`` (candidata por junction compartida) **y** (2)
existe al menos una ``<connection from=A to=B>`` real en la red (colapsada a nivel
edge, salteando from/to internos). El paso (2) es la corrección que elimina las
aristas-fantasma: pares que comparten junction pero que SUMO no conecta de verdad.

Salida (formato PyG, sin dependencia de ``torch_geometric``):
- ``edge_index``  : ``torch.long``  de shape ``[2, E]`` (fila 0 = source idx, fila
  1 = target idx).
- ``edge_weight`` : ``torch.float`` de shape ``[E]``. **PLACEHOLDER uniforme = 1.0**,
  NO es un peso real. La ponderación de aristas (longitud, capacidad, flujo, etc.)
  es una decisión de modelado del STGNN que se difiere a Fase 4. Aquí 1.0 solo
  rellena la interfaz; ningún consumidor de producción lo usa hoy.
- ``mapping``     : dict serializable (puente node_index↔sumo_edge para Fase 2).

Orden de nodos: ``edge_id`` ordenados lexicográficamente; el ``node_index`` es la
posición en ese orden. Es **estable y NO reordenable**: la Fase 2 alinea las series
temporales de cada edge por ``node_index``, así que cambiar el orden rompería esa
alineación silenciosamente.

torch_geometric NO está instalado en el repo; no se necesita: el formato ``[2, N]``
se construye con torch puro. Dependencias: ``xml.etree`` (stdlib) + ``torch``.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Rutas por defecto (relativas a la raíz del repo CerebroVial).
# miraflores_graph_builder.py vive en ia_prediction_service/src/data/ ⇒ la raíz
# del repo es parents[3]. Se usa la red CANÓNICA, no la copia de scratch/.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NET_XML = _REPO_ROOT / "simulation" / "conf" / "network" / "miraflores.net.xml"
DEFAULT_MAPPING_OUT = Path(__file__).resolve().parent / "artifacts" / "miraflores_graph_mapping.json"
# Artefacto SEPARADO del de 381 para el modo componente-grande (LCC): no se
# sobrescribe el mapping completo. Ver build_miraflores_graph(largest_component_only=...).
DEFAULT_LCC_MAPPING_OUT = Path(__file__).resolve().parent / "artifacts" / "miraflores_graph_lcc_mapping.json"

# Aristas-fantasma esperadas sobre el grafo COMPLETO (los 381 nodos): pares que
# la regla ingenua ``to==from`` propone pero que no tienen ``<connection>`` real.
# Es un invariante de la red canónica; si cambia, algo se movió y hay que revisar.
EXPECTED_GHOST_EDGES = 12


# --------------------------------------------------------------------------- #
# Parsing de XML (semántica SUMO reusada del spike scratch/ghost_edges_audit/)
# --------------------------------------------------------------------------- #
def _lane_allows_passenger(lane: ET.Element) -> bool:
    """Semántica SUMO de la clase vehicular ``passenger`` sobre un ``<lane>``.

    - ``allow`` presente  → permitido si ``passenger`` o ``all`` está en los tokens.
    - ``disallow`` presente → permitido si ``passenger`` y ``all`` NO están.
    - ninguno → default SUMO: permite todas las clases vehiculares (incl. passenger).
    """
    allow = lane.get("allow")
    disallow = lane.get("disallow")
    if allow is not None:
        toks = allow.split()
        return ("passenger" in toks) or ("all" in toks)
    if disallow is not None:
        toks = disallow.split()
        return ("passenger" not in toks) and ("all" not in toks)
    return True


def _discover_nodes(root: ET.Element) -> dict[str, tuple[str, str]]:
    """Devuelve {edge_id: (from, to)} de los edges vehiculares (nodos del grafo).

    Un edge es nodo sii alguna de sus ``<lane>`` permite ``passenger``. Se excluyen
    los internos: ``id`` que empieza con ``:`` o ``function="internal"``.
    """
    veh: dict[str, tuple[str, str]] = {}
    for edge in root.findall("edge"):
        eid = edge.get("id")
        if eid is None or eid.startswith(":"):
            continue
        if edge.get("function") == "internal":
            continue
        lanes = edge.findall("lane")
        if any(_lane_allows_passenger(la) for la in lanes):
            veh[eid] = (edge.get("from"), edge.get("to"))
    return veh


def _parse_connections(root: ET.Element, node_set: set[str]) -> set[tuple[str, str]]:
    """Pares ``(from, to)`` colapsados a nivel edge desde los ``<connection>``.

    Saltea conexiones con from/to ausente o interno (empieza con ``:``), y retiene
    solo las que van entre dos nodos del conjunto (sin self-loops).
    """
    conn: set[tuple[str, str]] = set()
    for c in root.findall("connection"):
        a = c.get("from")
        b = c.get("to")
        if a is None or b is None:
            continue
        if a.startswith(":") or b.startswith(":"):
            continue
        if a in node_set and b in node_set and a != b:
            conn.add((a, b))
    return conn


# --------------------------------------------------------------------------- #
# Construcción del grafo
# --------------------------------------------------------------------------- #
def _build_edges(
    sumo_ids: list[str],
    idx_of: dict[str, int],
    veh: dict[str, tuple[str, str]],
    conn: set[tuple[str, str]],
) -> tuple[list[tuple[int, int]], int, int, int]:
    """Aristas A→B retenidas y los conteos del filtro de fantasmas.

    Candidata A→B sii ``veh[A].to == veh[B].from`` (sin self-loops). Se retiene sii
    además ``(A, B)`` aparece como ``<connection>`` real. Devuelve
    ``(graph_edges, n_candidates, n_retained, n_filtered)`` con ``graph_edges`` en
    índices de grafo y ordenado por (source, target).
    """
    graph_edges: list[tuple[int, int]] = []
    n_candidates = 0
    for a in sumo_ids:
        to_a = veh[a][1]
        for b in sumo_ids:
            if a == b:
                continue
            if to_a == veh[b][0]:
                n_candidates += 1
                if (a, b) in conn:
                    graph_edges.append((idx_of[a], idx_of[b]))
    graph_edges.sort()
    n_retained = len(graph_edges)
    n_filtered = n_candidates - n_retained
    return graph_edges, n_candidates, n_retained, n_filtered


def _connected_components(
    sumo_ids: list[str],
    graph_edges: list[tuple[int, int]],
) -> list[list[str]]:
    """Componentes conexas del grafo NO-dirigido inducido por ``graph_edges``.

    Trabaja sobre la conectividad real (aristas retenidas), no sobre las candidatas:
    la regla ``<connection>`` ya filtró las fantasmas. Un nodo sin aristas es su
    propia componente (singleton). Sin ``networkx`` (no instalado): BFS con stdlib.

    Devuelve una lista de componentes, cada una como lista de ``sumo_edge`` ordenada
    lexicográficamente. El orden de recorrido sigue ``sumo_ids`` (ya lexicográfico),
    así que la salida es determinista.
    """
    adj: dict[int, set[int]] = {i: set() for i in range(len(sumo_ids))}
    for s, t in graph_edges:
        adj[s].add(t)
        adj[t].add(s)  # no-dirigido: conectividad, no sentido

    seen: set[int] = set()
    components: list[list[str]] = []
    for start in range(len(sumo_ids)):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        comp: list[int] = []
        while queue:
            node = queue.popleft()
            comp.append(node)
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        components.append(sorted(sumo_ids[i] for i in comp))
    return components


def build_mapping(
    sumo_ids: list[str],
    graph_edges: list[tuple[int, int]],
    counts: dict[str, int],
    provenance: dict | None = None,
) -> dict:
    """Artefacto de mapeo persistente (puente node_index↔sumo_edge para Fase 2).

    Identificación dual ``sumo_edge`` + ``node_index`` posicional. Los ``counts`` se
    computan en esta corrida (nodos/candidatas/retenidas/filtradas), nunca literales.
    Si ``provenance`` no es ``None``, sus campos se mergean (procedencia del recorte
    LCC: ``derived_from``, ``full_graph_nodes``, ``dropped_component_nodes``,
    ``dropped_edges``); también computados en la corrida, nunca literales.
    """
    mapping = {
        "model": "edge-as-node",
        "network": "miraflores",
        "edge_rule": "to==from ∩ <connection> (nivel edge, sin internos)",
        "weight": "placeholder uniform=1.0 (no es peso real; ponderación → Fase 4)",
        "node_order": "edge_id lexicográfico ascendente (estable, no reordenable)",
    }
    if provenance is not None:
        mapping.update(provenance)
    mapping["counts"] = dict(counts)
    mapping["nodes"] = [
        {"node_index": i, "sumo_edge": eid} for i, eid in enumerate(sumo_ids)
    ]
    mapping["edges"] = [
        {
            "source_index": s,
            "target_index": t,
            "source_edge": sumo_ids[s],
            "target_edge": sumo_ids[t],
        }
        for s, t in graph_edges
    ]
    return mapping


def build_miraflores_graph(
    net_xml: Path | str = DEFAULT_NET_XML,
    edge_ids: list[str] | None = None,
    expected_nodes: int | None = None,
    largest_component_only: bool = False,
    expected_component_sizes: tuple[int, ...] | None = None,
    mapping_out: Path | str | None = DEFAULT_MAPPING_OUT,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Construye el grafo edge-as-node de Miraflores (función pública principal).

    Args:
        net_xml: ruta a ``miraflores.net.xml`` (canónica por defecto).
        edge_ids: ``None`` → todos los edges vehiculares descubiertos (grafo
            completo). Si se pasa un subconjunto, cada id debe estar entre los
            descubiertos (``ValueError`` ruidoso si alguno no lo está) y el grafo
            se construye solo sobre ese subconjunto. **Mutuamente excluyente con
            ``largest_component_only``** (son dos formas distintas de seleccionar
            nodos; pasar ambos lanza ``ValueError``).
        expected_nodes: gate de conteo del caso canónico. Si no es ``None`` y el
            nº de nodos descubiertos difiere, lanza ``ValueError`` (falla fuerte: un
            conteo distinto señala cambio en la red o en el criterio de parseo, no
            una curiosidad a logear). Si es ``None``, solo logea el conteo y sigue
            (modo exploratorio / red nueva). No hardcodea ningún número. Se evalúa
            siempre sobre el grafo COMPLETO (también en modo componente-grande).
        largest_component_only: si ``True``, tras construir y validar el grafo
            completo (descubrimiento, regla ``<connection>``, gate de 12 fantasmas
            —que NO se saltea—), computa las componentes conexas del grafo
            no-dirigido, se queda con la mayor, recorta nodos+aristas a esa
            componente, re-ordena los ``sumo_edge`` lexicográficamente y reasigna
            ``node_index`` 0..N-1 contiguos. **Criterio de "mayor componente"
            (determinista):** por nº de nodos; ante empate, la que contiene el
            ``sumo_edge`` lexicográficamente menor. El mapping se escribe a un
            artefacto SEPARADO (``DEFAULT_LCC_MAPPING_OUT``) para no sobrescribir el
            de 381, con campos de procedencia (``derived_from``, ``full_graph_nodes``,
            ``dropped_component_nodes``, ``dropped_edges``).
        expected_component_sizes: gate de componentes (asimetría opt-in, igual que
            ``expected_nodes``). Solo aplica con ``largest_component_only=True``. Si
            no es ``None``, afirma que los tamaños de las componentes (ordenados
            descendente) coinciden EXACTAMENTE; si no, ``ValueError`` ruidoso con los
            tamaños reales vs esperados. Si es ``None``, solo logea los tamaños y sigue.
        mapping_out: si no es ``None``, escribe el JSON de mapeo a esa ruta. En modo
            componente-grande, si se deja el default (``DEFAULT_MAPPING_OUT``) se
            redirige automáticamente a ``DEFAULT_LCC_MAPPING_OUT`` para no pisar el
            mapping de 381; una ruta explícita o ``None`` se respetan tal cual.

    Returns:
        ``(edge_index, edge_weight, mapping)`` donde ``edge_index`` es
        ``torch.long [2, E]``, ``edge_weight`` es ``torch.float [E]`` PLACEHOLDER
        uniforme 1.0 (ver docstring de módulo; ponderación real → Fase 4), y
        ``mapping`` el dict con la identificación dual y los conteos de la corrida
        (más procedencia si es modo componente-grande). En modo componente-grande
        las tensores y el mapping corresponden al subgrafo recortado.

    Filosofía de gates: parar fuerte en el caso canónico, ser flexible en el
    exploratorio. Sobre el grafo completo (``edge_ids=None``) las fantasmas filtradas
    deben ser exactamente ``EXPECTED_GHOST_EDGES`` (12); si no, lanza ``ValueError``
    (un edge_index sucio es justo lo que este módulo existe para evitar). Sobre un
    subconjunto el nº de fantasmas es otro legítimamente y la aserción no aplica.
    """
    if edge_ids is not None and largest_component_only:
        raise ValueError(
            "edge_ids y largest_component_only son mutuamente excluyentes: son dos "
            "formas distintas de seleccionar nodos y no se combinan. Pasá uno u otro."
        )

    net_xml = Path(net_xml)
    root = ET.parse(net_xml).getroot()

    veh = _discover_nodes(root)
    n_discovered = len(veh)
    logger.info(
        "Miraflores: %d edges vehiculares descubiertos (criterio: alguna lane "
        "permite passenger; sin internos)",
        n_discovered,
    )
    if expected_nodes is not None and n_discovered != expected_nodes:
        raise ValueError(
            f"Conteo de nodos inesperado: descubiertos {n_discovered}, "
            f"esperados {expected_nodes}. Posible cambio en la red o en el criterio "
            f"de parseo — parar y revisar antes de seguir."
        )

    is_full_graph = edge_ids is None
    if is_full_graph:
        selected = set(veh)
    else:
        selected = set(edge_ids)
        invalid = sorted(selected - set(veh))
        if invalid:
            raise ValueError(
                f"edge_id(s) no encontrados entre los {n_discovered} edges "
                f"vehiculares: {invalid}"
            )

    # Orden determinista: edge_id lexicográfico → node_index posicional.
    sumo_ids = sorted(selected)
    idx_of = {eid: i for i, eid in enumerate(sumo_ids)}

    conn = _parse_connections(root, set(sumo_ids))
    graph_edges, n_candidates, n_retained, n_filtered = _build_edges(
        sumo_ids, idx_of, veh, conn
    )
    logger.info(
        "edge_index: candidatas(to==from)=%d, retenidas=%d, filtradas=%d",
        n_candidates,
        n_retained,
        n_filtered,
    )

    if is_full_graph and n_filtered != EXPECTED_GHOST_EDGES:
        raise ValueError(
            f"Fantasmas filtradas={n_filtered} sobre el grafo completo, se esperaban "
            f"{EXPECTED_GHOST_EDGES}. Posible discrepancia de parseo o cambio en la "
            f"red — el edge_index podría estar sucio. Parar y revisar."
        )

    provenance: dict | None = None
    if largest_component_only:
        # El grafo completo (con su gate de 12 fantasmas) ya quedó validado arriba.
        # Ahora se computan las componentes sobre la conectividad REAL (aristas
        # retenidas) y se recorta a la mayor.
        components = _connected_components(sumo_ids, graph_edges)
        comp_sizes = tuple(sorted((len(c) for c in components), reverse=True))
        if expected_component_sizes is not None:
            if comp_sizes != tuple(expected_component_sizes):
                raise ValueError(
                    f"Tamaños de componentes inesperados: reales {comp_sizes}, "
                    f"esperados {tuple(expected_component_sizes)}. Posible cambio en "
                    f"la red o en el criterio de conectividad — parar y revisar."
                )
        else:
            logger.info("Componentes conexas (tamaños desc): %s", comp_sizes)

        # Mayor componente: por nº de nodos; ante empate, la que contiene el
        # sumo_edge lexicográficamente menor (cada componente viene ordenada, así
        # que c[0] es su menor). min sobre (-len, c[0]) elige más nodos y, a igualdad,
        # el menor sumo_edge.
        lcc = min(components, key=lambda c: (-len(c), c[0]))
        lcc_set = set(lcc)
        dropped = sorted(s for s in sumo_ids if s not in lcc_set)
        provenance = {
            "derived_from": "largest connected component of full 381-node graph",
            "full_graph_nodes": n_discovered,
            "dropped_component_nodes": len(dropped),
            "dropped_edges": dropped,
        }

        # Recortar = re-derivar el subgrafo inducido reusando el código probado:
        # node_index 0..N-1 contiguos sobre los sumo_edge de la componente, ordenados.
        sumo_ids = sorted(lcc)
        idx_of = {eid: i for i, eid in enumerate(sumo_ids)}
        conn = _parse_connections(root, set(sumo_ids))
        graph_edges, n_candidates, n_retained, n_filtered = _build_edges(
            sumo_ids, idx_of, veh, conn
        )
        logger.info(
            "LCC: %d nodos, candidatas=%d, retenidas=%d, filtradas=%d; %d nodos droppeados",
            len(sumo_ids), n_candidates, n_retained, n_filtered, len(dropped),
        )

        # Artefacto separado: si se dejó el default de 381, redirigir al de LCC para
        # no pisar el mapping completo. Ruta explícita o None se respetan tal cual.
        if mapping_out is not None and Path(mapping_out) == DEFAULT_MAPPING_OUT:
            mapping_out = DEFAULT_LCC_MAPPING_OUT

    counts = {
        "nodes": len(sumo_ids),
        "candidates": n_candidates,
        "retained": n_retained,
        "filtered": n_filtered,
    }
    mapping = build_mapping(sumo_ids, graph_edges, counts, provenance=provenance)

    edge_index = torch.tensor(
        [[s for s, _t in graph_edges], [t for _s, t in graph_edges]],
        dtype=torch.long,
    )
    edge_weight = torch.ones(len(graph_edges), dtype=torch.float)  # PLACEHOLDER → Fase 4

    if mapping_out is not None:
        mapping_out = Path(mapping_out)
        mapping_out.parent.mkdir(parents=True, exist_ok=True)
        mapping_out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))

    return edge_index, edge_weight, mapping


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("== Grafo completo (381) ==")
    ei, ew, m = build_miraflores_graph(expected_nodes=381)
    print("counts:", m["counts"])
    print("edge_index shape:", tuple(ei.shape))
    print("edge_weight (placeholder):", ew.shape, "todos 1.0:", bool((ew == 1.0).all()))
    print("mapping escrito en:", DEFAULT_MAPPING_OUT)

    print("\n== Componente conexa principal (LCC, 375) ==")
    ei_lcc, ew_lcc, m_lcc = build_miraflores_graph(
        largest_component_only=True, expected_component_sizes=(375, 6)
    )
    print("counts:", m_lcc["counts"])
    print("derived_from:", m_lcc["derived_from"])
    print("dropped_component_nodes:", m_lcc["dropped_component_nodes"])
    print("dropped_edges:", m_lcc["dropped_edges"])
    print("edge_index shape:", tuple(ei_lcc.shape))
    print("mapping escrito en:", DEFAULT_LCC_MAPPING_OUT)
