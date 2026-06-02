"""Dataset builder por-arista del corredor Larco (STGNN, Fase 2/3).

De un ``edgedata.xml`` (salida de meandata de una corrida SUMO; ver
``simulation/conf/corredor_larco/edgedata.add.xml``) produce una serie temporal
``tiempo × nodos × canales`` con ``jam_level`` como canal base/target.

Diseño (decisiones del Tramo 1):

* **Free-flow POR ARISTA**, NO VMAX global. Los límites del corredor varían
  (nodo Larco-Sur 16.67 m/s, nodo Schell-Este 11.11 m/s, resto 13.89), así que el
  ratio se calcula contra el free-flow de cada arista, topado por el ``maxSpeed``
  del vType (los autos nunca superan 13.89 m/s aunque el carril permita más):
  ``free_flow_edge = min(max(speed de los <lane> del net.xml), vmax_cap)``.
* **jam_level vía la función productiva D-009**: se importa
  ``ratio_to_jam_level`` de ``cerebrovial_simulation.jam_level`` (única fuente de
  verdad de los cortes Waze). Como ese paquete no está instalado en el venv de
  ``ia_prediction_service``, se bootstrapea ``simulation/src`` en ``sys.path``.
* **Topología**: los 6 nodos instrumentados y su orden (graph_index 0-5) salen de
  ``corridor_graph_mapping.json`` (Fase 1). El edgeData trae TODOS los edges; acá
  se filtra a los 6.
* **Bucket vacío** (``sampledSeconds <= 0`` o sin atributo ``speed``): convención
  heredada → flujo libre → ``jam_level = 0`` (NO se llama a ``ratio_to_jam_level``,
  que mapearía ratio<=0 a jam 5). Política configurable vía ``empty_policy``.

  LIMITACIÓN a re-verificar: un bucket vacío es AMBIGUO — puede ser "calle vacía
  por flujo libre" (jam 0, la convención que usamos) o "calle vacía porque está
  bloqueada aguas abajo y no entra nadie" (que sería jam alto). El edgeData solo
  por sí mismo no las distingue; con ``excludeEmpty="false"`` el bucket vacío sale
  con ``sampledSeconds=0`` y sin ``speed``. Mantener el flag para revisitar.

NO toca DB, ni red, ni producción. Sin ventaneo (lookback/horizonte es Fase 2/3):
este builder produce la serie completa por nodo. Forma de salida: ver ``build_corridor_series``.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# ── Bootstrap: importar la función productiva de jam_level desde simulation/src ──
# cerebrovial_simulation no está instalado en el venv de ia_prediction_service;
# jam_level.py es stdlib puro (sin torch), así que basta con ponerlo en el path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIM_SRC = _REPO_ROOT / "simulation" / "src"
if str(_SIM_SRC) not in sys.path:
    sys.path.insert(0, str(_SIM_SRC))
try:
    from cerebrovial_simulation.jam_level import ratio_to_jam_level
except ModuleNotFoundError as exc:  # pragma: no cover - guardia de entorno
    raise ModuleNotFoundError(
        f"No se pudo importar cerebrovial_simulation.jam_level desde {_SIM_SRC}. "
        "Verificá que simulation/src/cerebrovial_simulation/jam_level.py exista."
    ) from exc


DEFAULT_NET_XML = _REPO_ROOT / "simulation" / "conf" / "corredor_larco" / "corredor_larco.net.xml"
DEFAULT_MAPPING = Path(__file__).resolve().parent / "artifacts" / "corridor_graph_mapping.json"

# maxSpeed del vType `car` en demand_params.yaml (50 km/h, D-009). Los autos no
# superan esta velocidad aunque el carril permita más → tope del free-flow efectivo.
VTYPE_MAXSPEED_MPS = 13.89

# Canales derivados que van primero; índice 0 = jam_level (base/target).
_DERIVED_CHANNELS = ("jam_level", "ratio")


def load_node_order(mapping_path: Path | str = DEFAULT_MAPPING) -> list[str]:
    """Lee corridor_graph_mapping.json y devuelve los sumo_edge de los nodos
    instrumentados en orden de graph_index (0..N-1). Mismo orden que Fase 1."""
    with open(mapping_path) as f:
        mapping = json.load(f)
    nodes = sorted(mapping["nodes"], key=lambda n: n["graph_index"])
    return [n["sumo_edge"] for n in nodes]


def parse_freeflow(net_xml: Path | str, edges: set[str],
                   vmax_cap: float = VTYPE_MAXSPEED_MPS) -> dict[str, float]:
    """Free-flow efectivo por arista = min(max(speed de sus <lane>), vmax_cap).

    Replica la lectura mínima del net.xml (Fase 1 lee `length`, no `speed`). En
    este corredor todos los carriles de una arista comparten speed; tomar el max
    es robusto si difirieran.
    """
    tree = ET.parse(net_xml)
    out: dict[str, float] = {}
    for edge in tree.getroot().findall("edge"):
        eid = edge.get("id")
        if eid not in edges:
            continue
        speeds = [float(la.get("speed")) for la in edge.findall("lane") if la.get("speed")]
        if not speeds:
            continue
        out[eid] = min(max(speeds), vmax_cap)
    missing = edges - set(out)
    if missing:
        raise KeyError(f"aristas sin free-flow en el net.xml: {sorted(missing)}")
    return out


def _parse_edgedata(path: Path | str) -> dict[float, dict[str, dict[str, float]]]:
    """Parsea edgedata.xml (``<meandata>/<interval>/<edge>``) →
    ``{t_bucket_end: {edge_id: {attr_numérico: valor}}}``.

    Conserva TODOS los atributos numéricos de cada ``<edge>`` (speed, density,
    occupancy, sampledSeconds, departed, ...). Los no numéricos (p. ej. ``id``) se
    descartan. ``t_bucket`` es el ``end`` del intervalo (en segundos).
    """
    root = ET.parse(path).getroot()
    out: dict[float, dict[str, dict[str, float]]] = {}
    for interval in root.findall("interval"):
        t_end = float(interval.get("end"))
        bucket = out.setdefault(t_end, {})
        for edge in interval.findall("edge"):
            eid = edge.get("id")
            attrs: dict[str, float] = {}
            for key, val in edge.attrib.items():
                if key == "id":
                    continue
                try:
                    attrs[key] = float(val)
                except (TypeError, ValueError):
                    continue
            bucket[eid] = attrs
    return out


def build_corridor_series(
    edgedata_xml: Path | str,
    net_xml: Path | str = DEFAULT_NET_XML,
    mapping_path: Path | str = DEFAULT_MAPPING,
    vmax_cap: float = VTYPE_MAXSPEED_MPS,
    empty_policy: str = "free_flow",
) -> dict:
    """Construye la serie ``tiempo × nodos × canales`` del corredor.

    Args:
        edgedata_xml: salida meandata de una corrida (``edgedata.xml``).
        net_xml: red del corredor (para free-flow por arista).
        mapping_path: corridor_graph_mapping.json (orden de los nodos).
        vmax_cap: tope del free-flow (maxSpeed del vType; ver VTYPE_MAXSPEED_MPS).
        empty_policy: tratamiento del bucket vacío (sampledSeconds<=0 / sin speed):
            ``"free_flow"`` (default) → ratio 1.0, jam_level 0 (convención heredada);
            ``"blocked"`` → ratio 0.0, jam_level 5 (hipótesis de bloqueo, a re-verificar).

    Returns:
        dict con:
          - ``data``: np.ndarray ``[T, N, C]`` (float64).
          - ``channels``: list[str], nombres de canal; índice 0 = ``jam_level`` (target).
          - ``node_order``: list[str], los N sumo_edge en orden graph_index.
          - ``times``: list[float], los T fin-de-bucket (s) ascendentes.
          - ``free_flow``: dict {edge: free_flow_mps} usado.
          - ``empty_policy``: la política aplicada.
    """
    if empty_policy not in ("free_flow", "blocked"):
        raise ValueError(f"empty_policy desconocida: {empty_policy!r}")

    node_order = load_node_order(mapping_path)
    free_flow = parse_freeflow(net_xml, set(node_order), vmax_cap=vmax_cap)
    buckets = _parse_edgedata(edgedata_xml)
    times = sorted(buckets)

    # Canales raw = unión de atributos numéricos vistos en los nodos del corredor.
    raw_attrs: set[str] = set()
    for t in times:
        for edge in node_order:
            raw_attrs.update(buckets[t].get(edge, {}))
    raw_channels = sorted(raw_attrs)
    channels = list(_DERIVED_CHANNELS) + raw_channels
    ch_index = {name: i for i, name in enumerate(channels)}

    T, N, C = len(times), len(node_order), len(channels)
    data = np.full((T, N, C), np.nan, dtype=np.float64)

    empty_jam = 0 if empty_policy == "free_flow" else 5
    empty_ratio = 1.0 if empty_policy == "free_flow" else 0.0

    for ti, t in enumerate(times):
        for ni, edge in enumerate(node_order):
            rec = buckets[t].get(edge)
            sampled = rec.get("sampledSeconds", 0.0) if rec else 0.0
            has_speed = bool(rec) and "speed" in rec
            is_empty = (rec is None) or (sampled <= 0.0) or (not has_speed)

            if is_empty:
                jam = empty_jam
                ratio = empty_ratio
            else:
                speed = rec["speed"]
                ratio = speed / free_flow[edge]
                jam = ratio_to_jam_level(ratio)

            data[ti, ni, ch_index["jam_level"]] = float(jam)
            data[ti, ni, ch_index["ratio"]] = float(ratio)
            # Canales raw: lo que emitió SUMO; ausente → NaN (sampledSeconds marca el vacío).
            if rec:
                for name, val in rec.items():
                    data[ti, ni, ch_index[name]] = val

    return {
        "data": data,
        "channels": channels,
        "node_order": node_order,
        "times": times,
        "free_flow": free_flow,
        "empty_policy": empty_policy,
    }
