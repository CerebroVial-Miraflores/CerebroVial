"""Generador PURO de config de nodos para el control adaptativo, derivado del net (F3a).

Deriva automáticamente, de cada TLS de 2 fases de ``miraflores.net.xml``, la estructura
que hoy está hardcodeada a mano en ``corredor_adaptive.py`` (``NodeCfg`` / ``PhaseCfg`` +
los linkstates de ``linkstates_corredor.json``) para los 2 nodos de Larco. La auditoría
previa estableció que los 54 TLS de 2 fases son uniformemente derivables (fase→edges
disjunto 54/54, linkstates copiables del net, bus/bici auto-excluible).

ALCANCE F3a — generador + validación. NO toca el adaptador en vivo, NO resuelve el mapeo
al motor, NO corre TraCI (eso es F3b).

Decisiones de diseño (fijadas, no reabrir):
  - **Sensor: solo movimientos G protegidos.** ``PhaseSpec.edges`` y ``.lanes`` cuentan
    SOLO los links con 'G' (verde protegido), NO los 'g' permisivos/menores.
  - **linkstates copiados COMPLETOS del net** (con 'G' y 'g'): son para el bootstrap del
    programa de 6 sub-fases del adaptador, distinto del sensor.
  - **seed_green derivado del net:** el verde inicial de cada fase = la duración de esa
    fase verde en el programa de netconvert del TLS.
  - **carriles generales:** por fase, excluyendo bus/bici con ``lane_allows_passenger``
    (REUSADA de ``generate_b1_demand.py``, no reimplementada — ver nota de import abajo).

Campos PENDIENTES para F3b (se dejan vacíos/None acá):
  - ``NodeSpec.intersection_id`` = None — el mapeo tls_id→node_id del motor lo decide F3b
    (depende de qué pide ``resolve_node_id``: reusar ``sumo_<id>`` o sembrar nodos nuevos).
  - ``PhaseSpec.downstream_edges`` = [] y ``turn_ratio`` = 1.0 (default) — MP-red es scope
    opcional diferido; F3a no lo deriva.

Orden de fases: NET order (P0 = primera fase verde del programa, P1 = segunda). El corredor
reordenó a [LARCO, TRANSV] por semántica; los 54 genéricos no tienen esa semántica, así que
se preserva el orden del net. Lo que importa para el motor son los conteos de carriles, no
la etiqueta.

Uso (validación / inspección):
    SUMO_HOME=... python -m cerebrovial_simulation.traci_adapter.node_config_gen \\
        --net simulation/conf/network/miraflores.net.xml [--validate-larco] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
# scripts/ vive en simulation/scripts/. lane_allows_passenger se REUSA de ahí.
SCRIPTS_DIR = HERE.parent.parent.parent / "scripts"


def _lane_allows_passenger():
    """Import perezoso de ``lane_allows_passenger`` de generate_b1_demand.

    Se importa dentro de la función (no al tope del módulo) PORQUE
    ``generate_b1_demand.py`` hace ``sys.exit`` a nivel de módulo si ``SUMO_HOME`` no
    está seteado (líneas 41-46). Importarlo arriba rompería hasta el simple ``import`` de
    este generador sin SUMO_HOME. Así el módulo importa limpio; solo derivar config exige
    SUMO_HOME (por el import reusado, no por la lógica). Reuso, NO reimplementación.
    """
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from generate_b1_demand import lane_allows_passenger  # noqa: E402
    return lane_allows_passenger


# --- Estructura de salida: espeja NodeCfg/PhaseCfg de corredor_adaptive.py. F3b la enchufa.
@dataclass
class PhaseSpec:
    phase_id: str                 # "P0" | "P1" (orden del net)
    edges: list[str]              # from-edges servidos por links G protegidos (sensor)
    lanes: float                  # carriles GENERALES (bus/bici excluidos) con link G
    downstream_edges: list[str] = field(default_factory=list)  # PENDIENTE F3b (MP-red)
    turn_ratio: float = 1.0       # PENDIENTE F3b (MP-red)


@dataclass
class NodeSpec:
    tls_id: str
    intersection_id: Optional[str]   # PENDIENTE F3b (mapeo al motor)
    phases: list[PhaseSpec]          # orden = [P0, P1] del net
    seed_green: dict                 # {"P0": dur, "P1": dur} de netconvert
    linkstates: dict                 # {"P0_g","P0_y","P0_r","P1_g","P1_y","P1_r"} del net

    def to_dict(self) -> dict:
        return asdict(self)


def _is_green(state: str) -> bool:
    return ("G" in state) or ("g" in state)


def _is_yellow(state: str) -> bool:
    return ("y" in state) or ("Y" in state)


def _parse_net(net_path: Path):
    """Una pasada: lanes (ET elements por edge/idx), connections por tl, tlLogic phases."""
    root = ET.parse(str(net_path)).getroot()
    # edge -> {laneIdx: <lane> element} (para lane_allows_passenger)
    edge_lanes: dict[str, dict[int, ET.Element]] = {}
    for e in root.findall("edge"):
        if e.get("function") == "internal":
            continue
        lanes = {}
        for ln in e.findall("lane"):
            lanes[int(ln.get("index"))] = ln
        edge_lanes[e.get("id")] = lanes
    # tl -> {linkIndex: (from_edge, from_lane)}
    conns: dict[str, dict[int, tuple]] = {}
    for c in root.findall("connection"):
        tl = c.get("tl")
        if tl is None:
            continue
        conns.setdefault(tl, {})[int(c.get("linkIndex"))] = (
            c.get("from"), int(c.get("fromLane")))
    # tlLogic id -> list of (state, duration)
    programs: dict[str, list[tuple]] = {}
    for t in root.findall("tlLogic"):
        programs[t.get("id")] = [
            (p.get("state"), float(p.get("duration"))) for p in t.findall("phase")]
    return edge_lanes, conns, programs


def find_two_phase_tls(net_path: Path) -> list[str]:
    """Los TLS con exactamente 2 fases-verde (los 54 de la auditoría)."""
    root = ET.parse(str(net_path)).getroot()
    out = []
    for t in root.findall("tlLogic"):
        greens = [p.get("state") for p in t.findall("phase") if _is_green(p.get("state"))]
        if len(greens) == 2:
            out.append(t.get("id"))
    return out


def _derive_one(tls_id: str, edge_lanes, conns, program, allows_passenger) -> NodeSpec:
    states = [s for s, _ in program]
    durs = [d for _, d in program]
    green_idx = [i for i, s in enumerate(states) if _is_green(s)]
    if len(green_idx) != 2:
        raise ValueError(f"{tls_id}: esperaba 2 fases verdes, hay {len(green_idx)}")
    L = len(states[0])
    tl_conns = conns.get(tls_id, {})

    phases, seed_green, linkstates = [], {}, {}
    for k, gi in enumerate(green_idx):
        pid = f"P{k}"
        gstate = states[gi]
        # yellow = la fase siguiente (programa netconvert: G,y,G,y[,r]).
        yi = gi + 1
        ystate = states[yi] if yi < len(states) and _is_yellow(states[yi]) else "y" * L
        # SENSOR: solo links 'G' protegidos (no 'g' permisivos).
        edges_set, lane_pairs = [], set()
        for li, ch in enumerate(gstate):
            if ch != "G":
                continue
            if li not in tl_conns:
                continue
            fr, fl = tl_conns[li]
            ln = edge_lanes.get(fr, {}).get(fl)
            is_general = allows_passenger(ln) if ln is not None else True
            if not is_general:
                continue                       # bus/bici excluido
            if fr not in edges_set:
                edges_set.append(fr)
            lane_pairs.add((fr, fl))           # carril general distinto que descarga
        phases.append(PhaseSpec(phase_id=pid, edges=edges_set, lanes=float(len(lane_pairs))))
        seed_green[pid] = durs[gi]
        linkstates[f"{pid}_g"] = gstate        # COMPLETO (G y g) para el bootstrap
        linkstates[f"{pid}_y"] = ystate
        linkstates[f"{pid}_r"] = "r" * L

    return NodeSpec(tls_id=tls_id, intersection_id=None, phases=phases,
                    seed_green=seed_green, linkstates=linkstates)


def derive_node_specs(net_path: Path, tls_ids: Optional[list[str]] = None) -> list[NodeSpec]:
    """Deriva un NodeSpec por TLS de 2 fases. ``tls_ids=None`` ⇒ autodetecta los 54."""
    edge_lanes, conns, programs = _parse_net(net_path)
    allows_passenger = _lane_allows_passenger()
    if tls_ids is None:
        tls_ids = find_two_phase_tls(net_path)
    specs = []
    for tid in tls_ids:
        if tid not in programs:
            raise KeyError(f"TLS {tid} no está en el net")
        specs.append(_derive_one(tid, edge_lanes, conns, programs[tid], allows_passenger))
    return specs


# --- CLI: inspección / validación (no toca nada) ---
_LARCO = {
    "133925753": "schell",
    "cluster_108178119_263630444_2673400749_3245705958_#6more": "benavides",
}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Generador puro de NodeSpec desde el net (F3a).")
    ap.add_argument("--net", required=True, type=Path)
    ap.add_argument("--validate-larco", action="store_true",
                    help="deriva los 2 TLS de Larco y compara conteos vs hand-coding")
    ap.add_argument("--json", type=Path, default=None, help="vuelca los NodeSpec a JSON")
    args = ap.parse_args(argv)

    specs = derive_node_specs(args.net)
    print(f"TLS de 2 fases detectados: {len(specs)}")

    # resumen de carriles generales por fase + anomalías
    lane_counts = [ph.lanes for s in specs for ph in s.phases]
    zero_lane = [(s.tls_id, ph.phase_id) for s in specs for ph in s.phases if ph.lanes == 0]
    overlap = [s.tls_id for s in specs
               if set(s.phases[0].edges) & set(s.phases[1].edges)]
    print(f"  carriles generales/fase: min={int(min(lane_counts))} "
          f"max={int(max(lane_counts))}  (n={len(lane_counts)} fases)")
    print(f"  fases con 0 carriles generales (anomalía): {len(zero_lane)} {zero_lane[:5]}")
    print(f"  TLS con solape de edges entre fases (anomalía): {len(overlap)} {overlap[:5]}")

    if args.validate_larco:
        print("\n--- VALIDACIÓN LARCO (conteos de carriles, set por TLS) ---")
        by_id = {s.tls_id: s for s in specs}
        expected = {"133925753": {2, 3}, "cluster_108178119_263630444_2673400749_3245705958_#6more": {2, 4}}
        for tid, name in _LARCO.items():
            s = by_id[tid]
            got = {int(ph.lanes) for ph in s.phases}
            ok = got == expected[tid]
            detail = ", ".join(f"{ph.phase_id}={int(ph.lanes)}gen edges={ph.edges}" for ph in s.phases)
            print(f"  {name:9s} {tid[:42]:42s} lanes={sorted(got)} "
                  f"esperado={sorted(expected[tid])} {'OK' if ok else 'MISMATCH'}")
            print(f"            seed_green={s.seed_green}  {detail}")

    if args.json:
        args.json.write_text(json.dumps([s.to_dict() for s in specs], indent=2, ensure_ascii=False))
        print(f"\nJSON → {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
