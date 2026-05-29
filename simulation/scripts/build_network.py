"""Construye la red SUMO 4-vías genérica (CT-07.1) desde
``conf/network/network_params.yaml``.

Pasos:
1. Genera ``nodes.nod.xml`` y ``edges.edg.xml`` con los parámetros aprobados
   (asimetría NS 3 carriles × 300 m / EW 2 carriles × 200 m).
2. Invoca ``netconvert`` para producir ``miraflores_4way.net.xml`` con TLS en
   el centro.
3. Introspecciona el TLS con sumolib (corrección 4 del plan F0): determina
   la dirección (N/S/E/W) de cada link controlado y deriva los linkstates
   por sub-fase {NS_g, NS_y, NS_r, EW_g, EW_y, EW_r}.
4. Persiste ``linkstates.json`` y ``miraflores_4way.tllogic.add.xml`` con
   las 6 sub-fases del programa baseline (programID = "baseline").

Lefts permissive: el char es ``g`` minúscula durante la verde de su fase
(yield obligatorio al opuesto), ``r`` durante la opuesta.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import sumolib
import yaml


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
CONF_NETWORK = SIM_ROOT / "conf" / "network"
PARAMS_YAML = CONF_NETWORK / "network_params.yaml"
NODES_XML = CONF_NETWORK / "_generated_nodes.nod.xml"
EDGES_XML = CONF_NETWORK / "_generated_edges.edg.xml"
NET_XML = CONF_NETWORK / "miraflores_4way.net.xml"
TLLOGIC_XML = CONF_NETWORK / "miraflores_4way.tllogic.add.xml"
LINKSTATES_JSON = CONF_NETWORK / "linkstates.json"


def _load_params() -> dict[str, Any]:
    with open(PARAMS_YAML) as f:
        return yaml.safe_load(f)


def _write_nodes(params: dict[str, Any]) -> None:
    """Nodes: center (TLS), N/S/E/W endpoints."""
    ns_len = params["intersection"]["ns"]["approach_length_m"]
    ew_len = params["intersection"]["ew"]["approach_length_m"]
    content = f"""<nodes>
  <node id="center" x="0.0" y="0.0" type="traffic_light"/>
  <node id="N" x="0.0" y="{ns_len:.1f}" type="priority"/>
  <node id="S" x="0.0" y="-{ns_len:.1f}" type="priority"/>
  <node id="E" x="{ew_len:.1f}" y="0.0" type="priority"/>
  <node id="W" x="-{ew_len:.1f}" y="0.0" type="priority"/>
</nodes>
"""
    NODES_XML.write_text(content)


def _write_edges(params: dict[str, Any]) -> None:
    """8 edges: 4 inbound + 4 outbound. NS = 3 lanes, EW = 2 lanes. vmax = 13.89."""
    ns_lanes = params["intersection"]["ns"]["lanes_per_approach"]
    ew_lanes = params["intersection"]["ew"]["lanes_per_approach"]
    vmax = params["speeds"]["vmax_mps"]
    edges = [
        ("N_in",  "N",      "center", ns_lanes),
        ("N_out", "center", "N",      ns_lanes),
        ("S_in",  "S",      "center", ns_lanes),
        ("S_out", "center", "S",      ns_lanes),
        ("E_in",  "E",      "center", ew_lanes),
        ("E_out", "center", "E",      ew_lanes),
        ("W_in",  "W",      "center", ew_lanes),
        ("W_out", "center", "W",      ew_lanes),
    ]
    lines = ["<edges>"]
    for eid, frm, to, n_lanes in edges:
        lines.append(
            f'  <edge id="{eid}" from="{frm}" to="{to}" priority="1" '
            f'numLanes="{n_lanes}" speed="{vmax}"/>'
        )
    lines.append("</edges>")
    EDGES_XML.write_text("\n".join(lines) + "\n")


def _run_netconvert() -> None:
    cmd = [
        "netconvert",
        "--node-files", str(NODES_XML),
        "--edge-files", str(EDGES_XML),
        "--output-file", str(NET_XML),
        "--no-turnarounds", "true",
        # Que netconvert genere el TLS por defecto en "center"; lo
        # reemplazaremos por nuestro programa "baseline" via add-file.
        "--tls.default-type", "static",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("netconvert STDOUT:\n", proc.stdout, file=sys.stderr)
        print("netconvert STDERR:\n", proc.stderr, file=sys.stderr)
        raise RuntimeError("netconvert failed")


# --- Introspección y derivación de linkstates ---

# Mapeo edge_id → dirección "N/S/E/W" (de qué cara viene el flujo).
EDGE_DIR = {
    "N_in": "N",
    "S_in": "S",
    "E_in": "E",
    "W_in": "W",
}


def _link_direction(conn: sumolib.net.Connection) -> str:
    """Dirección desde la que llega el link controlado."""
    from_edge_id = conn.getFrom().getID()
    return EDGE_DIR.get(from_edge_id, "?")


def _link_movement(conn: sumolib.net.Connection) -> str:
    """Movimiento del link: 'straight', 'left', 'right', 'turn'."""
    d = conn.getDirection()
    if d == sumolib.net.Connection.LINKDIR_STRAIGHT:
        return "straight"
    if d in (sumolib.net.Connection.LINKDIR_LEFT, sumolib.net.Connection.LINKDIR_PARTLEFT):
        return "left"
    if d in (sumolib.net.Connection.LINKDIR_RIGHT, sumolib.net.Connection.LINKDIR_PARTRIGHT):
        return "right"
    return "turn"


def _derive_linkstates(net: sumolib.net.Net) -> dict[str, str]:
    """Construye los strings de estado del TLS para las 6 sub-fases.

    Reglas Option A (lefts permissive):
    - Fase NS verde: links cuyo from ∈ {N,S} con movimiento straight/right = 'G';
      lefts de N/S = 'g' (permissive, ceden a la corriente opuesta);
      links cuyo from ∈ {E,W} = 'r'.
    - Fase NS yellow: G/g → 'y'; r se mantiene 'r'.
    - Fase NS all_red: todo 'r'.
    - Simétrico para EW.
    """
    tls_list = net.getTrafficLights()
    if len(tls_list) != 1:
        raise RuntimeError(f"Esperado 1 TLS, encontrado {len(tls_list)}")
    tls = tls_list[0]
    # Connections ordenadas por TL link index.
    conns = list(tls.getConnections())
    # tls.getConnections() devuelve tuplas (fromLane, toLane, linkIndex).
    # Las normalizamos a estructuras con índice + objeto Connection real.
    indexed: list[tuple[int, sumolib.net.Connection]] = []
    for from_lane, to_lane, link_idx in conns:
        # Buscar el Connection real en outgoing del fromLane.
        edge = from_lane.getEdge()
        for c in edge.getConnections(to_lane.getEdge()):
            if c.getTLLinkIndex() == link_idx:
                indexed.append((link_idx, c))
                break
    indexed.sort(key=lambda t: t[0])

    n_links = max(i for i, _ in indexed) + 1
    if len(indexed) != n_links:
        raise RuntimeError(
            f"Connections indexadas {len(indexed)} != n_links {n_links}; "
            "algún TL link index quedó sin Connection asociada"
        )

    def _state_for(active_axis: str, sub: str) -> str:
        """active_axis ∈ {'NS','EW'}; sub ∈ {'g','y','r'}."""
        chars = []
        for _, c in indexed:
            d = _link_direction(c)
            m = _link_movement(c)
            on_axis = (active_axis == "NS" and d in ("N", "S")) or \
                      (active_axis == "EW" and d in ("E", "W"))
            if sub == "r":
                chars.append("r")
                continue
            if not on_axis:
                chars.append("r")
                continue
            # On axis activa.
            if sub == "g":
                # straight/right = G; left = g (permissive).
                chars.append("g" if m == "left" else "G")
            else:  # sub == 'y'
                chars.append("y")
        return "".join(chars)

    states = {
        "NS_g": _state_for("NS", "g"),
        "NS_y": _state_for("NS", "y"),
        "NS_r": _state_for("NS", "r"),
        "EW_g": _state_for("EW", "g"),
        "EW_y": _state_for("EW", "y"),
        "EW_r": _state_for("EW", "r"),
    }
    return states


def _write_tllogic_addfile(params: dict[str, Any], states: dict[str, str]) -> None:
    """Escribe additional-file con el programa 'baseline' (6 sub-fases)."""
    phases_yaml = {p["phase_id"]: p for p in params["tllogic"]["phases"]}
    sub_phases = [
        ("NS", "g", phases_yaml["NS"]["green_baseline_s"]),
        ("NS", "y", phases_yaml["NS"]["yellow_s"]),
        ("NS", "r", phases_yaml["NS"]["all_red_s"]),
        ("EW", "g", phases_yaml["EW"]["green_baseline_s"]),
        ("EW", "y", phases_yaml["EW"]["yellow_s"]),
        ("EW", "r", phases_yaml["EW"]["all_red_s"]),
    ]
    program_id = params["tllogic"]["program_id_baseline"]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<additional>',
        f'  <tlLogic id="center" type="static" programID="{program_id}" offset="0">',
    ]
    for phase_id, sub, dur in sub_phases:
        key = f"{phase_id}_{sub}"
        lines.append(f'    <phase duration="{dur}" state="{states[key]}"/>')
    lines.append("  </tlLogic>")
    lines.append("</additional>")
    TLLOGIC_XML.write_text("\n".join(lines) + "\n")


def main() -> int:
    params = _load_params()
    _write_nodes(params)
    _write_edges(params)
    _run_netconvert()

    net = sumolib.net.readNet(str(NET_XML), withPrograms=True)
    states = _derive_linkstates(net)
    LINKSTATES_JSON.write_text(json.dumps(states, indent=2) + "\n")
    _write_tllogic_addfile(params, states)

    print(f"Net written: {NET_XML}")
    print(f"Linkstates: {json.dumps(states, indent=2)}")
    print(f"tllogic add-file: {TLLOGIC_XML}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
