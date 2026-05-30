"""Genera ``conf/corredor_larco/routes/peak_s_n.rou.xml`` desde
``conf/corredor_larco/demand_params.yaml``.

Flujos OD EXPLÍCITOS: una ``<route>`` con lista de edges completa por par OD y
un ``<flow>`` con ``vehsPerHour`` fijo. El seed se controla al invocar sumo
(``--seed N``), NO al generar — el .rou.xml es determinístico.

Mismo patrón que ``generate_routes.py`` (TTH-07), adaptado a rutas con giros
sobre la red real del corredor en lugar de straight-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
PARAMS = SIM_ROOT / "conf" / "corredor_larco" / "demand_params.yaml"
ROUTES_DIR = SIM_ROOT / "conf" / "corredor_larco" / "routes"


def _vtype_xml(vt: dict) -> str:
    return (
        f'  <vType id="{vt["id"]}" accel="{vt["accel"]}" decel="{vt["decel"]}" '
        f'sigma="{vt["sigma"]}" length="{vt["length"]}" maxSpeed="{vt["maxSpeed"]}"/>'
    )


def _route_xml(route_id: str, edges: str) -> str:
    return f'  <route id="{route_id}" edges="{edges}"/>'


def _flow_xml(flow_id: str, route_id: str, vph: int, vtype_id: str, duration_s: int) -> str:
    return (
        f'  <flow id="f_{flow_id}" type="{vtype_id}" route="{route_id}" '
        f'begin="0" end="{duration_s}" vehsPerHour="{vph}" departLane="best" '
        f'departSpeed="max"/>'
    )


def generate(cfg: dict) -> Path:
    duration_s = cfg["duration_s"]
    vt = cfg["vtype"]
    routes = cfg["routes"]
    flows = cfg["flows_vph"]
    out_file = ROUTES_DIR / "peak_s_n.rou.xml"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!-- peak S→N corredor Larco — flujos OD explícitos (ETAPA 2 baseline). '
        'Generado desde demand_params.yaml; NO editar a mano. -->',
        '<routes>',
        _vtype_xml(vt),
    ]
    # Rutas (en el orden del YAML para trazabilidad).
    for route_id, edges in routes.items():
        lines.append(_route_xml(route_id, edges))
    # Flows.
    total = 0
    for flow_id, vph in flows.items():
        if vph <= 0:
            continue
        if flow_id not in routes:
            raise KeyError(f"flow '{flow_id}' sin ruta correspondiente en routes:")
        lines.append(_flow_xml(flow_id, flow_id, vph, vt["id"], duration_s))
        total += vph
    lines.append('</routes>')

    ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n")
    print(f"  → {out_file.relative_to(SIM_ROOT)}  ({len(flows)} flujos, {total} veh/h total)")
    return out_file


def main() -> int:
    with open(PARAMS) as f:
        cfg = yaml.safe_load(f)
    generate(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
