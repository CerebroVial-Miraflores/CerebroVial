"""Genera ``conf/scenarios/routes/{pattern}.rou.xml`` (uno por patrón) desde
``conf/scenarios/pattern_params.yaml``.

Usa `<flow>` elements con `vehsPerHour` y `route` straight-only por origen.
El seed se controla al invocar sumo (`--seed N`), NO al generar el archivo
de rutas — el .rou.xml es determinístico por patrón.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
PARAMS = SIM_ROOT / "conf" / "scenarios" / "pattern_params.yaml"
ROUTES_DIR = SIM_ROOT / "conf" / "scenarios" / "routes"


# Mapeo origin → destino opuesto (straight).
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _route_xml(orig: str, dest: str) -> str:
    return f'  <route id="r_{orig}_to_{dest}" edges="{orig}_in {dest}_out"/>'


def _flow_xml(orig: str, dest: str, vph: int, duration_s: int) -> str:
    return (
        f'  <flow id="f_{orig}" type="car" route="r_{orig}_to_{dest}" '
        f'begin="0" end="{duration_s}" vehsPerHour="{vph}" departLane="best" '
        f'departSpeed="max"/>'
    )


def generate_pattern(pattern_name: str, pattern: dict, duration_s: int) -> Path:
    flows = pattern["flows_vph"]
    out_file = ROUTES_DIR / f"{pattern_name}.rou.xml"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- {pattern_name}: {pattern["description"]} -->',
        '<routes>',
        '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.89"/>',
    ]
    # Rutas straight-only.
    for orig, dest in OPPOSITE.items():
        lines.append(_route_xml(orig, dest))
    # Flows.
    for orig, vph in flows.items():
        if vph <= 0:
            continue
        dest = OPPOSITE[orig]
        lines.append(_flow_xml(orig, dest, vph, duration_s))
    lines.append('</routes>')
    out_file.write_text("\n".join(lines) + "\n")
    return out_file


def main() -> int:
    with open(PARAMS) as f:
        cfg = yaml.safe_load(f)
    ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    duration_s = cfg["duration_s"]
    for name, pattern in cfg["patterns"].items():
        out = generate_pattern(name, pattern, duration_s)
        print(f"  → {out.relative_to(SIM_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
