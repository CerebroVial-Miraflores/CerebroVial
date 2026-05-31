"""Genera ``conf/scenarios/routes/{perfil}.rou.xml`` (uno por perfil-día) desde
``conf/scenarios/perfil_dia_params.yaml``.

A diferencia de ``generate_routes.py`` (demanda constante de TTH-07, un `<flow>`
por dirección), acá cada perfil tiene demanda VARIABLE en el tiempo: una secuencia
de fases con `vehsPerHour` distinto. Por cada (fase, dirección) emite un `<flow>`
con `begin/end` en segundos (`begin_h*3600`) e id único.

Eje dominante N-S con `vehsPerHour=vph`; secundario E-W = `round(vph / secondary_ratio)`.
El seed se controla al invocar sumo (`--seed N`), NO al generar el archivo de rutas
— el .rou.xml es determinístico por perfil.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
PARAMS = SIM_ROOT / "conf" / "scenarios" / "perfil_dia_params.yaml"
ROUTES_DIR = SIM_ROOT / "conf" / "scenarios" / "routes"


# Mapeo origin → destino opuesto (straight). Dominante = N-S, secundario = E-W.
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
DOMINANT = {"N", "S"}


def _route_xml(orig: str, dest: str) -> str:
    return f'  <route id="r_{orig}_to_{dest}" edges="{orig}_in {dest}_out"/>'


def _flow_xml(orig: str, dest: str, idx: int, vph: int, begin_s: int, end_s: int) -> str:
    return (
        f'  <flow id="f_{orig}_{idx}" type="car" route="r_{orig}_to_{dest}" '
        f'begin="{begin_s}" end="{end_s}" vehsPerHour="{vph}" departLane="best" '
        f'departSpeed="max"/>'
    )


def _vph_for(orig: str, base_vph: int, secondary_ratio: int) -> int:
    """vph dominante para N/S; derivado para E/W."""
    if orig in DOMINANT:
        return base_vph
    return round(base_vph / secondary_ratio)


def generate_pattern(name: str, pattern: dict, secondary_ratio: int) -> Path:
    out_file = ROUTES_DIR / f"{name}.rou.xml"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- {name}: {pattern["description"]} -->',
        '<routes>',
        '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.89"/>',
    ]
    # Rutas straight-only (una por dirección).
    for orig, dest in OPPOSITE.items():
        lines.append(_route_xml(orig, dest))
    # Flows: una secuencia por fase × dirección.
    for idx, phase in enumerate(pattern["phases"]):
        begin_s = int(phase["begin_h"] * 3600)
        end_s = int(phase["end_h"] * 3600)
        base_vph = int(phase["vph"])
        for orig, dest in OPPOSITE.items():
            vph = _vph_for(orig, base_vph, secondary_ratio)
            if vph <= 0:
                continue
            lines.append(_flow_xml(orig, dest, idx, vph, begin_s, end_s))
    lines.append('</routes>')
    out_file.write_text("\n".join(lines) + "\n")
    return out_file


def main() -> int:
    with open(PARAMS) as f:
        cfg = yaml.safe_load(f)
    ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    secondary_ratio = cfg["secondary_ratio"]
    for name, pattern in cfg["patterns"].items():
        out = generate_pattern(name, pattern, secondary_ratio)
        print(f"  → {out.relative_to(SIM_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
