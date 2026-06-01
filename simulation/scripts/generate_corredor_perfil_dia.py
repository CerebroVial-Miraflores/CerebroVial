"""Genera ``conf/corredor_larco/routes/perfil_dia_{profile}.rou.xml`` combinando
las 11 rutas OD de ``conf/corredor_larco/demand_params.yaml`` con un perfil-día
por FASES de ``conf/corredor_larco/perfil_dia_corredor.yaml``.

A diferencia de ``generate_corredor_routes.py`` (demanda constante de 30 min, un
``<flow>`` por OD con ``--scale`` GLOBAL), acá la demanda escala POR FASE: por
cada fase del perfil se emiten los 11 ``<flow>`` OD con
``vehsPerHour = round(base_vph_OD × scale_fase)``, ``begin/end`` en segundos
(``begin_h*3600``) e id único ``f_{flow_id}_{idx_fase}``. Reusa las mismas rutas,
``<vType>`` y turn ratios del baseline — NO los redefine, los lee.

El seed se controla al invocar sumo, NO al generar — el .rou.xml es determinístico
por perfil. ``--seed`` se acepta por paridad con los scripts de barrido, pero como
el archivo de rutas es independiente de la semilla NO entra en el nombre por
defecto (mismo contenido ⇒ mismo archivo); usá ``--out`` para diferenciar a mano.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
PARAMS = SIM_ROOT / "conf" / "corredor_larco" / "demand_params.yaml"
PERFIL = SIM_ROOT / "conf" / "corredor_larco" / "perfil_dia_corredor.yaml"
ROUTES_DIR = SIM_ROOT / "conf" / "corredor_larco" / "routes"


def _vtype_xml(vt: dict) -> str:
    return (
        f'  <vType id="{vt["id"]}" accel="{vt["accel"]}" decel="{vt["decel"]}" '
        f'sigma="{vt["sigma"]}" length="{vt["length"]}" maxSpeed="{vt["maxSpeed"]}"/>'
    )


def _route_xml(route_id: str, edges: str) -> str:
    return f'  <route id="{route_id}" edges="{edges}"/>'


def _flow_xml(flow_id: str, idx: int, route_id: str, vph: int, vtype_id: str,
              begin_s: int, end_s: int) -> str:
    return (
        f'  <flow id="f_{flow_id}_{idx}" type="{vtype_id}" route="{route_id}" '
        f'begin="{begin_s}" end="{end_s}" vehsPerHour="{vph}" departLane="best" '
        f'departSpeed="max"/>'
    )


def generate(demand_cfg: dict, perfil_cfg: dict, profile: str = "laborable",
             out_file: Path | None = None) -> Path:
    vt = demand_cfg["vtype"]
    routes = demand_cfg["routes"]
    flows = demand_cfg["flows_vph"]

    if profile not in perfil_cfg["profiles"]:
        raise KeyError(f"perfil '{profile}' no está en perfil_dia_corredor.yaml")
    pattern = perfil_cfg["profiles"][profile]
    phases = pattern["phases"]

    if out_file is None:
        out_file = ROUTES_DIR / f"perfil_dia_{profile}.rou.xml"

    header = (f'<!-- perfil-día {profile} corredor Larco — demanda por fases×OD '
              f'(STGNN). Generado desde demand_params.yaml + perfil_dia_corredor.yaml; '
              'NO editar a mano. -->')

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        header,
        '<routes>',
        _vtype_xml(vt),
    ]
    # Rutas (una vez, al tope, en el orden del YAML para trazabilidad).
    for route_id, edges in routes.items():
        lines.append(_route_xml(route_id, edges))
    # Flows: por cada fase, los 11 OD con vehsPerHour escalado por el scale de la fase.
    total = 0
    for idx, phase in enumerate(phases):
        begin_s = int(phase["begin_h"] * 3600)
        end_s = int(phase["end_h"] * 3600)
        scale = float(phase["scale"])
        for flow_id, base_vph in flows.items():
            vph = round(base_vph * scale)
            if vph <= 0:
                continue
            if flow_id not in routes:
                raise KeyError(f"flow '{flow_id}' sin ruta correspondiente en routes:")
            lines.append(_flow_xml(flow_id, idx, flow_id, vph, vt["id"], begin_s, end_s))
            total += vph
    lines.append('</routes>')

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n")
    rel = out_file.relative_to(SIM_ROOT) if out_file.is_relative_to(SIM_ROOT) else out_file
    print(f"  → {rel}  (perfil={profile}, {len(phases)} fases, {total} veh/h sumados)")
    return out_file


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Genera rutas del corredor con demanda variable por fases (perfil-día).")
    ap.add_argument("--profile", default="laborable",
                    help="perfil de perfil_dia_corredor.yaml (default laborable)")
    ap.add_argument("--seed", type=int, default=42,
                    help="semilla (paridad con barridos; el .rou.xml es determinístico, no entra en el nombre)")
    ap.add_argument("--out", type=Path, default=None,
                    help="ruta de salida (default conf/.../routes/perfil_dia_{profile}.rou.xml)")
    args = ap.parse_args()

    with open(PARAMS) as f:
        demand_cfg = yaml.safe_load(f)
    with open(PERFIL) as f:
        perfil_cfg = yaml.safe_load(f)
    generate(demand_cfg, perfil_cfg, profile=args.profile, out_file=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
