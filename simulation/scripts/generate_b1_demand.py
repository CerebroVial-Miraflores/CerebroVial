"""B1 — Generador de demanda perfil-día para Miraflores COMPLETO (descartable, GATE).

Estrategia (Opción B, nivel GRUESO — acordada 2026-06-01):
  - Ruteo con randomTrips.py + duarouter. NO od2trips, NO TAZ, NO turn-probs a mano.
  - Pesos por edge según CLASE VIAL (atributo `type`/`priority` del net), NO por `name`:
    el atributo `name` está VACÍO en los 381 edges vehiculares de miraflores.net.xml
    (auditoría 2026-06-01), así que el match por nombre de avenida es imposible.
    Pivote aprobado: arterias (primary/secondary/motorway) pesan ALTO, resto BAJO.
    Semántica: "arterias pesan alto", NO "las 6 avenidas piloto pesan alto".
  - Curva horaria 24h: se REUSA la FORMA del perfil `laborable` de
    conf/scenarios/perfil_dia_params.yaml (10 fases, doble spike AM 07-09 / PM 18-20,
    meseta diurna, valles nocturnos).

  ⚠️ NIVELES SIN CALIBRAR. Los vph por fase son PLACEHOLDER (heredados de la forma
     laborable del 4-way, que es vph POR DIRECCIÓN de UN cruce, NO una tasa de
     inserción de red completa). Acá se usan como tasa de inserción network-wide
     sólo para validar que B1 rutea. Recalibrar contra la capacidad de Miraflores
     en un paso posterior. NO usar este .rou.xml para dataset ni KPIs.

Salidas (todas en este dir scratch):
  miraflores_b1.src.xml / .dst.xml   weight-files para randomTrips --weights-prefix
  pNN_<fase>.trips.xml               trips crudos por fase (intermedios)
  miraflores_b1_laborable.rou.xml    rutas finales (duarouter)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
# scripts/ -> simulation/ -> repo-root. Todas las rutas se resuelven desde acá,
# nada machine-specific.
REPO_ROOT = HERE.parent.parent
NET = REPO_ROOT / "simulation" / "conf" / "network" / "miraflores.net.xml"
DATASET_DIR = REPO_ROOT / "simulation" / "data" / "datasets" / "miraflores_laborable_60d"

_sumo_home = os.environ.get("SUMO_HOME")
if not _sumo_home:
    sys.exit("ERROR: la variable de entorno SUMO_HOME no está seteada "
             "(necesaria para localizar randomTrips.py). Configurala antes de correr.")
SUMO_HOME = Path(_sumo_home)
RANDOMTRIPS = SUMO_HOME / "tools" / "randomTrips.py"

# Clases viales que cuentan como ARTERIA (peso alto). Las 6 avenidas piloto
# (Arequipa, Angamos, Larco, Benavides, Pardo, Cmte Espinar) caen en estas clases;
# pesar por clase sube TODAS las arterias, no sólo esas 6 (tradeoff aceptado).
ARTERIAL_TYPES = {
    "highway.motorway", "highway.motorway_link",
    "highway.primary", "highway.primary_link",
    "highway.secondary", "highway.secondary_link",
}

# Forma laborable reusada (begin_h, end_h, vph PLACEHOLDER SIN CALIBRAR).
LABORABLE_PHASES = [
    (0, 4, 150), (4, 6, 400), (6, 7, 1500), (7, 9, 2600), (9, 10, 1800),
    (10, 17, 1500), (17, 18, 1900), (18, 20, 2700), (20, 22, 1000), (22, 24, 400),
]


def lane_allows_passenger(lane: ET.Element) -> bool:
    allow = lane.get("allow")
    dis = lane.get("disallow")
    if allow is not None:
        return "passenger" in allow.split()
    if dis is not None:
        return "passenger" not in dis.split()
    return True  # sin allow/disallow => permite todo


def classify_edges(net_path: Path):
    """Devuelve (arterial_ids, other_ids) sobre los edges vehiculares."""
    root = ET.parse(net_path).getroot()
    arterial, other = [], []
    for e in root.findall("edge"):
        if e.get("function") == "internal":
            continue
        if not any(lane_allows_passenger(l) for l in e.findall("lane")):
            continue
        (arterial if e.get("type") in ARTERIAL_TYPES else other).append(e.get("id"))
    return arterial, other


def write_weight_file(path: Path, arterial, other, hi: float, lo: float, end_s: int):
    lines = ['<edgedata>', f'  <interval begin="0" end="{end_s}">']
    for eid in arterial:
        lines.append(f'    <edge id="{eid}" value="{hi}"/>')
    for eid in other:
        lines.append(f'    <edge id="{eid}" value="{lo}"/>')
    lines += ['  </interval>', '</edgedata>']
    path.write_text("\n".join(lines) + "\n")


def run_randomtrips(begin_s, end_s, vph, prefix, weights_prefix, out_trips, seed=None):
    # vph ya viene escalado por el caller; period = 3600/vph_efectivo.
    period = 3600.0 / vph
    cmd = [
        sys.executable, str(RANDOMTRIPS), "-n", str(NET),
        "--weights-prefix", str(weights_prefix), "--vclass", "passenger",
        "-b", str(begin_s), "-e", str(end_s), "-p", f"{period:.4f}",
        "--prefix", prefix, "--fringe-factor", "1", "-o", str(out_trips),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]  # reproducibilidad explícita (el gate NO seedeaba)
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="B1 demand generator (descartable gate).")
    ap.add_argument("--ratio", type=float, default=5.0, help="peso arteria:resto (default 5:1)")
    ap.add_argument("--low", type=float, default=1.0, help="peso base del resto (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="semilla randomTrips (reproducibilidad)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplicador global de vph, uniforme sobre las 10 fases (C1: 0.35)")
    ap.add_argument("--outdir", type=str, default=str(DATASET_DIR),
                    help="directorio de salida (default: el dataset versionado "
                         "simulation/data/datasets/miraflores_laborable_60d/). "
                         "gen_day.sh siempre pasa --outdir explícito a un _work dir.")
    args = ap.parse_args()

    hi, lo = args.ratio * args.low, args.low
    end_day = 24 * 3600
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    arterial, other = classify_edges(NET)
    print(f"edges vehiculares: {len(arterial) + len(other)}  "
          f"(arteria={len(arterial)} peso {hi}, resto={len(other)} peso {lo})")
    print(f"scale={args.scale} seed={args.seed} outdir={outdir}")

    src = outdir / "miraflores_b1.src.xml"
    dst = outdir / "miraflores_b1.dst.xml"
    write_weight_file(src, arterial, other, hi, lo, end_day)
    write_weight_file(dst, arterial, other, hi, lo, end_day)
    print(f"  weight-files: {src.name}, {dst.name}")

    weights_prefix = outdir / "miraflores_b1"
    trip_files = []
    for idx, (bh, eh, vph) in enumerate(LABORABLE_PHASES):
        vph_eff = vph * args.scale  # scale uniforme: baja el nivel, preserva la forma
        out = outdir / f"p{idx:02d}_{bh:02d}-{eh:02d}h.trips.xml"
        run_randomtrips(bh * 3600, eh * 3600, vph_eff, f"p{idx}_", weights_prefix, out,
                        seed=args.seed)
        trip_files.append(out)
        print(f"  fase {idx} {bh:02d}-{eh:02d}h vph={vph:4d}*{args.scale}={vph_eff:7.1f} -> {out.name}")

    print(f"\ntrips por fase generados: {len(trip_files)}")
    print("Siguiente: duarouter para rutear.")
    # Lista de trip files para duarouter, en orden.
    (outdir / "trip_files.txt").write_text(",".join(t.name for t in trip_files) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
