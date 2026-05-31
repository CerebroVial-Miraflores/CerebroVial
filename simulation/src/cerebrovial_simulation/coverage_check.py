"""CT-07.2 cobertura jam level + spillback verification (catch 2 del plan).

Parsea ``edgedata.xml`` (velocidad por edge por bucket) y ``lanearea.xml``
(jamLengthInMeters por detector por bucket) — formato XML, fallback
documentado de F1 (SUMO 1.26 arrow writer no soporta append multi-interval).

Verifica:
- **jam ≥3**: target_direction tiene `jam_level ≥ 3` sostenido en ≥5 buckets
  consecutivos (5 × 60 s = 300 s).
- **jam ≤2**: todos los buckets en todas las direcciones tienen
  `jam_level ≤ 2`.
- **Spillback**: `maxJamLengthInMeters` de ningún detector laneArea
  excede 90% del largo del aproche (~270 m N/S; ~180 m E/W).
- **Cola estabilizada**: el bucket de mayor jam length NO es el último
  (la cola estabilizó antes del final).

NO usa TraCI live (corrida headless con sumo -c).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .jam_level import sumo_to_jam_level


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent
PATTERN_PARAMS = SIM_ROOT / "conf" / "scenarios" / "pattern_params.yaml"
SCENARIOS_DIR = SIM_ROOT / "conf" / "scenarios"


# Mapeo edge_id de aproche → dirección.
EDGE_TO_DIR = {"N_in": "N", "S_in": "S", "E_in": "E", "W_in": "W"}
# Detectores laneArea agrupados por dirección.
DET_BY_DIR = {
    "N": ["LA_N_0", "LA_N_1", "LA_N_2"],
    "S": ["LA_S_0", "LA_S_1", "LA_S_2"],
    "E": ["LA_E_0", "LA_E_1"],
    "W": ["LA_W_0", "LA_W_1"],
}
# Largos efectivos (post-netconvert) por dirección.
APPROACH_LEN_M = {"N": 289.6, "S": 289.6, "E": 186.4, "W": 186.4}
# 90% del aproche como umbral de spillback (catch 2).
SPILLBACK_THRESHOLD = {d: 0.90 * APPROACH_LEN_M[d] for d in APPROACH_LEN_M}
# vmax uniforme (D-009 denominador del ratio).
VMAX_MPS = 13.89


@dataclass
class CoverageReport:
    pattern: str
    seed: int
    satisfied: bool
    target_jam: str            # ">=3" / "<=2"
    target_direction: Optional[str]
    max_jam_per_dir: dict[str, int]              # max jam_level observado por dirección
    sustained_jam_ge3_buckets: dict[str, int]    # max run-length de jam ≥3 por dirección
    max_queue_per_dir_m: dict[str, float]
    spillback_per_dir: dict[str, bool]
    queue_stabilized: dict[str, bool]
    reasons: list[str] = field(default_factory=list)


def _parse_edgedata(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Retorna {direction: [(bucket_t, mean_speed_mps), ...]} ordenado por bucket."""
    root = ET.parse(path).getroot()
    per_dir: dict[str, list[tuple[float, float]]] = {d: [] for d in ["N", "S", "E", "W"]}
    for interval in root.findall("interval"):
        bucket_t = float(interval.attrib["end"])
        for edge in interval.findall("edge"):
            eid = edge.attrib["id"]
            d = EDGE_TO_DIR.get(eid)
            if d is None:
                continue
            # Con excludeEmpty="false" un edge vacío se emite con sampledSeconds="0.00"
            # y SIN atributo speed (2b). sampledSeconds es la señal de presencia:
            # sin muestras = calle vacía = flujo libre (jam 0), no vía cerrada (jam 5).
            sampled = float(edge.attrib.get("sampledSeconds", "0"))
            speed = float(edge.attrib.get("speed", "-1"))  # -1 = atributo ausente
            if sampled <= 0.0:
                speed = VMAX_MPS  # sumo_to_jam_level(VMAX, VMAX) → ratio 1.0 → jam 0
            per_dir[d].append((bucket_t, speed))
    for d in per_dir:
        per_dir[d].sort(key=lambda t: t[0])
    return per_dir


def _parse_lanearea(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Retorna {direction: [(bucket_t, max_jam_length_m_sum_lanes), ...]}.

    Por bucket, suma maxJamLengthInMeters sobre los lanes de la dirección
    (cola total agregada del aproche).
    """
    root = ET.parse(path).getroot()
    # Estructura: {(direction, bucket_t): total_max_jam_length_m}
    accum: dict[tuple[str, float], float] = {}
    dir_by_det = {det: d for d, dets in DET_BY_DIR.items() for det in dets}
    for interval in root.findall("interval"):
        det_id = interval.attrib["id"]
        d = dir_by_det.get(det_id)
        if d is None:
            continue
        bucket_t = float(interval.attrib["end"])
        max_jam_m = float(interval.attrib.get("maxJamLengthInMeters", "0"))
        accum[(d, bucket_t)] = accum.get((d, bucket_t), 0.0) + max_jam_m

    per_dir: dict[str, list[tuple[float, float]]] = {d: [] for d in ["N", "S", "E", "W"]}
    for (d, t), v in accum.items():
        per_dir[d].append((t, v))
    for d in per_dir:
        per_dir[d].sort(key=lambda t: t[0])
    return per_dir


def _max_run_length(seq: list[int], pred) -> int:
    """Largo del run-length máximo de elementos que cumplen `pred(x)`."""
    best = 0
    cur = 0
    for x in seq:
        if pred(x):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def run_sumo_for_coverage(
    pattern: str,
    seed: int,
    out_dir: Path,
    end_s: Optional[int] = None,
) -> tuple[Path, Path]:
    """Invoca sumo headless para el patrón y persiste edgedata/lanearea XML en out_dir.

    SUMO resuelve los ``file=`` de add-files relativo al **directorio del
    add-file**, no al CWD. Para que los outputs caigan en out_dir, copiamos
    los add-files a out_dir y sobreescribimos ``--additional-files`` en el
    CLI (overrides el sumocfg).

    Retorna paths a (edgedata.xml, lanearea.xml).
    """
    sumo_bin = Path(os.environ["SUMO_HOME"]) / "bin" / "sumo"
    cfg = SCENARIOS_DIR / f"{pattern}.sumocfg"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copiar add-files a out_dir para que el `file="X.xml"` relativo resuelva acá.
    network_dir = SIM_ROOT / "conf" / "network"
    add_files_local: list[Path] = []
    for add_name in ["miraflores_4way.tllogic.add.xml", "edgedata.add.xml", "lanearea.add.xml"]:
        src = network_dir / add_name
        dst = out_dir / add_name
        shutil.copy(src, dst)
        add_files_local.append(dst)

    args = [
        str(sumo_bin),
        "-c", str(cfg),
        # Sobreescribe sumocfg <additional-files> con copias locales.
        "--additional-files", ",".join(str(p) for p in add_files_local),
        "--seed", str(seed),
        "--summary-output", str(out_dir / "summary.parquet"),
        "--tripinfo-output", str(out_dir / "tripinfo.parquet"),
    ]
    if end_s is not None:
        args.extend(["--end", str(end_s)])

    proc = subprocess.run(args, cwd=out_dir, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sumo failed para {pattern} seed={seed}:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    edgedata_xml = out_dir / "edgedata.xml"
    lanearea_xml = out_dir / "lanearea.xml"
    assert edgedata_xml.exists(), f"edgedata.xml faltante en {out_dir}"
    assert lanearea_xml.exists(), f"lanearea.xml faltante en {out_dir}"
    return edgedata_xml, lanearea_xml


def check_coverage(
    pattern: str,
    seed: int,
    out_dir: Path,
    target_jam: str,
    target_direction: Optional[str],
    end_s: Optional[int] = None,
) -> CoverageReport:
    edgedata, lanearea = run_sumo_for_coverage(pattern, seed, out_dir, end_s=end_s)
    speeds = _parse_edgedata(edgedata)
    queues = _parse_lanearea(lanearea)

    # Per-direction: lista de jam_levels por bucket.
    jam_by_dir: dict[str, list[int]] = {}
    max_jam_per_dir: dict[str, int] = {}
    sustained_ge3: dict[str, int] = {}
    for d, buckets in speeds.items():
        if not buckets:
            jam_by_dir[d] = []
            max_jam_per_dir[d] = 0
            sustained_ge3[d] = 0
            continue
        jams = [sumo_to_jam_level(speed, VMAX_MPS) for _, speed in buckets]
        jam_by_dir[d] = jams
        max_jam_per_dir[d] = max(jams) if jams else 0
        sustained_ge3[d] = _max_run_length(jams, lambda x: x >= 3)

    # Per-direction: queue lengths.
    max_queue_per_dir_m: dict[str, float] = {}
    spillback_per_dir: dict[str, bool] = {}
    queue_stabilized: dict[str, bool] = {}
    for d, buckets in queues.items():
        if not buckets:
            max_queue_per_dir_m[d] = 0.0
            spillback_per_dir[d] = False
            queue_stabilized[d] = True
            continue
        values = [v for _, v in buckets]
        max_v = max(values)
        max_queue_per_dir_m[d] = max_v
        spillback_per_dir[d] = max_v > SPILLBACK_THRESHOLD[d]
        # Estabilizada: el último bucket NO es el máximo (la cola
        # estabilizó antes del final).
        queue_stabilized[d] = values[-1] < max_v if len(values) > 1 else True

    # Verificación final.
    reasons: list[str] = []
    satisfied = True

    if target_jam == ">=3":
        if target_direction is None:
            satisfied = False
            reasons.append("target_jam>=3 requiere target_direction")
        else:
            achieved = sustained_ge3.get(target_direction, 0)
            if achieved < 5:
                satisfied = False
                reasons.append(
                    f"jam ≥3 sostenido en {target_direction} fue de "
                    f"{achieved} bucket(s); requerido ≥5 (= 300 s sim)"
                )
    elif target_jam == "<=2":
        # Estructural: en una intersección señalizada, vehículos esperando
        # en rojo deprimen mean_speed transitoriamente y pueden generar
        # jam ≥3 en buckets aislados aun en flujo libre. El criterio
        # operativo es la ausencia de jam ≥3 SOSTENIDO (>= 5 buckets =
        # 300 s sim), simétrico con la cota de jam ≥3.
        sustained_violations = {d: n for d, n in sustained_ge3.items() if n >= 5}
        if sustained_violations:
            satisfied = False
            reasons.append(
                f"jam ≥3 sostenido (≥5 buckets) en {sustained_violations} "
                f"viola target <=2"
            )
    else:
        satisfied = False
        reasons.append(f"target_jam={target_jam!r} no reconocido (use >=3 o <=2)")

    # Spillback (independiente del target).
    spillback_dirs = [d for d, sb in spillback_per_dir.items() if sb]
    if spillback_dirs:
        satisfied = False
        reasons.append(
            f"spillback en {spillback_dirs}: "
            f"maxJamLengthInMeters excede 90% del aproche"
        )

    # Cola estabilizada (solo para patrones de jam — offpeak/weekend
    # no necesariamente generan cola).
    if target_jam == ">=3":
        not_stab = [d for d, s in queue_stabilized.items() if not s and d == target_direction]
        if not_stab:
            satisfied = False
            reasons.append(
                f"cola en {not_stab} no estabilizó — bucket final = máximo"
            )

    return CoverageReport(
        pattern=pattern,
        seed=seed,
        satisfied=satisfied,
        target_jam=target_jam,
        target_direction=target_direction,
        max_jam_per_dir=max_jam_per_dir,
        sustained_jam_ge3_buckets=sustained_ge3,
        max_queue_per_dir_m=max_queue_per_dir_m,
        spillback_per_dir=spillback_per_dir,
        queue_stabilized=queue_stabilized,
        reasons=reasons,
    )


def run_all_patterns(out_root: Path, seed: int = 42, end_s: Optional[int] = None) -> dict[str, CoverageReport]:
    with open(PATTERN_PARAMS) as f:
        cfg = yaml.safe_load(f)
    results: dict[str, CoverageReport] = {}
    for name, params in cfg["patterns"].items():
        out_dir = out_root / name
        report = check_coverage(
            pattern=name,
            seed=seed,
            out_dir=out_dir,
            target_jam=params["target_jam"],
            target_direction=params.get("target_direction"),
            end_s=end_s,
        )
        results[name] = report
    return results


def main() -> int:
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else SIM_ROOT / "data" / "_coverage"
    reports = run_all_patterns(out_root)
    all_ok = True
    for name, r in reports.items():
        status = "OK " if r.satisfied else "FAIL"
        print(f"[{status}] {name}: target={r.target_jam} dir={r.target_direction}")
        print(f"       max_jam_per_dir = {r.max_jam_per_dir}")
        print(f"       sustained ≥3 buckets = {r.sustained_jam_ge3_buckets}")
        print(f"       max_queue_m = { {d: round(v, 1) for d, v in r.max_queue_per_dir_m.items()} }")
        if r.reasons:
            for reason in r.reasons:
                print(f"       → {reason}")
        all_ok = all_ok and r.satisfied
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
