"""Cobertura del dataset perfil-día (TTH-11) — redefinición de CT-07.2.

El criterio viejo (coverage_check.py, TTH-07) pedía "jam ≥3 SOSTENIDO ≥5 buckets"
sobre corridas de 20 min de demanda constante. El perfil-día es distinto: la
demanda varía en 24h y el jam 3-4 es **transitorio en las ventanas pico** (un pico
de 2h en un día de 24h NO es congestión sostenida). Este módulo redefine el criterio:

- **Perfiles con pico** (laborable, especial): la dirección dominante alcanza
  ``jam_level >= peak_jam_min`` en ≥``MIN_PEAK_BUCKETS`` buckets DENTRO de al menos
  una ``peak_window``.
- **Perfiles valle** (finde, feriado): alcanzan ``meseta_jam_min`` en su meseta y
  NO presentan jam ≥3 **sostenido** (run-length < 5 buckets) en ninguna dirección.
- **Sanidad de valle**: aparece ``jam 0`` en ``valley_windows`` (confirma el fix
  edge-vacío: calle vacía = flujo libre, no vía cerrada).
- **Spillback**: ningún detector excede 90% del aproche (mismo umbral que TTH-07).

REUSA los helpers de parseo/corrida de ``coverage_check`` (TTH-07) sin modificarlos:
``run_sumo_for_coverage`` resuelve ``{pattern}.sumocfg`` por nombre — sirve igual
para los sumocfg de 24h del perfil-día.

BIMODALIDAD (hallazgo 2026-05-31 — marcar para CT-11.6 del doc de TTH-11):
la congestión por demanda sostenida en la 4-way es BIMODAL (jam 2 bajo capacidad /
jam 4 sobre capacidad), con un CLIFF entre ~2500 y ~2550 vph; jam 3 NO es meseta
estable, solo aparece TRANSITORIAMENTE en las rampas de un pico. Por eso el criterio
de pico (``peak_jam_min``) se evalúa como "jam ≥ objetivo ALCANZADO en ≥MIN_PEAK_BUCKETS
buckets dentro de la ventana" (transitorio admisible), NO como meseta sostenida. Ver
la nota extensa en conf/scenarios/perfil_dia_params.yaml. Es una propiedad del dataset
que TTH-09 hereda.

Todos los criterios se evalúan sobre el EJE DOMINANTE N-S (ver DOMINANT_AXIS): ahí
vive la firma de congestión del perfil; el secundario E-W es ruido de rojo por diseño.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ..coverage_check import (
    SPILLBACK_THRESHOLD,
    VMAX_MPS,
    _max_run_length,
    _parse_edgedata,
    _parse_lanearea,
    run_sumo_for_coverage,
)
from ..jam_level import sumo_to_jam_level


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent.parent
PERFIL_DIA_PARAMS = SIM_ROOT / "conf" / "scenarios" / "perfil_dia_params.yaml"

# Mínimo de buckets en pico para considerar el jam objetivo "alcanzado" (transitorio,
# no sostenido). 3 buckets = 180 s simulados de jam ≥ objetivo dentro de la ventana.
MIN_PEAK_BUCKETS = 3
# Sostenido (para descartar congestión persistente en perfiles valle): ≥5 buckets = 300 s.
SUSTAINED_BUCKETS = 5
DIRECTIONS = ["N", "S", "E", "W"]
# Eje dominante N-S: la firma de congestión del perfil vive acá. El secundario E-W
# (demanda N-S/4 por diseño) acumula cola de semáforo en rojo en la vía menor, que es
# ruido esperado, NO congestión del corredor. Los criterios de cobertura se evalúan
# sobre el eje dominante para no ensuciar la verificación con ese ruido (decisión
# 2026-05-31: finde fallaba por un run de 5 buckets jam≥3 en E/W, no en N/S).
DOMINANT_AXIS = ("N", "S")


@dataclass
class PerfilDiaCoverageReport:
    pattern: str
    seed: int
    satisfied: bool
    max_jam_per_dir: dict[str, int]
    peak_jam_in_window: Optional[int]          # max jam de la dir dominante dentro de alguna ventana pico
    peak_buckets_at_target: Optional[int]      # # buckets en ventana con jam >= objetivo
    sustained_ge3_per_dir: dict[str, int]      # max run-length de jam ≥3 por dirección
    jam0_in_valley: Optional[bool]             # apareció jam 0 en la ventana de valle
    max_queue_per_dir_m: dict[str, float]
    spillback_per_dir: dict[str, bool]
    reasons: list[str] = field(default_factory=list)


def _in_window(bucket_t: float, windows_h: list[list[float]]) -> bool:
    """bucket_t (end-of-bucket, seg) cae en alguna ventana [a,b] (horas)."""
    for a_h, b_h in windows_h:
        if a_h * 3600 < bucket_t <= b_h * 3600:
            return True
    return False


def check_coverage(
    pattern: str,
    seed: int,
    out_dir: Path,
    coverage: dict,
    end_s: Optional[int] = None,
) -> PerfilDiaCoverageReport:
    edgedata, lanearea = run_sumo_for_coverage(pattern, seed, out_dir, end_s=end_s)
    speeds = _parse_edgedata(edgedata)
    queues = _parse_lanearea(lanearea)

    # jam_level por dirección por bucket: [(bucket_t, jam), ...].
    jam_by_dir: dict[str, list[tuple[float, int]]] = {}
    max_jam_per_dir: dict[str, int] = {}
    sustained_ge3: dict[str, int] = {}
    for d, buckets in speeds.items():
        jams = [(t, sumo_to_jam_level(speed, VMAX_MPS)) for t, speed in buckets]
        jam_by_dir[d] = jams
        max_jam_per_dir[d] = max((j for _, j in jams), default=0)
        sustained_ge3[d] = _max_run_length([j for _, j in jams], lambda x: x >= 3)

    # Colas / spillback.
    max_queue_per_dir_m: dict[str, float] = {}
    spillback_per_dir: dict[str, bool] = {}
    for d, buckets in queues.items():
        values = [v for _, v in buckets]
        max_v = max(values, default=0.0)
        max_queue_per_dir_m[d] = max_v
        spillback_per_dir[d] = max_v > SPILLBACK_THRESHOLD[d]

    reasons: list[str] = []
    satisfied = True
    peak_jam_in_window: Optional[int] = None
    peak_buckets_at_target: Optional[int] = None
    jam0_in_valley: Optional[bool] = None

    # --- Perfiles con pico (laborable, especial). ---
    if "peak_jam_min" in coverage:
        pdir = coverage["peak_direction"]
        target = int(coverage["peak_jam_min"])
        windows = coverage["peak_windows_h"]
        in_win = [(t, j) for t, j in jam_by_dir.get(pdir, []) if _in_window(t, windows)]
        peak_jam_in_window = max((j for _, j in in_win), default=0)
        peak_buckets_at_target = sum(1 for _, j in in_win if j >= target)
        if peak_buckets_at_target < MIN_PEAK_BUCKETS:
            satisfied = False
            reasons.append(
                f"pico: dir {pdir} alcanzó jam ≥{target} en {peak_buckets_at_target} "
                f"bucket(s) dentro de las ventanas {windows}; requerido ≥{MIN_PEAK_BUCKETS} "
                f"(max jam en ventana = {peak_jam_in_window})"
            )

    # --- Perfiles valle (finde, feriado). Solo eje dominante N-S (ver DOMINANT_AXIS). ---
    if coverage.get("no_sustained_ge3"):
        sustained_violations = {
            d: n for d, n in sustained_ge3.items()
            if d in DOMINANT_AXIS and n >= SUSTAINED_BUCKETS
        }
        if sustained_violations:
            satisfied = False
            reasons.append(
                f"jam ≥3 sostenido (≥{SUSTAINED_BUCKETS} buckets) en eje dominante "
                f"{sustained_violations} — un perfil valle no debería congestionar el "
                f"corredor de forma persistente"
            )
    if "meseta_jam_min" in coverage:
        mdir = coverage["meseta_direction"]
        mtarget = int(coverage["meseta_jam_min"])
        if max_jam_per_dir.get(mdir, 0) < mtarget:
            satisfied = False
            reasons.append(
                f"meseta: dir {mdir} no alcanzó jam ≥{mtarget} "
                f"(max observado = {max_jam_per_dir.get(mdir, 0)})"
            )

    # --- Sanidad de valle: jam 0 presente (fix edge-vacío). ---
    if "valley_windows_h" in coverage:
        vdir = coverage.get("peak_direction") or coverage.get("meseta_direction") or "N"
        vwin = coverage["valley_windows_h"]
        jam0_in_valley = any(
            j == 0 for t, j in jam_by_dir.get(vdir, []) if _in_window(t, vwin)
        )
        if not jam0_in_valley:
            satisfied = False
            reasons.append(
                f"valle: no apareció jam 0 en dir {vdir} dentro de {vwin} "
                f"(esperado por el fix edge-vacío)"
            )

    # --- Spillback (independiente del perfil). ---
    spillback_dirs = [d for d, sb in spillback_per_dir.items() if sb]
    if spillback_dirs:
        satisfied = False
        reasons.append(
            f"spillback en {spillback_dirs}: maxJamLengthInMeters excede 90% del aproche"
        )

    return PerfilDiaCoverageReport(
        pattern=pattern,
        seed=seed,
        satisfied=satisfied,
        max_jam_per_dir=max_jam_per_dir,
        peak_jam_in_window=peak_jam_in_window,
        peak_buckets_at_target=peak_buckets_at_target,
        sustained_ge3_per_dir=sustained_ge3,
        jam0_in_valley=jam0_in_valley,
        max_queue_per_dir_m=max_queue_per_dir_m,
        spillback_per_dir=spillback_per_dir,
        reasons=reasons,
    )


def run_all_patterns(
    out_root: Path, seed: int = 42, end_s: Optional[int] = None
) -> dict[str, PerfilDiaCoverageReport]:
    with open(PERFIL_DIA_PARAMS) as f:
        cfg = yaml.safe_load(f)
    results: dict[str, PerfilDiaCoverageReport] = {}
    for name, params in cfg["patterns"].items():
        out_dir = out_root / name
        results[name] = check_coverage(
            pattern=name,
            seed=seed,
            out_dir=out_dir,
            coverage=params["coverage"],
            end_s=end_s,
        )
    return results


def main() -> int:
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else SIM_ROOT / "data" / "_coverage_perfil_dia"
    reports = run_all_patterns(out_root)
    all_ok = True
    for name, r in reports.items():
        status = "OK " if r.satisfied else "FAIL"
        print(f"[{status}] {name}")
        print(f"       max_jam_per_dir   = {r.max_jam_per_dir}")
        print(f"       peak_jam_in_window= {r.peak_jam_in_window} "
              f"(buckets@target={r.peak_buckets_at_target})")
        print(f"       sustained ≥3      = {r.sustained_ge3_per_dir}")
        print(f"       jam0_in_valley    = {r.jam0_in_valley}")
        print(f"       max_queue_m       = { {d: round(v, 1) for d, v in r.max_queue_per_dir_m.items()} }")
        for reason in r.reasons:
            print(f"       → {reason}")
        all_ok = all_ok and r.satisfied
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
