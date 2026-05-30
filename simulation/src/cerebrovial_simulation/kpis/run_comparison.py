"""Orquesta corridas adaptive vs fixed-Webster × 4 patrones × N seeds.

Modos:
- ``fixed_webster``: tllogic estático precomputado offline (Catch B).
  Usa el add-file ``conf/network/webster_fixed.add.xml`` (reemplaza
  el baseline en el sumocfg via ``--additional-files``). NO requiere motor.
- ``adaptive``: corrida e2e con `run_e2e.run_e2e` consumiendo el motor
  por HTTP. Requiere motor corriendo (`invoke up`).

Output: ``data/kpis/comparison.csv`` con columnas KPIs.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

from ..fixed_control import webster_fixed
from ..traci_adapter.run_e2e import run_e2e
from . import collect as kpis_collect


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent.parent
SCENARIOS_DIR = SIM_ROOT / "conf" / "scenarios"
NETWORK_DIR = SIM_ROOT / "conf" / "network"
WEBSTER_ADD = NETWORK_DIR / "webster_fixed.add.xml"
DATA_KPIS = SIM_ROOT / "data" / "kpis"
COMPARISON_CSV = DATA_KPIS / "comparison.csv"

PATTERNS = ["am_peak", "pm_peak", "offpeak", "weekend"]
DEFAULT_END_S = 600
DEFAULT_SEEDS = [1, 2, 3]


def _run_fixed_webster(pattern: str, seed: int, end_s: int, out_dir: Path) -> kpis_collect.KPIs:
    """Corrida headless con el programa fixed Webster precomputado."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sumo_bin = Path(os.environ["SUMO_HOME"]) / "bin" / "sumo"
    cfg = SCENARIOS_DIR / f"{pattern}.sumocfg"

    # Add-files locales (file= relativo resuelve al directorio del add).
    add_files = []
    for name in ["webster_fixed.add.xml", "edgedata.add.xml", "lanearea.add.xml"]:
        src = NETWORK_DIR / name
        dst = out_dir / name
        shutil.copy(src, dst)
        add_files.append(dst)

    args = [
        str(sumo_bin),
        "-c", str(cfg),
        # Sobreescribe sumocfg additional-files: USAR webster_fixed en lugar de baseline.
        "--additional-files", ",".join(str(p) for p in add_files),
        "--seed", str(seed),
        "--end", str(end_s),
        "--summary-output", str(out_dir / "summary.parquet"),
        "--tripinfo-output", str(out_dir / "tripinfo.parquet"),
        "--no-step-log", "--no-warnings",
    ]
    proc = subprocess.run(args, cwd=out_dir, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sumo (fixed_webster) failed {pattern} seed={seed}:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return kpis_collect.collect(
        mode="fixed_webster",
        pattern=pattern,
        seed=seed,
        out_dir=out_dir,
    )


def _run_adaptive(pattern: str, seed: int, end_s: int, out_dir: Path) -> kpis_collect.KPIs:
    """Corrida e2e con engine HTTP (motor real)."""
    stats = run_e2e(pattern=pattern, seed=seed, end_s=end_s, out_dir=out_dir)
    if stats["engine_errors"]:
        raise RuntimeError(
            f"adaptive {pattern} seed={seed} tuvo errores: {stats['engine_errors']}"
        )
    return kpis_collect.collect(
        mode="adaptive",
        pattern=pattern,
        seed=seed,
        out_dir=out_dir,
    )


def _engine_health(timeout_s: float = 5.0) -> bool:
    url = os.environ.get("ENGINE_URL", "http://localhost:8001/control/recommend")
    health = url.replace("/recommend", "/health")
    try:
        resp = requests.get(health, timeout=timeout_s)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1 seed per (mode, pattern)")
    parser.add_argument("--end", type=int, default=DEFAULT_END_S)
    parser.add_argument("--skip-adaptive", action="store_true",
                        help="Solo correr fixed_webster (útil si el motor no está disponible).")
    parser.add_argument("--out", default=str(DATA_KPIS / "_runs"),
                        help="Scratch dir para outputs sumo intermedios.")
    args = parser.parse_args()

    seeds = [DEFAULT_SEEDS[0]] if args.quick else DEFAULT_SEEDS

    # Precomputar Webster fijo si add-file no existe o pattern_params cambió.
    if not WEBSTER_ADD.exists():
        print("Precomputando Webster fijo offline (am_peak flows)...")
        plan = webster_fixed.compute_from_pattern("am_peak")
        webster_fixed.write_addfile(plan)

    # Health check engine si vamos a correr adaptive.
    run_adaptive = not args.skip_adaptive
    if run_adaptive and not _engine_health():
        print(
            "⚠️  Motor no responde en /control/health. "
            "Skipping adaptive runs. Usá --skip-adaptive para suprimir este aviso.",
            file=sys.stderr,
        )
        run_adaptive = False

    scratch_root = Path(args.out)
    scratch_root.mkdir(parents=True, exist_ok=True)
    DATA_KPIS.mkdir(parents=True, exist_ok=True)

    all_kpis: list[kpis_collect.KPIs] = []
    modes_to_run = ["fixed_webster"]
    if run_adaptive:
        modes_to_run.append("adaptive")

    for mode in modes_to_run:
        for pattern in PATTERNS:
            for seed in seeds:
                out_dir = scratch_root / f"{mode}_{pattern}_seed{seed}"
                print(f"[{mode}] {pattern} seed={seed} → {out_dir}")
                t0 = time.time()
                if mode == "fixed_webster":
                    kpi = _run_fixed_webster(pattern, seed, args.end, out_dir)
                else:
                    kpi = _run_adaptive(pattern, seed, args.end, out_dir)
                dt = time.time() - t0
                print(f"  {dt:.1f}s | travel={kpi.mean_travel_time_s:.1f}s "
                      f"delay_total={kpi.total_delay_s:.0f}s "
                      f"throughput={kpi.throughput_veh_per_h:.0f}v/h "
                      f"max_q_NS={max(kpi.max_queue_m_by_dir['N'], kpi.max_queue_m_by_dir['S']):.0f}m")
                all_kpis.append(kpi)

    # Persiste CSV.
    rows = [k.to_dict() for k in all_kpis]
    if not rows:
        print("Sin filas — nada que persistir.")
        return 0
    fieldnames = list(rows[0].keys())
    with open(COMPARISON_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nKPIs persistidos en {COMPARISON_CSV}")
    print(f"  {len(rows)} corridas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
