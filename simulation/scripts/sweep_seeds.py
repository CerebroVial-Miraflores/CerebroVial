"""Barrido multi-semilla (EXPLORACIÓN) — robustez del RD% adaptativo vs fijo.

El RD% de una sola semilla no se reporta. Corre N semillas, comparación PAREADA
(fijo Y adaptativo con el MISMO seed → única diferencia = el control), y agrega
RD% medio ± SD. Confirma si el mecanismo (drenaje del cuello de entrada de Benavides
+ relocalización del tapón al link interno Benavides→Schell) se sostiene entre semillas.

Todo idéntico salvo la semilla: scale=1.0, end=1800, warmup=600, mismos detectores,
mismo adaptador (F1, no se toca). Reúsa collect_corredor para KPIs apples-to-apples.

CERO core (motor por HTTP). Requiere motor vivo en :8001 (fail-rápido).

Uso:
    invoke up        # (raíz del repo, otra terminal) — levanta db + core
    simulation/.venv/bin/python scripts/sweep_seeds.py 42 43 44 45 46 47 48 49 50 51
"""
from __future__ import annotations

import os
import statistics
import subprocess
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
CONF = SIM_ROOT / "conf" / "corredor_larco"
CFG = CONF / "peak_s_n.sumocfg"
DET = CONF / "corredor_larco.det.add.xml"
PARAMS = CONF / "demand_params.yaml"
DATA = SIM_ROOT / "data" / "corredor_larco"
SUMO_PKG = SIM_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "sumo"
HEALTH = os.environ.get("ENGINE_HEALTH", "http://localhost:8001/control/health")

END, WARMUP = 1800, 600

# SUMO_HOME debe estar en el entorno ANTES de importar corredor_adaptive (traci.start).
os.environ.setdefault("SUMO_HOME", str(SUMO_PKG))
os.environ["PATH"] = f"{SUMO_PKG}/bin:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(SIM_ROOT / "src"))

from cerebrovial_simulation.kpis import collect_corredor  # noqa: E402
from cerebrovial_simulation.traci_adapter import corredor_adaptive  # noqa: E402


def _edge_mean(kpis, label: str) -> float:
    for eq in kpis.edge_queues:
        if eq.label == label:
            return eq.mean_queue_m
    return 0.0


def _edge_max(kpis, label: str) -> float:
    for eq in kpis.edge_queues:
        if eq.label == label:
            return eq.max_queue_m
    return 0.0


def _run_fixed(seed: int):
    out_dir = DATA / f"peak_s_n_seed{seed}_end{END}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / DET.name).write_bytes(DET.read_bytes())
    env = dict(os.environ, SUMO_HOME=str(SUMO_PKG))
    args = [
        str(SUMO_PKG / "bin" / "sumo"), "-c", str(CFG),
        "--additional-files", str(out_dir / DET.name),
        "--seed", str(seed), "--end", str(END),
        "--summary-output", str(out_dir / "summary.parquet"),
        "--tripinfo-output", str(out_dir / "tripinfo.parquet"),
        "--no-step-log", "--no-warnings",
    ]
    p = subprocess.run(args, cwd=out_dir, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"sumo fijo seed={seed} falló:\n{p.stderr[-800:]}")
    return collect_corredor.collect_corredor(out_dir, WARMUP, PARAMS)


def _run_adaptive(seed: int):
    out_dir = DATA / f"peak_s_n_adaptive_seed{seed}_end{END}"
    stats = corredor_adaptive.run_adaptive(seed=seed, end_s=END, out_dir=out_dir)
    errs = sum(len(s["errors"]) for s in stats.values())
    if errs:
        print(f"  ⚠️ seed {seed}: {errs} errores de motor en el adaptativo", file=sys.stderr)
    return collect_corredor.collect_corredor(out_dir, WARMUP, PARAMS), stats


def main(seeds: list[int]) -> int:
    # Fail-rápido: motor vivo.
    try:
        r = requests.get(HEALTH, timeout=5)
        assert r.status_code == 200
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: motor no responde 200 en {HEALTH} ({exc}).\n"
              "       Levantalo con `invoke up` antes de correr el barrido.", file=sys.stderr)
        return 1

    # Demanda determinista (scale=1.0) una sola vez.
    subprocess.run([str(SIM_ROOT / ".venv/bin/python"),
                    str(HERE / "generate_corredor_routes.py")], check=True, capture_output=True)

    rows = []
    for seed in seeds:
        f = _run_fixed(seed)
        a, _ = _run_adaptive(seed)
        tl_f = f.net_mean_timeloss_s_postwarm
        tl_a = a.net_mean_timeloss_s_postwarm
        rd = (tl_f - tl_a) / tl_f * 100.0 if tl_f else 0.0
        rows.append({
            "seed": seed, "rd": rd,
            "tl_f": tl_f, "tl_a": tl_a,
            "larcoS_f": _edge_mean(f, "larcoS_approach"),
            "larcoS_a": _edge_mean(a, "larcoS_approach"),
            "benSch_f": _edge_max(f, "benavides_to_schell(INT)"),
            "benSch_a": _edge_max(a, "benavides_to_schell(INT)"),
            "thr_f": f.throughput_veh_per_h, "thr_a": a.throughput_veh_per_h,
        })
        print(f"  seed {seed}: RD%={rd:+.1f}  larcoS {rows[-1]['larcoS_f']:.0f}→{rows[-1]['larcoS_a']:.0f}  "
              f"benSch {rows[-1]['benSch_f']:.0f}→{rows[-1]['benSch_a']:.0f}", file=sys.stderr)

    W = 92
    print("\n" + "=" * W)
    print(f"BARRIDO MULTI-SEMILLA — MP adaptativo per-node vs fijo | corredor Larco | n={len(rows)} "
          f"| end={END} warmup={WARMUP}")
    print("=" * W)
    print(f"{'seed':>5} {'RD%':>7} | {'timeLoss_fij':>12} {'timeLoss_adp':>12} | "
          f"{'larcoS_fij':>10} {'larcoS_adp':>10} | {'benSch_fij':>10} {'benSch_adp':>10} | "
          f"{'thr_fij':>8} {'thr_adp':>8}")
    for r in rows:
        print(f"{r['seed']:>5} {r['rd']:>+7.1f} | {r['tl_f']:>12.1f} {r['tl_a']:>12.1f} | "
              f"{r['larcoS_f']:>10.1f} {r['larcoS_a']:>10.1f} | {r['benSch_f']:>10.1f} {r['benSch_a']:>10.1f} | "
              f"{r['thr_f']:>8.0f} {r['thr_a']:>8.0f}")

    rds = [r["rd"] for r in rows]
    mean = statistics.mean(rds)
    sd = statistics.stdev(rds) if len(rds) > 1 else 0.0
    se = sd / (len(rds) ** 0.5) if rds else 0.0
    lo, hi = mean - sd, mean + sd
    pos = sum(1 for x in rds if x > 0)
    # mecanismo por semilla
    drain = sum(1 for r in rows if r["larcoS_a"] < r["larcoS_f"])
    reloc = sum(1 for r in rows if r["benSch_a"] > r["benSch_f"])
    both = sum(1 for r in rows if r["larcoS_a"] < r["larcoS_f"] and r["benSch_a"] > r["benSch_f"])

    print("-" * W)
    print(f"RD% medio = {mean:+.2f}%  ± SD {sd:.2f}  (SE {se:.2f})   "
          f"[media±SD = {lo:+.2f}% … {hi:+.2f}%]")
    crosses = lo <= 0 <= hi
    print(f"  ¿media±SD cruza el cero? {'SÍ → EMPATE ESTADÍSTICO' if crosses else 'NO → señal'}  "
          f"| semillas con RD%>0: {pos}/{len(rds)}")
    print("MECANISMO (¿se sostiene entre semillas?):")
    print(f"  drenaje cuello Benavides (larcoS_adp < larcoS_fij): {drain}/{len(rows)} semillas")
    print(f"  relocalización a Benavides→Schell (benSch_adp > benSch_fij): {reloc}/{len(rows)} semillas")
    print(f"  AMBOS a la vez: {both}/{len(rows)} semillas")
    print("=" * W)
    return 0


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or list(range(42, 52))
    sys.exit(main(seeds))
