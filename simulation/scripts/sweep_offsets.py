"""Onda verde Benavides↔Schell — barrido de offsets + multi-semilla del óptimo.

Experimento: imponer un CICLO COMÚN FIJO (90 s, el del baseline) sobre el adaptativo
y coordinar el arranque de Schell respecto a Benavides (offset) para que el pelotón
S→N que sale de Benavides en verde llegue a Schell también en verde. La onda verde va
ENCIMA del control adaptativo: el motor sigue decidiendo el split por presión; acá solo
se le fija el ciclo y se desfasa el arranque. CERO core (motor por HTTP).

Espejo de sweep_seeds.py: reúsa _run_fixed / _run_adaptive / _edge_* / health-check /
generación determinista de rutas, y collect_corredor para KPIs apples-to-apples.

Dos fases:
  FASE 2 (barrido):  offset 0..80 paso 10 en seed 42. offset=0 = CONTROL (ciclo fijo
                     sin coordinación) → aísla el efecto puro del offset. Reporta demora
                     de red post-warmup + spillback/cola del link interno Benavides→Schell.
  FASE 3 (multi):    semillas 42..51 en offset óptimo + offset=0. RD% pareado vs FIJO,
                     vs MP per-node (ciclo variable) y vs offset=0 (efecto puro de coord).

Uso:
    invoke up                                         # (raíz, otra terminal) — motor :8001
    simulation/.venv/bin/python scripts/sweep_offsets.py sweep
    simulation/.venv/bin/python scripts/sweep_offsets.py multiseed --offset-opt 10
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# sweep_seeds setea SUMO_HOME + sys.path e importa collect_corredor/corredor_adaptive.
import sweep_seeds as ss  # noqa: E402
from cerebrovial_simulation.kpis import collect_corredor  # noqa: E402
from cerebrovial_simulation.traci_adapter import corredor_adaptive  # noqa: E402

CYCLE = 90.0                      # ciclo común fijo (= baseline del corredor)
DEFAULT_OFFSETS = list(range(0, 81, 10))
BEN_SCH = "benavides_to_schell(INT)"   # label del link interno (edge 279893875#1)


def _edge_spill(kpis, label: str) -> bool:
    for eq in kpis.edge_queues:
        if eq.label == label:
            return eq.spillback
    return False


def _run_adaptive_offset(seed: int, cycle: float, offset: float):
    """Corre el adaptativo con ciclo fijo + offset y recolecta KPIs."""
    out_dir = ss.DATA / f"peak_s_n_adaptive_seed{seed}_cyc{int(cycle)}_off{int(offset)}_end{ss.END}"
    stats = corredor_adaptive.run_adaptive(
        seed=seed, end_s=ss.END, out_dir=out_dir, cycle=cycle, offset=offset)
    errs = sum(len(s["errors"]) for s in stats.values())
    if errs:
        print(f"  ⚠️ seed {seed} off {offset}: {errs} errores de motor", file=sys.stderr)
    return collect_corredor.collect_corredor(out_dir, ss.WARMUP, ss.PARAMS), stats


def _check_engine() -> bool:
    try:
        r = requests.get(ss.HEALTH, timeout=5)
        assert r.status_code == 200
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: motor no responde 200 en {ss.HEALTH} ({exc}).\n"
              "       Levantalo con `invoke up` antes de correr.", file=sys.stderr)
        return False
    # demanda determinista (scale=1.0) una sola vez
    import subprocess
    subprocess.run([str(ss.SIM_ROOT / ".venv/bin/python"),
                    str(HERE / "generate_corredor_routes.py")], check=True, capture_output=True)
    return True


def sweep(seed: int, offsets: list[int]) -> int:
    """FASE 2 — barrido de offsets en una semilla. Encuentra el offset óptimo."""
    if not _check_engine():
        return 1
    rows = []
    for off in offsets:
        k, _ = _run_adaptive_offset(seed, CYCLE, off)
        rows.append({
            "off": off,
            "tl_pw": k.net_mean_timeloss_s_postwarm,
            "corr_pw": k.corridor_mean_travel_time_s_postwarm,
            "benSch_max": ss._edge_max(k, BEN_SCH),
            "benSch_mean": ss._edge_mean(k, BEN_SCH),
            "spill": _edge_spill(k, BEN_SCH),
            "thr": k.throughput_veh_per_h,
        })
        print(f"  off {off:>2}: timeLoss_pw={rows[-1]['tl_pw']:6.1f}  "
              f"corrTT_pw={rows[-1]['corr_pw']:6.1f}  benSch max={rows[-1]['benSch_max']:5.1f}"
              f"{'  SPILL' if rows[-1]['spill'] else ''}", file=sys.stderr)

    W = 86
    print("\n" + "=" * W)
    print(f"FASE 2 — BARRIDO DE OFFSETS | onda verde Benavides↔Schell | seed {seed} | "
          f"ciclo fijo {CYCLE:.0f}s | end={ss.END} warmup={ss.WARMUP}")
    print("=" * W)
    print(f"{'offset':>6} {'timeLoss_pw':>12} {'corrTT_pw':>10} {'benSch_max':>11} "
          f"{'benSch_mean':>12} {'spill':>6} {'thr':>7}")
    base = next(r for r in rows if r["off"] == 0)
    for r in rows:
        dpct = (base["tl_pw"] - r["tl_pw"]) / base["tl_pw"] * 100.0 if base["tl_pw"] else 0.0
        tag = "  <- CONTROL (off=0)" if r["off"] == 0 else f"  Δ vs off0 = {dpct:+.1f}%"
        print(f"{r['off']:>6} {r['tl_pw']:>12.1f} {r['corr_pw']:>10.1f} {r['benSch_max']:>11.1f} "
              f"{r['benSch_mean']:>12.1f} {('SÍ' if r['spill'] else 'no'):>6} {r['thr']:>7.0f}{tag}")
    print("-" * W)
    best = min(rows, key=lambda r: r["tl_pw"])
    base_corr = base["corr_pw"]
    best_corr = min(rows, key=lambda r: r["corr_pw"])
    gain = (base["tl_pw"] - best["tl_pw"]) / base["tl_pw"] * 100.0 if base["tl_pw"] else 0.0
    print(f"ÓPTIMO (min timeLoss_pw): offset={best['off']}s  "
          f"timeLoss_pw={best['tl_pw']:.1f}  (vs off0 {base['tl_pw']:.1f} → {gain:+.1f}%)")
    print(f"ÓPTIMO (min corrTT_pw):   offset={best_corr['off']}s  "
          f"corrTT_pw={best_corr['corr_pw']:.1f}  (vs off0 {base_corr:.1f})")
    print("SANITY-CHECK: tiempo de viaje Benavides→Schell ≈ 144.92 m / 13.89 m/s ≈ 10.4 s "
          "(cota inf.) → óptimo esperado ~10–20 s")
    print("=" * W)
    return 0


def multiseed(seeds: list[int], off_opt: int) -> int:
    """FASE 3 — multi-semilla en offset óptimo + offset=0. RD% pareado vs 3 refs."""
    if not _check_engine():
        return 1
    rows = []
    for seed in seeds:
        f = ss._run_fixed(seed)                              # baseline fijo (net.xml estático)
        mp, _ = ss._run_adaptive(seed)                       # MP per-node (ciclo variable)
        coord, _ = _run_adaptive_offset(seed, CYCLE, off_opt)  # onda verde (óptimo)
        ctrl, _ = _run_adaptive_offset(seed, CYCLE, 0)         # ciclo fijo sin coordinación

        def rd(ref):
            r = ref.net_mean_timeloss_s_postwarm
            return (r - coord.net_mean_timeloss_s_postwarm) / r * 100.0 if r else 0.0

        rows.append({
            "seed": seed,
            "rd_fijo": rd(f), "rd_mp": rd(mp), "rd_ctrl": rd(ctrl),
            "tl_fijo": f.net_mean_timeloss_s_postwarm,
            "tl_mp": mp.net_mean_timeloss_s_postwarm,
            "tl_ctrl": ctrl.net_mean_timeloss_s_postwarm,
            "tl_coord": coord.net_mean_timeloss_s_postwarm,
            "benSch_ctrl": ss._edge_max(ctrl, BEN_SCH),
            "benSch_coord": ss._edge_max(coord, BEN_SCH),
            "spill_ctrl": _edge_spill(ctrl, BEN_SCH),
            "spill_coord": _edge_spill(coord, BEN_SCH),
            "thr_coord": coord.throughput_veh_per_h,
        })
        print(f"  seed {seed}: RD% vs fijo={rows[-1]['rd_fijo']:+.1f} "
              f"vs MP={rows[-1]['rd_mp']:+.1f} vs off0={rows[-1]['rd_ctrl']:+.1f}  "
              f"benSch off0→opt {rows[-1]['benSch_ctrl']:.0f}→{rows[-1]['benSch_coord']:.0f}",
              file=sys.stderr)

    W = 100
    print("\n" + "=" * W)
    print(f"FASE 3 — MULTI-SEMILLA | onda verde offset={off_opt}s vs fijo / MP per-node / "
          f"off=0 | ciclo {CYCLE:.0f}s | n={len(rows)} | end={ss.END} warmup={ss.WARMUP}")
    print("=" * W)
    print(f"{'seed':>5} | {'RD%fijo':>8} {'RD%MP':>8} {'RD%off0':>8} | "
          f"{'tl_fijo':>8} {'tl_MP':>8} {'tl_off0':>8} {'tl_coord':>9} | "
          f"{'benSch0':>8} {'benSchO':>8} {'thr':>7}")
    for r in rows:
        print(f"{r['seed']:>5} | {r['rd_fijo']:>+8.1f} {r['rd_mp']:>+8.1f} {r['rd_ctrl']:>+8.1f} | "
              f"{r['tl_fijo']:>8.1f} {r['tl_mp']:>8.1f} {r['tl_ctrl']:>8.1f} {r['tl_coord']:>9.1f} | "
              f"{r['benSch_ctrl']:>8.0f} {r['benSch_coord']:>8.0f} {r['thr_coord']:>7.0f}")
    print("-" * W)

    def agg(key, label):
        xs = [r[key] for r in rows]
        mean = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        se = sd / (len(xs) ** 0.5) if xs else 0.0
        lo, hi = mean - sd, mean + sd
        crosses = lo <= 0 <= hi
        pos = sum(1 for x in xs if x > 0)
        print(f"  RD% {label:<10} = {mean:+.2f}% ± SD {sd:.2f} (SE {se:.2f})  "
              f"[{lo:+.2f}…{hi:+.2f}]  ¿cruza 0? {'SÍ→EMPATE' if crosses else 'NO→señal'}  "
              f"| >0: {pos}/{len(xs)}")

    agg("rd_fijo", "vs FIJO")
    agg("rd_mp", "vs MP node")
    agg("rd_ctrl", "vs off=0")   # efecto PURO de la coordinación
    spill_ctrl = sum(1 for r in rows if r["spill_ctrl"])
    spill_coord = sum(1 for r in rows if r["spill_coord"])
    print("TERMÓMETRO spillback link interno Benavides→Schell (¿el pelotón fluye?):")
    print(f"  semillas con spillback sostenido: off=0 {spill_ctrl}/{len(rows)}  →  "
          f"óptimo {spill_coord}/{len(rows)}")
    print("=" * W)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    p_sw = sub.add_parser("sweep", help="FASE 2: barrido de offsets en una semilla")
    p_sw.add_argument("--seed", type=int, default=42)
    p_sw.add_argument("--offsets", type=int, nargs="+", default=DEFAULT_OFFSETS)
    p_ms = sub.add_parser("multiseed", help="FASE 3: multi-semilla en offset óptimo")
    p_ms.add_argument("--offset-opt", type=int, required=True)
    p_ms.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    args = ap.parse_args()

    if args.cmd == "multiseed":
        return multiseed(args.seeds, args.offset_opt)
    # default = sweep (FASE 2)
    seed = getattr(args, "seed", 42)
    offsets = getattr(args, "offsets", DEFAULT_OFFSETS)
    return sweep(seed, offsets)


if __name__ == "__main__":
    sys.exit(main())
