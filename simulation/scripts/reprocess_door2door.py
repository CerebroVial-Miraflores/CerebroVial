"""Re-procesa KPIs con métrica PUERTA-A-PUERTA robusta a censura sobre outputs YA
corridos (NO re-corre SUMO). Dos conjuntos:

  IE05    — MP per-node vs fijo, 10 semillas pareadas (42–51). RD% nuevo (RED robusto
            vía Little's law + CORREDOR S→N) media±SD, LADO A LADO con el RD% viejo
            (timeLoss dentro de red), para ver que la diferencia es CÓMO se mide.
  OFFSETS — barrido onda verde (seed 42, off 0–80). Puerta-a-puerta por offset →
            ¿sigue offset=0 óptimo con la métrica correcta?

Métrica (ver collect_corredor): RED = W robusto (cuenta a TODOS, circulan + esperan
entrar; desglose espera vs adentro). CORREDOR = puerta-a-puerta de completados +
ratio de completación (pairing: más autos pasan Y tardan menos = gana). Siempre se
reporta throughput + no-completados (censura).

Uso:
    simulation/.venv/bin/python scripts/reprocess_door2door.py            # ambos
    simulation/.venv/bin/python scripts/reprocess_door2door.py ie05
    simulation/.venv/bin/python scripts/reprocess_door2door.py offsets
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
sys.path.insert(0, str(SIM_ROOT / "src"))

from cerebrovial_simulation.kpis import collect_corredor  # noqa: E402

DATA = SIM_ROOT / "data" / "corredor_larco"
PARAMS = SIM_ROOT / "conf" / "corredor_larco" / "demand_params.yaml"
END, WARMUP = 1800, 600


def _k(out_dir: Path):
    if not out_dir.exists():
        return None
    return collect_corredor.collect_corredor(out_dir, WARMUP, PARAMS)


def _agg(xs: list[float]):
    mean = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    se = sd / (len(xs) ** 0.5) if xs else 0.0
    lo, hi = mean - sd, mean + sd
    return mean, sd, se, lo, hi


def _rd(ref: float, new: float) -> float:
    return (ref - new) / ref * 100.0 if ref else 0.0


def ie05(seeds: list[int]) -> int:
    rows = []
    for s in seeds:
        f = _k(DATA / f"peak_s_n_seed{s}_end{END}")
        a = _k(DATA / f"peak_s_n_adaptive_seed{s}_end{END}")
        if f is None or a is None:
            print(f"  ⚠️ faltan outputs para seed {s} (fijo o adaptativo) — salteado", file=sys.stderr)
            continue
        rows.append({
            "seed": s,
            # RD% viejo (timeLoss dentro de red, completados)
            "rd_old": _rd(f.net_mean_timeloss_s_postwarm, a.net_mean_timeloss_s_postwarm),
            # RD% nuevo RED robusto (Little's law)
            "rd_red": _rd(f.w_red_door2door_s_postwarm, a.w_red_door2door_s_postwarm),
            # RD% nuevo CORREDOR S→N (completados)
            "rd_corr": _rd(f.d2d_corr_s_postwarm, a.d2d_corr_s_postwarm),
            # valores crudos para desglose
            "wf": f.w_red_door2door_s_postwarm, "wa": a.w_red_door2door_s_postwarm,
            "wf_wait": f.w_red_wait_s_postwarm, "wa_wait": a.w_red_wait_s_postwarm,
            "wf_in": f.w_red_inside_s_postwarm, "wa_in": a.w_red_inside_s_postwarm,
            "tlf": f.net_mean_timeloss_s_postwarm, "tla": a.net_mean_timeloss_s_postwarm,
            "ni_f": f.never_inserted_veh, "ni_a": a.never_inserted_veh,
            "thr_f": f.throughput_veh_per_h, "thr_a": a.throughput_veh_per_h,
            "cc_f": f.corr_completion_ratio, "cc_a": a.corr_completion_ratio,
            "corrf": f.d2d_corr_s_postwarm, "corra": a.d2d_corr_s_postwarm,
        })

    W = 100
    print("\n" + "=" * W)
    print(f"IE05 — MP per-node vs FIJO | PUERTA-A-PUERTA robusta | n={len(rows)} "
          f"semillas pareadas | end={END} warmup={WARMUP}")
    print("=" * W)
    print("RD% > 0 = adaptativo MEJOR. 'viejo'=timeLoss dentro de red (completados); "
          "'RED'=W robusto (Little's law, TODOS); 'CORR'=S→N completados.")
    print("-" * W)
    print(f"{'seed':>5} | {'RD%viejo':>9} {'RD%RED':>8} {'RD%CORR':>8} | "
          f"{'W_fijo':>7} {'W_adap':>7} | {'nunca_ins f→a':>14} | {'thr f→a':>13}")
    for r in rows:
        print(f"{r['seed']:>5} | {r['rd_old']:>+9.1f} {r['rd_red']:>+8.1f} {r['rd_corr']:>+8.1f} | "
              f"{r['wf']:>7.1f} {r['wa']:>7.1f} | {r['ni_f']:>6d} → {r['ni_a']:<5d} | "
              f"{r['thr_f']:>5.0f} → {r['thr_a']:<5.0f}")
    print("-" * W)
    for key, label in [("rd_old", "viejo (timeLoss red)"),
                       ("rd_red", "RED robusto  ⭐"),
                       ("rd_corr", "CORREDOR S→N")]:
        m, sd, se, lo, hi = _agg([r[key] for r in rows])
        crosses = lo <= 0 <= hi
        pos = sum(1 for r in rows if r[key] > 0)
        print(f"  RD% {label:<22} = {m:>+6.2f}% ± SD {sd:5.2f} (SE {se:.2f})  "
              f"[{lo:+.2f}…{hi:+.2f}]  ¿cruza 0? {'SÍ→EMPATE' if crosses else 'NO→señal'}  "
              f">0: {pos}/{len(rows)}")
    print("-" * W)
    print("DESGLOSE espera-para-entrar vs manejar-adentro (RED robusto, media de las semillas):")
    fw = statistics.mean([r["wf_wait"] for r in rows])
    aw = statistics.mean([r["wa_wait"] for r in rows])
    fi = statistics.mean([r["wf_in"] for r in rows])
    ai = statistics.mean([r["wa_in"] for r in rows])
    print(f"  ESPERA entrar:  fijo {fw:6.1f}s → adap {aw:6.1f}s   ({_rd(fw, aw):+.1f}%)")
    print(f"  ADENTRO red:    fijo {fi:6.1f}s → adap {ai:6.1f}s   ({_rd(fi, ai):+.1f}%)")
    print(f"  → historia: el adaptativo {'BAJA fuerte la espera' if aw < fw else 'sube la espera'}; "
          f"adentro {'baja' if ai < fi else 'sube (tapón relocalizado)'}.")
    nif = statistics.mean([r["ni_f"] for r in rows])
    nia = statistics.mean([r["ni_a"] for r in rows])
    ccf = statistics.mean([r["cc_f"] for r in rows])
    cca = statistics.mean([r["cc_a"] for r in rows])
    print(f"CENSURA (media): nunca-insertados fijo {nif:.0f} → adap {nia:.0f} | "
          f"completación corredor S→N fijo {ccf:.0%} → adap {cca:.0%}")
    print("=" * W)
    return 0


def offsets(offs: list[int]) -> int:
    rows = []
    for o in offs:
        k = _k(DATA / f"peak_s_n_adaptive_seed42_cyc90_off{o}_end{END}")
        if k is None:
            print(f"  ⚠️ falta output offset {o} — salteado", file=sys.stderr)
            continue
        rows.append({
            "off": o,
            "w": k.w_red_door2door_s_postwarm,
            "w_wait": k.w_red_wait_s_postwarm, "w_in": k.w_red_inside_s_postwarm,
            "corr": k.d2d_corr_s_postwarm, "cc": k.corr_completion_ratio,
            "ni": k.never_inserted_veh, "thr": k.throughput_veh_per_h,
            "old_tl": k.net_mean_timeloss_s_postwarm,
        })

    W = 96
    print("\n" + "=" * W)
    print(f"BARRIDO OFFSETS — onda verde | PUERTA-A-PUERTA robusta | seed 42 | ciclo 90s | "
          f"end={END} warmup={WARMUP}")
    print("=" * W)
    print(f"{'offset':>6} | {'W_RED':>7} {'espera':>7} {'adentro':>8} | {'CORR_S→N':>9} {'compl%':>7} | "
          f"{'nunca_ins':>9} {'thr':>6} | {'viejo_tl':>8}")
    base = next((r for r in rows if r["off"] == 0), None)
    for r in rows:
        d = _rd(base["w"], r["w"]) if base else 0.0
        tag = "  <- CONTROL" if r["off"] == 0 else f"  ΔW vs off0={d:+.1f}%"
        print(f"{r['off']:>6} | {r['w']:>7.1f} {r['w_wait']:>7.1f} {r['w_in']:>8.1f} | "
              f"{r['corr']:>9.1f} {r['cc']:>6.0%} | {r['ni']:>9d} {r['thr']:>6.0f} | "
              f"{r['old_tl']:>8.1f}{tag}")
    print("-" * W)
    best = min(rows, key=lambda r: r["w"])
    print(f"ÓPTIMO con métrica robusta (min W_RED): offset={best['off']}s  W={best['w']:.1f}s")
    if base:
        print(f"  (control offset=0: W={base['w']:.1f}s → "
              f"{'algún offset MEJORA' if best['off'] != 0 else 'offset=0 SIGUE óptimo'})")
    print("=" * W)
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "all"
    if cmd in ("ie05", "all"):
        ie05(list(range(42, 52)))
    if cmd in ("offsets", "all"):
        offsets(list(range(0, 81, 10)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
