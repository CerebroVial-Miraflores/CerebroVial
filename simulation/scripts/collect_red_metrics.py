"""Métricas de RED (agnósticas a la topología) de una corrida SUMO del experimento de
validación fijo-vs-adaptativo sobre Miraflores completo (F2, DHU-030).

Lee ``summary.parquet`` + ``tripinfo.parquet`` de ``--out`` y reporta:
  - net_timeLoss medio  → LA MÉTRICA del experimento (demora de red, todos los completados).
  - censura (loaded/inserted/never-inserted, %), teleports, completados → señales de salud.

A diferencia de ``cerebrovial_simulation.kpis.collect_corredor``, NO depende del corredor:
sin ``lanearea.xml``, sin ``demand_params.yaml``, sin eje S→N, sin los 6 edges instrumentados.
Solo summary + tripinfo, que cualquier corrida produce. Escala a 2 o 99 TLS sin cambios.

Uso:
    python collect_red_metrics.py --out <dir-con-summary-y-tripinfo> [--json <ruta>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def _f(v) -> float:
    """SUMO 1.26 emite los campos parquet como strings — coerción robusta a float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def collect_red(out_dir: Path) -> dict:
    summary, tripinfo = out_dir / "summary.parquet", out_dir / "tripinfo.parquet"
    for p in (summary, tripinfo):
        if not p.exists():
            raise FileNotFoundError(f"falta output: {p}")

    S = pq.read_table(str(summary))

    def col(c):
        return [_f(x) for x in S.column(c).to_pylist()]

    loaded, inserted = col("step_loaded"), col("step_inserted")
    running, ended = col("step_running"), col("step_ended")
    teleports, discarded = col("step_teleports"), col("step_discarded")
    sim_time = col("step_time")[-1] if S.num_rows else 0.0

    loaded_f, inserted_f = int(loaded[-1]), int(inserted[-1])
    never_inserted = loaded_f - inserted_f
    # step_teleports es ACUMULATIVO (serie monotónica no-decreciente; verificado en F1:
    # su último valor == nº de eventos "Teleporting vehicle" del log). El total es el
    # ÚLTIMO valor, NO la suma de la serie (sumar un acumulado infla el número).
    teleports_total = int(teleports[-1]) if teleports else 0
    discarded_total = int(sum(discarded))

    T = pq.read_table(str(tripinfo))
    durs = [_f(x) for x in T.column("tripinfo_duration").to_pylist()]
    tls = [_f(x) for x in T.column("tripinfo_timeLoss").to_pylist()]
    dds = [_f(x) for x in T.column("tripinfo_departDelay").to_pylist()]
    n_completed = len(durs)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "out_dir": str(out_dir),
        "sim_duration_s": sim_time,
        # métrica del experimento
        "net_timeloss_mean_s": round(mean(tls), 4),
        "net_travel_time_mean_s": round(mean(durs), 4),
        "depart_delay_mean_s": round(mean(dds), 4),
        # censura / inserción
        "loaded": loaded_f,
        "inserted": inserted_f,
        "never_inserted": never_inserted,
        "censura_pct": round(never_inserted / loaded_f * 100, 4) if loaded_f else 0.0,
        "discarded_total": discarded_total,
        # salud
        "teleports_total": teleports_total,
        "final_running": int(running[-1]) if running else 0,
        "final_ended": int(ended[-1]) if ended else 0,
        "completed_n": n_completed,
        "completion_pct": round(n_completed / loaded_f * 100, 2) if loaded_f else 0.0,
        "throughput_veh_per_h": round(n_completed * 3600.0 / sim_time, 1) if sim_time else 0.0,
    }


def _print(k: dict) -> None:
    print("=" * 64)
    print(f"MÉTRICAS DE RED — {k['out_dir']}  (sim {k['sim_duration_s']:.0f}s)")
    print("=" * 64)
    print(f"  net_timeLoss medio   : {k['net_timeloss_mean_s']:.4f} s   <<< MÉTRICA")
    print(f"  travel_time medio    : {k['net_travel_time_mean_s']:.1f} s")
    print(f"  departDelay medio    : {k['depart_delay_mean_s']:.2f} s")
    print("  --- censura / salud ---")
    print(f"  loaded / inserted    : {k['loaded']} / {k['inserted']}")
    print(f"  never-inserted       : {k['never_inserted']}  ({k['censura_pct']:.2f}% censura)")
    print(f"  discarded            : {k['discarded_total']}")
    print(f"  teleports (total)    : {k['teleports_total']}")
    print(f"  completados          : {k['completed_n']} ({k['completion_pct']:.1f}%)  "
          f"throughput {k['throughput_veh_per_h']:.0f} veh/h")
    print(f"  final running/ended  : {k['final_running']} / {k['final_ended']}")
    print("=" * 64)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Métricas de red agnósticas (F2).")
    ap.add_argument("--out", required=True, type=Path, help="dir con summary.parquet + tripinfo.parquet")
    ap.add_argument("--json", type=Path, default=None, help="ruta opcional para volcar el dict JSON")
    args = ap.parse_args(argv)
    k = collect_red(args.out)
    _print(k)
    if args.json:
        args.json.write_text(json.dumps(k, indent=2, ensure_ascii=False))
        print(f"  JSON → {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
