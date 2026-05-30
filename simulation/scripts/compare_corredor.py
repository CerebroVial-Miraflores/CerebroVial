"""Compara KPIs adaptativo vs fijo del corredor Larco (IE05) y calcula RD%.

Lee kpis.json de ambas corridas (collect_corredor) + adaptive_stats.json del
adaptativo. RD% = (demora_fijo − demora_adaptativo) / demora_fijo en POST-WARMUP
(timeLoss medio). Reporta el número REAL (positivo, cero o negativo) y el desglose:
cola de entrada sur a Benavides (termómetro del cuello), throughput/completion/
teleports, modos elegidos por cada cruce, y tiempo perdido por all-red en cada
control (para separar mejor-asignación-de-verde de overhead estructural).

Uso:
    python scripts/compare_corredor.py --fixed <dir_fijo> --adaptive <dir_adaptativo>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(d: Path) -> dict:
    return json.loads((d / "kpis.json").read_text())


def _edge(kpis: dict, label: str) -> dict:
    for eq in kpis.get("edge_queues", []):
        if eq["label"] == label:
            return eq
    return {"max_queue_m": 0.0, "mean_queue_m": 0.0, "length_m": 0.0}


def _rd(fixed: float, adapt: float) -> float:
    return (fixed - adapt) / fixed * 100.0 if fixed else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True, type=Path)
    ap.add_argument("--adaptive", required=True, type=Path)
    args = ap.parse_args()

    f, a = _load(args.fixed), _load(args.adaptive)
    stats_path = args.adaptive / "adaptive_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    W = 74
    print("=" * W)
    print("IE05 — ADAPTATIVO (Max Pressure per-node) vs FIJO | corredor Larco")
    print(f"  fijo={args.fixed.name}")
    print(f"  adap={args.adaptive.name}")
    print("=" * W)

    def row(lbl, fv, av, unit="", rd=False):
        extra = f"   RD%={_rd(fv, av):+.1f}%" if rd else ""
        print(f"  {lbl:32s} fijo={fv:>9.1f}{unit}  adap={av:>9.1f}{unit}{extra}")

    print("DEMORA / TIEMPO DE VIAJE (post-warmup)")
    row("timeLoss medio red [RD%]", f["net_mean_timeloss_s_postwarm"],
        a["net_mean_timeloss_s_postwarm"], "s", rd=True)
    row("TT medio red", f["net_mean_travel_time_s_postwarm"],
        a["net_mean_travel_time_s_postwarm"], "s")
    row("TT medio corredor S→N", f["corridor_mean_travel_time_s_postwarm"],
        a["corridor_mean_travel_time_s_postwarm"], "s")

    print("CUELLO — cola de entrada sur a Benavides (termómetro directo)")
    fe, ae = _edge(f, "larcoS_approach"), _edge(a, "larcoS_approach")
    row("larcoS max (L=247.6m)", fe["max_queue_m"], ae["max_queue_m"], "m")
    row("larcoS media", fe["mean_queue_m"], ae["mean_queue_m"], "m")

    print("LINKS INTERNOS (pico de cola)")
    for lbl in ("benavides_to_schell(INT)", "tarata_to_schell", "schell_to_diezcanseco(INT)"):
        row(lbl, _edge(f, lbl)["max_queue_m"], _edge(a, lbl)["max_queue_m"], "m")

    print("THROUGHPUT / COMPLETION / GRIDLOCK")
    row("throughput", f["throughput_veh_per_h"], a["throughput_veh_per_h"], "")
    row("completion", f["completion_ratio"], a["completion_ratio"], "")
    print(f"  {'arrived / teleports':32s} "
          f"fijo={f['arrived_veh']}/{f['teleports_total']}  "
          f"adap={a['arrived_veh']}/{a['teleports_total']}")

    print("CONTROL — modos del motor + all-red (desglose del RD%)")
    allred_adap = 0
    for iid, s in stats.items():
        modes = s.get("modes", [])
        mp = sum(1 for m in modes if m == "max_pressure")
        web = sum(1 for m in modes if m == "webster")
        allred_adap += s.get("allred_steps", 0)
        print(f"  [{iid}] calls={s.get('calls', 0)} apps={s.get('applications', 0)} "
              f"MP={mp} Webster={web} all-red={s.get('allred_steps', 0)}s "
              f"errs={len(s.get('errors', []))}")
    print(f"  all-red perdido: fijo=0s (net.xml sin all-red)  adap={allred_adap}s "
          "→ overhead estructural del adaptativo")

    print("-" * W)
    rd = _rd(f["net_mean_timeloss_s_postwarm"], a["net_mean_timeloss_s_postwarm"])
    verdict = ("POSITIVO" if rd > 1 else "NULO/MARGINAL" if rd >= -1 else "NEGATIVO")
    print(f"RD% (demora red, post-warmup) = {rd:+.1f}%  → {verdict}")
    print("=" * W)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
