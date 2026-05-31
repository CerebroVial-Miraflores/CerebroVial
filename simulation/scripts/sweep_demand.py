"""Barrido de demanda (EXPLORACIÓN) — zanja si existe acoplamiento inter-cruce
Benavides→Schell sostenido antes de gridlock, o si el desacople es estructural
(cuello aguas arriba en Benavides).

NO commitear como escenario. Solo varía demand_scale; NO toca tiempos de control.
Determinista: seed=42, end=1800, warmup=600, mismos detectores y sumocfg.

Separa explícitamente:
  Dim 1  ¿la demanda extra LLEGA a Schell o se atasca en la entrada sur?
         - flujo cruzando Benavides hacia el link interno (e2_benSch nVehEntered/h)
         - backlog de inserción (summary step_waiting max/final; loaded vs inserted)
  Dim 2  ¿acoplamiento inter-cruce sostenido?
         - benSch#1 (97 m): jam ≥48 m (medio link) sostenido ≥3 buckets = spillback real
         - serie temporal de benSch#1 y tarSch#2
  Dim 3  ¿gridlock? teleports, completion (arrived/generados), running sin ended

Uso:
    simulation/.venv/bin/python scripts/sweep_demand.py 1.0 1.1 1.2 1.3
"""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
CONF = SIM_ROOT / "conf" / "corredor_larco"
CFG = CONF / "peak_s_n.sumocfg"
DET = CONF / "corredor_larco.det.add.xml"
PARAMS = CONF / "demand_params.yaml"
DATA = SIM_ROOT / "data" / "corredor_larco"
SUMO_PKG = SIM_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "sumo"

SEED, END, WARMUP = 42, 1800, 600

# Detectores por grupo (max de cola sobre carriles).
G_LARCOS = ["e2_larcoS_0", "e2_larcoS_1", "e2_larcoS_2"]
G_BENSCH = ["e2_benSch_0", "e2_benSch_1", "e2_benSch_2"]   # link interno #1 (97 m)
G_TARSCH = ["e2_tarSch_0", "e2_tarSch_1", "e2_tarSch_2"]   # aprox Schell #2 (48 m)
BENSCH_HALF = 48.0          # medio link de 97 m → umbral de acoplamiento
COUPLE_MIN_BUCKETS = 3


def _f(v) -> float:
    return 0.0 if v in (None, "") else float(v)


def _gen_routes(scale: float, out_dir: Path) -> Path:
    out = out_dir / f"peak_s_n_scale{int(round(scale*100))}.rou.xml"
    subprocess.run(
        [str(SIM_ROOT / ".venv/bin/python"), str(HERE / "generate_corredor_routes.py"),
         "--scale", str(scale), "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    return out


def _run_sumo(routes: Path, out_dir: Path) -> None:
    det_local = out_dir / DET.name
    det_local.write_bytes(DET.read_bytes())
    env = dict(os.environ, SUMO_HOME=str(SUMO_PKG), PATH=f"{SUMO_PKG}/bin:{os.environ['PATH']}")
    args = [
        str(SUMO_PKG / "bin" / "sumo"), "-c", str(CFG),
        "--route-files", str(routes),
        "--additional-files", str(det_local),
        "--seed", str(SEED), "--end", str(END),
        "--summary-output", str(out_dir / "summary.parquet"),
        "--tripinfo-output", str(out_dir / "tripinfo.parquet"),
        "--no-step-log", "--log", str(out_dir / "sumo.log"),
    ]
    p = subprocess.run(args, cwd=out_dir, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"sumo falló scale_dir={out_dir.name}:\n{p.stderr}")


def _parse_lanearea(out_dir: Path):
    """Series temporales (max cola sobre carriles por bucket) y flujos (nVehEntered)."""
    root = ET.parse(out_dir / "lanearea.xml").getroot()
    det2g = {}
    for g, ds in (("larcoS", G_LARCOS), ("benSch", G_BENSCH), ("tarSch", G_TARSCH)):
        for d in ds:
            det2g[d] = g
    jam = defaultdict(lambda: defaultdict(float))     # bucket_end -> g -> max jam
    entered = defaultdict(float)                        # g -> sum nVehEntered (todos los buckets)
    for iv in root.findall("interval"):
        g = det2g.get(iv.attrib["id"])
        if not g:
            continue
        end = _f(iv.attrib.get("end"))
        jam[end][g] = max(jam[end][g], _f(iv.attrib.get("maxJamLengthInMeters")))
        entered[g] += _f(iv.attrib.get("nVehEntered"))
    return jam, entered


def _parse_summary(out_dir: Path) -> dict:
    t = pq.read_table(str(out_dir / "summary.parquet"))
    cols = set(t.column_names)

    def s(name):
        c = f"step_{name}"
        return [_f(v) for v in t.column(c).to_pylist()] if c in cols else []

    waiting, loaded, inserted = s("waiting"), s("loaded"), s("inserted")
    teleports, running, ended = s("teleports"), s("running"), s("ended")
    return {
        "wait_max": max(waiting) if waiting else 0.0,
        "wait_final": waiting[-1] if waiting else 0.0,
        "loaded": loaded[-1] if loaded else 0.0,
        "inserted": inserted[-1] if inserted else 0.0,
        "teleports": int(sum(teleports)),
        "running_final": int(running[-1]) if running else 0,
        "ended_final": int(ended[-1]) if ended else 0,
        "sim_dur": max(s("time")) if s("time") else END,
    }


def _parse_tripinfo(out_dir: Path):
    t = pq.read_table(str(out_dir / "tripinfo.parquet"))
    cols = set(t.column_names)
    idc = "tripinfo_id" if "tripinfo_id" in cols else "id"
    dc = "tripinfo_duration" if "tripinfo_duration" in cols else "duration"
    ids = t.column(idc).to_pylist() if idc in cols else []
    durs = [_f(v) for v in t.column(dc).to_pylist()] if dc in cols else []
    n = len(durs)
    corr = [d for i, d in zip(ids, durs) if str(i).startswith("f_larco_S_N.")]
    corr_tt = sum(corr) / len(corr) if corr else 0.0
    return n, corr_tt


def _scaled_generated(scale: float, dur: float) -> float:
    cfg = yaml.safe_load(PARAMS.read_text())
    total_vph = sum(round(v * scale) for v in cfg["flows_vph"].values() if v > 0)
    return total_vph * dur / 3600.0


def run_level(scale: float) -> dict:
    out_dir = DATA / f"peak_s_n_scale{int(round(scale*100))}_seed{SEED}_end{END}"
    out_dir.mkdir(parents=True, exist_ok=True)
    routes = _gen_routes(scale, out_dir)
    _run_sumo(routes, out_dir)

    jam, entered = _parse_lanearea(out_dir)
    summ = _parse_summary(out_dir)
    n_arr, corr_tt = _parse_tripinfo(out_dir)
    dur = summ["sim_dur"]
    gen = _scaled_generated(scale, dur)

    # flujos veh/h (sobre la duración real)
    h = dur / 3600.0
    flow_cross_ben = entered["benSch"] / h if h else 0.0   # cruza Benavides → link interno
    flow_south_ins = entered["larcoS"] / h if h else 0.0   # insertados que entran al link sur

    # acoplamiento: benSch#1 ≥48 m sostenido (post-warmup)
    ends = sorted(e for e in jam if e > WARMUP)
    bensch_series = [(e, jam[e]["benSch"]) for e in ends]
    tarsch_series = [(e, jam[e]["tarSch"]) for e in ends]
    maxc = c = 0
    for _, v in bensch_series:
        if v >= BENSCH_HALF:
            c += 1
            maxc = max(maxc, c)
        else:
            c = 0
    coupled = maxc >= COUPLE_MIN_BUCKETS

    return {
        "scale": scale,
        "gen": gen, "arrived": n_arr,
        "completion": (n_arr / gen) if gen else 0.0,
        "throughput": n_arr / h if h else 0.0,
        "flow_cross_ben": flow_cross_ben,
        "flow_south_ins": flow_south_ins,
        "wait_max": summ["wait_max"], "wait_final": summ["wait_final"],
        "loaded": summ["loaded"], "inserted": summ["inserted"],
        "insert_deficit": summ["loaded"] - summ["inserted"],
        "teleports": summ["teleports"],
        "running_final": summ["running_final"], "ended_final": summ["ended_final"],
        "corr_tt": corr_tt,
        "bensch_max": max((v for _, v in bensch_series), default=0.0),
        "bensch_mean": (sum(v for _, v in bensch_series) / len(bensch_series)) if bensch_series else 0.0,
        "tarsch_mean": (sum(v for _, v in tarsch_series) / len(tarsch_series)) if tarsch_series else 0.0,
        "couple_consec": maxc, "coupled": coupled,
        "bensch_series": bensch_series, "tarsch_series": tarsch_series,
    }


def main(scales: list[float]) -> int:
    rows = [run_level(s) for s in scales]

    print("\n" + "=" * 100)
    print("BARRIDO DE DEMANDA — corredor Larco | control fijo | seed=42 end=1800 warmup=600")
    print("=" * 100)

    print("\nDIM 1 — ¿la demanda extra LLEGA a Schell o se atasca en la entrada sur?")
    print(f"{'scale':>6} {'gen':>6} {'demanda_vph':>11} {'cruzaBenav/h':>13} "
          f"{'insertSur/h':>12} {'wait_max':>9} {'wait_fin':>9} {'deficit_ins':>12}")
    for r in rows:
        demanda_vph = round(r["gen"] * 3600 / (END))
        print(f"{r['scale']:>6.2f} {r['gen']:>6.0f} {demanda_vph:>11} "
              f"{r['flow_cross_ben']:>13.0f} {r['flow_south_ins']:>12.0f} "
              f"{r['wait_max']:>9.0f} {r['wait_final']:>9.0f} {r['insert_deficit']:>12.0f}")

    print("\nDIM 2 — ¿acoplamiento inter-cruce sostenido? (benSch#1 97m, umbral ≥48m por ≥3 buckets)")
    print(f"{'scale':>6} {'benSch_max':>10} {'benSch_mean':>11} {'tarSch_mean':>11} "
          f"{'consec≥48':>10} {'ACOPLADO':>9}")
    for r in rows:
        print(f"{r['scale']:>6.2f} {r['bensch_max']:>10.1f} {r['bensch_mean']:>11.1f} "
              f"{r['tarsch_mean']:>11.1f} {r['couple_consec']:>10} {'SÍ' if r['coupled'] else 'no':>9}")

    print("\nDIM 3 — ¿gridlock?")
    print(f"{'scale':>6} {'arrived':>8} {'completion':>11} {'throughput/h':>13} "
          f"{'teleports':>10} {'run_fin':>8} {'end_fin':>8} {'corrTT_s':>9}")
    for r in rows:
        print(f"{r['scale']:>6.2f} {r['arrived']:>8} {r['completion']:>11.2f} "
              f"{r['throughput']:>13.0f} {r['teleports']:>10} {r['running_final']:>8} "
              f"{r['ended_final']:>8} {r['corr_tt']:>9.1f}")

    print("\nSERIE TEMPORAL benSch#1 (97m) por nivel — cola cada 120 s post-warmup (m):")
    for r in rows:
        pts = [v for (e, v) in r["bensch_series"] if int(e) % 120 == 0]
        print(f"  scale {r['scale']:.2f}: " + " ".join(f"{v:5.1f}" for v in pts))
    print("SERIE TEMPORAL tarSch#2 (48m) por nivel:")
    for r in rows:
        pts = [v for (e, v) in r["tarsch_series"] if int(e) % 120 == 0]
        print(f"  scale {r['scale']:.2f}: " + " ".join(f"{v:5.1f}" for v in pts))

    print("\n" + "-" * 100)
    coupled = [r for r in rows if r["coupled"] and r["teleports"] < 20 and r["completion"] > 0.5]
    if coupled:
        c = min(coupled, key=lambda r: r["scale"])
        print(f"VEREDICTO: ACOPLAMIENTO sostenido SIN gridlock a partir de scale={c['scale']:.2f}.")
    else:
        print("VEREDICTO: NINGÚN nivel dio acoplamiento sostenido sin gridlock → "
              "desacople ESTRUCTURAL (cuello aguas arriba en Benavides), no de demanda.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    scales = [float(a) for a in sys.argv[1:]] or [1.0, 1.1, 1.2, 1.3]
    sys.exit(main(scales))
