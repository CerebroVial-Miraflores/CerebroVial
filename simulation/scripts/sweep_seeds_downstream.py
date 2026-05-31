"""Barrido 42-51, TRES brazos pareados — Max Pressure de RED (downstream) vs
per-node vs fijo. Veredicto de la hipótesis de Etapa 2.

Brazos (pareados por semilla, misma demanda determinista):
  fijo      — control estático (net.xml).
  per-node  — MP simplificado P=s·q (downstream OFF). Reproduce IE05.
  MP-red    — MP de red P=s·(q − Στ·q_down) (downstream ON; Benavides lee #1).

Veredicto = DELTA PAREADO  Δ = W_RED(MP-red) − W_RED(per-node)  por semilla.
Se decide por si el IC del Δ EXCLUYE el cero (t pareado df=9) + Wilcoxon signed-rank
exacto, NO por la media a ojo ni por el conteo de semillas. Δ<0 ⇒ MP-red MEJOR
(menos demora RED). Las medias marginales por brazo (±~8) se solapan y no prueban nada.

No-regresión (compuerta dura): el brazo per-node fresco reproduce el W_RED histórico
de IE05 semilla por semilla, EXACTO al dígito. Si clava el agregado pero no por semilla,
se filtró algo no-motor.

Uso:
    invoke up      # motor vivo en :8001 CON el código de Etapa 1 (rebuildeado)
    simulation/.venv/bin/python scripts/sweep_seeds_downstream.py 42 43 44 45 46 47 48 49 50 51
"""
from __future__ import annotations

import math
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
T_CRIT_DF9 = 2.262  # t_{0.975, df=9} para el IC 95% pareado (n=10)

os.environ.setdefault("SUMO_HOME", str(SUMO_PKG))
os.environ["PATH"] = f"{SUMO_PKG}/bin:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(SIM_ROOT / "src"))

from cerebrovial_simulation.kpis import collect_corredor  # noqa: E402
from cerebrovial_simulation.traci_adapter import corredor_adaptive  # noqa: E402


def _collect(out_dir: Path):
    return collect_corredor.collect_corredor(out_dir, WARMUP, PARAMS) if out_dir.exists() else None


def _edge(kpis, label: str, attr: str) -> float:
    for eq in kpis.edge_queues:
        if eq.label == label:
            return getattr(eq, attr)
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
    return _collect(out_dir)


def _run_adaptive(seed: int, downstream: bool, tag: str):
    out_dir = DATA / f"peak_s_n_mpred_{tag}_seed{seed}_end{END}"
    stats = corredor_adaptive.run_adaptive(seed=seed, end_s=END, out_dir=out_dir, downstream=downstream)
    errs = sum(len(s["errors"]) for s in stats.values())
    if errs:
        print(f"  ⚠️ seed {seed} [{tag}]: {errs} errores de motor", file=sys.stderr)
    return _collect(out_dir)


# --- Estadística pareada sin dependencias (n=10) ---

def _paired_t_ci(d: list[float]):
    n = len(d)
    mean = statistics.mean(d)
    sd = statistics.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    half = T_CRIT_DF9 * se
    return mean, sd, se, mean - half, mean + half


def _wilcoxon_signed_rank_exact(d: list[float]):
    """p two-sided exacto por enumeración de signos (n pequeño). Devuelve (W+, p)."""
    nz = [x for x in d if x != 0.0]
    m = len(nz)
    if m == 0:
        return 0.0, 1.0
    order = sorted(range(m), key=lambda i: abs(nz[i]))
    ranks = [0.0] * m
    i = 0
    while i < m:
        j = i
        while j + 1 < m and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # rangos promedio en empates de |d|
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_obs = sum(ranks[i] for i in range(m) if nz[i] > 0)
    # distribución exacta de W+ bajo H0 (cada signo ±, equiprobable)
    from itertools import product
    dist = {}
    for signs in product([0, 1], repeat=m):
        wp = sum(ranks[i] for i in range(m) if signs[i])
        dist[wp] = dist.get(wp, 0) + 1
    n_perm = 2 ** m
    p_le = sum(c for wp, c in dist.items() if wp <= w_obs) / n_perm
    p_ge = sum(c for wp, c in dist.items() if wp >= w_obs) / n_perm
    p = min(1.0, 2.0 * min(p_le, p_ge))
    return w_obs, p


def main(seeds: list[int]) -> int:
    try:
        assert requests.get(HEALTH, timeout=5).status_code == 200
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: motor no responde 200 en {HEALTH} ({exc}). `invoke up` primero.", file=sys.stderr)
        return 1
    subprocess.run([str(SIM_ROOT / ".venv/bin/python"), str(HERE / "generate_corredor_routes.py")],
                   check=True, capture_output=True)

    rows = []
    for s in seeds:
        fx = _run_fixed(s)
        pn = _run_adaptive(s, downstream=False, tag="pernode")
        rd = _run_adaptive(s, downstream=True, tag="downon")
        hist = _collect(DATA / f"peak_s_n_adaptive_seed{s}_end{END}")  # IE05 histórico per-node
        rows.append({
            "seed": s,
            "w_fix": fx.w_red_door2door_s_postwarm,
            "w_pn": pn.w_red_door2door_s_postwarm,
            "w_rd": rd.w_red_door2door_s_postwarm,
            "w_hist": hist.w_red_door2door_s_postwarm if hist else None,
            "delta": rd.w_red_door2door_s_postwarm - pn.w_red_door2door_s_postwarm,
            # desglose trade-off
            "wait_pn": pn.w_red_wait_s_postwarm, "wait_rd": rd.w_red_wait_s_postwarm,
            "in_pn": pn.w_red_inside_s_postwarm, "in_rd": rd.w_red_inside_s_postwarm,
            "benSch_pn": _edge(pn, "benavides_to_schell(INT)", "max_queue_m"),
            "benSch_rd": _edge(rd, "benavides_to_schell(INT)", "max_queue_m"),
            "benSchMean_pn": _edge(pn, "benavides_to_schell(INT)", "mean_queue_m"),
            "benSchMean_rd": _edge(rd, "benavides_to_schell(INT)", "mean_queue_m"),
            "larcoS_pn": _edge(pn, "larcoS_approach", "mean_queue_m"),
            "larcoS_rd": _edge(rd, "larcoS_approach", "mean_queue_m"),
            "thr_rd": rd.throughput_veh_per_h, "ni_rd": rd.never_inserted_veh,
        })
        print(f"  seed {s}: W fijo={rows[-1]['w_fix']:.1f} pernode={rows[-1]['w_pn']:.1f} "
              f"MP-red={rows[-1]['w_rd']:.1f}  Δ(rd-pn)={rows[-1]['delta']:+.1f}", file=sys.stderr)

    W = 104
    print("\n" + "=" * W)
    print(f"BARRIDO MP-RED — downstream vs per-node vs fijo | RED puerta-a-puerta | n={len(rows)} pareadas | "
          f"end={END} warmup={WARMUP}")
    print("=" * W)

    # --- COMPUERTA: no-regresión per-node semilla a semilla, exacto ---
    print("\n[NO-REGRESIÓN per-node = IE05, semilla a semilla, EXACTO al dígito]")
    nonreg_fail = []
    for r in rows:
        ok = r["w_hist"] is not None and abs(r["w_pn"] - r["w_hist"]) < 1e-9
        if not ok:
            nonreg_fail.append(r["seed"])
        print(f"  seed {r['seed']}: per-node fresco={r['w_pn']:.1f}  histórico={r['w_hist']}  "
              f"{'OK' if ok else '✗ MISMATCH'}")
    agg_pn = statistics.mean([r["w_pn"] for r in rows])
    agg_hist = statistics.mean([r["w_hist"] for r in rows if r["w_hist"] is not None])
    print(f"  agregado per-node={agg_pn:.2f}  histórico={agg_hist:.2f}")
    if nonreg_fail:
        print(f"  ⚠️ MISMATCH en semillas {nonreg_fail} → algo no-motor se coló. VEREDICTO NO VÁLIDO.")

    # --- Brazos: W_RED por brazo (contexto, NO veredicto) ---
    print("\n[W_RED por brazo — CONTEXTO, las medias marginales se solapan]")
    for key, lab in [("w_fix", "fijo"), ("w_pn", "per-node"), ("w_rd", "MP-red")]:
        xs = [r[key] for r in rows]
        m, sd = statistics.mean(xs), statistics.stdev(xs)
        print(f"  {lab:<9} W_RED = {m:7.2f}s ± SD {sd:5.2f}")
    rd_pn_vs_fix = statistics.mean([(r["w_fix"] - r["w_pn"]) / r["w_fix"] * 100 for r in rows])
    rd_rd_vs_fix = statistics.mean([(r["w_fix"] - r["w_rd"]) / r["w_fix"] * 100 for r in rows])
    print(f"  RD% vs fijo: per-node {rd_pn_vs_fix:+.1f}%  |  MP-red {rd_rd_vs_fix:+.1f}%")

    # --- VEREDICTO: delta pareado Δ = MP-red − per-node ---
    deltas = [r["delta"] for r in rows]
    mean, sd, se, lo, hi = _paired_t_ci(deltas)
    w_obs, p_w = _wilcoxon_signed_rank_exact(deltas)
    excludes_zero = not (lo <= 0 <= hi)
    n_better = sum(1 for d in deltas if d < 0)  # Δ<0 ⇒ MP-red mejor
    print("\n[VEREDICTO — DELTA PAREADO Δ = W_RED(MP-red) − W_RED(per-node)]")
    print(f"  Δ por semilla: {[round(d, 1) for d in deltas]}")
    print(f"  media Δ = {mean:+.2f}s ± SD {sd:.2f} (SE {se:.2f})")
    print(f"  IC 95% pareado (t, df=9) = [{lo:+.2f}, {hi:+.2f}]  → ¿EXCLUYE el 0? "
          f"{'SÍ' if excludes_zero else 'NO'}")
    print(f"  Wilcoxon signed-rank exacto: W+={w_obs:.1f}  p(two-sided)={p_w:.4f}")
    print(f"  semillas con Δ<0 (MP-red mejor): {n_better}/{len(deltas)}")
    if not excludes_zero:
        verdict = "EL DOWNSTREAM NO APORTA (IC cruza 0): no se distingue de per-node"
    elif mean < 0:
        verdict = "EL DOWNSTREAM APORTA: MP-red baja la demora RED (IC<0)"
    else:
        verdict = "EL DOWNSTREAM EMPEORA: MP-red sube la demora RED (IC>0)"
    print(f"  → {verdict}")

    # --- Trade-off (mecanismo): ¿baja benSch a costa de w_wait? ---
    print("\n[TRADE-OFF — mecanismo per-node → MP-red (media de semillas)]")
    def _mean(k):
        return statistics.mean([r[k] for r in rows])
    print(f"  espera-entrar  w_wait : {_mean('wait_pn'):6.1f}s → {_mean('wait_rd'):6.1f}s")
    print(f"  adentro-red    w_inside: {_mean('in_pn'):6.1f}s → {_mean('in_rd'):6.1f}s")
    print(f"  tapón interno  benSch max : {_mean('benSch_pn'):6.1f}m → {_mean('benSch_rd'):6.1f}m "
          f"(mean {_mean('benSchMean_pn'):.1f}→{_mean('benSchMean_rd'):.1f})")
    print(f"  entrada Benav. larcoS mean: {_mean('larcoS_pn'):6.1f}m → {_mean('larcoS_rd'):6.1f}m")
    print("  (benSch: si max≫mean en MP-red → cola burstea/oscila; si max≈mean → saturación sostenida)")
    print("=" * W)
    return 2 if nonreg_fail else 0


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or list(range(42, 52))
    sys.exit(main(seeds))
