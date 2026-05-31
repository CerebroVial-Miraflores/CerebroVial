"""Barrido del eje de acoplamiento τ (turn_ratio) del MP de RED — corredor Larco.

Caracteriza el eje completo: τ=0 (≡ per-node, +15.7%) · τ=0.5 · τ=0.75 · τ=1.0 (−9.2%).
τ pondera cuánto de la cola del link interno Benavides→Schell entra en la presión de
la fase Larco de Benavides: P(φ)=s·(x_local − τ·x_down). τ=0 apaga el término (per-node);
τ=1.0 acopla al máximo.

Reutiliza:
  - per-node (τ=0): histórico IE05 ya validado (peak_s_n_adaptive_seed*).
  - τ=1.0: corrida previa del sweep 3-brazos (peak_s_n_mpred_downon_seed*).
  - estadística pareada (t-CI, Wilcoxon exacto) de sweep_seeds_downstream.

Corre fresco solo τ∈{0.5, 0.75} (10 semillas c/u). Veredicto por τ = delta pareado
Δ=W_RED(MP-red τ) − W_RED(per-node), IC pareado (t, df=9) + Wilcoxon. Δ<0 ⇒ mejor que per-node.

Pre-flight ligero (el cableado ya se probó a τ=1.0):
  (a) per-node reproduce IE05 exacto al dígito (integridad del harness entre sesiones).
  (b) τ=0.5, 0.75, 1.0 dan W_RED DISTINTO entre sí y de τ=0 (la perilla turn_ratio opera).

Uso:
    invoke up   # motor :8001 con Etapa 1 deployada
    simulation/.venv/bin/python scripts/sweep_tau.py 42 43 44 45 46 47 48 49 50 51
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
PARAMS = CONF / "demand_params.yaml"
DATA = SIM_ROOT / "data" / "corredor_larco"
SUMO_PKG = SIM_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "sumo"
HEALTH = os.environ.get("ENGINE_HEALTH", "http://localhost:8001/control/health")
END, WARMUP = 1800, 600

os.environ.setdefault("SUMO_HOME", str(SUMO_PKG))
os.environ["PATH"] = f"{SUMO_PKG}/bin:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(SIM_ROOT / "src"))

from cerebrovial_simulation.kpis import collect_corredor  # noqa: E402
from cerebrovial_simulation.traci_adapter import corredor_adaptive  # noqa: E402
from sweep_seeds_downstream import _paired_t_ci, _wilcoxon_signed_rank_exact, _edge  # noqa: E402

TAUS = [0.5, 0.75]


def _collect(out_dir: Path):
    return collect_corredor.collect_corredor(out_dir, WARMUP, PARAMS) if out_dir.exists() else None


def _w(out_dir: Path):
    k = _collect(out_dir)
    return k.w_red_door2door_s_postwarm if k else None


def _run_mpred_tau(seed: int, tau: float):
    tag = f"tau{int(round(tau*100)):03d}"
    out_dir = DATA / f"peak_s_n_mpred_{tag}_seed{seed}_end{END}"
    corredor_adaptive.run_adaptive(seed=seed, end_s=END, out_dir=out_dir,
                                   downstream=True, turn_ratio=tau)
    return _collect(out_dir)


def _delta_block(label: str, deltas: list[float]):
    mean, sd, se, lo, hi = _paired_t_ci(deltas)
    w_obs, p_w = _wilcoxon_signed_rank_exact(deltas)
    excl = not (lo <= 0 <= hi)
    nbetter = sum(1 for d in deltas if d < 0)
    print(f"  [{label}] Δ por semilla: {[round(d,1) for d in deltas]}")
    print(f"           media Δ={mean:+.2f}s ± SD {sd:.2f}  IC95 t=[{lo:+.2f},{hi:+.2f}] "
          f"¿excluye 0? {'SÍ' if excl else 'NO'}  Wilcoxon W+={w_obs:.1f} p={p_w:.4f}  "
          f"Δ<0(mejor): {nbetter}/{len(deltas)}")
    return mean, excl, nbetter


def main(seeds: list[int]) -> int:
    try:
        assert requests.get(HEALTH, timeout=5).status_code == 200
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: motor no responde 200 en {HEALTH} ({exc}). `invoke up` primero.", file=sys.stderr)
        return 1
    subprocess.run([str(SIM_ROOT / ".venv/bin/python"), str(HERE / "generate_corredor_routes.py")],
                   check=True, capture_output=True)

    # --- PRE-FLIGHT (a): per-node reproduce IE05 exacto (integridad del harness) ---
    s0 = seeds[0]
    hist = _w(DATA / f"peak_s_n_adaptive_seed{s0}_end{END}")
    chk_dir = DATA / f"peak_s_n_tau_pernodecheck_seed{s0}_end{END}"
    corredor_adaptive.run_adaptive(seed=s0, end_s=END, out_dir=chk_dir, downstream=False)
    w_chk = _w(chk_dir)
    a_ok = hist is not None and abs(w_chk - hist) < 1e-9
    print(f"\n[PRE-FLIGHT a] per-node seed {s0}: fresco={w_chk} histórico={hist} "
          f"→ {'OK exacto' if a_ok else '✗ MISMATCH (harness derivó)'}")
    if not a_ok:
        print("PARAR: el harness no reproduce IE05.", file=sys.stderr)
        return 2

    # --- correr τ∈{0.5,0.75} fresco; reusar τ=0 (hist) y τ=1.0 (downon previo) ---
    rows = []
    for s in seeds:
        r = {"seed": s}
        r["fijo"] = _w(DATA / f"peak_s_n_seed{s}_end{END}")
        r[0.0] = _w(DATA / f"peak_s_n_adaptive_seed{s}_end{END}")          # per-node = τ0
        r[1.0] = _w(DATA / f"peak_s_n_mpred_downon_seed{s}_end{END}")      # τ=1.0 previo
        for tau in TAUS:
            k = _run_mpred_tau(s, tau)
            r[tau] = k.w_red_door2door_s_postwarm
            if tau == 0.5:
                r["benSchMean_05"] = _edge(k, "benavides_to_schell(INT)", "mean_queue_m")
                r["wait_05"], r["in_05"] = k.w_red_wait_s_postwarm, k.w_red_inside_s_postwarm
            if tau == 0.75:
                r["benSchMean_075"] = _edge(k, "benavides_to_schell(INT)", "mean_queue_m")
                r["wait_075"], r["in_075"] = k.w_red_wait_s_postwarm, k.w_red_inside_s_postwarm
        rows.append(r)
        print(f"  seed {s}: τ0={r[0.0]:.1f} τ.5={r[0.5]:.1f} τ.75={r[0.75]:.1f} τ1={r[1.0]:.1f}",
              file=sys.stderr)

    axis = [0.0, 0.5, 0.75, 1.0]
    aggW = {t: statistics.mean([r[t] for r in rows]) for t in axis}
    aggFijo = statistics.mean([r["fijo"] for r in rows])

    # --- PRE-FLIGHT (b): los τ dan W_RED distinto (la perilla opera) ---
    vals = [round(aggW[t], 2) for t in axis]
    distinct = len(set(vals)) == len(vals)
    print(f"\n[PRE-FLIGHT b] W_RED agregado por τ {dict(zip(axis, vals))} "
          f"→ {'OK los 4 distintos (turn_ratio opera)' if distinct else '✗ algún τ no cambia nada'}")

    W = 96
    print("\n" + "=" * W)
    print(f"BARRIDO τ (eje de acoplamiento) — MP de red | RED puerta-a-puerta | n={len(rows)} pareadas")
    print("=" * W)
    print("[EJE τ — W_RED agregado y RD% vs fijo]")
    print(f"  fijo            W_RED = {aggFijo:7.2f}s")
    for t in axis:
        rd = (aggFijo - aggW[t]) / aggFijo * 100
        lab = "per-node (τ=0)" if t == 0.0 else f"MP-red τ={t}"
        print(f"  {lab:<16} W_RED = {aggW[t]:7.2f}s   RD% vs fijo = {rd:+5.1f}%")

    print("\n[VEREDICTO por τ — delta pareado Δ = W_RED(MP-red τ) − W_RED(per-node)]")
    verdicts = {}
    for t in [0.5, 0.75, 1.0]:
        deltas = [r[t] - r[0.0] for r in rows]
        mean, excl, nb = _delta_block(f"τ={t}", deltas)
        verdicts[t] = (mean, excl, nb)

    any_better = any(excl and mean < 0 for (mean, excl, nb) in verdicts.values())
    print("\n[CONCLUSIÓN DEL EJE]")
    seq = " → ".join(f"τ{t}:{(aggFijo-aggW[t])/aggFijo*100:+.1f}%" for t in axis)
    print(f"  RD% vs fijo a lo largo del eje: {seq}")
    if any_better:
        print("  ⚠️ ALGÚN τ SUPERA al per-node (IC<0) — REABRE la conclusión, NO refutación.")
    else:
        print("  Ningún τ supera al per-node (ningún IC<0). per-node (τ=0) es el ÓPTIMO del eje.")

    print("\n[TRADE-OFF por τ (media) — alivio del link vs espera de entrada]")
    def m(k):
        return statistics.mean([r[k] for r in rows])
    print(f"  benSch mean: τ0=(ver IE05) τ.5={m('benSchMean_05'):.1f}m τ.75={m('benSchMean_075'):.1f}m")
    print(f"  w_wait:      τ.5={m('wait_05'):.1f}s τ.75={m('wait_075'):.1f}s")
    print(f"  w_inside:    τ.5={m('in_05'):.1f}s τ.75={m('in_075'):.1f}s")
    print("=" * W)
    return 0


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or list(range(42, 52))
    sys.exit(main(seeds))
