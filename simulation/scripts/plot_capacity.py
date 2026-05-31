"""Figura + tabla de la caracterización demanda↔capacidad (lee capacity_sweep.csv).

Separado del barrido para regenerar la figura sin re-simular. Produce:
  data/corredor_larco/capacity_sweep/capacity_sweep.png  (+ .svg)
  data/corredor_larco/capacity_sweep/capacity_sweep_summary.csv

Panel A — W_RED puerta-a-puerta vs demanda (fijo y per-node, barras ±SD):
  lo arreglable por temporizado vive acá. Línea vertical en scale=1.0. Overlay
  teórico = piso de capacidad (W_libre + demora determinística de sobre-saturación
  (λ−μ̂)·T/(2λ), cantidades del CORREDOR, T=ventana post-warmup). Régimen contaminado
  por teleports sombreado (optimista).

Panel B — THROUGHPUT vs demanda: descarga MEDIDA por Schell (e2_schDc) y throughput
  de red; línea de demanda corredor ofrecida λ_corr(s); tasa de teleports en eje
  derecho con marca de onset. μ̂ = PICO de la descarga en el régimen teleport≈0
  (no el valor en 2.0; en colapso la descarga puede caer).

HONESTIDAD DEL TECHO: los teleports inflan throughput y deprimen demora. μ̂ se estima
SOLO donde teleports≈0; más allá la curva es optimista y se marca como tal.

Uso:
    simulation/.venv/bin/python scripts/plot_capacity.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
DATA = SIM_ROOT / "data" / "corredor_larco"
CSV_IN = DATA / "capacity_sweep" / "capacity_sweep.csv"
PNG_OUT = DATA / "capacity_sweep" / "capacity_sweep.png"
SVG_OUT = DATA / "capacity_sweep" / "capacity_sweep.svg"
SUMMARY_OUT = DATA / "capacity_sweep" / "capacity_sweep_summary.csv"

END, WARMUP = 1800, 600
T_PW = END - WARMUP  # ventana post-warmup (s), consistente con W_RED y el overlay

# Umbral de teleports para declarar el régimen "contaminado/optimista" (media por punto).
TELEPORT_CLEAN = 1.0

ARMS = ["fijo", "pernode"]
ARM_STYLE = {
    "fijo":    {"color": "#c44e52", "marker": "s", "label": "fijo (TLLogic estático)"},
    "pernode": {"color": "#4c72b0", "marker": "o", "label": "per-node (Max-Pressure)"},
}
NUM_METRICS = ["w_red", "throughput_net", "schell_discharge_vph",
               "completion", "teleports", "never_inserted", "benSch_max"]


def _load() -> list[dict]:
    if not CSV_IN.exists():
        sys.exit(f"ERROR: no existe {CSV_IN}. Corré sweep_capacity.py primero.")
    with CSV_IN.open() as fh:
        return list(csv.DictReader(fh))


def _aggregate(rows: list[dict]):
    """(scale, arm) → {metric: (mean, sd)}; además scales ordenadas y λ_corr por escala."""
    buckets: dict[tuple[float, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    lam_corr: dict[float, float] = {}
    for r in rows:
        scale = float(r["scale"])
        key = (scale, r["arm"])
        lam_corr[scale] = float(r["demand_corr_vph"])
        for m in NUM_METRICS:
            buckets[key][m].append(float(r[m]))
    agg: dict[tuple[float, str], dict[str, tuple[float, float]]] = {}
    for key, mets in buckets.items():
        agg[key] = {m: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
                    for m, v in mets.items()}
    scales = sorted({s for s, _ in agg})
    return agg, scales, lam_corr


def _series(agg, scales, arm, metric):
    xs, means, sds = [], [], []
    for s in scales:
        if (s, arm) in agg:
            xs.append(s)
            mu, sd = agg[(s, arm)][metric]
            means.append(mu)
            sds.append(sd)
    return xs, means, sds


def _teleport_onset(agg, scales) -> float | None:
    """Menor escala donde ALGÚN brazo supera TELEPORT_CLEAN teleports (media)."""
    for s in scales:
        for arm in ARMS:
            if (s, arm) in agg and agg[(s, arm)]["teleports"][0] > TELEPORT_CLEAN:
                return s
    return None


def _capacity(agg, scales, onset) -> tuple[float, float]:
    """μ̂ = techo de descarga por Schell (per-node) en el régimen limpio (= pico de la
    meseta); s* = ESCALA DE LA RODILLA = menor escala que ya alcanza ≥98% de μ̂ (donde
    la descarga deja de subir), NO el argmax (la meseta es plana y el argmax cae por
    ruido en el extremo)."""
    clean = [s for s in scales if onset is None or s < onset]
    if not clean:
        clean = scales  # sin teleports: todo el rango es limpio
    discharge = {s: agg[(s, "pernode")]["schell_discharge_vph"][0]
                 for s in clean if (s, "pernode") in agg}
    mu_hat = max(discharge.values())
    s_star = min(s for s, v in discharge.items() if v >= 0.98 * mu_hat)
    return mu_hat, s_star


def _theory_floor(agg, scales, lam_corr, mu_hat) -> tuple[list[float], list[float]]:
    """Piso de capacidad = W_libre + (λ−μ̂)·T/(2λ). W_libre = per-node W_RED en la
    escala más chica (referencia ~libre). λ,μ̂ del corredor, T post-warmup."""
    w_anchor = agg[(scales[0], "pernode")]["w_red"][0]
    xs, ys = [], []
    for s in scales:
        lam = lam_corr[s]
        add = max(0.0, (lam - mu_hat) * T_PW / (2.0 * lam)) if lam > 0 else 0.0
        xs.append(s)
        ys.append(w_anchor + add)
    return xs, ys


def _write_summary(agg, scales) -> None:
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scale", "arm"] + [f"{m}_{stat}" for m in NUM_METRICS for stat in ("mean", "sd")]
    with SUMMARY_OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in scales:
            for arm in ARMS:
                if (s, arm) not in agg:
                    continue
                row = {"scale": s, "arm": arm}
                for m in NUM_METRICS:
                    mu, sd = agg[(s, arm)][m]
                    row[f"{m}_mean"] = round(mu, 3)
                    row[f"{m}_sd"] = round(sd, 3)
                w.writerow(row)
    print(f"→ tabla: {SUMMARY_OUT.relative_to(SIM_ROOT)}")


def main() -> int:
    rows = _load()
    agg, scales, lam_corr = _aggregate(rows)
    onset = _teleport_onset(agg, scales)
    mu_hat, s_star = _capacity(agg, scales, onset)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    # ---- Panel A: W_RED vs demanda ----
    for arm in ARMS:
        xs, mu, sd = _series(agg, scales, arm, "w_red")
        st = ARM_STYLE[arm]
        axA.errorbar(xs, mu, yerr=sd, color=st["color"], marker=st["marker"],
                     capsize=3, lw=1.8, label=st["label"])
    tx, ty = _theory_floor(agg, scales, lam_corr, mu_hat)
    axA.plot(tx, ty, color="0.35", ls="--", lw=1.5,
             label="piso teórico de capacidad\n(W_libre + (λ−μ̂)·T/2λ)")
    axA.axvline(1.0, color="0.6", ls=":", lw=1.2)
    axA.text(1.0, axA.get_ylim()[1], " scale=1.0", color="0.4", va="top", fontsize=8)
    if onset is not None:
        axA.axvspan(onset, scales[-1], color="#c44e52", alpha=0.08)
        axA.text(onset, axA.get_ylim()[1], " teleports → optimista",
                 color="#c44e52", va="top", fontsize=8)
    axA.set_ylabel("W_RED puerta-a-puerta (s)")
    axA.set_title("Demora vs demanda — lo arreglable por temporizado vs el piso de capacidad")
    axA.grid(alpha=0.25)
    axA.legend(fontsize=8, loc="upper left")

    # ---- Panel B: throughput / descarga vs demanda ----
    for arm in ARMS:
        xs, mu, sd = _series(agg, scales, arm, "schell_discharge_vph")
        st = ARM_STYLE[arm]
        axB.errorbar(xs, mu, yerr=sd, color=st["color"], marker=st["marker"],
                     capsize=3, lw=1.8, label=f"descarga Schell — {arm}")
        xs2, mu2, _ = _series(agg, scales, arm, "throughput_net")
        axB.plot(xs2, mu2, color=st["color"], marker=st["marker"], ls=":", lw=1.0,
                 alpha=0.6, label=f"throughput red — {arm}")
    lam_xs = scales
    lam_ys = [lam_corr[s] for s in scales]
    axB.plot(lam_xs, lam_ys, color="0.3", ls="--", lw=1.3, label="demanda corredor ofrecida λ_corr")
    axB.axhline(mu_hat, color="green", ls="-.", lw=1.2)
    axB.text(scales[0], mu_hat, f" μ̂≈{mu_hat:.0f} veh/h (s*={s_star:.1f})",
             color="green", va="bottom", fontsize=8)
    if onset is not None:
        axB.axvspan(onset, scales[-1], color="#c44e52", alpha=0.08)

    # teleports en eje derecho (per-node) — la prueba de contaminación.
    axBr = axB.twinx()
    xs_t, mu_t, _ = _series(agg, scales, "pernode", "teleports")
    axBr.bar(xs_t, mu_t, width=0.06, color="#c44e52", alpha=0.35, label="teleports (per-node)")
    axBr.set_ylabel("teleports (per-node, media)", color="#c44e52")
    axBr.tick_params(axis="y", labelcolor="#c44e52")

    axB.set_xlabel("escala de demanda  (1.0 = baseline IE05)")
    axB.set_ylabel("flujo (veh/h)")
    axB.set_title("Throughput vs demanda — el techo de capacidad y su contaminación por teleports")
    axB.grid(alpha=0.25)
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axBr.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    fig.tight_layout()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=150)
    fig.savefig(SVG_OUT)
    print(f"→ figura: {PNG_OUT.relative_to(SIM_ROOT)}  (+ .svg)")
    print(f"  μ̂≈{mu_hat:.0f} veh/h en s*={s_star:.1f}; "
          f"onset de teleports={'(ninguno)' if onset is None else onset}")

    _write_summary(agg, scales)
    return 0


if __name__ == "__main__":
    sys.exit(main())
