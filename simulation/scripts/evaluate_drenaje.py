"""Evaluador de drenaje multi-señal por-día (criterio D-014, v2). B3.2.a.

Juzga si un día simulado del net v2 **drena** (acepta) o **colapsa** (bandera), con las tres
señales de D-014. Reemplaza, para v2, la racha sub-8 km/h v1-específica de ``analyze24.py``
(que queda intacta por compatibilidad con C2). Este módulo NO importa ``analyze24.py``: copia
sus patrones (``iterparse`` streaming sobre edgeData; lectura de ``<teleports>`` y
``<vehicleTripStatistics>`` de stats), porque aquél está acoplado a su layout por-scale y a
buckets horarios, incompatibles con el criterio por-día y sub-horario de D-014.

Tres señales (D-014):
  1. **Teleports** ``≤ 50`` — de ``stats.xml`` ``<teleports @total>``. *Señal primaria* (la menos
     contaminada por el ciclo de semáforo; manda ante señales en desacuerdo).
  2. **Δduración media** ``≤ +10 %`` — de ``stats.xml`` ``vehicleTripStatistics @duration``
     (media agregada de duración de viaje). Umbral **absoluto ≤ 280 s** (baseline ~254 s).
  3. **Dip acotado** — la velocidad media de red **no** permanece **< 20 km/h por > 15 min
     consecutivos** en ninguno de los dos picos (AM 07-09h / PM 18-20h). mean_kmh por intervalo de
     60 s ponderado por ``sampledSeconds`` (igual que C3); racha consecutiva de intervalos sub-20.

Regla de combinación (D-014): el día **falla** si ``teleports > 50`` **O** dispara cualquiera de
las otras dos. El veredicto es el AND de las tres; ``teleports`` es la primaria al reportar.

Importante (contrato que verifica el test de Fase 4): el dip se computa **solo sobre las ventanas
de pico** ``[25200,32400)`` (AM) y ``[64800,72000)`` (PM). Cualquier intervalo fuera de esas
ventanas — p. ej. el margen ±30 min de los fixtures C3 — **NO entra al cómputo de racha**. El
evaluador acota a la ventana de pico explícitamente; el margen del fixture existe para testear
justamente eso.

``evaluate_day(stats, edgedata)`` devuelve una **estructura** (dict), no imprime: el test asserta
sobre el retorno y el modo batch tabula sobre él. La presentación (tabla) está separada del cómputo.

Uso (modo batch, B3.2.c):
    python evaluate_drenaje.py [--dataset-dir DIR] [--seeds 42 .. 101] [--no-cleanup]
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]  # scripts/ -> simulation/ -> repo-root
DEFAULT_DATASET_DIR = _REPO / "simulation/data/datasets/miraflores_laborable_60d"
DEFAULT_SEEDS = tuple(range(42, 102))  # 42..101 = 60 días

# --- Umbrales D-014 (net-específicos de v2; una reconstrucción del net los invalida) ---
TELEPORTS_MAX = 50          # señal 1: drena si total ≤ 50
DURATION_MAX_S = 280.0      # señal 2: drena si duración media ≤ 280 s (baseline ~254 s, +10%)
DIP_SPEED_KMH = 20.0        # señal 3: umbral de velocidad de red
DIP_MAX_RUN_MIN = 15        # señal 3: falla si racha sub-20 > 15 min consecutivos (intervalos de 60 s)

# Ventanas de PICO donde se evalúa el dip (begin en s). NO incluyen margen: el cómputo de racha
# se acota a estos rangos exactos, así el margen de los fixtures queda fuera.
DIP_WINDOWS = {"AM": (25200, 32400), "PM": (64800, 72000)}  # 07-09h, 18-20h


def _read_stats(stats_path: Path) -> tuple[int, float]:
    """(teleports_total, duración_media_s) de stats.xml. Patrón de ``analyze24.totals``."""
    root = ET.parse(str(stats_path)).getroot()
    tel = root.find("teleports")
    vts = root.find("vehicleTripStatistics")
    if tel is None or vts is None:
        raise ValueError(f"{stats_path}: falta <teleports> o <vehicleTripStatistics>")
    return int(tel.get("total")), float(vts.get("duration"))


def _network_mean_kmh_in_peaks(edgedata_path: Path) -> dict[str, list[tuple[int, float | None]]]:
    """Por cada intervalo cuyo ``begin`` cae en una ventana de PICO, media de velocidad de red
    (km/h) ponderada por ``sampledSeconds`` sobre edges con ``sampledSeconds>0``. Los intervalos
    fuera de las ventanas (margen del fixture, resto del día) se **ignoran**. Streaming (el
    edgeData es voluminoso). Devuelve ``{win: [(begin, mean_kmh|None), ...]}`` ordenado por begin.
    """
    def _window_of(begin: int) -> str | None:
        for name, (lo, hi) in DIP_WINDOWS.items():
            if lo <= begin < hi:
                return name
        return None

    series: dict[str, list[tuple[int, float | None]]] = {name: [] for name in DIP_WINDOWS}
    cur_win: str | None = None
    cur_begin = 0
    num = den = 0.0
    for ev, el in ET.iterparse(str(edgedata_path), events=("start", "end")):
        if ev == "start" and el.tag == "interval":
            cur_begin = int(float(el.get("begin")))
            cur_win = _window_of(cur_begin)
            num = den = 0.0
        elif ev == "end" and el.tag == "edge":
            if cur_win is not None:
                ss = float(el.get("sampledSeconds", "0") or 0)
                sp = el.get("speed")
                if ss > 0.0 and sp is not None:
                    num += ss * float(sp)
                    den += ss
            el.clear()
        elif ev == "end" and el.tag == "interval":
            if cur_win is not None:
                series[cur_win].append((cur_begin, (3.6 * num / den) if den > 0 else None))
            el.clear()
            cur_win = None
    for name in series:
        series[name].sort()
    return series


def _max_sub20_run(window_series: list[tuple[int, float | None]]) -> int:
    """Racha máxima de intervalos consecutivos con mean_kmh < umbral. Un intervalo sin tráfico
    (mean_kmh None) NO cuenta como sub-20 y corta la racha. Asume intervalos contiguos de 60 s
    (SUMO emite todos), así posiciones consecutivas en la serie = minutos consecutivos."""
    run = mx = 0
    for _begin, mean_kmh in window_series:
        if mean_kmh is not None and mean_kmh < DIP_SPEED_KMH:
            run += 1
            mx = max(mx, run)
        else:
            run = 0
    return mx


def _min_mean_kmh(window_series: list[tuple[int, float | None]]) -> float | None:
    vals = [m for _b, m in window_series if m is not None]
    return min(vals) if vals else None


def evaluate_day(stats_path: Path | str, edgedata_path: Path | str) -> dict:
    """Evalúa un día con las tres señales de D-014. Devuelve una **estructura** (no imprime):

    ``{verdict, drains, signals:{teleports, duration_s, dip:{windows:{AM,PM}}}, reasons}``
    con los números intermedios del dip (racha máxima sub-20 + mean_kmh mínimo por ventana).
    """
    stats_path = Path(stats_path)
    edgedata_path = Path(edgedata_path)

    teleports, duration_s = _read_stats(stats_path)
    series = _network_mean_kmh_in_peaks(edgedata_path)

    # Insumo presente: si una ventana de pico no tiene intervalos, el input es anómalo (archivo
    # equivocado / sin datos de pico). No pasar el dip en silencio sobre una ventana vacía.
    empty = [w for w, s in series.items() if not s]
    if empty:
        raise ValueError(
            f"{edgedata_path.name}: ventanas de pico sin intervalos {empty} "
            f"(¿edgeData incompleto o ventana mal definida?)"
        )

    dip_windows: dict[str, dict] = {}
    for name, (lo, hi) in DIP_WINDOWS.items():
        run = _max_sub20_run(series[name])
        dip_windows[name] = {
            "begin": lo,
            "end": hi,
            "max_sub20_run_min": run,
            "min_mean_kmh": _min_mean_kmh(series[name]),
        }

    teleports_ok = teleports <= TELEPORTS_MAX
    duration_ok = duration_s <= DURATION_MAX_S
    dip_ok = all(w["max_sub20_run_min"] <= DIP_MAX_RUN_MIN for w in dip_windows.values())
    drains = teleports_ok and duration_ok and dip_ok

    reasons: list[str] = []
    if not teleports_ok:
        reasons.append(f"teleports {teleports} > {TELEPORTS_MAX} (señal primaria)")
    if not duration_ok:
        reasons.append(f"duración media {duration_s:.1f}s > {DURATION_MAX_S:.0f}s")
    if not dip_ok:
        parts = [
            f"{name} racha sub-20 {w['max_sub20_run_min']}min > {DIP_MAX_RUN_MIN}"
            for name, w in dip_windows.items()
            if w["max_sub20_run_min"] > DIP_MAX_RUN_MIN
        ]
        reasons.append("dip: " + "; ".join(parts))
    if drains:
        reasons.append("las tres señales pasan")

    return {
        "verdict": "drena" if drains else "colapsa",
        "drains": drains,
        "signals": {
            "teleports": {"value": teleports, "threshold": TELEPORTS_MAX, "ok": teleports_ok},
            "duration_s": {"value": duration_s, "threshold": DURATION_MAX_S, "ok": duration_ok},
            "dip": {
                "ok": dip_ok,
                "threshold_kmh": DIP_SPEED_KMH,
                "max_run_min": DIP_MAX_RUN_MIN,
                "windows": dip_windows,
            },
        },
        "reasons": reasons,
    }


def evaluate_dataset(
    dataset_dir: Path | str,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    *,
    cleanup: bool = True,
) -> list[dict]:
    """Modo batch (B3.2.c): evalúa cada ``day_seedNNN`` del dir, consumiendo
    ``stats_seedNNN.xml`` + ``edgedata_seedNNN.xml``. Con ``cleanup=True`` **borra cada
    ``edgedata_seedNNN.xml`` tras consumirlo** (limpieza incremental → pico de disco ~1 edgeData,
    no los ~25 GB de los 60 simultáneos). Robusto a días anómalos: un día se marca y el batch sigue,
    no se absorbe en silencio (D-014 §492):
      - ``"FALTA"``: faltan los insumos (stats o edgedata ausentes).
      - ``"ERROR"``: insumo presente pero anómalo (ventana de pico vacía, XML corrupto, atributo
        malformado). Su ``edgedata`` NO se borra — se conserva como evidencia para inspección.
    Devuelve la lista de resultados (cada uno + ``seed``)."""
    dataset_dir = Path(dataset_dir)
    results: list[dict] = []
    for seed in seeds:
        stats = dataset_dir / f"stats_seed{seed:03d}.xml"
        edge = dataset_dir / f"edgedata_seed{seed:03d}.xml"
        if not stats.exists() or not edge.exists():
            results.append({"seed": seed, "verdict": "FALTA", "drains": False,
                            "reasons": [f"faltan insumos: stats={stats.exists()} edge={edge.exists()}"]})
            continue
        try:
            res = evaluate_day(stats, edge)
        except Exception as exc:
            # Insumo presente pero anómalo: bandera "ERROR" (distinto de "FALTA"), el batch sigue.
            # NO se borra el edgedata del día que erró: se conserva como evidencia del problema.
            results.append({"seed": seed, "verdict": "ERROR", "drains": False,
                            "reasons": [f"evaluación falló: {exc}"]})
            continue
        res["seed"] = seed
        results.append(res)
        if cleanup:
            edge.unlink()  # efímero: solo se limpia tras una evaluación exitosa
    return results


def _format_table(results: list[dict]) -> str:
    """Presentación (separada del cómputo): tabla por-seed + resumen de banderas."""
    hdr = (f"{'seed':>4} | {'tel':>4} | {'dur_s':>6} | {'AM run/min':>10} | "
           f"{'PM run/min':>10} | veredicto")
    lines = [hdr, "-" * len(hdr)]
    flagged = []
    for r in results:
        if "signals" not in r:  # FALTA o ERROR: sin cómputo, fila placeholder
            lines.append(
                f"{r['seed']:>4} | {'—':>4} | {'—':>6} | {'—':>10} | {'—':>10} | {r['verdict']}"
            )
            flagged.append(r["seed"])
            continue
        s = r["signals"]
        am = s["dip"]["windows"]["AM"]
        pm = s["dip"]["windows"]["PM"]
        am_min = f"{am['min_mean_kmh']:.1f}" if am["min_mean_kmh"] is not None else "—"
        pm_min = f"{pm['min_mean_kmh']:.1f}" if pm["min_mean_kmh"] is not None else "—"
        lines.append(
            f"{r['seed']:>4} | {s['teleports']['value']:>4} | {s['duration_s']['value']:>6.1f} | "
            f"{am['max_sub20_run_min']:>3}/{am_min:>6} | {pm['max_sub20_run_min']:>3}/{pm_min:>6} | "
            f"{r['verdict']}"
        )
        if not r["drains"]:
            flagged.append(r["seed"])
    n = len([r for r in results if "signals" in r])
    drained = len([r for r in results if r.get("drains")])
    lines.append("-" * len(hdr))
    lines.append(f"drenan {drained}/{n}.  Banderas (no-drena/falta): "
                 f"{flagged if flagged else 'ninguna'}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluador de drenaje multi-señal por-día (D-014).")
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR),
                    help="dir con day_seedNNN + stats_seedNNN.xml + edgedata_seedNNN.xml")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                    help="seeds a evaluar (default: 42..101)")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="NO borrar los edgedata_seedNNN.xml tras evaluar (default: limpieza incremental)")
    args = ap.parse_args()
    results = evaluate_dataset(
        args.dataset_dir, tuple(args.seeds), cleanup=not args.no_cleanup
    )
    print(_format_table(results))


if __name__ == "__main__":
    main()
