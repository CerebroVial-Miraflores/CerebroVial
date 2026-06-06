"""Orquestador multi-seed PAREADO del experimento fijo-vs-adaptativo sobre Miraflores
completo, brazo adaptativo sobre los 54 TLS de 2 fases (F5, DHU-030).

Por cada seed corre AMBOS brazos con la MISMA demanda (los runners comparten el .rou.xml
regenerado en runs/_demand_seed<NNN> → comparación pareada con demanda idéntica, única
diferencia = el control) y persiste el resultado de ESE seed inmediatamente al CSV
consolidado. ROBUSTO A CORTES (~horas de cómputo): si se interrumpe, retoma salteando los
seeds ya completos (fila en el CSV) y reusa los metrics.json de brazos ya corridos.

Análisis pareado estilo IE05 (misma metodología que la validación de Max Pressure del
corredor): Δ pareado por seed, IC 95% (t, df=9), Wilcoxon signed-rank exacto. Helpers
``_paired_t_ci`` / ``_wilcoxon_signed_rank_exact`` REUSADOS de ``sweep_seeds_downstream.py``.

Reusa los runners de F2/F3b (``run_fijo_red.sh``, ``run_adaptivo_red.sh``) y la métrica de
``collect_red_metrics.py``. El motor se asegura UP una vez al arranque (no se bouncea entre
corridas; los runners adaptativos lo ven vivo y lo dejan como está).

Uso:
    python run_multiseed_red.py --seeds 42                 # SMOKE de 1 seed
    python run_multiseed_red.py --seeds 42-51              # los 10 (default); retoma si cortó
    python run_multiseed_red.py --analyze-only             # solo re-imprime el análisis del CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
REPO_ROOT = SIM_ROOT.parent
EXP_DIR = SIM_ROOT / "conf" / "miraflores_red_completa"
RUNS = EXP_DIR / "runs"
HEALTH = "http://localhost:8001/control/health"
FIJO_SH = HERE / "run_fijo_red.sh"
ADAP_SH = HERE / "run_adaptivo_red.sh"
DEFAULT_CSV = EXP_DIR / "multiseed_54tls_resultado.csv"

sys.path.insert(0, str(HERE))
from sweep_seeds_downstream import (  # noqa: E402  (REUSO metodología IE05)
    T_CRIT_DF9,
    _paired_t_ci,
    _wilcoxon_signed_rank_exact,
)

CSV_FIELDS = [
    "seed", "fijo_timeloss", "adap_timeloss", "delta_fijo_menos_adap", "rd_pct",
    "fijo_neverins", "adap_neverins", "fijo_teleports", "adap_teleports",
    "fijo_completed", "adap_completed", "adap_webster", "adap_maxpressure", "adap_calls",
]


def _seedp(seed: int) -> str:
    return f"{seed:03d}"


def _fijo_dir(seed: int) -> Path:
    return RUNS / f"fijo_seed{_seedp(seed)}"


def _adap_dir(seed: int) -> Path:
    return RUNS / f"adaptivo_seed{_seedp(seed)}_end86400"


def _metrics(out_dir: Path):
    p = out_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text())
        return m if "net_timeloss_mean_s" in m else None
    except (json.JSONDecodeError, OSError):
        return None


def _modes(adap_dir: Path):
    """(webster, max_pressure, total_calls) sumados sobre los nodos del adaptive_stats."""
    p = adap_dir / "adaptive_stats.json"
    if not p.exists():
        return 0, 0, 0
    d = json.loads(p.read_text())
    modes = [m for s in d.values() for m in s.get("modes", [])]
    web = sum(1 for m in modes if m == "webster")
    mp = sum(1 for m in modes if m == "max_pressure")
    return web, mp, len(modes)


def ensure_motor() -> None:
    """Asegura el motor UP en :8001 (idempotente). No lo bouncea entre corridas."""
    try:
        if urllib.request.urlopen(HEALTH, timeout=4).status == 200:
            print(">> motor ya está arriba en :8001")
            return
    except Exception:  # noqa: BLE001
        pass
    print(">> levantando el motor (invoke up --service=core_management_api)")
    subprocess.run(["invoke", "up", "--service=core_management_api"], cwd=str(REPO_ROOT), check=True)
    for i in range(90):
        try:
            if urllib.request.urlopen(HEALTH, timeout=3).status == 200:
                print(f"   motor OK (intento {i + 1})")
                return
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise RuntimeError("el motor no respondió en ~3 min")


def _run_arm(sh: Path, seed: int, out_dir: Path) -> dict:
    """Corre el brazo si su metrics.json no existe (reuso/idempotencia); devuelve metrics."""
    m = _metrics(out_dir)
    if m is not None:
        print(f"   [{sh.name}] seed {seed}: reuso metrics.json existente")
        return m
    print(f"   [{sh.name}] seed {seed}: corriendo...")
    subprocess.run(["bash", str(sh), str(seed)], check=True)
    m = _metrics(out_dir)
    if m is None:
        raise RuntimeError(f"{sh.name} seed {seed} no produjo metrics.json válido")
    return m


def _row_for_seed(seed: int) -> dict:
    fijo = _run_arm(FIJO_SH, seed, _fijo_dir(seed))
    adap = _run_arm(ADAP_SH, seed, _adap_dir(seed))
    web, mp, calls = _modes(_adap_dir(seed))
    f, a = fijo["net_timeloss_mean_s"], adap["net_timeloss_mean_s"]
    delta = round(f - a, 4)                  # >0 = adaptativo ahorra demora
    return {
        "seed": seed, "fijo_timeloss": f, "adap_timeloss": a,
        "delta_fijo_menos_adap": delta, "rd_pct": round(delta / f * 100, 4) if f else 0.0,
        "fijo_neverins": fijo["never_inserted"], "adap_neverins": adap["never_inserted"],
        "fijo_teleports": fijo["teleports_total"], "adap_teleports": adap["teleports_total"],
        "fijo_completed": fijo["completed_n"], "adap_completed": adap["completed_n"],
        "adap_webster": web, "adap_maxpressure": mp, "adap_calls": calls,
    }


def _read_csv(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as fh:
        return {int(r["seed"]): r for r in csv.DictReader(fh)}


def _append_csv(csv_path: Path, rows: dict) -> None:
    """Reescribe el CSV ordenado por seed (write-then-flush por seed completado)."""
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for s in sorted(rows):
            w.writerow(rows[s])
        fh.flush()


def collect(seeds: list[int], csv_path: Path) -> dict:
    ensure_motor()
    rows = _read_csv(csv_path)
    for seed in seeds:
        if seed in rows:
            print(f">> seed {seed}: ya en el CSV — salteado (resume)")
            continue
        print(f">> seed {seed}: procesando ambos brazos")
        rows[seed] = _row_for_seed(seed)
        _append_csv(csv_path, rows)          # persistir ESTE seed inmediatamente
        print(f"   seed {seed} persistido: RD% {float(rows[seed]['rd_pct']):+.2f}%")
    return rows


def analyze(rows: dict, json_path: Path | None) -> dict:
    seeds = sorted(rows)
    R = [rows[s] for s in seeds]
    deltas = [float(r["delta_fijo_menos_adap"]) for r in R]   # >0 = adaptativo mejor
    rds = [float(r["rd_pct"]) for r in R]
    W = 92
    print("\n" + "=" * W)
    print(f"F5 — ADAPTATIVO vs FIJO sobre los 54 TLS de 2 fases | PAREADO | n={len(seeds)} seeds")
    print("=" * W)
    print(f"{'seed':>5} | {'fijo_TL':>9} {'adap_TL':>9} | {'Δ(f-a)':>8} {'RD%':>7} | "
          f"{'tel f→a':>9} {'never f→a':>10} | {'web/mp(adap)':>13}")
    for r in R:
        print(f"{r['seed']:>5} | {float(r['fijo_timeloss']):>9.3f} {float(r['adap_timeloss']):>9.3f} | "
              f"{float(r['delta_fijo_menos_adap']):>8.3f} {float(r['rd_pct']):>+6.2f}% | "
              f"{int(r['fijo_teleports']):>3}→{int(r['adap_teleports']):<3} "
              f"{int(r['fijo_neverins']):>3}→{int(r['adap_neverins']):<3}      | "
              f"{int(r['adap_webster'])}/{int(r['adap_maxpressure'])}")
    print("-" * W)
    if len(seeds) < 2:
        print("(n<2 → sin análisis pareado; smoke de 1 seed)")
        result = {"seeds": seeds, "rows": rows}
    else:
        d_mean, d_sd, d_se, d_lo, d_hi = _paired_t_ci(deltas)
        w_obs, p_w = _wilcoxon_signed_rank_exact(deltas)
        rd_mean = sum(rds) / len(rds)
        rd_sd = (sum((x - rd_mean) ** 2 for x in rds) / (len(rds) - 1)) ** 0.5
        pos = sum(1 for x in deltas if x > 0)
        crosses = d_lo <= 0 <= d_hi
        tot_web = sum(int(r["adap_webster"]) for r in R)
        tot_mp = sum(int(r["adap_maxpressure"]) for r in R)
        print(f"Δ pareado (fijo−adap, s): media {d_mean:+.3f} ± SD {d_sd:.3f} (SE {d_se:.3f})  "
              f"IC95% [{d_lo:+.3f}, {d_hi:+.3f}]")
        print(f"  ¿IC cruza 0?  {'SÍ → no concluyente' if crosses else 'NO → efecto REAL'}")
        print(f"RD% (reducción de demora): media {rd_mean:+.2f}% ± SD {rd_sd:.2f}  "
              f"(umbral DHU-030 ≥15%)")
        print(f"Wilcoxon signed-rank exacto: W+={w_obs:.1f}  p(two-sided)={p_w:.4f}")
        print(f"seeds con adaptativo MEJOR (Δ>0): {pos}/{len(seeds)}")
        print(f"modos adaptativos AGREGADOS (10 seeds): webster={tot_web}  max_pressure={tot_mp}  "
              f"(mp = {tot_mp/(tot_web+tot_mp)*100:.3f}% de las decisiones)")
        # salud agregada
        print("salud: censura (never-inserted) fijo Σ={} adap Σ={} | teleports fijo Σ={} adap Σ={}".format(
            sum(int(r["fijo_neverins"]) for r in R), sum(int(r["adap_neverins"]) for r in R),
            sum(int(r["fijo_teleports"]) for r in R), sum(int(r["adap_teleports"]) for r in R)))
        result = {
            "n": len(seeds), "seeds": seeds,
            "delta_mean_s": round(d_mean, 4), "delta_sd_s": round(d_sd, 4),
            "delta_ci95": [round(d_lo, 4), round(d_hi, 4)], "ci_excludes_zero": not crosses,
            "rd_pct_mean": round(rd_mean, 4), "rd_pct_sd": round(rd_sd, 4),
            "wilcoxon_Wplus": w_obs, "wilcoxon_p_two_sided": round(p_w, 6),
            "seeds_positive": pos, "modes_total": {"webster": tot_web, "max_pressure": tot_mp},
            "t_crit_df9": T_CRIT_DF9, "rows": rows,
        }
    print("=" * W)
    if json_path:
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"JSON consolidado → {json_path}")
    return result


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Orquestador multi-seed pareado red-completa (F5).")
    ap.add_argument("--seeds", default="42-51", help="rango 'a-b' o lista 'a,b,c' (default 42-51)")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--json", type=Path, default=DEFAULT_CSV.with_suffix(".json"))
    ap.add_argument("--analyze-only", action="store_true",
                    help="no corre nada; re-imprime el análisis del CSV existente")
    args = ap.parse_args(argv)

    if args.analyze_only:
        rows = _read_csv(args.csv)
        if not rows:
            print(f"CSV vacío o inexistente: {args.csv}", file=sys.stderr)
            return 1
        analyze(rows, args.json)
        return 0

    seeds = _parse_seeds(args.seeds)
    rows = collect(seeds, args.csv)
    analyze({s: rows[s] for s in seeds}, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
