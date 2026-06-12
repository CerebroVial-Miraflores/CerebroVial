"""Wrapper de siembra del calendario de congestión (Fase 2.5, HU-23).

Itera un mapeo determinista fecha→seed e invoca el sembrador existente
(``scripts/replay_congestion.py``) UNA VEZ POR DÍA, en modo presiembra. NO extiende
ni modifica el sembrador: es un loop alrededor del CLI. Cada subproceso ya hace
``preseed`` + ``populate_geom_from_edges`` + commit por su cuenta.

El calendario son 10 días (1 jun → 10 jun 2026, inclusive) — el histórico observado de
la semana de la demo + el día vivo (lunes 8) + dos días extra (mar 9, mié 10). Decisión
de alcance: 10 días ≈ 24M filas ≈ ~16 GB en disco (manejable en BD local), histórico =
visualización de observado. Findes (6–7 jun) con placeholder laborable (no hay perfil
finde calibrado; deuda registrada). Lunes 8 = seed051 (día vivo + día de predicción de
Fase 3; seed051 ∈ TEST/no-visto) — excepción fija fuera del secuencial. Los días 9–10 se
añadieron como extensión a 10 días con los siguientes seeds libres (049, 050).

──────────────────────────────────────────────────────────────────────────────
COMANDO EXACTO (lo dispara Cesar, NO se lanza solo):

    # Vista previa (NO toca la BD) — imprime el plan + estimación:
    .venv/bin/python scripts/seed_congestion_calendar.py

    # Siembra la semana completa (8 días) — requiere --execute explícito:
    .venv/bin/python scripts/seed_congestion_calendar.py --execute

    # Siembra un solo día (usado en el stage-gate):
    .venv/bin/python scripts/seed_congestion_calendar.py --execute --only 2026-06-01

ESTIMACIÓN (base: 1 día = 2,390,400 filas = ~1.6 GB medido en la BD actual):
    • Filas:  2,390,400/día × 10 ≈ 24M filas.
    • Disco:  ~1.6 GB/día × 10 ≈ ~16 GB (hypertable TimescaleDB, tabla + índices).
    • Tiempo: ~<T>/día × 10  (T = el medido en el stage-gate de un día).
──────────────────────────────────────────────────────────────────────────────

Idempotente: re-correr un día ya sembrado es no-op limpio (la fecha desambigua la PK
``(event_uuid, snapshot_timestamp)`` y el ``ON CONFLICT DO NOTHING`` del sembrador
descarta los duplicados; el UPDATE-join de geom no toca filas ya pobladas).

Logs (en ``logs/seed_calendar/``, gitignored):
    • <fecha>_<seed>.log  — stdout/stderr completo del subproceso de ese día.
    • master.log          — una línea por día: inicio, fin, duración, fecha, seed,
                            filas, geom, count total, estado (OK/FALLÓ).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SEEDER = _REPO / "scripts" / "replay_congestion.py"
_LOG_DIR = _REPO / "logs" / "seed_calendar"

# ── Mapeo fecha→seed (explícito, trazable; NO calculado mágicamente) ───────────
# Procedencia: secuencial desde seed042 por orden de calendario (1–7 jun → 042–048),
# con seed051 RESERVADO para el 8-jun (día vivo, excepción fija fuera del secuencial).
# 8 fechas, seeds = {042..048, 051} = 8 distintos, sin repetición, todos en disco.
CALENDAR: dict[str, str] = {
    "2026-06-01": "seed042",  # lunes
    "2026-06-02": "seed043",  # martes
    "2026-06-03": "seed044",  # miércoles
    "2026-06-04": "seed045",  # jueves
    "2026-06-05": "seed046",  # viernes
    "2026-06-06": "seed047",  # sábado  (placeholder laborable)
    "2026-06-07": "seed048",  # domingo (placeholder laborable)
    "2026-06-08": "seed051",  # lunes   (día vivo + predicción Fase 3 — excepción fija)
    "2026-06-09": "seed049",  # martes  (extensión a 10 días; siguiente seed libre)
    "2026-06-10": "seed050",  # miércoles (extensión a 10 días; siguiente seed libre)
}


def _validate_calendar() -> None:
    """Falla ruidosamente si el mapeo tiene fechas inválidas o seeds repetidos."""
    seeds = list(CALENDAR.values())
    if len(set(seeds)) != len(seeds):
        dupes = sorted({s for s in seeds if seeds.count(s) > 1})
        raise SystemExit(f"Mapeo inválido: seeds repetidos {dupes}")
    for d in CALENDAR:
        datetime.strptime(d, "%Y-%m-%d")  # valida formato/calendario


def _print_plan() -> None:
    print("Calendario de siembra (fecha → seed):")
    for d, s in CALENDAR.items():
        print(f"  {d}  →  {s}")
    n = len(CALENDAR)
    print(f"\n{n} días, {len(set(CALENDAR.values()))} seeds distintos (sin repetición).")
    print(f"Estimación: 2,390,400 filas/día × {n} ≈ {2_390_400 * n / 1e6:.0f}M filas; "
          f"~1.6 GB/día × {n} ≈ ~{1.6 * n:.0f} GB en disco.")


_RE_ROWS = re.compile(r"'rows_processed':\s*(\d+)")
_RE_GEOM = re.compile(r"'geom_updated':\s*(\d+)")
_RE_COUNT = re.compile(r"count\(waze_jams\)\s*=\s*(\d+)")


def _seed_one(date_str: str, seed: str, master) -> bool:
    """Siembra un día vía subproceso del sembrador. Devuelve True si OK."""
    day_log = _LOG_DIR / f"{date_str}_{seed}.log"
    cmd = [sys.executable, str(_SEEDER),
           "--mode", "presiembra", "--date", date_str, "--day", seed]
    t0 = datetime.now()
    print(f"[{t0:%H:%M:%S}] sembrando {date_str} → {seed} …", flush=True)
    proc = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True)
    t1 = datetime.now()
    out = (proc.stdout or "") + (proc.stderr or "")
    day_log.write_text(out)

    rows = (m.group(1) if (m := _RE_ROWS.search(out)) else "?")
    geom = (m.group(1) if (m := _RE_GEOM.search(out)) else "?")
    total = (m.group(1) if (m := _RE_COUNT.search(out)) else "?")
    ok = proc.returncode == 0
    status = "OK" if ok else f"FALLÓ(rc={proc.returncode})"
    dur = (t1 - t0).total_seconds()

    line = (f"{t0:%Y-%m-%d %H:%M:%S}\t{t1:%H:%M:%S}\tdur={dur:.1f}s\t"
            f"{date_str}\t{seed}\trows={rows}\tgeom={geom}\tcount={total}\t{status}\n")
    master.write(line)
    master.flush()
    print(f"[{t1:%H:%M:%S}] {date_str} → {seed}: {status} "
          f"(dur={dur:.1f}s, rows={rows}, geom={geom}, count={total})", flush=True)
    if not ok:
        print(f"  ↳ ver {day_log} para el detalle del error", flush=True)
    return ok


def main() -> None:
    _validate_calendar()
    ap = argparse.ArgumentParser(
        description="Wrapper de siembra del calendario de congestión (Fase 2.5, HU-23).")
    ap.add_argument("--execute", action="store_true",
                    help="siembra de verdad. Sin esto, solo imprime el plan (dry-run).")
    ap.add_argument("--only", default=None, metavar="YYYY-MM-DD",
                    help="siembra un solo día del calendario (stage-gate).")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="siembra solo los primeros N días del calendario.")
    args = ap.parse_args()

    items = list(CALENDAR.items())
    if args.only:
        if args.only not in CALENDAR:
            raise SystemExit(f"--only {args.only} no está en el calendario. "
                             f"Fechas válidas: {', '.join(CALENDAR)}")
        items = [(args.only, CALENDAR[args.only])]
    elif args.limit is not None:
        items = items[:args.limit]

    if not args.execute:
        _print_plan()
        sel = (f" (filtrado: {len(items)} día/s)" if args.only or args.limit is not None else "")
        print(f"\n[DRY-RUN]{sel} — NO se tocó la BD. Agregá --execute para sembrar.")
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    master_path = _LOG_DIR / "master.log"
    run_t0 = datetime.now()
    print(f"=== Siembra: {len(items)} día/s — inicio {run_t0:%Y-%m-%d %H:%M:%S} ===", flush=True)
    failures = []
    with open(master_path, "a") as master:
        master.write(f"# === corrida {run_t0:%Y-%m-%d %H:%M:%S} — {len(items)} día/s ===\n")
        for date_str, seed in items:
            if not _seed_one(date_str, seed, master):
                failures.append(date_str)
    run_t1 = datetime.now()
    dur = (run_t1 - run_t0).total_seconds()
    print(f"=== Fin {run_t1:%H:%M:%S} — duración {dur:.1f}s, "
          f"{len(items) - len(failures)}/{len(items)} OK ===", flush=True)
    if failures:
        print(f"FALLARON: {', '.join(failures)} — ver logs en {_LOG_DIR}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
