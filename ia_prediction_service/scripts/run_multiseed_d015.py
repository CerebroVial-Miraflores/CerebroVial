"""Orquestador multi-seed D-015/D-011 — robustece el veredicto GRU baseline vs STGNN.

Corre seeds 1..4 de AMBOS modelos (seed 0 ya está hecho) de forma SECUENCIAL (una a la
vez — el STGNN es CPU-bound y saturaría la máquina, y queremos logs limpios), persistiendo
cada resultado APENAS termina, idempotente/reanudable, con log por seed + log maestro con
timestamps, y al final consolida las métricas de las 5+5 corridas en un CSV/JSON agregable
(las MISMAS métricas que abrió D-015 — sin métricas nuevas).

NO se lanza solo. Cesar lo dispara explícitamente (desde la raíz del repo):

    cd /Users/rasec/Tesis/CerebroVial
    nohup ia_prediction_service/.venv/bin/python \
        ia_prediction_service/scripts/run_multiseed_d015.py \
        > ia_prediction_service/scripts/multiseed/launch.out 2>&1 &

Duración estimada: ~10 h (8 corridas CPU-bound secuenciales). Reanudable: si se interrumpe,
relanzar el MISMO comando — saltea lo ya completado ("seed N ya completada, salteando").

Sólo reconsolidar métricas existentes (no entrena):
    ia_prediction_service/.venv/bin/python \
        ia_prediction_service/scripts/run_multiseed_d015.py --consolidate-only

Previsualizar sin entrenar:
    ia_prediction_service/.venv/bin/python \
        ia_prediction_service/scripts/run_multiseed_d015.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYBIN = REPO_ROOT / "ia_prediction_service" / ".venv" / "bin" / "python"
SCRIPTS_DIR = REPO_ROOT / "ia_prediction_service" / "scripts"
MODELS_DIR = REPO_ROOT / "ia_prediction_service" / "models"
MULTISEED_DIR = SCRIPTS_DIR / "multiseed"
LOGS_DIR = MULTISEED_DIR / "logs"
STGNN_OUT_DIR = MULTISEED_DIR / "stgnn"

NEW_SEEDS = [1, 2, 3, 4]          # la 0 ya está hecha en ambos modelos
ALL_SEEDS = [0, 1, 2, 3, 4]       # para la consolidación (incluye la canónica)

# Mismas 4 particiones × 3 métricas que abrió D-015. No se agregan métricas nuevas.
REGIMES = ["test_all", "test_severe_dia", "test_severe_pico", "test_normal"]
METRICS = ["MAE", "RMSE", "R2"]


def metrics_path_for(model: str, seed: int) -> Path:
    """JSON de métricas de una (modelo, seed). seed 0 = ruta canónica existente."""
    if model == "baseline":
        stem = "miraflores_baseline_metrics" + ("" if seed == 0 else f"_seed{seed}")
        return SCRIPTS_DIR / f"{stem}.json"
    stem = "miraflores_stgnn_metrics" + ("" if seed == 0 else f"_seed{seed}")
    base = SCRIPTS_DIR if seed == 0 else STGNN_OUT_DIR
    return base / f"{stem}.json"


def ckpt_path_for(model: str, seed: int) -> Path:
    if model == "baseline":
        stem = "miraflores_gru_baseline" + ("" if seed == 0 else f"_seed{seed}")
        return MODELS_DIR / f"{stem}.pt"
    stem = "miraflores_stgnn" + ("" if seed == 0 else f"_seed{seed}")
    base = MODELS_DIR if seed == 0 else STGNN_OUT_DIR
    return base / f"{stem}.pt"


def metadata_path_for(model: str, seed: int) -> Path:
    if model == "baseline":
        stem = "miraflores_baseline_metadata" + ("" if seed == 0 else f"_seed{seed}")
        return SCRIPTS_DIR / f"{stem}.json"
    stem = "miraflores_stgnn_metadata" + ("" if seed == 0 else f"_seed{seed}")
    base = SCRIPTS_DIR if seed == 0 else STGNN_OUT_DIR
    return base / f"{stem}.json"


def stale_artifacts(model: str, seed: int) -> list[Path]:
    """Los 3 artefactos (.pt + metadata + metrics) de una (modelo, seed)."""
    return [ckpt_path_for(model, seed), metadata_path_for(model, seed),
            metrics_path_for(model, seed)]


def run_cmd_for(model: str, seed: int) -> list[str]:
    if model == "baseline":
        return [str(PYBIN), str(SCRIPTS_DIR / "train_miraflores_baseline.py"),
                "--seed", str(seed)]
    return [str(PYBIN), str(SCRIPTS_DIR / "train_miraflores_stgnn.py"),
            "--seed", str(seed), "--out-dir", str(STGNN_OUT_DIR)]


def log_master(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    with (LOGS_DIR / "master.log").open("a") as f:
        f.write(line + "\n")


def already_done(model: str, seed: int) -> bool:
    # El JSON de métricas se escribe al FINAL de cada corrida → su presencia (junto al .pt)
    # marca corrida completa. Si murió a mitad, falta el JSON y se re-corre.
    return metrics_path_for(model, seed).exists() and ckpt_path_for(model, seed).exists()


def run_one(model: str, seed: int) -> bool:
    # GUARDA anti-corrupción: run_one sólo se llama para seeds NO completas (main saltea las
    # already_done). Antes de re-correr, borra cualquier artefacto parcial de un intento muerto
    # (.pt/metadata/metrics de ESTA seed) para que el reintento arranque limpio y nunca mezcle un
    # .pt nuevo con un JSON viejo del mismo nombre (lo que daría already_done=True con modelo malo).
    assert seed != 0, "run_one nunca corre la seed 0 canónica"  # NEW_SEEDS no incluye 0
    for p in stale_artifacts(model, seed):
        if p.exists():
            log_master(f"CLEAN  {model} seed={seed} borra artefacto parcial previo: {p.name}")
            p.unlink()
    log_path = LOGS_DIR / f"{model}_seed{seed}.log"
    cmd = run_cmd_for(model, seed)
    log_master(f"START  {model} seed={seed}  -> logs/{log_path.name}")
    with log_path.open("w") as logf:
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0 and already_done(model, seed)
    if ok:
        log_master(f"DONE   {model} seed={seed}  (rc=0, artefactos OK)")
    else:
        log_master(f"FAIL   {model} seed={seed}  (rc={proc.returncode}; ver logs/{log_path.name})")
    return ok


def consolidate() -> None:
    """Aplana las métricas de las 5+5 corridas a formato tidy (modelo×seed×régimen×métrica×horizonte)."""
    rows, missing = [], []
    for model in ("baseline", "stgnn"):
        for seed in ALL_SEEDS:
            mp = metrics_path_for(model, seed)
            if not mp.exists():
                missing.append((model, seed, mp))
                continue
            payload = json.loads(mp.read_text())
            for regime in REGIMES:
                block = payload.get(regime, {})
                for metric in METRICS:
                    for hstr, val in block.get(metric, {}).items():
                        rows.append({"model": model, "seed": seed, "regime": regime,
                                     "metric": metric, "horizon": int(hstr), "value": val})
    csv_path = MULTISEED_DIR / "consolidated_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", "regime", "metric", "horizon", "value"])
        w.writeheader()
        w.writerows(rows)
    (MULTISEED_DIR / "consolidated_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    log_master(f"CONSOLIDATE  {len(rows)} filas -> multiseed/consolidated_metrics.{{csv,json}}")
    for model, seed, mp in missing:
        log_master(f"WARN   métricas faltantes: {model} seed={seed} ({mp.name}) — fila omitida, NO inventada")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orquestador multi-seed D-015 (GRU baseline + STGNN)")
    ap.add_argument("--consolidate-only", action="store_true",
                    help="No entrena; sólo reconsolida las métricas que ya existan.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Muestra qué correría/saltearía sin entrenar.")
    args = ap.parse_args()

    for d in (MULTISEED_DIR, LOGS_DIR, STGNN_OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if args.consolidate_only:
        consolidate()
        return

    # El check del intérprete sólo aplica a la corrida real: --dry-run previsualiza sin tocar nada
    # y no debe morir si el venv no estuviera presente.
    if not args.dry_run and not PYBIN.exists():
        raise SystemExit(f"No existe el intérprete del venv de training: {PYBIN}")

    log_master(f"==== multiseed D-015 START (seeds {NEW_SEEDS} × baseline,stgnn) ====")
    plan = [(m, s) for m in ("baseline", "stgnn") for s in NEW_SEEDS]
    for model, seed in plan:
        if already_done(model, seed):
            log_master(f"SKIP   {model} seed={seed} ya completada, salteando")
            continue
        if args.dry_run:
            log_master(f"DRYRUN {model} seed={seed} -> {' '.join(run_cmd_for(model, seed))}")
            continue
        run_one(model, seed)   # persiste apenas termina; si una falla, sigue con la próxima

    if not args.dry_run:
        consolidate()
    log_master("==== multiseed D-015 END ====")


if __name__ == "__main__":
    main()
