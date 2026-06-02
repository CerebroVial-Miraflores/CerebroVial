"""TTH-09 Fase 2b — barrido de estrategias de class weights (material CT-09.6 / tesis).

La eval full mostró que el inverso de frecuencia puro SOBRE-CORRIGE bajo desbalance
extremo: en N/S la accuracy cayó por debajo del majority baseline porque el modelo
abandona la mayoritaria j1 para perseguir las raras (peso j3 ~27×). Este barrido
entrena+evalúa 5 estrategias de peso para encontrar la calibración correcta y
documentar el trade-off.

Aislamiento de la variable: SOLO se barre la estrategia de peso. ``epochs=12`` fijo,
todo el resto de hiperparámetros idéntico al entrenamiento real (se reutiliza
``train_direction`` y ``build_checkpoint`` de ``tth09_train``, y ``evaluate`` del
evaluador — cero duplicación de entrenamiento ni de métricas).

Para cada estrategia entrena los 4 modelos de cero (determinista, seed 0), los guarda
en un dir temporal por estrategia (``/tmp/tth09_weight_sweep/<estrategia>/``) para NO
pisar los ``models/gru_<dir>.pt`` actuales, evalúa sobre valid y consolida las métricas
en un único JSON comparable.

LIMITACIÓN METODOLÓGICA (D-005, honestidad): la estrategia ganadora se selecciona
sobre la partición de validación. No hay test set held-out (el dataset solo tiene
train/valid), así que el F1 del ganador puede ser ligeramente OPTIMISTA por selección
de modelo. Queda registrado en ``meta.selection_note`` del JSON de salida.

Uso (correr desde la raíz del repo):
  ia_prediction_service/.venv/bin/python ia_prediction_service/scripts/tth09_weight_sweep.py
  ia_prediction_service/.venv/bin/python ia_prediction_service/scripts/tth09_weight_sweep.py --keep-models
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Mismo patrón que tth09_train.py / tth09_evaluate.py: apuntar a src/ sin instalar paquete.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "ia_prediction_service" / "src"
sys.path.insert(0, str(SRC))
# scripts/ ya está en sys.path[0] al ejecutar el módulo → import directo del hermano.
from tth09_train import BATCH, SEED, build_checkpoint, train_direction  # noqa: E402
from data.tth09_temporal_loader import (  # noqa: E402
    DIRECTIONS,
    N_CLASSES,
    WEIGHT_STRATEGIES,
    build_sequences,
    compute_class_weights,
    load_partition,
)
from evaluation.evaluator import evaluate  # noqa: E402

# Estrategias a barrer (orden estable para lectura lado a lado).
STRATEGIES: tuple[str, ...] = ("none", "inverse", "sqrt", "effective", "cap")
SWEEP_EPOCHS = 12  # FIJO: aislar la variable de peso. NO tocar otros hiperparámetros.
BETA = 0.999  # solo lo usa 'effective' (Cui et al. 2019).
LOOKBACK_MIN = 30  # C2 (cerrado TTH-11)
HORIZONTE_MIN = 30  # C2 (cerrado TTH-11)
EXPECTED_VALID_RUNS = 20  # 5 seeds × 4 patrones (Fase 0/1). Mismatch ⇒ PARAR.

TRAIN_DIR = REPO_ROOT / "simulation" / "data" / "train"
VALID_DIR = REPO_ROOT / "simulation" / "data" / "valid"
SWEEP_ROOT = Path("/tmp/tth09_weight_sweep")  # checkpoints efímeros, fuera del repo.
# JSON consolidado committeado como evidencia del gate (mismo trato que tth09_eval_metrics.json).
OUT_JSON = REPO_ROOT / "ia_prediction_service" / "scripts" / "tth09_weight_sweep_metrics.json"

# Métricas pedidas por estrategia/dirección y para el corte global.
METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "majority_baseline_accuracy",
    "f1_macro_6",
    "f1_macro_present",
    "f1_per_class",
    "mae_ordinal",
)
# Pivote global métrica→estrategia (escalares; f1_per_class queda solo en el detalle).
COMPARE_KEYS: tuple[str, ...] = (
    "accuracy",
    "majority_baseline_accuracy",
    "f1_macro_6",
    "f1_macro_present",
    "mae_ordinal",
)

SELECTION_NOTE = (
    "Estrategia ganadora seleccionada sobre valid; sin test set held-out, el F1 del "
    "ganador puede ser optimista por seleccion de modelo."
)


def set_seeds(seed: int) -> None:
    """Higiene de reproducibilidad a nivel script (train_direction re-siembra por dirección)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _slim(metrics: dict) -> dict:
    """Recorta el reporte del evaluador a las métricas pedidas (descarta extras)."""
    return {k: metrics[k] for k in METRIC_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="TTH-09 barrido de class weights")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--valid-dir", type=Path, default=VALID_DIR)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument(
        "--keep-models",
        action="store_true",
        help=f"No borrar los checkpoints efímeros de {SWEEP_ROOT} al terminar.",
    )
    args = parser.parse_args()

    set_seeds(SEED)
    print(f"[sweep] strategies={list(STRATEGIES)} epochs={SWEEP_EPOCHS} batch={BATCH} "
          f"beta={BETA} train={args.train_dir} valid={args.valid_dir}")

    # Guard temprano: contar valid antes de entrenar nada (mismatch ⇒ PARAR, no malgastar).
    valid_runs = load_partition(args.valid_dir)
    valid_n_runs = len(valid_runs)
    if valid_n_runs != EXPECTED_VALID_RUNS:
        raise SystemExit(
            f"[sweep] PARAR: valid tiene {valid_n_runs} corridas, esperadas "
            f"{EXPECTED_VALID_RUNS} (5 seeds × 4 patrones). No cuadra con Fase 0/1."
        )

    # train_seqs se construye UNA vez y se comparte entre estrategias: el input es el
    # mismo, lo único que cambia entre estrategias es el vector de pesos.
    train_runs = load_partition(args.train_dir)
    if not train_runs:
        raise SystemExit(f"[sweep] PARAR: sin corridas de entrenamiento en {args.train_dir}.")
    train_seqs = build_sequences(train_runs, LOOKBACK_MIN, HORIZONTE_MIN)

    results: dict = {
        "meta": {
            "seed": SEED,
            "epochs": SWEEP_EPOCHS,
            "batch": BATCH,
            "beta": BETA,
            "lookback_min": LOOKBACK_MIN,
            "horizonte_min": HORIZONTE_MIN,
            "strategies": list(STRATEGIES),
            "valid_n_runs": valid_n_runs,
            "selection_note": SELECTION_NOTE,
        },
        "strategies": {},
        "comparison_global": {},
    }

    for strat in STRATEGIES:
        assert strat in WEIGHT_STRATEGIES, f"estrategia no soportada por el loader: {strat}"
        strat_dir = SWEEP_ROOT / strat
        strat_dir.mkdir(parents=True, exist_ok=True)
        t_strat = time.time()

        for d in DIRECTIONS:
            X, y = train_seqs[d]
            counts = np.bincount(
                np.asarray(y).reshape(-1).astype(int), minlength=N_CLASSES
            )[:N_CLASSES]
            weights = compute_class_weights(y, N_CLASSES, strategy=strat, beta=BETA)
            # Invariante (gate): jam5 estructural siempre en peso 0, en toda estrategia.
            assert weights[5] == 0.0, f"[{strat}/{d}] weight[5] (jam5) debe ser 0.0, got {weights[5]}"

            horizonte = y.shape[1]
            model = train_direction(X, y, weights, epochs=SWEEP_EPOCHS, batch=BATCH)
            ckpt = build_checkpoint(
                model, d, weights, counts, horizonte, BATCH, SWEEP_EPOCHS, strat, BETA,
            )
            torch.save(ckpt, strat_dir / f"gru_{d}.pt")

        # Reusa el evaluador tal cual: carga los 4 ckpt del dir de la estrategia.
        report = evaluate(strat_dir, args.valid_dir)
        results["strategies"][strat] = {
            "per_direction": {d: _slim(report["per_direction"][d]) for d in DIRECTIONS},
            "global": _slim(report["global"]),
        }
        g = report["global"]
        print(
            f"[sweep] {strat}: GLOBAL acc={g['accuracy']:.4f} "
            f"(base={g['majority_baseline_accuracy']:.4f}) "
            f"f1m6={g['f1_macro_6']:.4f} f1mp={g['f1_macro_present']:.4f} "
            f"mae_ord={g['mae_ordinal']:.4f}  [{time.time() - t_strat:.0f}s]"
        )

    # Pivote para comparar estrategias lado a lado en el corte global.
    results["comparison_global"] = {
        k: {strat: results["strategies"][strat]["global"][k] for strat in STRATEGIES}
        for k in COMPARE_KEYS
    }

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[sweep] metrics -> {args.out}")

    if args.keep_models:
        print(f"[sweep] checkpoints efímeros conservados en {SWEEP_ROOT}")
    else:
        shutil.rmtree(SWEEP_ROOT, ignore_errors=True)
        print(f"[sweep] checkpoints efímeros borrados ({SWEEP_ROOT}); usar --keep-models para conservarlos")


if __name__ == "__main__":
    main()
