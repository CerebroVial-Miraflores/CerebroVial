"""TTH-09 Fase 2a — runner de evaluación del GRU multi-output (cierra CT-09.6).

Carga los 4 modelos full entrenados en Fase 1 (``ia_prediction_service/models/
gru_<dir>.pt``), evalúa sobre la partición de validación (``simulation/data/valid``,
seeds 21-25, sin fuga) y persiste las métricas de clasificación a un JSON.

La lógica vive en ``src/evaluation/evaluator.py`` (importable, testeable); este script
es solo el CLI: arma sys.path, llama ``evaluate(...)``, escribe el JSON e imprime un
resumen por dirección. Determinista, CPU, sin MLOps.

Uso (correr desde la raíz del repo):
  ia_prediction_service/.venv/bin/python ia_prediction_service/scripts/tth09_evaluate.py
  ia_prediction_service/.venv/bin/python ia_prediction_service/scripts/tth09_evaluate.py --out /tmp/eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# La lógica y el loader viven en ia_prediction_service/src/. Apuntamos explícito
# (mismo patrón que tth09_train.py / tth11_sweep.py: sin glob, sin instalar paquete).
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "ia_prediction_service" / "src"
sys.path.insert(0, str(SRC))
from evaluation.evaluator import evaluate  # noqa: E402

MODELS_DIR = REPO_ROOT / "ia_prediction_service" / "models"
VALID_DIR = REPO_ROOT / "simulation" / "data" / "valid"
# JSON committeado como evidencia del gate (mismo lugar/trato que tth11_sweep_metrics.json).
OUT_JSON = REPO_ROOT / "ia_prediction_service" / "scripts" / "tth09_eval_metrics.json"

SEED = 0


def set_seeds(seed: int) -> None:
    """Higiene de reproducibilidad (la inferencia ya es determinista; esto la blinda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="TTH-09 evaluación GRU multi-output")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--valid-dir", type=Path, default=VALID_DIR)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    set_seeds(SEED)
    print(f"[tth09-eval] models={args.models_dir} valid={args.valid_dir}")

    report = evaluate(args.models_dir, args.valid_dir)

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Resumen legible: accuracy vs baseline, ambos F1, jam5_in_pred, por dirección + global.
    print(f"[tth09-eval] version={report['version']} n_runs={report['n_runs']}")
    for d, m in report["per_direction"].items():
        print(
            f"  {d}: acc={m['accuracy']:.4f} (base={m['majority_baseline_accuracy']:.4f}) "
            f"f1m6={m['f1_macro_6']:.4f} f1mp={m['f1_macro_present']:.4f} "
            f"mae_ord={m['mae_ordinal']:.4f} jam5_pred={m['jam5_in_pred']}"
        )
    g = report["global"]
    print(
        f"  GLOBAL: acc={g['accuracy']:.4f} (base={g['majority_baseline_accuracy']:.4f}) "
        f"f1m6={g['f1_macro_6']:.4f} f1mp={g['f1_macro_present']:.4f} "
        f"mae_ord={g['mae_ordinal']:.4f}"
    )
    print(f"[tth09-eval] metrics -> {args.out}")


if __name__ == "__main__":
    main()
