"""TTH-09 Fase 2a — evaluador del GRU multi-output por dirección (cierra CT-09.6).

Carga los 4 checkpoints entrenados en Fase 1 (``gru_<dir>.pt``), corre inferencia
sobre la partición de VALIDACIÓN (``simulation/data/valid``, seeds 21-25, sin fuga:
train usa seeds 1-20) y calcula métricas de CLASIFICACIÓN ordinal multi-output.

El target es ``jam_level`` (0-5, escala Waze) — es un CLASIFICADOR, no un regresor:
NO se calcula MAE/RMSE sobre el ratio. Las métricas son accuracy, F1-macro (dos
lecturas, ver abajo), matriz de confusión 6×6 (convención scikit-learn) y MAE
ORDINAL sobre la clase (distancia |pred - real| tratando las clases como orden).

NO reutiliza ``metrics.py`` de este paquete: ese módulo es del STGNN descartado
(regresión TSL ``MaskedR2``/``MaskedRMSE``), inaplicable a clasificación ordinal.

Honestidad del número (lecturas con contexto, no métricas sueltas):
- F1-macro: se reportan AMBOS. ``f1_macro_6`` (las 6 clases, ``zero_division=0``;
  jam5 tiene soporte 0 por diseño del dataset y cuenta como 0 → pesimista pero
  comparable al spike TTH-11, es el headline) y ``f1_macro_present`` (solo clases
  con soporte>0 en ese corte).
- Majority baseline: ``majority_baseline_accuracy`` = accuracy de predecir SIEMPRE
  la clase mayoritaria. Con N/S ~47% en j1, sin este baseline la accuracy sola
  engaña; con él, "acc 0.52 vs baseline 0.47" se lee al instante.

Tres cortes de reporte: (1) por dirección N/S/E/W consolidado sobre el horizonte,
(2) por paso t+1..t+30 (solo escalares, para ver degradación con el horizonte),
(3) global consolidado (4 direcciones juntas).

Determinista, CPU, sin MLOps. Reusa ``load_partition``/``build_sequences`` del
loader de PRODUCCIÓN (mismo ventaneo multi-output que el entrenamiento).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# src/ está en sys.path (lo inserta el runner scripts/tth09_evaluate.py), igual que
# en tth09_train.py / tth11_sweep.py. Imports relativos a la raíz de src/.
from data.tth09_temporal_loader import (  # noqa: E402
    DIRECTIONS,
    N_CLASSES,
    build_sequences,
    load_partition,
)
from models.gru_multioutput import GRUMultiOutput  # noqa: E402

# Etiquetas fijas 0..5 (escala Waze completa). jam5 (clase 5) nunca se activa en el
# dataset, pero la matriz y el F1 cubren las 6 para no ocultar su ausencia.
LABELS: list[int] = list(range(N_CLASSES))

# Batch de inferencia (no afecta el resultado; solo memoria).
INFER_BATCH: int = 512


def load_model(ckpt_path: Path) -> tuple[GRUMultiOutput, dict]:
    """Carga un checkpoint-dict de Fase 1 y reconstruye el modelo en CPU (eval).

    ``weights_only=False`` es NECESARIO: el checkpoint es un dict autocontenido con
    metadata no-tensor (version, hyperparams, class_weights, counts), no solo el
    state_dict. PyTorch 2.6+ default ``weights_only=True`` rechazaría esos objetos.
    Es seguro: el artefacto lo produce nuestro propio entrenamiento (tth09_train.py),
    fuente confiable; no se cargan checkpoints de terceros.
    """
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Falta el checkpoint del modelo: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    horizonte = int(ckpt["horizonte"])
    n_classes = int(ckpt["n_classes"])
    hidden = int(ckpt["hyperparams"]["hidden"])

    # Guarda: la arquitectura del ckpt debe cuadrar con lo cerrado en Fase 0/1.
    if n_classes != N_CLASSES:
        raise ValueError(
            f"{ckpt_path}: n_classes={n_classes} != {N_CLASSES} (escala Waze). PARAR."
        )

    model = GRUMultiOutput(hidden=hidden, horizonte=horizonte, n_classes=n_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model: GRUMultiOutput, X: np.ndarray) -> np.ndarray:
    """Inferencia determinista: X (n, lookback) -> preds (n, horizonte).

    Replica la normalización del entrenamiento (jam_level /5.0, 1 feature por paso):
    X -> (n, lookback, 1) float32. Toma argmax sobre la dimensión de clase.
    """
    model.eval()
    preds: list[np.ndarray] = []
    n = X.shape[0]
    for i in range(0, n, INFER_BATCH):
        xb = torch.tensor(X[i : i + INFER_BATCH], dtype=torch.float32).unsqueeze(-1) / 5.0
        logits = model(xb)  # (b, H, C)
        preds.append(logits.argmax(-1).cpu().numpy())  # (b, H)
    if not preds:
        return np.empty((0, model.horizonte), dtype=int)
    return np.concatenate(preds, axis=0).astype(int)


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas de clasificación ordinal sobre vectores APLANADOS (1D).

    Sirve para un corte consolidado (toda una dirección, o el global). Incluye el
    majority baseline para que la accuracy sea interpretable bajo desbalance fuerte.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    n = int(y_true.shape[0])

    present = sorted(int(c) for c in np.unique(y_true))
    support = np.bincount(y_true, minlength=N_CLASSES)[:N_CLASSES]

    # Majority baseline: accuracy de predecir siempre la clase mayoritaria en y_true.
    maj_class = int(support.argmax()) if n else -1
    maj_baseline = float(support.max() / n) if n else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)  # filas=real, cols=pred

    return {
        "n": n,
        "accuracy": float(accuracy_score(y_true, y_pred)) if n else 0.0,
        "majority_class": maj_class,
        "majority_baseline_accuracy": maj_baseline,
        # Headline: las 6 clases (jam5 soporte 0 -> F1 0 -> baja el macro, a propósito).
        "f1_macro_6": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        # Secundaria: solo clases con soporte>0 en ESTE corte.
        "f1_macro_present": float(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        )
        if present
        else 0.0,
        "f1_per_class": [
            round(float(v), 4)
            for v in f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
        ],
        "mae_ordinal": float(np.abs(y_pred - y_true).mean()) if n else 0.0,
        "confusion_6x6": cm.tolist(),
        "support_per_class": support.tolist(),
        "classes_true": present,
        "classes_pred": sorted(int(c) for c in np.unique(y_pred)),
        "jam5_in_true": bool((y_true == 5).any()),
        "jam5_in_pred": bool((y_pred == 5).any()),
    }


def step_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Escalares por paso del horizonte t+1..t+H (degradación con el horizonte).

    y_true / y_pred shape (n, H). Por cada paso solo escalares (sin matriz, para no
    inflar el JSON ~30×). f1_macro_6 es la serie COMPARABLE entre pasos (denominador
    fijo = 6); f1_macro_present es referencia secundaria (su denominador varía).
    """
    H = y_true.shape[1]
    out: list[dict] = []
    for t in range(H):
        yt = y_true[:, t].astype(int)
        yp = y_pred[:, t].astype(int)
        present = sorted(int(c) for c in np.unique(yt))
        out.append(
            {
                "step": t + 1,
                "accuracy": float(accuracy_score(yt, yp)),
                "f1_macro_6": float(
                    f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0)
                ),
                "f1_macro_present": float(
                    f1_score(yt, yp, labels=present, average="macro", zero_division=0)
                )
                if present
                else 0.0,
                "mae_ordinal": float(np.abs(yp - yt).mean()),
            }
        )
    return out


def evaluate(
    models_dir: Path,
    valid_dir: Path,
    lookback_min: int = 30,
    horizonte_min: int = 30,
) -> dict:
    """Orquesta carga, inferencia y arma el reporte de los tres cortes.

    Acumula y_true/y_pred de las 4 direcciones para el corte global consolidado.
    Para y reporta si falta un modelo, si una dirección no tiene secuencias, o si
    los shapes no cuadran con lo cerrado en Fase 0/1.
    """
    valid_runs = load_partition(valid_dir)
    if not valid_runs:
        raise FileNotFoundError(f"Sin corridas de validación en {valid_dir}. PARAR.")
    seqs = build_sequences(valid_runs, lookback_min, horizonte_min)

    version: str | None = None
    per_direction: dict[str, dict] = {}
    glob_true: list[np.ndarray] = []
    glob_pred: list[np.ndarray] = []

    for d in DIRECTIONS:
        model, ckpt = load_model(models_dir / f"gru_{d}.pt")
        version = version or str(ckpt.get("version"))

        X, y = seqs[d]
        if X.shape[0] == 0:
            raise ValueError(
                f"[{d}] build_sequences(valid) dio n_seq=0. Datos insuficientes. PARAR."
            )

        y_pred = predict(model, X)
        # Guarda de shapes: target e inferencia deben ser (n, H) coherentes.
        if y_pred.shape != y.shape:
            raise ValueError(
                f"[{d}] shape mismatch: pred {y_pred.shape} != target {y.shape}. "
                "No cuadra con Fase 0/1. PARAR."
            )

        rep = class_metrics(y, y_pred)
        rep["per_step"] = step_metrics(y, y_pred)
        per_direction[d] = rep

        glob_true.append(y)
        glob_pred.append(y_pred)

    gt = np.concatenate(glob_true, axis=0)
    gp = np.concatenate(glob_pred, axis=0)
    global_rep = class_metrics(gt, gp)
    global_rep["per_step"] = step_metrics(gt, gp)

    return {
        "version": version,
        "partition": "valid",
        "n_runs": len(valid_runs),
        "lookback_min": lookback_min,
        "horizonte_min": horizonte_min,
        "n_classes": N_CLASSES,
        "device": "cpu",
        "f1_note": (
            "f1_macro_6 = headline (6 clases, jam5 soporte 0 cuenta como 0). "
            "f1_macro_present = solo clases con soporte>0 en el corte. "
            "accuracy se lee contra majority_baseline_accuracy."
        ),
        "per_direction": per_direction,
        "global": global_rep,
    }
