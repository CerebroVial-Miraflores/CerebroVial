"""TTH-09 Fase 1 — entrenamiento GRU multi-output por dirección (CT-09.1/.2/.3).

Entrena 4 GRU univariadas multi-output (una por N/S/E/W, D-006), maneja el
desbalance con class weights POR DIRECCIÓN (inverso de frecuencia sobre train),
y persiste los 4 modelos como checkpoint dict autocontenido + un metadata.json
índice bajo ``ia_prediction_service/models/``.

Arquitectura (b1, cerrada): GRU univariada (input_size=1, jam_level /5.0) → cabeza
ancha Linear(hidden, horizonte*n_classes) → reshape (batch, horizonte, n_classes).
Multi-output DIRECTO (NO autoregresivo, NO seq2seq).

Loss: CrossEntropyLoss con weight= por dirección, promediada sobre los
``horizonte`` pasos × batch. Shape: se reshapea logits (batch, H, C) → (batch*H, C)
y target (batch, H) → (batch*H,); CrossEntropyLoss promedia sobre los batch*H
elementos = promedio sobre H pasos × batch (lo pedido). Equivale a permutar a
(batch, C, H) vs target (batch, H); se elige el reshape por ser inequívoco.

Hiperparámetros temporales (cerrados TTH-11, C2): lookback=30, horizonte=30, Δt=60s.

Determinista: seeds fijas (random, numpy, torch). Sin MLOps.

Uso:
  python ia_prediction_service/scripts/tth09_train.py            # full (80 corridas)
  python ia_prediction_service/scripts/tth09_train.py --quick    # smoke / subset
(correr desde la raíz del repo; usa simulation/data/{train,valid}).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# El loader y el modelo viven en ia_prediction_service/src/. Apuntamos explícito.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "ia_prediction_service" / "src"
sys.path.insert(0, str(SRC))
from data.tth09_temporal_loader import (  # noqa: E402
    DIRECTIONS,
    N_CLASSES,
    WEIGHT_STRATEGIES,
    build_sequences,
    compute_class_weights,
    load_partition,
)
from models.gru_multioutput import GRUMultiOutput  # noqa: E402

# --- Config base ---
# NOTA (Fase 2): batch 512 / lr 1e-3 / hidden 64 vienen del spike TTH-11, que era
# single-step (target ESCALAR). Esta loss es MULTI-OUTPUT (target de 30 pasos),
# ~30× más densa en señal por ejemplo. Son un punto de partida razonable, pero NO
# hay garantía de que transfieran. El tuning fino es Fase 2: si el modelo no
# converge, estos hiperparámetros son sospechoso explícito.
HIDDEN = 64
LR = 1e-3
BATCH = 512
EPOCHS = 12
LOOKBACK_MIN = 30  # C2 (cerrado TTH-11)
HORIZONTE_MIN = 30  # C2 (cerrado TTH-11)
SEED = 0
VERSION = "tth09-fase1-b1"

# Subset reducido para el gate end-to-end (--quick): pocas corridas, pocos epochs.
QUICK_RUNS = 2
QUICK_EPOCHS = 1
QUICK_BATCH = 128

MODELS_DIR = REPO_ROOT / "ia_prediction_service" / "models"

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def set_seeds(seed: int) -> None:
    """Fija las seeds de random, numpy y torch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensors(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """jam_level normalizado /5.0, 1 feature por paso -> X (n, lookback, 1).

    y se mantiene entero (n, horizonte) para CrossEntropyLoss.
    """
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) / 5.0
    yt = torch.tensor(y, dtype=torch.long)
    return Xt, yt


def describe_zero_weight_classes(weights: np.ndarray, counts: np.ndarray) -> dict[int, str]:
    """Mapea cada clase con peso 0.0 a la CAUSA, distinguiendo:

      - clase 5 sin muestras  -> 'jam5-estructural' (siempre, por diseño del dataset).
      - otra clase con conteo 0 -> 'ausente-en-subset-quick' (artefacto del subset
        reducido; en full esa clase aparece y recibe su peso alto).
    """
    out: dict[int, str] = {}
    for c in range(len(weights)):
        if weights[c] != 0.0:
            continue
        if c == 5:
            out[c] = "jam5-estructural"
        elif int(counts[c]) == 0:
            out[c] = "ausente-en-subset-quick"
        else:
            out[c] = "peso-0-inesperado"  # no debería ocurrir; señal para investigar
    return out


def train_direction(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    epochs: int,
    batch: int,
) -> GRUMultiOutput:
    """Entrena UN modelo (una dirección) de forma determinista.

    Loss multi-output con weights por dirección. Reshape (batch*H, C) vs (batch*H,).
    """
    set_seeds(SEED)
    Xt, yt = to_tensors(X, y)
    n = Xt.shape[0]
    horizonte = yt.shape[1]

    model = GRUMultiOutput(hidden=HIDDEN, horizonte=horizonte, n_classes=N_CLASSES).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    w = torch.tensor(weights, dtype=torch.float32, device=DEV)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb = Xt[idx].to(DEV)
            yb = yt[idx].to(DEV)
            opt.zero_grad()
            logits = model(xb)  # (b, H, C)
            # Reshape para CrossEntropy: (b*H, C) vs (b*H,). Promedia sobre H × batch.
            loss = loss_fn(logits.reshape(-1, N_CLASSES), yb.reshape(-1))
            loss.backward()
            opt.step()
    return model


def build_checkpoint(
    model: GRUMultiOutput,
    direction: str,
    weights: np.ndarray,
    counts: np.ndarray,
    horizonte: int,
    batch: int,
    epochs: int,
    strategy: str,
    beta: float,
) -> dict:
    """Arma el dict de checkpoint persistido por dirección.

    Reutilizado por el entrenamiento (``main``) y por el barrido de pesos
    (``tth09_weight_sweep``) para que ambos artefactos compartan EXACTAMENTE la
    misma estructura. ``weight_strategy`` queda siempre registrado; ``weight_beta``
    solo cuando la estrategia lo usa (``effective``).
    """
    ckpt = {
        "state_dict": model.state_dict(),
        "version": VERSION,
        "direction": direction,
        "seed": SEED,
        "lookback": (LOOKBACK_MIN * 60) // 60,
        "horizonte": horizonte,
        "n_classes": N_CLASSES,
        "input_norm": "/5.0",
        "hyperparams": {
            "hidden": HIDDEN,
            "lr": LR,
            "batch": batch,
            "epochs": epochs,
        },
        "weight_strategy": strategy,
        "class_weights": weights.tolist(),
        "class_counts": counts.tolist(),
    }
    if strategy == "effective":
        ckpt["weight_beta"] = beta
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="TTH-09 entrenamiento GRU multi-output")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo smoke: subset reducido + pocos epochs para el gate end-to-end.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override de epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Override de batch.")
    parser.add_argument(
        "--weight-strategy",
        choices=WEIGHT_STRATEGIES,
        default="inverse",
        help="Estrategia de class weights (default: inverse, comportamiento histórico).",
    )
    parser.add_argument(
        "--weight-beta",
        type=float,
        default=0.999,
        help="Beta para la estrategia 'effective' (Cui et al. 2019). Ignorado por el resto.",
    )
    args = parser.parse_args()

    quick = args.quick
    epochs = args.epochs if args.epochs is not None else (QUICK_EPOCHS if quick else EPOCHS)
    batch = args.batch if args.batch is not None else (QUICK_BATCH if quick else BATCH)
    weight_strategy = args.weight_strategy
    weight_beta = args.weight_beta
    max_runs = QUICK_RUNS if quick else None

    set_seeds(SEED)
    print(f"[tth09] device={DEV} quick={quick} epochs={epochs} batch={batch} "
          f"max_runs={max_runs} lookback={LOOKBACK_MIN}min horizonte={HORIZONTE_MIN}min "
          f"weight_strategy={weight_strategy}"
          + (f" weight_beta={weight_beta}" if weight_strategy == "effective" else ""))

    train_dir = REPO_ROOT / "simulation" / "data" / "train"
    train_runs = load_partition(train_dir, max_runs=max_runs)
    seqs = build_sequences(train_runs, LOOKBACK_MIN, HORIZONTE_MIN)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metadata: dict = {
        "version": VERSION,
        "seed": SEED,
        "device": DEV,
        "quick": quick,
        "hyperparams": {
            "hidden": HIDDEN,
            "lr": LR,
            "batch": batch,
            "epochs": epochs,
            "lookback_min": LOOKBACK_MIN,
            "horizonte_min": HORIZONTE_MIN,
            "n_classes": N_CLASSES,
            "input_norm": "/5.0",
            "weight_strategy": weight_strategy,
            **({"weight_beta": weight_beta} if weight_strategy == "effective" else {}),
        },
        "directions": {},
    }

    for d in DIRECTIONS:
        X, y = seqs[d]
        counts = np.bincount(np.asarray(y).reshape(-1).astype(int), minlength=N_CLASSES)[:N_CLASSES]
        weights = compute_class_weights(y, N_CLASSES, strategy=weight_strategy, beta=weight_beta)
        zero_causes = describe_zero_weight_classes(weights, counts)

        # Log explícito de qué clases quedaron en peso 0 y POR QUÉ (legible en voz alta).
        print(f"[{d}] n_seq={X.shape[0]} X{tuple(X.shape)} y{tuple(y.shape)} "
              f"counts={counts.tolist()}")
        print(f"[{d}] class_weights: clases con peso 0 -> {zero_causes if zero_causes else '(ninguna)'}")

        # Shape asserts (gate): logits (b,H,C), target (b,H), weight[5]==0.
        assert y.ndim == 2 and y.shape[1] == (HORIZONTE_MIN * 60) // 60, \
            f"[{d}] target multi-output esperado (n,{HORIZONTE_MIN}), got {y.shape}"
        assert len(weights) == N_CLASSES, f"[{d}] weights len {len(weights)} != {N_CLASSES}"
        assert weights[5] == 0.0, f"[{d}] weight[5] (jam5) debe ser 0.0, got {weights[5]}"

        t0 = time.time()
        model = train_direction(X, y, weights, epochs=epochs, batch=batch)
        dt = time.time() - t0

        horizonte = y.shape[1]
        ckpt = build_checkpoint(
            model, d, weights, counts, horizonte, batch, epochs,
            weight_strategy, weight_beta,
        )
        out_path = MODELS_DIR / f"gru_{d}.pt"
        torch.save(ckpt, out_path)
        print(f"[{d}] entrenado en {dt:.1f}s -> {out_path}")

        metadata["directions"][d] = {
            "n_seq": int(X.shape[0]),
            "class_counts": counts.tolist(),
            "class_weights": weights.tolist(),
            "zero_weight_classes": {str(k): v for k, v in zero_causes.items()},
            "train_time_s": round(dt, 1),
            "checkpoint": out_path.name,
        }

    meta_path = MODELS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[tth09] metadata -> {meta_path}")
    print(f"[tth09] OK: 4 modelos + metadata en {MODELS_DIR}")


if __name__ == "__main__":
    main()
