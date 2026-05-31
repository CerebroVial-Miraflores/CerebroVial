"""TTH-11 CT-11.4 — barrido exploratorio lookback x horizonte (modelos EFIMEROS).

4 direcciones (N/S/E/W) x 4 combos (C1..C4) = 16 GRUs univariadas (D-006).
Entrena, mide las 4 metricas por direccion, DESCARTA el modelo (ningun checkpoint
a disco) y persiste SOLO las metricas a un JSON en /tmp.

Config fija para los 16 (comparacion limpia): GRU 1 capa hidden=64, input_size=1
(jam_level normalizado /5.0), Linear(64->6), CrossEntropyLoss SIN class weights,
Adam lr=1e-3, batch 512, EPOCHS identicas para todos. 6 clases (escala Waze 0-5;
jam5 fuera-de-soporte, no se remapea). SIN rebalanceo (el desbalance es dato para
CT-11.6, no se corrige).

Uso:  .venv/bin/python ia_prediction_service/scripts/tth11_sweep.py
(correr desde la raiz del repo; usa simulation/data/{train,valid}).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# El loader vive en ia_prediction_service/src/data/. Apuntamos explicito (sin glob).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ia_prediction_service" / "src"))
from data.tth11_temporal_loader import load_partition, build_sequences  # noqa: E402

# --- Config del barrido (fija) ---
NUM_CLASSES = 6  # escala Waze 0-5 completa; jam5 nunca se activa (fuera-de-soporte)
HIDDEN = 64
LR = 1e-3
BATCH = 512
EPOCHS = 12  # 10-15; identico para los 16 (el nº de epocas NO varia entre combos)
DIRECTIONS = ("N", "S", "E", "W")
COMBOS = {  # nombre -> (lookback_min, horizonte_min)
    "C1": (15, 30),
    "C2": (30, 30),
    "C3": (30, 60),
    "C4": (60, 60),
}
SEED = 0
OUT_JSON = Path("/tmp/tth11_sweep_metrics.json")

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


class GRUClf(nn.Module):
    def __init__(self, hidden: int = HIDDEN, n_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)  # softmax implicito en CrossEntropyLoss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.fc(h[-1])


def to_tensors(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """jam_level normalizado /5.0, 1 feature por paso -> (n, lookback, 1)."""
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) / 5.0
    yt = torch.tensor(y, dtype=torch.long)
    return Xt, yt


def train_one(Xtr: np.ndarray, ytr: np.ndarray) -> GRUClf:
    torch.manual_seed(SEED)
    Xt, yt = to_tensors(Xtr, ytr)
    n = Xt.shape[0]
    model = GRUClf().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()  # SIN class weights
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            xb = Xt[idx].to(DEV)
            yb = yt[idx].to(DEV)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model: GRUClf, X: np.ndarray) -> np.ndarray:
    model.eval()
    Xt, _ = to_tensors(X, np.zeros(len(X)))
    out = []
    for i in range(0, Xt.shape[0], BATCH):
        out.append(model(Xt[i : i + BATCH].to(DEV)).argmax(1).cpu())
    return torch.cat(out).numpy()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    acc = float((y_pred == y_true).mean())
    mae = float(np.abs(y_pred - y_true).mean())  # MAE ordinal

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    f1s = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    present = [c for c in range(NUM_CLASSES) if cm[c].sum() > 0]
    f1_macro_6 = float(np.mean(f1s))  # PRIMARIA (6 clases)
    f1_macro_present = float(np.mean([f1s[c] for c in present]))  # secundaria

    vals, cnts = np.unique(y_true, return_counts=True)
    maj_class = int(vals[cnts.argmax()])
    baseline = float(cnts.max() / cnts.sum())

    return {
        "accuracy": acc,
        "f1_macro_6": f1_macro_6,
        "f1_macro_present": f1_macro_present,
        "f1_per_class": [round(f, 4) for f in f1s],
        "mae_ordinal": mae,
        "confusion_6x6": cm.tolist(),
        "majority_class": maj_class,
        "majority_baseline_acc": baseline,
        "classes_true": sorted(int(v) for v in vals),
        "classes_pred": sorted(int(v) for v in np.unique(y_pred)),
        "jam5_in_true": bool((y_true == 5).any()),
        "jam5_in_pred": bool((y_pred == 5).any()),
        "n_valid": int(len(y_true)),
    }


def main() -> None:
    print(f"device={DEV} torch={torch.__version__} epochs={EPOCHS}")
    train_runs = load_partition(REPO_ROOT / "simulation" / "data" / "train")
    valid_runs = load_partition(REPO_ROOT / "simulation" / "data" / "valid")
    print(f"runs: train={len(train_runs)} valid={len(valid_runs)}")

    results: dict = {"config": {
        "hidden": HIDDEN, "lr": LR, "batch": BATCH, "epochs": EPOCHS,
        "num_classes": NUM_CLASSES, "class_weights": False, "rebalance": False,
        "input": "jam_level/5.0", "device": DEV,
    }, "models": {d: {} for d in DIRECTIONS}}

    t_start = time.time()
    # Cache de ventanas por combo (build_sequences devuelve las 4 direcciones de una)
    for combo, (lb, hz) in COMBOS.items():
        tr_seqs = build_sequences(train_runs, lb, hz)
        va_seqs = build_sequences(valid_runs, lb, hz)
        for d in DIRECTIONS:
            Xtr, ytr = tr_seqs[d]
            Xva, yva = va_seqs[d]
            t0 = time.time()
            model = train_one(Xtr, ytr)
            y_pred = predict(model, Xva)
            m = metrics(yva, y_pred)
            m["train_time_s"] = round(time.time() - t0, 2)
            m["n_train"] = int(len(ytr))
            results["models"][d][combo] = m
            del model  # efimero: descartar
            print(f"  {d} {combo} ({lb}/{hz}): acc={m['accuracy']:.4f} "
                  f"f1m6={m['f1_macro_6']:.4f} mae={m['mae_ordinal']:.4f} "
                  f"({m['train_time_s']}s)")

    results["total_time_s"] = round(time.time() - t_start, 1)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\ntotal={results['total_time_s']}s  metrics -> {OUT_JSON}")


if __name__ == "__main__":
    main()
