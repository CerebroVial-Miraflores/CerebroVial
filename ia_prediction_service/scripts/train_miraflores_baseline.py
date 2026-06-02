"""STGNN Fase 3 — entrenamiento del baseline GRU univariado solo-temporal.

Loop manual en torch puro (espejo de ``tth09_train.py``, sin tsl). Entrena UN modelo
de pesos compartidos sobre los 375 nodos: cada ventana es la serie de ``timeLoss`` de
un nodo (sin vecinos). Target = ``timeLoss`` TOTAL continuo (D-013 + enmienda
2026-06-01); regresión, NO clasificación.

Pipeline:
  npz Fase 2 → split compartido (miraflores_split) → índices de ventanas por fold
  (miraflores_windower) → scaler train-only → entrenamiento con MSE enmascarado en
  espacio estandarizado + early stopping sobre val → evaluación en TEST des-
  estandarizada (MAE/RMSE/R² por horizonte), con desglose día-081 vs días normales.

Artefactos:
  - models/miraflores_gru_baseline.pt           checkpoint (gitignored, regenerable)
  - scripts/miraflores_baseline_metadata.json   config + split + scaler + historia (versionado)
  - scripts/miraflores_baseline_metrics.json    tablas de métricas test/081/normales (versionado)

Nota de eficiencia: con stride=1 hay ~1381 ventanas/nodo/día (≈22 M en train), muy
redundantes (solapan en 29/30 pasos). Para que el baseline corra en tiempo razonable
se entrena/evalúa con un ``stride`` configurable (default 12 = 1 ventana cada 12 min);
queda REGISTRADO en metadata para que la Fase 4 use el idéntico (comparación justa).

Uso (desde la raíz del repo, venv raíz):
  ../.venv/bin/python ia_prediction_service/scripts/train_miraflores_baseline.py
  ../.venv/bin/python ia_prediction_service/scripts/train_miraflores_baseline.py --quick
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

# Los módulos del baseline viven en ia_prediction_service/src/. Apuntamos explícito.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "ia_prediction_service" / "src"
sys.path.insert(0, str(SRC))
from data.miraflores_split import get_split  # noqa: E402
from data.miraflores_windower import (  # noqa: E402
    DEFAULT_NPZ,
    HORIZON,
    LOOKBACK,
    build_window_index,
    compute_timeloss_scaler,
    gather_batch,
    load_npz,
)
from evaluation.miraflores_metrics import (  # noqa: E402
    DEFAULT_CUTS,
    format_metrics_table,
    masked_metrics_by_horizon,
)
from models.miraflores_gru_baseline import MirafloresGRUBaseline  # noqa: E402

# --- Config base ---
HIDDEN = 64
LR = 1e-3
BATCH = 512
MAX_EPOCHS = 60
PATIENCE = 6          # épocas sin mejora de val antes de parar
TRAIN_STRIDE = 12     # 1 ventana cada 12 min en train (ver nota de eficiencia)
EVAL_STRIDE = 12      # idem en val/test (métricas estables, evaluación rápida)
SEED = 0
VERSION = "stgnn-fase3-baseline-b1"

TARGET_AUTHORITY = (
    "D-013 + enmienda 2026-06-01: target = timeLoss TOTAL por arista por intervalo "
    "de 60 s (columna timeLoss del Parquet, sin derivación per-vehículo)."
)

MODELS_DIR = REPO_ROOT / "ia_prediction_service" / "models"
SCRIPTS_DIR = REPO_ROOT / "ia_prediction_service" / "scripts"

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def set_seeds(seed: int) -> None:
    """Fija las seeds de random, numpy y torch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize_inputs(X: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Estandariza el canal 0 (timeLoss) y RE-ZERA celdas vacías vía el indicador.

    X: ``[B, L, 2]`` crudo (canal0 timeLoss con vacío=0, canal1 validez 1/0).
    Válido → (timeLoss-mean)/std; vacío → 0 (porque canal1=0 multiplica). El canal 1
    (indicador) no se escala.
    """
    valid = X[..., 1]
    ch0 = (X[..., 0] - mean) / std * valid
    return torch.stack([ch0, valid], dim=-1)


def masked_mse_sums(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Devuelve (suma de errores²·máscara, suma de máscara) para promediar exacto."""
    m = mask.to(pred.dtype)
    sq = ((pred - target) ** 2 * m).sum()
    return sq, m.sum()


def run_epoch(model, opt, timeloss, mask, index, mean, std, batch, *, train: bool):
    """Una pasada sobre ``index``. Devuelve el MSE enmascarado exacto (espacio std).

    Si ``train``: shuffle con np.random.permutation (determinista por seed) + backward.
    Si no: orden secuencial, sin grad.
    """
    model.train(train)
    n = index.shape[0]
    order = np.random.permutation(n) if train else np.arange(n)
    sq_sum = 0.0
    mask_sum = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i in range(0, n, batch):
            rows = index[order[i : i + batch]]
            X, y, ymask = gather_batch(timeloss, mask, rows)
            Xt = torch.from_numpy(X).to(DEV)
            yt = torch.from_numpy(y).to(DEV)
            mt = torch.from_numpy(ymask).to(DEV)
            Xt = standardize_inputs(Xt, mean, std)
            yt_std = (yt - mean) / std
            if train:
                opt.zero_grad()
            pred = model(Xt)
            sq, msum = masked_mse_sums(pred, yt_std, mt)
            denom = msum.clamp_min(1.0)
            loss = sq / denom
            if train:
                loss.backward()
                opt.step()
            sq_sum += float(sq.item())
            mask_sum += float(msum.item())
    return sq_sum / max(mask_sum, 1.0)


def predict_collect(model, timeloss, mask, index, mean, std, batch):
    """Predice sobre ``index`` y des-estandariza a segundos.

    Devuelve (pred_sec ``[N,H]``, y_raw ``[N,H]``, ymask ``[N,H]``, row_seeds ``[N]``).
    """
    model.eval()
    n = index.shape[0]
    preds, ys, ms = [], [], []
    with torch.no_grad():
        for i in range(0, n, batch):
            rows = index[i : i + batch]
            X, y, ymask = gather_batch(timeloss, mask, rows)
            Xt = standardize_inputs(torch.from_numpy(X).to(DEV), mean, std)
            pred_std = model(Xt)
            pred_sec = (pred_std * std + mean).cpu()
            preds.append(pred_sec)
            ys.append(torch.from_numpy(y))
            ms.append(torch.from_numpy(ymask))
    pred_sec = torch.cat(preds, dim=0)
    y_raw = torch.cat(ys, dim=0)
    ymask = torch.cat(ms, dim=0)
    return pred_sec, y_raw, ymask


def main() -> None:
    parser = argparse.ArgumentParser(description="STGNN Fase 3 — baseline GRU univariado")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke: pocas épocas y stride grande para el gate end-to-end.")
    parser.add_argument("--epochs", type=int, default=None, help="Override de max épocas.")
    parser.add_argument("--patience", type=int, default=None, help="Override de paciencia.")
    parser.add_argument("--batch", type=int, default=None, help="Override de batch.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--hidden", type=int, default=HIDDEN, help="Tamaño hidden GRU.")
    parser.add_argument("--train-stride", type=int, default=None, help="Stride de ventanas train.")
    parser.add_argument("--eval-stride", type=int, default=None, help="Stride de ventanas val/test.")
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ), help="Ruta del .npz Fase 2.")
    args = parser.parse_args()

    quick = args.quick
    max_epochs = args.epochs if args.epochs is not None else (1 if quick else MAX_EPOCHS)
    patience = args.patience if args.patience is not None else (1 if quick else PATIENCE)
    batch = args.batch if args.batch is not None else BATCH
    train_stride = args.train_stride if args.train_stride is not None else (60 if quick else TRAIN_STRIDE)
    eval_stride = args.eval_stride if args.eval_stride is not None else (60 if quick else EVAL_STRIDE)
    lr = args.lr
    hidden = args.hidden

    set_seeds(SEED)
    print(f"[baseline] device={DEV} quick={quick} max_epochs={max_epochs} patience={patience} "
          f"batch={batch} lr={lr} hidden={hidden} lookback={LOOKBACK} horizonte={HORIZON} "
          f"train_stride={train_stride} eval_stride={eval_stride}")

    # --- Datos + split + scaler train-only ---
    data = load_npz(args.npz)
    timeloss, mask, seeds = data["timeloss"], data["mask"], data["seeds"]
    n_days, n_t, n_nodes = timeloss.shape
    train_seeds, val_seeds, test_seeds = get_split()
    print(f"[baseline] dataset {timeloss.shape} | split train/val/test = "
          f"{len(train_seeds)}/{len(val_seeds)}/{len(test_seeds)} (081 in test: {81 in test_seeds})")

    scaler = compute_timeloss_scaler(timeloss, mask, seeds, train_seeds)
    mean, std = scaler["mean"], scaler["std"]
    print(f"[baseline] scaler train-only (timeLoss): mean={mean:.3f}s std={std:.3f}s")

    idx_kw = dict(n_nodes=n_nodes, n_timesteps=n_t, lookback=LOOKBACK, horizon=HORIZON)
    train_index = build_window_index(seeds, train_seeds, stride=train_stride, **idx_kw)
    val_index = build_window_index(seeds, val_seeds, stride=eval_stride, **idx_kw)
    test_index = build_window_index(seeds, test_seeds, stride=eval_stride, **idx_kw)
    print(f"[baseline] ventanas train/val/test = "
          f"{train_index.shape[0]:,}/{val_index.shape[0]:,}/{test_index.shape[0]:,}")

    # --- Modelo / optimizador ---
    model = MirafloresGRUBaseline(hidden=hidden, horizonte=HORIZON, input_size=2).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Loop con early stopping sobre val ---
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict] = []
    print(f"[baseline] entrenando (MSE enmascarado, espacio estandarizado)…")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, opt, timeloss, mask, train_index, mean, std, batch, train=True)
        val_loss = run_epoch(model, opt, timeloss, mask, val_index, mean, std, batch, train=False)
        dt = time.time() - t0
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        history.append({"epoch": epoch, "train_mse_std": round(train_loss, 6),
                        "val_mse_std": round(val_loss, 6), "secs": round(dt, 1)})
        flag = " *mejor*" if improved else ""
        print(f"  ep {epoch:>3}/{max_epochs}  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"({dt:.1f}s){flag}")
        if epochs_no_improve >= patience:
            print(f"[baseline] early stopping en ep {epoch} (sin mejora hace {patience}); "
                  f"mejor val={best_val:.5f} @ ep {best_epoch}")
            break

    # Restaurar mejor modelo.
    model.load_state_dict(best_state)

    # --- Evaluación en TEST (des-estandarizada, segundos) ---
    pred_sec, y_raw, ymask = predict_collect(model, timeloss, mask, test_index, mean, std, batch)
    row_seeds = np.asarray([int(seeds[d]) for d in test_index[:, 0]])
    is081 = torch.from_numpy(row_seeds == 81)
    cuts = DEFAULT_CUTS

    metrics_all = masked_metrics_by_horizon(pred_sec, y_raw, ymask, cuts)
    metrics_081 = masked_metrics_by_horizon(pred_sec[is081], y_raw[is081], ymask[is081], cuts)
    norm = ~is081
    metrics_normal = masked_metrics_by_horizon(pred_sec[norm], y_raw[norm], ymask[norm], cuts)

    print("\n[baseline] ===== MÉTRICAS TEST (des-estandarizadas, segundos) =====")
    print(f"\nTEST completo (10 días, n_ventanas={test_index.shape[0]:,}):")
    print(format_metrics_table(metrics_all, cuts))
    print(f"\nDía 081 (congestión máxima, n_ventanas={int(is081.sum())}):")
    print(format_metrics_table(metrics_081, cuts))
    print(f"\nDías normales de test (9 días, n_ventanas={int(norm.sum())}):")
    print(format_metrics_table(metrics_normal, cuts))

    # --- Persistencia ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    arch = {"input_size": 2, "hidden": hidden, "horizonte": HORIZON}
    hyperparams = {
        "lr": lr, "batch": batch, "max_epochs": max_epochs, "patience": patience,
        "epochs_run": history[-1]["epoch"] if history else 0, "best_epoch": best_epoch,
        "train_stride": train_stride, "eval_stride": eval_stride,
        "lookback": LOOKBACK, "horizonte": HORIZON,
    }
    split_dict = {"train": train_seeds, "val": val_seeds, "test": test_seeds}

    ckpt = {
        "state_dict": best_state,
        "version": VERSION,
        "model": "MirafloresGRUBaseline",
        "arch": arch,
        "scaler": scaler,
        "hyperparams": hyperparams,
        "split": split_dict,
        "seed": SEED,
        "device": DEV,
        "target_authority": TARGET_AUTHORITY,
    }
    ckpt_path = MODELS_DIR / "miraflores_gru_baseline.pt"
    torch.save(ckpt, ckpt_path)

    metadata = {
        "version": VERSION,
        "model": "MirafloresGRUBaseline",
        "arch": arch,
        "scaler": scaler,
        "hyperparams": hyperparams,
        "split": split_dict,
        "seed": SEED,
        "device": DEV,
        "target_authority": TARGET_AUTHORITY,
        "best_val_mse_std": round(best_val, 6),
        "history": history,
        "checkpoint": ckpt_path.name,
        "npz": str(Path(args.npz).relative_to(REPO_ROOT)) if str(args.npz).startswith(str(REPO_ROOT)) else str(args.npz),
    }
    meta_path = SCRIPTS_DIR / "miraflores_baseline_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    metrics_payload = {
        "version": VERSION,
        "units": "segundos (timeLoss total des-estandarizado)",
        "cuts_steps": list(cuts),
        "n_windows": {"test_all": int(test_index.shape[0]),
                      "test_081": int(is081.sum()), "test_normal": int(norm.sum())},
        "test_all": metrics_all,
        "test_081": metrics_081,
        "test_normal": metrics_normal,
    }
    metrics_path = SCRIPTS_DIR / "miraflores_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n")

    print(f"\n[baseline] checkpoint -> {ckpt_path}  (gitignored)")
    print(f"[baseline] metadata  -> {meta_path}  (versionado)")
    print(f"[baseline] metrics   -> {metrics_path}  (versionado)")
    print(f"[baseline] OK. best val(MSE std)={best_val:.5f} @ ep {best_epoch}. PARADA: revisar números antes de commitear.")


if __name__ == "__main__":
    main()
