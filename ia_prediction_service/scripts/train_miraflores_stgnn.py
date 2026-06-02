"""STGNN Fase 4 — entrenamiento del STGNN Time-then-Space (base B, DiffConv espacial).

Espejo **byte-a-byte** del baseline GRU de Fase 3 (``train_miraflores_baseline.py``):
mismo split, mismo scaler train-only FIJO, mismo índice de ventanas (mismos
``(day, start_t)``), misma loss (MSE enmascarado en espacio estandarizado), mismo
early stopping y las mismas métricas por horizonte. **La única variable nueva frente al
GRU es la componente espacial**: los 375 nodos van en el eje N (no colapsados en batch) y
entra ``DiffConv`` (k=2) entre el RNN y el decoder. Base B: solo capas ``tsl``
(``NodeEmbedding``/``RNN``/``DiffConv``/``MLPDecoder``); ningún engine ``Predictor`` ni
PyTorch-Lightning. El grafo es estático (``edge_index``/``edge_weight`` constantes del LCC).

Salida DESCARTABLE: por defecto a ``/tmp`` (configurable con ``--out-dir``), con nombres
propios (``miraflores_stgnn_*``). **NUNCA** escribe sobre los artefactos versionados del
baseline (``scripts/miraflores_baseline_*.json``, ``models/miraflores_gru_baseline.pt``);
hay un guardia explícito que lo impide. Esto es smoke de que ENTRENA; batir al baseline
es Fase 5.

Device: por defecto **cpu** (los kernels scatter de PyG/DiffConv no están completos en
Apple MPS); configurable con ``--device``. El device no afecta la paridad (split/scaler/
índice/loss/métricas son idénticos al baseline).

Uso (desde la raíz del repo, venv de training):
  ia_prediction_service/.venv/bin/python ia_prediction_service/scripts/train_miraflores_stgnn.py --quick
  ia_prediction_service/.venv/bin/python ia_prediction_service/scripts/train_miraflores_stgnn.py
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

# Los módulos del track viven en ia_prediction_service/src/. Apuntamos explícito.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "ia_prediction_service" / "src"
sys.path.insert(0, str(SRC))
from data.miraflores_split import get_split  # noqa: E402
from data.miraflores_windower import (  # noqa: E402
    DEFAULT_NPZ,
    HORIZON,
    LOOKBACK,
    build_graph_window_index,
    compute_timeloss_scaler,
    gather_graph_batch,
    load_lcc_edge_index,
    load_npz,
)
from evaluation.miraflores_metrics import (  # noqa: E402
    DEFAULT_CUTS,
    format_metrics_table,
    masked_metrics_by_horizon,
)
from models.time_then_space import TimeThenSpaceModel  # noqa: E402

# --- Config base (alineada al baseline; agrega hiperparámetros espaciales) ---
HIDDEN = 64
EMB = 10
GNN_KERNEL = 2
RNN_LAYERS = 1
LR = 1e-3
BATCH = 64            # snapshots de grafo por batch (cada uno = 375 nodos)
MAX_EPOCHS = 60
PATIENCE = 6          # épocas sin mejora de val antes de parar
TRAIN_STRIDE = 12     # idéntico al baseline (comparación justa)
EVAL_STRIDE = 12
SEED = 0
VERSION = "stgnn-fase4-timethenspace-b1"

TARGET_AUTHORITY = (
    "D-013 + enmienda 2026-06-01: target = timeLoss TOTAL por arista por intervalo "
    "de 60 s (columna timeLoss del Parquet, sin derivación per-vehículo)."
)

# Artefactos versionados del baseline que el STGNN NO debe tocar (guardia Paso 8).
BASELINE_PROTECTED = {
    (REPO_ROOT / "ia_prediction_service" / "scripts" / "miraflores_baseline_metadata.json").resolve(),
    (REPO_ROOT / "ia_prediction_service" / "scripts" / "miraflores_baseline_metrics.json").resolve(),
    (REPO_ROOT / "ia_prediction_service" / "models" / "miraflores_gru_baseline.pt").resolve(),
}


def set_seeds(seed: int) -> None:
    """Fija las seeds de random, numpy y torch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize_inputs(X: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Estandariza el canal 0 (timeLoss) y RE-ZERA celdas vacías vía el indicador.

    Idéntico al baseline; funciona con cualquier número de ejes líder. Aquí
    ``X``: ``[B, L, N, 2]``. Canal 0 → (timeLoss-mean)/std·validez; canal 1 (validez) intacto.
    """
    valid = X[..., 1]
    ch0 = (X[..., 0] - mean) / std * valid
    return torch.stack([ch0, valid], dim=-1)


def masked_mse_sums(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Devuelve (suma de errores²·máscara, suma de máscara) para promediar exacto."""
    m = mask.to(pred.dtype)
    sq = ((pred - target) ** 2 * m).sum()
    return sq, m.sum()


def run_epoch(model, opt, timeloss, mask, gindex, mean, std, batch, edge_index, edge_weight,
              dev, *, train: bool):
    """Una pasada sobre ``gindex``. Devuelve el MSE enmascarado exacto (espacio std).

    Espejo del baseline: shuffle determinista + backward si ``train``. Único agregado:
    ``edge_index``/``edge_weight`` constantes al forward; ``pred`` ``[B,H,N,1]`` → ``[B,H,N]``.
    """
    model.train(train)
    n = gindex.shape[0]
    order = np.random.permutation(n) if train else np.arange(n)
    sq_sum = 0.0
    mask_sum = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i in range(0, n, batch):
            rows = gindex[order[i : i + batch]]
            X, y, ymask = gather_graph_batch(timeloss, mask, rows)   # [B,L,N,2],[B,H,N],[B,H,N]
            Xt = torch.from_numpy(X).to(dev)
            yt = torch.from_numpy(y).to(dev)
            mt = torch.from_numpy(ymask).to(dev)
            Xt = standardize_inputs(Xt, mean, std)
            yt_std = (yt - mean) / std
            if train:
                opt.zero_grad()
            pred = model(Xt, edge_index, edge_weight).squeeze(-1)    # [B,H,N]
            sq, msum = masked_mse_sums(pred, yt_std, mt)
            denom = msum.clamp_min(1.0)
            loss = sq / denom
            if train:
                loss.backward()
                opt.step()
            sq_sum += float(sq.item())
            mask_sum += float(msum.item())
    return sq_sum / max(mask_sum, 1.0)


def predict_collect(model, timeloss, mask, gindex, mean, std, batch, edge_index, edge_weight, dev):
    """Predice sobre ``gindex`` y des-estandariza a segundos.

    Devuelve (pred_sec ``[B,H,N]``, y_raw ``[B,H,N]``, ymask ``[B,H,N]``).
    """
    model.eval()
    n = gindex.shape[0]
    preds, ys, ms = [], [], []
    with torch.no_grad():
        for i in range(0, n, batch):
            rows = gindex[i : i + batch]
            X, y, ymask = gather_graph_batch(timeloss, mask, rows)
            Xt = standardize_inputs(torch.from_numpy(X).to(dev), mean, std)
            pred_std = model(Xt, edge_index, edge_weight).squeeze(-1)   # [B,H,N]
            pred_sec = (pred_std * std + mean).cpu()
            preds.append(pred_sec)
            ys.append(torch.from_numpy(y))
            ms.append(torch.from_numpy(ymask))
    return torch.cat(preds, dim=0), torch.cat(ys, dim=0), torch.cat(ms, dim=0)


def to_metric_shape(t: torch.Tensor) -> torch.Tensor:
    """``[B, H, N]`` → ``[B·N, H]`` (cada nodo de cada snapshot es una fila de evaluación).

    Reusa ``masked_metrics_by_horizon`` (que espera ``[filas, H]``) sin tocarla.
    """
    b, h, n = t.shape
    return t.permute(0, 2, 1).reshape(b * n, h)


def main() -> None:
    parser = argparse.ArgumentParser(description="STGNN Fase 4 — Time-then-Space (DiffConv espacial)")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke: 1 época y stride grande para el gate end-to-end.")
    parser.add_argument("--epochs", type=int, default=None, help="Override de max épocas.")
    parser.add_argument("--patience", type=int, default=None, help="Override de paciencia.")
    parser.add_argument("--batch", type=int, default=None, help="Override de batch (snapshots).")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--hidden", type=int, default=HIDDEN, help="Tamaño hidden.")
    parser.add_argument("--emb", type=int, default=EMB, help="Tamaño embedding de nodo.")
    parser.add_argument("--gnn-kernel", type=int, default=GNN_KERNEL, help="K de DiffConv.")
    parser.add_argument("--train-stride", type=int, default=None, help="Stride de ventanas train.")
    parser.add_argument("--eval-stride", type=int, default=None, help="Stride de ventanas val/test.")
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ), help="Ruta del .npz Fase 2.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu (default; PyG/MPS incompleto) | mps | cuda.")
    parser.add_argument("--out-dir", type=str, default="/tmp/miraflores_stgnn",
                        help="Directorio DESCARTABLE de salida (ckpt/metadata/metrics).")
    args = parser.parse_args()

    quick = args.quick
    max_epochs = args.epochs if args.epochs is not None else (1 if quick else MAX_EPOCHS)
    patience = args.patience if args.patience is not None else (1 if quick else PATIENCE)
    batch = args.batch if args.batch is not None else (16 if quick else BATCH)
    train_stride = args.train_stride if args.train_stride is not None else (60 if quick else TRAIN_STRIDE)
    eval_stride = args.eval_stride if args.eval_stride is not None else (60 if quick else EVAL_STRIDE)
    lr = args.lr
    hidden = args.hidden
    emb = args.emb
    gnn_kernel = args.gnn_kernel
    dev = args.device

    set_seeds(SEED)
    print(f"[stgnn] device={dev} quick={quick} max_epochs={max_epochs} patience={patience} "
          f"batch={batch} lr={lr} hidden={hidden} emb={emb} gnn_kernel={gnn_kernel} "
          f"lookback={LOOKBACK} horizonte={HORIZON} train_stride={train_stride} eval_stride={eval_stride}")

    # --- Guardia: la salida NO puede pisar artefactos versionados del baseline ---
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "miraflores_stgnn.pt"
    meta_path = out_dir / "miraflores_stgnn_metadata.json"
    metrics_path = out_dir / "miraflores_stgnn_metrics.json"
    for p in (ckpt_path, meta_path, metrics_path):
        if p.resolve() in BASELINE_PROTECTED:
            raise SystemExit(f"[stgnn] ABORT: la salida {p} pisaría un artefacto versionado del baseline.")

    # --- Datos + split + scaler train-only FIJO (compartido byte-a-byte con el baseline) ---
    data = load_npz(args.npz)
    timeloss, mask, seeds = data["timeloss"], data["mask"], data["seeds"]
    n_days, n_t, n_nodes = timeloss.shape
    train_seeds, val_seeds, test_seeds = get_split()
    print(f"[stgnn] dataset {timeloss.shape} | split train/val/test = "
          f"{len(train_seeds)}/{len(val_seeds)}/{len(test_seeds)} (081 in test: {81 in test_seeds})")

    scaler = compute_timeloss_scaler(timeloss, mask, seeds, train_seeds)
    mean, std = scaler["mean"], scaler["std"]
    print(f"[stgnn] scaler train-only (timeLoss): mean={mean:.3f}s std={std:.3f}s")

    # --- Grafo estático (edge_index/edge_weight constantes del LCC) ---
    ei_np, ew_np = load_lcc_edge_index()
    edge_index = torch.from_numpy(ei_np).to(dev)
    edge_weight = torch.from_numpy(ew_np).to(dev)
    print(f"[stgnn] grafo LCC: edge_index {tuple(edge_index.shape)}  edge_weight {tuple(edge_weight.shape)}")

    # --- Índices de snapshots de grafo (mismos starts que el baseline) ---
    gidx_kw = dict(n_timesteps=n_t, lookback=LOOKBACK, horizon=HORIZON)
    train_gindex = build_graph_window_index(seeds, train_seeds, stride=train_stride, **gidx_kw)
    val_gindex = build_graph_window_index(seeds, val_seeds, stride=eval_stride, **gidx_kw)
    test_gindex = build_graph_window_index(seeds, test_seeds, stride=eval_stride, **gidx_kw)
    print(f"[stgnn] snapshots train/val/test = "
          f"{train_gindex.shape[0]:,}/{val_gindex.shape[0]:,}/{test_gindex.shape[0]:,} "
          f"(c/u = grafo de {n_nodes} nodos)")

    # --- Modelo / optimizador ---
    model = TimeThenSpaceModel(
        input_size=2, n_nodes=n_nodes, horizon=HORIZON,
        emb_size=emb, hidden_size=hidden, rnn_layers=RNN_LAYERS,
        gnn_kernel=gnn_kernel, output_size=1,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Loop con early stopping sobre val (idéntico al baseline) ---
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict] = []
    print(f"[stgnn] entrenando (MSE enmascarado, espacio estandarizado)…")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, opt, timeloss, mask, train_gindex, mean, std, batch,
                               edge_index, edge_weight, dev, train=True)
        val_loss = run_epoch(model, opt, timeloss, mask, val_gindex, mean, std, batch,
                             edge_index, edge_weight, dev, train=False)
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
            print(f"[stgnn] early stopping en ep {epoch} (sin mejora hace {patience}); "
                  f"mejor val={best_val:.5f} @ ep {best_epoch}")
            break

    # Restaurar mejor modelo.
    model.load_state_dict(best_state)

    # --- Evaluación en TEST (des-estandarizada, segundos) ---
    pred_sec, y_raw, ymask = predict_collect(model, timeloss, mask, test_gindex, mean, std, batch,
                                             edge_index, edge_weight, dev)
    # [B,H,N] → [B·N,H] para reusar masked_metrics_by_horizon tal cual.
    pred_m = to_metric_shape(pred_sec)
    y_m = to_metric_shape(y_raw)
    mask_m = to_metric_shape(ymask)
    snap_seeds = np.asarray([int(seeds[d]) for d in test_gindex[:, 0]])  # [B]
    row_seeds = np.repeat(snap_seeds, n_nodes)                            # [B·N], alineado a permute(0,2,1)
    is081 = torch.from_numpy(row_seeds == 81)
    cuts = DEFAULT_CUTS

    metrics_all = masked_metrics_by_horizon(pred_m, y_m, mask_m, cuts)
    metrics_081 = masked_metrics_by_horizon(pred_m[is081], y_m[is081], mask_m[is081], cuts)
    norm = ~is081
    metrics_normal = masked_metrics_by_horizon(pred_m[norm], y_m[norm], mask_m[norm], cuts)

    print("\n[stgnn] ===== MÉTRICAS TEST (des-estandarizadas, segundos) =====")
    print(f"\nTEST completo (10 días, n_filas={pred_m.shape[0]:,}):")
    print(format_metrics_table(metrics_all, cuts))
    print(f"\nDía 081 (congestión máxima, n_filas={int(is081.sum())}):")
    print(format_metrics_table(metrics_081, cuts))
    print(f"\nDías normales de test (9 días, n_filas={int(norm.sum())}):")
    print(format_metrics_table(metrics_normal, cuts))

    # --- Persistencia (DESCARTABLE; nunca pisa los artefactos del baseline) ---
    arch = {"input_size": 2, "output_size": 1, "hidden": hidden, "emb": emb,
            "gnn_kernel": gnn_kernel, "rnn_layers": RNN_LAYERS, "horizonte": HORIZON,
            "n_nodes": n_nodes}
    hyperparams = {
        "lr": lr, "batch": batch, "max_epochs": max_epochs, "patience": patience,
        "epochs_run": history[-1]["epoch"] if history else 0, "best_epoch": best_epoch,
        "train_stride": train_stride, "eval_stride": eval_stride,
        "lookback": LOOKBACK, "horizonte": HORIZON,
    }
    split_dict = {"train": train_seeds, "val": val_seeds, "test": test_seeds}

    ckpt = {
        "state_dict": best_state, "version": VERSION, "model": "TimeThenSpaceModel",
        "arch": arch, "scaler": scaler, "hyperparams": hyperparams, "split": split_dict,
        "seed": SEED, "device": dev, "target_authority": TARGET_AUTHORITY,
    }
    torch.save(ckpt, ckpt_path)

    metadata = {
        "version": VERSION, "model": "TimeThenSpaceModel", "base": "B (solo capas tsl)",
        "arch": arch, "scaler": scaler, "hyperparams": hyperparams, "split": split_dict,
        "seed": SEED, "device": dev, "target_authority": TARGET_AUTHORITY,
        "best_val_mse_std": round(best_val, 6), "history": history, "checkpoint": ckpt_path.name,
        "quick": quick,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    metrics_payload = {
        "version": VERSION, "units": "segundos (timeLoss total des-estandarizado)",
        "cuts_steps": list(cuts),
        "n_rows": {"test_all": int(pred_m.shape[0]),
                   "test_081": int(is081.sum()), "test_normal": int(norm.sum())},
        "test_all": metrics_all, "test_081": metrics_081, "test_normal": metrics_normal,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n")

    print(f"\n[stgnn] checkpoint -> {ckpt_path}  (descartable)")
    print(f"[stgnn] metadata  -> {meta_path}  (descartable)")
    print(f"[stgnn] metrics   -> {metrics_path}  (descartable)")
    print(f"[stgnn] OK. best val(MSE std)={best_val:.5f} @ ep {best_epoch}. "
          f"Smoke de que ENTRENA; batir al baseline es Fase 5.")


if __name__ == "__main__":
    main()
