"""Windower univariado solo-temporal para el baseline GRU (STGNN Fase 3).

Lee el ``.npz`` consolidado de Fase 2 (``tensor [60,1440,375,1]`` + ``mask`` +
``seeds``) y genera ventanas ``lookback=30 / horizonte=30`` **por día y por nodo,
sin cruzar el borde entre días** (el eje 0 son días separados por seed, nunca una
serie continua). Univariado: cada nodo genera sus ventanas independientes, sin ver
vecinos (eso lo agrega el STGNN en Fase 4).

Diseño **memory-safe**: NO materializa las ~22 M ventanas (≈5 GB). Mantiene el
tensor residente en RAM (~130 MB) y construye un **índice** ``[(day_idx, node_idx,
start_t)]`` por fold; ``gather_batch`` arma cada batch por *slicing* bajo demanda.
El trainer hace ``randperm`` sobre las filas del índice.

Por ventana:
- input ``[lookback, 2]``: canal 0 = ``timeLoss`` (celda vacía → 0), canal 1 =
  indicador de validez (1=válido, 0=vacío, desde ``mask``).
- target ``[horizonte]`` = ``timeLoss`` de los pasos siguientes (vacío → 0).
- target_mask ``[horizonte]`` = validez de esos pasos (para enmascarar el loss).

Entrega **crudo (segundos)**; la estandarización (train-only) la aplica el trainer.
``compute_timeloss_scaler`` calcula media/std sobre celdas válidas del TRAIN, para
que el trainer y la Fase 4 usen el idéntico scaler.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Repo root: data → src → ia_prediction_service → CerebroVial
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NPZ = (
    REPO_ROOT
    / "simulation"
    / "data"
    / "datasets"
    / "miraflores_laborable_60d"
    / "tensors"
    / "miraflores_laborable_60d.npz"
)

LOOKBACK = 30  # pasos de 60 s (D-011/TTH-11: 30 min)
HORIZON = 30   # pasos de 60 s (30 min multi-output)


def load_npz(path: Path | str = DEFAULT_NPZ) -> dict:
    """Carga el ``.npz`` de Fase 2 y devuelve ``timeloss``/``mask``/``seeds``.

    - ``timeloss`` ``[D, T, N]`` float32 — canal 0 del tensor (NaN en celdas vacías).
    - ``mask`` ``[D, T, N]`` bool — True=válido, False=vacío.
    - ``seeds`` ``[D]`` int — seed por índice de día (eje 0).
    """
    z = np.load(path)
    tensor = z["tensor"]              # [D, T, N, 1]
    timeloss = np.ascontiguousarray(tensor[..., 0].astype(np.float32))  # [D, T, N]
    mask = np.ascontiguousarray(z["mask"].astype(bool))                 # [D, T, N]
    seeds = z["seeds"].astype(int)                                      # [D]
    return {"timeloss": timeloss, "mask": mask, "seeds": seeds}


def fold_day_indices(seeds_array: np.ndarray, fold_seeds) -> np.ndarray:
    """Índices de día (eje 0) cuyas seeds pertenecen a ``fold_seeds``, ordenados."""
    fold = set(int(s) for s in fold_seeds)
    return np.asarray(
        [di for di, s in enumerate(np.asarray(seeds_array).tolist()) if int(s) in fold],
        dtype=np.int64,
    )


def build_window_index(
    seeds_array: np.ndarray,
    fold_seeds,
    *,
    n_nodes: int,
    n_timesteps: int,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    stride: int = 1,
) -> np.ndarray:
    """Índice ``[N, 3]`` int32 de ventanas ``(day_idx, start_t, node_idx)`` del fold.

    Las ventanas NO cruzan el borde entre días: ``start_t`` recorre
    ``0 .. n_timesteps - lookback - horizon`` (inclusive) DENTRO de cada día.
    Orden determinista: día → start_t → nodo.
    """
    days = fold_day_indices(seeds_array, fold_seeds)
    last_start = n_timesteps - lookback - horizon  # inclusive
    if last_start < 0:
        return np.empty((0, 3), dtype=np.int32)
    starts = np.arange(0, last_start + 1, stride, dtype=np.int32)
    nodes = np.arange(n_nodes, dtype=np.int32)

    # Producto cartesiano (día, start, nodo) sin bucles Python sobre las ~22M filas.
    sa, na = np.meshgrid(starts, nodes, indexing="ij")  # [S, N]
    starts_flat = sa.reshape(-1)
    nodes_flat = na.reshape(-1)
    per_day = starts_flat.shape[0]

    rows = np.empty((days.shape[0] * per_day, 3), dtype=np.int32)
    for k, di in enumerate(days):
        sl = slice(k * per_day, (k + 1) * per_day)
        rows[sl, 0] = di
        rows[sl, 1] = starts_flat
        rows[sl, 2] = nodes_flat
    return rows


def gather_batch(
    timeloss: np.ndarray,
    mask: np.ndarray,
    index_rows: np.ndarray,
    *,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arma un batch por *slicing* desde el tensor residente.

    Devuelve (crudo, en segundos):
    - ``X`` ``[B, lookback, 2]`` float32 — canal 0 timeLoss (vacío→0), canal 1 validez.
    - ``y`` ``[B, horizon]`` float32 — timeLoss objetivo (vacío→0).
    - ``ymask`` ``[B, horizon]`` float32 — 1=válido, 0=inválido (para enmascarar loss).
    """
    rows = np.asarray(index_rows)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(f"index_rows debe ser [N,3], got {rows.shape}")
    di = rows[:, 0].astype(np.int64)[:, None]    # [B,1]
    t0 = rows[:, 1].astype(np.int64)[:, None]    # [B,1]
    no = rows[:, 2].astype(np.int64)[:, None]    # [B,1]

    in_t = t0 + np.arange(lookback, dtype=np.int64)[None, :]            # [B, lookback]
    tg_t = t0 + lookback + np.arange(horizon, dtype=np.int64)[None, :]  # [B, horizon]

    in_vals = timeloss[di, in_t, no]   # [B, lookback]
    in_mask = mask[di, in_t, no]       # [B, lookback] bool
    tg_vals = timeloss[di, tg_t, no]   # [B, horizon]
    tg_mask = mask[di, tg_t, no]       # [B, horizon] bool

    # Canal 0: timeLoss con celda vacía → 0 (y cualquier NaN residual → 0).
    ch0 = np.where(in_mask, in_vals, 0.0).astype(np.float32)
    ch0 = np.nan_to_num(ch0, copy=False)
    ch1 = in_mask.astype(np.float32)
    X = np.stack([ch0, ch1], axis=-1)  # [B, lookback, 2]

    y = np.where(tg_mask, tg_vals, 0.0).astype(np.float32)  # vacío → 0
    y = np.nan_to_num(y, copy=False)
    ymask = tg_mask.astype(np.float32)
    return X, y, ymask


def compute_timeloss_scaler(
    timeloss: np.ndarray,
    mask: np.ndarray,
    seeds_array: np.ndarray,
    train_seeds,
) -> dict:
    """Media/std de ``timeLoss`` sobre celdas **válidas del TRAIN** (train-only).

    Devuelve ``{"mean": float, "std": float}``. ``std`` se piso-limita a 1.0 si la
    varianza es degenerada (evita división por ~0 al estandarizar).
    """
    days = fold_day_indices(seeds_array, train_seeds)
    if days.size == 0:
        raise ValueError("train_seeds no mapea a ningún día del dataset.")
    sub_t = timeloss[days]          # [Dtr, T, N]
    sub_m = mask[days]             # [Dtr, T, N]
    vals = sub_t[sub_m]            # 1D, sólo válidas
    vals = vals[~np.isnan(vals)]  # guardia: descarta NaN residual
    if vals.size == 0:
        raise ValueError("No hay celdas válidas en TRAIN para calcular el scaler.")
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return {"mean": mean, "std": std}
