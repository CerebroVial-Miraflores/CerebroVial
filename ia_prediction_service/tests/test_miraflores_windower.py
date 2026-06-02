"""Tests del windower univariado (miraflores_windower.py).

Verifica: las ventanas NO cruzan el borde entre días; el input tiene 2 canales
(timeLoss + indicador de validez); celda vacía → canal0=0 y canal1=0; shapes
correctos; el scaler train-only ignora celdas inválidas.

Fixtures sintéticas en memoria (no tocan disco). Corre con el venv raíz:
``cd ia_prediction_service && .venv/bin/python -m pytest tests/test_miraflores_windower.py``
"""
from __future__ import annotations

import numpy as np

from src.data.miraflores_windower import (
    build_window_index,
    compute_timeloss_scaler,
    fold_day_indices,
    gather_batch,
)

NAN = float("nan")


def _tiny_dataset():
    """2 días (seeds 42,43), T=8, N=2. Día 0 nodo 1 con una celda vacía en t=5."""
    D, T, N = 2, 8, 2
    timeloss = np.zeros((D, T, N), dtype=np.float32)
    mask = np.ones((D, T, N), dtype=bool)
    # Día 0 (seed 42)
    timeloss[0, :, 0] = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.float32)
    timeloss[0, :, 1] = np.array([20, 21, 22, 23, 24, NAN, 26, 27], dtype=np.float32)
    mask[0, 5, 1] = False  # celda vacía
    # Día 1 (seed 43): valores distintos (no deben mezclarse con el día 0)
    timeloss[1, :, 0] = np.array([100, 101, 102, 103, 104, 105, 106, 107], dtype=np.float32)
    timeloss[1, :, 1] = np.array([200, 201, 202, 203, 204, 205, 206, 207], dtype=np.float32)
    seeds = np.array([42, 43], dtype=np.int32)
    return timeloss, mask, seeds, T, N


def test_fold_day_indices():
    _, _, seeds, _, _ = _tiny_dataset()
    assert fold_day_indices(seeds, [42]).tolist() == [0]
    assert fold_day_indices(seeds, [43]).tolist() == [1]
    assert fold_day_indices(seeds, [42, 43]).tolist() == [0, 1]


def test_ventanas_no_cruzan_dias():
    timeloss, mask, seeds, T, N = _tiny_dataset()
    lookback, horizon = 3, 2
    # Fold = sólo día 0 (seed 42).
    idx = build_window_index(seeds, [42], n_nodes=N, n_timesteps=T,
                             lookback=lookback, horizon=horizon)
    # Todas las filas apuntan al día 0 (ninguna al día 1).
    assert set(idx[:, 0].tolist()) == {0}
    # Ninguna ventana se sale del día: start + lookback + horizon <= T.
    assert np.all(idx[:, 1] + lookback + horizon <= T)
    # Conteo: n_starts (0..T-L-H) * n_nodes.
    n_starts = T - lookback - horizon + 1  # 8-3-2+1 = 4
    assert idx.shape[0] == n_starts * N

    # Fold de ambos días: día 1 también aparece, sin que ninguna ventana cruce.
    idx2 = build_window_index(seeds, [42, 43], n_nodes=N, n_timesteps=T,
                              lookback=lookback, horizon=horizon)
    assert set(idx2[:, 0].tolist()) == {0, 1}
    assert idx2.shape[0] == 2 * n_starts * N


def test_input_dos_canales_y_celda_vacia():
    timeloss, mask, seeds, T, N = _tiny_dataset()
    lookback, horizon = 3, 2
    # Ventana explícita: día 0, start=3, nodo 1 → input steps 3,4,5 (5 es vacío).
    rows = np.array([[0, 3, 1]], dtype=np.int32)
    X, y, ymask = gather_batch(timeloss, mask, rows, lookback=lookback, horizon=horizon)

    # 2 canales en el input.
    assert X.shape == (1, lookback, 2)
    # canal 0 = timeLoss (vacío→0), canal 1 = indicador validez.
    assert X[0, :, 0].tolist() == [23.0, 24.0, 0.0]   # step 5 vacío → 0
    assert X[0, :, 1].tolist() == [1.0, 1.0, 0.0]      # step 5 → indicador 0
    # Celda vacía: canal0==0 Y canal1==0.
    assert X[0, 2, 0] == 0.0 and X[0, 2, 1] == 0.0
    # Sin NaN en el input.
    assert not np.isnan(X).any()

    # target = steps 6,7 (válidos) → 26,27; ymask = 1,1.
    assert y.shape == (1, horizon)
    assert ymask.shape == (1, horizon)
    assert y[0].tolist() == [26.0, 27.0]
    assert ymask[0].tolist() == [1.0, 1.0]


def test_target_vacio_se_enmascara():
    timeloss, mask, seeds, T, N = _tiny_dataset()
    lookback, horizon = 3, 3
    # Ventana día 0, start=3, nodo1 → target steps 6,7 y... start+L=6, horizon3 → 6,7,?
    # Usamos start=2: input 2,3,4; target 5,6,7 (step5 vacío).
    rows = np.array([[0, 2, 1]], dtype=np.int32)
    X, y, ymask = gather_batch(timeloss, mask, rows, lookback=lookback, horizon=horizon)
    # target step5 vacío → y=0 y ymask=0 en esa posición.
    assert ymask[0].tolist() == [0.0, 1.0, 1.0]
    assert y[0, 0] == 0.0           # vacío → 0 (no NaN)
    assert not np.isnan(y).any()


def test_shapes_batch():
    timeloss, mask, seeds, T, N = _tiny_dataset()
    lookback, horizon = 3, 2
    idx = build_window_index(seeds, [42], n_nodes=N, n_timesteps=T,
                             lookback=lookback, horizon=horizon)
    X, y, ymask = gather_batch(timeloss, mask, idx, lookback=lookback, horizon=horizon)
    B = idx.shape[0]
    assert X.shape == (B, lookback, 2)
    assert y.shape == (B, horizon)
    assert ymask.shape == (B, horizon)
    assert X.dtype == np.float32


def test_scaler_train_only_ignora_invalidas():
    timeloss, mask, seeds, T, N = _tiny_dataset()
    # Scaler sobre día 0 (seed 42): usa SOLO celdas válidas (excluye la vacía t=5,n=1).
    scaler = compute_timeloss_scaler(timeloss, mask, seeds, [42])
    valid_vals = timeloss[0][mask[0]]
    valid_vals = valid_vals[~np.isnan(valid_vals)]
    assert abs(scaler["mean"] - float(valid_vals.mean())) < 1e-5
    assert abs(scaler["std"] - float(valid_vals.std())) < 1e-5
    # std > 0 (datos con varianza) → no se piso-limita.
    assert scaler["std"] > 1.0
