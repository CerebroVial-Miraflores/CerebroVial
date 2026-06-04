"""Single-window builder para inferencia GRU baseline (Fase 3).

Reproduce el contrato de ``gather_graph_batch``
(``ia_prediction_service/src/data/miraflores_windower.py``) para UNA sola ventana, sobre el
slice de un día ya seleccionado. Numpy-puro, sin torch, sin tsl.

Dado el slice del día (``timeloss[1440, 1660]`` + ``mask[1440, 1660]`` en orden
``node_index``) y un corte ``t``, produce la ventana ``[30, 1660, 2]`` que el modelo vio en
training:
- canal 0 = ``timeLoss`` con celda vacía → 0 (y NaN residual → 0),
- canal 1 = indicador de validez (1=válido, 0=vacío).

La ventana cubre ``t-29 .. t`` inclusive (lookback=30 terminando en ``t``). El gate de
fidelidad (``test_prediction_window_gate.py``) verifica que esto es byte-igual a
``gather_graph_batch`` para el mismo ``(day_idx, start=t-29)`` sobre el ``.npz`` original.
"""
from __future__ import annotations

import numpy as np

LOOKBACK = 30
N_TIMESTEPS = 1440  # pasos de 60 s del día (0..1439 = minuto del día)


def build_window(timeloss: np.ndarray, mask: np.ndarray, t: int) -> np.ndarray:
    """Arma la ventana ``[30, 1660, 2]`` cortada en ``t`` (cubre ``t-29..t`` inclusive).

    Args:
        timeloss: ``[1440, 1660]`` float — ``timeLoss`` (s) del día, NaN en celdas vacías.
        mask: ``[1440, 1660]`` bool — True=válido, False=vacío.
        t: corte (último paso del lookback). Válido en ``[LOOKBACK-1, N_TIMESTEPS-1]``
           = ``[29, 1439]`` (la ventana no puede empezar antes de 0).

    Returns:
        ``np.ndarray [30, 1660, 2]`` float32 — canal 0 timeLoss (vacío→0), canal 1 validez.
    """
    if timeloss.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            f"timeloss y mask deben ser [T, N]; got {timeloss.shape} / {mask.shape}"
        )
    if timeloss.shape != mask.shape:
        raise ValueError(f"timeloss {timeloss.shape} != mask {mask.shape}")
    t = int(t)
    if not (LOOKBACK - 1 <= t <= N_TIMESTEPS - 1):
        raise ValueError(
            f"t={t} fuera de rango [{LOOKBACK - 1}, {N_TIMESTEPS - 1}]: "
            "la ventana t-29..t debe caer dentro del día."
        )

    start = t - LOOKBACK + 1            # inclusive; start..t = 30 pasos
    in_vals = timeloss[start : t + 1, :]   # [30, 1660]
    in_mask = mask[start : t + 1, :]       # [30, 1660] bool

    # Misma semántica de canales que gather_graph_batch (windower.py:256-260):
    # canal 0 = timeLoss con vacío→0 y NaN residual→0; canal 1 = validez.
    ch0 = np.where(in_mask, in_vals, 0.0).astype(np.float32)
    ch0 = np.nan_to_num(ch0, copy=False)
    ch1 = in_mask.astype(np.float32)
    return np.stack([ch0, ch1], axis=-1)   # [30, 1660, 2]
