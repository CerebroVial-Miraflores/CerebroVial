"""Pre/post-procesamiento de inferencia GRU baseline (Fase 3).

VENDORIZADO de ``ia_prediction_service/src/inference/preprocessing.py`` (D-010, patrón
TTH-09). Copia byte-a-byte. Mantener en sync con el original; el gate de reproducción
verifica que el camino vendorizado produce la misma salida que el loader de Fase 1.

Contrato del scaler: escalar global ``{mean, std}`` (un solo par para las 1660 aristas),
aplicado SOLO al canal 0 (``timeLoss``); el canal 1 (indicador de validez 1/0) queda
intacto. Operan sobre tensores torch float32 para reproducir el cómputo del eval exacto.
El scaler viaja DENTRO del checkpoint (``ckpt["scaler"]``), no hay archivo aparte.
"""
from __future__ import annotations

import torch


def standardize_window(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Estandariza el canal 0 (``timeLoss``) y RE-ZERA vacíos vía el indicador.

    Espejo exacto de ``standardize_inputs`` de los scripts de training: canal 0 →
    ``(timeLoss - mean) / std * validez`` (las celdas vacías, validez=0, vuelven a 0);
    canal 1 (validez) NO se escala. Funciona con cualquier número de ejes líder:
    ``x[..., 2] -> [..., 2]`` (sirve para ``[L, N, 2]``, ``[N, L, 2]``, ``[B, L, N, 2]``).
    """
    valid = x[..., 1]
    ch0 = (x[..., 0] - mean) / std * valid
    return torch.stack([ch0, valid], dim=-1)


def destandardize(pred_std: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Des-estandariza la predicción a segundos: ``pred_std * std + mean``."""
    return pred_std * std + mean
