"""Pre/post-procesamiento de inferencia — re-extracción de la lógica del eval offline.

Estas dos funciones son la **misma lógica** que usan los scripts de training
(``standardize_inputs`` en ``scripts/train_miraflores_baseline.py`` y
``train_miraflores_stgnn.py``, más la des-estandarización de ``predict_collect``), pero
RE-EXTRAÍDA acá para que el loader de inferencia no dependa de los scripts. La igualdad
byte-a-byte con el eval offline se verifica en ``tests/test_inference_loader.py`` (gate de
reproducción de Fase 1).

Contrato del scaler (auditoría de contrato de inferencia): escalar global
``{mean, std}`` (un solo par para las 1660 aristas), aplicado SOLO al canal 0
(``timeLoss``); el canal 1 (indicador de validez 1/0) queda intacto. Operan sobre tensores
torch float32 para reproducir el cómputo del eval exactamente.
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
    """Des-estandariza la predicción a segundos: ``pred_std * std + mean``.

    Espejo de la línea ``pred_sec = pred_std * std + mean`` de ``predict_collect``.
    """
    return pred_std * std + mean
