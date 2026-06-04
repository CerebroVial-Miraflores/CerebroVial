"""Entrada única del loader de inferencia model-agnostic.

``load_inference_adapter(ckpt_path)`` lee un ``.pt`` (GRU baseline o STGNN) por ruta
EXPLÍCITA, elige el adaptador según ``ckpt["model"]`` vía el registry y lo reconstruye. La
selección de QUÉ checkpoint servir no es responsabilidad de esta capa: el loader carga la
ruta que se le pase.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .adapters import InferenceAdapter
from .registry import resolve


def load_inference_adapter(ckpt_path: Path | str, device: str = "cpu") -> InferenceAdapter:
    """Carga un ``.pt`` y devuelve el adaptador de inferencia correspondiente.

    - Lee el ckpt con ``weights_only=False`` (es un dict con ``arch``/``scaler``/… además
      del ``state_dict``).
    - Valida que ``ckpt["model"]`` esté en el registry (error claro si no; no adivina).
    - Reconstruye el modelo con el remapeo de ``arch`` propio de cada adaptador.

    Args:
        ckpt_path: ruta al ``.pt`` (GRU o STGNN).
        device: device de inferencia (default ``"cpu"``).

    Returns:
        ``InferenceAdapter`` listo para ``predict(window [30, 1660, 2]) -> [1660, 30]``.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise ValueError(
            f"ckpt en {ckpt_path} no tiene la llave 'model'; no es un checkpoint de "
            f"entrenamiento de este track (esperado dict con model/arch/scaler/state_dict)."
        )
    adapter_cls = resolve(ckpt["model"])
    return adapter_cls.from_checkpoint(ckpt, device=device)
