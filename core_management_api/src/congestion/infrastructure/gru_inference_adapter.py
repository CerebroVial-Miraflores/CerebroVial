"""Adaptador de inferencia GRU baseline (camino GRU-only, sin tsl) — Fase 3.

VENDORIZADO de ``ia_prediction_service/src/inference/adapters.py`` (D-010, patrón TTH-09),
pero SOLO el camino GRU: la base ``InferenceAdapter`` + ``GruAdapter``. El original importa
``TimeThenSpaceModel`` (STGNN) y ``load_lcc_edge_index`` en el top del módulo, que arrastran
``tsl`` — esos imports NO se vendorizan acá (el core sirve solo el baseline GRU). El
``StgnnAdapter`` queda fuera por construcción.

Contrato (idéntico al original): ``predict(window [30, 1660, 2]) -> [1660, 30]`` en segundos,
orden ``node_index``. El scaler viaja dentro del checkpoint (``ckpt["scaler"]``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from cerebrovial_shared.lfs_check import assert_real_binary

from .gru_inference_model import MirafloresGRUBaseline
from .gru_inference_preprocessing import destandardize, standardize_window

_SUPPORTED_MODEL = "MirafloresGRUBaseline"

N_NODES = 1660   # aristas-como-nodo del LCC net v2 (orden node_index, invariante sagrada)
LOOKBACK = 30
HORIZON = 30


class InferenceAdapter(ABC):
    """Contrato común de inferencia. Subclases: una por familia de modelo.

    ``predict(window [30, 1660, 2]) -> [1660, 30]`` en segundos, orden ``node_index``.
    """

    def __init__(self, model: torch.nn.Module, mean: float, std: float, device: str = "cpu"):
        self.model = model
        self.mean = float(mean)
        self.std = float(std)
        self.device = device
        self.model.eval()

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, ckpt: dict, device: str = "cpu") -> "InferenceAdapter":
        """Reconstruye el modelo desde ``ckpt`` (``arch``/``state_dict``/``scaler``)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, window: np.ndarray) -> np.ndarray:
        """``window [30, 1660, 2]`` (segundos crudos) -> ``[1660, 30]`` segundos."""
        raise NotImplementedError

    @staticmethod
    def _check_window(window: np.ndarray) -> np.ndarray:
        """Valida el shape del contrato y devuelve un float32 contiguo."""
        w = np.asarray(window, dtype=np.float32)
        if w.shape != (LOOKBACK, N_NODES, 2):
            raise ValueError(
                f"window debe ser [{LOOKBACK}, {N_NODES}, 2] (lookback, aristas, canales); "
                f"got {w.shape}"
            )
        return np.ascontiguousarray(w)


class GruAdapter(InferenceAdapter):
    """Baseline GRU univariado (``MirafloresGRUBaseline``). Nodo-agnóstico, sin grafo.

    El modelo ve una serie por nodo. Convierte el snapshot de grafo ``[30, 1660, 2]`` en
    1660 series ``[1660, 30, 2]`` (eje 0 = ``node_index`` por construcción del window),
    forward batcheado ``[1660, 30, 2] -> [1660, 30]``, des-estandariza. Sin reordenamiento:
    el eje 1660 del window YA está en orden ``node_index``.
    """

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str = "cpu") -> "GruAdapter":
        arch = ckpt["arch"]
        # arch matchea 1:1 el ctor: {input_size, hidden, horizonte}.
        model = MirafloresGRUBaseline(**arch).to(device)
        model.load_state_dict(ckpt["state_dict"])
        scaler = ckpt["scaler"]
        return cls(model, scaler["mean"], scaler["std"], device=device)

    def predict(self, window: np.ndarray) -> np.ndarray:
        w = self._check_window(window)                          # [30, 1660, 2]
        x = torch.from_numpy(w).to(self.device)                 # [30, 1660, 2]
        x = standardize_window(x, self.mean, self.std)          # canal 0 estandarizado
        # [30, 1660, 2] -> [1660, 30, 2]: una serie temporal por nodo (eje 0 = node_index).
        x = x.permute(1, 0, 2).contiguous()                     # [1660, 30, 2]
        with torch.no_grad():
            pred_std = self.model(x)                            # [1660, 30]
        pred_sec = destandardize(pred_std, self.mean, self.std)
        return pred_sec.cpu().numpy().astype(np.float32)        # [1660, 30]


def load_gru_adapter(ckpt_path: Path | str, device: str = "cpu") -> GruAdapter:
    """Carga un ``.pt`` del baseline GRU y devuelve el ``GruAdapter`` listo.

    GRU-only: a diferencia del loader de Fase 1 (registry string→adaptador), acá solo se
    sirve ``MirafloresGRUBaseline``; cualquier otro ``ckpt["model"]`` es error explícito (no
    se arrastra el STGNN ni tsl). Guarda LFS: rechaza punteros sin materializar.
    ``weights_only=False`` es necesario (el ckpt es un dict con arch/scaler/metadata).
    """
    assert_real_binary(str(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model")
    if model_name != _SUPPORTED_MODEL:
        raise ValueError(
            f"ckpt['model']={model_name!r} no soportado por el core (Fase 3 GRU-only); "
            f"esperado {_SUPPORTED_MODEL!r}."
        )
    return GruAdapter.from_checkpoint(ckpt, device=device)
