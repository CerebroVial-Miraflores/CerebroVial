"""Adaptadores de inferencia model-agnostic (GRU baseline / STGNN Time-then-Space).

Una base abstracta ``InferenceAdapter`` con contrato único —``predict(window) ->
[1660, 30]`` en segundos, orden ``node_index``— y dos implementaciones que absorben las
TRES asimetrías de los modelos congelados (auditoría de contrato de inferencia):

1. firma del forward (GRU sin grafo / STGNN con ``edge_index``/``edge_weight``),
2. layout de entrada (GRU: una serie por nodo, nodo colapsado en batch / STGNN: snapshot
   de grafo con los nodos en el eje N),
3. reshape de salida.

El caller ve una sola interfaz; la diferencia GRU-vs-STGNN queda ENTERAMENTE adentro del
adaptador. No se tocan las clases de modelo (``MirafloresGRUBaseline``,
``TimeThenSpaceModel``): se importan y se reconstruyen desde el ``arch`` del ckpt.

Contrato de entrada/salida (idéntico para ambos):
- ``predict(window)``: ``window`` ``np.ndarray [30, 1660, 2]`` (30 pasos lookback, 1660
  aristas, 2 canales: timeLoss + validez), crudo en segundos.
- devuelve ``np.ndarray [1660, 30]`` float32: ``timeLoss`` predicho en segundos por arista
  (eje 0 = ``node_index`` 0..1659) × 30 pasos de horizonte (eje 1).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from ..data.miraflores_windower import load_lcc_edge_index
from ..models.miraflores_gru_baseline import MirafloresGRUBaseline
from ..models.time_then_space import TimeThenSpaceModel
from .preprocessing import destandardize, standardize_window

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


class StgnnAdapter(InferenceAdapter):
    """STGNN Time-then-Space (``TimeThenSpaceModel``). Forward de grafo.

    Procesa el snapshot completo en un forward. Añade el eje batch ``[1, 30, 1660, 2]``,
    pasa el grafo estático (``edge_index``/``edge_weight`` cacheados del LCC), squeeze y
    transpone a ``[1660, 30]``. Orden ``node_index`` nativo del eje N del tensor.
    """

    def __init__(self, model, mean, std, edge_index, edge_weight, device="cpu"):
        super().__init__(model, mean, std, device=device)
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str = "cpu") -> "StgnnAdapter":
        arch = ckpt["arch"]
        # arch NO matchea el ctor: remapear nombres. rnn_cell/dropout NO están en arch
        # (el training nunca los pasó) -> defaults del ctor ('gru' / 0.0).
        model = TimeThenSpaceModel(
            input_size=arch["input_size"],
            n_nodes=arch["n_nodes"],
            horizon=arch["horizonte"],      # horizonte -> horizon
            emb_size=arch["emb"],           # emb       -> emb_size
            hidden_size=arch["hidden"],     # hidden    -> hidden_size
            rnn_layers=arch["rnn_layers"],
            gnn_kernel=arch["gnn_kernel"],
            output_size=arch["output_size"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        scaler = ckpt["scaler"]
        # Grafo estático del LCC (mismo camino que el eval: load_lcc_edge_index()).
        ei_np, ew_np = load_lcc_edge_index()
        edge_index = torch.from_numpy(ei_np).to(device)
        edge_weight = torch.from_numpy(ew_np).to(device)
        return cls(model, scaler["mean"], scaler["std"], edge_index, edge_weight, device=device)

    def predict(self, window: np.ndarray) -> np.ndarray:
        w = self._check_window(window)                          # [30, 1660, 2]
        x = torch.from_numpy(w).to(self.device)
        x = standardize_window(x, self.mean, self.std)
        x = x.unsqueeze(0)                                      # [1, 30, 1660, 2]
        with torch.no_grad():
            out = self.model(x, self.edge_index, self.edge_weight)  # [1, 30, 1660, 1]
        pred_std = out.squeeze(-1).squeeze(0)                  # [30, 1660]
        pred_std = pred_std.permute(1, 0).contiguous()        # [1660, 30] (eje N -> filas)
        pred_sec = destandardize(pred_std, self.mean, self.std)
        return pred_sec.cpu().numpy().astype(np.float32)        # [1660, 30]
