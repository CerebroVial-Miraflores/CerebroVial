"""Fuente del día para la predicción servida: carga el slice seed051 y corta ventanas.

Carga EAGER al startup el artefacto auto-contenido
(``core_management_api/models/day_seed051_tensor.npz``, ~1.8 MB en disco, ~12 MB en RAM):
``timeloss[1440, 1660]`` + ``mask[1440, 1660]`` + ``node_order[1660]`` (sumo_edge en orden
``node_index``). Generado offline por ``ia_prediction_service/scripts/extract_day_slice.py``
desde el ``.npz`` de training (``day_idx=9 ↔ seed051``, byte-idéntico, stage-gate 1).

``window_at(t)`` delega al single-window builder; ``node_order`` se expone para mapear la
salida ``[1660, 30]`` (orden node_index) a ``edge_id`` en la respuesta — el MISMO orden con
el que el modelo fue entrenado (círculo entrada→salida cerrado).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .gru_window_builder import build_window

# Path repo-relativo (mismo patrón que GRUModelEngine): CWD /app en contenedor por
# ``COPY core_management_api/ .`` -> /app/models/...; en local desde core_management_api/.
_DEFAULT_SLICE_PATH = "models/day_seed051_tensor.npz"


class PredictionDaySource:
    """Slice del día seed051 residente en RAM; corta ventanas ``[30, 1660, 2]`` en ``t``."""

    def __init__(self, slice_path: str | os.PathLike | None = None):
        # Precedencia: argumento explícito > env PREDICTION_SLICE_PATH > default repo-relativo.
        self.slice_path = Path(
            slice_path
            or os.environ.get("PREDICTION_SLICE_PATH")
            or _DEFAULT_SLICE_PATH
        )
        self._loaded = False
        self.timeloss: np.ndarray | None = None   # [1440, 1660]
        self.mask: np.ndarray | None = None        # [1440, 1660]
        self.node_order: list[str] = []            # [1660] sumo_edge en orden node_index
        self.seed: int | None = None
        self.day_idx: int | None = None

    def load(self) -> None:
        """Carga eager el slice. Idempotente. Fail-fast si falta el artefacto."""
        if self._loaded:
            return
        if not self.slice_path.exists():
            raise FileNotFoundError(
                f"Slice del día no encontrado: {self.slice_path}. "
                "Se genera con ia_prediction_service/scripts/extract_day_slice.py y se hornea "
                "a la imagen del core (core_management_api/models/)."
            )
        z = np.load(self.slice_path, allow_pickle=True)
        # tensor del slice es [1440, 1660, 1]; el canal 0 es timeLoss.
        self.timeloss = np.ascontiguousarray(z["tensor"][..., 0].astype(np.float32))  # [1440, 1660]
        self.mask = np.ascontiguousarray(z["mask"].astype(bool))                       # [1440, 1660]
        self.node_order = [str(e) for e in z["node_order"]]
        self.seed = int(z["seed"])
        self.day_idx = int(z["day_idx"])
        self._loaded = True

    @property
    def available(self) -> bool:
        return self._loaded and self.timeloss is not None

    def window_at(self, t: int) -> np.ndarray:
        """Ventana ``[30, 1660, 2]`` cortada en ``t`` (cubre ``t-29..t``)."""
        if not self.available:
            raise RuntimeError("PredictionDaySource sin cargar: llamar load() antes de window_at().")
        return build_window(self.timeloss, self.mask, t)
