"""Servicio de predicción de congestión por arista (Fase 3 — GRU baseline servido).

Orquesta el camino vendorizado GRU-only: dado un corte ``t``, corta la ventana del día
seed051 desde el slice residente, corre el ``GruAdapter`` (``[1660, 30]`` segundos), y mapea
a niveles de congestión 0-4 por arista (``timeloss_to_jam_level``, capa de presentación).

Análogo a ``GruInferenceService`` (TTH-09) pero SEPARADO y por arista: aquel clasifica
jam_level 0-5 por intersección; éste predice ``timeLoss`` continuo por arista y lo presenta
0-4. Dos predictores coexistiendo. Si no puede servir, lanza ``PredictionUnavailableError``
(-> 503 en el handler).

Invariante de coherencia: el ``t`` lo decide el handler (param explícito o derivado del feed
vivo), NO este servicio — así la inferencia es determinista y testeable.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..infrastructure.gru_inference_adapter import GruAdapter
from ..infrastructure.gru_inference_jam_mapping import timeloss_to_jam_level
from ..infrastructure.prediction_day_source import PredictionDaySource


class PredictionUnavailableError(RuntimeError):
    """La predicción no puede servirse: slice o modelo no cargados (-> 503 en el handler)."""


@dataclass(frozen=True)
class EdgePrediction:
    """Predicción de UNA arista: niveles 0-4 para los pasos t+1..t+horizon."""
    edge_id: str
    levels: list[int]


class CongestionPredictionService:
    """Inferencia GRU baseline por arista, a ``horizon`` pasos, desde el slice del día."""

    def __init__(self, day_source: PredictionDaySource, adapter: GruAdapter):
        self.day_source = day_source
        self.adapter = adapter

    @property
    def available(self) -> bool:
        return self.day_source.available and self.adapter is not None

    def predict(self, t: int) -> list[EdgePrediction]:
        """Predice los 30 pasos (t+1..t+30) por arista, niveles 0-4, en orden ``node_index``.

        El ``edge_id`` de cada arista sale de ``day_source.node_order`` — el MISMO orden con
        el que el modelo fue entrenado (círculo entrada→salida cerrado): la fila ``i`` de la
        salida ``[1660, 30]`` corresponde a ``node_order[i]``.
        """
        if not self.available:
            raise PredictionUnavailableError(
                "Predicción no disponible: slice del día o modelo GRU no cargados."
            )
        window = self.day_source.window_at(t)        # [30, 1660, 2]
        pred_seconds = self.adapter.predict(window)  # [1660, 30] timeLoss (s)
        levels = timeloss_to_jam_level(pred_seconds)  # [1660, 30] enteros 0-4
        node_order = self.day_source.node_order
        return [
            EdgePrediction(edge_id=node_order[i], levels=levels[i].tolist())
            for i in range(len(node_order))
        ]
