"""Loader de inferencia model-agnostic (Fase 1).

Carga un ``.pt`` (GRU baseline o STGNN) por ruta explícita y expone un contrato único de
inferencia: ``predict(window [30, 1660, 2]) -> [1660, 30]`` en segundos, orden
``node_index``. Las asimetrías entre modelos quedan dentro de los adaptadores.

API pública::

    from src.inference import load_inference_adapter
    adapter = load_inference_adapter("models/miraflores_gru_baseline.pt")
    pred = adapter.predict(window)   # [1660, 30] segundos, orden node_index
"""
from __future__ import annotations

from .adapters import GruAdapter, InferenceAdapter, StgnnAdapter
from .jam_mapping import timeloss_to_jam_level
from .loader import load_inference_adapter
from .registry import resolve

__all__ = [
    "load_inference_adapter",
    "InferenceAdapter",
    "GruAdapter",
    "StgnnAdapter",
    "resolve",
    "timeloss_to_jam_level",
]
