"""Factory para extraer `zones_config` del cfg de visión y armar `OpenCVVisualizer`.

Acepta ambos formatos de zona: legacy (list de puntos) y nuevo (dict con `polygon`).
"""
from typing import Any

from omegaconf import OmegaConf

from .opencv_visualizer import OpenCVVisualizer


def _extract_zones_config(vision_cfg: Any) -> dict:
    zones_config: dict = {}
    zones = getattr(vision_cfg, "zones", None)
    if not zones:
        return zones_config
    for k, v in zones.items():
        if isinstance(v, list) or OmegaConf.is_list(v):
            zones_config[k] = list(v)
        elif hasattr(v, "polygon"):
            zones_config[k] = list(v.polygon)
        elif isinstance(v, dict) and "polygon" in v:
            zones_config[k] = list(v["polygon"])
    return zones_config


def build_visualizer_from_vision_cfg(vision_cfg: Any) -> OpenCVVisualizer:
    return OpenCVVisualizer(zones_config=_extract_zones_config(vision_cfg))
