"""Mapeo del nivel de congestión 0-5 (D-009) al ``status`` del marcador del mapa (B2).

El DashboardView pinta el marcador (y el sidebar/popup) por un string
``fluid | moderate | critical`` — NO por el nivel 0-5 crudo. El mapeo vive en el
backend para que el front reciba el dato ya resuelto y su lógica de color quede sin
tocar.

Umbrales (B2 D4, escala Waze D-009): 0-1 → fluid · 2-3 → moderate · 4-5 → critical.
``None`` = intersección sin ninguna arista con dato Waze → fluid (verde, sin alarma).
"""
from __future__ import annotations


def congestion_status_from_level(level: int | None) -> str:
    """Traduce el nivel agregado 0-5 (o ``None`` = sin dato) al status del marcador."""
    if level is None:
        return "fluid"
    if level >= 4:
        return "critical"
    if level >= 2:
        return "moderate"
    return "fluid"
