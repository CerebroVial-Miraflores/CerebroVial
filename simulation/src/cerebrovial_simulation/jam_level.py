"""Mapeo SUMO → jam level 0-5 (constructo Waze).

Transcripción literal de D-009 (documentation/lean-inception/4-decisiones/DECISIONS.md:211-225).
Única definición productiva de este mapeo en el repo — cualquier
consumidor (F2 coverage_check, F3 dataset, F4 adapter) lo importa de
aquí.
"""
from __future__ import annotations


def sumo_to_jam_level(mean_speed_mps: float, max_speed_mps: float) -> int:
    """Mapea velocidad SUMO a jam level (Waze 0-5).

    Args:
        mean_speed_mps: velocidad promedio observada (m/s). De
            ``traci.edge.getLastStepMeanSpeed`` o agregado equivalente.
        max_speed_mps: velocidad de flujo libre (m/s). De
            ``traci.lane.getMaxSpeed`` o leído de la red.

    Returns:
        Entero en [0, 5] según D-009.
    """
    if mean_speed_mps == 0:
        return 5
    ratio = mean_speed_mps / max_speed_mps
    if ratio >= 0.90:
        return 0
    if ratio >= 0.70:
        return 1
    if ratio >= 0.50:
        return 2
    if ratio >= 0.30:
        return 3
    return 4


def ratio_to_jam_level(ratio: float) -> int:
    """Variante que recibe el ratio directo (útil para tests / agregados Parquet)."""
    if ratio <= 0:
        return 5
    if ratio >= 0.90:
        return 0
    if ratio >= 0.70:
        return 1
    if ratio >= 0.50:
        return 2
    if ratio >= 0.30:
        return 3
    return 4
