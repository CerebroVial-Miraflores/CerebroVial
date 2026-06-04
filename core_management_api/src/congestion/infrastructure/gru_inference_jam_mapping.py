"""Capa de presentación: ``timeLoss`` (segundos) → nivel de congestión 0–4 (Fase 3).

VENDORIZADO de ``ia_prediction_service/src/inference/jam_mapping.py`` (D-010, patrón
TTH-09). Copia byte-a-byte. Mantener en sync con el original.

El adaptador devuelve ``timeLoss`` crudo en segundos (model-faithful); el mapa muestra
niveles 0–5 (constructo Waze). Esta función convierte la salida del modelo a ese nivel
como capa de presentación, SEPARADA del adaptador y aplicada post-``predict()``.

Devuelve **0–4**. El nivel **5 ("vía cerrada") queda reservado** e inalcanzable desde
acá: es un estado observable, no algo que el modelo prediga desde la demora.
"""
from __future__ import annotations

import numpy as np

# Cortes timeLoss (s) -> nivel de congestion 0-4. Derivados del cruce empirico
# timeLoss <-> nivel-velocidad (ratio_to_jam_level sobre speedRelative) por
# MATCHING DE DISTRIBUCION (M2) sobre el dataset completo miraflores_laborable_60d
# (60 dias, 30.8M celdas validas). Decision Cesar 2026-06-03.
_TIMELOSS_CUTS = np.array([1.34, 3.17, 6.32, 26.16], dtype=np.float64)


def timeloss_to_jam_level(seconds):
    """Convierte ``timeLoss`` en segundos a nivel de congestión entero **0–4**.

    Función pura, sin estado, vectorizable. Acepta un escalar (``float``) y devuelve un
    ``int``; o un ``np.ndarray`` (p.ej. ``[1660, 30]``) y devuelve un ``np.ndarray`` de
    enteros con la misma forma.

    Semántica de frontera: **demora ``>= corte`` ⇒ más congestión ⇒ nivel superior**.

    Bordes: ``timeLoss`` negativo → 0 (ruido tras des-estandarizar); ``NaN`` → 0 (forzado
    explícito vía ``np.isnan``, no se confía en ``np.digitize`` con NaN). El nivel **5
    ("cerrado") es reservado y fuera de scope**: esta función nunca lo devuelve.
    """
    arr = np.asarray(seconds, dtype=np.float64)
    scalar_input = arr.ndim == 0

    flat = np.atleast_1d(arr)
    nan_mask = np.isnan(flat)

    safe = np.where(nan_mask, 0.0, flat)
    levels = np.digitize(safe, _TIMELOSS_CUTS, right=False).astype(np.int64)
    levels[nan_mask] = 0  # forzado explicito e independiente de digitize

    if scalar_input:
        return int(levels[0])
    return levels.reshape(arr.shape)
