"""Capa de presentación: ``timeLoss`` (segundos) → nivel de congestión 0–4.

El loader de inferencia devuelve ``timeLoss`` crudo en segundos (model-faithful); el
mapa observado muestra niveles de congestión 0–5 (constructo Waze) derivados de la
velocidad relativa vía ``ratio_to_jam_level`` (módulo ``cerebrovial_simulation.jam_level``).
Esta función convierte la salida del modelo a ese nivel **como capa de presentación**,
SEPARADA del adaptador y aplicada **post-``predict()``** — el loader sigue devolviendo
segundos crudos; el nivel es presentación.

Devuelve **0–4**. El nivel **5 ("vía cerrada") queda reservado** y es inalcanzable desde
esta función: es un estado *observable* (la vía está cerrada), no algo que el modelo
*prediga* desde la demora. Por construcción (4 cortes), una demora enorme satura en 4,
nunca 5.
"""
from __future__ import annotations

import numpy as np

# Cortes timeLoss (s) -> nivel de congestion 0-4. Derivados del cruce empirico
# timeLoss <-> nivel-velocidad (ratio_to_jam_level sobre speedRelative) por
# MATCHING DE DISTRIBUCION (M2) sobre el dataset completo miraflores_laborable_60d
# (60 dias, 30.8M celdas validas). Decision Cesar 2026-06-03. Sin redondear: son
# los valores que M2 produce sobre el universo completo (trazabilidad tesis).
# M2 (igualar la distribucion marginal de niveles predichos a la observada) se eligio
# sobre M1 (median-midpoint) por continuidad con el mapa observado + estabilidad
# submuestra<->60d (M2 reproduce la auditoria previa de 6 dias ~1.3/3/6/24; M1 se aparta
# en las fronteras bajas). Ver plan Fase 2.
_TIMELOSS_CUTS = np.array([1.34, 3.17, 6.32, 26.16], dtype=np.float64)


def timeloss_to_jam_level(seconds):
    """Convierte ``timeLoss`` en segundos a nivel de congestión entero **0–4**.

    Función pura, sin estado, vectorizable. Acepta un escalar (``float``) y devuelve un
    ``int``; o un ``np.ndarray`` (p.ej. ``[1660, 30]``) y devuelve un ``np.ndarray`` de
    enteros con la misma forma.

    Semántica de frontera: **demora ``>= corte`` ⇒ más congestión ⇒ nivel superior**
    (justificada por sí misma; nótese que ``ratio_to_jam_level`` va en dirección inversa
    —ratio alto = nivel bajo—, así que NO es un espejo de su ``>=``). Los 4 cortes salen
    del cruce empírico ``timeLoss ↔ nivel-velocidad`` por matching de distribución (ver
    ``_TIMELOSS_CUTS``).

    Bordes:

    - ``timeLoss`` negativo → 0 (no es físicamente posible; ruido tras des-estandarizar
      → flujo libre).
    - ``NaN`` → 0. Se fuerza **explícitamente** vía ``np.isnan`` (no se confía en el
      comportamiento de ``np.digitize`` con NaN, que mandaría el valor al último bin → 4).
      **Simplificación de presentación, no una verdad:** NaN ("sin dato") se asume flujo
      libre por consistencia con el modo observado, pero "sin dato" ≠ "flujo libre" (una
      arista sin observación podría estar congestionada). Un tratamiento más fiel
      (gris / sin-dato) queda como refinamiento posterior.

    El nivel **5 ("cerrado") es reservado y fuera de scope**: esta función nunca lo devuelve.
    """
    arr = np.asarray(seconds, dtype=np.float64)
    scalar_input = arr.ndim == 0

    # Trabajar siempre sobre >=1D para evitar casos borde de arrays 0-D.
    flat = np.atleast_1d(arr)
    nan_mask = np.isnan(flat)

    # NaN se neutraliza ANTES del digitize (no se depende de como digitize trata NaN);
    # el sustituto 0.0 es irrelevante porque se sobre-escribe a 0 explicitamente abajo.
    safe = np.where(nan_mask, 0.0, flat)
    levels = np.digitize(safe, _TIMELOSS_CUTS, right=False).astype(np.int64)
    levels[nan_mask] = 0  # forzado explicito e independiente de digitize

    if scalar_input:
        return int(levels[0])
    return levels.reshape(arr.shape)
