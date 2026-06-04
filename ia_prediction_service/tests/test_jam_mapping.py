"""Tests de ``timeloss_to_jam_level`` — capa de presentación timeLoss(s) → nivel 0–4.

Verifica el contrato sustantivo de Fase 2: los 4 cortes derivados por matching de
distribución, la monotonía, el tope estructural en 4 (el 5 reservado es inalcanzable),
la vectorización escalar↔array, y el manejo de bordes (negativo→0, NaN→0). El test
``test_nan_mixto_array`` es el que protege contra el bug real de producción: un array
``[1660, 30]`` con NaN dispersos (aristas sin dato) junto a valores válidos.

Correr: ``cd ia_prediction_service && .venv/bin/python -m pytest tests/test_jam_mapping.py -v``
"""
from __future__ import annotations

import numpy as np

from src.inference import timeloss_to_jam_level as timeloss_to_jam_level_public
from src.inference.jam_mapping import _TIMELOSS_CUTS, timeloss_to_jam_level

N_NODES, HORIZON = 1660, 30


def test_api_publica_es_la_misma_funcion():
    """La función exportada por ``src.inference`` es la del módulo (sin wrappers)."""
    assert timeloss_to_jam_level_public is timeloss_to_jam_level


def test_piso_cero_da_nivel_cero():
    assert timeloss_to_jam_level(0.0) == 0


def test_tope_demora_enorme_da_cuatro_nunca_cinco():
    assert timeloss_to_jam_level(1000.0) == 4
    assert timeloss_to_jam_level(1e9) == 4


def test_cuatro_cortes_exactos():
    """Justo debajo / exacto / justo encima de cada frontera cae donde debe.

    Convención: ``timeLoss >= corte`` ⇒ nivel superior (el valor exacto sube de nivel).
    """
    eps = 1e-3
    for i, cut in enumerate(_TIMELOSS_CUTS):
        lo = float(cut) - eps
        hi = float(cut) + eps
        assert timeloss_to_jam_level(lo) == i, f"justo debajo de corte {i} ({cut}) → {i}"
        assert timeloss_to_jam_level(float(cut)) == i + 1, f"exacto en corte {i} ({cut}) → {i+1}"
        assert timeloss_to_jam_level(hi) == i + 1, f"justo encima de corte {i} ({cut}) → {i+1}"


def test_rangos_intermedios():
    """Un valor representativo de cada banda cae en el nivel correcto (0..4)."""
    # bandas: (-inf,1.34) (1.34,3.17) (3.17,6.32) (6.32,26.16) (26.16,inf)
    casos = [(0.5, 0), (2.0, 1), (5.0, 2), (10.0, 3), (50.0, 4)]
    for x, nivel in casos:
        assert timeloss_to_jam_level(x) == nivel, f"{x}s → nivel {nivel}"


def test_monotonia():
    """timeLoss creciente ⇒ nivel no decreciente."""
    xs = np.linspace(0.0, 60.0, 5000)
    niveles = timeloss_to_jam_level(xs)
    assert np.all(np.diff(niveles) >= 0)


def test_escalar_devuelve_int_python():
    out = timeloss_to_jam_level(5.0)
    assert isinstance(out, int)
    assert not isinstance(out, np.integer)


def test_negativo_da_cero():
    assert timeloss_to_jam_level(-1.0) == 0
    assert timeloss_to_jam_level(-1000.0) == 0
    arr = np.array([-5.0, -0.1, 0.0, 2.0])
    assert list(timeloss_to_jam_level(arr)) == [0, 0, 0, 1]


def test_nan_solo_da_cero():
    assert timeloss_to_jam_level(np.nan) == 0
    arr = np.array([np.nan, np.nan, np.nan])
    assert list(timeloss_to_jam_level(arr)) == [0, 0, 0]


def test_cinco_no_alcanzable():
    """Barrido amplio: la salida NUNCA es 5 (reservado para 'cerrado', sólo observado)."""
    barrido = np.array(
        [-1e9, -10.0, -0.001, 0.0, *(_TIMELOSS_CUTS), 1.34, 3.17, 6.32, 26.16,
         100.0, 1e6, 1e12, np.nan]
    )
    niveles = timeloss_to_jam_level(barrido)
    assert niveles.max() <= 4
    assert not np.any(niveles == 5)


def test_rango_siempre_0_a_4():
    """Cualquier entrada finita → nivel en {0,1,2,3,4}."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(-50, 200, size=10000)
    niveles = timeloss_to_jam_level(xs)
    assert niveles.min() >= 0 and niveles.max() <= 4


def test_vectorizacion_array_igual_a_elemento_a_elemento():
    """Un array [1660,30] da el mismo resultado que aplicar la función elemento a elemento."""
    rng = np.random.default_rng(42)
    grid = rng.uniform(0.0, 50.0, size=(N_NODES, HORIZON))
    vect = timeloss_to_jam_level(grid)
    assert vect.shape == (N_NODES, HORIZON)
    elem = np.array(
        [[timeloss_to_jam_level(float(grid[i, j])) for j in range(HORIZON)] for i in range(N_NODES)]
    )
    assert np.array_equal(vect, elem)


def test_nan_mixto_array():
    """[1660,30] con NaN dispersos + válidos: NaN→0 Y los válidos en su nivel correcto.

    Réplica del caso real de producción (aristas sin dato junto a aristas con demora).
    Protege contra el bug de ``np.digitize`` (que mandaría NaN al último bin → 4).
    """
    rng = np.random.default_rng(7)
    grid = rng.uniform(0.0, 50.0, size=(N_NODES, HORIZON))
    nan_mask = rng.random(size=grid.shape) < 0.15  # ~15% de celdas sin dato, dispersas
    grid_nan = grid.copy()
    grid_nan[nan_mask] = np.nan

    out = timeloss_to_jam_level(grid_nan)

    # NaN → 0 en todas sus posiciones.
    assert np.all(out[nan_mask] == 0)
    # Hay NaN dispersos (no es el caso trivial de array sin NaN).
    assert nan_mask.any() and not nan_mask.all()
    # Los válidos coinciden con la conversión del valor original (no contaminados por los NaN).
    expected_valid = timeloss_to_jam_level(grid)
    assert np.array_equal(out[~nan_mask], expected_valid[~nan_mask])


def test_no_muta_la_entrada():
    """La función no modifica el array de entrada."""
    grid = np.array([np.nan, 2.0, 100.0, -1.0])
    antes = grid.copy()
    timeloss_to_jam_level(grid)
    np.testing.assert_array_equal(grid[~np.isnan(grid)], antes[~np.isnan(grid)])
    assert np.isnan(grid[0])
