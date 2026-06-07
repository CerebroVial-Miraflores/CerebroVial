"""Tests unitarios de la geometría PURA del matching (Fase B-2).

Corren SIEMPRE (sin DB, sin PostGIS). Cubren bearing, diferencia angular, ``same_direction``
(incluyendo los casos del brief: ~80° pasa, ~100° no) y, sobre todo, ``local_bearing`` —
la pieza del ajuste de Cesar: el rumbo del segmento TomTom debe ser LOCAL al tramo cercano a
la arista, no extremo→extremo, para no mentir en avenidas curvas.
"""
import pytest

from src.corridors.application.geometry import (
    angular_diff,
    bearing,
    local_bearing,
    midpoint,
    same_direction,
)


@pytest.mark.parametrize(
    "p1,p2,expected",
    [
        ((0.0, 0.0), (0.0, 1.0), 0.0),    # norte
        ((0.0, 0.0), (1.0, 0.0), 90.0),   # este
        ((0.0, 0.0), (0.0, -1.0), 180.0), # sur
        ((0.0, 0.0), (-1.0, 0.0), 270.0), # oeste
    ],
)
def test_bearing_cardinales(p1, p2, expected):
    assert bearing(p1, p2) == pytest.approx(expected, abs=0.5)


def test_bearing_ida_y_vuelta_difieren_180():
    a, b = (-77.040, -12.120), (-77.030, -12.120)
    assert angular_diff(bearing(a, b), bearing(b, a)) == pytest.approx(180.0, abs=0.5)


@pytest.mark.parametrize(
    "b1,b2,expected",
    [
        (350.0, 10.0, 20.0),
        (10.0, 350.0, 20.0),
        (0.0, 180.0, 180.0),
        (90.0, 90.0, 0.0),
    ],
)
def test_angular_diff_normaliza_a_0_180(b1, b2, expected):
    assert angular_diff(b1, b2) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "b1,b2,threshold,expected",
    [
        (10.0, 80.0, 90.0, True),    # ~70° de diferencia → mismo sentido
        (0.0, 80.0, 90.0, True),     # curva ~80° → PASA (brief)
        (0.0, 100.0, 90.0, False),   # curva ~100° → NO pasa (brief)
        (10.0, 110.0, 90.0, False),  # 100° → no
        (0.0, 180.0, 90.0, False),   # sentido opuesto → no
        (90.0, 270.0, 90.0, False),  # calzada opuesta → no
    ],
)
def test_same_direction(b1, b2, threshold, expected):
    assert same_direction(b1, b2, threshold) is expected


def test_midpoint():
    assert midpoint((0.0, 0.0), (2.0, 4.0)) == (1.0, 2.0)


# --- local_bearing: el ajuste de Cesar (rumbo local, no extremo→extremo) ---

# Polilínea que arranca yendo al NORTE y termina yendo al ESTE (avenida que dobla).
# Extremo→extremo daría ~45° (NE) — que MENTIRÍA respecto del tramo donde está la arista.
_CURVA = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 2.0), (2.0, 2.0)]


def test_local_bearing_tramo_norte_da_norte_no_NE():
    # ref en el tramo norte de la curva → rumbo local ≈ Norte (0°), NO el ~45° global.
    assert local_bearing(_CURVA, (0.0, 0.5)) == pytest.approx(0.0, abs=1.0)


def test_local_bearing_tramo_este_da_este_no_NE():
    # ref en el tramo este de la curva → rumbo local ≈ Este (90°), NO el ~45° global.
    assert local_bearing(_CURVA, (1.5, 2.0)) == pytest.approx(90.0, abs=1.0)


def test_local_bearing_difiere_del_extremo_a_extremo():
    # El bearing global (extremo→extremo) es ~45°; el local en el tramo norte es ~0°.
    global_bearing = bearing(_CURVA[0], _CURVA[-1])
    local_norte = local_bearing(_CURVA, (0.0, 0.5))
    assert global_bearing == pytest.approx(45.0, abs=2.0)
    assert angular_diff(global_bearing, local_norte) > 30.0


def test_local_bearing_requiere_dos_puntos():
    with pytest.raises(ValueError):
        local_bearing([(0.0, 0.0)], (0.0, 0.0))
