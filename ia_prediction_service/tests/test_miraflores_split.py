"""Tests del split compartido seed→fold (miraflores_split.py), universo v2.

Verifica: determinismo (dos llamadas → mismo split); folds sin solapamiento;
unión == 60 seeds (42..101); proporciones 42/8/10; y —lo nuevo de B4— la
**estratificación por régimen de congestión**: los 12 severos repartidos sin
pérdida, cada fold ve congestión severa, test cubre el régimen severo (≥3), y los
duros NO se concentran en un fold. El test protege la estratificación: si alguien
reordena los folds sin respetar el régimen, rebota.

Corre con el venv raíz:
``cd ia_prediction_service && .venv/bin/python -m pytest tests/test_miraflores_split.py``
"""
from __future__ import annotations

from src.data.miraflores_split import (
    ALL_SEEDS,
    SEVERE_DUROS,
    SEVERE_SEEDS,
    get_split,
)


def test_determinista():
    a = get_split()
    b = get_split()
    assert a == b
    # Cada llamada devuelve copias nuevas (no comparte estado mutable).
    train_a, _, _ = a
    train_a.append(999)
    train_b, _, _ = get_split()
    assert 999 not in train_b


def test_sin_solapamiento():
    train, val, test = get_split()
    st, sv, sx = set(train), set(val), set(test)
    assert not (st & sv)
    assert not (st & sx)
    assert not (sv & sx)


def test_union_60_seeds():
    train, val, test = get_split()
    union = set(train) | set(val) | set(test)
    assert union == set(ALL_SEEDS)
    assert union == set(range(42, 102))
    assert len(train) + len(val) + len(test) == 60


def test_proporciones_42_8_10():
    train, val, test = get_split()
    assert (len(train), len(val), len(test)) == (42, 8, 10)


def test_ordenado():
    train, val, test = get_split()
    assert train == sorted(train)
    assert val == sorted(val)
    assert test == sorted(test)


# ---------------------- Estratificación por régimen de congestión (B4) -------------
def test_severos_repartidos_sin_perdida():
    """Los 12 severos están todos cubiertos, cada uno en exactamente un fold."""
    train, val, test = get_split()
    severe = set(SEVERE_SEEDS)
    assert len(severe) == 12
    union = set(train) | set(val) | set(test)
    assert severe <= union, "algún severo no está en ningún fold"
    # Sin duplicados entre folds ya lo garantiza test_sin_solapamiento; acá: cobertura.


def test_cada_fold_ve_congestion_severa():
    """Ningún fold queda ciego al régimen severo (ni entrena ni evalúa sin él)."""
    train, val, test = get_split()
    severe = set(SEVERE_SEEDS)
    assert severe & set(train), "train sin severos (se entrenaría ciego al régimen)"
    assert severe & set(val), "val sin severos"
    assert severe & set(test), "test sin severos (evaluaría solo el caso fácil)"


def test_test_cubre_regimen_severo():
    """Test contiene ≥3 severos — el régimen que decide D-011 se evalúa, no por azar."""
    train, val, test = get_split()
    severe_in_test = set(SEVERE_SEEDS) & set(test)
    assert len(severe_in_test) >= 3, f"test con <3 severos: {sorted(severe_in_test)}"


def test_duros_no_se_concentran():
    """Los 4 colapsos duros se reparten: cada fold ve ≥1 (gradiente no sesgado)."""
    train, val, test = get_split()
    duros = set(SEVERE_DUROS)
    assert len(duros) == 4
    assert duros & set(train), "ningún duro en train"
    assert duros & set(val), "ningún duro en val"
    assert duros & set(test), "ningún duro en test (no evaluaría el caso más severo)"
    # Ningún fold acapara los 4 duros.
    for fold in (train, val, test):
        assert not duros <= set(fold), f"un fold concentra los 4 duros: {sorted(set(fold) & duros)}"
