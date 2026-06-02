"""Tests del split compartido seed→fold (miraflores_split.py).

Verifica: determinismo (dos llamadas → mismo split); seed-081 en test; folds sin
solapamiento; unión == 60 seeds (42..101); proporciones 42/8/10.

Corre con el venv raíz:
``cd ia_prediction_service && .venv/bin/python -m pytest tests/test_miraflores_split.py``
"""
from __future__ import annotations

from src.data.miraflores_split import (
    ALL_SEEDS,
    FORCED_TEST_SEED,
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


def test_081_en_test():
    _, _, test = get_split()
    assert FORCED_TEST_SEED == 81
    assert 81 in test


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
