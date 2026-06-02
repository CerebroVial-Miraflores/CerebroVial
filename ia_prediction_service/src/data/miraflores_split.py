"""Split compartido seed→fold del track STGNN (Fase 3 baseline / Fase 4 STGNN).

FUENTE ÚNICA DE VERDAD del split. El baseline GRU (Fase 3) y el STGNN (Fase 4)
importan este módulo para evaluar sobre EXACTAMENTE el mismo train/val/test, de
modo que la comparación aísle la contribución de la vecindad del grafo (D-011).

El dataset Fase 2 tiene 60 días, uno por seed entera 42..101 (eje 0 del tensor;
día N ↔ seed 42+N). seed-081 = día de congestión máxima.

Split por **lista explícita** (no regla aritmética), elegido para BALANCEAR los
regímenes de congestión —verificado contra la demora media por día (valid
``timeLoss``) del ``.npz``—:

  test cubre el rango de congestión (081 extremo ~75 s, 067/061 altos naturales
  ~50/42 s, resto moderado ~14-26 s) para que la evaluación en régimen severo no
  dependa de un solo día; train conserva congestión alta de sobra (050, 089, 059,
  073, 055, 082 …). seed-081 forzado a test.

Proporciones: 42 train / 8 val / 10 test (incluye 081).
"""
from __future__ import annotations

# --- Split explícito (seeds enteras 42..101) -------------------------------------
# 10 días: 081 (extremo, forzado) + 067/061 (altos naturales) + 7 moderados.
TEST_SEEDS: list[int] = [45, 51, 57, 61, 63, 67, 69, 75, 81, 93]
# 8 días (sin cambios respecto al esquema mod-6 original).
VAL_SEEDS: list[int] = [48, 54, 60, 66, 78, 84, 90, 96]
# 42 días restantes (incorpora los ex-test suaves 087 y 099; conserva alta congestión).
TRAIN_SEEDS: list[int] = sorted(
    set(range(42, 102)) - set(TEST_SEEDS) - set(VAL_SEEDS)
)

# Rango completo del dataset (60 días, seeds 42..101).
ALL_SEEDS: list[int] = list(range(42, 102))

# seed-081 forzada a test (día de congestión máxima) — invariante del track.
FORCED_TEST_SEED: int = 81


def _validate() -> None:
    """Invariantes del split (se ejecuta al importar; falla rápido si se rompe)."""
    train, val, test = set(TRAIN_SEEDS), set(VAL_SEEDS), set(TEST_SEEDS)
    assert FORCED_TEST_SEED in test, "seed-081 debe estar en test (congestión máxima)."
    assert not (train & val), f"train∩val no vacío: {sorted(train & val)}"
    assert not (train & test), f"train∩test no vacío: {sorted(train & test)}"
    assert not (val & test), f"val∩test no vacío: {sorted(val & test)}"
    assert train | val | test == set(ALL_SEEDS), "la unión de folds != seeds 42..101"
    assert len(train) + len(val) + len(test) == 60, "los folds no suman 60 (¿solapamiento?)"
    assert (len(train), len(val), len(test)) == (42, 8, 10), (
        f"proporciones esperadas 42/8/10, got "
        f"{len(train)}/{len(val)}/{len(test)}"
    )


_validate()


def get_split() -> tuple[list[int], list[int], list[int]]:
    """Devuelve ``(train_seeds, val_seeds, test_seeds)``, cada uno ordenado.

    Determinista: copias nuevas de las listas módulo-nivel en cada llamada (dos
    llamadas → mismo split, sin estado mutable compartido).
    """
    return sorted(TRAIN_SEEDS), sorted(VAL_SEEDS), sorted(TEST_SEEDS)
