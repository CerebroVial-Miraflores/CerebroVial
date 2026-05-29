"""Particiones train/valid del dataset TTH-07 (CT-07.4).

Declaración explícita de seeds por partición, sin overlap entre train y
valid. Cumple D-008 ("escenarios SUMO distintos para evitar fuga de
información") via seed-disjointness.

Patrones × seeds = corridas:
    Train: 4 patrones × 20 seeds = 80 corridas (seeds 1-20).
    Valid: 4 patrones × 5  seeds = 20 corridas (seeds 21-25).

Sin overlap por construcción. Documentado en el docstring (CT-07.4 b).
"""
from __future__ import annotations

from typing import Iterator


PATTERNS = ["am_peak", "pm_peak", "offpeak", "weekend"]
TRAIN_SEEDS = list(range(1, 21))      # 1..20 inclusive
VALID_SEEDS = list(range(21, 26))     # 21..25 inclusive


def iter_train() -> Iterator[tuple[str, int]]:
    """Yield (pattern, seed) para cada corrida de la partición de entrenamiento."""
    for pattern in PATTERNS:
        for seed in TRAIN_SEEDS:
            yield pattern, seed


def iter_valid() -> Iterator[tuple[str, int]]:
    """Yield (pattern, seed) para cada corrida de la partición de validación."""
    for pattern in PATTERNS:
        for seed in VALID_SEEDS:
            yield pattern, seed


def assert_no_overlap() -> None:
    """Failsafe — sin overlap entre seeds train y valid."""
    train_set = set(TRAIN_SEEDS)
    valid_set = set(VALID_SEEDS)
    overlap = train_set & valid_set
    if overlap:
        raise AssertionError(f"Overlap entre train y valid: {overlap}")


# Failsafe al import.
assert_no_overlap()
