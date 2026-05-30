"""Schema canónico del dataset de TTH-07 (CT-07.3).

Una fila por (seed, pattern, t_sim_s, direction). t_sim_s en buckets de
60 s simulados (Δt_in provisional per TTH-11 CT-11.8). 4 direcciones
N/S/E/W por bucket. Carriles del aproche se agregan a la dirección
(ponderación por vehículos para mean_speed; suma para n_vehicles y
queue_length_m).

Columnas seleccionadas a partir del enunciado de CT-07.3:
    "velocidad media, número de vehículos, longitud de cola; ratio
    velocidad/free-flow; jam level 0-5 derivado D-009; marca de seed,
    patrón y tiempo simulado".
"""
from __future__ import annotations

import pyarrow as pa


SCHEMA = pa.schema([
    ("seed", pa.int32()),
    ("pattern", pa.string()),
    ("t_sim_s", pa.float32()),       # end-of-bucket en seg simulados (múltiplo de 60).
    ("direction", pa.string()),      # "N" | "S" | "E" | "W"
    ("mean_speed_mps", pa.float32()),
    ("n_vehicles", pa.int32()),
    ("queue_length_m", pa.float32()),
    ("max_speed_mps", pa.float32()),   # vmax (constante 13.89 — documentado).
    ("ratio", pa.float32()),           # mean_speed / max_speed
    ("jam_level", pa.int8()),          # 0-5 según D-009
])


DIRECTIONS = ["N", "S", "E", "W"]
VMAX_MPS = 13.89


def validate(table: pa.Table) -> None:
    """Verifica que `table` cumple SCHEMA mínimamente.

    Raises ValueError con mensaje específico si algo no cuadra.
    """
    if not isinstance(table, pa.Table):
        raise ValueError(f"Esperaba pa.Table, recibí {type(table)}")
    required = set(SCHEMA.names)
    actual = set(table.schema.names)
    missing = required - actual
    if missing:
        raise ValueError(f"Faltan columnas en dataset: {sorted(missing)}")

    # Tipos compatibles (relajado: int32/int64 ambos OK donde declaramos int*).
    for name, expected_type in zip(SCHEMA.names, SCHEMA.types):
        actual_type = table.schema.field(name).type
        if not _types_compatible(actual_type, expected_type):
            raise ValueError(
                f"Columna {name!r} tipo {actual_type} no compatible con {expected_type}"
            )

    # Rango básico de jam_level.
    jam_col = table.column("jam_level").to_pylist()
    if any((j is None) or (j < 0) or (j > 5) for j in jam_col):
        raise ValueError("jam_level fuera de [0,5] o nulo")

    # ratio en [0, 1+ε].
    ratio_col = table.column("ratio").to_pylist()
    if any((r is not None and not (0 <= r <= 1.05)) for r in ratio_col):
        raise ValueError("ratio fuera de [0, 1.05]")

    # Direction ∈ DIRECTIONS.
    dir_col = table.column("direction").to_pylist()
    bad = set(dir_col) - set(DIRECTIONS)
    if bad:
        raise ValueError(f"direction inválido: {bad}")


def _types_compatible(actual: pa.DataType, expected: pa.DataType) -> bool:
    """Compat type check: acepta int* compatible con int* declarado, float compatible con float, etc."""
    if actual.equals(expected):
        return True
    # Integer types compatibles.
    if pa.types.is_integer(actual) and pa.types.is_integer(expected):
        return True
    # Float types compatibles.
    if pa.types.is_floating(actual) and pa.types.is_floating(expected):
        return True
    if pa.types.is_string(actual) and pa.types.is_string(expected):
        return True
    return False
