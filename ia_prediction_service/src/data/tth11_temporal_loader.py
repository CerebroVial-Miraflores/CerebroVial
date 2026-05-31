"""Loader y ventaneo temporal para el spike TTH-11 (hiperparámetros temporales).

Spike DESECHABLE: este módulo solo carga el dataset D-009, separa la serie de
``jam_level`` por dirección (N/S/E/W) y genera ventanas (X, y) parametrizables por
``lookback`` y ``horizonte``. NO entrena ni define modelos (eso es la sub-tarea
siguiente y, en producción, TTH-09).

Definición del problema (cerrada):
- Target: ``jam_level`` (0-5), clasificación ordinal.
- GRU univariada por dirección (D-006): 4 series independientes, una por N/S/E/W.
- Entrada autoregresiva pura sobre ``jam_level`` (sin features auxiliares).
- Δt_in fijo = 60 s (cada bucket del dataset = 60 s ⇒ 1 paso = 1 minuto).

Dependencias: pyarrow + numpy (pandas NO está en el venv del repo). Probado con
``simulation/.venv/bin/python``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Esquema D-009: 10 columnas, en orden. Fuente:
# simulation/src/cerebrovial_simulation/dataset/schema.py
D009_COLUMNS: tuple[str, ...] = (
    "seed",
    "pattern",
    "t_sim_s",
    "direction",
    "mean_speed_mps",
    "n_vehicles",
    "queue_length_m",
    "max_speed_mps",
    "ratio",
    "jam_level",
)

DIRECTIONS: tuple[str, ...] = ("N", "S", "E", "W")

# Resolución temporal del dataset (D-009 / cierre CT-11.8).
BUCKET_SECONDS: int = 60


class SchemaError(ValueError):
    """Un Parquet no cumple el esquema D-009."""


@dataclass(frozen=True)
class DirectionSeries:
    """Serie temporal de ``jam_level`` de una dirección dentro de una corrida.

    Una corrida = un (pattern, seed). ``values`` está ordenado por ``t_sim_s``
    ascendente; su longitud es el número de buckets de 60 s de la corrida.
    """

    pattern: str
    seed: int
    direction: str
    values: np.ndarray  # shape (n_buckets,), dtype int

    @property
    def n_buckets(self) -> int:
        return int(self.values.shape[0])


def validate_d009_schema(table: pq.lib.Table, source: Path) -> None:
    """Valida que ``table`` tenga exactamente las 10 columnas D-009.

    Lanza :class:`SchemaError` si falta/sobra alguna columna. (Es una guarda
    explícita: solo apuntamos a train/ y valid/, pero verificamos igual para no
    arrastrar Parquets ajenos —p. ej. corredor_larco— por error.)
    """
    names = tuple(table.schema.names)
    if set(names) != set(D009_COLUMNS) or len(names) != len(D009_COLUMNS):
        raise SchemaError(
            f"{source}: esquema no-D009. Esperado {D009_COLUMNS}, encontrado {names}"
        )


def load_run(path: Path) -> dict[str, DirectionSeries]:
    """Carga UN Parquet (una corrida) y devuelve la serie por dirección.

    Devuelve un dict {direction -> DirectionSeries} con las 4 direcciones
    presentes en el archivo. Valida el esquema D-009 antes de procesar.
    """
    table = pq.read_table(path)
    validate_d009_schema(table, path)

    t_sim = np.asarray(table.column("t_sim_s").to_pylist(), dtype=float)
    direction = np.asarray(table.column("direction").to_pylist(), dtype=object)
    jam = np.asarray(table.column("jam_level").to_pylist(), dtype=int)
    pattern_col = table.column("pattern").to_pylist()
    seed_col = table.column("seed").to_pylist()
    pattern = str(pattern_col[0]) if pattern_col else ""
    seed = int(seed_col[0]) if seed_col else -1

    out: dict[str, DirectionSeries] = {}
    for d in DIRECTIONS:
        mask = direction == d
        if not mask.any():
            continue
        order = np.argsort(t_sim[mask], kind="stable")
        values = jam[mask][order]
        out[d] = DirectionSeries(pattern=pattern, seed=seed, direction=d, values=values)
    return out


def load_partition(partition_dir: Path) -> list[dict[str, DirectionSeries]]:
    """Carga TODAS las corridas de una partición (carpeta train/ o valid/).

    Apunta EXPLÍCITO a ``partition_dir`` y solo lee ``*.parquet`` de ESA carpeta
    (sin recursión). Nunca usa glob('data/**'): eso arrastraría corredor_larco/.
    """
    partition_dir = Path(partition_dir)
    if not partition_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de partición: {partition_dir}")
    runs: list[dict[str, DirectionSeries]] = []
    for parquet in sorted(partition_dir.glob("*.parquet")):  # acotado a la carpeta
        runs.append(load_run(parquet))
    return runs


def minutes_to_steps(minutes: int) -> int:
    """Convierte minutos a pasos del dataset (÷ 60 s por bucket)."""
    return (minutes * 60) // BUCKET_SECONDS


def window_series(
    values: np.ndarray, lookback_steps: int, horizonte_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Ventanea UNA serie 1D en (X, y) autoregresivos.

    - ``X`` = los ``lookback_steps`` valores de la ventana histórica.
    - ``y`` = el ``jam_level`` en el bucket a ``+horizonte_steps`` pasos del FINAL
      de la ventana.

    Para una serie de largo ``n``: ventana en [i, i+lookback-1], target en
    ``i+lookback-1+horizonte``. Se requiere target <= n-1, de modo que el número
    de secuencias es ``max(0, n - lookback - horizonte + 1)``. Si la serie es muy
    corta (caso esperado con lookback/horizonte grandes) devuelve arrays vacíos.
    """
    n = int(values.shape[0])
    n_seq = n - lookback_steps - horizonte_steps + 1
    if n_seq <= 0 or lookback_steps <= 0 or horizonte_steps <= 0:
        return (
            np.empty((0, max(lookback_steps, 0)), dtype=values.dtype),
            np.empty((0,), dtype=values.dtype),
        )
    X = np.stack([values[i : i + lookback_steps] for i in range(n_seq)])
    y = np.array(
        [values[i + lookback_steps - 1 + horizonte_steps] for i in range(n_seq)],
        dtype=values.dtype,
    )
    return X, y


def build_sequences(
    runs: list[dict[str, DirectionSeries]],
    lookback_min: int,
    horizonte_min: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Genera (X, y) por dirección a partir de todas las corridas de una partición.

    Concatena las secuencias de todas las corridas (no se ventanea a través de
    corridas: cada corrida es una simulación independiente). Devuelve
    {direction -> (X, y)} con X shape (n_seq, lookback_steps) e y shape (n_seq,).
    """
    lookback_steps = minutes_to_steps(lookback_min)
    horizonte_steps = minutes_to_steps(horizonte_min)

    acc: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {d: [] for d in DIRECTIONS}
    for run in runs:
        for d, series in run.items():
            X, y = window_series(series.values, lookback_steps, horizonte_steps)
            if X.shape[0] > 0:
                acc[d].append((X, y))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for d in DIRECTIONS:
        if acc[d]:
            Xs = np.concatenate([x for x, _ in acc[d]], axis=0)
            ys = np.concatenate([y for _, y in acc[d]], axis=0)
        else:
            Xs = np.empty((0, lookback_steps), dtype=int)
            ys = np.empty((0,), dtype=int)
        out[d] = (Xs, ys)
    return out


def count_sequences(
    runs: list[dict[str, DirectionSeries]], lookback_min: int, horizonte_min: int
) -> dict[str, int]:
    """Cuenta secuencias por dirección sin materializar los arrays grandes."""
    seqs = build_sequences(runs, lookback_min, horizonte_min)
    return {d: int(seqs[d][0].shape[0]) for d in DIRECTIONS}
