"""Loader y ventaneo temporal MULTI-OUTPUT para TTH-09 (GRU productivo).

Loader de PRODUCCIÓN para el clasificador GRU univariado por dirección (D-006).
Carga el dataset D-009, separa la serie de ``jam_level`` por dirección (N/S/E/W) y
genera ventanas (X, y) **vectoriales**: para cada ventana histórica de ``lookback``
pasos, el target son los ``horizonte`` pasos inmediatamente siguientes (multi-output
directo, NO autoregresivo, NO seq2seq).

Diferencia con el spike TTH-11 (``tth11_temporal_loader.py``, DESECHABLE, intacto):
- Spike: target ESCALAR ``y`` shape ``(n_seq,)`` — el jam_level en UN bucket a
  ``+horizonte`` del final de la ventana (single-step).
- Producción (este módulo): target VECTORIAL ``y`` shape ``(n_seq, horizonte)`` —
  los ``horizonte`` buckets +1..+horizonte tras la ventana.

Este módulo es auto-contenido a propósito: NO importa del spike para que el código
de producción no dependa de un prototipo desechable.

Definición del problema (cerrada):
- Target: ``jam_level`` (0-5), clasificación. jam5 ausente en datos (0 muestras por
  diseño del dataset); se conserva como clase válida (n_classes=6).
- GRU univariada por dirección (D-006): 4 series independientes, una por N/S/E/W.
- Entrada autoregresiva pura sobre ``jam_level`` (sin features auxiliares).
- Δt_in fijo = 60 s (cada bucket del dataset = 60 s ⇒ 1 paso = 1 minuto).

Dependencias: pyarrow + numpy (pandas NO está en el venv del repo).
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

# Escala Waze 0-5 completa. jam5 nunca se activa en el dataset, pero la cabeza del
# modelo y los class weights cubren las 6 clases.
N_CLASSES: int = 6

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

    Guarda explícita: solo apuntamos a train/ y valid/, pero verificamos igual para
    no arrastrar Parquets ajenos (p. ej. corredor_larco) por error.
    """
    names = tuple(table.schema.names)
    if set(names) != set(D009_COLUMNS) or len(names) != len(D009_COLUMNS):
        raise SchemaError(
            f"{source}: esquema no-D009. Esperado {D009_COLUMNS}, encontrado {names}"
        )


def load_run(path: Path) -> dict[str, DirectionSeries]:
    """Carga UN Parquet (una corrida) y devuelve la serie por dirección.

    Devuelve un dict {direction -> DirectionSeries} con las direcciones presentes
    en el archivo. Valida el esquema D-009 antes de procesar.
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


def load_partition(
    partition_dir: Path, max_runs: int | None = None
) -> list[dict[str, DirectionSeries]]:
    """Carga las corridas de una partición (carpeta train/ o valid/).

    Apunta EXPLÍCITO a ``partition_dir`` y solo lee ``*.parquet`` de ESA carpeta
    (sin recursión). Nunca usa glob('data/**'): eso arrastraría corredor_larco/.

    ``max_runs`` limita el número de corridas cargadas (subset reducido para el modo
    smoke ``--quick``); ``None`` carga todas.
    """
    partition_dir = Path(partition_dir)
    if not partition_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de partición: {partition_dir}")
    parquets = sorted(partition_dir.glob("*.parquet"))  # acotado a la carpeta
    if max_runs is not None:
        parquets = parquets[:max_runs]
    return [load_run(parquet) for parquet in parquets]


def minutes_to_steps(minutes: int) -> int:
    """Convierte minutos a pasos del dataset (÷ 60 s por bucket)."""
    return (minutes * 60) // BUCKET_SECONDS


def window_series_multioutput(
    values: np.ndarray, lookback_steps: int, horizonte_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Ventanea UNA serie 1D en (X, y) MULTI-OUTPUT.

    - ``X`` = los ``lookback_steps`` valores de la ventana histórica
      ``[i, i+lookback-1]`` → shape ``(n_seq, lookback_steps)``.
    - ``y`` = los ``horizonte_steps`` valores INMEDIATAMENTE siguientes a la ventana,
      ``[i+lookback, i+lookback+horizonte-1]`` → shape ``(n_seq, horizonte_steps)``.
      (Predice los pasos +1..+horizonte tras el final de la ventana.)

    Para una serie de largo ``n``: el último target válido termina en ``n-1``, así
    que ``n_seq = max(0, n - lookback - horizonte + 1)``. Si la serie es muy corta
    devuelve arrays vacíos con shape 2D coherente ``(0, lookback)`` y ``(0, horizonte)``.
    """
    n = int(values.shape[0])
    n_seq = n - lookback_steps - horizonte_steps + 1
    if n_seq <= 0 or lookback_steps <= 0 or horizonte_steps <= 0:
        return (
            np.empty((0, max(lookback_steps, 0)), dtype=values.dtype),
            np.empty((0, max(horizonte_steps, 0)), dtype=values.dtype),
        )
    X = np.stack([values[i : i + lookback_steps] for i in range(n_seq)])
    y = np.stack(
        [
            values[i + lookback_steps : i + lookback_steps + horizonte_steps]
            for i in range(n_seq)
        ]
    )
    return X, y


def build_sequences(
    runs: list[dict[str, DirectionSeries]],
    lookback_min: int,
    horizonte_min: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Genera (X, y) multi-output por dirección a partir de todas las corridas.

    Concatena las secuencias de todas las corridas (no se ventanea a través de
    corridas: cada corrida es una simulación independiente). Devuelve
    {direction -> (X, y)} con X shape (n_seq, lookback_steps) e y shape
    (n_seq, horizonte_steps).
    """
    lookback_steps = minutes_to_steps(lookback_min)
    horizonte_steps = minutes_to_steps(horizonte_min)

    acc: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {d: [] for d in DIRECTIONS}
    for run in runs:
        for d, series in run.items():
            X, y = window_series_multioutput(series.values, lookback_steps, horizonte_steps)
            if X.shape[0] > 0:
                acc[d].append((X, y))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for d in DIRECTIONS:
        if acc[d]:
            Xs = np.concatenate([x for x, _ in acc[d]], axis=0)
            ys = np.concatenate([y for _, y in acc[d]], axis=0)
        else:
            Xs = np.empty((0, lookback_steps), dtype=int)
            ys = np.empty((0, horizonte_steps), dtype=int)
        out[d] = (Xs, ys)
    return out


def compute_class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Class weights por inverso de frecuencia (balanceado) sobre el target ``y``.

    Fórmula (estilo sklearn 'balanced', sobre clases PRESENTES):
        w[c] = n_muestras / (n_clases_presentes * conteo[c])
    de modo que las clases raras reciben peso alto y las frecuentes peso bajo.

    Clases con conteo 0 → peso 0.0 (evita la división por cero / inf). Esto cubre:
      - jam5 SIEMPRE (clase 5 sin muestras por diseño del dataset). El peso nunca se
        usa: la clase 5 jamás es target, así que CrossEntropyLoss no lo indexa.
      - Bajo ``--quick`` es ESPERADO Y CORRECTO que las clases raras (j3 en N/S
        ~0.7%, j4 en E/W ~2.8%) no aparezcan en el subset reducido y queden con peso
        0. NO es un bug: es artefacto del subset chico. En modo full (80 corridas)
        esas clases sí aparecen y reciben su peso alto.

    ``y`` puede ser 1D o 2D (multi-output (n_seq, horizonte)): se aplana para contar.
    """
    flat = np.asarray(y).reshape(-1).astype(int)
    counts = np.bincount(flat, minlength=n_classes)[:n_classes]
    n_present = int((counts > 0).sum())
    n_samples = int(counts.sum())

    weights = np.zeros(n_classes, dtype=np.float64)
    if n_present == 0 or n_samples == 0:
        return weights
    present = counts > 0
    weights[present] = n_samples / (n_present * counts[present])
    return weights
