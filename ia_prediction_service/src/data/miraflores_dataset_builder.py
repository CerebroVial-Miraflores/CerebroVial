"""Dataset builder por-arista de Miraflores (STGNN, Fase 2) — target = timeLoss TOTAL.

De los 60 Parquet diarios de ``simulation/data/datasets/miraflores_laborable_60d/``
(un día por seed 42..101, cada uno 1440 intervalos de 60 s × 381 aristas vehiculares)
consolida un tensor ``[60, 1440, 375, 1]`` sobre el grafo edge-as-node del LCC
(componente conexa mayor, 375 nodos) + una máscara de validez ``[60, 1440, 375]``.

Decisiones que este builder materializa:

* **Target = ``timeLoss`` TOTAL, tal cual la columna del Parquet.** Por D-013 y su
  enmienda 2026-06-01 (``documentation/lean-inception/4-decisiones/DECISIONS.md``
  § D-013): el ``timeLoss`` del edgeData es el tiempo perdido TOTAL agregado por
  arista por intervalo (sumado sobre vehículos), no el promedio por vehículo. El
  promedio per-vehicle está indefinido en el atasco (entered≈0 → división por
  casi-cero) y exigiría regenerar las 60 simulaciones; el total es estable y
  preserva el orden (Spearman 0.94). Acá NO se deriva, NO se divide por
  flow/entered, NO se maneja div-por-cero: se usa la columna timeLoss directamente.
  Demora continua en segundos.

* **Orden de nodos = ``node_index`` del LCC mapping** (``artifacts/
  miraflores_graph_lcc_mapping.json``, clave ``nodes``), 0..374. La alineación de
  cada arista del Parquet a su posición en el tensor es un ``reindex`` EXPLÍCITO
  por ``sumo_edge`` (NO por orden de aparición; la clave es ``node_index``, NO un
  ``graph_index``). Las 6 aristas de la islita (fuera del LCC) quedan excluidas
  por construcción. Fail-fast: arista del LCC ausente en el Parquet → ``ValueError``
  nombrándola; seed faltante → ``ValueError``.

* **Máscara de 3 estados.** ``False`` (vacío, sin tráfico) donde
  ``density == 0 AND speed.isna()`` — el mismo predicado de generación
  (``sampledSeconds == 0`` ⇒ SUMO no emite métricas ⇒ compactado rellena
  density=0, speed=NaN). ``True`` donde hay tráfico. En las celdas vacías el target
  se fija a ``NaN`` (NO 0, NO imputado): "sin tráfico" no es "sin congestión".

* **Features mínimas: C=1, solo ``timeLoss``.** La lista de columnas-feature es
  parametrizable (``feature_columns``) para enriquecimiento futuro (density, flow,
  speedRelative; deuda (iii) de D-013) sin rediseño.

Desacople: NO toca DB, NO toca producción, NO toca el grafo builder ni
``jam_level.py``. Lee Parquet + el LCC mapping; escribe ``.npz`` + metadata JSON +
README de regeneración. Nada más.

Uso CLI: ``cd ia_prediction_service && .venv/bin/python -m src.data.miraflores_dataset_builder``
(usa los defaults). Ver ``build_miraflores_dataset`` / ``save_dataset``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Raíz del repo: este archivo vive en ia_prediction_service/src/data/.
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PARQUET_DIR = _REPO_ROOT / "simulation/data/datasets/miraflores_laborable_60d"
DEFAULT_LCC_MAPPING = (
    Path(__file__).resolve().parent / "artifacts/miraflores_graph_lcc_mapping.json"
)
DEFAULT_OUT_DIR = DEFAULT_PARQUET_DIR / "tensors"
DEFAULT_SEEDS = tuple(range(42, 102))  # 42..101 = 60 días
DEFAULT_FEATURES = ("timeLoss",)

# Columnas que la máscara de 3 estados necesita siempre, además de las features.
_MASK_COLUMNS = ("density", "speed")

# Autoridad del target — referencia citable en metadata.
_TARGET_AUTHORITY = (
    "D-013 + enmienda 2026-06-01 "
    "(documentation/lean-inception/4-decisiones/DECISIONS.md § D-013): "
    "target = timeLoss TOTAL por arista por intervalo de 60 s, sin derivación per-vehículo."
)
_MASK_RULE = (
    "valido (True) = hay trafico; invalido (False, vacio) = density==0 AND speed.isna() "
    "(reproduce sampledSeconds==0 de la generacion). En celdas invalidas target=NaN."
)


def load_node_order(lcc_mapping_path: Path) -> list[str]:
    """Los 375 ``sumo_edge`` del LCC, ordenados por ``node_index`` ascendente (0..374)."""
    mapping = json.loads(Path(lcc_mapping_path).read_text())
    nodes = mapping["nodes"]
    ordered = sorted(nodes, key=lambda n: n["node_index"])
    indices = [n["node_index"] for n in ordered]
    if indices != list(range(len(ordered))):
        raise ValueError(
            f"node_index del LCC mapping no es contiguo 0..{len(ordered) - 1}: {indices[:5]}..."
        )
    return [n["sumo_edge"] for n in ordered]


def _parquet_path(parquet_dir: Path, seed: int) -> Path:
    return Path(parquet_dir) / f"day_seed{seed:03d}.parquet"


def _pivot_day(
    df: pd.DataFrame, column: str, timesteps: list[int], node_order: list[str], seed: int
) -> np.ndarray:
    """[T, N] de ``column`` alineado a (timesteps × node_order) por reindex EXPLÍCITO."""
    piv = df.pivot(index="timestep", columns="edge_id", values=column)
    # reindex explícito: filas=timesteps, columnas=node_order (orden node_index, NO
    # orden de aparición). Aristas fuera del LCC desaparecen; faltantes → NaN (se
    # validan aparte vía density).
    piv = piv.reindex(index=timesteps, columns=node_order)
    return piv.to_numpy(dtype=np.float64)


def build_miraflores_dataset(
    parquet_dir: Path | str = DEFAULT_PARQUET_DIR,
    lcc_mapping_path: Path | str = DEFAULT_LCC_MAPPING,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
) -> dict:
    """Consolida los Parquet diarios en tensor + máscara alineados al LCC.

    Returns dict con:
      - ``tensor``: np.ndarray ``[D, T, N, C]`` float32 (D=días/seeds, T=timesteps,
        N=375 nodos LCC, C=len(feature_columns)). Celdas vacías = NaN.
      - ``mask``: np.ndarray ``[D, T, N]`` bool (True = tráfico, False = vacío).
      - ``seeds``: np.ndarray ``[D]`` int — seed por índice de día (ascendente).
      - ``node_order``: list[str] — los N ``sumo_edge`` en orden ``node_index``.
      - ``channels``: list[str] — nombres de canal (= feature_columns).
      - ``timesteps``: list[int] — los T ``begin`` de intervalo (s), ascendente.
    """
    parquet_dir = Path(parquet_dir)
    node_order = load_node_order(lcc_mapping_path)
    node_set = set(node_order)
    channels = list(feature_columns)
    if not channels:
        raise ValueError("feature_columns vacío: se requiere al menos un canal (default 'timeLoss').")
    seeds = sorted(seeds)
    read_cols = ["edge_id", "timestep", *dict.fromkeys([*channels, *_MASK_COLUMNS])]

    tensor: np.ndarray | None = None
    mask: np.ndarray | None = None
    ref_timesteps: list[int] | None = None

    for di, seed in enumerate(seeds):
        path = _parquet_path(parquet_dir, seed)
        if not path.exists():
            raise ValueError(f"seed {seed} faltante: no existe el Parquet {path}")

        # Validar el schema ANTES del read restringido: pd.read_parquet(columns=...)
        # lanzaría un error crudo de pyarrow si una columna-feature no existe; este
        # chequeo da un ValueError claro nombrando la columna y el seed.
        available = set(pq.read_schema(path).names)
        missing_cols = [c for c in read_cols if c not in available]
        if missing_cols:
            raise ValueError(f"seed {seed}: columnas ausentes en {path.name}: {missing_cols}")

        df = pd.read_parquet(path, columns=read_cols)

        present = set(df["edge_id"].astype("string").unique())
        missing_edges = [e for e in node_order if e not in present]
        if missing_edges:
            raise ValueError(
                f"seed {seed}: {len(missing_edges)} sumo_edge del LCC ausente(s) en "
                f"{path.name}: {missing_edges[:10]}"
            )

        timesteps = sorted(int(t) for t in df["timestep"].unique())
        if ref_timesteps is None:
            ref_timesteps = timesteps
            T, N, C, D = len(ref_timesteps), len(node_order), len(channels), len(seeds)
            tensor = np.full((D, T, N, C), np.nan, dtype=np.float32)
            mask = np.zeros((D, T, N), dtype=bool)
        elif timesteps != ref_timesteps:
            raise ValueError(
                f"seed {seed}: timesteps difieren del primer día "
                f"({len(timesteps)} vs {len(ref_timesteps)}); días no homogéneos."
            )

        df = df[df["edge_id"].isin(node_set)]

        density = _pivot_day(df, "density", ref_timesteps, node_order, seed)
        speed = _pivot_day(df, "speed", ref_timesteps, node_order, seed)
        # density NaN ⟺ celda (timestep,edge) ausente: la grilla debe estar completa.
        if np.isnan(density).any():
            raise ValueError(
                f"seed {seed}: grilla incompleta (faltan celdas timestep×edge); "
                "el compactado debería emitir todas las aristas en todos los intervalos."
            )
        mask_day = ~((density == 0.0) & np.isnan(speed))  # [T, N]
        mask[di] = mask_day

        for ci, col in enumerate(channels):
            tensor[di, :, :, ci] = _pivot_day(df, col, ref_timesteps, node_order, seed)
        # Vacío → NaN en todos los canales (no imputar, no 0).
        tensor[di][~mask_day] = np.nan

    return {
        "tensor": tensor,
        "mask": mask,
        "seeds": np.asarray(seeds, dtype=np.int32),
        "node_order": node_order,
        "channels": channels,
        "timesteps": ref_timesteps,
    }


def save_dataset(
    result: dict,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    *,
    lcc_mapping_path: Path | str = DEFAULT_LCC_MAPPING,
    npz_name: str = "miraflores_laborable_60d.npz",
) -> dict[str, Path]:
    """Escribe ``.npz`` (gitignored) + ``metadata.json`` + ``README.md`` (versionados).

    Returns dict {clave: Path} de los archivos escritos.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor, mask = result["tensor"], result["mask"]
    npz_path = out_dir / npz_name
    meta_path = out_dir / "metadata.json"
    readme_path = out_dir / "README.md"

    np.savez_compressed(
        npz_path,
        tensor=tensor,
        mask=mask,
        seeds=result["seeds"],
        node_order=np.asarray(result["node_order"], dtype=object),
        channels=np.asarray(result["channels"], dtype=object),
        timesteps=np.asarray(result["timesteps"], dtype=np.int32),
    )

    meta = {
        "artifact": npz_name,
        "shape": {
            "tensor": list(tensor.shape),
            "mask": list(mask.shape),
            "axes": ["dia/seed", "timestep", "nodo (node_index LCC)", "canal"],
        },
        "channels": list(result["channels"]),
        "target_authority": _TARGET_AUTHORITY,
        "mask_rule": _MASK_RULE,
        "empty_cells_value": "NaN",
        "seed_by_day_index": {str(i): int(s) for i, s in enumerate(result["seeds"])},
        "n_timesteps": len(result["timesteps"]),
        "timestep_step_s": 60,
        "node_order_source": str(Path(lcc_mapping_path).name) + " (clave 'nodes', orden node_index 0..N-1)",
        "n_nodes": len(result["node_order"]),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    readme_path.write_text(_README_TEXT)
    return {"npz": npz_path, "metadata": meta_path, "readme": readme_path}


_README_TEXT = """\
# Tensores consolidados — Miraflores laborable 60d (STGNN Fase 2)

**Regenerable** desde los 60 Parquet de `../` (seeds 42..101) + el LCC mapping.
El `.npz` está **gitignored** (voluminoso, regenerable); `metadata.json` y este
README **sí** se versionan.

## Contenido del `.npz`
- `tensor` `[60, 1440, 375, 1]` float32 — canal 0 = `timeLoss` TOTAL (s) por arista
  por intervalo de 60 s. Celdas vacías (sin tráfico) = `NaN`.
- `mask` `[60, 1440, 375]` bool — `True` = tráfico, `False` = vacío
  (`density==0 AND speed.isna()`). ~82% de las celdas son vacías.
- `seeds` `[60]` int — seed por índice de día (eje 0). Día N ↔ seed 42+N.
- `node_order`, `channels`, `timesteps` — alineación (orden = `node_index` del LCC).

## Target
`timeLoss` **TOTAL** (no promedio por-vehículo), por **D-013 + enmienda 2026-06-01**
(`documentation/lean-inception/4-decisiones/DECISIONS.md` § D-013). Se usa la columna
`timeLoss` del Parquet tal cual: sin derivación, sin dividir por flow/entered, sin
regeneración. Eje 0 = días separados por seed (nunca apilados como serie continua).

## Cómo regenerar
```bash
cd ia_prediction_service
.venv/bin/python -m src.data.miraflores_dataset_builder
```
Defaults: lee `simulation/data/datasets/miraflores_laborable_60d/day_seed0{42..101}.parquet`
y `src/data/artifacts/miraflores_graph_lcc_mapping.json`; escribe acá
(`tensors/miraflores_laborable_60d.npz` + `metadata.json` + este README).

Parámetros (ver `build_miraflores_dataset`): `parquet_dir`, `lcc_mapping_path`,
`seeds`, `feature_columns` (default `["timeLoss"]`; ampliar para enriquecer features).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet-dir", default=str(DEFAULT_PARQUET_DIR))
    parser.add_argument("--lcc-mapping", default=str(DEFAULT_LCC_MAPPING))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--features", nargs="+", default=list(DEFAULT_FEATURES),
        help="columnas-feature (default: timeLoss)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="seeds de los días (default: 42..101)",
    )
    args = parser.parse_args()

    result = build_miraflores_dataset(
        parquet_dir=args.parquet_dir,
        lcc_mapping_path=args.lcc_mapping,
        seeds=args.seeds,
        feature_columns=args.features,
    )
    written = save_dataset(result, args.out_dir, lcc_mapping_path=args.lcc_mapping)
    t = result["tensor"]
    valid = float(result["mask"].mean())
    print(f"OK tensor={tuple(t.shape)} mask_validas={100 * valid:.1f}% vacias={100 * (1 - valid):.1f}%")
    for k, p in written.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
