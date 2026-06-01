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
../.venv/bin/python -m src.data.miraflores_dataset_builder
```
Defaults: lee `simulation/data/datasets/miraflores_laborable_60d/day_seed0{42..101}.parquet`
y `src/data/artifacts/miraflores_graph_lcc_mapping.json`; escribe acá
(`tensors/miraflores_laborable_60d.npz` + `metadata.json` + este README).

Parámetros (ver `build_miraflores_dataset`): `parquet_dir`, `lcc_mapping_path`,
`seeds`, `feature_columns` (default `["timeLoss"]`; ampliar para enriquecer features).
