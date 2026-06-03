# Dataset multi-día LABORABLE — Miraflores

**Fecha:** 2026-06-03 (regenerado v2, scale 1.1) · **Estado:** dataset para construir/validar
el pipeline end-to-end. Homogéneo (solo perfil laborable) — ver deuda abajo.

## Versionado
Promovido a ubicación versionada: `simulation/data/datasets/miraflores_laborable_60d/`.
Los 60 `day_seed0NN.parquet` están **ignorados por git** (binarios regenerables vía la
receta — ver `simulation/.gitignore`, regla global `*.parquet`); lo único trackeado de
esta carpeta es **este README**, los `calibracion/*.md` (registros de calibración + gate de
drenaje) y `tensors/{metadata.json, README.md}` (el `.npz` es gitignored). La receta que
regenera los Parquet vive en `simulation/scripts/` (parametrizada, sin paths machine-specific):
`generate_b1_demand.py`, `gen_day.sh`, `compact_day.py`, `batch_days.sh`, `aggregate_sanity.py`.

**Registro de calibración del scale:** `calibracion/SWEEP_C1_RESULTS.md`, `SWEEP_C2_RESULTS.md`
(barridos v1 → elección **0.20**) y `SWEEP_C3_RESULTS.md` (barrido v2 → cliff entre 1.2 y 1.3,
elección **1.1** como borde del cliff con margen). La **aceptación de scale 1.1 para v2** (gate de
drenaje D-014 sobre los 60 días: 48 drenan / 12 de congestión severa) está en
`calibracion/DRENAJE_GATE_60D_RESULTS.md`.

## Qué es
60 días del perfil **laborable** de demanda para `miraflores.net.xml` (v2, 1664 edges
vehiculares; LCC 1660), a **scale=1.1** (el régimen de borde del cliff fijado en C3 y aceptado
para v2 por el gate de drenaje D-014 sobre los 60 días; ver `calibracion/DRENAJE_GATE_60D_RESULTS.md`
y `calibracion/SWEEP_C3_RESULTS.md`). Cada día = una corrida SUMO 24h headless,
control fijo del net (sin TraCI), con `edgeData freq=60` (1440 intervalos de 60s),
compactado a Parquet.

## Generación (reproducible)
- Método B1: `randomTrips.py` con pesos por clase vial (arteria primary/secondary/motorway
  peso 5, resto 1) → `duarouter`. Maquinaria: `simulation/scripts/generate_b1_demand.py` +
  `simulation/scripts/gen_day.sh`.
- **Seeds: 42..101** (60 días). El seed se pasa a **randomTrips, duarouter Y sumo** —varía
  tanto el ruteo como la dinámica de simulación. Día N ↔ `day_seed0NN.parquet`.
- Compactación: `simulation/scripts/compact_day.py` (corre con el **root `.venv`**:
  pandas + pyarrow, mismo stack que el builder F3). El edgeData XML crudo de cada día
  (~434 MB, `edgedata_seed042.xml` = 434.4 MB) se **persiste** como `edgedata_seedNNN.xml`
  (insumo del evaluador de drenaje D-014, gitignored) y se limpia tras evaluar.
- **Muestra cruda de auditoría (v1, scale 0.20 — muestra histórica):** `SAMPLE_edgedata_seed042.xml`
  (+ `SAMPLE_stats_seed042.xml`) NO se promovieron a esta carpeta (pesan ~228 MB, tamaño v1);
  quedan como muestra local de auditoría del mapeo de columnas en
  `scratch/b1_miraflores/dataset_laborable_60d/`.

## Esquema de cada `day_seed0NN.parquet`
- **2 396 160 filas = 1664 edges vehiculares × 1440 timesteps** (medido sobre `day_seed042.parquet`).
- Edges: el edgeData emite los **1664 no-internal**; en el net v2 todos resultan vehiculares
  (passenger), así que el filtro de clase vial no descarta nada hoy (en v1 separaba 381 de 1044).
  El tensor del builder usa los **1660** del LCC (excluye 4 aristas de la islita).
- 8 columnas:

| columna | tipo | unidad / nota |
|---|---|---|
| `edge_id` | string | id del edge SUMO |
| `timestep` | int32 | **begin del intervalo en segundos** (0..86340, paso 60). hora = timestep//3600 |
| `speed` | float32 | velocidad media en el edge (**m/s**) |
| `timeLoss` | float32 | tiempo perdido vs free-flow (s, agregado del intervalo) |
| `traveltime` | float32 | tiempo medio de traversía (s) |
| `flow` | float32 | flujo (veh/h) |
| `density` | float32 | densidad (veh/km) |
| `speedRelative` | float32 | speed / speedLimit (≈1 = free-flow, →0 = atasco) |

- Formato: **Parquet, compresión zstd**. ~7.0 MB/día (`day_seed042` = 6.98 MB), **~420 MB** los 60
  (vs ~434 MB de edgeData XML por día / ~24 GB los 60; **~62× más chico** que el XML por día).

## Convención de edges VACÍOS (sin tráfico en el intervalo, `sampledSeconds=0`)
SUMO no emite métricas para un edge vacío. Se rellena así:
- `flow=0, density=0, timeLoss=0` — **exactos** (sin vehículos no hay flujo/densidad/demora).
- `speed=NaN, speedRelative=NaN, traveltime=NaN` — **indefinidos** (no hay vehículos que promediar).
- **NUNCA `speed`→0** en vacío: eso simularía atasco máximo (jam 5). El vacío es jam 0.

**Señal de presencia / distinción de 3 estados** (validada en datos):
- **Vacío** (jam 0): `density==0` & `timeLoss==0` & `speed` es NaN.
- **Fluido** (jam 0 con tráfico): `density>0`, `speedRelative≈1`, `timeLoss≈0`.
- **Atascado** (jam alto): `density>0`, `speed≈0`, `speedRelative≈0`, `timeLoss` alto;
  `flow` puede ser 0 (nada sale) y `traveltime` NaN (nada completa) — pero `density>0` lo
  separa del vacío. → El discriminador robusto es `density`/`speed-NaN`, **no** `flow`
  (flow=0 ocurre tanto en vacío como en atasco total — ramas del diagrama fundamental).

## Sanity agregado (v2: PENDIENTE de recaracterización)
> **Sección pendiente de recaracterización sobre el dataset v2 (scale 1.1, N=1660).** Los números
> previos eran de **scale 0.20 / v1** (velocidades ~31–33 km/h, perfil jam% bimodal, el episodio
> del seed-081) y **NO aplican** al dataset actual. La recaracterización requiere correr
> `aggregate_sanity.py` (con su hardcode `381`/`548640` corregido a v2) sobre la data nueva + la
> narrativa de régimen del gate de drenaje. **Deuda asignada post-B3.2.c.**
>
> **Lo que ya se sabe del régimen v2** (del gate de drenaje, `calibracion/DRENAJE_GATE_60D_RESULTS.md`):
> 48/60 días drenan; 12/60 (20%) son de **congestión severa de pico PM**
> (`53,55,58,60,62,63,71,83,85,90,97,99`), con un gradiente **continuo (no bimodal)** en el filo;
> ningún día alcanza el colapso-franco del C3. Los 12 NO se descartan (son la cola superior del
> fenómeno, señal para un predictor de demora). **Consecuencia para B4: el split train/val/test
> debe estratificarse por régimen de congestión** (los 12 días severos distribuidos a conciencia),
> NO aleatorio — es la generalización del trato que esta sección daba al seed-081, pero ahora son 12.

## Alcance y deuda
Dataset **solo laborable** (homogéneo, variación por seed). Suficiente para construir/validar
el pipeline end-to-end; NO representa finde/feriado/especial (sin calibrar — ver deuda en
`documentation/ESTADO_Y_PROXIMOS_PASOS.md`). Enriquecer tras cerrar el pipeline.
