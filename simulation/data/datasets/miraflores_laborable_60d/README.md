# Dataset multi-día LABORABLE — Miraflores

**Fecha:** 2026-06-01 · **Estado:** dataset para construir/validar el pipeline end-to-end.
Homogéneo (solo perfil laborable) — ver deuda abajo.

## Versionado
Promovido a ubicación versionada: `simulation/data/datasets/miraflores_laborable_60d/`.
Los 60 `day_seed0NN.parquet` están **ignorados por git** (binarios regenerables vía la
receta — ver `simulation/.gitignore`, regla global `*.parquet`); lo único trackeado de
esta carpeta es **este README**. La receta que los regenera vive en `simulation/scripts/`
(parametrizada, sin paths machine-specific): `generate_b1_demand.py`, `gen_day.sh`,
`compact_day.py`, `batch_days.sh`, `aggregate_sanity.py`.

**Registro de calibración del scale (barridos C1/C2):** `calibracion/SWEEP_C1_RESULTS.md`
y `calibracion/SWEEP_C2_RESULTS.md` — documentan el barrido de `scale`, el cliff de
colapso y por qué se fijó **0.20** (rescatados del scratch de calibración a esta ubicación
versionada para que el registro metodológico no se pierda al limpiar el debris).

## Qué es
60 días del perfil **laborable** de demanda para `miraflores.net.xml`, a **scale=0.20**
(el nivel validado en C2 sobre la 24h continua contra el colapso por carryover; ver
`scratch/b1_miraflores/SWEEP_C2_RESULTS.md`). Cada día = una corrida SUMO 24h headless,
control fijo del net (sin TraCI), con `edgeData freq=60` (1440 intervalos de 60s),
compactado a Parquet.

## Generación (reproducible)
- Método B1: `randomTrips.py` con pesos por clase vial (arteria primary/secondary/motorway
  peso 5, resto 1) → `duarouter`. Maquinaria: `simulation/scripts/generate_b1_demand.py` +
  `simulation/scripts/gen_day.sh`.
- **Seeds: 42..101** (60 días). El seed se pasa a **randomTrips, duarouter Y sumo** —varía
  tanto el ruteo como la dinámica de simulación. Día N ↔ `day_seed0NN.parquet`.
- Compactación: `simulation/scripts/compact_day.py` (corre con el `.venv` del proyecto:
  pandas + pyarrow, mismo stack que el builder F3). El XML crudo de 220–230 MB de cada día
  se DESCARTA tras compactar.
- **Muestra cruda de auditoría:** `SAMPLE_edgedata_seed042.xml` (+ `SAMPLE_stats_seed042.xml`)
  NO se promovieron a esta carpeta (pesan ~228 MB); quedan como muestra local de auditoría
  del mapeo de columnas en `scratch/b1_miraflores/dataset_laborable_60d/`.

## Esquema de cada `day_seed0NN.parquet`
- **548 640 filas = 381 edges vehiculares × 1440 timesteps.**
- Edges: se filtran los 1044 que emite el edgeData (todos los no-internal, incl. peatonales)
  a los **381 vehiculares (passenger)** — los demás son ruido siempre-vacío para autos.
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

- Formato: **Parquet, compresión zstd**. ~1.3 MB/día, **~76 MB** los 60 (vs ~13.7 GB de XML;
  **174× más chico** que el XML por día).

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

## Sanity agregado (60 días; ver `simulation/scripts/aggregate_sanity.py`)
- **Forma:** los 60 días = 548 640 filas, 381 edges, 1440 timesteps. Consistente.
- **Sin colapso tipo 0.35:** velocidad global media ~31–33 km/h por día; ninguna se clava en
  un dígito sostenido. Perfil de jam% **bimodal y con recuperación** (valle ~15% → pico AM
  ~20.5%@09-10h → meseta ~17% → pico PM ~22.8%@20h → noche ~16%): congestión real que drena.
- **Firma traveltime-NaN** (gridlock): NaN entre edges-con-presencia se mantiene bajo
  (~3–9% en pico) en 59/60 días.
- **seed 081 — día de congestión máxima del set, conservado deliberadamente:** episodio
  TRANSITORIO de gridlock en el pico PM (20h cae a 13.4 km/h, NaN-pres 30.5%, jam 62.9%) que
  **RECUPERA** (24→28 km/h hacia 21–23h). NO es un colapso 0.35 (ése era 0.4 km/h sostenido
  14h sin recuperar); es la cola pesada de la variación normal del pico PM a scale 0.20. Es
  el día más congestionado del set; se conserva a propósito. **Tener presente al hacer el
  split train/val/test para distribuirlo conscientemente** (no dejarlo aislado en un solo
  fold sin querer).

## Alcance y deuda
Dataset **solo laborable** (homogéneo, variación por seed). Suficiente para construir/validar
el pipeline end-to-end; NO representa finde/feriado/especial (sin calibrar — ver deuda en
`documentation/ESTADO_Y_PROXIMOS_PASOS.md`). Enriquecer tras cerrar el pipeline.
