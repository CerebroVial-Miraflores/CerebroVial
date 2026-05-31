# TTH-11 — Handoff: dataset perfil-día generado + fixes de pipeline

**Rama**: `feature/tth-11-hiperparametros-temporales`.
**Fecha de cierre**: 2026-05-31.
**Estado al cierre**: dataset perfil-día completo y verificado (100 corridas);
cobertura 25/25 con criterio temporal nuevo; `pytest test_perfil_dia_coverage`
5/5. Próximo paso = Fase 1 del loader de TTH-11 (ver §5).

---

## 1. Estado del dataset

El dataset de entrenamiento del predictor **cambió de NATURALEZA**, no es una
actualización del mismo dato:

- **Antes**: 4 patrones de demanda **constante de 20 min** (am_peak/pm_peak/
  offpeak/weekend, 20 buckets/corrida).
- **Ahora**: 4 **perfiles-día de 24h** con dinámica temporal real (valle→pico→
  valle, rampas explícitas): `laborable` / `finde` / `feriado` / `especial`.

**Forma**: 100 corridas (80 train + 20 valid, seeds 1-20 / 21-25 sin overlap),
**1440 buckets × 4 dirs = 5760 filas/corrida**. `schema.validate` (D-009) OK en
las 100. Eje dominante N-S, secundario E-W = round(N-S / 4).

**Histograma jam (verificación post-full, 25 seeds, dir N)**:
- `laborable` {0:145, 1:12151, 2:17075, 3:**895**, 4:**5734**} — dos spikes agudos
  con hombro jam-3 **transitorio** + ápice jam-4 (rise/fall 1→2→3→4→3→2).
- `especial` {1:13218, 2:4832, 3:133, 4:**17817**} — meseta jam-4 sostenida (12h).
- `finde` / `feriado` — tope jam-2; jam-0 presente en valles.
- **Cobertura 25/25** en los 4 perfiles con el criterio temporal nuevo.

La separación `laborable`↔`especial` es **temporal/dinámica** (dos picos agudos vs
meseta larga), no de nivel de pico — que es justo la señal que el predictor
lookback×horizonte debe aprender.

**Los parquets viven en `data/`** (gitignored, regenerables):
`python -m cerebrovial_simulation.dataset.generate` (full, ~31 min) o `--quick`
(8 corridas). Rutas: `python scripts/generate_perfil_dia_routes.py`.

Archivos clave (commiteados, ver §6):
[perfil_dia_params.yaml](../../../simulation/conf/scenarios/perfil_dia_params.yaml),
[generate_perfil_dia_routes.py](../../../simulation/scripts/generate_perfil_dia_routes.py),
4 `*.sumocfg` + 4 `routes/*.rou.xml`,
[partitions.py](../../../simulation/src/cerebrovial_simulation/dataset/partitions.py),
[coverage_perfil_dia.py](../../../simulation/src/cerebrovial_simulation/dataset/coverage_perfil_dia.py),
[test_perfil_dia_coverage.py](../../../simulation/tests/test_perfil_dia_coverage.py).

**Desacople deliberado**: el set perfil-día es **nuevo y separado**; los 4 patrones
constantes de TTH-07 (`pattern_params.yaml`, sus `*.sumocfg`/`*.rou.xml`,
`webster_fixed.py`, `kpis/`, `traci_adapter/`, `test_patterns_run.py`,
`test_kpis.py`, `test_traci_e2e.py`) quedaron **intactos** — los comparte el
pipeline de comparación de control de TTH-07, que quiere carga estable.

## 2. Decisiones de fondo tomadas esta sesión

- **D-009 realineado a escala Waze (80/60/40/20)**, reemplazando 90/70/50/30.
  *Motivo*: paridad con la fuente de producción de TTH-09 (jam_level real de Waze).
  *Impacto*: los conteos "sustained jam≥3" del handoff de TTH-07 quedaron en escala
  vieja; **NO se recalcularon** (integridad histórica) — ya tienen nota de reenvío.
  Andres al tanto.
- **Bug edge-vacío arreglado**: un bucket sin observación (`sampledSeconds=0`) ya
  **NO** se mapea a jam 5; ahora → jam 0 / ratio 1.0 (calle vacía = flujo libre).
  Distingue del caso velocidad-cero-genuina-con-vehículos (jam 5 = atasco real).
  Aplicado en `generate.py` y `coverage_check.py`.
- **Bimodalidad de la 4-way**: solo **jam 2** (bajo capacidad) y **jam 4** (sobre
  capacidad) son regímenes **estables**; **jam 3 es intrínsecamente transitorio**
  (cliff de capacidad ~2520 vph, inestable — verificado en un barrido de 7 valores).
  Por eso `laborable` cruza el cliff a propósito (picos jam-4 con hombros jam-3) en
  vez de buscar un jam-3 sostenido inexistente. Documentado en `perfil_dia_params.yaml`
  y `coverage_perfil_dia.py`, **marcado para CT-11.6**.
- **jam 5 fuera de scope del dataset**: requiere mecanismo de **bloqueo** (spillback
  forzado / incidente / cierre), no demanda. Tarea de **TTH-09**.
- **Mapa OSM de Andres (`miraflores.net.xml`) NO se usó**: el pipeline está acoplado
  a la 4-way (naming `*_in`/`*_out`, esquema NS/EW, detectores LA_*). Migrar a la red
  OSM multi-intersección es trabajo de **TTH-09**. El archivo queda **untracked**
  esperando.
- **Fix spillback per-lane** en `coverage_perfil_dia.py`: el checker comparaba la cola
  **sumada sobre los 3 carriles** del aproche vs el largo de **un** aproche (bug de
  unidades; disparaba a ~30% de ocupación por carril). Corregido a **per-lane**
  (carril peor vs 90% del aproche). No hay spillback real: el peor carril llega a
  **71%** (especial, ~205m de 289.6m), sin teleports, demanda ≈ capacidad.
- **Criterio de cobertura sobre el eje dominante N-S**: la firma de congestión del
  perfil vive en la vía dominante; el E-W secundario acumula cola de rojo (ruido por
  diseño). Aplicado consistente en `coverage_perfil_dia.py`.

## 3. Deuda pendiente (explícita)

- **Bibliografía**: 4 refs (Chung 2014, Wang 2022, Singh 2025, Wen 2023) agregadas a
  mano a `markdownToDocx/referencias.bib` (rama `comparedocs`) — **se PIERDEN en el
  próximo re-export de Zotero**. *Qué hacer*: cargarlas en Zotero para que sobrevivan.
- **`test_b2` de TTH-07 sigue en `xfail(strict)`**: los patrones constantes no llegan
  a jam≥3 bajo la escala Waze. Recalibrar esos flujos es **tarea aparte** (no se hizo,
  decisión consciente — desacople de TTH-07).
- **jam 5** y **mapa OSM multi-intersección**: ambos para **TTH-09** (ver §2).

## 4. Estado de TTH-11 (los CTs del spike)

Doc: [INVESTIGACION_HIPERPARAMETROS_TEMPORALES.md](../../docs/INVESTIGACION_HIPERPARAMETROS_TEMPORALES.md).

**Hecho (parte académica, commit `e9ff8d9a`)**:
- **CT-11.1** — propósito y estructura.
- **CT-11.2** — los 4 hiperparámetros (Δt_in, lookback, horizonte, cadencia):
  definición, bibliografía, rango candidato, recomendación preliminar
  (lookback=30min, horizonte=60min).
- **CT-11.3** — revisión bibliográfica (≥5 fuentes).
- **CT-11.8** — cierre de Δt_in = 60 s (cerraba CT-07.3).

**Pendiente (empírico, requiere entrenamiento — placeholders en el doc)**:
- **CT-11.4** — exploración empírica: ≥3 combinaciones lookback×horizonte entrenadas
  con 4 métricas. **Ahora desbloqueado** por el dataset perfil-día (ver §5).
- **CT-11.5** — tabla resumen / contrato para TTH-09.
- **CT-11.6** — limitaciones y trabajo futuro: incorporar la **bimodalidad** (§2) y el
  **81.3% heredado** (ver §5).
- **CT-11.7** — cierre de doble propósito.

## 5. Punto exacto de retome (lo más importante)

**Próximo paso: Fase 1 del loader de TTH-11** —
[tth11_temporal_loader.py](../../../ia_prediction_service/src/data/tth11_temporal_loader.py)
(ya escrito, **untracked**).

**RE-CORRER el gate de shapes**: el barrido lookback×horizonte daba **CERO secuencias**
sobre las corridas de 20 min. `window_series` calcula
`n_seq = n_buckets - lookback_steps - horizonte_steps + 1`; con n=20 y, p. ej.,
lookback=30 + horizonte=60 pasos → `n_seq < 0` → arrays vacíos. **Ahora con 1440
buckets/corrida** las combinaciones candidatas C1-C4 **SÍ** tienen datos
(`1440 - 30 - 60 + 1 = 1351` secuencias/serie). *Confirmar shapes no vacíos por
combinación*, después:
1. **CT-11.4** — entrenamiento exploratorio (≥3 combinaciones, 4 métricas).
2. **CT-11.5** — tabla resumen / contrato para TTH-09.
3. **CT-11.6** — limitaciones: bimodalidad jam2/jam4 + jam-3 transitorio (§2).
4. **CT-11.7** — cierre.

**81.3% heredado (D-005)** sigue **pendiente de validar**: cuando se entrene, comparar
contra el número real y marcar **reproducido o no** en el doc.

## 6. Commits de esta sesión

```
1ebdaf0a  jam_level alineado a escala Waze 80/60/40/20 (D-009 realineado)
5af501b0  fix edge-vacío: bucket sin dato ≠ velocidad cero (jam 0 vs jam 5)
664ff904  tests de regresión: cortes nuevos + ausente-vs-cero; xfail cobertura
3c324829  nota de reenvío en handoff TTH-07 por realineación de escala
ee889539  feat: perfiles-día de demanda variable 24h (params + generador + 4 rou.xml)
dac02261  feat: escenarios SUMO 24h + repunte de partitions (cambio de naturaleza)
c9e744ef  test: cobertura perfil-día con criterio temporal (CT-07.2 redefinido)
29718d51  fix: spillback per-lane (bug de unidades) en la cobertura
```

Árbol limpio salvo 2 untracked esperando tarea futura: `tth11_temporal_loader.py`
(Fase 1, §5) y `miraflores.net.xml` (mapa OSM, TTH-09, §2). TTH-07 sin tocar.
Sin push/PR/merge.
