# TTH-07 — Cierre de Fase 3 (dataset Parquet + particiones train/valid)

**Rama**: `feature/tth-07`.
**Fecha de cierre**: 2026-05-29.
**Estado al cierre**: F3 verde, 18/18 tests pasando (8 F1 + 5 F2 + 5 F3).

---

## 1. Lo que F3 entregó

```
simulation/
├── src/cerebrovial_simulation/dataset/
│   ├── __init__.py
│   ├── schema.py             columnas declaradas + validate()
│   ├── generate.py           corre seeds × patrones → Parquet
│   └── partitions.py         CT-07.4 — train (seeds 1-20) + valid (21-25) sin overlap
├── scripts/
│   └── regenerate_dataset.sh CT-07.7 punto c — reproducible
└── tests/
    └── test_dataset_generates.py  5 tests (c1-c5)
```

## 2. Schema canónico (CT-07.3)

Una fila por (seed, pattern, t_sim_s, direction). Columnas:

| Columna | Tipo | Notas |
|---------|------|-------|
| seed | int32 | Determinístico via `--seed N` en sumo |
| pattern | string | "am_peak", "pm_peak", "offpeak", "weekend" |
| t_sim_s | float32 | end-of-bucket en seg sim (múltiplo de 60) |
| direction | string | "N", "S", "E", "W" |
| mean_speed_mps | float32 | `<edge speed="...">` del bucket |
| n_vehicles | int32 | `<edge departed="...">` (vehículos spawned en el bucket) |
| queue_length_m | float32 | Suma de `maxJamLengthInMeters` sobre lanes del aproche |
| max_speed_mps | float32 | 13.89 constante |
| ratio | float32 | mean_speed / max_speed |
| jam_level | int8 | 0-5 derivado D-009 |

**Decisión** sobre `n_vehicles`: usar `departed` (no `entered`). Las
edges de aproche en este net son source-edges alimentadas por `<flow>`;
`entered` siempre es 0 porque no hay edge anterior. `departed` es el
count de spawns en el bucket — para AM-peak (N=2400 v/h), bucket 60s:
~40 spawns/bucket esperados, validado en test (c4): N=40, S=40, E=10,
W=10.

Documentado in-line en `generate.py` línea 64-70.

## 3. Particiones (CT-07.4)

| Partición | Patrones | Seeds | Corridas |
|-----------|----------|-------|----------|
| Train | am_peak, pm_peak, offpeak, weekend | 1-20 | 80 |
| Valid | am_peak, pm_peak, offpeak, weekend | 21-25 | 20 |

**Sin overlap por construcción**. `partitions.assert_no_overlap()` es
failsafe al import. Test c1 confirma.

D-008 ("escenarios SUMO distintos para evitar fuga de información") se
satisface vía seed-disjointness. El "preferentemente patrones distintos"
queda como problema de TTH-09 si necesita pattern-holdout para
generalización (no es requisito duro de CT-07.4).

## 4. Decisión de implementación — XML + pyarrow downstream

Confirma corrección 2 del plan (fallback documentado en F1):
- SUMO escribe `edgedata.xml` + `lanearea.xml` (multi-interval; arrow
  writer no soporta append).
- `generate.py` parsea ambos con `xml.etree.ElementTree`, agrega por
  dirección/bucket, deriva ratio + jam_level con `jam_level.py` (D-009),
  y persiste el dataset final como Parquet (single-write OK).

Tiempo total smoke `--quick` (8 corridas, 4 train + 4 valid): ~10s
wallclock total. Full (100 corridas): ~2 min estimados.

## 5. Tests verdes

```
$ .venv/bin/pytest tests/ -v
... 18 passed in 3.74s ...
tests/test_dataset_generates.py::test_c1_train_valid_no_overlap PASSED
tests/test_dataset_generates.py::test_c2_generate_one_produces_valid_parquet PASSED
tests/test_dataset_generates.py::test_c3_ranges_within_bounds PASSED
tests/test_dataset_generates.py::test_c4_am_peak_seed1_invariants PASSED
tests/test_dataset_generates.py::test_c5_generate_all_quick PASSED
```

## 6. Smoke ejecutable

```bash
$ cd simulation && bash scripts/regenerate_dataset.sh --quick
Regenerando dataset desde /Users/rasec/Tesis/CerebroVial/simulation
  SUMO_HOME=...../sumo
  python=...../python
  train: am_peak seed=1
  train: pm_peak seed=1
  train: offpeak seed=1
  train: weekend seed=1
  valid: am_peak seed=21
  valid: pm_peak seed=21
  valid: offpeak seed=21
  valid: weekend seed=21

Train: 4 archivos en .../data/train
Valid: 4 archivos en .../data/valid
OK
```

## 7. Próximo paso — F4

F4 abre el contrato del motor (`engine_recommend_contract.md`) y
construye el adaptador TraCI↔HTTP con:
- Sensado del motor cada 30s sim (TraCI vehicle/edge state).
- Aplicación del plan en borde de **ciclo** (Catch A), no de sub-fase.
- Expansión PhaseTiming → 3 sub-fases SUMO (corrección 3).
- Flow como tasa de paso, no ocupación instantánea (Catch C).

Stop-conditions a vigilar (Catch C): bajo AM-peak, engine debe rutear a
`mode="max_pressure"` (flow_total > 1500). Si rutea a "webster", la
estimación de flow está mal.
