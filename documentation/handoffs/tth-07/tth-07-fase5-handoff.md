# TTH-07 — Cierre de Fase 5 (Webster offline + KPIs comparativos)

**Rama**: `feature/tth-07`.
**Fecha de cierre**: 2026-05-29.
**Estado al cierre**: F5 verde, 27/27 tests pasando (8 F1 + 5 F2 + 5 F3
+ 5 F4 + 4 F5). Catch B implementado.

---

## 1. Lo que F5 entregó

```
simulation/
├── src/cerebrovial_simulation/
│   ├── fixed_control/
│   │   ├── __init__.py
│   │   └── webster_fixed.py     Catch B: Webster offline directo, no engine HTTP
│   └── kpis/
│       ├── __init__.py
│       ├── collect.py           KPIs desde tripinfo.parquet + summary.parquet + lanearea.xml
│       └── run_comparison.py    Orquesta adaptive vs fixed × 4 patrones × N seeds
├── conf/network/
│   └── webster_fixed.add.xml    Generado por webster_fixed.compute_from_pattern("am_peak")
├── scripts/
│   └── run_comparison.sh        CLI reproducible
└── tests/
    └── test_kpis.py             4 tests (e1-e4)
```

## 2. Catch B verificado — Webster fijo NO via engine

`webster_fixed.compute(...)` implementa la fórmula directa transcrita
literalmente de [webster.py:7-12](../../../core_management_api/src/control/application/webster.py#L7-L12)
**sin invocar HTTP**:

```
y_NS = flow_NS / sat_NS    = 4800 / 10800 = 0.444
y_EW = flow_EW / sat_EW    = 1200 / 7200  = 0.167
Y    = 0.611
C_opt = (1.5 × 10 + 5) / (1 − 0.611) = 51.4 s
green_NS = (51.4 − 10) × (0.444 / 0.611) = 30.1 s
green_EW = (51.4 − 10) × (0.167 / 0.611) = 11.3 s
```

Resultado persistido en `conf/network/webster_fixed.add.xml` con
`programID="webster_fixed"` y las 6 sub-fases expandidas (yellow + all_red
preservados — corrección 3 aplicada también acá).

**Por qué importa (Catch B)**: AM-peak `flow_total = 6000 ≥ 1500
PEAK_THRESHOLD`. Si llamáramos al engine HTTP, rutearía a `max_pressure`
y el "baseline Webster" sería en realidad `max_pressure` congelado —
viciaría la comparación adaptive vs. fixed-time-Webster. Catch B
elimina esta dependencia: F5 fixed NO requiere `invoke up`.

Test e1 verifica MTC bounds; e3 verifica consistencia con flujos del
YAML; e4 verifica `WebsterInfeasible` para Y ≥ 0.95.

## 3. KPIs producidos (CT-07.6)

Por corrida (mode × pattern × seed):

| KPI | Fuente |
|-----|--------|
| `sim_duration_s` | `summary.parquet` última `step_time` |
| `mean_travel_time_s` | `tripinfo.parquet` mean `tripinfo_duration` |
| `total_delay_s` | `tripinfo.parquet` sum `tripinfo_timeLoss` |
| `throughput_veh_per_h` | `n_arrived × 3600 / sim_duration` |
| `max_queue_m_{N,S,E,W}` | `lanearea.xml` max `maxJamLengthInMeters` por bucket, sum sobre lanes |
| `mean_queue_m_{N,S,E,W}` | `lanearea.xml` mean por bucket |

**Hallazgo**: SUMO 1.26 prefija columnas Parquet con `step_` (summary)
y `tripinfo_` (tripinfo), y serializa numéricos como strings ("46.00").
`collect.py` tolera ambos prefijos y aplica `float()` para coercer.
Documentado in-line.

## 4. Smoke ejecutable

```bash
$ bash scripts/run_comparison.sh --quick --skip-adaptive --end 300
[fixed_webster] am_peak  seed=1: travel=56.2s delay_total=5737s throughput=4912v/h max_q_NS=99m
[fixed_webster] pm_peak  seed=1: travel=56.5s delay_total=6157s throughput=5141v/h max_q_NS=96m
[fixed_webster] offpeak  seed=1: travel=50.6s delay_total=429s  throughput=518v/h  max_q_NS=0m
[fixed_webster] weekend  seed=1: travel=52.1s delay_total=897s  throughput=1096v/h max_q_NS=4m
KPIs persistidos en .../data/kpis/comparison.csv
```

Sin spillback (max_q_NS=99m < 261m threshold), throughput diferenciado
por patrón (am_peak ~5000 v/h vs offpeak ~500), delay escala con
demanda. KPIs comparables limpiamente con la rama adaptive cuando el
motor esté corriendo.

## 5. Modo adaptive (requiere motor)

```bash
# Terminal 1
invoke up

# Terminal 2
bash scripts/run_comparison.sh           # 3 seeds × 4 patrones × 2 modes = 24 corridas
# o
bash scripts/run_comparison.sh --quick   # 1 seed × 4 × 2 = 8 corridas
```

El script health-checkea `/control/health` antes de arrancar el modo
adaptive. Si el motor no responde, skip adaptive y reporta warning
(`--skip-adaptive` suprime el warning).

## 6. Tests verdes

```
$ .venv/bin/pytest tests/ -v
... 27 passed in 4.98s ...

tests/test_kpis.py::test_e1_webster_offline_within_mtc_bounds PASSED
tests/test_kpis.py::test_e2_fixed_webster_run_produces_finite_kpis PASSED
tests/test_kpis.py::test_e3_compute_from_pattern_consistent_with_manual PASSED
tests/test_kpis.py::test_e4_infeasible_raises PASSED
```

## 7. Próximo paso — F6

F6 cierra TTH-07:
- `simulation/README.md` final con instrucciones reproducibles por un tercero (CT-07.7).
- `documentation/handoffs/tth-07/tth-07-cierre.md` con estado de los 8 CTs (07.1-07.8).
- Tests CT-07.8.a/b/c verdes (ya cubiertos por F1/F2/F4).
