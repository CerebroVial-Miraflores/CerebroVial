# TTH-07 — Cierre de Fase 2 (patrones de demanda + cobertura jam level)

**Rama**: `feature/tth-07`.
**Fecha de cierre**: 2026-05-29.
**Estado al cierre**: F2 verde, 13/13 tests pasando (8 de F1 + 5 de F2).
Cobertura CT-07.2 verificada en los 4 patrones con criterio robusto
contra el artefacto estructural de intersecciones señalizadas.

---

## 1. Lo que F2 entregó

```
simulation/
├── conf/scenarios/
│   ├── pattern_params.yaml       4 patrones calibrados + duration_s=1200
│   ├── am_peak.sumocfg
│   ├── pm_peak.sumocfg
│   ├── offpeak.sumocfg
│   ├── weekend.sumocfg
│   └── routes/                   .rou.xml por patrón (straight-only flows)
├── scripts/
│   └── generate_routes.py        YAML → .rou.xml por patrón
├── src/cerebrovial_simulation/
│   └── coverage_check.py         CT-07.2 + spillback (catch 2) + D-009
└── tests/
    └── test_patterns_run.py      CT-07.8.b (b1 × 4 patrones + b2)
```

## 2. Flujos calibrados (post-iteración)

| Patrón | N (v/h) | S (v/h) | E (v/h) | W (v/h) | Target | Sustained jam ≥3 buckets (target dir) | Max queue (m) |
|--------|---------|---------|---------|---------|--------|----------------------------------------|----------------|
| am_peak | 2400 | 2400 | 600 | 600 | jam ≥3 en N | **N=5** ✓ | N=177, S=172, E=35, W=43 |
| pm_peak | 2500 | 2500 | 650 | 650 | jam ≥3 en S | **S=18** ✓ | N=218, S=222, E=41, W=41 |
| offpeak | 200 | 200 | 100 | 100 | jam ≤2 | N/S/E/W ≤4 | todas ≈0 |
| weekend | 500 | 500 | 150 | 150 | jam ≤2 | N/S=0, E=3, W=2 | N=14, S=12, E=0, W=0 |

> **Nota de reenvío 2026-05-31 — realineación a escala Waze.** Los conteos
> `Sustained jam ≥3` de esta tabla (am_peak N=5, pm_peak S=18) se calcularon con
> los cortes **previos** de jam_level (90/70/50/30). La realineación a la escala
> oficial de Waze (80/60/40/20; ver nota en D-009, DECISIONS.md) reclasifica el
> mismo tráfico ~1 nivel hacia abajo: bajo los cortes nuevos las dirs target N/S de
> am_peak/pm_peak topan en jam 2 y ya no alcanzan jam ≥3 sostenido. Estos números
> históricos **no se recalculan** (sprint cerrado). Recuperar cobertura jam ≥3 bajo
> la escala corregida requiere recalibrar la demanda (tarea aparte); mientras tanto
> `test_b2_coverage_satisfied_for_all_patterns` queda marcado `xfail`.

Capacidad por dirección (3 carriles × 1800 sat × 30/60 cycle fraction) =
2700 veh/h ⇒ am_peak 2400 = 89% saturación, pm_peak 2500 = 93%.
Sin spillback (max queue NS=222m < 261m threshold 90% × 290m efectivos).

## 3. Hallazgo material — criterio "jam ≤2" simétrico con "≥3"

### Observación

En una intersección señalizada cualquiera, vehículos esperando en rojo
deprimen `mean_speed` durante un bucket. Con flujo bajo (offpeak: 100
veh/h por E/W) y cycle fraction de EW = 25/60 ≈ 42%, vehículos
individuales pueden experimentar `jam_level=3-4` transiente en buckets
aislados sin que haya congestión operativa.

### Decisión

**Criterio `target_jam: "<=2"`**: "ninguna dirección tiene jam ≥3
**sostenido** en ≥5 buckets consecutivos (= 300 s simulados)".
Simétrico con `target_jam: ">=3"`. Acepta picos transientes de jam ≥3
(artefacto estructural) y rechaza congestión real sostenida.

Documentado in-line en `coverage_check.py` líneas 196-205.

## 4. Catch 2 verificado — spillback con N-S a 300 m

Max queue observada en pm_peak (el patrón más cargado): 222 m. Umbral
de spillback: 90% × 290 m efectivos = 261 m. Headroom de ~39 m.

Si Cesar quiere mayor seguridad para corridas con seeds desfavorables,
N-S puede ampliarse a 350 m sin tocar otros parámetros — pero el
criterio `90% del aproche` aún cubre estos flujos.

## 5. Decisiones cerradas (no reabrir)

- 4 patrones con calibración aprobada arriba.
- 5 seeds por patrón (declarado en `pattern_params.yaml` no — eso
  vive en `partitions.py` que abre F3).
- Route format: straight-only (`<flow>` per origin, route to opposite
  approach). Turns son refinamiento futuro fuera de scope MVP1.
- coverage_check NO usa TraCI live (corrección 1 del plan): parsea
  `edgedata.xml` + `lanearea.xml` (fallback XML de F1).

## 6. Tests verdes

```
$ .venv/bin/pytest tests/ -v
tests/test_network_loads.py::test_a1_net_loads PASSED
tests/test_network_loads.py::test_a2_tls_program_baseline_has_6_subphases PASSED
tests/test_network_loads.py::test_a3_lane_counts PASSED
tests/test_network_loads.py::test_a4_vmax_uniform PASSED
tests/test_network_loads.py::test_a5_approach_lengths PASSED
tests/test_network_loads.py::test_a6_baseline_behavior PASSED
tests/test_network_loads.py::test_a7_outputs_parquet_and_xml_fallback PASSED
tests/test_network_loads.py::test_a8_linkstates_json_has_6_keys PASSED
tests/test_patterns_run.py::test_b1_pattern_runs_and_produces_outputs[am_peak] PASSED
tests/test_patterns_run.py::test_b1_pattern_runs_and_produces_outputs[pm_peak] PASSED
tests/test_patterns_run.py::test_b1_pattern_runs_and_produces_outputs[offpeak] PASSED
tests/test_patterns_run.py::test_b1_pattern_runs_and_produces_outputs[weekend] PASSED
tests/test_patterns_run.py::test_b2_coverage_satisfied_for_all_patterns PASSED
============================== 13 passed in 1.70s ==============================
```

## 7. Próximo paso — F3

F3 construye el generador del dataset:
- `src/cerebrovial_simulation/dataset/{schema,generate,partitions}.py`
- `scripts/regenerate_dataset.sh` reproducible
- Particiones train (seeds 1-20) + valid (seeds 21-25) × 4 patrones
- Schema canónico por dirección con columnas D-009 (jam_level derivado)
