# TTH-07 — Handoff de cierre del sprint

**Rama**: `feature/tth-07` (desde `feature/tth-07-fase0-docs` ←
`master@85d56bb4` = merge PR #34 cierre TTH-08 F9).
**Fecha de cierre**: 2026-05-29.
**Estado al cierre**: **TTH-07 Done** — 8/8 CTs cubiertos, 27/27 tests
verdes. Sprint ejecutado en 6 fases (F0-F6) sobre una sola rama, un
commit por fase.

---

## 1. Estado de los 8 CTs (CT-07.1 a CT-07.8)

| CT | Descripción | Estado | Dónde se valida |
|----|-------------|--------|-----------------|
| **CT-07.1** | Topología 4-vías cargable, parámetros legibles (no hardcode) | ✅ Validado | F1 — `test_network_loads.py::a1-a8` |
| **CT-07.2** | ≥4 patrones de demanda + cobertura jam (≥1 ≥3, ≥1 ≤2) + seeds reproducibles | ✅ Validado | F2 — `test_patterns_run.py::b1,b2` |
| **CT-07.3** | Generador dataset reproducible + esquema documentado + Parquet | ✅ Validado | F3 — `test_dataset_generates.py::c1-c5` |
| **CT-07.4** | Particiones train/valid con seeds (y patrones) sin overlap | ✅ Validado | F3 — `partitions.py` + test c1 |
| **CT-07.5** | Integración TraCI ↔ motor bidireccional; corrida e2e demostrable | ✅ Validado | F4 — `test_traci_e2e.py::d1-d5` (mock engine) |
| **CT-07.6** | KPIs comparativos adaptive vs fixed (Webster fijo) | ✅ Validado | F5 — `test_kpis.py::e1-e4` + `comparison.csv` |
| **CT-07.7** | Documentación reproducible por un tercero (cargar topo, correr patrones, regenerar dataset, e2e, comparación) | ✅ Validado | F6 — `simulation/README.md` |
| **CT-07.8** | Tests integración mínimos (carga net, corre patrones N steps, e2e mecánico) | ✅ Validado | F1 (a) + F2 (b) + F4 (c) — 27 tests verdes |

**Resumen**: 8 de 8 CTs validados. **TTH-07 Done**.

## 2. Mapa fase → commit

| Fase | Commit | Alcance |
|------|--------|---------|
| F0 | `149d991b` | handoff de kickoff + smoke toolchain S0 |
| F1 | `993f336d` | scaffold simulation/ + CT-07.1 topología genérica + tests a1-a8 |
| F2 | `21cc82ce` | patrones AM/PM/offpeak/weekend + cobertura jam + tests b1-b2 |
| F3 | `939361b9` | dataset Parquet + particiones train/valid + tests c1-c5 |
| F4 | `d8194a44` | contrato motor + adaptador TraCI ↔ HTTP + e2e + tests d1-d5 |
| F5 | `64aae34a` | Webster offline (Catch B) + KPIs comparativos + tests e1-e4 |
| F6 | (este) | README final CT-07.7 + handoff de cierre |

## 3. Catches incorporados (planificación → ejecución)

| Catch | Detección | Implementación | Verificación |
|-------|-----------|----------------|--------------|
| **C1 — Puerto motor 8001** | Cesar al revisar plan; default 8000 era error mío. Verificado `docker-compose.yml:37`. | `ENGINE_URL=http://localhost:8001/control/recommend` default. | `engine_recommend_contract.md` §1 + `.env.example`. |
| **C2 — Spillback con N-S a 300m** | Análisis Cesar: capacidad 2700 v/h vs target jam ≥3 requiere ~85% sat ⇒ colas grandes ⇒ N-S debe ser 300m no 250m. | `network_params.yaml` ns approach_length_m=300. | F2 `coverage_check.py` verifica `maxJamLengthInMeters ≤ 0.9 × largo`. Test b2 con pm_peak 2500 v/h max_queue=222m < 261m threshold. |
| **C3 — Aplicar en borde de fase (luego CICLO)** | Plan original aplicaba en cada borde sub-fase ⇒ reset a NS_g cada 30s ⇒ hambreado EW. Cesar (Catch A) refinó a borde de ciclo (EW_r → NS_g). | `tllogic_applier.maybe_apply`: solo aplica si `getPhase==5` Y `getNextSwitch==sim_time`. | F4 test d3: 180s sim, engine llamado 6 veces, setProgramLogic ≤ 3 aplicaciones (no 6). |
| **A — Borde de ciclo, no de sub-fase** | Cesar al revisar plan. Mismo que C3 después del refinamiento. | Idem. | Idem. |
| **B — Webster offline directo, NO via motor** | Cesar: AM-peak ≥ PEAK_THRESHOLD ⇒ engine rutea a max_pressure ⇒ "Webster fijo via engine" sería realmente max_pressure congelado. | `fixed_control/webster_fixed.py` implementa fórmula transcrita literal de `webster.py:7-12` sin HTTP. | F5 test e1: MTC bounds + suma consistente. Test e3: consistencia con flujos YAML. F5 NO requiere `invoke up`. |
| **C — Flow es tasa de paso, no ocupación** | Cesar: si flow se estima con `getLastStepVehicleNumber` (ocupación instantánea), `flow_total` se subestima ⇒ engine no rutea a max_pressure bajo pico ⇒ "adaptive" sería Webster-dinámico y se pierde el punto de la tesis. | `state_reader.StateTracker.observe()/commit_window()` cuenta IDs únicos en ventana 30s ⇒ `flow_vph = N × 3600/30`. | F4 test d5: aggregator con flujos am_peak produce `flow_total > 1500`. `run_e2e` exit code 2 con STOP-CONDITION si bajo am/pm peak engine nunca ruteó a max_pressure. |

## 4. Correcciones de método (todas aterrizadas)

| Corrección | Implementación | Verificación |
|------------|----------------|--------------|
| **1 — Fuentes canónicas por dirección y cola** | `coverage_check.py` lee `edgedata.xml` (speed/edge) + `lanearea.xml` (jamLengthInMeters), no `--summary-output` (que es agregado a red). Sin TraCI live. | F2 test b2. |
| **2 — Parquet edgeData/laneArea no verificado** | Hallazgo material: SUMO 1.26 arrow writer NO soporta append multi-interval. Adoptado XML fallback para outputs con `freq < end`. F3 `dataset.generate.py` parsea XML, persiste Parquet final (single-write OK). | F1 test a7: summary/tripinfo Parquet OK; edgeData/laneArea XML producen `<interval>` válidos. |
| **3 — Expansión a 6 sub-fases SUMO** | `tllogic_applier.expand_timings_to_sumo_phases` mapea cada `PhaseTiming(green,yellow,all_red)` a 3 Phases SUMO (G→y→r). 2 timings motor ⇒ 6 fases SUMO con clearance preservado. | F4 test d4: durations `[g,y,r,g,y,r]` con linkstates correctos. |
| **4 — Linkstates derivados via sumolib** | `build_network.py` invoca `netconvert` para geometría base, luego `sumolib.net.readNet` introspecciona el TLS, determina dirección de cada link controlado y asigna chars `G/g/r/y` por sub-fase. Persiste `linkstates.json`. | F1 test a6: corrida baseline 120s confirma que vehículos NS y EW completan ruta (linkstates son operacionalmente correctos, no solo sintácticamente). |

## 5. Tests verdes finales

```
$ cd simulation && .venv/bin/pytest tests/ -v
======================== 27 passed in 4.98s ========================

CT-07.8.a (F1, 8 tests): network loads + behavior baseline + Parquet outputs
CT-07.8.b (F2, 5 tests): cada patrón corre + cobertura jam verificada
CT-07.8.c (F4, 5 tests): adaptador TraCI e2e con engine mockeado
F3 (5 tests):            partitions sin overlap + dataset schema + ranges
F5 (4 tests):            Webster offline MTC bounds + KPIs finitos
```

## 6. Decisiones lockeadas (no reabrir post-cierre)

- Pin `eclipse-sumo==1.26.0`, `traci==1.26.0`. `libsumo` opción C
  (no se usa).
- Topología genérica (sin OSM). Larco × Schell solo referencia
  documentada.
- tlLogic Option A: 2 fases NS+EW lefts permissive. Linkstates
  derivados.
- Motor HTTP externo, puerto 8001. Cero código en
  `core_management_api/`.
- Transporte SUMO: TraCI cross-process.
- Dataset: Parquet single-write final, XML fallback intermedio para
  multi-interval outputs.
- KPIs comparativos: adaptive (engine) vs fixed-Webster (offline).

## 7. Deudas heredadas (fuera de scope MVP1)

| Item | Tipo | Origen | Tracking |
|------|------|--------|----------|
| Δt_in cierre formal | Provisional 60s — pendiente cierre TTH-11 CT-11.8 | F3 schema | TTH-11 |
| Turns en demanda | Routes straight-only por simplicidad | F2 | F41 / refinamiento posterior |
| libsumo in-process | Toolchain alternativo si TraCI no alcanza throughput | F0 opción C | Trabajos futuros |
| CT-10.11 integración R2 | Adaptador con persistencia + estado vigente | core (TTH-10) | R2 |

**Adenda post-cierre (smoke vivo).** El contrato del motor transcrito en F4
(`engine_recommend_contract.md`) estaba **incompleto**: el motor valida el
`intersection_id` contra la tabla `graph_nodes` (fail-fast **422
`unknown_intersection`**, `DHU-021 V1`) y depende de **`invoke seed`** para poblarla
(`invoke up` solo migra; tabla vacía → 422 con *cualquier* ID). F4 lo capturó con un
mock que devolvía un `intersection_id` inventado (`miraflores_4way`), lo que ocultó la
validación hasta el primer smoke con motor real. Corregido en `fix(tth-07)`:
`INTERSECTION_ID = "larco_schell"` (nodo del seed), contrato + README + mock alineados,
y el README documenta `invoke seed` como paso obligatorio del e2e (CT-07.7). El smoke
vivo (sin 422, `max_pressure` bajo am_peak, `setProgramLogic > 0`) lo corre Cesar.

## 8. Cross-refs

- Plan ejecutado: `~/.claude/plans/auditor-a-read-only-de-arranque-virtual-spark.md`
  (último estado post-catches A/B/C + correcciones 1-4).
- Handoffs por fase: `documentation/handoffs/tth-07/tth-07-faseN-handoff.md`.
- Contrato motor: `documentation/contracts/engine_recommend_contract.md`.
- README reproducible (CT-07.7): `simulation/README.md`.

## 9. Cierre

Sprint TTH-07 cierra con 8/8 CTs cubiertos sobre `feature/tth-07`. La
rama tiene 7 commits desde `master@85d56bb4`. Listo para el PR único
que abre Cesar.

**No push, no merge desde el agente** ([CLAUDE.md](../../CLAUDE.md)
§"Flujo de trabajo"). El cuerpo del PR usa este handoff como `--body-file`.
