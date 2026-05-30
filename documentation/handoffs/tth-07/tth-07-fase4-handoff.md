# TTH-07 — Cierre de Fase 4 (contrato motor + adaptador TraCI ↔ HTTP + e2e)

**Rama**: `feature/tth-07`.
**Fecha de cierre**: 2026-05-29.
**Estado al cierre**: F4 verde, 23/23 tests pasando (8 F1 + 5 F2 + 5 F3 + 5 F4).
Catches A, C, y corrección 3 implementados + verificados por tests con
engine mockeado.

---

## 1. Lo que F4 entregó

### Contrato canónico del motor (abre F4)

`documentation/contracts/engine_recommend_contract.md`:
- POST http://localhost:8001/control/recommend (Catch 1).
- Request `IntersectionState` + Response `ControlRecommendation` shape verbatim del Pydantic.
- Routing engine documentado: flow_total < 1500 → webster; ≥1500 → max_pressure.
- MTC constants lockeadas.
- `has_pedestrian` semantics: etiqueta sin efecto con defaults.
- F4 pasa `lost_time=10s` explícito (no usa default 8.0).

### Adaptador TraCI

```
simulation/src/cerebrovial_simulation/traci_adapter/
├── __init__.py
├── state_reader.py        TraCI live → DirectionState (Catch C: flow = TASA DE PASO)
├── phase_aggregator.py    4 dirs → 2 PhaseFlow (Option A: NS, EW)
├── engine_client.py       POST a ENGINE_URL con timeout/retry
├── tllogic_applier.py     Catch A: aplica en borde de CICLO (no sub-fase) + corrección 3 (expansión 6 sub-fases SUMO)
└── run_e2e.py             Orquestador sumo + TraCI + motor cada 30s

simulation/scripts/
└── run_e2e_with_engine.sh CLI reproducible (valida /control/health antes de arrancar)

simulation/tests/
└── test_traci_e2e.py      CT-07.8.c + d4/d5 unit (Catch A, corrección 3, Catch C)
```

## 2. Catch C — flow es tasa de paso, no ocupación

Implementación en `state_reader.StateTracker`:
- `observe(sim_time)`: en cada step, agrega `traci.edge.getLastStepVehicleIDs(N_in/S_in/E_in/W_in)` al set acumulado por dirección.
- `commit_window(sim_time)`: al cerrar ventana (30s sim), `flow_vph = len(seen_vehicles) × (3600 / window_s)`.

Verificación unit en `test_d5_phase_aggregator_flow_total_crosses_peak_threshold`:
con flujos N=2400 S=2400 E=600 W=600, `flow_total = 6000 > 1500` ⇒
engine debe rutear a max_pressure.

Smoke e2e con motor real (NO ejecutado en F4 — requiere `invoke up`):
si bajo am_peak `engine_modes` no contiene `"max_pressure"`, `run_e2e`
retorna exit code 2 con mensaje "Catch C STOP-CONDITION" — falla
explícita visible.

## 3. Catch A — aplicar en borde de CICLO, no de sub-fase

Implementación en `tllogic_applier.TllogicApplier.maybe_apply(sim_time)`:
- Solo aplica si `getPhase(tls_id) == 5` (índice de EW_r = última sub-fase) Y `getNextSwitch(tls_id) - sim_time ≤ 1` (este step cierra el ciclo).
- Con `currentPhaseIndex=0` (NS_g), el reemplazo es continuidad natural (cierre→apertura del ciclo siguiente).

Verificación en `test_d3_applications_only_at_cycle_boundary`:
- Corrida 180s sim con engine mockeado.
- Engine se invoca 6 veces (cada 30s).
- `setProgramLogic` se aplica solo en bordes de ciclo: **≤ 3 aplicaciones** en 180s (ciclos baseline ~60s), confirmado por assert.

Si Catch A no estuviera implementado correctamente (aplicación en cada
sub-fase), `n_applications` sería ≥ 6 — el test lo rechaza explícitamente.

## 4. Corrección 3 — Expansión a 6 sub-fases SUMO

`expand_timings_to_sumo_phases(timings, linkstates)` mapea:

```
PhaseTiming("NS", green=30, yellow=3, all_red=2)
→ [Phase(30, linkstates["NS_g"]),
   Phase(3,  linkstates["NS_y"]),
   Phase(2,  linkstates["NS_r"])]

PhaseTiming("EW", green=25, yellow=3, all_red=2)
→ [Phase(25, linkstates["EW_g"]),
   Phase(3,  linkstates["EW_y"]),
   Phase(2,  linkstates["EW_r"])]
```

2 timings motor ⇒ 6 sub-fases SUMO. Verificado en `test_d4_expand_timings_produces_6_phases`.

Esto preserva el clearance (yellow + all_red entre fases) que un naive
`Phase(duration=g+y+r, state=G_state)` rompería — corrupción silenciosa
del throughput de F5.

## 5. Tests verdes

```
$ .venv/bin/pytest tests/ -v
... 23 passed in 5.01s ...

tests/test_traci_e2e.py::test_d4_expand_timings_produces_6_phases PASSED
tests/test_traci_e2e.py::test_d5_phase_aggregator_flow_total_crosses_peak_threshold PASSED
tests/test_traci_e2e.py::test_d1_e2e_runs_and_applies_plan PASSED
tests/test_traci_e2e.py::test_d2_engine_invoked_every_30s PASSED
tests/test_traci_e2e.py::test_d3_applications_only_at_cycle_boundary PASSED
```

## 6. Smoke manual e2e (NO ejecutado en F4 — requiere motor)

```bash
# Terminal 1: motor
cd /Users/rasec/Tesis/CerebroVial
invoke up

# Terminal 2: e2e
cd simulation
bash scripts/run_e2e_with_engine.sh --pattern am_peak --seed 1 --end 600
```

El script:
1. Valida `curl /control/health` (falla rápido si motor no responde).
2. Setea SUMO_HOME al wheel.
3. Invoca `run_e2e.py` con args.
4. Reporta `n_engine_calls`, `n_applications`, `engine_modes`, errores.
5. Sale con código 2 si Catch C STOP-CONDITION (am_peak no ruteó a max_pressure).

## 7. Decisiones cerradas (no reabrir en F5+)

- HTTP externo locked para invocación del motor.
- TraCI cross-process para sensado + actuación (libsumo NO se usa).
- Ciclo de sensado = 30s sim; aplicación = borde de ciclo.
- `lost_time=10s` explícito en cada POST.
- `has_pedestrian=True` en ambas fases (etiqueta inactiva con MTC defaults).

## 8. Próximo paso — F5

F5 implementa Webster fijo offline (Catch B — fórmula directa, NO via
motor) + KPIs comparativos adaptive vs fixed.
