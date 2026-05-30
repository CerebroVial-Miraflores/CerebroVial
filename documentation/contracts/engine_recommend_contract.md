# Contrato canónico — `POST /control/recommend`

> Contrato del motor adaptativo (TTH-10) consumido por TTH-07 F4 vía
> HTTP externo. Transcripción del shape Pydantic vivo en
> [core_management_api/src/control/presentation/api/schemas.py:13-56](../../core_management_api/src/control/presentation/api/schemas.py#L13-L56)
> y la semántica del engine en
> [adaptive_engine.py](../../core_management_api/src/control/application/adaptive_engine.py)
> + [mtc_constraints.py](../../core_management_api/src/control/application/mtc_constraints.py)
> + [webster.py](../../core_management_api/src/control/application/webster.py).
>
> Este contrato **abre F4**: el adaptador TraCI se codea contra esta
> interfaz estable. Cualquier cambio al shape del motor en el core
> requiere actualizar este archivo Y los consumidores. Trabajos
> futuros: si el contrato evoluciona, abrir DHU con migración explícita.

## 1. Endpoint

```
POST http://localhost:8001/control/recommend
Content-Type: application/json
```

**Puerto 8001 lockeado** en SP3/TB1
([docker-compose.yml:37](../../docker-compose.yml#L37) `"8001:8001"`,
[.env.example:41](../../.env.example#L41) `VITE_API_BASE_URL=http://localhost:8001`).
El router del motor monta con `prefix="/control"` sin `/api/`
([routes.py:55](../../core_management_api/src/control/presentation/api/routes.py#L55)).

`ENGINE_URL=http://localhost:8001/control/recommend` es el default
documentado en `simulation/.env.example`, env-configurable.

## 2. Request — `IntersectionState`

```json
{
  "intersection_id": "larco_schell",
  "timestamp": "2026-05-29T10:00:00Z",
  "phases": [
    {
      "phase_id": "NS",
      "flow": 4800.0,
      "saturation_flow": 10800.0,
      "queue": 12,
      "has_pedestrian": true
    },
    {
      "phase_id": "EW",
      "flow": 1200.0,
      "saturation_flow": 7200.0,
      "queue": 4,
      "has_pedestrian": true
    }
  ],
  "lost_time": 10.0,
  "predicted_demand": null
}
```

Campos del top-level:

| Campo | Tipo | Notas |
|-------|------|-------|
| `intersection_id` | string | Identificador único; ≥ 1 char |
| `timestamp` | string | ISO-8601 |
| `phases` | `list[PhaseFlow]` | ≥1 fase. Para Option A: 2 fases (`NS`, `EW`) |
| `lost_time` | float | Segundos perdidos por ciclo (L de Webster). **F4 pasa L=10s explícito** (2 fases × 5s inter-green = yellow+all_red). NO usar el default 8.0 del schema (asume 2 fases con interrupts menores). |
| `predicted_demand` | `PredictedDemand?` | Optional. F4 lo deja `null` en MVP1 (TTH-09 lo cabla en operación). |

`PhaseFlow`:

| Campo | Tipo | Notas |
|-------|------|-------|
| `phase_id` | string | Para Option A: `"NS"` o `"EW"` |
| `flow` | float | **Tasa de paso en veh/h** — vehículos cruzando el aproche por hora. **CRÍTICO (Catch C)**: NO es ocupación instantánea. F4 lo deriva contando arribos en una ventana de sensado de 30s y multiplicando por 120. Underestimación rompe el routing del engine. |
| `saturation_flow` | float | Capacidad veh/h en saturación. Para Option A: NS=10800 (6 lanes × 1800), EW=7200 (4 × 1800). |
| `queue` | int | Cola actual en vehículos. F4 lo deriva de `jamLengthInMeters` del detector laneArea agregado a la dirección, dividido por 7.5 m/veh, redondeado a entero. |
| `has_pedestrian` | bool | Para Option A: `True` en ambas fases. **Semántica del engine**: con MTC defaults (`min_pedestrian=7 == min_green=7`) el flag es **etiqueta sin efecto comportamental**. Solo cambiaría comportamiento si MTCConstants se parametriza con `min_pedestrian > min_green`. |

## 3. Response — `ControlRecommendation`

```json
{
  "data": {
    "intersection_id": "larco_schell",
    "mode": "max_pressure",
    "cycle_seconds": 84.5,
    "phase_timings": [
      {"phase_id": "NS", "green": 42.0, "yellow": 3.0, "all_red": 2.0},
      {"phase_id": "EW", "green": 32.5, "yellow": 3.0, "all_red": 2.0}
    ],
    "next_phase": "NS",
    "reasoning": "Peak (Σ flow = 6000 veh/h ≥ 1500). Max Pressure on Webster base cycle = 84.5s; next phase = NS; MTC constraints applied.",
    "adjustments": [],
    "decision_id": "..."
  }
}
```

Campos:

| Campo | Tipo | Notas |
|-------|------|-------|
| `intersection_id` | string | Eco del request |
| `mode` | `"webster"` \| `"max_pressure"` | **Routing** (ver §4) |
| `cycle_seconds` | float | Tras MTC constraints (clamp a [min_cycle, 120]) |
| `phase_timings` | `list[PhaseTiming]` | Una entrada por phase_id del request. **Expandir a fases SUMO**: cada `PhaseTiming` se expande en 3 sub-fases SUMO (green con linkstate G/g, yellow con y, all_red con r) — F4 corrección 3. Para 2 timings motor ⇒ 6 sub-fases SUMO. |
| `next_phase` | `str?` | Para max_pressure: cuál fase activar next. Para webster: `null`. F4 lo usa solo informativamente; la actuación va por `phase_timings`. |
| `reasoning` | string | Texto human-readable. **F4 verifica via Catch C** que bajo am_peak `reasoning` contenga `"Peak"`. |
| `adjustments` | `list[str]` | MTC adjustments aplicados (clamp floors/ceilings, cycle_capped) |
| `decision_id` | `str?` | UUID del registro persistido en `motor_decisions`. F4 no lo usa. |

`PhaseTiming`:

| Campo | Tipo |
|-------|------|
| `phase_id` | string |
| `green` | float ≥ 0 |
| `yellow` | float ≥ 0 |
| `all_red` | float ≥ 0 |

## 4. Routing del engine

[adaptive_engine.py:69-71](../../core_management_api/src/control/application/adaptive_engine.py#L69):

```python
self.PEAK_THRESHOLD = peak_threshold  # 1500 veh/h default
if flow_total < self.PEAK_THRESHOLD:
    return self._webster_branch(...)
return self._max_pressure_branch(...)
```

donde `flow_total = sum(p.flow for p in state.phases)`.

**Implicancia para F4** (Catch C): con am_peak (flujo Σ = 4800 NS + 1200 EW
= 6000), `flow_total` debe estimarse correctamente como tasa de paso. Si
F4 underestima `flow` (e.g. usando `getLastStepVehicleNumber` =
ocupación instantánea ≈ 10-15), `flow_total ≈ 30` ⇒ engine rutea
`mode="webster"` durante el pico — **incorrecto**. Smoke e2e debe
confirmar `mode="max_pressure"` bajo am_peak.

## 5. MTC constraints lockeadas

[mtc_constraints.py:23-29](../../core_management_api/src/control/application/mtc_constraints.py#L23-L29):

```python
@dataclass(frozen=True)
class MTCConstants:
    min_green: int = 7
    max_green: int = 60
    min_yellow: int = 3
    all_red: int = 2
    min_pedestrian: int = 7
```

`max_cycle = 120` (parámetro del `MTCRestrictionApplier`).

Implicancia: cualquier `phase_timings` del engine cumple `7 ≤ green ≤
60`, `yellow = 3`, `all_red = 2`. F4 puede asumir estos invariantes.

## 6. Estado actual del motor (R1 / no-TraCI)

CT-10.11 (integración TraCI directa) está **diferido a R2** según
[ESTADO_Y_PROXIMOS_PASOS.md:43](../../documentation/ESTADO_Y_PROXIMOS_PASOS.md#L43).
F4 implementa la ruta **HTTP externa** locked en el plan: el adaptador
TraCI vive en `simulation/`, llama al motor via HTTP, y aplica las
respuestas al semáforo SUMO. **Cero código en `core_management_api/`**.

## 7. Versionado del contrato

Este contrato refleja el motor al cierre de TTH-08 F9 (commit
`85d56bb4` master). Si el motor evoluciona en HU futuras (HU-15
parámetros configurables, HU-07 transiciones notificadas, etc.), este
contrato debe actualizarse en bloque junto al consumidor (`simulation/`).

Cross-refs:
- [TAREAS_TECNICAS_HABILITADORAS.md TTH-07 CT-07.5](../lean-inception/2-backlog/TAREAS_TECNICAS_HABILITADORAS.md#L383)
  declara el requisito de integración bidireccional.
- [vision_contract.md](vision_contract.md) precedente de "contrato
  canónico" en `documentation/contracts/`.
