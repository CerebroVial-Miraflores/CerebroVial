# Data Model: CerebroVial — Control adaptativo de semáforos (MVP1)

**Date**: 2026-05-20 | **Feature**: `001-cerebrovial-mvp`

> **Adopción brownfield (DHU-021).** Este `data-model.md` **mapea** el modelo de datos ya
> documentado; no se regenera. Fuentes: `documentation/sdd/SDD_CEREBROVIAL.md` §4 (extensión
> del motor) y `documentation/docs/DATA_MODEL.md` (modelo heredado, canónico columna por
> columna). Para el *porqué* y qué llena cada tabla, ver `DATA_MODEL_AUDIT.md`.
>
> Stack: PostgreSQL 15 + TimescaleDB + PostGIS; ORM SQLAlchemy + GeoAlchemy2 en
> `shared/cerebrovial_shared/database/models.py`; migraciones Alembic.

## Visión general

El dominio se organiza en tres familias heredadas —**topología vial** (grafo PostGIS),
**feeds de Waze** (series temporales) y **visión computacional**— más **dos entidades
nuevas** que la operación del motor adaptativo requiere y que **aún no existen** en el
código (estado *as-designed*; Delta-10 / DHU-020 §E): `motor_decisions` y
`engine_active_state`.

```text
graph_nodes ──< graph_edges            (topología; columna vertebral espacial)
graph_nodes ──< cameras
graph_edges ──< waze_jams / waze_alerts
cameras     ──< vision_tracks / vision_flows / vision_aggregates
graph_nodes ──< motor_decisions ──1 engine_active_state   (NUEVAS, §4 SDD)
```

---

## 1. Modelo heredado (canónico en `DATA_MODEL.md`)

Resumen por dominio; el detalle columna por columna es canónico en `DATA_MODEL.md`.

### Topología vial (PostGIS)
- **`graph_nodes`** — intersección física. PK `node_id` (**String**, ej. `"larco_diagonal"`), `lat`/`lon`, `has_camera`, `geom` POINT 4326. Estática (seed).
- **`graph_edges`** — calle dirigida entre dos nodos (doble sentido = dos aristas). PK `edge_id`, FKs `source_node`/`target_node` → `graph_nodes`, `distance_m`, `lanes`, `geom` LINESTRING 4326.

### Feeds de Waze (series temporales; hypertables)
- **`waze_jams`** — snapshots de congestión. PK (`event_uuid`, `snapshot_timestamp`), FK `edge_id` (nullable), `congestion_level` 1-5 (**ground truth del GRU**), etc.
- **`waze_alerts`** — eventos puntuales (accidente, peligro, cierre…). PK (`alert_uuid`, `timestamp`).

### Visión computacional
- **`cameras`** — metadata espacial. PK `camera_id`, FK `node_id` (nullable), `heading`, `fov`, `geom`.
- **`vision_tracks`** / **`vision_flows`** — modeladas, **no llenadas** (integración futura del pipeline a BD).
- **`vision_aggregates`** — **a crear (E18)**; alineada con el CSV del pipeline, persistencia sin refactor de visión (regla de CLAUDE.md). Conteos por tipo, ocupación, flujo por ventana.

### Hypertables (TimescaleDB)
`waze_jams`, `waze_alerts`, `vision_tracks`, `vision_flows` y (a futuro) `vision_aggregates`, particionadas por su timestamp.

### Índices y tablas internas
Todos los `geom` con índice GIST (GeoAlchemy2). Las tablas internas de PostGIS
(`spatial_ref_sys`, `layer`, `topology`) se excluyen del autogenerate de Alembic vía
`include_object` en `env.py`.

---

## 2. Extensión para el motor adaptativo (SDD §4.2 — *as-designed*, aún no implementada)

Dos entidades nuevas, ancladas a `graph_nodes`. Mandato: DHU-020 §E (estado vigente) y
Delta-10 (historial de decisiones).

### 2.1 `motor_decisions` — historial de decisiones (append-only)

Registro inmutable: cada recomendación del motor se inserta y no se modifica (RNF-SEC-01).

| Columna | Tipo | Notas |
|---|---|---|
| `decision_id` | uuid PK | Identificador único de la decisión. |
| `node_id` | string FK → `graph_nodes` | La intersección. El motor emite hoy un `intersection_id` **opaco y sin restricción** (calculadora sin estado, no consulta la BD); la **capa de persistencia resuelve y valida** ese id contra `graph_nodes.node_id` al escribir (DHU-021 V1). |
| `decided_at` | datetime | Momento del cálculo. |
| `mode` | string | `webster` \| `max_pressure`. |
| `cycle_seconds` | float | Ciclo final, compuesto por la capa MTC. |
| `flow_total` | float | Suma de flujos del input; discriminante peak/off-peak. **No serializado por el endpoint**; se captura del cálculo interno al persistir (DHU-021 V2). |
| `y_load_factor` | float, nullable | Factor de carga Y de Webster; nulo en saturación severa (`webster_infeasible`). **No serializado**; capturado del cálculo interno (DHU-021 V2). |
| `next_phase` | string, nullable | Fase que MaxPressure entra primero; nulo en Webster. |
| `reasoning` | text | Razonamiento textual; sustrato de HU-06. |
| `phase_timings` | jsonb | Arreglo de `{phase_id, green, yellow, all_red}` por fase (no normalizado — DHU-021 #10). |
| `adjustments` | jsonb | Lista de descripciones de texto de los ajustes MTC (el motor las emite como `list[str]`); vacía si no hubo. |
| `inputs_snapshot` | jsonb, nullable | Snapshot del payload de fases que originó la decisión (`flow`, `saturation`, `queue`, `pedestrian`). **No devuelto por el endpoint**; capturado del payload al persistir, hace la decisión reproducible (DHU-021 V2). |

- **Índice:** `(node_id, decided_at DESC)` — sirve el historial de una intersección (HU-08) y su decisión más reciente.
- **Relacional, no hypertable** (DHU-021 #11): el volumen de la intersección piloto no justifica TimescaleDB; conversión diferida a productivización.
- **Contrato fuente:** `POST /control/recommend` (respuesta envuelta en `{data: …}`) expone `mode`, `cycle_seconds`, `phase_timings`, `next_phase`, `reasoning`, `adjustments`. La brecha entre lo que el endpoint emite y lo que la decisión persiste la cierra el componente de control (Delta-10).

### 2.2 `engine_active_state` — estado vigente (mutable, uno por intersección)

Exactamente una fila por intersección; materializa el "estado vigente" de DHU-020 §E.

| Columna | Tipo | Notas |
|---|---|---|
| `node_id` | string PK FK → `graph_nodes` | La intersección. Como PK, garantiza una única estrategia vigente por intersección. |
| `active_decision_id` | uuid FK → `motor_decisions` | Puntero a la decisión actualmente activada. |
| `activated_at` | datetime | Timestamp de activación (exigido por DHU-020 §E). |
| `activated_by` | string, nullable | Origen de la activación (operador o automático); soporta HU-05/HU-07 sin acoplar aún el modelo de usuarios. |

**Por qué entidad propia y no vista derivada** (DHU-021 #13): separa el evento de *cálculo*
(`motor_decisions.decided_at`) del de *activación* (`engine_active_state.activated_at`). El
motor puede calcular recomendaciones que no se activan; "última calculada" ≠ "vigente".

### 2.3 Frontera grafo↔intersección (diferida, SDD §4.3 / DHU-021 #12)

El adaptador cámara→approach→fase y la conversión nivel→flujo **no** se esquematizan: el
motor consume las fases por payload, no las persiste. Las decisiones se anclan a
`graph_nodes`; el interior de la intersección (approaches/fases) queda como extensión futura.

---

## Reglas de validación y consistencia

- `motor_decisions` es **append-only** (sin UPDATE/DELETE) — auditabilidad (RNF-SEC-01, §8.3).
- `engine_active_state.node_id` único por construcción (PK) — una estrategia vigente por intersección.
- `node_id` de ambas entidades debe existir en `graph_nodes` (FK); el `intersection_id` del motor se resuelve a `node_id` en el write-path antes de insertar.
- `mode ∈ {webster, max_pressure}`; `y_load_factor` nulo ⇔ caso `webster_infeasible`.
- Migraciones **solo** con Alembic (nunca `Base.metadata.create_all()`); las dos tablas nuevas entran como una migración relacional con FK de una sola columna.
