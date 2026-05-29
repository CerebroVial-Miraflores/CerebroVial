# Contrato del módulo de visión — TTH-08

**Estado al cierre de TTH-08:** Parcial — F8 (validación de precisión / CT-08.9) diferida; C7.6 (pin CPU de torch) reabierta como F9.z.
**Última actualización:** 2026-05-29 (Fase 9 del sprint TTH-08).
**Autoridad:** DHU-024 §5 (`documentation/lean-inception/4-decisiones/DECISIONS_HU.md` líneas 2491–2503) compromete este documento. El addendum F9 al pie de DHU-024 refleja el estado real al cierre.

Este es el documento contractual de referencia para integradores del módulo de visión: shape de los endpoints HTTP, shape de la persistencia (`vision_aggregates`), shape del payload SSE, semántica de cada campo, alcance honesto de lo validado, deudas heredadas y referencia explícita a F41 como integración futura.

---

## 1. Endpoint `GET /vision/state/{intersection_id}` (CT-08.6)

Endpoint canónico de consulta sincrónica del estado de una intersección. Implementado en [edge_device/src/vision/presentation/api/routes/state.py](../edge_device/src/vision/presentation/api/routes/state.py). El campo `camera_id` del dominio se serializa como `intersection_id` para alinear con la nomenclatura del consumidor (DHU-024 §5; ver Fase 1 §6.5).

### 1.1 Respuesta 200 (estado vigente)

```json
{
  "intersection_id": "cam_javier_prado_01",
  "timestamp": "2026-05-27T22:48:30+00:00",
  "directions": [
    {
      "direction": "north_approach",
      "count": 24,
      "queue": null,
      "flow": 2880.0,
      "density": 48.0
    }
  ]
}
```

| Campo | Tipo | Semántica |
|---|---|---|
| `intersection_id` | string | Identificador de cámara/intersección. Coincide con `camera_id` del dominio. |
| `timestamp` | ISO-8601 tz-aware UTC | `window_end` máximo entre las zonas activas. Marca cuán reciente es el estado servido. |
| `directions` | array | Una entrada por zona configurada. Vacío durante el warm-up de la primera ventana. |
| `directions[].direction` | string | `zone_id` (string libre del YAML; en MVP1, p. ej. `"north_approach"`). |
| `directions[].count` | int ≥ 0 | `unique_vehicles` de la zona en la última ventana cerrada. |
| `directions[].queue` | int / null | **NULL en MVP1.** F41 lo poblará cuando se implemente cálculo de cola (ver §6.1). |
| `directions[].flow` | float ≥ 0 | `flow_vehicles_per_hour`. Unidad: vehículos/hora (HCM / Webster, alineado con `core_management_api/src/control/`). |
| `directions[].density` | float / null | `density_vehicles_per_km`. NULL si la zona no tiene `segment_length_meters` configurado (calibración de campo opcional). |

### 1.2 Respuestas de error

| Código | Cuándo | Significado |
|---|---|---|
| `404 Not Found` | `intersection_id` no está registrada en el `CameraManager`. | Cámara desconocida; semánticamente distinto de "módulo caído". |
| `503 Service Unavailable` | Cámara registrada pero el módulo no procesa frames (no hay `TrafficData` publicado y la cámara no está running). | CT-08.10 mandata 5xx en este caso. |
| `200 OK` con `directions: []` | Cámara running pero la primera ventana de agregación aún no cerró. | Warm-up esperado; no es error. |

### 1.3 Alcance NO cubierto en este endpoint

- **Estado discreto OK / Degradado / Fuera de servicio** vive en `GET /vision/health` (§2), no acá.
- **Eventos push** en tiempo real viven en el canal SSE (§3), no acá.

---

## 2. Endpoint `GET /vision/health` (CT-08.10 + CT-08.11f)

Endpoint **separado** de `/vision/state`. Reporta el estado discreto del módulo como payload estructurado, consumible por TTH-04 (fallback en cascada) conforme a CT-04.1. Implementado en [edge_device/src/vision/presentation/api/routes/health.py](../edge_device/src/vision/presentation/api/routes/health.py).

### 2.1 Respuesta

```json
{
  "status": "OK",
  "checked_at": "2026-05-27T22:48:30+00:00",
  "cameras": {
    "cam_javier_prado_01": {
      "running": true,
      "last_frame_age_seconds": 0.4,
      "aggregation_errors": 0,
      "data_dropped": 0
    }
  }
}
```

| Campo | Tipo | Semántica |
|---|---|---|
| `status` | enum | `"OK"` / `"Degradado"` / `"Fuera de servicio"`. |
| `checked_at` | ISO-8601 tz-aware UTC | Momento de la consulta. |
| `cameras` | dict | Telemetría por cámara registrada. |
| `cameras[id].running` | bool | Pipeline de la cámara está activo. |
| `cameras[id].last_frame_age_seconds` | float / null | Edad en segundos del último frame recibido; NULL si nunca se recibió frame. |
| `cameras[id].aggregation_errors` | int | Saves fallidos del aggregator (Fase 5b §11.1). Suma monotónica. |
| `cameras[id].data_dropped` | int | Eventos dropeados por output queue llena (Fase 5b §11.2). Suma monotónica. |

### 2.2 Reglas de estado (worst-of-fleet)

- Sin cámaras registradas → **Fuera de servicio** (HTTP 503).
- Ninguna cámara sana (running ∧ `last_frame_age_seconds < 5.0`) → **Fuera de servicio** (503).
- Hay cámaras sanas pero fleet parcialmente sano, o cualquier cámara con `aggregation_errors > 0` o `data_dropped > 0` → **Degradado** (200).
- Todas sanas, contadores en 0 → **OK** (200).

Umbral de frescura `LAST_FRAME_FRESHNESS_SECONDS = 5.0` (constante de implementación, no fijada por diseño).

---

## 3. Payload SSE — broadcaster `traffic_update`

Canal SSE de push del módulo. Único tipo de evento en MVP1: `traffic_update`. Política de suscriptor nuevo: recibe inmediatamente el último `TrafficData` conocido por zona suscrita (cache de último estado). Ver Fase 1 §6.2–§6.4.

### 3.1 Shape

```json
{
  "schema_version": "1.0",
  "event_type": "traffic_update",
  "server_timestamp": "2026-05-27T22:48:30Z",
  "camera": {
    "id": "cam_javier_prado_01",
    "street_monitored": null
  },
  "zone": {
    "id": "north_approach"
  },
  "window": {
    "start": "2026-05-27T22:48:00Z",
    "end": "2026-05-27T22:48:30Z",
    "duration_seconds": 30.0
  },
  "metrics": {
    "unique_vehicles": 24,
    "vehicles_by_type": {"car": 18, "bus": 2, "truck": 1, "motorcycle": 3},
    "mean_speed_kmh": 32.5,
    "flow_vehicles_per_hour": 2880.0,
    "mean_occupancy": 0.47,
    "density_vehicles_per_km": 48.0
  }
}
```

### 3.2 Notas honestas sobre el shape

- **`camera.street_monitored: null` en MVP1.** Auditoría F6 (cierre Fase 6) confirmó cero consumidores frontend del campo: ni `CameraDetailView`, ni `DashboardView`, ni `predictionService`, ni `TrafficHistoryWidget` lo leen. La premisa original de F1 §6.2 ("el frontend lo necesita para HU-02") no se materializó — HU-02 se cablea a SUMO (D-007), no a vision. El broadcaster emite `null` y los consumidores lo ignoran. Si en F41 aparece un consumidor del nombre humano, las dos rutas (CameraMetadataProvider inyectable, registry frontend) siguen abiertas sin tocar el dominio.
- **`mean_speed_kmh`** y **`density_vehicles_per_km`** son `null` cuando faltan los datos de calibración requeridos (ver §4).
- **`schema_version: "1.0"`** explícito desde el principio para permitir evolución sin romper consumidores.
- **`server_timestamp` vs `window.end`**: el primero es cuándo el broadcaster emite (útil para latencia RNF-PERF-01); el segundo es el fin de la ventana de datos.

---

## 4. Tabla `vision_aggregates` — persistencia canónica

Única tabla de persistencia del módulo (DHU-024 §2). Materializa Fase 1 §6.5. Migración Alembic activa: `5b4beac1055d_vision_aggregates_and_drop_legacy_vision.py`. Mapping ORM real en [edge_device/src/vision/infrastructure/persistence/postgres_repository.py](../edge_device/src/vision/infrastructure/persistence/postgres_repository.py) (`_to_row()`).

```sql
CREATE TABLE vision_aggregates (
    camera_id TEXT NOT NULL,
    zone_id   TEXT NOT NULL,

    window_start            TIMESTAMPTZ NOT NULL,
    window_end              TIMESTAMPTZ NOT NULL,
    window_duration_seconds DOUBLE PRECISION NOT NULL,

    unique_vehicles  INTEGER NOT NULL CHECK (unique_vehicles  >= 0),
    car_count        INTEGER NOT NULL DEFAULT 0 CHECK (car_count        >= 0),
    bus_count        INTEGER NOT NULL DEFAULT 0 CHECK (bus_count        >= 0),
    truck_count      INTEGER NOT NULL DEFAULT 0 CHECK (truck_count      >= 0),
    motorcycle_count INTEGER NOT NULL DEFAULT 0 CHECK (motorcycle_count >= 0),

    mean_speed_kmh         DOUBLE PRECISION,   -- NULL si la cámara no tiene calibración pixel→metro
    flow_vehicles_per_hour DOUBLE PRECISION NOT NULL CHECK (flow_vehicles_per_hour >= 0),

    mean_occupancy           DOUBLE PRECISION NOT NULL CHECK (mean_occupancy BETWEEN 0 AND 1),
    density_vehicles_per_km  DOUBLE PRECISION,  -- NULL si zona sin segment_length_meters

    queue INTEGER,                              -- NULL en MVP1; F41 lo poblará

    PRIMARY KEY (camera_id, zone_id, window_start)
);
SELECT create_hypertable('vision_aggregates', 'window_start');
```

### 4.1 Decisiones clave del schema

- **Un row por zona** (no array de `directions[]`). El shape `{intersection_id, directions: [...]}` de DHU-024 §5 es representación API; la BD almacena planas y agrupa al servir.
- **Desempaquetado de `vehicles_by_type`** a cuatro columnas en la tabla. En el dominio (`TrafficData`) es dict; en la tabla son columnas (`WHERE bus_count > 5` directo). `_to_row()` desempaqueta con `.get(default=0)`, por lo que tipos ausentes del dict aterrizan como 0.
- **PK compuesta `(camera_id, zone_id, window_start)`** garantiza idempotencia. `INSERT … ON CONFLICT DO NOTHING` absorbe reintentos sin duplicar.
- **`mean_occupancy` y `density_vehicles_per_km` como columnas separadas**, no un único campo `density`. Reinterpretación honesta de DHU-024 §5: cuando se redactó no se distinguían las dos métricas; Fase 1 §6.9 las separó.
- **`queue INTEGER` nullable** en MVP1. F41 lo poblará cuando se implemente cálculo de cola.
- **`camera_id` ≡ `intersection_id` del shape API.** Dos nomenclaturas, un mismo identificador (ver Fase 1 §6.5).
- **`TIMESTAMPTZ`** deliberado para `window_start`/`window_end`/`server_timestamp`, alineado con `datetime` tz-aware UTC del dominio.
- **`CheckConstraint('mean_occupancy BETWEEN 0 AND 1')` activo en BD**, validado por el e2e de Fase 7 contra Postgres real (bypaseando el validador `__post_init__` del dominio).

### 4.2 Campos eliminados del schema legacy `CameraTrafficData`

`CameraTrafficData` (en `shared/cerebrovial_shared/schemas/camera.py`) **queda huérfano**: sin consumidor runtime (solo su definición + un comentario en `core_management_api/scripts/generate_camera_data.py`). Su borrado requiere coordinación de `shared/` y queda **fuera de TTH-08** (ver Fase 1 §5.8).

`total_vehicles`, `vehicle_types: dict` redundante, `street_monitored` por agregado, campos de debug (`raw_detection_count`, `display_queue`, `interpolate`), `occupancy_rate` con `× 100` previo: todos eliminados conforme Fase 1 §5.6.

---

## 5. Alcance honesto de validación al cierre de TTH-08 (D-005)

Esta sección rastrea **exactamente** qué CTs están validados y cuáles diferidos. Es lectura obligatoria antes de citar resultados en la tesis o ante el jurado.

### 5.1 CTs validados al cierre de Fase 7

| CT | Qué valida | Dónde |
|---|---|---|
| **CT-08.1** | Detección YOLO produce `list[DetectedVehicle]` | `tests/vision/unit/test_yolo_detector.py` (Fase 4b) |
| **CT-08.2** | Conteo por zona / ROI con `mean_occupancy` (DHU-025) | `tests/vision/unit/test_zone_counter_basic.py`, `test_zones.py` (Fase 4a) |
| **CT-08.3** | Métricas direccionales (`flow`, `density`, `mean_speed_kmh`) | `tests/vision/unit/test_compute_traffic_data.py` (11 tests, Fase 5b) |
| **CT-08.4** | Input modes (file/youtube/ip_cam/auto dispatcher) | Tests de fuentes y dispatcher (Fase 4b) |
| **CT-08.5** | Persistencia a `vision_aggregates` — mapping de columnas, NULLs, idempotencia ON CONFLICT | `tests/vision/unit/test_postgres_repository.py` (Fase 4c) |
| **CT-08.6** | Endpoint `GET /vision/state/{intersection_id}` con shape §1.1 + branch 5xx | `tests/vision/unit/test_state_endpoint.py` (Fase 6e) |
| **CT-08.7** | Stream procesado (`/vision/streaming/...` y derivados) | Suite de streaming (Fase 6) |
| **CT-08.8** | Componente demostrable end-to-end (smoke 4c contra compose vivo) | Smoke manual + integration tests |
| **CT-08.10** | Health check con estados OK / Degradado / Fuera de servicio | `tests/vision/integration/test_health.py` (Fase 6f) |
| **CT-08.11(a–d, f)** | Detección + asignación direccional + derivación + endpoint + health | Cubiertos en Fases 4–6 |
| **CT-08.11(e)** | Integración persistencia repo↔modelo↔Postgres vivo (testcontainers) | `tests/vision/integration/test_persistence_e2e.py` (Fase 7). **Alcance acotado:** valida repo↔BD-real, **NO** migración↔modelo, **NO** pipeline-de-video end-to-end. Ver handoff F7 §4.1–§4.2. |

### 5.2 CT diferido — CT-08.9

**No validado al cierre de TTH-08.** Diferido por decisión del usuario (handoff F7 §6.1 línea 157). Requiere:
- Dataset etiquetado propio ≥200 frames.
- Cómputo honesto de precisión / recall / mAP del detector YOLO con `conf_threshold = 0.3`.
- Trabajo de **datos**, no de código (Roboflow / CVAT / labelImg).

Estado del **número 88.2%** (D-005 — `documentation/lean-inception/4-decisiones/DECISIONS.md` líneas 70–79): es el valor aspiracional del documento de tesis original. **No tiene sustento reproducible al día de hoy** — sin dataset etiquetado ni medición cuantitativa, no puede afirmarse como validado. Se sustituye por el mAP real medido cuando corra F8. Si la realidad medida es peor que 88.2%, **se reporta la realidad** (integridad académica, no marketing).

### 5.3 Estado consolidado

| Aspecto | Estado |
|---|---|
| Entrega operativa (código + tests + endpoints + persistencia) | **Validada** por Fases 0–7 |
| Contrato documental (este archivo, DHU-024 §5) | **Entregado en F9** |
| Validación cuantitativa de detección (CT-08.9, 88.2%) | **Diferida** — pendiente F8 |
| **Estado TTH-08 al cierre** | **Parcial — F8 diferida** (NO Done) |

---

## 6. Integración futura con módulo predictivo — F41 (Trabajos Futuros)

**TTH-08 mantiene CERO acoplamiento con `ia_prediction_service/`.** Ningún import, ningún endpoint compartido, ninguna dependencia transitiva. La integración vision→predictivo se documenta como **F41 (Trabajos Futuros)** y queda explícitamente fuera del scope de TTH-08 y de MVP1. Esta decisión preserva D-007 — el módulo de visión es componente **demostrable**, no participa del loop cuantitativo de MVP1 (alimentado por SUMO).

### 6.1 Ámbito reservado a F41

- **Cálculo de `queue` (longitud de cola).** La columna `queue INTEGER` en `vision_aggregates` y el campo `directions[].queue` del endpoint están reservados; en MVP1 son siempre `NULL`. F41 los poblará cuando se implemente el cálculo.
- **Calibración direccional por zona** (mapear zonas YOLO a fases del controlador adaptativo con métricas direccionales reales). En MVP1 la calibración direccional **no** es métrica del detector — el mAP de F8 será a nivel detector global, no por dirección. La calibración direccional fina queda como trabajo de F41.
- **Tipos adicionales en `vehicles_by_type`** (p. ej. `"pedestrian"`, `"bicycle"`) si la integración lo requiere.
- **Enriquecimiento del payload SSE con `camera.street_monitored`** si aparece un consumidor frontend que lo necesite (ver §3.2).

### 6.2 Trazabilidad de F41

- `documentation/lean-inception/1-contexto/EVOLUCION_TESIS.md` §8 (Trabajos Futuros), fila F41.
- `documentation/lean-inception/1-contexto/LEAN_INCEPTION_CEREBROVIAL.md` sección "Trabajos Futuros", fila F41.
- `DECISIONS_HU.md` DHU-024 §5 (compromiso de no acoplar en TTH-08) y addendum F9 (reafirmación al cierre).

---

## 7. Deudas heredadas al cierre

| Deuda | Estado al cierre F9 | Cierre planeado |
|---|---|---|
| **C7.6 — pin CPU de `torch` en `edge_device/requirements.txt`** | **NO aplicada en TTH-08.** DHU-024 §7 declaró cierre dentro del refactor pero `edge_device/requirements.txt:5` sigue siendo `torch` sin `--index-url https://download.pytorch.org/whl/cpu`. F9 lo reconoce honestamente con addendum a DHU-024 §7. | **Reabierta como F9.z** — sub-fase de infra separable post-F9, fuera del scope F9. |
| **Paridad migración Alembic ↔ modelo SQLAlchemy** | El e2e de F7 valida repo↔modelo↔BD pero **NO** migración↔modelo. Si la migración `5b4beac1055d_vision_aggregates_and_drop_legacy_vision.py` y `shared/cerebrovial_shared/database/models.py:86-115` divergen, el test pasa y producción rompe. | **Nominada como C9.7** en `TODO.md`. Forma de cierre: test chico con `alembic.autogenerate.api.compare_metadata`. |
| **Wiring de `edge_device/tests` a CI** | El workflow `.github/workflows/ci.yml` corre solo `core_management_api/tests/`. Las 124 tests de `edge_device/tests/` (120 heredadas + 4 e2e F7) están fuera de CI desde TTH-03. | **Nominada como C9.8** en `TODO.md` (o TTH-03 retomado). Requiere job Docker, caché TimescaleDB, decisión sobre deps pesadas. |
| **Barrido de código huérfano** (`smart_detection.py` sin consumidor, tests legacy `multi_camera_manager`) | DHU-024 §3 declaró lógica muerta; el source vive aún sin consumidor runtime. F9 nomina pero **no** borra (es código productivo). | **Nominada como F9.y** en `TODO.md` — sub-fase de barrido separable. |
| **`CameraTrafficData` huérfana en `shared/`** | Sin consumidor runtime; borrado requiere coordinación de `shared/`. Fuera de TTH-08 (Fase 1 §5.8). | Fuera de TTH-08. |

Estas cinco deudas + **F8 (CT-08.9)** se agrupan como **"Backlog post-TTH-08 (infra/cleanup separable)"** en el handoff F9 §[Backlog post-TTH-08]. Esa sección consolidada hace descubrible el conjunto en vez de quedar como labels sueltos.

---

## 8. Cross-refs

- **Compromiso original**: `DECISIONS_HU.md` DHU-024 §5 (líneas 2491–2503) + addendum F9 al pie de DHU-024.
- **Diseño que materializa este contrato**: `documentation/docs/tth-08-fase1-diseno.md` §5 (TrafficData), §6.2 (SSE), §6.5 (vision_aggregates).
- **Implementación**: [edge_device/src/vision/](../edge_device/src/vision/) — capas `domain/`, `application/`, `infrastructure/`, `presentation/`.
- **Honestidad del 88.2%**: `DECISIONS.md` D-005 (líneas 70–79).
- **Alcance acotado de CT-08.11(e)**: `documentation/handoffs/tth-08/tth-08-fase7-handoff.md` §4.1–§4.2 (qué SÍ valida, qué NO valida) y §2 fila (e).
- **Diferimiento de F8**: `documentation/handoffs/tth-08/tth-08-fase7-handoff.md` §6.1, `documentation/handoffs/tth-08/tth-08-fase9-handoff.md` §[diferimientos], addendum F9 de DHU-024.
- **F41 como Trabajos Futuros**: `EVOLUCION_TESIS.md` §8 (línea 161), `LEAN_INCEPTION_CEREBROVIAL.md` sección "Trabajos Futuros" (línea 288).
- **Deudas nominadas F9.x / F9.y / F9.z**: `documentation/docs/TODO.md` (C7.6 reabierta, C9.7 paridad, C9.8 CI, F9.y barrido).
- **Cierre del sprint**: `documentation/handoffs/tth-08/tth-08-fase9-handoff.md`.
