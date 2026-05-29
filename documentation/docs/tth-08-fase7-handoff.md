# TTH-08 — Cierre de Fase 7 + handoff a Fases 8-9

**Rama**: `feature/tth-08-fase7-tests` (desde `master@39793c18` = merge PR #32,
Fase 6).
**Fecha de cierre**: 2026-05-29.
**Estado al cierre**: suite `tests/vision/` en **124 passed / 0 failed / 0 xfailed**
con Docker disponible (4 e2e nuevos + 120 heredados); **120 passed / 4 skipped**
sin Docker, con mensaje claro. Ruff verde sobre el archivo nuevo. Sin warnings
de marker desconocido.

---

## 1. Lo que Fase 7 entregó

Alcance único de F7 por decisión del usuario: cerrar el sub-test **CT-08.11(e)**
diferido por el handoff de Fase 6 (§1 y §4.3). Los otros cinco sub-tests
(a, b, c, d, f) ya estaban verdes desde Fases 4-6.

| Sub-fase | Commits | Entregable |
|---|---|---|
| **7a** | `3b3be977` | Dep nueva `testcontainers[postgres]>=4.0` en `requirements-dev.txt` con comentario inline del rationale. Commit aislado del test para que cualquier conflicto de resolución de deps se atrape antes. |
| **7b** | `0b89f7fd` | Marker `e2e` registrado en `pyproject.toml` bajo `[tool.pytest.ini_options].markers`. Nuevo `tests/vision/integration/test_persistence_e2e.py` con fixture session-scoped que levanta `timescale/timescaledb-ha:pg15` efímero vía testcontainers + cuatro tests sobre el repo. Skip-graceful si Docker no está. |
| **7c** | (este commit) | Handoff Fase 7 + nominación explícita de las dos deudas que F7 deja abiertas para F9. |

---

## 2. Mapa de cobertura CT-08.11 al cierre de Fase 7

Comparado con la matriz del handoff de Fase 6 §1 — la única fila que cambia
es (e), que pasa de "**cubierto en dos rutas, e2e diferido**" a "**cerrado para
repo↔BD-real**", con asterisco que aclara el alcance.

| Sub-test | Descripción | Estado al cierre de Fase 7 |
|---|---|---|
| (a) Detección | YOLO produce `list[DetectedVehicle]` | Cubierto desde Fase 4b. Sin cambio. |
| (b) Asignación direccional | `ZoneCounter.count()` con occupancy DHU-025 | Cubierto desde Fase 4a. Sin cambio. |
| (c) Derivación de métricas | `compute_traffic_data()` produce `TrafficData` §5.4 | Cubierto desde Fase 5b. Sin cambio. |
| (d) Endpoint `GET /vision/state` | Shape §6.5 + branch 5xx CT-08.10 | Cubierto en 6e. Sin cambio. |
| **(e) Integración persistencia** | `TrafficData` aterriza en `vision_aggregates` | **Cerrado para repo↔BD-real***. Cuatro tests en `test_persistence_e2e.py` corren contra Postgres/Timescale efímero (testcontainers): shape canónico §6.5, idempotencia `ON CONFLICT DO NOTHING`, Optionals→NULL, `CheckConstraint('mean_occupancy BETWEEN 0 AND 1')` activo en BD. *Asterisco*: paridad migración Alembic↔modelo SQLAlchemy NO validada — deuda nominada para F9.x. |
| (f) Caída del módulo | Health check + 5xx en `GET /vision/state` | Cubierto en 6f + 6e. Sin cambio. |

**Resumen**: los seis sub-tests cerrados al cierre de Fase 7. CT-08.11 entrega
completa, con la deuda residual de migración↔modelo declarada honestamente.

---

## 3. Decisión de diseño F7 — testcontainers vs. compose vivo

| Criterio | (a) testcontainers — **adoptada** | (b) Apuntar al compose vivo |
|---|---|---|
| Reproducibilidad | Test arranca su propio Postgres; cero estado externo. | Requiere `invoke up` previo + DB healthy. |
| Contaminación | DB efímera, descartada al final. | Contamina `cerebrovial`; limpieza al final crítica. |
| CI-readiness | GitHub runners tienen Docker — solo falta wirear el job (ver §5). | Requiere `services: postgres` en CI o levantar compose. |
| Patrón de proyecto | El smoke 4c es manual a propósito. Promoverlo a test exige autosuficiencia. | Repite el patrón smoke en lugar de mejorarlo. |
| Costo de arranque | ~7s overhead/sesión (medido: 1.97s → 9.27s). | 0s (asume compose corriendo). |
| Dep nueva | +1 dep dev: `testcontainers[postgres]>=4.0`. | Cero. |

**Imagen**: `timescale/timescaledb-ha:pg15` — la misma del compose. El repo NO
depende de hypertable-behavior (INSERT/SELECT ORM es transparente sobre
hypertables), pero usamos la misma imagen para no introducir divergencia
silenciosa de versión.

**Schema**: `VisionAggregateDB.__table__.create(engine)` desde el modelo
SQLAlchemy. NO se ejecuta la migración Alembic dentro del test — la
hipertabilización (`create_hypertable`) es perf-optimization, no cambia el
contrato de inserción/lectura. La tabla queda como regular table; los
constraints (`PRIMARY KEY`, `CheckConstraint`, columnas) se crean igual
desde el modelo y son lo que CT-08.11(e) verifica.

**Skip graceful**: `docker.from_env().ping()` en el fixture de sesión.
Si Docker no está, `pytest.skip()` con mensaje:
*"Docker no disponible — e2e CT-08.11(e) requiere Docker (…)"*. La suite
no rompe: 4 skipped en vez de 4 errors, propiedad "suite verde en cualquier
entorno" preservada.

---

## 4. Alcance honesto del e2e (D-005)

Dos claims acotados, explícitos para evitar que el cierre de F7 se sobre-afirme
en F9 o en la tesis:

### 4.1 Lo que el e2e SÍ valida

- `repo ↔ modelo SQLAlchemy ↔ Postgres vivo`. Concretamente, contra una BD
  real corriendo en un container efímero:
  - mapping de columnas del `_to_row()` — desempaque `vehicles_by_type →
    car/bus/truck/motorcycle_count` (incluyendo `motorcycle_count = 0` por
    `.get(default=0)` cuando el tipo está ausente del dict);
  - preservación TIMESTAMPTZ tz-aware UTC en `window_start`/`window_end`;
  - invariante §5.7 — `queue IS NULL` siempre (el `_to_row()` lo omite
    deliberadamente);
  - idempotencia `ON CONFLICT DO NOTHING` sobre la PK compuesta
    `(camera_id, zone_id, window_start)` — la unit test
    `test_postgres_repository.py` valida esto a nivel de SQL compilado;
    el e2e lo ejecuta contra Postgres real;
  - Optionals → NULL — `mean_speed_kmh=None` y `density_vehicles_per_km=None`
    aterrizan como `NULL` en las columnas nullable;
  - `CheckConstraint('mean_occupancy BETWEEN 0 AND 1')` **declarado en el
    modelo SQLAlchemy** está activo en Postgres — bypaseando el validador
    `__post_init__` del dominio.

### 4.2 Lo que el e2e NO valida

- **Paridad migración Alembic ↔ modelo SQLAlchemy.** El test crea la tabla
  desde el modelo (`VisionAggregateDB.__table__.create()`), NO ejecutando la
  cadena Alembic (`alembic upgrade head`). Si la migración
  `5b4beac1055d_vision_aggregates_and_drop_legacy_vision.py` y el modelo
  `shared/cerebrovial_shared/database/models.py:86-115` divergen, el test
  pasa y producción rompe. Mismo patrón de bit-rot que el csv legacy de
  F5b y la divergencia §5.8. **Deuda nominada para F9.x** (ver §6).

- **Pipeline e2e** `video → detección → zonas → agregación → TrafficData`.
  El e2e llama `repo.save()` directo, NO arranca el pipeline completo. Esa
  producción ya está verde por:
  - (a) detección — `test_yolo_detector.py`;
  - (b) asignación direccional — `test_zones.py` + `test_zone_counter_basic.py`;
  - (c) derivación de métricas — `test_compute_traffic_data.py` (11 tests).

  El wiring builder↔repo se cubre en `test_pipeline_wiring.py` (F6 paso
  bit-rot). El gap real que quedaba era repo↔BD-real, y eso es lo que F7
  cierra. La redacción del handoff de F6 §1 ("el e2e pipeline → Postgres
  real") sobre-afirmaba — el cierre real es más acotado y honesto.

---

## 5. CI — fuera de scope F7

El workflow [.github/workflows/ci.yml](.github/workflows/ci.yml) corre:
- `backend-checks` → `cd core_management_api && pytest tests/` (NO toca
  `edge_device/tests`).
- `frontend-checks` → vitest + eslint.
- `docker-build` → `docker compose build`.

`edge_device/tests/` (las 120 heredadas + los 4 nuevos = 124 hoy) está
fuera de CI desde siempre — decisión histórica TTH-03
([DECISIONS_HU.md:2548-2551](../lean-inception/4-decisiones/DECISIONS_HU.md#L2548-L2551)).

F7 hereda esa propiedad: **NO se cablea a CI dentro de F7**. Razones:
1. F7 (1.5 SP DHU-024) cierra una deuda de test, no la deuda más grande de
   wirear todo `edge_device/tests` a CI.
2. Wirear el e2e solo requiere un job nuevo con Docker; wirear los 124
   requiere decisiones (dependencias YOLO/torch en CI, tiempo de build,
   caching de imagen TimescaleDB) que pertenecen a una sub-fase de F9 o a
   un TTH-03 retomado.

Nominado en §6 como deuda separable F9.x.

---

## 6. Deuda nominada para Fases 8-9

### 6.1 Diferida explícitamente por el usuario

| Deuda | Estado | Cierre planeado |
|---|---|---|
| **F8 — dataset etiquetado ≥200 frames + métricas mAP/precisión/recall (CT-08.9)** | NO se etiqueta dataset, NO se mide precisión, NO se toca validación de detección dentro de F7. | F8 cuando el usuario lo decida. Es trabajo de **datos**, no de código — el grueso es etiquetar manualmente con Roboflow/CVAT/labelImg. |

### 6.2 Para F9 (documentación contractual y cross-refs)

| Deuda | Detalle | Forma de cierre sugerida |
|---|---|---|
| **`javier_prado.yaml` muerta** | Auditoría F7 confirmó: `grep -rn "javier_prado.yaml\|conf/vision/javier_prado" --include="*.py"` → 0 matches. Hydra root (`conf/config.yaml`) compone `vision: default`, no `javier_prado`. Las referencias a `javier_prado` en código son al string literal `cam_javier_prado_01` (camera_id de tests), no al YAML. Además tiene `persistence.type: "csv"` — bit-rot post-F5b. | Decidir en F9: (i) borrar; (ii) reescribir a postgres + zona Javier Prado real y documentar como "config de demo en vivo"; (iii) mover a `legacy/` con header explicando que quedó como referencia histórica de F1. |
| **Paridad migración Alembic ↔ modelo SQLAlchemy** | El e2e de F7 valida repo↔modelo↔BD pero NO migración↔modelo. Mismo patrón de bit-rot que el proyecto viene pisando (csv legacy F5b, divergencia §5.8). | Test chico que use `alembic.autogenerate.api.compare_metadata` contra una BD post-`upgrade head` y reviente si el diff no está vacío. Forma barata; cabe en F9 sin engordar 1.5 SP. |

### 6.3 Para F9.x (separable, opcional)

| Deuda | Detalle | Dueño |
|---|---|---|
| **Wirear `edge_device/tests` a CI** | TTH-03 históricamente postergado *"hasta que TTH-08 entregue módulo y tests estables"* — F7 cumple esa condición. Falta job CI nuevo con Docker, caché de imagen TimescaleDB, y decisión sobre deps pesadas (YOLO/torch). | F9.x o TTH-03 retomado. |

---

## 7. Estado de la rama al cierre

- **Branch**: `feature/tth-08-fase7-tests`.
- **Working tree clean** al cierre del último commit (este handoff).
- **Commits desde `master@39793c18`** (3 totales):
  1. `3b3be977` — 7a `chore(tth-08): testcontainers[postgres] en requirements-dev.txt`.
  2. `0b89f7fd` — 7b `test(tth-08): e2e CT-08.11(e) repo↔Postgres real (testcontainers + 4 tests)`.
  3. (este) — 7c `docs(tth-08): handoff Fase 7 + nominación de deudas a F9`.

**Pre-requisitos antes de mergear** (decisión humana, no del agente):
- Verificar suite verde en el entorno del revisor (`pytest tests/vision/` →
  124 passed con Docker, 120 passed + 4 skipped sin Docker).
- Abrir PR con este handoff como descripción.
- El agente **no mergea ni hace push** — esa decisión queda fuera de su scope.

Con Fase 7 mergeada y F8 diferida por decisión del usuario, TTH-08 entra en su
último tramo: **Fase 9 (documentación contractual + cross-refs + retiro de
C1.x)**, donde se cierra el sprint con el `vision_contract.md`, el cierre
formal de C7.6, la reafirmación de F41 (vision→predictivo como Trabajos
Futuros) y las dos deudas chicas nominadas acá (`javier_prado.yaml`,
paridad migración↔modelo).
