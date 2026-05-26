# Discovery formal pre-adopción SDD+BDD+TDD — CerebroVial

> **Fecha:** 2026-05-10
> **Branch:** `master`
> **Último commit (HEAD):** `69acbcd151e95cc54727edc467d1347068339509` — 2026-05-09 23:03:13 -0500 — *"Merge pull request #9 from CerebroVial-Miraflores/fase-2-andres-frontend-ci"*
> **Modo:** solo lectura — ningún archivo del código fue modificado durante esta auditoría. El único archivo creado es este reporte. **No commiteado.**
> **Discovery previo de referencia:** [`20260430_ARCHITECTURE_DISCOVERY.md`](20260430_ARCHITECTURE_DISCOVERY.md) (2026-04-30, 317 líneas). Las secciones 1–4 reportan **delta** sobre ese discovery; las secciones 5–11 son completas a fondo.
> **Tesis tomada como fuente oficial:** [`tesis/TB1-251-223-03-BH-u20241c919-u202418685 (2).docx`](../tesis/) (29.6 MB, mod. 2026-05-10).

---

## Índice

1. [Estructura general — delta](#1-estructura-general--delta)
2. [Documentación existente — delta](#2-documentación-existente--delta)
3. [Stack tecnológico real — delta](#3-stack-tecnológico-real--delta)
4. [Código de producción — delta + estado de Fase 2](#4-código-de-producción--delta--estado-de-fase-2)
5. [Tests](#5-tests)
6. [Datos y modelos de ML](#6-datos-y-modelos-de-ml)
7. [Historias de Usuario y trazabilidad — CRÍTICA](#7-historias-de-usuario-y-trazabilidad--crítica)
8. [Git y historial](#8-git-y-historial)
9. [Contradicciones y hallazgos](#9-contradicciones-y-hallazgos)
10. [Estado real de las fases del PLAN](#10-estado-real-de-las-fases-del-plan)
11. [Recomendación honesta](#11-recomendación-honesta)

---

## 1. Estructura general — delta

> Comparar contra `20260430_ARCHITECTURE_DISCOVERY.md` §1 (topología) y §3 (servicios).

### 1.1 Árbol de carpetas (master, 3 niveles, sin `node_modules`/`.venv`/`__pycache__`/`.git`)

```
CerebroVial/
├── core_management_api/
│   ├── alembic/{versions/}
│   ├── conf/vision/
│   ├── data/traffic_logs/
│   ├── docs/specs/
│   ├── models/                     # 8 .joblib RF (LFS)
│   ├── scripts/
│   ├── src/
│   │   ├── prediction/             # domain/application/infrastructure/presentation
│   │   └── control/                # domain/application/presentation  ← NUEVO Fase 3b
│   └── tests/{common, prediction, control}
├── edge_device/
│   ├── conf/vision/
│   ├── scripts/
│   ├── src/vision/                 # DDD completo
│   └── tests/vision/{unit, integration}
├── ia_prediction_service/
│   ├── config/, data/, notebooks/{data, logs}, scripts/
│   ├── src/{data, models, training, evaluation, utils, visualization}
│   └── tests/
├── frontend_ui/
│   ├── public/, src/{assets, components/{layout,modals,ui,views,widgets,views/control}, services, types}
├── shared/                         # ← NUEVO Fase 1, paquete pip `cerebrovial_shared`
│   └── cerebrovial_shared/{database, schemas, config}
├── infra/docker/initdb/
├── documentation/{docs/, legacy/, tesis/}
├── scripts/                        # ← seed.py global
├── scratch/                        # ← extract_thesis.py (1 archivo, posiblemente borrable)
├── .github/workflows/              # ← NUEVO Fase 2
└── (archivos raíz — ver 1.3)
```

### 1.2 Cambios estructurales por módulo desde 2026-04-30

| Módulo | Estado al 2026-04-30 | Estado actual | Delta |
|---|---|---|---|
| `core_management_api/` | Dockerfile roto (`src.main:app` inexistente). Solo `prediction/` con `control/domain.py` stub. Sin alembic real. | `src/main.py` creado, entrypoint correcto, `control/` ahora con `application/{webster,max_pressure,mtc_constraints,adaptive_engine}.py` + `presentation/api/routes.py` + 4 archivos de tests. Alembic con 3 migrations. | **MEJORÓ**: motor adaptativo implementado (HU-15, Webster + MaxPressure + MTC), alembic operativo, main.py real |
| `edge_device/` | Vision DDD funcional, common duplicado, `legacy_api.py` con imports rotos | Vision intacto, `common/` migrado a `shared/`, `legacy_api.py` aún presente con sus 4 endpoints (`/video_feed`, `/status`, `/health`, `/metrics`) | **PARCIAL**: common deduplicado vía shared/, legacy_api.py sigue presente |
| `ia_prediction_service/` | En compose, Dockerfile roto (`main.py` inexistente) | Sacado del compose (C3), Dockerfile sigue para entrenamiento manual | **MEJORÓ**: ya no rompe `docker compose up` |
| `frontend_ui/` | Sin Dockerfile, sin servicio en compose, URLs hardcoded, sin VITE env vars | Dockerfile multi-stage nginx, servicio frontend en compose, **sí usa `import.meta.env`** (VITE_CORE_API_URL, VITE_EDGE_API_URL, VITE_API_BASE_URL — defaults a localhost), AdminView/AlertsView siguen mock, **ControlView (motor adaptativo) completo en `views/control/` (9 archivos, 1034 líneas)**, integración E2E de intersecciones implementada (commit `f72d3ce0` 2026-05-05) | **MEJORÓ**: Dockerizado, parcialmente configurable por entorno, control adaptativo UI completo |
| `shared/` | No existía | Paquete pip `cerebrovial_shared` con `database/`, `schemas/`, `config/`, `logging`, `lfs_check`, `metrics`. Tests en `shared/tests/test_lfs_check.py` | **NUEVO**: Fase 1 cerrada (C1) |
| `infra/` | Solo SQL extensions | Sin cambios | Sin cambios |
| `scripts/` | No existía como carpeta raíz | `scripts/seed.py` global (carga 5 nodos + 6 edges + 4 cámaras + 1 admin de Miraflores) | **NUEVO**: Fase 2 E5 cerrada |

### 1.3 Archivos raíz (todos confirmados con `ls -la`)

| Archivo | Tamaño | Último commit toca | Existencia esperada |
|---|---|---|---|
| `README.md` | 17 KB (525 líneas) | `235a0721` 2026-05-09 | ✓ existe |
| `CLAUDE.md` | 5.9 KB (~120 líneas) | `037cc0b0` 2026-05-09 | ✓ existe |
| `pyproject.toml` | 1.7 KB (65 líneas) | reciente | ✓ existe |
| `requirements-dev.txt` | 67 B | C11 (2026-05-03) | ✓ existe |
| `docker-compose.yml` | 1.8 KB | 2026-05-09 | ✓ existe |
| `docker-compose.dev.yml` | 1.7 KB | `2c0422d7` 2026-05-09 | ✓ existe — **NUEVO** desde 2026-04-30 |
| `tasks.py` | 12 KB (362 líneas) | reciente | ✓ existe — **NUEVO** desde 2026-04-30 (C11) |
| `.env.example` | 1.9 KB | C0 (2026-05-03) | ✓ existe — *(Explore inicial reportó falsamente que no existía)* |
| `.gitignore` | 246 B | Fase 0 (2026-05-09) | ✓ existe |
| `.gitattributes` | 270 B | Fase 1 (C9) | ✓ existe (Git LFS rules) |
| `.dockerignore` | 95 B | reciente | ✓ existe |
| `.env` | 1.9 KB | local (no commiteable) | ✓ presente local — *verificar `.gitignore` lo cubra* |

**Notas:**
- El working tree de master está limpio; `feature/sdd-bdd-tdd-framework` apunta al mismo commit (0 diff).
- El único archivo untracked en master al checkout fue `documentation/tesis/TB1-...(2).docx` (no commiteado todavía).

---

## 2. Documentación existente — delta

> Comparar contra `20260430_ARCHITECTURE_DISCOVERY.md` §11 (preguntas abiertas).

### 2.1 Inventario `documentation/`

```
documentation/
├── docs/
│   ├── 20260430_ARCHITECTURE_DISCOVERY.md   (317 líneas, 32.9 KB)  — pre-Fase 1
│   ├── 20260503_PHASE1_CLOSURE.md           (125 líneas, 4.9 KB)   — NUEVO
│   ├── ARCHITECTURE_TARGET.md               (494 líneas, 25.5 KB)  — spec target
│   ├── BACKLOG_V2.md                        (104 líneas)           — NUEVO, DESACTUALIZADO (ver §9.2)
│   ├── COLLABORATION_NOTES.md               (171 líneas)           — NUEVO, método de trabajo
│   ├── DATA_MODEL.md                        (297 líneas, 10.3 KB)  — NUEVO
│   ├── DATA_MODEL_AUDIT.md                  (~250 líneas, 11.6 KB) — NUEVO 2026-05-03
│   ├── DECISIONS.md                         (90 líneas)            — NUEVO, 5 cerradas + 1 pendiente
│   ├── PLAN.md                              (128 líneas)           — NUEVO
│   ├── RNF02_LATENCY_REPORT.md              (~50 líneas)           — NUEVO 2026-05-04
│   └── TODO.md                              (156 líneas)           — NUEVO, granular
├── legacy/
│   ├── README.md                            (15 líneas)            — NUEVO, justifica preservación
│   └── DOCUMENTACION.md                     (773 líneas, 24.8 KB)  — pre-refactor
├── tesis/
│   ├── TB1-...(1).docx                      (6.2 MB, 2026-05-09)   — versión pre-refinamiento de backlog [legacy]
│   └── TB1-...(2).docx                      (29.6 MB, 2026-05-10)  — **OFICIAL** (HU000-HU019 originales + HU-01..HU-18 refinadas)
└── CONTROL.md              (552 líneas, 32.3 KB)  — NUEVO, teoría Webster+MaxPressure
```

### 2.2 Existencia de archivos esperados (del prompt original)

| Archivo solicitado | Estado | Contenido resumido |
|---|---|---|
| `documentation/docs/PLAN.md` | ✓ existe | Fases 0–4 con criterios de cierre. Fase 0 ✓, Fase 1 ✓, Fase 2 en curso, Fase 3+4 futuras |
| `documentation/docs/TODO.md` | ✓ existe | Bloques A–K, ~150 ítems, granular, fuente de verdad operativa |
| `documentation/docs/DECISIONS.md` | ✓ existe | D-001 a D-005 cerradas, D-PENDING-001 abierta (modelo RNN: reusar TimeThenSpace o GRU desde cero) |
| `documentation/docs/DATA_MODEL.md` | ✓ existe | 8 tablas (graph_nodes, graph_edges, cameras, waze_jams, waze_alerts, vision_tracks, vision_flows, vision_aggregates), ER, hypertables, índices PostGIS |
| `documentation/docs/ARCHITECTURE_TARGET.md` | ✓ existe | Spec target detallada (origen: CLAUDE.md.old reubicado). Incluye backlog **HUs ORIGINALES sin adaptar** (HU001-HU019, ver §7) |
| `documentation/docs/MODEL.md` | **NO existe** | Entregable de Fase 3a (F1 del TODO), pendiente |
| `documentation/docs/CONTROL.md` | **NO existe** | Entregable de Fase 3b (H1 del TODO). El motor adaptativo ya está implementado, pero la doc formal del CONTROL.md no se escribió todavía. La teoría vive en `CONTROL.md` raíz `documentation/` (ver §9.4) |

### 2.3 ADRs (registro decisional formal)

- No hay carpeta `adr/`, `adrs/`, `decisions/` ni patrón `ADR-*.md` o `0001-*.md`.
- `DECISIONS.md` actúa como ADR ligero: 5 decisiones cerradas + 1 pendiente.
- Decisiones D-006, D-007, D-008 mencionadas en `CLAUDE.md` (visión persistencia, datos del GRU) **no aparecen en `DECISIONS.md`**. Probablemente referidas en `DATA_MODEL_AUDIT.md` (que no tiene formato D-XXX). `[REVISAR JUNTOS]`

### 2.4 `documentation/tesis/`

- 2 archivos `.docx` presentes:
  - **(1)** versión pre-refinamiento del backlog (HU001-HU027) — 6.2 MB. Confirmado por el usuario como obsoleta.
  - **(2)** versión oficial — 29.6 MB. **Esta es la fuente de verdad para HUs.** Declara 20 HUs (HU000-HU019 originales) y refina a 18 HUs (HU-01..HU-18). Tiene Tabla 18 de trazabilidad refinada↔original.

`[REVISAR JUNTOS]`: ¿qué hacer con `(1).docx`? Recomendación en §11.

---

## 3. Stack tecnológico real — delta

> Comparar contra `20260430_ARCHITECTURE_DISCOVERY.md` §4 (stack).

### 3.1 Por módulo Python — cambios

| Módulo | Python | Stack principal | Delta clave |
|---|---|---|---|
| `core_management_api/` | 3.11 (Dockerfile) | FastAPI + Uvicorn + SQLAlchemy + GeoAlchemy2 + Alembic + psycopg2 + scikit-learn (RF) + Torch (deuda C7.5) + Pydantic + PassLib + python-jose + httpx | **Sacó** ultralytics, opencv-python, supervision, hydra-core, cap_from_youtube, imageio-ffmpeg, streamlink, shapely (C7). **Mantiene** torch + scikit-learn por STGNN residual (C7.5 deuda). 23→16 paquetes |
| `edge_device/` | 3.11 | FastAPI + ultralytics (YOLO) + supervision + opencv + Hydra + sse-starlette + SQLAlchemy + GeoAlchemy2 + shapely + scikit-learn + httpx + pandas + numpy | Sin cambios mayores. Mismo set que pre-Fase 1, deps necesarias para visión |
| `ia_prediction_service/` | 3.11 | PyTorch 2.3+ + PyTorch Lightning 2.0+ + tsl (git) + torch-geometric + tensorboard + pandas + numpy + pyyaml | Sin cambios desde 2026-04-30 |
| `shared/` | — | Pydantic 2.0+ + SQLAlchemy 2.0+ + GeoAlchemy2 + omegaconf (paquete pip `cerebrovial_shared`) | **NUEVO Fase 1** |

**Hallazgos de imports vs requirements** (heurística): no se detectaron deps obvias declaradas sin usar tras C7. **Pendiente verificar**: `torch` en `core_management_api` se usa solo por `core_management_api/src/prediction/infrastructure/models.py` (STGCNModel residual). Deuda C7.5 abierta. `[REVISAR JUNTOS]`

### 3.2 Frontend

- React 19.2.0, TypeScript 5.9.3, Vite 7.2.4, Tailwind 4.1.17 (vía `@tailwindcss/vite`), Vitest 4.0.15, Leaflet 1.9.4, react-leaflet 5.0.0, recharts 3.5.1, lucide-react 0.556.0, axios 1.13.2.
- Bundler: Vite.
- Dockerfile multi-stage (`node:20-alpine` build → `nginx:stable-alpine` serve).
- `.env.example` propio en `frontend_ui/` configurable.
- Vulnerabilidades npm: 0 (C10.2 cerrado).
- Build TS hoy falla por 3 tipos de errores preexistentes (C10.2.1): import React no usado (TS6133), global no tipado en tests (TS2304), test key en vite.config.ts (TS2769). **Bloquea build prod, no dev.** `[REVISAR JUNTOS]`

### 3.3 Base de datos

- Motor real: **PostgreSQL 15** con TimescaleDB-HA (single container).
- Extensiones inicializadas: PostGIS, postgis_topology, TimescaleDB (`infra/docker/initdb/01_extensions.sql`).
- **Sin MongoDB** (eliminado en C5). El compose previo lo declaraba sin uso.
- **3 migraciones Alembic** (todas del 2026-05-04, Bloque E):
  - `775d2d1db8b4_initial_schema.py` — crea 8 tablas core
  - `daec5fdcfcdd_timescaledb_hypertables.py` — convierte `waze_jams`, `waze_alerts`, `vision_tracks`, `vision_flows` en hypertables
  - `99319147948b_add_users_table.py` — tabla `users` (id, email, password_hash, role, created_at)
- `init_db()` eliminado (E4). `alembic upgrade head` corre en el entrypoint del core (`core_management_api/entrypoint.sh`).

### 3.4 Infraestructura — `docker-compose.yml` vs `docker-compose.dev.yml`

**Servicios en `docker-compose.yml`** (4):
| Servicio | Imagen / build | Puerto host | Depende de |
|---|---|---|---|
| `db` | `timescale/timescaledb-ha:pg15` | 5432 | — |
| `core_management_api` | build local | 8001 | `db` (healthy) |
| `edge_device` | build local | 8000 | `db` (healthy) |
| `frontend` | `./frontend_ui/Dockerfile` | 5173→80 | `core_management_api` |

**`docker-compose.dev.yml`** (override): solo modifica `core_management_api` (override entrypoint a `uvicorn --reload`, monta `core_management_api/` y `shared/` como volúmenes, **NO corre `alembic upgrade head` automáticamente**). Edge y frontend NO se overridean.

**Diferencias clave del compose vs 2026-04-30**:
- Eliminados: `db_mongo`, `api_gateway`, `ia_prediction_service`.
- Renombrado: `db_postgres` → `db` (consistente con `.env`).
- Añadido: `frontend` con Dockerfile real, `entrypoint.sh` que corre Alembic en el core.
- `docker-compose.dev.yml` nuevo para hot-reload del core.

---

## 4. Código de producción — delta + estado de Fase 2

> Las novedades del último delta son: control adaptativo completo (backend + frontend), integración E2E del dashboard, JWT/auth pendiente, persistencia visión→BD pendiente.

### 4.1 Resumen de cada módulo (versus 2026-04-30)

#### `core_management_api/` — 26 archivos `.py`

```
src/
├── main.py                                  # ← NUEVO C2 (entry FastAPI con routers)
├── prediction/
│   ├── domain.py
│   ├── application/{predictor.py, builder.py}
│   ├── infrastructure/{models.py, csv_loader.py, data_loader.py, engine.py, graph_builder.py, repository.py}
│   └── presentation/api/{routes.py, schemas.py}
└── control/                                 # ← NUEVO Fase 3b (HU-15) — fuera de orden del PLAN
    ├── domain.py                            # Stub previo
    ├── application/
    │   ├── webster.py                       # WebsterCalculator (ciclo óptimo off-peak)
    │   ├── max_pressure.py                  # MaxPressureController (peak + fallback round-robin)
    │   ├── mtc_constraints.py               # MTCRestrictionApplier (cumplimiento normativo)
    │   └── adaptive_engine.py               # AdaptiveEngine orquestador
    └── presentation/api/{routes.py, schemas.py}  # POST /control/recommend
```

Patrón arquitectónico: DDD ligero (domain/application/infrastructure/presentation). Coherente con `edge_device/src/vision/`.

#### `edge_device/` — 7 src files

Sin cambios estructurales mayores. Sigue siendo el subsistema mejor armado.

#### `ia_prediction_service/` — 32 archivos `.py`

Sin cambios. STGNN deprecated por D-002 (modelo será RNN/GRU). Pendiente decisión D-PENDING-001 (reusar TimeThenSpace o GRU desde cero).

#### `shared/` — 17 src files

```
cerebrovial_shared/
├── __init__.py
├── logging.py, metrics.py, utils.py, exceptions.py, lfs_check.py
├── database/{models.py, database.py}
├── config/manager.py
└── schemas/{vision.py, graph.py, camera.py, waze.py}
```

`lfs_check.py` valida en runtime que los binarios no sean pointers de LFS (deuda C9.6 cerrada parcialmente, integrada en `invoke check_lfs`).

#### `frontend_ui/src/` — frontend completo

```
components/
├── views/
│   ├── DashboardView.tsx       (380 líneas, real, mapa Leaflet, consume API real)
│   ├── CameraDetailView.tsx    (339 líneas, real, SSE + MJPEG)
│   ├── AnalyticsView.tsx       (113 líneas, parcialmente conectada)
│   ├── AlertsView.tsx          (47 líneas, MOCK)
│   ├── AdminView.tsx           (46 líneas, MOCK)
│   └── control/                 ← NUEVO HU-15 (motor adaptativo), 9 archivos, 1034 líneas
│       ├── ControlView.tsx (215)
│       ├── ModeSelector.tsx (67)
│       ├── Pedagogical422Card.tsx (75)
│       ├── PhaseEditor.tsx (122)
│       ├── PresetButtons.tsx (31)
│       ├── RecommendationPanel.tsx (209)
│       ├── Slider.tsx (36)
│       ├── TimingBar.tsx (51)
│       └── TrafficLightCycle.tsx (228)
├── widgets/{TrafficHistoryWidget.tsx, AIChatWidget.tsx, ...}
├── modals/ReportModal.tsx
├── ui/, layout/
services/
├── predictionService.ts        (usa VITE_CORE_API_URL)
└── controlService.ts           ← NUEVO HU-15 (usa VITE_API_BASE_URL)
```

### 4.2 Endpoints expuestos (grep de decoradores `@router`/`@app`)

**`core_management_api/` (3 endpoints) — `core_management_api/src/main.py` los monta:**
- `POST /predictions/predict` — `prediction/presentation/api/routes.py:25`
- `GET /predictions/history/{camera_id}` — `prediction/presentation/api/routes.py:64`
- `POST /control/recommend` — `control/presentation/api/routes.py:81`

**`edge_device/` (12 endpoints):**
- `POST /cameras/{camera_id}/start`
- `POST /cameras/{camera_id}/stop`
- `GET /cameras/status`
- `POST /cameras/{camera_id}` (registro)
- `GET /cameras` (lista)
- `GET /video/{camera_id}` (MJPEG)
- `GET /stream/{camera_id}` (SSE)
- `GET /snapshot/{camera_id}`
- Legacy en `legacy_api.py`: `GET /video_feed`, `GET /status`, `GET /health`, `GET /metrics`

**Endpoints CONSUMIDOS por el frontend:**
| Endpoint | Archivo frontend |
|---|---|
| `${VITE_CORE_API_URL}/predictions/predict` | `services/predictionService.ts` |
| `${VITE_CORE_API_URL}/predictions/history/{id}` | `widgets/TrafficHistoryWidget.tsx` |
| `${VITE_CORE_API_URL}/intersections` (inferido) | `views/DashboardView.tsx` (intersecciones) |
| `${VITE_EDGE_API_URL}/cameras` | `views/DashboardView.tsx` |
| `${VITE_EDGE_API_URL}/stream/{id}` (SSE) | `views/CameraDetailView.tsx` |
| `${VITE_EDGE_API_URL}/video/{id}` (MJPEG) | `views/CameraDetailView.tsx` |
| `${VITE_API_BASE_URL}/control/recommend` | `services/controlService.ts` |

**Endpoints expuestos pero NO consumidos (hoy):** `/snapshot/{id}`, los legacy de `edge_device`. `[REVISAR JUNTOS]` — ¿se mantienen, se borran, se documentan?

**Endpoints declarados en TODO/PLAN pero NO existentes:**
- `POST /api/auth/login` (E8, pendiente)
- `POST /api/ai/chat` (E14, Gemini moverse a backend, pendiente)
- `GET /api/alerts`, `POST /api/alerts` (I1, Fase 4, pendiente)
- `GET /api/reports/daily` (I4, Fase 4, pendiente)
- `GET /api/control/intersections/{id}/current-plan` (H4 del PLAN — el endpoint actual `/control/recommend` cumple parcialmente, pero no hay persistencia de planes ni historial)

### 4.3 Modelo de datos

- **8 tablas creadas en master** (3 migrations):
  - `graph_nodes`, `graph_edges` — geometrías PostGIS, llenadas por `scripts/seed.py`
  - `cameras` — llenada por seed
  - `waze_jams`, `waze_alerts` — hypertables vacías (sin ingestor Waze, fuera de alcance per `DATA_MODEL.md`)
  - `vision_tracks`, `vision_flows` — hypertables modeladas pero **NO llenadas hoy** (pipeline visión sigue persistiendo a CSV vía `csv_repository.py`)
  - `users` — para JWT (E7 hecha, pero E8-E10 pendientes; tabla existe vacía salvo seed admin)
- **Tabla planificada (E18, no existe aún):** `vision_aggregates` — pendiente, schema diseñado en `DATA_MODEL.md`.
- **Inconsistencia menor**: `DATA_MODEL.md` describe `vision_aggregates` con detalle pero la tabla **no existe** todavía. Es spec, no estado real.

### 4.4 Estado de Fase 2 (Bloque E del TODO) — recorrido por tarea

| Tarea | Descripción | Estado real (commits + archivos) | Marca TODO.md |
|---|---|---|---|
| E1 | `alembic init` + configurar | Hecho (master tiene `alembic.ini`, `env.py`, etc.) | [x] |
| E2 | Migración inicial autogenerate | Hecho (`775d2d1db8b4_initial_schema.py` 2026-05-04) | [x] |
| E3 | TimescaleDB hypertables migration | Hecho (`daec5fdcfcdd_timescaledb_hypertables.py`) | [x] |
| E4 | Borrar `init_db()` | Hecho (no aparece en `shared/`) | [x] |
| E5 | `scripts/seed.py` con datos Miraflores | Hecho (`scripts/seed.py` global) | [x] |
| E6 | Frontend lee intersecciones del backend | **HECHO según commit** `f72d3ce0 2026-05-05 feat(frontend): completada HU010 - integracion End-to-End de intersecciones (Tarea 1)` — pero **el TODO sigue marcado [ ]** | [ ] ← desactualizado |
| E7 | Modelo `User` en BD | Hecho (`99319147948b_add_users_table.py`) | [x] |
| E8 | Endpoint `POST /api/auth/login` | **NO existe en código** (no aparece en endpoints `core_management_api/src/`) | [ ] |
| E9 | `get_current_user` dependency | **NO existe** | [ ] |
| E10 | Aplicar auth a rutas | **NO aplicado** | [ ] |
| E11 | Cerrar CORS a `localhost:5173` + prod | **No verificable sin levantar el servidor**; código en `core_management_api/src/main.py` indeterminado en esta auditoría `[REVISAR JUNTOS]` | [ ] |
| E12 | LoginView + AuthContext frontend | **NO existe** (no hay `LoginView.tsx`) | [ ] |
| E13 | URLs configurables `VITE_API_BASE_URL` | **PARCIALMENTE HECHO** — frontend usa `import.meta.env.VITE_CORE_API_URL`, `VITE_EDGE_API_URL`, `VITE_API_BASE_URL` con defaults. `.env.example` en frontend existe | [ ] ← desactualizado parcial |
| E14 | Gemini key al backend | **NO hecho** (`AIChatWidget.tsx`, `ReportModal.tsx` aún llaman Gemini directo desde el browser) | [ ] |
| E15 | Frontend Dockerfile + servicio compose | Hecho | [x] |
| E16 | Verificar end-to-end `docker compose up` | Hecho (parcial: sin login todavía) | [x] |
| E17 | Commit ceremonial Fase 2 | **No emitido todavía** | [ ] |
| E18-E21 | Visión→BD persistencia | **Ningún commit relevante** | [ ] todos |

**Resumen Fase 2:** 9 de 21 tareas cerradas + 2 parcialmente cerradas (E6, E13). El bloque JWT/auth (E7-E12) está parado a la mitad. La persistencia visión→BD (E18-E21) no se inició.

### 4.5 Trabajo fuera del orden del PLAN

El motor adaptativo (Bloque H del TODO, declarado como Fase 3b en `PLAN.md`) **ya está implementado en código** mientras la Fase 2 sigue abierta:
- Backend: `src/control/application/` completo (webster, max_pressure, mtc_constraints, adaptive_engine), `routes.py` con `POST /control/recommend`, **tests pytest** completos (`test_engine.py`, `test_webster.py`, `test_mtc.py`, `test_max_pressure.py`), `pytest-cov` agregado al `requirements-dev.txt`.
- Frontend: `views/control/` con 9 componentes (1034 líneas) — wiring completo en `App.tsx + Sidebar + Header`.
- Documentación: `CONTROL.md` (552 líneas, Webster + Max Pressure + escenarios de fallo + conexión futura a predicción).
- Commits del 2026-05-07 al 2026-05-09 en branches `controladaptativo` y `upgrade-motor` (rasec1106), mergeados.

**Implicación:** el orden de ejecución no respeta `PLAN.md` (Fase 3b > Fase 2 incompleta). Esto no es bloqueante pero rompe el criterio "no avanzar a Fase 3 sin cerrar Fase 2" declarado en `PLAN.md` §"Reglas de avance" L122-128. `[REVISAR JUNTOS]`

---

## 5. Tests

### 5.1 Ubicación

Tests viven en cada módulo bajo `<module>/tests/`. No hay carpeta `tests/` raíz. `shared/` también tiene su propia carpeta de tests.

### 5.2 Por módulo

| Módulo | # archivos test | Framework | Tipo | Ratio test/src |
|---|---|---|---|---|
| `core_management_api/` | 7 (3 prediction + 4 control + 1 schemas) | pytest | unit + algunos integration (rutas) | 7/26 ≈ 27% |
| `edge_device/` | 14 (11 unit + 3 integration) | pytest | unit + integration | 14/7 (más test que src del subdir visión, pero `src/vision/` tiene más archivos que el conteo solo top-level) |
| `shared/` | 1 (`test_lfs_check.py`) | pytest | unit | 1/17 ≈ 6% |
| `ia_prediction_service/` | 3 (data_loader, preprocessing, models) | pytest | unit (uno antes vacío) | 3/32 ≈ 9% |
| `frontend_ui/` | 2 (`CameraDetailView.test.tsx`, `TrafficHistoryWidget.test.tsx`) | vitest + @testing-library/react + jsdom | unit | 2/~30 archivos `.tsx` ≈ 7% |

**Cobertura estimada (heurística sin correr coverage):** edge_device es el módulo mejor cubierto (~50–70%). core_management_api con control adaptativo recientemente sumado, cobertura razonable del control (~70% por archivos) pero predicción más floja (3 tests vs 6 archivos src). frontend_ui sub-cubierto.

### 5.3 Tests BDD / Gherkin

**Cero archivos `.feature`**, cero deps `behave`, `pytest-bdd`. Coincide con el contexto del proyecto (no se adoptó BDD todavía — eso es precisamente lo que se va a evaluar con este discovery).

### 5.4 Tests E2E

**No existen tests end-to-end** que levanten varios servicios. El TODO los planifica en J5 (Fase 4b): `pytest+httpx`: login → consulta intersecciones → trigger predicción → consulta plan → log auditoría.

### 5.5 CI

`.github/workflows/ci.yml` (creado en Fase 2):
- **Jobs:**
  - `backend-checks`: Python 3.12, `pip install ruff pytest` + `pip install -r core_management_api/requirements.txt`, `ruff check .`, `pytest tests/`.
  - `frontend-checks`: Node 24, `npm ci`, `npm run lint`, `npm run test -- --run`.
  - `docker-build`: `docker compose build`.
- **Triggers:** push a `master` o `fase-*`; PR a `master`.
- **Estado:** según commits recientes (PR #9 merge), el CI **está pasando** (no se ven mensajes "CI broken" en commits).

**Observación:** CI corre tests de `core_management_api` y frontend, pero **NO de `edge_device`, `shared/`, ni `ia_prediction_service/`**. La cobertura CI es selectiva. `[REVISAR JUNTOS]`

### 5.6 Deuda en tests preexistente (xfail)

Documentada en `PHASE1_CLOSURE.md` y `TODO.md`:
- C1.5 — race condition en `test_pipeline_processing_flow`
- C1.6 — `MultiCameraManager` (API CameraInstance cambió)
- C1.7 — `SmartDetectionProcessor` interpolación + trayectorias (PRIORIDAD ALTA, bloquea entrenamiento de GRU)
- C1.8 — `ZoneCounter` polygon contains (PRIORIDAD ALTA, conteo por zonas)

Tests marcados xfail según la convención de `COLLABORATION_NOTES.md`.

---

## 6. Datos y modelos de ML

### 6.1 Modelos entrenados presentes (Git LFS confirmado vía `git lfs ls-files`)

| Archivo | Path | Tamaño | Estado | Cargado por |
|---|---|---|---|---|
| `yolo11n.pt` | `edge_device/yolo11n.pt` | 5.4 MB | **VIVO** | `edge_device/conf/config_models.py`, tests `edge_device/tests/vision/` |
| `traffic_rf_class_{15min,30min,45min,current}.joblib` | `core_management_api/models/` | ~3 MB c/u | **VIVOS** (RandomForest baseline temporal per D-002) | `core_management_api/src/prediction/infrastructure/engine.py` (load via joblib) |
| `traffic_rf_reg_{15min,30min,45min,current}.joblib` | `core_management_api/models/` | ~3 MB c/u | **VIVOS** | mismo path |
| `epoch=79-step=30800.ckpt` | `ia_prediction_service/notebooks/logs/` | 244 KB | **LEGACY** — STGNN deprecated por D-002; conservado como referencia |  No cargado en runtime de prod |

**Total binarios LFS:** 13 (8 RF .joblib + 1 yolo .pt + 1 STGNN .ckpt + metr_la datasets + tesis (1).docx).

### 6.2 Datasets

- `metr_la.h5` (11 MB, `ia_prediction_service/notebooks/data/`) — dataset estándar METR-LA (Los Ángeles, 207 sensores). Per D-008 en `CLAUDE.md`, se mantiene en LFS como **input de calibración** del dataset sintético del GRU futuro.
- `metr_la_dist.npy` — matriz de distancias para METR-LA.
- `ia_prediction_service/data/locations.csv` — coordenadas de los nodos METR-LA.
- **No hay datasets reales de Miraflores**: las cámaras YouTube siguen siendo la única fuente de datos en vivo (3 de 4 streams ya no existen según C3.5 deuda).

### 6.3 Pipeline de entrenamiento

- Vive en `ia_prediction_service/`.
- Scripts: `scripts/{train,predict,evaluate,visualize_network}.py`.
- Notebooks: 4 en `notebooks/` + `notebooks/logs/` con 1 checkpoint.
- **No es reproducible end-to-end hoy:** no hay un script `generate_synthetic_data.py` (F2 del TODO), el dataset de entrenamiento del GRU **no existe**. Lo que existe entrena con METR-LA, no con datos de Miraflores.
- **D-PENDING-001 abierta**: reusar `time_then_space.py` (GRU embebido + DiffConv) o GRU desde cero. Bloquea inicio de Bloque F.

---

## 7. Historias de Usuario y trazabilidad — CRÍTICA

> Esta sección es central para diseñar el plan retroactivo SDD+BDD+TDD. Aquí está el caos.

### 7.1 HUs declaradas en la tesis oficial (`TB1-...(2).docx`)

La tesis declara **20 HUs originales** (HU000–HU019) y las refina a **18 HUs vigentes** (HU-01..HU-18). 7 HUs originales fueron **explícitamente diferidas a trabajo futuro**: HU003, HU007, HU009, HU011, HU013, HU016, HU017. La justificación se desarrolla en el Capítulo 6 de la tesis.

### 7.2 Tabla 18 — Trazabilidad refinada↔original (extraída del .docx)

| HU vigente (tesis 2) | HU original | Tipo de cambio | Título (refinado) |
|---|---|---|---|
| HU-01 | HU008 | Reformulación | (modelo IA detección de vehículos) |
| HU-02 | HU010 (parte) | Desglose | (visualización tiempo real — parte) |
| HU-03 | HU001 + HU002 (parte) | Consolidación | (obtención + organización de datos — parte) |
| HU-04 | HU005 (versión inicial) | Acotamiento a baseline sintético | Modelo predictivo baseline sintético |
| HU-05 | HU010 (parte) | Desglose | (visualización tiempo real — otra parte) |
| HU-06 | (transversal nueva) | HU emergente | (transversal) |
| HU-07 | HU001 + HU019 | Consolidación | (obtención + estabilización repo) |
| HU-08 | HU002 (cierre) | Cierre técnico | (organización datos — cierre) |
| HU-09 | HU008 (refuerzo) | Cierre técnico | (modelo detección — refuerzo) |
| HU-10 | HU005 (rama investigativa) | Desglose | Investigación de modelo predictivo avanzado (STGNN) |
| HU-11 | HU005 (cierre) | Cierre productivo | **Modelo predictivo productivo con datos del piloto** (= GRU final) |
| HU-12 | HU010 (cierre) | Cierre técnico | **Integración frontend-backend con datos productivos** |
| HU-13 | HU015 | Reformulación: contenedores | Despliegue containerizado del sistema completo |
| HU-14 | HU014 | Sin cambios sustantivos | Acceso seguro con autenticación y control de roles |
| HU-15 | HU004 | Reformulación: recomendación | **Motor de reglas adaptativo para gestión semafórica** |
| HU-16 | HU018 | Reformulación: validación simulada | Validación del motor adaptativo en simulador SUMO |
| HU-17 | HU006 | Sin cambios sustantivos | Comparación cuantitativa de KPIs con/sin sistema |
| HU-18 | HU012 | Reformulación: reporte único | Reporte ejecutivo de resultados del piloto |

### 7.3 Roadmap del producto declarado en la tesis

| Sprint | Foco | HUs | SP | Fecha |
|---|---|---|---|---|
| Sprint 1 | Fundamentos y modelos de IA | 5 | 42 | cerrado dic-2025 |
| Sprint 2 | Refactor arquitectónico e investigación | 5 | 39 | cerrado feb–mar 2026 |
| Sprint 3 | Cierre de deuda y nuevas capacidades | 5 | 39 | en curso abril–mayo 2026 |
| Sprint 4 | Validación SUMO y KPIs comparativos | 3 | 26 | por iniciar mayo–junio 2026 |
| **Total** | | **18** | **146** | |

### 7.4 Trazabilidad en el repo (grep exhaustivo)

| Tipo de mención | Numeración usada | Ubicación | Implicación |
|---|---|---|---|
| `HU-12` (refinada con guión) | **predicción de congestión** | `documentation/docs/CONTROL.md` L3, 274, 276, 278, 294, 300, 304, 322, 330, 340, 346, 348, 388, 390 (14 menciones) | **NO calza con la tesis (2)**: HU-12 refinada = "Integración frontend-backend con datos productivos", NO "predicción". El documento usa una numeración propia/intermedia |
| `HU-12` (texto literal) | Reportes | `documentation/docs/TODO.md` L126 ("I4. HU12 — Reportes...") | **Calza con HU-12 = HU012 original = "Reportes Automáticos"** según `ARCHITECTURE_TARGET.md` L273. Numeración ORIGINAL sin guión |
| `HU010`, `HU012`, `HU015`, `HU016` (sin guión) | Identificadores en commits | `[HU015]` (motor), `HU010` (integración E2E), `[HU016-fix]` (tests) | **Mezcla**: `HU015` aquí se usa para motor adaptativo, lo que **calza con la refinada HU-15** (motor) y NO con la original HU015 (containerización). `HU010` parece ser original (integración E2E ≈ original HU010 "Visualización tiempo real") |
| `HU0XX` (con tres dígitos) | Backlog | `documentation/docs/ARCHITECTURE_TARGET.md` L247-277 (lista completa) | Numeración **ORIGINAL sin adaptar** (HU004 = "Gestión Semafórica", HU015 = "Infra Nube") |
| `HU0XX` (con adaptaciones) | Backlog | `documentation/docs/BACKLOG_V2.md` | Numeración **ORIGINAL adaptada** (HU004 = "Gestión semafórica"; HU015 = "Containerización Docker"). Contiene tabla de cambios |
| `HU03`, `HU04`, `HU06`, `HU09`, `HU17` (con guión, 2 dígitos sin zero-pad) | TODO.md L4, L115, L123, L125-127 | Numeración ambigua. `HU03` (contingencia) calza con original `HU003`. `HU06` (comparación) calza con `HU006`. `HU17` (alertas) calza con `HU017`. **Pero estas HUs originales están explícitamente DIFERIDAS por la tesis (2)** | **CONFLICTO**: el TODO trabaja HUs que la tesis difirió |
| Branches | `controladaptativo`, `upgrade-motor`, `fase-*` | No nombran HUs explícitamente | — |

### 7.5 Síntesis del caos de numeración

**Tres numeraciones operando simultáneamente, sin convención clara:**

1. **Tesis (2) refinada**: `HU-01` a `HU-18` (2 dígitos zero-padded, con guión). Esta es la **fuente de verdad académica**.
2. **Tesis (2) original**: `HU001` a `HU019` (3 dígitos zero-padded, sin guión). Mapeada en Tabla 18.
3. **Documentos internos**: mezcla arbitraria de variantes:
   - `ARCHITECTURE_TARGET.md`: original sin adaptar.
   - `BACKLOG_V2.md`: original adaptada (HU015 reinterpretada como containerización).
   - `CONTROL.md`: numeración intermedia inventada (`HU-12` = predicción, no calza con ninguna).
   - `TODO.md`: mezcla original (`HU17`, `HU03`...) con refinada (en commits referidos: `[HU015]` = motor).
   - Commits: `[HU015]` = motor adaptativo (refinada); `HU010` = integración E2E (original).

**Consecuencia operativa:** sin una tabla de equivalencias maestra, cualquier afirmación tipo "HU-X está implementada" es ambigua. Si el jurado pregunta "muéstreme dónde está HU-15", la respuesta correcta depende de qué versión de "HU-15" estás citando (motor adaptativo refinada vs containerización original). `[REVISAR JUNTOS]`

### 7.6 Tabla de trazabilidad HU↔código (intento de mapeo a numeración refinada)

| HU refinada (tesis 2) | Original | Título | ¿En código? | Archivos clave | Estado real |
|---|---|---|---|---|---|
| HU-01 | HU008 | Modelo IA detección vehículos | ✓ | `edge_device/src/vision/` (YOLO), `yolo11n.pt` | **Implementado** |
| HU-02 | HU010 (parte) | Visualización tiempo real (parte) | ✓ | `frontend_ui/src/components/views/DashboardView.tsx`, `CameraDetailView.tsx`, SSE en `edge_device` | **Implementado** |
| HU-03 | HU001 + HU002 (parte) | Obtención + organización de datos | 🟡 | seed.py, schemas en `shared/`, sin ingestor Waze | Parcial |
| HU-04 | HU005 inicial | Predictor baseline sintético | ✓ | `core_management_api/models/*.joblib` (8 RF), `prediction/infrastructure/engine.py` | **Implementado** (RF) |
| HU-05 | HU010 (parte) | Visualización tiempo real (otra parte) | ✓ | mismo que HU-02 | Parcial |
| HU-06 | transversal | HU emergente | ? | indeterminado sin lectura tesis | `[REVISAR JUNTOS]` |
| HU-07 | HU001 + HU019 | Obtención + estabilización repo | ✓ | toda la estabilización Fase 1 (shared, alembic, compose, LFS) | **Implementado** |
| HU-08 | HU002 cierre | Cierre datos | 🟡 | DATA_MODEL.md + Alembic + seed.py, sin ingesta real | Parcial |
| HU-09 | HU008 refuerzo | Cierre detección | ✓ | mismo HU-01 (detección + medición) | Parcial (falta medición real D-005) |
| HU-10 | HU005 rama | Investigación STGNN | ✓ (legacy) | `ia_prediction_service/src/models/time_then_space.py` + checkpoint | **Implementado** (deprecated per D-002, conservado) |
| HU-11 | HU005 cierre | **Modelo predictivo productivo con datos del piloto (GRU)** | ❌ | — | **NO implementado**. Bloque F del TODO pendiente. D-PENDING-001 abierta |
| HU-12 | HU010 cierre | **Integración frontend-backend con datos productivos** | 🟡 | `frontend_ui/src/services/predictionService.ts` + `views/DashboardView` (commit `f72d3ce0`) | **Parcial** (integración E2E hecha, falta consumir GRU productivo) |
| HU-13 | HU015 | Despliegue containerizado | ✓ | `docker-compose.yml`, todos los Dockerfile, `frontend_ui/Dockerfile` multi-stage | **Implementado** |
| HU-14 | HU014 | Auth con roles | 🟡 | `users` table existe (E7), JWT NO implementado (E8-E12) | **Parcial** (tabla creada, código auth NO) |
| HU-15 | HU004 | **Motor adaptativo** | ✓ | `core_management_api/src/control/{webster,max_pressure,mtc_constraints,adaptive_engine}.py` + tests + `frontend_ui/src/components/views/control/` + `services/controlService.ts` + `CONTROL.md` | **Implementado completo** (backend + frontend + teoría + tests) |
| HU-16 | HU018 | Validación SUMO | ❌ | — | **NO implementado**. Cero SUMO en el repo. Solo 1 commit fix de PYTHONPATH `[HU016-fix]` |
| HU-17 | HU006 | Comparación cuantitativa KPIs | ❌ | — | **NO implementado**. Depende de HU-16 |
| HU-18 | HU012 | Reporte ejecutivo | ❌ | — | **NO implementado** |

**Resumen cobertura:** 6 HUs implementadas, 5 parciales, 5 NO implementadas, 2 indeterminadas. Total: 7/18 con código maduro (~39%).

`[REVISAR JUNTOS]` — confirmar mapeo HU-03, HU-05, HU-06 abriendo el .docx en las secciones específicas.

---

## 8. Git y historial

### 8.1 Métricas

- **Commits totales:** 91 (`git rev-list --count HEAD`).
- **Branches locales:** master, feature/sdd-bdd-tdd-framework, controladaptativo, upgrade-motor.
- **Branches remotas activas:** master, fase-1-estabilizacion, fase-2-{alembic, andres-frontend-ci, cimientos, cimientos-e, gru}, controladaptativo, upgrade-motor.
- **Last commit:** `69acbcd1` — 2026-05-09.
- **Remote:** `https://github.com/AndresBR2003/CerebroVial.git`.

### 8.2 Autores

| Autor | Commits | Notas |
|---|---|---|
| `rasec1106` | 73 | ~80% — autor principal, owner de Fase 1 y motor adaptativo |
| `AndresBR2003` | 13 | Co-autor — frontend + CI (PR #9), owner del repo en GitHub |
| `Cesar Herrera` | 7 | Tercer contribuyente — bajo perfil pero presente |

### 8.3 Últimos 20 commits (delta desde 2026-04-30, 39 commits)

```
69acbcd1 2026-05-09 Merge PR #9 (fase-2-andres-frontend-ci)
8430a522 2026-05-09 Merge master into fase-2-andres-frontend-ci
9e715123 2026-05-09 Merge PR #8 (upgrade-motor)
[muchos commits 2026-05-09] Fase 10c.2-bis: refinamientos UI motor adaptativo (sliders, animaciones, presets)
753182d4..a1875efa 2026-05-09  Fase 10c motor adaptativo frontend (PhaseEditor, RecommendationPanel, etc.)
1416b8ae 2026-05-09 [Hotfix][Infra] alembic env.py @db→localhost fuera del container
dd056cbc 2026-05-09 [Docs] CONTROL.md (teoría completa)
958a81c6 2026-05-09 [Docs] invoke up-build --service docs
0381b7ec 2026-05-09 [Tooling] invoke up-build --service
fb500e90 2026-05-09 Merge PR #7 (controladaptativo)
037cc0b0 2026-05-09 [Docs] CLAUDE.md tasks invoke + alembic notes
235a0721 2026-05-09 [Docs] Reescribir README.md flujos
1630b5d1 2026-05-09 [Tooling] tasks: migrate, seed, db-reset, up-dev, shells
2c0422d7 2026-05-09 [Infra] docker-compose.dev.yml hot-reload
55466093 2026-05-09 [Infra] alembic upgrade head en entrypoint del core
[7 commits Fase 10c HU015 Frontend control]
[8 commits Fase 10b HU015 Backend control (Webster, MaxPressure, MTC, AdaptiveEngine, /control/recommend, tests, pytest-cov)]
83602722 2026-05-05 fix(frontend): linter any en DashboardView
f72d3ce0 2026-05-05 feat(frontend): completada HU010 - integración E2E intersecciones (Tarea 1)
566cbc98 2026-05-05 feat(frontend): completadas 6 tareas (3,5,6,7,8,9) estabilización UI y tiempo real
d1b60295 2026-05-04 [HU016-fix] Ajustar PYTHONPATH y filtrar tests backend
```

(Antes del 2026-05-04, son commits de Fase 1 — C0 a C12, todos del 2026-05-03 y 2026-05-04.)

### 8.4 Branches abandonadas (heurística)

Branches `origin/fase-1-estabilizacion`, `origin/fase-2-{alembic, cimientos, cimientos-e, gru}` están en commits ≤2026-05-04. Probablemente ya mergeadas o reemplazadas — `fase-2-gru` no aparece tener trabajo nuevo todavía. `controladaptativo` y `upgrade-motor` están en HEAD del 2026-05-09 — probablemente ya merged (PRs #7 y #8 mostraban ese mensaje).

### 8.5 PRs históricos visibles

- PR #7 — controladaptativo (motor backend)
- PR #8 — upgrade-motor (motor pulido)
- PR #9 — fase-2-andres-frontend-ci (frontend control + CI)

---

## 9. Contradicciones y hallazgos

### 9.1 ★ HALLAZGO SEVERIDAD ALTA — Inconsistencia de numeración HU entre código y tesis oficial

**Descripción:** Tres numeraciones HU operando simultáneamente en el repo, ninguna documentada como convención maestra. Lo más grave: `CONTROL.md` introduce una cuarta numeración intermedia (`HU-12` = predicción) que no calza con la tesis (2) refinada ni con la original.

**Citas exactas:**
- `documentation/docs/CONTROL.md:3`: *"se conectará con el modelo predictivo de congestión (HU-12) en SP4"*.
- `documentation/docs/CONTROL.md:274`: *"## 7. Conexión con HU-12 (predicción de congestión)"*.
- Tesis (2) Tabla 18 L1519: *"HU-12 | HU010 (cierre) | Cierre técnico | Integración frontend-backend con datos productivos"*.
- `documentation/docs/TODO.md:126`: *"I4. HU12 — Reportes: endpoint GET /api/reports/daily..."* → calza con original HU012 = "Reportes" → en refinada eso sería HU-18.

**Mapeo del `HU-12` de `CONTROL.md`**: probablemente se refiere a lo que **debería** ser HU-11 refinada ("Modelo predictivo productivo con datos del piloto" = GRU) — pero NO está implementada y se sigue usando el RF de HU-04. El término "predicción de congestión" tampoco calza con HU-04 (baseline sintético) exactamente.

**Propuesta concreta de reconciliación:**
1. Decidir como equipo **una única numeración de referencia**. Recomendado: la **refinada de la tesis (2)** (`HU-01..HU-18`).
2. Crear `documentation/docs/HU_MAPPING.md` permanente con:
   - Tabla refinada↔original (copia de Tabla 18 con título completo).
   - Para cada HU, el módulo y archivos del repo que la implementan.
   - Notas sobre numeración legacy en commits (cómo interpretar `HU010` en `f72d3ce0`).
3. Reescribir `documentation/docs/CONTROL.md` reemplazando todas las menciones a `HU-12` por la correcta (probablemente HU-11 si "predicción productiva" o HU-04 si "baseline").
4. Actualizar `TODO.md` y `BACKLOG_V2.md` para usar nomenclatura refinada (`HU-XX` con guión y zero-pad).
5. Convención de commits a futuro: `[HU-XX]` (refinada) en mensajes.

**Plazo sugerido:** **antes de iniciar el plan retroactivo de SDD+BDD+TDD**. No se puede hacer trazabilidad de tests/especs sobre una numeración inconsistente.

### 9.2 BACKLOG_V2.md desactualizado

`documentation/docs/BACKLOG_V2.md` (fecha declarada 2026-05-02):
- Dice "Fase 1 — Estabilización **❌ Pendiente**" y "Fase 2 — Cimientos **❌ Pendiente**" en su tabla "Resumen por fase".
- Pero `PHASE1_CLOSURE.md` y `CLAUDE.md` afirman que Fase 1 cerró el 2026-05-03.
- HU010 marcado 🟡 pero commit `f72d3ce0 2026-05-05` la declara "completada" en frontend.
- HU004 (motor) marcado ❌ pero `[HU015]` está implementado (backend + frontend completos) en master.

**Acción sugerida:** o regenerar BACKLOG_V2.md con estado al 2026-05-10, o explícitamente deprecarlo y reemplazar por una tabla en `PLAN.md` o `TODO.md`. Sin actualizar, da información incorrecta al primer lector que abra el repo.

### 9.3 PLAN.md vs orden real de ejecución

`documentation/docs/PLAN.md` §"Reglas de avance" L122-128:
> - No iniciar Fase 2 sin `docker compose up` verde (criterio Fase 1).
> - No iniciar Fase 3 sin la BD con datos reales y login funcional (criterio Fase 2).

**Pero**: el motor adaptativo (Fase 3b según el PLAN, Bloque H del TODO) **se implementó antes de cerrar Fase 2**. JWT/login (E8-E12) sigue pendiente. La regla "no iniciar Fase 3 sin login" se rompió.

Esto **no necesariamente es un error** — el motor adaptativo es lo más visible de la tesis y trabajarlo temprano es defendible. Pero el PLAN no refleja la realidad del orden de ejecución. `[REVISAR JUNTOS]`

### 9.4 `motor_adaptativo_teoria.md` no es `CONTROL.md` — RESUELTO

> ✅ Resuelto en `chore/orden-repo` (2026-05-25): `documentation/motor_adaptativo_teoria.md`
> fue movido y renombrado a `documentation/docs/CONTROL.md`, alineando el archivo con la
> declaración de `PLAN.md` Fase 3.

`PLAN.md` Fase 3 declara el entregable `documentation/docs/CONTROL.md` (con las reglas del motor justificadas). Existía en su lugar `documentation/motor_adaptativo_teoria.md` en la **raíz de `documentation/`** (no en `documentation/docs/`). 552 líneas, cubre Webster + Max Pressure + escenarios de fallo + conexión con predicción.

Acción tomada: renombrar/mover a `documentation/docs/CONTROL.md` por consistencia con el PLAN.

### 9.5 HUs diferidas por la tesis pero presentes en TODO.md

La tesis (2) declara explícitamente **diferidas a trabajo futuro**: HU003, HU007, HU009, HU011, HU013, HU016, HU017.

`TODO.md` tiene tareas asignadas a algunas de estas HUs:
- I1 — HU17 — Alertas (línea 123). La tesis difirió HU017 → "Alertas Automáticas".
- I3 — HU03 — Contingencia (L125). La tesis difirió HU003 → "Reglas de contingencia ante fallos".
- I5 — HU06 — Comparación antes/después (L127). HU006 corresponde a "Comparación de resultados" → en la refinada es HU-17 (NO diferida).

**Implicación:** algunas HUs que el equipo cree estar trabajando (I1, I3) son contradicciones explícitas con el alcance de la tesis. Cabe reconfirmar:
- ¿Se mantiene el alcance interno del TODO (trabajar HU17 alertas aunque la tesis lo difiera)?
- ¿O se ajusta el TODO para respetar lo que la tesis diferió?

Esto afecta el IE03 (≥95% HU aceptadas). `[REVISAR JUNTOS]`

### 9.6 Python version mismatch menor

- `Dockerfile` de los servicios Python: `python:3.11-slim`.
- `pyproject.toml` ruff: `target-version = "py312"`.
- CI `.github/workflows/ci.yml`: `python-version: '3.12'`.
- `README.md`: "Python 3.11 venv local".
- `tasks.py setup_dev`: `python3.11 -m venv .venv`.

El runtime es 3.11 pero el CI lint usa 3.12. Probablemente sin efectos prácticos para ruff con las reglas seleccionadas (E4, E7, E9, F), pero técnicamente inconsistente.

### 9.7 STGNN residual en `core_management_api` (deuda C7.5 vigente)

`core_management_api/src/prediction/infrastructure/models.py` aún contiene código STGCNModel + torch — la deuda C7.5 documentada en `TODO.md` sigue abierta. `requirements.txt` del core mantiene `torch` solo por este motivo. Impacto: imagen Docker de core ~+1.5 GB de lo necesario.

### 9.8 Streams YouTube rotos (C3.5 documentada)

Según `TODO.md` Bloque C deuda C3.5: *"3 de 4 streams YouTube ya no existen"*. Impacta directamente el demo del pipeline de visión.

### 9.9 Build TypeScript del frontend roto (C10.2.1 documentada)

`npm run build` falla por errores TS preexistentes. `npm run dev` y `npm run test` funcionan. CI usa `npm run test -- --run` y `npm run lint`, no `npm run build` — por eso el CI pasa pese a que el build prod no compila. `docker-compose.yml` arranca el frontend con el build estático nginx, lo cual sí ejecuta `npm run build` en el Dockerfile multi-stage. **Verificar si el Dockerfile del frontend está fallando hoy.** `[REVISAR JUNTOS]`

### 9.10 `D-006`, `D-007`, `D-008` mencionadas en CLAUDE.md pero ausentes de DECISIONS.md

`CLAUDE.md` raíz cita decisiones D-006, D-007, D-008 sobre persistencia visión a `vision_aggregates` y datos del GRU (METR-LA + Waze). Estas no aparecen en `DECISIONS.md` (que solo tiene D-001..D-005 + D-PENDING-001). Probablemente fueron documentadas en `DATA_MODEL_AUDIT.md` con formato distinto. El registro decisional está fragmentado. `[REVISAR JUNTOS]`

### 9.11 `scratch/extract_thesis.py`

Archivo de un solo uso, queda en el repo. Probablemente residuo de extracción anterior del .docx. `[REVISAR JUNTOS]` ¿borrar o mover a `documentation/legacy/`?

### 9.12 `legacy_api.py` en edge_device sigue presente

El discovery 2026-04-30 §10.4 reportó que `edge_device/src/vision/presentation/legacy_api.py` tenía imports rotos. **Persiste** en master. Aporta 4 endpoints legacy duplicados (`/video_feed`, `/status`, `/health`, `/metrics`). No queda claro si se carga o no en runtime.

---

## 10. Estado real de las fases del PLAN

### 10.1 Fase 1 — ✓ Cerrada el 2026-05-03

Verificado contra `PHASE1_CLOSURE.md`:
- `docker compose up` levanta 3 servicios sin crashes (db + core + edge): ✓
- Frontend `npm run dev` arranca: ✓
- `common/` consolidado en `shared/`: ✓ (paquete `cerebrovial_shared`)
- `requirements.txt` limpio: ✓ (C7) — deuda C7.5 sigue abierta (torch residual)
- Repo limpio: ✓
- Git LFS configurado: ✓ (13 binarios)
- README + tasks.py invoke: ✓

**Conclusión: Fase 1 cumple los criterios declarados en `PLAN.md`.** Deuda residual conocida y trackeada (C1.5-C1.8, C7.5, C9.6, C10.1, C10.2.1, C10.2.2).

### 10.2 Fase 2 — En curso, parcialmente completa (ver §4.4 para el detalle por tarea)

**Hecho (9 tareas + 2 parciales):**
- Bloque E1-E5 (Alembic + seed) ✓
- E7 (User model) ✓
- E15-E16 (Frontend Dockerizado, compose end-to-end) ✓
- E6 (intersections desde API) ✓ según commit (TODO desactualizado)
- E13 (URLs configurables) parcial — el frontend lee `VITE_*_API_URL` con defaults, pero algunos archivos siguen con literales

**Pendiente:**
- E8-E12 (JWT, login, frontend auth) — **NO iniciado**
- E14 (Gemini al backend) — **NO iniciado**
- E17 (commit ceremonial) — pendiente
- E18-E21 (visión → BD persistencia) — **NO iniciado**

**Criterio de "Fase 2 lista" del PLAN:** *"se puede hacer login en el frontend"*. **NO se cumple hoy.**

### 10.3 Fase 3 — Iniciada fuera de orden

**Fase 3a (Bloque F del TODO — GRU):** NO iniciada.
- F1 (MODEL.md spec) — no escrito.
- F2 (generate_synthetic_data.py) — no existe.
- F3 (gru_model.py) — no existe.
- F4-F9 — pendientes.
- Bloqueada por `D-PENDING-001` (reusar TimeThenSpace o GRU desde cero).

**Fase 3b (Bloque H del TODO — Control adaptativo):** **AVANZADA**.
- H2 (rules_engine) ✓ — `core_management_api/src/control/application/adaptive_engine.py` + `webster.py` + `max_pressure.py` + `mtc_constraints.py`.
- H3 (plan_repository) **NO HECHA** — no hay persistencia de planes en BD.
- H4 (endpoints) ✓ parcial — `POST /control/recommend` existe; `GET /control/intersections/{id}/current-plan`, `GET /control/history` NO existen.
- H5 (audit table) **NO HECHA**.
- H6 (frontend ControlView) ✓ — completo.
- H7 (tests motor) ✓ — `tests/control/test_{engine,webster,max_pressure,mtc}.py`.

### 10.4 Tareas hechas que NO aparecen en el PLAN

- **`CONTROL.md`** (552 líneas, defensa). Es entregable de Fase 3b según PLAN (H1 → `CONTROL.md`), pero en otra ubicación y con otro nombre.
- **`RNF02_LATENCY_REPORT.md`** — auditoría de latencia SSE. No estaba planificada explícitamente; corresponde al cumplimiento de RNF-02 declarado en la tesis.
- **`DATA_MODEL_AUDIT.md`** — auditoría empírica preparatoria de E2.
- **`COLLABORATION_NOTES.md`** — método de trabajo documentado.

Todo esto enriquece el repo. Pero no estaba previsto en el PLAN, lo cual sugiere que el PLAN debería actualizarse al cierre de Fase 2.

---

## 11. Recomendación honesta

### 11.1 Estado del proyecto

**Lectura sin filtro:** el proyecto está **sano pero con deuda metodológica**, NO en crisis. La fundación técnica es sólida; el problema es de **trazabilidad y coherencia narrativa**, no de calidad del código.

**Fortalezas observables:**
- Arquitectura clara (monolito modular, DDD aplicado en `vision/` y `control/`).
- Fase 1 cerrada con todos los criterios cumplidos y deuda registrada.
- Motor adaptativo (HU-15) **completo** backend + frontend + tests + teoría documentada — es lo más valioso del repo, y es el corazón académico de la tesis.
- Tooling de desarrollo de primera (invoke + docker-compose + alembic + LFS + entrypoint con migraciones + CI).
- Documentación cuidada y honesta (CLAUDE.md, COLLABORATION_NOTES, decisiones registradas).
- Método de trabajo formalizado y demostrado (Plan Mode + commits acotados + xfail + deuda registrada).

**Debilidades observables:**
- **Trazabilidad HU↔código rota** (3 numeraciones simultáneas, ver §7 y §9.1). Es el problema más serio para una defensa.
- **JWT/auth no implementado** (Fase 2 a la mitad).
- **GRU no empezado** (HU-11 productiva). Hoy se sirve RF baseline.
- **No hay SUMO** (HU-16 validación) → bloquea HU-17 comparación KPIs.
- **AdminView y AlertsView siguen mocks** en el frontend (46/47 líneas).
- **Cámaras YouTube rotas** (3 de 4) según deuda C3.5.

### 11.2 Qué preservar (no tocar sin motivo grave)

- `edge_device/src/vision/` — pipeline DDD con tests reales. Regla ya en `CLAUDE.md`.
- `core_management_api/src/control/` — implementación reciente, tests cubren los 4 algoritmos. Es la pieza de defensa.
- `shared/cerebrovial_shared` — Fase 1 ya validada, no romper la abstracción.
- `tasks.py` + `invoke` flow — la mejor herramienta de productividad del repo.
- `CLAUDE.md` y `COLLABORATION_NOTES.md` — método de trabajo demostrado.

### 11.3 Atención urgente (antes de cualquier otra cosa)

1. **Reconciliar la numeración HU** (§9.1). Sin esto, cualquier plan retroactivo SDD+BDD+TDD parte de bases inestables.
2. **Cerrar Fase 2: JWT/login (E8-E12)**. El criterio de cierre del PLAN exige login funcional. Tiempo estimado: 1–2 sesiones de Claude Code.
3. **Mover/eliminar tesis (1).docx** (ver §11.7). Reduce confusión y peso del repo.
4. **Actualizar o deprecar BACKLOG_V2.md** (§9.2). Hoy da estado incorrecto.

### 11.4 Puede esperar

- STGNN residual en core (C7.5) — molestia, no bloqueante.
- Build TS roto en frontend (C10.2.1) — no afecta dev ni CI; afecta solo build prod fuera de Docker.
- `legacy_api.py` en edge — limpio cuando se ataque módulo `edge_device`.
- Renombrar `CONTROL.md` → `docs/CONTROL.md` (§9.4) — cosmético.
- `scratch/extract_thesis.py` — borrarlo en limpieza ocasional.

### 11.5 Candidatos a retroactividad selectiva (nivel 2 — escenarios BDD ejecutables)

**Criterios usados:**
- (a) Comportamiento estable y probado.
- (b) Central para la defensa académica.
- (c) Endpoint/salida observable.
- (d) Escenario Gherkin tendría sentido sin inventar requisitos.

**Ranking de candidatos (ordenados por valor de defensa):**

#### Candidato #1 — HU-15 (Motor adaptativo) ★★★★★

Motivos: (a) ✓ tests pytest pasando, (b) ✓ corazón de la tesis (OE03), (c) ✓ endpoint `POST /control/recommend` observable, (d) ✓ CAs ya documentados implícitamente en `CONTROL.md`.

**Escenarios Gherkin de muestra:**

```gherkin
Feature: HU-15 — Motor de reglas adaptativo para gestión semafórica
  # CA por confirmar al revisar el .docx tesis (2) capítulo de HU-15

  Scenario: Cálculo de ciclo óptimo en off-peak con Webster
    Given una intersección con 4 fases declaradas y volúmenes de tráfico bajos
      And el modo del motor está configurado en "off-peak"
    When se invoca POST /control/recommend con esos datos
    Then el response trae un ciclo entre 60 y 120 segundos
      And cada fase tiene un tiempo de verde proporcional a su flujo según la fórmula de Webster
      And el modo retornado es "webster"
  # → TRIVIAL: el código existente cubre el escenario. Solo falta wirear el test.

  Scenario: Restricciones MTC se aplican sobre el ciclo calculado
    Given un cálculo Webster que sugiere ciclo de 130 segundos
    When MTCRestrictionApplier procesa el resultado
    Then el ciclo final está acotado al máximo MTC (típicamente 120s)
      And el response incluye un campo "mtc_overrides" listando los cambios aplicados
  # → TRIVIAL: lo cubre tests/control/test_mtc.py — solo falta promoverlo a Gherkin.

  Scenario: Fallback a round-robin cuando MaxPressure no converge
    Given una intersección en modo "peak" con datos de presión incompletos
    When el MaxPressureController detecta presión inválida
    Then el response cae al fallback round-robin con tiempos iguales por fase
      And el response incluye un campo "fallback_reason"
  # → TRIVIAL: el código tiene tests cubriendo esto.
```

#### Candidato #2 — HU-13 (Despliegue containerizado) ★★★★☆

Motivos: (a) ✓ probado via `invoke up`, (b) ✓ defendible como "el sistema arranca con un comando", (c) ✓ verificable via curl a `/health`, (d) ✓ CA típica "el sistema arranca con un solo comando".

```gherkin
Feature: HU-13 — Despliegue containerizado del sistema completo
  # CA por confirmar (probable CA-13.1: invoke up arranca el sistema)

  Scenario: Arranque exitoso con un solo comando
    Given Docker Desktop está corriendo
      And el archivo .env existe con credenciales válidas
      And git-lfs descargó los binarios
    When se ejecuta `invoke up`
    Then 4 servicios están healthy: db, core_management_api, edge_device, frontend
      And `invoke health` retorna 200
      And el frontend responde en http://localhost:5173
  # → REVELA FALTA: no hay test automatizado de este flujo (E2E pendiente, J5 del TODO).

  Scenario: `invoke check-lfs` aborta si los binarios no se descargaron
    Given el repo se clonó SIN tener git-lfs instalado
    When se ejecuta `invoke up`
    Then check_lfs detecta que .joblib son pointers
      And el comando aborta con mensaje claro apuntando a instalación de git-lfs
  # → TRIVIAL: tasks.py:check_lfs cubre esta lógica.
```

#### Candidato #3 — HU-04 (Predictor baseline RF) ★★★★☆

Motivos: (a) ✓ tests existentes en `tests/prediction/test_predictor.py`, (b) ✓ es lo que efectivamente sirve hoy, (c) ✓ endpoint `POST /predictions/predict`, (d) ✓ CA observable.

```gherkin
Feature: HU-04 — Modelo predictivo baseline (RandomForest sobre datos sintéticos)
  # CA por confirmar al revisar tesis (2) HU-04

  Scenario: Predicción de congestión a 15 minutos retorna nivel ordinal 1-5
    Given una cámara con datos de tráfico de los últimos 10 minutos
    When se invoca POST /predictions/predict con horizon=15
    Then el response incluye un campo congestion_level entero entre 1 y 5
      And el response incluye confidence entre 0 y 1
  # → TRIVIAL: lo cubre tests/prediction/test_routes.py.

  Scenario: Endpoint /predictions/history retorna histórico ordenado
    Given el módulo de visión persistió 100 predicciones para la cámara CAM_001
    When se invoca GET /predictions/history/CAM_001?interval=5
    Then el response incluye al menos 20 puntos
      And los puntos están ordenados cronológicamente
  # → REVELA FALTA: el histórico hoy se sirve desde CSV, no de Postgres — depende de E18-E21 visión→BD.
```

#### Candidato #4 — HU-12 (Integración frontend-backend con datos productivos) ★★★☆☆

Motivos: (a) ✓ integración E2E commited (`f72d3ce0`), (b) — defensa media, (c) ✓ verificable via UI, (d) ✓ CA "el dashboard muestra coordenadas reales".

```gherkin
Feature: HU-12 — Integración frontend-backend con datos productivos
  # CA por confirmar al revisar tesis (2) HU-12

  Scenario: Dashboard lista las intersecciones de Miraflores cargadas por seed
    Given la BD tiene los 5 nodos cargados por scripts/seed.py
    When se carga el frontend en /
    Then el mapa Leaflet muestra 5 marcadores
      And las coordenadas son las reales de Miraflores (Av. Larco, Pardo, Angamos, Arequipa, Ejército)
  # → TRIVIAL EN PARTE: el commit f72d3ce0 lo cubre. Falta endpoint /api/intersections — verificar.

  Scenario: CameraDetailView recibe SSE en tiempo real
    Given una cámara CAM_001 con stream activo
    When se abre CameraDetail
    Then la vista recibe eventos analysis vía EventSource
      And el contador de vehículos se actualiza al menos cada 5 segundos
  # → REVELA FALTA: RNF02_LATENCY_REPORT documenta bloqueo backend — el SSE no fluye en tiempo real hoy.
```

#### Candidato #5 — HU-01 (Detección vehículos con YOLO) ★★★☆☆

Motivos: (a) ✓ pipeline real con tests, (b) ✓ entrada de la cadena, (c) ✓ stream observable, (d) ✓ CA "el sistema detecta vehículos en el video".

```gherkin
Feature: HU-01 — Modelo IA de detección de vehículos (YOLO11n)
  # CA por confirmar — RNF02_LATENCY_REPORT relacionado

  Scenario: Pipeline detecta vehículos del frame y los publica
    Given el edge_device tiene una cámara YouTube en streaming
    When YOLO procesa un frame
    Then el detector retorna bounding boxes con class_id ∈ {2,3,5,7} (COCO)
      And el tracker asigna track_ids estables
      And el ZoneCounter emite eventos por zona
  # → REVELA FALTA: tests C1.7 y C1.8 (SmartDetectionProcessor, ZoneCounter) están rotos. Implementar este BDD obliga a cerrar la deuda.
```

**No candidatos para retroactividad (no implementadas):**
- HU-11 (GRU productivo) — no existe código.
- HU-14 (auth) — no existe código.
- HU-16 (SUMO) — no existe.
- HU-17 (KPIs comparación) — depende de HU-16.
- HU-18 (reportes) — no existe.

### 11.6 Riesgos para defensa académica (observables desde el código)

Ordenados por severidad:

| # | Riesgo | Evidencia | Mitigación posible |
|---|---|---|---|
| R1 | **HU-11 (GRU productivo) sin empezar.** El predictor declarado en la tesis es RNN/GRU, no RF. Hoy se sirve solo RF. | `models/*.joblib`, no hay `gru_model.py`, D-PENDING-001 abierta | Decidir D-PENDING-001 cuanto antes. Bloque F del TODO 6-8 días estimados |
| R2 | **HU-16 (SUMO) inexistente.** Sin validación cuantitativa simulada, no hay KPIs comparativos. | Cero SUMO en repo. Solo 1 commit fix de PYTHONPATH | Definir alcance acotado: simulación sintética sin SUMO si no llega al plazo |
| R3 | **HU-17 (comparación con/sin sistema) imposible sin HU-16.** Esto era parte de la integridad académica (D-005). | Sin datos baseline ni de comparación | Aclarar con asesor (A2) si "comparación contra Webster fijo" basta sin SUMO |
| R4 | **Numeración HU rota** (§9.1) → riesgo presentacional. Si jurado pregunta "muéstreme HU-X", la respuesta es ambigua. | `CONTROL.md` HU-12 = predicción ≠ tesis (2) | Reconciliar antes del primer ensayo de defensa |
| R5 | **HU-14 (auth) ausente** pero la tesis declara roles. | E8-E12 pendientes, sin LoginView | Implementar JWT mínimo (E8-E12) — 1-2 días si se prioriza |
| R6 | **RNF02 latencia: bloqueo backend documentado.** SSE no fluye real-time hoy. | `RNF02_LATENCY_REPORT.md` declara backend reprobado | Resolver concurrencia en edge_device (threading) antes de medir |
| R7 | **D-005 (números reales tras validación) compromiso académico.** Tesis declara 88.2%/81.3%/<2s; hoy no se pueden medir esos números. | C1.7, C1.8 deudas en visión bloquean medición de detección. Sin GRU no hay accuracy predictivo | Documentar limitaciones honestamente — el commit en D-005 ya prepara el camino |
| R8 | **Cámaras YouTube rotas** (3 de 4). | C3.5 deuda en TODO | Documentar como limitación del demo o reemplazar streams |
| R9 | **AdminView y AlertsView siguen mocks.** Visibles en demo si jurado pide tour completo. | 46/47 líneas, sin conexión a backend | Implementar mínimo o esconder en demo (degradación honesta) |
| R10 | **HUs trabajadas que la tesis difirió** (§9.5). Si TODO trabaja HU17 alertas, gastas tiempo en algo descopeado. | TODO L123-127 vs tesis cap 6 | Alinear scope: o re-incluir las HUs en tesis o sacarlas del TODO |

### 11.7 Recomendación fundamentada sobre `documentation/tesis/TB1-...(1).docx`

**Recomiendo: mover a `documentation/tesis/legacy/` con un README de 2 líneas.**

**Pasos concretos:**
1. `mkdir documentation/tesis/legacy`.
2. `git mv "documentation/tesis/TB1-...(1).docx" documentation/tesis/legacy/`.
3. Crear `documentation/tesis/legacy/README.md`:
   ```markdown
   # Versiones previas del documento de tesis

   - `TB1-...(1).docx` — versión con backlog HU001-HU027 (pre-refinamiento).
     La versión vigente vive en `documentation/tesis/TB1-...(2).docx`
     (HU000-HU019 originales + HU-01..HU-18 refinadas, ver Tabla 18).
   ```
4. Commit junto con la creación de `HU_MAPPING.md` (acción del §9.1) — todo en el mismo commit ceremonial de "estandarización trazabilidad pre-SDD".

**Justificación:**
- El patrón `legacy/` ya existe en el repo (`documentation/legacy/` con `DOCUMENTACION.md` antigua + README justificando preservación). Replicar el patrón en `tesis/legacy/` es consistente con la arquitectura informacional ya establecida.
- Preserva el histórico, útil si en defensa hay que explicar cómo evolucionó el scope (es buen material narrativo del proceso académico).
- Deja `documentation/tesis/` con un solo archivo activo, eliminando ambigüedad para futuros lectores o asistentes IA.
- El peso del repo no cambia (Git LFS ya cubre ambos `.docx` por la regla en `.gitattributes`).

**Alternativas evaluadas (peores):**
- Renombrar con sufijo `_LEGACY`: aumenta entropía en `tesis/`, no escalable si en el futuro hay v(3) v(4).
- Eliminar: pierde trazabilidad histórica, irreversible.

---

## Apéndice — Marcadores `[REVISAR JUNTOS]` consolidados

Los siguientes ítems requieren conversación con el usuario antes de cerrar el plan retroactivo:

1. **§2.3** — D-006, D-007, D-008 mencionadas en `CLAUDE.md` pero no en `DECISIONS.md`. ¿Migrar?
2. **§2.4** — Versión (1) del .docx — confirmada como legacy. Plan en §11.7.
3. **§3.1** — STGNN residual en core (C7.5). ¿Sacar ahora o en Fase 3a?
4. **§3.2** — Build TS roto en frontend (C10.2.1). ¿Confirmar que el Dockerfile multi-stage del frontend está corriendo el build hoy?
5. **§4.2** — Endpoints expuestos sin consumir (`/snapshot/{id}`, legacy). ¿Mantener/borrar/documentar?
6. **§4.4** — Estado real de E11 (CORS): pendiente verificar `core_management_api/src/main.py`.
7. **§4.5** — Motor adaptativo se hizo fuera de orden del PLAN. ¿Actualizar PLAN.md para reflejar el orden real?
8. **§5.5** — CI no corre tests de `edge_device`, `shared/`, `ia_prediction_service/`. ¿Ampliar coverage CI?
9. **§7.5** — Reconciliación de numeración HU (acción prioritaria).
10. **§7.6** — Confirmar mapeo HU-03, HU-05, HU-06 abriendo .docx en secciones específicas.
11. **§9.2** — BACKLOG_V2.md desactualizado. ¿Actualizar o deprecar?
12. **§9.3** — Fase 3b antes que Fase 2. ¿Actualizar PLAN.md?
13. **§9.4** — `motor_adaptativo_teoria.md` no era `CONTROL.md`. ✅ Resuelto: renombrado en chore/orden-repo (2026-05-25).
14. **§9.5** — HUs diferidas por tesis (HU017, HU003) en TODO. ¿Alinear scope?
15. **§9.6** — Python 3.11 vs 3.12 mismatch CI/runtime.
16. **§9.10** — D-006/007/008 fragmentadas.
17. **§9.11** — `scratch/extract_thesis.py`. ¿Borrar?
18. **§9.12** — `legacy_api.py` con imports rotos. ¿Limpiar?
