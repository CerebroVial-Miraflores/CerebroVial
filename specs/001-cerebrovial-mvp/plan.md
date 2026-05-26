# Implementation Plan: CerebroVial — Control adaptativo de semáforos (MVP1)

**Branch**: `feature/SDD` | **Date**: 2026-05-20 | **Spec**: backlog en `documentation/lean-inception/` (21 HU, 11 TTH, 22 RF, 53 RNF)

**Input**: Arquitectura objetivo en `documentation/sdd/SDD_CEREBROVIAL.md` (SDD verificado contra el repo).

> **Adopción brownfield (DHU-021).** Este `plan.md` **mapea** el SDD existente al formato de
> Spec Kit; no se regenera con `/speckit-plan`. El SDD es la fuente canónica de la
> descomposición y del diseño; aquí se resume y se enlaza, evitando duplicar texto.
> La fuente de cada afirmación es la sección citada del SDD.

## Summary

CerebroVial es un sistema de **control adaptativo de semáforos** para una intersección de
Miraflores (Lima), construido como **monolito modular** desplegado con Docker Compose
(D-001, D-003). Integra cuatro capacidades sobre un núcleo común: un **sensor de estado**
por visión (YOLO11n en `edge_device`), un **modelo predictivo** (GRU univariado por
intersección, con RandomForest de respaldo — D-002/D-006), un **motor adaptativo** de
control (Webster + MaxPressure + capa de cumplimiento normativo MTC — TTH-10) y un **entorno
de validación** con SUMO/TraCI (D-008, diseño objetivo). El sistema está desacoplado de la
fuente de su variable de estado (jam level / ratio velocidad-flujo-libre), lo que permite
intercambiar visión, SUMO o Waze sin alterar predicción ni control (§3.3 del SDD).

Detalle completo: `documentation/sdd/SDD_CEREBROVIAL.md` §1–§3.

## Technical Context

Valores verificados contra los manifiestos del repo (SDD §7).

**Language/Version**: Backend Python 3.11 (por `Dockerfile`); Frontend TypeScript 5.9.

**Primary Dependencies**:
- Backend (`core_management_api`): FastAPI, SQLAlchemy + GeoAlchemy2, Alembic — **sin pin de versión** en `requirements.txt` (se resuelven al construir la imagen).
- Visión (`edge_device`): `ultralytics` (YOLO11n, artefacto `yolo11n.pt`), `supervision`, OpenCV — sin pin.
- ML offline (`ia_prediction_service`): PyTorch `>=2.3.0`, PyTorch Lightning `>=2.0.0`.
- Frontend (`frontend_ui`): React 19.2, Vite 7, Tailwind 4, Leaflet 1.9 — **versiones pineadas** en `package.json`.
- Base común: `cerebrovial_shared` (paquete pip local, `pip install -e ../shared`) — aloja los modelos ORM.

**Storage**: PostgreSQL 15 con TimescaleDB + PostGIS, en una sola instancia (imagen `timescale/timescaledb-ha:pg15`); migraciones con Alembic (exclusión de tablas internas PostGIS vía `include_object` en `env.py`). Volumen `db_data`.

**Testing**: pytest (backend; `edge_device/tests/vision/` tiene cobertura real); Vitest + Testing Library (frontend).

**Target Platform**: Despliegue local en una máquina con Docker Compose (D-003); mapeo conceptual edge/servidor documentado, no entregado (D-004, SDD §6.2).

**Project Type**: Web — backend (monolito modular FastAPI) + frontend SPA, + nodo edge de visión + entrenamiento ML offline.

**Performance Goals / Constraints / Safety**: latencia de respuesta del motor objetivo <2 s; operación **fail-safe** a tiempos preconfigurados ante caída de la fuente (RNF-SAF-01/02/03); cumplimiento normativo MTC en cada decisión (capa MTC). Detalle en SDD §8.

**Scale/Scope**: una intersección piloto; modelo univariado por intersección (sin grafo espacial, D-006). MVP1 = 21 HU / 11 TTH; ≈25% implementado al 2026-05-18 (SDD §11).

## Constitution Check

> Las decisiones rectoras viven en `DECISIONS.md` (D-001…D-009) y `DECISIONS_HU.md`
> (DHU-001…DHU-022). El `constitution.md` de Spec Kit está **poblado** (2026-05-20,
> 22 artículos vinculantes: Título I D-001…D-009, Título II DHU-001…DHU-022) y
> ratificado; ver `.specify/memory/constitution.md`. Gates derivados de las
> restricciones fundacionales (SDD §2):

- **D-001 monolito modular** — sin API HTTP entre módulos internos; `ia_prediction_service` fuera del runtime. ✔ verificado en `docker-compose.yml` (4 servicios, sin el de entrenamiento).
- **Restricción de no-refactor de `edge_device/src/vision/`** (CLAUDE.md) — vigente; no se toca sin sprint dedicado. ✔
- **Migraciones solo con Alembic** (nunca `create_all`) — ✔ entrypoint corre `alembic upgrade head`.
- **DHU-020 §E** — persistencia de "estado vigente del motor" autorizada como cambio estructural; diseño en §4 del SDD. ✔

## Project Structure

### Documentation (this feature)

```text
specs/001-cerebrovial-mvp/
├── plan.md          # Este archivo (mapeado del SDD)
├── data-model.md    # ✓ poblado (mapeado de SDD §4 + DATA_MODEL.md)
├── spec.md          # ✓ poblado 2026-05-20 (← backlog HU/RF/RNF: 21 HU + 22 RF + 53 RNF)
└── tasks.md         # ✓ poblado 2026-05-20 (← REPORTE_PLANIFICACION_SPRINT_4.md, inventario de 32 elementos)
```

### Source Code (repository root)

Estructura real verificada (SDD vista de desarrollo; convención DDD por capas
`domain` / `application` / `infrastructure` / `presentation`, **no uniforme**):

```text
core_management_api/      # núcleo FastAPI: predicción + control + consumo de visión
  src/
    prediction/           # DDD completo (domain/application/infrastructure/presentation)
    control/              # motor adaptativo — SIN capa infrastructure (calculadora sin estado)
                          #   application/: webster.py, max_pressure.py, mtc_constraints.py, adaptive_engine.py
                          #   presentation/api/: POST /control/recommend
edge_device/              # visión (YOLO11n + supervision); src/vision/ con DDD completo + tests
ia_prediction_service/    # entrenamiento offline del GRU (PyTorch), fuera del runtime
frontend_ui/              # SPA React 19 + TS + Vite + Tailwind + Leaflet
shared/                   # cerebrovial_shared: modelos ORM + utilidades (paquete pip local)
infra/                    # SQL de extensiones, configs
core_management_api/alembic/  # migraciones (env.py excluye tablas PostGIS)
```

**Structure Decision**: monolito modular (D-001). El módulo `control/` es la desviación
DDD conocida: hoy sin `infrastructure/` porque el motor es una calculadora sin estado; esa
capa aparecerá con la persistencia de `motor_decisions`/`engine_active_state` (§4 del SDD;
Delta-10).

## Complexity Tracking

| Decisión de diseño | Por qué se necesita | Alternativa más simple, descartada porque |
|---|---|---|
| `node_id` FK a `graph_nodes` con resolución en el write-path (DHU-021 V1) | Auditabilidad y anclaje al grafo de cada decisión del motor | String opaco sin FK: pierde trazabilidad al grafo; el motor hoy emite `intersection_id` sin restricción |
| Conservar `flow_total`/`y_load_factor`/`inputs_snapshot` capturados del cálculo interno (DHU-021 V2) | Reproducibilidad/auditabilidad de cada decisión (RNF-SEC-01) | Persistir solo lo que el endpoint serializa: decisión no reproducible |
| `motor_decisions` relacional, no hypertable (DHU-021 #11) | Volumen de intersección piloto no justifica TimescaleDB; FKs limpias | Hypertable desde el inicio: complejidad sin beneficio a esta escala |
| Fases en `jsonb`, no normalizadas (DHU-021 #10) | Fidelidad al sistema real (el motor recibe fases por payload) | Tabla hija de fases: inventa estructura que el código no produce |

## Estado y brecha

El estado de construcción por HU/TTH y los 13 deltas viven en SDD §10 (matriz) y §11
(brecha), con fuente en `AUDITORIA_HU_CODIGO.md`. Resumen: 1 completo, 5 parciales, 25 no
iniciados, 1 fuera de scope (≈25% del backlog vivo al 2026-05-18). El `tasks.md` derivará de
`REPORTE_PLANIFICACION_SPRINT_4.md`.
