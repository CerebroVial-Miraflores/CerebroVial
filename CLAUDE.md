# CerebroVial — Sistema de gestión de tráfico Miraflores

> 📍 **Estado actual y próximos pasos:** ver `documentation/ESTADO_Y_PROXIMOS_PASOS.md`.

## Contexto
Proyecto de tesis. Sistema predictivo de tráfico para el distrito de Miraflores 
(Lima, Perú). Detección por visión computacional + predicción GRU + control 
adaptativo de semáforos.

## Arquitectura
**Monolito modular**, organizado en carpetas que sugieren microservicios pero 
NO lo son. Todas las carpetas se entienden como módulos del mismo sistema, 
desplegado como un único proceso FastAPI.

```
core_management_api/   # módulo: API + predicción + control
edge_device/           # módulo: visión computacional (YOLO + tracking + SSE)
ia_prediction_service/ # módulo: entrenamiento del modelo GRU (offline)
frontend_ui/           # SPA React (proceso separado)
shared/                # paquete pip-instalable cerebrovial_shared (código transversal,
                       # instalado como dependencia en cada módulo; reemplaza al ex-`common/`)
infra/                 # SQL, configs de infra
```

## Stack
Backend: Python 3.11, FastAPI, SQLAlchemy + GeoAlchemy2 + Alembic, 
PostgreSQL + PostGIS + TimescaleDB. Vision: ultralytics YOLO + supervision + 
opencv. ML: PyTorch + PyTorch Lightning (modelo GRU). Frontend: React 19 + 
TypeScript + Vite + Tailwind 4 + Leaflet.

## Cómo levantar el proyecto

El repo usa `invoke` (gestor de tareas en Python) para envolver los
comandos frecuentes. Ver lista completa con `invoke --list`.

Setup primer-uso:
1. Tener instalados los prerequisitos (ver README.md raíz).
2. `cp .env.example .env` y completar valores.
3. `pip install invoke`
4. `invoke setup-dev` (crea venv local con deps de dev)
5. `invoke up` (las tablas se crean solas — alembic corre en el entrypoint del core)
6. `invoke seed` (carga datos iniciales de Miraflores)

Día a día:
- `invoke up` / `invoke down` / `invoke logs` / `invoke ps` / `invoke test`
- `invoke up --service=<nombre>` para levantar un servicio suelto
- `invoke up-build` para rebuildear imágenes (con cache); acepta `--service=<nombre>`.
  Necesario cuando cambia código que se compila al imagen (p.ej. el frontend,
  servido como build estático con nginx) o cambian Dockerfile/requirements.
- `invoke up-dev` para hot-reload del core (usa docker-compose.dev.yml).
  En este modo alembic NO corre automáticamente; usar `invoke migrate` a mano.
- `invoke migrate` después de un git pull con migraciones nuevas (sin rebuild)
- `invoke db-reset` cuando el schema cambió de forma incompatible
- `invoke shell-api` / `invoke shell-db` para debugging interactivo

NO usar `docker compose ...` directo — `invoke` agrega validaciones
(check de LFS, .env presente) que evitan errores crípticos.

Migraciones:
- El entrypoint de core_management_api ([core_management_api/entrypoint.sh](core_management_api/entrypoint.sh))
  corre `alembic upgrade head` antes de uvicorn. Apto para single-node.
  En multi-instance habría que sacar las migraciones a un job separado.
- `invoke up-dev` cancela ese entrypoint (override de docker-compose.dev.yml),
  por eso ahí hay que migrar a mano con `invoke migrate`.

## Tests y calidad

- **Backend**: pytest en `core_management_api/tests/` y `edge_device/tests/`.
  Correr con `invoke test` (requiere venv activo de `invoke setup-dev`).
- **Lint Python**: `ruff check .` desde la raíz. Config en `pyproject.toml`.
- **Frontend**: `cd frontend_ui && npm run lint && npm run test -- --run`
  (ESLint + Vitest).
- **CI**: [.github/workflows/ci.yml](.github/workflows/ci.yml) corre en push a
  `master` y `fase-*`, y en PRs a `master`. Tres jobs: backend (ruff + pytest),
  frontend (lint + test), y `docker compose build`. Lo que rompa CI rompe el merge.
- No hay pre-commit hooks configurados.

## Frontend (rápido)

- Stack: React 19 + TS + Vite + Tailwind 4 + Leaflet.
- Dev con HMR: `cd frontend_ui && npm run dev` (puerto 5173).
- En docker se sirve como build estático con nginx — por eso requiere
  `invoke up-build --service=frontend` cuando cambia el código.
- Tests: Vitest (`npm run test`). Lint: ESLint (`npm run lint`).

## Decisiones tomadas
- **Arquitectura**: monolito modular, NO microservicios. Las carpetas separadas 
  son herencia de cuando había 3 repos.
- **Modelo predictivo**: GRU (RNN, alineado al documento de tesis). El STGNN 
  explorado se descarta. El RandomForest actual es temporal hasta que el 
  GRU esté servido.
- **Deploy**: docker local únicamente. No Azure por ahora.
- **Datos del GRU**: dataset generado por SUMO (D-008, 2026-05-11). Calibración
  contra Waze es trabajo futuro. `metr_la.h5` (LFS) se conserva sólo como
  referencia histórica pre-D-008. Ver `documentation/lean-inception/4-decisiones/DECISIONS.md`
  (decisiones D-001 a D-009).
- **Visión y BD**: el pipeline de visión **persistirá** agregados a una tabla
  `vision_aggregates` (pendiente, tareas E18-E21 / SAN-03). **Hoy** la persistencia
  es a CSV. No se migra a las tablas `vision_tracks`/`vision_flows` (modeladas para
  futuro, requieren refactor del pipeline). Ver D-006, D-007.
- **Spec Kit (DHU-021)**: el proyecto adoptó Spec Kit v0.8.11 brownfield. Constitución
  en `.specify/memory/constitution.md`; artefactos vivos en `specs/001-cerebrovial-mvp/`
  (spec.md, plan.md, tasks.md, data-model.md, quickstart.md). Mapeo de adopción en
  `documentation/sdd/SPECKIT_MAPPING.md`.
- **HU-01 (RBAC) — alcance entregado y deuda declarada**. La HU instaura la
  **maquinaria** RBAC. Backend: dependency `require_role(*allowed: Role)` en
  `core_management_api/src/auth/presentation/api/dependencies.py`, con cuerpo 403
  genérico `"Acceso denegado"` byte-idéntico para no filtrar el recurso ni el rol
  esperado (RNF-SEC-04). Frontend: `RoleGate` + `roles.ts` (mapas TABS_BY_ROLE,
  DEFAULT_TAB_BY_ROLE, ROLE_LABEL_ES) en `frontend_ui/src/auth/`. El enforcement
  se demuestra sobre un único endpoint de muestra (`GET /api/health` con
  `require_role(Role.ADMIN)`). Esto es consistente con la fila *Aplicabilidad*
  de RNF-SEC-03 (REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md): *"la matriz endpoint
  × rol se materializa al implementar y se valida con prueba automatizada
  exhaustiva"*. La cobertura endpoint × rol sobre las rutas restantes
  (`/api/intersections`, `/predictions/*`, `/control/*`) es **responsabilidad
  acumulativa** de HU-15..HU-21 (cada HU futura aplica `require_role(...)` a sus
  endpoints). **CA-01.6 (auto-logout por token expirado) — smoke e2e en vivo
  diferido**: la cobertura ejecutable está completa por dos vías independientes —
  el interceptor + flash de sesión expirada se verifican en `httpClient.test.ts`,
  `SessionContext.test.tsx` y `LoginView.test.tsx` (CT-01.11); el rechazo 401
  por token inválido/expirado se verifica sobre `GET /api/health` en los 4
  escenarios pytest-bdd de CA-01.4. Lo que **no** es ejecutable en HU-01 es
  unir las dos mitades en un smoke e2e en vivo: el único endpoint protegido
  por `require_role` es `/api/health`, que la UI no consume en su flujo
  natural, así que dejar expirar el token y "seguir usando la app" no dispara
  el auto-logout porque las demás rutas siguen abiertas (la misma frontera
  de RNF-SEC-03 ya declarada arriba). El smoke e2e en vivo queda diferido a
  la primera HU que aplique `require_role` a una ruta consumida por la UI
  (HU-03 o HU-05, lo que llegue primero). Cuando llegue, el protocolo correcto
  **no** es `JWT_EXPIRATION_HOURS` fraccionario (el servicio JWT lee horas
  enteras vía `int(...)` y revienta con decimales); el camino válido es
  invocar `create_access_token(..., expires_delta=timedelta(seconds=N))` desde
  una shell del backend. Primer Gherkin del proyecto: `features/hu-01-rbac/`
  con `rbac_api.feature` ejecutable vía `pytest-bdd` (declarado en
  `requirements-dev.txt`; CI actualizado en `.github/workflows/ci.yml` para
  instalar `requirements-dev.txt`).

## Reglas para Claude Code

### Zonas que NO se tocan sin pedirlo
- `edge_device/src/vision/` — subsistema mejor armado, con tests reales.
- `ia_prediction_service/src/models/time_then_space.py` — STGNN descartado, queda
  como referencia hasta que el GRU esté funcional. En `notebooks/logs/` solo queda
  `epoch=79-step=30800.ckpt` (LFS); los 4 intermedios se borraron en C9.
  <!-- TODO: verificar con `git lfs ls-files | grep epoch=79` si el ckpt sigue presente -->
- **ThesisModal + su botón en `Sidebar.tsx`** — documentación viva de la tesis
  (autores, objetivo, stack, KPIs). NO es parte de la arquitectura de control;
  NO marcar como componente huérfano ni proponer su remoción.
- **`CerebroVial/.gemini/settings.json`** — configuración intencional del flujo
  Gemini CLI del equipo. NO marcar como huérfano ni proponer remoción/`.gitignore`.

### Deuda técnica a respetar
- **No instalar `torch` ni `ultralytics` en `core_management_api`**. El endpoint
  vivo de predicción usa RandomForest baseline; el GRU vive en
  `ia_prediction_service/` (entrenamiento off-line), no en el backend. La
  visión YOLO vive en `edge_device/`. Esta regla aplica anti-regresión —
  cualquier HU futura que necesite torch en el core debe revisarse primero.
  *(Deuda C7.5 / SAN-01 cerrada 2026-05-26 en rama `san-06`.)*
- No migrar el pipeline de visión a `vision_tracks` / `vision_flows`
  (ver D-006/D-007 en "Decisiones tomadas" arriba).

### Convenciones
- Endpoints nuevos: ubicarlos en el módulo correspondiente y registrarlos en el
  router unificado de `core_management_api`.
- Cambios estructurales (mover carpetas, renombrar paquetes, cambiar el modelo
  de la BD): parar y preguntar al usuario.
- Cambios al modelo predictivo: leer primero el documento de tesis en
  `documentation/tesis/`.
- **Migraciones de BD**: siempre con Alembic. Nunca usar
  `Base.metadata.create_all()`. Schema actual en `documentation/docs/DATA_MODEL.md`.

## Estado del proyecto

Estado vivo y sprint en curso: `documentation/ESTADO_Y_PROXIMOS_PASOS.md`.
Plan técnico canónico: `documentation/sdd/SDD_CEREBROVIAL.md` (Spec Kit v0.8.11 brownfield).
Decisiones vigentes (D-001 a D-009): `documentation/lean-inception/4-decisiones/DECISIONS.md`.

**D-009** (variable predicha: jam level ordinal 0-5, constructo Waze) es lectura
obligatoria antes de tocar predicción o métricas de estado.

Fase 1 cerrada (`documentation/docs/20260503_PHASE1_CLOSURE.md`).
`documentation/docs/PLAN.md` es histórico, no operativo.

## Git LFS (requerido)
Este repo usa Git LFS para binarios (.joblib, .pt, .ckpt, .h5, .npy, .docx).
Antes de clonar o pull, instalar git-lfs y configurarlo:

  brew install git-lfs   # macOS
  # o: apt install git-lfs   # Linux Debian/Ubuntu
  git lfs install

Sin LFS, los archivos binarios van a venir como pointers de texto y
`docker compose up` va a fallar al cargar modelos.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure, shell commands, and other
important information, read the current plan at `specs/001-cerebrovial-mvp/plan.md`. Related artifacts:
- `specs/001-cerebrovial-mvp/spec.md` (feature specification)
- `specs/001-cerebrovial-mvp/tasks.md` (dependency-ordered tasks)
- `.specify/memory/constitution.md` (project constitution)
<!-- SPECKIT END -->
