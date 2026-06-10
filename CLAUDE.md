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
- **Modelo predictivo**: GRU (RNN, alineado al documento de tesis) como **baseline
  solo-temporal** (Fase 3, 375 nodos colapsados en batch). El **STGNN Time-then-Space**
  (DiffConv espacial sobre el grafo LCC de 375 nodos; **base B**: solo capas `tsl` —
  `NodeEmbedding`/`RNN`/`DiffConv`/`MLPDecoder`, sin engine `Predictor` ni Lightning) es
  **track activo de Fase 4**, comparado byte-a-byte contra el baseline GRU (mismo split,
  scaler, índice de ventanas, loss y métricas). El RandomForest actual es temporal hasta que
  el modelo neuronal esté servido.
- **Deploy**: docker local únicamente. No Azure por ahora.
- **Datos del GRU**: dataset generado por SUMO (D-008, 2026-05-11). Calibración
  contra Waze es trabajo futuro. `metr_la.h5` (LFS) se conserva sólo como
  referencia histórica pre-D-008. Ver `documentation/lean-inception/4-decisiones/DECISIONS.md`
  (decisiones D-001 a D-009).
- **Visión y BD**: el pipeline de visión **persistirá** agregados a una tabla
  `vision_aggregates` (pendiente, tareas E18-E21 / SAN-03). **Hoy** la persistencia
  es a CSV. No se migra a las tablas `vision_tracks`/`vision_flows` (modeladas para
  futuro, requieren refactor del pipeline). Ver D-006, D-007; el shape definitivo de
  `vision_aggregates` y el borrado de `vision_tracks`/`vision_flows` vía Alembic
  quedan fijados en DHU-024 (2026-05-27), ejecutables dentro del sprint de TTH-08.
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
  CERRADO por HU-05 (Sprint 4, fase 7)**: la cobertura ejecutable estaba
  completa por dos vías independientes — el interceptor + flash de sesión
  expirada en `httpClient.test.ts`, `SessionContext.test.tsx` y
  `LoginView.test.tsx` (CT-01.11); el rechazo 401 por token
  inválido/expirado sobre `GET /api/health` en los 4 escenarios pytest-bdd
  de CA-01.4. Lo que no era ejecutable en HU-01 era unir las dos mitades
  en un smoke e2e en vivo: el único endpoint con `require_role` era
  `/api/health`, que la UI no consume en su flujo natural. **HU-05 trajo
  `GET /control/active-state/{node_id}` con `require_role(OPERATOR, ADMIN)`
  y `ActiveStrategyView` la consume en su flujo natural** — esa es la ruta
  sobre la que se ejerce el smoke. Cobertura ejecutable del cierre:
  `features/hu-01-rbac/ca_01_6_token_expirado_endpoint_vivo.feature` +
  `core_management_api/tests/bdd/hu_05/test_ca_01_6_smoke.py` (backend:
  token con `exp` ya pasado devuelve 401 sobre `/control/active-state/larco_schell`),
  y `frontend_ui/src/services/__tests__/controlActiveStateService.test.ts`
  (frontend: 401 sobre la misma ruta dispara `authBridge.onUnauthorized`
  → `SessionProvider.performLogout({reason:'session-expired'})` →
  `navigate('/login')` con flash). Protocolo correcto: **no** usar
  `JWT_EXPIRATION_HOURS` fraccionario (el servicio JWT lee horas enteras
  vía `int(...)` y revienta con decimales); el camino válido es invocar
  `create_access_token(..., expires_delta=timedelta(seconds=N))` con `N`
  negativo para forjar un token ya expirado sin depender del wall-clock. Primer Gherkin del proyecto: `features/hu-01-rbac/`
  con `rbac_api.feature` ejecutable vía `pytest-bdd` (declarado en
  `requirements-dev.txt`; CI actualizado en `.github/workflows/ci.yml` para
  instalar `requirements-dev.txt`).

## Reglas para Claude Code

### Zonas que NO se tocan sin pedirlo
- **ThesisModal + su acceso en AppShell (rail ≥md / botón en topbar <md)** — documentación viva de la tesis
  (autores, objetivo, stack, KPIs). NO es parte de la arquitectura de control;
  NO marcar como componente huérfano ni proponer su remoción.
- **`CerebroVial/.gemini/settings.json`** — configuración intencional del flujo
  Gemini CLI del equipo. NO marcar como huérfano ni proponer remoción/`.gitignore`.

### Reglas levantadas (histórico)
- ~~`ia_prediction_service/src/models/time_then_space.py` — STGNN descartado, queda
  como referencia hasta que el GRU esté funcional.~~ **LEVANTADA (Fase 4, STGNN
  Time-then-Space, 2026-06-01).** El archivo pasa a ser **código vivo**: reescrito in-place
  como el STGNN base B (DiffConv espacial sobre el LCC de 375 nodos), comparado contra el
  baseline GRU. Deja de ser zona protegida. (El clúster tutorial Lightning que lo rodea —
  `src/training/predictor.py` + `scripts/train.py`/`predict.py`/`evaluate.py` — sigue muerto
  y desconectado; su limpieza es deuda diferida en `documentation/ESTADO_Y_PROXIMOS_PASOS.md`,
  disparador "limpieza del Dockerfile".)
- ~~`edge_device/src/vision/` — subsistema mejor armado, con tests reales.~~
  **LEVANTADA (TTH-08 Fase 2, 2026-05-27).** Fue zona protegida hasta este punto;
  TTH-08 reescribe el módulo de visión desde cero (DDD), por lo que la guarda
  queda sin efecto a partir de este commit —el primero del sprint TTH-08, que es
  exactamente donde DHU-024 §8 dispuso su levantamiento formal. Contexto en
  `documentation/lean-inception/4-decisiones/DECISIONS_HU.md` § DHU-024 decisión 8.

### Deuda técnica a respetar
- **No instalar `torch` ni `ultralytics` en `core_management_api` por defecto.**
  La visión YOLO vive en `edge_device/`; el entrenamiento del GRU vive en
  `ia_prediction_service/` (off-line) — el core no entrena. Esta regla sigue
  vigente como **guardia anti-regresión general**: cualquier HU futura que quiera
  meter una dependencia pesada en el core (o torch para algo distinto de la
  excepción de abajo) debe revisarse primero.
  *(Deuda C7.5 / SAN-01 cerrada 2026-05-26 en rama `san-06`.)*
  - **Excepción registrada (D-010, 2026-05-31):** se admite **`torch` CPU-only**
    en el core **exclusivamente para servir el predictor GRU de TTH-09**
    (inferencia in-process; la clase `GRUMultiOutput` se vendoriza en el core).
    No habilita entrenamiento en el core, ni torch en `cerebrovial_shared`, ni
    CUDA, ni `ultralytics`. El RandomForest baseline se preserva como respaldo
    Nivel 2. Justificación completa en
    `documentation/lean-inception/4-decisiones/DECISIONS.md` § D-010.
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

### Ubicación de documentos
Cada tipo de doc vive en su carpeta. No mezclar:

- **`documentation/handoffs/<tth-o-hu>/`** — handoffs de cierre de fase, uno por
  fase del sprint (p. ej. `documentation/handoffs/tth-08/tth-08-fase7-handoff.md`).
  Una sub-carpeta por feature/sprint.
- **`documentation/contracts/`** — contratos de módulo: shape de endpoints,
  schemas de tablas, semántica de campos, alcance honesto de validación. Documentos
  vivos del producto, no del proceso (p. ej.
  `documentation/contracts/vision_contract.md`).
- **`documentation/docs/`** — diseño técnico, lecciones de fase, discovery, planes
  históricos, schema canónico (`DATA_MODEL.md`), `TODO.md`. **NO handoffs sueltos,
  NO contratos sueltos**.
- **`documentation/lean-inception/`** — decisiones (`4-decisiones/DECISIONS.md`,
  `DECISIONS_HU.md`), planificación (`planificacion/ESTIMACION_SP.md`, etc.),
  contexto (`1-contexto/EVOLUCION_TESIS.md`, `LEAN_INCEPTION_CEREBROVIAL.md`).
- **`documentation/legacy/`** — referencias históricas (configs muertas, docs
  OBSOLETOS). Cada archivo movido a `legacy/` debe tener header explicando
  contexto, motivo del retiro y condición de reactivación.

### Flujo de trabajo (plan, commits, push, PR)
- **Plan-antes-de-ejecutar**: ante cualquier tarea no trivial, el agente primero
  hace auditoría read-only (grep, lectura de archivos, reporte de estado actual) y
  escribe un plan en `~/.claude/plans/` antes de tocar nada. ExitPlanMode con OK
  explícito del usuario antes de ejecutar. **Stage-gates** entre fases: el agente
  para entre fases largas (p. ej. F1 → F2) y reporta antes de seguir.
- **Commits granulares en español**, una unidad atómica por commit, **sin línea
  `Co-Authored-By:`** (decisión del usuario — historial limpio).
- **Push / PR — defecto**: el agente commitea + verifica end-to-end + **PARA**.
  **NO `git push`, NO `gh pr create`, NO merge** desde el agente.
- **Push / PR — solo con pedido explícito del usuario**:
  `git push origin <feature-branch>` (nunca `master`/`main`) y
  `gh pr create --body-file <handoff-de-cierre>` usando el handoff de cierre de
  la fase como cuerpo del PR. Reportar la URL al usuario.
- **Cuerpo del PR**: usar el handoff de cierre como base del `--body-file`. El
  handoff es la fuente de verdad técnica y debe estar libre de framing académico
  (`jurado`, `tribunal`, `defensa`, "documento de tesis" como audiencia,
  "el equipo verá") y libre de menciones a `Co-Authored-By`. El cuerpo del PR
  describe entregables, alcance, follow-ups técnicos y commits — nada más. Si
  el handoff todavía tiene framing, scrubealo en un commit antes de abrir el PR.
- **El cuerpo del PR se redacta en voz neutra de equipo**, no debe delatar cómo se
  produjo internamente. Aunque los handoffs se generan con asistencia de IA, el
  body del PR **no** menciona quién o qué hizo cada paso. Prohibido:
  - Atribuciones de autoría humano-vs-máquina: `IA`, `agente`, `el agente`,
    `humano`, `Claude`, `Codex`, `Gemini`, `asistente`, `co-pilot`.
  - Primera persona que delata al autor automático: `error mío`, `mi default`,
    `yo apliqué`, `me equivoqué`.
  - Framing "X [persona] revisó/corrigió mi error" que implica un autor-máquina
    supervisado. Si un catch o decisión importa, descríbelo en pasiva o en voz
    de equipo: *"se detectó que el default 8000 era incorrecto"*, no
    *"al revisar el plan se vio que mi default era error mío"*.
  - Líneas de proceso que asumen un agente: *"este handoff es el cuerpo del PR"*,
    *"lo corre <persona>"*, *"la decisión de mergear es humana"*. El "merge lo hace
    una persona" es una regla de este repo, no contenido del PR — no va en el body.
  No nombrar personas del equipo en el body (ni autor ni revisor): el trabajo se
  describe en voz de equipo, no atribuido a un individuo. Lo que se elimina es
  tanto el contraste autor-IA ↔ revisor-humano como la firma personal.

  Ejemplo (scrub de una celda de la tabla de catches):
  - ❌ *"Cesar al revisar el plan; default 8000 era error mío. Verificado contra
    `docker-compose.yml:37`."*
  - ✅ *"Al revisar el plan se detectó que el default 8000 era incorrecto.
    Verificado contra `docker-compose.yml:37`."*
- **Merge a `master`**: **siempre humano**, fuera del scope del agente, incluso
  con permiso. El agente no mergea.
- **Herramientas de pregunta vs Bash**: **NO usar `AskUserQuestion` en paralelo
  con `Bash`**. Si hay que preguntar al usuario, esperar primero a que los
  resultados de Bash estén disponibles, sintetizarlos y entonces preguntar.

## Estado del proyecto

Estado vivo y sprint en curso: `documentation/ESTADO_Y_PROXIMOS_PASOS.md`.
Plan técnico canónico: `documentation/sdd/SDD_CEREBROVIAL.md` (Spec Kit v0.8.11 brownfield).
Decisiones vigentes (D-001 a D-009): `documentation/lean-inception/4-decisiones/DECISIONS.md`.

**D-009** (variable predicha: jam level ordinal 0-5, constructo Waze) es lectura
obligatoria antes de tocar predicción o métricas de estado.

Fase 1 cerrada (`documentation/docs/20260503_PHASE1_CLOSURE.md`).
`documentation/docs/PLAN.md` es histórico, no operativo.

## Registro de cierres de rama

**Mantenimiento (hacia adelante):** de ahora en más, cada cierre de rama se documenta aquí
AL HACER EL MERGE, con la columna *Deuda/follow-ups* poblada. Las columnas `Rama`/`PR`/`Merge`/
`Fecha` son git-grounded; *Qué entregó* es un resumen breve; *Deuda/follow-ups* va **vacía en
las entradas históricas** (git no la registra — no se rellena retroactivamente con suposiciones).

| Rama | PR | Merge | Fecha | Qué entregó | Deuda/follow-ups |
|---|---|---|---|---|---|
| feature/net-miraflores-completo | #44 | `c73e3976` | 2026-06-03 | Net Miraflores v2 (1660/2948): reentrena GRU+STGNN sobre universo real, re-estratifica split, D-015 (STGNN gana en severo). | |
| feat/hu-23-recorrido-temporal-congestion | #43 | `81c463c4` | 2026-06-02 | HU-23: recorrido temporal — `/congestion/series` + modo histórico (slider, repintado por índice). | |
| feature/hu-22-mapa-congestion | #42 | `bd6f5b0a` | 2026-06-02 | HU-22: mapa de congestión operator-only (`CongestionMapView`, live + SSE; cableado tab/roles/Sidebar). | |
| feature/tth-12-congestion-aristas | #41 | `7ba5e2d3` | 2026-06-02 | TTH-12: infra de congestión por arista — `/congestion/{geometry,state,state/stream}` sobre graph_edges/waze_jams. | |
| fase-backlog-hu22 | #40 | `1c16a332` | 2026-06-02 | Precursor de HU-22 (backlog). | |
| feature/stgnn-corredor-larco | #39 | `3379066d` | 2026-06-02 | Track STGNN (investigación). | |
| feature/tth-09-gru-predictor | #38 | `7b89c85d` | 2026-05-31 | TTH-09: GRU 4-way servido (`POST /predictions/predict`, jam_level 0-5 × N/S/E/W × 30 pasos) + persistencia durable. | |
| feature/tth-11-hiperparametros-temporales | #37 | `9ec02649` | 2026-05-31 | TTH-11: spike hiperparámetros temporales. | |
| feature/corredor-larco-mp-red | #36 | `fe3cd9bd` | 2026-05-30 | Corredor Larco: MP-red (investigación, descartado). | |
| feature/tth-07 | #35 | `77426705` | 2026-05-29 | TTH-07: integración SUMO. | |
| feature/tth-08-fase* | #22–#34 | varios | 2026-05-27→29 | TTH-08: visión computacional (DDD). Cerrado en 13 PRs consecutivos por fase (fase0-auditoría → fase9-docs). | |
| docs/dhu-023-024-encuadre-hu07-tth08 | #21 | `ab017de7` | 2026-05-27 | Docs DHU-023/024 (encuadre HU-07/TTH-08). | |
| feature/hu-05-vista-pasiva-estrategia | #20 | `af8cd5b6` | 2026-05-27 | HU-05: vista pasiva de estrategia activa (`/control/active-state` + SSE, `ActiveStrategyView`); cierre cruzado CA-01.6. | |
| san-06-purge-torch-core | #19 | `ec948502` | 2026-05-26 | SAN-01: purga de torch muerto del core. | |
| feat/hu-01-rbac | #18 | `2a3a2441` | 2026-05-26 | HU-01: maquinaria RBAC — `require_role` (backend) + `RoleGate`/`roles.ts` (frontend), primer Gherkin pytest-bdd. | |
| san-05 | #17 | `81c482ba` | 2026-05-26 | Saneamiento. | |
| feat/tth-01-frontend-login | #16 | `8c7c626c` | 2026-05-26 | TTH-01: login frontend + manejo de sesión (CT-01.6–01.13). | |
| feat/tth-01-auth-jwt | #15 | `534c228f` | 2026-05-26 | TTH-01: backend auth — JWT (jose HS256) + bcrypt (cost 12), `POST /auth/login`, `get_current_user`. | |
| chore/saneamiento-documental | #14 | `4214ff8f` | 2026-05-25 | Saneamiento documental. | |
| fix/consolidar-decisiones | #13 | `954e2cd8` | 2026-05-25 | SAN-04: consolidación de DECISIONS. | |
| chore/orden-repo | #12 | `d3994e22` | 2026-05-25 | Orden del repo (limpieza ligera). | |
| feature/SDD | #11 | `5085b595` | 2026-05-25 | Adopción Spec Kit v0.8.11 brownfield (DHU-021). | |
| inception-agile | #10 | `20106a5d` | 2026-05-19 | Lean Inception / backlog ágil. | |
| fase-2-* (fork AndresBR2003/) | #2–#9 | varios | 2026-05-03→09 | Fase 2 (cimientos, alembic, motor, frontend-ci). **Detalle no derivable desde el log de merges.** | |

**Cierres NO rastreables por PR (incluidos con salvedad):**
- **TTH-10 (cierre parcial)** — commit directo `6df521e4` (2026-05-26), **SIN PR**: persistencia + config + health.
- **Fase 0** — `3f1bacef` (2026-05-02), merge sin PR (`analysis/initial-discovery`).
- **PRs #2–#9** — del fork `AndresBR2003/`; el detalle de entrega no es derivable desde el log de merges.

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

## Rediseño UI (feature/rediseno-ui)
- Spec visual: design/cerebrovial-ui-concept.html. Antes de construir cualquier vista o
  componente, leer la sección equivalente del prototipo y calcar estética e interacciones.
- Mobile-first OBLIGATORIO: primero layout 390px, después md:/lg:. Rail solo ≥md; <md usa
  bottom-nav. Drawers = sheet de pantalla completa en <md. Prohibido h-[calc(100vh-*)]:
  alturas por flex/grid con min-h-0 y las vars --h-topbar/--w-rail/--h-bottomnav.
- Tokens: todo color/radio/animación sale de src/styles/tokens.css (@theme). Prohibido hex
  inline y paleta slate/indigo default en código nuevo. Única escala de estado: ok/warn/bad/
  sev/info.
- Reusar sin reescribir: services/*, auth/* (SessionContext, RoleGate, roles, authBridge),
  utils/* (congestion, markerVisual, trafficLabels), HlsPlayer, TrafficLightCycle, Slider,
  TimingBar, types/*.
- Datos: todo acceso al core vía httpClient (JWT). Hooks de datos en src/hooks/. Lo que no
  tenga backend real se mockea SEÑALIZADO con badge "Demo · datos simulados" (no verde con
  asterisco). El edge (EventSource y POST /cameras) queda como está: su auth es deuda backend.
- ThesisModal y su acceso en la navegación: zona protegida. Migra, nunca se elimina.
- Tests: componente nuevo = test nuevo. Mock de react-leaflet y stubs (IntersectionObserver,
  EventSource) globales en setupTests.ts.
- Disciplina de fases: plan mode antes de ejecutar; commit por fase completa validada, en
  español, sin Co-Authored-By; PARAR al cerrar cada fase; sin push/PR sin pedido explícito.
