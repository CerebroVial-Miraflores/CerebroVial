# CerebroVial — Sistema de gestión de tráfico Miraflores

## Contexto
Proyecto de tesis. Sistema predictivo de tráfico para el distrito de Miraflores 
(Lima, Perú). Detección por visión computacional + predicción GRU + control 
adaptativo de semáforos.

## Arquitectura
**Monolito modular**, organizado en carpetas que sugieren microservicios pero 
NO lo son. Todas las carpetas se entienden como módulos del mismo sistema, 
desplegado como un único proceso FastAPI.

```
core_management_api/   # módulo: API + predicción + control + common
edge_device/           # módulo: visión computacional (YOLO + tracking + SSE)
ia_prediction_service/ # módulo: entrenamiento del modelo GRU (offline)
frontend_ui/           # SPA React (proceso separado)
shared/                # paquete pip-instalable cerebrovial_shared
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

## Decisiones tomadas
- **Arquitectura**: monolito modular, NO microservicios. Las carpetas separadas 
  son herencia de cuando había 3 repos.
- **Modelo predictivo**: GRU (RNN, alineado al documento de tesis). El STGNN 
  explorado se descarta. El RandomForest actual es temporal hasta que el 
  GRU esté servido.
- **Deploy**: docker local únicamente. No Azure por ahora.
- **Datos del GRU**: dataset sintético calibrado contra distribuciones de Waze 
  + METR-LA (`metr_la.h5` ya en LFS). Sin acceso a API real de Waze. 
  Ver `DATA_MODEL_AUDIT.md` decisión D-008.
- **Visión y BD**: el pipeline de visión persiste agregados a una tabla 
  nueva `vision_aggregates` (no a las tablas `vision_tracks`/`vision_flows` 
  modeladas que requieren refactor del pipeline). Ver D-006, D-007.

## Reglas para Claude Code
- NO refactorizar `edge_device/src/vision/`. Es el subsistema mejor armado y 
  con tests reales. Tocar sólo cuando se pida explícitamente.
- NO modificar `ia_prediction_service/src/models/time_then_space.py`. El STGNN
  se descarta pero el código queda como referencia hasta que el GRU esté 
  funcional. En `notebooks/logs/` solo queda `epoch=79-step=30800.ckpt` 
  (en LFS); los 4 checkpoints intermedios se borraron en C9.
- NO instalar `torch` ni `ultralytics` en `core_management_api`. Los 
  `requirements.txt` están limpios desde C7.
- NO migrar el pipeline de visión a las tablas `vision_tracks` / 
  `vision_flows`. Esas tablas quedan modeladas para futuro pero sin 
  refactor del pipeline. La persistencia a BD se hace vía 
  `vision_aggregates` (tareas E18-E21 del TODO).
- Cuando agregues un endpoint, ubicarlo en el módulo correspondiente y 
  registrarlo en el router unificado de `core_management_api`.
- Antes de cualquier cambio estructural (mover carpetas, renombrar paquetes, 
  cambiar el modelo de la BD), parar y preguntar al usuario.
- Para cambios al modelo predictivo, leer primero el documento de tesis en 
  `documentation/tesis/`.
- **Migraciones de BD**: siempre con Alembic. Nunca usar `Base.metadata.create_all()`. 
  Para entender el schema actual, leer `documentation/docs/DATA_MODEL.md`.

## Estado del proyecto

**Fase 1 (Estabilización del repo):** ✓ Cerrada el 2026-05-03.
Ver `documentation/docs/20260503_PHASE1_CLOSURE.md` para detalle.

**Auditoría de modelo de datos:** ✓ Completada el 2026-05-03.
Ver `documentation/docs/DATA_MODEL_AUDIT.md` y `DATA_MODEL.md`.

**Fase 2 (Cimientos: Alembic, JWT, frontend conectado, persistencia visión→BD):**
en curso.

**Fase 3 (Predictor GRU + control adaptativo + métricas comparativas):**
después de Fase 2.

Detalle en `documentation/docs/PLAN.md`.

## Git LFS (requerido)
Este repo usa Git LFS para binarios (.joblib, .pt, .ckpt, .h5, .npy, .docx).
Antes de clonar o pull, instalar git-lfs y configurarlo:

  brew install git-lfs   # macOS
  # o: apt install git-lfs   # Linux Debian/Ubuntu
  git lfs install

Sin LFS, los archivos binarios van a venir como pointers de texto y
`docker compose up` va a fallar al cargar modelos.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->
