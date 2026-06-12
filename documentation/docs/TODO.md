BLOQUE A — Decisiones y comunicación (no es código, es lo más importante)

 [x]A1. Llamar a tu compañero. Mostrarle el assessment, este plan, las decisiones sensibles. Acordar reparto de trabajo: vos código, él documento de tesis (recomendado).
 [x]A2. Llamar al asesor de tesis. Tres preguntas concretas: (1) ¿podemos actualizar los números 88.2% / 81.3% / <2s tras validación real?; (2) ¿es aceptable demostrar "arquitectura desplegable en Pi" sin entregar Pi física?; (3) ¿demo local con plan de deploy en Azure cubre la "arquitectura híbrida" del documento?
 [x]A3. Decidir formalmente: el modelo es GRU (una arquitectura RNN). Documentarlo en un commit de tesis o en docs/DECISIONS.md.
 [x]A4. Acordar con el compañero el refinamiento del backlog (mover HU04 al Sprint 2, reducir alcance de HU07, distribuir SP de forma pareja).


BLOQUE B — Fase 0: Documentación honesta del estado actual (objetivo: lunes 4)

 [x] B1. Mergear analysis/initial-discovery a main con el assessment incluido como docs/ARCHITECTURE_DISCOVERY.md.
 [x] B2. Mover el CLAUDE.md antiguo del compañero a docs/SPEC.md (es la especificación target, no el estado actual).
 [x] B3. Crear el nuevo CLAUDE.md raíz con el contenido del mensaje anterior, ajustado a las decisiones tomadas (monolito modular, GRU, docker local).
 [x] B4. Crear docs/PLAN.md con las 4 fases del plan que te pasé.
 [x] B5. Crear docs/TODO.md con esta misma lista para que sea trackeable.
 [x] B6. Crear docs/DECISIONS.md con las 5 decisiones tomadas (monolito, GRU, local docker, Pi conceptual, números reales tras validación). Cada una con fecha y justificación de una línea.
 [x] B7. Crear docs/BACKLOG_V2.md con el backlog refinado (resultado de A4).
 [x] B8. Verificar que .env esté en .gitignore. Si está versionado, rotar credenciales y agregarlo al ignore.
 [x] B9. Agregar al .gitignore: .claude/settings.local.json, .claude/projects/, tmp_*.txt, tmp_*.py, *.docx excepto los que estén en documentation/tesis/.
 [x] B10. Borrar CLAUDE.md.old (ya no es necesario, el contenido vive en docs/SPEC.md).
 [x] B11. Commit final del bloque B con mensaje [Fase 0] Documentación de estado y plan.


BLOQUE C — Fase 1: Estabilización del repo (objetivo: lunes 4 o domingo 3)
Cada ítem es una sesión separada de Claude Code, en plan mode primero.

 [x] C1. Consolidar common/ en un solo lugar: crear shared/ (o cerebrovial_common/) en la raíz con pyproject.toml mínimo, mover el contenido de core_management_api/src/common/ ahí, instalar como paquete pip local en ambos servicios (pip install -e ../shared), borrar el common/ duplicado de edge_device/. Verificar que tests siguen pasando.
 [x] C1.1. Resolver duplicación de setup_logger: definida en logging.py (con param `level`) y en utils.py (INFO hardcoded). Decidir cuál queda, eliminar la otra, actualizar imports.
 [x] C1.4 — Marcar `test_pipeline_processing_flow` como xfail (deuda preexistente). Race condition confirmada en commit 0e20b0b4. Decorador aplicado, tracked como C1.5.
 [~] C1.6 — **OBSOLETA por TTH-08 F4b (2026-05-29).** El refactor desde cero reemplazó la capa `application/` que contenía `MultiCameraManager`; el `CameraManager` nuevo vive en `edge_device/src/vision/application/services/camera_manager.py`. Los tests legacy en `edge_device/tests/vision/unit/test_multi_camera_manager.py` apuntan a una API extinta — el archivo queda como pieza muerta nominada a **F9.y** (barrido de tests legacy huérfanos, sub-fase de cleanup post-F9). No se borra en F9 (es código productivo, fuera del alcance de F9 = solo documentación). **Cross-ref:** TODO.md F9.y; `tth-08-fase9-handoff.md` §[backlog post-TTH-08]; DECISIONS_HU.md addendum F9 a DHU-024.
 [~] C1.7 — **OBSOLETA por TTH-08 (DHU-024 §3).** DHU-024 explícitamente declaró `SmartDetectionProcessor.get_analysis_for_frame()` como lógica muerta que el refactor descarta. Los tests del processor ya fueron eliminados (no xfailed) cuando se confirmó la obsolescencia. El source `edge_device/src/vision/application/processors/smart_detection.py` sigue presente pero sin consumidor runtime — pieza muerta nominada a **F9.y** (barrido) junto con C1.6. No se borra en F9 (fuera de alcance). **Cross-ref:** TODO.md F9.y; `tth-08-fase9-handoff.md` §[backlog post-TTH-08].
 [x] C1.8 — **RESUELTA por TTH-08 F4a + DHU-025 (2026-05-28).** F4a preservó el `ZoneCounter` en el dominio y lo extendió con `mean_occupancy` (DHU-025 abrió formalmente la divergencia: el Protocol cerrado en Fase 3 no contenía la info necesaria para computar occupancy; F4a lo amplió). Test verde en `edge_device/tests/vision/unit/test_zone_counter_basic.py`. CT-08.2 cubierto.
 [x] C1.5 — **RESUELTA por TTH-08 F5c.** La capa `application/` reescrita rediseñó la concurrencia del pipeline (`AsyncVisionPipeline` reemplazado por nuevo contrato `FrameProducer.read()` que evita la race del `finally: pipeline.stop()`). Test verde en `edge_device/tests/vision/unit/test_async_pipeline.py:101` (`test_pipeline_processing_flow_drains_all_frames_C1_5`). xfail original retirado.
 [x] C2. Crear core_management_api/src/main.py como entry point real de FastAPI con routers de prediction y control montados. Actualizar el Dockerfile de core_management_api para que apunte a src.main:app. Mantener scripts/run_prediction.py como entry alternativo de dev.
 [x] C3. Sacar ia_prediction_service del docker-compose.yml. Documentar en su README cómo correrlo manualmente para entrenar.
 [x] C4. Renombrar el servicio compose db_postgres → db (o cambiar el .env para que use db_postgres). Lo que sea menos invasivo.
 [x] C5. Sacar db_mongo del docker-compose.yml. Documentar en docs/DECISIONS.md que MongoDB se reemplaza por PostgreSQL+TimescaleDB para todo (incluyendo logs).
 [x] C6. Sacar api_gateway del docker-compose.yml. Borrar la referencia al directorio inexistente.
 [x] C7. Limpiar core_management_api/requirements.txt: sacar torch, ultralytics, opencv-python, supervision, hydra-core, cap_from_youtube, imageio-ffmpeg, streamlink, shapely. Mantener fastapi, uvicorn, sqlalchemy, geoalchemy2, alembic, numpy, pandas, scikit-learn, psycopg2-binary, sse-starlette, python-jose[cryptography], passlib[bcrypt], httpx, python-multipart.
 [x] **C7.5** — ✓ resuelta (2026-05-26, rama `san-06`): purgado el código STGNN muerto de `core_management_api/src/prediction/` (6 archivos: `domain.py`, `infrastructure/{models,graph_builder,data_loader,repository}.py`, `application/builder.py`) y eliminado `torch` de `core_management_api/requirements.txt`. El predictor vivo (`predictor.py → engine.py`) sigue intacto sobre joblib + sklearn RandomForest. Cierra simultáneamente SAN-01 (regla CLAUDE.md vs requirements). La prohibición de torch en core permanece como guardia anti-regresión.
 [ ] **C7.6** — **REABIERTA como F9.z al cierre de TTH-08 F9 (2026-05-29).** DHU-024 §7 declaró que C7.6 se cerraría dentro del refactor "al reescribir `edge_device/requirements.txt` desde cero con `--index-url https://download.pytorch.org/whl/cpu`". Auditoría F9 confirmó que **NO se aplicó**: `edge_device/requirements.txt:5` sigue siendo `torch` sin --index-url cpu. El addendum F9 al pie de DHU-024 §7 lo reconoce honestamente y reabre la deuda. Cierre actual = no es cierre, sino una sub-fase de infra separable post-F9 que aplique el pin CPU (~200MB vs ~2GB CUDA) + smoke build/import. Aplicación bloqueada por la restricción dura de F9 ("cero código productivo, cero tests nuevos") — no cabe en 0.5 SP. **Restricciones a verificar antes de tocar (heredadas del item original)**: (1) `edge_device` usa YOLO/ultralytics — confirmar que la variante CPU-only basta para la inferencia esperada del demo; (2) `ia_prediction_service/requirements.txt:1,6-8` (`torch>=2.3.0`, `torch-geometric`, `torch-scatter`, `torch-sparse`) sigue como deuda independiente fuera de TTH-08 (re-evaluar al definir TTH-09/GRU). **Cross-ref:** DECISIONS_HU.md addendum F9 a DHU-024 §7; vision_contract.md §7; tth-08-fase9-handoff.md §[backlog post-TTH-08].
 [ ] **C7.7** — Deuda de empaquetado anotada en TTH-08 Fase 5a (2026-05-28), **no ejecutar todavía**: `edge_device/requirements.txt` declara `opencv-python` (variante con bindings de GUI, arrastra Qt y X11). Pero el `edge_device` de producción corre headless (sin display, contenedor sin servidor X) — esos bindings son peso muerto en la imagen final. La variante `opencv-python-headless` (~46 MB, sin GUI) sería suficiente para inferencia, tracking y geometría de zonas. **Restricciones a verificar antes de tocar**: (1) `edge_device/src/vision/infrastructure/interaction.py` (script de calibración de zonas) usa GUI de cv2 (`cv2.namedWindow`, `cv2.imshow`, `cv2.setMouseCallback`) — si se migra a headless, ese script deja de funcionar y hay que (a) sacarlo del paquete principal, (b) instalarlo en un venv separado del developer, o (c) reescribirlo sin GUI; (2) Dockerfile de `edge_device` puede contener pasos para instalar libs de sistema (libgl1, libglib2.0-0) que se vuelven innecesarios al migrar a headless — auditar y limpiar; (3) cualquier CI job que ejercite GUI (no hay hoy, pero verificar) se rompería. Prioridad baja — la divergencia local (`opencv-python-headless` instalada en `.venv` para correr tests, `opencv-python` declarado en `requirements.txt` para producción) es inofensiva mientras ningún test toque GUI, y al 2026-05-28 ningún test lo hace. **Trazabilidad:** detectada al instalar localmente la variante headless para correr los tests de occupancy de DHU-025 en Fase 5a de TTH-08; el venv local quedó con headless, `requirements.txt` no se modificó.
 [x] C8. Limpieza de raíz: mover evidence_report.md, diagrama_vial*.html, DOCUMENTACION.md a documentation/. Borrar tmp_docx_output.txt, tmp_docx_utf8.txt, tmp_read_docx.py, generate_evidence.py.
 [x] C9. Configurar Git LFS para binarios + limpiar checkpoints intermedios. LFS aplicado a 13 binarios (.joblib, .pt, .ckpt, .h5, .npy, .docx). Borrados 4 .ckpt intermedios del STGNN (~48 MB). Solo queda epoch=79-step=30800.ckpt como referencia.
 [~] **C9.5** — ~~Migración de metr_la.h5 a download-on-demand~~ **NO APLICA**: D-008 confirma que `metr_la.h5` se mantiene en LFS como input del pipeline de calibración del dataset sintético del GRU.
 [x] C10. Verificar que docker compose up levanta db, core_management_api y edge_device sin crashes. Frontend con npm run dev puede llamar a los endpoints existentes.
 [x] C11. Crear un Makefile o tasks.py raíz con comandos: make up, make down, make test, make lint. Trivial pero ahorra mucho tiempo a futuro.
 [x] C12. Commit final del bloque C: [Fase 1] Estabilización: docker compose up funciona end-to-end.
 [ ] **C9.6** — Validación al arranque: detectar binarios LFS como pointers en lugar de archivos reales y fallar con mensaje claro si git-lfs no está instalado. Hoy si un dev clona sin git-lfs, los modelos `.joblib` y `yolo11n.pt` vienen como punteros de texto y `joblib.load()` / torch fallan con errores crípticos (`UnpicklingError: invalid load key, 'v'.`). Implementar check al load del modelo en `core_management_api` (predictor) y `edge_device` (yolo). Si el archivo empieza con `version https://git-lfs...`, fallar con mensaje accionable que apunte a CLAUDE.md sección "Git LFS (requerido)". Prioridad media — no bloquea hoy pero ahorra horas de soporte futuro.
 [ ] **C10.1** — Setup de dev: ~~documentar cómo correr tests fuera del container~~. **Parcialmente resuelto en C11**: `invoke setup-dev` automatiza la creación del venv y la instalación de pytest desde `requirements-dev.txt`. Pendiente: si en algún momento se decide correr tests dentro del container Docker, agregar un stage de build separado al Dockerfile. Por ahora, el flujo dev está cubierto. Prioridad: baja.
 [x] **C10.2** — Vulnerabilidades npm en frontend_ui. `npm audit fix` conservador aplicado. Antes: 10 vulns (4 moderate + 6 high). Después: 0 vulnerabilidades. Solo `package-lock.json` modificado (291 ins / 243 del). Node v24.11.0 / npm 11.6.1. Sin `.nvmrc` (ver C10.2.2). Build no regresó — errores pre-existentes (ver C10.2.1).
 [ ] **C10.2.1** — Build del frontend falla con errores TypeScript pre-existentes (no introducidos por C10.2). Tres tipos de error: (1) TS6133: imports `React` no usados en ~9 archivos .tsx (React 19 usa JSX transform automático, el import explícito es redundante); (2) TS2304: `global` no reconocido en 2 archivos de tests (.test.tsx) — falta tipado de entorno Vitest (`@vitest/globals` o `types` en tsconfig); (3) TS2769 en `vite.config.ts`: clave `test` no reconocida en el tipo `UserConfigExport` — falta `/// <reference types="vitest" />` o la importación correcta del plugin. Confirmado preexistente: `npm run build` fallaba idénticamente antes de C10.2. Prioridad media — bloquea el build de producción pero no el dev (`npm run dev`). Resolver antes de Fase 4b (CI/CD) o antes de defensa si se demuestra el build.
 [ ] **C10.2.2** — Sin `.nvmrc` en `frontend_ui/`. El proyecto usa Node v24.11.0 localmente pero no hay fichero que lo fije. Riesgo: diferente versión de Node en CI/CD futuro puede producir comportamientos distintos. Agregar `.nvmrc` con `24` (o la versión exacta) antes de configurar CI en J6. Prioridad baja.
 [ ] **C9.7** — Nominada por TTH-08 handoff F7 §6.2 (2026-05-29), promovida en F9. **Paridad migración Alembic ↔ modelo SQLAlchemy** para `vision_aggregates`: el e2e de F7 valida repo↔modelo↔Postgres vivo pero **NO** migración↔modelo. Si la migración `5b4beac1055d_vision_aggregates_and_drop_legacy_vision.py` y `shared/cerebrovial_shared/database/models.py:86-115` divergen, el test pasa y producción rompe (mismo patrón de bit-rot que csv legacy F5b o divergencia §5.8). Forma de cierre sugerida: test chico con `alembic.autogenerate.api.compare_metadata` contra BD post-`upgrade head` que reviente si el diff no está vacío. **Cross-ref:** tth-08-fase7-handoff.md §6.2; vision_contract.md §7; tth-08-fase9-handoff.md §[backlog post-TTH-08].
 [ ] **C9.8** — Nominada por TTH-08 handoff F7 §6.3 (2026-05-29), promovida en F9. **Wirear `edge_device/tests` a CI.** El workflow `.github/workflows/ci.yml` corre solo `core_management_api/tests/`. Las 124 tests de `edge_device/tests/` (120 heredadas + 4 e2e CT-08.11(e) de F7) están fuera de CI desde TTH-03 (decisión histórica). TTH-03 había declarado *"hasta que TTH-08 entregue módulo y tests estables"* — F7 cumple esa condición. Falta job CI nuevo con Docker, caché de imagen TimescaleDB, y decisión sobre deps pesadas (YOLO/torch). Dueño: F9.x o TTH-03 retomado. **Cross-ref:** tth-08-fase7-handoff.md §5 + §6.3; vision_contract.md §7; tth-08-fase9-handoff.md §[backlog post-TTH-08].
 [ ] **F9.y** — Nominada por TTH-08 F9 (2026-05-29). **Barrido de código huérfano del refactor de visión.** Dos piezas confirmadas sin consumidor runtime tras el refactor:
   1. `edge_device/src/vision/application/processors/smart_detection.py` — DHU-024 §3 declaró `SmartDetectionProcessor.get_analysis_for_frame()` como lógica muerta; F9 confirmó que no hay caller.
   2. `edge_device/tests/vision/unit/test_multi_camera_manager.py` — apunta a `MultiCameraManager` reescrito por F4b a `CameraManager` con API distinta; el archivo entero es referencia histórica.
   Sub-fase de cleanup separable post-F9 (es código productivo, fuera del alcance de F9 = solo docs). Forma de cierre: borrar ambos archivos, verificar que la suite `tests/vision/` sigue verde (124 passed) y que ningún import quedó colgado. **Cross-ref:** TODO.md C1.6 y C1.7 (origen); vision_contract.md §7; tth-08-fase9-handoff.md §[backlog post-TTH-08].
 [ ] **F9.z** — Nominada por TTH-08 F9 (2026-05-29). **Pin CPU de `torch` en `edge_device/requirements.txt`** (ex-C7.6, reabierta). Cambiar línea 5 de `torch` a `torch --index-url https://download.pytorch.org/whl/cpu` (~200MB) o equivalente con `--extra-index-url`, + smoke `invoke up-build --service=edge_device` y arranque del pipeline para confirmar que YOLO carga. Sub-fase de infra separable post-F9. **Cross-ref:** TODO.md C7.6 (reabierta); DECISIONS_HU.md addendum F9 a DHU-024 §7; vision_contract.md §7; tth-08-fase9-handoff.md §[backlog post-TTH-08].

BLOQUE D — Avance del lunes 4 (preparación)

 D1. Ensayo del avance: levantar docker compose up en una máquina limpia. Si falla, arreglar antes de presentar.
 D2. Preparar 5 capturas: (1) docker compose up corriendo, (2) frontend mostrando dashboard, (3) detalle de cámara con stream, (4) árbol de directorios limpio, (5) docs/PLAN.md abierto con las fases.
 D3. Slide o documento de 1 página resumiendo: "encontramos deuda técnica del refactor, hicimos assessment, plan de remediación, fase 1 completa, próximas fases hasta el 11 de mayo".
 D4. Presentar el avance.


BLOQUE E — Fase 2: Cimientos reales (objetivo: lunes 11)
Cada ítem es sesión separada de Claude Code.

 [x] E1. Inicializar Alembic en core_management_api: alembic init alembic, configurar alembic.ini con la URL de .env, configurar env.py para leer los modelos de shared/database/models.py.
 [x] E2. Generar la primera migración con todas las tablas modeladas: alembic revision --autogenerate -m "initial schema". Revisar el SQL generado antes de aplicar.
 [x] E3. Generar segunda migración para activar TimescaleDB hypertables sobre vision_tracks, vision_flows, waze_jams, waze_alerts. Esto es SQL manual: SELECT create_hypertable('vision_tracks', 'timestamp');.
 [x] E4. Borrar la función init_db() que nadie llama. Las tablas ahora se crean con alembic upgrade head.
 [x] E5. Crear scripts/seed.py con datos reales de Miraflores: 5 intersecciones (Av. Larco, Av. José Pardo, Av. Angamos, Av. Arequipa, Av. del Ejército) con sus coordenadas reales, las 4 cámaras con sus URLs YouTube, un usuario admin de prueba.
 E6. Modificar el frontend DashboardView.tsx para que las coordenadas de cámaras vengan de GET /api/intersections en lugar de estar hardcoded.
 [x] E7. Crear modelo User en shared/database/models.py: id, email, password_hash, role (operador / analista / admin), created_at.
 E8. Implementar endpoint POST /api/auth/login: recibe email + password, valida con passlib, retorna JWT con python-jose.
 E9. Crear dependency get_current_user que valida el JWT en headers. Crear require_role(role) para endpoints protegidos por rol.
 E10. Aplicar get_current_user a las rutas existentes de prediction y vision. Decidir cuáles van por rol (admin para CRUD usuarios, todos los roles para lectura, etc.).
 E11. Cerrar CORS: cambiar allow_origins=["*"] por ["http://localhost:5173"] (Vite dev) y la URL prod cuando exista.
 E12. Crear LoginView.tsx en frontend, AuthContext con el JWT, apiClient axios con interceptor que agrega el token a cada request.
 E13. Reemplazar URLs hardcoded localhost:8000/localhost:8001 por import.meta.env.VITE_API_BASE_URL. Crear frontend_ui/.env.example.
 E14. Mover la API key de Gemini a core_management_api: crear endpoint POST /api/ai/chat que recibe el prompt y llama a Gemini con la key del .env del backend. Frontend llama a este endpoint, no directo a Gemini.
 [x] E15. Crear frontend_ui/Dockerfile multi-stage: build con node:20, serve con nginx:alpine. Agregarlo a docker-compose.yml como servicio frontend.
 [x] E16. Verificar end-to-end: docker compose up levanta todo, podés hacer login, ver dashboard con datos de seed.
 E17. Commit final del bloque E: [Fase 2] Cimientos: alembic, seed, JWT, frontend configurable.
[x] **E18** — Crear modelo `VisionAggregateDB` en `cerebrovial_shared.database.models`. Schema alineado con `csv_repository.py`: timestamp, camera_id, street_monitored, conteos por clase, total, occupancy, flow_rate, avg_speed, avg_density, zone_id, duration_seconds. PK compuesta (id uuid, timestamp) para hypertable de TimescaleDB. FK camera_id → cameras. Generar migración Alembic.
[x] **E19** — Implementar `PostgresAggregateRepository(TrafficRepository)` en `edge_device/src/vision/infrastructure/persistence/postgres_repository.py`. Implementa el método `save(data: TrafficData)` escribiendo a `vision_aggregates`. Usa SQLAlchemy con la sesión configurada en `cerebrovial_shared.database`.
[x] **E20** — Configurar `pipeline_builder.py` para inyectar Postgres repo. Decidir: (a) reemplazar CSV por Postgres, (b) cascada (escribir a ambos). Recomendado: cascada con flag de configuración para mantener CSV como respaldo durante validación.
[ ] **E21** — Verificación end-to-end de visión→BD. `invoke up`, una cámara procesa video real durante 5 minutos, query a `vision_aggregates` confirma filas insertadas con datos coherentes. Capturar query de ejemplo para defensa. **⚠ Bloqueada por DEUDA-ZONAS-ONDEMAND**: el path on-demand manda `zones: {}`; sin zonas no hay filas, así que correr una cámara no basta como está escrito.
[ ] **DEUDA-ZONAS-ONDEMAND** — El alta on-demand de cámaras (`_build_camera_config`, `edge_device/src/vision/presentation/api/routes/cameras.py:51`) arma el cfg desde cero y toma `zones` del body. Tanto el frontend (`CameraDetailView.tsx:247`) como cualquier POST mandan `zones: {}`, y `_build_camera_config` NO mergea `conf/vision/default.yaml`, así que las `zone1`–`zone4` genéricas tampoco entran. Sin zonas el `zone_counter` no emite `TrafficData` y `vision_aggregates` nunca recibe filas en el flujo on-demand — aunque la persistencia esté "wired". **Bloquea E21**: correr una cámara no basta. Salidas (dev/calibración): (a) pasar zonas reales en el body; (b) inyectar zonas en el path on-demand; (c) fallback a `default.yaml` cuando el body manda `{}`. Definir zonas reales de una cámara usa el script de calibración interactivo (`interaction.py`, ver C7.7). **Cross-ref:** E21; C7.7.

BLOQUE F — Fase 3a: GRU básico funcional (objetivo: lunes 11, junto con E)
Esta es la parte que más vamos a discutir en chat antes de delegar a Claude Code.

 F1. Conmigo en chat: definir la especificación del modelo GRU. Inputs (ventana temporal de qué features), outputs (clasificación de 5 niveles de congestión a 15/30/45 min), arquitectura (capas, hidden size), función de pérdida, métricas. Salida: docs/MODEL.md.
 F2. Crear ia_prediction_service/scripts/generate_synthetic_data.py: genera dataset sintético para entrenar el GRU. Patrones realistas (hora pico AM/PM, fines de semana distintos, ruido). Output: CSV con la misma estructura que produce el módulo de visión.
 F3. Crear ia_prediction_service/src/models/gru_model.py: clase CongestionGRU con PyTorch Lightning, nn.GRU interno, encoder/decoder lineales.
 F4. Adaptar ia_prediction_service/src/training/predictor.py (o crear nuevo) para entrenar el GRU sobre el dataset sintético.
 F5. Entrenar la primera versión. Métrica objetivo: accuracy ≥ 70% (luego iteramos para llegar al 80% del IE04). Guardar el .pt en ia_prediction_service/models/gru_v1.pt.
 F6. En core_management_api: crear prediction/infrastructure/gru_predictor.py que carga el .pt y hace inferencia. Mantener el RandomForestPredictor actual como fallback con flag de configuración.
 F7. Modificar el endpoint POST /api/predictions/predict para usar GRUPredictor por defecto. Documentar el fallback a RF en docs/DECISIONS.md.
 F8. Validar que el frontend sigue funcionando con el nuevo predictor.
 F9. Commit: [Fase 3a] GRU básico entrenado y servido.


BLOQUE G — Entregable del lunes 11 (preparación)

 G1. Escribir el README raíz definitivo con quickstart de 5 pasos y diagrama de arquitectura.
 G2. Probar entregable en máquina limpia: git clone, docker compose up, npm install && npm run dev, login, ver predicción del GRU.
 G3. Grabar video corto de 2-3 min mostrando el flujo. Útil para entrega y para defensa.
 G4. Entregar.


BLOQUE H — Fase 3b: Control adaptativo (semana del 12)
Acá está OE03 demostrando.

 H1. Conmigo en chat: definir las reglas del control adaptativo. Qué umbrales, qué acciones. Salida: docs/CONTROL.md.
 H2. Implementar control/application/rules_engine.py: dado un estado de tráfico + predicción GRU, retorna un IntersectionControlPlan modificado.
 H3. Implementar control/infrastructure/plan_repository.py: CRUD de planes en BD.
 H4. Endpoints GET /api/control/intersections/{id}/current-plan, POST /api/control/recompute, GET /api/control/history.
 H5. Tabla de auditoría: cada cambio de plan se registra con timestamp, intersección, plan anterior, plan nuevo, predicción que lo motivó, usuario (o "sistema"). Esto cubre HU09.
 H6. Frontend: vista nueva o widget en IntersectionDetail que muestra plan actual, próximo cambio sugerido, historial.
 H7. Tests del motor de reglas (escenarios concretos: alta saturación → +verde en avenida principal, etc.).
 H8. Commit: [Fase 3b] Control adaptativo funcional.


BLOQUE I — Fase 4a: HU pendientes priorizadas (semana del 19)

 I1. HU17 — Alertas: tabla alerts en BD, endpoint POST /api/alerts, GET /api/alerts, conexión SSE para push en tiempo real al frontend.
 I2. Frontend AlertsView.tsx con datos reales (hoy es de 45 líneas con mocks).
 I3. HU03 — Contingencia: health checks periódicos a cámaras, si una falla → fallback a plan fijo + alerta automática.
 I4. HU12 — Reportes: endpoint GET /api/reports/daily?date=... que genera PDF con KPIs del día. Cron que lo genere a las 00:00.
 I5. HU06 — Comparación antes/después: endpoint que compara métricas con y sin sistema (con datos sintéticos del simulador). Vista frontend con gráficos.
 I6. Frontend AdminView.tsx con datos reales (hoy también es 46 líneas mock): CRUD usuarios, panel de salud (GET /api/health).
 I7. Frontend AnalyticsView.tsx conectado a GET /api/traffic/history (TimescaleDB).
 I7.1. Frontend: Lógica de colores dinámica para marcadores del mapa según nivel de tráfico (CA2).
 I7.2. Frontend: Mecanismo de actualización automática (polling) para datos del dashboard (CA2).
 I7.3. Frontend: Implementar estados de carga (spinners/skeletons) para cumplir con tiempos de respuesta (CA1/CA3).
 I7.4. Frontend: Popups en mapa con datos reales de flujo y velocidad (CA1).
 I8. Commit: [Fase 4a] HU pendientes integradas.


BLOQUE J — Fase 4b: Validación, tests, CI (semana del 26)

 J1. Mejorar dataset sintético del GRU para que llegue a 80% accuracy. Iterar arquitectura/hyperparams si es necesario. Documentar el experimento.
 J2. Medir latencia real de YOLO + transmisión + dashboard. Si supera 2s, optimizar (frame skipping, resolución más baja). Documentar.
 J3. Medir precisión de detección de YOLO en video de prueba. Documentar.
 J4. Decisión documental: actualizar números del documento de tesis con los reales (no usar 88.2% / 81.3% si los reales son distintos). Esto es lo más importante para la integridad académica.
 J5. Tests E2E con pytest+httpx: login → consulta intersecciones → trigger predicción → consulta plan → log de auditoría.
 J6. GitHub Actions workflow: lint con ruff, tests backend con pytest, tests frontend con vitest, build de imágenes Docker. Sin deploy.
 J7. Configurar pre-commit: ruff, black, eslint. Opcional pero útil.
 J8. Documentar limitaciones del demo en el README (datos sintéticos, no Pi real, no Azure, etc.) y plan de productivización.
 J9. Commit: [Fase 4b] Validación + CI + métricas reales.


BLOQUE K — Cierre y defensa (semana del 2 de junio)

 K1. Ensayo de defensa con el sistema corriendo. Probar que se levanta en máquina limpia.
 K2. Actualizar capítulos de la tesis: arquitectura entregada, modelo GRU con resultados reales, control adaptativo justificado, métricas de validación.
 K3. Capturas finales y diagrama de arquitectura limpio para el documento.
 K4. Video demo de 5 minutos: levantar el sistema, login, dashboard, detalle de cámara con YOLO en vivo, predicción GRU, plan adaptativo, auditoría.
 K5. Buffer para arreglos de último momento.
 K6. Defender.


DEUDAS FASE A — modelo de datos de intersecciones (D-016, 2026-06-05)

 [ ] **DEUDA-CAM-GEO** — La asociación cámara-Claro ↔ intersección es **nominal**: en `scripts/seed_intersections.py` los 11 `stream_url` de Claro se asignan 1:1 a las 11 intersecciones **arbitrariamente** (por orden del mapeo), sin concordancia geográfica real. Falta verificar qué stream de Claro corresponde geográficamente a cada intersección de Miraflores y reasignar. El modelo (`cameras.intersection_id` + `cameras.stream_url`) ya lo soporta; es solo cuestión de datos. **Cross-ref:** D-016; `documentation/contracts/intersections_contract.md`.
 [ ] **DEUDA-CTRL-TLS** — 10 de 11 intersecciones quedan **sin `tls_id`** (solo `larco_benavides` lo tiene, verificado contra `corredor_adaptive.py`). El modelo (`intersections.tls_id` nullable) soporta poblarlo; falta una fase de control futura que verifique el `tls_id` SUMO de cada intersección antes de cargarlo (no se mete a la base un `tls_id` sin verificar). **`arequipa_angamos` es el caso "casi listo":** ya es nodo de control sembrado en `graph_nodes` (el motor genérico puede apuntarlo); falta solo verificar que su `tls_id` SUMO es su `junction_id` del mapeo. Las otras 9 no tienen nodo de control sembrado. **Cross-ref:** D-016; `documentation/contracts/intersections_contract.md`.


DEUDA FRONTEND — switch de modos del mapa (track feature/tomtom, 2026-06-07)

 [ ] **DEUDA-SWITCH-MODE** — El switch de modos de `CongestionMapView.tsx` (`live`/`historic`/`prediction`) es un **union inline** en el `useState` (`frontend_ui/src/components/views/CongestionMapView.tsx:249`) + **patrón disperso**: 7 `useEffect` con guard `if (mode !== X) return;` y acoples de UI condicionales (leyendas, paneles, slider, título). Candidato a extracción a un `type Mode` nombrado + un componente de control de modo reutilizable, para que vistas nuevas con modos (p. ej. el track TomTom) no repliquen el patrón ad-hoc. **NO se sanea en este track** (sería un san-NN aparte). Solo registrado.


DEUDA BD — índice GiST fuera de Alembic (track feature/tomtom, 2026-06-07)

 [x] **DEUDA-GIST-MIGRACION** — **CERRADA** por la revisión `29ae3a133d00` (Fase B-1). El índice GiST sobre `graph_edges.geom` existía en la BD de dev viva pero estaba **comentado** en la migración inicial (`775d2d1db8b4:59`), así que una BD recreada desde Alembic NO lo tenía → el matching geométrico de Fase B haría seq scan. La revisión lo crea con `CREATE INDEX IF NOT EXISTS ... USING gist` y guard de dialecto (PostgreSQL-only), idempotente contra la BD que ya lo tiene. El downgrade NO lo dropea (preexistía fuera de Alembic).


DEUDAS REFUNDACIÓN VISIÓN — B1 Paso 3 (migración al scheduler, 2026-06-09)

 [ ] **DEUDA-ALTA-SINCRONICA** — La apertura de la fuente es **sincrónica** dentro de `CameraInstance.__init__` (`edge_device/src/vision/application/services/multi_camera.py:95` → `build_pipeline` → `build_source` → `VideoFileSource._initialize`). Consecuencia: una fuente muerta (HLS 404, stream caído) **revienta el alta con HTTP 500** en vez de registrar la cámara como running-pero-degradada. Es exactamente la vía de corte controlado que la observación (b) de 2-A1 daba por inexistente — ahora se sabe **por qué** no existe: el pipeline se construye eager, sin un estado intermedio "registrada, esperando fuente". Manifestado en el Paso 3: `cam_benavides_panama` (panamericana_peaje1, 404 persistente) tiró 500 al alta mientras las otras 10 levantaron. **Trigger:** diseño de la degradación controlada de streams (post-refundación) — alta tolerante a fuente no disponible + reintento/backoff. **Cross-ref:** `documentation/handoffs/refundacion-vision/b1-2a-scheduler-handoff.md` § Paso 3; D-018.
 [ ] **PENDIENTE-BENAVIDES-11A** — `cam_benavides_panama` quedó **fuera** de la migración (10/11) por 404 upstream persistente (3/3 + reintento único al cierre del gate). Es la **11ª cámara**. **B1 Paso 4 (scheduler único):** ya no hay gate de env — todas las cámaras corren por el scheduler por default, así que no requiere cambio de código ni de env, basta dar el alta cuando el stream reviva. **Trigger:** cuando `https://live.smartechlatam.online/claro/panamericana_peaje1/index.m3u8` vuelva a responder 200, alta en caliente y **verificar que `Loading model` NO incrementa** (prueba de que una cámara nueva en caliente toma el detector compartido — la evidencia bonus que el gate no pudo obtener con el stream muerto). **Cross-ref:** § Paso 3; DEUDA-CAM-GEO (la asociación stream↔intersección sigue siendo nominal).
 [ ] **DEUDA-CREEP-MEMORIA** — Creep de memoria **~1.5–2.1 MiB/min bajo carga** en el edge (10 cámaras, scheduler único, detector compartido). Medido en los gates de B1 Paso 3 (render-off, 30 min, ~2.1 MiB/min) y Paso 5 (render-on parcial, 51 min limpios, ~1.7 MiB/min decayendo): la pendiente **decae pero NO plateó** en 51 min — no se sabe si converge o crece sin límite. **Hallazgo clave (Paso 5):** es **workload-driven, no time-driven** — durante un corte de red real de ~17 min (sin frames → sin inferencia) la memoria quedó **plana** (1.701–1.702 GiB, 17 muestras); sin frames no hay creep. La caza apunta al **path de procesamiento de frames** (decode HLS / buffers / inferencia / render), no a un leak de reloj ni a estructuras que crecen en idle. El límite de 4 GiB del compose ya absorbe el creep proyectado a 8h con margen (aritmética en el handoff §11). **Trigger:** caza del leak o restart programado del edge **antes de operación 24/7**. **Cross-ref:** `documentation/handoffs/refundacion-vision/b1-2a-scheduler-handoff.md` §9/§11; D-018 (trigger de leak lento, ahora medido).
 [ ] **DEUDA-TIPOS-FRONTEND** — `tsc -b --noEmit` reporta **8 errores de tipo en archivos de test** (preexistentes en `rediseno-ui`, no inducidos por el merge): `LiveDataSection.test.tsx` (7 — mocks de `RestResource<T>` con shape laxo: `errorStatus` opcional y `data: unknown` vs. tipos estrictos) y `sseClient.test.ts` (1 — `onerror` con `null` de más). Invisibles porque `vite build` (esbuild) no type-checkea y el CI de frontend corre lint+test, no `tsc`. **Trigger:** agregar `tsc -b --noEmit` al CI de frontend (lo que los habría atrapado) y saldar los 8 en una sesión de estabilización de tipos de test. El error de producción (`SegmentedControl`/`CameraHistoryPanel`, 3 diagnósticos) ya fue resuelto en esta rama.
 [ ] **DEUDA-TEST-TOMTOM-ENV** — `TomTomView.test.tsx` asume `VITE_TOMTOM_KEY` ausente y verifica el aviso de degradación, pero no la stubea ni la limpia: en un entorno con la key definida (el `.env` local de desarrollo, o CI si la key se configura ahí), Vite la carga, la vista renderiza el mapa real y el test falla. Falla **ambiental**, no de runtime — el test debe limpiar/mockear `import.meta.env.VITE_TOMTOM_KEY` en su setup en vez de asumir el entorno. **Trigger:** incluir en la sesión de estabilización de tipos/tests de frontend (junto con DEUDA-TIPOS-FRONTEND) y agregar `tsc -b --noEmit` al CI de frontend.