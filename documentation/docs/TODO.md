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
 [ ] C1.6 — Arreglar tests de MultiCameraManager. CameraInstance perdió los atributos `camera_id` e `is_running` en el refactor de microservicios. Decidir: o se restauran los atributos en CameraInstance, o se actualizan los tests para reflejar la nueva API. Confirmar antes con compañero. Prioridad media — afecta 2 tests, no bloquea Fase 1. **Cross-ref:** este item se resuelve o se vuelve obsoleto al ejecutarse TTH-08 (DHU-024, 2026-05-27); el refactor desde cero reemplaza la capa `application/` que contiene MultiCameraManager. Puede adelantarse como deuda separada si el sprint de TTH-08 se demora.
 [ ] C1.7 — Arreglar SmartDetectionProcessor. La lógica de interpolación y de trayectorias de vehículos no funciona después del refactor (test_interpolation_logic, test_trajectory_update). Investigar si es bug real del processor o si los tests asumen API vieja. Prioridad alta — afecta el flujo de detección, debe resolverse antes de Fase 3 (donde se entrena el GRU con datos producidos por este pipeline). **Cross-ref:** SmartDetectionProcessor desaparece al ejecutarse TTH-08 (DHU-024, 2026-05-27); el DHU-024 §3 explícitamente cita `SmartDetectionProcessor.get_analysis_for_frame()` como lógica muerta que el refactor descarta. Si el sprint se demora más allá de Fase 3 del GRU, evaluar fix puntual; en otro caso queda obsoleto.
 [ ] C1.8 — Arreglar ZoneCounter. test_zone_manager_update muestra que no detecta vehículos dentro del polígono cuando debería. Investigar si es bug del polygon contains o de cómo se calcula el centroide del bbox. Prioridad alta — el conteo por zonas es funcionalidad central del módulo de visión. **Cross-ref:** ZoneCounter es reemplazado por el aggregator de la capa `application/` nueva en TTH-08 (DHU-024, 2026-05-27); el conteo por zona/ROI es CT-08.2 del refactor.
 [ ] C1.5 — Investigar y arreglar `test_pipeline_processing_flow` en edge_device. Race condition pre-existente entre `finally: pipeline.stop()` y el thread de procesamiento de `AsyncVisionPipeline`. Confirmado preexistente comparando con commit 0e20b0b4. Marcado xfail temporalmente. Prioridad media — no bloquea, pero conviene resolver antes de Fase 3. **Cross-ref:** `AsyncVisionPipeline` se reemplaza al ejecutarse TTH-08 (DHU-024, 2026-05-27); el refactor desde cero rediseña la concurrencia del pipeline en la capa `application/`. xfail queda obsoleto al merge del sprint.
 [x] C2. Crear core_management_api/src/main.py como entry point real de FastAPI con routers de prediction y control montados. Actualizar el Dockerfile de core_management_api para que apunte a src.main:app. Mantener scripts/run_prediction.py como entry alternativo de dev.
 [x] C3. Sacar ia_prediction_service del docker-compose.yml. Documentar en su README cómo correrlo manualmente para entrenar.
 [x] C4. Renombrar el servicio compose db_postgres → db (o cambiar el .env para que use db_postgres). Lo que sea menos invasivo.
 [x] C5. Sacar db_mongo del docker-compose.yml. Documentar en docs/DECISIONS.md que MongoDB se reemplaza por PostgreSQL+TimescaleDB para todo (incluyendo logs).
 [x] C6. Sacar api_gateway del docker-compose.yml. Borrar la referencia al directorio inexistente.
 [x] C7. Limpiar core_management_api/requirements.txt: sacar torch, ultralytics, opencv-python, supervision, hydra-core, cap_from_youtube, imageio-ffmpeg, streamlink, shapely. Mantener fastapi, uvicorn, sqlalchemy, geoalchemy2, alembic, numpy, pandas, scikit-learn, psycopg2-binary, sse-starlette, python-jose[cryptography], passlib[bcrypt], httpx, python-multipart.
 [x] **C7.5** — ✓ resuelta (2026-05-26, rama `san-06`): purgado el código STGNN muerto de `core_management_api/src/prediction/` (6 archivos: `domain.py`, `infrastructure/{models,graph_builder,data_loader,repository}.py`, `application/builder.py`) y eliminado `torch` de `core_management_api/requirements.txt`. El predictor vivo (`predictor.py → engine.py`) sigue intacto sobre joblib + sklearn RandomForest. Cierra simultáneamente SAN-01 (regla CLAUDE.md vs requirements). La prohibición de torch en core permanece como guardia anti-regresión.
 [ ] **C7.6** — Deuda derivada anotada en SAN-06 (2026-05-26), **no ejecutar todavía**: `edge_device/requirements.txt:5` (línea `torch`) y `ia_prediction_service/requirements.txt:1,6-8` (`torch>=2.3.0`, `torch-geometric`, `torch-scatter`, `torch-sparse`) descargan el wheel CUDA por default (~2GB+ cada uno). Si el target de despliegue es CPU local, evaluar `torch ... --index-url https://download.pytorch.org/whl/cpu` (~200MB) para reducir el peso de la imagen y el tiempo de build de ambos servicios. **Restricciones a verificar antes de tocar**: (1) `edge_device` usa YOLO/ultralytics — confirmar que la variante CPU-only es suficiente para la inferencia esperada del demo; (2) `ia_prediction_service` entrena el GRU — confirmar si el entrenamiento se hace en GPU real (en cuyo caso no aplica) o en CPU local. Prioridad baja — no afecta el dolor que SAN-06 ya resolvió en core_management_api. **Cross-ref:** resolución prevista dentro de TTH-08 (DHU-024 §7, 2026-05-27). Al reescribir `edge_device/requirements.txt` desde cero en el refactor, se define `--index-url https://download.pytorch.org/whl/cpu` desde el inicio; C7.6 deja de ser ticket separado para `edge_device` y se cierra como parte de la fase de infraestructura. Si el demo requiere GPU local, se agrega como configuración opcional, no como default. **Nota:** el lado de `ia_prediction_service` no está cubierto por TTH-08 — esa parte de C7.6 sigue como deuda independiente (re-evaluar al definir el sprint de TTH-09/GRU).
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
[ ] **E18** — Crear modelo `VisionAggregateDB` en `cerebrovial_shared.database.models`. Schema alineado con `csv_repository.py`: timestamp, camera_id, street_monitored, conteos por clase, total, occupancy, flow_rate, avg_speed, avg_density, zone_id, duration_seconds. PK compuesta (id uuid, timestamp) para hypertable de TimescaleDB. FK camera_id → cameras. Generar migración Alembic.
[ ] **E19** — Implementar `PostgresAggregateRepository(TrafficRepository)` en `edge_device/src/vision/infrastructure/persistence/postgres_repository.py`. Implementa el método `save(data: TrafficData)` escribiendo a `vision_aggregates`. Usa SQLAlchemy con la sesión configurada en `cerebrovial_shared.database`.
[ ] **E20** — Configurar `pipeline_builder.py` para inyectar Postgres repo. Decidir: (a) reemplazar CSV por Postgres, (b) cascada (escribir a ambos). Recomendado: cascada con flag de configuración para mantener CSV como respaldo durante validación.
[ ] **E21** — Verificación end-to-end de visión→BD. `invoke up`, una cámara procesa video real durante 5 minutos, query a `vision_aggregates` confirma filas insertadas con datos coherentes. Capturar query de ejemplo para defensa.

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