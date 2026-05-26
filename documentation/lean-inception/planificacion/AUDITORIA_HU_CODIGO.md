# Auditoría HU↔Código — CerebroVial

> Generado: 2026-05-18 por Claude Code
> Versión: v1 (en construcción incremental por batches; al cierre del Batch B1 — Foundations)
> Inputs:
> - `BACKLOG_OVERVIEW.md` (2026-05-16)
> - `HU_BLOQUE_A.md` v5 (2026-05-17), `HU_BLOQUE_E.md` (2026-05-15)
> - `TAREAS_TECNICAS_HABILITADORAS.md` (cierre MVP2 2026-05-16)
> - Repo Git en HEAD actual de `/Users/rasec/Tesis/CerebroVial/`
> Salida prescrita por: `PROMPT_ARRANQUE_CLAUDE_CODE_v2.md` Fase 4.1.

## 1. Resumen ejecutivo

**Estado final (Fase 4.1 cerrada — 32 de 32 elementos auditados):**

| Estado | Elementos | Cuenta |
|---|---|---|
| Completo | TTH-02 | 1 |
| Parcial | TTH-03, TTH-08, TTH-10, HU-05, HU-06 | 5 |
| No iniciado | TTH-01, TTH-04, TTH-05, TTH-07, TTH-09, TTH-11, HU-01, HU-02, HU-03, HU-04, HU-07, HU-08, HU-10, HU-11, HU-12, HU-13, HU-14, HU-15, HU-16, HU-17, HU-09, HU-18, HU-19, HU-20, HU-21 | 25 |
| Fuera de scope | TTH-06 (Trabajos Futuros) | 1 |
| Indeterminado | — | 0 |

**Deltas críticos:** 13 (ver §5).
**Pendientes de consulta humana:** 0.
**Cobertura del backlog implementada:** ≈ 25% (1 completo + 5 parciales sobre 31 elementos in-scope). Brecha estructural para MVP1.

## 2. Tareas Técnicas Habilitadoras

### TTH-01 — Implementación de autenticación JWT con bcrypt

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CT-01.1 (endpoint `POST /auth/login`), CT-01.2 (bcrypt hashing en runtime), CT-01.3 (JWT con claims `sub`/`role`/`exp`), CT-01.4 (dependency `get_current_user`), CT-01.5 (tests).
- **Archivos del repo:** evidencia preparatoria solamente:
  - [`core_management_api/requirements.txt`](Tesis/CerebroVial/core_management_api/requirements.txt) línea 8: declara dependencia `passlib[bcrypt]`.
  - [`core_management_api/alembic/versions/99319147948b_add_users_table.py`](Tesis/CerebroVial/core_management_api/alembic/versions/99319147948b_add_users_table.py): crea tabla `users` con columnas `id`, `email`, `password_hash`, `role`, `created_at` (la nota técnica del spec dice "La tabla User ya está creada").
  - Sin archivos de runtime: cero matches en `core_management_api/src/` para `jwt`, `get_current_user`, `HTTPBearer`, `create_access_token`, `/auth/login`.
- **Deltas:** ver Delta-02 en §5 sobre el tipo de la columna `role`.

### TTH-02 — Arquitectura Docker Compose multi-servicio

- **Estado:** Completo.
- **CAs cubiertos:** CT-02.1 a CT-02.6.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** ninguno.
- **Archivos del repo:**
  - [`docker-compose.yml`](Tesis/CerebroVial/docker-compose.yml) declara servicios `db` (timescale/timescaledb-ha:pg15), `core_management_api`, `edge_device`, `frontend_ui` con builds, ports, depends_on y volumes — satisface CT-02.1, CT-02.3, CT-02.4.
  - [`docker-compose.dev.yml`](Tesis/CerebroVial/docker-compose.dev.yml) override para hot-reload del core (per CLAUDE.md).
  - [`.env.example`](Tesis/CerebroVial/.env.example) (1919 bytes, con `.env` correspondiente de 1934 bytes) — satisface CT-02.6.
  - [`README.md`](Tesis/CerebroVial/README.md) raíz + [`CLAUDE.md`](Tesis/CerebroVial/CLAUDE.md) documentan quickstart vía `invoke setup-dev` → `invoke up` con verificación de prerequisitos (Docker, LFS), URLs locales, comandos de logs/reset/shell — satisface CT-02.5.
  - [`tasks.py`](Tesis/CerebroVial/tasks.py) (invoke task runner) envuelve `docker compose` con validaciones que evitan errores crípticos.
  - [`infra/docker/`](Tesis/CerebroVial/infra/docker/) configuraciones de infra.
- **Deltas:** ninguno significativo; la composición real va más allá del CT mínimo (entrypoint con alembic auto-run, dev override, invoke wrapper).

### TTH-03 — Repositorio Git y pipeline CI con cobertura completa

- **Estado:** Parcial.
- **CAs cubiertos:** CT-03.3 (dispara en push y PR), CT-03.5 (estados verde/rojo en GitHub Actions).
- **CAs parcialmente cubiertos:** CT-03.4 (cobertura incompleta — ver deltas).
- **CAs no cubiertos:** CT-03.4 totalmente (mypy ausente), CT-03.1 (usa rama `master`, no `main`/`develop` como sugiere el spec), CT-03.6 (protección de ramas no verificable desde el .yml; es config de GitHub).
- **Archivos del repo:**
  - [`.github/workflows/ci.yml`](Tesis/CerebroVial/.github/workflows/ci.yml): jobs `backend-checks` (ruff + pytest sobre `core_management_api/tests`), `frontend-checks` (npm lint + npm test), `docker-build` (docker compose build). Dispara en push a `master`/`fase-*` y PR a `master`.
- **Deltas:** ver Delta-03 en §5. El propio spec en "Estado actual" cita DISCOVERY_2026-05-10 §5.5 reconociendo que CI no corre tests de `edge_device`, `shared/`, `ia_prediction_service/`. Esta auditoría confirma que la situación se mantiene.

### TTH-04 — Lógica de fallback en cascada del sistema

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CT-04.1 a CT-04.10 (todos).
- **Archivos del repo:** ninguno. `grep` exhaustivo sobre `core_management_api/src/` y `frontend_ui/src/` para `operational[_-]state`, `/system/components`, `/system/operational`, `health[_-]check`, `circuit[_-]breaker`, `degraded_[123]`, `safe_3`, `total_failure` devuelve cero matches.
- **Deltas:** ninguno (ausencia total, no hay conflicto).

### TTH-05 — Configuración de tiempos preconfigurados para degradado nivel 3

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CT-05.1 a CT-05.8 (todos).
- **Archivos del repo:** ninguno. `grep` para `fallback[_-]times` y `/admin/fallback-times-config` devuelve cero.
- **Deltas:** ninguno.

### TTH-06 — Capa de DTOs transversal al backend

- **Estado:** Fuera de scope.
- **Razón:** Clasificada explícitamente como **Trabajos Futuros** por DHU-014. No se construye dentro del alcance del proyecto académico (spec sección "Por qué se clasifica como Trabajos Futuros y no MVP2"). No auditable.
- **Archivos del repo:** N/A.

### TTH-07 — Integración con SUMO para simulación del entorno

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CT-07.1 a CT-07.8 (todos).
- **Archivos del repo:** ninguno. `grep` para `sumo`/`traci`/`TraCI` sobre `core_management_api/`, `ia_prediction_service/`, `edge_device/` devuelve cero. `find` para `*.net.xml`/`*.rou.xml` no devuelve nada bajo `CerebroVial/`.
- **Deltas:** ninguno. El propio spec confirma: "🆕 Por construir desde cero. No hay código SUMO ni topología cargada en el repositorio."

### TTH-08 — Módulo de visión computacional que produce métricas de estado

- **Estado:** Parcial.
- **CAs cubiertos:** CT-08.1 (detección YOLO11n.pt configurada — [`edge_device/yolo11n.pt`](Tesis/CerebroVial/edge_device/yolo11n.pt)), CT-08.2 (tracking vía `supervision` por inventario), CT-08.3 (ROI polígonos vía `InteractiveZoneSelector` + `zone_counter` en `infrastructure/zones/`), CT-08.7 (input video grabado y stream — soporte multi-fuente en `infrastructure/sources/`), CT-08.8 (stream procesado con bounding boxes vía `presentation/legacy_api.py`).
- **CAs parcialmente cubiertos:** CT-08.4 (métricas conteo/cola/flujo/densidad — aggregators existen pero la persistencia es a CSV, no a las tablas/contratos prescritos), CT-08.11 (tests — existe [`edge_device/tests/vision/test_yolo_detector.py`](Tesis/CerebroVial/edge_device/tests/vision/test_yolo_detector.py) limitado al detector).
- **CAs no cubiertos:** CT-08.5 (persistencia BD — tabla `vision_aggregates` no existe; el pipeline persiste a CSV en lugar de BD, decisión D-006/D-007 de CLAUDE.md), CT-08.6 (endpoint canónico `GET /vision/state` — solo expone stream MJPEG, no métricas estructuradas), CT-08.9 (dataset etiquetado ≥200 frames con precisión/recall/mAP — no documentado), CT-08.10 (health check consumible por TTH-04 — TTH-04 no existe).
- **Archivos del repo:**
  - [`edge_device/src/vision/`](Tesis/CerebroVial/edge_device/src/vision/) con estructura DDD (domain/application/infrastructure/presentation).
  - [`edge_device/src/vision/application/`](Tesis/CerebroVial/edge_device/src/vision/application/) pipelines, builders, aggregators, processors.
  - [`edge_device/src/vision/presentation/legacy_api.py`](Tesis/CerebroVial/edge_device/src/vision/presentation/legacy_api.py) FastAPI separada del core con streaming endpoint.
- **Restricciones aplicables:** CLAUDE.md de CerebroVial prohíbe refactorizar `edge_device/src/vision/`. Por tanto cualquier acción correctiva en sprint 4 debe respetar esta restricción y operar sobre el wrapper/adapter, no sobre el código de visión interno.
- **Deltas:** ver Delta-04 (refactor vs evolución) y Delta-05 (vision_aggregates planeada pero no existe).

### TTH-09 — Modelo predictivo GRU servido vía API

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno con el contrato prescrito por la TTH.
- **CAs parcialmente cubiertos:** CT-09.4 (existe un endpoint en el path canónico `POST /predictions/predict` pero con contrato completamente distinto al spec — ver Delta-01).
- **CAs no cubiertos:** CT-09.1 (GRU univariado por dirección — solo existe `STGCNModel` descartado por D-006 y `RandomForestPredictor` baseline), CT-09.2 (arquitectura multi-output ventana/horizonte), CT-09.3 (script reproducible de entrenamiento — existe [`ia_prediction_service/scripts/train.py`](Tesis/CerebroVial/ia_prediction_service/scripts/train.py) pero no integrado al GRU servido vía API), CT-09.5 (persistencia de predicciones), CT-09.6 (evaluación 4 métricas), CT-09.7 (accuracy ≥80% aspiracional), CT-09.8 (fallback HTTP estándar — el endpoint actual lanza HTTP 500 vía `except Exception` pero no está cableado a TTH-04 Nivel 2), CT-09.9 (tests del GRU servido).
- **Archivos del repo:**
  - [`core_management_api/src/prediction/presentation/api/routes.py`](Tesis/CerebroVial/core_management_api/src/prediction/presentation/api/routes.py) líneas 25-62: endpoint `POST /predictions/predict` con `PredictionInput` (camera_id + 6 métricas) → `PredictionResponse` con `current/predicted_15min/30min/45min` discretos (Normal/High/Heavy) + alert + message.
  - [`core_management_api/src/prediction/infrastructure/models.py`](Tesis/CerebroVial/core_management_api/src/prediction/infrastructure/models.py): `STGCNModel` (GraphConv+LSTM, descartado por D-006), `RandomForestPredictor` baseline activo con 6 joblib (`traffic_rf_class/reg_15/30/45min.joblib` + current).
  - [`ia_prediction_service/src/models/`](Tesis/CerebroVial/ia_prediction_service/src/models/): `base_model.py`, `model_factory.py`, `time_then_space.py` (STGNN preservado como referencia histórica per CLAUDE.md).
  - [`ia_prediction_service/scripts/`](Tesis/CerebroVial/ia_prediction_service/scripts/) train.py, evaluate.py, predict.py existen pero su lifecycle no está integrado al servidor.
- **Deltas:** ver Delta-01 (contrato divergente endpoint).

### TTH-10 — Motor adaptativo de control semafórico

- **Estado:** Parcial.
- **CAs cubiertos:** CT-10.1 (AdaptiveEngine en `core_management_api/src/control/application/adaptive_engine.py`), CT-10.2 (Webster en `webster.py`), CT-10.3 (Max Pressure en `max_pressure.py`), CT-10.6 (constantes MTC en `mtc_constraints.py`), CT-10.7 (operaciones MTC elevar/recortar/componer), CT-10.8 (output endpoint `POST /control/recommend` retorna estructura con razonamiento), CT-10.14 (tests automatizados — [`core_management_api/tests/control/`](Tesis/CerebroVial/core_management_api/tests/control/) con tests de Webster, MaxPressure, MTC y conftest).
- **CAs parcialmente cubiertos:** CT-10.4 (selección por umbral `flow_total>1500` — implementación presente en AdaptiveEngine pero parametrizabilidad por archivo de configuración no auditada en detalle), CT-10.5 (4 casos de la matriz cubiertos por tests por inventario), CT-10.9 (persistencia de decisiones — no se detectó tabla `motor_decisions` en migrations; puede ser parcial o ausente; requiere verificación en B2 con HU-08).
- **CAs no cubiertos:** CT-10.10 (integración con TTH-09 — el motor consume del RandomForest, no del GRU; GRU no servido), CT-10.11 (integración TraCI con TTH-07 — TTH-07 no existe), CT-10.12 (integración con HU-15 parámetros configurables — HU-15 no auditada aún, pero el inventario no muestra UI ni endpoint de configuración de parámetros operativos), CT-10.13 (health check para TTH-04 — TTH-04 no existe).
- **Archivos del repo:**
  - [`core_management_api/src/control/application/webster.py`](Tesis/CerebroVial/core_management_api/src/control/application/webster.py)
  - [`core_management_api/src/control/application/max_pressure.py`](Tesis/CerebroVial/core_management_api/src/control/application/max_pressure.py)
  - [`core_management_api/src/control/application/mtc_constraints.py`](Tesis/CerebroVial/core_management_api/src/control/application/mtc_constraints.py)
  - [`core_management_api/src/control/application/adaptive_engine.py`](Tesis/CerebroVial/core_management_api/src/control/application/adaptive_engine.py)
  - [`core_management_api/src/control/presentation/api/`](Tesis/CerebroVial/core_management_api/src/control/presentation/api/) endpoint `POST /control/recommend`.
  - [`documentation/docs/CONTROL.md`](Tesis/CerebroVial/documentation/docs/CONTROL.md) (552 líneas, referencia normativa según TTH-10 spec).
  - [`core_management_api/tests/control/`](Tesis/CerebroVial/core_management_api/tests/control/) suite de tests.
- **Concordancia con auto-clasificación del spec:** el spec mismo se auto-clasifica "✓✓ Construido, integración pendiente con otras TTH del Bloque E" y enumera 5 pendientes idénticos a los detectados aquí.

### TTH-11 — Spike de calibración de hiperparámetros temporales

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CT-11.1 (documento entregable en `documentation/docs/INVESTIGACION_HIPERPARAMETROS_TEMPORALES.md` o similar) a CT-11.8 (todos).
- **Archivos del repo:** ninguno. `find` para `*hiperparametro*` devuelve cero matches; `documentation/docs/` no contiene tal archivo.
- **Deltas:** ninguno (ausencia total).

## 3. Historias de Usuario MVP1

### Bloque A — Acceso

#### HU-01 — Acceso diferenciado por rol

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-01.1, CA-01.2, CA-01.3 (vistas segregadas por rol — frontend renderiza todas las vistas sin distinción), CA-01.4 (API 403 sin filtrar — sin enforcement en backend), CA-01.5 (redirección a login — no hay login), CA-01.6 (manejo de token expirado — no hay tokens en uso).
- **Archivos del repo:**
  - [`frontend_ui/src/App.tsx`](Tesis/CerebroVial/frontend_ui/src/App.tsx): el routing usa estado local `activeTab` con sentencia condicional sobre `'dashboard' | 'analytics' | 'alerts' | 'admin' | 'control'`. **Todas las vistas son accesibles a cualquier usuario, sin verificación de rol.**
  - [`frontend_ui/src/components/layout/Sidebar.tsx`](Tesis/CerebroVial/frontend_ui/src/components/layout/Sidebar.tsx) línea 79: muestra usuario hardcoded "Operador C4" sin lógica de sesión.
  - [`frontend_ui/src/components/views/AdminView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/AdminView.tsx) línea 13: lista de usuarios hardcoded ("Ana Torres (Admin)", "Jorge Salazar (Analista)", "Lucía Ramos (Operador)") — mock visual, sin lógica funcional.
  - [`frontend_ui/src/services/predictionService.ts`](Tesis/CerebroVial/frontend_ui/src/services/predictionService.ts), [`frontend_ui/src/services/controlService.ts`](Tesis/CerebroVial/frontend_ui/src/services/controlService.ts): no envían header `Authorization` ni manejan token.
  - Backend: cero matches para `RBAC`, `require_role`, `Depends.*role` en `core_management_api/src/`. Sin enforcement.
- **Deltas:** ver Delta-02.
- **Dependencia bloqueante:** HU-01 requiere TTH-01 completada (autenticación). Por construcción, sin TTH-01 no puede haber enforcement de rol.

### Bloque B — Operador tiempo real

#### HU-02 — Monitoreo del estado actual de la intersección en tiempo real

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno con la semántica que la HU exige.
- **CAs parcialmente cubiertos:** CA-02.1 (existe un panel de "monitoreo" pero muestra una lista de intersecciones/cámaras en un mapa Leaflet, no flujo+cola por acceso de UNA intersección como exige la HU).
- **CAs no cubiertos:** CA-02.2 (auto-update ≤5s — no hay SSE/WebSocket ni polling; el `useEffect` de [`DashboardView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/DashboardView.tsx) hace un `fetch` único al `/api/intersections` al montar), CA-02.3 (umbrales verde/amarillo/rojo — no implementados), CA-02.4 (marca "desactualizado" con tiempo desde último update — DHU-005 Caso A, no implementado), CA-02.5 (redirect login — sin login).
- **Archivos del repo:**
  - [`frontend_ui/src/components/views/DashboardView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/DashboardView.tsx): mapa Leaflet con cámaras + fetch único de `/api/intersections`.
  - [`frontend_ui/src/components/views/CameraDetailView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/CameraDetailView.tsx): vista detalle de una cámara individual (presunto navegador desde la lista).
  - [`frontend_ui/src/components/widgets/TrafficHistoryWidget.tsx`](Tesis/CerebroVial/frontend_ui/src/components/widgets/TrafficHistoryWidget.tsx): widget complementario.
  - Endpoint backend: `GET /api/intersections` en [`core_management_api/src/main.py`](Tesis/CerebroVial/core_management_api/src/main.py).
- **Deltas:** ver Delta-06 (semántica del dashboard) y Delta-07 (ausencia de mecanismo realtime).

#### HU-03 — Visualización de predicción de congestión a corto plazo

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno con el contrato exigido.
- **CAs parcialmente cubiertos:** CA-03.3 (existe lógica de "alert" en el handler `predict_traffic` basada en severidad Normal/High/Heavy → flag bool + mensaje; pero esto NO es resaltado visual por acceso superando umbral 0-5).
- **CAs no cubiertos:** CA-03.1 (nivel 0-5 por acceso en horizonte configurado — el endpoint devuelve `Normal/High/Heavy` discreto a 15/30/45 min, per-camera, no per-dirección, no en escala 0-5 jam_level), CA-03.2 (auto-update ≤5s), CA-03.4 (marca "no confirmada" DHU-005 Caso B), CA-03.5 (login).
- **Archivos del repo:**
  - [`core_management_api/src/prediction/presentation/api/routes.py`](Tesis/CerebroVial/core_management_api/src/prediction/presentation/api/routes.py) líneas 25-62: endpoint `POST /predictions/predict`.
  - [`frontend_ui/src/services/predictionService.ts`](Tesis/CerebroVial/frontend_ui/src/services/predictionService.ts): cliente del endpoint.
- **Deltas:** ver Delta-01 (heredado de TTH-09: contrato divergente impide cumplir HU-03 sin refactor del endpoint).

#### HU-04 — Vista combinada del estado actual y la predicción de tráfico

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-04.1 (no existe vista que integre HU-02+HU-03; las vistas actuales son independientes y ninguna combina), CA-04.2 (resaltado de discrepancia — no implementado), CA-04.3 (auto-update ≤5s — sin canal realtime), CA-04.4 (robustez independiente para cada fuente), CA-04.5 (login).
- **Archivos del repo:** ninguno con la semántica integrada exigida. La feature distintiva del Operador no existe en el frontend.
- **Deltas:** ninguno propio; deriva de la ausencia de HU-02 y HU-03.

#### HU-05 — Visualización de la estrategia de control activa

- **Estado:** Parcial.
- **CAs cubiertos:** ninguno completo con la semántica "vista pasiva del estado vigente"; ver delta.
- **CAs parcialmente cubiertos:** CA-05.1 (nombre estrategia + tiempos de verde por acceso — el endpoint `POST /control/recommend` retorna `modo` y `tiempos por fase`, y [`RecommendationPanel.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/RecommendationPanel.tsx) los renderiza vía `TrafficLightCycle.tsx`; pero ocurre solo cuando el usuario presiona "Play" después de editar fases manualmente — no es "estrategia vigente del sistema en producción"), CA-05.4 (manejo de error `webster_infeasible`/`invalid_state`/genérico en `useRecommendControl` cubre parcialmente la idea de "marca no confirmada", aunque la semántica difiere).
- **CAs no cubiertos:** CA-05.2 (timestamp activación de la estrategia vigente), CA-05.3 (auto-update ≤5s ante cambio), CA-05.5 (login).
- **Archivos del repo:**
  - [`frontend_ui/src/components/views/control/ControlView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/ControlView.tsx) coordina edición de fases + invocación del endpoint.
  - [`frontend_ui/src/components/views/control/RecommendationPanel.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/RecommendationPanel.tsx) muestra el output.
  - [`frontend_ui/src/components/views/control/TrafficLightCycle.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/TrafficLightCycle.tsx), [`Pedagogical422Card.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/Pedagogical422Card.tsx), [`PhaseEditor.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/PhaseEditor.tsx), [`PresetButtons.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/PresetButtons.tsx), [`Slider.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/Slider.tsx), [`TimingBar.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/TimingBar.tsx), [`ModeSelector.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/ModeSelector.tsx).
  - [`frontend_ui/src/services/controlService.ts`](Tesis/CerebroVial/frontend_ui/src/services/controlService.ts).
- **Deltas:** ver Delta-08 (semántica de ControlView).

#### HU-06 — Explicación de la razón de selección de estrategia

- **Estado:** Parcial.
- **CAs cubiertos:** ninguno completo.
- **CAs parcialmente cubiertos:** CA-06.1 (el motor produce un `reasoning` textual con valores del estado, expuesto en el output del endpoint según TTH-10 CT-10.8, y `RecommendationPanel` lo renderiza con "Log técnico (para operador C4)"; falta sustitución plantillada con catálogo acotado, lenguaje del dominio en lugar de "log técnico", y la mecánica pasiva), CA-06.3 (existe fallback textual cuando el cálculo falla — `parseControlError` con `webster_infeasible`/`invalid_state`/genérico — aunque no es exactamente "plantilla genérica cuando no hay match").
- **CAs no cubiertos:** CA-06.2 (auto-update ≤5s al cambiar estrategia), CA-06.4 (marca "no confirmada" DHU-005 Caso B), CA-06.5 (login).
- **Archivos del repo:**
  - [`core_management_api/src/control/application/adaptive_engine.py`](Tesis/CerebroVial/core_management_api/src/control/application/adaptive_engine.py) — produce el `reasoning`.
  - [`frontend_ui/src/components/views/control/RecommendationPanel.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/RecommendationPanel.tsx) línea ~189: "Log técnico (para operador C4)".
  - [`frontend_ui/src/components/views/control/Pedagogical422Card.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/control/Pedagogical422Card.tsx) — texto explicativo del caso patológico `webster_infeasible`.
- **Deltas:** ver Delta-09 (lenguaje técnico vs. dominio del operador). Delta arquitectónico compartido con HU-05.

#### HU-07 — Notificación de cambios de estrategia del motor

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-07.1 a CA-07.6 (todos). No existe infraestructura de toasts/notificaciones en frontend, no hay canal de eventos en tiempo real, no hay agrupamiento, no hay indicador de canal degradado.
- **Archivos del repo:** ninguno. `grep` para `toast`/`notification`/`notif` en `frontend_ui/src/components/` devuelve cero.
- **Deltas:** ver Delta-07 (ausencia canal realtime impacta también HU-07).

#### HU-08 — Consulta del historial de decisiones del motor

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-08.1 (F31 inglobada — persistencia de cada decisión con timestamp/estrategia/razón/parámetros/estrategia anterior: cero matches para `motor_decisions`/`decision_log` en migrations; el motor no persiste sus decisiones), CA-08.2 (paginación), CA-08.3 (filtros por rango+estrategia), CA-08.4 (orden por defecto últimas 24h), CA-08.5 (resiliencia: cola/fallback ante fallo escritura), CA-08.6 (login).
- **Archivos del repo:**
  - Existe `GET /predictions/history/{camera_id}` ([`prediction/presentation/api/routes.py`](Tesis/CerebroVial/core_management_api/src/prediction/presentation/api/routes.py) línea 64) — pero retorna historial de tráfico (CSV logs), no decisiones del motor adaptativo. Tipos distintos.
  - No hay archivo análogo para `control/history` o `motor/decisions`.
- **Deltas:** ver Delta-10 (persistencia F31 inglobada ausente).

### Bloque C — Operador degradado

#### HU-10 — Alerta activa transversal del estado operativo del sistema

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno. Existe [`AlertsView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/AlertsView.tsx) con cards de alertas mock, pero la semántica está desviada: HU-10 exige un **banner transversal persistente** visible desde cualquier vista, no una vista dedicada con cards de alertas históricas tipo notificación de seguridad. Ver Delta-11.
- **CAs no cubiertos:** CA-10.1 a CA-10.10 (todos). Sin componente banner en `App.tsx`, sin distinción visual por nivel (amarillo/naranja/rojo/falla total), sin lógica de reconocimiento+colapso, sin re-escalada automática, sin persistencia de transiciones (CA-10.7 inglobada presupone tabla `operational_state_transitions` que no existe), sin resiliencia ante fallo del log (CA-10.8), sin marca "estado no confirmado" (CA-10.9).
- **Archivos del repo:**
  - [`frontend_ui/src/components/views/AlertsView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/AlertsView.tsx): cards mock con 3 alertas hardcoded (Congestión Av. Larco / Predicción IA / Latencia Edge).
  - Sin componente banner en [`frontend_ui/src/App.tsx`](Tesis/CerebroVial/frontend_ui/src/App.tsx) ni en [`Header.tsx`](Tesis/CerebroVial/frontend_ui/src/components/layout/Header.tsx).
- **Dependencia bloqueante:** HU-10 consume CT-04.4 de TTH-04. TTH-04 está No iniciado, lo que precluye cualquier avance funcional.
- **Deltas:** ver Delta-11 (semántica desviada de AlertsView).

#### HU-11 — Vista del estado operativo de los componentes del sistema

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-11.1 a CA-11.9 (todos). Sin vista dedicada del operador con lista de componentes en escala OK/Degradado/Fuera de servicio, sin textos de impacto operativo, sin resaltado de no-OK, sin canal realtime, sin acceso desde el banner de HU-10 (que tampoco existe).
- **Archivos del repo:** ninguno con la semántica exigida. La segunda card de [`AdminView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/AdminView.tsx) ("Estado del Sistema") presenta visualmente el concepto pero pertenece al Bloque D / HU-13, no a HU-11 (operador).
- **Dependencia bloqueante:** consume CT-04.5 de TTH-04 (compartido con HU-13). TTH-04 No iniciado.

#### HU-12 — Explicación del modo degradado activo y sus implicaciones operativas

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-12.1 a CA-12.6 (todos). Sin catálogo de plantillas para explicación compuesta del modo degradado, sin componente UI para mostrar el texto compuesto, sin canal realtime para CA-12.2.
- **Archivos del repo:** ninguno.
- **Dependencia bloqueante:** misma cadena que HU-10/HU-11 — requiere TTH-04 + catálogo de plantillas.

### Bloque D — Administrador

#### HU-13 — Vista técnica de salud de los componentes del sistema

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno con datos funcionales.
- **CAs parcialmente cubiertos:** existe un **mockup visual** del concepto en la segunda card de [`AdminView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/AdminView.tsx) ("Estado del Sistema") con labels hardcoded `API Gateway (Azure) ONLINE (34ms)`, `Database (PostgreSQL + Timescale) ONLINE`, `Servicio Predicción (PyTorch) IDLE`, `Nodos Edge (Raspberry Pi) 33/34 ONLINE`. Cubre superficialmente el concepto de CA-13.1 (lista de componentes con estado + latencia) pero los datos son ficticios, no hay backend, no hay actualización (CA-13.2), no hay RBAC (CA-13.6), no hay diferenciación de "sin datos" vs cero (CA-13.5).
- **CAs no cubiertos:** todos los relacionados con datos reales y consumo del endpoint `GET /system/components/status` (CT-04.5) que no existe.
- **Archivos del repo:**
  - [`frontend_ui/src/components/views/AdminView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/AdminView.tsx) líneas ~25-45: scaffolding visual.
- **Dependencia bloqueante:** requiere TTH-04 CT-04.5 con los 7 campos por componente.
- **Deltas:** ver Delta-12 (UI scaffolding como deuda invertida).

#### HU-14 — Vista de métricas de desempeño del modelo predictivo

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno con datos reales.
- **CAs parcialmente cubiertos:** existe **mockup visual** en [`AnalyticsView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/AnalyticsView.tsx) — `recharts` `AreaChart` con `trafficData` hardcoded (`08:00 a 14:00` con `real` y `prediccion`), título "Congestión Real vs Predicción (Red Neuronal GRU)". Cubre el "look and feel" de la vista de métricas, pero ninguna de las 4 métricas exigidas por CA-14.5 (MAE, RMSE, accuracy, matriz de confusión 6×6) aparece en el componente. No hay tooltips de ayuda (CA-14.7), no hay persistencia de predicciones para comparar contra observación real (CA-14.1, CA-14.2, sustrato inglobado), no hay cálculo en backend (CA-14.3).
- **CAs no cubiertos:** todos los CAs sustantivos (CA-14.1 a CA-14.6, CA-14.8 a CA-14.14).
- **Archivos del repo:**
  - [`frontend_ui/src/components/views/AnalyticsView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/AnalyticsView.tsx) scaffolding.
  - El motor de entrenamiento offline en [`core_management_api/src/prediction/infrastructure/engine.py`](Tesis/CerebroVial/core_management_api/src/prediction/infrastructure/engine.py) usa `accuracy_score` de sklearn (línea 33) durante el training del RandomForest, pero esto es accuracy de fitting, no de evaluación operacional comparando predicción vs observación real.
- **Dependencia bloqueante:** consume TTH-09 (modelo + persistencia de predicciones CT-09.5). TTH-09 No iniciado.
- **Deltas:** ver Delta-12.

#### HU-15 — Configuración de parámetros operativos del sistema

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-15.1 a CA-15.13 (todos). Sin tabla persistente de parámetros operativos, sin endpoints `GET/PUT /admin/parameters` o equivalentes, sin UI de configuración por familia (visualización tráfico / predicción modelo / monitor salud), sin auditoría de cambios, sin restauración de defaults, sin control de concurrencia entre admins.
- **Archivos del repo:** ninguno. Los parámetros que HU-15 debería gestionar (umbrales de cola de CA-02.3, horizonte de predicción de CA-03.1, umbral de congestión de CA-03.3, ventana de cálculo de métricas de CA-14.4, frecuencia del monitor de salud de CT-04.1) tampoco están implementados como configurables en sus respectivos componentes — están hardcoded o ausentes.
- **Dependencia bloqueante:** múltiples HUs río abajo (HU-02/03/14) consumen parámetros que HU-15 debería gestionar; ninguna está implementada.

### Bloque F — Gerente

#### HU-16 — Consulta de KPIs operativos sobre periodo seleccionable

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno. AnalyticsView contiene scaffolding visual con un `AreaChart` y un `<select>` con opciones "Últimas 24 Horas"/"Última Semana", pero no implementa ningún KPI específico ni el selector con las 5 opciones de CA-16.4 (`Esta semana`/`Semana anterior`/`Este mes`/`Mes anterior`/`Rango personalizado`).
- **CAs no cubiertos:** todos (CA-16.1 a CA-16.21).
  - CA-16.1 (F30 inglobada — tabla persistente con flujo/cola/velocidad/densidad a granularidad 30s): no existe tabla `traffic_history`/`state_history`/equivalente en `alembic/versions/`.
  - CA-16.9 a CA-16.12 (definiciones operacionales de los 4 KPIs: tiempo espera, longitud cola por dirección, throughput, demora promedio): no existe endpoint `/kpi`/`/reports`/`/analytics`; `grep` para `throughput`/`demora`/`wait_time` en `core_management_api/src/` devuelve cero.
  - CA-16.13 (4 cards principales con valor agregado + gráfico temporal): AnalyticsView muestra UN solo AreaChart genérico con título "Congestión Real vs Predicción", no las 4 cards específicas.
  - CA-16.14 (toggle por dirección para 2 indicadores): ausente.
  - CA-16.20 (HTTP 403 para no-Gerente): sin RBAC.
- **Archivos del repo:**
  - [`frontend_ui/src/components/views/AnalyticsView.tsx`](Tesis/CerebroVial/frontend_ui/src/components/views/AnalyticsView.tsx) (113 líneas, todo mock).
  - `GET /predictions/history/{camera_id}` existe pero es para traffic logs CSV per-cámara, no para KPIs agregados del Gerente.
- **Dependencia bloqueante:** la persistencia F30 (granularidad 30s del estado observado) no existe. Toda la HU es nueva construcción.

#### HU-17 — Vista comparativa entre periodos

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-17.1 a CA-17.16 (todos). Sin vista comparativa, sin lógica de "periodo previo equivalente" (CA-17.4), sin paneles con dos series temporales superpuestas (CA-17.6), sin semántica de mejora/empeoramiento (CA-17.7).
- **Archivos del repo:** ninguno.
- **Dependencia bloqueante:** hereda toda la cadena de HU-16 (F30 inglobada, KPIs, selector de periodo) que tampoco está construida.

## 4. HUs MVP2

### HU-09 — Registro de notas e incidencias del turno

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-09.1 a CA-09.6 (todos). Sin formulario de creación, sin tabla `notes`/`turn_notes`, sin listado paginado con filtros, sin ventana de edición de 24h.
- **Archivos del repo:** ninguno. `grep` para `/notes`/`note_log` en backend y frontend devuelve cero.
- **Dependencia bloqueante:** HU autocontenida; no bloquea ni depende de otras HU. Pura construcción nueva.

### HU-18 — Vista detallada de periodo específico (drill-down del Gerente)

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-18.1 a CA-18.21 (todos). Sin vista drill-down, sin tres carriles temporales integrados, sin acceso desde HU-16/HU-17 (que tampoco existen).
- **Archivos del repo:** ninguno. [`TrafficHistoryWidget.tsx`](Tesis/CerebroVial/frontend_ui/src/components/widgets/TrafficHistoryWidget.tsx) es widget aparte sin relación semántica con el drill-down compuesto.
- **Dependencia bloqueante:** consume 3 registros que no existen: histórico F30 (HU-16), registro decisiones motor (HU-08), registro transiciones estado (TTH-04 CT-04.3). Triple bloqueo estructural.

### HU-19 — Exportación de reportes del Gerente a PDF o Excel

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno con la semántica exigida.
- **CAs parcialmente cubiertos:** ninguno. Existe [`ReportModal.tsx`](Tesis/CerebroVial/frontend_ui/src/components/modals/ReportModal.tsx) pero su semántica es **completamente distinta**: usa Gemini API (`generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025`) para generar un **reporte IA textual** desde una `Alert` del AlertsView, no exportación PDF/Excel de KPIs del Gerente. Es feature huérfana (ver Delta-13).
- **CAs no cubiertos:** CA-19.1 a CA-19.28 (todos). Sin botón "Exportar reporte" en HU-16/HU-17 (que no existen), sin generación PDF, sin generación Excel, sin política conservadora ante fuente caída.
- **Archivos del repo:**
  - [`frontend_ui/src/components/modals/ReportModal.tsx`](Tesis/CerebroVial/frontend_ui/src/components/modals/ReportModal.tsx): integración Gemini para reporte IA de incidente — fuera del scope de HU-19.
- **Dependencia bloqueante:** consume HU-16 y HU-17 que no existen.
- **Deltas:** ver Delta-13.

### HU-20 — Vista comparativa de métricas del modelo predictivo principal vs respaldo

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-20.1 a CA-20.20 (todos). Sin ejecución paralela del modelo de respaldo, sin persistencia paralela (CA-20.2 extiende CA-14.1 que tampoco existe), sin vista comparativa con 4 paneles.
- **Archivos del repo:** RandomForest existe como predictor único activo en `core_management_api/src/prediction/infrastructure/models.py`, pero no hay "modelo principal" (GRU) servido contra el cual compararlo.
- **Dependencia bloqueante:** TTH-09 (GRU) + HU-14 (registro predicciones inglobada) + extensión paralela. Cero de las tres existe.

### HU-21 — Escalamiento de incidentes del Operador al Administrador en operación degradada

- **Estado:** No iniciado.
- **CAs cubiertos:** ninguno.
- **CAs parcialmente cubiertos:** ninguno.
- **CAs no cubiertos:** CA-21.1 a CA-21.37 (todos). Sin botón "Escalar al Administrador" en HU-10/HU-12 (que no existen), sin captura automática de contexto, sin tabla `escalated_incidents`, sin vista de incidentes para Admin con badge de pendientes, sin vista de escalamientos enviados para Operador.
- **Archivos del repo:** ninguno. `grep` para `/incidents`/`/escalat`/`incident_log` devuelve cero.
- **Dependencia bloqueante:** consume HU-10 + HU-12 + CT-04.4 + CT-04.5 de TTH-04, ninguna existe.

## 5. Deltas consolidados

| ID | Descripción | Tipo | HUs/TTH afectadas | Severidad | Acción propuesta |
|---|---|---|---|---|---|
| **Delta-01** | Endpoint canónico `POST /predictions/predict` existe pero con contrato divergente del spec de TTH-09. Actual: `PredictionInput` per-camera (camera_id + 6 features de tráfico crudo) → `PredictionResponse` con niveles discretos Normal/High/Heavy a 15/30/45 min, usando RandomForest. Spec: per-intersección, input series temporales por dirección, output multi-output con ratio continuo + jam_level 0-5 por cada paso del horizonte, usando GRU univariado por dirección. | Conflicto código vs backlog | TTH-09, HU-03, HU-04, HU-14 (consumidores río abajo) | Alta | Sprint 4 debe ejecutar refactor del endpoint y sustituir el modelo (preservando RandomForest como respaldo de Nivel 2 invocable por TTH-04). El refactor toca también el frontend (`predictionService.ts` interfaces) y migraciones (persistencia CT-09.5). |
| **Delta-02** | Inconsistencia de roles entre fuentes. Spec TTH-01 CT-01.3 dice claims `role` con valores `operator/manager/admin` (inglés). Migration `99319147948b_add_users_table` define `role` como `sa.String()` sin enum constraint. AdminView frontend muestra hardcoded "Ana Torres (Admin) / Jorge Salazar (Analista) / Lucía Ramos (Operador)" (español + "Analista" en lugar de "Manager"). Las 3 Personas del producto definidas en BACKLOG_OVERVIEW son "Operador / Gerente / Administrador". | Conflicto entre múltiples fuentes | TTH-01, HU-01, y toda HU que dependa de role-based UI | Media | Cerrar la nomenclatura antes de implementar TTH-01: elegir un set (recomendado: `operator/manager/admin` para claims técnicos, mapeo a labels español en frontend). Documentar en DECISIONS_HU.md. |
| **Delta-03** | CI no corre tests de `edge_device/`, `shared/`, `ia_prediction_service/` ni verificación de tipos (mypy). El propio spec de TTH-03 lo reconoce citando DISCOVERY_2026-05-10 §5.5. Cobertura actual: ruff + pytest core + npm test/lint + docker build. | Backlog define más de lo construido | TTH-03 | Media | Sprint 4 (o sprint dedicado): agregar jobs CI para los 3 módulos faltantes y mypy. Es trabajo de configuración pura, bajo riesgo. |
| **Delta-04** | TTH-08 spec declara reconstruir el módulo de visión "desde cero" preservando solo conceptualmente el código existente. CLAUDE.md del repo establece restricción dura: "NO refactorizar `edge_device/src/vision/`". Estas dos posiciones son contradictorias. Adicionalmente, el código actual implementa CT-08.1 a CT-08.3 + CT-08.7 + CT-08.8 con calidad razonable (tests, DDD), lo que cuestiona la economía del "rebuild desde cero". | Conflicto entre backlog y CLAUDE.md | TTH-08 | Alta | **Resolución (decisión humana, 2026-05-18):** el refactor de visión se ejecutará, pero no inmediatamente. La restricción de CLAUDE.md se levantará cuando el sprint correspondiente lo aborde. Hasta entonces, el código actual queda como base operativa. La auditoría conserva la clasificación "Parcial" de TTH-08 reflejando el estado al día de hoy; el refactor cambia el plan de sprint, no el estado auditado. |
| **Delta-05** | Tabla `vision_aggregates` está planeada (CLAUDE.md D-006 + tareas E18-E21 del TODO) pero no existe en BD ni en código. Las tablas modeladas `vision_tracks`/`vision_flows` (migration `775d2d1db8b4_initial_schema` + `daec5fdcfcdd_timescaledb_hypertables`) tampoco se rellenan; el pipeline persiste a CSV local. | Backlog/migrations declaran más de lo conectado | TTH-08 (CT-08.5) | Media | **Encapsulado como SAN-03 en `specs/001-cerebrovial-mvp/tasks.md` y declarado fuera del Sprint 4 vigente** (ver `REPORTE_PLANIFICACION_SPRINT_4.md` §7 — Trabajos Futuros). La acción técnica al ejecutarse: migration Alembic para `vision_aggregates` + cableado del aggregator del pipeline para persistir a BD. CLAUDE.md prohíbe migrar el pipeline a `vision_tracks`/`vision_flows`; esa restricción se preserva. |
| **Delta-06** | El "dashboard de monitoreo" actual (DashboardView) es un mapa Leaflet con lista de cámaras/intersecciones de Miraflores. La HU-02 describe un panel de **una sola intersección con flujo+cola por acceso direccional**. Son dos semánticas distintas: vista multi-intersección de "supervisión de red" vs. vista intra-intersección "control de cruce". | Conflicto código vs backlog | HU-02, HU-04 | Alta | Sprint 4 debe construir la vista intra-intersección con paneles por acceso. La vista multi-intersección puede preservarse como vista adicional (no descrita en backlog actual), o eliminarse si no es útil para las Personas definidas. Decisión metodológica con estudiante antes de implementar. |
| **Delta-07** | Cero infraestructura de tiempo real: sin SSE, sin WebSocket, sin polling. `DashboardView` hace un único `fetch('/api/intersections')` al montar. Sin canal realtime, las CAs `*.2`/`*.3` de HU-02/HU-03/HU-04/HU-05/HU-06/HU-07 son imposibles. | Backlog declara más de lo construido | HU-02, HU-03, HU-04, HU-05, HU-06, HU-07 | Alta | Sprint 4 debe decidir SSE vs WebSocket y cablear endpoints de eventos (`/predictions/stream`, `/control/strategy/stream`, `/system/events`). Notas técnicas de HU-02/HU-07 sugieren SSE como default razonable. |
| **Delta-08** ✓ **Cerrado por DHU-020 (2026-05-20)** | `ControlView` + subcomponentes implementan UI rica e interactiva tipo "playground": el usuario edita fases (PhaseEditor), presets, slider, presiona Play, recibe `RecommendationPanel` con razonamiento + manejo de `webster_infeasible`. **Pero la semántica es request-response del usuario, no vista pasiva del estado vigente del motor en producción**. HU-05 describe un panel pasivo que muestra qué estrategia está corriendo ahora en la intersección operativa, no un simulador interactivo. | Conflicto código vs backlog | HU-05, HU-06 | Alta | **Resolución cerrada por DHU-020 (`DECISIONS_HU.md`):** prevalece la semántica pasiva de HU-05; el playground se preserva como herramienta de Administrador (no se elimina); Delta-07/08/09 se abordan en un único refactor en bloque; la persistencia de "estado vigente del motor" queda autorizada deliberadamente como cambio estructural (per CLAUDE.md y Art. 20 de constitution). Evidencia consolidada en `DELTA_08_ANALISIS.md`. Ejecución pendiente: Sprint 4 item #4 (HU-05, 3 SP). |
| **Delta-09** | El `reasoning` que produce el motor adaptativo se renderiza en `RecommendationPanel.tsx` etiquetado como "Log técnico (para operador C4)" — lenguaje y formato técnico, no del dominio operativo del Operador (HU-06 exige texto plantillado en lenguaje del dominio, ej. *"Estrategia X activada porque la cola en Norte (38 veh) supera el umbral de 25"*). | Backlog define una semántica que el código no respeta | HU-06 | Media | Sprint 4 debe agregar capa de presentación humana sobre el `reasoning` técnico, con catálogo de plantillas según CA-06.1/CA-06.3. Esto NO requiere tocar el motor (que ya produce el dato bruto); es trabajo de frontend + diccionario de plantillas. |
| **Delta-10** | El motor adaptativo no persiste sus decisiones. No existe tabla `motor_decisions`/`control_decisions`/`decision_log` en migrations. CT-10.9 (persistencia ampliada) está marcado como pendiente en TTH-10. F31 inglobada en CA-08.1 de HU-08 (per Bloque A) no tiene sustrato técnico. | Backlog define más de lo construido | TTH-10 (CT-10.9), HU-08 (CA-08.1), HU-18 (consumidora MVP2) | Alta | Sprint 4 debe crear migration `motor_decisions` (timestamp, intersection_id, modo, reasoning, parámetros, estrategia_anterior, ajustes_mtc) y agregar persistencia en el endpoint `POST /control/recommend`. Resiliencia (CA-08.5) requiere mecanismo de cola/fallback adicional. |
| **Delta-11** | `AlertsView.tsx` existe como vista propia con cards de alertas mock históricas (Congestión Av. Larco, Predicción IA, Latencia Edge). HU-10 exige semánticamente algo distinto: un **banner transversal persistente** visible en todas las vistas, con código de color por nivel (amarillo/naranja/rojo + falla total), no una vista dedicada de notificaciones. Las cards actuales son lo que sería un "centro de notificaciones" tipo dashboard de seguridad, no la alerta operativa exigida. | Conflicto código vs backlog | HU-10 | Alta | Sprint 4 debe construir banner transversal en `App.tsx`/`Header.tsx`. Decidir si `AlertsView` se reaprovecha como vista de "centro de eventos históricos" (sin HU que lo respalde — sería nuevo) o se elimina. Decisión metodológica con estudiante. |
| **Delta-12** | UI scaffolding mock existe para tres HUs del Bloque D sin sustento de datos reales: `AdminView` (HU-13 segunda card "Estado del Sistema" con labels hardcoded ONLINE/IDLE), `AnalyticsView` (HU-14 con recharts AreaChart sobre `trafficData` hardcoded), parcialmente también `AlertsView` (HU-10). Este patrón es **deuda de UI invertida**: se construyó la cáscara visual antes que la integración con backend, sin reflejar los CAs reales de las HUs respectivas (HU-14 exige MAE/RMSE/accuracy/matriz 6×6, AnalyticsView no muestra ninguno). | Deuda invertida | HU-10, HU-13, HU-14 | Media | Sprint 4 puede reaprovechar el scaffolding como punto de partida visual (acelera implementación) pero debe sustituir todos los datos hardcoded y reorganizar componentes para reflejar los CAs específicos. Especialmente AnalyticsView necesita rediseño para mostrar MAE/RMSE/accuracy/confusion matrix, no AreaChart "Real vs Predicción". |
| **Delta-13** | Existen **features huérfanas en frontend** sin HU/TTH que las respalde en el backlog: (a) [`ReportModal.tsx`](Tesis/CerebroVial/frontend_ui/src/components/modals/ReportModal.tsx) invoca Gemini API (`generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025`) para generar "reporte IA" textual de incidentes desde AlertsView — funcionalidad no descrita por HU-19 (que es export PDF/Excel de KPIs del Gerente) ni ninguna otra. (b) [`AIChatWidget.tsx`](Tesis/CerebroVial/frontend_ui/src/components/widgets/AIChatWidget.tsx) "Asistente CerebroVial" — sin HU. (c) [`ThesisModal.tsx`](Tesis/CerebroVial/frontend_ui/src/components/modals/ThesisModal.tsx) — sin HU. Estas features podrían tener valor académico/demo pero no están priorizadas por el backlog y no han sido auditadas contra criterios formales. | Código sin backlog | Ninguna HU formal | Media | Sprint 4 decide explícitamente: (a) preservar como "extras demo" sin criterios de aceptación, (b) elevar a HU formal en una nueva sesión metodológica, o (c) remover si no aportan valor académico. Especialmente sensible: la integración con Gemini API tiene implicaciones de privacidad/datos y debería discutirse antes de mantenerla. |

## 6. Pendientes de consulta humana

Al cierre del Batch B1, ninguno. Todos los 12 elementos clasificados con confianza razonable a partir de evidencia textual y de código. Los conflictos identificados se documentaron como Deltas (sección 5) para resolución metodológica, no como pendientes que bloqueen el avance del audit.

**Nota:** Delta-04 (conflicto TTH-08 spec vs CLAUDE.md sobre refactor de visión) podría considerarse pendiente de consulta humana en lugar de delta. Lo dejo como delta porque la decisión es metodológica de producto, no necesaria para clasificar el estado del código.
