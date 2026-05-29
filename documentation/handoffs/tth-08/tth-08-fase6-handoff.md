# TTH-08 — Cierre de Fase 6 + handoff a Fases 7-9

**Rama**: `feature/tth-08-fase6-presentation` (desde `master@310a383a` = merge PR #31,
Fase 5).
**Fecha de cierre**: 2026-05-29.
**Estado al cierre**: suite `tests/vision/` en **120 passed / 0 failed / 0 xfailed**;
suite frontend Vitest en **125 passed / 0 failed**; ruff verde sobre `edge_device/`;
ESLint sin errores en `frontend_ui/src/`. Validación en vivo: pipeline contra `.mp4`
local (volumen `/app/videos/:ro`), SSE §6.2 emitido a `interval_seconds: 5`, frontend
mostrando `CameraDetailView` + `DashboardView` con discretización ES client-side, video
MJPEG fluido con bbox + zonas anotadas.

---

## 1. Mapa de cobertura CT-08.11 al cierre de Fase 6

CT-08.11 declara seis sub-tests de aceptación. Fase 6 cierra los dos que dependían de
presentación y deja explícitamente diferida la integración con BD viva (CT-08.11(e) e2e)
a Fases 7-9, donde los CT pasan de "conformes" a "validados".

| Sub-test | Descripción | Estado al cierre de Fase 6 |
|---|---|---|
| **(a) Detección** | YOLO produce `list[DetectedVehicle]` | Cubierto desde Fase 4b (`tests/vision/test_yolo_detector.py`). |
| **(b) Asignación direccional** | `ZoneCounter.count()` con occupancy DHU-025 | Cubierto desde Fase 4a (`tests/vision/unit/test_zones.py` + `integration/test_zone_counter_basic.py`). |
| **(c) Derivación de métricas** | `compute_traffic_data()` produce `TrafficData` §5.4 | Cubierto desde Fase 5b (`unit/test_compute_traffic_data.py`, 11 tests). |
| **(d) Integración endpoint `GET /vision/state`** | Shape §6.5 + branch 5xx CT-08.10 | **Cerrado en 6e**: `integration/test_state_endpoint.py` (6 tests: 404, 503, 200 warm-up, 200 con shape §6.5, intersection_id-serialization, timestamp=max(window_end)). |
| **(e) Integración persistencia automatizada** | `TrafficData` aterriza en `vision_aggregates` | **Cubierto en dos rutas complementarias al cierre de Fase 6**: (1) `unit/test_postgres_repository.py` valida mapping de columnas sin BD viva; (2) smoke en vivo de 4c contra Timescale. **El e2e pipeline→Postgres real (BD viva) queda diferido a Fases 7-9** (deuda nominada). En Fase 6 se agregó `integration/test_pipeline_wiring.py` que cierra la deuda de proceso (alineación config↔código), no la de persistencia real. |
| **(f) Caída del módulo** | Health check + 5xx en `GET /vision/state` | **Cerrado en 6f + 6e**: `integration/test_health.py` (10 tests: 503 sin cámaras / sin frames / sin recientes; Degradado por errores / drops / fleet parcial; OK; payload estructurado; aggregator None) + branch 5xx del state endpoint cubierto en `integration/test_state_endpoint.py::test_returns_503_when_camera_not_running_and_no_data`. |

**Resumen**: (a)–(d), (f) cerrados con tests automatizados al cierre de Fase 6.
(e) cerrado para shape de persistencia y wiring; el e2e contra Postgres vivo es
deuda nominada de Fases 7-9.

---

## 2. Divergencias diseño-vs-código registradas en Fase 6

Mismo patrón de registro que DHU-025 (Σ→unión en §5.4) y DHU-026 (caller→worker en
§4.4/§11). Documentadas in-doc (no son DHUs nuevos porque son **decisiones de no-construir**,
no de reabrir un Protocol).

| Divergencia | Sub-fase | Resolución |
|---|---|---|
| **Adapter §5.8 a `CameraTrafficData` obsoleto** | 6a | Nota al inicio de §5.8 documentando que la migración Fase 2/3 + `PostgresTrafficRepository._to_row()` (Fase 4c) volvieron obsoleto el adapter como objeto intermedio. `vision_aggregates` ya está en shape canónico; el mapping real vive en `_to_row()` sin intermediario. La tabla de mapeo de §5.8 queda como referencia histórica, no contrato vivo. `CameraTrafficData` (en `shared/cerebrovial_shared/schemas/camera.py`) queda **huérfana** sin consumidor runtime — borrado nominado fuera de TTH-08 (territorio común). Comentario corto agregado en `_to_row()` apuntando a la nota. |
| **`camera.street_monitored: null` en payload §6.2** | 6d → 6g | Nota agregada al inicio de §6.2 al cierre del paso-0 de 6g. La premisa original de §6.2 era "el frontend lo necesita para HU-02", pero auditoría del frontend mostró cero consumidores (HU-02 cableada a SUMO en MVP1 per D-007/§10.4). Enriquecer en el broadcaster sería YAGNI. Implementación 6d emite `null`; frontend lo ignora. **Dos rutas siguen abiertas para HU futuras** sin tocar dominio: (i) `CameraMetadataProvider` inyectable en broadcaster (volver a §6.2 al pie), (ii) registry en frontend. |

---

## 3. Lo que Fase 6 entregó (presentation + frontend, validado en runtime)

| Sub-fase | Commits | Entregable |
|---|---|---|
| **6a** | `42b5e603` | Registro de la divergencia §5.8 (no construcción de adapter). Nota en `tth-08-fase1-diseno.md` + comentario en `_to_row()`. |
| **6b** | `fcc8d1c6` | `OpenCVVisualizer` contra Protocol `FrameRenderer` §7.2 (`render(Frame, FrameAnalysis) -> np.ndarray`, no muta entity). Factory `build_visualizer_from_vision_cfg()`. Cableado en `run_server.py` y POST `/cameras/{id}`. Barrido de bit-rot Fase 3 (`total_count`/`raw_detection_count`/`vehicle_count` → `len(analysis.vehicles)`/`zvc.count`). 9 tests nuevos. |
| **6c** | `7ee8ac0d` | `legacy_api.py` borrado (huérfano confirmado). Tres propiedades públicas en `RealtimeBroadcaster` (`subscribed_cameras()`, `latest_state(camera_id)`, `latest_states()`). `streaming.py` consume las propiedades. Cero acceso a `_subscribers`/`_latest_state` desde presentation. |
| **6d** | `5eab1383` | `RealtimeBroadcaster` reescrito contra Protocol §6.10/§6.11 (`publish(TrafficData)`, `subscriber_count()`, `is_subscribed()`). Payload §6.2 agrupado. Prohibiciones §9.2 cumplidas (sin strings ES, sin %, sin umbrales 30/70). Callsite reorientado de por-frame a por-ventana (drena `aggregator.flush()` y publica). Cruce thread→async resuelto vía `queue.Queue` thread-safe + coroutine main; sin `run_coroutine_threadsafe`. 12 tests reescritos sin acceso a privados. |
| **6e** | `be2caabf` | `GET /vision/state/{intersection_id}` con shape §6.5 (`{intersection_id, timestamp, directions[]}`). Branch 5xx (CT-08.10): 404 si intersection_id desconocida, 503 si cámara no procesa, 200 warm-up con `directions=[]`, 200 con shape §6.5. Cache `_latest_traffic_data: dict[camera_id, dict[zone_id, TrafficData]]` en broadcaster + helper `traffic_data_for()`. 6 tests. |
| **6f** | `4188897d` | `GET /vision/health` separado (endpoint distinto de `/vision/state` per CT-08.10). Payload estructurado `{status, checked_at, cameras{}}`. Estado discreto worst-of-fleet OK/Degradado/Fuera de servicio. Consume `aggregator.aggregation_errors`, `aggregator.data_dropped` (§11.3) y `pipeline.get_latest()` para frame freshness. 503 cuando fuera de servicio. 10 tests. |
| **6g** | `e3e6ef24` | Frontend SSE migrado al shape §6.2: listeners de `CameraDetailView` + `DashboardView` parsean payload agrupado y consumen `event: traffic_update`. Tipo `frontend_ui/src/types/visionStream.ts`. Helper `frontend_ui/src/utils/trafficLabels.ts` con discretización ES (`congestionLabel`, `congestionUiStatus`, `densityPercent`) — umbrales 0.30/0.70 sobre `mean_occupancy`. `predictionService.ts` y `TrafficHistoryWidget.tsx` NO se tocan (invariante 1 sostenida + Caso A confirmado en paso-0). Nota agregada a §6.2 sobre `street_monitored: null`. 22 tests nuevos en Vitest. |

### Bug-fix de bit-rot destapado al validar Fase 6

| | Commit | Entregable |
|---|---|---|
| **Fix** | `76e4ef0c` | Alineación config↔código de Fase 5: `default.yaml` (`source_type: auto`, `camera_id: null` declarado, `persistence.type: postgres`, `output_dir` eliminado, `interval_seconds: 5` con comentario inline registrando el rationale para multi-cámara), `config_models.py` (`VisionConfig.source_type` default `auto`, `camera_id: Optional[str]` agregado), `run_server.py` (`cam_cfg.vision.camera_id = cam_info["id"]` por cámara). |
| **Test** | `e1ec802f` | `integration/test_pipeline_wiring.py` — wiring test del builder real con config representativa. Atrapa los cuatro bit-rots de hoy (csv legacy en config, camera_id ausente, ZoneCounter firma desalineada, AsyncTrafficAggregator firma desalineada). Confirmación empírica del valor inyectando cada bit-rot temporalmente y observando el rojo (ver mensaje del commit). 4 tests. |

### Validación visual en vivo (2026-05-29)

Reproducible con `invoke up-build --service=edge_device` + `.mp4` local en `./videos/`
(después de aplicar el bootstrap de validación local — ver §6 abajo).

- ✓ Endpoint `GET /vision/state/{intersection_id}` retorna shape §6.5 con 200 cuando hay
  TrafficData publicado; 503 cuando la cámara no procesa frames.
- ✓ Endpoint `GET /vision/health` retorna estado discreto OK/Degradado/Fuera de servicio
  con telemetría por cámara.
- ✓ SSE `/stream/{camera_id}` emite payload §6.2 puro (sin `density "X%"`, sin
  `congestion_level` ES, sin umbrales hardcoded). Cadencia: 5s. Sentido visual:
  **vivo** (12 actualizaciones por minuto se sienten responsivas).
- ✓ Frontend `CameraDetailView`: barra de congestión muestra `Bajo`/`Moderado`/`Alto`
  (computado client-side desde `mean_occupancy`), densidad formateada como `X%`.
- ✓ Frontend `DashboardView`: markers del mapa cambian color (verde/ámbar/rojo) según
  `congestionUiStatus(mean_occupancy)`.
- ✓ Video MJPEG `/video/{camera_id}?type=processed` fluido con bbox + zonas anotadas
  por el `OpenCVVisualizer` cableado en 6b.
- ✓ Persistencia: `vision_aggregates` recibe rows con shape canónico en cada ventana.
  Al EOF del `.mp4` el panel se congela (loop no soportado, ver deuda §5).

---

## 4. Deuda nominada para Fases 7-9 y posteriores

Categorizada por origen y dueño.

### 4.1 Integración front-back (HU-02 / panel de operador)

Detectadas durante la validación visual del paso-0 de 6g. Son brechas entre lo que el
backend de vision emite y lo que el frontend necesita para un panel "completo" —
ortogonales a Fase 6 (que cerró el contrato del shape) pero parte del trayecto hacia HU-02 real.

| Deuda | Detalle | Dueño / Fase |
|---|---|---|
| **Cámaras hardcodeadas en `run_server.py`** | Las 4 URLs YouTube viven en código (`scripts/run_server.py:23-28`); el frontend no descubre vía `/cameras/status`. Contra §6.4 (discovery dinámico). | HU-02 / Fases 7-9. |
| **Endpoint POST `/cameras/{id}` roto** | Hardcodea `persistence.type: "csv"` ([routes/cameras.py:84](edge_device/src/vision/presentation/api/routes/cameras.py#L84)) que el builder de Fase 5b rechaza. No usa zonas del YAML. Defaults distintos a producción (`conf_threshold: 0.5` vs `0.3`). Sub-fase aparte de presentation: alinear con builder real + usar config del perfil activo. | TTH-08 deuda residual o post-Fase 6. |
| **KPIs mock en `DashboardView`** | Cards "Vehículos detectados (Hora) 1,245", "Velocidad Promedio 22 km/h", "Predicción Congestión ALTA", "Semáforos Conectados 34/34" son strings hardcodeados ([DashboardView.tsx:160-187](frontend_ui/src/components/views/DashboardView.tsx#L160)). | HU-02. |
| **UI vieja con campos no emitidos por visión** | `CameraDetailView` muestra "Peatones" e "Incidentes" que §5.7 declara fuera del scope MVP1 (el broadcaster no los emite). Los listeners de 6g los setean en 0; las cards quedan vacías. | HU-02 o sub-fase de cleanup UI. |
| **"Insights de IA (CerebroVial)" en strings inglés mock** | `prediction.message` ([CameraDetailView.tsx:242](frontend_ui/src/components/views/CameraDetailView.tsx#L242)) viene del predictor del core; en inglés. Localización + alineación con D-009 (jam level 0-5 Waze). | HU-02 / TTH-09. |

### 4.2 Calibración / datos de demo

| Deuda | Detalle | Dueño |
|---|---|---|
| **Zonas default no calibradas para videos reales** | `conf/vision/default.yaml` define cuatro zonas (`zone1` Av. Javier Prado real, `zone2`–`zone4` con polígonos cuadrados [100,100]-[1180,620] genéricos). Al validar con un MP4 arbitrario, los conteos no cuadran "a ojo" (el módulo cuenta bien; la métrica es por-ventana de 5s, no instantánea, y las zonas de demo no corresponden al video). | Calibrar con `scripts/calibrate_zones.py` por video específico antes de demos en vivo. |

### 4.3 TTH-08 — items diferidos explícitamente

| Deuda | Estado | Cierre planeado |
|---|---|---|
| **CT-08.11(e) e2e pipeline→Postgres vivo** | Test de wiring (`test_pipeline_wiring.py`) cubre alineación config↔código; el e2e con BD viva (`vision_aggregates` recibiendo rows del pipeline real, no del repo directo) queda diferido. | Fases 7-9 (CT pasan de "conformes" a "validados"). |
| **`camera.street_monitored: null`** | Divergencia §6.2 registrada (ver §2). Implementación benigna en MVP1. | Si un consumidor lo requiere en F41 / HU futura: `CameraMetadataProvider` inyectable o registry frontend. |
| **Loop del video source** | NO soportado en `OpenCVSource.read()`; al EOF retorna `None` y el pipeline para. El panel se congela. | Si se quiere validación sostenida con `.mp4` corto en Fases 7-9, agregar flag `loop: bool` en `SourceConfig` (~5 líneas en `video_source.py`). |
| **C7.7 `opencv-python` vs `opencv-python-headless`** | Documentado en `TODO.md` C7.7 desde Fase 5a. Venv local con headless para tests; `requirements.txt` mantiene `opencv-python` para `interaction.py` (GUI). | Auditar Dockerfile de edge_device, decidir empaquetado por target (producción headless vs developer GUI). Fuera de TTH-08. |
| **`CameraTrafficData` huérfana en `shared/`** | Documentada en §5.8 al cierre de 6a. Sin consumidor runtime. | Borrado coordinado de `shared/` (territorio común). Item separado, no urgente. |

### 4.4 Otros módulos — afectados colateralmente

| Deuda | Detalle | Dueño |
|---|---|---|
| **Predictor lee CSVs muertos** | `core_management_api/src/prediction/infrastructure/csv_loader.py` lee `traffic_log_*.csv` cuyo writer Fase 5f eliminó. No lee de `vision_aggregates`. Consistente con MVP1 SUMO-no-visión (D-007 + §10.4): el sistema se alimenta de SUMO, no de visión; el predictor RandomForest queda como fallback de Nivel 2. Alimentar el RF fallback **pertenece a TTH-04 (cascada) / TTH-09 (modelo)**, no a TTH-08. | TTH-04 / TTH-09. |
| **6h migración predictor (`PredictionInput` shape viejo)** | Migrar `schemas.py`, `predictor.py`, `csv_loader.py`, `engine.py` al shape canónico forzaría retraining del RandomForest que TTH-09 reemplaza con GRU. Trabajo negativo en MVP1. | Post-GRU (TTH-09). |
| **Migración predictor + scripts offline** (`generate_training_data.py`, `generate_camera_data.py`, `train_models.py`) | Mismos motivos que 6h. Mecánicos pero fuera de scope TTH-08. | Cuando el predictor migre. |

### 4.5 Patrón de proceso — bit-rot config/bootstrap (cerrado)

**Causa raíz**: Fase 5 cambió contratos del builder (`persistence.type` solo Postgres,
`vision.camera_id` requerido, `ZoneCounter.__init__` sin `resolution`,
`AsyncTrafficAggregator` con worker-persiste) **sin actualizar consumidores de config
ni tener cobertura de test que ejercitara el builder real con `persistence.enabled=True`**.
Cuatro desincronizaciones distintas se acumularon hasta la validación en vivo de Fase 6.

**Cerrado por**:
- Commit `76e4ef0c` (fix de las cuatro desincronizaciones).
- Commit `e1ec802f` (test `test_pipeline_wiring.py` que atrapa los cuatro bit-rots con
  confirmación empírica inyectando cada uno temporalmente). El test **construye el builder
  real** sobre config representativa; mockea solo recursos externos (YOLO weights, cv2 con
  URL, tracker que carga torch). Cualquier desalineación futura de firma o config se atrapa
  en CI antes de tocar Docker.

**Raíz no cerrada — nominada para resolución arquitectónica**:

> **`VisionConfig` dataclass existe pero no está registrado con `ConfigStore`.**
> El YAML carga "abierto" (sin enforcement contra el schema), entonces el dataclass es
> documentación que se desincroniza silenciosamente del runtime. Eso es structured config
> a medias y es **exactamente la clase de desincronización contrato-código** que causó
> los bit-rots de Fase 5. Hoy se mitigó con el wiring test, pero la causa raíz queda.
> **Dos rutas, decisión diferida**:
> (i) **Activar enforcement** vía `cs.store(name="vision_config", node=VisionConfig)` en el
>     entrypoint de Hydra. Pro: el YAML no puede divergir del dataclass; cualquier campo
>     nuevo en código requiere campo nuevo en YAML. Contra: rigidez — overrides ad-hoc
>     de Hydra requieren declararse en el dataclass.
> (ii) **Documentar `VisionConfig` como type-hint decorativo** explícitamente en su
>      docstring + agregar en el dataclass un comentario indicando que la validación real
>      vive en `pipeline_builder.build_*()`. Pro: simple, preserva flexibilidad. Contra:
>      no previene futuros bit-rots — confía en disciplina del dev + el wiring test.

---

## 5. Decisiones autónomas de Fase 6 (registradas para Fases 7-9)

| Decisión | Sub-fase | Razón |
|---|---|---|
| **`interval_seconds: 5`** como default en `conf/vision/default.yaml` | Validación de 6g | 5s × 30 FPS = ~150 frames/ventana, estadísticamente estable. Validado visualmente como "vivo" para 1 cámara. **Reconsiderar para multi-cámara** (12 cámaras × 12 ventanas/min = 144 INSERTs/min a Timescale). Comentario inline en el YAML registra el rationale. |
| **`source_type: "auto"`** como default | 6 (bug-fix `76e4ef0c`) | El dispatcher de `infrastructure/sources/__init__.py` matchea URLs YouTube por contenido — `auto` preserva el comportamiento de producción sin restringirlo a un único tipo. |
| **No revertir volumen `./edge_device/conf:/app/conf:ro`** en `docker-compose.yml` (no se commitea) | Cierre Fase 6 | Convención del repo: `docker-compose.dev.yml` es el lugar canónico para overrides de iteración. Mezclar dev-tooling en el compose principal diluye la convención. Si en Fases 7-9 se quiere iterar config sin rebuild, va a `docker-compose.dev.yml`. |

---

## 6. Bootstrap de validación local (no commiteado a la rama)

Para reproducir la validación con `.mp4` local en lugar de YouTube (los streams están
bloqueados desde el contenedor edge_device):

1. **Volumen `videos/`** en `docker-compose.yml` bajo `edge_device` (no commiteado al
   merge — es config de iteración, va a `docker-compose.dev.yml` en una fase futura
   si se quiere consolidar):
   ```yaml
   volumes:
     - ./videos:/app/videos:ro
   ```

2. **Cámara única en `run_server.py`** (bloque marcado con banderas
   `▼▼▼ VALIDACIÓN LOCAL ▲▲▲` para que el revert sea explícito):
   ```python
   CAMERAS = [
       {"id": "cam_larco_schell", "source": "/app/videos/trafico.mp4"},
       # URLs YouTube originales comentadas.
   ]
   ```

3. **Poner el video**: `mkdir -p ./videos && cp trafico.mp4 ./videos/`.

4. **Rebuild + recreate**:
   ```bash
   docker compose build --no-cache edge_device
   docker compose up -d --force-recreate --no-deps edge_device
   ```

El `.mp4` debe durar al menos 60s (ventana × 12 actualizaciones) para una validación
significativa del SSE. Al EOF el panel se congela (loop no soportado).

---

## 7. Estado de la rama al cierre

- **Branch**: `feature/tth-08-fase6-presentation`.
- **Working tree clean** (validación local revertida tras la sesión; queda `videos/`
  como untracked `.gitignore`-equivalente — no afecta el merge).
- **Commits desde `master@310a383a`** (9 totales):
  1. `fcc8d1c6` — 6b visualizer + Protocol FrameRenderer.
  2. `7ee8ac0d` — 6c limpieza (legacy_api borrado, privados detrás de propiedades).
  3. `42b5e603` — 6a registro divergencia §5.8.
  4. `5eab1383` — 6d broadcaster + Protocol §6.10/§6.11 + payload §6.2.
  5. `be2caabf` — 6e `GET /vision/state` + branch 5xx.
  6. `4188897d` — 6f `GET /vision/health` separado.
  7. `e3e6ef24` — 6g frontend SSE §6.2 + helper client-side + nota §6.2.
  8. `76e4ef0c` — fix bit-rot config/bootstrap de Fase 5.
  9. `e1ec802f` — test wiring que atrapa los cuatro bit-rots.

**Pre-requisitos antes de mergear** (decisión humana, no del agente):
- Verificar coordinación con la rama HU-14 de Andrés (componentes compartidos de
  `shared/`, frontend).
- Abrir PR con este handoff como descripción.
- El agente **no mergea ni hace push** — esa decisión queda fuera de su scope.

Con Fase 6 mergeada, TTH-08 entra en su tramo final: **Fases 7-9 (validación de
dataset + documentación contractual)**, donde los CT pasan de "conformes" a "validados"
contra datos y BD vivos.
