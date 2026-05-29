# TTH-08 — Cierre de Fase 5 + handoff a Fase 6

**Rama**: `feature/tth-08-fase5-application` (desde `master@c0945fa0` = merge PR #30,
Fase 4c).
**Fecha de cierre**: 2026-05-28.
**Estado al cierre**: suite `tests/vision/` en **0 failed / 0 xfailed**, ruff verde
sobre todos los archivos tocados, sin imports de `presentation/` desde
`application/`.

---

## 1. Mapa de cobertura CT-08.11 al cierre de Fase 5

CT-08.11 declara 6 sub-tests de aceptación. Fase 5 cubre parcialmente los que caen en
su alcance (application + infraestructura) y deja explícitamente para Fase 6 los que
dependen de presentación / endpoint / health check.

| Sub-test | Descripción | Cobertura Fase 5 | Falta para Fase 6 |
|---|---|---|---|
| **(a) Detección** | YOLO produce `list[DetectedVehicle]` con tipos canónicos | `tests/vision/test_yolo_detector.py` (Fase 4b). Ningún cambio en Fase 5. | — |
| **(b) Asignación direccional** (zonas / ROI) | `ZoneCounter.count(detections, frame_id)` retorna `dict[ZoneId, ZoneVehicleCount]` con `occupancy` (DHU-025) | `tests/vision/unit/test_zones.py` (9 tests, incluye los 4 casos de occupancy + validación entity) + `tests/vision/integration/test_zone_counter_basic.py` (regresión §6.1 / C1.8) | — |
| **(c) Derivación de métricas** | `compute_traffic_data(...)` produce `TrafficData` canónico §5.4 | `tests/vision/unit/test_compute_traffic_data.py` (11 tests: voto mayoritario, mean_speed count-weighted, flow veh/h, density con/sin segment_length, mean_occupancy desde occupancies, casos vacíos/edge) | — |
| **(d) Integración endpoint `GET /vision/state`** | Endpoint canónico expuesto vía FastAPI | **NO cubierto en Fase 5 (presentation)** | **Fase 6**: implementar endpoint + tests `tests/vision/integration/test_state_endpoint.py` con `httpx.AsyncClient` + repo/aggregator fake. Consume el `get_components()` del builder y los counters del aggregator. |
| **(e) Integración persistencia** | `TrafficData` aterriza en `vision_aggregates` | **Cubierto vía dos rutas complementarias**: (1) `tests/vision/unit/test_postgres_repository.py` (CT-08.5) sin BD viva valida el mapping de columnas; (2) smoke en vivo de 4c contra Timescale del compose (CT-08.11(e) canónico). El test viejo `test_persistence_integration.py` (CSV + shape viejo) fue **eliminado** en 5f por inválido. Fase 5 además cubre `aggregator → repo` con repo fake en `tests/vision/unit/test_async_aggregator.py` (DHU-026 independencia, §11.1, §11.2). | **Fase 6 (opcional)**: smoke e2e ampliado al pipeline entero (no solo el repo) contra Timescale, paralelo al de 4c. |
| **(f) Caída del módulo** | Health check del módulo de visión | **NO cubierto en Fase 5 (presentation)** | **Fase 6**: implementar health check que consume `aggregator.aggregation_errors` y `aggregator.data_dropped` (counters expuestos como properties en 5b, §11.3 listo). Tests `tests/vision/integration/test_health.py`. |

**Resumen**: (a), (b), (c), (e) cubiertos al cierre de Fase 5. (d) y (f) son Fase 6
porque dependen de la capa de presentación.

---

## 2. DHUs abiertos en Fase 5

| DHU | Sub-fase | Qué reabrió | Razón |
|---|---|---|---|
| **DHU-025** | 5a | Fase 3 (`domain/entities.py` — extensión de `ZoneVehicleCount` con `occupancy: float ∈ [0.0, 1.0]`) + Fase 4a (`infrastructure/zones/zone_counter.py` — cómputo geométrico) | `TrafficData.mean_occupancy` (§5.4) requiere overlap bbox∩polígono; el Protocol cerrado en Fase 3 no exponía occupancy. Se eligió opción A (extender la entity) sobre B/C por DDD-ortodoxo. Incluye **decisión interpretativa de §5.4**: `Σ` se lee como **unión** (no suma + clip), preservando el rango `[0.0, 1.0]` sin clip y la semántica "fracción cubierta". Nota cruzada agregada a §5.4 del doc de diseño en 5f. |
| **DHU-026** | 5b (paso 0) | Fase 3 (`domain/protocols.py` — docstring del Protocol `AsyncAggregator`) + nota cruzada en §4.4 Cambio 2 del doc de diseño | Conflicto interno del diseño: §4.4 Cambio 2 (Sesión 1) y el docstring del Protocol decían "caller persiste"; §11 (Sesión 3) hablaba del "worker" capturando errores de `save`. Se eligió opción (a) = worker-persiste (las (a) y (b) no difieren en runtime; (b) era más arquitectura por la "letra" de §4.4). Incluye **refinamiento de Cesar**: save y push a output queue son **paths independientes best-effort**, no secuenciales (`flush()` retorna lo computado-y-no-dropeado, no lo persistido). |

Ambos DHU registran la apertura formal del cambio en `DECISIONS_HU.md` con índice
actualizado, fila Bloque E del Resumen de impacto, y "Última actualización" del header.

---

## 3. Deudas y decisiones autónomas que quedan para fases siguientes

| Item | Sub-fase de origen | Estado | Acción pendiente |
|---|---|---|---|
| **Pre-buffering del `AsyncVisionPipeline.run()`** | 5c | Eliminado. Comentario explicativo agregado in-code donde estaba el bloque. | Validar con **stream real** en smoke Fase 6 / Fases 7-9. Si aparece micro-jitter visible a 30 FPS con streams HLS reales, considerar re-introducir un pre-buffer condicionable por config con timeout total para no romper shutdown limpio. |
| **`opencv-python` vs `opencv-python-headless` para producción** | 5a → 5b (C7.7) | Documentado en `TODO.md` C7.7. El venv local tiene headless para correr tests; `requirements.txt` mantiene `opencv-python` para que el script de calibración (`interaction.py`, usa GUI cv2) siga funcionando. | Tarea propia más adelante: auditar Dockerfile de `edge_device`, decidir empaquetado por target (producción headless vs developer con GUI), separar `interaction.py` en venv del developer o reescribirlo sin GUI. Fuera de TTH-08. |
| **Cobertura `catch-up §10.5`** | 5c → 5f | Agregada en 5f: `test_catch_up_skips_frames_when_lag_exceeds_threshold` + `test_catch_up_inactive_when_no_lag`. Manipulan `_latest_capture_ts` + frame_queue manualmente para forzar lag artificial sin depender de timing real. | — (cerrado). |
| **Nota cruzada §5.4 → DHU-025** | 5a → 5f | Agregada en 5f (una línea en la tabla de §5.4). | — (cerrado). |
| **`DetectionProcessor` eliminado** | 5d | Código muerto preexistente, ningún caller lo importaba. | — (decisión limpia, no deuda). |
| **Tests `test_aggregation_consistency.py` (sync) y `test_persistence.py` (CSV)** | 5b, 5f | Migrado el primero al shape §5.4 + AsyncTrafficAggregator (5b); eliminado el segundo en 5f (CSV inválido, cobertura está en `test_async_aggregator.py` + smoke en vivo de 4c). | — (cerrado). |

---

## 4. Qué queda para Fase 6 (presentación)

Fuera del alcance de Fase 5, explícitamente declarado en el plan y en este handoff:

1. **Adapter §5.8** (`TrafficData` → `CameraTrafficData`): mapeo para que consumidores
   actuales (BD legacy, predictor, payload SSE viejo) sigan funcionando durante la
   transición. Vive en `presentation/`.
2. **Endpoint `GET /vision/state`** (CT-08.11(d) + CT-08.6): expone el estado actual
   del pipeline (último `FrameAnalysis` + telemetría del aggregator). Consume el
   `get_components()` del builder y los `aggregator.get_latest()` / `aggregator.flush()`.
3. **Health check del módulo** (CT-08.11(f) + CT-08.10): expone counters
   `aggregation_errors` y `data_dropped` del aggregator (ya disponibles como
   properties desde 5b, §11.3 listo).
4. **`OpenCVVisualizer` adaptado al Protocol `FrameRenderer`**: renombrar `.draw()` a
   `.render(frame, analysis) -> np.ndarray`. Hecho esto, el builder lo inyectará al
   `MultiCameraManager` desde fuera (5e ya lo dejó listo: `add_camera(...,
   renderer=...)` y `CameraInstance.__init__(..., renderer=None)`).
5. **`Broadcaster` concreto** contra el Protocol nuevo: `publish(data: TrafficData)`
   + `subscriber_count()` + `is_subscribed(subscriber_id)` (§6.10 / §6.11). Eliminar el
   acceso a `_subscribers` desde `routes/streaming.py` y tests.
6. **Migración de consumidores** que aún esperan el shape viejo (frontend, predictor,
   csv_loader) — enumerados en §5.8 del doc de diseño. Cada consumidor se actualiza en
   Fase 6 según el adapter que se elija (directo al schema nuevo o vía
   `CameraTrafficData`).

---

## 5. Smoke en vivo recomendado para Fase 6

Reproducible con `invoke up` + un video file o stream real:

1. Configurar `vision.persistence.enabled: true`, `vision.persistence.type: 'postgres'`,
   `vision.camera_id: '<algun id>'`, `vision.persistence.interval_seconds: 5.0`.
2. Levantar el compose con Timescale.
3. Correr el pipeline (a través del entry point del módulo de visión, una vez Fase 6
   lo cablee).
4. Verificar en `vision_aggregates` que los `TrafficData` aparecen con el schema §5.4
   y con `mean_occupancy` distinto de 0 cuando hay vehículos en zona.
5. Verificar el endpoint `GET /vision/state` con un cliente HTTP simple.
6. Verificar el health check: con repo intencionalmente caído (e.g., parar Timescale
   por unos segundos), el counter `aggregation_errors` debe subir y el módulo NO debe
   morir (§11.1).
