# TTH-08 Fase 0 — Lecciones de la auditoría crítica del módulo `vision`

Auditoría de lectura completa de `edge_device/src/vision/` realizada el
2026-05-27 para alimentar el diseño de Fase 1 (DDD + contratos). El módulo
permanece intocable durante toda Fase 0 según regla CLAUDE.md; este
documento es la única escritura permitida.

## 1. Inventario crudo del estado actual

**52 lecturas totales** = 32 archivos productivos + 14 tests + 1 conftest
compartido + 5 configs YAML.

### 1.1 Módulo productivo (32 archivos `.py`)

| Capa | # archivos | Notas |
|------|------------|-------|
| Raíz (`vision/__init__.py`) | 1 | Vacío — no exporta API pública. |
| `domain/` | 4 | Entidades, protocolos, repositorios. Sin imports salientes. |
| `application/` | 8 | Processors (chain), aggregators (sync+async), pipelines (sync+async), builder, multi_camera. |
| `infrastructure/` | 13 | Sources, detection (YOLO), tracking, zones, broadcast, persistence + un `interaction.py` que en realidad es UI. |
| `presentation/` | 6 | API FastAPI (3 routers), visualizer OpenCV, `legacy_api.py` roto. |

**Discrepancia con el encuadre**: el encuadre cita 36 archivos como
aproximación no verificada; el conteo real es **32**. Adicionalmente
faltan 13 `__init__.py` en subpaquetes
(`application/`, `application/aggregators/`, `application/builders/`,
`application/pipelines/`, `application/services/`, los 5 subpaquetes
de `infrastructure/` excepto `sources/`, `presentation/`,
`presentation/api/routes/`, `presentation/visualization/`). El módulo
funciona porque Python 3 trata estos directorios como
*namespace packages*; el problema es de consistencia declarativa.
Adicionalmente, los namespace packages son frágiles (resolución
inconsistente con herramientas de packaging, mypy y empaquetado).
En el módulo nuevo, todos los subpaquetes deben tener `__init__.py`
explícito.

### 1.2 Tests (14 archivos + `conftest.py`)

Viven en `edge_device/tests/vision/`, fuera del módulo productivo:
1 archivo suelto, 3 en `integration/`, 10 en `unit/`.
El `conftest.py` compartido vive en `edge_device/tests/conftest.py`
y expone fixtures `mock_frame` y `mock_analysis` triviales.

**6 tests `xfail` confirmados** (todos con razón documentada y TODO
de tracking):

| Test | TODO | Causa |
|------|------|-------|
| `test_smart_detection.py::test_interpolation_logic` | C1.7 | Tests funcionalidad **eliminada del productivo** (interpolación). |
| `test_smart_detection.py::test_trajectory_update` | C1.7 | Tests `_vehicle_trajectories`, atributo **muerto** en productivo. |
| `test_async_pipeline.py::test_pipeline_processing_flow` | C1.5 | **Bug real productivo**: race condition `stop()` vs processing thread. |
| `test_multi_camera_manager.py::test_add_camera` | C1.6 | Test desactualizado: `camera.camera_id` se movió a `camera.state.camera_id`. |
| `test_multi_camera_manager.py::test_start_stop_camera` | C1.6 | Mismo issue + mock incompleto de cv2.putText. |
| `test_zones.py::test_zone_manager_update` | C1.8 | **Bug real productivo**: zone_counter no detecta vehículos dentro del polígono. Ver §6. |

### 1.3 Configs YAML (5 archivos)

`edge_device/conf/vision/`: `default.yaml`, `balanced.yaml`,
`low_latency.yaml`, `javier_prado.yaml`, `vehicle_classes.yaml`.

Existe copia con los **mismos nombres** en
`core_management_api/conf/vision/`. **Contenido no verificado en Fase 0**
(por decisión expresa al cerrar el plan).

## 2. Patrones bien hechos a replicar

- **Capa de dominio limpia**: `domain/` no importa nada de
  application/infrastructure/presentation. Usa `typing.Protocol`
  (structural subtyping) para invertir dependencias. Este invariante
  debe mantenerse en el módulo nuevo.

- **Excepciones tipadas** importadas desde
  `cerebrovial_shared.exceptions`: `SourceError`, `DetectionError`. Las
  capas internas no atrapan `Exception` desnudo cuando hay un tipo
  específico disponible.

- **Pydantic + field_validator en config de sources**
  (`infrastructure/sources/base.py:9-21`): valida resolución par,
  rangos. Patrón a usar para todos los configs estructurados del
  módulo nuevo.

- **Factory + Registry para sources**
  (`infrastructure/sources/__init__.py`): selección dinámica por tipo
  o por matching del URL. Decoupling correcto entre quién construye
  y quién consume.

- **Chain of Responsibility para procesamiento por frame**: el patrón
  `FrameProcessor` con `set_next` es la abstracción correcta; el problema
  está en dónde vive el código y a qué se acopla (ver §4).

- **Worker thread + queue para persistencia no-bloqueante**
  (`async_aggregator.py:178-198`): el pipeline principal nunca espera I/O.

- **Smart catch-up logic** en async pipeline
  (`async_pipeline.py:118-138`): cuando el lag de red sube, se descartan
  frames de procesamiento manteniendo la última captura. Apropiado para
  streams HLS/YouTube.

- **Pub/sub asyncio con cache de latest state**
  (`realtime_broadcaster.py:36-42`): un nuevo subscriber recibe el estado
  actual inmediatamente al suscribirse, sin esperar al próximo broadcast.

- **Majority vote para estabilizar identidades**: el tracker
  (`supervision_tracker.py:58-70`) y el aggregator
  (`sync_aggregator.py:80-94`) usan el mismo patrón para resolver
  ambigüedad de clase a lo largo del tiempo. El patrón es correcto;
  está duplicado y debería ser una utilidad común.

- **`assert_real_binary` para validar LFS** antes de cargar el modelo
  (`yolo_detector.py:27`): impide cargar pointers de Git LFS como si
  fueran pesos reales. Convención del proyecto, replicarla.

- **Detector usa `setup_logger` + `log_execution_time`** de
  `cerebrovial_shared.logging` (`yolo_detector.py:8, 33, 35`). Es el
  **único archivo del módulo que loggea correctamente**; debería ser
  el estándar para todos los demás (ver §3.6).

## 3. Patrones mal hechos a NO replicar

Las siete observaciones que afectan decisiones de Fase 1 van con párrafo
propio (§3.1–§3.7). El resto (§3.8) va en línea.

### 3.1 `presentation/legacy_api.py` es código roto e inalcanzable

El archivo importa `from ..application.pipeline import VisionPipeline`
(`legacy_api.py:7`) y `from ..infrastructure.visualization import
OpenCVVisualizer` (`legacy_api.py:8`). **Ninguno de los dos paths
existe**: el pipeline real está en `application/pipelines/sync_pipeline.py`
y el visualizer en `presentation/visualization/opencv_visualizer.py`.
Si alguien intentara importarlo, crashea con `ImportError`.

El archivo expone `VisionService` + 4 endpoints (`/video_feed`,
`/status`, `/health`, `/metrics`) — los conceptos `/health` y
`/metrics` valen, pero deben re-implementarse correctamente en
`presentation/api/routes/` en el módulo nuevo. **Eliminar
`legacy_api.py` entero en TTH-08.**

### 3.2 Duplicación DRY masiva entre sync y async aggregator

El bloque de cómputo de agregados de
`AsyncTrafficDataAggregator._compute_aggregates` (`async_aggregator.py:80-176`)
es **copy-paste byte-a-byte** del cuerpo de `TrafficDataAggregator.flush`
(`sync_aggregator.py:27-131`): mismo majority vote, mismo weighted
average, mismo unique IDs, mismo breakdown por tipo. Si se corrige un
bug en uno, hay que recordar replicarlo en el otro.

**Decisión para Fase 1**: extraer el cómputo a una función pura (toma
`List[FrameAnalysis]`, devuelve `List[TrafficData]`) y reusarla en
ambos aggregators. La diferencia entre sync y async debe ser
exclusivamente el dispatching a I/O, no el algoritmo.

### 3.3 `__init__.py` con código productivo de cientos de líneas

Dos casos:
- `application/processors/__init__.py` (159 líneas): define la ABC
  `FrameProcessor` + 5 procesadores concretos (DetectionProcessor,
  TrackingProcessor, SpeedEstimationProcessor, ZoneProcessor,
  AggregationProcessor).
- `infrastructure/sources/__init__.py` (72 líneas): define 3 Factory
  classes + `SourceRegistry` + función global `create_source` + un
  singleton `_registry` global mutable.

Un `__init__.py` debe ser un barrel de exports. Mezclar barrel con
implementación dificulta la navegación y rompe la convención. **En el
módulo nuevo, cada clase concreta va en su propio archivo**; el
`__init__.py` re-exporta y nada más.

### 3.4 CORS abierto con comentario "for development" en master

`presentation/api/__init__.py:13-19` configura `allow_origins=["*"]`,
`allow_methods=["*"]`, `allow_headers=["*"]` con comentario *"Allow
all origins for development"*. Si está mergeado en master, está
expuesto en producción. **Fase 1 debe decidir el origen permitido
explícitamente** (puede ser una env var leída en boot) y nunca
dejarse `["*"]` con comentario aspiracional.

### 3.5 Dos definiciones inconsistentes de "density"

- `TrafficData.avg_density` (`domain/entities.py:71`): número medio
  de vehículos por frame en la ventana (float).
- `realtime_broadcaster.serialize_analysis` (`broadcast/realtime_broadcaster.py:91-92`):
  porcentaje basado en occupancy promedio de zonas (`* 100`).

Frontend, CSV y SSE usan el segundo. Persistencia usa el primero (en
columna `avg_density`). **Fase 1 debe darles nombres distintos a estos
dos conceptos** (p.ej. `mean_vehicle_count` y `mean_occupancy_pct`) y
documentar el contrato en el dominio.

### 3.6 `print()` generalizado (>50 ocurrencias) en lugar de logger

Conteo aproximado: **>50 `print()` en el módulo productivo**, repartidos
por todas las capas. Incluye `[ERROR]`, `[WARNING]`, `[INFO]` y
`[DEBUG]`. Casos destacables:
- `zone_counter.py:104-107`: dos `print("[DEBUG] Zone ...")` que se
  disparan por cada frame con detecciones. Spam masivo de stdout.
- `async_pipeline.py` (9 prints), `pipeline_builder.py` (8 prints),
  `video_source.py` (10 prints), `multi_camera.py` (6 prints).
- Errores en workers (`async_aggregator.py:193, 198`) solo se
  imprimen, no se reportan ni reintentan.

El único archivo correcto es `yolo_detector.py` (usa
`setup_logger`/`log_execution_time` de `cerebrovial_shared`). **Fase 1
debe estandarizar el uso del logger compartido en todo el módulo**
y eliminar `print` salvo en scripts CLI explícitos.

### 3.7 Scratchpad del autor dejado en código productivo

Tres archivos con comentarios de exploración/duda del propio autor:
- `youtube_source.py:13-26`: 14 líneas de "Let's check...wait...
  Actually let's look at the file content of __init__.py from
  previous turns" sobre el orden de argumentos del factory.
- `realtime_broadcaster.py:4-6`: "No domain imports here, but let's
  check if it uses any. ... Wait, let's check the content."
- `csv_repository.py:40-45`: discusión interna sobre formato de
  timestamp.

Estos comentarios reflejan trabajo en progreso, no decisiones cerradas.
**Política para Fase 1**: si una duda sigue abierta cuando se mergea,
se documenta como issue/TODO con tracking explícito, no como
comentario inline. Si se cerró, se borra.

### 3.8 Observaciones de bajo impacto (una línea cada una)

- Lógica muerta confirmada en `smart_detection.py`:
  `_vehicle_trajectories` (línea 31, nunca poblado),
  `get_analysis_for_frame` (línea 76, siempre `None`),
  `interpolate` (parámetro de `__init__` nunca leído).
- Atributos muertos en `async_pipeline.py`: `source_fps` (línea 35),
  `display_queue` (línea 40).
- Nombre engañoso: "SmartDetection" no significa nada concreto; es
  *throttled detection*. Renombrar.
- Debug field renderizado en UI: `opencv_visualizer.py:67-69` pinta
  `f"Raw Detections: {analysis.raw_detection_count}"` en pantalla.
- Speed estimator muta entidad de dominio in-place
  (`speed_estimator.py:53`: `vehicle.speed = speed_kmh`).
- `set_pipeline()` no-op disfrazado de hook de compatibilidad
  (`presentation/api/__init__.py:32-42`: cuerpo `pass`).
- Sentinel strings `"unknown"` para `camera_id`/`street` en
  `domain/entities.py:32-33` y `infrastructure/zones/zone_counter.py:23, 27-28`.
- `id: str` plano en lugar de Value Objects en el dominio.
- Sin `__post_init__`/validación en entidades (confidence puede ser >1,
  speed negativa, bbox con x1>x2).
- `frame: object` en lugar de `np.ndarray` (`entities.py:54`,
  `protocols.py:11`).
- Subclases vacías solo para naming: `VideoFileSource`, `WebcamSource`,
  `ZoneSelector` (legacy alias).
- Magic numbers hardcoded (ByteTrack params, scaling 1280x720,
  congestion 30/70, sleep 1.0, time_diff 0.1, fps 24/30).
- Threading + asyncio mezclados sin cuidado:
  `multi_camera._run_camera_pipeline` itera `pipeline.run()` síncrono
  dentro de `async def` (bloquea event loop).
- Mutación concurrente sin locks en `CameraState`.
- Encapsulación violada: routes acceden a `manager.cameras[...]` y
  `camera.state.*` directamente; también a `broadcaster._subscribers`
  (ver §6.10).
- Singletons globales mutables en routes (`_manager`, `_broadcaster`,
  `_service`).
- Tests anti-pattern: `test_async_pipeline_drop.py` valida su propia
  simulación, no el productivo.

## 4. Responsabilidades mal distribuidas

Cinco casos cambian decisiones de capa de Fase 1:

**Caso A — `infrastructure/zones/zone_counter.ZoneCounter` no tiene
Protocol en el dominio.** Existe `VehicleDetector`, `VehicleTracker`,
`SpeedEstimator`, `FrameProducer` y `TrafficRepository`, pero no
`ZoneCounter` ni `Aggregator` ni `Broadcaster`. La consecuencia directa
es que `application/processors/__init__.py:6` importa la implementación
concreta (`from ...infrastructure.zones.zone_counter import ZoneCounter`),
violando la dirección DDD. Decisión formal en §6.2.

**Caso B — `application/services/multi_camera.py` importa
presentación.** Línea 11: `from ...presentation.visualization.opencv_visualizer
import OpenCVVisualizer`. La aplicación NO debe conocer la presentación.
Decisión formal en §6.8.

**Caso C — `infrastructure/interaction.py` es UI, no infraestructura.**
Usa `cv2.namedWindow`, `cv2.imshow`, `cv2.setMouseCallback`,
`cv2.waitKey`. Probablemente es un script offline para definir zonas a
mano. Decisión formal en §6.8 (que también cubre el lugar del
visualizer).

**Caso D — `realtime_broadcaster.serialize_analysis` mezcla transporte
y presentación.** `broadcast/realtime_broadcaster.py:75-132` produce
strings localizadas (`"Bajo"/"Moderado"/"Alto"`), formatos con `%`,
umbrales semánticos (30/70), y campos placeholder (`"incidents": 0`).
Es lógica de presentación. Decisión formal en §6.11.

**Caso E — Builder construido siempre como `AsyncVisionPipeline`,
sync queda sin caller.** `pipeline_builder.py:162` siempre instancia
`AsyncVisionPipeline`. `VisionPipeline` (sync) existe en
`application/pipelines/sync_pipeline.py` y tiene tests, pero ningún
caller en código de producción. Decisión formal en §6.6.

## 5. Tests rescatables

**Rescatables tal cual** (conceptualmente válidos para el módulo nuevo):
- `test_detection_frequency` (throttled detection cada N frames).
- `test_aggregation_consistency` + `test_aggregation_multiple_vehicles`
  (contratos del aggregator: majority vote, sum=total).
- `test_pipeline_*` del sync (delegación analysis previo → siguiente
  frame).
- `test_pipeline_initialization`, `test_pipeline_start_stop`,
  `test_pipeline_stop_event_propagation` del async.
- `test_builder_constructs_complete_pipeline` (template de integración).
- `test_add_duplicate_camera`, `test_start_camera_not_found`,
  `test_get_status` (multi_camera).
- `test_create_source_*` (factory completo).
- `test_class_stabilization`, `test_history_limit` (tracker majority
  vote + window cap).
- `test_zone_manager_initialization`, `test_zone_manager_empty_detections`.
- `test_subscribe_unsubscribe`, `test_broadcast`,
  `test_broadcast_slow_consumer`, `test_latest_state` (broadcaster).
- `test_yolo_detector_*` (todos: init, detect, filter classes).
- `test_persistence_integration` (integración end-to-end CSV, con dos
  cambios: path relativo y timing arbitrario).

**Rescatables con actualización menor** (rename de atributo):
- Los dos xfail C1.6 (`test_add_camera`, `test_start_stop_camera`):
  cambiar `camera.camera_id` por `camera.state.camera_id` y
  `camera.is_running` por `camera.state.is_running`.

**Rescatables como contrato del módulo nuevo** (documentan bugs):
- `test_pipeline_processing_flow` (xfail C1.5 race condition).
- `test_zone_manager_update` (xfail C1.8 bug productivo, ver §6).

**A tirar**:
- `test_interpolation_logic`, `test_trajectory_update`: testean código
  removido (interpolación, `_vehicle_trajectories`).
- `test_drop_newest_strategy`, `test_rate_limited_logging`: tests
  autotautológicos que simulan su propia lógica, no el productivo.

**A reescribir mejor**:
- `test_broadcaster.py` no debería tocar `broadcaster._subscribers`
  (atributo protected); el módulo nuevo debe exponer un método público
  equivalente y los tests usarlo (ver §6.10 para la decisión de API
  pública).
- `test_tracker_stabilization` tiene scratchpad del autor en
  comentarios (líneas 41-46): reescribir sin "Let's see" inline.

## 6. Decisiones que Fase 1 debe tomar

### 6.1 PRIORITARIO — Test de aceptación temprano para zone counter

**Hallazgo**: el xfail TODO C1.8
(`test_zones.py::test_zone_manager_update`) **no es un test
desactualizado**. Testea el comportamiento más básico imaginable —
un vehículo cuyo bbox cae dentro del polígono debe contar 1 — y falla.
Razón documentada: *"ZoneCounter not detecting vehicles inside polygon.
Possibly related to coordinate system change in refactor."*

**Impacto downstream**: si el zone_counter cuenta 0, el aggregator
agrega 0 vehículos por ventana, el broadcaster reporta densidad
cero/baja consistentemente, y el CSV persiste 0s. **Toda la cadena de
valor del módulo está rota en el productivo actual**, aunque el
pipeline corra "sin errores".

Esto **refuerza** las decisiones cerradas en DHU-024 (refactor desde
cero); no las contradice.

**Acción concreta para Fase 1**: el módulo nuevo debe tener, como
**primer test de aceptación implementado**, un test end-to-end que:
1. Construye un `ZoneCounter` con un polígono conocido.
2. Le pasa una detección sintética cuyo bbox cae claramente dentro.
3. Asserta que el conteo devuelto es **no-cero** (la pregunta es
   "¿el contrato más básico se cumple?", no "¿cuál es el conteo
   exacto?").

Hasta que este test pase, ningún otro trabajo de Fase 1 avanza. Es
gate de salida del primer commit funcional.

### 6.2 Establecer el set completo de Protocols en el dominio

Hoy hay 5 (`VehicleDetector`, `VehicleTracker`, `SpeedEstimator`,
`FrameProducer`, `TrafficRepository`). Faltan al menos `ZoneCounter`,
`Aggregator`, `Broadcaster`, `FrameRenderer` (ver §4 casos A y B).
La decisión incluye: ¿`Repository` es solo write, o se agrega query
interface? El actual solo tiene `save` (`domain/repositories.py:11`).

### 6.3 Política de logging unificada

El estándar del proyecto (cuando se usa bien) es `setup_logger` +
`log_execution_time` de `cerebrovial_shared.logging`. Decisión:
- ¿Se permite algún `print()` en código productivo? (Recomendado:
  no.)
- ¿Cuál es el nivel mínimo loggeado en boot por default?
- ¿Cómo se loggea desde workers de threading sin perder contexto?

### 6.4 Estrategia de overrides de configs YAML

Contexto: `balanced.yaml` y `low_latency.yaml` son configs parciales
(solo `performance`+`model`+`display`). El código del builder no
muestra merge con `default.yaml`.

**Decisión de fondo**: ¿los YAMLs en `conf/vision/` son fuente de
verdad o son ejemplo? Evidencia del conflicto: `cameras.py:70-85`
construye una OmegaConf en código que **duplica textualmente** valores
de los YAMLs. Si los YAMLs son la fuente de verdad, el endpoint debe
leerlos. Si el endpoint es la fuente de verdad, los YAMLs son ejemplo
y deberían vivir en `docs/`.

Según la respuesta, la decisión técnica de overrides cae en una de
tres opciones: OmegaConf merge explícito, perfil = nombre con merge
en boot, o cada perfil completo.

### 6.5 Manejo de errores en workers async

`async_aggregator._flush_worker` captura `Exception` genérico y solo
hace `print(...)` (líneas 193, 198). Decisión:
- ¿Se reintenta `save` con backoff?
- ¿Se acumulan errores y se reportan vía métrica/healthcheck?
- ¿Cuál es la política cuando se pierden datos por queue full
  (`async_aggregator.py:78`: `data_dropped` actualmente solo se
  imprime)?

### 6.6 Modos de pipeline (sync vs async)

Hoy el builder solo construye `AsyncVisionPipeline`; `VisionPipeline`
(sync) tiene tests propios pero ningún caller. Decisión: conservar
ambos modos exponiéndolos en el builder, o eliminar el sync.

### 6.7 Identidades como Value Objects o strings

Hoy `DetectedVehicle.id`, `ZoneVehicleCount.zone_id`,
`TrafficData.camera_id` son todos `str`. Decisión: introducir VOs
(`VehicleId`, `ZoneId`, `CameraId`) con validación, o mantener `str`
y validar en boundaries.

### 6.8 Lugar del visualizer y del módulo `interaction`

Caso C de §4 (mover `interaction.py` a `presentation/`) y caso B
(decidir si `OpenCVVisualizer` se inyecta o vive en una capa
distinta). Estas son decisiones de capa, no menores.

### 6.9 Definiciones canónicas de métricas

`density`, `congestion_level`, `occupancy`, `flow_rate_per_min`,
`avg_density`: hoy hay duplicaciones (§3.5) y placeholders
(`"incidents": 0` hardcoded en broadcaster). El dominio debe definir
canónicamente qué significa cada métrica antes de que la
serialización de la API se reescriba.

### 6.10 API pública de `Broadcaster`

Hoy tanto las routes (`presentation/api/routes/streaming.py`) como el
test (`test_broadcaster.py`) acceden a `broadcaster._subscribers`
directamente porque no existe un método público equivalente. Fase 1
debe definir la API pública mínima del `Broadcaster` (al menos:
`subscriber_count()`, `is_subscribed(subscriber_id)`) antes de
reescribir el componente. Lo que hoy es atributo protected debe quedar
privado del todo en el módulo nuevo.

### 6.11 Separación transporte/presentación en el Broadcaster

El broadcaster actual (`broadcast/realtime_broadcaster.serialize_analysis`,
líneas 75-132) emite strings localizadas
(`"Bajo"/"Moderado"/"Alto"`), formatos con `%`, umbrales semánticos
(30/70) y campos placeholder (`"incidents": 0`). Eso es lógica de
presentación incrustada en una capa de transporte. Fase 1 debe
decidir que el broadcaster transporta estructuras puras (entidades o
DTOs del dominio) y que el formateo localizado / semántico vive en
`presentation/`, no en `infrastructure/broadcast/`. Esto incluye
eliminar el cálculo `pedestrians = sum(... if v.type == 'person')`
(el detector está configurado solo para car/bus/truck/motorcycle, el
sum siempre es 0).

## 7. Candidatos a DHU-025

**Ninguno en esta auditoría.** Los hallazgos refuerzan las decisiones
cerradas en DHU-024 (refactor desde cero está justificado, regla de
no-tocar está justificada hasta que TTH-08 la levante formalmente);
no las contradicen.

Si Fase 1 al diseñar destapa contradicciones, se abre DHU-025 en ese
momento, no acá.
