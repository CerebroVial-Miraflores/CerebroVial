# TTH-08 Fase 1 — Diseño DDD del módulo `vision` reescrito

> **Documento parcial.** Esta versión cubre los cimientos del dominio (Bloques 0
> y 1 del grafo de dependencias de §6 del documento de lecciones de Fase 0).
> Las decisiones que aún no se han tomado en Fase 1 están marcadas como
> **pendiente** y se cerrarán en sesiones posteriores antes de que Fase 3 (Domain
> layer) arranque. Este documento NO debe leerse como el diseño completo de
> Fase 1.
>
> **Alcance de esta versión** (Sesión 1 de Fase 1):
> §6.1 (test de aceptación temprano del `ZoneCounter`),
> §6.7 (Value Objects),
> §6.2 (set completo de Protocols del dominio, parcialmente — con adelanto parcial de §6.6).

## 1. Contexto y marco

Fase 1 de TTH-08 toma diseño DDD del módulo `vision` reescrito. Su insumo
primario es el documento `tth-08-fase0-lecciones.md`, que destaca once
hallazgos (siete con párrafo propio en §3, cinco responsabilidades mal
distribuidas en §4) y once decisiones que Fase 1 debe tomar antes de que
Fase 3 implemente el dominio.

El marco arquitectónico está fijado por DHU-024 (alcance operativo completo,
DDD completo, capas espejo de `core_management_api`) y por el SDD canónico,
que reconoce al `vision/` actual como módulo donde la disciplina DDD del
proyecto se cumple. La reescritura mantiene esa disciplina como invariante;
los cambios que introduce son los que la auditoría de Fase 0 justifica
explícitamente (corrección de bugs, encapsulación, tipado, Protocols
faltantes), no desviaciones gratuitas. Donde el módulo nuevo se aparta del
módulo actual, este documento lo declara como **cambio deliberado** y cita
la evidencia de Fase 0 que lo justifica.

Fase 1 produce diseño (documento), no código. Fase 3 implementa el
dominio a partir de este documento.

### 1.1 Convenciones del proyecto que aplican

- **TTH ≠ HU.** TTH-08 es Tarea Técnica Habilitadora (DHU-004, Artículo 12
  de la constitución). Sus criterios de "terminado" son los CT-08.1 a
  CT-08.11, no escenarios BDD/Gherkin. Los tests del módulo siguen el
  patrón pytest puro del repo, en la estructura `unit/`/`integration/`
  que `vision/` ya usa.
- **Convención objetivo del repo: sin `xfail`/`skip` en código de
  producción.** `core_management_api` lo cumple. `vision/` actual no:
  tiene 6 `xfail` (test_zones.py:12-14, test_smart_detection.py:45 y :94,
  test_multi_camera_manager.py:26 y :48, test_async_pipeline.py:56). La
  reescritura debe eliminar todos. Por la misma convención, el test de
  aceptación §2 se difiere a Fase 4 en lugar de escribirse ahora con
  `xfail`.
- **Patrón de fixtures**: factory pattern (closures que retornan funciones
  parametrizadas), heredado de `core_management_api/tests/{auth,control}/`.
- **Asserts**: forma estricta + valores exactos invariantes + rangos solo
  donde haya tolerancia numérica esperada (patrón observado en
  `core_management_api/tests/control/test_engine.py`).

### 1.2 Referencia DDD del proyecto

El SDD identifica a `vision/` actual como el módulo donde la disciplina
DDD se cumple. `core_management_api/src/control/` es desviación reconocida
por el propio SDD (el SDD lo atribuye a la falta de capa `infrastructure/`
en `control/`, mientras que `domain/`, `application/` y `presentation/`
sí están presentes).
`core_management_api/src/prediction/` **no tiene capa `domain/`**: solo
tiene `application/`, `infrastructure/` y `presentation/`. Por lo tanto,
**la única referencia DDD comparable disponible en el repo es el `vision/`
actual mismo, después de las correcciones que esta reescritura aplica**.

Cuando este documento cita "el patrón del repo" para algo del dominio,
se refiere a `vision/` actual + la auditoría de Fase 0 que lo evaluó.
No se citan otros módulos como referencia DDD positiva porque no los hay.

## 2. §6.1 — Test de aceptación temprano del `ZoneCounter`

### 2.1 Hallazgo origen

El xfail TODO C1.8 del módulo actual
(`edge_device/tests/vision/unit/test_zones.py:12-14`,
`test_zone_manager_update`) **no es un test desactualizado**: testea el
contrato más básico imaginable del `ZoneCounter` —un vehículo cuyo bbox
cae dentro del polígono debe contar 1— y falla. Documentado en lecciones
de Fase 0 §6.1: el zone counter roto rompe toda la cadena de valor
downstream (aggregator, broadcaster, persistencia reportan ceros
consistentemente).

### 2.2 Decisión

| Atributo | Valor |
|---|---|
| **Ubicación** | `edge_device/tests/vision/integration/test_zone_counter_basic.py` |
| **Categoría** | Test de integración (no unit). Construye `ZoneCounter` real con polígono real y le pasa una detección sintética. |
| **Naturaleza** | Pytest puro. No BDD (TTH-08 no es HU). |
| **Cuándo se escribe** | En Fase 4 (Infrastructure), junto con la implementación del `ZoneCounter` nuevo. NO se escribe ahora con `xfail`/`skip` (no es convención objetivo del repo). |
| **Rol en TTH-08** | Gate de salida del primer commit funcional de Fase 4. Hasta que pase, ningún otro trabajo de Fase 4 avanza. |

### 2.3 Contrato del test

```
Dado:
  - Un ZoneCounter inicializado con un polígono cuadrado conocido,
    p.ej. vértices (100,100), (200,100), (200,200), (100,200).
  - Una zona identificada como ZoneId("zone_test").
  - Una única detección sintética de tipo DetectedVehicle con:
      * id        = VehicleId("vehicle_test")
      * bbox      = (140, 140, 160, 160)   # centroide claramente dentro
      * confidence = 0.95
      * type      = "car"

Cuando:
  - Se invoca ZoneCounter.count(detections=[detection], frame_id=1).

Entonces:
  - El resultado contiene una entrada para ZoneId("zone_test").
  - El conteo de la zona es EXACTAMENTE 1.
  - La entrada incluye el VehicleId del vehículo contado.
```

**Assert principal:** el conteo de la zona es `== 1`.

**Asserts secundarios** (no bloqueantes para el gate, opcionales para
Fase 4): preservación de identidad del vehículo en la lista. La forma
exacta de este assert depende del shape final de `ZoneVehicleCount`
en Fase 3; el contrato del test se alinea con el campo final cuando
exista.

### 2.4 Justificación de la elección de assert

- `>= 1` es demasiado laxo: count=5 con una sola detección sería "no-cero"
  pero estaría mal.
- `== 1` es discreto y exacto. El conteo no tiene tolerancia numérica.
- Alinea con el patrón de asserts del repo: forma estricta donde no hay
  variabilidad esperada
  (`core_management_api/tests/control/test_engine.py:15-22`).

### 2.5 Pre-requisitos para Fase 4

El archivo `edge_device/src/vision/infrastructure/zones/zone_counter.py`
**ya existe** (es el que tiene el bug). Fase 4 lo **reescribe**, no lo
construye desde cero. Si la reescritura crea un archivo con nombre
distinto, el viejo se borra en el mismo commit. La regla de no-tocar de
CLAUDE.md está levantada para entonces (DHU-024 §8).

Cuando Fase 4 escriba el test, la infraestructura mínima que debe estar
en su lugar es:

1. Entidades del dominio (`Frame`, `DetectedVehicle`, `ZoneVehicleCount`,
   `FrameAnalysis`) en `domain/entities.py`. Fase 3.
2. VOs (`VehicleId`, `ZoneId`) en `domain/value_objects.py`. Fase 3.
3. Protocol `ZoneCounter` en `domain/protocols.py`. Fase 3.
4. Implementación concreta reescrita del `ZoneCounter` en
   `infrastructure/zones/zone_counter.py`. Fase 4.

El test es lo primero que Fase 4 escribe, contra el esqueleto del
`ZoneCounter` que esa misma fase implementa.

## 3. §6.7 — Value Objects en el dominio

### 3.1 Decisión

Tres VOs estrictos en `domain/value_objects.py`:

- `VehicleId`
- `ZoneId`
- `CameraId`

Implementación: `@dataclass(frozen=True)` con validación en `__post_init__`.

### 3.2 Implementación de referencia

```python
# domain/value_objects.py
from dataclasses import dataclass

_MAX_ID_LEN = 64


@dataclass(frozen=True)
class VehicleId:
    """Identifier for a detected and tracked vehicle within a session."""
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("VehicleId no puede ser vacío")
        if len(self.value) > _MAX_ID_LEN:
            raise ValueError(f"VehicleId excede {_MAX_ID_LEN} caracteres")


@dataclass(frozen=True)
class ZoneId:
    """Identifier for a configured zone (polygon) in a camera frame."""
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("ZoneId no puede ser vacío")
        if len(self.value) > _MAX_ID_LEN:
            raise ValueError(f"ZoneId excede {_MAX_ID_LEN} caracteres")


@dataclass(frozen=True)
class CameraId:
    """Identifier for a camera (intersection access point)."""
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("CameraId no puede ser vacío")
        if len(self.value) > _MAX_ID_LEN:
            raise ValueError(f"CameraId excede {_MAX_ID_LEN} caracteres")
```

### 3.3 Justificación

1. **Los tres ids cruzan seis capas** del módulo (detector → tracker →
   aggregator → broadcaster → routes → persistencia). Es exactamente el
   caso donde los VOs ganan: el dominio se vuelve más legible y el riesgo
   de mezclar ids entre capas se elimina por tipado estático.
2. **La auditoría de Fase 0 identificó problemas concretos** que los VOs
   resuelven: encapsulación violada (Fase 0 §3.8), sentinel strings
   `"unknown"` para `camera_id`/`street` (`domain/entities.py:32-33`,
   `infrastructure/zones/zone_counter.py:23,28,62`), ids como `str` plano
   sin validación.
3. **`@dataclass(frozen=True)` con `__post_init__` es la forma más liviana
   de Python**: sin dependencias adicionales, validación al construir,
   inmutabilidad garantizada, hashable por defecto (apto para usar como
   key en dict y set, lo que el aggregator necesita).
4. **Política aplicada solo donde se necesita**: VOs para los tres ids
   que cruzan capas. Otros campos (`type`, `confidence`, `street`) NO
   se promueven a VOs ahora; eso se evalúa caso por caso en Sesiones
   posteriores si la auditoría lo justifica.

### 3.4 Política de boundaries

Los VOs son del dominio. En boundaries externas (API entrante, persistencia
saliente) se envuelven y desenvuelven explícitamente:

- **API entrante** (`presentation/api/routes/`): el endpoint recibe `str`
  desde JSON. Pydantic valida estructura básica; el adapter de presentación
  envuelve antes de pasar a `application/`:
  `VehicleId(value=request.vehicle_id)`. Si `__post_init__` levanta
  `ValueError`, se traduce a HTTP 422 en el handler.
- **Persistencia saliente** (`infrastructure/persistence/`): al escribir
  a BD/CSV, se desenvuelve: `row["vehicle_id"] = vehicle.id.value`.
- **Logs y serialización para broadcasting**: misma política de boundary,
  `.value` al salir.

### 3.5 Trade-offs aceptados

| Costo | Mitigación |
|---|---|
| Más código (3 clases vs 3 type aliases). | Las clases son ~12 líneas cada una; trivial. |
| Boilerplate `VehicleId(value=...)` en boundaries. | Limitado a routes y persistencia; el resto del módulo trabaja con VOs ya envueltos. |
| Indirección `vehicle.id.value` en vez de `vehicle.id`. | Lectura ligeramente más verbosa, pero el tipo del campo es ahora autodocumentado. |
| Verificación de no-colisión con código existente. | Verificada al 2026-05-27: `VehicleId`/`ZoneId`/`CameraId` no existen como símbolos en `shared/`, `edge_device/`, ni `core_management_api/`. |

### 3.6 Nota sobre el resto del proyecto

Otros módulos del repo NO usan VOs estrictos:
`core_management_api/src/control/` (desviación DDD reconocida por el SDD)
maneja ids como `str` plano en sus dataclasses sufijadas `_DC`.
`core_management_api/src/prediction/` no tiene capa `domain/` con la cual
comparar. La decisión de introducir VOs en `vision/` no se justifica por
"seguir el patrón del repo" (el repo no tiene patrón consolidado sobre
esto) sino por la estructura específica de cruce de capas del módulo
`vision/`. Es decisión deliberada local al módulo, no extensión de
convención global.

## 4. §6.2 — Set de Protocols en el dominio

### 4.1 Decisión

**9 Service Protocols** en `domain/protocols.py` y **1 Repository Protocol**
en `domain/repositories.py` (manteniendo la separación arquitectónica del
patrón Repository según DDD ortodoxo y el `vision/` actual).

Tres del módulo actual se preservan **con misma firma**. Dos se preservan
**con firma reescrita** (cambio deliberado documentado). Cinco son nuevos
(cubren huecos identificados en Fase 0).

| # | Protocol | Origen | Cubre |
|---|----------|--------|-------|
| 1 | `VehicleDetector` | Preservado mismo nombre, firma reescrita | Detección por frame. Cambio: `frame: object` → `np.ndarray`. |
| 2 | `VehicleTracker` | Preservado mismo nombre, misma firma | Tracking entre frames. |
| 3 | `SpeedEstimator` | Preservado mismo nombre, misma firma | Estimación de velocidad. |
| 4 | `FrameProducer` | Preservado nombre, **firma cambiada** | Fuente de frames. **Breaking**: ver §4.4. |
| 5 | `ZoneCounter` | **Nuevo** | Conteo por polígono. Cierra Caso A de §4 de Fase 0. |
| 6 | `SyncAggregator` | **Nuevo** | Cómputo síncrono de `TrafficData`. Cierra Caso A. |
| 7 | `AsyncAggregator` | **Nuevo** | Cómputo asíncrono (worker thread). Cierra Caso A. |
| 8 | `Broadcaster` | **Nuevo** | Publicación de estado en tiempo real. Cierra Caso A + §6.10. Caso D queda parcialmente cubierto; cierre completo en §6.11 (Sesión 3). |
| 9 | `FrameRenderer` | **Nuevo** | Anotación visual de frames. Cierra Caso B. |
| R | `TrafficRepository` | Preservado nombre, firma reescrita | Persistencia de agregados. Cambio: explicitar `-> None`. Vive en `repositories.py`. |

### 4.2 Firmas detalladas

```python
# domain/protocols.py
from typing import Protocol, Optional
import numpy as np

from .entities import (
    DetectedVehicle, Frame, FrameAnalysis,
    TrafficData, ZoneVehicleCount,
)
from .value_objects import VehicleId, ZoneId


class VehicleDetector(Protocol):
    """Protocol for vehicle detection in a single frame."""
    def detect(self, frame: np.ndarray, frame_id: int) -> list[DetectedVehicle]: ...


class VehicleTracker(Protocol):
    """Protocol for assigning stable identities across frames."""
    def update(self, detections: list[DetectedVehicle]) -> list[DetectedVehicle]: ...


class SpeedEstimator(Protocol):
    """Protocol for speed estimation of tracked vehicles."""
    def estimate(self, vehicles: list[DetectedVehicle]) -> list[DetectedVehicle]: ...


class FrameProducer(Protocol):
    """Protocol for a source of frames (file, webcam, stream)."""
    def read(self) -> Optional[Frame]: ...
    def release(self) -> None: ...


class ZoneCounter(Protocol):
    """Protocol for counting vehicles per configured zone (polygon)."""
    def count(
        self,
        detections: list[DetectedVehicle],
        frame_id: int,
    ) -> dict[ZoneId, ZoneVehicleCount]: ...


class SyncAggregator(Protocol):
    """Protocol for synchronous aggregation of frame analyses into traffic data.

    Caller is responsible for persistence: `add()` accumulates, `flush()`
    returns computed `TrafficData`, persistence is delegated to the caller.
    """
    def add(self, analysis: FrameAnalysis) -> None: ...
    def flush(self) -> list[TrafficData]: ...


class AsyncAggregator(Protocol):
    """Protocol for asynchronous aggregation using a worker thread + queue.

    Same compute contract as `SyncAggregator` (`add` + `flush`) plus
    `force_flush` (synchronous drain) and `stop` (worker lifecycle).

    Semantics of the four methods:

    - `add(analysis)`: non-blocking enqueue. The worker thread picks the
      analysis from its input queue asynchronously.
    - `flush() -> list[TrafficData]`: non-blocking. Returns the
      `TrafficData` items already computed and waiting in the output queue
      at call time. Items still being processed by the worker are NOT
      included in this call's return. Use case: periodic telemetry pull
      from the pipeline thread without pausing the worker.
    - `force_flush() -> list[TrafficData]`: blocking. Forces the worker to
      drain its input queue, computes everything pending, and returns the
      full result. Use case: clean shutdown before `stop()`, or test
      synchronization.
    - `stop() -> None`: signals the worker to terminate. Does NOT return
      data; callers that need final data should invoke `force_flush()`
      first.

    The caller (typically a use case in `application/`) is responsible for
    persistence via the `TrafficRepository` Protocol.
    """
    def add(self, analysis: FrameAnalysis) -> None: ...
    def flush(self) -> list[TrafficData]: ...
    def force_flush(self) -> list[TrafficData]: ...
    def stop(self) -> None: ...


class Broadcaster(Protocol):
    """Protocol for publishing traffic state to real-time consumers."""
    async def publish(self, data: TrafficData) -> None: ...
    def subscriber_count(self) -> int: ...
    def is_subscribed(self, subscriber_id: str) -> bool: ...


class FrameRenderer(Protocol):
    """Protocol for visually annotating frames (bbox, labels, ROI overlays).

    Returns annotated frame as `np.ndarray` of shape (H, W, 3), dtype uint8,
    BGR channel order (OpenCV convention), same H/W as input frame.
    """
    def render(self, frame: Frame, analysis: FrameAnalysis) -> np.ndarray: ...
```

```python
# domain/repositories.py
from typing import Protocol
from .entities import TrafficData


class TrafficRepository(Protocol):
    """Protocol for persisting traffic aggregates. Write-only in MVP1."""
    def save(self, data: TrafficData) -> None: ...
```

### 4.3 Decisiones secundarias

| Tópico | Decisión | Razón |
|---|---|---|
| **Firmas con VOs vs `str`** | VOs en todos los ids estructurales. | Consecuencia directa de §3 (VOs en el dominio). |
| **Return types** | Explícitos en todos los métodos. | Auditoría detectó que `TrafficRepository.save` no lo tenía. |
| **`@runtime_checkable`** | No usar. | No surge necesidad de `isinstance(x, Protocol)` en el diseño. Se introduce solo si un caso de uso concreto lo requiere. |
| **`Repository` write-only** | Solo `save`. Sin query interface. | Las queries históricas las consume `core_management_api`, no `vision/`. F41 (Trabajos Futuros) reevaluará. |
| **Organización en archivos** | 9 Service Protocols en `protocols.py`, 1 Repository en `repositories.py`. | Separación DDD ortodoxa Service vs Repository. Coincide con el `vision/` actual. |
| **Docstrings** | Una línea para Protocols simples; varias líneas para los que tienen contrato no obvio (`SyncAggregator`, `AsyncAggregator`, `FrameRenderer`). | El `vision/` actual usa una línea para todos. Acá agregamos detalle solo cuando el contrato lo requiere (no es ornamento). |
| **`frame: object` → `np.ndarray`** | Tipar explícitamente. | **Cambio deliberado** respecto al `vision/` actual. Justificación: `object` renuncia al tipado. El frame es nativo numpy en toda la infraestructura. La auditoría de Fase 0 §3.8 lo identificó como problema. Esto introduce un import de numpy en `domain/`. Es una desviación consciente del principio "dominio sin dependencias técnicas", aceptada porque numpy es el tipo del dato, no una biblioteca de infraestructura. |
| **Excepciones tipadas** | `VehicleDetector.detect` puede levantar `DetectionError`; `FrameProducer.read` puede levantar `SourceError`. Documentado como nota informal en los docstrings cuando se redacten; no como cláusula estructurada. | `cerebrovial_shared.exceptions` ya las define. Es el patrón del proyecto. |

### 4.4 Cambios respecto al `vision/` actual (declarados explícitamente)

Este documento NO presenta los Protocols como una formalización trivial
de lo que el `vision/` actual ya tenía. Hay tres cambios deliberados que
rompen consumidores actuales. Quedan documentados acá para que Fase 3 y
Fase 5 los implementen sin sorpresa:

**Cambio 1 — `FrameProducer`: `__iter__(self) -> Iterator[Frame]` →
`read() -> Optional[Frame]` + `release()`.**

Estado actual (`domain/protocols.py:28-36`): `__iter__` retorna iterador.
El consumidor itera con `for frame in producer`.

Cambio: el productor expone `read()` que retorna `Optional[Frame]`
(`None` para fin de stream o error transitorio recuperable) y `release()`
para cerrar recursos. Excepciones tipadas (`SourceError`) para errores
fatales.

Razón: `__iter__` no permite distinguir fin-de-stream de error transitorio;
`StopIteration` cubre ambos. Para streams HLS/RTMP/YouTube que pueden
desconectarse intermitentemente, esa distinción es necesaria. La metáfora
`read()` + `release()` alinea con la API estándar de OpenCV
(`cv2.VideoCapture.read()`) que la infraestructura usa nativamente.
Reduce impedance mismatch en el adapter.

Impacto: `application/pipelines/` (sync y async) consumen el producer
actual con `for frame in producer`. Fase 5 los reescribe con la nueva
API.

**Cambio 2 — `Aggregator`: `aggregate_and_persist()` + `flush() -> None`
→ `add()` + `flush() -> list[TrafficData]`, separados en `SyncAggregator`
y `AsyncAggregator`.**

Estado actual: `aggregate_and_persist`
(`application/aggregators/sync_aggregator.py:17` y
`async_aggregator.py:44`) acumula la `FrameAnalysis` recibida y delega a
`flush`/`_compute_aggregates`
(`sync_aggregator.py:27-131` y `async_aggregator.py:80-176`), que es donde
ocurre el cómputo real **y la persistencia interna como side-effect**
(p.ej. `sync_aggregator.py:131` invoca `self.repository.save(data)`).
`flush()` retorna `None`.

Cambio: `add(analysis)` solo acumula; `flush()` calcula y **retorna**
`list[TrafficData]`. La persistencia es responsabilidad del caller (use
case en `application/`), que pasa el resultado al `TrafficRepository`.

Razón: separar cómputo de persistencia es lo que prescribe DDD ortodoxo.
La auditoría de Fase 0 §3.2 identificó la duplicación masiva entre sync y
async aggregator: ambos contenían cómputo idéntico + persistencia mezclada.
Separar permite (a) una función pura de cómputo reusable entre los dos
modos, (b) tests sin mockear el repository, (c) cumplir Single
Responsibility.

Además se separa en dos Protocols (`SyncAggregator`, `AsyncAggregator`)
porque el `AsyncAggregator` tiene lifecycle propio (`force_flush`,
`stop`) que el síncrono no requiere. Unificar bajaría el valor del
tipado.

Esto adelanta parcialmente §6.6 a Sesión 1: votamos "dos contratos
separados", no "qué modo se conserva". §6.6 sigue pendiente en cuanto a
si Fase 5 implementa ambos modos o elimina uno.

Impacto: los aggregators concretos, los pipelines y el use case que los
orquesta se reescriben en Fases 4 y 5.

**Cambio 3 — `Broadcaster`: `broadcast(camera_id: str, analysis_data: dict)`
→ `publish(data: TrafficData)` + API pública con `subscriber_count()` e
`is_subscribed(subscriber_id)`.**

Estado actual
(`infrastructure/broadcast/realtime_broadcaster.py:52`): firma con dos
parámetros (`camera_id` separado del `analysis_data` que es un `dict`
crudo). Subscribers accedidos vía atributo privado `_subscribers`
(`realtime_broadcaster.py:16, :31-33, :47-49, :61` como uso interno
legítimo de la clase; pero también
`presentation/api/routes/streaming.py:60` y
`tests/vision/unit/test_broadcaster.py:13, :14, :17` como acceso externo
a atributo privado — esto es lo que Fase 0 §3.8 y §5 documentan).

Cambio: un solo parámetro tipado (`TrafficData`), que ya contiene
`camera_id` como campo. Métodos públicos `subscriber_count()` e
`is_subscribed(subscriber_id)` que cubren los casos por los cuales
hoy se accede al atributo privado.

Razón:

- **Single source of truth.** `TrafficData` ya tiene `camera_id` como
  campo. Pasarlo aparte invita a inconsistencias
  (`camera_id != data.camera_id`).
- **Eliminar acceso a atributos privados** (Fase 0 §6.10): si la API
  pública existe, ni el productivo ni el test necesitan tocar
  `_subscribers`.
- **El `TrafficData` reescrito en Fase 3** tiene `camera_id: CameraId`
  como campo obligatorio (sin sentinel `"unknown"`). Eso ya es
  consecuencia de §3.

Impacto: el broadcaster concreto, las routes que lo invocan y los
tests que validan suscriptores se reescriben en Fases 4, 6 y 7.

### 4.5 Cobertura de huecos de Fase 0

| Hueco de Fase 0 | Cubierto por |
|---|---|
| §3.8 (encapsulación violada, sentinels `"unknown"`, `frame: object`, ids como `str` plano) | §3 (VOs) + §4.2 (firmas con VOs + `np.ndarray`) |
| §4 Caso A: `ZoneCounter`, `Aggregator`, `Broadcaster` sin Protocol. | Protocols #5, #6, #7, #8 |
| §4 Caso B: `multi_camera` importa `OpenCVVisualizer` de presentación. | Protocol #9 `FrameRenderer`. `OpenCVVisualizer` se reescribe como adaptador de infraestructura. |
| §6.10: API pública del Broadcaster (no más `broadcaster._subscribers`). | Métodos `subscriber_count()` e `is_subscribed()` en Protocol #8. |

### 4.6 Lo que esto NO resuelve todavía

Tres decisiones de Fase 1 que afectan a los Protocols pero NO se cierran
en esta sesión:

- **§6.11** (separación transporte/presentación en `Broadcaster`): el
  Protocol #8 tipa `publish(data: TrafficData)` que ya es una entidad
  del dominio. Pero la decisión formal de qué shape se serializa hacia
  suscriptores (¿`TrafficData` directo? ¿un DTO de presentación?) y
  dónde se traduce a strings localizadas vive en §6.11. **Pendiente.**
- **§6.9** (definiciones canónicas de métricas): los métodos del
  `Aggregator` retornan `TrafficData`, cuyos campos (`density`,
  `congestion_level`, `flow_rate_per_min`, etc.) se definen
  canónicamente en §6.9. **Pendiente.**
- **§6.6** (sync vs async pipeline, decisión final): esta Sesión 1
  adelantó la decisión de "dos Protocols separados" pero NO la decisión
  de "qué modo se conserva en Fase 5". §6.6 sigue pendiente para
  Sesión 3.

## 5. Estructura objetivo del `domain/` para Fase 3

```
edge_device/src/vision/domain/
├── __init__.py              # Barrel export
├── entities.py              # DetectedVehicle, ZoneVehicleCount, FrameAnalysis,
│                            # Frame, TrafficData
│                            # — sin sentinel "unknown"
│                            # — sin frame: object (Frame.image: np.ndarray)
│                            # — sin campos de debug (raw_detection_count, etc.)
│                            # — TrafficData.camera_id: CameraId (no opcional)
│                            # — con __post_init__ donde aplique validación
├── value_objects.py         # VehicleId, ZoneId, CameraId
│                            # — @dataclass(frozen=True) con __post_init__
├── protocols.py             # Los 9 Service Protocols listados en §4.2
│                            # — firmas con VOs
│                            # — return types explícitos
│                            # — frame: np.ndarray (no object)
└── repositories.py          # TrafficRepository (write-only en MVP1)
```

Tamaño estimado: ~250-350 LOC totales en `domain/`.

Cinco archivos. Cuatro ya existen en el módulo actual (`__init__.py`,
`entities.py`, `protocols.py`, `repositories.py`). Uno se agrega
(`value_objects.py`). El contenido de los cuatro existentes se reescribe.

## 6. Lo pendiente para Sesiones 2 y 3 de Fase 1

| Decisión | Bloque | Sesión |
|---|---|---|
| §6.9 — Definiciones canónicas de métricas | Bloque 2 | Sesión 2 |
| §6.10 — API pública del Broadcaster (cierre detallado más allá de los métodos votados acá) | Bloque 2 | Sesión 2 |
| §6.8 — Lugar del visualizer y de `interaction` | Bloque 2 | Sesión 2 |
| §6.11 — Separación transporte/presentación en Broadcaster | Bloque 3 | Sesión 3 |
| §6.6 — Modos de pipeline sync vs async (decisión final: cuál conservar) | Bloque 3 | Sesión 3 |
| §6.5 — Manejo de errores en workers async | Bloque 3 | Sesión 3 |
| §6.3 — Política de logging unificada | Bloque 4 | Sesión 3 |
| §6.4 — Configs YAML: fuente de verdad o ejemplo | Bloque 4 | Sesión 3 |

Sesión 2 cierra contratos por capa (entrega primera versión de OpenAPI
de `GET /vision/state` y de `vision_contract.md`). Sesión 3 cierra
decisiones derivadas y transversales.

## 7. Trazabilidad con Fase 0

Esta sesión cierra (parcialmente, los puntos listados en el aviso del
inicio del documento) las decisiones que Fase 0 dejó abiertas en
`tth-08-fase0-lecciones.md` §6:

- §6.1 → cerrada en §2 de este documento.
- §6.7 → cerrada en §3 de este documento.
- §6.2 → cerrada en §4 de este documento (con adelanto parcial de §6.6:
  decisión de "dos Protocols separados" para Aggregator).

Hallazgos de Fase 0 consumidos directamente:

- §3.2 (duplicación DRY entre sync y async aggregator) → resuelto por
  la decisión §4.4 Cambio 2 (separar cómputo de persistencia).
- §3.8 (encapsulación violada, sentinels `"unknown"`, `frame: object`,
  ids como `str` plano) → resueltos por §3 (VOs) y §4 (Protocols con
  VOs y tipado explícito de `frame: np.ndarray`).
- §4 Casos A, B (Protocols faltantes; `application` importa de
  `infrastructure` y `presentation`) → resueltos por §4.1 con los cinco
  nuevos Protocols.
- §6.1 (test de aceptación zone_counter) → cerrado con contrato preciso
  en §2.
- §6.10 (API pública del Broadcaster) → cerrado parcialmente con los
  métodos votados en §4.2 Protocol #8; el cierre detallado del shape
  serializado queda para Sesión 2.

Sin candidatos a DHU-025 emergentes de esta sesión. Las ocho decisiones
de DHU-024 siguen sin contradicción.
