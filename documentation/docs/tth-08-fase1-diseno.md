# TTH-08 Fase 1 — Diseño DDD del módulo `vision` reescrito

> **Documento parcial.** Esta versión cubre los cimientos del dominio
> (Bloque 1 del grafo de dependencias de §6 del documento de lecciones de
> Fase 0) y los contratos por capa (Bloque 2). Las decisiones que aún no
> se han tomado en Fase 1 están marcadas como **pendiente** y se cerrarán
> en Sesión 3 antes de que Fase 3 (Domain layer) arranque. Este documento
> NO debe leerse como el diseño completo de Fase 1.
>
> **Alcance cubierto:**
>
> Sesión 1 (Bloque 1):
> §6.1 (test de aceptación temprano del `ZoneCounter`),
> §6.7 (Value Objects),
> §6.2 (set completo de Protocols del dominio, parcialmente — con
> adelanto parcial de §6.6).
>
> Sesión 2 (Bloque 2):
> §6.9 (definiciones canónicas de métricas),
> §6.10 (cierre detallado del Broadcaster),
> §6.8 (lugar del visualizer y de `interaction.py`).

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

> **Nota (Sesión 3):** conteo al cierre de Sesión 1. El Protocol
> `SyncAggregator` se revirtió en §10 (§6.6); el conteo vigente es **8
> Service Protocols**. Ver §8.1 y §15.1.

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
| 6 | `SyncAggregator` | **Nuevo** | Cómputo síncrono de `TrafficData`. Cierra Caso A. **`SyncAggregator` — superseded por §10 (§6.6, Sesión 3): eliminado del dominio (async-only).** |
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


# Superseded por §10 (§6.6, Sesión 3): SyncAggregator se elimina del dominio
# (async-only). Bloque conservado como registro de Sesión 1; no implementar
# en Fase 3.
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
| **Organización en archivos** | 9 Service Protocols en `protocols.py`, 1 Repository en `repositories.py`. **Nota (Sesión 3): 8 vigentes tras revertir `SyncAggregator` en §10. Ver §8.1.** | Separación DDD ortodoxa Service vs Repository. Coincide con el `vision/` actual. |
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

> **Nota (Sesión 3):** §6.6 se cerró en §10 — async-only. La separación
> `SyncAggregator`/`AsyncAggregator` que esta sección anticipó se resolvió
> eliminando `SyncAggregator`; solo `AsyncAggregator` sobrevive en el
> dominio.

> **Nota (Sesión 3 / DHU-026, 2026-05-28):** el principio "caller persiste"
> declarado arriba quedó **superseded** en modo async-only (que es el único
> modo en MVP1 tras §10) por §11 + DHU-026: el **worker** del
> `AsyncAggregator` es responsable de la persistencia y del manejo de
> errores §11.1/§11.2. Save y output-queue son paths **independientes** del
> mismo `TrafficData` computado (un fallo de save no saca el item de la
> output queue; `flush()` retorna lo computado-y-no-dropeado, no lo
> persistido). Las otras consecuencias de este Cambio 2 (cómputo separado en
> función pura `_compute.py`, repositorio inyectado por constructor, tests
> con repo fake, `flush()` retorna `list[TrafficData]`) siguen vigentes. Ver
> DHU-026 para el razonamiento completo y el supersede explícito sobre §4.4
> Cambio 2 y sobre el docstring del Protocol `AsyncAggregator`.

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
| §4 Caso A: `ZoneCounter`, `Aggregator`, `Broadcaster` sin Protocol. | Protocols #5, #7, #8 (el #6 `SyncAggregator` se eliminó en §10; el rol de Aggregator lo cubre `AsyncAggregator` = #7). Ver §8.1. |
| §4 Caso B: `multi_camera` importa `OpenCVVisualizer` de presentación. | Protocol #9 `FrameRenderer`. `OpenCVVisualizer` se reescribe como adaptador de infraestructura. |
| §6.10: API pública del Broadcaster (no más `broadcaster._subscribers`). | Métodos `subscriber_count()` e `is_subscribed()` en Protocol #8. |

### 4.6 Lo que esto NO resuelve todavía (al cierre de Sesión 1)

Tres decisiones de Fase 1 que afectan a los Protocols pero NO se cierran
en Sesión 1:

- **§6.11** (separación transporte/presentación en `Broadcaster`): el
  Protocol #8 tipa `publish(data: TrafficData)` que ya es una entidad
  del dominio. Pero la decisión formal de qué shape se serializa hacia
  suscriptores y dónde se traduce a strings localizadas vive en §6.11.
  **Cerrado en Sesión 3 (ver §9)** (el shape estructurado del payload, §6,
  lo dejaba parcialmente cubierto).
- **§6.9** (definiciones canónicas de métricas): los métodos del
  `Aggregator` retornan `TrafficData`, cuyos campos se definen
  canónicamente en §6.9. **Cerrado en Sesión 2** (ver §5).
- **§6.6** (sync vs async pipeline, decisión final): Sesión 1 adelantó la
  decisión de "dos Protocols separados" pero NO la decisión de "qué modo
  se conserva en Fase 5". §6.6 **cerrado en Sesión 3 (ver §10)**:
  async-only, con eliminación del Protocol `SyncAggregator` (#6) introducido
  acá.

## 5. §6.9 — Definiciones canónicas de métricas del módulo `vision`

### 5.1 Hallazgo origen

La auditoría de Fase 0 identificó cuatro definiciones distintas de
"congestión/densidad" conviviendo en el repo (Fase 0 §3.5), el placeholder
`"incidents": 0` hardcoded (Fase 0 §3.6), el nombre engañoso
`flow_rate_per_min` que computa cardinalidad de IDs únicos en ventana
configurable (no veh/min), y la convivencia de tres escalas de congestión
(`Normal/High/Heavy`, `Bajo/Moderado/Alto`, escala 0-5 de D-009). §6.9
cierra las definiciones que vision emite para que estas inconsistencias
dejen de propagarse a los consumidores.

### 5.2 Decisión metodológica de fondo

**Vision reporta métricas nativas del sensor (conteos, IDs únicos, tipos,
velocidades) más derivables localmente con metadata disponible (ocupación
siempre; densidad real solo si la zona tiene calibración de longitud).**
Vision NO emite `congestion_level` discreto. La discretización a la escala
0-5 declarada en D-009 ocurre en la capa de presentación según el SDD
(SDD §2.3 y §3.2: "la discretización al nivel 0-5 ocurre solo en la capa
de presentación"). De ahí se sigue que vision no debe usurpar esa
responsabilidad emitiendo la escala discreta — el dato que vision podría
aportar es el ratio velocidad/free-flow continuo, no su discretización,
y eso queda como evaluación para F41 si llega a integrarse el módulo en
el loop predictivo.

Justificación:

1. Respeta la decisión arquitectural del SDD sobre dónde vive la
   discretización a 0-5 (ver párrafo anterior).
2. DHU-024 §5 declara que `vision_aggregates` incluye `count`, `queue`,
   `flow`, `density` por dirección — no incluye `congestion_level`.
3. Eliminar la lógica de discretización del módulo cierra el placeholder
   `"incidents": 0` y los umbrales 30/70 hardcoded del broadcaster
   (Fase 0 §3.6).

### 5.3 Schema canónico de `TrafficData`

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .value_objects import CameraId, ZoneId


@dataclass(frozen=True)
class TrafficData:
    """Aggregated traffic metrics for a single zone over a closed time window.

    This is the canonical output shape of the vision module's aggregation
    layer. It is the value object that the application use case persists
    via `TrafficRepository` and publishes via `Broadcaster`.

    All fields are required unless explicitly typed Optional.
    """

    # Identifiers
    camera_id: CameraId
    zone_id: ZoneId

    # Temporal window (closed interval)
    window_start: datetime           # tz-aware UTC
    window_end: datetime             # tz-aware UTC; window_end > window_start
    window_duration_seconds: float   # equal to (window_end - window_start).total_seconds(); > 0

    # Counts
    unique_vehicles: int                   # cardinality of distinct VehicleIds in window; >= 0
    vehicles_by_type: dict[str, int]       # keys subset of {"car", "bus", "truck", "motorcycle"};
                                           # sum of values equals unique_vehicles

    # Temporal aggregates
    mean_speed_kmh: Optional[float]        # count-weighted average; None if no frames have count > 0
                                           # or if camera has no pixel-to-meter calibration
    flow_vehicles_per_hour: float          # unique_vehicles / window_duration_seconds * 3600; >= 0

    # Local derived
    mean_occupancy: float                  # mean over frames of (bbox_area ∩ polygon_area) / polygon_area;
                                           # in [0.0, 1.0]
    density_vehicles_per_km: Optional[float]  # unique_vehicles / zone.segment_length_meters * 1000;
                                              # None if zone has no segment_length_meters configured;
                                              # >= 0 when not None
```

### 5.4 Definiciones campo por campo

**Identificadores:**

| Campo | Tipo | Definición |
|---|---|---|
| `camera_id` | `CameraId` (VO) | Identificador único de cámara. Obligatorio. Sin sentinel `"unknown"`. |
| `zone_id` | `ZoneId` (VO) | Identificador único de zona dentro de la cámara. Obligatorio. |

**Ventana temporal:**

| Campo | Tipo | Definición |
|---|---|---|
| `window_start` | `datetime` UTC tz-aware | Inicio del intervalo de agregación. |
| `window_end` | `datetime` UTC tz-aware | Fin del intervalo. Validación: `window_end > window_start`. |
| `window_duration_seconds` | `float` | `(window_end - window_start).total_seconds()`. Validación: `> 0`. Redundante pero útil para consumidores que no quieran calcular. |

**Cambio respecto al actual**: `timestamp: float` (Unix epoch) → `datetime`
tz-aware UTC + intervalo explícito (`window_start`, `window_end`). El módulo
actual usa un único punto temporal y deja la duración implícita por
configuración. Hacer el intervalo explícito elimina la ambigüedad que
causaba que `flow_rate_per_min` no fuera realmente veh/min.

**Conteos:**

| Campo | Tipo | Definición | Validación |
|---|---|---|---|
| `unique_vehicles` | `int` | Cardinalidad del set de `VehicleId` distintos observados en la ventana, tras resolución de tipo por voto mayoritario. | `>= 0` |
| `vehicles_by_type` | `dict[str, int]` | Para cada tipo, cantidad de `VehicleId` únicos cuyo tipo (voto mayoritario) es ese. | Keys ⊆ `{"car", "bus", "truck", "motorcycle"}` en MVP1; suma de valores = `unique_vehicles`. |

**Cambios respecto al actual:**

- `total_vehicles` → `unique_vehicles`. El nombre original sugería suma de
  detecciones; el cómputo real es cardinalidad de IDs únicos.
- `car_count` / `bus_count` / `truck_count` / `motorcycle_count` (cuatro
  campos) → `vehicles_by_type: dict[str, int]` (un dict). Si en F41 se
  agregan tipos (p.ej. "bicycle"), el schema no rompe. El `__post_init__`
  valida que las keys estén en el set permitido.

**Agregados temporales:**

| Campo | Tipo | Definición | Validación |
|---|---|---|---|
| `mean_speed_kmh` | `Optional[float]` | Promedio ponderado por count: `Σ(frame_avg_speed × frame_vehicle_count) / Σ(frame_vehicle_count)` sobre frames con `vehicle_count > 0` y `avg_speed > 0`. | `None` si no hay frames válidos o si la cámara carece de calibración pixel→metro. `>= 0` cuando no es `None`. |
| `flow_vehicles_per_hour` | `float` | `unique_vehicles / window_duration_seconds * 3600`. | `>= 0` |

**Cambios respecto al actual:**

- `avg_speed` → `mean_speed_kmh` con unidad en el nombre.
- `flow_rate_per_min: int` → `flow_vehicles_per_hour: float`. La unidad
  cambia a veh/h por convención de ingeniería de tráfico (HCM,
  Webster 1958), por alineación con el motor adaptativo
  (`PhaseFlowDC.flow` y `PEAK_THRESHOLD = 1500.0 veh/h` en
  `core_management_api/src/control/`), y porque el cómputo deja de mentir
  sobre la ventana temporal.

**Derivados locales:**

| Campo | Tipo | Definición | Validación |
|---|---|---|---|
| `mean_occupancy` | `float` | Promedio sobre frames de la ventana de `Σ(bbox_area ∩ polygon_area) / polygon_area`. Métrica primaria de "qué tan lleno está el carril visible". Ver **DHU-025 (Sesión 3 / Fase 5a, 2026-05-28)** para la interpretación del `Σ` como **unión** de bboxes (no suma aritmética con clip): el rango `[0.0, 1.0]` sin clip explícito + la semántica "fracción cubierta" prevalecen sobre la lectura literal del `Σ`. | `[0.0, 1.0]` |
| `density_vehicles_per_km` | `Optional[float]` | `unique_vehicles / zone.segment_length_meters * 1000`. Requiere que la zona configurada tenga el atributo `segment_length_meters` (calibración de campo). | `None` si la zona carece de `segment_length_meters`. `>= 0` cuando no es `None`. |

**Cambios respecto al actual:**

- `avg_density` (count promedio por frame, mal nombrado como densidad) →
  eliminado. Si un consumidor necesita count promedio por frame, lo deriva
  de `unique_vehicles` y la duración. No es métrica canónica.
- `avg_occupancy` → `mean_occupancy` y movido al rango fraccional [0.0, 1.0]
  (no `× 100` ni serializado como string). El broadcaster actual hacía la
  multiplicación a la salida (Fase 0 §3.5); esa traducción es trabajo de
  presentación, no del dominio.
- `density_vehicles_per_km` es campo nuevo. Implementa la decisión de
  Sesión 2 (ocupación primaria + densidad real opcional). Permite que
  vision reporte una métrica de ingeniería estándar cuando la zona está
  calibrada, sin obligar a que todas lo estén.

### 5.5 Configuración de zona requerida para densidad real

Para que `density_vehicles_per_km` se calcule, cada zona (polígono) debe
tener `segment_length_meters` en su configuración YAML:

```yaml
# edge_device/conf/vision/javier_prado.yaml (ejemplo)
zones:
  - id: "north_approach"
    polygon: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    segment_length_meters: 50.0   # opcional; sin esto, density_vehicles_per_km = None
```

Sin `segment_length_meters`, vision reporta `density_vehicles_per_km = None`.
Con él, reporta el valor calculado. La calibración es trabajo de campo
fuera del scope de TTH-08; las zonas existentes en MVP1 pueden quedar sin
calibración inicial.

### 5.6 Campos eliminados respecto al `TrafficData` actual

| Campo eliminado | Razón |
|---|---|
| `street_monitored` | Metadata de la cámara, no del agregado temporal. Repetirlo por agregado rompe normalización. Consumidores que necesiten el nombre de la calle lo resuelven por `camera_id` contra metadata de cámara (entidad aparte). Implica que la capa de presentación enriquece el payload del broadcaster con metadata de cámara cuando la HU lo requiere (ver §6.4 de este documento). |
| `total_vehicles` redundante con `unique_vehicles` | Mismo cómputo, mejor nombre. |
| `vehicle_types: dict` redundante con `vehicles_by_type` | Mismo contenido, eliminado del schema canónico. |
| `duration_seconds` como único campo temporal | Reemplazado por `window_start` + `window_end` + `window_duration_seconds` derivado. |
| Campos de debug (`raw_detection_count`, `display_queue`, `interpolate`) | No son del dominio (Fase 0 §3.8 atributos muertos). |

### 5.7 Métricas que vision NO emite

| Métrica | Razón |
|---|---|
| `congestion_level` (escala discreta) | D-009 declara escala 0-5 como variable de estado canónica del sistema; la discretización vive en presentación o en consumidor (predictor de TTH-09), no en vision. |
| `"incidents"` | El placeholder hardcoded del broadcaster actual desaparece. La detección de incidentes no es responsabilidad de vision en MVP1. |
| `pedestrians` | El detector YOLO está configurado solo para `car/bus/truck/motorcycle` (Fase 0 §3.8); contar peatones siempre dio 0. Si en F41 se quiere contar peatones, se agrega al detector y a `vehicles_by_type` (con key `"pedestrian"`). |
| `queue` (longitud de cola) | DHU-024 §5 lo declara como campo de `vision_aggregates`, pero el módulo actual no lo calcula y MVP1 no lo requiere (el motor adaptativo recibe `queue` desde SUMO en validación cuantitativa, no de vision; D-007). Postergado a F41. El schema de `vision_aggregates` mantiene la columna `queue` como `NULL` en MVP1. |

### 5.8 Compatibilidad con `CameraTrafficData` y `vision_aggregates`

> **Nota de Fase 6 (2026-05-28) — divergencia diseño-vs-código resuelta a favor del código.** Esta sección fue escrita en Sesión 1 asumiendo que la BD seguiría siendo legacy (shape `CameraTrafficData`) y que Fase 6 tendría que construir un adapter `TrafficData → CameraTrafficData → BD`. La migración Fase 2/3 (Alembic + DDD) **rehízo `vision_aggregates` con shape canónico** (`unique_vehicles`, `mean_occupancy`, `flow_vehicles_per_hour`, etc., sin `total_vehicles`/`occupancy_rate`/`flow_rate_per_min`/`street_monitored`), y Fase 4c materializó el mapping directo `TrafficData → columnas canónicas` en `PostgresTrafficRepository._to_row()` (`edge_device/src/vision/infrastructure/persistence/postgres_repository.py`). El paso intermedio "→ `CameraTrafficData`" del texto original nunca se implementó porque dejó de tener sentido cuando la BD se rehízo canónica. Por tanto: **no existe ni se construirá un adapter `TrafficData → CameraTrafficData` en Fase 6** — el mapping real vive en `_to_row()` contra el shape canónico de `vision_aggregates`, sin intermediario. La tabla de mapeo de campos a `CameraTrafficData` que sigue queda como **referencia histórica**, no como contrato vivo. `CameraTrafficData` (en `shared/cerebrovial_shared/schemas/camera.py`) queda **huérfana**: solo es referenciada por su definición + un comentario en `core_management_api/scripts/generate_camera_data.py`, sin consumidor runtime. Su borrado requiere coordinación de `shared/` (territorio común al proyecto) y queda nominado fuera de TTH-08. Mismo patrón de registro de divergencia que DHU-025 (Σ→unión en §5.4) y DHU-026 (caller→worker en §4.4/§11).

El `TrafficData` canónico **no es idéntico** a `CameraTrafficData` (schema
actual en `shared/cerebrovial_shared/schemas/camera.py`). DHU-024 §5 declara
compatibilidad, lo que significa que existe un adapter que traduce — no
que los schemas son iguales campo por campo.

Mapeo del adapter (referencia histórica — *no implementado* como objeto; el mapping real vive en `_to_row()` contra columnas canónicas):

| Campo `TrafficData` canónico | Campo `CameraTrafficData` actual | Traducción |
|---|---|---|
| `camera_id: CameraId` | `camera_id: str` | `td.camera_id.value` |
| `window_end: datetime` | `timestamp: float` | `td.window_end.timestamp()` |
| `unique_vehicles: int` | `total_vehicles: int` | directo |
| `vehicles_by_type["car"]` | `car_count: int` | desempaquetado del dict (`vehicles_by_type.get("car", 0)`) |
| `vehicles_by_type["bus"]` | `bus_count: int` | idem |
| `vehicles_by_type["truck"]` | `truck_count: int` | idem |
| `vehicles_by_type["motorcycle"]` | `motorcycle_count: int` | idem |
| `mean_occupancy: float` | `occupancy_rate: float` | directo (mismo rango [0.0, 1.0]) |
| `flow_vehicles_per_hour: float` | `flow_rate_per_min: int` | `int(td.flow_vehicles_per_hour / 60)` |
| — | `street_monitored: str` | Enriquecimiento desde metadata de cámara |

**Alcance del adapter**: este mapping cubre **el camino de persistencia**
(`TrafficData` → `CameraTrafficData` → BD). NO cubre otros consumidores
que también reciben campos por nombre y que cambian con esta sesión:

- **Payload SSE del broadcaster** (§6.2 de este documento): el frontend
  consume el JSON enriquecido directamente. Consumidores conocidos:
  `frontend_ui/src/components/views/CameraDetailView.tsx`,
  `DashboardView.tsx`, `TrafficHistoryWidget.tsx`,
  `frontend_ui/src/services/predictionService.ts`.
- **Endpoint del predictor** (`core_management_api/src/prediction/`):
  `PredictionInput` referencia `total_vehicles`, `flow_rate_per_min`,
  `avg_speed`, `occupancy_rate` por nombre exacto.
- **`csv_loader.py` del predictor**: feature columns con nombres
  antiguos.

Estos consumidores se ajustan en Fase 6 (presentación) — no se cubren
con el adapter de persistencia de §5.8. La trazabilidad del cambio
queda en este párrafo para que Fase 6 enumere y actualice los archivos
afectados sin tener que redescubrirlos.

**Estado real al cierre del batch 1 de Fase 6** (ver nota al inicio de la sección): el frontend SSE (CameraDetailView, DashboardView, TrafficHistoryWidget, predictionService) se migra en la sub-fase 6g al payload §6.2. El predictor (`PredictionInput`, `csv_loader.py`) queda **fuera de Fase 6**: lee CSVs históricos sin writer (Fase 5f eliminó el CSV-persistencia) y no consume `vision_aggregates`; alimentar el RandomForest fallback es responsabilidad de TTH-04/TTH-09. Documentado en el handoff de cierre de Fase 6 como hallazgo colateral.

## 6. §6.10 — Cierre detallado del Broadcaster

### 6.1 Lo que Sesión 1 ya cerró

Sesión 1 (este documento, §4.4 Cambio 3) votó:

1. Firma del Protocol: `async def publish(data: TrafficData)`,
   `subscriber_count() -> int`, `is_subscribed(subscriber_id) -> bool`.
2. Cero acceso externo a `_subscribers` (atributo privado).
3. `TrafficData.camera_id` no-opcional (consecuencia de §3 VOs).

Esta sección extiende esas decisiones con el shape exacto del payload SSE,
política de eventos, política de suscriptores nuevos, y schema concreto de
`vision_aggregates`.

### 6.2 Shape del payload SSE

El broadcaster publica un payload **enriquecido** (TrafficData + metadata
de cámara), no el `TrafficData` puro. Razón: el frontend que consume el
SSE necesita `street_monitored` para HU-02 (panel de la intersección).
Mandar solo `TrafficData` obligaría a una segunda llamada para metadata.
El enriquecimiento ocurre dentro del broadcaster concreto (o en un adapter
justo antes de publicar) — NO cambia el Protocol del dominio.

```json
{
  "schema_version": "1.0",
  "event_type": "traffic_update",
  "server_timestamp": "2026-05-27T22:48:30Z",
  "camera": {
    "id": "cam_javier_prado_01",
    "street_monitored": "Av. Javier Prado Este"
  },
  "zone": {
    "id": "north_approach"
  },
  "window": {
    "start": "2026-05-27T22:48:00Z",
    "end": "2026-05-27T22:48:30Z",
    "duration_seconds": 30.0
  },
  "metrics": {
    "unique_vehicles": 24,
    "vehicles_by_type": {"car": 18, "bus": 2, "truck": 1, "motorcycle": 3},
    "mean_speed_kmh": 32.5,
    "flow_vehicles_per_hour": 2880.0,
    "mean_occupancy": 0.47,
    "density_vehicles_per_km": 48.0
  }
}
```

Decisiones de shape implícitas:

- **`schema_version`** explícito desde el principio. Cuando F41 agregue
  `queue` o nuevos campos, los consumidores pueden chequear versión.
- **Agrupación por concepto** (`camera`, `zone`, `window`, `metrics`) en
  vez de flat. Más legible, más estructurado.
- **Valores Optional como ausencia explícita**: `mean_speed_kmh: null`
  cuando no hay calibración, `density_vehicles_per_km: null` cuando no hay
  `segment_length_meters`. JSON nativo, sin convención propietaria.
- **`server_timestamp`** además del `window.end`: el primero es cuándo el
  broadcaster emite; el segundo es el fin de la ventana de datos. Útil
  para medir latencia broadcaster→cliente (RNF-PERF-01).

### 6.3 Política de tipos de evento

**Único tipo de evento en MVP1: `traffic_update`.**

Razones:
- Mínimo viable. No agregar tipos de eventos sin caso de uso concreto.
- Los frameworks SSE (EventSource del browser, sse-starlette en el backend)
  ya manejan reconexión nativamente; no se necesitan eventos de control
  (`connected`, `heartbeat`).
- `heartbeat` se evalúa si en operación aparece evidencia de conexiones
  muertas no detectadas — no se anticipa.

### 6.4 Política de suscriptores nuevos

**Cache de último estado por zona preservada.** Cuando un suscriptor se
conecta, el broadcaster le envía inmediatamente el último `TrafficData`
conocido para cada zona suscrita. El suscriptor no espera al próximo
broadcast natural.

Razones:
- Es lo que el broadcaster actual ya implementa y Fase 0 §2 lo identificó
  como "patrón bien hecho".
- Buena UX: el operador que abre HU-02 ve datos inmediatamente, no espera
  hasta el próximo agregado (que puede ser hasta 30s después).

**Implicación para la implementación concreta**: el broadcaster mantiene
internamente un dict `{ZoneId: TrafficData}` con el último valor. Este es
atributo privado de la implementación, NO parte del Protocol del dominio.
El Protocol del dominio sigue siendo el de Sesión 1.

### 6.5 Schema de `vision_aggregates`

DHU-024 §5 fija el shape conceptual: `{intersection_id, timestamp,
directions: [{direction, count, queue, flow, density}]}`. Este documento
materializa ese shape como tabla TimescaleDB con un row por zona (no array).

```sql
CREATE TABLE vision_aggregates (
    -- Identificadores
    camera_id TEXT NOT NULL,
    zone_id TEXT NOT NULL,

    -- Ventana temporal
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    window_duration_seconds DOUBLE PRECISION NOT NULL,

    -- Conteos
    unique_vehicles INTEGER NOT NULL CHECK (unique_vehicles >= 0),
    car_count INTEGER NOT NULL DEFAULT 0 CHECK (car_count >= 0),
    bus_count INTEGER NOT NULL DEFAULT 0 CHECK (bus_count >= 0),
    truck_count INTEGER NOT NULL DEFAULT 0 CHECK (truck_count >= 0),
    motorcycle_count INTEGER NOT NULL DEFAULT 0 CHECK (motorcycle_count >= 0),

    -- Agregados temporales
    mean_speed_kmh DOUBLE PRECISION,                            -- NULL si sin calibración
    flow_vehicles_per_hour DOUBLE PRECISION NOT NULL CHECK (flow_vehicles_per_hour >= 0),

    -- Derivados locales
    mean_occupancy DOUBLE PRECISION NOT NULL CHECK (mean_occupancy BETWEEN 0 AND 1),
    density_vehicles_per_km DOUBLE PRECISION,                   -- NULL si zona sin segment_length

    -- Cola (F41, NULL en MVP1)
    queue INTEGER,                                              -- NULL en MVP1; F41 lo poblará

    PRIMARY KEY (camera_id, zone_id, window_start)
);

-- TimescaleDB: hypertable por window_start
SELECT create_hypertable('vision_aggregates', 'window_start');
```

Decisiones del schema:

- **Un row por zona** (no array de `directions[]` por cámara). TimescaleDB
  optimiza queries sobre rows planos por `(camera_id, zone_id, timestamp)`.
  El shape `{intersection_id, directions: [...]}` de DHU-024 §5 es
  representación de respuesta API (el endpoint `GET /vision/state` agrupa
  zonas de una cámara al servir); la BD las almacena planas.
- **Desempaquetado de `vehicles_by_type`** a cuatro columnas en la tabla.
  En el dominio (`TrafficData`) es dict porque facilita evolución; en la
  tabla son columnas porque facilita queries (`WHERE bus_count > 5` es
  directo). El adapter desempaqueta al persistir.
- **PK compuesta `(camera_id, zone_id, window_start)`** garantiza
  idempotencia. Si el aggregator reenvía la misma ventana por reintento,
  `INSERT ... ON CONFLICT DO NOTHING` lo absorbe sin duplicar.
- **`mean_occupancy` y `density_vehicles_per_km` como columnas separadas**,
  no un único campo `density`. Reinterpretación honesta de DHU-024 §5
  (cuando se redactó, no se distinguían las dos métricas; §6.9 las separó).
- **`queue INTEGER` nullable** en MVP1. F41 lo poblará cuando se calcule
  cola; por ahora `NULL` explícito.
- **`camera_id` en la tabla equivale a `intersection_id` del shape de
  DHU-024 §5**. La nomenclatura difiere por contexto: DHU-024 usa
  `intersection_id` desde la perspectiva del consumidor (motor y
  presentación piensan en intersecciones); el módulo `vision` usa
  `camera_id` porque cada cámara monitorea una intersección y el VO
  ya está nombrado `CameraId` (§3). En el endpoint `GET /vision/state`,
  el campo se serializa como `intersection_id` para alinear con la
  nomenclatura del consumidor; en BD y dominio se mantiene `camera_id`.
- **`TIMESTAMPTZ` desviación deliberada respecto a la convención del
  repo**. Las migraciones actuales (`775d2d1db8b4_initial_schema.py`
  y similares) usan `sa.DateTime()` que traduce a `TIMESTAMP WITHOUT
  TIME ZONE`. Esta sesión elige `TIMESTAMPTZ` para `vision_aggregates`
  porque (a) los `datetime` del dominio (`window_start`/`window_end`)
  son tz-aware UTC por decisión de §5.4, y mapearlos a `TIMESTAMP`
  sin tz pierde la garantía; (b) `TIMESTAMPTZ` es la forma idiomática
  para series temporales en TimescaleDB. Si Fase 2 (migración Alembic
  real) detecta inconsistencia con otras tablas del schema, se
  reconsidera entonces.

La migración Alembic que crea `vision_aggregates` es trabajo de Fase 2
(DHU-024 §2 junto con el borrado de `vision_tracks`/`vision_flows`).
El SQL listado arriba es **ilustrativo del contrato**; la migración
Alembic real lo materializa con SQLAlchemy + `op.execute("SELECT
create_hypertable(...)")` siguiendo el patrón del repo
(`daec5fdcfcdd_timescaledb_hypertables.py` como referencia), con
`chunk_time_interval => INTERVAL '1 day'` e `if_not_exists => TRUE`.

### 6.6 Cobertura de hallazgos de Fase 0

| Hallazgo de Fase 0 | Cubierto por |
|---|---|
| §3.5 (dos definiciones de "density") | §5.4 (`mean_occupancy` canónica como fracción [0,1]; `avg_density` count-por-frame eliminada). El broadcaster en §6.2 ya emite la versión canónica sin `× 100`. |
| §3.6 (placeholder `"incidents": 0` hardcoded) | §5.7 (vision no emite `incidents`). |
| §4 Caso D (broadcaster mezcla transporte y presentación) | Parcialmente cubierto: el shape del payload de §6.2 es estructurado y tipado; queda pendiente §6.11 (Sesión 3) para la decisión formal de qué se considera presentación pura vs transporte del broadcaster. |
| §6.10 (API pública del Broadcaster) | Cerrado: §6.1 (Sesión 1) define los métodos públicos; §6.2-§6.4 (esta sesión) extienden con shape, eventos y política de suscriptores. |

## 7. §6.8 — Lugar del visualizer y de `interaction.py`

### 7.1 Hallazgos origen (Fase 0)

- **Caso B (§4 de Fase 0)**: `application/services/multi_camera.py:11`
  importa `OpenCVVisualizer` de `presentation/visualization/opencv_visualizer.py`.
  La aplicación NO debe conocer la presentación.
- **Caso C (§4 de Fase 0)**: `infrastructure/interaction.py` usa
  `cv2.namedWindow`, `cv2.imshow`, `cv2.setMouseCallback`, `cv2.waitKey`.
  Es UI, no infraestructura.

### 7.2 Decisión 1 — Visualizer (Caso B)

**`OpenCVVisualizer` se queda en `presentation/`, pero se inyecta vía
Protocol `FrameRenderer` del dominio (definido en Sesión 1, §4.2 Protocol #9).**

Estructura final:

- `domain/protocols.py` declara `FrameRenderer` Protocol (ya hecho en Sesión 1).
- `presentation/visualization/opencv_visualizer.py` implementa
  `FrameRenderer` con OpenCV.
- `application/services/multi_camera.py` recibe un `FrameRenderer` por
  constructor (inyección de dependencia). NO importa la clase concreta.

Razón: invierte la dirección de la dependencia. `application` depende del
Protocol del dominio (correcto); la implementación concreta vive en
`presentation` (correcto); la composición ocurre en el `pipeline_builder.py`
o en el entry point que ya hace wiring.

**Detalle de implementación**: el `FrameRenderer.render()` devuelve
`np.ndarray` (no `Frame`). Eso significa que el visualizer NO muta entidades
del dominio — produce un array nuevo. Esto previene el bug que la auditoría
destapó en otro lugar (Fase 0 §3.8: `speed_estimator.py:53` mutaba
`vehicle.speed = ...` in-place).

### 7.3 Decisión 2 — `interaction.py` (Caso C)

**`interaction.py` se mueve a `edge_device/scripts/calibrate_zones.py`
(rename + ubicación fuera del módulo `vision`).**

Razones:

- `interaction.py` es un **script offline** para calibrar polígonos a mano.
  No es parte del runtime de vision.
- `presentation/` en el módulo `vision` es la capa de exposición runtime
  (FastAPI routes, visualizer durante operación). Mezclar `interaction.py`
  con presentation runtime ensucia ambos conceptos.
- La herramienta es útil: §5.5 establece que la densidad real requiere
  `segment_length_meters` por zona, lo que va a requerir trabajo de
  calibración. Eliminar la herramienta obligaría a reescribirla.
- El rename de `interaction.py` a `calibrate_zones.py` no es cosmético —
  el nombre actual no dice qué hace. Fase 0 §3.8 identificó nombres
  engañosos como patrón a no replicar.

### 7.4 Trazabilidad con DHU-024

DHU-024 §3 dice que infraestructura técnica reutilizable puede preservarse.
`calibrate_zones.py` es exactamente eso: infraestructura técnica genérica
que se reaprovecha **fuera del módulo `vision`**, no en él. La decisión es
coherente.

## 8. Estructura objetivo del módulo para Fase 3

### 8.1 Árbol del `domain/` (Fase 3)

```
edge_device/src/vision/domain/
├── __init__.py              # Barrel export
├── entities.py              # DetectedVehicle, ZoneVehicleCount, FrameAnalysis,
│                            # Frame, TrafficData (schema de §5.3)
│                            # — sin sentinel "unknown"
│                            # — sin frame: object (Frame.image: np.ndarray)
│                            # — sin campos de debug (raw_detection_count, etc.)
│                            # — TrafficData.camera_id: CameraId (no opcional)
│                            # — sin street_monitored (movido a metadata de cámara)
│                            # — con __post_init__ donde aplique validación
├── value_objects.py         # VehicleId, ZoneId, CameraId
│                            # — @dataclass(frozen=True) con __post_init__
├── protocols.py             # 8 Service Protocols (§4.2 listaba 9; SyncAggregator
│                            #   eliminado en §10 / §6.6 — async-only)
│                            # — firmas con VOs
│                            # — return types explícitos
│                            # — frame: np.ndarray (no object)
└── repositories.py          # TrafficRepository (write-only en MVP1)
```

Tamaño estimado: ~250-350 LOC totales en `domain/`.

Cinco archivos. Cuatro ya existen en el módulo actual (`__init__.py`,
`entities.py`, `protocols.py`, `repositories.py`). Uno se agrega
(`value_objects.py`). El contenido de los cuatro existentes se reescribe.

### 8.2 Árbol del módulo completo (consecuencia de §6.8)

```
edge_device/src/vision/
├── domain/               # 8 Service Protocols + 1 Repository (§4 menos
│                         #   SyncAggregator por §10; ver §8.1)
├── application/          # Use cases, aggregator async, pipeline async
│                         # — sin SyncAggregator ni sync_pipeline (async-only, §10)
│                         # — sin imports a presentation (Caso B resuelto en §7.2)
│                         # — sin interaction.py (Caso C resuelto en §7.3)
├── infrastructure/       # YOLO, tracker, sources, persistence, broadcast, zones,
│                         #   config (Settings VISION_ + loader de calibración, §13)
│                         # — interaction.py YA NO está acá
└── presentation/         # FastAPI routes, visualizer (implementa FrameRenderer)

edge_device/scripts/
└── calibrate_zones.py    # — antiguo interaction.py, movido y renombrado (§7.3)
```

Notas de Sesión 3 sobre esta estructura:

- **Config (§13)**: la config operativa (`Settings` con `env_prefix="VISION_"`)
  y el loader de calibración espacial viven en `infrastructure/config/`,
  sin introducir capas nuevas. La operativa se instancia como singleton al
  boot; la calibración se lee de YAML (fuente de verdad) vía loader
  explícito.
- **Logging (§12)**: el entrypoint del módulo invoca `setup_logger` de
  `cerebrovial_shared` **una sola vez** al boot (antes de levantar
  pipeline/app); los `getLogger(__name__)` de cada archivo heredan handler
  y formato.

## 9. §6.11 — Separación transporte/presentación en el Broadcaster

### 9.1 Decisión

El broadcaster **no contiene traducción de presentación**. Transporta
estructuras puras: el payload SSE definido en §6.2, es decir `TrafficData`
enriquecido con metadata de cámara, tipos numéricos puros, `datetime`
ISO-8601 tz-aware, y Value Objects serializados como su `str`. La
localización y el formateo viven aguas abajo del transporte.

### 9.2 Prohibido dentro de `infrastructure/broadcast/`

Queda explícitamente fuera del broadcaster (es transporte, no presentación):

- Strings localizadas de estado (`"Bajo"` / `"Moderado"` / `"Alto"`).
- Niveles semánticos con umbrales hardcoded (los `30` / `70` actuales).
- Formateos con `%` o sufijos de unidad (`"32.5 km/h"`, `"47%"`).
- Campos placeholder (`incidents`, `pedestrians`).
- Cualquier mapeo a la escala 0-5 de Waze (vive en presentación /
  predictivo, D-009).

### 9.3 Consecuencia para Fases 5-6

Cualquier localización ocurre en el cliente o en un adapter de
`presentation/api/` separado del transporte. **Hipótesis fuerte**: ningún
consumidor del MVP1 requiere strings localizadas en el SSE — el frontend
consume números crudos (SDD §5.2). Si esa hipótesis cae, el adapter de
presentación es el único lugar donde se introduce traducción; el
broadcaster no se toca.

### 9.4 Campos eliminados

- **`pedestrians`**: no se reimplementa el cálculo. El detector está
  configurado para `car` / `bus` / `truck` / `motorcycle`; no hay clase
  peatón en el pipeline.
- **`incidents`**: se elimina del módulo (era placeholder hardcoded,
  Fase 0 §3.6).

### 9.5 Cierre

Cierra el **Caso D de §4 de Fase 0** (broadcaster mezcla transporte y
presentación), que el shape estructurado del payload (§6.2) había dejado
parcialmente cubierto.

## 10. §6.6 — Modos de pipeline: async-only en MVP1

### 10.1 Decisión

**Async-only.** Se elimina `VisionPipeline` (síncrono) del módulo nuevo
por completo. Se elimina también el Protocol `SyncAggregator` del dominio
(introducido en §4.1 / §4.2 como Protocol #6): un tipo sin consumidor es
ruido. Si en el futuro vuelve a hacer falta el modo síncrono, se
reintroduce explícitamente, con su caller.

### 10.2 Razón

El modo síncrono no tiene caller productivo hoy. Los dos requisitos
operativos de CT-08.7 — video grabado y stream en vivo — los cubre el
pipeline async. Eliminar el sync **cierra de raíz** la duplicación DRY de
§3.2 de Fase 0 (cómputo de agregados copy-paste entre sync y async
aggregator), no solo la mitiga: al quedar un único aggregator, no hay dos
implementaciones que mantener en sincronía.

### 10.3 Validación CT-08.9 (≥200 frames)

Script aparte que invoca el detector directamente, sin pipeline completo.
No necesita broadcaster, aggregator ni persistencia: mide detección, no el
flujo end-to-end.

### 10.4 Nota de frontera con HU-02

`GET /vision/state` expone un shape compatible con el contrato agnóstico de
fuente que HU-02 consume. HU-02 no está acoplada a la procedencia del dato;
en MVP1 se cablea a SUMO (D-007, D-008), y el path visión→HU-02 queda como
operación hipotética / trabajo futuro (F41). Async-only sostiene que el
contrato de estado sea robusto en tiempo real, condición para que sea
intercambiable con SUMO como fuente.

### 10.5 Catch-up logic (requisito de fluidez, NO optimización opcional)

El descarte de frames intermedios de procesamiento para mantener sincronía
con el reloj de reproducción es **requisito de fluidez visual** del stream
de frames anotados (CT-08.8), no una mejora opcional. Fase 5 debe
preservarlo. El mecanismo **difiere por modo**:

- **STREAM en vivo**: se dropean frames para no atrasarse del tiempo real.
- **VIDEO GRABADO**: se regula el ritmo de salida al fps objetivo (24/30
  fps) en vez de dropear.

Fase 5 no debe asumir un único mecanismo para ambos modos.

## 11. §6.5 — Manejo de errores en workers async

Aplica porque §6.6 conservó el modo async. Tres reglas.

### 11.1 Regla 1 — errores de `save`

El worker captura la excepción, hace `logger.exception` (captura el
traceback, **no** `print`), incrementa el contador `aggregation_errors`, y
**continúa el loop**. No reintenta. El worker nunca mata el pipeline
principal de captura/detección: un fallo de persistencia degrada la serie
histórica, no la operación en vivo.

### 11.2 Regla 2 — cola llena (DROP-NEWEST)

Política **drop-newest**, idéntica al patrón del `ActiveStateBroadcaster`
del core (`core_management_api/src/control/infrastructure/broadcaster.py:60-68`):
ante `asyncio.QueueFull` se descarta la ventana **entrante** con
`logger.warning` + contador `data_dropped`; se conserva lo ya encolado.

**Diferencia de contexto con el broadcaster** (importante): el broadcaster
SSE tolera drop-newest porque el cliente re-lee el estado autoritativo de
la BD al reconectar (BD = fuente de verdad). El worker de persistencia **no
tiene ese fallback** — la BD es el destino, no una fuente alterna. Por eso:

- La cola del aggregator se dimensiona **más grande** que la del
  broadcaster SSE (`maxsize=32`): valor parametrizable por env, default
  sugerido **~256**, porque la persistencia tolera más latencia que un
  canal de tiempo real.
- Drop-newest mantiene la serie histórica **contigua** hasta el punto de
  saturación, en vez de intercalar huecos.

### 11.3 Regla 3 — observabilidad

Los contadores `aggregation_errors` y `data_dropped` se exponen en el
health check del módulo (CT-08.10). Sin Prometheus ni métricas formales en
MVP1: son enteros en el payload del health check, señal de degradación.

### 11.4 Lo que NO se hace

- Sin `tenacity` / `backoff`: el repo no usa ninguna librería de reintentos
  en ningún módulo (confirmado en la auditoría de Sesión 3).
- Sin dead-letter queue.

## 12. §6.3 — Política de logging unificada

### 12.1 Hallazgo que reencuadra Fase 0

El patrón canónico del backend (core) es `logging.getLogger(__name__)`
directo: 4 archivos productivos lo usan, **cero** consumen
`cerebrovial_shared.logging`. Hoy `setup_logger` solo lo usan
`edge_device/src/vision/` (a reescribir) y scripts de
`ia_prediction_service/`. No hay setup central en ningún entrypoint —
`core_management_api/src/main.py` ni siquiera importa `logging`.

### 12.2 Decisión

El módulo nuevo usa `logging.getLogger(__name__)` por archivo, e invoca
`setup_logger` de `cerebrovial_shared` **una sola vez** en el boot del
módulo de visión (antes de levantar pipeline/app), con nivel `INFO`
configurable por ENV. Los `getLogger(__name__)` heredan handler y formato
de la raíz.

`print()` queda prohibido en código productivo. Único lugar permitido:
scripts CLI explícitos — la calibración de zonas
(`edge_device/scripts/calibrate_zones.py`, §7.3) y el script de validación
CT-08.9. Aun ahí, preferir logger.

### 12.3 Matices confirmados en la confrontación con el repo

- **(a) Formatter hardcodeado.** `setup_logger` fija
  `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`, sin
  `threadName`. Para MVP1 se acepta tal cual: con un solo worker de
  persistencia, el `%(name)s` (módulo del aggregator) basta para
  identificar el origen. Si en el futuro hay múltiples workers, se
  parametriza el formatter en `cerebrovial_shared` (no vedado). Se registra
  como nota; no se actúa ahora.
- **(b) `log_execution_time` es decorador síncrono.** Aplicado a una
  coroutine mide la creación del coroutine, no su ejecución. **No
  aplicarlo a código async** (workers, handlers async). Sí sirve para
  métodos síncronos (el detector ya lo usa correctamente).

### 12.4 Nota al margen (deuda fuera de alcance de TTH-08)

El core no tiene configuración central de logging (`main.py` no importa
`logging`). Se registra para que un sprint futuro pueda unificarlo; **no se
actúa sobre el core en TTH-08**.

## 13. §6.4 — Configs: híbrido por clase de configuración

### 13.1 Hallazgo

Tres patrones sin convergencia en el repo: el core con `pydantic-settings`
env-only; el `ConfigManager` de `cerebrovial_shared` con OmegaConf,
importado en **0 sitios**; `ia_prediction_service` con `yaml.safe_load`
directo. El patrón canónico del backend (core) **no usa YAML**: usa
`pydantic-settings` con env vars + defaults en código.

### 13.2 Decisión: separar dos clases de config

#### Clase 1 — config OPERATIVA

Nivel de logging, fps objetivo, umbral de cola (km/h), frecuencia de
persistencia, tamaño de queue del aggregator, origen CORS, detección cada N
frames.

- `pydantic-settings` con `env_prefix="VISION_"`, paralelo al `CONTROL_`
  del core.
- Precedencia **env > defaults en código**.
- Instanciado como **singleton al boot** (una vez a nivel módulo, como
  `ControlSettings` en `main.py:36`), **no por request**.
- **Consecuencia**: los perfiles operativos `balanced.yaml` /
  `low_latency.yaml` **dejan de ser fuente de verdad**; sus valores pasan a
  defaults en código o a env. Esos YAMLs se eliminan o se mueven a `docs/`
  como ejemplos.

#### Clase 2 — config ESPACIAL CALIBRADA por intersección

`javier_prado.yaml` (polígonos ROI de CT-08.3, escala espacial de CT-08.2,
longitud de accesos para densidad) y `vehicle_classes.yaml` (clases YOLO).

- Archivo **YAML, FUENTE DE VERDAD**, leído al boot vía loader explícito.
- **No se duplica en código** (elimina el problema de `cameras.py:70-85`,
  que reconstruye en código valores ya presentes en los YAMLs).
- La ruta del archivo de calibración activo se indica por env var (p.ej.
  `VISION_CALIBRATION_PATH`).

### 13.3 `ConfigManager` y dependencias

`ConfigManager` / `load_vision_config` de `cerebrovial_shared` (importado
en 0 sitios, confirmado): se **descarta**. El loader de calibración se
escribe con `pydantic.BaseModel` (validación de estructura), coherente con
el core y sin arrastrar OmegaConf.

**Nota de dependencia**: el `Settings` `VISION_` requiere
`pydantic-settings` en `edge_device/`, que **hoy no lo tiene**
(`edge_device/requirements.txt` no incluye `pydantic` ni
`pydantic-settings`). Agregar la dependencia es trabajo de **Fase 4
(infraestructura)**, no de Sesión 3. El loader de calibración con
`BaseModel` puro depende también de `pydantic`, igualmente ausente hoy en
`edge_device/` (misma nota).

### 13.4 Duplicación de YAMLs

Los 5 YAMLs de `core_management_api/conf/vision/` son **byte-idénticos**
(md5 confirmado) a los de `edge_device/conf/vision/`. Ubicación canónica
del YAML de calibración: `edge_device/conf/vision/` (el módulo vive ahí; el
core no lo consume en MVP1). La copia en `core_management_api/conf/vision/`
se elimina **en Fase 2** (Sesión 3 solo lo decide; el borrado físico es
trabajo de Fase 2).

### 13.5 CORS — corrección de alcance

§6.4 **no "cierra"** el CORS de §3.4 de Fase 0. Lo que hace: el módulo nuevo
de visión configura CORS por env var (origen explícito, default
restrictivo, **nunca `["*"]` con credenciales**). Crea el patrón en visión
porque el repo **no tiene precedente** de CORS-por-env-var.

**Nota al margen (deuda fuera de alcance de TTH-08)**: el core arrastra el
mismo defecto — `allow_origins=["*"]` hardcodeado en `main.py:23-29` — y
además lo combina con `allow_credentials=True`, combinación que los
navegadores rechazan. Se registra para unificación futura; **no se actúa
sobre el core en TTH-08**.

## 14. Cierre de Fase 1

Al cierre de Sesión 3, Fase 1 está completa: **no quedan decisiones
pendientes**. Las cinco decisiones derivadas y transversales (§6.11, §6.6,
§6.5, §6.3, §6.4) quedan cerradas en §9-§13. Fase 2 (migración Alembic +
levantamiento formal de la regla CLAUDE.md sobre `edge_device/src/vision/`)
puede arrancar.

## 15. Trazabilidad con Fase 0

### 15.1 Decisiones de §6 de Fase 0 cerradas en este documento

| Decisión Fase 0 | Cerrada en | Sesión |
|---|---|---|
| §6.1 — Test de aceptación zone_counter | §2 de este documento | Sesión 1 |
| §6.7 — Value Objects | §3 | Sesión 1 |
| §6.2 — Set de Protocols | §4 (con adelanto parcial de §6.6) | Sesión 1 |
| §6.9 — Definiciones canónicas de métricas | §5 | Sesión 2 |
| §6.10 — API pública del Broadcaster | §4.4 Cambio 3 (parcial) + §6 (detalle) | Sesiones 1 y 2 |
| §6.8 — Lugar del visualizer y de `interaction` | §7 | Sesión 2 |
| §6.11 — Separación transporte/presentación en Broadcaster | §9 | Sesión 3 |
| §6.6 — Modos de pipeline (async-only) | §10 | Sesión 3 |
| §6.5 — Manejo de errores en workers async | §11 | Sesión 3 |
| §6.3 — Logging unificado | §12 | Sesión 3 |
| §6.4 — Configs (híbrido por clase) | §13 | Sesión 3 |

### 15.2 Hallazgos de Fase 0 consumidos directamente

- §3.2 (duplicación DRY entre sync y async aggregator) → resuelto por
  §4.4 Cambio 2 (separar cómputo de persistencia).
- §3.5 (dos definiciones de "density") → resuelto por §5.4
  (`mean_occupancy` canónica) y §6.2 (broadcaster emite la versión
  canónica sin `× 100` ni string).
- §3.6 (placeholder `"incidents": 0` hardcoded, umbrales 30/70) →
  resuelto por §5.7 (vision no emite `incidents` ni `congestion_level`).
- §3.8 (encapsulación violada, sentinels `"unknown"`, `frame: object`,
  ids como `str` plano, nombres engañosos) → resueltos por §3 (VOs),
  §4 (Protocols con VOs y tipado explícito), §5 (renombres
  `total_vehicles → unique_vehicles`, `flow_rate_per_min → flow_vehicles_per_hour`,
  `avg_speed → mean_speed_kmh`).
- §4 Casos A, B (Protocols faltantes; `application` importa de
  `infrastructure` y `presentation`) → resueltos por §4.1 con los cinco
  nuevos Protocols y por §7.2 (visualizer inyectado vía `FrameRenderer`).
- §4 Caso C (`interaction.py` como UI en infraestructura) → resuelto por
  §7.3 (movido a `edge_device/scripts/calibrate_zones.py`).
- §4 Caso D (broadcaster mezcla transporte y presentación) → cerrado en §9
  (§6.11): el broadcaster transporta estructuras puras; toda presentación
  (strings localizadas, umbrales, escala Waze) vive aguas abajo del
  transporte.
- §6.1 (test de aceptación zone_counter) → cerrado con contrato preciso
  en §2.
- §6.10 (API pública del Broadcaster) → cerrado en §4.4 Cambio 3 (métodos
  públicos) + §6 (detalle del payload, eventos y política de suscriptores).

Sin candidatos a DHU-025 emergentes de Sesiones 1, 2 ni 3. Las ocho
decisiones de DHU-024 siguen sin contradicción. Las observaciones de la
auditoría de Sesión 3 (el `ConfigManager` muerto de `cerebrovial_shared` y
la duplicación de los YAMLs de visión) eran insumos de §6.4 y quedan
resueltas en §13; la deuda de logging central y la de CORS del core son
notas al margen fuera del alcance de TTH-08 (§12.4, §13.5), no
contradicciones.
