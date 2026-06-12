# Resultados — Benchmark de captura de visión

Corrida de referencia: **2026-06-12**, Apple Silicon (Darwin 25.5.0), `mps` disponible.
Stream(s) HLS de Claro reales (segmentos 2 s, 1 keyframe/segmento → ~0.5 fps de keyframes,
ventana viva ~6 s). CPU en **% de un core** (user+sys / wall, árbol de proceso). Reproducción
y entorno: ver [README.md](README.md). Datos crudos: [data/](data/). Frames: [frames/](frames/).

> **Método de frescura:** se lee el reloj quemado del frame guardado a los ~60 s
> ([frames/](frames/)) menos el reloj del sistema. El reloj de la cámara corre ~+5 s
> adelantado respecto al sistema (medido sobre `native`, que es live edge continuo), así
> que la precisión es de ±varios segundos: la distinción que importa es **live (segundos)
> vs. rancio (decenas de s)**.

---

## Eje 1 — Mecanismo de captura para inferencia

### 1a. Single stream — CPU, fps efectivo, frescura, RSS  ([data/eje1_single.csv](data/eje1_single.csv), [data/eje1_freshness.csv](data/eje1_freshness.csv))

| mecanismo | fps efectivo | CPU/cámara | RSS (min–max MB) | frescura-a-vivo |
|---|--:|--:|--:|--:|
| native (decode completo) | 26.7 | **10.5 %** | 150–155 | **~0 s (live)** |
| sleep1 (cv2 +throttle 1 fps) | 1.02 | 0.9 % | 111–147 | **~66 s detrás, creciendo** |
| **kf_ffmpeg** (`-skip_frame nokey`) | 0.51 | **1.3 %** | 92–138 | **~live (±s)** |
| **kf_pyav** (`skip_frame='NONKEY'`) | 0.51 | **1.2 %** | 68–101 | **~live (±s)** |

### 1b. Concurrencia 8 streams, inferencia 1 fps  ([data/eje1_concurrency.csv](data/eje1_concurrency.csv))

| mecanismo | CPU total | peor edad_max | cams > 2.5 s | muertes | RSS pico |
|---|--:|--:|--:|--:|--:|
| native | **83 %** | 3.09 s | **4/8** | 0 | 983 MB |
| sleep1 | 23 % | **1.01 s** | **0/8** | 0 | 1087 MB |
| kf_ffmpeg | **25 %** | 3.15 s | 6/8 | 0 | 1023 MB |
| kf_pyav | 25 % | 3.15 s | 5/8 | 0 | **559 MB** |

**Conclusión Eje 1.** El baseline `native` reproduce el freeze: a 8 streams el decode completo
satura **83 % de un core** y empuja 4/8 cámaras sobre el umbral de frescura de 2.5 s (una
hasta 3.09 s) — exactamente el mecanismo del congelamiento en producción (11 decoders × 25 fps).
Los tres mecanismos throttled bajan el CPU a 23–25 %. Pero hay un **trade-off real entre
frescura y cadencia**: `sleep1` mantiene la edad < 2.5 s (0/8) **pero entrega video ~66 s
atrasado y creciendo** (rancio, con buffer del demuxer sin tope) → inservible para métricas.
`kf_ffmpeg`/`kf_pyav` **sí se quedan en el live edge** (±segundos, confirmado por reloj quemado)
con el mismo CPU bajo, **pero su cadencia de 0.5 fps (1 keyframe/2 s) deja la sierra de edad en
~2–3 s**, que **cruza el umbral actual de 2.5 s en 5–6/8 cámaras**. Es decir: keyframe-only es
el único que da *bajo CPU + frescura-a-vivo*, pero **exige relajar `_DEFAULT_FRESH_THRESHOLD_S`
a ~4–5 s** para acompañar su cadencia (cambio acoplado, no swap puro). PyAV se comporta igual
que ffmpeg CLI en frescura, fps y CPU, con **menor RSS** y entrega `ndarray (720,1280,3) uint8
C-contiguous` sin conversión; su contra es operacional: al cargar `av` y `cv2` en el mismo
proceso aparece un warning de clase ObjC duplicada (ambos traen libav) — benigno aquí, a vigilar
si conviven en producción.

---

## Eje 2 — Cadencia de inferencia (sobre kf_ffmpeg, 8 streams)  ([data/eje1_concurrency.csv](data/eje1_concurrency.csv), filas kf_ffmpeg)

| inferencia | CPU total (8 streams) | CPU/cámara | proyección 11 cám | peor edad_max |
|---|--:|--:|--:|--:|
| 0.5 fps | 18 % | ~2.3 % | ~25 % | 3.09 s |
| 1 fps | 25 % | ~3.1 % | ~34 % | 3.15 s |
| 2 fps | 35 % | ~4.4 % | ~48 % | 3.95 s |

**Conclusión Eje 2.** Cada salto de cadencia cuesta **~+8–10 % de un core** sobre los 8 streams
(~+1 % por cámara y paso). Con inferencia a imgsz 320 (~8 ms/frame, medido en el benchmark de
detección), el presupuesto a 1 Hz/cámara entra con margen enorme: **11 cámaras × 1 inferencia ×
8 ms = 88 ms por ronda ≪ 1000 ms** (8.8 % del presupuesto); a 2 fps son ~176 ms (17.6 %). El CPU
total proyectado a 11 cámaras (~34 % a 1 fps, ~48 % a 2 fps de un solo core) confirma que la
cadencia **no es el cuello** — el dominante era el decode, ya resuelto por keyframe-only. La
edad_max no mejora subiendo la cadencia de inferencia (la fija la cadencia de *captura* de 0.5 fps,
no la de inferencia). La elección de 0.5/1/2 fps es por **fluidez visual de las cajas**, que se
evalúa aparte en validación visual; en costo, cualquiera entra.

---

## Eje 3 — Vista del operador: ¿directa al browser o por el edge?  ([data/eje3_cors.csv](data/eje3_cors.csv))

| target | HTTP | `Access-Control-Allow-Origin` | reproducible cross-origin |
|---|--:|---|:--:|
| master `.m3u8` | 200 | `*` | **sí** |
| segmento `.ts` | 200 | `*` | **sí** |

**Conclusión Eje 3.** El HLS de Claro responde **`Access-Control-Allow-Origin: *`** tanto en el
playlist como en los segmentos → **es reproducible directo desde el browser** (un `<video>`/hls.js
en el origen del frontend no es bloqueado por CORS). Esto **desbloquea desacoplar vista de
inferencia**: el video fluido lo reproduce el browser tomándolo directo de Claro, y el edge deja
de tener que decodificar para *mostrar*. [INFERIDO] El ahorro de CPU en el edge es sustancial:
hoy el feed MJPEG `/video/{id}` re-decodifica/re-encodea por consumidor; sacándole esa
responsabilidad, el edge solo necesita el decode **keyframe-only para inferir** (~1.3 %/cámara,
Eje 1). Caveat honesto: esto vale mientras Claro mantenga el CORS wildcard y el stream sea de
acceso público sin token (verificado en el diagnóstico previo: la URL no lleva token/expiry); si
Claro endurece CORS o mete tokens con TTL, la vista volvería a requerir proxy del edge y habría
que dimensionar ese costo aparte.

---

## Eje 4 — Costura `FrameProducer`: ¿source-agnóstica?

### Contrato actual  [OBSERVADO]
Existe una **abstracción explícita**, no acoplamiento directo a OpenCV. Hay **dos costuras**:

- **`FrameProducer`** (pull) — [domain/protocols.py:44](../../../edge_device/src/vision/domain/protocols.py#L44): `read() -> Optional[Frame]`, `release() -> None`. Lo implementa `OpenCVSource`; es "cualquier productor de `Frame`", no "una URL que OpenCV abre".
- **`LiveFrameSource`** (push) — [domain/protocols.py:50](../../../edge_device/src/vision/domain/protocols.py#L50): `start()/snapshot()/stop()`. **Es contra esto que programa el scheduler** ([camera_scheduler.py:69](../../../edge_device/src/vision/application/services/camera_scheduler.py#L69): `ThreadedCapture(pipeline.source)`), no contra `read()`.
- **`Frame`** — [domain/entities.py:63](../../../edge_device/src/vision/domain/entities.py#L63): `id:int, timestamp:float, image:np.ndarray`. Genérico, sin supuesto de fuente.
- **`FrameSnapshot`** — [domain/entities.py:71](../../../edge_device/src/vision/domain/entities.py#L71): `frame, age_seconds (monotónico), live`. Contrato de frescura/liveness **explícitamente agnóstico** ("un archivo y un stream responden según su naturaleza sin que el scheduler los distinga").
- **Dispatch por tipo:** `SourceRegistry` + `SourceFactory.can_handle/create` ([infrastructure/sources/__init__.py:64](../../../edge_device/src/vision/infrastructure/sources/__init__.py#L64)); el builder llama `create_source(source, source_type, ...)` ([pipeline_builder.py:110](../../../edge_device/src/vision/application/builders/pipeline_builder.py#L110)) — **no hardcodea `OpenCVSource`**.

### Qué tan agnóstico es hoy  [OBSERVADO]
**Bastante.** El scheduler trata la fuente como caja negra (`snapshot()`), `ThreadedCapture`
trata al productor como caja negra (`read()`), y **la lógica HLS-específica (reconnect,
Streamlink, Referer, ventana de segmentos) está encapsulada dentro de `OpenCVSource`** —
no se filtra hacia arriba. Agregar una fuente es: implementar `FrameProducer` + un
`SourceFactory` + `register("tipo", ...)`; `ThreadedCapture` la envuelve y el scheduler no se toca.

### Supuestos que SÍ se filtran  [OBSERVADO/INFERIDO]
1. **Umbral de frescura (2.5 s)** — [camera_scheduler.py:45](../../../edge_device/src/vision/application/services/camera_scheduler.py#L45). Asume implícitamente una fuente que entrega frames **sub-2.5 s**; una fuente lenta (keyframe-only 0.5 fps, o un `APISource` de baja tasa) la viola (Eje 1: 5–6/8 cruzan). Hoy es constante del scheduler, no parámetro de la fuente. **Es el principal acople a romper.**
2. **Resolución de zonas** — los polígonos de zonas están en **coordenadas de píxel** (config) → dependen de la resolución de la fuente. No lo asume el scheduler ni el detector (toman cualquier `ndarray`), pero sí el cómputo de zonas. [INFERIDO]
3. Nada más relevante: `target_width/height` es config de la fuente, no supuesto upstream; el `Frame.image` es `ndarray` genérico.

### Interfaz mínima para fuentes muni sin refactor aguas arriba  [INFERIDO — propuesta, no implementada]
La que **ya existe** alcanza; solo hay que ajustar 2 cosas:
- **Contrato de la fuente:** implementar `FrameProducer` (`read()->Frame|None`, `release()`) — y, si la fuente es nativamente push/async (ONVIF/SDK con callback), implementar directamente `LiveFrameSource` (`start/snapshot/stop`) en vez de envolver con `ThreadedCapture`.
- **Forma del `Frame`:** `id` incremental, `timestamp` wall-clock de captura, `image` = `ndarray (H,W,3) uint8 BGR`. (Ya es lo que entregan cv2, el pipe ffmpeg y PyAV — verificado.)
- **Ciclo de vida / salud:** vía `FrameSnapshot(frame, age_seconds, live)` — `live=False` señaliza fuente muerta; el scheduler ya lo interpreta como NULL-con-motivo. Una fuente nueva lo obtiene gratis al pasar por `ThreadedCapture`.
- **Ajuste 1 (necesario):** hacer **`fresh_threshold_s` un parámetro por-fuente/config** (no constante) para que una fuente de baja cadencia declare su propia tolerancia. Sin esto, keyframe-only y cualquier fuente lenta tripean el umbral.
- **Ajuste 2 (registro):** registrar la nueva factory en `SourceRegistry` (`register("munirtsp"/"onvif"/"api", ...)`); el builder ya despacha por `source_type` sin cambios.

### ¿keyframe-only encaja como una implementación más?  [INFERIDO]
**Sí, naturalmente — como `HlsKeyframeSource(FrameProducer)`, no como parche dentro de
`OpenCVSource`.** Su `read()` bloquea hasta el próximo keyframe (~2 s) y devuelve un `Frame`
con `ndarray` BGR; `ThreadedCapture` lo absorbe igual que cualquier `read()` bloqueante, y se
registra con `register("hls_keyframe", HlsKeyframeFactory())`. El único acople a tocar es el
**Ajuste 1** (umbral de frescura), que es independiente del mecanismo de captura y beneficia a
toda fuente lenta futura. Conclusión: keyframe-only entra por la costura existente como una
fuente más; **no re-acopla a HLS/ffmpeg** si se implementa como su propia clase `FrameProducer`
detrás de la factory.

---

## Síntesis para la decisión
- **Mecanismo:** keyframe-only (ffmpeg CLI o PyAV) es el único que da **bajo CPU (1.3 %/cám,
  25 % a 8 streams) + frescura-a-vivo + memoria acotada**. `sleep1` se descarta (rancio, −66 s).
  `native` se descarta (satura, freeze).
- **Condición acoplada:** subir `_DEFAULT_FRESH_THRESHOLD_S` a ~4–5 s (la cadencia de 0.5 fps
  cruza el 2.5 s actual en 5–6/8). Hacerlo **parámetro por-fuente** deja la costura agnóstica.
- **ffmpeg CLI vs PyAV:** equivalentes en frescura/fps/CPU; PyAV menor RSS y `ndarray` directo,
  pero suma dependencia pesada y el conflicto libav con cv2. Decisión de empaque, no de
  performance → **resuelta a favor de ffmpeg CLI** (ver "Decisión: ffmpeg CLI sobre PyAV" abajo).
- **Vista del operador:** CORS `*` permite servirla **directa al browser**, sacándole al edge
  el decode de visualización (el edge solo decodifica keyframe-only para inferir).
- **Watchdog:** sigue necesario para stalls de origen (independiente del fps; ver diagnóstico).

---

## Decisión: ffmpeg CLI sobre PyAV

El benchmark dejó el `ffmpeg-CLI`-vs-`PyAV` como empate de performance (frescura, fps y CPU
equivalentes — Eje 1). La decisión se cierra **sobre la evidencia ya medida**; es [INFERIDO]
a partir de hechos [OBSERVADOS].

**Decisión.** El productor keyframe-only se implementa sobre **ffmpeg CLI (subproceso)**, no PyAV.

**Criterio decisivo — aislamiento de fallas.** La captura es la capa más expuesta a streams
impredecibles (Claro hoy; fuentes muni a futuro). ffmpeg como **subproceso aísla el fallo**: un
stream corrupto o un decode trabado mata solo esa cámara — el pipe se cierra, el `read()` lo
detecta (en el benchmark, los streams que se cortaron aparecieron como `len(buf) < FRAME_BYTES`,
sin tumbar el proceso) y se reinicia el subproceso, sin afectar al resto del edge. PyAV corre
**libav en proceso**: un estado malo o segfault de libav ante un stream corrupto se llevaría el
**edge entero**. Para un sistema desatendido cuyo defecto original fue justamente un modo de
falla (el freeze), el aislamiento pesa más que el ahorro de RSS.

**Riesgo evitado adicional.** PyAV trae su propia copia de libav coexistiendo con la de OpenCV
en el mismo proceso — warning ObjC de clase duplicada **[OBSERVADO en el Eje 1]**
(`AVFFramereceiver ... implemented in both .../cv2/.dylibs/libavdevice... and
.../av/.dylibs/libavdevice...`). Dos copias de la misma librería nativa en proceso es riesgo
latente de mantenimiento (símbolos/versores divergentes). ffmpeg CLI lo evita: el binario y
OpenCV usan libav por separado, en procesos distintos, y nunca se tocan.

**Trade explícito (lo que se sacrifica).** PyAV midió **~45 % menos RSS** (559 vs 1023 MB de pico
a 8 streams — [data/eje1_concurrency.csv](data/eje1_concurrency.csv)) y entrega `ndarray` sin
subproceso. Se acepta el mayor RSS de ffmpeg CLI porque **hoy la memoria no es el cuello**: el
presupuesto de inferencia entra holgado (88/1000 ms por ronda, Eje 2) y la vista directa al
browser (CORS `*`, Eje 3) le quita al edge el decode de visualización. **Condición de revisión:**
si el auto-scaling futuro muestra que la memoria por contenedor es el límite de densidad de
cámaras, re-evaluar PyAV con ese dato.

**Costo asumido.** ffmpeg CLI requiere (a) el binario en la imagen del edge —`apt-get install
ffmpeg` en el Dockerfile (hoy `python:3.11-slim` + `opencv-python-headless` no lo incluye)— y
(b) escribir la supervisión del subproceso: detectar proceso muerto/colgado → reiniciar, sin
dejar zombies. Trabajo acotado y conocido, dentro del reemplazo del `FrameProducer` (Eje 4).
