# Cierre — Métricas de visión, flujo por cruce y caracterización de la medición

**Fecha:** 2026-06-07
**Autor:** Cesar (con Claude.ai como estratega/auditor; mediciones ejecutadas por Claude Code)
**Estado:** Documento de cierre de una línea de trabajo de diseño y medición. No es entrega de código; es la base empírica y de decisiones sobre la que se diseñarán el saneamiento del flujo, los canales de métricas y la validación de conteo.

---

## Cómo leer este documento

Tiene dos partes con audiencias distintas:

- **Parte A — Hallazgos con valor de tesis.** Material defendible ante el jurado, escrito para poder elevarse al capítulo de resultados/discusión. Cada afirmación distingue lo medido de lo aspiracional o preliminar.
- **Parte B — Bitácora de ingeniería.** Decisiones técnicas y mediciones de recursos que fijan el diseño del sistema. Audiencia: el propio equipo (Cesar, Andrés) y futuras sesiones.

Disciplina transversal aplicada en todo el trabajo aquí registrado: **métricas honestas**. Donde una medición salió ruidosa o no concluyente, se reporta como tal; no se presenta ningún número sin su alcance y sus caveats.

---

# PARTE A — Hallazgos con valor de tesis

## A.1 La calibración por cámara es una condición de validez de la medición, no un paso accesorio

El hallazgo central de esta línea de trabajo: el sistema no produce "el conteo de tráfico" en abstracto. Produce una **medición cuya confiabilidad depende de variables físicas específicas de cada cámara**. Caracterizar esa dependencia — establecer bajo qué condiciones el conteo por visión es confiable sobre infraestructura de tránsito real — es un resultado, no una tarea previa.

Variables que influyen en la validez de la medición, identificadas en este trabajo:

- **Ángulo de la cámara.** Las cámaras de tránsito tienen vistas cenitales o muy oblicuas, distintas a las imágenes frontales/laterales sobre las que se entrenan los detectores estándar (COCO). El ángulo afecta la separación entre vehículos, la oclusión mutua y la estabilidad del seguimiento.
- **Iluminación y hora del día.** La luz solar, las sombras duras, el contraluz y la luz urbana nocturna cambian la tasa de detección. Una cámara confiable de día puede no serlo de madrugada (ver A.4).
- **Condiciones climáticas.** Lluvia, neblina y reflexiones sobre asfalto mojado degradan la detección y el seguimiento.
- **Reflexiones.** Superficies reflectantes (vidrios, carrocerías, agua) generan detecciones espurias o fragmentan trayectorias.
- **Posición de la línea de conteo.** Dónde se coloca la línea virtual de cruce sobre el plano de la imagen determina si los vehículos se observan suficientes veces a ambos lados para contarse correctamente.

**Consecuencia metodológica:** cada cámara requiere un trabajo de calibración propio (posición de zonas/línea, parámetros de seguimiento, eventualmente umbral de detección) antes de que sus números puedan usarse como medida. El tuneo no es global; es por cámara. Esto eleva una deuda que el proyecto ya tenía registrada como "deuda de demo" (zonas default no calibradas para videos reales) al estatus de **condición de validez científica**.

**Enganches en la documentación existente:**
- Deuda de calibración de zonas registrada en el handoff de Fase 6 §4.2 (zonas default genéricas; existe `scripts/calibrate_zones.py`).
- F39 (Trabajos Futuros — Raspberry Pi en intersección real) ya menciona "calibración de cámara en condiciones reales" como complejidad; este hallazgo le da el fundamento de por qué es no-trivial.

## A.2 La precisión de detección (YOLO) no es la confiabilidad del conteo de flujo

Distinción que conviene defender explícitamente, porque suele confundirse:

- **Precisión de detección** (mAP, precision, recall): mide si el detector encuentra vehículos en imágenes sueltas. Es una propiedad del modelo YOLO, medida sobre datasets de imágenes.
- **Confiabilidad del conteo de flujo**: mide si el *sistema completo* — detección + seguimiento + asociación de ID entre frames + cruce de línea — cuenta correctamente cuántos vehículos pasan. Es una propiedad de la cadena entera, no del detector.

Un detector con mAP alto puede producir un conteo de flujo pobre si el seguimiento pierde o fragmenta IDs, o si el ángulo de la cámara difiere de las condiciones de entrenamiento. **La confiabilidad del conteo se valida sobre el sistema integrado y sobre video real, no heredando la métrica del detector.**

Esto reorienta y amplía CT-08.9 (que pedía dataset etiquetado ≥200 frames con precisión/recall/mAP del detector, actualmente diferido): la métrica relevante para el caso de uso no es solo el mAP de detección, sino el **error de conteo de flujo contra verdad de campo** (ver A.3). El número aspiracional histórico de 88.2% (D-005) no tiene sustento reproducible y debe sustituirse por medición real reportada honestamente.

## A.3 Presencia ≠ flujo: el conteo de cruce es la métrica correcta para flujo

Hay tres métricas físicamente distintas que el sistema produce o debe producir, y conflación entre ellas es deuda:

1. **Conteo instantáneo** — cuántos vehículos hay en el cuadro ahora. Densidad/ocupación puntual. Sale de un frame, sin seguimiento.
2. **Presencia agregada / vehículos únicos por ventana** — cuántos vehículos distintos vio la cámara en una ventana (p. ej. 60s). Requiere seguimiento para no recontar el mismo vehículo. **No es flujo.**
3. **Flujo** — cuántos vehículos cruzaron una sección de la vía por unidad de tiempo. Métrica canónica de ingeniería de tráfico (HCM, Webster).

**El error vigente en el sistema:** `flow_vehicles_per_hour` se calcula hoy como `unique_vehicles / window_duration_s * 3600` (`edge_device/src/vision/application/.../_compute.py:90`). Eso reescala presencia, no mide cruce. En vía fluida presencia y flujo correlacionan; **en congestión divergen** — muchos vehículos presentes y detenidos dan presencia alta y flujo real ≈ 0. El sistema reportaría flujo alto justo cuando no pasa nadie, que es el régimen de mayor interés para un sistema de gestión de congestión. El umbral `PEAK_THRESHOLD = 1500 veh/h` del motor adaptativo está definido sobre flujo de cruce, no sobre presencia: comparar presencia contra ese umbral mezcla dos magnitudes con la misma unidad nominal.

**La geometría correcta para flujo es una línea de conteo (cruce), no una zona poligonal (presencia).** La zona mide quién *está dentro*; la línea mide quién *cruzó*. Para flujo se necesita cruce. Ventaja adicional: una línea perpendicular a los carriles no requiere calibración pixel→metro (a diferencia de velocidad o densidad), solo IDs de seguimiento estables.

## A.4 Límites del conteo de cruce — qué está firme y qué sigue abierto

Dos campañas de medición sobre clips reales de cámaras Claro (Benchmarks 3 y 3.5, ver Parte B) acotaron el problema. Conviene separar lo que quedó firme de lo que sigue condicionado.

**Firme (no depende de la hora ni de la escena):**

- **A 1 frame/segundo el conteo de cruce es estructuralmente imposible.** Un vehículo se observa una sola vez antes de salir del encuadre; cruzar requiere verlo a ambos lados de la línea (≥2 observaciones). El conteo de cruce no es viable a 1 Hz — esto es geometría, no medición.
- **La hipótesis de configuración quedó refutada.** Igualar `frame_rate` del seguidor al FPS real de muestreo (30→15) **no mueve la fragmentación** (reasignaciones 11→11, 17→18, 18→19 en tres clips diurnos). El re-tuneo de ByteTrack en el rango probado (`lost_track_buffer` 30/60/90) tampoco la reduce; `30` la empeora, `90` es idéntico a `60`. La configuración de producción ya está cerca del óptimo alcanzable con este seguidor. **No hay tuning de ByteTrack que rescate el conteo de cruce.**

**Evidencia preliminar (pendiente de confirmación):**

- **La fragmentación parece gobernada por la escena, no por el seguidor.** Con configuración idéntica, la vida media del track varía 3.6× entre cámaras (urbano 6.4s vs autopista 1.8s) y los fragmentos van de 11% a 32% de los IDs únicos según la cámara (ángulo, velocidad, oclusión). Esto apunta a que la confiabilidad es scene-driven. **Pero los clips son de amanecer (06:08–06:13), tráfico moderado** — falta confirmar sobre condiciones diurnas plenas y tráfico denso antes de elevarlo a hallazgo firme.

**Defecto experimental que afecta a ambos benchmarks (importante):** la línea de conteo usada fue **sintética — una horizontal en y=altura/2, puesta a ciegas**, sin relación con la forma real de la calle, los carriles ni la dirección de flujo. Por tanto, la fragmentación medida *en el cruce* mezcla dos efectos inseparables: la fragilidad del método de conteo y el hecho de que la línea cae en un lugar arbitrario (posiblemente malo) de la escena. **No se puede concluir "la línea fina no sirve" de forma limpia**, porque la línea nunca estuvo bien puesta. Esto tiene una implicación más profunda (ver A.4.1).

### A.4.1 Corolario: la calibración es precondición de la evaluación, no solo de la confiabilidad

El defecto de la línea sintética revela algo que refuerza A.1 a un nivel más fundamental: **no se puede evaluar ninguna geometría de conteo —línea fina o zona de paso— sin antes calibrar la cámara a su escena real** (forma de la calle, posición de carriles, dirección de flujo, dónde una línea o banda tiene sentido físico). Sin esa calibración, todo benchmark de conteo mide una mezcla indistinguible de "método malo" y "geometría mal ubicada".

La calibración por cámara, entonces, no es solo lo que hace confiable el número final (A.1). Es **anterior**: es condición para siquiera poder medir si un método de conteo funciona. Cualquier comparación futura de geometrías de conteo debe partir de cámaras calibradas, o sus resultados serán inconcluyentes por construcción.

### A.4.2 Deuda aceptada

El autor **acepta como deuda documentada** que las cámaras requieren calibración por sitio para un conteo de flujo confiable, y **acepta el comportamiento de flujo actual tal como está** (con la fórmula vigente descrita en A.3, reconocida como presencia-disfrazada-de-flujo) sin sanearlo en esta iteración. La caracterización completa —comparación de geometrías sobre cámaras calibradas, en condiciones diurnas, con verdad de campo— queda definida como spike (A.5) para una iteración futura. Esto es una decisión consciente de alcance, no un descuido: el saneo del flujo no se ejecuta porque no hay base calibrada sobre la cual decidir la geometría correcta.

## A.5 Spike — caracterización del conteo de flujo por visión (definido, no ejecutado)

Lo abierto por A.4 no se cierra con un benchmark más: es una **pregunta de investigación** — *¿cómo se mide flujo confiable por visión sobre cámaras de tránsito reales, y bajo qué condiciones?* — con varias dimensiones entrelazadas. Se registra como **spike** (investigación acotada, output = conocimiento + este doc, no código merged). Absorbe la validación de confiabilidad de conteo (antes ítem separado): ambas necesitan verdad de campo sobre los mismos clips, son el mismo trabajo.

**Precondición dura:** el spike parte de **cámaras calibradas** (A.4.1). Sin calibración previa de cada cámara a su escena (forma de calle, carriles, dirección de flujo, ubicación física de la línea/banda), cualquier comparación de geometrías es inconcluyente. La calibración es el primer paso del spike, no un supuesto.

**Dimensiones a cubrir (lo más detallado posible cuando se ejecute):**
- **Calibración por cámara primero**: definir, por sitio, la geometría de medición real (no sintética).
- **Geometría de conteo**: zona-de-paso (banda) vs línea fina, sobre cámaras ya calibradas. Modos de error distintos — la línea pierde cruces por fragmentación; la zona puede sobrecontar fragmentos que aparecen dentro de la banda. Cuál domina es empírico.
- **Verdad de campo**: etiquetado manual del autor (patrón oro) — conteo real de cruces por cámara, define qué cuenta (motos, vehículos detenidos sobre la línea, etc.). Comparar conteo del sistema vs humano → error (%) y sesgo (sobre/subconteo) por cámara y condición.
- **Condiciones**: clips diurnos plenos (mediodía), tráfico denso real. Los de amanecer usados hasta ahora no alcanzan.
- **Ángulo de cámara como variable de primer orden**: la diferencia urbano-vs-autopista (vida de track 3.6×) sugiere que el ángulo manda. Incluir cámaras de ángulos deliberadamente distintos.
- **Pregunta de fondo a resolver en el spike**: ¿vale sostener flujo-por-visión, o se documenta como límite caracterizado y el sistema se apoya en conteo instantáneo/ocupación (robustos, no dependen de tracking entre frames) + `jam_level` de Waze para la dimensión de flujo/congestión? D-009 ya deja la arquitectura preparada para tomar `jam_level` de Waze.

---

# PARTE B — Bitácora de ingeniería

## B.1 Restricción de cómputo

Docker en Mac **no accede a GPU/MPS**: YOLO (yolo11n) corre en CPU dentro del contenedor `edge_device`. Todo el dimensionamiento parte de esta restricción. Contexto medido del contenedor: 14 CPU lógicos, torch en CPU.

**Hallazgo de método crítico:** importar y usar `cv2` colapsa `torch.get_num_threads()` a 1 por colisión de OpenMP. Como el server real usa `cv2` en cada frame junto a YOLO, **producción corre la inferencia a 1 thread**. Toda medición que ignore esto sobreestima el rendimiento. Las mediciones de abajo usan 1 thread como representativo de prod.

## B.2 Benchmark 1 — costo de inferencia (`t_inf`)

Frame denso (peor caso de NMS, ~27 detecciones), 1 thread, ms por inferencia:

| imgsz | media | p95 |
|---|---|---|
| 640 | 48.3 | 53.4 |
| 416 | 35.8 | 37.8 |
| 320 | 26.2 | 27.8 |
| 256 | 19.2 | 20.9 |

El contenido del frame casi no movió `t_inf` a imgsz fijo (el costo lo domina la inferencia, no el NMS). Decode JPEG 1280×720 ≈ 1.8ms (despreciable). **No mide el grab de red del HLS** (demux + H.264), que es mayor y va tratado por diseño (B.5).

**Conclusión:** un muestreador de 1 inferencia/segundo sobre 11 cámaras entra holgado a cualquier resolución (peor caso 640: 11 × 53ms ≈ 0.6s < 1s). No hace falta bajar resolución para sostener 1 Hz.

## B.3 Benchmark 2 — MJPEG full-FPS concurrente con el muestreador

Dos cargas concurrentes 30s, frame denso: (A) muestreador 11 inferencias @320 a 1 Hz; (B) MJPEG @640 en loop, lo más rápido posible.

| modo | B: FPS@640 | A ciclos >1s | A p95 |
|---|---|---|---|
| 1 thread (prod) | 15.3 fps | 0 / 30 | 784 ms |
| 14 threads (forzado) | 7.5 fps | 21 / 30 | 1547 ms |

**Conclusiones:**
- A 1 thread, el sistema objetivo es viable: MJPEG sostiene ~15 fps@640 y el muestreador no falla un solo ciclo de 1s.
- **Forzar 14 threads es contraproducente bajo concurrencia.** Dos inferencias pidiendo 14 threads cada una sobre 14 cores → oversubscription de OpenMP: el MJPEG cae a la mitad y el muestreador revienta el segundo. El "14 threads mejora la media" del Benchmark 1 era cierto solo en aislamiento; con dos cargas se invierte.

## B.4 Decisión de diseño de recursos (cerrada)

- **Muestreador (conteo instantáneo):** 11 cámaras × 1 inferencia/segundo @ imgsz 320. Canal vivo, no persiste (un valor que envejece en 1s no se guarda).
- **MJPEG anotado (cámara abierta en el detalle):** imgsz 640 fijo, ~15 fps. El 640 fijo es decisión del autor (nitidez sobre fluidez); 15 fps es lo que la CPU sostiene con el muestreador encima.
- **Threads: 1, deliberadamente.** No es un default accidental. Subir threads rompe el sistema bajo concurrencia (B.3). **Esta decisión debe documentarse en `CLAUDE.md` con su porqué**, para que nadie la "optimice" subiéndola sin entender la colisión cv2/OpenMP.
- **Capacidad ociosa aceptada:** se usan ~2 de 14 cores. Repartir threads asimétricamente entre cargas es espacio de diseño futuro, no pendiente — solo se justifica si se sube el número de cámaras o el FPS del MJPEG, y debe medirse, no asumirse.

## B.5 Captura desacoplada (invariante de diseño)

El grab de red del HLS (demux + H.264) **no está medido** y es el único costo no caracterizado. Se neutraliza por diseño: cada cámara mantiene un hilo de captura que conserva "el último frame disponible"; el muestreador toma ese último frame e infiere, sin esperar a la red. El camino crítico del ciclo de 1s nunca depende de la latencia de red. **Pendiente de verificación:** un smoke con los 11 streams reales abiertos en background (ancho de banda, hilos de captura, memoria de buffers) cuando se construya el canal vivo.

## B.6 Arquitectura de dos (tres) canales de métricas

| Métrica | Canal | Frecuencia observación | Transporte | Persiste |
|---|---|---|---|---|
| Conteo instantáneo | vivo | 1 Hz, 11 cámaras | SSE (push) | No |
| Vehículos únicos / ocupación | agregado | por ventana 60s | — | `vision_aggregates` |
| Flujo por cruce | agregado | FPS alto, **solo cámara abierta** | — | `vision_aggregates` |

- **SSE para el canal vivo**, no polling: para un valor por segundo por 11 cámaras que el usuario mira en tiempo real, una conexión push es más correcta y más liviana que polling a 1 Hz. Reusa el patrón SSE existente (HU-22 / `openCongestionStream`).
- **El flujo por cruce queda solo en la cámara abierta** (no en las 11), porque requiere FPS de observación alto que el muestreador a 1 Hz no provee y que 11 cámaras a FPS alto no entran en la CPU (ver A.4 y B.2). Las 11 tienen conteo instantáneo y presencia; el flujo real es de la cámara que se está observando.

## B.7 Benchmarks 3 y 3.5 — estabilidad de ID al cruzar

**Benchmark 3** (clips madrugada ~05:35): señal robusta = vida del track (monótona con FPS): 15fps→~30 steps, 5fps→9, 3fps→5, 1fps→1 step. Tasa de cruce estable = no concluyente (denominadores 1-9, ruido). Línea sintética y=H/2.

**Benchmark 3.5** (clips diurnos ~06:08-06:13, tráfico real): re-tuneo acotado de ByteTrack a 15fps. Resultado clave — **igualar `frame_rate` 30→15 no movió la fragmentación** (11→11, 17→18, 18→19); `lost_track_buffer` 30 empeora, 90 idéntico a 60. Reasignaciones por cámara: 11 (escuela_pnp, urbano denso), 17 (lamarina, urbano medio), 18 (panamericana, autopista). Vida de track 6.4s vs 1.8s entre cámaras con misma config. Fragmentos = 11%/32%/21% de IDs únicos.

**Interpretación:** ver A.4. La hipótesis de config quedó refutada (firme); la fragmentación scene-driven es evidencia preliminar (clips de amanecer, pendiente confirmación diurna plena); ambos benchmarks usaron línea sintética mal puesta (defecto que impide concluir limpio sobre la geometría).

**Clips conservados** en `edge_device/benchmarks/clips/` (madrugada + diurnos), `.mp4` gitignored, README versionado. Reutilizables para el spike (A.5).

## B.8 Saneamiento del flujo — DIFERIDO (deuda aceptada)

El saneo del flujo (reemplazar `unique_vehicles/window*3600` por contador de cruce) **no se ejecuta en esta iteración**. Razón: no hay base calibrada sobre la cual decidir la geometría correcta de conteo (A.4.1, A.4.2). Sanear con línea fija en YAML sobre cámaras no calibradas reproduciría el defecto de la línea sintética. El saneo queda **bloqueado por el spike (A.5)**, que debe primero establecer, sobre cámaras calibradas y con verdad de campo, qué geometría usar (línea vs zona) y si el flujo-por-visión se sostiene o se sustituye por Waze.

Cuando el spike resuelva la geometría, el saneo será una rama `san-NN` con su DHU (reabre §5.4). Patrón ya existente reutilizable: track de posiciones por ID en `SimpleSpeedEstimator`. Shape direccional por `vision_aggregates` (DHU-024 §5).

## B.9 Cola de trabajo (reordenada tras Benchmark 3.5)

1. **Canales de métricas** — vivo SSE 1Hz/320 (conteo instantáneo, robusto) + agregado 60s de presencia/ocupación a `vision_aggregates`. **No incluye flujo por cruce** (diferido). Esto es lo que sí se puede construir hoy con confianza.
2. **Auditoría de dependencias** — 4 módulos.
3. **Spike de flujo por visión (A.5)** — investigación futura: calibración por cámara → geometría línea-vs-zona → verdad de campo → decisión flujo-visión vs Waze. Desbloquea el saneo del flujo.
4. **Saneo del flujo (B.8)** — diferido, bloqueado por el spike.

**Features posteriores:** calibración-de-línea-desde-front (parte natural del spike cuando se ejecute); reparto asimétrico de threads (solo si sube cámaras/FPS).

**Deuda aceptada y registrada:** el flujo actual se mantiene como está (presencia-disfrazada-de-flujo, A.3) hasta que el spike provea base calibrada. Decisión consciente de alcance del autor.

---

## Trazabilidad y enganches con documentación existente

- **CT-08.9** (diferido): esta línea de trabajo lo amplía/reorienta hacia confiabilidad de conteo de flujo. Ver `documentation/contracts/vision_contract.md` §5.2.
- **Calibración de zonas**: deuda en handoff Fase 6 §4.2; `scripts/calibrate_zones.py`.
- **88.2% aspiracional** (D-005): sin sustento reproducible; se sustituye por medición real.
- **F39** (Raspberry Pi, Trabajos Futuros): "calibración de cámara en condiciones reales".
- **`flow_vehicles_per_hour`**: fórmula vigente en `_compute.py:90`; consumidores en entidad, persistencia, SSE, `state.py`.
- **DHU-024 §5**: shape direccional de `vision_aggregates`.
