# Refundación de visión — B1 Paso 2-A: migración al scheduler + detector compartido

**Rama**: `feature/refundacion-vision`.
**Fecha**: 2026-06-09.
**Estado**: **track EN VUELO, no es un cierre.** Pasos 0 (medición), 2-A1 (gate de ruteo),
1 (andamio 4 GiB) y 2 (detector + executor compartidos) hechos y commiteados; sub-pasos 3–5
pendientes. Este handoff fija la **secuencia** y su **rationale**; el §8 documenta la
implementación del Paso 2 (orden b→c→a + inyección condicional).
**Implementa en ejecución**: D-018 (scheduler único, modelo compartido, instancia dueña
de cámaras) — ver `documentation/lean-inception/4-decisiones/DECISIONS.md` § D-018.
**Contrasta con**: `documentation/docs/CIERRE-metricas-vision-flujo.md` (Benchmark 1,
costo de inferencia; spike de carga sostenida de D-018).

---

## 1. Dónde estamos parados (estado por sub-paso)

La secuencia de 2-A son seis sub-pasos. No es un cierre: marcamos el estado real de cada uno.

| Sub-paso | Qué | Estado |
|---|---|---|
| **0** | Medición de memoria (base + marginal por cámara) para resolver el fork de orden | ✅ **Hecho** (2026-06-09) — números en §3 |
| **2-A1** | Gate de ruteo: env `VISION_SCHEDULER_CAMERA_IDS` rutea cámaras designadas al `CameraScheduler`; el resto sigue por el path viejo | ✅ **Commiteado** (`ad42ee04`) |
| **1** | Andamio de memoria: subir el límite del contenedor edge a **4 GiB** en compose (temporal) | ✅ **Commiteado** (`35226f88`) |
| **2** | Detector + executor compartidos: hoist de singletons al `MultiCameraManager` + lifecycle | ✅ **Commiteado** (`c7b81c7e`) — implementación en §8 |
| **3** | Migrar las 11 al scheduler (nacen compartiendo detector) | ✅ **Hecho 10/11** (2026-06-09) — gate aprobado; `cam_benavides_panama` excluida por 404 upstream. Detalle en §9 |
| **4** | Retirar path viejo: invertir dispatch a scheduler-por-default, borrar el gate | ⬜ Pendiente |
| **5** | Re-medir con detector compartido + 11 y bajar el límite de memoria al consumo real | ⬜ Pendiente |

---

## 2. ⚠ Hallazgo contraintuitivo — leer antes de cuestionar "¿por qué compartir detector?"

**La pared de memoria de las 11 cámaras era el límite artificial de 2 GiB del contenedor,
NO los modelos.** La medición del paso 0 (§3) lo prueba: 11 cámaras con **modelos separados**
proyectan **~2.4–2.9 GiB** — entran holgadas bajo cualquier techo sano. El OOM observado con
6 cámaras (91 % de 2 GiB) era contra ese límite artificial, no contra una pared real de modelos.

**Consecuencia directa:** el detector compartido ahorra solo **~300–400 MiB** (la porción de
pesos del modelo; el marginal por cámara está dominado por buffers de frame, no por pesos).
Si alguien — probablemente vos, dentro de tres meses — mira ese ahorro y pregunta *"¿por qué
compartir detector si ahorra tan poco?"*, la respuesta es:

> **El detector compartido NO se justifica por ahorro de memoria. Se justifica por
> CORRECCIÓN DE CONCURRENCIA + HIGIENE de arquitectura para el edge real.** Ver el invariante
> de concurrencia en §5. El ahorro de memoria es un efecto lateral menor, no la razón.

Esto **refuerza** la conclusión de D-018 (riesgos conocidos): *"el cuello de botella de escala
de esta arquitectura es la I/O de red, NO el cómputo"*. El spike de D-018 lo mostró desde el
modelo compartido (inferencia con margen de sobra); el paso 0 lo confirma desde los modelos
separados (la memoria tampoco es la pared). Ni cómputo ni memoria son el límite — lo es la
captura HLS concurrente en vivo (riesgo aún no medido, trigger en D-018).

---

## 3. Medición del paso 0 (números medidos, no asumidos)

Protocolo: restart del edge a cero cámaras (bootea vacío) → medir base → +1 cámara → +1 cámara,
con `docker stats` estabilizado en cada escalón. Restart vía `docker restart` del container
puntual (no `docker compose`: no recrea, no toca volúmenes, no pasa por el guard de `invoke`).

| Escalón | MEM edge | Δ |
|---|---|---|
| **B** — boot vacío, 0 cámaras | **74.4 MiB** | — |
| 1 cámara (`cam_paseo_angamos`) | 681 MiB | **Δ₁ = 607 MiB** |
| 2 cámaras (+ `cam_larco_benavides`) | 856 MiB | **Δ₂ = 175 MiB** |

**Descomposición:**
- **Import único** = Δ₁ − Δ₂ = **432 MiB**. torch/cv2/ultralytics se importan **lazy** (al
  crear el primer detector, no al boot). Por eso B es tan bajo (74 MiB) y Δ₁ tan alto.
- **Marginal real por cámara** = Δ₂ = **175 MiB**, **dominado por buffers de frame**, no por
  pesos: `yolo11n` en RAM ≈ 30–40 MiB (el `.pt` son 5.4 MB en disco); el resto (~135 MiB) es
  decode HLS 720p + ThreadedCapture + `frame_buffer_size`×(1280×720×3 ≈ 2.6 MiB) + tracker +
  copias de render.
- **Piso residente:** tras `DELETE` de las 2 cámaras la memoria bajó a **778 MiB, NO a B=74**:
  torch/cv2 quedan importados (Python no descarga módulos). **B=74 es solo el proceso que nunca
  cargó una cámara**; el piso práctico de un edge que ya corrió algo es ~500–780 MiB permanente.
  El andamio debe contar ese piso pegado.

**Proyecciones:**
- **11 separados** = B + Δ₁ + 10·Δ₂ = 74 + 607 + 1750 ≈ **2.4 GiB**. Versión conservadora
  (tasa de las 6-con-render del audit, ~220 MiB/cam con render de navegador activo): **~2.9 GiB**.
- **11 compartido** ≈ **2.0 GiB** (ahorro ~300–400 MiB = 10 modelos × ~30–40 MiB).

**Caveats de la medición (alcance honesto):**
- Medido bajo la **config demo** del alta on-demand (`imgsz` 1280×720, `detect_every_n_frames=1`),
  NO la config de fondo del scheduler (320 @ 1 Hz sin render, per D-018 §5/§6). El footprint
  real de producción por cámara sería **más bajo** → la conclusión "entran bajo 4 GiB" es aún
  más segura.
- **Asimetría de path:** `cam_paseo_angamos` corrió por el scheduler (está en la env); 
  `cam_larco_benavides` por el path viejo (no está). Ambas construyen pipeline + modelo propio,
  así que el footprint por cámara es comparable; Δ₂ se tomó de la cámara por path viejo.
- Contraste con el spike de D-018: aquél midió **crecimiento de RSS** (+388 MB en 5 min, Δ
  decreciente 388→152→27, sin deriva de tiempo de ciclo) con **modelo compartido** sobre `.mp4`.
  No es el mismo experimento (RSS sostenido vs footprint absoluto; compartido vs separado; .mp4
  vs HLS), pero ambos apuntan a lo mismo: memoria no es la pared. El leak lento sigue siendo
  trigger abierto de D-018 (corrida de horas antes de 24/7).

---

## 4. Secuencia fija: "detector primero" — y por qué, aunque la memoria no lo fuerce

La medición **habilitaba** invertir a "migrar primero" (11 separadas entran bajo 4 GiB). Se
decidió **NO invertir**: el orden queda **detector primero**. Razones:

1. **No crear un estado que después hay que desmontar.** "Migrar primero" pasaría por un estado
   intermedio de 11 cámaras × 11 modelos que luego habría que colapsar a 1. Compartir antes de
   migrar evita construir y desarmar esa duplicación.
2. **Validar la concurrencia del detector compartido con 2 cámaras antes de meter 11.** El riesgo
   real del detector compartido es de concurrencia (§5), no de memoria. Se valida barato con 2
   cámaras sobre el modelo compartido; recién con esa garantía se escala a 11.
3. **Cada sub-paso construye sobre el anterior sin volver atrás.** Andamio → detector compartido
   (probado con 2) → migrar las 11 (que ya nacen compartiendo) → retirar path viejo → re-medir.
   Monótono, sin estados intermedios que revertir.

**Costuras ya existentes (de B1 Paso 1a) — por eso el Paso 2 es mediano, no rearquitectura:**
- Inyección de detector: `VisionApplicationBuilder.build_pipeline(detector=...)` +
  factory standalone `create_detector()` (`pipeline_builder.py`). El chain ya acepta un
  detector externo; `build_detector()` es solo fallback de llamadores viejos.
- Inyección de executor de inferencia: `CameraScheduler(infer_executor=...)` con flag
  `_owns_executor` (no apaga un executor inyectado al parar la cámara).

Lo único sin cablear hoy: el detector **nace por-cámara** en `CameraInstance.__init__`
(`multi_camera.py`, `create_detector(config.vision)`) y cada scheduler crea su propio executor.
El Paso 2 es hoist de esos dos singletons al `MultiCameraManager` + lifecycle (que la baja de
una cámara NO libere el modelo compartido — hoy `remove_camera` llama `detector.release()`
por-cámara; pasa a release-on-shutdown).

---

## 5. Invariante de concurrencia (no depende del orden)

Un modelo YOLO compartido **no es thread-safe bajo llamadas concurrentes** desde threads
distintos. Por lo tanto el detector compartido exige, sí o sí, las tres cosas a la vez:

1. **UN executor `max_workers=1` global** para todas las cámaras → la inferencia se serializa
   globalmente (no solo dentro de una cámara, como hoy).
2. **El path viejo muerto.** `_run_camera_pipeline` infiere en su propio thread vía
   `pipeline.run()` (el `SmartDetectionProcessor` llama al detector sincrónicamente, **fuera**
   del executor del scheduler). Si una cámara por path viejo comparte el modelo con una
   scheduled → llamadas concurrentes NO serializadas → unsafe.
3. **Todas las cámaras en el scheduler** (corolario de 1+2).

**Atadura invariante:** el detector compartido y la muerte del path viejo están atados — no se
puede compartir mientras el path viejo vive. El orden del fork mueve *cuándo* se comparte, NO
esta atadura. (Nota: el estado "11 migradas, cada una con su modelo" SÍ es thread-safe —no hay
concurrencia sobre un modelo compartido; el problema aparece solo al compartir.)

Esto alinea con la advertencia de D-018 §"Advertencia para la implementación de B1": el
`POST /cameras/{id}` cambia de semántica single-slot a "sumá esta cámara al conjunto del
scheduler". El Paso 4 (retirar path viejo) cierra esa redefinición invirtiendo el dispatch a
scheduler-por-default.

---

## 6. Ortogonalidad con la persistencia (no esperar que esto destrabe E21)

Las 11 cámaras migradas correrán con `zones: {}` (el alta on-demand no inyecta zonas), así que
`vision_aggregates` **seguirá en 0 filas** — la migración al scheduler y la persistencia son
ortogonales. **E21 sigue bloqueada por DEUDA-ZONAS-ONDEMAND**, aparte de este track. Migrar al
scheduler no es condición suficiente para que aparezcan filas; hace falta resolver las zonas del
path on-demand (ver `documentation/docs/TODO.md` § DEUDA-ZONAS-ONDEMAND y § E21).

---

## 7. Andamio de memoria (Paso 1) — número, no redondeo arbitrario

Límite actual: `docker-compose.yml` → `edge_device.deploy.resources.limits.memory: 2G` (una
línea; no hay `mem_limit` legacy ni ulimit en el Dockerfile). Andamio propuesto: **4 GiB**.
Justificación: sobre la proyección peor-caso de 11-separados (~2.9 GiB con render), 4 GiB da
38–66 % de headroom; con el host de 24 GiB cualquier valor razonable es seguro. **Es temporal**:
el Paso 5 re-mide con detector compartido + 11 y baja el límite al consumo real. Anotar en el
commit del andamio que el número es provisional para que no quede como techo mágico permanente.

---

## 8. Paso 2 implementado — orden b→c→a + inyección condicional (`c7b81c7e`)

Paso 2 aterrizó en `multi_camera.py` (+ una línea de wiring en `api/__init__.py`). **Alcance
real: NO fueron "tres archivos"** — `camera_scheduler.py` y `pipeline_builder.py` no cambiaron
(sus seams de Paso 1a ya recibían el parámetro inyectado). Tres cambios: **(a)** detector
singleton del manager, inyectado condicionalmente; **(b)** executor `max_workers=1` singleton,
global; **(c)** guard de release en `remove_camera` vía `_owns_detector`.

### Orden b→c→a — por qué (a) aterriza ÚLTIMO
**(a) es el único cambio que introduce el riesgo** (compartir el modelo). Sin protección, dos fallas:
- **F1 (concurrencia):** modelo compartido + executors por-cámara → inferencia concurrente sobre
  el mismo modelo desde threads distintos → corrupción silenciosa. **Lo previene (b)**, no (c).
- **F2 (release-kill):** modelo compartido + release por-cámara → bajar una cámara mata el modelo
  bajo las otras. **Lo previene (c)**.

Por eso el orden es **(b) → (c) → (a)**: (a) entra en un entorno ya doblemente protegido.
"a+c sin b" todavía tendría F1. Y (b), (c) son seguros de aterrizar solos: (b) serializa modelos
aún separados (correcto, y por §B.3 del Benchmark es el óptimo anti-oversubscription); (c) con
`_owns_detector=True` por defecto es andamio dormido hasta que (a) lo ponga en False. Ningún
estado de runtime inseguro, ni durante el desarrollo.

### Inyección condicional — el aislamiento estructural de Q3
El detector compartido se inyecta **solo si `use_scheduler=True`** (`add_camera`); las cámaras del
path viejo pasan `detector=None` y construyen el suyo (`CameraInstance`). Resuelve un riesgo que la
auditoría destapó: una cámara con id **fuera de la env** (typo, cámara nueva, POST de un id no
listado) cae al path viejo e infiere **fuera** del executor global. Si se hubiera inyectado el
compartido a TODAS, esa cámara accidental inferiría sobre el modelo compartido fuera del executor
→ corrupción silenciosa (F1).

Con la condicional, una cámara fuera del scheduler **nunca recibe el modelo compartido** → no puede
tocarlo. **Refina el invariante del §5:** la garantía "nada infiere sobre el compartido fuera del
executor" se sostiene durante la ventana de coexistencia **por la inyección condicional, no por
matar el path viejo**. Por eso el Paso 2 (compartir) y el Paso 3 (migrar las 11) conviven con
seguridad **mientras el path viejo sigue presente pero dormido**. El Paso 4 después lo borra
(limpieza estructural definitiva), pero 2+3 ya son seguros sin él.

### Lifecycle, teardown y carrera
- El guard `_owns_detector` libera el modelo en `remove_camera` **solo si es propio** (path viejo).
  Las scheduled (`_owns_detector=False`) no liberan el compartido al bajar.
- `MultiCameraManager.shutdown()` libera el detector compartido + apaga el executor **una vez**,
  cableado a `@app.on_event("shutdown")` en `api/__init__.py` — contrapeso de haber sacado el
  release por-cámara. En contenedor el stop es SIGKILL y no corre; queda para shutdown graceful y
  deploys no-contenedor. (`on_event` está deprecado pero es consistente con el archivo, que no usa
  lifespan; meter lifespan sería refactor colado.)
- **Singletons lazy race-free:** `if X is None: X = create()` es seguro porque los helpers son
  síncronos y se invocan sin `await` entre el check y el set (el event loop no interleavea).
  Invariante anotada en sus docstrings: si un refactor los vuelve async, hace falta lock.

### Validación en vivo (gate de 5 criterios, 2026-06-09)
Edge **rebuildeado** (`invoke up-build` — el código va COPIADO a la imagen, no montado; correr con
`up` a secas habría validado código viejo), env con 2 cámaras scheduler:
- **C1 (comparten, duro):** 2 scheduler → **1 solo `Loading model`**.
- **C2 (baja-no-rompe, duro):** DELETE de una scheduled → la otra sigue tickeando
  (`sensor_status:ok`, age 1.12s), modelo no reconstruido.
- **C4 (condicional, duro):** cámara fuera de env → **+1 `Loading model` propio** (1→2), path
  viejo aislado del compartido.
- **C5 (release rama owns):** DELETE de la owns=True liberó (~60 MiB de vuelta, RSS parcial en
  torch); el compartido sobrevivió a esa baja.
- **C3 (confirmatorio):** sin blowup, bajo el andamio de 4 GiB. El sharing lo prueba C1, no el
  número de memoria (delta mezclado scheduler@320 + viejo@nativo, en el ruido).

Suite `edge_device/tests/vision`: **170 passed**.

---

## 9. Paso 3 ejecutado — migración a 10/11 + gate de validación (2026-06-09)

Paso 3 **NO cambió código**: fue env (los 11 `camera_id` en `VISION_SCHEDULER_CAMERA_IDS`),
recreate del edge (cambio de env, sin rebuild), alta on-demand de las cámaras y una ventana de
observación de 30 min con gate. El veredicto del gate se cerró en chat (**APROBADO, alcance 10/11
declarado**). Lo que sigue es la evidencia, no checkmarks.

### Alcance — 10/11, declarado sin asteriscos escondidos
Validado a **10 streams concurrentes, NO 11**. `cam_benavides_panama` quedó **excluida** por
**FUENTE_NO_DISPONIBLE upstream**: su `.m3u8` (`panamericana_peaje1`) devolvió **404 persistente
(3/3 + un reintento único al cierre del gate)**, ajeno al código y al gate de env (el id estaba
correcto en la env, verificado in-container). La apertura sincrónica de la fuente hace que esa
fuente muerta **reviente el alta con HTTP 500** en vez de registrar la cámara degradada (ver
`TODO.md` § DEUDA-ALTA-SINCRONICA). **Pendiente con trigger:** cuando el stream reviva, alta en
caliente y verificar que `Loading model` **NO** incrementa — prueba de que una cámara nueva en
caliente toma el detector compartido (la evidencia bonus que el gate no pudo obtener con el stream
muerto). Registrado en `TODO.md` § PENDIENTE-BENAVIDES-11A.

### C1 — Detector único ✅
`grep -c "Loading model"` desde el recreate = **1**, con **10 cámaras scheduled**. Un solo modelo
YOLO cargado y compartido por las 10. El criterio C1 del Paso 2 (probado allí con 2 cámaras) queda
confirmado formalmente a escala 10.

### Memoria — 4 muestras (uso / % de 4 GiB / CPU)
| Marca | MEM | % de 4 GiB | CPU |
|---|---|---|---|
| **t0**  | 1.471 GiB | 36.77 % | 166 % |
| **t10** | 1.498 GiB | 37.45 % | 125 % |
| **t20** | 1.512 GiB | 37.80 % | 179 % |
| **t30** | 1.535 GiB | 38.37 % | 167 % |

Creep **lineal, sin escalones ni sawtooth** (muestreo cada 60s): +64 MiB en 30 min ≈ **~2.1 MiB/min**.
El **trigger de leak lento de D-018 pasó de hipótesis a medición**: presente, **benigno a 30 min**, a
38 % del cap. **Pero 30 min no distinguen creep-que-platea de creep-ilimitado** — la magnitud quedó
**insuficientemente caracterizada para el Paso 5**. Consecuencia directa para el Paso 5: el límite real
**no puede ser "consumo observado + epsilon"**; necesita margen explícito para el creep, o una ventana
más larga que muestre plateau.

### Condición de medición — render-off (insumo crítico para el Paso 5)
La ventana corrió con el **render MJPEG apagado**: el watchdog lo apagó en las 10 cámaras a los ~50s
por **falta de consumidores** (`"sin consumidor MJPEG … apagando render (el muestreo sigue)"`). Es
correcto y esperado (config de fondo de D-018), pero significa que **1.53 GiB es el consumo render-off
— sin nadie mirando**. Con operadores conectados al dashboard (miniplayers HLS del frontend) el render
enciende y el perfil cambia. **El Paso 5 debe decidir**: medir el escenario **render-on**, o **documentar
que el límite asume render-off + headroom**. Esta decisión (render-on/render-off + creep) se cierra
**antes** de medir el Paso 5, con el insumo de este gate.

### Observación HLS (D-018) — cierre parcial de la observación (b) de 2-A1
- **(a) Pendiente de memoria:** creep lineal ~2.1 MiB/min, sin saltos (arriba).
- **(b) Edad de frames:** t0/t10/t20 → **10/10 `sensor_status:ok`**, edades < 2s, 0 errores, 0 drops.
  Al **t30** el snapshot marcó `"Degradado"` con **blip transitorio `sin_frame_fresco`** en
  `cam_28julio_reducto` (2.69s) y `cam_paseo_angamos` (5.03s), **auto-recuperado** (post-ventana las
  10 volvieron a `ok` < 1.4s, `status:OK`), con **`aggregation_errors:0` y `data_dropped:0`**.
  **Hallazgo positivo:** el mecanismo de frescura funciona como **detector transitorio, no como falla**
  — cierra parcialmente la observación (b) que 2-A1 dejó abierta.
- **(c) Logs de captura:** ventana limpia salvo lo upstream-ajeno. `reconnect`: 0 · `timeout` (evento):
  0 · `drop/descart`: 0 · `SourceError`: 1 (benavides, ya contado). El apagado de render por falta de
  consumidor es comportamiento esperado, no anomalía.

### Matiz de endpoints (trampa para gates futuros)
`GET /cameras` (`streaming.py:51`) devuelve `broadcaster.subscribed_cameras()` + `latest_states()` —
**cuenta suscriptores SSE, no cámaras corriendo**; da `[]` aunque las 10 corran si no hay cliente SSE.
La **vista autoritativa de running es `GET /cameras/status`** (registro + `running`) y la telemetría de
frescura está en `GET /vision/health`. **Gates futuros deben usar estas dos, no `GET /cameras`.**

### Estado resultante
10/10 cámaras `running`, `status global: OK`, 0 errores / 0 drops, ~1.53 GiB (38 % del cap de 4 GiB).
El edge quedó **levantado con las 10 cámaras activas** tras el gate (no se bajó). Próximo: **Paso 4**
(invertir dispatch a scheduler-por-default, borrar el gate de env, eliminar `_run_camera_pipeline` y la
inyección condicional) y **Paso 5** (re-medir con la decisión render-on/render-off + creep ya tomada).

---

## Referencias cruzadas
- **D-018** (`documentation/lean-inception/4-decisiones/DECISIONS.md`) — decisión canónica que
  este handoff narra en ejecución (scheduler único, modelo compartido, instancia dueña).
- **CIERRE-metricas-vision-flujo.md** (`documentation/docs/`) — Benchmark 1 (costo de
  inferencia) y spike de carga sostenida; base empírica contra la que se contrastan los
  B/Δ₁/Δ₂ de este paso 0.
- **TODO.md** (`documentation/docs/`) — § E21 y § DEUDA-ZONAS-ONDEMAND (persistencia, ortogonal).
