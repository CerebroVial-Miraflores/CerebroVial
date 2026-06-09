# Refundación de visión — B1 Paso 2-A: migración al scheduler + detector compartido

**Rama**: `feature/refundacion-vision`.
**Fecha**: 2026-06-09.
**Estado**: **track EN VUELO, no es un cierre.** Paso 0 (medición de memoria) hecho y
2-A1 (gate de ruteo al scheduler) commiteado; sub-pasos 1–5 pendientes. Este handoff
fija la **secuencia** y su **rationale** ahora que el orden quedó decidido — se venía
difiriendo a propósito para no versionar un orden que la medición del paso 0 podía dar
vuelta (y que finalmente no dio).
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
| **1** | Andamio de memoria: subir el límite del contenedor edge a **4 GiB** en compose (temporal) | ⬜ Pendiente |
| **2** | Detector + executor compartidos: hoist de singletons al `MultiCameraManager` + lifecycle | ⬜ Pendiente |
| **3** | Migrar las 11 al scheduler (nacen compartiendo detector) | ⬜ Pendiente |
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

## Referencias cruzadas
- **D-018** (`documentation/lean-inception/4-decisiones/DECISIONS.md`) — decisión canónica que
  este handoff narra en ejecución (scheduler único, modelo compartido, instancia dueña).
- **CIERRE-metricas-vision-flujo.md** (`documentation/docs/`) — Benchmark 1 (costo de
  inferencia) y spike de carga sostenida; base empírica contra la que se contrastan los
  B/Δ₁/Δ₂ de este paso 0.
- **TODO.md** (`documentation/docs/`) — § E21 y § DEUDA-ZONAS-ONDEMAND (persistencia, ortogonal).
