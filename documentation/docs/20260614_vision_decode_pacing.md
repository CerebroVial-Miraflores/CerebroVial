# Pacing del decode HLS, techo de inferencia y escalado N-cámaras (2026-06-14)

Contexto: cierre del trabajo de fluidez del feed de visión (rama
`feature/camara-fluidez-toggle-inferencia`). Documenta un **hallazgo de arquitectura** y
**dos deudas** que surgieron al medir el techo de inferencia con una cámara dedicada.

## Hallazgo — el HLS live entrega frames en ráfagas, no parejo

El stream HLS de Claro (segmentos de 2s, `EXT-X-TARGETDURATION:2`) **no entrega frames a
cadencia constante**: ffmpeg vuelca cada segmento como una **ráfaga de ~50-67 frames a
sub-milisegundo** y luego **stallea ~2s** esperando el próximo segmento publicado en el live edge.

Medido (cam_larco_benavides, `FullDecodeSource.read()`): gaps p50 0.8ms, p99 1859ms, **13 gaps
>1s en 25s** (uno por segmento). Sin pacear, esa entrega bursty **estrangula el pipeline**: con
la cola keep-latest el worker de inferencia descarta casi toda la ráfaga, infiere ~1-2/s y luego
idlea ~2s con el CPU/GPU ocioso. El detector solo (sin pipeline) hace 56/s en CPU y 122/s en MPS
a imgsz=640 — o sea **el cuello NO era cómputo ni GIL** (ambos descartados por medición), era la
**forma de entrega del stream**.

**Solución aplicada:** `-re` como input option de ffmpeg en `FullDecodeSource` (lee el input a su
cadencia nativa) → la entrega pasa a gaps parejos de ~40ms (p99 47ms, cero gaps >1s, estable en
runtime de 3 min sin drift ni pérdida de segmentos). Con eso + operar a 25fps (nativo de Claro) +
ventana de gather a 5ms, la inferencia sube a **24.8/s** y ByteTrack **confirma tracks** (de 0 a
57 tracks con ≥5 frames de persistencia; persistencia máx ~989 frames ≈ 40s).

**Costo:** `-re` agrega latencia acotada (~≤2s, una duración de segmento) porque libera los frames
a ritmo real-time en vez de en ráfaga instantánea. Tolerado.

**Aplica a cualquier consumo futuro del stream**, no solo a este caso: cualquier componente que
lea el HLS live debe pacear (o bufferear) o sufrirá la misma entrega bursty.

## Deuda — escalado a múltiples cámaras

La ventana de gather del `BatchInferenceWorker` quedó **fija en 5ms** (decisión por simplicidad,
óptima para 1-2 cámaras) y el worker es **serial** (~29 inferencias/s totales en MPS). Esto **no
escala a N cámaras simultáneas**: con 25fps × N cámaras el worker se satura (p.ej. las 11 cámaras
del corredor lo exceden de largo). Además el `analyze_fps=25` por default aplica también a
producción en Docker (CPU), donde el margen es menor que nativo (caveat nativo-CPU vs Docker-CPU)
— con 2+ cámaras a 25fps en Docker el worker serial se ajusta primero. Es override por env
`VISION_ANALYZE_FPS`, así que se puede bajar sin tocar código.

- **Trigger:** cuando se requiera inferir más de ~2-3 cámaras activas a la vez.
- **Qué falta:** **batching adaptativo** — cortar la ventana de gather apenas haya un frame por
  cámara activa (en vez del cap fijo), para batchear N cámaras en una sola pasada GPU sin pagar
  latencia de más cuando hay pocas.

## Trabajo futuro — multiprocessing (decode ≠ inferencia)

Separar el decode (N procesos, CPU, no tocan GPU) de la inferencia (1 proceso, GPU) en procesos
distintos, comunicados por memoria compartida. **IMPORTANTE — no confundir las causas:** esto
**NO es para el cuello de este trabajo** (la entrega bursty, ya resuelta con `-re`). Es para
cuando el cuello sea **realmente el cómputo con muchas cámaras** (escalado N-cámaras). La
hipótesis inicial de que el cuello era contención de GIL entre el thread de decode y el loop de
inferencia fue **medida y refutada** (el thread de decode no starvea ni un loop asyncio puro ni un
loop de inferencia con torch). El material de investigación GIL/multiproceso ya está recopilado;
se registra acá con ese contexto para no volver a confundir las dos causas.
