# Captura HLS keyframe-only sobre ffmpeg CLI — Fase 1

## Contexto

El módulo de visión sufría un freeze en producción: con varias cámaras HLS activas, el
decode completo (`cv2.VideoCapture` ~25 fps por cámara) saturaba CPU y empujaba los frames
sobre el umbral de frescura, congelando el muestreo. El benchmark
`documentation/benchmarks/vision-capture/` cerró el diagnóstico: a 8 streams concurrentes el
mecanismo `native` satura 83 % de un core y cruza el umbral de 2.5 s en 4/8 cámaras. La
captura keyframe-only (`ffmpeg -skip_frame nokey`) es el único mecanismo medido que da bajo
CPU (~1.3 %/cámara, 25 % a 8 streams) + frescura-a-vivo + memoria acotada. Se eligió ffmpeg
CLI sobre PyAV por aislamiento de fallas: el subproceso muerto/colgado se detecta como
lectura corta y se reinicia sin tumbar el resto del edge.

Esta fase implementa el fix como un nuevo `FrameProducer`, más el umbral de frescura
por-fuente, la supervisión del subproceso y el ruteo por defecto.

## Entregables

1. **Benchmark de captura** (`documentation/benchmarks/vision-capture/`): scripts, datos
   crudos y frames de evidencia que fundan la decisión (native vs sleep1 vs kf_ffmpeg vs
   kf_pyav sobre HLS de Claro real). Read-only, fuera de producción.

2. **`fresh_threshold_s` por-fuente**: el scheduler deriva el umbral de frescura de la fuente
   en vez de una constante propia. `OpenCVSource` declara 2.5 s (heredado por Webcam/VideoFile);
   `HlsKeyframeSource` declara 4.5 s (su cadencia de 0.5 fps deja la sierra de edad en ~2–3 s).
   Override per-cámara vía `SourceConfig.fresh_threshold_s`. El param explícito del scheduler
   sigue ganando; si no, `getattr(source, "fresh_threshold_s", 2.5)`. `ThreadedCapture` no cambia.

3. **`HlsKeyframeSource`** (`infrastructure/sources/hls_keyframe_source.py`): nuevo
   `FrameProducer` keyframe-only sobre un subproceso ffmpeg (rawvideo bgr24 por pipe).
   Supervisión dentro de `read()`: lectura corta (`len(buf) < FRAME_BYTES`) → reap
   (`kill` + `wait`, sin zombies) + respawn con backoff exponencial capado; devuelve `None`
   solo en give-up duro (20 respawns sin frame), que `ThreadedCapture` interpreta como fuente
   muerta. Puente `-rw_timeout` (8 s) pliega los stalls de origen al mismo camino de respawn.
   Geometría forzada con `-vf scale=W:H` derivada de la misma config que dimensiona
   `FRAME_BYTES` (evita el desgarro silencioso si el origen cambiara de resolución). Referer
   de Claro vía `-headers` (reusa `_needs_claro_referer`). Costura de test: el 3.er posicional
   se aliasa a un spawner inyectable → los tests ejercitan respawn/reap sin lanzar ffmpeg real.

4. **Ruteo por defecto (Design A)**: `source_type` "hls"/"stream" (lo que manda el front)
   rutea al productor keyframe-only; OpenCV deja de ser el productor HLS por defecto pero queda
   registrado como escape hatch explícito "hls_opencv" (para una eventual URL HLS no-directa que
   requiera resolución Streamlink — las de Claro son `.m3u8` directas). Alias explícito
   "hls_keyframe". La decisión vive entera en el registry de fuentes; la capa de API no cambia.

5. **ffmpeg en la imagen edge**: agregado al `apt-get` del Dockerfile (`python:3.11-slim` +
   `opencv-python-headless` no traen el binario).

## Validación

- **Tests**: suite edge completa **186 passed**, incluyendo 12 tests del productor (vía la
  costura del spawner inyectable, sin ffmpeg real), 3 de resolución del umbral por-fuente, y
  4 de ruteo. `ruff check .` verde repo-wide.

- **E2E sobre streams reales de Claro en el contenedor** (`cerebrovial-edge_device-1`):
  - **Captura por keyframe confirmada**: 3 procesos ffmpeg vivos con
    `-skip_frame nokey -fps_mode passthrough -rw_timeout 8000000 -vf scale=1280:720`.
  - **Aislamiento de fallas**: matar el ffmpeg de una cámara (SIGTERM) la respawnea con PID
    nuevo, sin afectar las otras dos, sin zombies (PID viejo reapeado), con el respawn
    registrado en el log (`HlsKeyframeSource: subproceso ffmpeg murió/colgó (...); respawn 1/20`).
  - **No-freeze**: `GET /vision/health` reporta las 3 cámaras en `sensor_status=ok`, edad de
    frame 0.5–1.6 s (muy por debajo de 4.5 s), `aggregation_errors=0`, `data_dropped=0`; sin
    cascada de no-frescos ni errores/excepciones en logs.

## Follow-ups técnicos

- **Watchdog stale→rebuild completo**: diferido a su propia fase. Cubre el residual de
  stalls de origen donde ffmpeg mantiene el pipe abierto goteando bajo el `-rw_timeout`.
  Necesita una costura de rebuild que hoy no existe (`ThreadedCapture` recibe la fuente, no la
  fábrica) → se diseña a nivel `ThreadedCapture`/manager, transversal a todas las fuentes. El
  read-colgado ya queda *contenido* (no congela el proceso) y el puente `-rw_timeout` cubre el
  caso común, así que el residual es de segundo orden.
- **`-rw_timeout`**: validado en ffmpeg 8.1.1 (local) y en el binario apt del contenedor
  (los 3 procesos lo llevan y capturan a vivo). Opción AVIO estándar; aislado en
  `_RW_TIMEOUT_US` por si una imagen futura usara un build que la rechace.
- **Capa 3 (`SourceConfig.fresh_threshold_s`)**: incluida; sin consumidor por config todavía
  (el default por clase es el mecanismo activo). Deja la puerta abierta a tuning per-cámara.

## Commits

- `docs(vision)`: benchmark de captura keyframe-only — cierre del diagnóstico de la regresión
- `feat(vision)`: fresh_threshold_s como propiedad por-fuente
- `feat(vision)`: HlsKeyframeSource — captura keyframe-only sobre ffmpeg CLI
- `feat(vision)`: HlsKeyframeSource como productor HLS por defecto (Design A)
- `build(edge)`: instalar ffmpeg en la imagen del edge
- `chore(ruff)`: excluir documentation/benchmarks del lint
