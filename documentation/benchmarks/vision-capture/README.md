# Benchmark de captura de visión — reproducción

Material de sustento de la decisión de **mecanismo de captura** para el módulo de visión
(edge_device). Mide, sobre las cámaras HLS reales de Claro, el costo y la frescura de
distintos mecanismos de captura para inferencia, la cadencia de inferencia, y la
viabilidad de servir la vista del operador directo al browser.

> **Solo medición. No toca el pipeline de producción** (`edge_device/src/vision/`). Los
> scripts instancian las clases reales del módulo solo para leer; el resto usa subprocesos
> ffmpeg / PyAV aislados.

## Contexto / qué se mide
- **Eje 1** — mecanismo de captura: `native` (decode completo ~25 fps, baseline), `sleep1`
  (cv2 + throttle a 1 fps), `kf_ffmpeg` (keyframe-only vía ffmpeg CLI `-skip_frame nokey
  -fps_mode passthrough`), `kf_pyav` (keyframe-only vía PyAV `skip_frame='NONKEY'`).
- **Eje 2** — cadencia de inferencia (0.5 / 1 / 2 fps) sobre el mecanismo ganador.
- **Eje 3** — ¿la URL HLS de Claro es reproducible directo desde el browser (CORS) o debe
  proxearse por el edge?

## Entorno de la corrida de referencia
- **Fecha:** 2026-06-12. **Hora local:** los frames de frescura llevan el reloj quemado de
  la cámara (~UTC-5, Lima); se comparan contra el reloj del sistema al guardar.
- **Máquina:** Apple Silicon (macOS, Darwin 25.5.0), `mps` disponible. CPU se mide como
  fracción de **un core** (user+sys / wall); 100 % = un core saturado.
- **Stream(s):** `https://live.smartechlatam.online/claro/{name}/index.m3u8`, nombres reales
  sembrados en `scripts/seed_intersections.py`. Single-stream usa `escuela_pnp`;
  concurrencia usa los primeros 8 nombres. Segmentos de 2 s, **1 keyframe (I) por segmento
  → ~0.5 fps de keyframes**, ventana viva de 3 segmentos (~6 s).
- **Versiones:** Python 3.11.15 · ffmpeg 8.1.1 (CLI) · streamlink 8.4.0 · PyAV 17.1.0 ·
  opencv 4.13.0 · torch 2.9.1 · psutil 7.2.2 · ultralytics (ver `pip show ultralytics`).

## Dos venvs (PyAV NO va en el venv de producción)
```bash
# 1) venv de producción (ya existe): cv2 + módulo vision + ultralytics
PROD_VENV=./.venv

# 2) venv aparte para PyAV (NO tocar el de producción):
python3 -m venv ~/.venvs/pyav-bench
~/.venvs/pyav-bench/bin/pip install av numpy psutil opencv-python-headless ultralytics
#   av 17.1.0 trae sus libs ffmpeg; opencv para guardar frames; ultralytics para la
#   carga de inferencia bajo concurrencia.
```

## Reproducir
```bash
cd <repo>/CerebroVial
# todo el benchmark (secuencial, ~18 min; requiere red a Claro):
PROD_VENV="$PWD/.venv" PYAV_VENV="$HOME/.venvs/pyav-bench" \
  bash documentation/benchmarks/vision-capture/scripts/run_all.sh

# o pasos sueltos:
.venv/bin/python .../scripts/cors_check.py
.venv/bin/python .../scripts/run_single.py --mech kf_ffmpeg --dur 120
.venv/bin/python .../scripts/run_concurrency.py --mech kf_ffmpeg --streams 8 --infer-fps 1 --dur 80
~/.venvs/pyav-bench/bin/python .../scripts/run_single.py --mech kf_pyav --dur 120
```
Duraciones override: `SINGLE_DUR`, `CONC_DUR` (segundos).

## Salidas (en `../data/` y `../frames/`)
- `eje1_single.csv` — una fila por mecanismo (single stream): CPU, fps efectivo, RSS min/max/end.
- `eje1_rss_trajectory.csv` — RSS(MB) cada 10 s por mecanismo (formato largo, para graficar).
- `eje1_concurrency.csv` — una fila por (mecanismo × infer_fps) bajo 8 streams: CPU total,
  peor edad_max, nº cámaras sobre umbral, muertes de stream, RSS pico. **Sirve a Eje 1 e
  Eje 2** (filtrar por `infer_fps`).
- `eje1_concurrency_percam.csv` — edad_max y frames por cámara (crudo, sin agregar).
- `eje3_cors.csv` — cabeceras CORS del master playlist y de un segmento.
- `frames/freshness_<mech>.jpg` — frame guardado a los ~60 s; la frescura-a-vivo se lee del
  **reloj quemado** vs `system_clock_at_save` de `eje1_single.csv`.

## Método de frescura-a-vivo
No hay `tesseract` en el entorno → la frescura se mide **leyendo el reloj quemado** del
frame guardado (`frames/freshness_<mech>.jpg`) y restándolo de `system_clock_at_save`. El
valor consolidado vive en `RESULTS.md` (tabla Eje 1). Los frames quedan versionados como
evidencia; re-correr regenera el frame para re-verificar visualmente.

## Notas de fidelidad
- CPU se mide sobre el **árbol de proceso** (parent + children) para capturar el decode
  tanto in-process (native/sleep/pyav) como en subproceso (ffmpeg).
- `kf_ffmpeg`/`kf_pyav` emiten solo keyframes reales (passthrough / NONKEY): **sin** la
  duplicación a 25 fps que mete vsync CFR si se omite `-fps_mode passthrough`.
- La inferencia bajo concurrencia es **serializada** (un worker), como producción.
