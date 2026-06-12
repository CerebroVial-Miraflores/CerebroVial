# Clips de benchmark — estabilidad de ID por FPS

Insumo reproducible para el **Benchmark 3**: medir a qué FPS de observación ByteTrack
mantiene el ID de un vehículo estable de un lado al otro de una línea de conteo. Estos
clips se conservan (no son scratch) porque son reutilizables para futuros ajustes del
contador-línea direccional que reemplazará la fórmula `unique_vehicles / window * 3600`.

Los `.mp4` están **gitignorados** (binarios); este README es la fuente versionable que
documenta y permite regenerarlos.

## Captura

- **Fuente:** streams HLS de Claro (`https://live.smartechlatam.online/claro/{name}/index.m3u8`),
  el mismo catálogo que siembra `scripts/seed_intersections.py`.
- **Fecha/hora:** 2026-06-07, ~05:35–05:40 (hora Lima, UTC-5). Pre-amanecer; iluminación
  urbana presente (brillo medio 91–128 sobre 255). **No hay clip de pleno día** — la
  captura fue de madrugada; condición reportada con honestidad.
- **Formato:** 1280×720, 15 fps, ~60 s (900 frames), `mp4v`. Resampleado on-the-fly a una
  grilla uniforme de 15 fps (cada tick toma el frame decodificado más reciente), para que
  el submuestreo posterior a 5/3/1 fps sea limpio.

| Archivo | Cámara (Claro) | Condición | Brillo medio | Propósito |
|---|---|---|---|---|
| `clip_paseodelarepublica.mp4` | `paseodelarepublica` | Tráfico denso, luz media | ~94 | Caso con varios cruces |
| `clip_panamericana_peaje1.mp4` | `panamericana_peaje1` | Mejor iluminada, tráfico moderado | ~128 | Mejor condición visual |
| `clip_avfaustinocarrion.mp4` | `avfaustinocarrion` | Baja luz, tráfico escaso | ~91 | Peor caso (poca señal) |

> En la captura, `avfaustinocarrion` tenía ~0 vehículos en cuadro → sirve como caso de
> "sin señal" a esta hora, no como condición de tráfico denso en penumbra.

### Clips diurnos (Benchmark 3.5 — re-tuneo de ByteTrack)

- **Fecha/hora:** 2026-06-07, ~06:08–06:13 (hora Lima, UTC-5). Amanecer, luz en aumento,
  tráfico visible. Mismo formato (1280×720, 15 fps, 900 frames, `mp4v`).

| Archivo | Cámara (Claro) | Condición / ángulo | Brillo medio | Tráfico aparente | Propósito |
|---|---|---|---|---|---|
| `clip_dia_escuela_pnp.mp4` | `escuela_pnp` | Urbano, luz diurna | ~102 | Alto (~10 veh/frame) | Más cruces |
| `clip_dia_lamarina.mp4` | `lamarina` | Av. La Marina, urbano | ~107 | Medio (~5) | Tráfico medio |
| `clip_dia_panamericana.mp4` | `panamericana` | Autopista, ángulo distinto | ~112 | Medio, alta velocidad (~4) | Variar ángulo/velocidad |

### Grilla completa de mediodía — `2026-06-07/` (las 11 cámaras)

Captura masiva de las **11 cámaras Claro** para acumular material del spike del
contador-línea. Guardadas en la subcarpeta fechada
[`2026-06-07/`](2026-06-07/), nombre `clip_<camara>_<YYYYMMDDHHMMSS>` (timestamp en
hora Lima, UTC-5). Mismo formato (1280×720, 15 fps, 900 frames, `mp4v`).
`gallinazos` es nativo 1920×1080 → reescalado a 1280×720 en captura.

> **No procesadas en este paso** (sin YOLO, sin inspección de frames): `tráfico aparente`
> y `ángulo` quedan pendientes del spike. El único campo objetivo medido en captura es el
> brillo medio. El `tipo de vía` se infiere del **nombre del stream Claro** (lo que el feed
> muestra), no del mapeo intersección↔cámara (que es arbitrario — DEUDA-CAM-GEO).

| Archivo | Cámara | Hora (Lima) | Brillo | Tipo de vía (por nombre de stream) | Tráfico aparente | Ángulo |
|---|---|---|--:|---|---|---|
| `clip_escuela_pnp_20260607121345.mp4` | `escuela_pnp` | 12:13:45 | 145.2 | Punto urbano | pendiente | no inspeccionado |
| `clip_gallinazos_20260607121447.mp4` | `gallinazos` | 12:14:47 | 117.2 | Bajada/zona (no confirmado) | pendiente | no inspeccionado |
| `clip_prolongaciontacna_20260607121547.mp4` | `prolongaciontacna` | 12:15:47 | 114.6 | Avenida (Prolongación Tacna) | pendiente | no inspeccionado |
| `clip_lamarina_20260607121648.mp4` | `lamarina` | 12:16:48 | 111.0 | Avenida (La Marina) | pendiente | no inspeccionado |
| `clip_avfaustinocarrion_20260607121748.mp4` | `avfaustinocarrion` | 12:17:48 | 115.8 | Avenida (Faustino Carrión) | pendiente | no inspeccionado |
| `clip_paseodelarepublica_20260607121851.mp4` | `paseodelarepublica` | 12:18:51 | 113.8 | Vía expresa (Paseo de la República) | pendiente | no inspeccionado |
| `clip_angamos_20260607121959.mp4` | `angamos` | 12:19:59 | 111.4 | Avenida (Angamos) | pendiente | no inspeccionado |
| `clip_panamericana_20260607122117.mp4` | `panamericana` | 12:21:17 | 118.1 | Autopista (Panamericana) | pendiente | no inspeccionado |
| `clip_javierprado_20260607122321.mp4` | `javierprado` | 12:23:21 | 104.3 | Vía expresa/avenida (Javier Prado) | pendiente | no inspeccionado |
| `clip_derby_20260607122825.mp4` | `derby` | 12:28:25 | 114.9 | Óvalo/zona (Derby) | pendiente | no inspeccionado |
| `clip_panamericana_peaje1_20260607122927.mp4` | `panamericana_peaje1` | 12:29:27 | 132.3 | Autopista/peaje (Panamericana) | pendiente | no inspeccionado |

## Metodología (resumen)

1. De cada clip base (15 fps) se derivan por submuestreo: 15 fps (todos), 5 fps (1/3),
   3 fps (1/5), 1 fps (1/15) — **mismos vehículos en cada FPS**.
2. Se corre el pipeline real (`YoloDetector` + `SupervisionTracker`, parámetros de
   `edge_device/src/vision/infrastructure/tracking/supervision_tracker.py`:
   ByteTrack `frame_rate=30`, `lost_track_buffer=60`, `track_activation_threshold=0.15`).
   La detección se computa una vez por frame y se replaya a un tracker nuevo por cada FPS.
3. Línea de medición **sintética** horizontal en `y = alto/2` (solo para medir; NO es la
   geometría de producción).

## Regenerar

Los clips dependen del tráfico/luz del momento, así que una recaptura **no será idéntica**;
el README fija cámara, hora y condición de la corrida original. Para recapturar: abrir cada
URL con OpenCV (`cv2.VideoCapture`), resamplear a 15 fps y escribir 900 frames `mp4v`.
