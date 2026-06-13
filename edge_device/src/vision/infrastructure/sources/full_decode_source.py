"""FullDecodeSource — productor full-decode a fps fijo (topología B, 15Hz).

A >0.5Hz no alcanza con keyframes: hay que decodificar el segmento ENTERO y
muestrear a cadencia fija. Reusa toda la maquinaria de `HlsKeyframeSource`
(subproceso ffmpeg supervisado, reap/respawn con backoff, geometría
determinística por `-vf scale`, lectura exacta de FRAME_BYTES) y solo cambia:

- **SIN `-skip_frame nokey`** → decode completo (todos los frames del GOP).
- **`-vf fps=N,scale=W:H`** → muestrea a N fps fijo (default 15).
- **timestamp = `frame_index / fps`** (FRAME-CLOCK), no wall-clock de lectura.

Por qué el frame-clock (cuidado (a) del plan): `_frame_id` persiste a través del
reap/respawn (el reap solo nulea `_proc`, no el contador), así que el ts es
MONOTÓNICO entre respawns. Un wall-clock de lectura saltaría con el jitter de
entrega de ffmpeg (burst de arranque, catch-up post-stall) → Δt erróneos →
velocidad inflada y Kalman del tracker roto. El frame-clock lo blinda. (No es
fully-correct: subcuenta el Δt a través de stalls <2s; el correcto de verdad es
el PTS del video, diferido a cuando la precisión importe — uso del motor.)
"""
from __future__ import annotations

from typing import Optional

from .base import SourceConfig
from .hls_keyframe_source import HlsKeyframeSource, Spawner

# Cadencia de muestreo por defecto (15 fps; topología B).
_DEFAULT_FPS = 15
# Umbral de frescura por-fuente: a 15fps la cadencia es ~67ms; 1.0s da margen
# holgado sobre un par de frames perdidos sin marcar la fuente como rancia.
_FULLDECODE_FRESH_THRESHOLD_S = 1.0


class FullDecodeSource(HlsKeyframeSource):
    """Productor `FrameProducer` full-decode a fps fijo sobre ffmpeg supervisado."""

    fresh_threshold_s: float = _FULLDECODE_FRESH_THRESHOLD_S

    def __init__(
        self,
        source,
        config: SourceConfig,
        capture: Optional[Spawner] = None,
        fps: int = _DEFAULT_FPS,
    ) -> None:
        if fps < 1:
            raise ValueError("fps debe ser >= 1")
        # El base aplica el override per-cámara de fresh_threshold si viene en config;
        # si no, queda el class-attr full-decode (1.0s), no el 4.5s keyframe.
        super().__init__(source, config, capture=capture)
        self._fps = fps

    def _input_frame_opts(self) -> list:
        return []  # decode completo (sin -skip_frame nokey)

    def _vf_filter(self) -> str:
        return f"fps={self._fps},scale={self._w}:{self._h}"

    def _frame_timestamp(self, frame_index: int) -> float:
        # FRAME-CLOCK: cadencia fija, monotónica entre respawns.
        return frame_index / self._fps
