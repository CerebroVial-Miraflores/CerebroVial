"""HlsKeyframeSource — productor keyframe-only de HLS vía subproceso ffmpeg CLI (B-fase1).

Reemplazo del decode completo (`OpenCVSource`/cv2) para fuentes HLS: ffmpeg
decodifica **solo keyframes** (`-skip_frame nokey`) y vuelca rawvideo `bgr24` por un
pipe; cada `read()` arma un `Frame` con el `ndarray` BGR. El benchmark
(`documentation/benchmarks/vision-capture/`) lo midió: ~0.5 fps (1 keyframe/segmento
de 2 s), CPU ~1.3 %/cámara, frescura-a-vivo — el único mecanismo que da bajo CPU sin
quedar rancio. Por eso `read()` **bloquea ~2 s** entre frames (cadencia del keyframe);
`ThreadedCapture` lo absorbe en su daemon thread y el scheduler usa el umbral por-fuente
(`fresh_threshold_s = 4.5`, sobre la sierra de edad ~2–3 s).

**Aislamiento de fallas (criterio de la decisión ffmpeg CLI sobre PyAV):** el subproceso
muerto/colgado se detecta como lectura corta (`len(buf) < FRAME_BYTES`) y se reinicia
sin tumbar el edge. La supervisión vive dentro de `read()`: reap (kill+wait, sin zombies)
+ respawn con backoff; `read()` solo devuelve `None` en give-up duro (reintentos agotados),
que `ThreadedCapture` interpreta como fuente muerta. Un origen estancado (socket abierto,
bytes sin fluir) se pliega al mismo camino vía `-rw_timeout` (ffmpeg cierra el pipe →
lectura corta → respawn). El watchdog stale→rebuild completo es fase aparte (la costura
de rebuild no vive acá: `ThreadedCapture` recibió la fuente, no la fábrica).

**Geometría determinística:** el pipe `bgr24` no lleva framing; si ffmpeg emitiera otra
resolución, el reshape "funciona" pero todo frame posterior queda desgarrado. Se fuerza
la geometría de salida con `-vf scale=W:H` derivada de la misma config que dimensiona
`FRAME_BYTES` (target_width/height, o 1280×720 por defecto) → el framing siempre coincide.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np

from ...domain.entities import Frame
from ...domain.protocols import FrameProducer
from .base import SourceConfig
from .video_source import _needs_claro_referer

logger = logging.getLogger(__name__)

# Geometría por defecto (Claro 720p; coincide con el benchmark y con _build_camera_config).
_DEFAULT_W = 1280
_DEFAULT_H = 720
_REFERER = "https://claro.com.pe/"

# Timeout de I/O de ffmpeg (µs): un origen estancado provoca cierre del pipe → lectura
# corta → respawn (puente barato del watchdog de origen). 8 s ≫ cadencia de 2 s del
# keyframe, así que la operación normal 0.5 fps nunca lo dispara. Constante compartida
# con el watchdog completo (fase aparte) para afinarlos en concierto.
_RW_TIMEOUT_US = 8_000_000

# Backoff máximo de reconexión in-process de ffmpeg (P1). Ante corte de red, ffmpeg
# reintenta con backoff capado a este valor y resume al volver la red, sin morir.
_RECONNECT_DELAY_MAX_S = 4

# Supervisión del subproceso.
_MAX_RESPAWN = 20          # respawns CONSECUTIVOS sin frame bueno antes del give-up
_BACKOFF_BASE_S = 0.25     # backoff exponencial capado: 0.25, 0.5, 1.0, 2.0, 2.0...
_BACKOFF_CAP_S = 2.0
_REAP_TIMEOUT_S = 5.0      # wait() tras kill() — reapea el exit status (sin zombies)

# Umbral de frescura por-fuente (cadencia 0.5 fps → sierra ~2–3 s; 4.5 da margen).
_FRESH_THRESHOLD_S = 4.5

# Tipo del spawner inyectable (costura de test): cmd argv → objeto proceso con
# .stdout.read(n), .kill(), .wait(timeout). Default = subprocess.Popen.
Spawner = Callable[[list], object]


class HlsKeyframeSource(FrameProducer):
    """Productor `FrameProducer` keyframe-only sobre un subproceso ffmpeg supervisado."""

    # Default por-fuente; override per-cámara vía SourceConfig.fresh_threshold_s.
    fresh_threshold_s: float = _FRESH_THRESHOLD_S

    def __init__(self, source, config: SourceConfig, capture: Optional[Spawner] = None):
        self._url = source
        self.config = config
        # Geometría desde config (o default); FRAME_BYTES se deriva de los MISMOS
        # valores que se fuerzan en ffmpeg (`-vf scale`) → framing siempre coincidente.
        self._w = config.target_width or _DEFAULT_W
        self._h = config.target_height or _DEFAULT_H
        self._frame_bytes = self._w * self._h * 3
        self._needs_referer = _needs_claro_referer(source)
        # Costura de test: `capture` se aliasa al spawner (no a un proceso vivo) para
        # poder devolver una SECUENCIA de procesos fake y ejercitar el respawn.
        self._spawn: Spawner = capture or self._default_spawn
        self._proc = None  # spawn lazy (primer read()) → factory/tests no lanzan ffmpeg
        self._frame_id = 0
        # Capa 3: override per-cámara desde config; si no, el default de clase.
        if config.fresh_threshold_s is not None:
            self.fresh_threshold_s = config.fresh_threshold_s

    # ---- argv / spawn --------------------------------------------------

    def _build_cmd(self) -> list:
        """argv de ffmpeg. Input-opts antes de `-i`; output-opts después.

        `-headers Referer` solo si la URL es de Claro (paridad con OpenCVSource, SPIKE-D5).
        `-vf scale=W:H` fuerza la geometría que dimensiona FRAME_BYTES.
        """
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "nobuffer",
            # Resiliencia de red (P1): ante corte/EOF del CDN, ffmpeg RECONECTA in-process
            # en vez de morir → evita el respawn storm (baseline: ~13 respawns/9s, blackout
            # ~10-12s con frames dropeados). Opciones del protocolo HTTP(S), aplican al
            # fetch de la playlist y los segmentos HLS. Deben ir ANTES de `-i` (input opts).
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_delay_max", str(_RECONNECT_DELAY_MAX_S),
            "-rw_timeout", str(_RW_TIMEOUT_US),
        ]
        if self._needs_referer:
            cmd += ["-headers", f"Referer: {_REFERER}\r\n"]
        cmd += self._input_frame_opts()
        cmd += [
            "-i", self._url,
            "-an", "-fps_mode", "passthrough",
            "-vf", self._vf_filter(),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        return cmd

    def _input_frame_opts(self) -> list:
        """Opciones de selección de frame ANTES de `-i`. Keyframe-only por defecto
        (`-skip_frame nokey`). `FullDecodeSource` lo override a `[]` (decode completo)."""
        return ["-skip_frame", "nokey"]

    def _vf_filter(self) -> str:
        """Filtro `-vf`. Solo escala (geometría determinística) por defecto.
        `FullDecodeSource` antepone `fps=N` para muestrear a cadencia fija."""
        return f"scale={self._w}:{self._h}"

    def _frame_timestamp(self, frame_index: int) -> float:
        """Timestamp del frame. Keyframe-only: wall-clock de captura.
        `FullDecodeSource` lo override al frame-clock (`frame_index / fps`)."""
        return time.time()

    def _default_spawn(self, cmd: list):
        """Spawner real (path de producción). cv2/ffmpeg viven en procesos distintos."""
        import subprocess

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes,
        )

    def _spawn_proc(self) -> bool:
        """Arranca un subproceso fresco. True si quedó vivo; False si el spawn falló
        (p. ej. ffmpeg ausente). NO lanza: `read()` corre en el daemon thread de
        `ThreadedCapture` y una excepción lo mataría en silencio — degradamos visible."""
        try:
            self._proc = self._spawn(self._build_cmd())
            return True
        except (OSError, ValueError) as e:  # FileNotFoundError es OSError
            logger.error("HlsKeyframeSource: fallo al spawnear ffmpeg (%s): %s", self._url, e)
            self._proc = None
            return False

    # ---- lectura / supervisión ----------------------------------------

    def _read_full(self, n: int) -> Optional[bytes]:
        """Lee EXACTAMENTE n bytes acumulando lecturas cortas del pipe (un short-read
        no es muerte). `b''` == EOF → proceso terminado. Garantiza entregar un frame
        entero al reshape (preserva el framing)."""
        stdout = self._proc.stdout
        chunks = []
        got = 0
        while got < n:
            b = stdout.read(n - got)
            if not b:
                return None
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)

    def _reap(self) -> None:
        """Mata y REAPEA el subproceso (kill + wait) → sin zombies acumulados a lo
        largo de los respawns. Idempotente (nulea `_proc`, guarda None)."""
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            p.kill()
        except Exception:  # noqa: BLE001 - kill sobre pid ya reapeado puede lanzar
            pass
        try:
            p.wait(timeout=_REAP_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - el wait no debe propagar en teardown
            pass

    @staticmethod
    def _backoff(attempts: int) -> float:
        """Backoff exponencial capado: evita el hot-loop de spawns ante un origen que
        flapea, sin dejar de ser responsivo ante un blip transitorio."""
        return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** (attempts - 1)))

    def read(self) -> Optional[Frame]:
        """Bloquea hasta el próximo keyframe y devuelve un `Frame`. Auto-sana caídas
        transitorias del subproceso (reap + respawn); devuelve `None` solo tras
        `_MAX_RESPAWN` fallas consecutivas (give-up → fuente muerta para ThreadedCapture)."""
        attempts = 0
        while True:
            if self._proc is None and not self._spawn_proc():
                attempts += 1
                if attempts > _MAX_RESPAWN:
                    logger.error(
                        "HlsKeyframeSource: %d spawns fallidos (%s); se rinde (read->None)",
                        _MAX_RESPAWN, self._url,
                    )
                    return None
                time.sleep(self._backoff(attempts))
                continue

            buf = self._read_full(self._frame_bytes)
            if buf is not None:
                # .copy(): np.frombuffer es vista read-only que aliasa `buf` (transitorio);
                # el Frame debe ser dueño de su imagen.
                img = np.frombuffer(buf, np.uint8).reshape(self._h, self._w, 3).copy()
                frame = Frame(
                    id=self._frame_id,
                    timestamp=self._frame_timestamp(self._frame_id),
                    image=img,
                )
                self._frame_id += 1
                return frame

            # Lectura corta / EOF → subproceso muerto o colgado (vía -rw_timeout).
            self._reap()
            attempts += 1
            if attempts > _MAX_RESPAWN:
                logger.error(
                    "HlsKeyframeSource: %d respawns sin frame (%s); se rinde (read->None)",
                    _MAX_RESPAWN, self._url,
                )
                return None
            logger.warning(
                "HlsKeyframeSource: subproceso ffmpeg murió/colgó (%s); respawn %d/%d",
                self._url, attempts, _MAX_RESPAWN,
            )
            time.sleep(self._backoff(attempts))

    def release(self) -> None:
        """Cierra el subproceso (kill + wait). Idempotente."""
        self._reap()
