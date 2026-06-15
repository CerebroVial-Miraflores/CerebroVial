"""Unit tests de FullDecodeSource (topología B, 15Hz).

Reusan los fakes de spawner/proc del test keyframe (NUNCA lanzan ffmpeg real).
Cubren lo que cambia vs HlsKeyframeSource: argv (sin `-skip_frame nokey`, con
`fps=N` en el `-vf`), el FRAME-CLOCK (`timestamp = frame_index / fps`,
MONOTÓNICO entre respawns), y la resolución por el registry.
"""
import numpy as np

from src.vision.infrastructure.sources import create_source
from src.vision.infrastructure.sources.base import SourceConfig
from src.vision.infrastructure.sources.full_decode_source import (
    FullDecodeSource,
    _FULLDECODE_FRESH_THRESHOLD_S,
)
from src.vision.infrastructure.sources.hls_keyframe_source import HlsKeyframeSource

_CLARO = "https://live.smartechlatam.online/claro/escuela_pnp/index.m3u8"


# ---- Fakes (proceso + spawner) — mismo patrón que test_hls_keyframe_source ----


class _FakeStdout:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, n):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _FakeProc:
    def __init__(self, chunks):
        self.stdout = _FakeStdout(chunks)

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None


class _Spawner:
    def __init__(self, procs):
        self._seq = list(procs)
        self.cmds = []

    def __call__(self, cmd):
        self.cmds.append(cmd)
        return self._seq.pop(0)


def _cfg(w=4, h=2, fresh=None):
    # Dimensiones chicas y pares (validador de SourceConfig): FRAME_BYTES = 4*2*3 = 24.
    return SourceConfig(target_width=w, target_height=h, fresh_threshold_s=fresh)


def _full_frame(w=4, h=2):
    return bytes(w * h * 3)


# ---- argv (_build_cmd, sin spawnear) ---------------------------------


def test_build_cmd_drops_skip_frame_and_adds_fps():
    src = FullDecodeSource(_CLARO, _cfg(w=640, h=480), fps=15)
    cmd = src._build_cmd()
    # Decode completo: SIN keyframe-only.
    assert "-skip_frame" not in cmd
    assert "nokey" not in cmd
    # Muestreo a fps fijo + escala (geometría determinística) en un solo -vf.
    assert "fps=15,scale=640:480" in cmd
    assert src._frame_bytes == 640 * 480 * 3


def test_build_cmd_keeps_referer_for_claro():
    """Hereda la lógica de Referer del base (paridad con keyframe)."""
    cmd = FullDecodeSource(_CLARO, _cfg(), fps=15)._build_cmd()
    assert "-headers" in cmd


def test_build_cmd_adds_re_before_input():
    """`-re` (pacing del decode bursty del HLS live) va como input option ANTES de -i."""
    cmd = FullDecodeSource(_CLARO, _cfg(), fps=15)._build_cmd()
    assert "-re" in cmd
    assert cmd.index("-re") < cmd.index("-i")  # input option: antes de -i


def test_keyframe_base_has_no_re():
    """Regresión: el `-re` es solo del path live full-decode, no del base keyframe."""
    assert "-re" not in HlsKeyframeSource(_CLARO, _cfg())._build_cmd()


def test_keyframe_source_unchanged_still_skip_frame():
    """Regresión: el default keyframe NO cambió (sigue keyframe-only)."""
    cmd = HlsKeyframeSource(_CLARO, _cfg())._build_cmd()
    assert "-skip_frame" in cmd and "nokey" in cmd
    assert "scale=4:2" in cmd
    assert "fps=" not in " ".join(cmd)


# ---- frame-clock (timestamp = frame_index / fps) ---------------------


def test_timestamp_is_frame_clock():
    proc = _FakeProc([_full_frame(), _full_frame(), _full_frame()])
    src = FullDecodeSource(_CLARO, _cfg(), capture=_Spawner([proc]), fps=15)
    f0, f1, f2 = src.read(), src.read(), src.read()
    assert (f0.id, f1.id, f2.id) == (0, 1, 2)
    # ts = id / fps, NO wall-clock.
    assert f0.timestamp == 0 / 15
    assert f1.timestamp == 1 / 15
    assert f2.timestamp == 2 / 15


def test_frame_clock_monotonic_across_respawn():
    """Cuidado (a): el contador (y el ts) NO se resetea cuando ffmpeg muere y
    respawnea — el ts sigue creciendo (monótono), no salta hacia atrás."""
    # proc1 entrega 2 frames y luego EOF (b"") → muerte → respawn.
    proc1 = _FakeProc([_full_frame(), _full_frame(), b""])
    proc2 = _FakeProc([_full_frame(), _full_frame()])
    src = FullDecodeSource(_CLARO, _cfg(), capture=_Spawner([proc1, proc2]), fps=15)

    f0 = src.read()   # proc1 frame 0
    f1 = src.read()   # proc1 frame 1
    f2 = src.read()   # proc1 EOF → respawn → proc2 frame
    f3 = src.read()   # proc2 frame

    ids = [f0.id, f1.id, f2.id, f3.id]
    ts = [f0.timestamp, f1.timestamp, f2.timestamp, f3.timestamp]
    assert ids == [0, 1, 2, 3]            # contador NO reseteado por el respawn
    assert ts == [0 / 15, 1 / 15, 2 / 15, 3 / 15]
    assert ts == sorted(ts)               # monótono creciente


def test_full_frame_geometry():
    proc = _FakeProc([_full_frame()])
    src = FullDecodeSource(_CLARO, _cfg(), capture=_Spawner([proc]), fps=15)
    f = src.read()
    assert f.image.shape == (2, 4, 3)
    assert f.image.dtype == np.uint8


def test_rejects_bad_fps():
    import pytest

    with pytest.raises(ValueError):
        FullDecodeSource(_CLARO, _cfg(), fps=0)


# ---- fresh threshold + registry --------------------------------------


def test_fresh_threshold_default_is_fulldecode():
    src = FullDecodeSource(_CLARO, SourceConfig())
    assert src.fresh_threshold_s == _FULLDECODE_FRESH_THRESHOLD_S  # 1.0, no el 4.5 keyframe


def test_fresh_threshold_config_override_wins():
    src = FullDecodeSource(_CLARO, SourceConfig(fresh_threshold_s=2.0))
    assert src.fresh_threshold_s == 2.0


def test_registry_resolves_hls_fulldecode():
    src = create_source(_CLARO, source_type="hls_fulldecode", target_width=4, target_height=2)
    assert isinstance(src, FullDecodeSource)


def test_registry_hls_default_still_keyframe():
    """Regresión: "hls"/"stream" siguen ruteando al keyframe (no al full-decode)."""
    assert isinstance(
        create_source(_CLARO, source_type="hls", target_width=4, target_height=2),
        HlsKeyframeSource,
    )
    assert not isinstance(
        create_source(_CLARO, source_type="stream", target_width=4, target_height=2),
        FullDecodeSource,
    )
