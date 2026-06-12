"""Unit tests de HlsKeyframeSource (B-fase1).

Ejercitan la costura del spawner inyectable (`capture=`) — NUNCA lanzan ffmpeg real.
Cubren: frame feliz + id incremental, ensamblado de lecturas cortas, auto-sanación ante
muerte transitoria del subproceso (reap+respawn sin zombies), give-up duro, backoff
capado, release idempotente, construcción del argv (referer/scale/FRAME_BYTES), umbral
por-fuente, y degradación visible ante fallo de spawn (ffmpeg ausente).
"""
import numpy as np

from src.vision.infrastructure.sources.base import SourceConfig
from src.vision.infrastructure.sources.hls_keyframe_source import (
    HlsKeyframeSource,
    _BACKOFF_CAP_S,
    _FRESH_THRESHOLD_S,
    _MAX_RESPAWN,
)

_CLARO = "https://live.smartechlatam.online/claro/escuela_pnp/index.m3u8"
_NON_CLARO = "https://cdn.example.com/live.m3u8"


# ---- Fakes (proceso + spawner) ---------------------------------------

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
        self.killed = False
        self.waited = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def poll(self):
        return None


class _Spawner:
    """Spawner llamable: devuelve procs de una cola; si se agota, usa `default_factory`."""

    def __init__(self, procs=None, default_factory=None):
        self._seq = list(procs or [])
        self._default = default_factory
        self.cmds = []
        self.procs = []

    def __call__(self, cmd):
        self.cmds.append(cmd)
        if self._seq:
            p = self._seq.pop(0)
        elif self._default is not None:
            p = self._default()
        else:
            raise AssertionError("spawner agotado")
        self.procs.append(p)
        return p

    @property
    def count(self):
        return len(self.cmds)


def _cfg(w=4, h=2, fresh=None):
    # Dimensiones chicas y pares (validador de SourceConfig): FRAME_BYTES = 4*2*3 = 24.
    return SourceConfig(target_width=w, target_height=h, fresh_threshold_s=fresh)


def _full_frame(w=4, h=2):
    return bytes(w * h * 3)


# ---- read(): frames y ensamblado -------------------------------------

def test_read_returns_frames_with_incrementing_id():
    proc = _FakeProc([_full_frame(), _full_frame()])
    sp = _Spawner([proc])
    src = HlsKeyframeSource(_CLARO, _cfg(), capture=sp)

    f0 = src.read()
    f1 = src.read()
    assert f0.id == 0 and f1.id == 1
    assert f0.image.shape == (2, 4, 3)
    assert f0.image.dtype == np.uint8
    assert sp.count == 1  # un solo spawn, sin respawns


def test_read_assembles_partial_reads():
    fb = _full_frame()
    proc = _FakeProc([fb[:10], fb[10:]])  # frame entregado en 2 lecturas cortas
    sp = _Spawner([proc])
    src = HlsKeyframeSource(_CLARO, _cfg(), capture=sp)

    f = src.read()
    assert f.id == 0
    assert f.image.shape == (2, 4, 3)
    assert sp.count == 1  # un short-read NO es muerte: no hubo respawn


# ---- Supervisión: auto-sanación, give-up, zombies, backoff -----------

def test_read_self_heals_on_transient_death(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    dead = _FakeProc([b""])             # EOF inmediato (subproceso muerto)
    alive = _FakeProc([_full_frame()])
    sp = _Spawner([dead, alive])
    src = HlsKeyframeSource(_CLARO, _cfg(), capture=sp)

    f = src.read()
    assert f is not None and f.id == 0  # devolvió frame, NO None
    assert dead.killed and dead.waited  # proc muerto reapeado (kill+wait, sin zombie)
    assert sp.count == 2                 # respawn


def test_read_gives_up_after_max_respawn(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    sp = _Spawner(default_factory=lambda: _FakeProc([b""]))  # infinitos EOF
    src = HlsKeyframeSource(_CLARO, _cfg(), capture=sp)

    assert src.read() is None
    assert sp.count == _MAX_RESPAWN + 1                  # 21 spawns y se rinde
    assert all(p.killed and p.waited for p in sp.procs)  # todos reapeados (sin zombies)


def test_backoff_is_non_decreasing_and_capped(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    sp = _Spawner(default_factory=lambda: _FakeProc([b""]))
    src = HlsKeyframeSource(_CLARO, _cfg(), capture=sp)

    src.read()
    assert sleeps == sorted(sleeps)        # no-decreciente
    assert max(sleeps) <= _BACKOFF_CAP_S    # capado
    assert sleeps[0] < sleeps[-1]           # creció (exponencial hasta el cap)


def test_read_gives_up_on_spawn_failure(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    def boom(cmd):
        raise FileNotFoundError("ffmpeg not found")

    src = HlsKeyframeSource(_CLARO, _cfg(), capture=boom)
    assert src.read() is None  # degrada visible (None), NO crashea el daemon thread


# ---- release ---------------------------------------------------------

def test_release_is_idempotent():
    proc = _FakeProc([_full_frame()])
    sp = _Spawner([proc])
    src = HlsKeyframeSource(_CLARO, _cfg(), capture=sp)

    src.read()  # spawnea proc
    src.release()
    assert proc.killed and proc.waited
    src.release()  # segundo release: no-op, sin excepción


# ---- argv (_build_cmd, sin spawnear) ---------------------------------

def test_build_cmd_includes_referer_before_input_for_claro():
    src = HlsKeyframeSource(_CLARO, _cfg())  # capture=None; lazy → no se spawnea
    cmd = src._build_cmd()
    assert "-headers" in cmd
    i_headers = cmd.index("-headers")
    assert cmd[i_headers + 1] == "Referer: https://claro.com.pe/\r\n"
    assert i_headers < cmd.index("-i")  # input-opt antes de -i


def test_build_cmd_omits_referer_for_non_claro():
    src = HlsKeyframeSource(_NON_CLARO, _cfg())
    assert "-headers" not in src._build_cmd()


def test_build_cmd_scale_matches_frame_bytes():
    src = HlsKeyframeSource(_CLARO, _cfg(w=640, h=480))
    assert "scale=640:480" in src._build_cmd()
    assert src._frame_bytes == 640 * 480 * 3


def test_defaults_to_720p_when_dims_absent():
    src = HlsKeyframeSource(_CLARO, SourceConfig())  # sin target_width/height
    assert src._frame_bytes == 1280 * 720 * 3
    assert "scale=1280:720" in src._build_cmd()


# ---- umbral por-fuente -----------------------------------------------

def test_fresh_threshold_default_and_config_override():
    assert HlsKeyframeSource(_CLARO, SourceConfig()).fresh_threshold_s == _FRESH_THRESHOLD_S
    over = HlsKeyframeSource(_CLARO, SourceConfig(fresh_threshold_s=6.0))
    assert over.fresh_threshold_s == 6.0
