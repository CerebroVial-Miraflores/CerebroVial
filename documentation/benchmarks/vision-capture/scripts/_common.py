"""Helpers compartidos del benchmark de captura de visión (read-only, fuera de producción).

Mecanismos de captura comparados (todos entregan ndarray BGR (H,W,3) uint8):
  - native      : cv2.VideoCapture, lazo apretado (decode completo ~25fps)  [requiere cv2]
  - sleep1      : cv2.VideoCapture, read()+sleep -> 1 fps de lectura          [requiere cv2]
  - kf_ffmpeg   : subproceso ffmpeg -skip_frame nokey -fps_mode passthrough   [requiere ffmpeg CLI]
  - kf_pyav     : PyAV con codec_context.skip_frame='NONKEY'                  [requiere av]

CPU se mide sobre el ÁRBOL de proceso (parent + children) para capturar el
decode tanto in-process (native/sleep/pyav) como en subproceso (ffmpeg).
"""
import os, subprocess, time
import numpy as np
import psutil

W, H = 1280, 720
FRAME_BYTES = W * H * 3
REFERER = "https://claro.com.pe/"
BASE = "https://live.smartechlatam.online/claro/{}/index.m3u8"

# 11 nombres reales sembrados (seed_intersections.py). Para 8-streams se usan los primeros 8.
STREAM_NAMES = [
    "escuela_pnp", "gallinazos", "prolongaciontacna", "lamarina", "avfaustinocarrion",
    "paseodelarepublica", "angamos", "panamericana", "javierprado", "derby", "panamericana_peaje1",
]

def url(name): return BASE.format(name)

# ---- medición de CPU/RSS sobre el árbol ----------------------------------
def tree_cpu_times(proc):
    u = s = 0.0
    for p in [proc] + proc.children(recursive=True):
        try:
            ct = p.cpu_times(); u += ct.user; s += ct.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return u, s

def tree_rss_mb(proc):
    r = 0
    for p in [proc] + proc.children(recursive=True):
        try: r += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    return r / 1e6

# ---- fuentes de frames por mecanismo -------------------------------------
# Cada una es un context manager-ish: .read() -> ndarray|None ; .close()

class Cv2Source:
    """native / sleep1 vía cv2.VideoCapture (decode completo). throttle_fps=None => nativo."""
    def __init__(self, name, throttle_fps=None):
        import cv2
        self._cv2 = cv2
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"referer;{REFERER}"
        self.cap = cv2.VideoCapture(url(name))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.period = (1.0 / throttle_fps) if throttle_fps else 0.0
    def read(self):
        t0 = time.monotonic()
        ok, img = self.cap.read()
        if not ok:
            return None
        if img.shape[1] != W or img.shape[0] != H:
            img = self._cv2.resize(img, (W, H))
        if self.period:
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)))
        return img
    def close(self):
        try: self.cap.release()
        except Exception: pass

class FfmpegKeyframeSource:
    """keyframe-only vía subproceso ffmpeg CLI (-skip_frame nokey -fps_mode passthrough)."""
    def __init__(self, name):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-fflags", "nobuffer",
               "-headers", f"Referer: {REFERER}\r\n", "-skip_frame", "nokey",
               "-i", url(name), "-an", "-fps_mode", "passthrough",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=FRAME_BYTES)
    def read(self):
        buf = self.p.stdout.read(FRAME_BYTES)
        if len(buf) < FRAME_BYTES:
            return None
        return np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    def close(self):
        try: self.p.kill()
        except Exception: pass

class PyAvKeyframeSource:
    """keyframe-only vía PyAV (codec_context.skip_frame='NONKEY')."""
    def __init__(self, name):
        import av
        self.container = av.open(url(name), options={"headers": f"Referer: {REFERER}\r\n"})
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.skip_frame = "NONKEY"
        self._gen = self.container.decode(self.stream)
    def read(self):
        try:
            frame = next(self._gen)
        except StopIteration:
            return None
        img = frame.to_ndarray(format="bgr24")
        return img
    def close(self):
        try: self.container.close()
        except Exception: pass

def make_source(mech, name):
    if mech == "native":   return Cv2Source(name, throttle_fps=None)
    if mech == "sleep1":   return Cv2Source(name, throttle_fps=1.0)
    if mech == "kf_ffmpeg":return FfmpegKeyframeSource(name)
    if mech == "kf_pyav":  return PyAvKeyframeSource(name)
    raise ValueError(mech)
