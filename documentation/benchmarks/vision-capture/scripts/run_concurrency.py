"""Eje 1 (concurrencia) y Eje 2 (cadencia de inferencia) — N streams simultáneos.
Cada cámara: thread de captura (estilo ThreadedCapture) que actualiza un slot.
Main loop: muestrea edad por cámara y, si fresca, corre inferencia serializada a
la cadencia indicada (un worker, como producción). Mide CPU total del árbol,
peor edad_max, nº de cámaras que cruzan el umbral, y muertes de stream.

Uso: python run_concurrency.py --mech kf_ffmpeg --streams 8 --infer-fps 1 [--dur 80] [--thresh 2.5]
Escribe (append): data/eje1_concurrency.csv (resumen) y data/eje1_concurrency_percam.csv (crudo por cámara)
"""
import argparse, csv, datetime, os, sys, threading, time
import psutil
sys.path.insert(0, os.path.dirname(__file__))
from _common import make_source, STREAM_NAMES, tree_cpu_times, tree_rss_mb

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")

def append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)

class Cap(threading.Thread):
    def __init__(self, mech, name):
        super().__init__(daemon=True)
        self.mech = mech; self.name_ = name
        self.lock = threading.Lock(); self.img=None; self.ts=None; self.n=0; self.died=False
        self._stop=False
    def run(self):
        try: src = make_source(self.mech, self.name_)
        except Exception as e:
            self.died=True; print("open fail", self.name_, e, flush=True); return
        while not self._stop:
            img = src.read()
            if img is None:
                self.died=True; break
            with self.lock: self.img=img; self.ts=time.monotonic(); self.n+=1
        src.close()
    def snap(self):
        with self.lock: return self.img, self.ts
    def stop(self): self._stop=True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", required=True)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--infer-fps", type=float, default=1.0)
    ap.add_argument("--dur", type=int, default=80)
    ap.add_argument("--thresh", type=float, default=2.5)
    a = ap.parse_args()

    # Inferencia serializada (un worker) como producción.
    from ultralytics import YOLO
    model = YOLO(os.path.join(HERE, "..", "..", "..", "..", "edge_device", "yolo11n.pt"))
    ilock = threading.Lock()
    infer_period = 1.0 / a.infer_fps
    last_infer = {}

    caps = [Cap(a.mech, n) for n in STREAM_NAMES[:a.streams]]
    for c in caps: c.start()
    time.sleep(5)  # warmup

    proc = psutil.Process()
    maxage = {c.name_: 0.0 for c in caps}
    crossed = {c.name_: 0 for c in caps}
    rss_max = 0.0
    u0, s0 = tree_cpu_times(proc)
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.dur:
        now = time.monotonic()
        for c in caps:
            img, ts = c.snap()
            if ts is None: continue
            age = now - ts
            maxage[c.name_] = max(maxage[c.name_], age)
            if age > a.thresh: crossed[c.name_] += 1
            if img is not None and age <= a.thresh:
                if now - last_infer.get(c.name_, 0) >= infer_period:
                    with ilock:
                        model(img, device="cpu", imgsz=320, conf=0.3, verbose=False)
                    last_infer[c.name_] = now
        rss_max = max(rss_max, tree_rss_mb(proc))
        time.sleep(0.5)
    u1, s1 = tree_cpu_times(proc); wall = time.monotonic() - t0
    for c in caps: c.stop()
    time.sleep(2)

    cpu_cores = (u1 - u0 + s1 - s0) / wall
    worst = max(maxage.values()); ncross = sum(1 for v in crossed.values() if v > 0)
    deaths = sum(1 for c in caps if c.died)
    ts_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    append_csv(os.path.join(DATA, "eje1_concurrency.csv"),
        ["timestamp_utc","mechanism","n_streams","infer_fps","duration_s","fresh_thresh_s",
         "cpu_total_cores","cpu_total_pct","worst_age_max_s","n_cams_over_thresh","stream_deaths","rss_mb_max"],
        [ts_utc, a.mech, a.streams, a.infer_fps, round(wall,1), a.thresh,
         round(cpu_cores,2), round(cpu_cores*100,0), round(worst,2), ncross, deaths, round(rss_max,1)])
    for c in caps:
        append_csv(os.path.join(DATA, "eje1_concurrency_percam.csv"),
            ["timestamp_utc","mechanism","n_streams","infer_fps","camera","age_max_s","frames","died"],
            [ts_utc, a.mech, a.streams, a.infer_fps, c.name_, round(maxage[c.name_],2), c.n, int(c.died)])
    print(f"[{a.mech} x{a.streams} infer={a.infer_fps}fps] cpu={cpu_cores*100:.0f}% "
          f"worst_age={worst:.2f}s over_thresh={ncross}/{a.streams} deaths={deaths} rss_max={rss_max:.0f}MB", flush=True)

if __name__ == "__main__":
    main()
