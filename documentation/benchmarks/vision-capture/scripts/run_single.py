"""Eje 1 — caracterización de UN stream por mecanismo.
Mide: CPU/cámara (user+sys), fps efectivo, trayectoria de RSS, y guarda un frame
a los ~60s para verificar frescura-a-vivo por el reloj quemado.

Uso:  python run_single.py --mech {native|sleep1|kf_ffmpeg|kf_pyav} [--dur 120]
Escribe (append): data/eje1_single.csv, data/eje1_rss_trajectory.csv, frames/freshness_<mech>.jpg
"""
import argparse, csv, datetime, os, sys, threading, time
import cv2, psutil
sys.path.insert(0, os.path.dirname(__file__))
from _common import make_source, STREAM_NAMES, tree_cpu_times, tree_rss_mb

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
FRAMES = os.path.join(HERE, "..", "frames")

def append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", required=True)
    ap.add_argument("--dur", type=int, default=120)
    ap.add_argument("--stream", default=STREAM_NAMES[0])
    a = ap.parse_args()

    src = make_source(a.mech, a.stream)
    slot = {"img": None, "ts": None, "n": 0}
    lock = threading.Lock()
    stop = threading.Event()
    def reader():
        while not stop.is_set():
            img = src.read()
            if img is None:
                time.sleep(0.05); continue
            with lock:
                slot["img"] = img; slot["ts"] = time.monotonic(); slot["n"] += 1
    th = threading.Thread(target=reader, daemon=True); th.start()
    time.sleep(3)  # warmup (primer keyframe puede tardar)

    proc = psutil.Process()
    u0, s0 = tree_cpu_times(proc)
    t0 = time.monotonic()
    rss_traj = []
    saved_clock = None
    next_rss = 0.0
    while time.monotonic() - t0 < a.dur:
        el = time.monotonic() - t0
        if el >= next_rss:
            rss_traj.append((round(el, 0), round(tree_rss_mb(proc), 1)))
            next_rss += 10.0
        if saved_clock is None and el >= 60.0:
            with lock: img = slot["img"]
            if img is not None:
                cv2.imwrite(os.path.join(FRAMES, f"freshness_{a.mech}.jpg"), img)
                saved_clock = datetime.datetime.now().strftime("%H:%M:%S")
        time.sleep(1)
    u1, s1 = tree_cpu_times(proc)
    wall = time.monotonic() - t0
    with lock: nframes = slot["n"]
    stop.set(); time.sleep(0.5); src.close()

    cpu_user = u1 - u0; cpu_sys = s1 - s0
    cpu_cores = (cpu_user + cpu_sys) / wall
    rss_vals = [r for _, r in rss_traj]
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    append_csv(os.path.join(DATA, "eje1_single.csv"),
        ["timestamp_utc","mechanism","stream","duration_s","frames","fps_effective",
         "cpu_user_s","cpu_sys_s","cpu_cores","cpu_pct_per_cam",
         "rss_mb_min","rss_mb_max","rss_mb_end","system_clock_at_save","freshness_frame"],
        [ts, a.mech, a.stream, round(wall,1), nframes, round(nframes/wall,2),
         round(cpu_user,2), round(cpu_sys,2), round(cpu_cores,3), round(cpu_cores*100,1),
         round(min(rss_vals),1), round(max(rss_vals),1), round(rss_vals[-1],1),
         saved_clock or "NA", f"freshness_{a.mech}.jpg"])
    for el, r in rss_traj:
        append_csv(os.path.join(DATA, "eje1_rss_trajectory.csv"),
            ["mechanism","t_s","rss_mb"], [a.mech, el, r])
    print(f"[{a.mech}] frames={nframes} fps={nframes/wall:.2f} cpu/core={cpu_cores*100:.1f}% "
          f"rss={min(rss_vals):.0f}-{max(rss_vals):.0f}MB save_clock={saved_clock}", flush=True)

if __name__ == "__main__":
    main()
