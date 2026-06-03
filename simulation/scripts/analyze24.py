"""C2 — analizador del barrido 24h continuo. Por cada scale: perfil horario de
velocidad (ground-truth edgeData, ponderado por sampledSeconds), presencia
(sampledSeconds/h), y totales del stats. Streaming (edgeData es grande)."""
import argparse, xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def profile(edge_xml):
    w = [0.0]*24; sv = [0.0]*24  # sampledSeconds-sum y sampledSeconds*speed-sum por hora
    cur = 0
    ctx = ET.iterparse(edge_xml, events=("start", "end"))
    for ev, el in ctx:
        if ev == "start" and el.tag == "interval":
            cur = min(int(float(el.get("begin"))//3600), 23)
        elif ev == "end" and el.tag == "edge":
            ss = float(el.get("sampledSeconds", "0") or 0)
            if ss > 0 and el.get("speed") is not None:
                w[cur] += ss; sv[cur] += ss*float(el.get("speed"))
            el.clear()
        elif ev == "end" and el.tag == "interval":
            el.clear()
    kmh = [(sv[h]/w[h]*3.6 if w[h] > 0 else 0.0) for h in range(24)]
    return kmh, w


def totals(stats_xml):
    r = ET.parse(stats_xml).getroot()
    v = r.find("vehicles"); t = r.find("teleports"); s = r.find("safety")
    ts = r.find("vehicleTripStatistics")
    return dict(loaded=int(v.get("loaded")), inserted=int(v.get("inserted")),
                waiting=int(v.get("waiting")), tel=int(t.get("total")),
                jam=int(t.get("jam")), yld=int(t.get("yield")), wl=int(t.get("wrongLane")),
                col=int(s.get("collisions")), em=int(s.get("emergencyStops")),
                kmh=float(ts.get("speed"))*3.6)


def shape_verdict(kmh, w):
    """Heurísticas: ¿drena? (nunca clavado <8 sostenido en meseta/picos) y
    ¿presencia tiene forma? (no monótona creciente)."""
    # tramo 07-23h (donde antes se clavaba). sostenido<8 si >=3h consecutivas <8.
    low = [h for h in range(7, 24) if kmh[h] < 8.0]
    # racha consecutiva máxima <8
    run = mx = 0
    for h in range(7, 24):
        run = run+1 if kmh[h] < 8.0 else 0
        mx = max(mx, run)
    # presencia monótona: ¿el pico de sampledSeconds está al final del día?
    peak_h = max(range(24), key=lambda h: w[h])
    drains = mx < 3
    return drains, mx, peak_h


def main():
    ap = argparse.ArgumentParser(description="Analizador del barrido 24h continuo.")
    ap.add_argument("--dir", type=str, default=str(HERE),
                    help="dir base donde viven los sweep24_s<tag>/ (default: el dir del "
                         "script; el sweep escribe en OUTDIR/sweep24_s<tag>/).")
    ap.add_argument("scales", nargs="*", help="scales a analizar (default: 0.25 0.20 0.15 0.10)")
    args = ap.parse_args()
    base = Path(args.dir).resolve()
    scales = args.scales or ["0.25", "0.20", "0.15", "0.10"]
    print(f"{'h':>3} | " + " | ".join(f"s{s:>5}" for s in scales))
    print("-"*(6+9*len(scales)))
    cols = {}
    for s in scales:
        tag = s.replace(".", "")
        d = base/f"sweep24_s{tag}"
        kmh, w = profile(d/f"edgedata_s{tag}.xml")
        cols[s] = (kmh, w, totals(d/f"stats_s{tag}.xml"))
    for h in range(24):
        row = " | ".join(f"{cols[s][0][h]:6.1f}" for s in scales)
        print(f"{h:02d}h | {row}")
    print("\n# Presencia sampledSeconds/h (miles) — ¿forma o monótona?")
    for h in range(24):
        row = " | ".join(f"{cols[s][1][h]/1000:6.0f}" for s in scales)
        print(f"{h:02d}h | {row}")
    print("\n# Totales 24h + veredicto de drenaje")
    hdr = f"{'scale':>6} | {'ins':>6} | {'wait':>5} | {'jam':>6} | {'yld':>5} | {'wL':>5} | {'col/em':>6} | {'km/h':>5} | {'maxrun<8':>8} | {'pico_pres_h':>11} | drena?"
    print(hdr); print("-"*len(hdr))
    for s in scales:
        kmh, w, t = cols[s]
        drains, mx, peak_h = shape_verdict(kmh, w)
        print(f"{s:>6} | {t['inserted']:>6} | {t['waiting']:>5} | {t['jam']:>6} | "
              f"{t['yld']:>5} | {t['wl']:>5} | {t['col']:>2}/{t['em']:<3} | {t['kmh']:>5.1f} | "
              f"{mx:>8} | {peak_h:>9}h | {'SÍ' if drains else 'NO'}")


if __name__ == "__main__":
    main()
