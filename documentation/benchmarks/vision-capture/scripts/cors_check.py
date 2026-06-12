"""Eje 3 — ¿la URL HLS de Claro es reproducible directo desde el browser, o requiere proxy del edge?
Hace un GET con cabecera Origin cruzado y reporta las cabeceras CORS de la respuesta del
master playlist y de un segmento .ts. Sin CORS permisivo, un <video>/fetch desde el browser
(origen del frontend) sería bloqueado y la vista debe servirse por el edge.

Uso: python cors_check.py     Escribe: data/eje3_cors.csv
"""
import csv, datetime, os, sys, urllib.request
from urllib.parse import urljoin

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
ORIG = "https://live.smartechlatam.online/claro/escuela_pnp/index.m3u8"
FRONTEND_ORIGIN = "http://localhost:5173"  # origen real del frontend (Vite/nginx)
CORS_HDRS = ["access-control-allow-origin","access-control-allow-methods",
             "access-control-allow-headers","access-control-allow-credentials",
             "access-control-expose-headers"]

def probe(u, label):
    req = urllib.request.Request(u, headers={
        "Origin": FRONTEND_ORIGIN,
        "Referer": "https://claro.com.pe/",
        "User-Agent": "Mozilla/5.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            h = {k.lower(): v for k, v in r.headers.items()}
            status = r.status
    except Exception as e:
        return {"label": label, "status": f"ERR {type(e).__name__}: {e}",
                "cors": {k: None for k in CORS_HDRS}, "content_type": None}
    return {"label": label, "status": status,
            "cors": {k: h.get(k) for k in CORS_HDRS},
            "content_type": h.get("content-type")}

def main():
    body = urllib.request.urlopen(urllib.request.Request(
        ORIG, headers={"Referer":"https://claro.com.pe/","User-Agent":"Mozilla/5.0"}), timeout=15
        ).read().decode("utf-8","replace")
    seg = next((l.strip() for l in body.splitlines() if l.strip().endswith(".ts")), None)
    targets = [(ORIG, "master_m3u8")]
    if seg: targets.append((urljoin(ORIG, seg), "segment_ts"))

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows = []
    for u, label in targets:
        r = probe(u, label)
        acao = r["cors"]["access-control-allow-origin"]
        cross_ok = acao in ("*", FRONTEND_ORIGIN)
        rows.append([ts, label, FRONTEND_ORIGIN, r["status"], r["content_type"],
                     acao, r["cors"]["access-control-allow-methods"],
                     r["cors"]["access-control-allow-credentials"], cross_ok])
        print(f"{label}: status={r['status']} ACAO={acao!r} cross_origin_playable={cross_ok}", flush=True)

    new = not os.path.exists(os.path.join(DATA, "eje3_cors.csv"))
    with open(os.path.join(DATA, "eje3_cors.csv"), "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["timestamp_utc","target","origin_sent","http_status","content_type",
                            "access_control_allow_origin","access_control_allow_methods",
                            "access_control_allow_credentials","cross_origin_playable"])
        w.writerows(rows)

if __name__ == "__main__":
    main()
