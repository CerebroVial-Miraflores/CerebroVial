"""C2/dataset — compacta un edgeData XML (24h, freq=60) a Parquet liviano.

Corre con el python del .venv del proyecto (pandas + pyarrow), igual que el
dataset builder downstream (F3).

Selección de columnas (8): edge_id, timestep, speed, timeLoss, traveltime,
flow, density, speedRelative.

Scope de edges: los 1664 edges vehiculares (passenger) no-internal. En el net v2
todos los no-internal resultan vehiculares (1664/1664), así que el filtro de clase
vial no descarta nada hoy; se mantiene por robustez (en v1 separaba 381 vehiculares
de 1044 no-internal que incluían peatonales/cruces, ruido siempre-vacío para un
dataset de congestión). Se filtra por la misma lógica de clase vial que la
generación de demanda.

Convención de edge VACÍO en un intervalo (sampledSeconds=0 → SUMO no emite las
métricas): flow=0, density=0, timeLoss=0 (exactos: sin vehículos no hay flujo,
densidad ni demora); speed=NaN, speedRelative=NaN, traveltime=NaN (indefinidos:
no hay vehículos que promediar). Presencia/vacío se distingue por flow==0 &
density==0; NUNCA se mapea speed ausente→0 (eso simularía jam máximo).

Uso: python compact_day.py <edgedata.xml> <out.parquet> <net.xml>
"""
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

METRICS = ["speed", "timeLoss", "traveltime", "flow", "density", "speedRelative"]
# Para edge vacío: ceros exactos vs NaN (indefinido).
EMPTY_FILL = {"flow": 0.0, "density": 0.0, "timeLoss": 0.0,
              "speed": np.nan, "speedRelative": np.nan, "traveltime": np.nan}

ARTERIAL_TYPES = {
    "highway.motorway", "highway.motorway_link",
    "highway.primary", "highway.primary_link",
    "highway.secondary", "highway.secondary_link",
}


def lane_allows_passenger(lane):
    allow = lane.get("allow")
    dis = lane.get("disallow")
    if allow is not None:
        return "passenger" in allow.split()
    if dis is not None:
        return "passenger" not in dis.split()
    return True


def vehicular_edges(net_path):
    root = ET.parse(net_path).getroot()
    ids = set()
    for e in root.findall("edge"):
        if e.get("function") == "internal":
            continue
        if any(lane_allows_passenger(l) for l in e.findall("lane")):
            ids.add(e.get("id"))
    return ids


def compact(xml_path, out_path, net_path):
    keep = vehicular_edges(net_path)
    edge_ids = []
    ts = []
    cols = {m: [] for m in METRICS}
    cur_begin = 0
    ctx = ET.iterparse(xml_path, events=("start", "end"))
    for ev, el in ctx:
        if ev == "start" and el.tag == "interval":
            cur_begin = int(float(el.get("begin")))
        elif ev == "end" and el.tag == "edge":
            eid = el.get("id")
            if eid in keep:
                edge_ids.append(eid)
                ts.append(cur_begin)
                empty = float(el.get("sampledSeconds", "0") or 0) == 0.0
                for m in METRICS:
                    v = el.get(m)
                    cols[m].append(EMPTY_FILL[m] if (empty or v is None) else float(v))
            el.clear()
        elif ev == "end" and el.tag == "interval":
            el.clear()

    df = pd.DataFrame({
        "edge_id": pd.array(edge_ids, dtype="string"),
        "timestep": np.asarray(ts, dtype=np.int32),
        **{m: np.asarray(cols[m], dtype=np.float32) for m in METRICS},
    })[["edge_id", "timestep"] + METRICS]
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    return df


if __name__ == "__main__":
    xml_path, out_path, net_path = sys.argv[1], sys.argv[2], sys.argv[3]
    df = compact(xml_path, out_path, net_path)
    n_edges = df["edge_id"].nunique()
    n_ts = df["timestep"].nunique()
    print(f"OK {out_path}")
    print(f"  filas={len(df)}  edges={n_edges}  timesteps={n_ts}  (esperado {n_edges*n_ts})")
    print(f"  columnas={list(df.columns)}")
