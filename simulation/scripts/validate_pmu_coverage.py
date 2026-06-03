"""Validación de cobertura PMU de un net SUMO contra el gazetteer de Miraflores (B1).

Parametrizado (``--net``, ``--out``): reusa la lógica geométrica del mapeo de Fase 0
(``build_pmu_edge_mapping.py``, rama chore/net-coverage-diagnosis) pero aplicable a
CUALQUIER net — en particular al net nuevo ``miraflores_v2.net.xml``, sin tocar el viejo.

Para cada intersección del gazetteer PMU (coords WGS84 verificadas a mano) busca la junction
``type="traffic_light"`` más cercana y mide la distancia en metros (UTM, sin haversine).
Criterio de gate: intersecciones VALIDABLES (<=40 m a un semáforo) >= 11/12.

Además reporta MÉTRICAS del net (nodos / traffic_light / edges vehiculares / convBoundary)
y una ESTIMACIÓN del LCC-N (tamaño del componente conexo mayor del grafo de aristas
vehiculares) — NO es el N autoritativo (eso lo produce B3 vía
``ia_prediction_service/.../miraflores_graph_builder.py``); es una estimación con método
explícito (BFS sobre ``edge.getOutgoing()`` = regla "to==from ∩ <connection>"). El número
``dropped`` (nodos fuera del LCC) es SEÑAL DE SALUD: muchos dropped = fragmentos sueltos.

Con ``--compare <net_viejo>`` imprime las métricas y la cobertura PMU lado a lado.

Uso (desde simulation/):
  .venv/bin/python scripts/validate_pmu_coverage.py \
      --net conf/network/miraflores_v2.net.xml \
      --out ../documentation/contracts/mapeo_pmu_edges_v2.yaml \
      --compare conf/network/miraflores.net.xml
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import sumolib
import yaml


# Umbrales de estado (metros al semáforo más cercano) — idénticos a Fase 0.
OK_MAX_M = 40.0       # <= ok (validable)
DUDOSA_MAX_M = 80.0   # > 80 m => dudosa por lejanía

# Gazetteer PMU (lat/long WGS84 verificadas manualmente). Idéntico a build_pmu_edge_mapping.py.
GAZETTEER = [
    {"nombre": "larco_benavides",    "lat": -12.124540, "lon": -77.029332, "nivel_los": "C/D", "nota": None},
    {"nombre": "ovalo_miraflores",   "lat": -12.119287, "lon": -77.029081, "nivel_los": None,  "nota": "Larco/Pardo/Arequipa, núcleo", "revisar": True},
    {"nombre": "pardo_espinar",      "lat": -12.118978, "lon": -77.037054, "nivel_los": "F/H", "nota": "crítico"},
    {"nombre": "arequipa_angamos",   "lat": -12.113838, "lon": -77.030074, "nivel_los": "E/H", "nota": None},
    {"nombre": "espinar_angamos",    "lat": -12.113999, "lon": -77.036845, "nivel_los": "F/F", "nota": None},
    {"nombre": "ovalo_gutierrez",    "lat": -12.112000, "lon": -77.037000, "nivel_los": None,  "nota": "Espinar/Santa Cruz; óvalo (coord aprox. corregida a mano)", "aprox": True},
    {"nombre": "28julio_reducto",    "lat": -12.129166, "lon": -77.023688, "nivel_los": "E/H", "nota": None},
    {"nombre": "28julio_lapaz",      "lat": -12.127884, "lon": -77.027016, "nivel_los": "D/F", "nota": None},
    {"nombre": "ricardopalma_paseo", "lat": -12.119221, "lon": -77.025545, "nivel_los": "D/F", "nota": None},
    {"nombre": "paseo_angamos",      "lat": -12.113408, "lon": -77.025577, "nivel_los": "E/H", "nota": "muy saturada"},
    {"nombre": "paseo_benavides",    "lat": -12.125343, "lon": -77.023909, "nivel_los": "D/E", "nota": None},
    {"nombre": "benavides_panama",   "lat": -12.126337, "lon": -77.018081, "nivel_los": None,  "nota": "al filo este", "filo": True},
]


def _round(x: float, n: int = 6) -> float:
    return round(float(x), n)


def load_tls_nodes(net) -> list:
    return [n for n in net.getNodes() if n.getType() == "traffic_light"]


def tls_bbox_lonlat(net, tls) -> list:
    """bbox real lon/lat de los nodos semáforo (footprint de la red de interés)."""
    lons, lats = [], []
    for n in tls:
        x, y = n.getCoord()
        lon, lat = net.convertXY2LonLat(x, y)
        lons.append(lon)
        lats.append(lat)
    return [min(lons), min(lats), max(lons), max(lats)]


def nearest_tls(net, tls, lon: float, lat: float):
    """Top-3 (node, distancia_m) por distancia euclídea en metros UTM."""
    gx, gy = net.convertLonLat2XY(lon, lat)
    ranked = [(n, math.hypot(gx - n.getCoord()[0], gy - n.getCoord()[1])) for n in tls]
    ranked.sort(key=lambda t: t[1])
    return ranked[:3]


def edges_of(node) -> tuple[list[str], list[str]]:
    inc = sorted(e.getID() for e in node.getIncoming())
    out = sorted(e.getID() for e in node.getOutgoing())
    return inc, out


def classify(dist_m: float, in_bbox: bool, aprox: bool) -> str:
    if not in_bbox and dist_m > DUDOSA_MAX_M:
        return "fuera"
    if aprox or dist_m > DUDOSA_MAX_M or dist_m > OK_MAX_M:
        return "dudosa"
    return "ok"


def map_gazetteer(net) -> dict:
    """Mapea cada intersección PMU a su TLS más cercano en `net`."""
    tls = load_tls_nodes(net)
    if not tls:
        return {"bbox": [0, 0, 0, 0], "entries": [], "n_tls": 0}
    bbox = tls_bbox_lonlat(net, tls)
    lon_min, lat_min, lon_max, lat_max = bbox
    entries = []
    for g in GAZETTEER:
        lon, lat = g["lon"], g["lat"]
        in_bbox = (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)
        top3 = nearest_tls(net, tls, lon, lat)
        chosen, dist = top3[0]
        aprox = bool(g.get("aprox"))
        estado = classify(dist, in_bbox, aprox)
        inc, out = edges_of(chosen)
        revisar = estado != "ok" or aprox or bool(g.get("revisar")) or bool(g.get("filo"))
        candidatos = (
            [{"junction_id": n.getID(), "distancia_m": _round(d, 2)} for n, d in top3]
            if revisar else None
        )
        entries.append({
            "g": g, "lon": lon, "lat": lat, "in_bbox": in_bbox,
            "junction_id": chosen.getID(), "distancia_m": _round(dist, 2),
            "inc": inc, "out": out, "estado": estado, "candidatos": candidatos,
        })
    return {"bbox": bbox, "entries": entries, "n_tls": len(tls)}


# --- Métricas + estimación LCC ---------------------------------------------

def _vehicular_edges(net) -> list:
    """Edges no-internas que permiten 'passenger' en algún lane (regla del builder canónico)."""
    out = []
    for e in net.getEdges():
        if e.isSpecial():  # internas ':'
            continue
        if any(lane.allows("passenger") for lane in e.getLanes()):
            out.append(e)
    return out


def lcc_estimate(net) -> dict:
    """Estimación del LCC-N: BFS no-dirigido sobre el grafo de aristas vehiculares.

    Adyacencia = edge.getOutgoing() (downstream por <connection>, equivale a la regla
    'to==from ∩ <connection>' del builder canónico). NO es el N autoritativo (B3).
    """
    veh = _vehicular_edges(net)
    veh_ids = {e.getID() for e in veh}
    idx = {eid: i for i, eid in enumerate(sorted(veh_ids))}
    adj: dict[int, set[int]] = {i: set() for i in range(len(idx))}
    for e in veh:
        s = idx[e.getID()]
        for nb in e.getOutgoing():  # dict {edge: [connections]}
            nid = nb.getID()
            if nid in idx:
                adj[s].add(idx[nid])
                adj[idx[nid]].add(s)  # no-dirigido (conectividad)
    seen: set[int] = set()
    comps: list[int] = []
    for start in range(len(idx)):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        size = 0
        while q:
            u = q.popleft()
            size += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(size)
    comps.sort(reverse=True)
    n_full = len(idx)
    lcc_n = comps[0] if comps else 0
    return {
        "full_graph_nodes": n_full,
        "lcc_n_estimado": lcc_n,
        "dropped_fuera_lcc": n_full - lcc_n,
        "n_componentes": len(comps),
        "top5_componentes": comps[:5],
    }


def net_metrics(net) -> dict:
    nodes = net.getNodes()
    tls = load_tls_nodes(net)
    veh = _vehicular_edges(net)
    bx = net.getBBoxXY()  # ((xmin,ymin),(xmax,ymax))
    (xmin, ymin), (xmax, ymax) = bx
    m = {
        "n_nodos": len(nodes),
        "n_traffic_light": len(tls),
        "n_edges_total_no_internas": len([e for e in net.getEdges() if not e.isSpecial()]),
        "n_edges_vehiculares": len(veh),
        "convBoundary_xy": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
        "ancho_m": round(xmax - xmin, 1),
        "alto_m": round(ymax - ymin, 1),
    }
    m.update(lcc_estimate(net))
    return m


# --- Salida -----------------------------------------------------------------

def to_yaml_doc(net_path: Path, result: dict, metrics: dict) -> dict:
    lon_min, lat_min, lon_max, lat_max = result["bbox"]
    validables = sum(1 for e in result["entries"] if e["estado"] == "ok")
    doc = {
        "meta": {
            "fase": "B1 — validación de cobertura PMU del net reconstruido (geométrica)",
            "fuente_pmu": "Plan de Movilidad Urbana de Miraflores 2017-2020",
            "red": str(net_path),
            "proyeccion": "+proj=utm +zone=18 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
            "tls_bbox_lonlat": {
                "lon_min": _round(lon_min), "lat_min": _round(lat_min),
                "lon_max": _round(lon_max), "lat_max": _round(lat_max),
            },
            "umbral_ok_m": OK_MAX_M,
            "umbral_dudosa_m": DUDOSA_MAX_M,
            "n_traffic_light": result["n_tls"],
            "intersecciones_validables": validables,
            "criterio_gate": ">= 11/12 validables (<=40 m a un traffic_light)",
            "gate_pasa": validables >= 11,
            "metricas_net": metrics,
            "nota_lcc": (
                "lcc_n_estimado es una ESTIMACIÓN (BFS sobre edge.getOutgoing()), NO el N "
                "autoritativo: B3 lo produce vía miraflores_graph_builder.build_miraflores_graph. "
                "dropped_fuera_lcc es señal de salud (fragmentos desconectados)."
            ),
        },
        "intersecciones": [],
    }
    for e in result["entries"]:
        g = e["g"]
        item = {
            "nombre": g["nombre"],
            "coord_gazetteer": {"lat": g["lat"], "lon": g["lon"]},
            "pmu": {"nivel_los": g["nivel_los"], "nota": g["nota"]},
            "junction_id": e["junction_id"],
            "distancia_m": e["distancia_m"],
            "in_bbox": e["in_bbox"],
            "edges_incoming": e["inc"],
            "edges_outgoing": e["out"],
            "n_edges": len(e["inc"]) + len(e["out"]),
            "estado": e["estado"],
        }
        if e["candidatos"] is not None:
            item["candidatos_top3"] = e["candidatos"]
        doc["intersecciones"].append(item)
    return doc


def print_metrics_table(label_a: str, ma: dict, label_b: str | None, mb: dict | None) -> None:
    rows = [
        ("n_nodos", "n_nodos"),
        ("n_traffic_light", "n_traffic_light"),
        ("n_edges_vehiculares", "n_edges_vehiculares"),
        ("ancho_m x alto_m", None),
        ("LCC-N (estimado)", "lcc_n_estimado"),
        ("dropped (fuera LCC)", "dropped_fuera_lcc"),
        ("full_graph_nodes", "full_graph_nodes"),
    ]
    print("\n=== MÉTRICAS DEL NET ===")
    if mb is not None:
        print(f"{'métrica':<24} {label_a:>16} {label_b:>16}")
        print("-" * 60)
    else:
        print(f"{'métrica':<24} {label_a:>16}")
        print("-" * 44)
    for disp, key in rows:
        if key is None:
            va = f"{ma['ancho_m']:.0f} x {ma['alto_m']:.0f}"
            vb = f"{mb['ancho_m']:.0f} x {mb['alto_m']:.0f}" if mb else None
        else:
            va = str(ma[key]); vb = str(mb[key]) if mb else None
        if mb is not None:
            print(f"{disp:<24} {va:>16} {vb:>16}")
        else:
            print(f"{disp:<24} {va:>16}")


def print_pmu_table(label: str, result: dict) -> int:
    print(f"\n=== COBERTURA PMU — {label} ===")
    hdr = f"{'interseccion':<20} {'junction_id':<14} {'dist(m)':>9}  estado"
    print(hdr); print("-" * len(hdr))
    validables = 0
    for e in result["entries"]:
        if e["estado"] == "ok":
            validables += 1
        print(f"{e['g']['nombre']:<20} {e['junction_id']:<14} {e['distancia_m']:>9.2f}  {e['estado']}")
    print(f"VALIDABLES (<=40 m): {validables}/12")
    return validables


def main() -> int:
    ap = argparse.ArgumentParser(description="Validación cobertura PMU de un net SUMO.")
    ap.add_argument("--net", required=True, type=Path, help="net a validar (p.ej. miraflores_v2.net.xml)")
    ap.add_argument("--out", required=True, type=Path, help="YAML de salida")
    ap.add_argument("--compare", type=Path, default=None, help="net viejo para comparar lado a lado")
    args = ap.parse_args()

    if not args.net.exists():
        print(f"ERROR: no existe {args.net}", file=sys.stderr)
        return 1

    net = sumolib.net.readNet(str(args.net))
    result = map_gazetteer(net)
    metrics = net_metrics(net)

    mb = None
    label_b = None
    result_b = None
    if args.compare and args.compare.exists():
        net_b = sumolib.net.readNet(str(args.compare))
        result_b = map_gazetteer(net_b)
        mb = net_metrics(net_b)
        label_b = args.compare.name

    # Reporte de gate
    print_metrics_table(args.net.name, metrics, label_b, mb)
    v_new = print_pmu_table(args.net.name, result)
    if result_b is not None:
        print_pmu_table(label_b, result_b)

    print("\n=== GATE ===")
    print(f"  {args.net.name}: {v_new}/12 validables  "
          f"({'PASA' if v_new >= 11 else 'NO PASA'} el criterio >=11/12)")
    if result_b is not None:
        v_old = sum(1 for e in result_b["entries"] if e["estado"] == "ok")
        print(f"  {label_b}: {v_old}/12 validables (referencia)")

    # Artefacto YAML
    doc = to_yaml_doc(args.net, result, metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    print(f"\n→ artefacto escrito: {args.out}")
    return 0 if v_new >= 11 else 2


if __name__ == "__main__":
    sys.exit(main())
