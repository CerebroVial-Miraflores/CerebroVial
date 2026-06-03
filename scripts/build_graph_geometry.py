"""
Builder de geometría de red para TTH-12 (Fase 1) — congestión por arista.

Puebla:
  - graph_nodes: los junctions SUMO del subgrafo de las 375 aristas del LCC,
    con node_id PREFIJADO ``sumo_<id>`` (p. ej. ``sumo_138854736``). El prefijo
    aísla este namespace del de los 5 nodos de CONTROL (``larco_schell``, etc.),
    que tienen colgando motor_decisions / engine_active_state / cameras y que
    este builder NO toca jamás.
  - graph_edges: las 375 aristas del LCC (edge_id == ``sumo_edge`` del mapping),
    source_node/target_node = ``sumo_<from>``/``sumo_<to>``, geometría LINESTRING.

Universo (375): leído del mapping canónico
``ia_prediction_service/src/data/artifacts/miraflores_graph_lcc_mapping.json``
(campo ``sumo_edge`` de cada entrada de ``nodes``). NO se usa el Parquet
(gitignored) ni los 1044 edges del .net.xml.

Geometría: extraída del .net.xml (atributo ``shape`` del edge; ``length`` y nº de
lanes de sus <lane>; coords x,y de los <junction>). Las coords SUMO son UTM zona
18 (SIN +south) desplazadas por ``netOffset``. La reproyección a WGS84 (4326) se
hace EN POSTGIS con ST_Transform y la cadena proj4 LITERAL del <location> — NO se
asume EPSG:32718 ni se inventa un SRID en spatial_ref_sys. Offset por RESTA:
``orig_utm = coord_red - netOffset``.

Idempotente: corrible N veces con el mismo resultado. Borra las graph_edges no-SUMO
(las 6 sintéticas) y las sumo_ previas, y solo los graph_nodes ``sumo_%``; reinserta.
Los nodos de control quedan intactos. Todo en una transacción (rollback total ante
cualquier fallo). NO push / PR / merge.

Uso:  .venv/bin/python scripts/build_graph_geometry.py
"""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from sqlalchemy import create_engine, text

_REPO = Path(__file__).resolve().parent.parent
_NET = _REPO / "simulation/conf/network/miraflores.net.xml"
_MAPPING = _REPO / "ia_prediction_service/src/data/artifacts/miraflores_graph_lcc_mapping.json"

EXPECTED_EDGES = 1660


class SpikeStop(RuntimeError):
    """Verificación intermedia falló: detener y reportar, no improvisar."""


def _load_dotenv() -> None:
    env_path = _REPO / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key not in os.environ:
                os.environ[key] = val


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL", "").replace("@db:", "@localhost:")
    if not url:
        raise SpikeStop("DATABASE_URL no encontrado en el entorno ni en .env")
    return url


def _read_location(net_path: Path) -> tuple[float, float, str]:
    """Lee netOffset (ox, oy) y projParameter del <location> del .net.xml."""
    with open(net_path) as f:
        for line in f:
            if "<location" in line:
                off = re.search(r'netOffset="([^"]+)"', line)
                proj = re.search(r'projParameter="([^"]+)"', line)
                if not off or not proj:
                    raise SpikeStop("location sin netOffset o projParameter")
                ox, oy = (float(v) for v in off.group(1).split(","))
                return ox, oy, proj.group(1)
    raise SpikeStop("no se encontró <location> en el .net.xml")


def _parse_net(net_path: Path, universe: set[str]) -> tuple[dict, dict]:
    """Una pasada: edges del universo (from/to/shape/length/numlanes) + coords de junctions."""
    edges: dict[str, dict] = {}
    junctions: dict[str, tuple[float, float]] = {}
    for _ev, el in ET.iterparse(str(net_path), events=("end",)):
        if el.tag == "edge":
            eid = el.get("id")
            if eid in universe:
                lanes = el.findall("lane")
                shape = el.get("shape") or (lanes[0].get("shape") if lanes else None)
                length = None
                if lanes and lanes[0].get("length") is not None:
                    length = float(lanes[0].get("length"))
                edges[eid] = {
                    "frm": el.get("from"),
                    "to": el.get("to"),
                    "shape": shape,
                    "length": length,
                    "numlanes": len(lanes),
                }
            el.clear()
        elif el.tag == "junction":
            jid = el.get("id")
            if jid is not None and el.get("x") is not None:
                junctions[jid] = (float(el.get("x")), float(el.get("y")))
            el.clear()
    return edges, junctions


def main() -> None:
    _load_dotenv()

    # --- PASO 1: universo de 375 aristas del mapping canónico ---
    mapping = json.loads(_MAPPING.read_text())
    universe = [n["sumo_edge"] for n in mapping["nodes"]]
    if len(universe) != EXPECTED_EDGES or len(set(universe)) != EXPECTED_EDGES:
        raise SpikeStop(
            f"mapping no da {EXPECTED_EDGES} sumo_edge únicos: "
            f"len={len(universe)} unique={len(set(universe))}"
        )
    universe_set = set(universe)
    print(f"[1] universo del mapping: {len(universe)} sumo_edge (únicos {len(set(universe))})")

    # --- PASO 2: geometría/atributos del .net.xml ---
    ox, oy, proj4 = _read_location(_NET)
    print(f"[loc] netOffset=({ox},{oy})  proj4={proj4!r}")
    edges, junctions = _parse_net(_NET, universe_set)

    missing_edges = sorted(e for e in universe if e not in edges)
    if missing_edges:
        raise SpikeStop(f"{len(missing_edges)} edges del mapping ausentes en .net.xml: {missing_edges[:10]}")
    no_shape = sorted(e for e in universe if not edges[e]["shape"])
    if no_shape:
        raise SpikeStop(f"{len(no_shape)} edges sin shape: {no_shape[:10]}")
    no_len = sorted(e for e in universe if edges[e]["length"] is None)
    if no_len:
        raise SpikeStop(f"{len(no_len)} edges sin length (distance_m NOT NULL): {no_len[:10]}")
    print("[2] 375 edges resueltos en el .net.xml (shape + length + lanes)")

    # --- PASO 3: subgrafo de nodos (junctions) ---
    node_ids: set[str] = set()
    for e in universe:
        node_ids.add(edges[e]["frm"])
        node_ids.add(edges[e]["to"])
    internal = sorted(j for j in node_ids if j.startswith(":"))
    if internal:
        raise SpikeStop(f"from/to internos (':') inesperados: {internal[:10]}")
    missing_j = sorted(j for j in node_ids if j not in junctions)
    if missing_j:
        raise SpikeStop(f"{len(missing_j)} junctions referenciados ausentes en .net.xml: {missing_j[:10]}")
    N = len(node_ids)
    print(f"[3] subgrafo de nodos: {N} junctions (todos presentes como <junction>)")

    # --- PASO 4: WKT en UTM (offset por resta) para reproyectar en PostGIS ---
    def to_utm(x: float, y: float) -> tuple[float, float]:
        return (x - ox, y - oy)

    node_rows = []
    for j in sorted(node_ids):
        x, y = junctions[j]
        ux, uy = to_utm(x, y)
        node_rows.append((f"sumo_{j}", f"POINT({ux} {uy})"))

    edge_rows = []
    for e in universe:
        d = edges[e]
        pts = [tuple(map(float, p.split(","))) for p in d["shape"].split()]
        upts = [to_utm(x, y) for x, y in pts]
        wkt = "LINESTRING(" + ", ".join(f"{x} {y}" for x, y in upts) + ")"
        edge_rows.append(
            (e, f"sumo_{d['frm']}", f"sumo_{d['to']}", round(d["length"], 1), d["numlanes"], wkt)
        )

    # --- PASO 5: escritura idempotente en transacción, en orden por FK ---
    engine = create_engine(_db_url(), pool_pre_ping=True)
    with engine.begin() as conn:
        # No tocamos waze_jams (Fase 2) ni waze_alerts; pero como vamos a borrar
        # graph_edges, exigimos que no haya filas colgando de ellas.
        wj = conn.execute(text("SELECT count(*) FROM waze_jams")).scalar_one()
        wa = conn.execute(text("SELECT count(*) FROM waze_alerts")).scalar_one()
        if wj != 0 or wa != 0:
            raise SpikeStop(f"waze_jams={wj} waze_alerts={wa} (debían ser 0): abortando")

        # a. graph_edges: namespace exclusivo de TTH-12 (0 FK entrantes con filas).
        #    Borra las 6 sintéticas y las sumo_ previas; reinserta abajo.
        deleted_edges = conn.execute(text("DELETE FROM graph_edges")).rowcount
        # b. graph_nodes: SOLO los sumo_ (los de control NUNCA se tocan).
        deleted_nodes = conn.execute(
            text("DELETE FROM graph_nodes WHERE node_id ~ '^sumo_'")
        ).rowcount
        print(f"[5a] borrados: graph_edges={deleted_edges}, graph_nodes(sumo_)={deleted_nodes}")

        ins_node = text(
            "INSERT INTO graph_nodes (node_id, lat, lon, has_camera, geom) "
            "SELECT :nid, ST_Y(g), ST_X(g), false, g "
            "FROM (SELECT ST_Transform(ST_GeomFromText(:wkt, 0), :proj4, 4326) AS g) s"
        )
        for nid, wkt in node_rows:
            conn.execute(ins_node, {"nid": nid, "wkt": wkt, "proj4": proj4})

        ins_edge = text(
            "INSERT INTO graph_edges (edge_id, source_node, target_node, distance_m, lanes, geom) "
            "VALUES (:eid, :src, :tgt, :dist, :lanes, "
            "ST_Transform(ST_GeomFromText(:wkt, 0), :proj4, 4326))"
        )
        for eid, src, tgt, dist, lanes, wkt in edge_rows:
            conn.execute(
                ins_edge,
                {"eid": eid, "src": src, "tgt": tgt, "dist": dist, "lanes": lanes,
                 "wkt": wkt, "proj4": proj4},
            )
        print(f"[5b] insertados: graph_nodes(sumo_)={len(node_rows)}, graph_edges={len(edge_rows)}")

    print("OK: carga TTH-12 Fase 1 completa.")


if __name__ == "__main__":
    main()
