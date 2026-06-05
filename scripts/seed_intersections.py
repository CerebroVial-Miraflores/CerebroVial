"""
Seed de las intersecciones semaforizadas del PMU de Miraflores (Fase A).

Puebla, desde el mapeo canónico
``documentation/contracts/mapeo_pmu_edges_v2.yaml``:

  - ``intersections``: las 11 intersecciones SEMAFORIZADAS (excluye ``ovalo_gutierrez``,
    rotonda sin TLS), con junction_id, lat/lon de coord_gazetteer, geom (POINT 4326) y
    los_pmu. tls_id se puebla SOLO para ``larco_benavides`` (= su junction_id, verificado
    contra simulation/.../corredor_adaptive.py); el resto queda NULL (DEUDA-CTRL-TLS).
  - ``intersection_edges``: el puente intersección → graph_edges, con dirección
    (incoming/outgoing), para todos los edges_incoming/outgoing de cada intersección.
  - ``cameras``: 1 cámara por intersección, con stream_url de Claro asignado 1:1 de forma
    arbitraria (asociación nominal — DEUDA-CAM-GEO).

ORDEN DE CARGA (importante): este seed asume el net real cargado. Los edge_id del mapeo
son ids SUMO crudos (p. ej. ``129466113#3``) y ``intersection_edges.edge_id`` es FK a
``graph_edges.edge_id``, que puebla ``scripts/build_graph_geometry.py`` (NO ``invoke seed``).
Secuencia:  invoke seed  →  build_graph_geometry.py  →  invoke seed-intersections.
Pre-check fail-fast: si algún edge del mapeo no está en graph_edges, aborta con mensaje
claro en vez de un error críptico de FK.

Idempotente: usa session.merge(). NO push / PR / merge.

Uso:  .venv/bin/python scripts/seed_intersections.py
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from geoalchemy2 import WKTElement
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from cerebrovial_shared.database.models import (
    CameraDB,
    GraphEdgeDB,
    IntersectionDB,
    IntersectionEdgeDB,
)

_REPO = Path(__file__).resolve().parent.parent
_MAPEO = _REPO / "documentation/contracts/mapeo_pmu_edges_v2.yaml"

# Rotonda sin semáforo: fuera del mapeo PMU por definición (nivel_los=None).
_EXCLUDE = {"ovalo_gutierrez"}

# tls_id verificado contra una config de control real. Solo larco_benavides hoy:
# su tls_id SUMO es idéntico a su junction_id (mapeo + corredor_adaptive.py). El resto
# queda NULL (DEUDA-CTRL-TLS): no metemos a la base un tls_id sin verificar.
_TLS_VERIFIED = {"larco_benavides"}

# URLs HLS de Claro vivas. Asignación 1:1 ARBITRARIA a las 11 intersecciones, por orden
# del mapeo (deuda DEUDA-CAM-GEO: falta concordancia geográfica real cámara↔intersección).
_CLARO_STREAMS = [
    "escuela_pnp",
    "gallinazos",
    "prolongaciontacna",
    "lamarina",
    "avfaustinocarrion",
    "paseodelarepublica",
    "angamos",
    "panamericana",
    "javierprado",
    "derby",
    "panamericana_peaje1",
]
_CLARO_BASE = "https://live.smartechlatam.online/claro/{name}/index.m3u8"


class SeedStop(RuntimeError):
    """Pre-check falló: detener y reportar, no improvisar."""


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
        raise SeedStop("DATABASE_URL no encontrado en el entorno ni en .env")
    return url


def _point(lat: float, lon: float) -> WKTElement:
    return WKTElement(f"POINT({lon} {lat})", srid=4326)


def load_mapeo() -> dict:
    return yaml.safe_load(_MAPEO.read_text())


def build_rows(mapeo: dict) -> dict:
    """Función pura: del mapeo a filas de intersections / intersection_edges / cameras.

    Testeable sin BD. Excluye ovalo_gutierrez; ordena la asignación de cámaras Claro por
    el orden de aparición de las 11 intersecciones semaforizadas en el mapeo.
    """
    seleccionadas = [
        it for it in mapeo["intersecciones"] if it["nombre"] not in _EXCLUDE
    ]
    if len(seleccionadas) != len(_CLARO_STREAMS):
        raise SeedStop(
            f"esperaba {len(_CLARO_STREAMS)} intersecciones semaforizadas, "
            f"el mapeo dio {len(seleccionadas)}"
        )

    intersections: list[dict] = []
    edges: list[dict] = []
    cameras: list[dict] = []

    for idx, it in enumerate(seleccionadas):
        iid = it["nombre"]
        lat = float(it["coord_gazetteer"]["lat"])
        lon = float(it["coord_gazetteer"]["lon"])
        junction_id = str(it["junction_id"])

        intersections.append(
            {
                "intersection_id": iid,
                "junction_id": junction_id,
                "lat": lat,
                "lon": lon,
                "los_pmu": it.get("pmu", {}).get("nivel_los"),
                "tls_id": junction_id if iid in _TLS_VERIFIED else None,
            }
        )

        for direction, key in (("incoming", "edges_incoming"), ("outgoing", "edges_outgoing")):
            for edge_id in it.get(key) or []:
                edges.append(
                    {
                        "intersection_id": iid,
                        "edge_id": str(edge_id),
                        "direction": direction,
                    }
                )

        cameras.append(
            {
                "camera_id": f"cam_{iid}",
                "intersection_id": iid,
                "stream_url": _CLARO_BASE.format(name=_CLARO_STREAMS[idx]),
                "lat": lat,
                "lon": lon,
            }
        )

    return {"intersections": intersections, "edges": edges, "cameras": cameras}


def _precheck_edges_present(session: Session, edges: list[dict]) -> None:
    """Fail-fast: todo edge_id del mapeo debe existir en graph_edges (FK satisfacible)."""
    wanted = {e["edge_id"] for e in edges}
    present = set(
        session.execute(
            select(GraphEdgeDB.edge_id).where(GraphEdgeDB.edge_id.in_(wanted))
        ).scalars()
    )
    missing = sorted(wanted - present)
    if missing:
        raise SeedStop(
            f"{len(missing)} edges del mapeo NO están en graph_edges "
            f"(ej. {missing[:5]}). El net real no está cargado: corré primero\n"
            f"  .venv/bin/python scripts/build_graph_geometry.py\n"
            f"y después este seed (orden: invoke seed → build_graph_geometry.py "
            f"→ invoke seed-intersections)."
        )


def main() -> None:
    _load_dotenv()
    rows = build_rows(load_mapeo())

    engine = create_engine(_db_url(), pool_pre_ping=True)
    with Session(engine) as session:
        _precheck_edges_present(session, rows["edges"])

        for it in rows["intersections"]:
            session.merge(
                IntersectionDB(
                    intersection_id=it["intersection_id"],
                    junction_id=it["junction_id"],
                    lat=it["lat"],
                    lon=it["lon"],
                    los_pmu=it["los_pmu"],
                    tls_id=it["tls_id"],
                    geom=_point(it["lat"], it["lon"]),
                )
            )

        for e in rows["edges"]:
            session.merge(
                IntersectionEdgeDB(
                    intersection_id=e["intersection_id"],
                    edge_id=e["edge_id"],
                    direction=e["direction"],
                )
            )

        for c in rows["cameras"]:
            session.merge(
                CameraDB(
                    camera_id=c["camera_id"],
                    intersection_id=c["intersection_id"],
                    stream_url=c["stream_url"],
                    lat=c["lat"],
                    lon=c["lon"],
                    heading=0.0,
                    fov=90.0,
                    geom=_point(c["lat"], c["lon"]),
                )
            )

        session.commit()

        n_int = session.scalar(select(func.count()).select_from(IntersectionDB))
        n_bridge = session.scalar(select(func.count()).select_from(IntersectionEdgeDB))
        n_cams = session.scalar(select(func.count()).select_from(CameraDB))
        n_tls = session.scalar(
            select(func.count()).select_from(IntersectionDB).where(
                IntersectionDB.tls_id.isnot(None)
            )
        )
        print(
            f"Intersections: {n_int}, intersection_edges: {n_bridge}, "
            f"cameras: {n_cams}, con tls_id: {n_tls}"
        )


if __name__ == "__main__":
    main()
