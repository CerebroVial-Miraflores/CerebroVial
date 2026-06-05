"""
Seed inicial de datos reales de Miraflores para graph_nodes, graph_edges y cameras.
Idempotente: usa session.merge() — se puede correr múltiples veces sin duplicar filas.
"""

import math
import os
from datetime import datetime
from pathlib import Path

from geoalchemy2 import WKTElement
import bcrypt
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from cerebrovial_shared.database.models import CameraDB, GraphEdgeDB, GraphNodeDB, UserDB

# ID fijo para que seed sea idempotente (session.merge lo reutiliza)
_ADMIN_ID = "00000000-0000-0000-0000-000000000001"


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
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


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _point(lat: float, lon: float) -> WKTElement:
    return WKTElement(f"POINT({lon} {lat})", srid=4326)


def _line(lat1: float, lon1: float, lat2: float, lon2: float) -> WKTElement:
    return WKTElement(f"LINESTRING({lon1} {lat1}, {lon2} {lat2})", srid=4326)


_NODES_RAW = [
    ("larco_schell",         -12.119500, -77.029800, True),
    ("larco_benavides",      -12.122700, -77.030100, True),
    ("benavides_miraflores", -12.122900, -77.028300, False),
    ("arequipa_angamos",     -12.115800, -77.023900, True),
    ("ejercito_sucre",       -12.109900, -77.057800, True),
]

_COORDS = {nid: (lat, lon) for nid, lat, lon, _ in _NODES_RAW}


def _edge_data(edge_id: str, src: str, tgt: str) -> dict:
    lat1, lon1 = _COORDS[src]
    lat2, lon2 = _COORDS[tgt]
    return dict(
        edge_id=edge_id,
        source_node=src,
        target_node=tgt,
        distance_m=round(_haversine(lat1, lon1, lat2, lon2), 1),
        lanes=2,
        geom=_line(lat1, lon1, lat2, lon2),
    )


_EDGES_RAW = [
    ("larco_schell_to_benavides",    "larco_schell",         "larco_benavides"),
    ("larco_benavides_to_schell",    "larco_benavides",      "larco_schell"),
    ("larco_benavides_to_miraflores","larco_benavides",      "benavides_miraflores"),
    ("benavides_to_arequipa",        "benavides_miraflores", "arequipa_angamos"),
    ("arequipa_to_larco",            "arequipa_angamos",     "larco_schell"),
    ("ejercito_to_arequipa",         "ejercito_sucre",       "arequipa_angamos"),
]

def main() -> None:
    _load_dotenv()
    url = os.environ.get("DATABASE_URL", "").replace("@db:", "@localhost:")
    if not url:
        raise RuntimeError("DATABASE_URL no encontrado en el entorno ni en .env")

    engine = create_engine(url, pool_pre_ping=True)
    with Session(engine) as session:
        for nid, lat, lon, has_cam in _NODES_RAW:
            session.merge(GraphNodeDB(
                node_id=nid,
                lat=lat,
                lon=lon,
                has_camera=has_cam,
                geom=_point(lat, lon),
            ))

        for e in [_edge_data(*row) for row in _EDGES_RAW]:
            session.merge(GraphEdgeDB(
                edge_id=e["edge_id"],
                source_node=e["source_node"],
                target_node=e["target_node"],
                distance_m=e["distance_m"],
                lanes=e["lanes"],
                geom=e["geom"],
            ))

        # Las cámaras ya NO se siembran acá (Fase A): pasaron a ser accesorio de
        # intersección y se cargan, junto a las 11 intersecciones del PMU y el puente
        # intersection_edges, en scripts/seed_intersections.py (requiere el net real
        # cargado con build_graph_geometry.py primero).
        session.commit()

        # TODO: cambiar password antes de demo
        session.merge(UserDB(
            id=_ADMIN_ID,
            email="admin@cerebrovial.pe",
            password_hash=_hash_password("admin123"),
            role="admin",
            created_at=datetime.utcnow(),
        ))
        session.commit()

        n_nodes  = session.scalar(select(func.count()).select_from(GraphNodeDB))
        n_edges  = session.scalar(select(func.count()).select_from(GraphEdgeDB))
        n_cams   = session.scalar(select(func.count()).select_from(CameraDB))
        n_users  = session.scalar(select(func.count()).select_from(UserDB))
        print(f"Nodes: {n_nodes}, Edges: {n_edges}, Cameras: {n_cams}, Users: {n_users}")


if __name__ == "__main__":
    main()
