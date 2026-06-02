"""Verificación de alineación de edge_id end-to-end (TTH-12, CT-12.8 / CT-12.9 g).

Confirma que los TRES conjuntos de edge_id coinciden EXACTAMENTE:
  1. geometría poblada   → graph_edges (Fase 1)
  2. feed de congestión  → waze_jams DISTINCT edge_id (Fase 2)
  3. mapping canónico    → ia_prediction_service/.../miraflores_graph_lcc_mapping.json (sumo_edge)

Una arista coloreada en el mapa corresponde inequívocamente a la misma arista en el
grafo, la geometría y el feed. Salida a stdout; exit 0 si alinean, 1 si no.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

_REPO = Path(__file__).resolve().parent.parent
_MAPPING = _REPO / "ia_prediction_service/src/data/artifacts/miraflores_graph_lcc_mapping.json"


def _load_dotenv() -> None:
    env = _REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k, v)


def main() -> int:
    _load_dotenv()
    url = os.environ.get("DATABASE_URL", "").replace("@db:", "@localhost:")
    engine = create_engine(url, pool_pre_ping=True)

    mapping = {n["sumo_edge"] for n in json.loads(_MAPPING.read_text())["nodes"]}
    with engine.connect() as conn:
        geom = {r[0] for r in conn.execute(text("SELECT edge_id FROM graph_edges"))}
        feed = {r[0] for r in conn.execute(text("SELECT DISTINCT edge_id FROM waze_jams"))}

    print(f"mapping (sumo_edge):      {len(mapping)}")
    print(f"geometría (graph_edges):  {len(geom)}")
    print(f"feed (waze_jams DISTINCT):{len(feed)}")

    ok = (mapping == geom == feed)
    if ok:
        print(f"ALINEACIÓN OK: {len(mapping)} = {len(geom)} = {len(feed)} (conjuntos idénticos)")
        return 0
    print("DESALINEACIÓN:")
    print("  en mapping y no en geom:", sorted(mapping - geom)[:10])
    print("  en geom y no en feed:   ", sorted(geom - feed)[:10])
    print("  en feed y no en mapping:", sorted(feed - mapping)[:10])
    return 1


if __name__ == "__main__":
    sys.exit(main())
