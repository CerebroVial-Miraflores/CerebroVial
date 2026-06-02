"""CLI de replay de congestión por arista (TTH-12 Fase 2, CT-12.4/CT-12.5).

Envuelve el ``SumoReplayAdapter`` + ``WazeJamsRepo``: lee un día del dataset SUMO,
deriva el nivel 0-5 por arista y lo persiste en ``waze_jams``. Dos modos:

  pre-siembra (batch, los 375×1440 de una):
      .venv/bin/python scripts/replay_congestion.py --mode presiembra --day seed062
  replay en vivo (gotea a cadencia configurable):
      .venv/bin/python scripts/replay_congestion.py --mode vivo --day seed062 --speedup 60

El día es parámetro (default seed062, elegido en el perfilado de Fase 2 Paso 2 por
amplitud de transición — 219 aristas alcanzan jam≥3, pico AM). NO se hardcodea en la
lógica del adaptador; acá es solo el default del CLI.

Requiere el stack de BD (sqlalchemy+psycopg2) y pyarrow en el venv. NO push/PR/merge.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

_REPO = Path(__file__).resolve().parent.parent
# El adaptador vive en el core; la derivación reusa cerebrovial_simulation.
sys.path.insert(0, str(_REPO / "core_management_api/src"))
sys.path.insert(0, str(_REPO / "simulation/src"))

from congestion.infrastructure.repositories import WazeJamsRepo  # noqa: E402
from congestion.infrastructure.sumo_replay_adapter import SumoReplayAdapter  # noqa: E402


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
            os.environ.setdefault(key, val)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay de congestión SUMO → waze_jams (TTH-12).")
    ap.add_argument("--mode", choices=["presiembra", "vivo"], default="presiembra")
    ap.add_argument("--day", default="seed062", help="día del dataset (default: seed062)")
    ap.add_argument("--speedup", type=float, default=60.0, help="aceleración del replay vivo (1=tiempo real)")
    ap.add_argument("--max-steps", type=int, default=None, help="limita pasos en modo vivo (debug)")
    args = ap.parse_args()

    _load_dotenv()
    url = os.environ.get("DATABASE_URL", "").replace("@db:", "@localhost:")
    if not url:
        raise RuntimeError("DATABASE_URL no encontrado en el entorno ni en .env")

    engine = create_engine(url, pool_pre_ping=True)
    adapter = SumoReplayAdapter(day=args.day)

    with Session(engine) as session:
        repo = WazeJamsRepo(session)
        if args.mode == "presiembra":
            metrics = adapter.preseed(repo)
            session.commit()
            print(f"PRE-SIEMBRA ok: {metrics}")
            print(f"count(waze_jams) = {repo.count()}")
        else:
            # CT-12.6: el replay en vivo dispara un wake por paso. Corriendo IN-PROCESS
            # dentro de la API, este callback publicaría en el NetworkCongestionBroadcaster
            # (que el cliente despierta para re-leer GET /congestion/state). Como CLI
            # (proceso aparte del uvicorn) el wake solo se registra; el broadcaster vive
            # en el proceso de la API.
            steps_done = {"n": 0}

            def _wake() -> None:
                steps_done["n"] += 1

            n = adapter.live_replay(
                repo, speedup=args.speedup, max_steps=args.max_steps, wake=_wake
            )
            print(f"REPLAY VIVO: {n} pasos emitidos, {steps_done['n']} wakes "
                  f"(día {args.day}, speedup {args.speedup}).")


if __name__ == "__main__":
    main()
