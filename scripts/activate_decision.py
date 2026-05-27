"""
CLI para activar una decisión registrada como estrategia vigente de un nodo
(HU-05 / Fase 4 del plan).

Propósito acotado: poblar la fila ``engine_active_state`` para que la vista
pasiva de HU-05 tenga algo que pintar y para poder ejercitar el GET
``/control/active-state/{node_id}`` contra la BD real.

**NO dispara SSE en vivo.** El broadcaster es un dict en memoria del proceso
``uvicorn``; este CLI corre en un proceso Python distinto, incluso cuando se
invoca con ``docker compose exec``. La invariante "publish post-commit" del
endpoint ``__internal/activate`` (que sí dispara SSE, intra-uvicorn, gated
por ``ENABLE_TEST_ACTIVATOR``) NO aplica acá: este script solo escribe a BD.
Si quieres ejercitar SSE en vivo, usa ``POST /control/__internal/activate``
con un Bearer de Administrador. El activador productivo será HU-07 (operador
o automático) y vivirá dentro del proceso uvicorn — no aquí.

Uso:
    docker compose exec core_management_api python scripts/activate_decision.py \\
        --node-id larco_schell \\
        --decision-id <uuid de motor_decisions> \\
        [--activated-by cli]

El script valida:
- ``--node-id`` existe en ``graph_nodes`` (fail-fast si no).
- ``--decision-id`` existe en ``motor_decisions`` (fail-fast si no).
- ``motor_decisions.node_id`` del decision-id coincide con ``--node-id``
  (rechaza activar la decisión de un nodo como vigente de otro nodo, evitando
  romper el audit trail).

Exit code 0 si la fila quedó actualizada (insertada o re-apuntada). Imprime el
estado resultante de ``engine_active_state`` por stdout para verificación.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from cerebrovial_shared.database.models import (
    EngineActiveStateDB,
    MotorDecisionDB,
)


def _load_dotenv() -> None:
    """Lee ``.env`` de la raíz del repo (mismo patrón que scripts/seed.py).

    Necesario cuando el script se invoca desde la máquina del usuario; dentro
    del contenedor las env vars vienen ya seteadas por docker-compose.
    """
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


def _build_engine_url() -> str:
    """Resuelve DATABASE_URL. Si el host es ``db`` (nombre del servicio en
    docker-compose) pero corremos fuera del contenedor, lo cambia a
    ``localhost``. Mismo truco que scripts/seed.py."""
    _load_dotenv()
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise SystemExit(
            "DATABASE_URL no encontrado en el entorno ni en .env"
        )
    # Si corremos fuera del contenedor pero el .env apunta a @db:, conmutamos
    # a @localhost: para que el conector llegue al puerto expuesto del compose.
    # Detectamos "estoy en el container" por la presencia del archivo
    # /.dockerenv (estándar de Docker).
    if not Path("/.dockerenv").exists():
        url = url.replace("@db:", "@localhost:")
    return url


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Activa una decisión existente como estrategia vigente de un nodo. "
            "No dispara SSE; solo escribe a engine_active_state."
        )
    )
    parser.add_argument(
        "--node-id",
        required=True,
        help='node_id de graph_nodes (ej. "larco_schell").',
    )
    parser.add_argument(
        "--decision-id",
        required=True,
        help="decision_id (UUID) de motor_decisions a marcar como vigente.",
    )
    parser.add_argument(
        "--activated-by",
        default="cli",
        help='Identificador opcional del actor que activa (default: "cli").',
    )
    return parser.parse_args()


def activate_decision(
    session: Session,
    *,
    node_id: str,
    decision_id: str,
    activated_by: str,
) -> EngineActiveStateDB:
    """Ejecuta la activación con validaciones.

    Importa el repo en runtime (no a nivel de módulo) para que los help
    messages de ``--help`` no requieran un DATABASE_URL válido para
    desplegarse — el import del repo arrastra ``src.control`` y sus
    dependencias.
    """
    from src.control.infrastructure import (
        EngineActiveStateRepo,
        resolve_node_id,
    )

    if resolve_node_id(session, node_id) is None:
        raise SystemExit(
            f'node_id {node_id!r} no existe en graph_nodes — '
            "corre `invoke seed` o usa un node_id válido."
        )

    decision = session.scalar(
        select(MotorDecisionDB).where(
            MotorDecisionDB.decision_id == decision_id
        )
    )
    if decision is None:
        raise SystemExit(
            f'decision_id {decision_id!r} no existe en motor_decisions — '
            "verifica el UUID o crea una decisión con POST /control/recommend."
        )
    if decision.node_id != node_id:
        raise SystemExit(
            f"inconsistencia: la decisión {decision_id!r} pertenece al nodo "
            f"{decision.node_id!r}, no a {node_id!r}. Activar una decisión "
            "de un nodo como vigente de otro rompe el audit trail."
        )

    repo = EngineActiveStateRepo(session)
    repo.activate(
        node_id=node_id,
        decision_id=decision_id,
        activated_by=activated_by,
    )
    session.commit()

    refreshed = session.get(EngineActiveStateDB, node_id)
    if refreshed is None:
        # No debería ocurrir tras el commit, pero defensivo.
        raise SystemExit(
            f"fallo inesperado: no se encontró engine_active_state para "
            f"{node_id!r} tras commit."
        )
    return refreshed


def main() -> None:
    args = _parse_args()
    engine_url = _build_engine_url()
    engine = create_engine(engine_url, pool_pre_ping=True)
    with Session(engine) as session:
        row = activate_decision(
            session,
            node_id=args.node_id,
            decision_id=args.decision_id,
            activated_by=args.activated_by,
        )
    print(
        f"activated node_id={row.node_id} "
        f"active_decision_id={row.active_decision_id} "
        f"activated_at={row.activated_at.isoformat()} "
        f"activated_by={row.activated_by!r}",
        file=sys.stdout,
    )
    print(
        "nota: este CLI NO publica el evento SSE. Si un cliente está "
        "conectado al stream, NO recibirá active-state-changed por este "
        "camino. Para verificar SSE en vivo usa POST /control/__internal/"
        "activate (gated por ENABLE_TEST_ACTIVATOR).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
