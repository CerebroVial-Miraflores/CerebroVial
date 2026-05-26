"""
Seed de tres usuarios de smoke (uno por rol) para probar HU-01 manualmente.

Credenciales de desarrollo (NO usar fuera de smoke local):
  operator@cv.pe / Smoke1234  → rol operator
  manager@cv.pe  / Smoke1234  → rol manager
  admin@cv.pe    / Smoke1234  → rol admin

Idempotente: si un usuario con el mismo email ya existe, se salta sin error.
Hash de contraseña vía core_management_api.src.auth.application.password.hash_password
(bcrypt cost=12, función canónica de TTH-01).
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Permitir importar `src.auth.application.password` desde core_management_api/,
# mismo patrón que usan los tests (tests/auth/conftest.py, tests/bdd/conftest.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "core_management_api"))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from cerebrovial_shared.database.models import UserDB
from src.auth.application.password import hash_password
from src.auth.domain import Role


_SMOKE_PASSWORD = "Smoke1234"

_SMOKE_USERS = [
    ("operator@cv.pe", Role.OPERATOR),
    ("manager@cv.pe",  Role.MANAGER),
    ("admin@cv.pe",    Role.ADMIN),
]


def _load_dotenv() -> None:
    env_path = _REPO_ROOT / ".env"
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


def main() -> None:
    _load_dotenv()
    url = os.environ.get("DATABASE_URL", "").replace("@db:", "@localhost:")
    if not url:
        raise RuntimeError("DATABASE_URL no encontrado en el entorno ni en .env")

    engine = create_engine(url, pool_pre_ping=True)
    with Session(engine) as session:
        created, skipped = [], []
        for email, role in _SMOKE_USERS:
            existing = session.scalar(select(UserDB).where(UserDB.email == email))
            if existing is not None:
                skipped.append((email, role.value))
                continue
            session.add(UserDB(
                id=str(uuid.uuid4()),
                email=email,
                password_hash=hash_password(_SMOKE_PASSWORD),
                role=role.value,
                created_at=datetime.utcnow(),
            ))
            created.append((email, role.value))
        session.commit()

    print("=== seed-rbac-smoke ===")
    print(f"Password de dev (para los tres): {_SMOKE_PASSWORD}")
    if created:
        print("Creados:")
        for email, role in created:
            print(f"  + {email}  (rol: {role})")
    if skipped:
        print("Ya existían (se saltaron):")
        for email, role in skipped:
            print(f"  · {email}  (rol: {role})")
    if not created and not skipped:
        print("Nada que hacer.")


if __name__ == "__main__":
    main()
