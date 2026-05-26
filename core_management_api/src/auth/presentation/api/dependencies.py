"""FastAPI dependencies for authenticated routes (CT-01.4, HU-01)."""
from collections.abc import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from cerebrovial_shared.database.database import get_db
from cerebrovial_shared.database.models import UserDB

from src.auth.application.jwt_service import decode_token
from src.auth.domain import Role

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

_INVALID_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Token inválido",
    headers={"WWW-Authenticate": "Bearer"},
)

# Cuerpo 403 byte-idéntico en todos los rechazos por rol (RNF-SEC-04):
# no menciona endpoint, no menciona rol esperado, no indica si el recurso
# existe — el 403 se lanza antes de cualquier lógica de recurso.
_FORBIDDEN = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="Acceso denegado",
)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> UserDB:
    try:
        claims = decode_token(token)
    except JWTError:
        raise _INVALID_CREDENTIALS

    user_id = claims.get("sub")
    if not user_id:
        raise _INVALID_CREDENTIALS

    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if user is None:
        raise _INVALID_CREDENTIALS
    return user


def require_role(*allowed: Role) -> Callable[..., UserDB]:
    """HU-01 / CA-01.4. Factory que devuelve una dependency FastAPI."""
    def _dep(current_user: UserDB = Depends(get_current_user)) -> UserDB:
        try:
            user_role = Role(current_user.role)
        except ValueError:
            raise _FORBIDDEN
        if user_role not in allowed:
            raise _FORBIDDEN
        return current_user

    return _dep
