"""FastAPI dependencies for authenticated routes (CT-01.4)."""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from cerebrovial_shared.database.database import get_db
from cerebrovial_shared.database.models import UserDB

from src.auth.application.jwt_service import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

_INVALID_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Token inválido",
    headers={"WWW-Authenticate": "Bearer"},
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
