"""Auth HTTP endpoints (CT-01.1)."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from cerebrovial_shared.database.database import get_db
from cerebrovial_shared.database.models import UserDB

from src.auth.application.jwt_service import create_access_token, expiration_seconds
from src.auth.application.password import verify_password
from src.auth.domain import Role
from src.auth.presentation.api.schemas import TokenResponse

auth_router = APIRouter(prefix="/auth", tags=["auth"])

# Same message and status on every failure path so the client can't tell
# whether the username or the password was wrong (CT-01.1).
_INVALID_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Credenciales incorrectas",
)


@auth_router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> TokenResponse:
    user = db.query(UserDB).filter(UserDB.email == form_data.username).first()
    if user is None or not verify_password(form_data.password, user.password_hash):
        raise _INVALID_CREDENTIALS

    try:
        role = Role(user.role)
    except ValueError:
        raise _INVALID_CREDENTIALS

    token = create_access_token(sub=user.id, role=role)
    return TokenResponse(access_token=token, expires_in=expiration_seconds())
