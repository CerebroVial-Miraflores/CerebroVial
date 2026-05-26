"""JWT issue/decode service (CT-01.3, CT-01.4).

Algorithm: HS256. Secret and expiration are read from environment variables
lazily, so tests that set os.environ after import still pick up overrides.
"""
import os
from datetime import datetime, timedelta, timezone

from jose import jwt

from src.auth.domain import Role

ALGORITHM = "HS256"
DEFAULT_EXPIRATION_HOURS = 8


def _secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise RuntimeError("JWT_SECRET environment variable is not set")
    return secret


def expiration_seconds() -> int:
    return int(os.environ.get("JWT_EXPIRATION_HOURS", DEFAULT_EXPIRATION_HOURS)) * 3600


def create_access_token(
    *,
    sub: str,
    role: Role,
    expires_delta: timedelta | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    if expires_delta is None:
        expires_delta = timedelta(seconds=expiration_seconds())
    claims = {
        "sub": sub,
        "role": role.value,
        "exp": now + expires_delta,
    }
    return jwt.encode(claims, _secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify a token. Raises jose.JWTError on failure."""
    return jwt.decode(token, _secret(), algorithms=[ALGORITHM])
