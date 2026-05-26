"""TTH-01 — Tests CT-01.5.

Five tests required by the spec, mapped to criteria:
    1. login with valid credentials (CT-01.1, CT-01.2, CT-01.3)
    2. login with invalid credentials returns 401 + generic message (CT-01.1)
    3. decode of a valid token returns claims (CT-01.3, CT-01.4)
    4. decode of an expired token is rejected with 401 (CT-01.4)
    5. decode of a token with invalid signature is rejected with 401 (CT-01.4)
"""
from datetime import timedelta

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt as jose_jwt

from src.auth.application.jwt_service import create_access_token, decode_token
from src.auth.domain import Role
from src.auth.presentation.api.dependencies import get_current_user


# --- 1. CT-01.1 + CT-01.2 + CT-01.3 ----------------------------------------
def test_login_with_valid_credentials_returns_200_and_token(
    client: TestClient, seed_user
) -> None:
    user = seed_user(email="alice@cv.pe", password="s3cret-pass", role="operator")

    response = client.post(
        "/auth/login",
        data={"username": "alice@cv.pe", "password": "s3cret-pass"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 8 * 3600
    assert isinstance(body["access_token"], str) and body["access_token"]

    claims = decode_token(body["access_token"])
    assert claims["sub"] == user.id
    assert claims["role"] == "operator"
    assert "exp" in claims


# --- 2. CT-01.1 (generic 401, no leak) -------------------------------------
@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("ghost@cv.pe", "anything"),       # user does not exist
        ("alice@cv.pe", "wrong-password"), # password mismatch
    ],
    ids=["unknown_user", "wrong_password"],
)
def test_login_with_invalid_credentials_returns_401_generic_message(
    client: TestClient, seed_user, username: str, password: str
) -> None:
    seed_user(email="alice@cv.pe", password="real-pass", role="operator")

    response = client.post(
        "/auth/login",
        data={"username": username, "password": password},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Credenciales incorrectas"


# --- 3. CT-01.3 + CT-01.4 (decode happy path) ------------------------------
def test_decode_valid_token_returns_claims() -> None:
    token = create_access_token(sub="user-123", role=Role.MANAGER)
    claims = decode_token(token)

    assert claims["sub"] == "user-123"
    assert claims["role"] == "manager"
    assert "exp" in claims


# --- 4. CT-01.4 (expired) --------------------------------------------------
def test_decode_expired_token_via_dependency_raises_401() -> None:
    expired_token = create_access_token(
        sub="user-123",
        role=Role.OPERATOR,
        expires_delta=timedelta(seconds=-10),
    )

    with pytest.raises(HTTPException) as exc:
        get_current_user(token=expired_token, db=None)
    assert exc.value.status_code == 401


# --- 5. CT-01.4 (invalid signature) ----------------------------------------
def test_decode_token_with_invalid_signature_raises_401() -> None:
    forged = jose_jwt.encode(
        {"sub": "user-123", "role": "operator", "exp": 9999999999},
        "totally-different-secret",
        algorithm="HS256",
    )

    with pytest.raises(HTTPException) as exc:
        get_current_user(token=forged, db=None)
    assert exc.value.status_code == 401
