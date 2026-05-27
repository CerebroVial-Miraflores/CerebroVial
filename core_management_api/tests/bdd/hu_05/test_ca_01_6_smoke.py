"""Step definitions del smoke CA-01.6 sobre ruta consumida por la UI (HU-05).

El smoke cierra la deuda diferida en HU-01: la cobertura ejecutable existía
en piezas (httpClient interceptor + SessionProvider.performLogout + flash en
LoginView + rechazo 401 sobre /api/health), pero UNIR las dos mitades en
vivo requería un endpoint con ``require_role`` que la UI consuma en su flujo
natural. HU-05 lo trajo: ``GET /control/active-state/{node_id}``.

Protocolo (de CLAUDE.md):
- NO usar ``JWT_EXPIRATION_HOURS`` fraccionario — el servicio JWT lee la env
  var con ``int(...)`` y revienta con decimales (p. ej. "0.001").
- USAR ``create_access_token(..., expires_delta=timedelta(seconds=-N))`` para
  forjar un token con ``exp`` ya en el pasado, evitando depender del wall
  clock del runner.

El step defs vive en ``tests/bdd/hu_05/`` para reutilizar el conftest BDD de
HU-05 (combinación de auth_router + control_router contra una BD SQLite en
memoria con UserDB + graph_nodes + motor_decisions + engine_active_state).
"""
from datetime import timedelta

from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenarios, then, when

from src.auth.application.jwt_service import create_access_token
from src.auth.domain import Role

scenarios(
    "../../../../features/hu-01-rbac/ca_01_6_token_expirado_endpoint_vivo.feature"
)


# === Given ================================================================

@given(
    "un Operador con un token JWT cuya expiración ya pasó",
    target_fixture="auth_headers",
)
def _expired_operator_token(seed_user) -> dict[str, str]:
    """Forja un token con ``exp`` 1 segundo en el pasado.

    No usamos ``time.sleep`` ni manipulamos ``JWT_EXPIRATION_HOURS`` (que el
    servicio lee con ``int(...)``); aprovechamos que ``create_access_token``
    acepta un ``expires_delta`` negativo y produce el ``exp`` directamente
    en el pasado.
    """
    email = "operator-expired@cv.pe"
    seed_user(
        email=email,
        password="passw0rd-smoke",
        role="operator",
    )
    token = create_access_token(
        sub=email,
        role=Role.OPERATOR,
        expires_delta=timedelta(seconds=-1),
    )
    return {"Authorization": f"Bearer {token}"}


# === When =================================================================

@when(
    parsers.parse('ese Operador hace "GET /control/active-state/{node_id}"'),
    target_fixture="response",
)
def _request_with_expired_token(
    node_id: str, client: TestClient, auth_headers: dict[str, str]
):
    return client.get(
        f"/control/active-state/{node_id}", headers=auth_headers
    )


# === Then =================================================================

@then(parsers.parse("la respuesta tiene código HTTP {code:d}"))
def _assert_status(response, code: int) -> None:
    assert response.status_code == code, response.text
