"""Step definitions para rbac_api.feature (CA-01.4 — HU-01).

Cubre los cuatro escenarios:
  - Sin token → 401
  - Token de Operador → 403 con cuerpo genérico (RNF-SEC-04)
  - Token de Gerente → 403 con cuerpo genérico (RNF-SEC-04)
  - Token de Administrador → 200
"""
from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("../../../../features/hu-01-rbac/rbac_api.feature")


# === Antecedentes =========================================================

@given('un endpoint de muestra "GET /api/health" protegido para el rol "admin"')
def _setup_protected_endpoint(client: TestClient) -> TestClient:
    # El endpoint está montado por la fixture app_with_rbac del conftest BDD.
    return client


# === Given: usuario autenticado ===========================================

@given(
    parsers.parse('un usuario autenticado con rol "{role}"'),
    target_fixture="auth_headers",
)
def _auth_headers(role: str, client: TestClient, seed_user) -> dict[str, str]:
    email = f"{role}@cv.pe"
    password = "passw0rd-rbac"
    seed_user(email=email, password=password, role=role)

    response = client.post(
        "/auth/login",
        data={"username": email, "password": password},
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# === When =================================================================

@when(
    'un cliente hace "GET /api/health" sin cabecera de autorización',
    target_fixture="response",
)
def _request_no_auth(client: TestClient):
    return client.get("/api/health")


@when('ese usuario hace "GET /api/health"', target_fixture="response")
def _request_with_auth(client: TestClient, auth_headers: dict[str, str]):
    return client.get("/api/health", headers=auth_headers)


# === Then: status =========================================================

@then(parsers.parse("la respuesta tiene código HTTP {code:d}"))
def _assert_status(response, code: int) -> None:
    assert response.status_code == code, response.text


# === Then: 200 con cuerpo del recurso =====================================

@then("la respuesta contiene un cuerpo JSON con el estado del sistema")
def _assert_health_body(response) -> None:
    body = response.json()
    assert body == {"status": "ok"}


# === Then: RNF-SEC-04 — no filtración =====================================

@then("la respuesta no incluye información del recurso solicitado")
def _assert_no_resource_info_401(response) -> None:
    body = response.json()
    detail = str(body.get("detail", "")).lower()
    for forbidden in ("health", "/api/", "status", "ok"):
        assert forbidden not in detail, (
            f"el cuerpo 401 filtra info del recurso: detail={body.get('detail')}"
        )


@then("el cuerpo de la respuesta no menciona el nombre del recurso")
def _assert_no_resource_leak(response) -> None:
    body = response.json()
    detail = str(body.get("detail", "")).lower()
    for forbidden in ("health", "/api/"):
        assert forbidden not in detail, (
            f"el cuerpo 403 filtra el nombre del recurso: detail={body.get('detail')}"
        )


@then("el cuerpo de la respuesta no indica si el recurso existe o no")
def _assert_no_existence_leak(response) -> None:
    body = response.json()
    detail = str(body.get("detail", "")).lower()
    for forbidden in ("existe", "no encontrado", "found", "exists", "missing"):
        assert forbidden not in detail, (
            f"el cuerpo 403 filtra existencia del recurso: detail={body.get('detail')}"
        )


@then("el cuerpo de la respuesta no menciona el rol esperado")
def _assert_no_role_leak(response) -> None:
    body = response.json()
    detail = str(body.get("detail", "")).lower()
    for forbidden in (
        "admin",
        "operator",
        "manager",
        "administrador",
        "operador",
        "gerente",
    ):
        assert forbidden not in detail, (
            f"el cuerpo 403 filtra el rol esperado: detail={body.get('detail')}"
        )
