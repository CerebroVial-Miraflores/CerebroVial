"""Frontera del endpoint dev-only ``POST /control/__internal/activate``.

Sin ``ENABLE_TEST_ACTIVATOR == "true"`` la ruta NO se registra: un GET/POST a
ella debe dar **404** (no 405). Esa diferencia es la que separa "endpoint
ausente del routing" de "endpoint registrado pero con método equivocado",
y es lo que evita que el andamiaje de prueba quede expuesto en prod.

Estos tests viven en ``tests/control/`` (no en ``tests/bdd/hu_05/``) a
propósito: ``tests/control/`` corre sin la env var seteada por la BDD conftest
(que la setea para que su suite use el endpoint). Así verificamos la frontera
con un build fresco del router.
"""
from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient


def test_setup_test_activator_returns_false_without_env_var(monkeypatch):
    """La función gate retorna False cuando la env var no está abierta.

    Esto es un test del CONTRATO de ``setup_test_activator()``: sin la env
    var, no registra nada, retorna False. Con monkeypatch garantizamos que
    cualquier estado leakeado entre suites no afecta este chequeo.
    """
    monkeypatch.delenv("ENABLE_TEST_ACTIVATOR", raising=False)
    from src.control.presentation.api.routes import (
        _is_test_activator_enabled,
        setup_test_activator,
    )

    assert _is_test_activator_enabled() is False
    assert setup_test_activator() is False


def test_setup_test_activator_returns_false_when_env_var_is_not_true(monkeypatch):
    """El gate exige exactamente ``"true"`` — valores como "1", "false",
    "True" no abren el endpoint (evita gates accidentales por typos)."""
    from src.control.presentation.api.routes import (
        _is_test_activator_enabled,
        setup_test_activator,
    )

    for value in ("1", "false", "yes", "True", "TRUE", "", "  true  "):
        monkeypatch.setenv("ENABLE_TEST_ACTIVATOR", value)
        assert _is_test_activator_enabled() is False, (
            f"el gate debió cerrarse para ENABLE_TEST_ACTIVATOR={value!r}"
        )
        assert setup_test_activator() is False


def test_unregistered_internal_activate_yields_404_not_405(monkeypatch):
    """Frontera observacional: con la env var cerrada y un router limpio,
    un GET y un POST a ``/control/__internal/activate`` deben devolver 404
    (ruta no existe) en lugar de 405 (ruta existe, método equivocado).

    Construimos un app fresco con un router vacío y solo le aplicamos
    ``setup_test_activator()`` sobre el router global (que no la registra
    porque el gate está off). Confirmamos el comportamiento de FastAPI
    para paths ausentes.
    """
    monkeypatch.delenv("ENABLE_TEST_ACTIVATOR", raising=False)
    # Limpiamos cualquier registración previa que otra suite haya filtrado
    # al router global (la BDD de hu_05 sí registra el endpoint).
    from src.control.presentation.api import routes as routes_module
    routes_module.router.routes = [
        r
        for r in routes_module.router.routes
        if getattr(r, "path", "") != "/control/__internal/activate"
    ]

    # Confirma que setup_test_activator NO re-registra (env var cerrada).
    assert routes_module.setup_test_activator() is False

    # Build app + client y verifica que un GET/POST a la ruta da 404.
    app = FastAPI()
    app.include_router(routes_module.router)
    client = TestClient(app)

    post_response = client.post(
        "/control/__internal/activate",
        json={"node_id": "x", "decision_id": "y"},
    )
    assert post_response.status_code == 404, (
        f"esperaba 404 (ruta no existe), obtuvo {post_response.status_code}"
    )

    get_response = client.get("/control/__internal/activate")
    assert get_response.status_code == 404, (
        f"esperaba 404 (ruta no existe), obtuvo {get_response.status_code}"
    )


def test_setup_test_activator_registers_route_when_env_var_true(monkeypatch):
    """Al abrir el gate, la función agrega el endpoint al router global.

    Es idempotente: una segunda llamada con el gate abierto no duplica la
    ruta (la verifica como ya presente y retorna True igual)."""
    from src.control.presentation.api import routes as routes_module

    # Limpieza preventiva: dejamos el router en estado "endpoint ausente".
    routes_module.router.routes = [
        r
        for r in routes_module.router.routes
        if getattr(r, "path", "") != "/control/__internal/activate"
    ]

    monkeypatch.setenv("ENABLE_TEST_ACTIVATOR", "true")
    assert routes_module.setup_test_activator() is True

    paths = [getattr(r, "path", "") for r in routes_module.router.routes]
    matches = [p for p in paths if p == "/control/__internal/activate"]
    assert len(matches) == 1, (
        f"esperaba exactamente 1 ruta __internal/activate, hay {len(matches)}"
    )

    # Idempotencia: segunda llamada con el gate abierto no duplica.
    assert routes_module.setup_test_activator() is True
    paths = [getattr(r, "path", "") for r in routes_module.router.routes]
    matches = [p for p in paths if p == "/control/__internal/activate"]
    assert len(matches) == 1, (
        f"segunda llamada duplicó la ruta: {len(matches)} matches"
    )

    # Limpieza para no leakear estado a tests siguientes en esta suite.
    routes_module.router.routes = [
        r
        for r in routes_module.router.routes
        if getattr(r, "path", "") != "/control/__internal/activate"
    ]


def test_unregistered_path_returns_404_not_405_in_fastapi():
    """Sanity: FastAPI/Starlette devuelve 404 para paths no registrados (no
    405). Test independiente de nuestra app para fijar la semántica que se
    espera de la frontera anterior."""
    app = FastAPI()
    router = APIRouter(prefix="/control")

    @router.get("/health")
    def _h() -> dict:
        return {"ok": True}

    app.include_router(router)
    client = TestClient(app)

    # /control/health existe (GET) → POST debería dar 405
    assert client.post("/control/health").status_code == 405

    # /control/__internal/activate NO existe → cualquier método da 404
    assert (
        client.post(
            "/control/__internal/activate",
            json={},
        ).status_code
        == 404
    )
    assert client.get("/control/__internal/activate").status_code == 404
