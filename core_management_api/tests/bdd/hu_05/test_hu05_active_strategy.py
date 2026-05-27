"""Step definitions para HU-05 (vista pasiva del estado vigente).

Cubre los .feature de ``features/hu-05-active-strategy/``:

- CA-05.1 (parcial): estado visible — contrato backend del campo crudo
  ``strategy_mode`` y forma de ``phase_timings``. La mitad "lo que ve el
  Operador" (etiqueta DHU-006 de dominio) la cierra el guard rail Vitest
  de Fase 5: el render NO debe mostrar ``"webster"``/``"max_pressure"``
  crudos.
- CA-05.2: timestamp de activación + decided_at (DHU-021 #13).
- CA-05.3: actualización en vivo vía SSE. En Fase 1 está en rojo con un
  GET sincrónico; Fase 3 reemplaza el step por streaming async real.
- CA-05.5: protección por rol (401/403/200).

Además cubrimos el contrato `404 no_active_state` para nodos sin fila en
``engine_active_state`` (estado inicial vacío). **Este contrato NO es
CA-05.4**: CA-05.4 es resiliencia stateful (DHU-005 Caso B: última
conocida + marca "no confirmada" + contador) y vive en el frontend
(Vitest sobre ``ActiveStrategyView`` en Fase 6).

En Fase 1 todos los escenarios fallan porque ``/control/active-state/{node_id}``
y su stream aún no existen. Fase 2 cierra CA-05.1 (contrato), CA-05.2,
CA-05.5 y el contrato 404. Fase 3 cierra CA-05.3.
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable

from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenarios, then, when

from cerebrovial_shared.database.models import EngineActiveStateDB

scenarios(
    "../../../../features/hu-05-active-strategy/ca_05_1_estado_visible.feature",
    "../../../../features/hu-05-active-strategy/ca_05_2_timestamp_activacion.feature",
    "../../../../features/hu-05-active-strategy/ca_05_5_proteccion_rol.feature",
    "../../../../features/hu-05-active-strategy/contrato_404_no_active_state.feature",
)
# CA-05.3 (SSE en vivo) requiere httpx.AsyncClient + pytest-asyncio para
# probarse de verdad sobre el stream HTTP. Vive en ``test_hu05_sse.py``.


# === Given: seed de decisiones / activación ===============================

@given(
    parsers.parse('existe una decisión "{mode}" registrada para "{node_id}"'),
    target_fixture="seeded_decision_id",
)
def _seed_decision(
    mode: str,
    node_id: str,
    seed_motor_decision: Callable[..., str],
) -> str:
    return seed_motor_decision(node_id=node_id, mode=mode)


@given(
    parsers.parse(
        'esa decisión fue activada como estrategia vigente para "{node_id}"'
    )
)
def _activate_seeded(
    node_id: str,
    seeded_decision_id: str,
    activator: Callable[..., None],
) -> None:
    activator(node_id, seeded_decision_id, activated_by="bdd-bg")


@given(
    parsers.parse('no existe ninguna estrategia activa para "{node_id}"')
)
def _no_active_state(node_id: str, hu05_session) -> None:
    existing = hu05_session.get(EngineActiveStateDB, node_id)
    assert existing is None, (
        f"precondición rota: ya hay engine_active_state para {node_id!r}"
    )


# === Given: autenticación =================================================

@given(
    parsers.parse('un usuario autenticado con rol "{role}"'),
    target_fixture="auth_headers",
)
def _auth_headers(
    role: str, client: TestClient, seed_user: Callable[..., object]
) -> dict[str, str]:
    email = f"{role}@cv.pe"
    password = "passw0rd-hu05"
    seed_user(email=email, password=password, role=role)

    response = client.post(
        "/auth/login",
        data={"username": email, "password": password},
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# === When: peticiones HTTP ===============================================

@when(
    parsers.parse(
        'un cliente hace "GET /control/active-state/{node_id}" '
        "sin cabecera de autorización"
    ),
    target_fixture="response",
)
def _request_no_auth(node_id: str, client: TestClient):
    return client.get(f"/control/active-state/{node_id}")


@when(
    parsers.parse('ese usuario hace "GET /control/active-state/{node_id}"'),
    target_fixture="response",
)
def _request_with_auth(
    node_id: str, client: TestClient, auth_headers: dict[str, str]
):
    return client.get(
        f"/control/active-state/{node_id}", headers=auth_headers
    )


# === Then: status =========================================================

@then(parsers.parse("la respuesta tiene código HTTP {code:d}"))
def _assert_status(response, code: int) -> None:
    assert response.status_code == code, response.text


# === Then: forma del cuerpo (CA-05.1, CA-05.2) ===========================

@then(
    parsers.parse(
        'la respuesta contiene el campo "{field}" con valor "{value}"'
    )
)
def _assert_field_eq(response, field: str, value: str) -> None:
    body = response.json()
    assert field in body, f"campo {field!r} ausente: keys={list(body.keys())}"
    assert str(body[field]) == value, (
        f"{field} esperaba {value!r}, obtuvo {body[field]!r}"
    )


@then(parsers.parse('la respuesta contiene el campo "{field}" con un número'))
def _assert_field_is_number(response, field: str) -> None:
    body = response.json()
    assert field in body, f"campo {field!r} ausente: keys={list(body.keys())}"
    assert isinstance(body[field], (int, float)), (
        f"{field} esperaba un número, obtuvo {type(body[field]).__name__}"
    )


@then(parsers.parse('la respuesta contiene una lista no vacía de "{field}"'))
def _assert_field_is_nonempty_list(response, field: str) -> None:
    body = response.json()
    assert field in body, f"campo {field!r} ausente: keys={list(body.keys())}"
    assert isinstance(body[field], list), (
        f"{field} esperaba una lista, obtuvo {type(body[field]).__name__}"
    )
    assert len(body[field]) > 0, f"{field} estaba vacío"


@then(
    parsers.parse(
        'cada elemento de "{field}" tiene los campos "{f1}", "{f2}", '
        '"{f3}", "{f4}"'
    )
)
def _assert_elements_have_fields(
    response, field: str, f1: str, f2: str, f3: str, f4: str
) -> None:
    body = response.json()
    required = {f1, f2, f3, f4}
    for idx, item in enumerate(body[field]):
        missing = required - set(item.keys())
        assert not missing, (
            f"elemento {idx} de {field!r} no tiene {missing}: keys={list(item.keys())}"
        )


@then(
    parsers.parse(
        'la respuesta contiene el campo "{field}" con un timestamp ISO-8601'
    )
)
def _assert_field_is_iso8601(response, field: str) -> None:
    body = response.json()
    assert field in body, f"campo {field!r} ausente: keys={list(body.keys())}"
    raw = body[field]
    assert isinstance(raw, str), (
        f"{field} esperaba string, obtuvo {type(raw).__name__}"
    )
    # ISO-8601 admite ``T`` o espacio como separador y opcionalmente offset.
    # ``fromisoformat`` cubre el subconjunto que devuelve datetime/json.
    try:
        dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AssertionError(
            f"{field} no es ISO-8601 válido: {raw!r} ({exc})"
        ) from exc


# === Then: error con código identificable (CA-05.4) =======================

@then(
    parsers.parse('el cuerpo de la respuesta tiene el código de error "{code}"')
)
def _assert_error_code(response, code: str) -> None:
    body = response.json()
    detail = body.get("detail")
    # ErrorDetail puede venir serializado como dict (modelo Pydantic) o como
    # string si se usó HTTPException(detail=str).
    if isinstance(detail, dict):
        assert detail.get("code") == code, (
            f'esperaba detail.code == {code!r}, obtuvo {detail!r}'
        )
    else:
        raise AssertionError(
            f"detail no es dict (esperaba ErrorDetail): {detail!r}"
        )


