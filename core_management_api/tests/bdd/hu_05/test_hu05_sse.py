"""Test de CA-05.3 (SSE en vivo) — streaming HTTP real con uvicorn + httpx.

**Por qué uvicorn y no httpx.ASGITransport**: ``ASGITransport`` BUFFERA las
respuestas streaming (verificado empíricamente: ``response.status_code`` no
queda disponible hasta que el generator emite el primer chunk). En CA-05.3 el
generator del endpoint queda colgado en ``await queue.get()`` hasta que el
test dispara ``publish`` — pero el test no puede disparar publish hasta saber
que el cliente está suscrito, y no sabe que está suscrito hasta recibir
status 200, que ASGITransport no expone. Deadlock por diseño del transport.
Solución: uvicorn en un puerto local libre, httpx contra ``http://127.0.0.1:N``
con streaming real.

**Por qué no es pytest-bdd**: pytest-bdd 8.1.0 no soporta steps ``async def``
nativamente (las coroutines no se awaitan, RuntimeWarning explícito). El
.feature ``ca_05_3_stream_actualiza.feature`` documenta el escenario; este
archivo es su ejecución autoritativa.

**Cobertura del archivo**:
1. ``test_ca_05_3_sse_active_state_changed`` — escenario completo del .feature:
   operator suscribe stream → activate + publish → recibe evento < 3s.
2. ``test_sse_unsubscribes_on_client_disconnect`` — verifica que el ``finally``
   del event_generator hace ``unsubscribe`` al cerrar la conexión (sin esto,
   cada cliente que cierra deja su Queue en el dict del broadcaster).
3. ``test_publish_runs_after_commit_in_internal_activate_handler`` —
   verificación estructural del orden ``commit → publish`` en el código.
"""
from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from datetime import datetime

import pytest
import pytest_asyncio
import uvicorn
from httpx import AsyncClient

from src.control.infrastructure import EngineActiveStateRepo


def _find_free_port() -> int:
    """Bind 0 → kernel asigna un puerto libre; lo soltamos para que uvicorn lo tome."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest_asyncio.fixture
async def uvicorn_server(
    app_with_rbac, broadcaster_instance
) -> AsyncIterator[str]:
    """Levanta un uvicorn en un puerto libre con la app del conftest BDD.

    Mismo proceso, mismo event loop, mismos singletons (broadcaster, engine):
    el ``broadcaster.publish`` del test alcanza al ``broadcaster.subscribe``
    del endpoint sin pasar por ningún transport intermedio falso.
    """
    del broadcaster_instance  # anchored: garantiza init_broadcaster antes de servir

    port = _find_free_port()
    config = uvicorn.Config(
        app_with_rbac,
        host="127.0.0.1",
        port=port,
        log_level="critical",
        lifespan="off",
    )
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())
    # Espera a que el server marque ``started``. Sin esto los primeros requests
    # pueden fallar con ConnectionRefused.
    for _ in range(200):
        await asyncio.sleep(0.02)
        if server.started:
            break
    else:
        server.should_exit = True
        await server_task
        raise RuntimeError("uvicorn no llegó a started")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=3.0)
        except asyncio.TimeoutError:
            server_task.cancel()
            try:
                await server_task
            except (asyncio.CancelledError, Exception):
                pass


async def _login_operator(
    base_url: str, seed_user, email: str = "operator@cv.pe"
) -> dict[str, str]:
    """Helper: crea un operator y devuelve los headers de Authorization."""
    password = "passw0rd-sse"
    seed_user(email=email, password=password, role="operator")
    async with AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/auth/login",
            data={"username": email, "password": password},
        )
        assert response.status_code == 200, response.text
        return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.mark.asyncio
async def test_ca_05_3_sse_active_state_changed(
    uvicorn_server: str,
    seed_user,
    seed_motor_decision,
    hu05_session,
    broadcaster_instance,
):
    """Escenario CA-05.3 end-to-end sobre el stream HTTP real."""
    node_id = "larco_schell"
    auth_headers = await _login_operator(uvicorn_server, seed_user)

    # Given: existe una decisión "webster" para larco_schell
    decision_id = seed_motor_decision(node_id=node_id, mode="webster")

    events: list[dict] = []
    ready = asyncio.Event()
    stop = asyncio.Event()

    async def collector() -> None:
        async with AsyncClient(base_url=uvicorn_server, timeout=10.0) as client:
            async with client.stream(
                "GET",
                f"/control/active-state/{node_id}/stream",
                headers=auth_headers,
            ) as response:
                assert response.status_code == 200, (
                    f"stream devolvió {response.status_code}"
                )
                ready.set()
                current_event: str | None = None
                async for line in response.aiter_lines():
                    if stop.is_set():
                        break
                    line = line.rstrip("\r")
                    if line.startswith("event:"):
                        current_event = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        raw = line[len("data:"):].strip()
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            payload = {"_raw": raw}
                        events.append(
                            {"event": current_event, "data": payload}
                        )
                        current_event = None
                        # Primer evento útil — cumplió, suelta la conexión.
                        break

    collector_task = asyncio.create_task(collector())
    # Espera a que el cliente reciba 200 (=> subscribe() del endpoint ejecutó).
    await asyncio.wait_for(ready.wait(), timeout=5.0)
    # Un pequeño yield extra: el endpoint termina de retornar EventSourceResponse
    # y la primera iteración del generator inicia ``await queue.get()``.
    await asyncio.sleep(0.05)

    # When: activate + publish — ORDEN CRÍTICO commit→publish
    repo = EngineActiveStateRepo(hu05_session)
    repo.activate(
        node_id=node_id,
        decision_id=decision_id,
        activated_by="sse-test",
    )
    hu05_session.commit()  # COMMIT first.
    await broadcaster_instance.publish(  # PUBLISH after commit.
        node_id,
        {
            "type": "active-state-changed",
            "node_id": node_id,
            "decision_id": decision_id,
            "activated_at": datetime.utcnow().isoformat() + "Z",
        },
    )

    # Then: evento recibido en < 3s
    try:
        await asyncio.wait_for(collector_task, timeout=3.0)
    except asyncio.TimeoutError:
        stop.set()
        collector_task.cancel()
        try:
            await collector_task
        except (asyncio.CancelledError, Exception):
            pass
        raise AssertionError(
            f"no se recibió evento active-state-changed en 3s; "
            f"events_collected={events}"
        )

    assert len(events) >= 1, f"no se recolectó ningún evento: {events}"
    last = events[-1]

    assert last["event"] == "active-state-changed", (
        f"evento recibido fue {last['event']!r}, esperaba "
        f"'active-state-changed'"
    )
    assert last["data"].get("node_id") == node_id
    assert "decision_id" in last["data"]
    assert last["data"]["decision_id"] == decision_id


@pytest.mark.asyncio
async def test_sse_unsubscribes_on_client_disconnect(
    uvicorn_server: str,
    seed_user,
    broadcaster_instance,
):
    """Verifica que el ``finally`` del event_generator hace ``unsubscribe`` al
    cerrar la conexión. Sin este cleanup, cada conexión cerrada deja su Queue
    en el dict de subscribers del broadcaster → fuga de memoria por conexión.
    """
    node_id = "larco_schell"
    auth_headers = await _login_operator(
        uvicorn_server, seed_user, email="operator-unsub@cv.pe"
    )

    async with AsyncClient(base_url=uvicorn_server, timeout=5.0) as client:
        async with client.stream(
            "GET",
            f"/control/active-state/{node_id}/stream",
            headers=auth_headers,
        ) as response:
            assert response.status_code == 200
            # Yield al loop para que el endpoint complete subscribe() y la
            # primera iteración del generator quede esperando queue.get().
            await asyncio.sleep(0.1)
            # En este punto el broadcaster tiene ≥1 subscriber para node_id.
            assert node_id in broadcaster_instance._subscribers, (
                f"se esperaba un subscriber para {node_id!r} tras 200 OK; "
                f"state={broadcaster_instance._subscribers}"
            )
            assert len(broadcaster_instance._subscribers[node_id]) >= 1

    # Tras cerrar el context manager (stream y client), uvicorn dispara
    # CancelledError sobre event_generator → finally → unsubscribe.
    # Damos varios yields para que el cleanup async complete.
    for _ in range(50):
        await asyncio.sleep(0.05)
        if node_id not in broadcaster_instance._subscribers:
            break

    assert node_id not in broadcaster_instance._subscribers, (
        f"subscriber para {node_id!r} quedó colgado tras desconexión: "
        f"{broadcaster_instance._subscribers}"
    )


def test_publish_runs_after_commit_in_internal_activate_handler():
    """Verificación estructural del orden ``db.commit() → broadcaster.publish``
    en el handler dev-only ``__internal/activate`` (ver routes.py).

    Lee el source de ``setup_test_activator`` y exige que la posición léxica
    de ``db.commit()`` aparezca ANTES que ``broadcaster.publish``. Probar el
    orden con monkey-patch sobre los dos métodos es viable pero pesado para
    una invariante puramente estructural.
    """
    import inspect

    from src.control.presentation.api import routes

    setup_src = inspect.getsource(routes.setup_test_activator)
    # Buscamos los call sites únicos (no las menciones en el docstring).
    commit_call = "db.commit()  # COMMIT first."
    publish_call = "await broadcaster.publish(  # PUBLISH after commit."
    commit_idx = setup_src.find(commit_call)
    publish_idx = setup_src.find(publish_call)

    assert commit_idx > -1, (
        f"no se encontró el call site marcado {commit_call!r} en setup_test_activator"
    )
    assert publish_idx > -1, (
        f"no se encontró el call site marcado {publish_call!r} en setup_test_activator"
    )
    assert commit_idx < publish_idx, (
        f"db.commit() debe ir ANTES que broadcaster.publish (race condition); "
        f"posiciones: commit={commit_idx}, publish={publish_idx}"
    )
    full_src = inspect.getsource(routes)
    assert "COMMIT first" in full_src and "PUBLISH after commit" in full_src, (
        "comentarios inline del orden commit→publish ausentes en routes.py"
    )
