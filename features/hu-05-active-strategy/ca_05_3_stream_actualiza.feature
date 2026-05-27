# language: es
Característica: Actualización en vivo de la estrategia vigente vía SSE (CA-05.3)
  Como Operador con el panel abierto
  Quiero que el panel refleje el cambio de estrategia automáticamente sin recargar
  Para no perder visibilidad de la decisión actual del sistema

  # DHU-021 fija SSE como canal. El stream emite un evento mínimo "active-state-changed";
  # el cliente RE-LEE GET /control/active-state para obtener el estado autoritativo.
  # Latencia objetivo: ≤ 5 s desde el cambio en BD.

  # EJECUCIÓN AUTORITATIVA: este .feature describe el escenario pero NO se
  # ejecuta como pytest-bdd. Razón: pytest-bdd 8.1.0 no soporta steps async
  # nativamente — invoca step funcs ``async def`` sin ``await`` y las coroutines
  # quedan colgadas (warning explícito de pytest-bdd). El escenario completo
  # (Given/When/Then) se ejecuta en
  # ``core_management_api/tests/bdd/hu_05/test_hu05_sse.py::test_ca_05_3_sse_active_state_changed``
  # como plain async test (pytest-asyncio) con httpx.AsyncClient y streaming
  # real. Cuando pytest-bdd suba a una versión con soporte async, este .feature
  # se vuelve a anclar via scenarios(...) en un test_hu05_sse_bdd.py.

  Escenario: El cliente conectado al stream recibe el evento al activar una nueva decisión
    Dado un usuario autenticado con rol "operator"
    Y existe una decisión "webster" registrada para "larco_schell"
    Y un cliente abierto al stream "GET /control/active-state/larco_schell/stream"
    Cuando el sistema activa esa decisión como estrategia vigente para "larco_schell"
    Entonces el cliente recibe un evento "active-state-changed" en menos de 3 segundos
    Y el evento incluye el campo "node_id" con valor "larco_schell"
    Y el evento incluye el campo "decision_id"
