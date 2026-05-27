# language: es
Característica: Contrato 404 cuando un nodo no tiene estrategia vigente registrada

  # ESTE .feature NO cubre CA-05.4.
  #
  # CA-05.4 es resiliencia (DHU-005 Caso B): "el motor adaptativo deja de
  # responder → el panel mantiene en pantalla la ÚLTIMA ESTRATEGIA CONOCIDA,
  # la marca como NO CONFIRMADA e indica el TIEMPO transcurrido desde la
  # última confirmación". Es un comportamiento STATEFUL en el cliente:
  # recordar lo último visto cuando la fuente cae. Esa lógica vive en el
  # frontend (ActiveStrategyView, sseClient con backoff + staleTimer) y se
  # verifica en Fase 6 con Vitest, NO con BDD backend.
  #
  # Lo que sí necesitamos del backend es un contrato identificable para el
  # caso ESTADO INICIAL VACÍO (un nodo válido para el que nunca se activó
  # una decisión todavía, o se reseteó la BD): debe responder 404 con un
  # ``code`` estable que el frontend pueda discriminar de otros 404 y de los
  # errores de red, para decidir si mostrar "sin estrategia activa todavía"
  # vs disparar el flujo de DHU-005 Caso B.
  #
  # CA-05.4 NO tiene cobertura BDD de backend en HU-05. Su verificación
  # ejecutable es el test de Fase 6 sobre ActiveStrategyView.

  Escenario: Nodo sin fila en engine_active_state responde 404 con código no_active_state
    Dado un usuario autenticado con rol "operator"
    Y no existe ninguna estrategia activa para "larco_schell"
    Cuando ese usuario hace "GET /control/active-state/larco_schell"
    Entonces la respuesta tiene código HTTP 404
    Y el cuerpo de la respuesta tiene el código de error "no_active_state"
