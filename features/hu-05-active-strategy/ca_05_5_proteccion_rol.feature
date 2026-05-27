# language: es
Característica: Protección por rol del panel de estrategia activa (CA-05.5)
  Como sistema CerebroVial
  Quiero exigir autenticación con rol Operador o Administrador para leer el estado vigente
  Para impedir accesos anónimos y restringir roles no autorizados

  # CA-05.5 declara la redirección a /login si el Operador no está autenticado.
  # En el backend eso se traduce como 401 sin token; el redirect en sí es
  # responsabilidad de ProtectedRoute (HU-01) ante 401 → onUnauthorized.
  # Además, HU-05 es la primera HU que aplica require_role a una ruta consumida
  # por la UI en su flujo natural (CLAUDE.md, deuda HU-01) — esta cobertura
  # habilita el smoke e2e de CA-01.6 declarado por separado.

  Antecedentes:
    Dado existe una decisión "webster" registrada para "larco_schell"
    Y esa decisión fue activada como estrategia vigente para "larco_schell"

  Escenario: Sin token la API responde 401
    Cuando un cliente hace "GET /control/active-state/larco_schell" sin cabecera de autorización
    Entonces la respuesta tiene código HTTP 401

  Escenario: Token de Gerente recibe 403
    Dado un usuario autenticado con rol "manager"
    Cuando ese usuario hace "GET /control/active-state/larco_schell"
    Entonces la respuesta tiene código HTTP 403

  Escenario: Token de Operador recibe 200
    Dado un usuario autenticado con rol "operator"
    Cuando ese usuario hace "GET /control/active-state/larco_schell"
    Entonces la respuesta tiene código HTTP 200

  Escenario: Token de Administrador recibe 200
    Dado un usuario autenticado con rol "admin"
    Cuando ese usuario hace "GET /control/active-state/larco_schell"
    Entonces la respuesta tiene código HTTP 200
