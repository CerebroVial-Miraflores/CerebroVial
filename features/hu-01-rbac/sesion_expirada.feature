# language: es
Característica: Auto-logout por token expirado (CA-01.6 — smoke)

  # Cobertura ejecutable parcial (entregada por TTH-01, regresión verificada
  # en HU-01):
  #   - Interceptor 401 → onUnauthorized:
  #       frontend_ui/src/services/__tests__/httpClient.test.ts
  #   - SessionProvider performLogout({reason: 'session-expired'}):
  #       frontend_ui/src/auth/__tests__/SessionContext.test.tsx
  #   - Flash "Tu sesión expiró" en LoginView:
  #       frontend_ui/src/auth/__tests__/LoginView.test.tsx (CT-01.11)
  #
  # El smoke e2e en vivo (loguear, dejar expirar el token, seguir usando la
  # app, ver auto-logout) no es ejecutable en HU-01 por una razón estructural:
  # el único endpoint protegido por require_role en HU-01 es GET /api/health,
  # que la UI no consume en su flujo normal. Las dos mitades del escenario
  # están cubiertas por separado y verdes (interceptor: CT-01.11; rechazo 401:
  # los 4 escenarios pytest-bdd de CA-01.4 sobre /api/health en rbac_api.feature),
  # pero unirlas en vivo requiere un endpoint protegido que la UI consuma —
  # diferido a HU-03 / HU-05. Cuando se haga, NO usar JWT_EXPIRATION_HOURS
  # fraccionario (el servicio lo lee con int(...) y revienta con decimales);
  # usar create_access_token(..., expires_delta=timedelta(seconds=N)) desde
  # shell del backend.

  Escenario: Token expirado pegando a endpoint protegido provoca redirección a login
    Dado un usuario autenticado con rol "admin"
    Y el token JWT de ese usuario ha expirado
    Cuando el frontend hace una petición a "GET /api/health"
    Entonces la respuesta es 401
    Y el frontend ejecuta logout automático
    Y el navegador es redirigido a "/login"
    Y la pantalla de login muestra el mensaje "Tu sesión expiró"
