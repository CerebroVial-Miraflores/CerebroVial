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
  # El smoke e2e real (con un endpoint backend protegido por require_role)
  # se ejecuta de forma manual ahora que GET /api/health emite 401 al token
  # expirado. Protocolo documentado en el plan §4.2.

  Escenario: Token expirado pegando a endpoint protegido provoca redirección a login
    Dado un usuario autenticado con rol "admin"
    Y el token JWT de ese usuario ha expirado
    Cuando el frontend hace una petición a "GET /api/health"
    Entonces la respuesta es 401
    Y el frontend ejecuta logout automático
    Y el navegador es redirigido a "/login"
    Y la pantalla de login muestra el mensaje "Tu sesión expiró"
