# language: es
Característica: Auto-logout e2e por token expirado sobre ruta consumida por la UI (CA-01.6 smoke)

  # Cierra la deuda declarada en CLAUDE.md ("HU-01 — alcance entregado y deuda
  # declarada"): el smoke e2e en vivo de CA-01.6 quedaba diferido a "la
  # primera HU que aplique require_role a una ruta consumida por la UI
  # (HU-03 o HU-05, lo que llegue primero)". HU-05 trajo
  # GET /control/active-state/{node_id} con require_role(OPERATOR, ADMIN) y
  # ActiveStrategyView la consume en su flujo natural; ese es el smoke
  # ejecutable que faltaba.
  #
  # Protocolo correcto (de CLAUDE.md): para forjar un token expirado NO usar
  # JWT_EXPIRATION_HOURS fraccionario — el servicio lee la env var con
  # int() y revienta con decimales. La vía correcta es invocar
  # ``create_access_token(..., expires_delta=timedelta(seconds=-N))`` desde
  # el shell del backend, que produce un token con ``exp`` ya pasado sin
  # tener que dormir el reloj del test.
  #
  # Contraparte frontend del lazo (httpClient 401 → authBridge.onUnauthorized
  # → SessionProvider.performLogout → navigate('/login')): tests vivos en
  #   frontend_ui/src/services/__tests__/httpClient.test.ts (CT-01.9)
  #   frontend_ui/src/services/__tests__/controlActiveStateService.test.ts
  #     (cubre el lazo sobre la ruta concreta de HU-05)
  #   frontend_ui/src/auth/__tests__/SessionContext.test.tsx (performLogout)
  #   frontend_ui/src/auth/__tests__/LoginView.test.tsx (CT-01.11, flash)

  Escenario: Token expirado sobre GET /control/active-state devuelve 401
    Dado un Operador con un token JWT cuya expiración ya pasó
    Cuando ese Operador hace "GET /control/active-state/larco_schell"
    Entonces la respuesta tiene código HTTP 401
