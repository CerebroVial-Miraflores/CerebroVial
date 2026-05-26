# language: es
Característica: Redirección al login sin sesión activa (CA-01.5)

  # Cobertura ejecutable (entregada por TTH-01, regresión verificada en HU-01):
  #   frontend_ui/src/auth/__tests__/ProtectedRoute.test.tsx (CT-01.8)
  # Este archivo es documentación viva del CA. No agrega step definitions
  # nuevos en HU-01.

  Escenario: Usuario sin sesión es redirigido al login
    Dado que no existe sesión activa
    Cuando intenta acceder a la ruta "/dashboard"
    Entonces el navegador es redirigido a "/login"
    Y la ruta original "/dashboard" se preserva en el estado de navegación
