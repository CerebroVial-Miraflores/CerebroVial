# language: es
Característica: Segregación de vistas del Operador (CA-01.1)
  Como Operador del C4
  Quiero ver únicamente las vistas de monitoreo operativo
  Para trabajar sin distractores fuera de mi rol

  # Cobertura ejecutable:
  #   - Sidebar tabs visibles: frontend_ui/src/components/layout/__tests__/Sidebar.test.tsx
  #   - Default tab y RoleGate: frontend_ui/src/__tests__/App.test.tsx
  #
  # Nota: el segundo escenario verifica la defensa de presentación del frontend
  # (RNF-SEC-06, validación dual lado cliente). NO prueba enforcement RBAC de
  # backend, que en esta HU se demuestra únicamente sobre GET /api/health vía
  # require_role (CA-01.4 / RNF-SEC-03).

  Escenario: El Operador solo ve sus vistas y no ve enlaces ajenos en la navegación
    Dado un usuario autenticado con rol "operator"
    Cuando ingresa al sistema
    Entonces ve la pestaña "Monitoreo" en la barra de navegación
    Y ve la pestaña "Motor Adaptativo" en la barra de navegación
    Y ve la pestaña "Alertas" en la barra de navegación
    Y no ve la pestaña "Analítica" en la barra de navegación
    Y no ve la pestaña "Administración" en la barra de navegación
    Y la vista activa por defecto es "Monitoreo"

  Escenario: El Operador no puede forzar el cambio a una vista de otro rol
    Dado un usuario autenticado con rol "operator"
    Cuando intenta activar manualmente la vista "analytics"
    Entonces el sistema no renderiza el contenido de "analytics"
    Y la vista activa termina en la vista por defecto del Operador
