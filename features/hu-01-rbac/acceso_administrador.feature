# language: es
Característica: Segregación de vistas del Administrador (CA-01.3)
  Como Administrador
  Quiero ver únicamente la vista de configuración y salud del sistema
  Para acceder al sistema con un menú coherente con mi rol

  # Cobertura ejecutable:
  #   - Sidebar tabs visibles: frontend_ui/src/components/layout/__tests__/Sidebar.test.tsx
  #   - Default tab y RoleGate: frontend_ui/src/__tests__/App.test.tsx
  #
  # Nota: la "configuración del motor" mencionada en el texto extendido del CA
  # pertenece a HUs futuras. ControlView es asignado al Operador en HU-01.
  #
  # El segundo escenario verifica la defensa de presentación del frontend
  # (RNF-SEC-06, validación dual lado cliente). NO prueba enforcement RBAC de
  # backend, que en esta HU se demuestra únicamente sobre GET /api/health vía
  # require_role (CA-01.4 / RNF-SEC-03).

  Escenario: El Administrador solo ve su vista y no ve enlaces ajenos en la navegación
    Dado un usuario autenticado con rol "admin"
    Cuando ingresa al sistema
    Entonces ve la pestaña "Administración" en la barra de navegación
    Y no ve la pestaña "Monitoreo" en la barra de navegación
    Y no ve la pestaña "Motor Adaptativo" en la barra de navegación
    Y no ve la pestaña "Alertas" en la barra de navegación
    Y no ve la pestaña "Analítica" en la barra de navegación
    Y la vista activa por defecto es "Administración"

  Escenario: El Administrador no puede forzar el cambio a una vista de otro rol
    Dado un usuario autenticado con rol "admin"
    Cuando intenta activar manualmente la vista "control"
    Entonces el sistema no renderiza el contenido de "control"
    Y la vista activa termina en la vista por defecto del Administrador
