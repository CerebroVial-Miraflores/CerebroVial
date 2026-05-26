# language: es
Característica: Segregación de vistas del Gerente (CA-01.2)
  Como Gerente
  Quiero ver únicamente la vista ejecutiva
  Para acceder al sistema con un menú coherente con mi rol

  # Cobertura ejecutable:
  #   - Sidebar tabs visibles: frontend_ui/src/components/layout/__tests__/Sidebar.test.tsx
  #   - Default tab y RoleGate: frontend_ui/src/__tests__/App.test.tsx
  #
  # Nota: este escenario NO valida el contenido de reportería de Analítica.
  # El contenido funcional de la vista del Gerente está reservado a HU-16 y HU-17.
  # Aquí se verifica únicamente que el Gerente llega a su zona y NO llega
  # a las zonas del Operador ni del Administrador.
  #
  # El segundo escenario verifica la defensa de presentación del frontend
  # (RNF-SEC-06, validación dual lado cliente). NO prueba enforcement RBAC de
  # backend, que en esta HU se demuestra únicamente sobre GET /api/health vía
  # require_role (CA-01.4 / RNF-SEC-03).

  Escenario: El Gerente solo ve su vista y no ve enlaces ajenos en la navegación
    Dado un usuario autenticado con rol "manager"
    Cuando ingresa al sistema
    Entonces ve la pestaña "Analítica" en la barra de navegación
    Y no ve la pestaña "Monitoreo" en la barra de navegación
    Y no ve la pestaña "Motor Adaptativo" en la barra de navegación
    Y no ve la pestaña "Alertas" en la barra de navegación
    Y no ve la pestaña "Administración" en la barra de navegación
    Y la vista activa por defecto es "Analítica"

  Escenario: El Gerente no puede forzar el cambio a una vista de otro rol
    Dado un usuario autenticado con rol "manager"
    Cuando intenta activar manualmente la vista "dashboard"
    Entonces el sistema no renderiza el contenido de "dashboard"
    Y la vista activa termina en la vista por defecto del Gerente
