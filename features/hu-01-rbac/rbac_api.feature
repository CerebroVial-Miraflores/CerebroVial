# language: es
Característica: Control de acceso por rol en la API (CA-01.4)
  Como sistema CerebroVial
  Quiero rechazar peticiones HTTP a recursos protegidos cuando el rol del usuario
  no está autorizado para acceder a ellos
  Para evitar accesos cruzados entre Operador, Gerente y Administrador

  Antecedentes:
    Dado un endpoint de muestra "GET /api/health" protegido para el rol "admin"

  Escenario: Sin token la API responde 401
    Cuando un cliente hace "GET /api/health" sin cabecera de autorización
    Entonces la respuesta tiene código HTTP 401
    Y la respuesta no incluye información del recurso solicitado

  Escenario: Token de Operador recibe 403 con cuerpo genérico
    Dado un usuario autenticado con rol "operator"
    Cuando ese usuario hace "GET /api/health"
    Entonces la respuesta tiene código HTTP 403
    Y el cuerpo de la respuesta no menciona el nombre del recurso
    Y el cuerpo de la respuesta no indica si el recurso existe o no
    Y el cuerpo de la respuesta no menciona el rol esperado

  Escenario: Token de Gerente recibe 403 con cuerpo genérico
    Dado un usuario autenticado con rol "manager"
    Cuando ese usuario hace "GET /api/health"
    Entonces la respuesta tiene código HTTP 403
    Y el cuerpo de la respuesta no menciona el nombre del recurso
    Y el cuerpo de la respuesta no indica si el recurso existe o no

  Escenario: Token de Administrador recibe 200
    Dado un usuario autenticado con rol "admin"
    Cuando ese usuario hace "GET /api/health"
    Entonces la respuesta tiene código HTTP 200
    Y la respuesta contiene un cuerpo JSON con el estado del sistema
