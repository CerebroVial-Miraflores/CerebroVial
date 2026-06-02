# Tareas Técnicas Habilitadoras (TTH)

> Trabajo técnico de infraestructura necesario para que las HUs del Product Backlog puedan ser implementadas. **Este documento NO contiene Historias de Usuario.** Las TTH no siguen el formato "Como X, quiero Y, para Z" ni Given-When-Then.
>
> **Fundamento metodológico:** Ver `DECISIONS_HU.md`, decisiones DHU-001, DHU-003, DHU-004 (Bloque A), DHU-010 (Bloque C), DHU-013 (clasificación HU/TTH del Bloque D), DHU-014 (decisiones de redacción del Bloque D) y DHU-015 (clasificación HU/TTH del Bloque E con ampliación 4 → 5 TTH).
>
> **Fecha de creación:** 2026-05-13
> **Última actualización:** 2026-06-02 (**TTH-13 agregada (DHU-029): ampliación del catálogo a 13 TTH (TTH-01 a TTH-13)** — infraestructura de consulta histórica de congestión por arista (estado por instante arbitrario + reporte de rango temporal disponible) que habilita HU-23 (recorrido temporal de la congestión de la red). Reutiliza la geometría y la persistencia en `waze_jams` de TTH-12 sin alterar sus contratos (CT-13.5). Es un cambio sustantivo (TTH nueva), no de metadata; las 12 TTH previas no se modifican. Previa: 2026-06-02 (**TTH-12 agregada (DHU-028): ampliación del catálogo a 12 TTH (TTH-01 a TTH-12)** — infraestructura de datos de congestión por arista que habilita HU-22 (vista panorámica de congestión de la red). Es un cambio sustantivo (TTH nueva), no de metadata; las 11 TTH previas no se modifican. Previa: 2026-05-18 (**DHU-019 aplicada documentalmente al backlog:** rango DHU referenciado actualizado a "DHU-001 a DHU-019" en cabecera y "Documentos relacionados". El cambio es exclusivamente de metadata documental; las 11 TTH no reciben modificaciones sustantivas con DHU-019. DHU-019 es decisión metodológica sobre la redacción del documento `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` recién creado; las TTH se referencian desde la matriz de trazabilidad de ese documento (sección 4.2) en lugar de modificarse. La pasada aditiva sobre las HUs (subsección G de DHU-019 que agregaría referencias `→ RNF-XXX-NN` a sus secciones "Candidatos a RNF") queda como tarea separada pendiente; las TTH no son destinatarias de esa pasada porque no tienen sección "Candidatos a RNF". Última previa: 2026-05-17, DHU-018 aplicada retroactivamente al backlog: rango DHU referenciado actualizado a "DHU-001 a DHU-018" en cabecera y "Documentos relacionados". El cambio fue exclusivamente de metadata documental; las 11 TTH no recibieron Resumen ejecutivo (las TTH no son HUs y su sección Descripción + Criterios técnicos de terminado ya sirven la función de orientación de lectura). Previa a esa: 2026-05-16, cierre del MVP2 (DHU-017): rango DHU actualizado a "DHU-001 a DHU-017" en cabeceras y "Documentos relacionados". Sin cambios sustantivos al contenido de las 11 TTH: el MVP2 no introduce TTH nuevas (las 4 HUs del MVP2 ingloban su sustrato técnico como CAs conforme a DHU-017 subsección H). HU-20 del MVP2 consume CT-09.5 de TTH-09 extendido inglobadamente con persistencia paralela del modelo de respaldo; HU-21 del MVP2 consume CT-04.4 y CT-04.5 de TTH-04 para captura automática del contexto operativo al disparar escalamientos. Previa a esa: 2026-05-15, cierre del Bloque E, DHU-015. Y previa: 2026-05-14, cierre del Bloque D, DHU-013 y DHU-014.)

---

## Contexto

Durante el cierre del Bloque A se identificó que tres trabajos originalmente redactados como HUs no cumplen los criterios para ser HUs (no tienen Persona del producto como beneficiaria, su valor es instrumental, son técnicos estándar). Siguiendo el principio metodológico de no disfrazar tareas técnicas como HUs (Lullabot, Scrum.org, SAFe Enabler Stories), se trasladaron a este documento. Durante el cierre del Bloque C, dos features adicionales (F26 y F27) se clasificaron como TTH por aplicación de DHU-010. Durante la preparación del Bloque D (DHU-013), se confirmó que no se crean TTH adicionales por las features F17, F18 y F20 (las 3 se modelan como HUs operativas) y se cerró a favor de mantener íntegra a TTH-05 (no dividir entre TTH instrumental y HU del Administrador). Durante la redacción del Bloque D (DHU-014) se introdujo **TTH-06** (capa de DTOs transversal al backend) como mejora técnica clasificada como Trabajos Futuros. Durante el cierre del Bloque E (DHU-015), las cuatro features del bloque (F32, F33, F34, F35) se clasificaron como TTH por aplicación de DHU-004 (las cuatro cumplen los cuatro criterios simultáneamente: sin Persona beneficiaria directa, valor instrumental, comportamiento técnico estándar, sin valor visible al usuario en aislamiento). Adicionalmente, durante la redacción del Bloque E se identificó la necesidad de **TTH-11** (spike de investigación de hiperparámetros temporales del modelo predictivo) como prerrequisito documental de TTH-09, ampliando DHU-015 de 4 a 5 TTH.

Las TTH son entregables del proyecto y son evaluables, pero su naturaleza es distinta a las HUs:

| Atributo | HU | TTH |
|---|---|---|
| **Sujeto** | Persona del producto | (no aplica) |
| **Valor** | De negocio, visible al usuario | Instrumental, habilita HUs |
| **Formato** | Como X, quiero Y, para Z | Enunciado imperativo |
| **Criterios** | Given-When-Then | Criterios técnicos de "terminado" |
| **Estimación** | Story points (Planning Poker) | Estimación técnica directa (horas/días) |
| **Priorización** | MoSCoW + valor de negocio | Por dependencia con HUs |

---

## Índice de TTH

| Código | Título | Bloque que habilita | Clasificación | Estado actual |
|---|---|---|---|---|
| TTH-01 | Implementación de autenticación JWT con bcrypt | A (transversal a todos) | MVP1 | Pendiente |
| TTH-02 | Arquitectura Docker Compose multi-servicio | A (transversal a todos) | MVP1 | Parcial |
| TTH-03 | Repositorio Git y pipeline CI con cobertura completa | A (transversal a todos) | MVP1 | Parcial |
| TTH-04 | Lógica de fallback en cascada del sistema | C (consumida también por HU-13 del Bloque D) | MVP1 | Pendiente |
| TTH-05 | Configuración de tiempos preconfigurados para degradado nivel 3 | C | MVP1 | Pendiente |
| TTH-06 | Capa de DTOs transversal al backend | Transversal (mejora técnica) | **Trabajos Futuros** | No se construye en MVP1 |
| TTH-07 | Integración con SUMO para simulación del entorno | E (consumida también por TTH-09 y TTH-10) | MVP1 | Pendiente |
| TTH-08 | Módulo de visión computacional que produce métricas de estado | E | MVP1 | Pendiente (refactor desde cero) |
| TTH-09 | Modelo predictivo GRU servido vía API | E (consumida por HU-03, HU-04, HU-14 y TTH-10) | MVP1 | Pendiente |
| TTH-10 | Motor adaptativo de control semafórico | E (consumida por HU-05, HU-06, HU-07, HU-08, HU-15) | MVP1 | Construido, integración pendiente |
| TTH-11 | Spike de calibración de hiperparámetros temporales del modelo predictivo | E (prerrequisito documental de TTH-09) | MVP1 | Pendiente |
| TTH-12 | Infraestructura de datos de congestión por arista para la vista de mapa de red | B (habilita HU-22; ampliación, DHU-028) | MVP1 — ampliación | Pendiente |
| TTH-13 | Infraestructura de consulta histórica de congestión por arista (estado por instante + rango temporal disponible) | B (habilita HU-23; ampliación, DHU-029) | MVP1 — ampliación | Pendiente |

---

## TTH-01 — Implementación de autenticación JWT con bcrypt

**Origen:** Reemplaza a HU-01 (Autenticación al sistema) de la versión inicial del Bloque A. Ver DHU-001 en `DECISIONS_HU.md`.

**Habilita a:** Todas las HUs operativas del backlog (toda HU requiere usuario autenticado).

### Descripción

Implementar el mecanismo de autenticación del sistema usando JSON Web Tokens (JWT) firmados, con hash de contraseñas mediante bcrypt. El endpoint `POST /auth/login` recibe credenciales, valida contra la tabla `User`, y retorna un token JWT con expiración configurable. La dependency `get_current_user` decodifica el token en cada request protegido y carga el usuario activo.

### Criterios técnicos de terminado

- **CT-01.1:** Existe endpoint `POST /auth/login` que acepta `username` y `password`, retorna `{access_token, token_type, expires_in}` con código 200 si las credenciales son válidas, y código 401 con mensaje genérico "Credenciales incorrectas" si no lo son (sin revelar si el error es de usuario o de contraseña).

- **CT-01.2:** Las contraseñas en la tabla `User` se almacenan hasheadas con bcrypt (cost factor ≥ 12). Nunca en texto plano.

- **CT-01.3:** El JWT incluye al menos los claims `sub` (user_id), `role` (operator/manager/admin) y `exp` (expiración). El tiempo de expiración es configurable vía variable de entorno (default propuesto: 8 horas).

- **CT-01.4:** Existe dependency FastAPI `get_current_user` que decodifica el token del header `Authorization: Bearer <token>`, valida la firma y la expiración, y retorna el usuario activo. Si el token es inválido o expiró, responde 401.

- **CT-01.5:** Hay tests unitarios para: login con credenciales válidas, login con credenciales inválidas, decode de token válido, decode de token expirado, decode de token con firma inválida.

### Notas técnicas

- La tabla `User` y el modelo Alembic ya están creados.
- La asignación inicial de roles se hace directamente en BD para MVP1 (no hay UI de gestión de roles, default 3 del Parking Lot del Inception).
- Decisión de algoritmo de firma JWT: HS256 con secret en variable de entorno (suficiente para MVP1; RS256 con par de claves se evalúa en MVP2 si se integra con sistema externo).

### Trazabilidad con HUs que dependen de esta TTH

Toda HU operativa del backlog tiene como precondición implícita "el usuario está autenticado". Esto se hace explícito en los criterios de aceptación de cada HU con redacciones del tipo:

> *Dado que el Operador no ha iniciado sesión, cuando intenta acceder al [recurso], entonces el sistema lo redirige a la pantalla de login.*

---

## TTH-02 — Arquitectura Docker Compose multi-servicio

**Origen:** Reemplaza a HU-03 (Preparación de arquitectura del sistema) de la versión inicial del Bloque A. Ver DHU-003 y DHU-004 en `DECISIONS_HU.md`.

**Habilita a:** Todas las HUs del backlog (no se puede desarrollar ni demostrar nada sin entorno funcional).

**Decisiones técnicas relacionadas:** D-001 (Monolito modular), D-003 (Deploy Docker local).

### Descripción

Consolidar el archivo `docker-compose.yml` para que orquesta todos los servicios del sistema, con sus dependencias, red interna, volúmenes persistentes y configuración para arranque desde máquina limpia. Incluye documentación de quickstart.

### Criterios técnicos de terminado

- **CT-02.1:** El archivo `docker-compose.yml` define los servicios `core_management_api`, `edge_device`, `frontend_ui` y `db` con sus respectivas imágenes/builds, puertos expuestos, variables de entorno y dependencias (`depends_on`).

- **CT-02.2:** En una máquina limpia con Docker y Docker Compose instalados, ejecutar `docker compose up` desde la raíz del repositorio levanta todos los servicios correctamente. El frontend es accesible vía `http://localhost:<puerto>`, la API responde a `http://localhost:<puerto>/health`, la BD acepta conexiones desde la API.

- **CT-02.3:** Los servicios se comunican entre sí por nombres internos de Docker (`db`, `core_management_api`) sin configuración manual de hosts.

- **CT-02.4:** Los datos de la BD persisten en un volumen nombrado (`postgres_data` o equivalente). Tras `docker compose down && docker compose up`, los datos previos siguen disponibles.

- **CT-02.5:** Existe documento `README.md` o `QUICKSTART.md` en la raíz del repositorio con: requisitos previos, comando de arranque, URLs locales de cada servicio, comandos comunes (logs, reset de BD), troubleshooting básico. Un nuevo miembro del equipo o el comité evaluador debe quedar con el sistema funcional siguiendo solo este documento.

- **CT-02.6:** El archivo `.env.example` documenta todas las variables de entorno necesarias con valores de ejemplo no sensibles.

### Estado actual

- Existe `docker-compose.yml` parcial (referencia: estado declarado en Bloque A original).
- Pendiente: consolidar todos los servicios, validar el quickstart en máquina limpia, documentar variables de entorno.
- El `edge_device` está contenerizado por separado, coherente con D-001 (Monolito modular).

### Notas técnicas

- El criterio de "menos de 15 minutos" del Bloque A original se reemplaza por un criterio binario (CT-02.5): el quickstart funciona o no funciona, sin umbral subjetivo. Si se requiere una métrica temporal para la sustentación, se registra en la ejecución real, no como criterio de aceptación.

---

## TTH-03 — Repositorio Git y pipeline CI con cobertura completa

**Origen:** Reemplaza a HU-04 (Configuración de repositorio y entrega continua) de la versión inicial del Bloque A. Ver DHU-003 y DHU-004 en `DECISIONS_HU.md`.

**Habilita a:** Todas las HUs del backlog (sin CI, no hay garantías de calidad sobre los entregables).

### Descripción

Consolidar el repositorio Git del proyecto con convenciones de versionamiento y configurar un pipeline de Integración Continua (CI) que ejecute tests, linting y verificación de tipos en cada push a las ramas principales y en cada Pull Request.

### Criterios técnicos de terminado

- **CT-03.1:** El repositorio tiene definidas las ramas `main` (estable) y `develop` (integración). Las features se desarrollan en ramas `feature/*` que se fusionan a `develop` vía Pull Request.

- **CT-03.2:** Existe `.gitignore` que excluye archivos binarios, virtualenvs, `.env`, `__pycache__`, `node_modules`, artefactos de build y checkpoints de modelos.

- **CT-03.3:** El pipeline de CI (GitHub Actions, GitLab CI, o equivalente) se dispara automáticamente en cada push a `main` o `develop`, y en cada Pull Request hacia esas ramas.

- **CT-03.4:** El pipeline ejecuta, como mínimo:
  - Tests unitarios de `core_management_api`.
  - Tests unitarios de `edge_device`.
  - Tests unitarios de `shared/`.
  - Tests unitarios de `ia_prediction_service/`.
  - Linting (ruff o equivalente).
  - Verificación de tipos (mypy o equivalente).

- **CT-03.5:** Si todos los checks pasan, el commit se marca con estado "verde" en la plataforma. Si alguno falla, el commit se marca "rojo" y la notificación es visible en el Pull Request asociado.

- **CT-03.6:** Las ramas protegidas (`main`, `develop`) requieren que los checks de CI estén en verde antes de permitir fusión.

- **CT-03.7:** Los conflictos de fusión deben resolverse manualmente; la plataforma no permite fusión con conflictos no resueltos.

### Estado actual

- Existe repositorio Git activo.
- Cobertura de CI parcial: según item §5.5 de `DISCOVERY_2026-05-10.md`, el CI **no** corre tests de `edge_device`, `shared/`, ni `ia_prediction_service/`.
- Esta TTH formaliza la cobertura completa.

### Notas técnicas

- El criterio original de "notificación visible del fallo" se reemplaza por dos criterios concretos (CT-03.5 y CT-03.6), eliminando el cualificador subjetivo "visible".
- La política de versionamiento semántico (SemVer) se deja como decisión técnica posterior; no es bloqueante para considerar la TTH terminada.

---

## TTH-04 — Lógica de fallback en cascada del sistema

**Origen:** Reemplaza a F26 (Lógica de fallback en cascada — backend). Clasificada como TTH por aplicación de los criterios de DHU-004 al Bloque C, formalizada en DHU-010.

**Habilita a:** HU-10 (alerta activa del estado operativo), HU-11 (vista de estado de componentes) y HU-12 (explicación del modo degradado activo). Sin esta TTH, las tres HUs operativas del Bloque C no tienen estado real que consumir.

**Decisiones técnicas relacionadas:** D-001 (Monolito modular). Decisiones del Bloque C: DHU-008 (separación conceptual entre componente caído, modo degradado y lógica de fallback), DHU-010 (clasificación de F26 como TTH), DHU-011 (cobertura de F25 por composición). DHU-012 (renombrado del estado "modo seguro" a "degradado nivel 3" y del identificador interno `safe_3` a `degraded_3`).

### Descripción

Implementar el mecanismo del backend que detecta caídas de componentes del sistema, evalúa qué fallback corresponde aplicar según la condición observada, y transita al sistema entre estados operativos (operación normal, degradado nivel 1, degradado nivel 2, degradado nivel 3, falla total). Esta lógica es el motor interno que materializa la propiedad de robustez del sistema y produce el estado operativo que consumen las HUs del Bloque C (HU-10, HU-11, HU-12).

La lógica opera de forma autónoma sin intervención del Operador. El Operador es consumidor del estado resultante (vía HU-10, HU-11, HU-12), no participante del proceso de fallback. Las transiciones de estado son automáticas y deterministas según la combinación de componentes caídos.

### Criterios técnicos de terminado

- **CT-04.1:** Existe un mecanismo de health check para cada componente del sistema con impacto operativo perceptible (al menos: módulo de detección de tráfico, módulo de predicción, motor adaptativo, componente de explicación, registro de eventos). El mecanismo determina para cada componente uno de tres estados: OK, Degradado o Fuera de servicio. La frecuencia de evaluación es configurable por componente vía variable de entorno (default propuesto: 5 segundos).

- **CT-04.2:** Existe una lógica determinista que, dada la combinación actual de estados de componentes, determina el estado operativo del sistema completo en uno de cinco valores: operación normal, degradado nivel 1, degradado nivel 2, degradado nivel 3, falla total. La regla concreta de mapeo está documentada en código y en documentación técnica del sprint, e implementa al menos los tres fallbacks declarados en F26:
  - **Nivel 1:** un componente periférico (detección de tráfico) no responde. El motor adaptativo opera sin esa fuente, usando el resto de información disponible.
  - **Nivel 2:** el componente predictivo principal no responde. El sistema activa un predictor de respaldo de menor precisión que sigue produciendo predicciones vigentes.
  - **Nivel 3 (degradado nivel 3):** el motor adaptativo no responde. El sistema aplica los tiempos preconfigurados (responsabilidad de TTH-05).
  - **Falla total:** condición sin fallback aplicable. El sistema no aplica decisiones nuevas al semáforo; el último estado conocido se mantiene hasta intervención del Administrador.

- **CT-04.3:** Cada transición de estado operativo del sistema completo (entre los cinco estados de CT-04.2) genera un evento persistido de forma durable con al menos: marca de tiempo, estado anterior, estado nuevo y causa raíz (identificador del componente o condición que disparó la transición). El esquema y el comportamiento de persistencia satisfacen CA-10.7 y CA-10.8 de HU-10.

- **CT-04.4:** El estado operativo actual del sistema está expuesto a la capa de presentación mediante un endpoint estable del backend (por ejemplo, `GET /system/operational-state`) que retorna al menos: estado actual, timestamp de entrada al estado actual, componente o condición disparadora (si no es operación normal), y nivel de degradación si aplica. Este endpoint es consumido por HU-10 (banner transversal) y como referencia por HU-12 (explicación compuesta).

- **CT-04.5:** El estado individual de cada componente está expuesto mediante un endpoint del backend (por ejemplo, `GET /system/components/status`) que retorna la lista de componentes con los siguientes campos por componente:
  1. **Nombre legible** (consumido por HU-11 y HU-13).
  2. **Estado cualitativo** (OK / Degradado / Fuera de servicio; consumido por HU-11 y HU-13).
  3. **Timestamp de último cambio de estado** (consumido por HU-11 y HU-13).
  4. **Identificador interno técnico del componente** (consumido por HU-13).
  5. **Latencia de respuesta del componente en la última evaluación de salud, en milisegundos** (consumido por HU-13). Si el componente no respondió en la última evaluación, este campo se reporta como "no aplicable" o el valor especial documentado para ese caso, no como cero (cero indica respuesta instantánea válida, no ausencia de respuesta).
  6. **Indicador de fallos recientes** definido como número de evaluaciones fallidas en una ventana temporal configurable, por defecto los últimos 5 minutos (consumido por HU-13).
  7. **Timestamp de la última evaluación de salud exitosa** (consumido por HU-13). Distinto del timestamp de último cambio de estado: el primero refleja "cuándo respondió bien por última vez"; el segundo refleja "cuándo cambió su estado cualitativo por última vez".

  Este endpoint es consumido por HU-11 (vista de estado de componentes del Operador) y por HU-13 (vista técnica de salud de componentes del Administrador), con la presentación adaptada en cada caso (DHU-013): HU-11 muestra los campos 1, 2 y 3 con resalte visual; HU-13 muestra los 7 campos. Los campos adicionales 4 a 7 se ampliaron al contrato original de CT-04.5 al cerrar HU-13 (cierre del Bloque D, DHU-014). Los campos técnicos adicionales no son sensibles; viajan en el wire incluso a consumidores con otros roles, pero el RBAC impide el acceso a las rutas que los renderizan.

- **CT-04.6:** La activación de un fallback no se detiene ni se posterga por fallos del registro de auditoría. Si la escritura del evento de transición (CT-04.3) falla momentáneamente, la transición de estado se aplica de todos modos a la operación del sistema y se registra en un mecanismo de respaldo (cola en memoria, archivo local, según se cierre en el sprint) para ser persistida cuando el registro vuelva a estar disponible. Esta resiliencia satisface CA-10.8 de HU-10.

- **CT-04.7:** Las transiciones de estado son **atómicas**: el sistema no queda en estados intermedios o indefinidos durante el cambio. Si la transición no puede completarse íntegramente (por ejemplo, el fallback no puede activarse), el sistema permanece en el estado anterior y registra una entrada de error en el registro de transiciones.

- **CT-04.8:** El comportamiento por defecto cuando el propio mecanismo de detección de salud falla (por ejemplo, el componente que ejecuta los health checks deja de responder) es **conservador**: el sistema reporta el estado operativo como "no confirmado" y notifica vía el endpoint de CT-04.4 con un flag específico, sin asumir falsamente operación normal. Esta resiliencia satisface CA-10.9 de HU-10.

- **CT-04.9:** Existen tests unitarios para: detección de cambio de estado de un componente individual, transición de operación normal a cada uno de los cuatro estados no normales, transición entre estados no normales (escaladas y recuperaciones parciales), recuperación a operación normal desde cada estado no normal, resiliencia del registro de auditoría (CT-04.6), atomicidad de transiciones fallidas (CT-04.7) y comportamiento por defecto ante fallo del mecanismo de salud (CT-04.8).

- **CT-04.10:** Existen tests de integración que validan, en un escenario simulado de degradación progresiva (componente periférico → componente predictivo → motor adaptativo), la activación correcta de cada nivel de fallback en cascada y la coherencia entre el estado operativo expuesto por el endpoint de CT-04.4 y los estados individuales expuestos por el endpoint de CT-04.5.

### Notas técnicas

- **Patrón arquitectónico sugerido:** Circuit Breaker para cada componente monitoreado, con orquestador central que agrega los estados individuales en estado operativo del sistema completo. Decisión final del patrón se cierra en el sprint, pero el principio de aislamiento de fallas y agregación centralizada queda anclado.
- **Detección de salud:** cada componente expone un endpoint `/health` o equivalente que la TTH-04 consulta periódicamente. La alternativa de tener cada componente reportando proactivamente su salud (push) se evalúa como variante del sprint, pero el contrato (cuándo se considera caído) debe ser explícito y configurable.
- **Diferencia entre el registro de TTH-04 y el de HU-08:** TTH-04 persiste transiciones de estado operativo del sistema completo. HU-08 persiste decisiones del motor adaptativo (estrategia A → estrategia B). Son registros distintos con esquemas distintos. Una transición a degradado nivel 3 registrada por TTH-04 puede coincidir temporalmente con el cese de decisiones del motor en HU-08; ambos registros coexisten y son consultables por separado.
- **Persistencia de configuración de fallbacks:** los criterios concretos que detonan cada fallback (qué componente, qué timeout, qué umbrales) son configurables sin redeploy. La configuración detallada se cierra en el sprint y queda documentada con la TTH.
- **Composición con TTH-05:** cuando el sistema entra en degradado nivel 3, la TTH-04 consulta la configuración de tiempos preconfigurados que provee TTH-05 y aplica esos tiempos al semáforo. Sin TTH-05, el nivel 3 no es completamente operativo. Las dos TTH deben terminarse antes de declarar el Bloque C operativo.
- **Catálogo de estados:** los identificadores internos de los cinco estados operativos (`normal`, `degraded_1`, `degraded_2`, `degraded_3`, `total_failure`) son los que aparecen en el registro de transiciones (CT-04.3) y en el endpoint de CT-04.4. Los nombres legibles que ve el Operador se traducen en presentación (HU-10, HU-12). El identificador `degraded_3` fue renombrado desde `safe_3` por DHU-012.

### Trazabilidad con HUs y otras TTH

| HU / TTH que depende | Cómo depende |
|---|---|
| HU-10 | Consume endpoint de estado operativo (CT-04.4). Persistencia de transiciones inglobada en CA-10.7 y CA-10.8 es responsabilidad de CT-04.3 y CT-04.6. Resiliencia ante caída del monitor (CA-10.9) es CT-04.8. |
| HU-11 | Consume endpoint de estado de componentes (CT-04.5), campos 1 a 3 (nombre, estado cualitativo, timestamp de último cambio). Ignora los campos técnicos adicionales 4 a 7. |
| HU-12 | Consume causa raíz (componente disparador y nivel) del endpoint de CT-04.4 para producir explicación compuesta. |
| HU-13 (Bloque D) | Consume el endpoint de estado de componentes (CT-04.5), los 7 campos. Misma fuente que HU-11 con presentación técnica distinta (DHU-013). |
| TTH-05 | TTH-04 consume la configuración de tiempos preconfigurados provista por TTH-05 al activar el nivel 3. |
| TTH-08 (Bloque E) | TTH-04 detecta caída del módulo de visión (vía health check de CT-04.1) y activa Nivel 1 de la cascada. Esta cascada es contractualmente preservada en MVP1 pero no se ejercita operativamente en validación cuantitativa, conforme a D-007 (visión no entra al loop cuantitativo). |
| TTH-09 (Bloque E) | TTH-04 detecta caída del modelo predictivo GRU (vía health check de CT-04.1) y activa Nivel 2 de la cascada, invocando al **RandomForest baseline preservado** por TTH-09 como predictor de respaldo. El RandomForest no es TTH propia; es componente preexistente preservado con rol declarado en TTH-09 nota técnica. |
| TTH-10 (Bloque E) | TTH-04 detecta caída del motor adaptativo (vía health check de CT-04.1 o por respuesta HTTP 422 `webster_infeasible` de CT-10.5) y activa Nivel 3 de la cascada, invocando los tiempos preconfigurados de TTH-05. |

---

## TTH-05 — Configuración de tiempos preconfigurados para degradado nivel 3

**Origen:** Reemplaza a F27 (Configuración de tiempos fijos para degradado nivel 3). Clasificada como TTH por aplicación de los criterios de DHU-004 al Bloque C, formalizada en DHU-010.

**Habilita a:** TTH-04 cuando el sistema entra en degradado nivel 3. Sin TTH-05, el nivel 3 de TTH-04 no tiene tiempos que aplicar al semáforo.

**Decisiones técnicas relacionadas:** D-001 (Monolito modular). DHU-012 (renombrado del estado y del identificador interno). DHU-013 (cierre de la pregunta abierta sobre división de TTH-05 entre TTH instrumental y HU del Administrador).

### Descripción

Implementar el mecanismo de configuración persistente de los tiempos fijos del semáforo que se aplican cuando el sistema entra en degradado nivel 3. El Administrador del Sistema define estos tiempos previamente, durante la configuración inicial del sistema y eventualmente al recalibrar los tiempos del nivel 3. Los tiempos se persisten en la base de datos y están disponibles para consulta por TTH-04 cuando se requiera activar el nivel 3.

A diferencia de las HUs operativas del Bloque C, F27 sí podría haber sido modelada como HU del Administrador (tiene Persona beneficiaria directa). Sin embargo, DHU-010 la clasificó como TTH por dos razones: (a) es configuración de un parámetro técnico interno del sistema (valores numéricos por dirección), no funcionalidad operativa, y (b) en aislamiento no entrega valor al usuario; su valor se materializa solo cuando se activa el degradado nivel 3, lo cual ocurre dentro de la lógica de TTH-04.

El alcance funcional de la configuración para MVP1 es deliberadamente simple: una tabla única de tiempos por dirección y fase, sin franjas horarias diferenciadas. La extensión a franjas horarias (mañana pico, valle, tarde pico, nocturno) se evalúa en trabajo futuro.

### Criterios técnicos de terminado

- **CT-05.1:** Existe una tabla persistente en la base de datos que almacena los tiempos preconfigurados del degradado nivel 3. El esquema mínimo incluye: identificador único, dirección o acceso de la intersección, fase del semáforo (verde, amarillo, opcionalmente rojo si se modela explícitamente), duración en segundos, marca de tiempo de creación y marca de tiempo de última modificación.

- **CT-05.2:** Existe un endpoint del backend (por ejemplo, `GET /admin/fallback-times-config` y `PUT /admin/fallback-times-config`) accesible exclusivamente al rol Administrador (RBAC según HU-01 del Bloque A) que permite consultar y actualizar la configuración de tiempos. Tanto la consulta como la actualización requieren autenticación válida (JWT vigente, según TTH-01).

- **CT-05.3:** Existe una interfaz mínima en el frontend que permite al Administrador visualizar la configuración actual y editarla. La interfaz valida que los tiempos sean numéricos, positivos y dentro de rangos razonables (por ejemplo: 5 a 120 segundos por fase). Valores fuera de rango son rechazados con mensaje claro al Administrador.

- **CT-05.4:** El endpoint de consulta de la configuración (`GET`) está disponible para consumo por TTH-04 cuando el sistema activa el degradado nivel 3. La latencia de la consulta debe ser suficiente para no demorar la transición al nivel 3 (criterio: ≤ 1 segundo en condiciones normales de carga).

- **CT-05.5:** Si la configuración no existe (sistema recién instalado, base de datos vacía), el endpoint de consulta retorna un conjunto de valores **por defecto seguros** (tiempos conservadores que se sabe que mantienen la intersección operativa, por ejemplo 30 segundos por fase principal). Esto garantiza que el degradado nivel 3 pueda activarse incluso antes de la primera configuración explícita del Administrador.

- **CT-05.6:** Cada actualización de la configuración registra una entrada de auditoría con: timestamp, identidad del Administrador que modificó, valores anteriores y valores nuevos. Esta auditoría es independiente del registro de transiciones de estado operativo de TTH-04.

- **CT-05.7:** Existen tests unitarios para: lectura de configuración existente, lectura cuando no existe configuración (valores por defecto de CT-05.5), actualización válida con auditoría correcta, rechazo de valores fuera de rango (CT-05.3), control de acceso por rol (un Operador o Gerente intentando leer o modificar la configuración recibe 403).

- **CT-05.8:** Existe un test de integración que valida: configuración guardada por el Administrador, posterior activación de degradado nivel 3 vía TTH-04, y aplicación efectiva de los tiempos configurados al semáforo simulado.

### Notas técnicas

- **Alcance MVP1 (tabla única vs franjas horarias):** F27 mencionaba como pregunta abierta si la tabla sería única o por franjas horarias. Para MVP1 se cierra en tabla única (un solo conjunto de tiempos aplicable siempre). La extensión a franjas horarias queda documentada como mejora futura, no en MVP2 explícitamente, pero como ítem identificado para iteraciones posteriores.
- **Valores por defecto seguros (CT-05.5):** los valores concretos del default se cierran en el sprint con criterio conservador: tiempos que un operador de tráfico clásico aceptaría como razonables para una intersección típica de 4 accesos. La definición exacta se documenta junto con la TTH al implementarse.
- **Composición con TTH-04:** cuando TTH-04 activa degradado nivel 3, consulta el endpoint de CT-05.4 y aplica los tiempos obtenidos. Si el endpoint no responde (caso patológico de la BD caída a la vez que el motor adaptativo), TTH-04 aplica los valores por defecto seguros como último recurso. Esta resiliencia se evalúa en el sprint y puede agregarse como CT adicional si se considera necesario.
- **Relación con el control de acceso (CA-01.4):** los endpoints de CT-05.2 están protegidos por RBAC según HU-01 del Bloque A. Un Operador o Gerente que intente acceder recibe HTTP 403. Esto se valida explícitamente en CT-05.7.
- **No es vista del Operador:** la configuración de tiempos preconfigurados es exclusivamente del Administrador. El Operador nunca interactúa con TTH-05 directamente; consume sus efectos cuando el degradado nivel 3 se activa, vía HU-12 (explicación) y HU-10 (alerta).
- **Auditoría de cambios (CT-05.6):** este registro es independiente del registro de transiciones de TTH-04 y del registro de decisiones de HU-08. Captura cambios de configuración del sistema, no eventos operativos. Se consulta principalmente para responder preguntas tipo "quién cambió la configuración del degradado nivel 3 y cuándo".
- **Pregunta abierta cerrada por DHU-013:** versiones anteriores de esta nota indicaban que TTH-05 podría dividirse en (a) parte instrumental que sigue siendo TTH y (b) HU del Administrador para el formulario y la auditoría, condicional a la revisión durante el Bloque D. DHU-013 cerró esta pregunta a favor de **mantener TTH-05 íntegra**, sin dividir, porque (i) mezclar la configuración de tiempos del nivel 3 con la HU de F20 ("Configuración del motor adaptativo") rompería la cohesión de esa HU, (ii) la separación arquitectónica entre TTH-04 (lógica de fallback) y TTH-05 (provee tiempos al nivel 3) se preserva mejor manteniéndolas juntas, y (iii) si en el futuro surge una necesidad concreta del Administrador (por ejemplo, una vista dedicada de auditoría de cambios), siempre se puede extraer una HU adicional sin perder nada.

### Trazabilidad con HUs y otras TTH

| HU / TTH que depende | Cómo depende |
|---|---|
| TTH-04 | Consume el endpoint de configuración (CT-05.4) cuando activa degradado nivel 3. |
| HU-01 (Bloque A) | Provee el control de acceso por rol que protege los endpoints de TTH-05 (CT-05.2). |
| TTH-01 | Provee la autenticación JWT que valida el acceso del Administrador (CT-05.2). |

---

## TTH-06 — Capa de DTOs transversal al backend

**Origen:** Identificada durante la redacción de HU-13 (Bloque D) en la discusión sobre el patrón de consumo del endpoint compartido de CT-04.5 por HU-11 (Operador) y HU-13 (Administrador). Formalizada en DHU-014.

**Habilita a:** mejora transversal de mantenibilidad y estabilidad del contrato API del backend. **No bloquea ninguna HU del MVP1.**

**Clasificación: Trabajos Futuros.** No se construye dentro del alcance del proyecto académico. Se menciona en el capítulo de trabajo futuro del documento de tesis si se considera relevante.

**Decisiones técnicas relacionadas:** D-001 (Monolito modular). DHU-014 (consolidación de decisiones del Bloque D, sección sobre creación de TTH-06).

### Descripción

La práctica actual del proyecto, observable en el código existente al cierre del Bloque D, es serializar directamente el modelo de dominio en algunos endpoints del backend (típicamente mediante respuestas que reflejan la estructura interna de los objetos persistidos). Esta práctica es funcional pero acopla el contrato de la API al modelo de dominio: cualquier evolución del modelo se propaga al contrato observable por los consumidores (frontend, otros servicios).

TTH-06 propone introducir una capa explícita de DTOs (Data Transfer Objects) entre el modelo de dominio y el contrato de la API. Cada endpoint del backend respondería con un DTO específico definido como tipo separado del modelo de dominio, con tres beneficios principales:

1. **Estabilidad del contrato:** el modelo de dominio puede evolucionar (refactorización, agregar campos internos, optimizaciones de persistencia) sin romper a los consumidores, mientras el DTO mantenga su contrato.

2. **Filtrado controlado:** si en el futuro aparecen campos sensibles en el modelo de dominio (información personal, IPs, logs internos), el DTO los excluye por construcción en lugar de depender de filtrado en el frontend o de configuración por rol del token.

3. **Documentación implícita:** los DTOs son la documentación más fiel del contrato de la API; un nuevo desarrollador entiende qué consume y qué expone cada endpoint con solo leer los tipos.

### Por qué se clasifica como Trabajos Futuros y no MVP2

Aplicando los criterios de DHU-012 sobre la semántica de MVP2 (HU completa documentada, construcción condicional a holgura tras MVP1):

1. **TTH-06 no realiza ningún Objetivo del Producto.** Los 4 Objetivos del Producto (observar, anticipar, adaptar, demostrar mejora) se cumplen con o sin esta capa. La introducción de DTOs es higiene técnica de mantenibilidad, no funcionalidad de valor.

2. **El alcance es transversal a todo el backend.** No es "una capa en un endpoint"; es refactor amplio que toca todos los endpoints del sistema. El costo de hacerlo "si hay holgura" es difícil de acotar.

3. **Naturalmente pertenece a la productivización del sistema.** En una iteración de producto real, donde el backend evoluciona y se expone a múltiples consumidores externos, la capa DTO se justifica claramente. En el alcance académico, donde el backend tiene un consumidor único (frontend propio) y vida útil acotada, el costo/beneficio es desfavorable.

4. **El sistema sin TTH-06 sigue siendo defendible académicamente.** La ausencia de capa DTO no es un defecto que el jurado vaya a observar; es una decisión razonable para un proyecto de tesis con alcance limitado.

Si se decidiera en el futuro construir TTH-06 (por ejemplo, en una iteración postacadémica del sistema para productivización), los criterios técnicos de terminado quedarían como esbozo:

### Criterios técnicos de terminado (esbozo, no se ejecuta en MVP1)

- **CT-06.1:** Cada endpoint del backend (de todos los routers) responde con un DTO Pydantic explícito, definido en un módulo separado del modelo de dominio. Ningún endpoint serializa directamente un modelo SQLAlchemy.
- **CT-06.2:** Para los endpoints de consulta (GET), los DTOs declaran explícitamente qué campos del modelo de dominio se exponen. Campos del modelo no listados en el DTO no aparecen en la respuesta.
- **CT-06.3:** Para los endpoints de mutación (POST, PUT, PATCH), los DTOs declaran explícitamente qué campos puede modificar el cliente. Campos del modelo no listados en el DTO no se aceptan en el body.
- **CT-06.4:** Existe documentación del patrón en el README técnico del backend, con un ejemplo guía para cada tipo de endpoint y los criterios de cuándo crear un DTO nuevo vs reutilizar uno existente.
- **CT-06.5:** Existen tests que verifican que los DTOs no exponen campos no declarados, incluso si el modelo de dominio cambia.

### Notas técnicas

- **No introduce TTH bloqueante:** todas las HUs y TTH del MVP1 son construibles sin TTH-06. La ausencia de DTOs no impide implementar HU-11, HU-13, HU-14, HU-15 ni cualquier otra HU del backlog.
- **No reabre decisiones de HUs del MVP1:** las HUs ya redactadas no mencionan DTOs en sus criterios de aceptación (son agnósticas a la implementación, DHU-006). La decisión sobre si el backend usa DTOs o no es 100% técnica y vive en este documento.
- **Relación con HU-13:** la discusión sobre el patrón de consumo de CT-04.5 (un endpoint, dos consumidores con presentación distinta) motivó identificar TTH-06 pero no la requiere. Para los campos técnicos adicionales que HU-13 expone (latencia, fallos recientes, identificador interno), se evaluó que **no son sensibles**, por lo cual no se justifica DTO específico ni filtrado en backend según el rol del token; el RBAC a nivel de ruta es suficiente.

### Trazabilidad con HUs y otras TTH

| HU / TTH que depende | Cómo depende |
|---|---|
| (ninguna del MVP1) | TTH-06 no es prerrequisito de ninguna HU ni TTH del MVP1. |

---

## TTH-07 — Integración con SUMO para simulación del entorno

**Origen:** Reemplaza a F32 (Integración con SUMO para simulación del entorno). Clasificada como TTH por aplicación de los criterios de DHU-004 al Bloque E, formalizada en DHU-015.

**Habilita a:** TTH-09 (provee dataset de entrenamiento), TTH-10 (provee escenarios de validación y entorno de simulación end-to-end al motor adaptativo) y al capítulo de validación cuantitativa de la tesis (provee los KPIs comparativos con/sin sistema). Sin esta TTH, ni el modelo predictivo puede entrenarse sobre dataset propio (D-008), ni el motor adaptativo puede validarse cuantitativamente, ni el documento de tesis tiene resultados experimentales que reportar.

**Decisiones técnicas relacionadas:** D-001 (Monolito modular: el componente SUMO vive como módulo del sistema, no como servicio independiente). D-005 (Números de tesis: reportar la realidad medida en validación, no las cifras de relleno). D-006 (GRU univariado por intersección: dictado por SUMO sobre una sola intersección). D-008 (SUMO como columna vertebral del sistema de datos: dataset de entrenamiento y escenarios de validación). D-009 (jam level Waze como variable predicha: el mapeo SUMO → jam_level vive aquí). DHU-015 (clasificación de F32 como TTH del Bloque E).

### Descripción

Implementar la integración del sistema con SUMO (Simulation of Urban MObility) como columna vertebral de datos del proyecto, conforme a D-008. La integración cubre tres responsabilidades técnicas distintas pero relacionadas, las tres internas al sistema y sin Persona del producto beneficiaria directa:

1. **Topología de la intersección de estudio** cargada en SUMO como red de simulación válida.
2. **Escenarios de demanda** que reproducen condiciones de tráfico distintas (al menos hora pico AM, hora pico PM, valle, fin de semana), con seeds reproducibles, para alimentar tanto la generación del dataset de entrenamiento del modelo predictivo (consumido por TTH-09) como los escenarios de validación cuantitativa del sistema integrado.
3. **Integración bidireccional vía TraCI** con el motor adaptativo (TTH-10): el motor consume el estado observado de la simulación (flujo, velocidad, longitud de cola por dirección, derivables al jam level 0-5 según D-009) y emite decisiones de fase del semáforo que la simulación ejecuta sobre el semáforo virtual.

El componente no es expuesto a ninguna Persona del producto directamente. Su salida se materializa en (a) el dataset persistido que TTH-09 consume, (b) el entorno de simulación que TTH-10 consume durante validación, y (c) los KPIs comparativos que se reportan en el capítulo de validación de la tesis.

**Alcance MVP1 sobre la intersección de estudio.** La intersección específica de Miraflores que se simula **no está elegida al momento de redactar esta TTH**. El alcance MVP1 declara una **topología genérica de cuatro accesos** con parámetros típicos de una intersección urbana semaforizada de Miraflores (carriles por acceso, velocidad de flujo libre, longitudes de aproximación). La calibración contra una intersección específica identificable se evalúa como ajuste posterior si surge en el cronograma; no condiciona la conclusión del MVP1. El capítulo de validación de la tesis declara explícitamente este alcance.

**Alcance MVP1 sobre datos reales.** Los datos reales de Waze sobre la intersección de estudio no son insumo de esta TTH en MVP1 (D-008). Quedan como posibilidad de validación adicional condicional a un acuerdo con la municipalidad de Miraflores, documentada como nota técnica al final de esta TTH y como trabajo futuro asociado a F38.

### Criterios técnicos de terminado

- **CT-07.1:** Existe una topología SUMO de una intersección genérica de cuatro accesos semaforizada, cargable por SUMO sin errores. La topología declara explícitamente al menos: número de carriles por acceso, longitud de cada acceso, velocidad máxima por carril (interpretable como velocidad de flujo libre para el mapeo de D-009), conexiones de giro permitidas y el programa del semáforo (fases, duraciones iniciales). Los parámetros utilizados son típicos de una intersección urbana semaforizada de Miraflores y están documentados en un archivo de configuración legible (no hardcodeados en un único script).

- **CT-07.2:** Existen al menos **cuatro patrones de demanda** definidos como escenarios SUMO independientes, correspondientes a: (a) hora pico de la mañana, (b) hora pico de la tarde, (c) periodo valle, (d) condiciones de fin de semana. Cada patrón declara explícitamente: distribución de flujos vehiculares por dirección y por giro, duración del escenario en tiempo simulado, y rango de seeds reproducibles. Los flujos elegidos son tales que **al menos un patrón** genera condiciones de congestión observables (jam level alcanzando valores ≥3 según D-009) y al menos un patrón mantiene condiciones de flujo libre o congestión baja (jam level ≤2). Esta diversidad es necesaria para que el dataset de entrenamiento de TTH-09 cubra todo el rango de la variable predicha.

- **CT-07.3:** Existe un mecanismo automatizado de generación del dataset de entrenamiento de TTH-09, ejecutable como script reproducible, que: (a) corre múltiples seeds por cada patrón de demanda de CT-07.2; (b) registra a frecuencia configurable (cierre por TTH-11: ver nota técnica de granularidad del dataset) las variables observadas por dirección y por carril relevantes para construir series temporales (al menos velocidad media, número de vehículos, longitud de cola); (c) calcula el ratio velocidad/free-flow y el jam level 0-5 derivado según el mapeo de D-009; (d) persiste el resultado en un formato tabular estable (CSV, Parquet o equivalente) con un esquema documentado. El dataset resultante incluye marca de seed, marca de patrón y marca de tiempo simulado en cada registro, lo cual permite particiones reproducibles de entrenamiento/validación por TTH-09.

- **CT-07.4:** Las particiones que TTH-09 utilizará para entrenamiento y validación se generan con **seeds distintos y, preferentemente, patrones de demanda con distinta combinación**, conforme a D-008 ("escenarios SUMO distintos para evitar fuga de información"). El script de generación del dataset declara explícitamente qué seeds y patrones componen cada partición; ningún seed aparece en ambas particiones.

- **CT-07.5:** Existe una integración bidireccional con SUMO vía TraCI accesible desde el motor adaptativo (TTH-10): (a) el motor puede consultar el estado observado de la simulación en tiempo de simulación (al menos velocidad media, número de vehículos y longitud de cola por dirección, suficiente para derivar el jam level según D-009); (b) el motor puede emitir decisiones de fase del semáforo y la simulación las ejecuta sobre el semáforo virtual en el paso correspondiente. La integración es funcional end-to-end: una corrida demostrable enciende SUMO, el motor adaptativo consume estado, el motor decide fase, la simulación responde, y los KPIs resultantes son capturables al final de la corrida.

- **CT-07.6:** El conjunto de KPIs comparativos del capítulo de validación de la tesis es producible mediante el componente. Mínimo: tiempo medio de viaje por vehículo, longitud máxima y media de cola por dirección, demora acumulada en la intersección, throughput (vehículos por unidad de tiempo). Los KPIs son capturables tanto para corridas "con sistema integrado" (motor adaptativo activo) como para corridas "control fijo" (programa Webster fijo o equivalente declarado en la tesis), con la misma topología y los mismos patrones de demanda, para permitir comparación cuantitativa.

- **CT-07.7:** Existe documentación técnica reproducible del componente, incluyendo: (a) cómo cargar la topología, (b) cómo ejecutar cada patrón de demanda, (c) cómo regenerar el dataset de entrenamiento, (d) cómo ejecutar una corrida end-to-end con el motor adaptativo conectado, (e) cómo ejecutar una corrida de control fijo para comparación. La documentación está en el repositorio (README técnico del módulo SUMO o equivalente) y permite a un tercero familiarizado con SUMO reproducir todos los resultados sin asistencia del autor.

- **CT-07.8:** Existen tests de integración mínimos: (a) la topología se carga sin errores; (b) cada patrón de demanda corre N pasos sin errores; (c) la integración TraCI con el motor adaptativo ejecuta una corrida corta end-to-end sin errores (motor consume estado, decide fase, simulación responde, KPIs capturados). Los tests no validan calidad de resultados (eso pertenece al capítulo de validación), solo que la mecánica end-to-end funciona.

### Estado actual

🆕 **Por construir desde cero.** No hay código SUMO ni topología cargada en el repositorio al momento de redactar esta TTH (mayo 2026). Se ha realizado exploración inicial de la herramienta SUMO sin entregables aún en el proyecto; el avance exploratorio no se considera material entregable y la TTH se redacta como si partiera de cero.

Esta TTH es el **cuello de botella cronológico declarado del MVP1** según la ficha F32. Su inicio temprano en el cronograma es prerrequisito de la conclusión del MVP1.

### Notas técnicas

- **Topología genérica como alcance MVP1.** La decisión de operar sobre una topología genérica de 4 accesos en lugar de calibrar contra una intersección específica es deliberada y conservadora para MVP1. Razones: (a) no hay decisión cerrada sobre qué intersección específica de Miraflores es la unidad de análisis; (b) el alcance académico (D-008) no requiere fidelidad geográfica fina; requiere consistencia metodológica y reproducibilidad; (c) mantener la topología genérica desacopla esta TTH de dependencias administrativas que pueden no concretarse en el cronograma.

- **Mapeo SUMO → jam level.** El mapeo concreto vive dentro de esta TTH como utilidad: `mean_speed_mps / max_speed_mps → ratio → jam_level (0-5)` según los umbrales declarados en D-009 (90%, 70%, 50%, 30%, 0%, con anclajes documentados en el paper JamVis). El máximo `max_speed_mps` se obtiene en MVP1 de `traci.lane.getMaxSpeed` o del archivo de red como aproximación inicial; la calibración fina del percentil 85/95 sobre histórico queda como mejora posterior, no bloqueante para MVP1.

- **Granularidad del dataset (frecuencia de muestreo).** La frecuencia de muestreo del dataset (CT-07.3) se cierra con la salida de TTH-11 (spike de hiperparámetros temporales). Hasta que TTH-11 cierre, el valor provisional a usar es **60 segundos simulados por muestra**, consistente con el Δt_in provisional declarado en TTH-09 CT-09.2. La frecuencia definitiva la confirma o ajusta TTH-11 antes del cierre del MVP1.

- **Reproducibilidad por seed.** El uso de seeds explícitos por escenario es lo que permite que TTH-09 declare particiones independientes sin riesgo de fuga de información. El script de generación del dataset documenta los seeds utilizados y las particiones resultantes son recreables byte-a-byte ante una nueva ejecución.

- **No hay validación cualitativa de "plausibilidad" como CT.** Se evaluó incluir un CT del tipo "los KPIs producidos son plausibles según criterio cualitativo de ingeniería de tráfico". La decisión fue **no** incluirlo como CT formal porque el criterio es subjetivo y la plausibilidad se evaluará naturalmente durante la calibración del trabajo y durante el capítulo de validación de la tesis. Si durante la implementación se detecta que los escenarios producen KPIs claramente aberrantes (cola infinita, tiempos de viaje absurdos), se ajustan los parámetros de demanda antes de generar el dataset final.

- **Datos reales de Waze como bono futuro.** D-008 menciona la posibilidad de obtener datos reales de Waze sobre la intersección de estudio mediante acuerdo con la municipalidad de Miraflores. Esta TTH **no depende** de obtener esos datos y su Done no se condiciona a su disponibilidad. Si los datos se obtienen durante o después del cierre del MVP1, su incorporación se aplica como (a) validación adicional del modelo entrenado en SUMO frente a una distribución real y (b) calibración fina de los umbrales del jam level y de los parámetros de demanda de SUMO. La conversación con el PO de la municipalidad se realiza después de tener el componente construido, no antes. Declarado como F38 / Trabajos Futuros.

- **Estado actual del motor adaptativo y consecuencias para CT-07.5.** El motor adaptativo (TTH-10) está significativamente construido al momento de redactar esta TTH según `EVOLUCION_TESIS.md` Fase 3. La integración TraCI declarada en CT-07.5 requiere coordinación con TTH-10 para definir la interfaz exacta de consumo. La decisión concreta entre adaptador externo y modificación del motor para consumir TraCI directamente se cierra al implementar las dos TTH juntas.

- **No se documenta el motor de control fijo como TTH separada.** CT-07.6 menciona corridas de "control fijo" para comparación cuantitativa. La implementación del control fijo es parte de esta TTH o del setup del experimento de validación; no se modela como TTH separada porque es un baseline trivial sin valor instrumental propio fuera del experimento de validación de la tesis.

### Trazabilidad con HUs y otras TTH

| HU / TTH que depende | Cómo depende |
|---|---|
| TTH-09 (Bloque E) | Consume el dataset persistido producido por CT-07.3, con particiones definidas por CT-07.4. Sin TTH-07 no hay dataset propio sobre el cual entrenar el modelo (D-008). |
| TTH-10 (Bloque E) | Consume el entorno de simulación end-to-end vía TraCI según CT-07.5. Sin TTH-07 el motor no tiene contra qué probarse cuantitativamente. |
| TTH-11 (Bloque E) | La parte empírica del spike (CT-11.4) consume el dataset producido por TTH-07. La parte bibliográfica es independiente. |
| HU-02 (monitoreo, Bloque B) | No depende directamente de TTH-07 en su contrato (HU-02 es agnóstica a la fuente según DHU-006). En MVP1, la fuente de medición que HU-02 consume es funcionalmente el estado de SUMO derivado a flujo/cola/jam level (D-008); en operación hipotética, sería TTH-08 (visión). El contrato de HU-02 no se modifica por TTH-07. |
| HU-03 (predicción, Bloque B), HU-04 (vista combinada) | Consumen indirectamente, vía TTH-09: las predicciones que HU-03 muestra al Operador provienen del modelo entrenado sobre el dataset que TTH-07 genera. |
| HU-14 (métricas del modelo, Bloque D) | Consume indirectamente, vía TTH-09: las métricas que el Administrador ve son las del modelo entrenado y evaluado sobre el dataset de TTH-07. |
| Capítulo de validación de la tesis | Consume directamente los KPIs comparativos producidos por CT-07.6. |

---

## TTH-08 — Módulo de visión computacional que produce métricas de estado

**Origen:** Reemplaza a F33 (Módulo de visión que produce métricas de estado). Clasificada como TTH por aplicación de los criterios de DHU-004 al Bloque E, formalizada en DHU-015.

**Habilita a:** El módulo de visión es **componente demostrable del sistema, no participa en el loop de validación cuantitativa del sistema integrado** conforme a D-007. Su validación es independiente mediante métricas estándar de detección sobre dataset etiquetado representativo. En operación hipotética alimentaría a HU-02 (monitoreo del Operador) con métricas de flujo y cola observados; en MVP1, HU-02 es agnóstica a la fuente (DHU-006) y se alimenta funcionalmente del estado producido por TTH-07. Por tanto, **ninguna HU operativa del backlog consume directamente TTH-08 en MVP1**; el rol del módulo es demostrar la viabilidad arquitectónica del sensor de estado en tiempo real.

**Decisiones técnicas relacionadas:** D-001 (Monolito modular: el módulo de visión vive como módulo del sistema, en `edge_device/` según la separación arquitectónica que distingue dispositivo de borde y servidor central, D-004). D-004 (Pi física como demostración conceptual: el módulo de visión es el componente arquetípico que correría en el dispositivo de borde en una operación real). D-005 (Números de tesis: reportar la realidad medida en validación, no las cifras de relleno). D-007 (Módulo de visión como componente demostrable, no en loop de validación cuantitativa). DHU-015 (clasificación de F33 como TTH del Bloque E).

**Encuadre del refactor (DHU-024, 2026-05-27):** alcance operativo completo (los 11 CTs, no solo los demostrables CT-08.8 + CT-08.9; sizing real ~11-13 SP, no 8); arquitectura DDD completa con capas espejo de `core_management_api` (`domain/`, `application/`, `infrastructure/`, `presentation/`); borrado vía migración Alembic de las tablas legacy `vision_tracks`/`vision_flows`; `vision_aggregates` como única tabla de persistencia del módulo nuevo; contrato técnico LISTO sin cableado real a `ia_prediction_service/` (la integración predictor↔visión queda como F41 explícitamente fuera de scope); C7.6 (deuda CUDA→CPU) resuelta dentro del refactor al reescribir `requirements.txt` desde cero; levantamiento formal de la regla CLAUDE.md sobre `edge_device/src/vision/` en el primer commit del sprint de TTH-08. Ver `DECISIONS_HU.md` § DHU-024 para las 8 decisiones detalladas y el plan de fases F0-F9.

### Descripción

Implementar el módulo de visión computacional del sistema CerebroVial como sensor de estado en tiempo real, capaz de procesar video de la intersección de estudio y producir dos outputs complementarios: un **output numérico** (métricas de estado por dirección) que en operación hipotética alimentaría al motor adaptativo, y un **output visual** (stream de frames procesados con bounding boxes y velocidad estimada por vehículo) que demuestra cualitativamente la viabilidad del componente.

El módulo opera sobre la base de una **arquitectura de tres etapas**:

**Etapa 1 — Detección por frame.** Cada frame del video se procesa con un modelo de detección de objetos (familia YOLO) que identifica vehículos y produce bounding boxes con etiquetas de clase básica (auto, bus, camión, etc.) heredadas de las clases nativas del modelo entrenado.

**Etapa 2 — Tracking entre frames.** Los vehículos detectados se asocian a identidades persistentes a través de frames consecutivos mediante un tracker de objetos (SORT, ByteTrack, BoT-SORT u otro algoritmo equivalente, decidido al implementar). El tracker permite (a) evitar doble conteo del mismo vehículo, (b) estimar la velocidad de cada vehículo individual midiendo su desplazamiento entre frames consecutivos en relación al tiempo transcurrido y a la escala espacial del frame.

**Etapa 3 — Asignación direccional y agregación de métricas.** Cada vehículo detectado y tracked se asigna a una de las direcciones de la intersección (norte, sur, este, oeste para una intersección genérica de cuatro accesos según TTH-07 CT-07.1) mediante **regiones de interés (ROI) configuradas como polígonos al desplegar**. Las métricas de estado por dirección (conteo, cola estimada, flujo, densidad) se derivan agregando los vehículos asignados a cada polígono.

El módulo se reconstruye desde cero como parte del refactor del Bloque E. Existe código predecesor en `edge_device/src/vision/` que sirvió como exploración funcional durante la Fase 1 del proyecto (`EVOLUCION_TESIS.md`) y que validó conceptualmente la viabilidad del pipeline de detección con streams de YouTube. Ese código no se preserva arquitectónicamente; queda como referencia histórica.

La validación del módulo es **independiente del loop de validación cuantitativa del sistema integrado** conforme a D-007. Se reportan métricas estándar de detección (precisión, recall, mAP) sobre dataset etiquetado representativo de tráfico vehicular en perspectiva de cámara de intersección. Los KPIs de tráfico del sistema integrado (tiempo de viaje, demora, longitud de cola) se miden sobre SUMO con métricas producidas por TTH-07, no por este módulo.

### Criterios técnicos de terminado

- **CT-08.1:** Existe un componente de detección de objetos basado en YOLO (versión a cerrar al implementar; YOLOv8 o YOLOv11 son candidatos defendibles) que: (a) consume frames de video provistos por la fuente configurada (CT-08.7), (b) produce bounding boxes con coordenadas, score de confianza y etiqueta de clase para cada vehículo detectado, (c) filtra detecciones con confidence inferior a un umbral configurable (default propuesto: 0.5), (d) soporta las clases vehiculares nativas del modelo entrenado (auto, motocicleta, autobús, camión). La clasificación detallada por tipo de vehículo para priorización operativa está fuera del alcance de MVP1 y se trata en F36 como Trabajos Futuros.

- **CT-08.2:** Existe un componente de tracking de objetos que asocia detecciones consecutivas a identidades persistentes (track_id) y estima la velocidad de cada vehículo tracked. El algoritmo concreto (SORT, ByteTrack, BoT-SORT u otro) se cierra al implementar; el CT solo declara que el tracker debe (a) producir track_id estable para un mismo vehículo a través de frames consecutivos, (b) estimar velocidad de cada vehículo en metros por segundo o kilómetros por hora derivada del desplazamiento del centroide del bounding box entre frames consecutivos, considerando el tiempo transcurrido entre frames y una escala espacial calibrada por intersección, (c) manejar grácil pérdidas temporales de tracking sin asignar track_id nuevo si el vehículo reaparece dentro de una ventana razonable de frames.

- **CT-08.3:** El módulo soporta configuración de regiones de interés (ROI) por intersección. Para una intersección genérica de cuatro accesos conforme a TTH-07 CT-07.1, la configuración declara cuatro polígonos sobre el frame de referencia, uno por cada dirección de entrada (norte, sur, este, oeste). Los polígonos se configuran como archivo de configuración del backend al desplegar (manualmente sobre un frame de referencia del video que se use en el demo, sin UI dedicada en MVP1). Cada vehículo detectado se asigna a la dirección cuyo polígono contiene el centroide de su bounding box. Vehículos cuyos centroides caen fuera de cualquier polígono configurado se reportan como "fuera de zona".

- **CT-08.4:** El módulo produce métricas de estado agregadas por dirección, derivadas de los vehículos asignados a cada polígono de ROI mediante CT-08.3. Las métricas mínimas son: (a) **conteo** de vehículos presentes en el polígono de la dirección en el frame procesado, (b) **cola estimada** definida como número de vehículos detectados con velocidad estimada por debajo de un umbral configurable (default propuesto: 5 km/h), (c) **flujo** definido como vehículos por unidad de tiempo que ingresan al polígono, derivado del tracking entre frames consecutivos, (d) **densidad** definida como conteo del polígono dividido por la longitud configurada del acceso. Las métricas se actualizan a una frecuencia configurable consistente con la frecuencia de procesamiento de frames del módulo.

- **CT-08.5:** Las métricas de estado producidas por CT-08.4 se persisten de forma durable en la base de datos con al menos: timestamp, identificador de intersección, dirección, valor de cada métrica (conteo, cola estimada, flujo, densidad). La frecuencia de persistencia es configurable y consistente con la frecuencia de producción de métricas. Esta persistencia es independiente de la persistencia del dataset de TTH-07 (CT-07.3) y de la persistencia de predicciones de TTH-09 (CT-09.5).

- **CT-08.6:** Existe un endpoint del backend **`GET /vision/state`** (path canónico; alternativa `GET /vision/metrics` aceptable al implementar) que retorna las métricas de estado actuales por dirección, con al menos: timestamp de las métricas reportadas, identificador de intersección, lista de direcciones con sus valores de conteo, cola estimada, flujo y densidad. El endpoint es consumido en operación hipotética por una capa funcional que alimentaría a HU-02, aunque en MVP1 HU-02 consume del estado de SUMO en lugar de visión.

- **CT-08.7:** El módulo soporta dos modos de input de video: (a) **video grabado** (archivo local en formato mp4, mov u otro estándar; modo principal para demos reproducibles), (b) **stream de video** (RTMP, HLS, WebRTC, o stream de YouTube vía URL; modo para operación hipotética en tiempo real, sujeto a la robustez de la fuente externa). El modo se configura al desplegar; el módulo es agnóstico al input una vez configurado.

- **CT-08.8:** Existe un componente de generación de **stream de frames procesados** que produce, para cada frame procesado por CT-08.1, una versión visualmente anotada con: (a) bounding box dibujado alrededor de cada vehículo detectado, (b) etiqueta visible junto al bounding box con la clase de vehículo detectada y la velocidad estimada del vehículo en km/h cuando el tracker (CT-08.2) provee tracking estable, (c) polígonos de ROI superpuestos al frame para visualización de las zonas direccionales configuradas. Este output es **distinto del output numérico de CT-08.6** y se expone vía endpoint separado (sugerencia: `GET /vision/stream` o similar; path concreto a cerrar al implementar). El output visual demuestra cualitativamente la viabilidad del componente y materializa el rol del módulo como "componente demostrable" declarado en D-007.

- **CT-08.9:** El módulo se valida independientemente del sistema integrado mediante métricas estándar de detección reportadas sobre **dataset etiquetado propio** producido para el caso de uso, con al menos **200 frames etiquetados manualmente** representativos del tipo de video que el módulo procesará en operación. Las métricas a reportar son: precisión, recall, mAP (mean Average Precision) según convención estándar del área (mAP@0.5, mAP@0.5:0.95). El **objetivo aspiracional declarado es accuracy ≥ 80%** sobre el conjunto de validación del dataset etiquetado, alineado con TTH-09. El umbral es aspiracional, no bloqueante: si el módulo no alcanza el objetivo, las métricas reales se reportan honestamente en el documento de tesis y en la documentación del sprint conforme a D-005. TTH-08 se considera Done cuando el dataset está etiquetado, las métricas están medidas y reportadas, no condicionada al cumplimiento del umbral aspiracional. Comparación con datasets públicos estándar (COCO, UA-DETRAC, BDD100K) se declara como trabajo futuro asociado, sin abrir ficha separada.

- **CT-08.10:** El módulo expone un health check consumible por TTH-04 conforme a CT-04.1 con estado uno de OK / Degradado / Fuera de servicio según la capacidad del módulo de procesar frames y producir métricas. Cuando el módulo no puede procesar frames (fuente de video caída, modelo no cargado, error interno no recuperable), el endpoint `GET /vision/state` retorna error HTTP estándar (5xx según el caso) y el health check reporta Fuera de servicio. La cascada de fallback declarada en TTH-04 (Nivel 1 con motor operando sin la fuente de visión) es **contractualmente preservada** aunque no se ejercite operativamente en validación cuantitativa de MVP1.

- **CT-08.11:** Existen tests automatizados que cubren los siguientes escenarios: (a) test unitario de detección que verifica que el componente de CT-08.1, dado un frame de prueba, produce bounding boxes con confidence sobre el umbral configurado; (b) test unitario de asignación direccional que verifica que dado un vehículo detectado con centroide en coordenadas conocidas y polígonos de ROI configurados, la asignación a la dirección correcta funciona; (c) test unitario de derivación de métricas; (d) test de integración del endpoint que verifica que `GET /vision/state` retorna métricas con shape esperado; (e) test de integración de persistencia; (f) test unitario del comportamiento ante caída del módulo (CT-08.10).

### Estado actual

🆕 **Por reconstruir desde cero como parte del refactor del Bloque E.** Existe código predecesor en `edge_device/src/vision/` originado en la Fase 1 del proyecto (`EVOLUCION_TESIS.md`) que validó conceptualmente la viabilidad del pipeline de detección con streams de YouTube. Ese código no se preserva arquitectónicamente; queda como referencia histórica y exploración descartada en el sentido de la decisión de refactor tomada al cerrar el Bloque E.

El refactor se justifica por dos razones convergentes: (a) el código predecesor no fue diseñado para producir los outputs estructurados que TTH-08 declara (output numérico estandarizado por dirección + output visual procesado con tracker); (b) la asignación direccional mediante polígonos ROI (CT-08.3), el tracker para velocidad estimada (CT-08.2) y la persistencia estandarizada de métricas (CT-08.5) son responsabilidades nuevas que conviene diseñar limpiamente desde el inicio.

### Notas técnicas

- **El módulo es componente demostrable, no entra al loop cuantitativo (D-007).** Esta separación es fundamental para defensa académica y se preserva intacta en TTH-08. La validación cuantitativa del sistema integrado se mide sobre KPIs producidos por SUMO (TTH-07 CT-07.6), no por este módulo. Cualquier resultado del módulo no afecta los resultados cuantitativos del sistema integrado porque el sistema integrado no consume del módulo durante validación.

- **Rastreo del valor 88.2% del documento de tesis.** Análogo al rastreo del 81.3% del predictor declarado en TTH-09. El documento de tesis menciona "88.2% accuracy de detección" según D-005. Durante la implementación de TTH-08 se rastrea el origen del valor con los mismos tres escenarios posibles: medida real reproducible, medida sobre dataset distinto, o cifra de relleno. Si es cifra de relleno se sustituye por las métricas reales medidas conforme a D-005.

- **Configuración de polígonos ROI sin UI en MVP1.** Los polígonos se configuran como archivo del backend al desplegar, sin UI dedicada en MVP1. Razones: no agrega HU al Bloque D ya cerrado; la configuración es operación rara; un dibujado manual sobre frame de referencia es trabajo de instalación pragmático.

- **Tracker como componente con elección abierta al implementar.** El CT-08.2 declara la responsabilidad del tracker sin atar el algoritmo concreto. Los candidatos defendibles son SORT, ByteTrack, BoT-SORT. La elección concreta es decisión de implementación.

- **Calibración espacial para estimación de velocidad.** La velocidad estimada requiere una calibración espacial del frame (cuántos píxeles del frame corresponden a un metro real). En MVP1, esta calibración se hace al desplegar para cada intersección, viviendo junto con los polígonos de ROI en el archivo de configuración del backend.

- **Clasificación por tipo de vehículo como capacidad heredada de YOLO.** El módulo soporta la detección de las clases vehiculares nativas del modelo YOLO entrenado. Esta clasificación se preserva en CT-08.1 y se expone en el output visual de CT-08.8 como etiqueta del bounding box. **El refinamiento operativo de la clasificación** (priorización de transporte público, detección específica de vehículos de emergencia) está fuera del alcance de MVP1 y se trata como F36 en Trabajos Futuros.

- **Output visual con velocidad estimada como característica demostrativa.** El output visual de CT-08.8 es **valor demostrativo** del componente. En sustentación permite mostrar literalmente que el módulo está procesando video en tiempo real. Materializa el rol del módulo como "componente demostrable" declarado en D-007.

- **Coherencia con D-004 (Pi física como demostración conceptual).** El módulo de visión es el componente arquetípico que correría en el dispositivo de borde en una operación real. En MVP1 corre en la laptop de demostración junto con el resto del sistema (D-003). La arquitectura que separa edge de servidor central se demuestra conceptualmente, no se entrega como hardware (D-004).

- **Limitaciones de los streams de YouTube documentadas honestamente.** Conforme a `EVOLUCION_TESIS.md` Fase 1 y a la justificación de D-007: los streams pueden apagarse, no son específicos de Miraflores, no proveen ground truth. Estas limitaciones se documentan honestamente en la documentación del módulo y en el capítulo de validación de la tesis.

- **Datos reales de Miraflores como trabajo futuro.** La obtención de cámaras propias o video específicamente de la intersección de estudio de Miraflores está fuera del alcance de MVP1 y se trata como trabajo futuro asociado a F41 (Integración cerrada del módulo de visión al loop de validación cuantitativa).

### Trazabilidad con HUs y otras TTH

| HU / TTH que depende | Cómo depende |
|---|---|
| HU-02 (monitoreo, Bloque B) | En operación hipotética consumiría las métricas de estado por dirección (CT-08.4, CT-08.6) como fuente de flujo y cola observados. En MVP1, HU-02 es agnóstica a la fuente (DHU-006) y se alimenta del estado de SUMO vía TTH-07; el módulo de visión no es consumido operativamente. La marca pasiva del panel ante caída de la fuente (CA-02.4, DHU-005 Caso A) se materializa con TTH-07 en MVP1, contractualmente preservada para visión en operación hipotética futura. |
| TTH-04 (Bloque C) | Detecta caída del módulo (mediante health check de CT-04.1) e invoca al Nivel 1 de la cascada. En MVP1, esta cascada es contractualmente preservada pero no se ejercita operativamente en validación cuantitativa (D-007). |
| Capítulo de validación de la tesis | Consume las métricas de detección reportadas por CT-08.9 como validación independiente del módulo. Estas métricas se reportan **separadamente** de los KPIs cuantitativos del sistema integrado (que vienen de SUMO vía TTH-07), preservando la separación metodológica de D-007. |
| Demostración del sistema en sustentación | Consume el output visual de CT-08.8 para demostrar visualmente que el componente está operativo durante la defensa. |

---

## TTH-09 — Modelo predictivo GRU servido vía API

**Origen:** Reemplaza a F34 (Módulo predictivo GRU servido vía API). Clasificada como TTH por aplicación de los criterios de DHU-004 al Bloque E, formalizada en DHU-015.

**Habilita a:** HU-03 (visualización de predicción de congestión a corto plazo, Bloque B), HU-04 (vista combinada del estado actual y la predicción, Bloque B), HU-14 (vista de métricas de desempeño del modelo predictivo, Bloque D). Sin esta TTH, HU-03 y HU-04 no tienen predicciones reales que mostrar al Operador, y HU-14 no tiene métricas reales que mostrar al Administrador.

**Decisiones técnicas relacionadas:** D-001 (Monolito modular). D-002 (Modelo predictivo RNN; refinada por D-006). D-005 (Números de tesis: reportar la realidad medida en validación). D-006 (GRU univariado por intersección: descarta STGNN). D-008 (SUMO como columna vertebral: provee dataset). D-009 (jam level Waze como variable predicha). DHU-015 (clasificación de F34 como TTH del Bloque E).

**Dependencias con otras TTH:** Consume el dataset de entrenamiento producido por TTH-07 (CT-07.3, CT-07.4). Consume la sustentación de hiperparámetros temporales producida por TTH-11 (CT-11.5). Es consumida por TTH-04 como componente predictivo principal cuyo fallo activa el Nivel 2 de la lógica de fallback en cascada.

### Descripción

Implementar el modelo predictivo del sistema, sirviendo predicciones a corto plazo del nivel de congestión por dirección de la intersección de estudio. Conforme a D-006, la arquitectura es **GRU univariado por intersección**; conforme a D-009, la variable predicha es el ratio velocidad/free-flow del cual se deriva el jam level 0-5; conforme a D-008, el modelo se entrena sobre dataset sintético generado por TTH-07.

El componente se materializa como **cuatro modelos GRU univariados independientes**, uno por cada dirección de entrada de la intersección genérica de cuatro accesos declarada en TTH-07 (CT-07.1). Cada GRU consume su propia serie temporal (la del acceso correspondiente) y produce la predicción para esa dirección. La separación en cuatro modelos univariados, en lugar de un modelo multi-output que prediga las cuatro direcciones simultáneamente, preserva la independencia entre direcciones declarada por D-006 y permite que cada modelo se entrene, evalúe y reemplace independientemente.

Cada modelo implementa la **arquitectura multi-output (seq2seq o equivalente)**: consume una ventana de entrada de los últimos pasos observados y produce un vector de predicciones futuras en una sola inferencia. La cantidad exacta de pasos en la ventana de entrada (lookback) y la cantidad de pasos en el horizonte de predicción son hiperparámetros temporales consumidos del documento de sustentación producido por TTH-11. El default operativo asumido para implementación inicial, si TTH-11 no ha cerrado al momento de comenzar TTH-09, es lookback de 30 minutos y horizonte de 60 minutos con paso de muestreo de 60 segundos; estos valores se confirman o ajustan con la salida de TTH-11 antes de declarar TTH-09 Done.

El componente expone sus predicciones vía endpoint HTTP del backend. Una sola llamada al endpoint devuelve las predicciones para las cuatro direcciones a lo largo de todo el horizonte temporal, suficiente para que HU-03 materialice la UX del slider del Operador sin necesidad de llamadas adicionales por paso temporal. Cada predicción se persiste de forma durable en el momento que se genera, lo cual habilita a HU-14 a comparar predicciones con observaciones reales una vez que el horizonte llega y calcular las métricas declaradas en HU-14.

El componente no implementa lógica de fallback ante caída propia ni del entorno. Si el modelo no responde, la detección y el fallback al predictor de respaldo (RandomForest existente) son responsabilidad de TTH-04, que declara el Nivel 2 de la lógica de fallback en cascada del sistema.

### Criterios técnicos de terminado

- **CT-09.1:** Existe un componente entrenable que implementa la arquitectura **GRU univariado por intersección** conforme a D-006. La implementación cubre cuatro modelos GRU independientes, uno por cada dirección de entrada de la intersección genérica declarada en TTH-07 (CT-07.1). Cada modelo consume su propia serie temporal univariada (ratio velocidad/free-flow del acceso correspondiente, según el mapeo declarado en D-009 y materializado por TTH-07) y produce predicciones para esa dirección de forma independiente del resto.

- **CT-09.2:** Cada modelo implementa **arquitectura multi-output**: consume una ventana de entrada de los últimos N pasos observados y produce un vector de M predicciones futuras en una sola inferencia. Los valores concretos de N (lookback) y M (horizonte) son consumidos del documento producido por TTH-11 (CT-11.5). Si TTH-11 no ha cerrado al momento de implementar TTH-09, se asumen valores provisionales (N = 30, M = 60, con paso de 60 segundos); TTH-09 no se considera Done hasta que los valores se confirman o ajustan con la salida de TTH-11 y el modelo se entrena o reentrena con los valores definitivos.

- **CT-09.3:** El componente incluye un **script reproducible de entrenamiento** que: (a) consume el dataset producido por TTH-07 (CT-07.3) usando las particiones declaradas en CT-07.4, (b) entrena los cuatro modelos GRU univariados con los hiperparámetros temporales de TTH-11 y los hiperparámetros no temporales que el equipo elija al implementar, (c) persiste los modelos entrenados en disco en una ubicación fija documentada, (d) reporta las cuatro métricas declaradas en HU-14 sobre la partición de validación al finalizar el entrenamiento, (e) es reejecutable: una nueva ejecución del script con el mismo dataset y los mismos hiperparámetros produce un modelo equivalente sin asistencia manual. El script no implementa pipeline MLOps automatizado (F21 es Trabajos Futuros); reentrenar es ejecutar el script manualmente.

- **CT-09.4:** Existe un endpoint del backend **`POST /predictions/predict`** (mismo path que el endpoint baseline existente, según ficha F34 y código actual del proyecto) que: (a) carga los cuatro modelos GRU entrenados desde la ubicación documentada en CT-09.3, (b) recibe como input las series temporales recientes necesarias para la ventana de entrada de cada dirección, (c) ejecuta una inferencia por dirección, (d) devuelve un objeto estructurado con una entrada por dirección, conteniendo un array de predicciones desde el paso t+1 hasta el paso t+horizonte, cada predicción con dos campos: el **ratio continuo** (valor numérico predicho por el modelo) y el **nivel discreto 0-5** (derivado del ratio según el mapeo de D-009, calculado en backend). Una sola llamada al endpoint cubre las cuatro direcciones y todo el horizonte temporal, suficiente para alimentar el slider del Operador de HU-03 sin llamadas adicionales.

- **CT-09.5:** Cada predicción producida por el endpoint se **persiste de forma durable** en el momento de generarse, con al menos: timestamp de generación de la predicción, dirección, paso futuro al que corresponde la predicción (t+1, t+2, ..., t+horizonte), ratio continuo predicho, nivel discreto derivado, identificador del modelo o versión del modelo que produjo la predicción. Esta persistencia es la fuente de datos que HU-14 consume para comparar predicciones con observaciones reales una vez que el horizonte llega y calcular las métricas declaradas (MAE, RMSE, accuracy, matriz de confusión 6×6). El registro de predicciones es independiente del registro de decisiones del motor adaptativo (CA-08.1 de HU-08) y del registro de transiciones de estado operativo (CT-04.3 de TTH-04).

  > **Nota de reconciliación (D-005, TTH-09 2026-05-31).** El texto de arriba asume un *regresor de ratio* (campo "ratio continuo predicho"; métricas "MAE, RMSE"). Esa arquitectura quedó **obsoleta**: la producción de TTH-09 es un **clasificador GRU multi-output** (6 clases, escala Waze 0-5; ver `documentation/contracts/prediction_contract.md` §4/§6/§8) que no emite ratio. Reconciliación: el campo "ratio continuo predicho" **no aplica** — se persiste el **`level` discreto (`argmax`) + el vector `probs` de 6 softmax**; se mantienen timestamp de generación, dirección, paso futuro y versión del modelo. Las métricas se reconcilian en CT-09.6. Materializado en la tabla `predictions` (modelo `PredictionDB`, migración `core_management_api/alembic/versions/f2c9d7a4b6e1_add_predictions_table.py`). Texto original conservado como rastro.

- **CT-09.6:** El modelo se evalúa sobre la partición de validación del dataset de TTH-07 (CT-07.4: seeds y patrones distintos de entrenamiento, para evitar fuga de información conforme a D-008). La evaluación reporta las **cuatro métricas declaradas en HU-14**: MAE y RMSE sobre el ratio continuo, accuracy sobre el nivel discreto 0-5, y matriz de confusión 6×6 (filas = nivel real, columnas = nivel predicho, convención de scikit-learn declarada en CA-14.8 de HU-14). Las métricas se reportan por dirección y consolidadas (promedio o equivalente que el equipo elija documentar) al finalizar el entrenamiento.

  > **Nota de reconciliación (D-005, TTH-09 2026-05-31).** "MAE y RMSE sobre el ratio continuo" asume un regresor de ratio, arquitectura **obsoleta**: la producción es un **clasificador** (contrato `prediction_contract.md` §6). Las métricas efectivas son de **clasificación**: accuracy y matriz de confusión 6×6 sobre el nivel 0-5 (ya enumeradas arriba), más **F1-macro** y **MAE ordinal** (`|nivel_pred − nivel_real|`). MAE/RMSE-sobre-ratio **no se reporta**. Misma reconciliación que CT-09.5 y §6 del contrato. Texto original conservado como rastro.

- **CT-09.7:** El **objetivo aspiracional del modelo es accuracy ≥ 80% sobre el nivel discreto 0-5** evaluada sobre la partición de validación. El umbral se establece como objetivo del modelo predictivo del sistema, no como criterio bloqueante de Done. Si el modelo entrenado no alcanza el objetivo aspiracional, las métricas reales medidas se reportan honestamente en el documento de tesis (capítulo de validación) y en la documentación del sprint, conforme a D-005. TTH-09 se considera Done cuando el modelo está entrenado, servido y las métricas reportadas, no condicionada al cumplimiento del umbral aspiracional.

- **CT-09.8:** Cuando el modelo no responde (proceso caído, modelo no cargado, timeout interno), el endpoint **`POST /predictions/predict`** retorna un **error HTTP estándar** (5xx según el caso) con cuerpo descriptivo del error. La detección de la caída y el fallback al predictor de respaldo (RandomForest existente, preservado como componente preexistente) son responsabilidad de TTH-04, no de este componente. TTH-09 no implementa lógica de fallback interno; expone únicamente el modelo principal GRU y delega la cascada a la lógica externa declarada en TTH-04 (Nivel 2).

- **CT-09.9:** Existen tests automatizados que cubren los siguientes escenarios: (a) test unitario del script de entrenamiento que verifica que una corrida corta sobre dataset reducido produce modelos persistidos en disco sin errores; (b) test unitario del endpoint que verifica que, dado input válido, devuelve respuesta con shape esperado; (c) test de integración que verifica que el endpoint, con modelos entrenados sobre el dataset de TTH-07, produce predicciones cuyos rangos son plausibles (ratio entre 0 y 1, nivel entre 0 y 5); (d) test unitario que verifica que cuando el modelo no se puede cargar, el endpoint retorna error HTTP esperado (CT-09.8); (e) test unitario que verifica que cada predicción generada por el endpoint queda persistida con los campos declarados en CT-09.5.

### Estado actual

🆕 **Por construir desde cero como GRU.** Existe un componente predictivo baseline (`RandomForestPredictor`) en el código actual, sirviendo predicciones vía endpoint `POST /predictions/predict`. Este componente baseline **se preserva sin modificación** como predictor de respaldo invocado por TTH-04 en Nivel 2 de la cascada de fallback. TTH-09 introduce el GRU como modelo principal **sin reemplazar arquitectónicamente** al RandomForest. La ruta del endpoint queda servida por el GRU como modelo principal; el RandomForest sigue accesible internamente para que TTH-04 lo invoque cuando el GRU no responde.

### Notas técnicas

- **Hiperparámetros no temporales (NO cubiertos por TTH-11).** TTH-11 cubre únicamente hiperparámetros temporales (paso, lookback, horizonte, frecuencia de re-inferencia). Los hiperparámetros no temporales del GRU (tamaño del estado oculto, número de capas, función de pérdida, optimizador, learning rate, regularización, batch size, número de épocas) se eligen al implementar TTH-09 por buenas prácticas estándar del área y se documentan junto con el modelo entrenado.

- **Rastreo del valor 81.3% del documento de tesis.** Durante la implementación de TTH-09 o TTH-11 se rastrea el origen del valor "81.3% accuracy del predictor" del documento de tesis. Tres escenarios: medida real reproducible (se preserva como referencia), medida del RandomForest sobre dataset distinto (se documenta como tal), o cifra de relleno (se sustituye por las métricas reales conforme a D-005). La acción es verificación documental, no bloqueante.

- **Persistencia del modelo en disco (versionado simple).** Los cuatro modelos entrenados se persisten en una ubicación fija del sistema de archivos del backend. Re-entrenar reemplaza los archivos. **No hay versionado de modelos** (no se preservan modelos anteriores, no hay timestamping, no hay metadata sofisticada de runs). Esta simplicidad es coherente con MLOps fuera de alcance (F21 es Trabajos Futuros).

- **Topología de cuatro direcciones como limitación de alcance MVP1, y extensión a múltiples intersecciones como trabajo futuro arquitectónico (no lineal).** El componente está diseñado y entrenado para una intersección de cuatro accesos según TTH-07 (CT-07.1). Dos escenarios de crecimiento son posibles y se tratan de forma distinta:

  *Crecimiento dentro de una misma intersección* (intersección de cinco o más accesos, glorietas, intersecciones complejas): es escalamiento trivial. Cada acceso adicional es un nuevo modelo GRU univariado entrenado independientemente, sin rediseño arquitectónico.

  *Crecimiento a múltiples intersecciones interrelacionadas* (red urbana): **escalar linealmente entrenando 4 modelos GRU univariados independientes por cada intersección adicional es técnicamente viable pero descarta la dependencia espacial entre intersecciones**, que es información con valor predictivo según la literatura del área (Yu et al. 2018, Li et al. 2018, Wu et al. 2019). El camino arquitectónico correcto para esta extensión son las arquitecturas Spatio-Temporal Graph Neural Networks (STGNN), declaradas como trabajo futuro del proyecto en F37 ("coordinación de ondas verdes entre intersecciones vecinas mediante arquitecturas espacio-temporales tipo STGNN") y fundamentadas por la exploración de Fase 2 documentada en `EVOLUCION_TESIS.md`. El componente preservado en `ia_prediction_service/src/models/legacy/time_then_space.py` (descartado por D-006 para MVP1) es punto de partida natural para esta extensión cuando se aborde. **TTH-09 no escala arquitectónicamente a red urbana; esa extensión es trabajo futuro declarado, no extensión lineal del componente actual.**

- **Comportamiento ante caída del modelo principal y TTH-04 (Nivel 2).** Como se declaró en CT-09.8, este componente expone únicamente el GRU. Cuando el GRU no responde, TTH-04 detecta la condición y activa Nivel 2 de la cascada de fallback, invocando al `RandomForestPredictor` preservado. Esta separación tiene una consecuencia importante: el endpoint puede estar respondiendo (con predicciones del RandomForest) aunque el GRU esté caído. La marca pasiva del panel de predicción (CA-03.4 de HU-03, DHU-005 Caso B) y la alerta activa transversal (HU-10) reflejan correctamente este estado: el componente predictor está degradado (Nivel 2), no caído del todo.

- **Continuidad del baseline RandomForest.** El `RandomForestPredictor` actual se preserva sin modificación funcional. La única coordinación con TTH-04 es asegurar que la ruta interna que TTH-04 invoca para el predictor de respaldo está disponible y produce respuestas con el mismo formato de output que el endpoint público (cuatro direcciones, ratio y nivel por predicción). La metadata de origen (qué modelo produjo la predicción) se preserva en CT-09.5.

- **Frecuencia de re-inferencia del endpoint.** La frecuencia con la que el backend recalcula predicciones se documenta y cierra con la salida de TTH-11 (CT-11.2). Hasta que TTH-11 cierre, el comportamiento provisional del endpoint es inferir por cada llamada externa recibida (sin caché ni recalculación programada).

### Trazabilidad con HUs y otras TTH

| HU / TTH que depende | Cómo depende |
|---|---|
| HU-03 (predicción, Bloque B) | Consume el endpoint `POST /predictions/predict` (CT-09.4) para mostrar la predicción al Operador. La UX del slider que muestra la predicción a distintos pasos futuros se materializa con el vector de horizonte completo que una sola llamada al endpoint devuelve. La marca pasiva del panel ante caída del componente (CA-03.4, DHU-005 Caso B) se materializa con CT-09.8 (error HTTP estándar) y con la cascada de TTH-04 (Nivel 2). |
| HU-04 (vista combinada, Bloque B) | Consume el mismo endpoint que HU-03; misma dinámica de slider y de marca pasiva. |
| HU-14 (métricas del modelo, Bloque D) | Consume el **registro persistente de predicciones** (CT-09.5) como fuente de datos para comparar predicciones con observaciones reales y calcular MAE, RMSE, accuracy, matriz de confusión 6×6 (CA-14.1 a CA-14.4 de HU-14). |
| TTH-04 (Bloque C) | Detecta caída del GRU (mediante health check del componente, CT-04.1) e invoca al predictor de respaldo `RandomForestPredictor` (Nivel 2 de la cascada). |
| TTH-07 (Bloque E) | Provee el dataset de entrenamiento y validación (CT-07.3, CT-07.4) que CT-09.3 consume. |
| TTH-11 (Bloque E) | Provee la tabla resumen de hiperparámetros temporales (CT-11.5) que CT-09.2 consume. Si TTH-11 no ha cerrado al momento de implementar TTH-09, se usan valores provisionales y se reentrena al cierre de TTH-11. |
| TTH-10 (Motor adaptativo, Bloque E) | Consume el endpoint `POST /predictions/predict` (CT-09.4) como input del motor adaptativo. |

---

## TTH-10 — Motor adaptativo de control semafórico

**Origen:** Reemplaza a F35 (Motor adaptativo: Webster + MaxPressure + MTC). Clasificada como TTH por aplicación de los criterios de DHU-004 al Bloque E, formalizada en DHU-015.

**Habilita a:** HU-05 (visualización de la estrategia de control activa, Bloque B), HU-06 (explicación de la razón de selección de estrategia, Bloque B), HU-07 (notificación de cambios de estrategia, Bloque B), HU-08 (consulta del historial de decisiones del motor, Bloque B), HU-15 (configuración de parámetros operativos del sistema, Bloque D, parámetros que afectan al motor). El motor adaptativo es el componente central del sistema y el aporte de ingeniería principal del trabajo según `EVOLUCION_TESIS.md` Fase 3.

**Decisiones técnicas relacionadas:** D-001 (Monolito modular: el motor vive como módulo del backend `core_management_api/src/control/`). D-005 (Números de tesis). D-006 (GRU univariado: fuente de predicciones del motor). D-008 (SUMO como entorno de operación durante validación cuantitativa). D-009 (jam level Waze como variable predicha). DHU-015 (clasificación de F35 como TTH del Bloque E).

**Marco normativo aplicable:** Manual de Dispositivos de Control del Tránsito Automotor para Calles y Carreteras, R.D. N.° 26-2024-MTC/18 (octubre 2024), Ministerio de Transportes y Comunicaciones del Perú. Define las restricciones legales que la capa MTC del motor aplica como reglas duras inviolables.

**Dependencias con otras TTH del Bloque E:** Consume las predicciones del modelo predictivo vía endpoint `POST /predictions/predict` (CT-09.4 de TTH-09). Consume el entorno de simulación SUMO vía TraCI (CT-07.5 de TTH-07) durante validación cuantitativa. Consume parcialmente las decisiones de hiperparámetros temporales de TTH-11 (frecuencia de re-inferencia del endpoint de predicciones, CT-11.2). Es consumido por TTH-04 como componente de control cuyo fallo activa el Nivel 3 de la lógica de fallback en cascada (tiempos preconfigurados de TTH-05).

### Descripción

Implementar el motor adaptativo de control semafórico del sistema CerebroVial, conforme al aporte de ingeniería central declarado en `EVOLUCION_TESIS.md` Fase 3. El motor es una **pipeline de dos etapas** que opera sobre el estado predicho (proveniente de TTH-09) y observado (proveniente de TTH-07 en validación, o del módulo de visión TTH-08 en operación hipotética):

**Etapa 1 — Selección de estrategia adaptativa.** El componente AdaptiveEngine elige entre dos estrategias intercambiables (Webster o Max Pressure) según el estado del sistema. La estrategia seleccionada calcula tiempos óptimos del semáforo a partir de la demanda observada y predicha.

**Etapa 2 — Aplicación de reglas duras (MTC).** La salida de la estrategia seleccionada en Etapa 1 pasa por una capa de cumplimiento normativo que corrige los tiempos calculados para que respeten las restricciones inviolables del marco regulatorio peruano. MTC no decide tiempos adaptativos; corrige los tiempos decididos en Etapa 1 y compone la secuencia final aplicada al semáforo.

La separación arquitectónica entre adaptabilidad (Webster/Max Pressure) y reglas duras (MTC) es deliberada: las dos estrategias adaptativas son algoritmos puros de optimización de capacidad que no incorporan regulación local; MTC es una capa de cumplimiento que hace los outputs legales y operativos. Si la regulación peruana cambia o se adopta otra norma, la capa MTC se reemplaza sin tocar los algoritmos adaptativos.

El motor no implementa lógica de fallback ante caída propia. Cuando el motor no puede decidir, responde con error HTTP estándar; la detección de la caída y el fallback al Nivel 3 (tiempos preconfigurados de TTH-05) es responsabilidad de TTH-04. Cada decisión del motor se persiste de forma durable para que HU-08 muestre el historial al Operador y para que el capítulo de validación de la tesis pueda analizar la evolución del comportamiento del motor.

### Criterios técnicos de terminado

- **CT-10.1:** Existe un componente AdaptiveEngine que opera como pipeline de dos etapas (selección de estrategia adaptativa + capa MTC de reglas duras) sobre el estado predicho y observado de la intersección. La arquitectura está implementada en `core_management_api/src/control/` y documentada en `CONTROL.md`. La frontera entre Etapa 1 (Webster/Max Pressure) y Etapa 2 (MTC) está separada en código de forma que la capa MTC sea reemplazable sin tocar las estrategias adaptativas y viceversa.

- **CT-10.2 (Etapa 1 — Estrategia Webster):** Existe implementación de la estrategia Webster (1958) que: (a) calcula el factor de carga `Y = Σ q_i/s_i` sobre las fases, (b) verifica feasibility (`Y < 0.95`), (c) calcula el ciclo óptimo según la fórmula `C = (1.5 · L + 5) / (1 - Y)`, (d) reparte verde efectivo proporcionalmente al peso de cada fase en Y. Cuando `Y ≥ 0.95`, Webster levanta excepción `WebsterInfeasible` que el AdaptiveEngine maneja según la combinación con `flow_total` (ver CT-10.5).

- **CT-10.3 (Etapa 1 — Estrategia Max Pressure):** Existe implementación de Max Pressure (Varaiya 2013) que: (a) calcula la presión de cada fase candidata según la versión simplificada `P(φ) ≈ cola(φ) · s(φ)/3600`, (b) elige la próxima fase como la de mayor presión, (c) usa ciclo base de Webster si Webster es feasible o ciclo default de 60 segundos si Webster es infeasible, (d) aplica round-robin alfabético determinístico cuando todas las colas son cero.

- **CT-10.4 (Etapa 1 — Selección entre Webster y Max Pressure):** El AdaptiveEngine selecciona la estrategia activa según el umbral `flow_total > 1500 veh/h`. Por debajo del umbral activa Webster (modo off-peak); por encima activa Max Pressure (modo peak). El umbral es **parametrizable** en archivo de configuración del backend, calibrable contra simulaciones SUMO en validación cuantitativa, no expuesto al Administrador en MVP1 conforme a DHU-014 subsección C.

- **CT-10.5 (Etapa 1 — Casos de la matriz de selección):** El motor maneja los cuatro casos posibles de combinación `flow_total` × `Y` según el cuadro de la sección 5.2 de `CONTROL.md`:

  | `flow_total` | `Y` | Comportamiento |
  |---|---|---|
  | ≤ 1500 | < 0.95 | Webster (off-peak), ciclo óptimo, verdes proporcionales. |
  | ≤ 1500 | ≥ 0.95 | Responde HTTP 422 con `code=webster_infeasible` (caso patológico raro). |
  | > 1500 | < 0.95 | Max Pressure con ciclo base de Webster. |
  | > 1500 | ≥ 0.95 | Max Pressure con ciclo default de 60s, verdes proporcionales a colas. |

  Estos cuatro casos están cubiertos por tests automatizados.

- **CT-10.6 (Etapa 2 — Capa MTC, constantes):** La capa MTC define cinco constantes normativas con origen en el Manual MTC peruano R.D. N.° 26-2024-MTC/18: `MIN_GREEN = 7 s` (verde mínimo por fase), `MAX_GREEN = 60 s` (verde máximo por fase), `MIN_YELLOW = 3 s` (amarillo mínimo), `ALL_RED = 2 s` (all-red obligatorio entre fases), `MIN_PEDESTRIAN = 7 s` (verde mínimo cuando la fase tiene cruce peatonal activo). Las constantes viven en archivo de configuración del backend, son ajustables al desplegar pero no expuestas al Administrador en MVP1 conforme a DHU-014 subsección C. Modificar reglas duras en runtime requiere conocimiento profundo de ingeniería de tráfico que excede el perfil del Administrador declarado, y crearía riesgo operativo.

- **CT-10.7 (Etapa 2 — Capa MTC, las tres operaciones):** La capa MTC aplica tres operaciones determinísticas sobre la salida de Etapa 1:
  1. **Elevar.** Si el verde calculado está por debajo de `MIN_GREEN` (o `MIN_PEDESTRIAN` si la fase tiene `has_pedestrian = true`), MTC lo eleva al mínimo correspondiente y registra el ajuste con descripción legible.
  2. **Recortar.** Si el verde calculado está por encima de `MAX_GREEN`, MTC lo recorta al máximo y registra el ajuste.
  3. **Componer.** MTC compone la secuencia final aplicada al semáforo agregando amarillo físico (`MIN_YELLOW`) y all-red (`ALL_RED`) tras cada verde corregido.

- **CT-10.8 (Output del motor):** El motor devuelve, para cada solicitud de recomendación, un objeto estructurado que incluye al menos: identificador de la intersección, modo activo (`webster` u `max_pressure`), ciclo final en segundos, lista de fases con sus tiempos finales (verde, amarillo, all-red), próxima fase a entrar si aplica (solo en modo Max Pressure), razonamiento legible que explica la decisión, y lista de ajustes aplicados por MTC. El formato JSON canónico está documentado en el Anexo de `CONTROL.md`.

- **CT-10.9 (Persistencia de decisiones):** Cada decisión del motor se persiste de forma durable en el momento de generarse, con al menos: timestamp, intersección, modo activo, estrategia previa si cambió, tiempos calculados antes de MTC (verdes "ideales" producidos por Webster o Max Pressure en Etapa 1), tiempos finales después de MTC (verdes corregidos + secuencia compuesta), lista de ajustes aplicados por MTC con su descripción legible, razonamiento. Esta persistencia es la fuente de datos que HU-08 consume para mostrar el historial al Operador (CA-08.1 ingloba F31 según el cierre del Bloque B) y que el capítulo de validación de la tesis consume para análisis cuantitativo. La persistencia es independiente del registro de transiciones de estado operativo de TTH-04 (CT-04.3) y del registro de predicciones de TTH-09 (CT-09.5).

- **CT-10.10 (Integración con TTH-09):** El motor consume las predicciones del modelo predictivo vía endpoint `POST /predictions/predict` de TTH-09 (CT-09.4) como input al razonamiento de selección de estrategia y al cálculo de tiempos. La frecuencia de re-inferencia consume el valor recomendado por TTH-11 (CT-11.2). Esta integración reemplaza la conexión actual con el baseline RandomForest, que queda preservado como predictor de respaldo invocado por TTH-04 en Nivel 2 de la cascada.

- **CT-10.11 (Integración con TTH-07 durante validación cuantitativa):** Durante el experimento de validación cuantitativa del sistema integrado, el motor opera dentro del entorno de simulación SUMO conforme a CT-07.5. El motor consulta el estado observado vía TraCI y emite decisiones de fase del semáforo que la simulación ejecuta sobre el semáforo virtual. La decisión concreta entre adaptador externo TraCI ↔ API del motor y modificación del motor para consumir TraCI directamente se cierra al implementar TTH-07 y TTH-10 conjuntamente.

- **CT-10.12 (Integración con HU-15, parámetros configurables del Administrador):** El motor consume desde el mecanismo de configuración declarado en HU-15 (CA-15.1 a CA-15.4) los parámetros operativos que afectan a su comportamiento: umbral de congestión (default ≥ nivel 3 según D-009) y horizonte de predicción (CA-03.1 de HU-03). Los parámetros internos de las estrategias del motor quedan fuera del alcance de HU-15 en MVP1 conforme a DHU-014 subsección C; se ajustan en archivo de configuración del backend al desplegar.

- **CT-10.13 (Integración con TTH-04, cascada de fallback):** El motor expone health check consumible por TTH-04 conforme a CT-04.1. Cuando el motor no puede decidir por condición patológica de inputs (caso `flow_total ≤ 1500 + Y ≥ 0.95` declarado en CT-10.5, ausencia de inputs obligatorios, error interno no recuperable), responde con HTTP 422 (caso `webster_infeasible`) o HTTP 5xx (errores genéricos) y no aplica decisiones al semáforo. TTH-04 detecta esta condición y activa Nivel 3 de la cascada (tiempos preconfigurados de TTH-05). El motor no implementa lógica de fallback interno; delega la cascada a la lógica externa declarada en TTH-04.

- **CT-10.14 (Tests):** Existen tests automatizados que cubren los siguientes escenarios: (a) tests unitarios de Webster; (b) tests unitarios de Max Pressure; (c) tests unitarios de MTC; (d) test de integración del AdaptiveEngine cubriendo los cuatro casos de CT-10.5; (e) test de integración con TTH-09; (f) test de integración con TTH-07; (g) test de integración con TTH-04; (h) test de persistencia. Los tests no validan calidad cuantitativa del sistema (eso pertenece al capítulo de validación de la tesis); validan únicamente que la mecánica funciona conforme a la teoría documentada.

### Estado actual

✓✓ **Construido, integración pendiente con otras TTH del Bloque E.** El motor está implementado en `core_management_api/src/control/` (las dos estrategias adaptativas + capa MTC + AdaptiveEngine), con tests unitarios y de integración pasando, frontend de visualización conectado (`views/control/`), y documentación teórica de 552 líneas en `CONTROL.md`.

Lo que falta para considerar TTH-10 Done:

1. **Integración con TTH-09** (CT-10.10): el motor consume hoy del baseline RandomForest; debe migrar al endpoint del GRU.
2. **Integración con TTH-07** (CT-10.11): la coordinación TraCI ↔ motor para el experimento de validación cuantitativa se construye junto con TTH-07.
3. **Integración con TTH-04** (CT-10.13): el health check expuesto por el motor y la cascada de fallback al Nivel 3 deben validarse end-to-end junto con TTH-04 y TTH-05.
4. **Consumo de parámetros configurables de HU-15** (CT-10.12).
5. **Persistencia ampliada de decisiones** (CT-10.9): el registro actual puede no cubrir todos los campos que HU-08 y el capítulo de validación necesitan; se amplía si es necesario al cerrar el sprint del Bloque B.

### Notas técnicas

- **El motor es el aporte de ingeniería principal de la tesis.** El detalle teórico de cada decisión arquitectónica está desarrollado en `CONTROL.md` con citas bibliográficas (Webster 1958, Varaiya 2013, FHWA 2008, Manual MTC peruano 2024). La defensa académica del componente se apoya en ese documento, no en TTH-10 sola.

- **Validación funcional vs validación cuantitativa.** TTH-10 se considera Done por integración funcional. La validación cuantitativa (mejora del motor adaptativo frente a control fijo Webster, medida en KPIs de tráfico) **no es CT de TTH-10**; pertenece al capítulo de validación de la tesis y se reporta conforme a D-005. El criterio de éxito sugerido en `CONTROL.md` (RD% ≥ 15% de mejora en demora promedio frente a tiempos fijos) es objetivo de la validación, no criterio de Done del componente.

- **Umbral peak (1500 veh/h) como decisión heurística parametrizable.** El valor es heurístico, no físico. Representa aproximadamente 40-50% de la capacidad teórica agregada de una intersección urbana de 2 fases. El valor es parametrizable en archivo de configuración del backend y se calibrará contra simulaciones SUMO durante validación cuantitativa.

- **Versión simplificada de Max Pressure.** El componente implementa la **versión simplificada** documentada en sección 4.3 de `CONTROL.md`, no la versión rigurosa de Varaiya 2013 con flujos upstream/downstream y matrices de doblar. La versión rigurosa se declara como trabajo futuro asociado a la extensión a red urbana (F37).

- **Predicción de congestión opcional como input del motor.** El motor declara `predicted_demand` como input **opcional**. Los inputs obligatorios son `flow`, `saturation_flow` y `queue` por fase. Cuando TTH-09 cierra y CT-10.10 se implementa, el motor consume las predicciones del GRU como contexto adicional para razonamiento anticipativo. Esto convierte al motor de **reactivo** a **proactivo**.

- **Trazabilidad regulatoria.** Las cinco constantes de MTC (CT-10.6) están justificadas individualmente en sección 6 de `CONTROL.md` con citas al Manual MTC peruano y al FHWA Traffic Signal Timing Manual (2008). El motor no inventa límites operativos; replica restricciones documentadas en normativa vigente y manuales operativos internacionales.

- **Visibilidad de los ajustes de MTC al Operador.** Las correcciones aplicadas por MTC se registran en la persistencia de decisiones (CT-10.9) y están disponibles en el output del motor (CT-10.8) como lista `adjustments`. El Operador, en MVP1, **no recibe notificación específica de cada corrección de MTC**; ve la decisión final aplicada al semáforo a través de las HUs del Bloque B.

- **Detalle conceptual sobre `lost_time`.** Webster usa `lost_time` como parámetro teórico de su fórmula (típicamente 8 segundos para 2 fases con 2 cambios por ciclo), mientras que MTC compone los amarillos y all-reds físicos completos (10 segundos para esa misma intersección). La diferencia representa la fracción del amarillo que los conductores aprovechan extendiendo el cruce. Esto explica por qué el ciclo final del demo (70 s) es distinto del ciclo "óptimo" calculado por Webster (68 s). Detalle relevante para defensa académica.

- **Soporte de N fases.** El motor está diseñado para soportar N fases por intersección. Webster suma sobre N términos en el cálculo de Y. Max Pressure elige cuál de las N fases entra primero por presión. La capa MTC aplica las cinco constantes a cada fase independientemente. El demo del frontend usa 2 fases (NS/EW); el caso de 4 fases u otro número es soportado funcionalmente.

- **Detección automática de demanda peatonal como trabajo futuro.** Cada fase tiene un flag `has_pedestrian` que activa `MIN_PEDESTRIAN`. En MVP1, el flag se configura por intersección al desplegar y permanece estático. La detección automática de demanda peatonal en tiempo real se declara como trabajo futuro implícito asociado a F33 (visión).

- **Recordatorio de la conexión con jam level (D-009).** El documento teórico describe `predicted_demand` como ordinal 1-5; la integración real con TTH-09 produce predicciones en términos del constructo **jam level 0-5 de Waze** (D-009). Al implementar CT-10.10, el adapter de dominio mencionado en sección 9 del documento teórico se concreta como mapeo `jam_level (0-5) → input del motor`.

### Trazabilidad con HUs y otras TTH

| HU / TTH que depende | Cómo depende |
|---|---|
| HU-05 (estrategia activa, Bloque B) | Consume el modo activo (`webster` u `max_pressure`) del output de CT-10.8. La marca pasiva del panel ante caída del motor (CA-05.4, DHU-005 Caso B) se materializa con CT-10.13 y con la cascada de TTH-04 (Nivel 3). |
| HU-06 (explicación, Bloque B) | Consume el `reasoning` del output de CT-10.8 para construir la explicación legible al Operador. |
| HU-07 (notificación de cambios, Bloque B) | Consume las transiciones de estrategia del registro persistido en CT-10.9 para notificar al Operador cuando el motor cambia de modo. |
| HU-08 (historial, Bloque B) | Consume el registro persistido en CT-10.9 para mostrar el historial al Operador. CA-08.1 ingloba F31 (persistencia de decisiones del motor) conforme al cierre del Bloque A; CT-10.9 es la materialización del sustrato técnico de esa CA. |
| HU-15 (configuración de parámetros, Bloque D) | Provee los parámetros operativos que el motor consume (CT-10.12). |
| TTH-04 (Bloque C) | Detecta caída del motor (mediante health check, CT-04.1) e invoca al Nivel 3 de la cascada (tiempos preconfigurados de TTH-05). |
| TTH-07 (Bloque E) | Provee el entorno de simulación SUMO end-to-end (CT-07.5) que el motor consume durante validación cuantitativa. |
| TTH-09 (Bloque E) | Provee las predicciones del modelo predictivo vía endpoint (CT-09.4) que el motor consume como input (CT-10.10). |
| TTH-11 (Bloque E) | Provee la frecuencia de re-inferencia recomendada (CT-11.2) que el motor usa al solicitar nuevas predicciones a TTH-09. |
| Capítulo de validación de la tesis | Consume el registro persistido en CT-10.9 y los KPIs producidos por el motor operando dentro de SUMO (vía TTH-07 CT-07.6) para análisis cuantitativo del comportamiento del motor frente a control fijo. |

---

## TTH-11 — Spike de calibración de hiperparámetros temporales del modelo predictivo

**Origen:** Identificada durante la redacción de TTH-09 (Bloque E) al cerrar la decisión arquitectónica del modelo predictivo. Su apertura formaliza la necesidad de sustentación bibliográfica y empírica de los hiperparámetros temporales del modelo. Formalizada en la actualización de DHU-015 (paso de 4 a 5 TTH del Bloque E).

**Habilita a:** TTH-09 (provee la sustentación documentada de los hiperparámetros temporales que TTH-09 consume al entrenar el modelo). Habilita también el cierre del Δt del dataset de TTH-07 (la decisión sobre granularidad de muestreo que CT-07.3 dejó parcialmente abierta se cierra con la salida de esta TTH).

**Decisiones técnicas relacionadas:** D-005 (Números de tesis: reportar la realidad medida en validación). D-006 (GRU univariado por intersección: el modelo cuyos hiperparámetros temporales son objeto de esta investigación). D-008 (SUMO como columna vertebral: la parte empírica de esta TTH consume el dataset producido por TTH-07). D-009 (jam level Waze como variable predicha: define la semántica del output del modelo cuyos hiperparámetros se calibran). DHU-015 (clasificación de F32-F35 como TTH del Bloque E, ampliada para incluir TTH-11).

### Descripción

Esta TTH es un **spike de investigación**, no una tarea de implementación. Su entregable es un **documento sustentado bibliográficamente y, condicionalmente, empíricamente** que cierra las decisiones de cuatro hiperparámetros temporales del modelo predictivo:

1. **Paso de muestreo (Δt_in):** intervalo de tiempo entre observaciones consecutivas de la serie temporal de entrada al modelo. Define la granularidad temporal del dataset que TTH-07 produce y, en cadena, la granularidad de la serie que el GRU procesa.
2. **Ventana de entrada (lookback):** número de pasos anteriores que el modelo "mira" para producir una predicción. Determina el contexto histórico que el GRU consume en cada inferencia.
3. **Horizonte de predicción:** número de pasos futuros que el modelo predice en una sola inferencia. Determina el alcance temporal del output del endpoint y, en consecuencia, el rango del slider del Operador en HU-03.
4. **Frecuencia de re-inferencia del endpoint:** cada cuánto tiempo el backend recalcula las predicciones que sirve. Es independiente de Δt_in (granularidad de la serie de entrada) y tiene impacto en HU-03 y en costo computacional.

Los cuatro hiperparámetros son **decisiones temporales acopladas**: cambiar uno afecta la interpretación de los demás. Por eso conviene sustentarlos en un mismo documento, con una metodología común y conclusiones coherentes entre sí.

El spike sigue el patrón estándar de "Enabler Story de tipo Exploration" (SAFe) o "Research Spike" (XP, SCRUM): trabajo de reducción de incertidumbre con entregable documental concreto y cuyo Done no es código sino conocimiento accionable.

### Criterios técnicos de terminado

- **CT-11.1:** Existe un documento entregable ubicado en `documentation/docs/` (junto a `DISCOVERY_2026-05-10.md` referenciado en `EVOLUCION_TESIS.md` sección 9), en formato Markdown, con un nombre claro y descriptivo (sugerencia: `INVESTIGACION_HIPERPARAMETROS_TEMPORALES.md`). El documento es legible por un revisor académico sin asistencia del autor y por un implementador técnico sin formación académica adicional.

- **CT-11.2:** El documento cubre los cuatro hiperparámetros temporales declarados en la descripción (Δt_in, lookback, horizonte, frecuencia de re-inferencia), cada uno con su propia sección. Ninguno queda sin sustentación. Para cada hiperparámetro, la sección presenta: (a) definición técnica precisa del hiperparámetro, (b) revisión bibliográfica del estado del arte en predicción de tráfico urbano, (c) rango de valores típicos reportados en la literatura, (d) análisis de implicancias prácticas para el caso CerebroVial, (e) recomendación final con valor concreto.

- **CT-11.3:** La revisión bibliográfica cita al menos **cinco fuentes académicas distintas** de predicción de tráfico urbano con redes neuronales recurrentes. Las citas se documentan con formato académico estándar (autor, año, título, venue/DOI) suficiente para que el documento sea referenciable como anexo del documento de tesis. Las fuentes son verificables; no se citan fuentes inventadas ni placeholders.

- **CT-11.4:** El documento incluye una sección de **exploración empírica** sobre el dataset producido por TTH-07. La exploración compara al menos **tres combinaciones de hiperparámetros** entrenando modelos completos (no fragmentos) y reporta las cuatro métricas declaradas en HU-14 (MAE, RMSE sobre ratio continuo; accuracy, matriz de confusión sobre nivel discreto) por cada combinación. Las combinaciones exploradas son razonadas (no aleatorias): cada una representa una hipótesis distinta sobre el trade-off entre granularidad y horizonte, sustentada por la revisión bibliográfica de CT-11.3.

- **CT-11.5:** El documento declara explícitamente la **recomendación final consolidada** de los cuatro hiperparámetros como una tabla resumen: nombre del hiperparámetro, valor recomendado, unidad, justificación corta (1-2 líneas), referencias bibliográficas que sustentan el valor. Esta tabla es el "contrato" de salida que TTH-09 consume al implementarse.

- **CT-11.6:** El documento declara una sección de **limitaciones y trabajo futuro** que cubre al menos: (a) qué decisiones podrían revisarse si se obtienen datos reales de Waze (D-008, F38), (b) qué sucede si la topología se extiende a más de 4 direcciones de entrada o a múltiples intersecciones interrelacionadas (referenciar F37 y nota técnica de TTH-09), (c) qué hiperparámetros adicionales no temporales (arquitectura del modelo, tamaño del estado oculto del GRU, optimizador, regularización) quedan fuera del alcance de esta TTH y se cierran al implementar TTH-09 sin investigación específica.

- **CT-11.7:** El documento se redacta con doble propósito: sirve como **sustentación interna del equipo** para tomar decisiones de implementación en TTH-09 y como **anexo formal del documento de tesis** para sustentación académica de los hiperparámetros. El nivel de rigor académico (citas, estructura, lenguaje) es consistente a lo largo del documento; no hay secciones internas/informales que rompan la consistencia con secciones formales.

- **CT-11.8:** El documento incluye una nota explícita sobre la **decisión de Δt_in que cierra el parámetro de TTH-07** (granularidad del dataset). La nota declara: el valor decidido y la razón. Esto materializa la dependencia declarada en TTH-07 nota técnica.

### Estado actual

🆕 **Por construir desde cero.** No hay documento de investigación de hiperparámetros temporales del modelo predictivo al momento de redactar esta TTH. La decisión sobre cuáles son los hiperparámetros temporales del modelo se cerró durante el diálogo de redacción del Bloque E.

### Notas técnicas

- **Dependencia con TTH-07 (parte empírica condicional).** La parte bibliográfica de TTH-11 (CT-11.2, CT-11.3) es **independiente** del dataset de TTH-07 y puede ejecutarse en paralelo con la construcción de TTH-07. La parte empírica (CT-11.4) requiere el dataset producido por CT-07.3 ejecutable. La dependencia se maneja como sigue: si TTH-07 está disponible cuando se ejecuta TTH-11, la parte empírica se incorpora antes de declarar TTH-11 Done. Si TTH-07 sufre retrasos, **TTH-11 puede cerrar con la parte bibliográfica completa** y la sección empírica se agrega como complemento al documento cuando TTH-07 esté disponible. El documento declara explícitamente en qué estado se encuentra cada sección al cerrar TTH-11 (versionado simple: v0.x con bibliográfica solamente, v1.0 con ambas).

- **Postura ante retraso de TTH-11 frente al cronograma de TTH-09.** Si por razones de cronograma TTH-09 necesita arrancar implementación antes de que TTH-11 cierre, TTH-09 **puede arrancar con valores provisionales** (Δt = 60 segundos, lookback = 30 minutos, horizonte = 60 minutos, frecuencia de re-inferencia a cerrar al implementar) bajo la premisa de que TTH-11 confirmará o ajustará esos valores antes del cierre del MVP1. Si TTH-11 confirma los valores provisionales, no hay trabajo adicional. Si TTH-11 ajusta los valores, TTH-09 reentrena el modelo con los valores correctos antes de declararse Done. Esta postura desacopla riesgo cronológico de las dos TTH preservando rigor académico al cierre del MVP1.

- **Cuatro combinaciones razonadas para la exploración empírica de CT-11.4.** El criterio "al menos tres combinaciones razonadas" se aterriza típicamente como cuatro combinaciones explorables, cada una con hipótesis distinta: (a) **referencia** (valores provisionales mencionados arriba: Δt=60s, lookback=30, horizonte=60), (b) **granularidad fina** (Δt menor, ej. 30s, manteniendo lookback y horizonte en tiempo), (c) **lookback corto** (lookback reducido para evaluar si el modelo aprovecha la historia adicional o no), (d) **horizonte reducido** (horizonte menor para evaluar si la calidad de predicción aumenta significativamente al sacrificar alcance temporal). El número final exacto de combinaciones puede ser tres o cuatro según el costo computacional observado al ejecutar el spike; el criterio mínimo de CT-11.4 es tres.

- **Costo computacional acotado.** La exploración empírica entrena modelos completos (no fragmentos), lo cual tiene costo. Para acotar el costo, la exploración se ejecuta sobre **una submuestra del dataset completo de TTH-07** suficiente para producir métricas estables (criterio sugerido: 30-50% del dataset de entrenamiento de TTH-09, suficiente para tendencias relativas entre configuraciones aunque las métricas absolutas sean inferiores a las del modelo final). El documento declara explícitamente el tamaño de submuestra usado y la limitación inherente.

- **Frecuencia de re-inferencia como hiperparámetro distinto.** La frecuencia de re-inferencia del endpoint (cada cuánto el backend recalcula predicciones) es decisión **de despliegue**, no de entrenamiento. Por eso su sustentación es distinta de la de Δt, lookback y horizonte. Su revisión bibliográfica cubre prácticas de sistemas de tráfico en producción reportadas en literatura; su exploración empírica puede limitarse a análisis de trade-off entre latencia percibida por el usuario, costo computacional y plausibilidad de cambio significativo del estado entre re-inferencias.

- **Hiperparámetros NO cubiertos por TTH-11.** Esta TTH cubre **únicamente hiperparámetros temporales**. Quedan fuera explícitamente, a cerrar al implementar TTH-09 sin investigación específica: tamaño del estado oculto del GRU, número de capas del GRU, función de pérdida concreta, optimizador y learning rate, regularización (dropout, weight decay), tamaño de batch de entrenamiento, número de épocas. Estos hiperparámetros se eligen por buenas prácticas estándar del área y se documentan al implementar el modelo, sin requerir spike de investigación dedicado.

- **Relación con D-005 (números reales).** Las métricas reportadas en la exploración empírica de CT-11.4 son **métricas reales** medidas sobre la submuestra del dataset de TTH-07. No son cifras de ejemplo ni placeholders. Si durante la exploración se observa que ninguna combinación alcanza el objetivo aspiracional del modelo final (≥80% accuracy, declarado para TTH-09 según DHU-015), esto se documenta honestamente en el documento de TTH-11 como **señal temprana** que TTH-09 deberá manejar al implementarse.

- **Rastreo del 81.3% del documento de tesis dentro de TTH-11.** La acción declarada como nota técnica de TTH-09 (rastrear el origen del valor 81.3% de accuracy mencionado en el documento de tesis) puede incorporarse a TTH-11 como subsección de la revisión bibliográfica de horizontes y métricas, **si el rastreo resulta en una referencia bibliográfica concreta**. Si el rastreo concluye que el 81.3% es relleno sin sustento, el documento de TTH-11 lo declara explícitamente y propone reemplazo conforme a D-005. La acción se ejecuta dentro de TTH-11 o dentro de TTH-09, según donde naturalmente caiga al implementarse.

### Trazabilidad con HUs y otras TTH

| HU / TTH que depende | Cómo depende |
|---|---|
| TTH-09 (Bloque E) | Consume la tabla resumen de hiperparámetros (CT-11.5) como contrato de entrada al entrenar el modelo. Sin TTH-11 cerrada, TTH-09 entrena con valores provisionales sujetos a revisión. |
| TTH-07 (Bloque E) | Recibe el cierre del Δt del dataset (CT-11.8) que materializa la decisión que la nota técnica de TTH-07 dejó abierta. |
| HU-03 (predicción, Bloque B) | Consume indirectamente vía TTH-09: el rango del slider del Operador está determinado por el horizonte recomendado por TTH-11. |
| HU-14 (métricas del modelo, Bloque D) | Consume indirectamente vía TTH-09: las métricas que el Administrador ve son las del modelo entrenado con los hiperparámetros recomendados por TTH-11. |
| Documento de tesis (capítulo de modelo predictivo) | Consume directamente el documento entregable de TTH-11 como anexo formal o como fuente de sustentación de la sección de metodología del modelo. |

---

## TTH-12 — Infraestructura de datos de congestión por arista para la vista de mapa de red

**Origen:** Habilitador de HU-22 (mapa de congestión de la red). Identificado durante el análisis de implementación del mapa de calor (2026-06-02), al constatar que la vista panorámica de red requiere datos de congestión por arista que el sistema no produce hoy. Ver DHU-028 en `DECISIONS_HU.md`.

**Habilita a:** HU-22 (vista de mapa de congestión de la red por tramo de vía). Sin esta TTH cerrada, HU-22 no tiene geometría que pintar ni feed de congestión que leer.

**Clasificación:** MVP1 — ampliación (vista panorámica de red).

**Estado actual:** Pendiente.

**Naturaleza (DHU-004):** cumple los cuatro criterios de TTH — no tiene Persona beneficiaria directa (su efecto se ve en HU-22), su valor es instrumental (habilita la vista), su comportamiento es técnico estándar (provisión de datos geoespaciales + feed en streaming + persistencia), y no entrega valor visible al usuario en aislamiento. Conforme a DHU-006, esta TTH sí nombra tecnologías concretas (SUMO, TraCI, PostGIS, `waze_jams`, SSE) por su naturaleza técnica.

---

### Contexto y alcance

HU-22 requiere, para cada tramo de vía (arista del grafo vial), dos cosas que hoy no existen como datos disponibles para el frontend:

1. **La geometría de la arista** (su trazado geográfico) para poder dibujarla en el mapa.
2. **Un nivel de congestión por arista, actualizándose en vivo**, para poder colorearla.

El estado del repositorio al momento de redactar esta TTH (auditoría 2026-06-02):

- La tabla `graph_edges` está modelada y migrada con columna `geom` (LINESTRING 4326), pero el seed carga solo 6 aristas de prueba — las ~375 aristas reales de la red de Miraflores **no están pobladas** en la base. La geometría real existe en `conf/network/miraflores.net.xml` y debe extraerse.
- La tabla `waze_jams` está modelada, migrada y convertida en hypertable TimescaleDB (`edge_id`, `congestion_level` 0-5, `speed_mps`, `geom`), pero está **huérfana**: sin repositorio, sin endpoint, sin ingestor.
- El nivel de congestión 0-5 por arista es derivable desde la velocidad de SUMO mediante `sumo_to_jam_level()` (ya implementada y testeada en `simulation/`).
- En producción, la fuente de este feed sería un ingestor de Waze (modelado pero fuera de alcance). Para la demostración, **SUMO actúa como stand-in del feed de Waze**, detrás de una interfaz de feed de congestión por arista que mantiene la tubería idéntica a la de producción (el día que se conecte Waze, solo cambia quién escribe en `waze_jams`).

**Alcance de esta TTH (V1):** estado de congestión **actual** por arista, persistido en `waze_jams` y servido al frontend en vivo. La capa de **predicción** por arista (servir un modelo predictivo por arista) queda fuera de alcance de esta TTH y se aborda en una TTH posterior cuando se implemente la extensión predictiva de HU-22.

**Decisión de fuente (V1):** se inicia con **replay de un dataset SUMO pre-generado** reproducido en streaming (opción B del análisis del 2026-06-02). El adaptador de fuente se diseña detrás de una interfaz que permite sustituir el replay por **SUMO en vivo vía TraCI** (opción A) como upgrade posterior sin alterar el resto de la tubería; la viabilidad de SUMO en vivo está confirmada por medición (real-time factor ~41× leyendo todas las aristas por paso). El upgrade a SUMO en vivo es una extensión registrada, no parte de la V1.

---

### Criterios técnicos de terminado

- **CT-12.1 (geometría poblada):** Las ~375 aristas de la red vial de Miraflores (las del componente conexo del grafo, alineadas con el mapping canónico `ia_prediction_service/src/data/artifacts/miraflores_graph_lcc_mapping.json`) están pobladas en la tabla `graph_edges` con su geometría real (`geom` LINESTRING 4326) extraída de `conf/network/miraflores.net.xml`. Existe un script de seed/ingesta idempotente, versionado, que realiza esta carga y es reejecutable sin duplicar. Los `edge_id` cargados coinciden exactamente con los `sumo_edge` del mapping canónico, de modo que el feed de congestión (que usa esos mismos IDs) alinea con la geometría sin traducción.

- **CT-12.2 (endpoint de geometría):** Existe un endpoint REST que sirve al frontend la geometría de las aristas de la red (identificador de arista + LINESTRING en un formato consumible por el mapa, p. ej. GeoJSON), protegido con `require_role(OPERATOR, ADMIN)` conforme al patrón de autorización del proyecto (TTH-01). El endpoint responde la red completa en un tiempo razonable para una carga de apertura de vista.

- **CT-12.3 (interfaz de feed de congestión por arista):** Existe una interfaz/contrato bien definido de "feed de congestión por arista" — el punto de desacople de la fuente. La interfaz expone, por `edge_id`, un `congestion_level` 0-5 con marca de tiempo. La implementación de la fuente (replay SUMO en V1) está detrás de esta interfaz, de modo que sustituirla (SUMO en vivo, o Waze real en producción) no requiere cambios en los consumidores. La derivación del nivel 0-5 usa `sumo_to_jam_level()` / `ratio_to_jam_level()` ya existentes.

- **CT-12.4 (adaptador de fuente — replay SUMO):** Existe un adaptador que reproduce en streaming un día de dataset SUMO pre-generado (Parquet por arista con `speed`/`speedRelative`, granularidad 60 s, 1440 pasos/día), emitiendo el nivel de congestión por arista a través de la interfaz de CT-12.3, a una cadencia configurable (tiempo real o acelerado para demostración). El día reproducido es seleccionable (p. ej. un día de congestión marcada para que el mapa muestre la transición de niveles). El adaptador NO requiere un proceso SUMO corriendo en tiempo de demostración (reproduce datos ya generados). Nota de robustez: el dataset Parquet está gitignored (regenerable); la TTH documenta la receta de regeneración y/o versiona el día concreto usado para la demostración como insumo reproducible.

- **CT-12.5 (persistencia en `waze_jams`):** El feed de congestión por arista se persiste en la tabla `waze_jams` (hypertable TimescaleDB): cada actualización de nivel por arista se escribe como una fila (`edge_id`, `congestion_level`, `timestamp`, y los campos disponibles). Existe un repositorio que encapsula la escritura y la lectura de `waze_jams`. Esto mantiene la tubería idéntica a la de producción: el día que un ingestor de Waze real reemplace al adaptador de replay, solo cambia el escritor de `waze_jams`; el resto de la tubería (lectura, endpoint, vista) permanece igual.

- **CT-12.6 (endpoint del feed + canal de tiempo real):** Existe el camino por el que el frontend obtiene el estado de congestión por arista en vivo, siguiendo el patrón establecido del proyecto (DHU-021 #15): un canal SSE que emite la señal de "hay nueva información de congestión" (wake-up) y un endpoint REST que el frontend re-lee para obtener el estado actual de congestión por arista desde `waze_jams`. El patrón pub/sub existente (`ActiveStateBroadcaster`, hoy keyed por `node_id`) se extiende o se replica para emitir por el dominio de aristas/red. Ambos endpoints protegidos con `require_role(OPERATOR, ADMIN)`. La latencia end-to-end (escritura del feed → disponibilidad para el frontend) es coherente con el umbral de tiempo real ≤ 5 s que HU-22 declara (CA-22.2).

- **CT-12.7 (robustez ante interrupción — soporte a DHU-005 Caso A):** El feed expone la marca de tiempo de la última actualización de congestión, de modo que la vista (HU-22, CA-22.4) pueda determinar y mostrar la antigüedad del dato y marcarlo como desactualizado si la fuente se interrumpe. La interrupción de la fuente (adaptador detenido) no produce error en el endpoint de lectura: este sigue devolviendo el último estado conocido con su marca de tiempo.

- **CT-12.8 (alineación de IDs end-to-end):** Se verifica explícitamente que los `edge_id` son consistentes a lo largo de toda la cadena: mapping canónico → geometría en `graph_edges` (CT-12.1) → feed de congestión (CT-12.3/12.4) → persistencia en `waze_jams` (CT-12.5) → consumo en el frontend (HU-22). Una arista coloreada en el mapa corresponde inequívocamente a la misma arista en el grafo, la geometría y el feed. Existe una verificación (test o script) que confirma que el conjunto de `edge_id` de la geometría poblada y el conjunto de `edge_id` que el feed puede emitir coinciden.

- **CT-12.9 (tests automatizados):** Existen tests que cubren: (a) la extracción/carga de geometría desde el `.net.xml` produce el número esperado de aristas con geometría válida; (b) el endpoint de geometría responde con el shape esperado y protegido por rol; (c) la derivación de nivel 0-5 desde el dato SUMO es correcta (reusa/extiende los tests existentes de `sumo_to_jam_level`); (d) el adaptador de replay emite por la interfaz la secuencia esperada para un día conocido; (e) la persistencia en `waze_jams` escribe y lee correctamente; (f) el endpoint del feed devuelve el estado por arista con su marca de tiempo; (g) la alineación de `edge_id` end-to-end (CT-12.8).

---

### Relación con HU y otras TTH / decisiones

| Depende de / se relaciona con | Cómo |
|---|---|
| HU-22 | Esta TTH la habilita por completo: sin CT-12.1 (geometría) y CT-12.5/12.6 (feed persistido y servido), HU-22 no tiene qué pintar ni qué actualizar. Orden de implementación: TTH-12 primero, HU-22 después. |
| TTH-01 (autenticación) | Los endpoints de geometría y de feed se protegen con el RBAC de TTH-01 (`require_role`). |
| TTH-07 (SUMO) | Reutiliza el dataset SUMO y la derivación de nivel de congestión producidos en el contexto de TTH-07; el adaptador de replay lee de ese dataset. |
| D-009 (jam level 0-5) | El nivel de congestión por arista usa la escala 0-5 de D-009. La derivación es por velocidad (`sumo_to_jam_level`), no por timeLoss. |
| DHU-005 (robustez ante interrupción, Caso A) | CT-12.7 provee la marca de tiempo que HU-22 (CA-22.4) usa para marcar datos desactualizados. |
| DHU-021 (#15, patrón SSE-wake/REST-read) | CT-12.6 sigue este patrón: SSE solo señaliza, el dato se re-lee por REST. |
| DHU-028 | Registra la ampliación del backlog (HU-22 + TTH-12) y la unidad de análisis "arista" para la vista de red. |

### Extensiones futuras (fuera de alcance de esta TTH)

- **SUMO en vivo (upgrade del adaptador de fuente):** sustituir el replay por un loop TraCI en vivo sobre la red completa de Miraflores, detrás de la misma interfaz de CT-12.3. Viabilidad confirmada por medición (RTF ~41×, cruce de procesos `simulation`↔`core` por HTTP localhost ya probado). Requiere un loop TraCI sobre la red completa (hoy solo existe sobre la red de una intersección) y el sentido inverso del puente de procesos.
- **Ingestor de Waze real:** reemplazar el adaptador de fuente por un ingestor que llene `waze_jams` desde la API de Waze. Es el destino de producción; la tubería de esta TTH está diseñada para recibirlo sin cambios aguas abajo.
- **Feed de congestión predicha por arista:** servir un modelo predictivo por arista para alimentar la capa de predicción de la extensión futura de HU-22. Requiere decidir y servir el predictor por arista correspondiente.

---

## TTH-13 — Infraestructura de consulta histórica de congestión por arista

**Origen:** Habilitador de HU-23 (recorrido temporal de la congestión de la red). Identificado durante el análisis de evolución de HU-22 (2026-06-02), al constatar que el recorrido temporal del mapa requiere servir la serie de congestión por arista de un día completo para que el frontend la recorra con fluidez, capacidad que TTH-12 no provee (su endpoint de estado está cableado al snapshot más reciente). Ver DHU-029 en `DECISIONS_HU.md`.

**Habilita a:** HU-23 (recorrido temporal de la congestión de la red por tramo de vía). Sin esta TTH cerrada, HU-23 no tiene la serie del día que el control temporal recorre ni el tope de cobertura con el que se acota.

**Clasificación:** MVP1 — ampliación (barrido temporal de la vista panorámica de red).

**Estado actual:** Pendiente.

**Naturaleza (DHU-004):** cumple los cuatro criterios de TTH — no tiene Persona beneficiaria directa (su efecto se ve en HU-23), su valor es instrumental (habilita el recorrido temporal), su comportamiento es técnico estándar (lectura de la serie temporal de un día desde la persistencia + serialización compacta), y no entrega valor visible al usuario en aislamiento. Conforme a DHU-006, esta TTH sí nombra tecnologías concretas (`waze_jams`, TimescaleDB, endpoint REST) por su naturaleza técnica.

### Contexto y alcance

HU-23 requiere recorrer la congestión de la red sobre el mapa a lo largo del día. Para que el recorrido sea fluido (el Operador arrastra un control temporal y el mapa repinta sin latencia perceptible), la vista necesita la **serie completa de un día** —el nivel de congestión 0-5 de cada arista en cada instante de 60 s— disponible para indexar en memoria, en lugar de una consulta por cada instante. Esto exige una capacidad que la infraestructura de TTH-12 no expone hoy: servir, en una sola llamada, la serie temporal de niveles por arista de un día dado, en un formato compacto consumible por el frontend.

El endpoint de estado de TTH-12 (`GET /congestion/state`) devuelve solo el snapshot más reciente por arista (`latest_per_edge()`) y se conserva intacto para HU-22 (vista en vivo). TTH-13 agrega un endpoint nuevo e independiente para la serie del día; no modifica el de estado actual.

El estado del repositorio al momento de redactar esta TTH (auditoría 2026-06-02):

- La tabla `waze_jams` está poblada y es hypertable TimescaleDB (chunk de 1 día sobre `snapshot_timestamp`), con índices simples sobre `event_uuid` y sobre `snapshot_timestamp`, y PK compuesta `(event_uuid, snapshot_timestamp)`. No existe índice compuesto `(edge_id, snapshot_timestamp)`, óptimo para leer la serie de un día ordenada por arista y tiempo.
- El repositorio de `waze_jams` expone `latest_per_edge()` (más reciente por arista) pero no existe un método que lea la serie completa de un día por arista (todas las filas del día, ordenadas por `edge_id, snapshot_timestamp`).
- La geometría de las aristas (CT-12.1/12.2) y la persistencia del feed (CT-12.5) ya existen y se reutilizan sin cambios.

**Alcance de esta TTH (V1):** servir la serie de congestión histórica y actual por arista de un día completo en una sola llamada (nivel 0-5 por arista por instante de 60 s), en formato compacto, con el tope de cobertura del día embebido en la respuesta. Bajo el supuesto de día sin huecos internos (ver "Contexto y alcance"). La consulta de estado **predicho** a futuro queda fuera de alcance (depende de la extensión predictiva por arista, ver DHU-029).

**Formato de la serie (decisión de diseño, verificada por medición):** la respuesta es compacta por arista — un objeto por `edge_id` con la metadata temporal del día (instante base `t0`, paso en segundos `step_s`) y un array ordenado de niveles `levels` (un entero 0-5 por instante del día). El frontend mapea la posición del control temporal a un índice del array (`levels[i]`) y repinta en O(1), sin nuevas llamadas. Medición sobre el día sembrado seed062 (540 000 filas): este formato compacto pesa ~1.58 MB crudo y ~55 KB bajo gzip de transporte, frente a ~50.8 MB de un formato de filas planas — servible en una sola llamada. No se aplica codificación adicional (RLE / sparse): la compresión gzip ya lleva el payload a ~55 KB (el 91 % de los niveles son 0), y un codec propio rompería el acceso O(1) por índice que el slider necesita, a cambio de una ganancia despreciable.

**Supuesto de cobertura (V1): día sin huecos internos.** Se asume que un día tiene cobertura temporal continua desde su inicio hasta su último instante con datos (sin huecos internos). Verificado para el día sembrado seed062: grilla densa 375 × 1440, 0 huecos, cada instante con las 375 aristas presentes, cobertura 00:00→23:59. Bajo este supuesto, el "tope de cobertura" de un día es simplemente su último instante con datos, que la respuesta de la serie ya contiene (no requiere endpoint ni consulta aparte). El frontend usa ese tope para limitar el control temporal (ver CT-13.2 y HU-23 CA-23.2). El manejo de **huecos internos** (un día con tramos sin dato intercalados) —y la UX asociada de saltos del control o zonas grises sobre el eje— queda **fuera de alcance de V1 como extensión futura de UX** (ver "Extensiones futuras").

### Criterios técnicos de terminado

- **CT-13.1 (lectura de la serie del día en el repositorio):** Existe un método en el
  repositorio de `waze_jams` que devuelve, para un día dado, todas las filas de ese día
  (`edge_id`, `snapshot_timestamp`, `congestion_level`) ordenadas por `edge_id,
  snapshot_timestamp`. Es una query nueva e independiente de `latest_per_edge()` (no agrega
  ni hace self-join: filtra por rango del día y ordena). El resultado, ya ordenado, se agrupa
  por arista para construir la serie compacta (Formato B): por cada `edge_id`, un array de
  niveles indexado por instante. La agrupación puede hacerse en Python iterando el resultado
  ordenado (cambio de `edge_id` = nueva arista) o empujarse a la DB con agregación; la decisión
  de implementación se justifica en el plan. Devuelve un DTO/estructura propia de la serie (no
  `EdgeCongestion`, que es por-instante).

- **CT-13.2 (endpoint de serie del día):** Existe un endpoint REST `GET /congestion/series`
  que recibe el día como **query param requerido** (`day`, fecha ISO `YYYY-MM-DD`, parseada
  como `date`) y devuelve, en una sola respuesta, la serie compacta del día: por cada arista,
  su `edge_id`, el instante base `t0`, el paso `step_s` y el array `levels` de niveles 0-5; más
  el tope de cobertura del día (el último instante con datos), que el frontend usa para limitar
  el control temporal. La respuesta admite compresión gzip de transporte. El endpoint está
  protegido con `require_role(OPERATOR, ADMIN)` conforme al patrón del proyecto (TTH-01), en el
  mismo router del dominio congestion. Nota de convención: TTH-13 es el primer endpoint del
  proyecto que recibe un selector temporal; se sigue la convención observada (selectores como
  query param, identificadores como path param — único precedente: `interval` en
  `/predictions/history`). Si el día solicitado no tiene datos, el endpoint responde
  explícitamente la ausencia (serie vacía o señal equivalente, sin error), de modo que el
  frontend pueda deshabilitar el control (HU-23 CA-23.7).

- **CT-13.3 (índice de soporte):** Existe una migración versionada que crea el índice compuesto
  `(edge_id, snapshot_timestamp)` sobre `waze_jams`. Este orden de columnas es óptimo para la
  query de CT-13.1 (`WHERE` por rango de día + `ORDER BY edge_id, snapshot_timestamp`): permite
  un index scan que entrega las filas ya ordenadas, sin sort en memoria, con cada arista como un
  rango contiguo. Se crea con `op.create_index` (patrón Alembic puro, calcando el índice
  compuesto existente de la tabla `predictions`). La migración verifica el head actual de Alembic
  antes de fijar su `down_revision`, e incluye `downgrade()` con `op.drop_index`. Hoy no existe
  índice sobre `edge_id` (es FK nullable sin índice); el compuesto es nuevo y no redundante.
  Nota de honestidad: `waze_jams` es hypertable TimescaleDB particionada por `snapshot_timestamp`;
  el índice incluye esa columna (como segunda), lo que Timescale acepta. El comportamiento exacto
  del `create_index` sobre la hypertable se verifica al ejecutar la migración (no antes).

- **CT-13.4 (no alteración de los contratos de TTH-12):** El endpoint `GET /congestion/state`
  (snapshot más reciente, consumido por HU-22), `GET /congestion/geometry`, el feed SSE y el
  repositorio (`latest_per_edge`, escritura) permanecen funcionalmente intactos. El endpoint de
  serie y el método de repo nuevos no modifican el comportamiento observable de los consumidores
  actuales. Existe verificación (test) de que `/congestion/state` sigue devolviendo el snapshot
  más reciente como antes.

- **CT-13.5 (robustez y coherencia con DHU-005):** La consulta de serie no produce error ante un
  día sin datos (devuelve serie vacía o señal de ausencia, no excepción). La respuesta incluye la
  marca temporal (instante base + paso, y/o el tope de cobertura), de modo que el frontend pueda
  derivar el rango recorrible y, coherente con TTH-12 (CT-12.7) y HU-23 (CA-23.4), comunicar la
  naturaleza del dato.

- **CT-13.6 (tests automatizados):** Existen tests que cubren: (a) el método de repo devuelve,
  para un día conocido sembrado, la serie por arista esperada (orden y niveles correctos);
  (b) el endpoint `GET /congestion/series?day=...` responde con el shape compacto esperado
  (objeto por arista con `edge_id`/`t0`/`step_s`/`levels`), protegido por rol, con el tope de
  cobertura; (c) un día sin datos responde la ausencia sin error (HU-23 CA-23.7); (d) test de RBAC
  (rol no autorizado → 403); (e) `/congestion/state` sigue funcionando sin cambios (CT-13.4). Los
  tests siguen el patrón del dominio (SQLite en memoria, helper de siembra de filas, `as_role`).
  El índice compuesto (CT-13.3) NO es verificable en el SQLite de test (su DDL no lleva índices);
  su verificación es manual/PostgreSQL, igual que `populate_geom_from_edges` — se declara así, sin
  fingir cobertura.

---

### Relación con HU y otras TTH / decisiones

| Depende de / se relaciona con | Cómo |
|---|---|
| HU-23 | Esta TTH la habilita: sin CT-13.2 (serie del día en una llamada, con su tope de cobertura) HU-23 no tiene cómo pintar el recorrido temporal ni cómo limitar el control. Orden de implementación: TTH-13 primero, HU-23 después. |
| TTH-12 | Reutiliza por completo la geometría (CT-12.1/12.2), la persistencia en `waze_jams` (CT-12.5) y el patrón de autorización. TTH-13 agrega solo la lectura de la serie del día y su endpoint; no altera los contratos de TTH-12 (CT-13.4). |
| TTH-01 (autenticación) | Los endpoints nuevos se protegen con el RBAC de TTH-01 (`require_role(OPERATOR, ADMIN)`). |
| D-009 (jam level 0-5) | La serie del día usa la misma escala 0-5 por arista de TTH-12; TTH-13 no introduce nueva derivación de nivel, solo lee lo persistido. |
| DHU-005 (robustez ante interrupción, Caso A) | CT-13.5 conserva la marca de tiempo por arista que la vista usa para mostrar antigüedad del dato recorrido. |
| DHU-029 | Registra la ampliación del backlog (HU-23 + TTH-13) y el alcance temporal histórico/actual de la vista de recorrido. |

### Extensiones futuras (fuera de alcance de esta TTH)

- **Consulta de estado predicho por arista:** servir el estado de congestión **proyectado a futuro** por arista (p. ej. próximos 30 min) para alimentar la extensión del eje temporal de HU-23 hacia el futuro. Requiere la capa de predicción por arista en escala 0-5, que hoy no existe servida (ver dimensionamiento en DHU-029: hay grafo edge-as-node y modelos por arista offline, pero predicen `timeLoss`, no 0-5). Requiere decidir y servir el predictor por arista correspondiente.
- **Agregación histórica para el Gerente:** una consulta histórica agregada por periodos prolongados (no instante a instante) que alimente una eventual vista de análisis geográfico del Gerente, distinta del recorrido operativo del Operador de HU-23. Es una necesidad de otra Persona y se redactaría como su propia HU si se aborda.
- **Manejo de huecos internos de cobertura (UX del control temporal):** soportar días con tramos sin dato intercalados (no solo cobertura continua hasta un tope). La mejor UX —que el control salte los tramos vacíos o los muestre como zonas grises sobre el eje temporal— queda fuera de V1, que asume cobertura continua sin huecos (validado para el día sembrado). Cuando se incorpore una fuente con cobertura irregular (p. ej. el día en curso parcial, o Waze real con llegadas intermitentes), se redactará la UX de huecos como ampliación.

---

## Relación con el Plan de Ejecución

Las TTH **no se estiman con Planning Poker** ni se priorizan con MoSCoW (ambas técnicas son para HUs con valor de negocio). Las TTH se planifican como trabajo técnico directo en el cronograma del proyecto, con estimaciones en horas/días.

Sin embargo, las TTH **sí son prerrequisitos de bloques completos de HUs**:

- TTH-01 (Autenticación) es prerrequisito de toda HU operativa.
- TTH-02 (Docker) es prerrequisito de toda HU (no se desarrolla nada sin entorno).
- TTH-03 (CI) es prerrequisito para considerar cualquier HU "Done" con calidad asegurada.
- TTH-04 (Lógica de fallback) es prerrequisito de toda HU del Bloque C (HU-10, HU-11, HU-12) y de HU-13 del Bloque D.
- TTH-05 (Configuración de tiempos preconfigurados para degradado nivel 3) es prerrequisito del nivel 3 de TTH-04.
- TTH-06 (Capa de DTOs) **no es prerrequisito de ninguna HU del MVP1**. Clasificada como Trabajos Futuros.

Por tanto, las TTH se ejecutan en la primera fase del proyecto, antes o en paralelo con las HUs operativas. Las TTH-04 y TTH-05 específicamente deben estar disponibles antes de comenzar el sprint del Bloque C. TTH-06 queda documentada pero no entra al cronograma del proyecto académico.

---

## Documentos relacionados

- `DECISIONS_HU.md` — Decisiones metodológicas (DHU-001 a DHU-022) que fundamentan la creación y clasificación de cada TTH.
- `HU_BLOQUE_A.md` — Bloque A del Product Backlog tras la reestructuración.
- `HU_BLOQUE_B.md` — Bloque B del Product Backlog (HU-02 a HU-09, HU-22, HU-23).
- `HU_BLOQUE_C.md` — Bloque C del Product Backlog (HU-10 a HU-12).
- `HU_BLOQUE_D.md` — Bloque D del Product Backlog (HU-13, HU-14, HU-15).
- `HU_BLOQUE_E.md` — Bloque E del Product Backlog (0 HUs operativas; mapeo a TTH-07 a TTH-11).
- `HU_BLOQUE_F.md` — Bloque F del Product Backlog (HU-16, HU-17; F30 inglobada como CAs).
- `HU_MVP2.md` — MVP2 del Product Backlog (HU-18, HU-19, HU-20, HU-21; HU-09 reside en `HU_BLOQUE_B.md`). HU-20 consume CT-09.5 de TTH-09 extendido inglobadamente; HU-21 consume CT-04.4 y CT-04.5 de TTH-04. Sin TTH nuevas.
- `DECISIONS.md` — Decisiones técnicas del producto (D-001 Monolito modular, D-003 Deploy Docker local, D-008 SUMO end-to-end, D-009 jam level Waze).
- `FEATURE_BACKLOG_DETALLADO.md` — Features de origen (F01 autenticación, F26 lógica de fallback, F27 configuración de tiempos preconfigurados para degradado nivel 3, F29 RBAC, y habilitadores transversales).
