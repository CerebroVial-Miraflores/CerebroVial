# Historias de Usuario — Bloque A (Cerradas)

> Primera entrega del Product Backlog del proyecto CerebroVial.
>
> **Estado:** Bloque A cerrado y aprobado (versión 4, tras higiene documental aplicada retroactivamente por DHU-012). Bloques B, C, D, E y F del MVP1 posteriormente cerrados, y MVP2 también cerrado el 2026-05-16 (DHU-017). **Con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 21 HUs operativas (HU-01 a HU-21) + 11 TTH (TTH-01 a TTH-11).** Pendiente: documento RF/RNF (DHU-007), Planning Poker, MoSCoW, implementación SCRUM del MVP1. Los Bloques B y C se cerraron posteriormente a las versiones v2/v3 de este documento, en la misma jornada del 2026-05-13; el Bloque D se cerró el 2026-05-14; el Bloque E se cerró el 2026-05-15; el Bloque F se cerró el 2026-05-16; el MVP2 se cerró el 2026-05-16.
>
> **Fecha de cierre v1:** 2026-05-11
> **Fecha de cierre v2:** 2026-05-13 (tras DHU-001 a DHU-004)
> **Fecha de cierre v3:** 2026-05-13 (DHU-007 aplicada retroactivamente: sección Candidatos a RNF en HU-01)
> **Fecha de cierre v4:** 2026-05-14 (higiene documental por extensión de DHU-012 a este documento: rangos DHU actualizados, residuo de copy-paste en "Próximos pasos" corregido, typos depurados)
> **Fecha de cierre v5:** 2026-05-17 (DHU-018 aplicada retroactivamente: Resumen ejecutivo en HU-01)

---

## Contexto

Este documento contiene las Historias de Usuario del **Bloque A — Acceso al sistema** del Sequencer del Lean Inception (ver `LEAN_INCEPTION_CEREBROVIAL.md`, sección 9, y `FEATURE_BACKLOG_DETALLADO.md`, Bloque A).

Las HUs se redactan en el formato del documento de referencia académica (`Desarrollo_Agil.pdf`, Tablas 9 y 13): "Como X, quiero Y, para Z" con criterios de aceptación Given-When-Then.

### Cambios respecto a versiones anteriores

**Versión 1 (2026-05-11):** contenía 4 HUs (HU-01 Autenticación, HU-02 Acceso por rol, HU-03 Arquitectura Docker, HU-04 Repo + CI).

**Versión 2 (2026-05-13):** tras revisión metodológica documentada en `DECISIONS_HU.md` (decisiones DHU-001 a DHU-004), tres de esas HUs se reclasificaron como **Tareas Técnicas Habilitadoras** (TTH-01, TTH-02, TTH-03) y se trasladaron a `TAREAS_TECNICAS_HABILITADORAS.md`. La HU-02 original (acceso por rol) se reformuló con un valor de negocio más fuerte y quedó como única HU operativa del bloque.

**Versión 3 (2026-05-13):** durante la redacción del Bloque B se aprobó DHU-007 (RNF declarados como tales en sección específica al final de cada HU). Esta versión aplica DHU-007 retroactivamente a HU-01, agregando la sección "Candidatos a RNF".

**Versión 4 (2026-05-14):** higiene documental aplicada retroactivamente como extensión de DHU-012. La tabla de "Documentos afectados" original de DHU-012 (ver `DECISIONS_HU.md` subsección Documentos afectados) no incluyó este documento, lo que dejó pendientes algunos residuos detectados posteriormente en una revisión cruzada con los demás documentos del backlog. Los cambios v4 son: (a) actualización del rango de DHU referenciado en "Documentos relacionados" (DHU-001 a DHU-007 → DHU-001 a DHU-013), (b) corrección del residuo de copy-paste en "Próximos pasos" (la frase "Esta sesión cerró el Bloque B" pertenecía al cierre del Bloque B y se reemplaza por una redacción coherente con que este es el documento del Bloque A), (c) corrección de typo "inglogan" → "ingloban" en la sección "Persistencias movidas a otros bloques". El contenido sustantivo de HU-01 y de las reglas metodológicas declaradas en el documento se mantiene intacto.

**Versión 5 (2026-05-17):** aplicación retroactiva de DHU-018 (patrón "Resumen ejecutivo" agregado al inicio de cada HU del Product Backlog). HU-01 recibe el bloque de Resumen ejecutivo entre la cabecera y la sección "Descripción" existente. Por ser HU corta sin subdivisión `####`, el bloque omite el campo "Estructura de CAs" conforme al criterio de DHU-018. El contenido sustantivo de HU-01 (CAs, notas técnicas, Candidatos a RNF, clasificación) no se modifica.

**Resumen del cambio v1 → v2:**

| Versión 1 | Versión 2 |
|---|---|
| HU-01 Autenticación | TTH-01 (no es HU) |
| HU-02 Acceso por rol | HU-01 (reformulada) |
| HU-03 Arquitectura Docker | TTH-02 (no es HU) |
| HU-04 Repositorio + CI | TTH-03 (no es HU) |

**Total Bloque A:** 1 HU operativa + 3 TTH.

---

## Regla de sujetos en HUs (aplica a TODO el backlog)

Esta regla fue cerrada durante la revisión del Bloque A y aplica a todas las HUs del proyecto. Sustituye a la regla original de la versión 1. Ver DHU-003 en `DECISIONS_HU.md` para la fundamentación.

**Sujetos válidos en HUs del Product Backlog:**

1. Una de las 3 Personas del producto: **Operador de Tráfico Municipal**, **Gerente de Tránsito Municipal**, **Administrador del Sistema**.
2. Enumeración explícita de Personas cuando la funcionalidad es transversal (por ejemplo: "Operador, Gerente o Administrador").

**Sujetos NO válidos en HUs:**

- "El sistema" → si aparece, es una tarea técnica disfrazada.
- "Equipo de Desarrollo" → su trabajo se documenta como Tarea Técnica Habilitadora (`TAREAS_TECNICAS_HABILITADORAS.md`), no como HU.
- "Usuario" o "Usuario del sistema" sin especificar Persona → demasiado genérico, debilita el valor.

**Justificación:**

- Mike Cohn (*Mountain Goat Software*): cuando se redactan HUs es preferible ser tan específico como sea posible sobre el tipo de usuario, evitando sujetos genéricos como "usuario" o stakeholders que no son consumidores reales del valor.
- El documento de referencia académica (`Desarrollo_Agil.pdf`) usa sujetos compuestos cuando aplica (HU20–HU24 con "Administrador de Sistemas, Desarrollador"), lo cual sustenta la enumeración explícita de Personas para funcionalidades transversales.
- El trabajo técnico de infraestructura se modela como TTH siguiendo el patrón de "Enabler Stories" de SAFe y la postura de Lullabot ("Not Everything is a User Story").

---

## HU-01 — Acceso diferenciado por rol

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal, Gerente de Tránsito Municipal o Administrador del Sistema |
| **Quiero** | acceder únicamente a las funcionalidades correspondientes a mi rol |
| **Para** | concentrarme en mis responsabilidades específicas sin la carga cognitiva de información ajena a mi trabajo |

**Tipo:** HU de Persona (transversal a las 3).
**Feature(s) origen:** F29 (Sistema RBAC). F01 (Autenticación) está implícita como TTH-01 prerrequisito.

### Resumen ejecutivo

**Qué entrega:** control de acceso basado en roles (Operador, Gerente, Administrador). Cada Persona del producto ve únicamente las vistas y endpoints correspondientes a su rol, con valor cognitivo (concentración en el contexto propio) además del valor defensivo (segregación de permisos).

**CAs críticos:** CA-01.1 a CA-01.3 (segregación por rol en vistas), CA-01.4 (segregación por rol en API con respuesta 403 sin revelar el recurso), CA-01.5 (redirección al login cuando no hay sesión).

**Dependencias:** requiere TTH-01 (autenticación JWT con bcrypt) completada. Toda HU operativa del backlog tiene como precondición implícita "el usuario está autenticado" y referencia CA-01.4 o CA-01.5 según corresponda.

**Notas clave:** F01 se modeló como TTH-01 por DHU-001 (login no es HU). F29 se modeló como HU por DHU-002 (acceso diferenciado por rol entrega valor cognitivo, no solo permisivo). Asignación de roles directa en BD para MVP1 (no hay UI de gestión de roles).

### Descripción

Sistema de control de acceso basado en roles (RBAC) con tres roles diferenciados. Cada Persona del producto accede solo a las vistas y endpoints que corresponden a su rol operativo. El valor de esta segregación no es defensivo (segregar permisos) sino positivo: cada Persona trabaja con un contexto cognitivo distinto y la información ajena al rol es ruido que degrada su capacidad de operar:

- El **Operador** trabaja en tiempo real con presión de respuesta inmediata. Ver reportes ejecutivos o pantallas de configuración del modelo lo distrae.
- El **Gerente** trabaja con horizonte semanal/mensual y necesita información agregada. Ver el detalle operativo en tiempo real lo abruma sin aportarle nada decisional.
- El **Administrador** trabaja con la salud técnica del sistema. Ver KPIs ejecutivos o pantallas operativas no le sirve para diagnosticar componentes.

### Criterios de aceptación

- **CA-01.1:** Dado que el Operador inicia sesión correctamente, cuando ingresa al sistema, entonces visualiza únicamente las vistas de monitoreo en tiempo real (dashboard principal, panel del motor adaptativo, panel de degradación) y no puede acceder a vistas del Gerente ni del Administrador.

- **CA-01.2:** Dado que el Gerente inicia sesión correctamente, cuando ingresa al sistema, entonces visualiza únicamente las vistas de reportería ejecutiva (dashboard ejecutivo, comparativas, selector de periodo) y no puede acceder a vistas de monitoreo operativo ni de configuración técnica.

- **CA-01.3:** Dado que el Administrador inicia sesión correctamente, cuando ingresa al sistema, entonces visualiza únicamente las vistas de configuración y salud del sistema (panel de componentes, métricas del modelo, configuración del motor) y no puede acceder a vistas operativas ni ejecutivas.

- **CA-01.4:** Dado que un usuario autenticado intenta acceder vía API a un endpoint que no corresponde a su rol, cuando la solicitud llega al backend, entonces el sistema responde con HTTP 403 Forbidden sin filtrar información del recurso solicitado.

- **CA-01.5:** Dado que un usuario no ha iniciado sesión, cuando intenta acceder a cualquier vista del sistema, entonces es redirigido a la pantalla de login.

- **CA-01.6:** Dado que la sesión del usuario ha expirado (token JWT vencido), cuando realiza una acción que requiere autenticación, entonces el sistema lo desconecta y lo redirige a la pantalla de login con un mensaje informativo.

### Notas técnicas

- **Dependencia:** Esta HU requiere TTH-01 (Autenticación JWT con bcrypt) completada. Ver `TAREAS_TECNICAS_HABILITADORAS.md`.
- **Modelo:** Campo `role` en la tabla `User` con valores enum (`operator`, `manager`, `admin`).
- **Asignación de roles:** Para MVP1 la asignación se hace directamente en BD; no hay UI de gestión de roles (default 3 del Parking Lot del Inception).
- **Implementación backend:** Decorators o dependencies de FastAPI por endpoint, validando el claim `role` del JWT.
- **Implementación frontend:** Routing condicional según el rol del usuario autenticado; las rutas no accesibles no se renderizan ni siquiera como enlaces.

### Candidatos a RNF (para futuro documento RF/RNF)

Sección agregada en v3 por aplicación retroactiva de DHU-007.

- **RNF de seguridad:** un usuario autenticado nunca debe acceder a recursos fuera de su rol vía API (CA-01.4). Este es probablemente el RNF más crítico de seguridad del sistema. → RNF-SEC-03 (Control de acceso por rol).
- **RNF de seguridad / sesión:** el token JWT debe tener tiempo de expiración configurable (CA-01.6 implica expiración; el valor por defecto vive en TTH-01). → RNF-SEC-02 (Autenticación del usuario al sistema).
- **RNF de usabilidad:** las rutas no accesibles no deben aparecer ni siquiera como enlaces (notas técnicas). Esto reduce confusión del usuario y refuerza el valor cognitivo declarado en la HU. → RNF-INT-07 (Ocultación de rutas no accesibles al rol).
- **RNF de privacidad:** la respuesta HTTP 403 no debe filtrar información del recurso solicitado (CA-01.4). Esto evita oracles que permitan a un atacante enumerar recursos por respuestas distintas. → RNF-SEC-04 (No filtración de información en respuestas de error de control de acceso).

---

## Resumen del Bloque A

| HU | Título | Sujeto | Tipo | Feature(s) origen |
|---|---|---|---|---|
| HU-01 | Acceso diferenciado por rol | Operador, Gerente o Administrador | Persona | F29 (F01 implícita como TTH-01) |

**Total Bloque A: 1 HU operativa + 3 TTH** (ver `TAREAS_TECNICAS_HABILITADORAS.md`).

---

## Tareas Técnicas Habilitadoras del Bloque A

Estas tres TTH **no son HUs** y se documentan en detalle en `TAREAS_TECNICAS_HABILITADORAS.md`. Se listan aquí solo para mantener trazabilidad del bloque.

| TTH | Título | Versión 1 era... | Estado actual |
|---|---|---|---|
| TTH-01 | Implementación de autenticación JWT con bcrypt | HU-01 | Pendiente |
| TTH-02 | Arquitectura Docker Compose multi-servicio | HU-03 | Parcial |
| TTH-03 | Repositorio Git y pipeline CI con cobertura completa | HU-04 | Parcial |

---

## Persistencias movidas a otros bloques

Las features F30 (persistencia de estados históricos) y F31 (persistencia de decisiones del motor) NO se redactan como HUs separadas, sino que se ingloban como criterios de aceptación de HUs de Personas reales:

- **F30 (persistencia de estados históricos)** → criterios de aceptación en las HUs del Gerente (Bloque F). En la redacción original del Bloque A esto era una promesa proyectada (las HUs derivarían de F12 dashboard ejecutivo y F14 comparativa entre periodos). Al cerrarse el Bloque F (DHU-016), la promesa se materializó así: **F30 quedó inglobada como CA-16.1 a CA-16.3 de HU-16** (que fusiona F12 y F13); HU-17 consume el sustrato vía HU-16, sin reinglobarlo. Ver `HU_BLOQUE_F.md`.

- **F31 (persistencia de decisiones del motor)** → criterios de aceptación en la HU del log de decisiones del Operador (Bloque B). Esto **ya ocurrió:** HU-08 del Bloque B contiene F31 inglobada en CA-08.1. Ver `HU_BLOQUE_B.md`.

---

## Próximos pasos

Este documento cierra el Bloque A. A la fecha actual, los Bloques B, C, D, E y F del MVP1 están cerrados y el MVP2 también está cerrado (DHU-017, 2026-05-16); **con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 21 HUs operativas (HU-01 a HU-21) y 11 TTH (TTH-01 a TTH-11).** Los siguientes pasos del proyecto, en sesiones futuras:

1. **Bloque B — Operador, núcleo de monitoreo** (8 HUs: HU-02 a HU-09; ya cerrado). Ver `HU_BLOQUE_B.md`.
2. **Bloque C — Operador, operación degradada** (3 HUs operativas + 2 TTH; ya cerrado). Ver `HU_BLOQUE_C.md`.
3. **Bloque D — Administrador, soporte técnico** (3 HUs operativas; ya cerrado). Ver `HU_BLOQUE_D.md`.
4. **Bloque E — Componentes centrales del sistema** (0 HUs operativas + 5 TTH: TTH-07 a TTH-11; ya cerrado el 2026-05-15 por DHU-015). Ver `HU_BLOQUE_E.md`.
5. **Bloque F — Gerente, reportería mínima** (F12 + F13 fusionadas en HU-16 con F30 inglobada, F14 en HU-17 → 2 HUs operativas + 0 TTH nuevas; ya cerrado el 2026-05-16 por DHU-016). Ver `HU_BLOQUE_F.md`.
6. **MVP2 — HUs documentadas con construcción condicional a holgura del cronograma tras cerrar MVP1** (5 HUs: F11→HU-09 del Bloque B; F15→HU-18; F16→HU-19; F19→HU-20; F28→HU-21; 0 TTH nuevas; cerrado el 2026-05-16 por DHU-017). Semántica refinada por DHU-012. Ver `HU_MVP2.md`.

Tras cerrar el MVP2 (ya hecho), los próximos pasos del proyecto, fuera del alcance del Product Backlog funcional, son:

1. **Documento de Requisitos Funcionales y No Funcionales (RF/RNF)** consolidando los "Candidatos a RNF" de todas las HUs (HU-01 a HU-21) en un documento único aprobado, conforme a DHU-007 pendiente. Sesión dedicada futura.
2. **Ceremonias de estimación (Planning Poker) y priorización (MoSCoW)** sobre el backlog completo.
3. **Implementación SCRUM del MVP1** (16 HUs operativas + 11 TTH del MVP1). El MVP2 (5 HUs adicionales) entra al sprint si hay holgura de cronograma, conforme a la semántica refinada por DHU-012.
4. **SDD (Software Design Document)**, siguiente entregable académico mayor del proyecto.

---

## Documentos relacionados

- `HU_BLOQUE_B.md` — Bloque B del Product Backlog (8 HUs: HU-02 a HU-09).
- `HU_BLOQUE_C.md` — Bloque C del Product Backlog (3 HUs operativas: HU-10, HU-11, HU-12).
- `HU_BLOQUE_D.md` — Bloque D del Product Backlog (3 HUs operativas: HU-13, HU-14, HU-15).
- `HU_BLOQUE_E.md` — Bloque E del Product Backlog (0 HUs operativas; mapeo a TTH-07 a TTH-11 y decisiones tomadas durante la redacción).
- `HU_BLOQUE_F.md` — Bloque F del Product Backlog (2 HUs operativas: HU-16, HU-17; F30 inglobada como CAs).
- `HU_MVP2.md` — MVP2 del Product Backlog (HU-18, HU-19, HU-20, HU-21; HU-09 reside en `HU_BLOQUE_B.md`).
- `DECISIONS_HU.md` — Decisiones metodológicas sobre redacción de HUs (DHU-001 a DHU-022). **Lectura obligatoria** antes de redactar nuevas HUs.
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-01, TTH-02, TTH-03 transversales; TTH-04 y TTH-05 del Bloque C; TTH-06 Trabajos Futuros; TTH-07 a TTH-11 del Bloque E.
- `DECISIONS.md` — Registro formal de decisiones técnicas del producto (D-001 a D-009).
- `EVOLUCION_TESIS.md` — Narrativa de las 4 fases del proyecto.
- `LEAN_INCEPTION_INVESTIGACION.md` — Fundamentación del marco metodológico.
- `LEAN_INCEPTION_CEREBROVIAL.md` — Inception completo aplicado al proyecto.
- `FEATURE_BACKLOG_DETALLADO.md` — Detalle completo de las 41 features identificadas (29 MVP1 + 5 MVP2 + 7 Trabajos Futuros).
- `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` — Documento normativo denso con catálogo de 22 RF y 53 RNF clasificados según ISO/IEC 25010:2023, redactado el 2026-05-18 ejecutando DHU-007 según las decisiones metodológicas consolidadas en DHU-019. `RF_RNF_LITE.md` — versión lite de lectura humana, derivada conforme al modelo de dos documentos cerrado en DHU-019 subsección H.
