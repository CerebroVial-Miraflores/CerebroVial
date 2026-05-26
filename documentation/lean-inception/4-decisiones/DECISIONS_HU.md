# DECISIONS_HU — Decisiones metodológicas sobre la redacción de Historias de Usuario

> Registro formal de decisiones que afectan la redacción del Product Backlog del proyecto CerebroVial.
>
> **Alcance:** Estas decisiones aplican a TODO el Product Backlog (Bloques A–F + MVP2). Cualquier HU redactada después de la fecha de cada decisión debe respetarla.
>
> **Relación con `DECISIONS.md`:** El documento `DECISIONS.md` registra decisiones técnicas del producto (arquitectura, modelo, datos). Este documento registra decisiones metodológicas sobre cómo se redacta el backlog. Los códigos no se solapan: `D-xxx` para técnicas, `DHU-xxx` para HUs.
>
> **Fecha de creación:** 2026-05-13
> **Última actualización:** 2026-05-20 (**DHU-021 y DHU-022 agregadas.** DHU-021 consolida las decisiones metodológicas de redacción del SDD: 17 de redacción más 4 ajustes derivados de la verificación del documento contra el repositorio vivo (`node_id` como FK a `graph_nodes` resuelto en el write-path; conservación de `flow_total`/`y_load_factor`/`inputs_snapshot` capturados del cálculo interno del motor; ratificación de que el sistema no opera en lazo cerrado autónomo; exclusión de la integración Gemini de la arquitectura objetivo con remoción diferida como saneamiento). DHU-022 cierra Delta-02 fijando `operator/manager/admin` como claims técnicos canónicos con mapeo a labels en español en el frontend. Ambas son del 2026-05-20. Última previa: DHU-020 (**DHU-020 agregada: semántica de ControlView, cierre de Delta-08.** Resuelve el riesgo R1 de la planificación del Sprint 4. Cierra cinco subsecciones: la semántica pasiva de HU-05 prevalece y la HU se mantiene sin enmiendas (se descarta legitimar el playground); el playground interactivo actual se preserva como herramienta de Administrador/validación en lugar de eliminarse, conservando su valor docente para la tesis; se declara pendiente un elemento de backlog propio que cubra ese playground; Delta-07, Delta-08 y Delta-09 se abordan en un único refactor en bloque por tocar los mismos archivos; y se reconoce explícitamente que construir la persistencia de "estado vigente del motor" es un cambio estructural autorizado deliberadamente conforme a la guardia de CLAUDE.md. Decisión de alineación especificación↔código; no modifica HUs, TTH ni decisiones técnicas. Última previa: 2026-05-18, DHU-019 (**DHU-019 agregada: decisiones metodológicas para la redacción del documento de Requisitos Funcionales y No Funcionales (RF/RNF).** Ejecuta la sesión dedicada que DHU-007 declaró pendiente. Cierra en un acto único nueve subsecciones de decisiones: adopción de ISO/IEC 25010:2023 como taxonomía formal (9 características), reasignación masiva de las categorías heterogéneas declaradas en DHU-007 a las características formales del estándar, resolución normativa de siete inconsistencias detectadas en los Candidatos a RNF de las 21 HUs, plantilla unificada de RF y RNF, política de derivación de RF desde CAs por composición transversal, política de prioridades MoSCoW sugeridas, política aditiva no destructiva sobre las HUs (los CAs preservan su redacción literal y los Candidatos a RNF reciben pasada aditiva con referencias `→ RNF-XXX-NN`), nota terminológica RF vs RNF-FUN y modelo de dos documentos (denso normativo + lite de lectura humana). Cambio metodológico sin alterar contenido sustantivo de HUs ni TTH. Última previa: 2026-05-17, DHU-018 (patrón "Resumen ejecutivo" retroactivo).

---

## Índice de decisiones

| Código | Título | Fecha | Estado |
|---|---|---|---|
| DHU-001 | El login no es una HU; es una tarea técnica habilitadora | 2026-05-13 | Cerrada |
| DHU-002 | Reformulación del valor en HU de acceso diferenciado por rol | 2026-05-13 | Cerrada |
| DHU-003 | Sujetos válidos en HUs y exclusión del Equipo de Desarrollo | 2026-05-13 | Cerrada |
| DHU-004 | Tareas Técnicas Habilitadoras como categoría separada del Product Backlog | 2026-05-13 | Cerrada |
| DHU-005 | Principio de robustez ante interrupción de fuente de información | 2026-05-13 | Cerrada (refinada con Casos A y B durante Bloque B) |
| DHU-006 | HUs agnósticas a la implementación | 2026-05-13 | Cerrada |
| DHU-007 | RNF declarados como tales en sección específica | 2026-05-13 | Cerrada |
| DHU-008 | Distinción arquitectónica entre componente caído, modo degradado y lógica de fallback | 2026-05-13 | Cerrada (nota agregada 2026-05-14: renombrado "modo seguro" → "degradado nivel 3") |
| DHU-009 | Relación entre marca pasiva (Bloque B) y alerta activa (Bloque C) | 2026-05-13 | Cerrada |
| DHU-010 | Criterios para clasificar trabajo del Bloque C como TTH | 2026-05-13 | Cerrada |
| DHU-011 | Eliminación de HU-13 y cobertura de F25 por composición | 2026-05-13 | Cerrada |
| DHU-012 | Auditoría de coherencia documental: semántica de MVP, eliminación de MVP3, corrección de conteos, alineación de vocabulario, limpieza de residuo pre-Inception | 2026-05-14 | Cerrada |
| DHU-013 | Clasificación HU/TTH de las features del Bloque D | 2026-05-14 | Cerrada |
| DHU-014 | Decisiones de redacción del Bloque D (numeración, dashboard, parámetros, métricas, concurrencia, ventana temporal, TTH-06) | 2026-05-14 | Cerrada |
| DHU-015 | Clasificación HU/TTH de las features del Bloque E (con ampliación 4 → 5 TTH durante la redacción) | 2026-05-15 | Cerrada |
| DHU-016 | Decisiones de redacción del Bloque F (numeración, F30 inglobada, fuente del histórico en MVP1, KPIs operacionales, granularidad, periodos, comparativa, concurrencia, dashboard integrador, robustez) | 2026-05-16 | Cerrada |
| DHU-017 | Decisiones de redacción del MVP2 (clasificación HU/TTH de las 4 features pendientes, numeración compactada, F16 como HU única, F19 sustrato inglobado, F28 como HU única con Operador protagonista, alcance del escalamiento, alcance del drill-down de F15, conexión F15 ↔ HU-16/HU-17, sustrato inglobado vs TTH, política MVP2 heredada, robustez Caso B) | 2026-05-16 | Cerrada |
| DHU-018 | Patrón "Resumen ejecutivo" agregado retroactivamente al inicio de cada HU del Product Backlog | 2026-05-17 | Cerrada |
| DHU-019 | Decisiones metodológicas para la redacción del documento de Requisitos Funcionales y No Funcionales (RF/RNF): adopción de ISO/IEC 25010:2023, reasignación de categorías DHU-007, plantillas unificadas, derivación de RF, resolución de inconsistencias, política aditiva, nota RF vs RNF-FUN, modelo de dos documentos | 2026-05-18 | Cerrada |
| DHU-020 | Semántica de ControlView: cierre de Delta-08 (vista pasiva de HU-05 prevalece, playground preservado como herramienta de Administrador, Delta-07/08/09 en bloque, persistencia de estado vigente autorizada como cambio estructural) | 2026-05-20 | Cerrada |
| DHU-021 | Decisiones metodológicas de redacción del SDD (17 de redacción + 4 ajustes derivados de la verificación SDD↔repo: node_id FK con resolución al persistir, campos del motor conservados, ratificación sin-lazo-cerrado, Gemini fuera de la arquitectura objetivo) | 2026-05-20 | Cerrada |
| DHU-022 | Nomenclatura de roles del sistema: `operator/manager/admin` como claims técnicos canónicos + labels en español en el frontend (cierre de Delta-02) | 2026-05-20 | Cerrada |

---

## DHU-001 — El login no es una HU; es una tarea técnica habilitadora

**Fecha:** 2026-05-13.
**Reemplaza:** HU-01 (Autenticación al sistema) de la versión inicial del Bloque A.

### Contexto

La versión inicial del Bloque A contenía HU-01 redactada como:

> *Como Usuario del sistema (Operador, Gerente o Administrador), quiero autenticarme con mi nombre de usuario y contraseña, para acceder a las funcionalidades del sistema según mi rol.*

Durante la revisión del Bloque A se cuestionó si el login debía modelarse como HU. La discusión se resolvió consultando bibliografía especializada.

### Bibliografía consultada

**Postura a favor de modelar login como HU:**

- **Mike Cohn** (referencia bibliográfica principal del tema) usa explícitamente el login como ejemplo de HU en *User Stories Applied: For Agile Software Development* y en Mountain Goat Software:
  - *"As a customer, I can regain access to my account when I forget my password."*
  - *"As a user, I can log in through my Facebook / LinkedIn / Twitter account."* (como ejemplo de splitting).

**Postura en contra de modelar login como HU:**

- **Lullabot** ("Not Everything is a User Story"): *"We like to surrender to the forces of common sense and call a user story that no longer involves a user what it actually is: a task for a developer to perform."*

- **Scrum.org** ("User Story or Stakeholder Story?"): *"As a user I want to login so I can use the service. At first it seems ok, but here the user is not getting the value (if I could use the service without logging in, I would be happy, after all I want to do my job and logging in brings no value)."*

- **Práctica habitual en certificaciones PMP/PMI-ACP:** "Login Story → Tasks: API · UI · Validation · Error handling · Testing." Es decir, el login se descompone como conjunto de tareas técnicas que cuelgan de HUs de mayor valor.

### Análisis aplicado al caso CerebroVial

El login en CerebroVial es **estándar y sin sofisticación de negocio**: JWT + bcrypt, sin recuperación de contraseña, sin SSO, sin doble factor, sin bloqueo por intentos. Aplicando el filtro INVEST a la HU original:

- **V (Valuable):** "Para acceder a las funcionalidades según mi rol" es tautológico. El valor real está río abajo (monitorear, reportar, configurar), no en el acto de loguearse.
- **I (Independent):** No se puede entregar valor con HU-01 sin HU-02 (autenticar sin diferenciar roles no entrega nada). Están atadas.
- **N (Negotiable):** El comportamiento del login es técnico estándar; no hay conversación de negocio que tener.

### Decisión

**El login se elimina del Product Backlog como HU.** Se documenta como **Tarea Técnica Habilitadora** (TTH-01) en el documento `TAREAS_TECNICAS_HABILITADORAS.md`.

Los requisitos de autenticación que afectan a HUs operativas se ingloban como **criterios de aceptación** de esas HUs (por ejemplo: "Dado que el Operador no ha iniciado sesión, cuando intenta acceder al dashboard, entonces el sistema lo redirige al login").

### Consecuencias

- HU-01 original (Autenticación) se elimina del Bloque A.
- Se crea TTH-01 (Implementación de autenticación JWT con bcrypt) en `TAREAS_TECNICAS_HABILITADORAS.md`.
- Las HUs operativas que requieran autenticación incluyen un CA específico de redirección al login.

---

## DHU-002 — Reformulación del valor en HU de acceso diferenciado por rol

**Fecha:** 2026-05-13.
**Reemplaza:** HU-02 (Acceso diferenciado por rol) de la versión inicial del Bloque A.

### Contexto

La versión inicial de HU-02 declaraba como valor: *"para usar el sistema dentro de mis responsabilidades sin interferir con otras áreas"*. La redacción es técnicamente correcta pero el "valor para el usuario" es débil: nadie se entusiasma con "no interferir con otras áreas".

Tras eliminar HU-01 (DHU-001), HU-02 queda como la HU de acceso al sistema y necesita un valor de negocio más fuerte y defendible.

### Análisis

El valor real del acceso diferenciado por rol no es operativo (segregación de permisos) sino **cognitivo**: cada Persona tiene un contexto de trabajo distinto y la información ajena a su rol es ruido que degrada su capacidad de operar.

- El **Operador** trabaja en tiempo real con presión de respuesta inmediata. Ver reportes ejecutivos o pantallas de configuración del modelo lo distrae.
- El **Gerente** trabaja con horizonte semanal/mensual y necesita información agregada. Ver el detalle operativo en tiempo real lo abruma sin aportarle nada decisional.
- El **Administrador** trabaja con la salud técnica del sistema. Ver KPIs ejecutivos o pantallas operativas no le sirve para diagnosticar componentes.

Este valor (concentración / reducción de carga cognitiva) es defendible en sustentación y conecta con principios de diseño centrado en el usuario.

### Decisión

HU-02 (renumerada como HU-01 en el Bloque A actualizado) se reformula así:

> *Como Operador, Gerente o Administrador, quiero acceder únicamente a las funcionalidades correspondientes a mi rol, para concentrarme en mis responsabilidades específicas sin la carga cognitiva de información ajena a mi trabajo.*

### Consecuencias

- HU-02 original se elimina y se reescribe como nueva HU-01 del Bloque A.
- El valor pasa de "no interferir" (defensivo) a "concentrarme" (positivo, defendible).

---

## DHU-003 — Sujetos válidos en HUs y exclusión del Equipo de Desarrollo

**Fecha:** 2026-05-13.
**Reemplaza:** La regla original de sujetos del Bloque A.

### Contexto

La regla original (cerrada en la sesión previa) admitía dos tipos de sujeto:

1. HUs operativas → Persona del producto.
2. HUs técnicas → "Equipo de Desarrollo" como Stakeholder.

Tras DHU-001 y DHU-004, la categoría (2) deja de ser necesaria: el trabajo técnico se modela como Tareas Técnicas Habilitadoras, no como HUs. La regla se simplifica.

### Decisión

**Sujetos válidos en HUs del Product Backlog:**

1. Una de las 3 Personas del producto: **Operador de Tráfico Municipal**, **Gerente de Tránsito Municipal**, **Administrador del Sistema**.
2. Enumeración explícita de Personas cuando la funcionalidad es transversal (por ejemplo: "Operador, Gerente o Administrador").

**Sujetos NO válidos en HUs:**

- "El sistema" (sería una tarea técnica disfrazada).
- "Equipo de Desarrollo" (su trabajo se documenta como Tarea Técnica Habilitadora, no como HU).
- "Usuario" o "Usuario del sistema" sin especificar Persona (demasiado genérico, debilita el valor).

### Justificación bibliográfica

- **Mike Cohn** (*Mountain Goat Software*): *"Note that you don't see any user story, 'As a product owner, I want a list of certification courses so that...'. The product owner is an essential stakeholder, but is not the end user/customer. When creating user stories, it's best to be as specific as possible about the type of user."*

- El documento de referencia académica (`Desarrollo_Agil.pdf`) usa sujetos compuestos cuando aplica (HU20–HU24 con "Administrador de Sistemas, Desarrollador"), lo cual sustenta la enumeración explícita de Personas.

### Consecuencias

- HU-03 y HU-04 originales del Bloque A (con sujeto "Equipo de Desarrollo") dejan de ser HUs y se convierten en Tareas Técnicas Habilitadoras (ver DHU-004).
- La sección "Regla de sujetos en HUs" del Bloque A se reescribe según esta decisión.

---

## DHU-004 — Tareas Técnicas Habilitadoras como categoría separada del Product Backlog

**Fecha:** 2026-05-13.

### Contexto

Tras DHU-001 (eliminación del login como HU) y DHU-003 (exclusión del Equipo de Desarrollo como sujeto), surge la necesidad de **documentar el trabajo técnico de infraestructura** (setup Docker, repositorio Git, CI, autenticación) en algún lugar visible y trazable, sin contaminarlo con el formato de HU.

### Bibliografía consultada

El concepto de **Enabler** o **Tarea Técnica Habilitadora** está formalizado en varias fuentes:

- **SAFe (Scaled Agile Framework)** distingue "Business Stories" (valor de negocio) de "Enabler Stories" (trabajo técnico habilitador): infraestructura, exploración, arquitectura, cumplimiento.
- **Lullabot** ("Not Everything is a User Story"): *"In general, it's better to surrender to common sense and not put these kinds of technical requirements into the user's voice. Instead, write simple, imperative statements that declare what must be done."*
- **Práctica común en certificaciones PMP/PMI-ACP:** distinguir explícitamente "User Story" (valor) de "Task" (trabajo técnico).

### Decisión

El proyecto CerebroVial mantiene **dos categorías separadas** dentro de su gestión de backlog:

1. **Product Backlog (HUs):** Contiene únicamente HUs con sujeto Persona del producto. Formato: "Como X, quiero Y, para Z" con criterios Given-When-Then.

2. **Tareas Técnicas Habilitadoras (TTH):** Contiene el trabajo técnico de infraestructura necesario para que las HUs puedan ser implementadas. Formato: enunciado imperativo + descripción + criterios técnicos de "terminado" (sin Given-When-Then).

Ambas categorías son entregables del proyecto y ambas son evaluables, pero no se mezclan en un mismo documento ni se priorizan con los mismos criterios.

### Ubicación física

- HUs → `HU_BLOQUE_A.md`, `HU_BLOQUE_B.md`, etc.
- TTH → `TAREAS_TECNICAS_HABILITADORAS.md` (documento único transversal a todos los bloques).

### Criterios para clasificar un trabajo como TTH y no como HU

Un trabajo se documenta como TTH (no como HU) si cumple **alguna** de estas condiciones:

1. No tiene una Persona del producto como beneficiaria directa.
2. Su valor es instrumental (habilita otras funcionalidades) y no de negocio.
3. Su comportamiento es técnico estándar sin negociación de negocio.
4. No se entrega valor visible al usuario al completarla en aislamiento.

### Consecuencias

- Se crea el documento `TAREAS_TECNICAS_HABILITADORAS.md`.
- HU-03 y HU-04 originales del Bloque A se convierten en TTH-02 (Arquitectura Docker) y TTH-03 (Repositorio + CI).
- HU-01 original del Bloque A se convierte en TTH-01 (Autenticación JWT) por DHU-001.
- El Bloque A queda con una sola HU (la antes llamada HU-02, ahora HU-01 reformulada).

---

## DHU-005 — Principio de robustez ante interrupción de fuente de información

**Fecha:** 2026-05-13.
**Estado:** Cerrada (versión inicial durante HU-02; refinada con Casos A y B durante HU-05).

### Contexto

Durante la redacción de HU-02 (monitoreo en tiempo real) se identificó la necesidad de un comportamiento explícito ante pérdida temporal de la fuente de datos: el sistema no debe mostrar datos viejos al Operador como si fueran actuales, porque eso lo lleva a tomar decisiones sobre información inválida.

Durante la redacción de HU-05 (visualización de estrategia activa) se descubrió que el principio original era insuficiente: el caso de un componente interno de decisión caído (motor adaptativo no responde) es distinto al caso de una fuente externa de medición caída (mediciones del tráfico no llegan). La diferencia semántica importa al usuario.

### Decisión

Toda HU operativa que muestre información dependiente de una fuente externa de medición o de un componente interno del sistema debe incluir un criterio de aceptación explícito sobre el comportamiento de la vista cuando esa fuente deja de actualizar el dato. La información no debe presentarse al Operador como si fuera vigente cuando no podemos garantizarlo.

**Casos cubiertos:**

**Caso A — Fuente externa de medición del mundo observado.** Aplica a HUs que muestran mediciones del estado real del tráfico (flujos, colas, velocidades). Cuando la fuente de medición deja de emitir, el sistema mantiene en pantalla los últimos valores conocidos, los marca visualmente como **"desactualizados"** e indica el tiempo transcurrido desde la última actualización. La palabra "desactualizado" comunica que el dato existe pero no refleja necesariamente lo que está pasando ahora.

**Caso B — Componente interno de decisión del sistema.** Aplica a HUs que muestran decisiones tomadas por componentes internos del sistema (estrategia de control activa, predicciones del modelo, explicaciones, eventos de notificación). Cuando el componente decisor deja de responder, el sistema mantiene en pantalla la última decisión conocida, la marca visualmente como **"no confirmada"** e indica el tiempo transcurrido desde la última confirmación. La palabra "no confirmada" comunica que no podemos garantizar que esa decisión siga vigente, porque el componente que la confirma no está respondiendo.

### Alcance de cada HU

Cada HU operativa es responsable únicamente de **marcar pasivamente** su propio panel según el caso que corresponda (A o B). La **notificación activa al Operador** ante una caída de componente que afecta la operación general del sistema es responsabilidad de las HUs del Bloque C (operación degradada), que actúan de forma transversal a todas las vistas. Esta separación se formaliza en DHU-009.

Las HUs operativas **no duplican** esa lógica ni la referencian explícitamente; cada bloque cumple su responsabilidad:

- **Bloque B (monitoreo):** marca pasiva en cada panel afectado.
- **Bloque C (degradación):** alerta activa transversal sobre el estado del sistema completo.

### Justificación

Separar marca pasiva (responsabilidad de cada vista) de alerta activa (responsabilidad transversal) evita duplicación de lógica, mantiene HUs cohesivas, y respeta la separación que ya estaba implícita en el Sequencer del Lean Inception (Bloque B = monitoreo, Bloque C = degradación).

### Aplicaciones del principio en el backlog

| HU | Caso aplicado | CA específico |
|---|---|---|
| HU-02 (monitoreo) | Caso A (fuente de medición) | CA-02.4 |
| HU-03 (predicción) | Caso B (componente predictivo) | CA-03.4 |
| HU-04 (vista combinada) | Casos A + B independientes | CA-04.4 |
| HU-05 (estrategia activa) | Caso B (motor adaptativo) | CA-05.4 |
| HU-06 (explicación) | Caso B (componente de explicación) | CA-06.4 |
| HU-07 (notificación) | Caso B aplicado a canal de eventos | CA-07.5 |
| HU-08 (log) | Variante de resiliencia (operación no detenida) | CA-08.5 |
| HU-09 (notas) | Información de error al Operador | CA-09.5 |

---

## DHU-006 — HUs agnósticas a la implementación

**Fecha:** 2026-05-13.

### Contexto

Durante la redacción de HU-02 se detectó la tentación de mencionar tecnologías concretas en la HU (por ejemplo, "el módulo de visión o el simulador SUMO emite métricas"). Esto contamina la HU con detalles de implementación y la ata a una arquitectura específica.

### Análisis

Mike Cohn formula el principio así: las HUs deben describir el **qué** (el comportamiento observable por el usuario) y no el **cómo** (la implementación). Las Personas del producto no son consumidoras de detalles técnicos; son consumidoras de comportamiento.

En el caso de CerebroVial, este principio tiene consecuencias prácticas importantes:

- "Módulo de visión computacional" → implementación. La HU dice "el sistema observa el tráfico".
- "Simulador SUMO" → implementación. La HU dice "fuente de datos del tráfico".
- "Modelo GRU" → implementación. La HU dice "el sistema predice".
- "Webster / MaxPressure / MTC" → implementación. La HU dice "el sistema selecciona la estrategia de control".
- "Waze jam level" → constructo técnico. La HU dice "nivel de congestión en escala 0-5".

### Decisión

Las HUs **NO** mencionan tecnologías, componentes técnicos, frameworks ni constructos específicos en su redacción (Como/Quiero/Para, Descripción, Criterios de aceptación). Estos detalles viven en:

- **Tareas Técnicas Habilitadoras** (`TAREAS_TECNICAS_HABILITADORAS.md`): especifican componentes técnicos con todo detalle.
- **Decisiones técnicas** (`DECISIONS.md`): registran las elecciones tecnológicas formales (D-001 a D-009).
- **Notas técnicas** dentro de cada HU: pueden referenciar decisiones técnicas por código (por ejemplo, "según D-008") sin nombrar tecnologías concretas.
- **Documento RF/RNF futuro**: especificará requisitos técnicos detallados.

### Excepciones

Se permite mencionar el constructo "nivel de congestión 0-5" (escala ordinal) en las HUs sin referenciar a Waze por nombre, porque la escala 0-5 ya es un concepto autónomo del sistema CerebroVial, aunque su origen sea Waze. El detalle de la adopción y mapeo vive en D-009.

### Consecuencias

- Toda HU del Bloque B (HU-02 a HU-09) cumple este principio.
- Si se detecta una HU previamente redactada que viola el principio, se reescribe sin perder su contenido funcional.
- Al redactar nuevos bloques, el redactor revisa cada CA para verificar que no se introdujeron menciones técnicas inadvertidas.

---

## DHU-007 — RNF declarados como tales en sección específica

**Fecha:** 2026-05-13.

### Contexto

Durante la redacción de HU-02 se observó que los CAs estaban absorbiendo **requisitos no funcionales** (RNF) que técnicamente pertenecen a otro tipo de documento: umbrales de latencia, comportamiento ante fallos, criterios de usabilidad, etc. Por ejemplo:

> *CA-02.2: ...con una latencia máxima de 5 segundos desde que la medición se genera.*

"Latencia máxima de 5 segundos" no es un criterio funcional (¿qué hace el sistema?); es un criterio de rendimiento (¿con qué calidad lo hace?). Esto pertenece a la familia de requisitos no funcionales según ISO/IEC 25010.

### Análisis

La práctica académica y de industria recomienda separar RF de RNF:

- **RF (Requisitos Funcionales):** qué hace el sistema. Se derivan de las HUs.
- **RNF (Requisitos No Funcionales):** cómo lo hace. Rendimiento, disponibilidad, seguridad, usabilidad, mantenibilidad, portabilidad.

Si los umbrales numéricos viven hardcodeados en cada CA:

- Cambiar un umbral implica modificar múltiples HUs.
- No hay un lugar único donde consultar todos los RNF del sistema.
- El jurado académico no ve trazabilidad explícita entre HU y RNF.

Sin embargo, eliminar los umbrales de los CAs ahora mismo (durante la redacción inicial) tendría el costo de **perder información** antes de tener el documento RF/RNF formal donde reubicarla. La solución es marcar explícitamente qué partes de cada HU son candidatas a RNF.

### Decisión

Cada HU del Product Backlog incluye al final una sección **"Candidatos a RNF (para futuro documento RF/RNF)"** que lista los criterios numéricos, de robustez, de usabilidad u otros que probablemente se moverán al documento RF/RNF cuando se redacte. Esto:

1. Da trazabilidad futura sin frenar el trabajo actual.
2. Permite mantener los umbrales en los CAs por ahora (para que los CAs sean autocontenidos durante la redacción).
3. Anticipa qué criterios se reemplazarán por referencias `RNF-XXX-NN` cuando exista el documento formal.

### Formato de la sección

```markdown
### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** [descripción] (referencia al CA donde aparece).
- **RNF de robustez:** [descripción] (referencia al CA donde aparece).
- **RNF de usabilidad:** [descripción]. Probablemente se valida con prueba de usuario.
- **RNF de [otra categoría]:** [descripción].
```

Las categorías típicas (siguiendo ISO/IEC 25010) son: rendimiento, robustez, usabilidad, seguridad, mantenibilidad, portabilidad, escalabilidad, configurabilidad, persistencia, auditoría, retención, trazabilidad, inmutabilidad.

### Trabajo futuro asociado a esta decisión

Tras cerrar todos los bloques del Product Backlog, se redactará el documento **`REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md`** que:

1. Consolida los "Candidatos a RNF" de todas las HUs en un documento único.
2. Numera cada RNF (`RNF-RENDIMIENTO-01`, etc.).
3. Define umbrales aprobados (que pueden ajustarse respecto a los valores tentativos de las HUs).
4. Reemplaza, en cada HU, los umbrales hardcodeados por referencias al RNF correspondiente.

Este trabajo es una **sesión dedicada futura**, no se hace simultáneamente con la redacción de HUs.

### Consecuencias

- Toda HU redactada a partir de Bloque B incluye la sección "Candidatos a RNF".
- HU-01 del Bloque A se actualiza para incluir esta sección (retroactivo).
- Se reconoce que el documento `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` es un entregable pendiente del proyecto.

---

## DHU-008 — Distinción arquitectónica entre componente caído, modo degradado y lógica de fallback

**Fecha:** 2026-05-13.
**Estado:** Cerrada.
**Aplica a:** Bloque C — Operador, operación degradada.

> **Nota agregada el 2026-05-14 (DHU-012):** El estado originalmente llamado "modo seguro" en esta decisión fue renombrado a "degradado nivel 3" para uniformidad del vocabulario de niveles de degradación. El renombrado aplica a esta DHU y a todas las referencias del backlog. Ver DHU-012 sección "Renombrado de 'modo seguro' a 'degradado nivel 3'".

### Contexto

Al cierre del Bloque B, DHU-005 quedó con una promesa abierta: "el Bloque C cubre la alerta activa transversal cuando un componente del sistema se cae". Al abrir el Bloque C para honrar esa promesa, se observó que el backlog detallado del Bloque C usa el término **"operación degradada"** en lugar de "componente caído". Son conceptos cercanos pero no idénticos, y mezclarlos genera HUs ambiguas.

### Análisis

Hay **tres conceptos distintos** que el Bloque C tiene que cubrir, y conviene separarlos explícitamente:

**Concepto 1 — Componente caído (estado binario, hecho técnico):**
Un componente específico del sistema dejó de responder. Ejemplos: el motor adaptativo no responde a solicitudes, el modelo predictivo no genera predicciones, la fuente de mediciones del tráfico no emite, la base de datos no acepta escrituras. Es atribuible a un componente específico y verificable por health check.

**Concepto 2 — Modo degradado (estado del sistema completo, condición operativa):**
El sistema como un todo está operando con capacidades reducidas. Ejemplos: "Sin predicción" (el sistema opera solo con estado observado), "Sin observación" (opera con histórico), "Degradado nivel 3" (aplica tiempos fijos preconfigurados). Es un estado derivado de qué componentes están caídos y qué fallbacks se aplican.

**Concepto 3 — Lógica de fallback en cascada (mecanismo interno):**
Es la regla automatizada que decide qué hacer cuando un componente cae. Define la transición entre el estado normal y un modo degradado específico, o entre un modo degradado y una falla total si no hay fallback aplicable.

### Relación entre los tres conceptos

```
Componente caído + Lógica de fallback aplicable  →  Modo degradado activo
Componente caído + Sin fallback aplicable        →  Falla total
Todos los componentes funcionando                →  Operación normal
```

El Operador necesita distinguir entre los tres estados resultantes:

- **Operación normal:** no hay alerta.
- **Modo degradado activo:** el sistema sigue operando pero con capacidades reducidas; el Operador debe saberlo.
- **Falla total:** el sistema no está operando; el Operador debe escalar.

### Decisión

El Bloque C distingue explícitamente los tres conceptos en HUs y TTH separadas:

| Concepto | Tipo de entrega | Features que cubre |
|---|---|---|
| Componente caído (estado técnico visible al Operador) | HU operativa (vista de estado de componentes) | F23 |
| Modo degradado (alerta activa transversal y explicación) | HUs operativas (alerta + mensaje + indicación contextual) | F22, F24, F25 |
| Lógica de fallback en cascada (mecanismo interno) | TTH (no HU; sin Persona beneficiaria directa) | F26 |
| Configuración del degradado nivel 3 (parámetro del sistema) | TTH (no HU; valor parametrizable interno) | F27 |

### Mapeo concreto

**HUs operativas del Bloque C (4 HUs):**

| HU | Cubre | Features |
|---|---|---|
| HU-10 | Alerta activa transversal cuando el sistema entra en modo degradado o falla total | F22 |
| HU-11 | Vista detallada del estado de cada componente del sistema | F23 |
| HU-12 | Explicación del modo degradado activo y sus implicaciones operativas | F24 |
| HU-13 | Indicación contextual del modo degradado en cada panel afectado | F25 |

**TTH del Bloque C (2 TTH):**

| TTH | Cubre | Features |
|---|---|---|
| TTH-04 | Lógica de fallback en cascada del sistema | F26 |
| TTH-05 | Configuración de tiempos fijos para degradado nivel 3 | F27 |

### Justificación de las TTH

F26 (lógica de fallback) y F27 (configuración del degradado nivel 3) cumplen los criterios de DHU-004 para clasificar como TTH:

- **F26:** mecanismo del backend que opera automáticamente sin participación del Operador. No tiene Persona del producto como beneficiaria directa (DHU-004 criterio 1). Su valor es instrumental (DHU-004 criterio 2). Su comportamiento es técnico estándar (DHU-004 criterio 3).
- **F27:** es un conjunto de parámetros de configuración del sistema, no una funcionalidad operativa. No se entrega valor visible al usuario al completarla en aislamiento (DHU-004 criterio 4).

### Lo que esta decisión deja abierto para la próxima sesión

La decisión arquitectónica está cerrada, pero las HUs concretas del Bloque C todavía deben redactarse. Tres cosas se resuelven al redactar las HUs:

1. **Niveles de severidad de la alerta activa.** ¿"Modo degradado" y "falla total" disparan la misma alerta o tienen estilos visuales distintos? Probablemente distintos.
2. **Persistencia del estado de modo degradado.** ¿Se registra en BD para reporte ejecutivo del Gerente (Bloque F)? Probable que sí, pero la decisión se cierra en el Bloque F.
3. **Capacidad del Operador de "reconocer" la alerta.** ¿Puede silenciarla mientras dura el modo degradado? Decisión de UX a cerrar en la redacción de HU-10.

### Consecuencias

- El Bloque C se redacta con esta estructura de 4 HUs + 2 TTH (refinado a 3 HUs + 2 TTH por DHU-011).
- TTH-04 y TTH-05 se agregan al documento `TAREAS_TECNICAS_HABILITADORAS.md` cuando se redacten formalmente.
- La promesa abierta en DHU-005 ("Bloque C cubre la alerta activa transversal") se cierra mediante HU-10 (ver DHU-009).

---

## DHU-009 — Relación entre marca pasiva (Bloque B) y alerta activa (Bloque C)

**Fecha:** 2026-05-13.
**Estado:** Cerrada.
**Aplica a:** Coordinación entre Bloque B y Bloque C.

### Contexto

DHU-005 cubrió el principio de robustez ante interrupción de fuente con dos casos (A: fuente externa, B: componente interno), pero dejó implícita una pregunta: si tanto el Bloque B como el Bloque C se ocupan del comportamiento ante caídas, ¿cómo se distinguen sus responsabilidades sin duplicarse?

DHU-008 estableció que el Bloque C cubre tres conceptos distintos, uno de ellos siendo la alerta activa transversal. Falta aclarar la relación entre la marca pasiva de cada panel (Bloque B) y la alerta activa transversal (Bloque C).

### Decisión

Las HUs del Bloque B y la HU-10 del Bloque C cumplen funciones complementarias, no duplicadas:

**Bloque B — Marca pasiva en el panel propio:**
- Notifica caída individual de la fuente o componente específico de ese panel.
- Es contextual: solo aparece en el panel afectado.
- Es pasiva: el Operador la descubre al mirar el panel.
- Propósito: ayudar al Operador a interpretar **qué panel específico** está afectado.

**Bloque C (HU-10) — Alerta activa transversal:**
- Notifica el estado del sistema completo (modo degradado o falla total).
- Es transversal: aparece en cualquier vista del sistema, independiente del panel que esté abierto.
- Es activa: busca la atención del Operador, no espera a que mire.
- Propósito: decirle al Operador **qué está haciendo el sistema completo** y si está operando con capacidades reducidas.

### Por qué no es duplicación

Un mismo evento físico (por ejemplo, motor adaptativo caído) puede disparar ambas señales legítimamente:

- **HU-05 del Bloque B** marca el panel de estrategia como "no confirmada" (DHU-005 Caso B). Esto le dice al Operador: "este dato específico ya no podemos garantizarlo".
- **HU-10 del Bloque C** dispara una alerta activa transversal "Degradado nivel 3 activo" cuando el fallback de TTH-04 aplica tiempos fijos. Esto le dice al Operador: "el sistema completo entró en degradado nivel 3; sabelo aunque estés mirando otra pantalla".

Las dos señales transportan **información distinta y útil al mismo tiempo**: la primera explica un panel específico; la segunda explica el estado del sistema completo. Eliminar una rompe una capacidad operativa real.

### Reglas operativas

1. **Las HUs del Bloque B no referencian explícitamente al Bloque C** en sus CAs. La marca pasiva del Bloque B es una responsabilidad autocontenida.

2. **La HU-10 del Bloque C no duplica los detalles** que cada panel del Bloque B marca pasivamente. La HU-10 describe el estado del sistema completo en términos de modo activo (degradado, falla total), no de qué pasa en cada panel.

3. **La HU-11 del Bloque C (vista de estado de componentes) sí muestra detalle por componente.** Es el lugar donde el Operador puede consultar específicamente qué componente está caído, sin tener que recorrer todos los paneles del Bloque B uno por uno.

### Consecuencias

- HU-10 del Bloque C se redacta como "alerta del estado del sistema completo", no como "alerta por cada componente".
- HU-11 del Bloque C se redacta como "vista por componente", complementaria al Bloque B.
- Las HUs del Bloque B se mantienen sin cambios (no necesitan referencia al Bloque C).
- DHU-005 queda cerrada y completada con esta decisión.

---

## DHU-010 — Criterios para clasificar trabajo del Bloque C como TTH

**Fecha:** 2026-05-13.
**Estado:** Cerrada.
**Aplica a:** Bloque C — Operador, operación degradada.

### Contexto

DHU-004 estableció criterios generales para clasificar trabajo como TTH (no HU). Al aplicarlos al Bloque C, dos features (F26 y F27) caen claramente en la categoría TTH. Esta decisión formaliza la aplicación específica al Bloque C, evitando ambigüedad futura.

### Decisión

Las features F26 (Lógica de fallback en cascada del backend) y F27 (Configuración de tiempos fijos para degradado nivel 3) del Bloque C se modelan como Tareas Técnicas Habilitadoras (TTH-04 y TTH-05), no como HUs.

### Aplicación de los criterios de DHU-004

**F26 — Lógica de fallback en cascada del backend:**

| Criterio DHU-004 | F26 cumple |
|---|---|
| 1. No tiene Persona del producto beneficiaria directa | Sí. Es lógica interna del backend que opera automáticamente. |
| 2. Su valor es instrumental, no de negocio | Sí. Habilita los modos degradados pero no genera valor visible al Operador en aislamiento. |
| 3. Comportamiento técnico estándar sin negociación de negocio | Sí. La regla "si X cae, aplicar Y" no requiere conversación con un Persona. |
| 4. Sin valor visible al usuario en aislamiento | Sí. El Operador nunca interactúa con la lógica de fallback; interactúa con sus resultados (modo degradado). |

**F27 — Configuración de tiempos fijos para degradado nivel 3:**

| Criterio DHU-004 | F27 cumple |
|---|---|
| 1. No tiene Persona del producto beneficiaria directa | Sí. Es un conjunto de parámetros del sistema. |
| 2. Su valor es instrumental, no de negocio | Sí. Solo se usa cuando se activa el degradado nivel 3. |
| 3. Comportamiento técnico estándar | Sí. Configuración de valores numéricos, no negociable funcionalmente. |
| 4. Sin valor visible al usuario en aislamiento | Sí. El Operador no usa F27 directamente; usa los efectos del degradado nivel 3 cuando se activa. |

### Lo que NO es TTH en el Bloque C

Para evitar confusión, lo siguiente del Bloque C NO se clasifica como TTH:

- **F22, F23, F24, F25** son funcionalidades visibles al Operador → son HUs (HU-10, HU-11, HU-12, HU-13).
- **El comportamiento esperado del sistema cuando entra en modo degradado** desde la perspectiva del Operador → es HU (HU-10).
- **La explicación del modo degradado** desde la perspectiva del Operador → es HU (HU-12).

### Consecuencias

- TTH-04 (Lógica de fallback en cascada del sistema) se agrega a `TAREAS_TECNICAS_HABILITADORAS.md` cuando se redacte formalmente.
- TTH-05 (Configuración de tiempos fijos para degradado nivel 3) se agrega a `TAREAS_TECNICAS_HABILITADORAS.md` cuando se redacte formalmente.
- El Bloque C queda con 4 HUs + 2 TTH como composición final (refinado a 3 HUs + 2 TTH por DHU-011).

---

## DHU-011 — Eliminación de HU-13 y cobertura de F25 por composición

**Fecha:** 2026-05-13.
**Estado:** Cerrada.
**Aplica a:** Bloque C — Operador, operación degradada.

### Contexto

DHU-008 estableció que el Bloque C cubriría tres conceptos distintos (componente caído, modo degradado, lógica de fallback) con un mapeo definitivo de 4 HUs operativas (HU-10 a HU-13) + 2 TTH (TTH-04, TTH-05). HU-13 estaba prevista para cubrir F25 (Indicación contextual en panel de modo degradado activo).

Durante la redacción detallada del Bloque C, al diferenciar HU-13 de las marcas pasivas del Bloque B (DHU-005 Casos A y B) se identificó que HU-13 **no aporta valor incremental real al Operador** dado los fallbacks declarados en F26 y la cobertura existente del Bloque B.

### Análisis

Los fallbacks declarados en F26 producen tres efectos distintos sobre los paneles del Operador:

| Nivel de fallback | Efecto sobre paneles del Operador | Cobertura existente |
|---|---|---|
| Nivel 1 (motor sin métricas de visión) | Los paneles que dependen de visión muestran datos viejos. | Marca pasiva DHU-005 Caso A en panel de visión. |
| Nivel 2 (predictor de respaldo activo) | Panel de predicción muestra datos **vigentes** pero de menor precisión. | Sin cobertura específica; HU-13 hubiese cubierto este caso. |
| Nivel 3 (tiempos preconfigurados) | Panel de estrategia activa congelado (no hay decisión nueva). | Marca pasiva DHU-005 Caso B en panel de estrategia. |

HU-13 con alcance amplio tendría valor real únicamente en el nivel 2 (predictor de respaldo activo). Para los niveles 1 y 3, las marcas pasivas del Bloque B ya cubren la información necesaria al Operador sin agregar nada que el Operador no sepa ya.

Mantener HU-13 con alcance acotado al nivel 2 sería redactar una HU completa para una única manifestación visual. Mantenerla con alcance amplio anticipando fallbacks futuros sería redactar una HU que en MVP1 sólo activa una etiqueta en un caso. Las dos opciones tienen baja relación valor/esfuerzo.

### Análisis adicional del Operador

Desde la perspectiva del Operador, la información que F25 buscaba comunicar ya está disponible por composición:

1. **Que el sistema está en modo degradado** → comunicado por la alerta transversal de HU-10.
2. **Qué componente específico falló** → consultable en la vista de HU-11.
3. **Qué significa operativamente el modo activo** → explicado por el texto compuesto de HU-12.
4. **Que un panel específico está afectado** → cubierto por la marca pasiva del Bloque B cuando el dato es viejo (Caso A o B según corresponda), y por la disponibilidad de las marcas en el panel afectado.

El único hueco residual es comunicar, en el panel específico de predicción del nivel 2, que el dato vigente proviene del predictor de respaldo. Este hueco es lo suficientemente acotado como para no justificar una HU dedicada en el alcance del MVP1.

### Decisión

Se elimina HU-13 del backlog del Bloque C. F25 queda cubierta funcionalmente por la composición de:

- HU-10 (alerta transversal del estado operativo).
- HU-11 (vista de estado de componentes, con refinamiento de resalte visual aprobado en esta decisión).
- HU-12 (explicación del modo degradado).
- Las marcas pasivas existentes del Bloque B (DHU-005 Casos A y B).

Refinamiento asociado a HU-11: se agrega un criterio de aceptación (CA-11.9) que declara explícitamente el resalte visual de las entradas de componentes en estado no-OK dentro de la vista de HU-11, para que el Operador pueda identificar de un vistazo qué componentes requieren atención. Este refinamiento absorbe el espíritu de F25 en la vista que ya tiene esa responsabilidad natural (HU-11), en lugar de crear una HU dedicada.

### Por qué este patrón es coherente con el resto del backlog

La cobertura por composición no es nueva en este Product Backlog. Ya se aplicó en:

- **F02 (Dashboard principal)** del Bloque B: cubierto por la composición visual de HU-02, HU-03, HU-04, HU-05 y HU-06, sin generar HU propia. Documentado en el cierre del Bloque B.
- **F30 (Persistencia de estados históricos)** del Bloque A: cubierto por inglobación como CA en HUs del Gerente (Bloque F). Documentado en el cierre del Bloque A.
- **F31 (Persistencia de decisiones del motor)** del Bloque A: inglobada como CA-08.1 de HU-08. Documentado en el cierre del Bloque A.

DHU-011 aplica el mismo principio a F25: cubrir una feature por composición de otras HUs cuando no se justifica una HU dedicada.

### Consecuencias

- HU-13 deja de existir en el Bloque C.
- HU-11 se refina con CA-11.9 (resalte visual de componentes en estado no-OK), una nota técnica que documenta la decisión, y un RNF de usabilidad ampliado.
- El Bloque C queda con composición final: **3 HUs operativas (HU-10, HU-11 refinada, HU-12) + 2 TTH (TTH-04, TTH-05)**.
- Esta decisión actualiza el mapeo de DHU-008, que originalmente preveía 4 HUs.
- F25 se documenta como "cubierta por composición" en la sección de mapeo del Bloque C.

### Lo que NO cambia

- DHU-008 sigue siendo válida en su separación conceptual de los tres conceptos (componente caído, modo degradado, lógica de fallback). DHU-011 solo refina el mapeo a HUs concretas.
- DHU-009 sigue siendo válida en su separación entre marca pasiva del Bloque B y alerta activa del Bloque C.
- DHU-010 sigue siendo válida en la clasificación de F26 y F27 como TTH.

### Documentos relacionados

- `HU_BLOQUE_C.md` — refleja la composición final 3 HUs + 2 TTH.
- `DECISIONS_HU.md` (este documento) — sección DHU-011.

---

## DHU-012 — Auditoría de coherencia documental: semántica de MVP, eliminación de MVP3, corrección de conteos, alineación de vocabulario y limpieza de residuo pre-Inception

**Fecha:** 2026-05-14.
**Estado:** Cerrada.
**Aplica a:** Todo el Product Backlog y los documentos relacionados.

### Contexto

Durante la preparación de la sesión de redacción del Bloque D se detectó un conjunto extenso de inconsistencias documentales entre los archivos del proyecto. Una auditoría sistemática identificó **10 inconsistencias específicas** (INC-01 a INC-10) cuyo origen es:

1. **Documentos generados en sesiones distintas** que evolucionaron sin sincronización (Inception del 11-mayo, fichas del backlog, cierres de Bloques A/B/C posteriores).
2. **Decisiones cerradas en conversación** que no quedaron documentadas formalmente (por ejemplo, omisión silenciosa de F18 en próximos pasos de Bloques B y C).
3. **Residuo del régimen previo al Inception** que no se limpió tras el cambio de marco metodológico (referencias a `PLAN.md`, "TODO", "Bloques K/J/F" del plan obsoleto, IDs "HU-16" y "HU-17" de hitos pre-Inception).
4. **Errores aritméticos** en conteos del Sequencer original (26 declarado vs 29 real al sumar bloques).

Esta decisión consolida la resolución de las 10 inconsistencias en un acto único de coherencia documental.

### Decisiones consolidadas

DHU-012 cierra los siguientes refinamientos, agrupados temáticamente:

#### A. Semántica de MVP refinada (INC-01)

La política previa establecía que "MVP2 se documenta como HU pero no se construye" (decisión cerrada #10 del Inception). La política nueva refina esto:

| Categoría | Política nueva |
|---|---|
| MVP1 | Se redacta como HU. Se construye. Es entregable comprometido del proyecto académico. |
| MVP2 | Se redacta como HU. Se construye **condicional a holgura** tras cerrar las MVP1. Es entregable redactado, no necesariamente entregable construido. |
| MVP3 | **Categoría eliminada.** Ver subsección B. |

**Consecuencia operativa para HUs MVP2 ya redactadas:** la nota técnica de HU-09 (única HU MVP2 hasta hoy) se suaviza para reflejar la nueva semántica (de "esta HU NO se implementa en MVP1" a "esta HU es candidata; su construcción se considera si el cronograma permite holgura").

#### B. Eliminación de la categoría MVP3 (INC-04)

La categoría "MVP3" del Sequencer original se renombra a **"Trabajos Futuros"** y se reformula su semántica:

- **Antes (MVP3):** "no se documenta como HU, solo se menciona como trabajo futuro" (Inception línea 263, redacción original).
- **Después (Trabajos Futuros):** las direcciones de trabajo futuro **se documentan como fichas de feature** en el backlog detallado (con ficha liviana), **NO se redactan como HU**, **NO se construyen**, **se mencionan en el capítulo de trabajo futuro del documento de tesis**.

**Razón del renombrado:** el término "MVP3" sugiere semánticamente "tercera iteración del MVP que eventualmente se construye". El término "Trabajos Futuros" refleja con precisión la naturaleza real de estas direcciones (líneas declaradas fuera del alcance académico, candidatas a futuras extensiones del producto o de la investigación).

**Composición resultante:** 7 direcciones de Trabajos Futuros, todas con ficha en el backlog detallado:

| ID | Título | Origen documental |
|---|---|---|
| F21 | Reentrenamiento del modelo predictivo (pipeline MLOps) | Brainstorming original del Inception (Artefacto 7) |
| F36 | Reconocimiento de tipos de vehículos para priorización | Sequencer del Inception MVP3 (original) |
| F37 | Coordinación de ondas verdes entre intersecciones vecinas | Sequencer del Inception MVP3 (cita D-006) |
| F38 | Procesamiento de datos reales de Waze | Sequencer del Inception MVP3 (cita D-008) |
| F39 | Despliegue real en Raspberry Pi como dispositivo de borde | Sequencer del Inception MVP3 (cita D-004) |
| F40 | Notificaciones push y monitoreo proactivo de cámaras | Sequencer del Inception MVP3 |
| F41 | Integración cerrada del módulo de visión al loop de validación cuantitativa | `EVOLUCION_TESIS.md` sección 8 (cita D-007) |

**Asimetría justificada:** F21 conserva ficha completa (entró al Brainstorming original con detalle); F36 a F41 son fichas livianas (estructura reducida sin "Revisión UX" ni "Estado actual en el repo"). Esta asimetría preserva el contenido histórico de F21 sin inflar artificialmente las 6 nuevas con detalle inventado.

**Reclasificación específica de F21:** la ficha de F21 estaba clasificada como "MVP2 — fuera del sprint" en el backlog detallado bajo la política previa. Con la política nueva, F21 no cabe en MVP2 (no es razonable construirla "si hay holgura": complejidad Alta y la propia ficha declaraba "no cabe en el cronograma"). Pasa a "Trabajos Futuros", consistente con su clasificación original en el Sequencer del Inception (MVP3).

#### C. Corrección de conteos de features (INC-03)

El Sequencer original del Inception declaraba "26 features MVP1" en su título (línea 233), pero la suma de los Bloques A-F enumerados era 4+9+6+3+4+3 = **29**. Es un error aritmético histórico del Inception.

**Conteo correcto consolidado tras DHU-012:**

| Categoría | Conteo | Detalle |
|---|---|---|
| MVP1 (con ficha) | **29** | 17 críticas ★ + 12 importantes ◆ |
| MVP2 (con ficha) | **5** | F11, F15, F16, F19, F28 |
| Trabajos Futuros (con ficha) | **7** | F21, F36, F37, F38, F39, F40, F41 |
| **Total features con ficha** | **41** | F01-F20 + F22-F41 (numeración con hueco en F21 inexistente: F21 sí está) |

**Nota sobre el conteo total:** el Brainstorming original tuvo 35 features (F01-F35). DHU-012 agrega 6 fichas livianas (F36-F41) para formalizar las direcciones de Trabajos Futuros que originalmente vivían como prosa en el Sequencer y en `EVOLUCION_TESIS.md`. Total post-DHU-012: **41 features con ficha**.

#### D. Composición del Bloque D y corrección de transcripción (INC-02)

Los cierres de los Bloques B y C declaraban en sus "Próximos pasos" que el Bloque D contendría las features **(F17, F20, F21)**, omitiendo F18. Esto era un error de transcripción no documentado como decisión formal. El Sequencer del Inception (artefacto formal aprobado) y la ficha de F18 declaran F18 = MVP1 — Bloque D.

**Decisión formalizada:** El Bloque D MVP1 contiene **F17 + F18 + F20**. F21 no es MVP1 (es Trabajos Futuros tras esta decisión). La omisión de F18 en los cierres B y C se corrige sin debate adicional.

#### E. Alineación de vocabulario de niveles de fallback (INC-06)

El modelo arquitectónico de degradación evolucionó entre el Inception (3 niveles, vocabulario técnico) y el cierre del Bloque C (3 niveles + falla total, vocabulario agnóstico por DHU-006). Los documentos previos al cierre del Bloque C quedaron con el vocabulario viejo.

**Acciones consolidadas:**

1. **Journey 4 del Inception** reescrito con 4 estados (Degradado nivel 1, 2, 3 + Falla total) y vocabulario agnóstico. Nota explicativa cita DHU-006 y DHU-008.
2. **Ficha F26 del backlog** reescrita eliminando la duplicación del listado de niveles. Referencia a TTH-04 (CT-04.2) como fuente canónica del modelo de fallback.

#### F. Renombrado de "modo seguro" a "degradado nivel 3" (INC-06)

El estado originalmente denominado "modo seguro" se renombra uniformemente a **"degradado nivel 3"** para cohesionar el vocabulario de niveles. El identificador interno técnico `safe_3` se renombra a `degraded_3`.

**Estructura final de estados operativos del sistema:**

| Estado | Identificador interno | Color visual | Descripción |
|---|---|---|---|
| Operación normal | `normal` | (sin banner) | Todos los componentes operativos |
| Degradado nivel 1 | `degraded_1` | Amarillo | Componente periférico de detección de tráfico no responde |
| Degradado nivel 2 | `degraded_2` | Naranja | Componente predictivo principal no responde; predictor de respaldo activo |
| Degradado nivel 3 | `degraded_3` | Rojo | Motor adaptativo no responde; tiempos preconfigurados aplicados |
| Falla total | `total_failure` | Rojo intenso | Sin fallback aplicable; sistema no aplica decisiones nuevas |

**Acciones:** la palabra "modo seguro" se reemplaza en `DECISIONS_HU.md` (DHU-008, DHU-010), `FEATURE_BACKLOG_DETALLADO.md` (ficha F27, ficha F26), `HU_BLOQUE_C.md` (HU-10, HU-11, HU-12, CAs y notas), y `TAREAS_TECNICAS_HABILITADORAS.md` (título de TTH-05, descripción de TTH-04, identificador `safe_3`).

**Vocabulario funcional que se mantiene:** expresiones como "valores por defecto seguros", "tiempos conservadores", "configuración inicial segura" se mantienen intactas: son vocabulario funcional, no nombres del estado.

#### G. Limpieza de residuo del régimen previo al Inception (INC-07, INC-08)

`DECISIONS.md` (decisiones D-001 a D-008, fechadas antes o durante el Inception) y otros documentos contenían referencias a artefactos del régimen de planificación previo al Inception, que ya no existen en el régimen vigente. Estos residuos generan confusión y rompen la trazabilidad.

**Referencias eliminadas:**

- `PLAN.md` y sus "Fase 1/2/3/N" (numeración del plan obsoleto).
- "Bloque J/K/F del TODO" o del PLAN. NO confundir con "Bloques A-F" del Sequencer del Inception, que son legítimos y se preservan.
- `TODO.md` y "F3 del TODO", "Bloque F del TODO", etc.
- "Llamada A2 del TODO" → reformulada como "Pendiente: confirmar con asesor" sin ID.
- "HU-16" y "HU-17" (numeración de hitos pre-Inception) → reescritas refiriéndose a las features del backlog actual.
- `tesis/(2).docx` → eliminada (era copia temporal del documento de tesis; el documento final no se ha cerrado).

**Referencias actualizadas:**

- "D-001 a D-008" → "D-001 a D-009" en los listados de "Documentos relacionados" de `LEAN_INCEPTION_CEREBROVIAL.md` y `FEATURE_BACKLOG_DETALLADO.md`.

**Preservación:**

- Contenido sustantivo de todas las decisiones D-001 a D-009 se mantiene intacto. Solo se limpia la prosa que referenciaba artefactos obsoletos.
- Fechas históricas de cada decisión se preservan.
- "Fase 1/2/3/4" en `EVOLUCION_TESIS.md` se mantiene cuando describe fases conceptuales de la evolución del proyecto (narrativa), no del PLAN obsoleto.

#### H. Higiene de documentos de fundamentación (INC-09, INC-10)

- **`LEAN_INCEPTION_INVESTIGACION.md`** sección 9 ("Próximo paso inmediato") eliminada por obsoleta. Documento sube a versión 1.1.
- **`LEAN_INCEPTION_CEREBROVIAL.md`** pasa a versión 1.1 (de "1.0, lista para Showcase"). Cabecera distingue "fecha del workshop original" (2026-05-11) de "fecha de última actualización" (2026-05-14). Se agrega nota al pie con los cambios desde v1.0, citando esta decisión DHU-012.
- **`EVOLUCION_TESIS.md`** sección 8 reescrita como tabla referencial que apunta a las fichas de Trabajos Futuros en el backlog detallado, en lugar de prosa larga.

### Documentos afectados por DHU-012

Esta decisión genera modificaciones en los siguientes 9 documentos del proyecto:

| Documento | Tipo de cambio |
|---|---|
| `DECISIONS_HU.md` (este documento) | Agregar DHU-012 (esta decisión) y DHU-013 (clasificación HU/TTH del Bloque D); nota en DHU-008; renombrado de vocabulario en DHU-008, DHU-009 y DHU-010. |
| `DECISIONS.md` | Limpieza completa de residuo del régimen pre-Inception; contenido sustantivo preservado. |
| `LEAN_INCEPTION_CEREBROVIAL.md` | Renombrado MVP3 → Trabajos Futuros; conteo MVP1 = 29; Journey 4 reescrito; limpieza de residuo PLAN; versión 1.1 con nota de cambios. Nota agregada en pase de higiene cruzada: aclaración sobre la convención del conteo del Bloque B (9 MVP1 vs 10 con F11 MVP2). |
| `LEAN_INCEPTION_INVESTIGACION.md` | Sección 9 eliminada; versión 1.1. Nota agregada en pase de higiene cruzada: renombrado de "MVP3 (trabajo futuro)" → "Trabajos Futuros" en la fila 9 de la tabla del plan de ejecución de la sección 5 (la referencia residual al término MVP3 en la versión adaptada a CerebroVial). Las referencias a "MVP3" en las secciones 3 y 4.3 del documento describen el método genérico de Caroli y se preservan intactas. |
| `EVOLUCION_TESIS.md` | Sección 8 reescrita como tabla referencial; limpieza de residuo PLAN. |
| `FEATURE_BACKLOG_DETALLADO.md` | Agregar fichas F36-F41; reclasificar F21 como Trabajos Futuros; recalcular tablas; renombrado vocabulario; limpieza de residuo PLAN; referencia ficha F26 a TTH-04; conteo D-001 a D-009. |
| `HU_BLOQUE_A.md` | **Agregado retroactivamente el 2026-05-14 (pase de higiene cruzada con los demás documentos del backlog).** Cambios v3 → v4: (a) rango de DHU referenciado en "Documentos relacionados" actualizado de DHU-001 a DHU-007 → DHU-001 a DHU-013; (b) corrección del residuo de copy-paste en "Próximos pasos" (la frase "Esta sesión cerró el Bloque B" se reemplaza por una redacción coherente con que el documento es del Bloque A, y se actualiza el listado de bloques pendientes para reflejar que Bloque B y Bloque C también están cerrados); (c) corrección de typo "inglogan" → "ingloban" en la sección "Persistencias movidas a otros bloques"; (d) referencia a `HU_BLOQUE_C.md` agregada en "Documentos relacionados"; (e) ampliación de "TTH-01, TTH-02, TTH-03 transversales" para incluir TTH-04 y TTH-05 del Bloque C. El contenido sustantivo de HU-01 y de las reglas metodológicas se mantiene intacto. |
| `HU_BLOQUE_B.md` | Próximos pasos actualizados; nota técnica de HU-09 suavizada. Nota agregada en pase de higiene cruzada: corrección del conteo total de features en "Documentos relacionados" (35 → 41); aclaración sobre la convención de conteo del Bloque B (10 features con F11 MVP2 por afinidad temática vs 9 features MVP1 del Sequencer); referencia a `HU_BLOQUE_C.md` agregada en "Documentos relacionados"; ampliación de "TTH-01, TTH-02, TTH-03" para incluir TTH-04 y TTH-05. |
| `HU_BLOQUE_C.md` | Próximos pasos del Bloque D actualizados; renombrado de vocabulario en HU-10, HU-11, HU-12, CAs y notas. |
| `TAREAS_TECNICAS_HABILITADORAS.md` | Título de TTH-05 actualizado; renombrado de vocabulario en TTH-04 y TTH-05; identificador `safe_3` → `degraded_3`; nota de línea 262 cerrada por DHU-013; limpieza de residuo PLAN. |

**Nota sobre el alcance temporal de DHU-012:** la decisión se cerró el 2026-05-14. El pase original cubrió 8 documentos. Una revisión cruzada posterior detectó tres residuos no resueltos en `HU_BLOQUE_A.md`, `HU_BLOQUE_B.md` y `LEAN_INCEPTION_INVESTIGACION.md` que se resolvieron en el mismo día como extensión natural del alcance de DHU-012 (no se abrió DHU separada porque los cambios son estrictamente de higiene documental y no involucran decisiones nuevas).

### Lo que NO cambia con DHU-012

- **Las decisiones DHU-001 a DHU-011 mantienen su contenido sustantivo.** Solo DHU-008 recibe una nota corta al inicio sobre el renombrado de vocabulario.
- **El alcance del producto (Personas, Objetivos, Journeys, Visión)** se mantiene intacto.
- **El contenido de las HUs redactadas (HU-01 a HU-12)** no se reabre; solo se ajustan referencias de vocabulario y notas de próximos pasos.
- **Las TTH redactadas (TTH-01 a TTH-05)** se mantienen en su contenido sustantivo; solo TTH-05 ajusta título y TTH-04/TTH-05 ajustan vocabulario.

### Trazabilidad de las 10 inconsistencias

Para referencia futura, las 10 inconsistencias del inventario auditado se resolvieron de la siguiente manera:

| Inconsistencia | Cobertura en DHU-012 |
|---|---|
| INC-01 — Clasificación de F21 | Subsección A (semántica MVP) + B (F21 = Trabajos Futuros) |
| INC-02 — F18 en el Bloque D | Subsección D |
| INC-03 — Conteo MVP1 26 vs 29 | Subsección C |
| INC-04 — Composición MVP3 | Subsección B |
| INC-05 — Composición del Bloque D | DHU-013 (decisión propia, ver más abajo) |
| INC-06 — Niveles de fallback | Subsecciones E + F |
| INC-07 — Residuo pre-Inception | Subsección G |
| INC-08 — Conteos D-001 a D-008 | Subsección G |
| INC-09 — Sección 9 obsoleta de INVESTIGACION | Subsección H |
| INC-10 — Versión y fecha del Inception | Subsección H |

INC-05 (composición del Bloque D y clasificación HU/TTH de F17, F18, F20) se documenta como **DHU-013** independiente porque pertenece al ciclo de redacción del Bloque D (paralela a DHU-008 + DHU-010 que cerraron lo mismo para el Bloque C).

---

## DHU-013 — Clasificación HU/TTH de las features del Bloque D

**Fecha:** 2026-05-14.
**Estado:** Cerrada.
**Aplica a:** Bloque D — Administrador, soporte técnico.

### Contexto

DHU-008 y DHU-010 cerraron la clasificación HU/TTH para el Bloque C. Esta decisión hace lo equivalente para el Bloque D, aplicando los criterios de DHU-004 a las tres features que componen el bloque tras el cierre de DHU-012 (INC-02 e INC-05).

**Composición del Bloque D MVP1 (cerrada por DHU-012 subsección D):** F17 (Panel de salud de componentes), F18 (Panel de métricas del modelo), F20 (Configuración del motor adaptativo).

### Análisis feature por feature

**F17 — Panel de salud de componentes del sistema (Administrador)**

| Criterio DHU-004 | F17 cumple |
|---|---|
| 1. Sin Persona beneficiaria directa | No. Administrador es Persona beneficiaria directa, paso 2 del Journey 3. |
| 2. Valor instrumental, no de negocio | No. Valor operativo claro: el Administrador necesita confirmar salud técnica del sistema. |
| 3. Comportamiento técnico estándar sin negociación | No. Qué métricas exponer, cómo presentar latencias/errores/logs, qué nivel de detalle, son decisiones de UX negociables con el Administrador. |
| 4. Sin valor visible al usuario en aislamiento | No. El Administrador la usa directamente. |

**Diagnóstico:** F17 cumple 0 de 4 criterios TTH. Es **HU operativa del Administrador**.

**Sustrato técnico:** ya existe vía CT-04.5 de TTH-04 (endpoint `GET /system/components/status`). No se crea TTH adicional. La HU del Administrador consume el mismo endpoint que HU-11 del Operador, con presentación distinta (técnica vs simplificada).

**F18 — Panel de métricas del modelo predictivo (Administrador)**

| Criterio DHU-004 | F18 cumple |
|---|---|
| 1. Sin Persona beneficiaria directa | No. Administrador es Persona beneficiaria directa, paso 3 del Journey 3, y necesidad declarada de la Persona ("Consultar métricas de desempeño del modelo predictivo"). |
| 2. Valor instrumental, no de negocio | No. Realiza el Objetivo 2 del producto desde la perspectiva del Administrador. |
| 3. Comportamiento técnico estándar | No. Qué métricas mostrar, qué ventana temporal, cómo visualizar evolución, son decisiones negociables. |
| 4. Sin valor visible al usuario en aislamiento | No. El Administrador la usa directamente. |

**Diagnóstico:** F18 cumple 0 de 4 criterios TTH. Es **HU operativa del Administrador**.

**Sustrato técnico:** el cálculo de métricas requiere (a) registrar las predicciones del modelo en el momento que se generan, (b) compararlas con observaciones reales una vez que el horizonte de predicción llega, (c) agregar métricas (MAE, RMSE) sobre ventana temporal configurable. Este sustrato se **ingloba como CAs dentro de la propia HU de F18**, siguiendo el patrón establecido para persistencias (F31 inglobada en HU-08 CA-08.1). No se crea TTH adicional.

**Justificación de la inglobación (no TTH separada):** TTH-04 y TTH-05 fueron justificadas como TTH porque ambas son lógica autónoma del sistema **consumida por múltiples HUs** (TTH-04 por HU-10, HU-11 y HU-12; TTH-05 por TTH-04). El sustrato de F18 es **consumido únicamente por la HU de F18 y por nadie más**. Esa diferencia favorece la inglobación, no la separación.

**F20 — Configuración de parámetros del motor adaptativo (Administrador)**

| Criterio DHU-004 | F20 cumple |
|---|---|
| 1. Sin Persona beneficiaria directa | No. Administrador es Persona beneficiaria directa, paso 5 del Journey 3. |
| 2. Valor instrumental, no de negocio | Discutible. Valor inmediato es "configurar sin redeploy"; valor de negocio último es calibración del sistema. |
| 3. Comportamiento técnico estándar | No. Qué parámetros exponer, qué rangos válidos, cómo validar, son decisiones negociables. |
| 4. Sin valor visible al usuario en aislamiento | No. El Administrador la usa directamente. |

**Diagnóstico:** F20 cumple 0 de 4 criterios TTH (criterio 2 discutible pero no decisivo). Es **HU operativa del Administrador**.

**Granularidad:** F20 se modela como **una sola HU** (no múltiples HUs por familia de parámetros). El Administrador trabaja con todos los parámetros en un mismo flujo (entra a la pantalla, ajusta, guarda); la organización por familias de parámetros se hace dentro de la HU con CAs estructurados. Patrón ya establecido: "una feature = una HU" salvo casos manifiestamente compuestos (como F02 cubierto por composición). F20 no es compuesta.

**Alcance:** la ficha original de F20 enumera "umbrales de cola, pesos entre estrategias, parámetros internos de Webster/MaxPressure/MTC". En la práctica, F20 cubre **más parámetros** según referencias cruzadas del backlog ya cerrado:

- Umbrales de cola verde/amarillo/rojo (CA-02.3 y nota de HU-02).
- Horizonte de predicción configurable (CA-03.1 y nota de HU-03).
- Umbral de congestión (CA-03.3 y nota de HU-03, default ≥ 3, atado a D-009 línea 244).

Estos parámetros son los que la HU debe cubrir en MVP1. Los parámetros internos de las estrategias (pesos entre Webster/MaxPressure/MTC) se evalúan caso a caso al redactar y pueden quedar fuera de MVP1 si exceden el alcance "3-5 parámetros críticos" sugerido por la propia ficha.

**Por DHU-006 (agnosticismo):** la HU de F20 NO menciona Webster/MaxPressure/MTC. Usa lenguaje funcional ("parámetros de las estrategias de control del motor" o equivalente).

### Reconsideración de TTH-05 a la luz de F20

La nota técnica de TTH-05 (al final de su entrada en `TAREAS_TECNICAS_HABILITADORAS.md`) dejaba abierta una pregunta:

> *"Si durante la redacción del Bloque D se considera que la configuración de degradado nivel 3 merece su propia HU del Administrador, esta TTH puede dividirse en (a) parte instrumental que sigue siendo TTH y (b) HU del Administrador para el formulario y la auditoría."*

DHU-013 cierra esta pregunta a favor de **mantener TTH-05 íntegra**, sin dividir. Razones:

1. **Cohesión de la HU de F20.** "Configuración del motor adaptativo" es un concepto coherente. Mezclarlo con "tiempos del degradado nivel 3" crea una HU heterogénea sin foco.

2. **Separación arquitectónica respetada.** TTH-04 maneja la lógica de fallback. TTH-05 provee los tiempos preconfigurados que TTH-04 consume en nivel 3. Ambas son lógica de backend; el formulario que las configura es incidental a su naturaleza.

3. **DHU-010 ya cerró F27 como TTH** con justificación específica. Reabrir esa decisión sin necesidad operativa concreta es agregar trabajo sin beneficio.

4. **La nota de TTH-05 dejó la puerta abierta, no obligaba a dividir.** Su redacción es *"puede dividirse"*, no *"se debe dividir"*. Esa puerta se cierra ahora con decisión expresa.

5. **Extensibilidad futura preservada.** Si en el futuro surge necesidad concreta (por ejemplo, el Administrador quiere ver auditoría de cambios de TTH-05 en una vista dedicada), siempre se puede extraer una HU adicional sin perder nada.

### Decisión

**Bloque D MVP1: 3 HUs operativas del Administrador + 0 TTH nuevas.**

| Feature | Modelado como | Sustrato técnico |
|---|---|---|
| F17 — Panel de salud de componentes | HU del Administrador | CT-04.5 de TTH-04 (existente, no se crea TTH adicional) |
| F18 — Panel de métricas del modelo | HU del Administrador | Sustrato (registro + cálculo de métricas) inglobado como CAs en la propia HU |
| F20 — Configuración del motor | HU del Administrador (única, agnóstica) | Sustrato (persistencia + auditoría) inglobado como CAs en la propia HU |

**TTH-05** se mantiene íntegra. La nota técnica de TTH-05 (su sección "Posible reconsideración futura") se actualiza para registrar el cierre de la pregunta.

### Consecuencias

- El Bloque D MVP1 se redacta con esta estructura: 3 HUs operativas. Numeración tentativa HU-13 (F17), HU-14 (F18), HU-15 (F20), sujeta a decisión menor al redactar (compactación vs numeración con hueco).
- F19 (Comparativa de métricas del modelo vs baseline) es MVP2 del Administrador, redactada en sesión MVP2 dedicada futura (no en el Bloque D).
- F21 (Reentrenamiento del modelo) es Trabajos Futuros (DHU-012), no se redacta como HU.
- TTH-05 mantiene título actualizado por DHU-012 ("Configuración de tiempos preconfigurados para degradado nivel 3"). Su nota técnica registra el cierre de la pregunta abierta sobre división.

### Documentos relacionados

- `HU_BLOQUE_D.md` — pendiente de redacción tras esta decisión.
- `TAREAS_TECNICAS_HABILITADORAS.md` — nota técnica de TTH-05 actualizada por DHU-013.
- `DECISIONS_HU.md` (este documento) — sección DHU-013.

---

## DHU-014 — Decisiones de redacción del Bloque D (numeración, dashboard, parámetros, métricas, concurrencia, ventana temporal, TTH-06)

**Fecha:** 2026-05-14.
**Estado:** Cerrada.
**Aplica a:** Bloque D — Administrador, soporte técnico.

### Contexto

DHU-013 cerró la clasificación HU/TTH de las features del Bloque D (F17, F18, F20 son HUs operativas; sin TTH nuevas) y la decisión sobre TTH-05 (no se divide). Quedaron pendientes varias decisiones menores de redacción que debían cerrarse antes o durante la redacción de las HUs concretas del bloque. DHU-014 consolida esas decisiones en un acto único, evitando la dispersión que hubiera resultado de cerrarlas por separado.

### Decisiones consolidadas

#### A. Numeración del Bloque D

La HU-13 original del Bloque C fue eliminada por DHU-011 antes de ser redactada formalmente. El número HU-13 no quedó "ocupado" en ningún documento vigente del backlog.

**Decisión:** el Bloque D reutiliza el número HU-13 para F17 (compactación de la numeración del Product Backlog), con HU-14 = F18 y HU-15 = F20. La traza histórica de la HU-13 eliminada vive en DHU-011, no en la numeración del backlog. Dejar un hueco en la numeración para preservar memoria es contaminar el backlog con metadata que pertenece a otro documento.

**Numeración final del Bloque D:**

| HU | Feature origen |
|---|---|
| HU-13 | F17 — Panel de salud de componentes del sistema (vista del Administrador) |
| HU-14 | F18 — Panel de métricas del modelo predictivo |
| HU-15 | F20 — Configuración de parámetros operativos del sistema |

#### B. Sin HU dedicada de dashboard del Administrador

**Decisión:** el Bloque D **no** introduce una HU equivalente a F02 del Bloque B (dashboard principal del Operador). Las tres HUs del Administrador (HU-13, HU-14, HU-15) se acceden desde la navegación del Administrador como tres vistas separadas, sin componerlas visualmente en un dashboard integrador.

**Justificación:** el Operador trabaja en tiempo real sobre un único objeto (la intersección), lo cual justifica un dashboard que muestre distintas caras del mismo objeto simultáneamente (F02 cubierto por composición de HU-02 a HU-06). El Administrador trabaja sobre objetos distintos en momentos distintos (consulta de componentes, análisis de métricas, ajuste de configuración); integrarlos visualmente no aporta valor cognitivo y agregaría una HU sin propósito claro.

Esta decisión queda registrada como decisión documental del Bloque D (DHU-014) y como nota técnica de cada HU del Bloque D. No requiere DHU separada por ser de UX/IA y no introducir regla metodológica nueva.

#### C. Selección concreta de parámetros de F20 en MVP1

**Decisión:** HU-15 (F20) cubre en MVP1 los siguientes parámetros, organizados en tres familias funcionales:

| Familia | Parámetros | Referencias cruzadas |
|---|---|---|
| Visualización del estado del tráfico | Umbrales de cola verde/amarillo/rojo | CA-02.3 de HU-02 |
| Predicción y evaluación del modelo | Horizonte de predicción | CA-03.1 de HU-03 |
| Predicción y evaluación del modelo | Umbral de congestión (default ≥ 3) | CA-03.3 de HU-03 + D-009 |
| Predicción y evaluación del modelo | Ventana temporal de cálculo de métricas | CA-14.4 de HU-14 (ver subsección F) |
| Monitor de salud del sistema | Frecuencia de evaluación de salud de componentes | CT-04.1 de TTH-04 |

**Fuera de MVP1 (parámetros internos del motor):** los parámetros internos de las estrategias de control del motor adaptativo (parámetros que afectan cómo cada estrategia decide los tiempos del semáforo) quedan **internos al sistema en MVP1**. Su exposición al Administrador requeriría conocimiento profundo de ingeniería de tráfico que excede el perfil de la Persona declarada y agregaría riesgo operativo sin valor proporcional. Su inclusión es trabajo futuro condicionado a (a) necesidad concreta de calibración fina, y (b) un Administrador con perfil técnico apropiado.

#### D. Métricas exactas en HU-14

**Decisión:** HU-14 (F18) cubre en MVP1 cuatro métricas de evaluación del modelo predictivo:

1. **MAE (Error Absoluto Medio)** sobre el ratio continuo.
2. **RMSE (Raíz del Error Cuadrático Medio)** sobre el ratio continuo.
3. **Accuracy (Exactitud)** sobre el nivel discreto 0-5.
4. **Matriz de confusión 6×6** del nivel discreto.

Cada una con **ícono de ayuda activable** que despliega una explicación breve de cómo interpretar la métrica.

**Convención de la matriz:** filas = nivel real observado, columnas = nivel predicho por el modelo (convención académica estándar, equivalente a la del módulo de métricas de scikit-learn). Declarada explícitamente en CA-14.8 y en el tooltip de la matriz.

**Presentación de la matriz:** valores absolutos (conteos) con totales de fila y columna, y un control toggle visible que permite alternar a porcentajes por fila para neutralizar el desbalance natural de clases. La diagonal principal (aciertos) es identificable visualmente respecto a las celdas fuera de la diagonal (errores).

**Justificación de incluir la matriz pese a su densidad visual:** la matriz aporta información que las tres métricas escalares no capturan (perfil de errores por nivel, no solo magnitud agregada), su costo de implementación es bajo, y los tooltips integrados mitigan la barrera cognitiva sin contaminar la vista con texto permanente.

#### E. Concurrencia entre Administradores en HU-15

**Decisión:** la concurrencia entre Administradores en la modificación de parámetros de HU-15 se resuelve con **last-write-wins con advertencia explícita al segundo Administrador** (mecanismo de control de concurrencia optimista con marca de versión).

Comportamiento detallado en CA-15.11: la primera modificación en guardarse se persiste normalmente; cuando el segundo Administrador intenta guardar, el sistema detecta que la configuración cambió desde su lectura inicial, le muestra una advertencia con detalles de la modificación intermedia (autor, timestamp, parámetros), y le ofrece confirmar la sobrescritura o cancelar y recargar para reevaluar. El registro de auditoría preserva ambas modificaciones, no solo la última.

**Justificación:** patrón estándar, bajo costo de implementación, suficiente para el escenario MVP1 donde la concurrencia será rara. Cumple el principio inegociable de no perder modificaciones silenciosamente.

#### F. Ventana temporal de cálculo de métricas de HU-14 inglobada en HU-15

**Contexto:** HU-14 (F18) declara una ventana temporal sobre la cual se calculan las métricas del modelo (CA-14.4, default sugerido 24 h). La nota técnica de HU-14 dejaba abierta la cuestión de si esta ventana se ajusta solo por configuración interna o se expone al Administrador.

**Decisión:** la ventana temporal se incluye como un parámetro configurable en HU-15, dentro de la familia "Predicción y evaluación del modelo".

**Justificación:** si el Administrador es responsable de evaluar el modelo (HU-14) y de configurar el sistema (HU-15), no tiene sentido que un parámetro central de la evaluación sea ajustable solo por variable de entorno. El costo marginal de agregar un parámetro más a HU-15 es bajo y cierra un cabo suelto sin abrir nuevos.

#### G. Creación de TTH-06 — Capa de DTOs transversal al backend

**Contexto:** durante la discusión sobre el patrón de consumo del endpoint compartido CT-04.5 por HU-11 (Operador) y HU-13 (Administrador), se evaluaron tres patrones posibles para manejar la diferencia de campos visibles: (1) un endpoint y un DTO completo, frontend filtra; (2) un endpoint con dos DTOs vía query parameter; (3) dos endpoints separados con dos DTOs.

**Decisión sobre el patrón de consumo concreto (CT-04.5):** patrón (1), un endpoint y los campos completos del DTO, frontend filtra según la vista. Los campos técnicos adicionales no son sensibles (son métricas operativas del propio sistema, no datos personales ni credenciales), por lo cual no se justifica un endpoint separado ni filtrado en backend según el rol del token; el RBAC a nivel de ruta es suficiente.

**Decisión transversal (TTH-06):** la cuestión más amplia de "introducir una capa explícita de DTOs en el backend o no" es una decisión transversal de arquitectura, no de HU-13. Se formaliza como **TTH-06 — Capa de DTOs transversal al backend**, clasificada como **Trabajos Futuros** (no se construye dentro del alcance del proyecto académico).

**Justificación de Trabajos Futuros y no MVP2:** TTH-06 no realiza ningún Objetivo del Producto; es higiene técnica de mantenibilidad. El alcance es transversal a todo el backend (no a un endpoint), lo cual hace difícil acotar el costo "si hay holgura". Naturalmente pertenece a la productivización del sistema, fuera del alcance académico. El sistema sin TTH-06 sigue siendo defendible académicamente.

#### H. Ampliación de CT-04.5 dentro de TTH-04

**Contexto:** el contrato original de CT-04.5 cubría nombre legible, estado cualitativo, timestamp del último cambio e identificador interno. HU-13 requiere campos adicionales no cubiertos: latencia de la última evaluación de salud, indicador de fallos recientes, timestamp de la última evaluación de salud exitosa.

**Decisión:** el contrato de CT-04.5 se amplía dentro de TTH-04 para cubrir los 7 campos requeridos por HU-13. HU-11 continúa consumiendo solo los campos básicos (1 a 3) e ignora los adicionales sin contradecir su contrato previo.

**No es TTH nueva ni decisión metodológica:** es refinamiento del contrato del endpoint existente. La modificación se documenta como ampliación de CT-04.5 dentro de TTH-04 y se cierra al cerrar el Bloque D.

### Documentos afectados por DHU-014

| Documento | Tipo de cambio |
|---|---|
| `HU_BLOQUE_D.md` (nuevo) | Documento nuevo con HU-13, HU-14, HU-15 redactadas. |
| `TAREAS_TECNICAS_HABILITADORAS.md` | Ampliación de CT-04.5 dentro de TTH-04 (subsección H); agregar TTH-06 (subsección G); actualización del índice y de la tabla de trazabilidad de TTH-04. |
| `DECISIONS_HU.md` (este documento) | Agregar DHU-014; actualizar índice, tabla de impacto en bloques y documentos relacionados. |
| `HU_BLOQUE_A.md`, `HU_BLOQUE_B.md`, `HU_BLOQUE_C.md` | Próximos pasos actualizados: Bloque D ya cerrado; restan Bloques E, F y MVP2. |
| `FEATURE_BACKLOG_DETALLADO.md` | Fichas de F17, F18 y F20 actualizan su columna "Modelado" para apuntar a HU-13, HU-14 y HU-15 respectivamente (estaban como "a redactar en el Bloque D"). |
| `LEAN_INCEPTION_CEREBROVIAL.md` | Documentos relacionados actualizado (referencia a `HU_BLOQUE_D.md`). |

### Lo que NO cambia con DHU-014

- **Las decisiones DHU-001 a DHU-013 mantienen su contenido sustantivo.** DHU-014 las cita pero no las reabre.
- **El alcance del producto** (Personas, Objetivos, Journeys, Visión) se mantiene intacto.
- **Las HUs del MVP1 redactadas en bloques previos** (HU-01 a HU-12) no se reabren; sus referencias a parámetros configurables (CA-02.3, CA-03.1, CA-03.3) ya apuntan al sistema de configuración que HU-15 ahora formaliza.
- **Las TTH previas (TTH-01 a TTH-05)** mantienen su contenido. TTH-04 recibe una ampliación de CT-04.5 (no contradice el contrato previo, solo lo extiende).

### Documentos relacionados

- `HU_BLOQUE_D.md` — Bloque D del Product Backlog.
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-04 ampliada y TTH-06 nueva.
- `DECISIONS_HU.md` (este documento) — sección DHU-014.

---

## DHU-015 — Clasificación HU/TTH de las features del Bloque E (con ampliación 4 → 5 TTH durante la redacción)

**Fecha:** 2026-05-15.
**Estado:** Cerrada.
**Aplica a:** Bloque E — Componentes centrales del sistema.

### Contexto

DHU-010 y DHU-013 cerraron la clasificación HU/TTH para los Bloques C y D respectivamente, aplicando los criterios de DHU-004 a las features de cada bloque. Esta decisión hace lo equivalente para el Bloque E, aplicando los mismos criterios a las cuatro features que lo componen según el Sequencer del Inception. Durante la redacción del bloque, al cerrar las decisiones arquitectónicas del modelo predictivo (TTH-09), se identificó la necesidad de una quinta TTH (TTH-11, spike de investigación de hiperparámetros temporales del modelo predictivo) que se incorporó como ampliación de esta decisión.

**Composición del Bloque E MVP1 (Sequencer del Inception, sin cambios desde el cierre del workshop original):**

- **F32** — Integración con SUMO para simulación del entorno.
- **F33** — Módulo de visión que produce métricas de estado.
- **F34** — Módulo predictivo GRU servido vía API.
- **F35** — Motor adaptativo (Webster + MaxPressure + MTC).

Las cuatro features están clasificadas como MVP1 y declaradas explícitamente con `Persona: SYS` en sus fichas (`FEATURE_BACKLOG_DETALLADO.md`), es decir, infraestructura/componentes del sistema sin Persona del producto beneficiaria directa.

### Análisis feature por feature

#### F32 — Integración con SUMO para simulación del entorno

| Criterio DHU-004 | F32 cumple |
|---|---|
| 1. Sin Persona beneficiaria directa | Sí. La ficha F32 declara `Persona: SYS` y la Revisión UX dice "no aplica (no expuesto al usuario directamente)". SUMO es infraestructura de validación cuantitativa y fuente del dataset de entrenamiento (D-008), no funcionalidad operativa visible. |
| 2. Valor instrumental, no de negocio | Sí. SUMO habilita el Objetivo 4 (demostrar mejora cuantificable) y habilita F34 (generación de dataset), pero no genera valor visible al Operador, Gerente ni Administrador en aislamiento. |
| 3. Comportamiento técnico estándar sin negociación de negocio | Sí. Cargar topología, generar escenarios de demanda, ejecutar simulación vía TraCI: son tareas técnicas sin negociación funcional con ninguna Persona. |
| 4. Sin valor visible al usuario en aislamiento | Sí. Ninguna Persona interactúa con SUMO. Su salida alimenta otras TTH (F34 dataset, validación del motor) y al capítulo de validación de la tesis. |

**Diagnóstico:** F32 cumple 4 de 4 criterios. Es **TTH (TTH-07)**.

#### F33 — Módulo de visión que produce métricas de estado

| Criterio DHU-004 | F33 cumple |
|---|---|
| 1. Sin Persona beneficiaria directa | Sí. Ficha F33 declara `Persona: SYS`. El Operador consume **derivadas funcionales** (flujo, cola observados) vía HU-02, pero HU-02 es agnóstica a la fuente (DHU-006); en MVP1 esa fuente es SUMO (D-007 + D-008), no visión. Visión es sensor en operación hipotética, no en el loop de validación cuantitativa (D-007). |
| 2. Valor instrumental, no de negocio | Sí. Habilita el Objetivo 1 en operación hipotética, pero su validación es independiente (métricas de detección sobre dataset etiquetado), no contribuye al loop de KPIs del sistema integrado. |
| 3. Comportamiento técnico estándar sin negociación de negocio | Sí. YOLO + tracking + exposición de métricas vía API son tareas técnicas estándar de visión computacional. Las decisiones de qué métricas exponer derivan de las HUs ya cerradas (HU-02, agnósticamente), no de una negociación abierta con la Persona. |
| 4. Sin valor visible al usuario en aislamiento | Sí. El Operador nunca interactúa con el módulo de visión directamente; consume sus derivadas funcionales cuando el sistema decide alimentarse de visión en operación. En MVP1 esas derivadas vienen de SUMO. |

**Diagnóstico:** F33 cumple 4 de 4 criterios. Es **TTH (TTH-08)**.

**Nota sobre el rol demostrativo:** F33 se modela como TTH porque HU-02 ya entrega el valor al Operador y HU-02 es agnóstica a la fuente. F33 es el sustrato técnico que en operación hipotética podría alimentar a HU-02. La separación arquitectónica de D-007 (visión demostrable, no en loop de validación cuantitativa) refuerza esta clasificación: la validación cuantitativa del sistema integrado se hace por SUMO; la validación de F33 se hace independientemente con métricas de detección.

#### F34 — Módulo predictivo GRU servido vía API

| Criterio DHU-004 | F34 cumple |
|---|---|
| 1. Sin Persona beneficiaria directa | Sí. Ficha F34 declara `Persona: SYS`. El Operador consume predicciones vía HU-03 (agnóstica al modelo); el Administrador consume métricas vía HU-14 (también agnóstica al modelo, declara MAE/RMSE/accuracy/matriz de confusión sin nombrar GRU). HU-03 y HU-14 son las consumidoras; el modelo es el sustrato. |
| 2. Valor instrumental, no de negocio | Sí. Habilita el Objetivo 2 (anticipar congestión), pero el valor al Operador llega vía HU-03 y al Administrador vía HU-14. El modelo en aislamiento es solo un endpoint que escupe números. |
| 3. Comportamiento técnico estándar sin negociación de negocio | Sí. GRU univariado por intersección (D-006), entrenado sobre dataset SUMO (D-008), prediciendo ratio continuo discretizable a jam level 0-5 (D-009): las tres decisiones técnicas que definen el modelo están cerradas. No queda negociación funcional con una Persona. |
| 4. Sin valor visible al usuario en aislamiento | Sí. Una Persona consume las predicciones presentadas por HU-03 o las métricas presentadas por HU-14; no consume el endpoint `/predictions/predict` directamente. |

**Diagnóstico:** F34 cumple 4 de 4 criterios. Es **TTH (TTH-09)**.

#### F35 — Motor adaptativo

| Criterio DHU-004 | F35 cumple |
|---|---|
| 1. Sin Persona beneficiaria directa | Sí. Ficha F35 declara `Persona: SYS / componente central`. El Operador consume la estrategia activa vía HU-05, su explicación vía HU-06, sus notificaciones vía HU-07, el historial vía HU-08; el Administrador configura parámetros vía HU-15. El motor es el sustrato; cinco HUs operativas ya cubren la superficie visible al usuario. |
| 2. Valor instrumental, no de negocio | Discutible pero diagnóstico claro. El valor al usuario llega vía HU-05/06/07/08 y HU-15, ya redactadas. El motor en aislamiento es lógica de decisión que escupe estrategia activa + parámetros aplicados al semáforo. El valor de negocio (aporte de ingeniería central de la tesis, según `EVOLUCION_TESIS.md`) se mide en el capítulo de validación, no como funcionalidad operativa visible. |
| 3. Comportamiento técnico estándar sin negociación de negocio | Sí. Las dos estrategias adaptativas (Webster, Max Pressure) y la capa de reglas duras MTC están en el código construido; la selección entre estrategias es lógica determinista según estado predicho + observado. La negociación con la Persona ya ocurrió al redactar HU-05/06/07/08 (qué ve, cómo se explica) y HU-15 (qué parámetros expone). |
| 4. Sin valor visible al usuario en aislamiento | Sí. El motor sin las HUs del Operador es un componente que toma decisiones invisibles. |

**Diagnóstico:** F35 cumple 4 de 4 criterios. Es **TTH (TTH-10)**.

**Nota sobre la arquitectura real del motor:** Durante la redacción de TTH-10 se clarificó la arquitectura real del motor según `CONTROL.md`: el motor es una **pipeline de dos etapas** (selección entre Webster y Max Pressure como estrategias adaptativas; aplicación de MTC como capa de reglas duras post-procesamiento), no un selector tripartita entre tres estrategias. Esta clarificación implica ajustes de coherencia documental en `EVOLUCION_TESIS.md` Fase 3 y, residualmente, en la descripción de F35 en `FEATURE_BACKLOG_DETALLADO.md`. No reabre decisiones técnicas previas; refina la descripción para coherencia con el código y el documento teórico.

**Nota sobre el aporte central:** Que F35 sea el aporte de ingeniería principal del trabajo no implica que deba modelarse como HU. La importancia académica de una pieza no determina su clasificación HU/TTH; la determina la presencia o ausencia de Persona del producto beneficiaria directa (DHU-004). El aporte central se documenta en el capítulo de validación de la tesis y en el video de demo, no en una HU operativa.

### Ampliación durante la redacción: TTH-11 (spike de hiperparámetros temporales)

Durante la redacción de TTH-09 se cerraron las decisiones arquitectónicas del modelo predictivo:

- Arquitectura multi-output (un GRU univariado por dirección, cada uno produce un vector de predicciones a múltiples horizontes en una sola inferencia).
- Cuatro modelos GRU univariados (uno por dirección de entrada de la intersección genérica de cuatro accesos).
- Endpoint devuelve ambos: ratio continuo + nivel discreto 0-5 derivado en backend.

Estas decisiones no atan los **hiperparámetros temporales** del modelo: paso de muestreo (Δt_in), ventana de entrada (lookback), horizonte de predicción, frecuencia de re-inferencia del endpoint. Los cuatro son hiperparámetros acoplados (cambiar uno afecta la interpretación de los otros) y merecen sustentación bibliográfica explícita para defensa académica.

**Decisión durante la redacción:** abrir **TTH-11** como spike de investigación con entregable documental, prerrequisito documental de TTH-09. El documento entregable se ubicará en `documentation/docs/` (sugerencia de nombre: `INVESTIGACION_HIPERPARAMETROS_TEMPORALES.md`). TTH-11 es **TTH**, no HU, conforme a DHU-004 (no tiene Persona beneficiaria directa; su valor es instrumental para reducir incertidumbre técnica). TTH-11 puede cerrar con su parte bibliográfica completa aun si TTH-07 sufre retrasos; la parte empírica se agrega como complemento del documento cuando TTH-07 esté disponible.

**Consecuencia formal:** DHU-015 se amplía de **4 TTH (TTH-07 a TTH-10) a 5 TTH (TTH-07 a TTH-11)**. El orden de redacción ajustado por dependencias técnicas es: TTH-07 → TTH-11 → TTH-09 → TTH-10 → TTH-08.

### Decisión final

**Bloque E MVP1: 0 HUs operativas + 5 TTH nuevas.**

| Feature | Modelado como | Identificador |
|---|---|---|
| F32 — Integración con SUMO | TTH | **TTH-07** |
| F33 — Módulo de visión | TTH | **TTH-08** |
| F34 — Módulo predictivo GRU | TTH | **TTH-09** |
| F35 — Motor adaptativo | TTH | **TTH-10** |
| (Derivada de TTH-09, sin feature asociada) | TTH | **TTH-11** |

### Granularidad: una TTH por componente, no agrupación

Cada feature del Bloque E se modela como TTH independiente, no como TTH compuesta ni agrupada. Justificación:

1. **Ciclos de implementación distintos.** F32 (SUMO) parte de cero con curva de aprendizaje real (ficha F32: "🆕 Por construir desde cero"); F33 (visión) se reconstruye desde cero como parte del refactor; F34 (GRU) parte de cero pero con RandomForest baseline preservado; F35 (motor) está significativamente construido. Agruparlas oscurece el plan de trabajo y dificulta el reporte de avance.

2. **Dependencias asimétricas.** F32 es prerrequisito de F34 (dataset de entrenamiento, D-008) y entrega los escenarios de validación que consumen F34 y F35. F33 es independiente del eje crítico. Una TTH compuesta no podría declarar correctamente estas dependencias internas.

3. **Validaciones independientes.** F32 (funcional, fidelidad de topología y simulación end-to-end), F33 (métricas de detección sobre dataset etiquetado independiente, D-007), F34 (MAE/RMSE/accuracy sobre escenarios SUMO no vistos, D-008), F35 (funcional con integraciones, validación cuantitativa en capítulo de tesis): cuatro criterios de Done independientes con instrumentación distinta.

4. **Precedente del backlog.** TTH-04 y TTH-05 del Bloque C son TTH separadas a pesar de que TTH-05 alimenta a TTH-04 internamente. El patrón "una TTH por componente con dependencia declarada en la sección de trazabilidad" ya está establecido.

### Orden de redacción aplicado

Por dependencias técnicas, con TTH-11 incorporada:

1. **TTH-07 (SUMO).** Prerrequisito de TTH-09 (dataset) y de la validación que TTH-10 consume.
2. **TTH-11 (Spike de hiperparámetros temporales).** Prerrequisito documental de TTH-09.
3. **TTH-09 (GRU).** Consume dataset de TTH-07 y sustentación de TTH-11.
4. **TTH-10 (Motor adaptativo).** Consume predicciones de TTH-09 y estado observado de TTH-07 en validación.
5. **TTH-08 (Visión).** Independiente del eje crítico; redactada al final por dependencia menor en MVP1.

### Validación de cada TTH

| TTH | Tipo de validación | Detalle |
|---|---|---|
| TTH-07 | Funcional | Topología cargada + simulación end-to-end vía TraCI + dataset generado + integración con motor adaptativo demostrable end-to-end. |
| TTH-08 | Independiente, métricas de detección | Precisión, recall, mAP sobre dataset etiquetado propio ≥200 frames. Objetivo aspiracional accuracy ≥ 80%. NO entra al loop de validación cuantitativa del sistema integrado (D-007). |
| TTH-09 | Funcional + cuantitativo de modelo | Endpoint sirviendo predicciones + cuatro métricas de HU-14 (MAE, RMSE sobre ratio continuo; accuracy, matriz de confusión sobre nivel discreto 0-5). **Objetivo aspiracional accuracy ≥ 80% sobre el nivel discreto 0-5, no bloqueante.** Si la realidad medida es peor, se reporta conforme a D-005. |
| TTH-10 | Funcional | Las dos estrategias adaptativas operan correctamente, AdaptiveEngine selecciona según criterios, MTC aplica reglas duras documentadas, integración con TTH-09/TTH-07/TTH-04 funciona end-to-end. La validación cuantitativa del sistema (mejora vs control fijo) pertenece al capítulo de validación de la tesis, no al Done de TTH-10. |
| TTH-11 | Documental | Documento entregable en `documentation/docs/` con revisión bibliográfica (mínimo 5 fuentes), exploración empírica mínima (3 combinaciones), recomendación final consolidada. |

### Consecuencias

- El Bloque E se redacta con 5 TTH y 0 HUs operativas. Numeración TTH-07 a TTH-11.
- El documento `HU_BLOQUE_E.md` se crea para mantener el patrón de un documento por bloque; su contenido principal es: mapeo de features → TTH, justificación de la ausencia de HUs operativas, decisiones tomadas durante la redacción, y referencias cruzadas a las TTH agregadas en `TAREAS_TECNICAS_HABILITADORAS.md`.
- `TAREAS_TECNICAS_HABILITADORAS.md` se actualiza con TTH-07 a TTH-11.
- La numeración de HUs del backlog no avanza con el Bloque E: la última HU operativa cerrada es HU-15 del Bloque D; la próxima HU operativa será HU-16 en el Bloque F (Gerente) o en sesión MVP2 dedicada.
- No se reabre ninguna decisión cerrada de los Bloques A, B, C ni D. HU-02 a HU-15 mantienen su contenido intacto; sus referencias agnósticas a "fuente de medición", "componente predictivo", "componente decisor" se materializan correctamente en TTH-07 a TTH-10.

### Lo que NO cambia con DHU-015

- **Las decisiones DHU-001 a DHU-014 mantienen su contenido sustantivo.** DHU-015 las aplica al Bloque E sin reabrirlas.
- **Las HUs ya redactadas (HU-01 a HU-15)** mantienen su contenido. Sus referencias a fuentes, componentes y modelos agnósticos se materializan en las TTH del Bloque E sin necesidad de modificar las HUs.
- **Las TTH previas (TTH-01 a TTH-06)** mantienen su contenido. TTH-04 (lógica de fallback) recibe referencias cruzadas adicionales desde TTH-09 (Nivel 2: predictor de respaldo es el RandomForest preservado por TTH-09) y TTH-10 (Nivel 3 y falla total invocan TTH-05 cuando TTH-10 cae), pero su contenido sustantivo no se reabre.

### Documentos afectados por DHU-015

| Documento | Tipo de cambio |
|---|---|
| `HU_BLOQUE_E.md` (nuevo) | Documento nuevo con mapeo de features F32-F35 → TTH-07 a TTH-11 y justificación de 0 HUs operativas. |
| `TAREAS_TECNICAS_HABILITADORAS.md` | Agregar TTH-07, TTH-08, TTH-09, TTH-10, TTH-11; actualización del índice. Referencias cruzadas adicionales en TTH-04 desde TTH-09 (Nivel 2 invoca RandomForest preservado por TTH-09) y TTH-10 (Nivel 3 invoca TTH-05 cuando TTH-10 cae). |
| `DECISIONS_HU.md` (este documento) | Agregar DHU-015; actualizar índice, tabla de impacto en bloques y documentos relacionados. |
| `HU_BLOQUE_A.md`, `HU_BLOQUE_B.md`, `HU_BLOQUE_C.md`, `HU_BLOQUE_D.md` | Próximos pasos actualizados: Bloque E ya cerrado; restan Bloque F y MVP2. |
| `FEATURE_BACKLOG_DETALLADO.md` | Fichas de F32, F33, F34 y F35 actualizan su columna "Modelado" para apuntar a TTH-07, TTH-08, TTH-09 y TTH-10 respectivamente (estaban como "A determinar al redactar el Bloque E"). Ajuste residual en la descripción de F35 para reflejar la arquitectura real del motor (2 estrategias + capa MTC). |
| `EVOLUCION_TESIS.md` | Fase 3 actualizada para reflejar la arquitectura real del motor (2 estrategias adaptativas + 1 capa de reglas duras), no "3 estrategias de control". Ajuste de coherencia documental. |
| `LEAN_INCEPTION_CEREBROVIAL.md` | Documentos relacionados actualizado (referencia a `HU_BLOQUE_E.md`). |

### Documentos relacionados

- `HU_BLOQUE_E.md` — Bloque E del Product Backlog (cierre de mapeo y decisiones).
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-07 a TTH-11 nuevas.
- `DECISIONS_HU.md` (este documento) — sección DHU-015.
- `DECISIONS.md` — D-006, D-007, D-008, D-009 fundamentan las decisiones técnicas internas de cada TTH del Bloque E.
- `EVOLUCION_TESIS.md` — Fase 4 (cierre arquitectónico) describe los cuatro componentes con roles separados.
- `CONTROL.md` — Sustentación teórica del motor adaptativo (consumido por TTH-10).

---

## DHU-016 — Decisiones de redacción del Bloque F (numeración, F30 inglobada, fuente del histórico en MVP1, KPIs operacionales, granularidad, periodos, comparativa, concurrencia, dashboard integrador, robustez)

**Fecha:** 2026-05-16.
**Estado:** Cerrada.
**Aplica a:** Bloque F — Gerente, reportería mínima.

### Contexto

DHU-015 cerró el Bloque E con 5 TTH operativas (TTH-07 a TTH-11) y 0 HUs operativas, dejando como próximo bloque pendiente el Bloque F — Gerente, reportería mínima. Las tres features que el Sequencer del Inception asigna a este bloque son F12 (Dashboard ejecutivo con KPIs agregados), F13 (Selector de periodo) y F14 (Vista comparativa entre periodos). Adicionalmente, la regla cerrada en el Bloque A determina que F30 (Persistencia de estados históricos) se ingloba como Criterios de Aceptación dentro de las HUs del Gerente, sin redactarse como HU dedicada.

Antes de iniciar la redacción de las HUs concretas del Bloque F era necesario cerrar un conjunto de decisiones de detalle que la sola aplicación de DHU-001 a DHU-015 no determina por sí misma. DHU-016 consolida esas decisiones en un acto único, siguiendo el patrón establecido por DHU-014 (decisiones consolidadas de redacción del Bloque D).

A diferencia de DHU-013 y DHU-015, esta decisión no requiere una sub-decisión de clasificación HU/TTH formal sobre las tres features del Bloque F: las tres tienen al Gerente como Persona beneficiaria directa (Journey 2, pasos 2 a 4) y cumplen las cuatro condiciones de HU operativa (DHU-004). La discusión metodológica sobre HU vs TTH para este bloque se cierra implícitamente al confirmar que las tres son HUs operativas y que F30 se mantiene como persistencia inglobada según la regla del Bloque A.

### Decisiones consolidadas

#### A. Numeración del Bloque F

El Bloque D cerró en HU-15. El Bloque E no avanzó la numeración de HUs (0 HUs operativas, según DHU-015). La numeración del Bloque F continúa secuencial desde HU-16.

**Decisión:** la numeración del Bloque F comienza en HU-16 (compactación secuencial desde el cierre del Bloque D, conforme a DHU-014 subsección A que estableció el principio de "no dejar huecos en la numeración para preservar memoria de HUs no redactadas o eliminadas; la traza histórica vive en `DECISIONS_HU.md`, no en el backlog").

La numeración final del Bloque F queda determinada por la subsección I (fusión F12 + F13 en una sola HU):

| HU | Feature(s) origen |
|---|---|
| HU-16 | F12 (Dashboard ejecutivo) + F13 (Selector de periodo) fusionadas, con F30 inglobada como CAs |
| HU-17 | F14 (Vista comparativa entre periodos), con consumo del histórico declarado por HU-16 |

#### B. F30 inglobada como Criterios de Aceptación, no como TTH del Bloque F

La regla cerrada en el Bloque A estableció que F30 (Persistencia de estados históricos) se modela como persistencia inglobada en HUs del Gerente, no como HU dedicada. La pregunta abierta para DHU-016 era si el cumplimiento de esa regla podía mantenerse en la forma de inglobación como CAs (patrón equivalente al de F31 en CA-08.1 de HU-08), o si la complejidad técnica de la persistencia histórica justificaba escalarla a TTH separada del Bloque F.

**Decisión:** F30 se ingloba como Criterios de Aceptación dentro de HU-16, no se extrae como TTH separada del Bloque F.

**Justificación:**

1. **Patrón previo equivalente.** F31 (persistencia de decisiones del motor) está inglobada en CA-08.1 de HU-08 y declarada como tal en su nota técnica. El sustrato técnico de F18 (registro de predicciones y cálculo de métricas) está inglobado en CA-14.1 a CA-14.4 de HU-14, sin TTH separada, conforme a DHU-013. El sustrato técnico de F20 (persistencia y auditoría de parámetros) está inglobado en CA-15.1 a CA-15.4 y CA-15.8 de HU-15. F30 sigue el mismo patrón.

2. **Criterio de "consumidores múltiples" no aplica.** TTH-04 fue justificada como TTH separada porque su salida (estado operativo del sistema) es consumida por HU-10, HU-11, HU-12 y HU-13: una pieza de lógica autónoma consumida por múltiples HUs heterogéneas. El histórico de F30 es consumido exclusivamente por las HUs del Bloque F (HU-16 y HU-17), que son cohesivas y comparten propósito (reportería ejecutiva al Gerente). No hay justificación de extracción por reuso.

3. **Complejidad técnica no justifica TTH.** La persistencia histórica es una tabla append-only de baja complejidad arquitectónica según la ficha F30 (complejidad Medio), comparable a la complejidad de los registros ya inglobados en HUs (predicciones de CA-14.1, parámetros de CA-15.1, decisiones del motor de CA-08.1). El umbral para extraer TTH no es la complejidad técnica sino la presencia de consumidores múltiples heterogéneos.

#### C. Fuente del histórico de F30 en MVP1

La ficha de F30 declara que la persistencia almacena "flujo, cola, velocidad, densidad por intersección y dirección, con timestamp". El sistema ya posee otros registros similares cerrados durante el Bloque E:

| Registro existente | Qué persiste | Origen |
|---|---|---|
| TTH-08 CT-08.5 | Métricas de estado observado (conteo, cola, flujo, densidad por dirección con timestamp) | Salida del módulo de visión |
| TTH-07 CT-07.3 | Dataset tabular del entorno simulado (velocidad, vehículos, cola, ratio, jam level) con marcas de seed/patrón/timestamp simulado | Generación offline de SUMO |
| TTH-09 CT-09.5 | Predicciones del modelo (timestamp, dirección, paso futuro, ratio, nivel) | Modelo predictivo |
| HU-08 CA-08.1 + TTH-10 CT-10.9 | Decisiones del motor (estrategia, razón, tiempos aplicados) | Motor adaptativo |
| TTH-04 CT-04.3 | Transiciones de estado operativo del sistema | Monitor de salud |

El registro de TTH-08 se solapa parcialmente con lo que F30 necesita persistir. Sin embargo, en MVP1 el módulo de visión no está en el loop de validación cuantitativa (D-007), y la cámara no corre continuamente en operación real durante el desarrollo académico. El dataset de TTH-07 sí contiene datos comparables, pero es offline (se genera por corridas reproducibles del script de CT-07.3), está pensado para entrenar TTH-09, y mezcla múltiples patrones de demanda y seeds, lo cual no es semánticamente una "operación histórica" del sistema sobre una intersección viva.

Se evaluaron tres opciones para resolver de dónde salen los datos de F30 en MVP1:

| Opción | Descripción | Consecuencia |
|---|---|---|
| A | F30 es persistencia operacional separada y agnóstica a la fuente, alimentada por la fuente de estado vigente del sistema en cada momento | Coherente con DHU-006 (HUs agnósticas a fuente). Independencia explícita respecto a TTH-07/TTH-08. En MVP1 la fuente vigente son corridas de validación cuantitativa en SUMO; el demo al jurado consulta el histórico generado en esas corridas. No se nombra a SUMO en las HUs. |
| B | F30 reusa CT-08.5 (persistencia de visión) y el Bloque F consume directamente de ahí | Ata el Bloque F a la salida de visión, que en MVP1 está fuera del loop por D-007. En MVP1 no habría datos para reportar. |
| C | F30 reusa el dataset de TTH-07 | Rompe la semántica: el dataset es offline para entrenamiento, no operacional para reportería; mezcla múltiples patrones y seeds. |

**Decisión:** se adopta la Opción A.

**Detalle de la decisión:**

1. F30 vive como persistencia operacional independiente, declarada como CAs dentro de HU-16. Su esquema mínimo incluye: marca de tiempo, identificador de intersección, dirección, y las cuatro variables observadas (flujo, longitud de cola, velocidad media, densidad). La persistencia es append-only, durable, y no se borra automáticamente en MVP1.

2. En MVP1, la fuente operacional vigente que alimenta F30 son las corridas de validación cuantitativa del sistema integrado en el entorno simulado de la intersección. Las HUs del Bloque F no nombran esta fuente (DHU-006); las notas técnicas y `TAREAS_TECNICAS_HABILITADORAS.md` la documentan.

3. La independencia de F30 respecto a TTH-07/TTH-08/TTH-09/HU-08/TTH-04 se declara explícitamente como nota técnica en HU-16, siguiendo la fórmula que CT-09.5 usa para declarar la independencia de TTH-09 respecto a TTH-04 y HU-08.

4. La conexión técnica entre F30 y la fuente vigente (cómo el sistema escribe a la tabla de F30 cuando corre una simulación de validación) es responsabilidad de implementación. No requiere TTH nueva: el comportamiento se declara en CAs de HU-16 y la implementación se resuelve al construir.

5. En operación hipotética posterior al alcance académico, la fuente vigente sería la salida del módulo de visión (TTH-08). La transición es transparente para las HUs del Bloque F porque su contrato es agnóstico a la fuente.

#### D. Definiciones operacionales de los 4 KPIs

El MVP Canvas (Bloque 6) cerró los 4 KPIs técnicos del sistema integrado: tiempo promedio de espera por vehículo, longitud máxima de cola por dirección, throughput de la intersección, demora promedio acumulada en periodo de simulación. La ficha de F12 identifica como riesgo "definir los KPIs específicos y su cálculo". DHU-016 cierra las definiciones operacionales para que HU-16 quede autocontenida:

| KPI | Definición operacional cerrada para MVP1 |
|---|---|
| Tiempo promedio de espera por vehículo | Media aritmética del tiempo, en segundos, que cada vehículo pasa con velocidad por debajo de un umbral bajo (sugerencia operativa: 0.1 m/s, cierre al implementar) durante su paso por la intersección. Agregado sobre todos los vehículos del periodo seleccionado, por dirección y total. |
| Longitud máxima de cola por dirección | Máximo de la longitud de cola, en número de vehículos, observado en cada dirección durante el periodo seleccionado. Se reporta por dirección, sin agregación al total de la intersección (el máximo de un agregado no es el agregado de los máximos). |
| Throughput de la intersección | Número total de vehículos que cruzan la intersección durante el periodo seleccionado, normalizado a vehículos por hora dividiendo por la duración del periodo. Agregado total, sin disgregación por dirección en la vista principal. |
| Demora promedio acumulada | Media aritmética, por vehículo, de la diferencia entre el tiempo real de paso del vehículo y el tiempo que tardaría en condiciones de free-flow (recorrido a velocidad libre sin detenciones). Agregado sobre todos los vehículos del periodo. |

**Decisión específica:** las cuatro definiciones se materializan como CAs específicos de cálculo en HU-16, y cada KPI lleva un tooltip de ayuda activable en la vista que despliega la definición operacional al Gerente (patrón establecido en CA-14.7 de HU-14 para las métricas del modelo predictivo).

**Detalles cerrados:**

1. **Cálculo: media aritmética, no percentiles.** Los promedios son aritméticos. Los percentiles (p50, p95) son trabajo futuro si se justifica.

2. **Disgregación por dirección.** Tiempo promedio de espera y longitud máxima de cola se reportan también por dirección de entrada. Throughput y demora promedio acumulada se reportan agregados a la intersección, sin disgregación por dirección en la vista principal (el agregado es lo que valida la tesis).

3. **Unidades:** segundos para tiempos, vehículos para conteos, vehículos/hora para throughput, segundos para demora acumulada por vehículo.

4. **Free-flow para demora:** se calcula como tiempo de cruce a velocidad libre del acceso (`longitud_acceso / max_speed_acceso`), tomando max_speed del archivo de red de la intersección, congruente con la nota técnica de TTH-07 sobre el mapeo SUMO → jam level.

#### E. Granularidad temporal del histórico persistido

La ficha de F30 deja abierto "¿estados cada 1s, 10s, 1min?" y sugiere 30 segundos para validación. Sin política de retención clara.

**Decisión:** la granularidad de agregación del histórico persistido es de 30 segundos por intersección y por dirección. No hay política de retención automática en MVP1: el histórico se acumula durante el alcance académico sin borrado programado.

**Justificación de los 30 segundos:** equilibra (a) resolución suficiente para reconstruir tendencias en periodos semanales y mensuales, (b) volumen razonable de filas para una intersección durante el alcance académico (aproximadamente 2880 filas por dirección y por día), (c) coherencia con la frecuencia típica de actualización de la vista del Operador (HU-02 actualiza en tiempo casi-real, no requiere persistir cada segundo).

**Exposición al Administrador:** la granularidad **no se expone** como parámetro configurable en HU-15 en MVP1. Cambiar la granularidad históricamente acumulada introduce complejidad de migración (filas de granularidad mixta) que no aporta valor en el alcance académico. Si en el futuro se justifica exposición, se evalúa entonces; no es trabajo del Bloque F.

#### F. Periodos predefinidos del selector

La ficha de F13 sugiere "esta semana, semana anterior, este mes, mes anterior + rango personalizado". DHU-016 cierra la lista exacta y las convenciones de cálculo.

**Decisión:** el selector ofrece cuatro presets más un rango personalizado:

1. **Esta semana** — desde el lunes 00:00 hasta el momento actual.
2. **Semana anterior** — desde el lunes de la semana previa 00:00 hasta el domingo previo 23:59:59.
3. **Este mes** — desde el día 1 del mes actual 00:00 hasta el momento actual.
4. **Mes anterior** — desde el día 1 del mes previo 00:00 hasta el último día del mes previo 23:59:59.
5. **Rango personalizado** — el Gerente selecciona fecha de inicio y fecha de fin mediante un componente date picker.

**Convenciones cerradas:**

- Semana inicia los **lunes** (convención ISO 8601, predominante en contexto académico peruano y latinoamericano).
- Mes natural calendario (no rolling 30 días).
- Zona horaria del sistema (la del despliegue del servidor; en MVP1 se asume zona horaria de Lima, Perú).
- El periodo "trimestre" mencionado en el título original de F13 **no se incluye en MVP1**: tres meses requieren especificación adicional (¿trimestre calendario natural Q1/Q2/Q3/Q4? ¿últimos 90 días?) que excede el alcance mínimo. Se evalúa como mejora si surge necesidad concreta.

#### G. Definición de "periodo previo equivalente" en HU-17

La ficha de F14 identifica como riesgo "decidir qué considera periodo previo equivalente". DHU-016 cierra:

**Decisión:** el periodo previo equivalente de la vista comparativa (HU-17) es el periodo del mismo tipo inmediatamente anterior al actual:

- Si el periodo seleccionado es "esta semana" → comparativo es "semana anterior".
- Si el periodo seleccionado es "este mes" → comparativo es "mes anterior".
- Si el periodo seleccionado es "semana anterior" → comparativo es "dos semanas atrás".
- Si el periodo seleccionado es "mes anterior" → comparativo es "dos meses atrás".
- Si el periodo seleccionado es "rango personalizado" → comparativo es el rango de igual duración inmediatamente anterior al rango actual (por ejemplo: si el rango actual es del 1 al 15 de marzo, el comparativo es del 14 al 28 de febrero).

**Justificación:** patrón estándar de herramientas analíticas (Google Analytics, Mixpanel, Tableau usan la misma convención). Bajo riesgo de implementación. Cubre los cuatro casos del selector sin agregar UI nueva.

#### H. Concurrencia entre Gerentes

**Diagnóstico:** las HUs del Bloque F son **read-only**. El Gerente consulta KPIs y comparativas, no edita configuración ni datos. Múltiples Gerentes consultando simultáneamente es un caso de carga, no de concurrencia funcional.

**Decisión:** las HUs del Bloque F no incluyen mecanismo de control de concurrencia (no hay last-write-wins porque no hay write). La concurrencia entre Gerentes se documenta explícitamente como **no aplicable** en una nota técnica de HU-16 para evitar ambigüedad.

**Justificación:** análoga a por qué HU-02 y HU-03 (consulta en tiempo real del Operador) no incluyen mecanismo de control de concurrencia, a diferencia de HU-15 (configuración de parámetros del Administrador) que sí lo requiere (CA-15.11).

#### I. Dashboard integrador del Gerente y composición de HUs

**Contexto:** el Sequencer del Inception lista tres features para el Bloque F (F12, F13, F14). Si cada feature se modela como HU separada, el Bloque F tendría tres HUs operativas (estimación inicial del usuario al iniciar la sesión).

**Análisis:** el selector de periodo (F13) no entrega valor en aislamiento; su único propósito es gobernar lo que muestra el dashboard ejecutivo (F12) y la comparativa (F14). Una HU dedicada al selector violaría el principio de cohesión de Mike Cohn ("una HU = un valor entregable autocontenido"). El selector no es una pieza de funcionalidad autónoma sino un componente de control sobre las vistas que sí entregan valor.

A diferencia del Administrador (DHU-014 subsección B), el Gerente sí trabaja sobre un único objeto compuesto: los KPIs del periodo seleccionado. F12 y F13 son altamente acoplados: la consulta de KPIs sin selector es un dashboard estático, y el selector sin KPIs es un componente vacío. F14 sí es separable: la comparativa es una vista distinta que reutiliza el periodo seleccionado, pero entrega un valor diferenciado (tendencia, no estado).

**Decisión:** el Bloque F se redacta con **2 HUs operativas, no 3**:

- **HU-16** fusiona F12 (Dashboard ejecutivo) y F13 (Selector de periodo) en una sola HU "Consulta de KPIs operativos sobre periodo seleccionable". El selector y el dashboard viven como CAs distintos dentro de la misma HU. F30 se ingloba como CAs adicionales en esta misma HU.

- **HU-17** mantiene F14 (Vista comparativa entre periodos) como HU separada. Reutiliza el selector definido en HU-16 (la selección de periodo es estado compartido entre las dos vistas del Gerente) y entrega valor diferenciado (comparativa con periodo previo equivalente).

**Sin HU dedicada de dashboard integrador.** Análogamente a DHU-014 subsección B (sin HU dedicada de dashboard del Administrador), el Gerente no requiere HU dedicada de "dashboard integrador" análoga a F02 del Bloque B. La navegación del Gerente da acceso a HU-16 (consulta principal) y HU-17 (comparativa), y eso es suficiente. La diferencia respecto al Operador es que el Gerente no monitorea en tiempo real: el valor agregado de un dashboard integrador único como F02 viene del simultaneismo del tiempo real, que no aplica al Gerente.

**Consecuencia formal:** el Bloque F cierra con **2 HUs operativas + 0 TTH nuevas**, no 3 HUs. La estimación inicial de "~3 HUs MVP1" en el mensaje de arranque queda revisada por DHU-016 a "2 HUs MVP1". La compactación preserva la cobertura funcional de las tres features y mejora la cohesión semántica de las HUs.

#### J. Aplicación de DHU-005 al Bloque F (robustez ante interrupción de fuente)

DHU-005 declara dos casos: Caso A (fuente externa de medición) y Caso B (componente interno de decisión). El Bloque F no opera en tiempo real, pero las HUs del Gerente dependen de que la persistencia histórica de F30 y el motor de cálculo de KPIs estén disponibles.

**Decisión:** las HUs del Bloque F aplican **DHU-005 Caso B** al motor de cálculo de KPIs y al subsistema de consulta del histórico:

- Cuando la persistencia de F30 deja de responder, la vista muestra los últimos KPIs calculados marcados como "no actualizados", indicando el timestamp del último cálculo exitoso.
- Cuando el motor de cálculo de KPIs no puede completar el cálculo del periodo solicitado, la vista comunica explícitamente la indisponibilidad temporal en lugar de mostrar KPIs en cero (que podrían confundirse con un periodo de tráfico cero, valor distinto a "no se pudo calcular").

**Patrón previo:** CA-14.12 de HU-14 aplica el mismo principio al motor de cálculo de métricas del modelo predictivo. Las HUs del Bloque F lo aplican análogamente al motor de cálculo de KPIs.

**Manejo de caso degenerado (sin datos en el periodo):** si el periodo seleccionado no contiene datos persistidos (por ejemplo, el Gerente selecciona "semana anterior" pero el sistema aún no estaba operativo entonces), la vista comunica explícitamente "no hay datos en el periodo seleccionado" en lugar de mostrar KPIs calculados sobre cero filas. Patrón análogo a CA-14.11 de HU-14.

### Decisión final

**Bloque F MVP1: 2 HUs operativas + 0 TTH nuevas.**

| Feature | Modelado como | Identificador |
|---|---|---|
| F12 (Dashboard ejecutivo) + F13 (Selector de periodo) | HU fusionada del Gerente | **HU-16** |
| F30 (Persistencia de estados históricos) | Inglobada como CAs en HU-16 | (CAs específicos) |
| F14 (Vista comparativa entre periodos) | HU del Gerente | **HU-17** |

**Total Bloque F:** 2 HUs operativas + 0 TTH nuevas. F30 inglobada en CAs de HU-16, conforme a la regla del Bloque A y al patrón establecido para F31 en CA-08.1 de HU-08.

### Lo que NO cambia con DHU-016

- **Las decisiones DHU-001 a DHU-015 mantienen su contenido sustantivo.** DHU-016 las cita y aplica al Bloque F sin reabrir ninguna.
- **El alcance del producto** (Personas, Objetivos, Journeys, Visión) se mantiene intacto.
- **Las HUs MVP1 redactadas en bloques previos (HU-01 a HU-15)** no se reabren. F30 inglobada en HU-16 es persistencia operacional independiente respecto a TTH-07, TTH-08, TTH-09, HU-08 CA-08.1 y TTH-04 CT-04.3; ninguno de esos registros se ve modificado.
- **Las TTH previas (TTH-01 a TTH-11)** mantienen su contenido. El cálculo de KPIs del Bloque F sobre el histórico persistido por F30 no requiere reabrir ningún CT.
- **Los 4 KPIs del MVP Canvas Bloque 6** se mantienen como base de validación cuantitativa. Las definiciones operacionales de la subsección D refinan su cálculo sin alterar su selección.

### Documentos afectados por DHU-016

| Documento | Tipo de cambio |
|---|---|
| `HU_BLOQUE_F.md` (nuevo) | Documento nuevo con HU-16 (F12 + F13 fusionadas + F30 inglobada) y HU-17 (F14). |
| `DECISIONS_HU.md` (este documento) | Agregar DHU-016; actualizar índice, tabla de impacto en bloques y documentos relacionados. |
| `FEATURE_BACKLOG_DETALLADO.md` | Fichas de F12, F13, F14 y F30 actualizan su columna "Modelado" para apuntar a HU-16, HU-17 y CAs específicos. La ficha de F13 incorpora la decisión sobre presets cerrados (subsección F) y la exclusión de "trimestre" en MVP1. La ficha de F14 incorpora la definición de "periodo previo equivalente" (subsección G). La ficha de F30 incorpora la granularidad cerrada y la independencia respecto a otros registros (subsecciones C y E). La ficha de F12 incorpora las definiciones operacionales de los 4 KPIs (subsección D). |
| `HU_BLOQUE_A.md`, `HU_BLOQUE_B.md`, `HU_BLOQUE_C.md`, `HU_BLOQUE_D.md`, `HU_BLOQUE_E.md` | Próximos pasos actualizados: Bloque F ya cerrado; resta MVP2. |
| `LEAN_INCEPTION_CEREBROVIAL.md` | Documentos relacionados actualizado (referencia a `HU_BLOQUE_F.md`). |

### Documentos relacionados

- `HU_BLOQUE_F.md` — Bloque F del Product Backlog (2 HUs operativas: HU-16, HU-17).
- `DECISIONS_HU.md` (este documento) — sección DHU-016.
- `LEAN_INCEPTION_CEREBROVIAL.md` — Persona Gerente, Journey 2, MVP Canvas Bloque 6 (KPIs).
- `FEATURE_BACKLOG_DETALLADO.md` — fichas F12, F13, F14, F30.
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-07/CT-07.3, TTH-08/CT-08.5, TTH-09/CT-09.5 (registros operacionales preexistentes; F30 declara independencia explícita respecto a estos).

---

## DHU-017 — Decisiones de redacción del MVP2

**Fecha:** 2026-05-16.
**Alcance:** Decisiones que afectan la redacción del MVP2 (HU-18, HU-19, HU-20, HU-21 redactadas en esta sesión; HU-09 ya cerrada en `HU_BLOQUE_B.md` desde el cierre del Bloque B). Decisiones agrupadas temáticamente en diez subsecciones (A a J).

### Contexto

El cierre del Bloque F (DHU-016) completó las 16 HUs operativas MVP1 (HU-01 a HU-08 + HU-10 a HU-17) y las 11 TTH del MVP1. Como cierre del Product Backlog en su componente funcional, restaba la sesión dedicada al MVP2 con las 4 features pendientes (F15, F16, F19, F28; F11 ya estaba redactada como HU-09 en `HU_BLOQUE_B.md` desde el cierre del Bloque B). Cada una de esas 4 features tenía decisiones específicas pendientes sobre clasificación, composición, alcance, sustrato técnico e integración con el MVP1 ya cerrado.

La redacción del MVP2 reproduce el patrón establecido por DHU-014 (Bloque D), DHU-015 (Bloque E) y DHU-016 (Bloque F): una decisión consolidada que agrupa las decisiones de redacción del conjunto en subsecciones temáticas. DHU-017 es la séptima y última decisión consolidada del Product Backlog.

### Decisiones aplicadas

#### A. Clasificación HU/TTH de las 4 features MVP2 pendientes

Cada una de las features F15, F16, F19 y F28 fue sometida individualmente a los cuatro criterios establecidos en DHU-004 para clasificarla como HU o TTH:

1. ¿Tiene Persona del producto como beneficiaria directa?
2. ¿Tiene valor de negocio claro y visible al usuario?
3. ¿Es comportamiento negociable con la Persona, o trabajo técnico no-negociable?
4. ¿Entrega valor en aislamiento, o solo es invisible habilitador?

Resultado del análisis:

| Feature | Criterio 1 (Persona) | Criterio 2 (Valor) | Criterio 3 (Negociable) | Criterio 4 (Visible) | Clasif. |
|---|---|---|---|---|---|
| F15 (Drill-down) | Sí (Gerente) | Sí (investigación de variaciones) | Sí (alcance, integración de carriles) | Sí (vista específica) | **HU** |
| F16 (Exportación PDF/Excel) | Sí (Gerente) | Sí (transferir reportes fuera del sistema) | Sí (formato, layout, política ante fallo) | Sí (archivo descargable) | **HU** |
| F19 (Comparativa vs baseline) | Sí (Administrador) | Sí (sustentar decisión sobre modelo) | Sí (alcance, métricas, ventana, tolerancia) | Sí (vista comparativa) | **HU** |
| F28 (Escalamiento) | Sí (Operador originador + Administrador destinatario) | Sí (transferir incidentes con trazabilidad) | Sí (alcance, captura, atención) | Sí (botón + flujo de gestión) | **HU** |

**Las cuatro cumplen 0 de 4 criterios para clasificarse como TTH. Las cuatro son HUs operativas, no TTH.**

**Decisión:** El MVP2 cierra con 4 HUs operativas nuevas (HU-18, HU-19, HU-20, HU-21) y **0 TTH nuevas**. El total del Product Backlog al cierre del MVP2 es 21 HUs operativas (HU-01 a HU-21, incluyendo HU-09 anticipada al Bloque B) y 11 TTH (TTH-01 a TTH-11). El sustrato técnico requerido por las 4 HUs MVP2 se ingloba como CAs conforme a DHU-013 y a la subsección H de esta decisión.

#### B. Numeración compactada del MVP2

Las HUs del MVP2 retoman la numeración secuencial desde HU-17 (última cerrada en el Bloque F), conforme al principio de DHU-014 subsección A: no dejar huecos en la numeración para preservar memoria de HUs no redactadas o eliminadas; la traza histórica vive en `DECISIONS_HU.md` y en `FEATURE_BACKLOG_DETALLADO.md`, no en el backlog.

**Decisión:**

- **HU-18** ← F15 (Drill-down).
- **HU-19** ← F16 (Exportación PDF/Excel).
- **HU-20** ← F19 (Comparativa vs baseline).
- **HU-21** ← F28 (Escalamiento).

**HU-09 conserva su número** en `HU_BLOQUE_B.md` (no se renumera ni se traslada físicamente a `HU_MVP2.md`). La cohesión temática del Bloque B (Operador, núcleo de monitoreo) se preserva manteniendo HU-09 en su ubicación original; `HU_MVP2.md` solo lista HU-09 en el mapeo con referencia cruzada explícita, sin duplicar contenido.

#### C. Composición de F16 (Exportación PDF/Excel) como HU única

F16 admite cuatro combinaciones potenciales de formato × vista origen: PDF × HU-16, PDF × HU-17, Excel × HU-16, Excel × HU-17. La discusión fue si modelarlas como HUs separadas, como dos HUs (una por formato o una por vista origen), o como una HU única que cubra las cuatro combinaciones mediante CAs estructurados.

**Decisión: HU única que cubre las cuatro combinaciones mediante CAs estructurados** (HU-19). El criterio de cohesión de Mike Cohn aplicado en DHU-016 subsección I a la fusión F12+F13 favorece la unificación cuando el valor entregable es uno solo (exportar el reporte del Gerente fuera del sistema) con variantes de presentación o configuración que el Gerente elige al momento de exportar. Descomponer en HUs separadas crearía fragmentación artificial: el Gerente no piensa en términos de "exportar HU-16 a PDF" como historia distinta de "exportar HU-17 a Excel"; piensa en términos de "necesito un reporte transferible" con elecciones de formato y vista.

**Patrón previo:** la fusión F12+F13 en HU-16 establecida en DHU-016 subsección I. DHU-017 subsección C extiende el patrón a fusión por variantes de presentación cuando el valor entregable es cohesivo.

#### D. Fuente y alcance del baseline de F19 (Comparativa)

F19 plantea la pregunta de cuál baseline comparar contra el modelo principal y cómo obtenerlo. Dos alternativas se contrastaron:

1. **Baseline estático**: dataset de evaluación congelado al momento del entrenamiento original. Refleja el estado del modelo en el laboratorio, no su operación reciente.
2. **Baseline operacional paralelo**: registro paralelo de predicciones del modelo de respaldo ejecutándose continuamente sobre los mismos inputs operativos que el modelo principal. Refleja la operación reciente sobre datos productivos.

La decisión técnica es relevante porque condiciona qué comparativa el Administrador ve y qué decisión puede sustentar con ella.

**Decisión: registro paralelo del baseline persistido como extensión de CA-14.1 de HU-14**. El modelo de respaldo (preservado en el sistema por TTH-09 y declarado en `EVOLUCION_TESIS.md` Fase 2 como baseline de comparación en validación) ejecuta predicciones en paralelo continuamente sobre los mismos inputs operativos que el modelo principal. Cada predicción del modelo de respaldo se persiste en el mismo registro declarado en CA-14.1 de HU-14 y materializado en CT-09.5 de TTH-09, con el campo "identificador del modelo o versión" como discriminante. Cuando llega el horizonte de cada predicción, el sistema asocia ambas (principal y respaldo) con la misma observación real, permitiendo calcular las métricas de ambos modelos sobre exactamente los mismos eventos.

**Sustrato inglobado, no TTH nueva:** la extensión del registro de predicciones para persistir también las predicciones del modelo de respaldo se ingloba como CAs de HU-20 (CA-20.1 a CA-20.4), conforme a la regla establecida por DHU-013. El esquema de CT-09.5 no se modifica: la extensión consiste en escribir adicionalmente predicciones del modelo de respaldo con identificador de modelo distinto, dentro del mismo esquema. HU-14 sigue mostrando solo el modelo principal en su vista individual; el modelo de respaldo se agrega al registro para que HU-20 lo compare.

**RandomForest preservado como único baseline declarado en TTH-09:** la identidad concreta del modelo de respaldo (RandomForest según TTH-09 nota técnica y `EVOLUCION_TESIS.md` Fase 2) vive en las TTH y decisiones técnicas. La HU-20 misma es **agnóstica** conforme a DHU-006: declara "modelo predictivo principal" y "modelo predictivo de respaldo", sin nombrar GRU ni RandomForest.

#### E. Composición de F28 (Escalamiento) y actor protagonista

F28 modela un flujo que tiene dos caras: el Operador que escala (originador del flujo) y el Administrador que recibe el escalamiento (destinatario y gestor del incidente). La discusión fue si modelar como una sola HU con sujeto compuesto o como dos HUs separadas (una por actor).

**Decisión: HU única con Operador protagonista y Administrador destinatario** (HU-21). Tres criterios sustentan la decisión:

1. **Cohesión del valor entregable**: el escalamiento es un solo flujo con dos caras que comparten un único objeto compuesto (el incidente escalado). Separar en dos HUs fragmentaría artificialmente algo que es operacionalmente una sola comunicación entre roles.
2. **Patrón de sujeto compuesto ya aceptado**: DHU-003 estableció el patrón de sujetos compuestos válidos (HU-01 del Bloque A tiene como sujeto "Usuario del sistema (Operador, Gerente, Administrador)"). DHU-017 subsección E extiende el patrón con un caso específico: protagonista + destinatario declarado.
3. **Cohesión del sustrato técnico**: la persistencia del incidente es un único registro consultado desde dos perspectivas (Operador ve los suyos; Administrador ve todos). Modelar dos HUs separadas requeriría declarar la persistencia compartida en alguna de las dos, generando dependencia artificial.

**Implementación formal del sujeto compuesto:** el "Como" de HU-21 declara "Operador de Tráfico Municipal" (protagonista). El "Para" declara como destinatario al Administrador del Sistema. Los CAs específicos de la vista del Administrador (CA-21.14 a CA-21.23) operacionalizan el rol del Administrador como sujeto operativo en esas vistas.

#### F. Alcance del escalamiento en MVP2

F28 admite alcance variable: desde un botón mínimo viable (registrar el evento) hasta un sistema completo de gestión de incidentes (conversación bidireccional, categorías, prioridades, notificaciones push, dashboards). La discusión fue acotar el alcance al mínimo viable que entregue valor.

**Decisión: alcance mínimo viable, coherente con el patrón establecido por HU-09**:

1. **Notificación unidireccional + persistencia incidente**: el Operador envía; el sistema persiste; el Administrador consulta. Sin conversación bidireccional. Sin respuesta textual del Administrador. El único feedback al Operador es el cambio de estado del incidente (de "Enviado" a "Atendido").
2. **Vista del Operador + vista del Administrador**: cada uno tiene su vista para consultar los incidentes (el Operador ve los suyos; el Administrador ve todos los recibidos con filtros).
3. **Transición irreversible "Atendido"**: una vez marcado como atendido, no hay reversión desde la vista del Administrador en MVP2. Casos de error son aceptables (queda registrado con identidad y timestamp); la reversión es trabajo futuro si se justifica.
4. **Sin conversación bidireccional**: las coordinaciones más profundas se hacen por canales externos al sistema en MVP2. F28 ampliada con conversación es trabajo futuro.
5. **Sin notificaciones push**: el Administrador detecta nuevos incidentes mediante un indicador pasivo (badge numérico) visible en su navegación. Las notificaciones activas (push, sonoras, correo) son alcance de F40 (Trabajos Futuros).

**Patrón previo:** HU-09 (notas e incidencias del turno) estableció el patrón de alcance mínimo viable en una HU del Operador: registro + consulta + flujo simple sin sofisticación. HU-21 hereda el patrón con la diferencia de que en HU-21 el flujo cruza dos roles, mientras que en HU-09 vive completamente dentro del rol Operador.

**Patrón de robustez:** la operación del motor adaptativo NUNCA depende del registro de incidentes (CA-21.13). El registro es de comunicación entre roles, no parte del control del semáforo; su indisponibilidad no afecta la operación. Patrón heredado de CA-09.6 de HU-09.

#### G. Alcance del drill-down de F15 y conexión con HU-16/HU-17

F15 (Vista detallada de periodo específico) plantea la pregunta de qué carriles temporales debe integrar el drill-down y cómo conectarse con las vistas agregadas del Gerente.

**Decisión 1: Tres carriles temporales integrados sobre la misma línea temporal**. El drill-down (HU-18) integra:

1. **Carril de evolución del tráfico**: alimentado por el histórico de estados de F30 inglobada en HU-16 (CA-16.1 a CA-16.3). Resolución temporal más fina que HU-16, con zoom interactivo que llega a la granularidad nativa de 30 segundos cerrada en DHU-016 subsección E.
2. **Carril de eventos del motor adaptativo**: alimentado por el registro de decisiones declarado en CA-08.1 de HU-08. Cada decisión renderizada como marcador activable.
3. **Carril de estado operativo del sistema**: alimentado por el registro de transiciones declarado en CT-04.3 de TTH-04. Intervalos renderizados como bandas coloreadas con los códigos visuales cerrados en DHU-012 subsección F.

El valor del drill-down está precisamente en la correlación visual entre los tres carriles: si una variación detectada en HU-16 o HU-17 coincide con un episodio de degradación o con un cambio de estrategia del motor, el Gerente puede distinguir explicaciones competidoras de una misma variación agregada.

**Decisión 2: Conexión bidireccional con HU-16 y HU-17 mediante estado compartido del selector**. El acceso al drill-down se ofrece desde dos puntos:

- **Desde HU-16**: botón visible "Ver detalle del periodo" (sobre el periodo actualmente seleccionado) + click sobre cualquier punto de los gráficos temporales (sub-periodo centrado en el momento del punto).
- **Desde HU-17**: botón "Ver detalle" con menú compacto de elección entre "Periodo actual" o "Periodo previo equivalente" (cada uno con sus fechas como subtítulo para evitar ambigüedad).

**Decisión 3: Distinción entre navegación local y cambio del selector global**. El periodo se hereda del estado compartido del selector entre HU-16, HU-17 y HU-18, pero la navegación a HU-18 con un sub-periodo o con el periodo previo equivalente **no muta el selector compartido** (es navegación local que pasa el periodo como contexto inicial de HU-18). En cambio, **cambiar el periodo desde el propio selector dentro de HU-18 sí muta el selector compartido**. Esta distinción operacional permite al Gerente navegar al detalle de un periodo concreto y regresar a su vista agregada original sin perder su contexto de trabajo previo, mientras que un cambio explícito del periodo de análisis se propaga consistentemente.

#### H. Sustrato técnico inglobado, sin TTH nuevas

Las 4 HUs del MVP2 requieren sustrato técnico de distintos tipos. El análisis fue si extraer ese sustrato como TTH separadas o inglobarlo como CAs en las HUs respectivas, conforme al criterio establecido por DHU-013.

**Decisión: el sustrato técnico requerido por las 4 HUs MVP2 se ingloba como CAs en las HUs respectivas. 0 TTH nuevas en el MVP2.**

Análisis caso por caso:

- **HU-18 (F15)**: no requiere sustrato técnico nuevo. Reutiliza tres registros existentes (histórico de F30 inglobada en HU-16; registro de decisiones del motor de CA-08.1 de HU-08; registro de transiciones de CT-04.3 de TTH-04). Es lógica de presentación que integra tres fuentes ya cerradas. No hay sustrato nuevo que extraer.
- **HU-19 (F16)**: no requiere sustrato técnico nuevo. La generación de PDF y Excel consume los mismos datos que HU-16 y HU-17 ya consumen (histórico de F30, motor de cálculo de KPIs inglobado, lógica de comparativa inglobada). Es lógica de presentación.
- **HU-20 (F19)**: requiere extensión del registro de predicciones para persistir también las predicciones del modelo de respaldo. La extensión se ingloba en CA-20.1 a CA-20.4 de HU-20. El esquema de CT-09.5 no se modifica (la extensión consiste en escribir filas adicionales con identificador de modelo distinto, dentro del mismo esquema). El sustrato extendido es **consumido únicamente** por HU-20; HU-14 sigue mostrando solo el modelo principal en su vista individual. Sin consumidor heterogéneo que justifique TTH separada.
- **HU-21 (F28)**: requiere persistencia nueva del registro de incidentes escalados. La persistencia se ingloba en CA-21.10 a CA-21.13 de HU-21. El registro es **consumido únicamente por HU-21** (vista del Operador para sus escalamientos + vista del Administrador para incidentes recibidos). Sin consumidor heterogéneo que justifique TTH separada.

**Patrón previo:** la inglobación de sustrato técnico operacional dentro de la HU que lo requiere cuando no hay consumidor heterogéneo está establecida desde DHU-013 (Bloque D) y aplicada en HU-14 (CA-14.1 a CA-14.4), HU-15 (CA-15.1 a CA-15.4 y CA-15.8), HU-16 (CA-16.1 a CA-16.3, F30 inglobada), HU-08 (CA-08.1, F31 inglobada). HU-20 y HU-21 extienden el patrón al MVP2.

**Consecuencia formal:** las 11 TTH del Product Backlog (TTH-01 a TTH-11) cubren todo el sustrato técnico habilitador del producto. El MVP2 no introduce TTH adicionales.

#### I. Política de construcción MVP2 heredada de DHU-012

DHU-012 refinó la semántica de "MVP2" desde la concepción inicial ("HUs fuera del sprint, sin compromiso de construcción") a la concepción operacional posterior ("HUs documentadas como completas con criterios de aceptación; candidatas a construcción condicional a la holgura del cronograma tras cerrar las HUs MVP1").

**Decisión: la política de construcción MVP2 declarada por DHU-012 aplica sin modificación a HU-18 a HU-21. Cada HU del MVP2 incluye en su sección "Notas técnicas" la siguiente declaración estándar:**

> *Política de construcción MVP2 (refinada por DHU-012):* Esta HU se documenta como Historia de Usuario completa y se considera candidata a entrar al sprint condicional a la holgura del cronograma tras cerrar las HUs MVP1. No es entregable comprometido del MVP1, pero tampoco está descartada a priori. Si el cronograma permite, se asignan sprints y puntos de historia normalmente.

Esta declaración estándar reemplaza la nota técnica de HU-09 ("fuera del sprint" pre-DHU-012); HU-09 conserva su nota técnica original tal cual está en `HU_BLOQUE_B.md`, dado que el contenido sustantivo de HU-09 no se reabre.

#### J. Aplicación de DHU-005 al MVP2 (robustez ante interrupción de fuente)

DHU-005 declara dos casos: Caso A (fuente externa de medición; ejemplo: detector de tráfico) y Caso B (componente interno de decisión; ejemplo: motor de cálculo, persistencia, modelo predictivo). El MVP2 no opera con fuentes externas de medición; todas sus HUs dependen de componentes internos.

**Decisión: las 4 HUs del MVP2 aplican DHU-005 Caso B con independencia por carril (HU-18) o por fuente (HU-19, HU-20, HU-21):**

- **HU-18**: Caso B aplicado **independientemente por carril** (CA-18.17, CA-18.18, CA-18.19). Las marcas de "no actualizado" son por carril (tráfico, motor, estado operativo), no por vista completa, porque las tres fuentes son independientes y un fallo de una no implica fallo de las otras. Esto contrasta deliberadamente con CA-17.14 de HU-17, donde la marca aplica simultáneamente a ambos periodos porque la causa raíz es la misma (motor de cálculo único de KPIs).
- **HU-19**: Caso B aplicado **con política conservadora** (CA-19.22). A diferencia de las vistas interactivas que muestran últimos valores conocidos marcados como "no actualizados", la exportación **rechaza** la generación cuando la fuente está caída. Justificación: un PDF o Excel descargado es un artefacto persistente que circula fuera del sistema; permitir generar reportes sobre datos no confirmados abriría la posibilidad de difundir datos "no actualizados" sin marca visible. La política conservadora protege la integridad del artefacto exportado.
- **HU-20**: Caso B aplicado al motor de cálculo de métricas comparativas (CA-20.18). Cuando el componente de cálculo no responde, se muestran las últimas métricas conocidas marcadas como "no actualizadas". Patrón análogo al de CA-14.12 de HU-14.
- **HU-21**: Caso B aplicado al disparo del escalamiento (CA-21.9; rechazo del inicio si los endpoints de CT-04.4 o CT-04.5 no responden) y al subsistema de consulta del registro (CA-21.28; últimos incidentes conocidos marcados como "no actualizados"). El rechazo en CA-21.9 es justificado por la integridad auditable del registro: escalar con campos automáticos vacíos comprometería el valor del registro.

### Decisión final

**MVP2: 4 HUs operativas nuevas + 0 TTH nuevas.**

| Feature | Modelado como | Identificador |
|---|---|---|
| F15 (Drill-down) | HU del Gerente con tres carriles temporales integrados | **HU-18** |
| F16 (Exportación PDF/Excel) | HU única del Gerente que cubre PDF/Excel × HU-16/HU-17 | **HU-19** |
| F19 (Comparativa vs baseline) | HU del Administrador con extensión inglobada de CA-14.1 | **HU-20** |
| F28 (Escalamiento) | HU única con Operador protagonista y Administrador destinatario, con persistencia inglobada | **HU-21** |
| F11 (Notas e incidencias) | Ya redactada como HU-09 en `HU_BLOQUE_B.md` desde el cierre del Bloque B | HU-09 (referencia cruzada) |

**Total MVP2:** 5 HUs operativas (incluyendo HU-09 anticipada) + 0 TTH nuevas. **Total Product Backlog al cierre del MVP2:** 21 HUs operativas (HU-01 a HU-21) + 11 TTH (TTH-01 a TTH-11).

### Lo que NO cambia con DHU-017

- **Las decisiones DHU-001 a DHU-016 mantienen su contenido sustantivo.** DHU-017 las cita y aplica al MVP2 sin reabrir ninguna.
- **El alcance del producto** (Personas, Objetivos, Journeys, Visión) se mantiene intacto. Las features MVP2 ya estaban declaradas en `LEAN_INCEPTION_CEREBROVIAL.md` sección 9 (Sequencer); DHU-017 las redacta como HUs sin modificar la sección del Inception.
- **Las HUs MVP1 redactadas en bloques previos (HU-01 a HU-17)** no se reabren. La extensión del registro de predicciones inglobada en CA-20.1 a CA-20.4 de HU-20 no modifica HU-14 (que sigue mostrando solo el modelo principal en su vista individual).
- **Las TTH previas (TTH-01 a TTH-11)** mantienen su contenido. El MVP2 no introduce TTH nuevas, conforme a la subsección H.
- **Las decisiones técnicas D-001 a D-009** se mantienen sin modificación. El MVP2 las consume sin reabrirlas.
- **HU-09 dentro de `HU_BLOQUE_B.md`** mantiene su contenido: su clasificación MVP2 ya estaba declarada desde el cierre del Bloque B y suavizada por DHU-012.

### Documentos afectados por DHU-017

| Documento | Tipo de cambio |
|---|---|
| `HU_MVP2.md` (nuevo) | Documento nuevo con HU-18 (F15), HU-19 (F16), HU-20 (F19), HU-21 (F28). Mapeo a HU-09 que reside en `HU_BLOQUE_B.md`. Resumen del MVP2 y próximos pasos. |
| `DECISIONS_HU.md` (este documento) | Agregar DHU-017; actualizar índice, tabla de impacto en bloques y documentos relacionados. |
| `FEATURE_BACKLOG_DETALLADO.md` | Fichas de F15, F16, F19 y F28 actualizan su columna "Modelado" para apuntar a HU-18, HU-19, HU-20 y HU-21 respectivamente. Referencia a DHU-017. |
| `HU_BLOQUE_A.md` a `HU_BLOQUE_F.md` | Próximos pasos actualizados: MVP2 ya cerrado; restan documento RF/RNF, Planning Poker, MoSCoW, e implementación SCRUM. Documentos relacionados actualizados con `HU_MVP2.md`. |
| `LEAN_INCEPTION_CEREBROVIAL.md` | Documentos relacionados actualizado (referencia a `HU_MVP2.md`). |
| `TAREAS_TECNICAS_HABILITADORAS.md` | Rango DHU referenciado actualizado: "DHU-001 a DHU-016" → "DHU-001 a DHU-017". Sin cambios sustantivos al contenido de las TTH. |

### Documentos relacionados

- `HU_MVP2.md` — MVP2 del Product Backlog (HU-18, HU-19, HU-20, HU-21). HU-09 mantenida en `HU_BLOQUE_B.md`.
- `DECISIONS_HU.md` (este documento) — sección DHU-017.
- `LEAN_INCEPTION_CEREBROVIAL.md` — sección 9 (Sequencer / MVP2), Persona Gerente, Persona Administrador, Persona Operador.
- `FEATURE_BACKLOG_DETALLADO.md` — fichas F11, F15, F16, F19, F28.
- `HU_BLOQUE_B.md` — HU-09 (anticipada al cierre del Bloque B).
- `HU_BLOQUE_F.md` — HU-16 y HU-17 (vistas agregadas del Gerente desde las cuales se accede al drill-down y a la exportación).
- `HU_BLOQUE_D.md` — HU-14 (vista de métricas del modelo principal, ampliada por HU-20 a comparativa con baseline).
- `HU_BLOQUE_C.md` — HU-10, HU-12 (vistas desde las cuales se dispara el escalamiento de HU-21).
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-04/CT-04.4 y CT-04.5 (consumidas por HU-21), TTH-09/CT-09.5 (consumida y extendida por HU-20).

---

## DHU-018 — Patrón "Resumen ejecutivo" agregado retroactivamente al inicio de cada HU del Product Backlog

**Fecha:** 2026-05-17.
**Estado:** Cerrada.
**Aplica a:** Las 21 HUs operativas del Product Backlog (HU-01 a HU-21).

### Contexto

Durante la redacción incremental del Product Backlog entre el 11 y el 16 de mayo de 2026 se observó que las HUs crecieron en densidad a medida que el equipo refinó el patrón de redacción. Las HUs iniciales del Bloque A y Bloque B se mantuvieron en torno a 5-7 CAs, sin subdivisión visual; las HUs del Bloque C empezaron a incluir notas técnicas extensas; las del Bloque D, F y MVP2 alcanzaron 15-33 CAs cada una con subsecciones `####` temáticas (Sustrato técnico, Presentación, Casos degenerados, Control de acceso).

Esta densidad creciente es legítima y deseable desde la perspectiva de **trazabilidad para implementación y para sustentación académica**: cada CA es referenciable por código, las decisiones DHU están entrelazadas, y los enlaces entre HUs son explícitos. Sin embargo, introduce dos costos:

1. **Lectura humana:** un lector que solo quiere entender qué hace una HU necesita leer entre 1 y 3 páginas por HU.
2. **Asistencia por agente de IA:** un agente con contexto limitado debe leer la HU completa para identificar los CAs ancla, las dependencias y las decisiones que orientan la lectura.

El backlog ya tiene un documento de lectura ligera (`HU_LITE.md`) con las 21 HUs en formato corto, pero esa solución vive en un documento separado y rompe la propiedad de "una HU es una unidad autocontenida". El refinamiento natural es agregar un bloque corto al inicio de cada HU densa que sirva como punto de entrada sin perder el detalle implementable.

### Análisis

Tres opciones evaluadas durante la sesión:

| Opción | Descripción | Decisión |
|---|---|---|
| A1 | Aplicar el patrón solo a HUs con ≥10 CAs (las "densas"). | Descartada: rompe uniformidad del backlog y mezcla HUs con y sin Resumen, inconsistente con la lectura por terceros. |
| A2 | Aplicar a HUs MVP2 + HUs densas MVP1. | Descartada: misma razón que A1, mantiene asimetría arbitraria entre bloques. |
| A3 | Aplicar a las 21 HUs uniformemente. | **Adoptada.** Costo marginal bajo, uniformidad total, lectura por terceros se beneficia consistentemente. |

### Decisión

Las 21 HUs del Product Backlog (HU-01 a HU-21) reciben retroactivamente un bloque **"Resumen ejecutivo"** insertado entre la cabecera (Tipo + Feature(s) origen) y la sección "Descripción" existente. El bloque contiene los siguientes campos en este orden:

1. **Qué entrega:** una frase que resume el valor entregable de la HU, sin jerga interna del backlog.
2. **CAs críticos:** los 3-4 CAs sin los cuales la HU no se sostiene. Típicamente: el CA principal de comportamiento, el CA de robustez (DHU-005 Caso A o B), el CA de presentación al usuario, y el CA de control de acceso.
3. **Estructura de CAs:** mapa de las subsecciones temáticas con rangos numéricos de CAs, para navegación rápida. Este campo se omite en HUs con menos de 8 CAs sin subdivisión visual, donde no aporta valor de navegación.
4. **Dependencias:** otras HUs, TTH o DHUs que esta HU consume, extiende o que la consumen.
5. **Notas clave:** 2-3 decisiones de diseño explícitas que orientan la lectura del resto de la HU (típicamente referencias a subsecciones de DHUs aplicables).

El bloque tiene una extensión típica de 80-150 palabras, presentado como cinco párrafos cortos con el campo en negrita seguido del contenido.

### Por qué no es contenido sustantivo nuevo

El "Resumen ejecutivo" es **redacción derivada de los CAs y notas técnicas existentes**, no contenido sustantivo nuevo. Para cada HU, el resumen se construye sintetizando lo que ya está declarado más abajo en la misma HU:

- "Qué entrega" sintetiza la cabecera Como/Quiero/Para con la sección Descripción.
- "CAs críticos" identifica entre los CAs existentes los que tienen impacto bloqueante o de robustez.
- "Estructura de CAs" enumera los headers `####` que ya estaban presentes (o las subsecciones implícitas en HUs sin subdivisión visual).
- "Dependencias" recupera referencias cruzadas declaradas en notas técnicas existentes.
- "Notas clave" cita decisiones DHU ya declaradas en notas técnicas o en la sección de "Decisiones que aplicaron a este bloque" del documento contenedor.

Por construcción, **ningún CA, ninguna nota técnica, ningún Candidato a RNF, ninguna clasificación MVP, ninguna feature de origen, ningún tipo, ninguna sección de Notas técnicas y ninguna sección de Candidatos a RNF se modifica con DHU-018**. La aplicación de esta decisión es estrictamente aditiva.

### Adaptación para HUs cortas

HU-01 (6 CAs sin subdivisión visual), HU-02 (5 CAs), HU-03 (5 CAs), HU-04 (5 CAs), HU-05 (5 CAs), HU-06 (5 CAs), HU-07 (6 CAs) y HU-12 (6 CAs) reciben Resumen ejecutivo con el campo "Estructura de CAs" omitido, dado que sin subdivisión `####` la enumeración de rangos no aporta valor de navegación. Los otros cuatro campos se aplican uniformemente.

HU-08 (6 CAs), HU-09 (6 CAs) y HU-11 (9 CAs) tienen subdivisión implícita por temática (CAs de presentación + CAs de robustez + CA de control de acceso) y reciben "Estructura de CAs" con la subdivisión declarada explícitamente en el Resumen, aunque la HU misma no use headers `####`.

### Consecuencias

- Las 21 HUs (HU-01 a HU-21) reciben el bloque "Resumen ejecutivo" en una pasada de redacción dedicada.
- HU-16 fue redactada primero como muestra del patrón antes de aprobar A3; su Resumen ejecutivo queda como está, sin reabrir.
- Los preámbulos de los seis documentos del backlog (`HU_BLOQUE_A.md` a `HU_BLOQUE_F.md`) y `HU_MVP2.md` reciben una mención breve al patrón en la sección "Contexto", referenciando esta DHU.
- El rango DHU referenciado en "Documentos relacionados" de los documentos del backlog se actualiza de "DHU-001 a DHU-017" a "DHU-001 a DHU-018".
- `HU_LITE.md` no se modifica: sigue siendo la versión corta paralela; el Resumen ejecutivo dentro de cada HU densa cumple una función diferente (orientar la lectura de la HU completa, no sustituirla).
- `BACKLOG_OVERVIEW.md` no requiere cambio sustantivo; opcionalmente se puede agregar una mención en la sección "Cómo navegar el backlog".

### Lo que NO cambia con DHU-018

- **El contenido sustantivo de las 21 HUs.** Ningún CA, ninguna nota técnica, ningún Candidato a RNF, ninguna clasificación MVP, ninguna feature de origen, ningún tipo se modifica.
- **El contenido sustantivo de las 11 TTH.** Las TTH no reciben Resumen ejecutivo: son entregables de naturaleza distinta (criterios técnicos de terminado, no CAs Given-When-Then), con audiencia distinta (equipo de desarrollo), y ya tienen una sección "Descripción" autocontenida que sirve la misma función.
- **El alcance del producto** (Personas, Objetivos, Journeys, Visión) se mantiene intacto.
- **Las decisiones DHU-001 a DHU-017 mantienen su contenido sustantivo.** DHU-018 las cita y las usa como insumo del Resumen ejecutivo, sin reabrir ninguna.
- **Las decisiones técnicas D-001 a D-009** se mantienen sin modificación.

### Documentos afectados por DHU-018

| Documento | Tipo de cambio |
|---|---|
| `DECISIONS_HU.md` (este documento) | Agregar DHU-018 al índice y cuerpo; actualizar fecha de última actualización; actualizar tabla de "Resumen de impacto en bloques redactados" con fila Transversal-DHU-018. |
| `HU_BLOQUE_A.md` | Resumen ejecutivo en HU-01 (campo "Estructura de CAs" omitido por HU corta sin subdivisión). Nota corta en sección "Contexto" referenciando DHU-018. Rango DHU referenciado actualizado a "DHU-001 a DHU-018". |
| `HU_BLOQUE_B.md` | Resumen ejecutivo en HU-02 a HU-09 (Estructura de CAs según corresponda). Nota corta en "Contexto". Rango DHU actualizado. |
| `HU_BLOQUE_C.md` | Resumen ejecutivo en HU-10, HU-11, HU-12. Nota corta en "Contexto". Rango DHU actualizado. |
| `HU_BLOQUE_D.md` | Resumen ejecutivo en HU-13, HU-14, HU-15. Nota corta en "Contexto". Rango DHU actualizado. |
| `HU_BLOQUE_E.md` | No requiere cambio (0 HUs operativas; solo TTH). Rango DHU actualizado en "Documentos relacionados" por coherencia con otros bloques. |
| `HU_BLOQUE_F.md` | Resumen ejecutivo en HU-16 (ya aplicado como muestra antes de DHU-018; queda como está) y HU-17. Nota corta en "Contexto". Rango DHU actualizado. |
| `HU_MVP2.md` | Resumen ejecutivo en HU-18, HU-19, HU-20, HU-21 (HU-09 vive físicamente en `HU_BLOQUE_B.md` y recibe Resumen ahí). Nota corta en "Contexto". Rango DHU actualizado. |
| `LEAN_INCEPTION_CEREBROVIAL.md` | Sin cambio sustantivo. Si se hace pase de higiene cruzada futura, el rango DHU referenciado en "Documentos relacionados" se actualiza por coherencia. |
| `TAREAS_TECNICAS_HABILITADORAS.md` | Rango DHU referenciado actualizado en cabecera y "Documentos relacionados". Sin cambios sustantivos al contenido de las 11 TTH. |
| `BACKLOG_OVERVIEW.md` | Opcional: mención del patrón en "Cómo navegar el backlog". |

### Documentos relacionados

- `HU_BLOQUE_A.md` a `HU_BLOQUE_F.md`, `HU_MVP2.md` — destinatarios del cambio (20 HUs).
- `HU_BLOQUE_F.md` — HU-16 redactada con el patrón antes de aprobar A3 (muestra).
- `HU_LITE.md` — documento de lectura corta paralela; función distinta al Resumen ejecutivo interno de cada HU.
- `DECISIONS_HU.md` (este documento) — sección DHU-018.

---

## DHU-019 — Decisiones metodológicas para la redacción del documento de Requisitos Funcionales y No Funcionales (RF/RNF)

**Fecha:** 2026-05-18.
**Estado:** Cerrada.
**Aplica a:** Documento `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` (nuevo entregable del proyecto).

### Contexto

DHU-007 (2026-05-13) estableció que cada HU del Product Backlog incluyera al final una sección **"Candidatos a RNF"** para preservar la trazabilidad futura de los requisitos no funcionales sin frenar la redacción inicial de las HUs. La línea 353 de DHU-007 declaró explícitamente como trabajo futuro asociado la redacción del documento `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` que consolidaría esos candidatos en un documento único, numerado y aprobado, con umbrales formales que reemplazarían los hardcodeados en los CAs por referencias `RNF-XXX-NN`.

Con el cierre del MVP2 (DHU-017) y la aplicación retroactiva del patrón "Resumen ejecutivo" (DHU-018), el componente funcional del Product Backlog quedó completo: 21 HUs operativas (HU-01 a HU-21) y 11 TTH (TTH-01 a TTH-11). Las 21 HUs incluyen su sección "Candidatos a RNF" y las 11 TTH incluyen criterios técnicos de terminado cuya naturaleza es híbrida entre funcional y de calidad. Es el momento operacional de ejecutar la sesión dedicada que DHU-007 declaró pendiente.

Antes de iniciar la redacción del documento RF/RNF era necesario cerrar un conjunto de decisiones metodológicas que la sola aplicación de DHU-007 no determina por sí misma: qué taxonomía de calidad usar y en qué versión, cómo reasignar las categorías heterogéneas declaradas en DHU-007 a la taxonomía elegida, qué plantilla unificada usar para cada RF y cada RNF, cómo derivar los RF desde los CAs existentes (composición transversal vs biyección), cómo resolver las inconsistencias detectadas durante la lectura de los Candidatos a RNF de las 21 HUs, y qué política seguir para el reemplazo retroactivo de umbrales hardcoded en las HUs por referencias al documento formal. DHU-019 consolida esas decisiones en un acto único, reproduciendo el patrón de decisiones consolidadas establecido por DHU-014 (Bloque D), DHU-016 (Bloque F) y DHU-017 (MVP2).

### Decisiones consolidadas

#### A. Elección de ISO/IEC 25010:2023 como taxonomía de RNF

DHU-007 línea 349 mencionaba ISO/IEC 25010 como referencia pero listaba categorías heterogéneas que no calzan literalmente con ninguna versión publicada del estándar ("robustez, configurabilidad, persistencia, auditoría, retención, trazabilidad, inmutabilidad" no son características formales de ISO 25010). Para el documento RF/RNF se requiere fijar versión exacta y vocabulario normativo.

Tres alternativas se contrastaron:

| Alternativa | Descripción | Decisión |
|---|---|---|
| ISO/IEC 9126 | Estándar predecesor, con 6 características (Functionality, Reliability, Usability, Efficiency, Maintainability, Portability). Sustituido oficialmente por ISO 25010 en 2011. | Descartada: obsoleta desde 2011, no recomendable en defensa académica de 2026. |
| ISO/IEC 25010:2011 | Versión más conocida y citada. 8 características (Functional Suitability, Performance Efficiency, Compatibility, Usability, Reliability, Security, Maintainability, Portability). | Descartada: superseded por la versión 2023 desde el 2023-11-01. |
| ISO/IEC 25010:2023 | Versión vigente. 9 características: las 8 de 2011 con renombramientos (Usability → **Interaction Capability**, Portability → **Flexibility**) más la adición de **Safety** como 9ª característica. Subcaracterísticas refinadas y ampliadas. | **Adoptada.** |

**Decisión: el documento RF/RNF clasifica los RNF según ISO/IEC 25010:2023** (la versión vigente al momento de redacción, mayo 2026).

**Justificación detallada:**

1. **Versión vigente al momento de la defensa académica.** La defensa de la tesis se realiza en o después de mayo de 2026. ISO/IEC 25010:2023 está vigente desde 2023-11-01. Usar la versión 2011 obligaría a justificar ante el jurado por qué se elige una versión superseded existiendo la vigente.

2. **Inclusión de Safety como característica explícita.** ISO 25010:2023 introduce Safety como 9ª característica de calidad, ausente en 2011. CerebroVial es un sistema de control de infraestructura urbana de tránsito vehicular: su comportamiento bajo fallo (degradado nivel 3 con tiempos preconfigurados, TTH-04 + TTH-05) y la inmutabilidad de los registros de auditoría (HU-08, HU-10, HU-14, HU-15, HU-16, HU-20, HU-21) son cuestiones de seguridad operacional con implicaciones tangibles sobre conductores y peatones. Tener Safety como característica de primer nivel permite agrupar estos RNF con vocabulario formal en lugar de dispersarlos entre Reliability y Security.

3. **Renombramientos terminológicos más cercanos al lenguaje del Product Backlog.** "Interaction Capability" comunica mejor que "Usability" la naturaleza del Operador trabajando con el sistema bajo presión de tiempo real (cubre appropriateness recognizability, learnability, operability, user error protection, UI aesthetics, accessibility). "Flexibility" cubre el espectro de adaptabilidad, escalabilidad, instalabilidad y reemplazabilidad que el backlog ya anticipa en notas técnicas (escalamiento futuro a multi-intersección documentado en notas de TTH-09, portabilidad del deploy Docker documentada en D-003).

4. **Subcaracterísticas más granulares.** ISO 25010:2023 refina algunas subcaracterísticas que el backlog ya distingue implícitamente. Por ejemplo, Reliability incluye **Faultlessness** (ausencia de fallos bajo operación normal), **Fault Tolerance** (operación correcta ante fallos de componentes — núcleo del Bloque C), **Availability** (proporción de tiempo disponible), **Recoverability** (capacidad de restaurar tras fallo). El backlog distingue las cuatro implícitamente; la versión 2023 las hace explícitas.

**Las 9 características adoptadas con su código RNF correspondiente:**

| # | Característica ISO 25010:2023 | Código RNF | Subcaracterísticas principales |
|---|---|---|---|
| 1 | Functional Suitability | **RNF-FUN** | Functional completeness, functional correctness, functional appropriateness |
| 2 | Performance Efficiency | **RNF-PERF** | Time behaviour, resource utilization, capacity |
| 3 | Compatibility | **RNF-COM** | Co-existence, interoperability |
| 4 | Interaction Capability | **RNF-INT** | Appropriateness recognizability, learnability, operability, user error protection, UI aesthetics, accessibility, self-descriptiveness |
| 5 | Reliability | **RNF-REL** | Faultlessness, availability, fault tolerance, recoverability |
| 6 | Security | **RNF-SEC** | Confidentiality, integrity, non-repudiation, accountability, authenticity, resistance |
| 7 | Maintainability | **RNF-MNT** | Modularity, reusability, analysability, modifiability, testability |
| 8 | Flexibility | **RNF-FLX** | Adaptability, scalability, installability, replaceability |
| 9 | Safety | **RNF-SAF** | Operational constraint, risk identification, fail safe, hazard warning, safe integration |

**Nota terminológica importante: Functional Suitability (RNF-FUN) no es sinónimo de Requisito Funcional (RF).**

La palabra "funcional" aparece en ambos términos y eso induce confusión común. La distinción es real y se preserva en este documento:

- **RF (Requisito Funcional)** responde a la pregunta *¿qué hace el sistema?*: declara comportamientos y servicios. Ejemplo: *"El sistema muestra el flujo vehicular y la longitud de cola por cada acceso de la intersección, con actualización automática"*.

- **RNF-FUN (Functional Suitability)** responde a la pregunta *¿con qué calidad lo hace?*: evalúa la corrección, completitud y apropiación del conjunto de funciones que el sistema ofrece. Es una característica de calidad de ISO 25010:2023, no una categoría de comportamientos del sistema. Ejemplo: *"Cuando el periodo seleccionado no contiene datos persistidos, el sistema comunica 'sin datos' en lugar de presentar KPIs calculados sobre cero filas (Functional Correctness)"*.

Las **tres subcaracterísticas de Functional Suitability** clarifican la diferencia: Functional Completeness (¿el catálogo de funciones cubre las tareas y objetivos del usuario?), Functional Correctness (¿los resultados producidos son correctos con la precisión necesaria, incluso en casos límite?) y Functional Appropriateness (¿las funciones facilitan la tarea sin pasos innecesarios?). Las tres son evaluaciones de calidad sobre el catálogo de RFs; no son RFs adicionales.

El código `RNF-FUN-NN` se conserva (en lugar de alternativas como `RNF-FSU-NN`) por trazabilidad léxica directa al estándar ISO 25010:2023. El prefijo `RNF-` distingue inequívocamente de `RF-NNN` en cualquier contexto donde ambos aparezcan. El preámbulo del documento `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` incluye una sección dedicada a esta distinción como orientación al lector (ver subsección H).

#### B. Reasignación de las categorías heterogéneas de DHU-007 a ISO 25010:2023

DHU-007 listó como "categorías típicas" un conjunto heterogéneo donde algunas son características ISO 25010 (rendimiento, usabilidad, seguridad, mantenibilidad, portabilidad), algunas son subcaracterísticas (escalabilidad, configurabilidad), y algunas son conceptos derivados sin un lugar canónico en ISO 25010 (robustez, persistencia, auditoría, retención, trazabilidad, inmutabilidad). La heterogeneidad fue funcional durante la redacción de las HUs (capturó lo que el redactor identificaba como RNF en cada caso); ahora debe normalizarse.

**Decisión: las categorías de DHU-007 se reasignan a ISO 25010:2023 conforme a la siguiente tabla. Esta reasignación es regla del proyecto, no caso a caso por RNF.**

| Categoría declarada en HUs (DHU-007) | Característica ISO 25010:2023 destino | Subcaracterística específica | Código |
|---|---|---|---|
| Rendimiento | Performance Efficiency | Time behaviour | RNF-PERF |
| Robustez | Reliability | Fault tolerance | RNF-REL |
| Disponibilidad | Reliability | Availability | RNF-REL |
| Continuidad operativa | Reliability | Fault tolerance + Recoverability | RNF-REL |
| Persistencia / durabilidad | Reliability | Recoverability (recuperación de datos) + Safety / Fail safe (cuando aplica a seguridad operacional) | RNF-REL / RNF-SAF |
| Retención | Reliability | Recoverability + Functional Suitability / Completeness | RNF-REL |
| Auditoría / auditabilidad | Security | Accountability + Integrity | RNF-SEC |
| Inmutabilidad / inmutabilidad parcial | Security | Integrity | RNF-SEC |
| Trazabilidad | Security | Accountability + Non-repudiation | RNF-SEC |
| Privacidad | Security | Confidentiality | RNF-SEC |
| Seguridad (acceso, RBAC) | Security | Authenticity + Resistance | RNF-SEC |
| Usabilidad | Interaction Capability | Appropriateness recognizability + Operability + UI aesthetics | RNF-INT |
| Accesibilidad | Interaction Capability | Accessibility | RNF-INT |
| Coherencia / consistencia entre vistas | Interaction Capability + Maintainability | Self-descriptiveness + Modifiability | RNF-INT |
| Mantenibilidad | Maintainability | Modifiability + Analysability | RNF-MNT |
| Configurabilidad | Maintainability | Modifiability | RNF-MNT |
| Manejabilidad de datos faltantes | Functional Suitability | Functional correctness | RNF-FUN |
| Manejabilidad de concurrencia | Functional Suitability + Reliability | Functional correctness + Fault tolerance | RNF-FUN / RNF-REL |
| Validación dual (frontend + backend) | Security + Maintainability | Integrity + Testability | RNF-SEC |
| Tolerancia a fallos del componente de generación | Reliability | Fault tolerance | RNF-REL |
| Cobertura (de catálogos de plantillas) | Functional Suitability | Functional completeness | RNF-FUN |
| Calidad de predicción (HU-03) | Functional Suitability + Performance Efficiency | Functional correctness + Time behaviour | RNF-FUN |
| Comparabilidad rigurosa (HU-20) | Functional Suitability | Functional correctness | RNF-FUN |
| Tolerancia parametrizada (HU-20) | Maintainability | Modifiability | RNF-MNT |
| No persistencia de reportes (HU-19) | Security | Confidentiality (no exposición de datos generados) | RNF-SEC |
| Identificabilidad del archivo (HU-19) | Interaction Capability | Self-descriptiveness | RNF-INT |
| Paralelización del cálculo (HU-18) | Performance Efficiency | Resource utilization | RNF-PERF |
| Independencia entre dimensiones (HU-21) | Functional Suitability | Functional correctness | RNF-FUN |
| Coherencia textual (catálogos de plantillas, HU-12) | Interaction Capability + Maintainability | Self-descriptiveness + Modifiability | RNF-INT |
| Separación de roles (HU-11/HU-13) | Security | Confidentiality + Accountability | RNF-SEC |
| Resiliencia de persistencia (HU-09, HU-21) | Reliability | Fault tolerance + Recoverability | RNF-REL |
| Seguridad operativa (valores por defecto, HU-15) | Safety | Fail safe | RNF-SAF |
| Granularidad de persistencia (HU-16) | Performance Efficiency | Capacity | RNF-PERF |
| Presentación visual (HU-19, impresión) | Interaction Capability | UI aesthetics + Accessibility | RNF-INT |

**Justificación de los cruces de característica (filas con más de una característica destino):**

- *Persistencia / durabilidad*: cuando se trata de "no perder datos por fallo de escritura", la naturaleza es Recoverability (Reliability). Cuando se trata de los tiempos preconfigurados del degradado nivel 3 que mantienen la intersección operativa ante falla del motor (TTH-05 + CT-04.6), la naturaleza es **Fail safe (Safety)**, porque el sistema falla hacia un estado seguro definido. Ambos códigos coexisten según el RNF concreto.

- *Auditoría / inmutabilidad / trazabilidad*: ISO 25010:2023 ubica estos conceptos dentro de Security (Accountability, Integrity, Non-repudiation). No están en Reliability ni en una característica propia. La decisión queda alineada con la taxonomía formal.

- *Manejabilidad de datos faltantes*: la decisión de comunicar "sin datos" en lugar de calcular sobre cero es una propiedad de Functional Correctness (Functional Suitability), no de Reliability. El sistema produce resultados correctos incluso ante datos ausentes; la corrección incluye no fabricar valores.

- *Manejabilidad de concurrencia (CA-15.11)*: el comportamiento last-write-wins con advertencia explícita es simultáneamente Functional Correctness (Functional Suitability, porque el resultado de cada modificación es correcto y predecible) y Fault Tolerance (Reliability, porque el sistema no pierde modificaciones silenciosamente). Se documenta con código primario RNF-FUN y referencia cruzada a RNF-REL.

- *Validación dual frontend + backend*: la propiedad es simultáneamente Integrity (Security, porque garantiza que las restricciones no se pueden bypassear) y Testability (Maintainability, porque la validación dual se valida con tests separados). Se documenta con código primario RNF-SEC.

- *Coherencia textual de catálogos de plantillas*: la propiedad es simultáneamente Self-descriptiveness (Interaction Capability, porque los textos son comprensibles y consistentes para el lector) y Modifiability (Maintainability, porque el catálogo se extiende sin tocar código). Se documenta con código primario RNF-INT.

**Notación de RNF con doble código:** cuando un RNF aplica simultáneamente a dos características, el documento usa el código primario (el más relevante para el RNF concreto) en el identificador `RNF-XXX-NN`, y declara la característica secundaria como referencia cruzada en el campo "Característica ISO" del RNF. Esto evita duplicar el RNF en dos secciones del documento.

#### C. Resolución de las inconsistencias detectadas durante la lectura de los Candidatos a RNF

Durante la lectura de los Candidatos a RNF de las 21 HUs se identificaron siete inconsistencias. DHU-019 las resuelve explícitamente como decisión metodológica de redacción del documento RF/RNF.

**C.1 — Consolidación del umbral "≤ 5 segundos" como un único RNF transversal de tiempo real.**

Diez HUs declaran latencia ≤ 5 s entre evento y actualización visible (HU-02 CA-02.2, HU-03 CA-03.2, HU-04 CA-04.3, HU-05 CA-05.3, HU-06 CA-06.2, HU-07 CA-07.1 implícito, HU-10 banner, HU-11 CA-11.2, HU-12 CA-12.2, HU-13 CA-13.2). El umbral es semánticamente uno solo: "actualización en tiempo real de la presentación visible al usuario, desde que el evento generador ocurre hasta que el dato actualizado aparece en pantalla".

**Decisión:** se consolida como un único **RNF-PERF de actualización de tiempo real** con umbral ≤ 5 segundos, aplicable a las diez HUs identificadas. La referencia desde cada CA al documento RF/RNF reemplaza la repetición del umbral; los CAs siguen redactados con su semántica de Given-When-Then pero el umbral pasa a referencia.

Los RNF de tiempo de respuesta para vistas no de tiempo real se documentan como RNFs adicionales con umbrales distintos: apertura ≤ 2 s (HU-15), apertura ≤ 3 s (HU-16, HU-17), apertura ≤ 5 s drill-down corto y ≤ 15 s drill-down largo (HU-18), generación ≤ 10-15 s para Excel y ≤ 15-60 s para PDF (HU-19), recálculo ≤ 10 s comparativo (HU-16, HU-17), latencia del badge ≤ 30 s (HU-21), latencia del cálculo de métricas ≤ 30 s (HU-14, HU-20).

**C.2 — Consolidación de los RNF de robustez ante interrupción como un único RNF transversal con dos modos (DHU-005 Caso A y Caso B).**

Veintiún declaraciones de "RNF de robustez" repiten estructuralmente el patrón de DHU-005: el sistema mantiene el último valor conocido, lo marca como "desactualizado" (Caso A: fuente externa de medición) o "no confirmado" (Caso B: componente interno de decisión), e indica el tiempo transcurrido desde la última actualización confirmada.

**Decisión:** se consolida como un único **RNF-REL transversal de robustez ante interrupción de fuente**, con dos modos formalmente declarados (A y B), y una tabla de aplicabilidad que enumera las HUs y CAs específicos donde cada modo aplica. La política de excepción para HU-19 (rechazo de generación en lugar de marca pasiva, por integridad del artefacto exportado) queda declarada como excepción explícita del RNF.

**C.3 — Reformulación del RNF de seguridad de HU-13 que mezcla normativa con descriptivo.**

El "RNF de seguridad" de HU-13 declara: *"Los campos viajan en el wire incluso para consumidores con otros roles, pero esos consumidores no tienen acceso a la ruta que los renderiza"*. La primera mitad es descripción de implementación; la segunda mitad es la propiedad normativa. La construcción mezcla niveles.

**Decisión:** el RNF correspondiente en el documento se redacta normativamente: *"El control de acceso por rol se aplica a nivel de ruta del backend; los campos no sensibles pueden estar presentes en payloads compartidos entre roles sin violar la confidencialidad porque el RBAC impide la materialización de las rutas que los renderizan"*. La afirmación descriptiva sobre el wire vive como nota técnica del RNF, no como cuerpo normativo. El RNF queda clasificado como RNF-SEC (Security / Confidentiality + Authenticity), con referencia cruzada a la decisión arquitectónica de DHU-014 subsección G (patrón "un endpoint, dos vistas con filtrado en presentación" y TTH-06 como Trabajos Futuros).

**C.4 — Reubicación del RNF de calidad de predicción de HU-03 al ámbito del componente predictivo.**

El RNF de HU-03 reconoce explícitamente que pertenece al Bloque E (TTH-09), no a la vista. El documento RF/RNF lo redacta así.

**Decisión:** el RNF correspondiente se redacta como **RNF-FUN de calidad del modelo predictivo**, con origen híbrido HU-03 (donde se identificó) y TTH-09 (donde reside operacionalmente: CT-09.7 con objetivo aspiracional accuracy ≥ 80% sobre el nivel discreto 0-5 evaluado sobre la partición de validación). La sección de Functional Suitability del documento RF/RNF agrupa este RNF con los demás RNF derivados de TTH-09. La referencia cruzada desde HU-03 al RNF queda preservada para sustentar la cadena "vista de predicción consume el modelo cuya calidad es el RNF correspondiente".

**C.5 — Unificación del RNF de inmutabilidad de logs como principio transversal único.**

Ocho HUs declaran cada una su propio RNF de "auditabilidad", "inmutabilidad" o "auditoría no modificable" (HU-08, HU-09, HU-10, HU-14, HU-15, HU-16, HU-20, HU-21). El principio subyacente es uno solo: *"los registros append-only del sistema son inmutables tras la escritura, preservando la confiabilidad de la consulta retroactiva y la auditabilidad del flujo"*.

**Decisión:** se consolida como un único **RNF-SEC transversal de inmutabilidad de logs append-only**, con tabla de aplicabilidad que enumera los registros afectados: registro de decisiones del motor (CT-10.9 / CA-08.1), registro de notas del Operador (HU-09 con inmutabilidad parcial por ventana de edición de CA-09.4), registro de transiciones de estado operativo (CT-04.3 / CA-10.7), registro de predicciones (CT-09.5 / CA-14.1), registro de auditoría de parámetros (CA-15.4), histórico de estados (CA-16.1), registro de predicciones del modelo de respaldo (CA-20.2 inglobada como extensión de CT-09.5), registro de incidentes escalados (CA-21.10 inglobada). Cada uno con su ventana de edición acotada cuando aplica (HU-09).

**C.6 — Aceptación formal de la reasignación masiva de categorías DHU-007 a ISO 25010:2023.**

La tabla de la subsección B ya cierra esta inconsistencia. DHU-019 declara que **ninguna HU se modifica retroactivamente para sustituir su vocabulario de Candidatos a RNF**; las HUs preservan su redacción original con vocabulario heterogéneo, y el documento RF/RNF aplica el vocabulario ISO 25010:2023 al consolidar. La tabla de la subsección B sirve como diccionario de traducción de un vocabulario al otro.

**C.7 — Clasificación de "Manejabilidad de datos faltantes" en Functional Correctness.**

Ocho HUs declaran RNF de "manejabilidad de datos faltantes" sin que el término tenga lugar canónico en ISO 25010. La propiedad descrita en todos los casos es: *"el sistema produce resultados correctos comunicando explícitamente la ausencia de datos en lugar de calcular sobre vacío"*.

**Decisión:** se clasifica como **RNF-FUN / Functional Correctness** (Functional Suitability). Esta es decisión cerrada de la subsección B y se documenta aquí solo por explicitud. La subcaracterística Functional Correctness cubre que el sistema produce los resultados correctos con el grado de precisión necesario; un sistema que calcula KPIs sobre cero filas y los presenta como "indicadores válidos en cero" no cumple Functional Correctness, mientras que un sistema que comunica "no hay datos en el periodo seleccionado" sí la cumple.

#### D. Plantilla unificada de RF y de RNF

**Plantilla de RF (Requisito Funcional):**

```markdown
### RF-NNN — [Título descriptivo]

| Campo | Contenido |
|---|---|
| Identificador | RF-NNN |
| Familia funcional | [una de las 7 familias declaradas en la subsección E] |
| Descripción | [una o dos frases describiendo qué hace el sistema; sin detalles de presentación ni de implementación] |
| HUs origen | HU-XX, HU-YY (lista de HUs cuyos CAs alimentan este RF) |
| CAs origen | CA-XX.N, CA-YY.M (referencias específicas a los CAs que materializan el comportamiento) |
| TTH relacionadas | TTH-XX (si aplica; típicamente cuando el RF tiene sustrato técnico declarado en una TTH) |
| Persona beneficiaria | Operador / Gerente / Administrador / Transversal a las tres |
| Objetivo del producto | 1 (reducir tiempos) / 2 (sustento técnico) / 3 (continuidad) / 4 (evidencia gerencial) / Soporte transversal |
| Prioridad MoSCoW sugerida | Must / Should / Could / Won't (sugerida en este documento; se refina en sesión dedicada de priorización MoSCoW posterior) |
| RNF asociados | RNF-XXX-NN, RNF-YYY-MM (referencias a los RNF que aplican a este RF) |
| Notas | (excepciones, dependencias entre RFs, casos límite documentados en notas técnicas de las HUs origen) |
```

**Plantilla de RNF (Requisito No Funcional):**

```markdown
### RNF-XXX-NN — [Título descriptivo]

| Campo | Contenido |
|---|---|
| Identificador | RNF-XXX-NN (XXX = código de característica de 3 letras; NN = correlativo dentro de la característica) |
| Característica ISO 25010:2023 | [una de las 9 declaradas en la subsección A; cuando aplica doble característica, se declara la primaria + secundaria con referencia cruzada conforme a la subsección B] |
| Subcaracterística | [subcaracterística específica dentro de la característica] |
| Descripción normativa | [una a tres oraciones declarando qué propiedad de calidad debe cumplir el sistema, redactadas normativamente ("el sistema debe...") sin descripciones de implementación] |
| Criterio de aceptación medible | [umbral numérico, condición binaria, o método de validación documentado; cuando no se puede cuantificar (típicamente Interaction Capability) se declara método de validación cualitativa: prueba de usuario, inspección, revisión por experto] |
| Método de validación | Prueba automatizada / Prueba de usuario / Inspección / Análisis estático / Revisión documental |
| HUs/TTH origen | HU-XX (CA-XX.N), TTH-YY (CT-YY.M) (con referencias específicas a los Candidatos a RNF o criterios técnicos que alimentan este RNF) |
| DHUs relacionadas | DHU-XXX (cuando una decisión metodológica orienta este RNF; típicamente DHU-005 para robustez, DHU-007 para origen, DHU-019 para clasificación) |
| Prioridad MoSCoW sugerida | Must / Should / Could / Won't |
| Aplicabilidad | (cuando el RNF es transversal: tabla o lista de HUs/TTH a las que aplica) |
| Excepciones | (cuando aplica: casos donde el RNF tiene comportamiento distinto, por ejemplo política conservadora de HU-19 respecto al RNF de robustez) |
| Notas | (notas técnicas, dependencias arquitectónicas, referencias cruzadas a otros RNF, decisiones técnicas D-XXX relacionadas) |
```

**Justificación del diseño de plantilla:**

1. **Identificador segmentado.** El código `RNF-XXX-NN` permite agrupar visualmente en el índice del documento todos los RNF de la misma característica (todos los RNF-PERF juntos, todos los RNF-SEC juntos), y permite que la referencia desde una HU sea autoexplicativa: leer "ver RNF-PERF-03" en un CA comunica que es un RNF de rendimiento sin abrir el documento.

2. **Doble campo de prioridad.** "MoSCoW sugerida" anticipa la ceremonia formal de priorización pendiente del proyecto (DHU-007 dejó esta ceremonia pendiente). El documento RF/RNF sugiere prioridades para que la ceremonia parta de una base argumentada; la ceremonia las ratifica o ajusta.

3. **Aplicabilidad explícita para RNF transversales.** La consolidación del umbral de 5 s (C.1) y del RNF de robustez (C.2) produce RNFs que aplican a 10 y 21 HUs respectivamente. El campo "Aplicabilidad" hace explícita esa lista para evitar que el lector tenga que reconstruirla.

4. **Excepciones como campo de primera clase.** La política conservadora de HU-19 respecto al RNF de robustez es un patrón que probablemente se repite (por ejemplo, HU-21 también rechaza el escalamiento en lugar de marca pasiva). Tener el campo "Excepciones" en la plantilla evita que estas variantes se diluyan en notas o se pierdan.

#### E. Política de derivación de RF desde los CAs existentes

Los RF se derivan de los CAs de las HUs siguiendo tres principios cerrados como decisión metodológica:

**E.1 — Composición transversal antes que biyección.** Un RF puede agrupar comportamientos coherentes declarados en CAs de varias HUs, cuando esos CAs describen el mismo qué del sistema desde perspectivas distintas (típicamente una HU del Operador y la HU del Administrador o del Gerente que consume el mismo sustrato técnico).

**Ejemplo de composición (hipotético, ver nota):**

> CA-02.1 + CA-03.1 + CA-04.1 → **RF hipotético — Presentación de variables de estado de la intersección por acceso**. El sistema expone, por cada acceso de la intersección, las variables observadas del estado del tráfico (flujo, longitud de cola, nivel de congestión) y las predicciones del nivel de congestión hasta el horizonte configurado, con actualización en tiempo real.

Los tres CAs alimentarían un solo RF porque describen el mismo comportamiento del sistema (presentación de variables por acceso) desde tres vistas distintas del Operador (HU-02 monitoreo, HU-03 predicción, HU-04 vista combinada). La presentación específica de cada vista (cómo se diseña el dashboard) no sería del RF; sería de las HUs.

**Nota sobre la materialización real:** el catálogo final de la sección 2 del documento RF/RNF aplicó un criterio más conservador para este caso: en lugar de consolidar los tres CAs en un único RF transversal, se redactaron tres RF separados (RF-003 derivado de HU-02, RF-004 derivado de HU-03, RF-005 derivado de HU-04) con la observación explícita de que RF-005 compone visualmente RF-003 y RF-004 sin duplicar lógica. La separación se prefirió porque cada HU original describe una vista distinta del Operador con criterios de aceptación diferenciados; agruparlas como un solo RF habría diluido la trazabilidad bidireccional HU ↔ RF. La composición transversal sí se aplicó en otros casos del catálogo donde la equivalencia funcional era inequívoca (por ejemplo, los CAs de control de acceso de las 21 HUs consolidados en RF-002 transversal según E.2; o RF-010 y RF-018 compartiendo sustrato técnico con presentación diferenciada por rol).

**E.2 — Control de acceso como precondición transversal, no como RF por HU.** Las 21 HUs incluyen un CA tipo "Dado que el [rol] no ha iniciado sesión, cuando intenta acceder, entonces el sistema lo redirige a la pantalla de login" y/o un CA de RBAC tipo "Dado que un usuario con rol no autorizado intenta acceder, el sistema responde HTTP 403". Estos CAs no producen 21 o 42 RFs separados; producen **dos RFs transversales**:

- **RF-001 — Autenticación al sistema** (consume TTH-01; precondición de cualquier acceso a HUs operativas).
- **RF-002 — Control de acceso por rol** (consume HU-01; aplicable a las 21 HUs operativas con tabla de aplicabilidad por endpoint).

Este patrón evita la inflación artificial del catálogo de RF.

**E.3 — Comportamiento de robustez ante interrupción no produce RF separados.** Los CAs de robustez (CA-XX.4 según DHU-005 Caso A o B) no se vuelven RFs porque la robustez es no funcional por naturaleza. Cada CA de robustez se referencia desde el RF correspondiente (el que captura el comportamiento normal de la vista) y se materializa en el RNF transversal de la subsección C.2.

**Ejemplo:**

> El RF hipotético del ejemplo de E.1 (presentación de variables por acceso) tendría como CAs origen CA-02.1, CA-03.1, CA-04.1 (presentación). Los CAs CA-02.4, CA-03.4, CA-04.4 (robustez) NO entrarían como CAs origen de ese RF; entran como aplicabilidad del **RNF-REL-01 — Robustez ante interrupción de fuente** (consolidado en C.2). Esta política se mantiene en el catálogo real con los RF efectivamente redactados (RF-003, RF-004, RF-005): los CAs de robustez no son CAs origen de cada RF de presentación, sino aplicabilidad del RNF-REL-01 transversal.

**Estimación de cantidad de RF al consolidar:** las 21 HUs contienen aproximadamente 130 CAs operativos (cifra aproximada al cierre del MVP2). Aplicando composición transversal, se estima **entre 25 y 35 RFs** en el catálogo final. La cifra exacta se confirma al redactar la sección 2 del documento RF/RNF; no se preestablece para no forzar la consolidación.

#### F. Política de prioridad MoSCoW sugerida en este documento

DHU-007 y el cierre del MVP2 declararon como entregables pendientes: documento RF/RNF, ceremonia de Planning Poker (estimación) y ceremonia MoSCoW (priorización). La ceremonia MoSCoW es sesión dedicada separada.

**Decisión:** el documento RF/RNF declara una **prioridad MoSCoW sugerida** para cada RF y cada RNF, no vinculante. La ceremonia formal MoSCoW posterior ratifica o ajusta. La función de la prioridad sugerida es:

1. **Anclar la ceremonia.** La ceremonia parte de una base argumentada en lugar de cero.
2. **Documentar el razonamiento.** Cada RF/RNF en el documento incluye implícitamente el razonamiento de su prioridad en sus campos "Persona beneficiaria" y "Objetivo del producto"; la prioridad sugerida es derivable de esos campos.

**Convención de prioridad sugerida:**

| RF/RNF que... | Sugerencia |
|---|---|
| Realiza directamente uno de los 4 Objetivos del Producto y aplica a una Persona MVP1 | **Must** |
| Realiza directamente uno de los 4 Objetivos del Producto y aplica a una HU MVP2 (incluyendo HU-09) | **Should** |
| Es soporte transversal del MVP1 (autenticación, control de acceso, auditoría) | **Must** |
| Es robustez del MVP1 (DHU-005, fallbacks de TTH-04, valores por defecto seguros) | **Must** (continuidad operativa = Objetivo 3) |
| Es accesibilidad WCAG 2.1 nivel AA | **Should** |
| Es coherencia textual de catálogos de plantillas | **Could** |
| Es paralelización del cálculo (HU-18) o tolerancia parametrizada (HU-20) | **Could** |
| Aplica a TTH-06 (Trabajos Futuros declarados) | **Won't** (en el ciclo del proyecto académico) |

#### G. Política de reemplazo retroactivo de umbrales hardcoded en las HUs por referencias al documento RF/RNF

DHU-007 línea 358 declaró como parte del trabajo futuro asociado: *"Reemplaza, en cada HU, los umbrales hardcodeados por referencias al RNF correspondiente"*. Esta política cierra el detalle de cómo se ejecuta ese reemplazo.

**Decisión:** **los CAs de las 21 HUs no se reescriben sustituyendo umbrales por referencias `RNF-XXX-NN`.** En su lugar, se aplica el siguiente patrón aditivo y no destructivo:

1. **Los CAs de las HUs preservan su redacción literal con umbrales hardcoded.** Esta redacción es resultado de tres semanas de trabajo cuidadoso (DHU-001 a DHU-018) y los CAs son autocontenidos para lectura humana sin tener que abrir el documento RF/RNF para entender una HU.

2. **Cada sección "Candidatos a RNF" de cada HU se actualiza retroactivamente** con una pasada de mantenimiento aditiva: cada candidato declarado en la HU recibe una referencia al RNF formal correspondiente (`→ RNF-XXX-NN`). El contenido sustantivo del candidato no se modifica; solo se agrega la referencia cruzada al final.

3. **El documento RF/RNF es la fuente normativa formal** de los umbrales. Si un umbral se ajusta en sesión posterior (típicamente tras Planning Poker, validación cuantitativa, o ajuste por feedback del jurado), se ajusta en el documento RF/RNF, no en las HUs. Los CAs de las HUs preservan los umbrales originales como referencia documental del momento de redacción.

4. **Las TTH no se modifican retroactivamente.** Las 11 TTH ya consumen los umbrales relevantes vía CTs (criterios técnicos de terminado); el documento RF/RNF referencia las CTs como origen sin pedir cambios a las TTH.

**Justificación de la política no destructiva:**

1. **Preservación de autocontención de las HUs.** Si los CAs perdieran sus umbrales y los reemplazaran por referencias, una HU dejaría de ser legible por sí misma; obligaría a abrir un segundo documento solo para entender qué umbral aplica.

2. **Coherencia con DHU-018.** DHU-018 estableció que la aplicación retroactiva del patrón "Resumen ejecutivo" sería estrictamente aditiva y nunca modificaría contenido sustantivo de las HUs. DHU-019 sostiene el mismo principio: los CAs no se modifican; solo se agrega referencia al RNF.

3. **Política análoga ya aplicada en el backlog.** Las notas técnicas de cada HU referencian decisiones D-XXX y DHU-XXX sin "borrar" el contenido relevante de la HU; agregan referencia. La política para RNF replica este patrón ya validado.

4. **Robustez ante evolución del documento RF/RNF.** Si en una futura iteración del documento RF/RNF se reorganiza la numeración o se agregan/eliminan RNF, las HUs no requieren pase de mantenimiento masivo: las referencias cruzadas son referencias, no umbrales propagados.

#### H. Estructura del documento RF/RNF

El documento `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` se organiza en seis secciones:

```
0. Preámbulo (propósito, alcance, convenciones, documentos relacionados, **distinción entre RF y RNF-FUN como nota orientadora al lector**)
1. Marco de referencia
   1.1 ISO/IEC 25010:2023 — las 9 características adoptadas
   1.2 Derivación de RF desde HUs (referencia a DHU-019 subsección E)
   1.3 Clasificación de RNF (referencia a DHU-019 subsecciones A y B)
   1.4 Trazabilidad bidireccional HU/TTH ↔ RF ↔ RNF

2. Requisitos Funcionales (RF)
   2.1 Familias funcionales declaradas (7 familias)
   2.2 Catálogo de RF (RF-001 a RF-NNN con plantilla unificada)
   2.3 Tabla de trazabilidad RF → HU(s) origen → CAs

3. Requisitos No Funcionales (RNF), clasificados por característica ISO 25010:2023
   3.1 Functional Suitability (RNF-FUN-NN)
   3.2 Performance Efficiency (RNF-PERF-NN)
   3.3 Compatibility (RNF-COM-NN)
   3.4 Interaction Capability (RNF-INT-NN)
   3.5 Reliability (RNF-REL-NN)
   3.6 Security (RNF-SEC-NN)
   3.7 Maintainability (RNF-MNT-NN)
   3.8 Flexibility (RNF-FLX-NN)
   3.9 Safety (RNF-SAF-NN)

4. Matriz de trazabilidad RNF → HUs/TTH origen

5. Glosario de términos del producto referenciados en el documento

6. Cierre y mantenimiento
   6.1 Cuándo se actualiza el documento
   6.2 Cómo se referencia desde las HUs y desde las TTH
   6.3 Relación con la ceremonia MoSCoW pendiente
```

**Las 7 familias funcionales declaradas en 2.1** se derivan de la composición natural del backlog:

1. **Control de acceso y autenticación** (insumo: HU-01 + TTH-01).
2. **Monitoreo operativo en tiempo real** (insumo: HU-02 a HU-07, HU-10, HU-11, HU-12).
3. **Decisiones del motor adaptativo** (insumo: HU-05, HU-06, HU-07, HU-08; sustento TTH-10).
4. **Predicción de tráfico** (insumo: HU-03, HU-04; sustento TTH-09, TTH-11).
5. **Soporte técnico y configuración del sistema** (insumo: HU-13, HU-14, HU-15, HU-20; sustento TTH-04, TTH-05).
6. **Reportería ejecutiva** (insumo: HU-16, HU-17, HU-18, HU-19).
7. **Soporte al Operador y trazabilidad de incidentes** (insumo: HU-09, HU-21).

#### I. Política de umbrales: respetar los valores de las HUs por defecto

DHU-007 línea 357 declaró que el documento RF/RNF podría ajustar umbrales respecto a los valores tentativos de las HUs. DHU-019 cierra esta política a favor de la conservación.

**Decisión:** **los umbrales del documento RF/RNF se inicializan idénticos a los umbrales declarados en los Candidatos a RNF y en los CAs de las 21 HUs.** Solo se admite ajuste en este momento de redacción cuando:

1. La HU declara explícitamente "criterio sugerido", "≤ X segundos sugerido", o vocabulario equivalente que reconoce la tentatividad del umbral (HU-15, HU-16, HU-17, HU-18, HU-19, HU-20, HU-21 son los casos típicos).

2. El umbral aparece consolidado por la subsección C.1 (≤ 5 s de tiempo real consolidado a partir de diez HUs que lo declaran idéntico).

3. Existe inconsistencia explícita entre dos HUs respecto al mismo concepto; en este caso se resuelve declarándolo como decisión específica en el documento RF/RNF con justificación.

**No se admite ajuste arbitrario.** Los umbrales que entren al documento son los del backlog, no umbrales calibrados ad hoc durante la redacción del RF/RNF. La calibración de umbrales reales contra mediciones del sistema integrado se reporta conforme a D-005 (números de tesis: actualizar tras validación real) en sesión posterior dedicada, no durante la redacción del documento RF/RNF.

**Justificación:**

1. **Los umbrales del backlog son resultado de trabajo cuidadoso.** Cada umbral fue discutido durante la redacción de la HU correspondiente; ajustarlos arbitrariamente durante la consolidación rompería la trazabilidad metodológica.

2. **Política coherente con DHU-005.** DHU-005 reportará comportamientos medidos en sesión de validación cuantitativa; los umbrales documentados son los esperados al diseño, no los medidos. Ajustar a posteriori los esperados pierde la separación entre diseño y validación.

3. **El documento RF/RNF es vivo.** Si una medición real demuestra que un umbral es irreal, el documento se actualiza en pasada futura referenciando el dato medido. La política de conservación no es perpetua; es para esta sesión de redacción.

### Decisión final

**DHU-019 cierra las nueve decisiones metodológicas (A a I) necesarias para iniciar la redacción del documento `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md`.**

Resumen ejecutivo de las nueve subsecciones:

| Subsección | Decisión |
|---|---|
| A | ISO/IEC 25010:2023 con sus 9 características (incluyendo Safety y los renombramientos Usability → Interaction Capability y Portability → Flexibility) como taxonomía única de clasificación de RNF. |
| B | Tabla de reasignación normativa de las categorías heterogéneas de DHU-007 a las 9 características ISO 25010:2023, con notación de doble código para RNF de característica primaria + secundaria. |
| C | Siete inconsistencias detectadas en los Candidatos a RNF cerradas explícitamente como decisiones de redacción (consolidación de ≤ 5 s, consolidación de robustez ante interrupción, reformulación normativa de HU-13, reubicación de calidad de predicción de HU-03 al ámbito de TTH-09, unificación de inmutabilidad de logs, aceptación formal de la reasignación masiva, clasificación de manejabilidad de datos faltantes en Functional Correctness). |
| D | Plantilla unificada de RF y de RNF con identificador segmentado por característica, doble campo de prioridad (sugerida + ratificada por ceremonia MoSCoW posterior), aplicabilidad explícita para RNF transversales y excepciones como campo de primera clase. |
| E | Política de derivación de RF desde CAs con composición transversal (no biyección), control de acceso como dos RFs transversales únicos (autenticación y RBAC) y robustez ante interrupción como RNF transversal único. Estimación 25 a 35 RFs. |
| F | Política de prioridad MoSCoW sugerida en el documento RF/RNF como anclaje argumentado para la ceremonia formal posterior; convención basada en Objetivos del Producto y clasificación MVP1/MVP2. |
| G | Política aditiva de reemplazo retroactivo de umbrales: los CAs de las HUs preservan su redacción literal; cada sección "Candidatos a RNF" se actualiza retroactivamente con referencias `→ RNF-XXX-NN`; el documento RF/RNF es la fuente normativa formal de umbrales. |
| H | Estructura de seis secciones del documento RF/RNF con 7 familias funcionales declaradas para la sección 2 (RF) y las 9 secciones de RNF por característica ISO 25010:2023. |
| I | Política de conservación de umbrales: los umbrales del documento RF/RNF se inicializan idénticos a los del backlog; ajustes solo cuando la HU declara explícita tentatividad, cuando la subsección C.1 consolida, o cuando existe inconsistencia entre HUs. |

### Lo que NO cambia con DHU-019

- **Las decisiones DHU-001 a DHU-018 mantienen su contenido sustantivo.** DHU-019 las cita y las usa como base sin reabrir ninguna.
- **El alcance del producto** (Personas, Objetivos, Journeys, Visión) se mantiene intacto.
- **Las 21 HUs operativas (HU-01 a HU-21)** no se reescriben. Sus CAs preservan los umbrales hardcoded. Sus secciones "Candidatos a RNF" reciben una pasada aditiva de referencias `→ RNF-XXX-NN` cuando el documento RF/RNF esté redactado (subsección G).
- **Las 11 TTH (TTH-01 a TTH-11)** no se modifican. Los criterios técnicos de terminado se referencian desde el documento RF/RNF como origen de RNFs específicos sin pedir cambios a las TTH.
- **Las decisiones técnicas D-001 a D-009** se mantienen sin modificación. El documento RF/RNF las consume y referencia sin reabrirlas.
- **La política de construcción MVP2 establecida por DHU-012 y aplicada por DHU-017** se mantiene. El documento RF/RNF refleja la clasificación MVP1/MVP2 en el campo de prioridad sugerida sin alterarla.
- **Las ceremonias Planning Poker y MoSCoW** declaradas como pendientes desde el cierre del MVP2 siguen pendientes. DHU-019 las anticipa con prioridad sugerida pero no las ejecuta.

### Documentos afectados por DHU-019

| Documento | Tipo de cambio |
|---|---|
| `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` (nuevo) | Documento nuevo. Se redacta tras aprobar DHU-019, siguiendo la estructura de la subsección H y aplicando las plantillas de la subsección D, las clasificaciones de las subsecciones A y B, y las políticas de las subsecciones E, F, G, I. |
| `DECISIONS_HU.md` | Agregar DHU-019 al índice y cuerpo; actualizar fecha de última actualización; actualizar tabla "Resumen de impacto en bloques redactados hasta la fecha" con fila Transversal-DHU-019. Actualizar `Documentos relacionados` incluyendo el nuevo documento RF/RNF. |
| `HU_BLOQUE_A.md` a `HU_BLOQUE_F.md`, `HU_MVP2.md` | Pasada aditiva (subsección G): cada sección "Candidatos a RNF" de cada HU recibe una referencia `→ RNF-XXX-NN` por candidato declarado, sin modificar contenido sustantivo. Rango DHU referenciado actualizado de "DHU-001 a DHU-018" a "DHU-001 a DHU-019". Esta pasada se ejecuta tras cerrar el documento RF/RNF, no simultáneamente. |
| `TAREAS_TECNICAS_HABILITADORAS.md` | Rango DHU referenciado actualizado en cabecera y "Documentos relacionados" de "DHU-001 a DHU-018" a "DHU-001 a DHU-019". Sin cambios sustantivos al contenido de las 11 TTH (los criterios técnicos relevantes se referencian desde el documento RF/RNF, no se modifican). |
| `BACKLOG_OVERVIEW.md` | Opcional: mención del documento RF/RNF como cierre del cabo suelto declarado al cierre del Product Backlog. La sección "Cómo navegar el backlog" puede incluir una línea adicional sobre el documento. |
| `LEAN_INCEPTION_CEREBROVIAL.md` | Documentos relacionados actualizado (referencia al nuevo documento RF/RNF). |

### Documentos relacionados

- `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` — Documento nuevo a redactar tras aprobar DHU-019.
- `DECISIONS_HU.md` (este documento) — sección DHU-019.
- `DECISIONS_HU.md` — DHU-007 (origen del trabajo pendiente que DHU-019 ejecuta).
- `DECISIONS_HU.md` — DHU-005 (Caso A y Caso B de robustez, consolidados como RNF transversal en subsección C.2).
- `DECISIONS_HU.md` — DHU-006 (HUs agnósticas a implementación, preservado en la redacción del documento RF/RNF).
- `DECISIONS_HU.md` — DHU-012 (semántica refinada de MVP2, reflejada en prioridad MoSCoW sugerida).
- `DECISIONS_HU.md` — DHU-014, DHU-016, DHU-017 (patrón de decisiones consolidadas que DHU-019 reproduce).
- `DECISIONS_HU.md` — DHU-018 (política aditiva no destructiva, replicada por DHU-019 en la subsección G).
- `DECISIONS.md` — D-005 (números de tesis: actualizar tras validación real), referenciada por la subsección I.
- ISO/IEC 25010:2023 — Norma adoptada como taxonomía única, declarada en la subsección A.

---

## DHU-020 — Semántica de ControlView: cierre de Delta-08 y refactor en bloque hacia vista pasiva

**Fecha:** 2026-05-20.
**Tipo:** Decisión metodológica de alineación especificación↔código (cierre de discrepancia de auditoría).
**Estado:** Cerrada.
**Origen:** Delta-08 de `AUDITORIA_HU_CODIGO.md`, clasificado como riesgo R1 en `REPORTE_PLANIFICACION_SPRINT_4.md`. Evidencia consolidada en `DELTA_08_ANALISIS.md` (análisis de campo sobre el repositorio vivo `CerebroVial/`).
**Documentos afectados al cierre:** `AUDITORIA_HU_CODIGO.md` (cierre de Delta-08), `REPORTE_PLANIFICACION_SPRINT_4.md` (resolución de R1), y este registro en `DECISIONS_HU.md`. No se modifica `HU_BLOQUE_B.md`: HU-05 se mantiene tal cual está redactada.

### Contexto

HU-05 ("Estrategia activa") especifica una **vista pasiva del estado vigente del motor adaptativo**: el Operador observa qué estrategia corre en producción, con qué parámetros y desde cuándo, sin disparar ni alterar la decisión del motor. La semántica pasiva está ratificada de forma redundante en la propia HU (resumen ejecutivo, notas clave, verbo "visualizar") y de forma cruzada por HU-06, HU-07 y HU-08, que referencian a HU-05 como "vista pasiva del estado actual".

El código actual (`frontend_ui/src/components/views/control/ControlView.tsx` y su familia) implementa, en cambio, un **playground interactivo request-response**: el usuario edita un estado de intersección hipotético (vía editor de fases, slider de tiempo perdido, presets de demostración) y pulsa "Recomendar" para obtener un cálculo del motor mediante `POST /control/recommend`. El endpoint es una función pura sin estado.

El análisis de campo (`DELTA_08_ANALISIS.md`, Hallazgo #3) estableció que el delta no es de presentación sino de **semántica de la fuente y dirección del flujo**, y reveló un hecho determinante: **el backend no tiene la noción de "estrategia vigente en producción"**. No existe un motor corriendo solo cuya decisión vigente se persista; existe una calculadora invocada a demanda sobre estados inventados. Por tanto, alinear el código a HU-05 no es solo retirar interactividad del frontend, sino *construir* el concepto de estado vigente persistido por intersección con timestamp de activación.

### Decisión

#### Subsección A — Semántica normativa: gana la vista pasiva. HU-05 se mantiene sin enmiendas.

La especificación es la fuente normativa del proyecto y prevalece sobre la implementación. HU-05 conserva su redacción actual (cabecera, descripción y CA-05.1 a CA-05.5). El código se alinea a la HU, no a la inversa.

Se descarta explícitamente la opción de enmendar HU-05 para legitimar la semántica playground, por tres razones:

1. HU-05 es el núcleo del Objetivo 3 del producto (naturaleza adaptativa). Convertirla en simulador vaciaría su justificación declarada: trazabilidad operativa y coherencia percibida del Operador.
2. HU-06, HU-07 y HU-08 referencian cruzadamente a HU-05 como vista pasiva. Enmendarla forzaría a deshacer la coherencia consolidada del Bloque B completo.
3. CA-05.2 (timestamp de activación) y CA-05.3 (auto-update ≤5 s) solo tienen sentido sobre un estado que evoluciona por sí mismo; bajo semántica playground quedarían huérfanas.

#### Subsección B — Destino del playground: se preserva como herramienta de Administrador / validación, no se elimina.

El playground actual (edición de estado, presets pedagógicos, métricas instructivas en vivo, tarjeta del caso `webster_infeasible`) **no se destruye**. Se reubica como vista interna separada, fuera del flujo del Operador, accesible bajo rol de Administrador o como herramienta de validación.

Fundamento:

- **Valor docente para la tesis.** Los presets, las métricas instructivas (Y = Σflow/sat, umbral 1500, badge PEAK/OFF-PEAK) y la tarjeta del caso patológico de Webster demuestran *cómo decide el motor*. Es evidencia construida, directamente útil en la defensa. Eliminarla es costo sin retorno.
- **Costo de backend idéntico.** Las opciones "eliminar" y "preservar" requieren exactamente el mismo trabajo nuevo de backend (estado vigente persistido + endpoint de lectura + infraestructura realtime). La diferencia es que eliminar además invierte esfuerzo en *destruir* lo que ya funciona. Preservar evita ese esfuerzo de destrucción.
- **Consistencia con la planificación.** Coincide con la nota informal de R1 del Sprint 4 ("vista pasiva + tab admin oculto con el playground actual").

`POST /control/recommend` se conserva como endpoint del playground. La nueva vista pasiva consume un endpoint distinto de lectura del estado vigente (ver Subsección D).

#### Subsección C — Pendiente declarado: el playground requiere un elemento de backlog propio.

El playground reubicado no puede vivir huérfano en el SDD. Queda pendiente de formalización un elemento de backlog (HU de Administrador o TTH de herramienta de validación) que lo cubra. **No se redacta en esta DHU**; se declara como pendiente explícito para que el SDD no documente un componente sin trazabilidad a backlog. Identificador provisional del pendiente: la herramienta de exploración del motor adaptativo, a clasificar HU/TTH en sesión dedicada.

#### Subsección D — Alcance del refactor: se abordan Delta-07, Delta-08 y Delta-09 en un solo bloque.

Los tres deltas tocan los mismos archivos (`ControlView.tsx`, `RecommendationPanel.tsx`, `controlService.ts`) y son interdependientes:

- **Delta-08** (este): semántica pasiva vs. playground.
- **Delta-07**: ausencia de infraestructura realtime. CA-05.3 (auto-update ≤5 s) es imposible sin polling o SSE/WebSocket, hoy inexistentes.
- **Delta-09**: el "Log técnico (para operador C4)" usa lenguaje técnico crudo, en conflicto con el lenguaje de dominio que piden HU-05 y HU-06.

Se decide abordarlos **en bloque**, en un único refactor, en lugar de secuencialmente. Abrir y reescribir los mismos componentes tres veces sería retrabajo. El refactor en bloque produce: (1) la vista pasiva de solo lectura sobre estado vigente, (2) con actualización automática vía el canal realtime nuevo, (3) con lenguaje de dominio en lugar de log técnico.

Componentes reutilizables sin reescritura semántica: `TrafficLightCycle.tsx`, `TimingBar.tsx`, `ModeSelector.tsx` (visualizan modo y tiempos, agnósticos al origen del dato).

#### Subsección E — Reconocimiento explícito del cambio estructural de backend.

Construir "estado vigente persistido por intersección con timestamp de activación" es un cambio estructural que probablemente implica modificación del modelo de persistencia. `CLAUDE.md` instruye parar y preguntar ante cambios estructurales o de modelo de BD. **Esta DHU es ese punto de decisión deliberada.** Queda registrado que la creación de la persistencia de estrategia vigente se autoriza conscientemente como parte del cierre de Delta-08, y no como un efecto colateral no examinado de un refactor de frontend.

El diseño concreto de esa persistencia (entidad, esquema, política de retención frente a HU-08 que ya cubre el registro histórico) se cierra en el SDD, no aquí. Esta DHU autoriza su existencia y fija su propósito; el SDD define su forma.

### Criterios de aceptación afectados

| CA | Estado tras DHU-020 |
|---|---|
| CA-05.1 (nombre + tiempos por acceso) | Se mantiene. La fuente pasa a ser el estado vigente real, no el formulario. Etiqueta de estrategia en lenguaje de dominio (DHU-006), no nombre técnico del algoritmo. |
| CA-05.2 (timestamp de activación) | Se mantiene. Requiere la persistencia de estado vigente autorizada en Subsección E. |
| CA-05.3 (auto-update ≤5 s) | Se mantiene. Requiere la infraestructura realtime del bloque Delta-07 (Subsección D). |
| CA-05.4 (última estrategia conocida "no confirmada", DHU-005 Caso B) | Se mantiene. El manejo de errores actual (semántica de fallo de cálculo de un request) se sustituye por semántica de fuente vigente no confirmada. |
| CA-05.5 (redirección a login) | Se mantiene. Hoy no implementada; entra en el alcance del refactor. |

### Relación con decisiones previas

- **DHU-005 (Caso B):** CA-05.4 aplica la política conservadora de fuente no confirmada. Sin cambios.
- **DHU-006 (vocabulario agnóstico a implementación):** refuerza el cambio de etiquetas técnicas (`webster`/`max_pressure`) a lenguaje de dominio en la vista del Operador.
- **DHU-013/014/015 (clasificación HU/TTH):** patrón seguido por esta DHU. El pendiente de Subsección C se clasificará con el mismo criterio.

### Preguntas que se cierran al implementar (no se renegocian, se resuelven en SDD)

- Diseño de la persistencia de estado vigente (entidad, esquema, relación con el registro histórico de HU-08).
- Mecanismo realtime concreto (polling vs. SSE vs. WebSocket) para CA-05.3.
- Ubicación exacta del playground reubicado (tab de AdminView vs. ruta separada gateada por rol).
- Clasificación HU/TTH del pendiente de Subsección C.

### Lo que NO cambia con DHU-020

- **HU-05 a HU-08 conservan su redacción.** DHU-020 no reescribe ninguna HU; ratifica la semántica pasiva ya redactada y alinea el código a ella.
- **Las TTH** no se modifican. El refactor consume sustrato técnico existente (TTH-10, motor adaptativo) sin reabrir su definición.
- **Las decisiones técnicas D-001 a D-009** se mantienen. El diseño concreto de la persistencia de estado vigente se cierra en el SDD citándolas, sin reabrirlas.
- **El alcance del Sprint 4** no cambia: el refactor de HU-05 ya estaba comprometido (item #4, 3 SP). DHU-020 fija su semántica, no agrega trabajo no planificado salvo el reconocimiento del backend nuevo que ya estaba implícito en CA-05.2/05.3.

### Documentos relacionados
- `AUDITORIA_HU_CODIGO.md` — Delta-08 (origen); se marca cerrado apuntando a DHU-020.
- `REPORTE_PLANIFICACION_SPRINT_4.md` — Riesgo R1 (origen); se resuelve apuntando a DHU-020.
- `HU_BLOQUE_B.md` — HU-05, HU-06, HU-07, HU-08 (semántica pasiva ratificada cruzadamente).
- `DECISIONS_HU.md` — DHU-005 (Caso B de robustez), DHU-006 (vocabulario agnóstico), DHU-013/014/015 (patrón de clasificación).
- `DECISIONS.md` — D-001 a D-009 (insumo del diseño de persistencia en SDD).
- `CLAUDE.md` — guardia de cambios estructurales / modelo de BD, invocada en la Subsección E.

---

## DHU-021 — Decisiones metodológicas de redacción del SDD

**Fecha:** 2026-05-20.
**Tipo:** Decisión metodológica consolidada sobre la redacción de un documento del proyecto (análoga a DHU-016, DHU-017 y DHU-019, que consolidaron las decisiones de redacción de los Bloques F, MVP2 y del documento RF/RNF respectivamente).
**Estado:** Cerrada.
**Origen:** La redacción del SDD (`documentation/sdd/SDD_CEREBROVIAL.md`) en el marco híbrido 4+1 / ISO 25010 / ADR generó decisiones de método —postura del documento, proceso, estructura, tratamiento de conflictos del corpus— que no son decisiones de producto (serie `D-`) ni del backlog (serie `DHU-` previas), sino del propio acto de documentar la arquitectura. Se acordó acumularlas durante la redacción y consolidarlas en una sola DHU al cerrar el SDD. Una segunda tanda de ajustes surgió de la sesión de verificación del SDD contra el repositorio vivo `CerebroVial/` (Claude Code, 2026-05-20).
**Hogar canónico:** Esta entrada es la fuente única de DHU-021. El SDD ya no reproduce el texto completo; conserva un puntero a este registro.

### Contexto

El SDD se redactó conversacionalmente y se cerró en dos movimientos: primero la redacción del cuerpo (§0–§12 + vista de desarrollo), que produjo decisiones de método; después una verificación punto por punto contra el código real, que corrigió afirmaciones inferidas y disparó ajustes de diseño. Ambos movimientos generan meta-decisiones sobre *cómo se documenta la arquitectura*, que esta DHU consolida sin reabrir el contenido sustantivo de HUs, TTH ni decisiones técnicas `D-`.

### Decisión — Grupo 1: Decisiones de redacción del SDD (1–17)

1. **Conciliación As-designed / matriz rica confinada a §10.** El cuerpo (§1–§9) es As-designed puro; el estado y los deltas se confinan a §10 (matriz bidireccional HU/TTH ↔ componente ↔ estado ↔ delta) y §11 (brecha). Honra simultáneamente la postura As-designed, la matriz con estado/delta y la separación de avance, y es la forma de menor confusión para los agentes de código.
2. **Formato Markdown, construcción incremental y convenciones de cita.** Único archivo Markdown construido sección por sección con verificación de coherencia antes de avanzar; convenciones de cita uniformes con el corpus (`D-00N`, `DHU-0NN`, `HU-NN`, `TTH-NN`, `RF-0NN`, `RNF-XXX-NN`, `CA/CT-NN.N`, `Delta-NN`; rutas en estilo de código).
3. **Proceso Spec Kit + estructura híbrida 4+1 / ISO 25010 / ADR.** Spec Kit gobierna el proceso (Spec→Plan→Tasks→Implement); el híbrido 4+1 (vistas) + ISO 25010 (calidad) + ADR ligero (decisiones) gobierna la estructura interna. El SDD corresponde a `plan.md` + `data-model.md`.
4. **Adopción brownfield de Spec Kit (mapear, no regenerar).** El corpus curado se mapea a las plantillas; no se ejecutan comandos generativos. Preserva la trazabilidad fina y las 20 DHU.
5. **`ARCHITECTURE_TARGET.md` archivado en legacy, no citado.** Versión pre-Inception (Azure/microservicios/YOLOv8/motor de reglas/MongoDB) que contradice D-001, D-003, TTH-10 y la auditoría. Se archiva en `legacy/` y no se menciona en el SDD; la narrativa de evolución vive en `EVOLUCION_TESIS.md`.
6. **Colisión de IDs `D-` entre el audit viejo y `DECISIONS.md` canónico.** El SDD usa la numeración canónica de `DECISIONS.md`; las decisiones del `DATA_MODEL_AUDIT.md` (2026-05-03) se citan por contenido y fecha, no por su ID-D, para evitar ambigüedad.
7. **Notación C4 reservada al informe; el SDD no la usa.** El SDD usa el formato del híbrido 4+1 (prosa + tablas + diagramas propios), sin C4; C4 se reserva para el informe/sustentación.
8. **SDD como fuente canónica de componentes; el C4 del informe deriva de él.** El SDD es la fuente de verdad de la descomposición; si el informe usa C4, deriva del SDD (mismos nombres, misma descomposición), evitando divergencia.
9. **Profundidad de §3 a dos niveles; detalle DDD a la vista de desarrollo.** §3 se descompone en contenedores + interior del núcleo; la estructura DDD interna se documenta en la vista de desarrollo, no en §3.
10. **Estado vigente e historial del motor como dos entidades, con `jsonb` para las fases (§4).** `motor_decisions` (historial append-only) y `engine_active_state` (puntero mutable, FK a la decisión activada); las fases se persisten como `jsonb` —no como tabla normalizada— por fidelidad al sistema real (el motor recibe las fases por payload, no las lee de BD).
11. **`motor_decisions` relacional pese a su naturaleza temporal (§4).** El volumen de una intersección piloto no justifica hypertable; la conversión se difiere a productivización. Defendible frente al contraste con `waze_jams` por volumen y origen (sistema propio vs. feed externo).
12. **Frontera grafo↔intersección diferida (§4.3).** El adaptador cámara→approach→fase y la conversión nivel→flujo no se esquematizan; las decisiones del motor se anclan a `graph_nodes` y el interior de la intersección queda como extensión futura, coherente con el alcance.
13. **Estado vigente como entidad propia con puntero, no como vista derivada (§4).** Separa el evento de cálculo (`decided_at`) del de activación (`activated_at`), semánticas que una vista no captura.
14. **SSE como mecanismo del canal de tiempo real (§5.2).** Elegido sobre WebSocket (excedente para flujo unidireccional) y polling (desperdicia peticiones); coherente con HTTP/SSE de D-004. Validado por Delta-07, que sugiere SSE como default razonable.
15. **Estado vigente leído vía API desde BD, sin cache en memoria (§5.2).** Suficiente para la intersección piloto; el cache se difiere a productivización.
16. **Dos loops de primera clase, operativo y de validación SUMO (§5).** Comparten núcleo y difieren solo en la fuente de estado; materializan la frontera de §3.3 en la vista de proceso.
17. **Topología vigente de una máquina con Docker Compose; mapeo edge/servidor documentado, no entregado (§6).** El plan de productivización no exige cambios estructurales, solo de configuración de despliegue.

### Decisión — Grupo 2: Ajustes derivados de la verificación SDD↔repo (V1–V4)

Estos cuatro ajustes nacieron de confrontar el SDD contra el código real (no de la redacción), y se registran separados para que su procedencia sea visible. Tocan el diseño de §4, §5 y §7/§11, y se aplicaron al cuerpo del SDD en la misma sesión.

- **V1 — `node_id` como FK a `graph_nodes` resuelto en el write-path (§4.2.1).** La verificación halló que el motor real emite un `intersection_id` opaco, sin FK ni consulta a la base de datos (es una calculadora sin estado). Se decide conservar `node_id` como FK a `graph_nodes` en `motor_decisions`/`engine_active_state` —por auditabilidad y anclaje al grafo— y hacer que la capa de persistencia resuelva y valide el `intersection_id` del motor contra `node_id` al escribir. El anclaje al grafo es responsabilidad del write-path, no del contrato del endpoint.
- **V2 — Conservar `flow_total`, `y_load_factor` e `inputs_snapshot` capturándolos del cálculo interno (§4.2.1).** La verificación halló que el endpoint `POST /control/recommend` no serializa esos tres campos (los calcula internamente pero no los devuelve). Se decide mantenerlos como columnas de `motor_decisions` —por la reproducibilidad/auditabilidad de RNF-SEC-01 (§8.3)— capturándolos del cálculo interno del motor y del snapshot del payload al persistir, no del cuerpo de la respuesta. Cerrar esa brecha es trabajo del componente de control (Delta-10).
- **V3 — Ratificación de que el sistema no opera en lazo cerrado autónomo (§5.1).** La verificación confirmó que el motor es una calculadora invocada a demanda. Se ratifica como **postura deliberada de la arquitectura objetivo** (no estado interino): el motor calcula a demanda y el operador activa; el lazo cerrado proactivo queda como trabajo futuro fuera del alcance del MVP (frontera de §4.3).
- **V4 — Integración Gemini fuera de la arquitectura objetivo; remoción diferida como saneamiento (§11; Delta-13).** La verificación confirmó una integración del frontend con la API de Gemini (`gemini-2.5-flash`) para reportes de incidentes, sin HU que la respalde y con envío de datos a un tercero (implicación de privacidad real). Se decide que la arquitectura objetivo del SDD **no la contempla**. La remoción del código del frontend es una **tarea de saneamiento diferida**, no ejecutada por esta decisión: basta con que el diseño declare que no es parte de la arquitectura objetivo. Se descarta "preservar y decidir luego" (deja viva una integración con riesgo de privacidad sin dueño) y "elevar a HU formal" (formalizar una dependencia de un tercero con datos de incidentes es una decisión de producto que no se toma de pasada en una verificación de SDD).

### Lo que NO cambia con DHU-021

- **No reescribe ninguna HU, TTH ni decisión técnica `D-`.** Es metodológica y de diseño de la documentación; el contenido sustantivo del backlog permanece.
- **La nomenclatura de roles (Delta-02) NO entra aquí.** Es una decisión de producto, no de redacción del SDD; se resuelve en DHU-022 para no mezclar naturalezas.

### Documentos relacionados
- `documentation/sdd/SDD_CEREBROVIAL.md` — documento cuya redacción y verificación consolida esta DHU; conserva un puntero a este registro.
- `documentation/sdd/SPECKIT_MAPPING.md` — mapeo brownfield del corpus a las plantillas de Spec Kit.
- `DECISIONS.md` — D-001 a D-009 (insumo del cuerpo del SDD).
- `AUDITORIA_HU_CODIGO.md` — origen de Delta-07 (realtime), Delta-10 (persistencia del motor) y Delta-13 (features huérfanas, Gemini).

---

## DHU-022 — Nomenclatura de roles del sistema

**Fecha:** 2026-05-20.
**Tipo:** Decisión de producto sobre nomenclatura del sistema (cierre de discrepancia de auditoría). Se registra en este documento por convención de trazabilidad de decisiones, pero no es metodológica de redacción de backlog.
**Estado:** Cerrada.
**Origen:** Delta-02 de `AUDITORIA_HU_CODIGO.md`.

### Contexto

La auditoría detectó una inconsistencia de nomenclatura de roles entre tres fuentes: la especificación de TTH-01 (CT-01.3) usa claims `role` con valores `operator/manager/admin` (inglés); la migración `99319147948b_add_users_table` define `role` como `sa.String()` sin restricción de enumerado; y el frontend (`AdminView`) muestra etiquetas en español con un tercer vocabulario ("Analista" en lugar de "Manager"/"Gerente"). Las tres Personas del producto definidas en `LEAN_INCEPTION_CEREBROVIAL.md`/`BACKLOG_OVERVIEW.md` son Operador, Gerente y Administrador.

### Decisión

Se fija un único conjunto canónico de identificadores de rol y una política de presentación:

1. **Claims técnicos canónicos en inglés:** `operator`, `manager`, `admin`. Son los valores que viajan en el token/claim y se validan en el backend.
2. **Mapeo a labels en español en el frontend:** la capa de presentación traduce `operator`→"Operador", `manager`→"Gerente", `admin`→"Administrador". Se elimina el vocabulario divergente ("Analista").
3. **Alcance de aplicación:** la nomenclatura se fija **antes** de implementar TTH-01 (autenticación), para que el modelo de usuarios, los claims y las vistas por rol nazcan alineados. La columna `role` debería restringirse al conjunto `{operator, manager, admin}` cuando se construya la autenticación.

### Lo que NO cambia con DHU-022

- No reescribe HUs ni TTH; fija la nomenclatura que TTH-01 y las HUs con acceso por rol consumirán.
- No ejecuta el cambio de código: es la decisión normativa; la implementación corresponde al sprint que aborde TTH-01 (hoy no iniciado, Delta-02).

### Documentos relacionados
- `AUDITORIA_HU_CODIGO.md` — Delta-02 (origen); se marca cerrado apuntando a DHU-022.
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-01 (autenticación JWT/bcrypt), consumidor de esta nomenclatura.
- `LEAN_INCEPTION_CEREBROVIAL.md` — Personas (Operador, Gerente, Administrador).
- `DHU-002` — reformulación del valor en HU de acceso diferenciado por rol (decisión relacionada de redacción).

---

---

## Resumen de impacto en los bloques redactados hasta la fecha

| Bloque | HUs | TTH | Decisiones aplicadas |
|---|---|---|---|
| Bloque A | HU-01 | TTH-01, TTH-02, TTH-03 | DHU-001, DHU-002, DHU-003, DHU-004, DHU-007 (retroactivo) |
| Bloque B | HU-02 a HU-09 | (ninguna nueva) | DHU-003, DHU-005 (refinada con A y B), DHU-006, DHU-007 |
| Bloque C | HU-10, HU-11, HU-12 (HU-13 eliminada por DHU-011) | TTH-04, TTH-05 | DHU-005, DHU-006, DHU-007, DHU-008, DHU-009, DHU-010, DHU-011 |
| Bloque D | HU-13, HU-14, HU-15 | (ninguna nueva del MVP1); TTH-06 agregada como Trabajos Futuros; CT-04.5 de TTH-04 ampliada | DHU-013 (clasificación), DHU-014 (decisiones de redacción) |
| Bloque E | (ninguna HU operativa) | TTH-07, TTH-08, TTH-09, TTH-10, TTH-11 | DHU-015 (clasificación HU/TTH del Bloque E con ampliación 4 → 5 TTH durante la redacción) |
| Bloque F | HU-16, HU-17 (F12+F13 fusionadas con F30 inglobada; F14) | (ninguna nueva) | DHU-016 (decisiones consolidadas de redacción del Bloque F en diez subsecciones) |
| MVP2 | HU-18, HU-19, HU-20, HU-21 (HU-09 cerrada previamente en `HU_BLOQUE_B.md`) | (ninguna nueva) | DHU-017 (decisiones consolidadas de redacción del MVP2 en diez subsecciones) |
| Transversal | — | — | DHU-012 (auditoría de coherencia documental, aplica a todos los bloques y documentos relacionados); DHU-018 (patrón "Resumen ejecutivo" aplicado retroactivamente a las 21 HUs, aditivo y sin modificar contenido sustantivo); DHU-019 (decisiones metodológicas para la redacción del documento RF/RNF, ejecuta la sesión dedicada que DHU-007 declaró pendiente; aditiva sobre las HUs en su pasada de referencias `→ RNF-XXX-NN` sobre los Candidatos a RNF); DHU-020 (semántica de ControlView, cierre de Delta-08; decisión de alineación especificación↔código que ratifica la semántica pasiva de HU-05 y alinea el código a ella, sin modificar la redacción de ninguna HU; afecta principalmente al Bloque B vía la cadena HU-05→HU-08); DHU-021 (decisiones metodológicas de redacción del SDD, 17 de redacción + 4 ajustes derivados de la verificación SDD↔repo; consolida el cierre del SDD sin reabrir HUs/TTH/`D-`); DHU-022 (nomenclatura de roles `operator/manager/admin` con labels en español, cierre de Delta-02; decisión de producto que TTH-01 y las HUs con acceso por rol consumirán) |

---

## Documentos relacionados

- `HU_BLOQUE_A.md` — Bloque A del Product Backlog (1 HU operativa).
- `HU_BLOQUE_B.md` — Bloque B del Product Backlog (8 HUs, 7 MVP1 + 1 MVP2).
- `HU_BLOQUE_C.md` — Bloque C del Product Backlog (3 HUs operativas: HU-10, HU-11, HU-12).
- `HU_BLOQUE_D.md` — Bloque D del Product Backlog (3 HUs operativas: HU-13, HU-14, HU-15).
- `HU_BLOQUE_E.md` — Bloque E del Product Backlog (0 HUs operativas; mapeo a TTH-07 a TTH-11 y decisiones tomadas durante la redacción).
- `HU_BLOQUE_F.md` — Bloque F del Product Backlog (2 HUs operativas: HU-16, HU-17; F30 inglobada como CAs).
- `HU_MVP2.md` — MVP2 del Product Backlog (HU-18, HU-19, HU-20, HU-21; HU-09 reside en `HU_BLOQUE_B.md`).
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-01 a TTH-11.
- `DECISIONS.md` — Decisiones técnicas del producto (D-001 a D-009). No se solapa con este documento.
- `LEAN_INCEPTION_CEREBROVIAL.md` — Personas, journeys, MVP Canvas (insumos para identificar sujetos válidos).
- `FEATURE_BACKLOG_DETALLADO.md` — Origen de las features que se mapean a HUs y TTH.
- `EVOLUCION_TESIS.md` — Narrativa de las 4 fases del proyecto; sección 8 contiene tabla de Trabajos Futuros.
- `CONTROL.md` — Sustentación teórica del motor adaptativo (consumido por TTH-10).
- `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` — Documento normativo denso con catálogo de 22 RF y 53 RNF clasificados según ISO/IEC 25010:2023, redactado el 2026-05-18 ejecutando DHU-007 según las decisiones metodológicas consolidadas en DHU-019.
- `RF_RNF_LITE.md` — Versión lite de lectura humana del documento RF/RNF, derivado conforme al modelo de dos documentos cerrado en DHU-019 subsección H.
- `AUDITORIA_HU_CODIGO.md` — Auditoría del estado del código por HU/TTH (Fase 4.1); origen de los deltas, incluido Delta-08 cerrado por DHU-020.
- `REPORTE_PLANIFICACION_SPRINT_4.md` — Síntesis de planificación del Sprint 4; origen del riesgo R1, resuelto por DHU-020.
