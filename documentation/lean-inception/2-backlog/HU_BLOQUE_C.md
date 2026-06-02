# Historias de Usuario — Bloque C (Cerradas)

> Tercera entrega del Product Backlog del proyecto CerebroVial.
>
> **Estado:** Bloque C cerrado y aprobado. Bloques A, B, D, E y F del MVP1 cerrados, y MVP2 también cerrado el 2026-05-16 (DHU-017). **Con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 22 HU operativas (HU-01 a HU-22) + 12 TTH (TTH-01 a TTH-12), tras la ampliación DHU-028 (HU-22 + TTH-12).** Pendiente: documento RF/RNF (DHU-007), Planning Poker, MoSCoW, implementación SCRUM del MVP1.
>
> **Fecha de cierre:** 2026-05-13
> **Fecha de actualización v2:** 2026-05-17 (DHU-018 aplicada retroactivamente: Resumen ejecutivo en HU-10, HU-11, HU-12)

---

## Contexto

Este documento contiene las Historias de Usuario del **Bloque C — Operador, operación degradada** del Sequencer del Lean Inception (ver `LEAN_INCEPTION_CEREBROVIAL.md`, sección 9, y `FEATURE_BACKLOG_DETALLADO.md`, Bloque C).

Las HUs se redactan en el formato del documento de referencia académica (`Desarrollo_Agil.pdf`, Tablas 9 y 13): "Como X, quiero Y, para Z" con criterios de aceptación Given-When-Then.

Las HUs del Bloque C siguen las reglas metodológicas establecidas y refinadas durante la redacción:

- **DHU-001 a DHU-004** (del Bloque A): sujetos válidos, exclusión del Equipo de Desarrollo, TTH como categoría separada.
- **DHU-005** (refinada en el Bloque B): principio de robustez ante interrupción de fuente, con Casos A y B.
- **DHU-006:** HUs agnósticas a la implementación.
- **DHU-007:** RNF declarados como tales en sección específica al final de cada HU.
- **DHU-008** (cerrada en el Bloque C): distinción arquitectónica entre componente caído, modo degradado y lógica de fallback.
- **DHU-009** (cerrada en el Bloque C): relación entre marca pasiva del Bloque B y alerta activa del Bloque C como complementarias y no duplicadas.
- **DHU-010** (cerrada en el Bloque C): criterios para clasificar F26 y F27 como TTH.
- **DHU-011** (cerrada en el Bloque C): eliminación de HU-13 y cobertura de F25 por composición.
- **DHU-018** (aplicada retroactivamente el 2026-05-17): patrón "Resumen ejecutivo" al inicio de cada HU para uniformidad de lectura. Aditiva, no modifica contenido sustantivo.

Ver `DECISIONS_HU.md` para fundamentación completa.

---

## Mapeo de features del Bloque C

Las 6 features del Bloque C (F22 a F27) se mapearon a 3 HUs operativas + 2 TTH según el siguiente criterio (DHU-008 + DHU-010 + DHU-011):

| HU / TTH | Título | Features que cubre |
|---|---|---|
| HU-10 | Alerta activa transversal del estado operativo del sistema | F22 |
| HU-11 | Vista del estado operativo de los componentes del sistema | F23 (+ F25 absorbida en CA-11.9, ver DHU-011) |
| HU-12 | Explicación del modo degradado activo y sus implicaciones operativas | F24 |
| TTH-04 | Lógica de fallback en cascada del sistema | F26 |
| TTH-05 | Configuración de tiempos preconfigurados para degradado nivel 3 | F27 |

**Total Bloque C:** 3 HUs operativas + 2 TTH.

**Sobre F25:** según DHU-011, F25 (indicación contextual del modo degradado en cada panel afectado) queda cubierta funcionalmente por la composición de HU-10 (alerta transversal) + HU-11 con el refinamiento de CA-11.9 (resalte visual de componentes no-OK) + HU-12 (explicación compuesta) + marcas pasivas del Bloque B (DHU-005 Casos A y B). No se redacta como HU independiente. Este patrón es coherente con la cobertura por composición de F02 (Bloque B), F30 y F31 (inglobaciones del Bloque A).

---

## HU-10 — Alerta activa transversal del estado operativo del sistema

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | recibir una alerta visual persistente y transversal cada vez que el sistema entra en un estado operativo degradado o de falla total, independientemente de la vista que esté usando |
| **Para** | saber en todo momento si el sistema está operando con capacidades completas, con capacidades reducidas, o si dejó de operar, y poder ajustar mi supervisión, calibrar mi confianza en los datos mostrados, y escalar oportunamente cuando corresponda |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** F22 (Indicador visible de estado degradado). Ingloba como CA la persistencia de transiciones de estado operativo, aplicando la misma regla cerrada en el Bloque A que llevó a inglobar F31 en HU-08.

### Resumen ejecutivo

**Qué entrega:** alerta visual persistente y transversal a todas las vistas del Operador cuando el sistema entra en estado degradado (niveles 1, 2 o 3) o falla total. Banner con distinción visual por nivel, reconocible en estados degradados (colapsa a forma discreta) pero no en falla total. Persiste transiciones para auditoría.

**CAs críticos:** CA-10.2 (alerta visible en estados degradados con código de color amarillo/naranja/rojo), CA-10.3 (alerta diferenciada en falla total, no reconocible), CA-10.7 (sustrato técnico inglobado: persistencia de transiciones con timestamp, estado anterior, estado nuevo, causa raíz), CA-10.8 (resiliencia: la activación del fallback nunca se detiene por fallo del registro de auditoría), CA-10.9 (DHU-005 Caso B aplicado al monitor del estado operativo).

**Estructura de CAs:** estados de la alerta (CA-10.1 a CA-10.3) → reconocimiento y reescalada (CA-10.4 a CA-10.5) → retorno a normal (CA-10.6) → persistencia inglobada y resiliencia (CA-10.7, CA-10.8) → robustez DHU-005 Caso B (CA-10.9) → control de acceso (CA-10.10).

**Dependencias:** consume el estado operativo expuesto por TTH-04 (CT-04.4). Complementaria, no duplicada, a las marcas pasivas del Bloque B (DHU-009: marca pasiva = "este dato específico"; alerta transversal = "el sistema completo"). HU-11 cubre detalle por componente; HU-12 cubre explicación compuesta. HU-21 (MVP2) dispara escalamiento al Administrador desde el banner expandido de esta HU.

**Notas clave:** 5 estados operativos formalizados en DHU-008 (normal + 3 degradados + falla total). Reconocimiento de la alerta se persiste con identidad y timestamp para auditoría futura. CA-10.8 codifica un principio operativo crítico: el log de auditoría nunca puede detener fallbacks. Aplica DHU-005 Caso B (CA-10.9), DHU-006, DHU-008, DHU-009, DHU-012 (vocabulario de niveles).

### Descripción

Las HUs del Bloque B marcan pasivamente cada panel del dashboard cuando la fuente o el componente específico de ese panel deja de responder (DHU-005 Casos A y B). Esa marca pasiva responde la pregunta *"¿qué pasa con este dato específico que estoy mirando?"*. Pero existe una pregunta operativa distinta que ninguna HU del Bloque B responde: *"¿en qué estado está operando el sistema completo en este momento?"*. Esta HU cubre esa pregunta.

El sistema puede encontrarse en uno de varios estados operativos según qué componentes están funcionando y qué mecanismos de fallback se han activado. Cuando todos los componentes operan, el sistema está en operación normal y no hay alerta. Cuando alguno falla, el sistema entra en un estado degradado (nivel 1, 2 o 3) o, si no hay fallback aplicable, en falla total. En cualquiera de esos estados no normales, el Operador necesita saberlo de forma inmediata y transversal: la información no puede depender de qué panel esté abierto en ese momento.

La alerta se manifiesta como un banner persistente en la parte superior del dashboard, visible desde cualquier vista del sistema. El estilo visual del banner distingue cuatro estados:

- **Degradado nivel 1** — color amarillo. El sistema opera con capacidades reducidas menores; típicamente un componente periférico no responde y el fallback aplicado preserva la mayoría de funcionalidades.
- **Degradado nivel 2** — color naranja. El sistema opera con capacidades reducidas significativas; un componente principal no responde y el fallback aplicado degrada la calidad de la operación.
- **Degradado nivel 3** — color rojo. El sistema opera pero al mínimo posible (tiempos preconfigurados, sin adaptación). Es el último escalón con operación.
- **Falla total** — estilo visual claramente distinto al de los tres anteriores (rojo intenso, ícono inequívoco, texto explícito tipo "Sistema fuera de operación"). Comunica que el sistema **no está operando** y requiere escalamiento del Operador.

El banner es persistente (no desaparece solo) y permanece visible hasta que el sistema retorne a operación normal. El Operador puede **reconocer** la alerta en los estados degradados (nivel 1, nivel 2, nivel 3), lo que colapsa el banner a una forma discreta pero todavía visible (barra delgada o ícono fijo en cabecera). La alerta de falla total **no** es reconocible: permanece en forma prominente hasta que el sistema sea restablecido, porque la inacción del Operador en ese estado no es aceptable.

Cualquier escalada del estado operativo (por ejemplo, de degradado 1 a degradado 2, o de cualquier estado degradado a falla total) restaura el banner a su forma prominente automáticamente, independientemente de si el Operador había reconocido el estado anterior. Esto garantiza que el Operador no pierda visibilidad de un deterioro adicional por haber reconocido un estado previo.

La HU también declara que cada transición de estado operativo del sistema completo se persiste de forma durable. La consulta de ese histórico es responsabilidad de HUs de otros bloques (Bloque F para reportería ejecutiva, eventualmente Bloque D para diagnóstico técnico); HU-10 sólo declara que la persistencia existe.

### Criterios de aceptación

- **CA-10.1:** Dado que el Operador ha iniciado sesión y el sistema opera en estado normal, cuando el Operador está observando cualquier vista del sistema, entonces no se muestra ninguna alerta del estado operativo del sistema completo.

- **CA-10.2:** Dado que el sistema transita desde operación normal a cualquier estado degradado (nivel 1, nivel 2 o nivel 3), cuando la transición ocurre, entonces el sistema muestra una alerta visual persistente en la parte superior del dashboard, visible desde cualquier vista del sistema, con un estilo visual distintivo según el nivel (amarillo para nivel 1, naranja para nivel 2, rojo para nivel 3) e incluye un texto identificable que comunica el estado actual.

- **CA-10.3:** Dado que el sistema transita a falla total, cuando la transición ocurre, entonces el sistema muestra una alerta visual persistente con un estilo claramente diferenciado de los estados degradados (combinación de color, ícono y texto inequívoca de "sistema fuera de operación"), visible desde cualquier vista del sistema. Esta alerta no se puede reconocer y permanece en forma prominente hasta que el sistema retorne a un estado operativo.

- **CA-10.4:** Dado que se muestra una alerta de estado degradado en su forma prominente, cuando el Operador la reconoce mediante la acción correspondiente, entonces el banner se colapsa a una forma discreta pero todavía visible que comunica el estado vigente, sin desaparecer del todo. El sistema registra el reconocimiento con la identidad del Operador y la marca de tiempo, para auditoría.

- **CA-10.5:** Dado que existe una alerta degradada en estado reconocido (colapsado), cuando el sistema escala a un estado degradado de mayor severidad o a falla total, entonces el banner vuelve automáticamente a su forma prominente, sin requerir acción del Operador y sin importar quién había reconocido el estado anterior.

- **CA-10.6:** Dado que el sistema retorna desde cualquier estado degradado o de falla total a operación normal, cuando la transición ocurre, entonces la alerta desaparece automáticamente del dashboard, sin requerir acción del Operador.

- **CA-10.7:** Dado que el sistema cambia de estado operativo (entre normal, cualquiera de los estados degradados y falla total), cuando la transición ocurre, entonces el sistema persiste de forma durable la transición con al menos: marca de tiempo, estado anterior, estado nuevo y causa raíz (identificación del componente o condición que disparó la transición). Esto presupone que existe un registro durable de transiciones de estado operativo del sistema completo.

- **CA-10.8:** Dado que el sistema necesita persistir una transición de estado operativo, cuando ocurre un fallo temporal en la escritura al registro, entonces la transición de estado se aplica de todos modos al sistema (por ejemplo, el sistema entra efectivamente en degradado nivel 3) y se registra en un mecanismo de respaldo para ser persistida cuando el registro vuelva a estar disponible. La operación del sistema y la activación de fallbacks nunca se detiene por una falla del registro de auditoría.

- **CA-10.9:** Dado que el componente responsable de determinar el estado operativo del sistema deja de responder por cualquier causa, cuando el Operador está observando el sistema, entonces se muestra de forma transversal una indicación de que el estado operativo del sistema no puede confirmarse en este momento, junto con el último estado conocido y el tiempo transcurrido desde la última confirmación (DHU-005 Caso B aplicado al monitor de estado operativo del sistema).

- **CA-10.10:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder a cualquier vista del sistema, entonces el sistema lo redirige a la pantalla de login. Las alertas de estado operativo no se entregan a usuarios no autenticados.

### Notas técnicas

- **Relación con HUs del Bloque B (DHU-009):** esta HU **no duplica** la marca pasiva de los paneles del Bloque B. Un mismo evento físico puede disparar legítimamente ambas señales y son complementarias: la marca pasiva del Bloque B le dice al Operador *"este dato específico ya no podemos garantizarlo"*; la alerta transversal de HU-10 le dice *"el sistema completo está operando con capacidades reducidas, sabelo aunque estés mirando otra pantalla"*. Las dos transportan información distinta y útil al mismo tiempo.
- **Relación con HU-11 (vista de estado de componentes):** HU-10 comunica el estado operativo del sistema completo en términos de modo activo (normal, degradado 1, degradado 2, degradado 3, falla total). HU-11 expone qué componente específico está caído. Es esperable que el banner de HU-10 ofrezca un enlace o acción "Ver detalle de componentes" que abra la vista de HU-11, pero esa integración se cierra en el sprint.
- **Relación con HU-12 (explicación del modo degradado):** HU-10 muestra el estado y su nivel; HU-12 explica qué significa operativamente. Igual que la relación HU-05/HU-06, el banner de HU-10 puede ofrecer un acceso directo a la explicación de HU-12.
- **Origen del estado operativo:** la HU es agnóstica al mecanismo que determina el estado del sistema. La lógica que detecta caídas de componentes y aplica cascadas de fallback es responsabilidad de TTH-04. La HU sólo consume el estado actual y las transiciones que ese mecanismo expone.
- **Forma colapsada (CA-10.4):** patrón típico es una barra delgada en la cabecera con color del estado, ícono y texto corto (por ejemplo *"Sistema en modo degradado nivel 1"*). El detalle exacto se define en el prototipado del dashboard. Lo importante es que **siga siendo visible y reconocible al ojo**; no es aceptable convertir el reconocimiento en una desaparición efectiva.
- **Persistencia del reconocimiento (CA-10.4):** se persiste la identidad del Operador y el timestamp del reconocimiento para auditoría. Esto permite reconstruir más tarde quién estaba al tanto del estado degradado en cada momento. La consulta de ese histórico no entra en MVP1; el dato queda capturado para uso futuro.
- **Mecanismo de entrega:** la alerta y sus transiciones se entregan al frontend por el mismo canal de eventos en tiempo real usado por otras notificaciones (SSE o WebSocket, decisión técnica concreta a cerrar en el sprint).
- **Accesibilidad (referencia F22):** el color **no** es el único indicador de severidad. Cada estado se distingue también por ícono y por texto descriptivo. Esto preserva la usabilidad para Operadores con limitaciones de visión cromática y refuerza la legibilidad del estado a primera vista.
- **Diferencia entre el registro de HU-10 y el registro de HU-08:** son dos registros distintos. HU-08 persiste decisiones del motor adaptativo (estrategia A → estrategia B). HU-10 persiste transiciones del estado de salud operativa del sistema (normal → degradado 1, degradado 1 → degradado 3, etc.). Ambos coexisten y son consultables por separado.
- **Esquema mínimo del registro de transiciones:** `timestamp`, `estado_anterior` (enum), `estado_nuevo` (enum), `causa_raiz` (identificador interno del componente o condición disparadora). Campos adicionales (fallback aplicado, métricas del momento, etc.) se evalúan más adelante; el MVP1 sostiene la trazabilidad con los cuatro campos básicos.
- **Texto humano de la causa raíz:** el `causa_raiz` del registro contiene el identificador técnico del componente o condición. El texto explicativo que ve el Operador (HU-12) se genera a partir de ese identificador con un catálogo de textos predefinidos similares a HU-06; no se almacena texto humano en el registro de transiciones.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** latencia máxima entre la transición real del estado operativo y la aparición de la alerta en el dashboard del Operador ≤ 5 s.
- **RNF de disponibilidad:** la alerta debe ser transversal y consistente en todas las vistas del sistema. No debe haber vistas donde la alerta no aparezca cuando corresponde.
- **RNF de robustez:** comportamiento ante caída del componente que determina el estado operativo (CA-10.9, DHU-005 Caso B). En ese caso la HU declara que se muestra una indicación de "estado no confirmado" en lugar de asumir falsamente estado normal.
- **RNF de persistencia / auditoría:** cada transición de estado operativo debe persistirse de forma durable en el momento en que se produce. La pérdida de transiciones por fallo de escritura es inaceptable (CA-10.8 cubre esto con mecanismo de respaldo).
- **RNF de continuidad operativa:** la activación de fallbacks y la operación del sistema no se detiene por fallos del registro de auditoría (CA-10.8).
- **RNF de usabilidad:** distinción visual entre los cuatro estados (degradado 1, degradado 2, degradado 3, falla total) debe ser inmediata, no requerir lectura detenida. Se valida con prueba de usuario.
- **RNF de accesibilidad:** el estado debe ser identificable sin depender exclusivamente del color (combinación color + ícono + texto). Probablemente conforme a WCAG 2.1 nivel AA.
- **RNF de inmutabilidad:** las entradas del registro de transiciones de estado operativo no deben modificarse después de escritas (auditoría confiable). En línea con el RNF equivalente de HU-08.
- **RNF de trazabilidad:** cada entrada del registro debe ser unívocamente identificable y referenciable desde otras partes del sistema (por ejemplo, desde el reporte ejecutivo del Gerente en el Bloque F).

---

## HU-11 — Vista del estado operativo de los componentes del sistema

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | consultar en una vista dedicada el estado operativo actual de cada componente del sistema, expresado en términos del impacto que tiene sobre mi operación |
| **Para** | identificar rápidamente qué componente está afectando la operación cuando el sistema entra en estado degradado o de falla, y también para confirmar proactivamente que todos los componentes están operando bien cuando no hay alertas |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** F23 (Vista simplificada de estado de componentes — Operador). Absorbe el espíritu visual de F25 vía CA-11.9 (DHU-011).

### Resumen ejecutivo

**Qué entrega:** vista dedicada para el Operador donde consulta el estado operativo de cada componente del sistema en escala cualitativa (OK / Degradado / Fuera de servicio) con descripción del impacto operativo en lenguaje no técnico. Resaltado visual de componentes no-OK. Accesible siempre, no solo bajo alerta.

**CAs críticos:** CA-11.1 (presentación cualitativa por componente), CA-11.3 (texto de impacto operativo cuando no-OK), CA-11.7 (DHU-005 Caso B aplicado al monitor de componentes), CA-11.9 (resaltado visual de no-OK absorbiendo F25 por DHU-011), CA-11.8 (redirección al login).

**Estructura de CAs:** presentación cualitativa por componente (CA-11.1 a CA-11.4) → integración con HU-10 y accesibilidad permanente (CA-11.5 a CA-11.6) → robustez DHU-005 Caso B (CA-11.7) → control de acceso (CA-11.8) → resaltado de no-OK por DHU-011 (CA-11.9).

**Dependencias:** consume CT-04.5 de TTH-04 (mismo endpoint que HU-13 del Administrador, con presentación distinta según DHU-013). Complementaria a HU-10 (estado global) y HU-12 (explicación compuesta). HU-13 del Bloque D expone los mismos componentes con campos técnicos adicionales (latencia, fallos recientes) para el Administrador.

**Notas clave:** F25 (indicación contextual del modo degradado en cada panel) quedó cubierta por composición de HU-10 + CA-11.9 + HU-12 + marcas pasivas del Bloque B, sin HU dedicada (DHU-011). La vista usa nombres legibles, no identificadores internos. Es de consulta solo, sin acciones correctivas (esas serían HUs separadas en futuro).

### Descripción

La alerta transversal de HU-10 le dice al Operador que el sistema está operando con capacidades reducidas, pero deliberadamente no entra al detalle de qué componente específico falló. Recorrer todos los paneles del Bloque B para inferirlo a partir de las marcas pasivas individuales (DHU-005 Casos A y B) es laborioso y propenso a malinterpretación. HU-11 provee un lugar único donde el Operador consulta directamente el estado de cada componente y obtiene la respuesta a *"¿qué componente está afectando la operación?"* en una sola mirada.

La vista es **complementaria al panel del Administrador** (responsabilidad del Bloque D, F17): la vista del Administrador expone detalle técnico (latencias, errores, métricas de salud, logs); la vista del Operador expone únicamente el estado operativo en términos de impacto perceptible para la operación. El Operador no consume métricas técnicas: si necesita ese detalle, escala al Administrador.

La vista lista los componentes con nombres legibles ("Detección de tráfico", "Predicción de congestión", "Motor de decisión", "Tiempos de respaldo", "Registro de eventos") y un estado cualitativo en tres niveles: **OK** (componente operando normalmente), **Degradado** (componente operando con limitaciones o con fallback aplicado, pero el sistema sigue funcionando) y **Fuera de servicio** (componente no responde, y el sistema opera sin él o ha escalado a un fallback que no lo requiere). Cada componente incluye una breve descripción en lenguaje no técnico de qué hace y, cuando no está OK, qué efecto tiene esa condición sobre la operación.

La vista es accesible **siempre**, tanto desde un punto fijo del menú principal del Operador como mediante un acceso directo desde el banner de alerta de HU-10 cuando éste está visible. La accesibilidad permanente permite al Operador confirmar proactivamente que el sistema está sano, no sólo reaccionar cuando aparece una alerta.

### Criterios de aceptación

- **CA-11.1:** Dado que el Operador ha iniciado sesión, cuando ingresa a la vista de estado de componentes, entonces el sistema muestra una lista de los componentes con impacto operativo perceptible, cada uno con su nombre legible, una breve descripción de su función y su estado operativo actual en escala cualitativa (OK, Degradado o Fuera de servicio).

- **CA-11.2:** Dado que el Operador está observando la vista, cuando un componente cambia de estado operativo, entonces el estado mostrado se actualiza automáticamente sin necesidad de recargar la página, con una latencia máxima de 5 segundos desde que el cambio se produce.

- **CA-11.3:** Dado que un componente está en estado Degradado o Fuera de servicio, cuando el Operador consulta su entrada en la vista, entonces el sistema muestra, junto al estado, un texto breve en lenguaje no técnico que describe el efecto operativo de esa condición (por ejemplo, *"La predicción no está disponible; el sistema opera sólo con observación en tiempo real"*).

- **CA-11.4:** Dado que el Operador puede acceder a la vista en cualquier momento, cuando todos los componentes están operando normalmente, entonces el sistema muestra cada componente con estado OK, sin ocultarlos ni colapsarlos, para que el Operador pueda confirmar visualmente que el sistema completo está sano.

- **CA-11.5:** Dado que el banner de alerta de HU-10 está visible, cuando el Operador interactúa con el acceso directo del banner hacia el detalle de componentes, entonces el sistema lo lleva a la vista de HU-11 sin requerir navegación adicional por el menú.

- **CA-11.6:** Dado que la vista es accesible permanentemente, cuando el Operador navega por el menú principal del dashboard, entonces existe un acceso permanente a la vista de estado de componentes, independientemente de si hay alerta de HU-10 activa o no.

- **CA-11.7:** Dado que la fuente que provee los estados de componentes deja temporalmente de responder por cualquier causa, cuando el Operador está observando la vista, entonces el sistema mantiene en pantalla los últimos estados conocidos de cada componente, los marca visualmente como "no confirmados" e indica el tiempo transcurrido desde la última confirmación (DHU-005 Caso B aplicado al monitor de componentes).

- **CA-11.8:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder a la vista de estado de componentes, entonces el sistema lo redirige a la pantalla de login.

- **CA-11.9:** Dado que uno o más componentes están en estado Degradado o Fuera de servicio, cuando el Operador observa la vista de estado de componentes, entonces el sistema resalta visualmente las entradas correspondientes de manera que el Operador pueda identificarlas de un vistazo sin tener que leer el estado de cada componente uno por uno. El resalte combina al menos color y posición destacada (componentes no-OK ordenados al inicio de la vista, o agrupados visualmente).

### Notas técnicas

- **Relación con HU-13 (Bloque D, vista técnica del Administrador):** las dos vistas comparten la lógica subyacente de health check de componentes (cuáles existen, cómo se monitorean, cómo se determina su estado), pero las **presentaciones son distintas**. La vista del Operador (HU-11) presenta únicamente nombre legible, estado cualitativo (OK/Degradado/Fuera de servicio) e impacto operativo. La vista del Administrador (HU-13) presenta detalle técnico (métricas de latencia, fallos recientes, identificador interno, timestamp de última evaluación exitosa). Es razonable que el backend exponga una API que sirva a ambas vistas y que cada frontend consuma lo que corresponde a su rol; en la práctica ambas HUs consumen el mismo endpoint CT-04.5 de TTH-04 con presentación adaptada (DHU-013).
- **Catálogo de componentes:** los componentes listados son aquellos cuyo estado tiene un efecto perceptible sobre lo que el Operador ve y opera. Componentes puramente internos sin impacto operativo perceptible (cachés, colas de mensajes, etc.) no aparecen en HU-11; pertenecen al detalle técnico del Administrador. La lista exacta se cierra en el sprint pero el principio queda anclado en esta HU.
- **Mapeo entre estado de componente y estado operativo del sistema (HU-10):** el estado operativo global del sistema (operación normal / degradado 1 / degradado 2 / degradado 3 / falla total) es derivado de los estados individuales de componentes vía la lógica de fallback en cascada (responsabilidad de TTH-04). La HU-11 no implementa esa derivación; sólo muestra los estados que la lógica de TTH-04 expone.
- **Textos de impacto operativo (CA-11.3):** catálogo predefinido de textos breves, similar al patrón de HU-06 y HU-12. Se mantiene un texto por combinación de componente y estado degradado. No es generación dinámica.
- **Distinción explícita respecto a HU-12:** los textos de CA-11.3 son una etiqueta por componente individual ("qué hace el sistema sin esta pieza"). HU-12 cubre la explicación compuesta del modo degradado del sistema completo, integrando componente disparador, fallback aplicado e implicación operativa. Las dos HUs son complementarias y no duplicadas.
- **Nombres legibles vs identificadores internos:** los nombres mostrados al Operador son legibles (*"Detección de tráfico"*); los identificadores internos del componente (*"vision_service"*) viven en backend y en el registro de transiciones de HU-10. El mapeo entre uno y otro se cierra en el sprint.
- **Acceso desde el banner (CA-11.5):** el banner de HU-10 ofrece un acceso directo a HU-11 que abrevia la navegación. Si el Operador llega a HU-11 desde el banner, es razonable que la vista resalte por defecto el componente que disparó la alerta vigente; este refinamiento se cierra en el prototipado del dashboard y no es bloqueante para el CA.
- **Resalte visual de componentes no-OK (CA-11.9):** la implementación combina color (coherente con la paleta de la alerta de HU-10), ícono y posición. La decisión exacta de ordenamiento (mover no-OK al inicio, agruparlos, ambos) se cierra en el prototipado del dashboard. Coherente con DHU-011, que estableció que HU-11 absorbe el espíritu visual de F25 en lugar de crear HU dedicada.
- **Mecanismo de actualización:** mismo canal de eventos en tiempo real que el resto del Bloque B y la HU-10 (SSE o WebSocket, decisión técnica concreta a cerrar en el sprint).

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** latencia máxima entre cambio de estado de un componente y actualización en la vista del Operador ≤ 5 s (CA-11.2).
- **RNF de robustez:** comportamiento ante caída del monitor de componentes (CA-11.7, DHU-005 Caso B). Coherente con CA-10.9 de HU-10.
- **RNF de usabilidad:** la vista debe permitir identificar de un vistazo qué componente está afectado, en caso de degradación. Probablemente se valida con prueba de usuario contrastando el tiempo de identificación con y sin la vista.
- **RNF de usabilidad ampliado (CA-11.9):** el resalte visual debe permitir identificar componentes no-OK en una vista panorámica sin lectura detallada de cada entrada. Probablemente se valida con prueba de usuario midiendo tiempo de identificación.
- **RNF de accesibilidad:** el estado de cada componente debe ser identificable sin depender exclusivamente del color (combinación de color + ícono + texto), coherente con el RNF de accesibilidad de HU-10.
- **RNF de mantenibilidad:** el catálogo de textos de impacto operativo (CA-11.3) debe ser extensible sin requerir cambios al código del frontend ni del backend de monitoreo (configurable como dato). Coherente con el RNF de mantenibilidad del catálogo de plantillas de HU-06.
- **RNF de separación de roles:** el contenido expuesto en HU-11 (Operador) y el de HU-13 (Administrador) deben mantenerse separados en presentación, aunque puedan compartir fuente de datos. La separación es por presentación, no por datos.

---

## HU-12 — Explicación del modo degradado activo y sus implicaciones operativas

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | leer una explicación contextual y comprensible del modo degradado en el que está operando el sistema, integrando qué falló, qué hace el sistema en respuesta y qué capacidad operativa se perdió |
| **Para** | entender de forma rápida y autocontenida el estado actual del sistema completo, calibrar el nivel de supervisión que requiere mi turno mientras dure esta condición, y decidir si corresponde escalar al Administrador |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** F24 (Mensaje explicativo del modo degradado activo).

### Resumen ejecutivo

**Qué entrega:** explicación textual compuesta al Operador del modo degradado activo, integrando tres elementos: qué disparó el modo (componente o condición), qué fallback está activo, y qué capacidad operativa se perdió con implicación para la supervisión. Texto curado, plantillado con catálogo predefinido.

**CAs críticos:** CA-12.1 (texto compuesto con los 3 elementos), CA-12.2 (actualización ≤ 5 s al transitar entre estados), CA-12.4 (texto genérico de respaldo si la combinación no tiene plantilla específica), CA-12.5 (robustez ante caída del componente de explicación — DHU-005 Caso B).

**Dependencias:** consume el estado operativo de TTH-04 y el catálogo de plantillas (probablemente el mismo componente que produce HU-06). Complementaria a HU-10 (alerta) y HU-11 (vista por componente). HU-21 (MVP2) dispara escalamiento al Administrador desde esta HU cuando el texto recomienda escalar. Aplica DHU-005 Caso B, DHU-006.

**Notas clave:** distinción explícita respecto a HU-06: HU-06 explica decisiones del motor adaptativo **en operación normal**; HU-12 explica el **estado degradado del sistema completo**. Las dos son catálogos de plantillas pero cubren universos disjuntos. **No** se usa NLP ni explicabilidad de IA: catálogo curado de 4-6 textos según backlog detallado. CA-12.4 garantiza que el Operador nunca vea panel vacío.

### Descripción

La alerta transversal de HU-10 comunica que el sistema entró en un estado operativo no normal y lo identifica por nivel (degradado 1, degradado 2, degradado 3, falla total). La vista de componentes de HU-11 le permite al Operador localizar qué pieza específica está afectada. Pero ninguna de las dos cubre la pregunta operativa que el Operador necesita responderse para ajustar su forma de trabajar mientras dure la condición: *"¿qué significa este modo, qué está haciendo el sistema en respuesta, y qué tengo que cambiar en mi manera de supervisar?"*. Esta HU cubre esa pregunta.

La explicación es un texto compuesto que integra tres elementos:

1. **Qué disparó el modo degradado.** Identifica el componente o condición que causó la transición, en lenguaje del dominio operativo (no técnico).
2. **Qué fallback está activo.** Describe qué hace el sistema en lugar del comportamiento normal: con qué fuente alternativa opera, con qué modelo de respaldo, o si está aplicando tiempos preconfigurados.
3. **Qué capacidad operativa se perdió y qué implica para la supervisión.** Indica al Operador qué funcionalidad ya no está disponible (predicción anticipada, adaptación al tráfico, etc.) y, cuando corresponde, una recomendación breve de supervisión más cercana o de escalamiento al Administrador.

El alcance del texto es deliberadamente **acotado y plantillado**, igual que en HU-06: catálogo predefinido de textos por combinación de componente disparador y nivel de degradación, con sustitución de variables cuando es relevante. No se trata de generación dinámica, ni de explicabilidad de IA, ni de razonamiento causal. Es texto curado que un humano (probablemente el equipo del proyecto, en coordinación con un Operador real) escribió previamente para cada caso típico.

La HU es independiente del componente que mostrará el texto en la UI. El texto puede mostrarse como un panel propio, como sección expandible del banner de HU-10, o como ambos. Esa decisión es de prototipado del dashboard. La HU sólo declara que el texto compuesto **debe estar disponible y debe ser legible cuando el sistema está en un estado degradado o de falla total**.

### Criterios de aceptación

- **CA-12.1:** Dado que el Operador ha iniciado sesión y el sistema está operando en un estado degradado (nivel 1, nivel 2 o nivel 3) o en falla total, cuando el Operador accede a la explicación del modo degradado activo, entonces el sistema muestra un texto compuesto que integra: el componente o condición que disparó el modo actual, el fallback aplicado en respuesta, y la capacidad operativa perdida con su implicación para la supervisión.

- **CA-12.2:** Dado que el sistema cambia entre estados operativos no normales (por ejemplo, escalada de degradado nivel 1 a degradado nivel 2, o de degradado nivel 3 a falla total), cuando la transición ocurre, entonces la explicación se actualiza automáticamente para reflejar el modo vigente, con una latencia máxima de 5 segundos desde que la transición se produce.

- **CA-12.3:** Dado que el sistema retorna desde un estado degradado o de falla total a operación normal, cuando la transición ocurre, entonces la explicación deja de mostrarse automáticamente, sin requerir acción del Operador.

- **CA-12.4:** Dado que el sistema entra en una combinación de componente disparador y nivel de degradación que no tiene una plantilla específica en el catálogo, cuando se muestra la explicación, entonces el sistema presenta un texto genérico de respaldo que indica al menos el nivel del modo degradado vigente y, si está disponible, el componente afectado, sin dejar el panel vacío.

- **CA-12.5:** Dado que el componente que produce la explicación deja de responder por cualquier causa, cuando el Operador accede a la vista de la explicación, entonces el sistema mantiene en pantalla la última explicación conocida, la marca visualmente como "no confirmada" e indica el tiempo transcurrido desde la última confirmación (DHU-005 Caso B aplicado al componente de explicación del modo degradado).

- **CA-12.6:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder a la explicación del modo degradado, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Distinción explícita respecto a HU-06:** HU-06 explica decisiones del motor adaptativo **en operación normal** (por qué eligió la estrategia A en lugar de la B). HU-12 explica el **estado degradado del sistema completo** (qué falló, qué fallback está activo, qué se perdió). Las dos son catálogos de textos predefinidos pero cubren universos disjuntos: HU-06 nunca se dispara en estado degradado, HU-12 nunca se dispara en estado normal.
- **Distinción explícita respecto a HU-11 (CA-11.3):** los textos de CA-11.3 son **etiquetas individuales por componente** ("qué hace el sistema sin esta pieza"). HU-12 es la **explicación compuesta del modo del sistema completo**, integrando disparador, fallback e implicación operativa. Las dos HUs son complementarias y no duplicadas; la regla se refleja también en la nota refinada de HU-11.
- **Catálogo de plantillas:** se define un conjunto acotado de textos predefinidos que cubren las combinaciones típicas de (componente disparador × nivel de degradación). El backlog detallado de F24 sugiere mantener el catálogo pequeño (4-6 mensajes) y no intentar cubrir todas las combinaciones posibles. Las combinaciones no cubiertas caen al texto genérico de respaldo (CA-12.4).
- **Variables disponibles para sustitución:** las plantillas pueden referenciar el nombre legible del componente afectado, el nombre del fallback activo, el tiempo transcurrido desde la entrada al modo, y otros valores contextuales. El conjunto exacto se cierra en el sprint, pero el principio es que el texto debe ser **autocontenido**: el Operador no debería tener que mirar otras vistas para entender la explicación.
- **Naturaleza del componente de explicación:** sistema de plantillas de texto con sustitución de variables, igual que HU-06. **No** se usa procesamiento de lenguaje natural, **no** se usa explicabilidad de IA, **no** se usan modelos generativos. La decisión es coherente con la postura ya tomada en HU-06.
- **Ubicación visual en el dashboard:** la HU es agnóstica al lugar físico donde se muestra el texto. Decisión razonable de prototipado es ofrecerlo simultáneamente como sección expandible desde el banner de HU-10 (acceso inmediato cuando el Operador interactúa con la alerta) y como panel propio en alguna vista persistente (acceso sin tener que abrir la alerta). La decisión exacta se cierra en el prototipado y no afecta los CAs.
- **Fallback de texto genérico (CA-12.4):** la existencia de un texto genérico de respaldo garantiza que el Operador nunca vea un panel vacío cuando el sistema está en estado degradado pero la combinación específica no fue anticipada. Coherente con CA-06.3 del Bloque B.
- **Origen del componente de explicación:** la HU es agnóstica a qué componente produce el texto. Razonable es que sea el mismo componente o servicio que produce las explicaciones de HU-06 (extensión del catálogo, no componente nuevo), pero esta integración se cierra en el sprint.
- **Caída del componente de explicación (CA-12.5):** este caso es particular porque, si el componente que explica el modo degradado deja de responder mientras el sistema está en modo degradado, el Operador queda momentáneamente sin contexto. El "no confirmado" del último texto conocido es la mejor aproximación. La marca pasiva no contradice la alerta transversal de HU-10, que sigue mostrando el estado correcto.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** latencia máxima entre transición a un nuevo modo degradado y actualización de la explicación visible ≤ 5 s (CA-12.2).
- **RNF de disponibilidad:** la explicación debe estar accesible siempre que el sistema esté en estado degradado o de falla total. Coherente con la disponibilidad transversal del banner de HU-10.
- **RNF de robustez:** comportamiento ante caída del componente de explicación (CA-12.5, DHU-005 Caso B). Coherente con CA-06.4 de HU-06.
- **RNF de cobertura:** el catálogo de plantillas debe cubrir al menos las combinaciones típicas de (componente × nivel de degradación) declaradas como esperadas por el equipo, sin pretender cubrir todas las posibles. CA-12.4 cubre el resto con texto genérico.
- **RNF de usabilidad:** la explicación debe ser comprensible por un Operador sin formación técnica en el sistema. Coherente con el RNF equivalente de HU-06. Probablemente se valida con prueba de usuario.
- **RNF de mantenibilidad:** el catálogo de plantillas debe ser extensible sin requerir cambios al código del frontend ni del backend (configurable como dato). Coherente con HU-06.
- **RNF de coherencia textual:** el lenguaje, tono y nivel de detalle de las plantillas de HU-12 deben ser consistentes con los de HU-06 y los textos de impacto operativo de CA-11.3. Probablemente se valida revisando el catálogo completo de textos en sesión dedicada (consistencia de voz del producto).

---

## Resumen del Bloque C

| HU | Título | Sujeto | Tipo | Feature(s) origen | Clasif. MVP |
|---|---|---|---|---|---|
| HU-10 | Alerta activa transversal del estado operativo del sistema | Operador | Persona | F22 | MVP1 |
| HU-11 | Vista del estado operativo de los componentes del sistema | Operador | Persona | F23 (+ F25 absorbida) | MVP1 |
| HU-12 | Explicación del modo degradado activo y sus implicaciones operativas | Operador | Persona | F24 | MVP1 |

**Total Bloque C: 3 HUs operativas + 2 TTH** (TTH-04, TTH-05 en `TAREAS_TECNICAS_HABILITADORAS.md`).

F25 (Indicación contextual en panel afectado) queda cubierta funcionalmente por la composición de HU-10 + HU-11 (CA-11.9) + HU-12 + marcas pasivas del Bloque B (DHU-005), sin generar HU propia. Decisión documentada en DHU-011.

---

## Tareas Técnicas Habilitadoras del Bloque C

Estas dos TTH **no son HUs** y se documentan en detalle en `TAREAS_TECNICAS_HABILITADORAS.md`. Se listan aquí solo para mantener trazabilidad del bloque.

| TTH | Título | Feature origen | Estado actual |
|---|---|---|---|
| TTH-04 | Lógica de fallback en cascada del sistema | F26 | Pendiente |
| TTH-05 | Configuración de tiempos preconfigurados para degradado nivel 3 | F27 | Pendiente |

---

## Decisiones que aplicaron a este bloque

Durante la redacción del Bloque C se cerraron las siguientes decisiones formales (todas en `DECISIONS_HU.md`):

- **DHU-008:** distinción arquitectónica entre componente caído, modo degradado y lógica de fallback. Define la composición inicial del bloque (4 HUs + 2 TTH antes de DHU-011).
- **DHU-009:** relación entre marca pasiva del Bloque B y alerta activa del Bloque C como complementarias y no duplicadas.
- **DHU-010:** criterios para clasificar F26 y F27 como TTH.
- **DHU-011:** eliminación de HU-13 y cobertura de F25 por composición. Ajusta la composición final a 3 HUs + 2 TTH.

---

## Próximos pasos

Esta sesión cerró el Bloque C. A la fecha actual, los Bloques D, E y F del MVP1 están cerrados y el MVP2 también está cerrado (DHU-017, 2026-05-16); **con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 22 HU operativas (HU-01 a HU-22) y 12 TTH (TTH-01 a TTH-12), tras la ampliación DHU-028 (HU-22 + TTH-12).** HU-21 (escalamiento del Operador al Administrador) consume directamente HU-10 y HU-12 de este bloque como puntos de invocación del botón "Escalar al Administrador". Los siguientes pasos del proyecto, en sesiones futuras:

1. **Bloque D — Administrador, soporte técnico** (F17, F18, F20 → 3 HUs operativas; ya cerrado con HU-13, HU-14, HU-15). Ver `HU_BLOQUE_D.md`. F21 fue reclasificado a Trabajos Futuros por DHU-012.
2. **Bloque E — Componentes centrales del sistema** (F32, F33, F34, F35 → 0 HUs operativas + 5 TTH: TTH-07 a TTH-11; ya cerrado el 2026-05-15 por DHU-015). Ver `HU_BLOQUE_E.md`.
3. **Bloque F — Gerente, reportería mínima** (F12 + F13 fusionadas en HU-16 con F30 inglobada, F14 en HU-17 → 2 HUs operativas + 0 TTH nuevas; ya cerrado el 2026-05-16 por DHU-016). Ver `HU_BLOQUE_F.md`.
4. **MVP2 — HUs documentadas con construcción condicional a holgura del cronograma tras cerrar MVP1** (F11→HU-09; F15→HU-18; F16→HU-19; F19→HU-20; F28→HU-21; 5 HUs operativas + 0 TTH nuevas; cerrado el 2026-05-16 por DHU-017). Semántica refinada por DHU-012. Ver `HU_MVP2.md`.

Tras cerrar el MVP2 (ya hecho), los próximos pasos del proyecto, fuera del alcance del Product Backlog funcional, son:

1. **Documento de Requisitos Funcionales y No Funcionales (RF/RNF)** consolidando los "Candidatos a RNF" de todas las HUs (HU-01 a HU-21) en un documento único aprobado, conforme a DHU-007 pendiente. Sesión dedicada futura.
2. **Ceremonias de estimación (Planning Poker) y priorización (MoSCoW)** sobre el backlog completo.
3. **Implementación SCRUM del MVP1** (16 HUs operativas + 11 TTH del MVP1). El MVP2 (5 HUs adicionales) entra al sprint si hay holgura de cronograma, conforme a la semántica refinada por DHU-012.
4. **SDD (Software Design Document)**, siguiente entregable académico mayor del proyecto.

---

## Documentos relacionados

- `HU_BLOQUE_A.md` — Bloque A del Product Backlog (acceso al sistema, 1 HU + 3 TTH).
- `HU_BLOQUE_B.md` — Bloque B del Product Backlog (8 HUs: HU-02 a HU-09).
- `HU_BLOQUE_D.md` — Bloque D del Product Backlog (3 HUs operativas: HU-13, HU-14, HU-15).
- `HU_BLOQUE_E.md` — Bloque E del Product Backlog (0 HUs operativas; mapeo a TTH-07 a TTH-11 y decisiones tomadas durante la redacción).
- `HU_BLOQUE_F.md` — Bloque F del Product Backlog (2 HUs operativas: HU-16, HU-17; F30 inglobada como CAs).
- `HU_MVP2.md` — MVP2 del Product Backlog (HU-18, HU-19, HU-20, HU-21; HU-09 reside en `HU_BLOQUE_B.md`). HU-21 escalamiento desde HU-10 y HU-12 de este bloque.
- `DECISIONS_HU.md` — Decisiones metodológicas sobre HUs (DHU-001 a DHU-022).
- `DECISIONS.md` — Decisiones técnicas del producto (D-001 a D-009).
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-01 a TTH-11.
- `LEAN_INCEPTION_CEREBROVIAL.md` — Inception completo aplicado al proyecto.
- `FEATURE_BACKLOG_DETALLADO.md` — Detalle completo de las 41 features identificadas (29 MVP1 + 5 MVP2 + 7 Trabajos Futuros).
- `EVOLUCION_TESIS.md` — Narrativa de las 4 fases del proyecto.
