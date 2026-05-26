# Historias de Usuario — Bloque F (Cerrado)

> Sexta entrega del Product Backlog del proyecto CerebroVial. Cierre del MVP1.
>
> **Estado:** Bloque F cerrado y aprobado. Bloques A, B, C, D y E previamente cerrados, y MVP2 también cerrado el 2026-05-16 (DHU-017). **Con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 21 HUs operativas (HU-01 a HU-21) + 11 TTH (TTH-01 a TTH-11).** Pendiente: documento RF/RNF (DHU-007), Planning Poker, MoSCoW, implementación SCRUM del MVP1. HU-16 y HU-17 son las vistas del Gerente desde las cuales se accede al drill-down (HU-18) y a la exportación (HU-19) del MVP2; el selector de periodo de HU-16 es estado compartido con HU-17 y HU-18 conforme a DHU-017 subsección G.
>
> **Fecha de cierre:** 2026-05-16
> **Fecha de actualización v2:** 2026-05-17 (DHU-018 aplicada retroactivamente: Resumen ejecutivo en HU-17; HU-16 ya tenía el bloque desde la muestra previa al cierre de DHU-018, queda como está)

---

## Contexto

Este documento contiene las Historias de Usuario del **Bloque F — Gerente, reportería mínima** del Sequencer del Lean Inception (ver `LEAN_INCEPTION_CEREBROVIAL.md`, sección 9, y `FEATURE_BACKLOG_DETALLADO.md`, Bloque F).

Las HUs se redactan en el formato del documento de referencia académica (`Desarrollo_Agil.pdf`, Tablas 9 y 13): "Como X, quiero Y, para Z" con criterios de aceptación Given-When-Then.

Las HUs del Bloque F siguen las reglas metodológicas establecidas y refinadas durante la redacción de los bloques previos:

- **DHU-001 a DHU-004** (del Bloque A): sujetos válidos, exclusión del Equipo de Desarrollo, TTH como categoría separada.
- **DHU-005** (refinada en el Bloque B): principio de robustez ante interrupción de fuente, con Casos A y B. Aplicado a las HUs del Bloque F en su forma Caso B (caída de motor de cálculo o persistencia del histórico), conforme a DHU-016 subsección J.
- **DHU-006:** HUs agnósticas a la implementación. Estricto en HU-16 y HU-17: no se nombran tecnologías de persistencia, librerías de gráficos, ni la fuente operacional concreta que alimenta el histórico en MVP1.
- **DHU-007:** RNF declarados como tales en sección específica al final de cada HU.
- **DHU-008** a **DHU-011** (cerradas en el Bloque C): modelo de degradación, marca pasiva vs alerta activa, clasificación TTH del Bloque C, eliminación de HU-13 original del Bloque C.
- **DHU-012** (transversal): auditoría de coherencia documental.
- **DHU-013** y **DHU-014** (cerradas en el Bloque D): clasificación HU/TTH del Bloque D, decisiones de redacción del Bloque D.
- **DHU-015** (cerrada en el Bloque E): clasificación HU/TTH del Bloque E, resultado 5 TTH y 0 HUs operativas.
- **DHU-016** (cerrada durante la redacción de este bloque): decisiones de redacción del Bloque F (numeración compactada continuando desde HU-15, F30 inglobada como CAs en HU-16, fuente del histórico operacional independiente y agnóstica, cuatro KPIs con definiciones operacionales cerradas, granularidad de treinta segundos, presets cerrados del selector, periodo previo equivalente, concurrencia entre Gerentes no aplicable, composición F12+F13 en HU única, robustez Caso B aplicado).
- **DHU-018** (aplicada retroactivamente el 2026-05-17): patrón "Resumen ejecutivo" al inicio de cada HU para uniformidad de lectura. Aditiva, no modifica contenido sustantivo.

Ver `DECISIONS_HU.md` para fundamentación completa.

---

## Mapeo de features del Bloque F

Las 3 features MVP1 del Bloque F (F12, F13, F14) más F30 (Persistencia de estados históricos, declarada por la regla del Bloque A como persistencia inglobada en HUs del Gerente) se mapearon a 2 HUs operativas, sin TTH nuevas, conforme a DHU-016:

| HU | Título | Feature(s) origen |
|---|---|---|
| HU-16 | Consulta de KPIs operativos sobre periodo seleccionable | F12 (Dashboard ejecutivo) + F13 (Selector de periodo) fusionadas, con F30 inglobada como CAs |
| HU-17 | Vista comparativa entre periodos | F14 (Vista comparativa entre periodos), reutiliza sustrato y definiciones de HU-16 |

**Total Bloque F:** 2 HUs operativas + 0 TTH nuevas.

**Sobre la fusión de F12 y F13 (DHU-016 subsección I):** el selector de periodo (F13) no entrega valor en aislamiento; su único propósito es gobernar lo que muestra el dashboard ejecutivo (F12) y la vista comparativa (F14). Una HU dedicada al selector violaría el principio de cohesión de Mike Cohn ("una HU = un valor entregable autocontenido"). El Gerente trabaja sobre un único objeto compuesto (los KPIs del periodo seleccionado), lo cual justifica fusionar F12 y F13 en una sola HU con secciones de CAs estructuradas. F14 sí queda como HU separada porque entrega valor diferenciado (tendencia, no estado).

**Sobre F30 inglobada como CAs (DHU-016 subsección B):** la persistencia del histórico de estados se modela como criterios de aceptación dentro de HU-16 (CA-16.1 a CA-16.3), no como TTH separada del Bloque F. Esto preserva la regla cerrada en el Bloque A y el patrón establecido por F31 inglobada en CA-08.1 de HU-08, sustrato de F18 inglobado en CA-14.1 a CA-14.4 de HU-14 y sustrato de F20 inglobado en CA-15.1 a CA-15.4 y CA-15.8 de HU-15. La justificación de no extraer a TTH es que el sustrato es consumido exclusivamente por las dos HUs cohesivas del Bloque F (HU-16 y HU-17), sin reuso desde otras HUs heterogéneas como sí ocurre con TTH-04.

**Sobre la fuente operacional del histórico en MVP1 (DHU-016 subsección C):** la HU-16 declara la persistencia del histórico de forma agnóstica a la fuente operacional, conforme a DHU-006. En MVP1, la fuente vigente que alimenta esa persistencia son las corridas de validación cuantitativa del sistema integrado en el entorno simulado de la intersección; en operación hipotética posterior, sería la salida del módulo sensor de estado. La transición es transparente para HU-16 porque su contrato es agnóstico a la fuente.

**Sobre la numeración (compactación):** el Bloque D cerró en HU-15. El Bloque E no avanzó la numeración (0 HUs operativas conforme a DHU-015). El Bloque F retoma la numeración secuencial desde HU-16 conforme a DHU-016 subsección A, manteniendo el principio establecido en DHU-014 de no dejar huecos en la numeración para preservar memoria de HUs no redactadas o eliminadas.

**Sobre la composición visual del dashboard del Gerente:** análogamente a DHU-014 subsección B (sin HU dedicada de dashboard del Administrador), el Bloque F no introduce una HU dedicada de "dashboard integrador del Gerente" análoga a F02 del Bloque B. La navegación del Gerente da acceso a HU-16 (consulta principal) y HU-17 (comparativa), y eso es suficiente. La diferencia respecto al Operador es que el Gerente no monitorea en tiempo real: el valor agregado de un dashboard integrador único como F02 viene del simultaneismo del tiempo real, que no aplica al Gerente.

---

## HU-16 — Consulta de KPIs operativos sobre periodo seleccionable

| Campo | Contenido |
|---|---|
| **Como** | Gerente de Tránsito Municipal |
| **Quiero** | consultar en una vista única los indicadores agregados de desempeño del control de tráfico sobre un periodo que yo selecciono (semana actual o anterior, mes actual o anterior, o un rango personalizado), viendo tanto el valor agregado de cada indicador como su evolución a lo largo del periodo |
| **Para** | evaluar la eficiencia del sistema de control de tráfico en periodos prolongados, identificar tendencias o problemas recurrentes para planificación futura, y disponer de datos cuantitativos que sustenten decisiones estratégicas y reportes a niveles superiores sin necesidad de operar el sistema en tiempo real ni interactuar con vistas técnicas |

**Tipo:** HU de Persona (Gerente de Tránsito Municipal).
**Feature(s) origen:** F12 (Dashboard ejecutivo con KPIs agregados) y F13 (Selector de periodo) fusionadas en una sola HU conforme a DHU-016 subsección I. Ingloba como CAs el sustrato técnico de persistencia histórica (F30 — Persistencia de estados históricos), conforme a la regla cerrada en el Bloque A y al patrón equivalente al de F31 inglobada en CA-08.1 de HU-08 y al sustrato inglobado en HU-14 y HU-15.

### Resumen ejecutivo

**Qué entrega:** vista única del Gerente con un selector de periodo y un dashboard de cuatro KPIs agregados del control de tráfico (tiempo de espera, longitud máxima de cola, throughput, demora) sobre el periodo seleccionado, cada uno con valor agregado y gráfico de evolución temporal.

**CAs críticos:** CA-16.1 (persistencia del histórico observado a granularidad de 30 s), CA-16.13 (presentación de las cuatro cards al Gerente), CA-16.19 (robustez ante caída del motor de cálculo o de la persistencia — DHU-005 Caso B), CA-16.20 (control de acceso por rol).

**Estructura de CAs:** sustrato técnico F30 inglobado (CA-16.1 a CA-16.3) → selección de periodo F13 inglobada (CA-16.4 a CA-16.8) → definiciones operacionales de los KPIs (CA-16.9 a CA-16.12) → presentación al Gerente (CA-16.13 a CA-16.16) → casos degenerados (CA-16.17 a CA-16.19) → control de acceso (CA-16.20 a CA-16.21).

**Dependencias:** consume su propio sustrato F30 inglobado. Comparte estado del selector con HU-17 y HU-18 (MVP2) durante la sesión activa. HU-19 (MVP2) consume sus CAs para generar reportes exportables. Aplica DHU-005 Caso B, DHU-006 (agnosticismo de fuente operacional), DHU-016 subsecciones A a J.

**Notas clave:** F12 y F13 se fusionaron por cohesión (selector sin dashboard no entrega valor; DHU-016 subsección I). La fuente operacional del histórico en MVP1 son las corridas de validación en el entorno simulado, sin nombrar tecnologías (DHU-006). Sin política de retención automática del histórico en MVP1.

### Descripción

El Gerente de Tránsito Municipal es la persona responsable de evaluar el desempeño global del sistema de control de tráfico, justificar inversiones y reportar a niveles superiores. A diferencia del Operador (que monitorea en tiempo real) y del Administrador (que mantiene la salud técnica), el Gerente trabaja con horizonte temporal semanal o mensual y consume información agregada sobre periodos prolongados. Su contacto con el sistema es esporádico y de naturaleza consultiva: entra, selecciona un periodo, lee los indicadores, eventualmente compara con un periodo previo, y sale.

HU-16 le entrega la vista principal de consulta. La vista se compone de dos partes integradas: un selector de periodo en la parte superior, persistente y siempre visible, y un dashboard ejecutivo en el cuerpo principal de la vista que muestra los indicadores del periodo seleccionado. El selector y el dashboard son componentes de una misma HU porque el selector no entrega valor en aislamiento (sin un dashboard que gobernar, el selector es un control vacío) y el dashboard no es operativo sin un periodo que lo alimente (un dashboard estático sobre "todo el tiempo" no responde a la necesidad declarada del Gerente de evaluar periodos específicos).

Los cuatro indicadores que la vista expone corresponden a los KPIs técnicos del sistema integrado cerrados en el MVP Canvas (Bloque 6) del Inception, que son simultáneamente la base de validación cuantitativa de la tesis (Objetivo 4 del producto):

1. **Tiempo promedio de espera por vehículo**, medido en segundos, calculado como la media aritmética del tiempo que cada vehículo permanece con velocidad por debajo de un umbral bajo durante su paso por la intersección, agregado sobre todos los vehículos del periodo. Se reporta como valor agregado de la intersección y, mediante un control de la vista, disgregado por dirección de entrada.

2. **Longitud máxima de cola por dirección**, medida en número de vehículos, calculada como el máximo observado de la longitud de cola en cada dirección durante el periodo. Se reporta por dirección sin agregación al total de la intersección, porque el máximo de un agregado no es el agregado de los máximos y la disgregación por dirección preserva información que se perdería al sumar.

3. **Throughput de la intersección**, medido en vehículos por hora, calculado como el número total de vehículos que cruzan la intersección durante el periodo dividido por la duración del periodo. Se reporta como valor agregado total sin disgregación por dirección en la vista principal, porque el agregado es lo que sustenta la mejora cuantitativa frente al control fijo que la tesis declara como criterio de éxito.

4. **Demora promedio acumulada por vehículo**, medida en segundos, calculada como la media aritmética de la diferencia entre el tiempo real que cada vehículo tarda en cruzar la intersección y el tiempo que tardaría a velocidad libre del acceso (sin detenciones). Se reporta como valor agregado total. Es complementaria al tiempo promedio de espera: este último mide tiempo de detención efectiva; la demora acumulada mide tiempo perdido respecto al óptimo teórico.

Cada uno de los cuatro indicadores se presenta en dos formas integradas: una card con el valor agregado del periodo (el número grande, central a la lectura del Gerente) y un gráfico temporal de líneas que muestra la evolución del indicador a lo largo del periodo, con granularidad adaptativa según la duración del periodo (puntos por hora para periodos cortos, por día para periodos de semanas a un mes, por semana para periodos personalizados largos). Cada card incluye un ícono de ayuda activable que despliega la definición operacional del indicador, para que el Gerente pueda leer la vista sin necesidad de consultar documentación externa, en línea con el patrón de tooltips establecido en CA-14.7 de HU-14.

El sustrato técnico (persistencia continua del estado observado del tráfico con timestamp, agregación sobre el periodo seleccionado, cálculo de los cuatro indicadores) es responsabilidad de esta misma HU, sin generar TTH adicional, conforme a DHU-016 subsección B. La fuente operacional que alimenta esta persistencia en MVP1 es la fuente de estado vigente del sistema en cada momento, sin que la HU nombre tecnologías concretas conforme a DHU-006.

### Criterios de aceptación

#### Sustrato técnico (persistencia del histórico de estados — F30 inglobada)

- **CA-16.1:** Dado que el sistema observa el estado del tráfico de la intersección durante su operación, cuando se produce una observación a lo largo del tiempo, entonces el sistema persiste de forma durable, por intersección y por dirección, con marca de tiempo agregada a granularidad de treinta segundos, al menos: flujo de vehículos, longitud de cola, velocidad media y densidad. La persistencia es independiente del registro de predicciones, del registro de decisiones del motor adaptativo, del registro de transiciones de estado operativo del sistema y del registro de métricas del módulo sensor, los cuales mantienen su propio esquema y su propio ciclo de escritura.

- **CA-16.2:** Dado que el sistema persiste el estado observado a granularidad de treinta segundos, cuando se solicita el histórico para un periodo de consulta, entonces el sistema dispone de los registros con esa granularidad sin necesidad de procesamiento adicional. La granularidad de treinta segundos por intersección y dirección es el valor por defecto de MVP1 y no se expone como parámetro configurable al Administrador en MVP1.

- **CA-16.3:** Dado que el histórico se acumula durante la operación del sistema, cuando el volumen de filas crece a lo largo del tiempo, entonces el sistema no aplica política de retención automática en MVP1 (no hay borrado programado). El histórico acumulado durante el alcance académico se preserva completo y disponible para consulta. La política de retención para operación posterior es trabajo futuro si se justifica.

#### Selección de periodo (F13 inglobada como CAs)

- **CA-16.4:** Dado que el Gerente ha iniciado sesión y accede a la vista de consulta, cuando la vista se carga por primera vez en la sesión, entonces el selector de periodo muestra cinco opciones: "Esta semana", "Semana anterior", "Este mes", "Mes anterior" y "Rango personalizado". La opción seleccionada por defecto es "Esta semana", sin que el sistema persista entre sesiones la última selección del Gerente; cada nueva sesión comienza con el default.

- **CA-16.5:** Dado que el Gerente cambia la selección del periodo entre los cuatro presets ("Esta semana", "Semana anterior", "Este mes" o "Mes anterior"), cuando confirma la selección, entonces la vista recalcula automáticamente los cuatro indicadores y sus gráficos temporales para el nuevo periodo, sin que el Gerente necesite pulsar un botón adicional. Durante el recálculo se muestra un indicador de carga visible.

- **CA-16.6:** Dado que el Gerente selecciona "Rango personalizado", cuando ingresa la fecha de inicio y la fecha de fin mediante el componente de selección de fechas, entonces el recálculo de los indicadores se dispara únicamente cuando el Gerente activa explícitamente la aplicación del rango (botón "Aplicar" o equivalente), no en cada cambio individual de cualquiera de las dos fechas. Esto evita recálculos intermedios cuando el rango aún está incompleto.

- **CA-16.7:** Dado que el Gerente está seleccionando el rango personalizado, cuando interactúa con el componente de fechas, entonces la fecha de fin no puede ser anterior a la fecha de inicio (validación en el formulario) y ninguna de las dos fechas puede situarse en el futuro respecto al momento actual del sistema (las fechas futuras aparecen desactivadas en el selector). Si por cualquier vía llega al backend una solicitud con fecha de fin posterior al momento actual, el backend recorta la fecha de fin al momento actual y devuelve el periodo efectivamente consultado.

- **CA-16.8:** Dado que las convenciones de cálculo de los periodos predefinidos deben ser inequívocas, cuando el sistema interpreta cada preset, entonces aplica las siguientes definiciones: "Esta semana" comprende desde el lunes de la semana actual a las 00:00 hasta el momento actual del sistema; "Semana anterior" comprende desde el lunes de la semana previa a las 00:00 hasta el domingo previo a las 23:59:59; "Este mes" comprende desde el día primero del mes actual a las 00:00 hasta el momento actual del sistema; "Mes anterior" comprende desde el día primero del mes previo a las 00:00 hasta el último día del mes previo a las 23:59:59. La zona horaria de referencia es la del despliegue del sistema. La semana comienza los lunes conforme a la convención ISO 8601.

#### Definiciones operacionales de los cuatro indicadores

- **CA-16.9:** Dado que la vista muestra el indicador "Tiempo promedio de espera por vehículo", cuando el sistema lo calcula sobre el periodo seleccionado, entonces aplica la siguiente definición operacional: media aritmética, en segundos, del tiempo que cada vehículo permanece con velocidad por debajo de un umbral bajo durante su paso por la intersección, agregado sobre todos los vehículos cuyo paso por la intersección se completó dentro del periodo. El umbral concreto de velocidad considerado "en espera" se cierra al implementar (sugerencia operativa documentada: 0.1 metros por segundo).

- **CA-16.10:** Dado que la vista muestra el indicador "Longitud máxima de cola por dirección", cuando el sistema lo calcula sobre el periodo seleccionado, entonces aplica la siguiente definición operacional: máximo, en número de vehículos, de la longitud de cola observada en cada dirección de entrada durante el periodo. El indicador se reporta como un valor por dirección, sin agregación a un único valor total de la intersección.

- **CA-16.11:** Dado que la vista muestra el indicador "Throughput de la intersección", cuando el sistema lo calcula sobre el periodo seleccionado, entonces aplica la siguiente definición operacional: número total de vehículos que cruzan completamente la intersección durante el periodo, dividido por la duración del periodo en horas, expresado en vehículos por hora. Se reporta como valor agregado total de la intersección, sin disgregación por dirección en la vista principal.

- **CA-16.12:** Dado que la vista muestra el indicador "Demora promedio acumulada por vehículo", cuando el sistema lo calcula sobre el periodo seleccionado, entonces aplica la siguiente definición operacional: media aritmética, en segundos, de la diferencia entre el tiempo real que cada vehículo tarda en cruzar la intersección y el tiempo que tardaría a velocidad libre del acceso correspondiente (longitud del acceso dividida por la velocidad máxima del acceso). Se reporta como valor agregado total de la intersección.

#### Presentación al Gerente

- **CA-16.13:** Dado que el Gerente ha confirmado un periodo, cuando la vista termina de calcular los indicadores, entonces el sistema muestra cuatro cards principales, una por cada indicador (tiempo promedio de espera, longitud máxima de cola, throughput, demora promedio acumulada). Cada card contiene: el nombre del indicador, el valor agregado del periodo destacado visualmente como número principal de la card, la unidad correspondiente, un ícono de ayuda activable y un gráfico temporal de líneas que muestra la evolución del indicador a lo largo del periodo.

- **CA-16.14:** Dado que las cards de "Tiempo promedio de espera" y "Longitud máxima de cola" admiten disgregación por dirección, cuando el Gerente quiere ver el desglose, entonces cada una de esas dos cards incluye un control visible (toggle, pestaña o equivalente) que permite alternar entre la presentación agregada (valor total o resumen) y la presentación por dirección (un valor por cada dirección de entrada de la intersección). La presentación inicial al abrir la vista es la agregada; el control desglose-agregado se ofrece dentro de la misma card sin abrir vistas adicionales.

- **CA-16.15:** Dado que cada indicador incluye un gráfico temporal, cuando el sistema renderiza ese gráfico, entonces la granularidad de los puntos del gráfico se determina adaptativamente según la duración del periodo seleccionado, de manera que el número de puntos resultante sea legible (criterio: el gráfico debe ser interpretable sin saturación visual ni puntos aislados). El criterio de adaptación cerrado para MVP1 es: periodos menores a cuarenta y ocho horas presentan puntos por hora; periodos entre dos y treinta y un días presentan puntos por día; periodos mayores a treinta y un días presentan puntos por semana. El backend agrega el histórico de treinta segundos al granularidad de presentación correspondiente, usando media para tiempo de espera y demora, máximo para longitud de cola y suma normalizada para throughput.

- **CA-16.16:** Dado que el Gerente no recuerda cómo interpretar alguno de los indicadores, cuando activa el ícono de ayuda visible en la card correspondiente, entonces el sistema despliega una explicación breve y autocontenida que reproduce la definición operacional del indicador (CA-16.9, CA-16.10, CA-16.11 o CA-16.12 según corresponda) en lenguaje accesible para el Gerente. Esto aplica a los cuatro indicadores.

#### Manejo de casos degenerados

- **CA-16.17:** Dado que el periodo seleccionado no contiene datos persistidos (por ejemplo, el Gerente selecciona "semana anterior" pero el sistema aún no estaba operativo durante esa semana, o el rango personalizado cubre un intervalo sin operación del sistema), cuando el Gerente confirma el periodo, entonces la vista comunica explícitamente "no hay datos en el periodo seleccionado" en lugar de mostrar los cuatro indicadores en cero. La comunicación distingue inequívocamente entre "no hubo datos durante el periodo" y "los datos fueron en cero" (que serían dos lecturas operacionales muy distintas).

- **CA-16.18:** Dado que el periodo seleccionado contiene datos solo parcialmente (por ejemplo, el sistema empezó a operar a mitad del periodo, o hubo intervalos sin observación dentro del periodo), cuando la vista presenta los indicadores, entonces los valores calculados se presentan junto con una indicación visible de que la cobertura del periodo es parcial, mostrando el porcentaje o intervalo efectivamente cubierto. Esto evita que el Gerente interprete como representativo del periodo completo un cálculo basado en una fracción.

- **CA-16.19:** Dado que el componente responsable de la persistencia del histórico o del cálculo de los indicadores deja de responder, cuando el Gerente consulta la vista, entonces el sistema indica explícitamente la indisponibilidad temporal, mostrando los últimos indicadores conocidos marcados como "no actualizados" e indicando el tiempo transcurrido desde el último cálculo exitoso (DHU-005 Caso B aplicado al motor de cálculo de indicadores). Si no hay indicadores previos disponibles porque la vista se abre por primera vez con el componente caído, la vista comunica "no se pueden calcular los indicadores en este momento" en lugar de mostrar la vista vacía o valores en cero.

#### Control de acceso

- **CA-16.20:** Dado que un usuario con rol Operador o Administrador intenta acceder a la vista de consulta de indicadores del Gerente, cuando solicita la ruta correspondiente, entonces el sistema deniega el acceso conforme a la política de control de acceso por rol (HU-01 del Bloque A), respondiendo con HTTP 403 si la solicitud llega vía API o redirigiéndolo fuera de la vista si el intento es vía navegación frontend.

- **CA-16.21:** Dado que el Gerente no ha iniciado sesión, cuando intenta acceder a la vista de consulta de indicadores, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Sustrato inglobado, no TTH:** conforme a DHU-016 subsección B, el sustrato de persistencia del histórico de estados (F30) y de cálculo de los indicadores se ingloba como CAs de esta HU (CA-16.1 a CA-16.3 para la persistencia; CA-16.9 a CA-16.12 para el cálculo) en lugar de extraerse como TTH separada del Bloque F. La justificación es la misma que para HU-14 y HU-15: este sustrato es consumido únicamente por las HUs del Bloque F (HU-16 y HU-17), que son cohesivas y comparten propósito (reportería al Gerente). Patrones previos equivalentes: F31 inglobada en CA-08.1 de HU-08, sustrato de F18 inglobado en CA-14.1 a CA-14.4 de HU-14, sustrato de F20 inglobado en CA-15.1 a CA-15.4 y CA-15.8 de HU-15.

- **Independencia respecto a otros registros del sistema (CA-16.1):** la persistencia de F30 es operacional independiente respecto a los demás registros del sistema. Esa independencia es deliberada: cada registro tiene su propio esquema, su propio ciclo de escritura y sus propios consumidores, y mezclarlos contaminaría modelos y dificultaría su evolución independiente.

- **Fuente operacional vigente en MVP1 (conforme a DHU-016 subsección C):** la HU no nombra la fuente operacional vigente que alimenta el histórico de F30 en MVP1, conforme a DHU-006 (HUs agnósticas a la implementación). En el alcance académico, esa fuente operacional son las corridas de validación cuantitativa del sistema integrado en el entorno simulado de la intersección. Cuando el sistema corre una simulación, el estado observado por intersección y dirección se persiste en la tabla de F30 con granularidad de treinta segundos. El demo del MVP1 al jurado consulta el histórico generado en esas corridas. En operación hipotética posterior al alcance académico, la fuente vigente sería la salida del módulo sensor de estado; la transición es transparente para HU-16 porque su contrato es agnóstico a la fuente. La implementación concreta de la escritura del histórico durante las corridas de validación es responsabilidad de implementación, no requiere TTH nueva.

- **Composición F12 + F13 en una sola HU (conforme a DHU-016 subsección I):** la fusión de F12 (Dashboard ejecutivo) y F13 (Selector de periodo) en una sola HU responde al principio de cohesión de Mike Cohn: una HU entrega un valor autocontenido, y un selector aislado sin lo que selecciona no entrega valor. El Gerente trabaja sobre un único objeto compuesto (los indicadores del periodo seleccionado); descomponer en dos HUs (una de selector y otra de dashboard) crearía dos HUs cuyo valor solo se materializa al consumirlas juntas. Esto es análogo a por qué F02 del Bloque B se cubre por composición visual de HU-02 a HU-06 sin generar HU dedicada de dashboard: el dashboard del Operador es un objeto compuesto cuya descomposición ya está cubierta por las HUs constituyentes.

- **Sin HU dedicada de dashboard integrador del Gerente:** análogamente a DHU-014 subsección B (sin HU dedicada de dashboard del Administrador), el Gerente no requiere HU dedicada de "dashboard integrador" que componga HU-16 y HU-17 en una vista única. La navegación del Gerente da acceso a HU-16 (consulta principal) y HU-17 (comparativa), y eso es suficiente. La diferencia respecto al Operador es que el Gerente no monitorea en tiempo real: el valor agregado de un dashboard integrador único como F02 viene del simultaneismo del tiempo real, que no aplica al Gerente.

- **Selección de periodo sin persistencia entre sesiones (CA-16.4):** la decisión de no persistir la última selección del Gerente entre sesiones se justifica por (a) complejidad de modelo añadida (tabla de preferencias por usuario o columna en la tabla de usuarios) sin valor proporcional al caso de uso primario del Gerente, que es revisión semanal con default razonable, (b) el default "Esta semana" cubre el caso de uso primario declarado en el Journey 2 ("Revisión semanal de desempeño"). La persistencia de preferencias por usuario es mejora natural si más adelante se justifica con un caso de uso concreto.

- **Granularidad adaptativa del gráfico temporal (CA-16.15):** el criterio de adaptación se cierra para MVP1 en tres tramos (horario, diario, semanal) según la duración del periodo. Otros criterios posibles (por ejemplo, ajuste continuo del número de puntos al ancho del gráfico) son refinamiento de UX al implementar y no requieren cambio de la HU mientras se respete el principio de legibilidad. La función de agregación a aplicar al pasar del histórico de treinta segundos a la granularidad de presentación depende del indicador: media para tiempo de espera y demora promedio, máximo para longitud máxima de cola, suma normalizada por la duración del intervalo agregado para throughput. Estas agregaciones se documentan en CA-16.15 para que sean inequívocas.

- **Disgregación por dirección con toggle en card (CA-16.14):** la decisión de presentar la disgregación por dirección mediante un control dentro de la misma card, en lugar de cards adicionales o vistas separadas, responde a tres consideraciones: (a) ocho cards adicionales por la disgregación de dos indicadores saturarían la vista visual del Gerente; (b) el valor primario para el Gerente es el agregado de la intersección (le interesa la eficiencia global, no la dirección específica salvo que detecte algo anómalo); (c) el drill-down por dirección detallado es exactamente F15 (Vista detallada de periodo específico), clasificada como MVP2 fuera del alcance del Bloque F. El toggle agregado-desglose en MVP1 cubre la necesidad mínima sin invadir el alcance de F15.

- **Latencias aceptables:** la apertura de la vista con el periodo por defecto debería completarse en un tiempo razonable para que el Gerente no perciba demora (criterio sugerido: tres segundos), y el recálculo al cambiar de periodo debería completarse con un indicador de carga visible y dentro de un tiempo razonable para una operación de consulta sobre histórico (criterio sugerido: diez segundos). Si la realidad medida es peor en MVP1 por volumen de datos persistidos, se reporta conforme a D-005 (umbrales aspiracionales, no bloqueantes). Estos umbrales son candidatos a RNF.

- **Validación de fechas futuras en doble capa (CA-16.7):** la validación tanto en el formulario como en el backend sigue el patrón de CA-15.6 de HU-15 (validación en formulario y servidor). El recorte del backend cuando llega una fecha de fin futura, en lugar de devolver error, hace más amable la experiencia: el Gerente puede haber seleccionado un rango ligeramente fuera de límite por descuido, y devolver el resultado del periodo efectivamente consultado es más útil que rechazar la solicitud completa.

- **Concurrencia entre Gerentes (conforme a DHU-016 subsección H):** múltiples Gerentes consultando la vista simultáneamente no requiere mecanismo de control de concurrencia. HU-16 es read-only: el Gerente consulta indicadores, no edita configuración ni datos. La concurrencia entre lecturas simultáneas es un caso de carga, no de concurrencia funcional, y no requiere las salvaguardas de last-write-wins que HU-15 sí requiere para la edición del Administrador (CA-15.11).

- **Relación con HU-17:** la comparativa entre periodos (HU-17) reutiliza el periodo seleccionado en esta HU. Esa relación se documenta en HU-17; aquí basta con declarar que la selección del periodo es estado de la vista del Gerente, accesible para la comparativa cuando el Gerente navega a esa otra HU.

- **Trabajo futuro adyacente:** el drill-down sobre eventos específicos del periodo (F15), la exportación de reportes a PDF o Excel (F16) y el periodo "trimestre" mencionado en el título original de F13 son trabajo fuera del alcance del Bloque F: F15 y F16 son MVP2; el trimestre se evalúa como mejora si surge necesidad concreta. La persistencia por usuario de la última selección del periodo, la calibración fina del umbral de "vehículo en espera" en CA-16.9, y la política de retención del histórico son refinamientos posteriores no bloqueantes para MVP1.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de durabilidad:** el histórico persistido por CA-16.1 no puede perderse por fallo temporal del sistema. La pérdida de filas históricas degrada irrecuperablemente la capacidad del Gerente de evaluar periodos pasados.
- **RNF de granularidad de persistencia:** treinta segundos por intersección y dirección como granularidad operativa del histórico (CA-16.2). Probablemente se valida con prueba de carga en MVP1.
- **RNF de auditabilidad:** las filas del histórico no deben modificarse después de escritas (preserva la confiabilidad de la consulta retroactiva del Gerente). En línea con el RNF equivalente de HU-08, HU-10, HU-14 y HU-15.
- **RNF de rendimiento (apertura):** la vista debería abrirse con el periodo por defecto en tiempo razonable para el Gerente (criterio sugerido: ≤ 3 segundos).
- **RNF de rendimiento (recálculo):** el recálculo al cambiar de periodo, incluso para periodos largos, debería completarse en tiempo razonable con indicador de carga visible (criterio sugerido: ≤ 10 segundos).
- **RNF de robustez:** comportamiento ante caída del motor de cálculo de indicadores o de la persistencia del histórico (CA-16.19, DHU-005 Caso B). En lugar de mostrar la vista vacía o indicadores en cero, se muestran los últimos indicadores conocidos marcados como "no actualizados".
- **RNF de manejabilidad de datos faltantes:** periodos sin datos (CA-16.17) o con cobertura parcial (CA-16.18) se comunican explícitamente, no se muestran como indicadores calculados sobre cero.
- **RNF de usabilidad:** la vista debe permitir al Gerente interpretar los indicadores sin documentación externa, mediante los tooltips de ayuda integrados (CA-16.16). Las cards con número grande deben ser legibles a primer golpe de vista (typografía contrastante, escala numérica adecuada). El control de desglose por dirección (CA-16.14) debe estar visualmente claro sin ser intrusivo.
- **RNF de accesibilidad:** los tooltips deben ser activables tanto con teclado como con dispositivo apuntador. La distinción visual entre presentación agregada y presentación desglosada por dirección no debe depender exclusivamente del color.
- **RNF de seguridad:** la consulta de la vista está restringida al rol Gerente (CA-16.20). Un usuario sin ese rol no debe poder consultar los indicadores bajo ninguna circunstancia, incluso si conoce los endpoints subyacentes.
- **RNF de validación dual:** las restricciones sobre fechas (CA-16.7) se validan tanto en el formulario del frontend como en el backend, conforme al patrón establecido en CA-15.6 de HU-15.

---

## HU-17 — Vista comparativa entre periodos

| Campo | Contenido |
|---|---|
| **Como** | Gerente de Tránsito Municipal |
| **Quiero** | comparar los cuatro indicadores agregados de desempeño del control de tráfico entre el periodo que tengo seleccionado y el periodo previo equivalente, viendo simultáneamente la evolución temporal de ambos periodos y la variación numérica entre ellos |
| **Para** | identificar tendencias de mejora o deterioro del sistema entre periodos, justificar la inversión en el sistema con datos cuantitativos de evolución y reportar a niveles superiores apoyado en evidencia comparativa, sin necesidad de hacer drill-down sobre eventos puntuales |

**Tipo:** HU de Persona (Gerente de Tránsito Municipal).
**Feature(s) origen:** F14 (Vista comparativa entre periodos). Reutiliza el sustrato técnico de persistencia del histórico (CA-16.1 a CA-16.3 de HU-16) y las definiciones operacionales de los cuatro indicadores (CA-16.9 a CA-16.12 de HU-16), sin redeclarar contenido ni introducir TTH nuevas.

### Resumen ejecutivo

**Qué entrega:** vista comparativa del Gerente entre el periodo seleccionado y el periodo previo equivalente. Cuatro paneles, uno por KPI, con gráfico de dos series temporales superpuestas, dos valores agregados y variación porcentual con semántica visual de mejora o empeoramiento (no signo aritmético).

**CAs críticos:** CA-17.4 (definición de "periodo previo equivalente"), CA-17.6 (presentación con gráfico + agregados + variación por panel), CA-17.7 (semántica visual de mejora/empeoramiento, no signo aritmético), CA-17.14 (DHU-005 Caso B aplicado simultáneamente a ambos periodos por causa raíz compartida), CA-17.15 (HTTP 403 para roles no-Gerente).

**Estructura de CAs:** estado del selector compartido con HU-16 (CA-17.1 a CA-17.3) → definición de periodo previo equivalente (CA-17.4) → presentación al Gerente (CA-17.5 a CA-17.10) → casos degenerados (CA-17.11 a CA-17.14) → control de acceso (CA-17.15, CA-17.16).

**Dependencias:** consume el sustrato F30 inglobado en HU-16 y las definiciones de KPI de CA-16.9 a CA-16.12. Comparte estado del selector con HU-16 (volátil, sin persistencia entre sesiones; CA-17.1 a CA-17.3). HU-18 (MVP2, drill-down) se accede desde cada panel comparativo con elección entre "Periodo actual" o "Periodo previo". HU-19 (MVP2) exporta esta comparativa a PDF/Excel.

**Notas clave:** semántica de variación traducida a mejora/empeoramiento por construcción de los KPIs (3 mejoran al disminuir, throughput mejora al aumentar). Redundancia visual (signo numérico + flecha direccional + color) para no depender del color (accesibilidad). **Sin disgregación por dirección** en MVP1 (asimetría intencional con HU-16, justificada por carga visual de dos periodos). Significancia estadística rigurosa fuera del alcance MVP1.

### Descripción

La vista comparativa es la segunda vista del Gerente, complementaria a HU-16. Mientras HU-16 responde a la pregunta "¿cómo le va a la intersección en el periodo seleccionado?", HU-17 responde a la pregunta "¿cómo evoluciona la intersección respecto al periodo previo equivalente?". Ambas vistas son consultivas, agregadas y orientadas al horizonte semanal o mensual del Gerente, conforme al Journey 2 paso 4 del Inception.

La vista presenta los cuatro indicadores cerrados en el MVP Canvas (tiempo promedio de espera por vehículo, longitud máxima de cola por dirección, throughput de la intersección, demora promedio acumulada por vehículo) comparados entre dos periodos: el periodo actualmente seleccionado y el periodo previo equivalente al actual. La definición de "periodo previo equivalente" se cierra en CA-17.4 y se deriva de la convención cerrada en DHU-016 subsección G.

Cada uno de los cuatro indicadores se presenta como un panel comparativo que combina tres elementos integrados: dos series temporales superpuestas en un mismo gráfico (la serie del periodo actual y la del periodo previo equivalente, distinguidas por color), los valores agregados de ambos periodos como números legibles, y un indicador prominente de la variación entre los dos agregados expresado en porcentaje, acompañado de una semántica visual que comunica si la variación representa una mejora o un empeoramiento del desempeño según la naturaleza de cada indicador.

El selector de periodo de esta vista comparte estado con el selector de HU-16 durante la sesión activa del Gerente: si el Gerente seleccionó un periodo en HU-16 y navega a HU-17, la comparativa se abre con ese mismo periodo preseleccionado y su correspondiente periodo previo equivalente, sin que el Gerente deba reseleccionar. La compartición de estado no se persiste entre sesiones: cada nueva sesión arranca con el default "Esta semana" en ambas vistas, conforme al comportamiento ya cerrado en CA-16.4.

A diferencia de HU-16, la vista comparativa no admite disgregación por dirección de los indicadores: presenta exclusivamente los agregados de la intersección. Esta asimetría es intencional. HU-16 responde al caso de uso "estado del sistema en el periodo", donde la disgregación por dirección puede aportar contexto local cuando el Gerente detecta algo anómalo. HU-17 responde al caso de uso "tendencia global del sistema", donde el valor agregado de la intersección es lo que sustenta la lectura comparativa, y agregar disgregación por dirección sobre dos periodos saturaría la vista sin servir al caso de uso. El drill-down detallado por dirección o por evento específico es alcance de F15 (Vista detallada de periodo específico), clasificada como MVP2 fuera del Bloque F.

La vista no calcula significancia estadística rigurosa de las variaciones en MVP1: muestra la variación porcentual cualquiera sea su magnitud, con su semántica de mejora o empeoramiento, dejando a la interpretación del Gerente si la magnitud observada es relevante para su decisión. El refinamiento estadístico se evalúa como mejora posterior si se justifica con casos concretos de uso.

### Criterios de aceptación

#### Estado del selector de periodo compartido con HU-16

- **CA-17.1:** Dado que el Gerente ha iniciado sesión y navega a la vista comparativa, cuando la vista se carga, entonces el selector de periodo refleja la última selección realizada por el Gerente en su sesión activa (sea desde HU-16 o desde HU-17), con default "Esta semana" si es la primera vista del Gerente en la sesión actual. La compartición de estado entre HU-16 y HU-17 es válida solo durante la sesión activa; cada nueva sesión arranca con el default, sin persistencia entre sesiones, conforme al patrón establecido en CA-16.4.

- **CA-17.2:** Dado que el Gerente cambia la selección del periodo en la vista comparativa, cuando confirma el cambio (selección de uno de los cuatro presets, o aplicación de un rango personalizado), entonces el sistema recalcula automáticamente ambos periodos (actual y previo equivalente) y actualiza los cuatro paneles comparativos. El selector de la vista comparativa respeta los mismos cinco modos cerrados en CA-16.4: "Esta semana", "Semana anterior", "Este mes", "Mes anterior" y "Rango personalizado", con las mismas convenciones de cálculo de CA-16.8, las mismas reglas de validación de fechas de CA-16.7, y la misma mecánica de aplicación diferenciada entre presets y rango personalizado de CA-16.5 y CA-16.6.

- **CA-17.3:** Dado que el Gerente regresa a HU-16 después de haber cambiado el periodo en HU-17, cuando la vista de HU-16 se renderiza, entonces refleja la última selección activa en la sesión (la realizada en HU-17), conservando la coherencia del estado compartido durante toda la sesión.

#### Definición de "periodo previo equivalente"

- **CA-17.4:** Dado que el periodo seleccionado determina su periodo previo equivalente, cuando el sistema calcula este último, entonces aplica la convención cerrada en DHU-016 subsección G: si el periodo seleccionado es "Esta semana", el periodo previo equivalente es "Semana anterior"; si es "Este mes", el previo es "Mes anterior"; si es "Semana anterior", el previo es la semana anterior a esa, es decir, dos semanas atrás respecto a la semana actual; si es "Mes anterior", el previo es el mes anterior a ese, es decir, dos meses atrás respecto al mes actual; si es "Rango personalizado", el previo es el rango de igual duración inmediatamente anterior al rango seleccionado, calculado como las fechas que terminan justo antes de la fecha de inicio del rango actual y cubren la misma cantidad de tiempo.

#### Presentación al Gerente de la comparativa

- **CA-17.5:** Dado que el Gerente ha confirmado un periodo, cuando la vista termina de calcular ambos periodos, entonces el sistema muestra cuatro paneles comparativos, uno por cada indicador del MVP Canvas. Los cuatro paneles se presentan simultáneamente en la vista, sin requerir que el Gerente seleccione cuál indicador comparar; el Gerente ve los cuatro indicadores comparados en una sola lectura.

- **CA-17.6:** Dado que cada panel comparativo muestra un indicador, cuando el sistema renderiza ese panel, entonces incluye tres elementos integrados: (a) un gráfico de líneas con dos series temporales superpuestas en el mismo eje temporal, una para el periodo actual y otra para el periodo previo equivalente, con colores claramente distinguibles y leyenda visible que identifica a cuál periodo corresponde cada serie; (b) los dos valores agregados del indicador, uno por cada periodo, mostrados como números legibles junto al gráfico; (c) un indicador prominente de la variación entre los dos agregados, expresado en porcentaje, con semántica visual conforme a CA-17.7.

- **CA-17.7:** Dado que cada panel comparativo muestra una variación porcentual entre periodos, cuando el sistema renderiza esa variación, entonces el código visual de la variación comunica **dirección de mejora o empeoramiento del desempeño**, no signo aritmético del cambio. La traducción de signo aritmético a mejora o empeoramiento depende de cada indicador: para "Tiempo promedio de espera", "Longitud máxima de cola" y "Demora promedio acumulada" la disminución entre periodos representa mejora; para "Throughput" el aumento entre periodos representa mejora. La variación se acompaña simultáneamente de tres pistas visuales redundantes: un signo numérico explícito en el porcentaje, una flecha o icono direccional, y un color que distingue mejora de empeoramiento. La redundancia es deliberada para no depender exclusivamente del color (RNF de accesibilidad).

- **CA-17.8:** Dado que cada panel comparativo presenta gráficos temporales, cuando el sistema renderiza esos gráficos, entonces la granularidad de los puntos de cada serie temporal se determina adaptativamente según la duración del periodo, conforme al mismo criterio cerrado en CA-16.15 de HU-16: periodos menores a cuarenta y ocho horas presentan puntos por hora, periodos entre dos y treinta y un días presentan puntos por día, periodos mayores a treinta y un días presentan puntos por semana. Ambas series del panel (actual y previo equivalente) usan la misma granularidad para preservar la comparabilidad visual.

- **CA-17.9:** Dado que los paneles muestran tanto valores agregados como series temporales, cuando el Gerente consulta la vista, entonces el orden visual de los cuatro paneles es coherente con el orden de las cards de HU-16 (Tiempo promedio de espera, Longitud máxima de cola, Throughput, Demora promedio acumulada), para facilitar la navegación cognitiva entre vistas.

- **CA-17.10:** Dado que el Gerente quiere recordar la definición operacional de algún indicador comparado, cuando activa el ícono de ayuda visible en el panel correspondiente, entonces el sistema despliega la misma explicación autocontenida descrita en CA-16.16 de HU-16 para el indicador correspondiente. La consistencia de los tooltips entre ambas vistas asegura que el Gerente reciba la misma definición operacional sin ambigüedades.

#### Manejo de casos degenerados

- **CA-17.11:** Dado que el periodo previo equivalente no contiene datos persistidos (por ejemplo, el sistema aún no estaba operativo durante ese periodo, o el rango personalizado calculado como previo equivalente cae fuera de la operación del sistema), cuando la vista intenta calcular la comparación, entonces los paneles muestran únicamente los valores del periodo actual y comunican explícitamente "no hay datos en el periodo previo equivalente para comparar" en lugar de mostrar la variación porcentual sobre datos inexistentes. Si ni el periodo actual ni el previo equivalente tienen datos, la vista comunica "no hay datos en ninguno de los dos periodos" sin renderizar los paneles comparativos.

- **CA-17.12:** Dado que alguno de los valores agregados de un indicador es cero en el periodo previo equivalente (por ejemplo, throughput cero porque ningún vehículo cruzó la intersección durante una franja sin operación), cuando el sistema calcula la variación porcentual entre periodos, entonces, en lugar de mostrar un valor indefinido o infinito, el panel comunica "sin datos suficientes para comparar" para ese indicador específico, mostrando los dos valores absolutos individualmente sin la variación porcentual ni la semántica de mejora o empeoramiento.

- **CA-17.13:** Dado que alguno de los dos periodos comparados tiene cobertura parcial de datos (por ejemplo, el sistema empezó a operar a mitad del periodo previo equivalente), cuando la vista presenta los paneles, entonces los valores calculados se acompañan de una indicación visible de que la cobertura del periodo es parcial, mostrando el porcentaje o intervalo efectivamente cubierto por cada periodo. La indicación es análoga a la de CA-16.18 de HU-16 y aplica a ambos periodos comparados.

- **CA-17.14:** Dado que el componente responsable de la persistencia del histórico o del cálculo de los indicadores deja de responder, cuando el Gerente consulta la vista, entonces el sistema indica explícitamente la indisponibilidad temporal, mostrando los últimos resultados comparativos conocidos marcados como "no actualizados" e indicando el tiempo transcurrido desde el último cálculo exitoso (DHU-005 Caso B aplicado al motor de cálculo y a la persistencia, igual que CA-16.19 de HU-16). La marca "no actualizados" aplica simultáneamente a los dos periodos del panel, porque ambos dependen del mismo componente caído. Si no hay resultados comparativos previos disponibles porque la vista se abre por primera vez con el componente caído, la vista comunica "no se pueden calcular los indicadores en este momento" sin renderizar los paneles.

#### Control de acceso

- **CA-17.15:** Dado que un usuario con rol Operador o Administrador intenta acceder a la vista comparativa del Gerente, cuando solicita la ruta correspondiente, entonces el sistema deniega el acceso conforme a la política de control de acceso por rol (HU-01 del Bloque A), respondiendo con HTTP 403 si la solicitud llega vía API o redirigiéndolo fuera de la vista si el intento es vía navegación frontend.

- **CA-17.16:** Dado que el Gerente no ha iniciado sesión, cuando intenta acceder a la vista comparativa, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Reutilización del sustrato técnico de HU-16:** HU-17 no introduce sustrato técnico nuevo. La persistencia del histórico de estados (declarada en CA-16.1 a CA-16.3) y las definiciones operacionales de los cuatro indicadores (declaradas en CA-16.9 a CA-16.12) son compartidas. La vista comparativa hace dos consultas independientes sobre el mismo histórico (una por periodo) y calcula los cuatro indicadores sobre cada consulta usando las mismas funciones de agregación documentadas en HU-16. Esta reutilización es coherente con la cohesión del Bloque F como entrega cohesiva al Gerente.

- **Estado compartido del selector entre HU-16 y HU-17 (CA-17.1, CA-17.3):** la compartición de estado se implementa en el frontend (por ejemplo, mediante context global, store, o equivalente del framework elegido), sin tabla en base de datos ni columna en la tabla de usuarios. El estado es volátil: vive durante la sesión activa y se descarta al cerrar sesión o al expirar el JWT (CA-01.6 de HU-01). Esta volatilidad es deliberada: persistir la última selección por usuario en base de datos abre un caso de uso (continuidad entre sesiones) que no está cerrado para MVP1 y agregaría complejidad de modelo sin valor proporcional, conforme a la justificación de CA-16.4.

- **Definición de "periodo previo equivalente" (CA-17.4):** la convención es del tipo "periodo del mismo tipo inmediatamente anterior", siguiendo el patrón estándar de herramientas analíticas profesionales que la implementan en su funcionalidad equivalente. Para el caso del rango personalizado, el cálculo del previo equivalente consiste en restar la duración del rango actual a su fecha de inicio: el nuevo rango va desde "fecha de inicio actual menos duración" hasta "fecha de inicio actual menos una unidad mínima de tiempo". Este cálculo es determinista y reproducible.

- **Semántica de mejora y empeoramiento por indicador (CA-17.7):** la decisión de que el código visual de la variación comunique dirección de mejora o empeoramiento (en lugar de signo aritmético) responde a la naturaleza distinta de los cuatro indicadores: tiempo promedio de espera, longitud máxima de cola y demora promedio acumulada mejoran al disminuir; throughput mejora al aumentar. Si todos los indicadores compartieran el mismo código visual basado en signo aritmético (verde si aumenta, rojo si disminuye), tres de los cuatro indicadores se leerían sistemáticamente al revés. La traducción a mejora o empeoramiento la hace el sistema explícitamente en la presentación, no se deja al Gerente, para reducir carga cognitiva y prevenir errores de interpretación. Esta convención es la que adoptan herramientas analíticas profesionales con indicadores heterogéneos (por ejemplo, en Google Analytics el "bounce rate" se marca con rojo aunque aumente, porque aumentar bounce rate es empeoramiento).

- **Sin disgregación por dirección en la vista comparativa:** la asimetría con HU-16 (que sí admite disgregación por dirección de Tiempo promedio de espera y Longitud máxima de cola mediante toggle dentro de la card, conforme a CA-16.14) es intencional. HU-16 puede admitir disgregación porque trabaja sobre un solo periodo y la información a presentar es manejable. HU-17 trabaja sobre dos periodos simultáneamente; cada panel comparativo ya muestra dos series temporales, dos valores agregados y una variación porcentual, y agregar disgregación por dirección duplicaría las series y los valores en cada panel saturando la vista sin servir al caso de uso del Gerente (tendencia global del sistema, no qué dirección específica varió). El drill-down detallado por dirección o por evento específico es alcance de F15 (MVP2).

- **Significancia estadística rigurosa no calculada en MVP1:** la decisión de mostrar la variación porcentual cualquiera sea su magnitud, sin marcar "sin cambio significativo" por debajo de un umbral, responde a que (a) calcular significancia estadística sobre series temporales agregadas requiere supuestos distribucionales que no están justificados en MVP1, (b) un umbral fijo arbitrario podría ocultar tendencias reales relevantes, (c) el Gerente puede interpretar visualmente si la magnitud observada es significativa para su decisión gerencial. El caso degenerado de variación matemáticamente indefinida (división por cero cuando el agregado del periodo previo es cero) se maneja explícitamente en CA-17.12: el panel muestra los dos valores absolutos sin la variación porcentual ni la semántica visual de mejora, comunicando "sin datos suficientes para comparar".

- **Latencias aceptables y paralelización del cálculo:** la apertura de la vista y el recálculo al cambiar de periodo se rigen por los mismos umbrales sugeridos en HU-16 (≤ 3 segundos para apertura, ≤ 10 segundos para recálculo), candidatos a RNF. Aunque la vista comparativa hace el doble de trabajo respecto a HU-16 (calcular dos periodos en lugar de uno), ese trabajo es naturalmente paralelizable: las dos consultas al histórico son independientes y pueden ejecutarse simultáneamente. Si el cuello de botella es la base de datos, la segunda consulta puede aprovechar caché de la primera (mismo esquema, misma agregación, distinto rango temporal). Si la realidad medida es peor en MVP1 por volumen de datos persistidos, se reporta conforme a D-005.

- **Robustez ante interrupción (CA-17.14):** aplica el mismo Caso B de DHU-005 que CA-16.19 de HU-16, adaptado al hecho de que esta vista depende de dos consultas al histórico (una por periodo). Si la persistencia o el motor de cálculo caen, ambas consultas se ven afectadas simultáneamente; la marca "no actualizados" aplica a ambos periodos del panel comparativo, no a uno solo, porque la causa raíz es compartida.

- **Coherencia visual con HU-16 (CA-17.9 y CA-17.10):** el orden de los paneles y los tooltips de definición operacional reutilizan exactamente los de HU-16. Esta consistencia entre las dos vistas del Gerente reduce la carga cognitiva de navegación: el Gerente reconoce los indicadores en la misma posición y con la misma explicación.

- **Sin selector independiente entre vistas:** la decisión de compartir estado del selector entre HU-16 y HU-17 (en lugar de tener un selector independiente en cada vista) es deliberada y responde al caso de uso primario del Gerente: cuando navega entre las dos vistas, lo hace para "ver lo mismo desde dos ángulos" (estado del periodo vs tendencia respecto al previo), no para "ver dos periodos distintos". Si en el futuro surge un caso de uso de selectores independientes (por ejemplo, comparar "esta semana" en una vista contra "el mes anterior" en otra), el cambio es local a la implementación del estado del frontend y no afecta el contrato de las HUs.

- **Trabajo futuro adyacente:** el drill-down sobre eventos específicos identificados en la comparativa (F15), la exportación de la comparativa a PDF o Excel (F16), la significancia estadística rigurosa de las variaciones, y la comparativa contra periodos no inmediatamente anteriores (por ejemplo, "esta semana vs misma semana del año pasado") son trabajos fuera del alcance del Bloque F. F15 y F16 son MVP2; los dos últimos se evalúan como mejoras posteriores si se justifican con casos concretos.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento (apertura):** la vista comparativa debería abrirse en tiempo razonable, aprovechando la paralelización de las dos consultas al histórico (criterio sugerido: ≤ 3 segundos, igual que HU-16).
- **RNF de rendimiento (recálculo):** el recálculo al cambiar de periodo debería completarse con indicador de carga visible en tiempo razonable (criterio sugerido: ≤ 10 segundos, igual que HU-16).
- **RNF de robustez:** comportamiento ante caída del motor de cálculo o de la persistencia del histórico (CA-17.14, DHU-005 Caso B). Los resultados comparativos se marcan como "no actualizados" en ambos periodos simultáneamente, no en uno solo.
- **RNF de manejabilidad de datos faltantes:** periodos sin datos (CA-17.11), variaciones matemáticamente indefinidas por agregado previo cero (CA-17.12) y cobertura parcial (CA-17.13) se comunican explícitamente al Gerente, sin mostrar valores espurios o calculados sobre vacío.
- **RNF de usabilidad:** la vista debe permitir al Gerente interpretar la comparativa sin documentación externa. La semántica de mejora y empeoramiento (CA-17.7) debe ser inequívoca a primer golpe de vista, sin requerir leer la definición de cada indicador. La distinción visual entre las dos series del gráfico (actual vs previo equivalente) debe ser clara.
- **RNF de accesibilidad:** la semántica de mejora o empeoramiento no debe depender exclusivamente del color (CA-17.7 redundancia con signo numérico y flecha direccional). Los tooltips deben ser activables tanto con teclado como con dispositivo apuntador. Las dos series del gráfico deben ser distinguibles más allá del color (por ejemplo, mediante estilo de línea o marcador).
- **RNF de consistencia entre vistas:** el orden de los indicadores y los tooltips de definición operacional son consistentes entre HU-16 y HU-17 (CA-17.9, CA-17.10), reduciendo carga cognitiva al navegar entre las dos vistas del Gerente.
- **RNF de seguridad:** la consulta de la vista comparativa está restringida al rol Gerente (CA-17.15). Un usuario sin ese rol no debe poder consultar la comparativa bajo ninguna circunstancia, incluso si conoce los endpoints subyacentes.

---

## Resumen del Bloque F

| HU | Título | Sujeto | Tipo | Feature(s) origen | Clasif. MVP |
|---|---|---|---|---|---|
| HU-16 | Consulta de KPIs operativos sobre periodo seleccionable | Gerente | Persona | F12 + F13 fusionadas, F30 inglobada | MVP1 |
| HU-17 | Vista comparativa entre periodos | Gerente | Persona | F14, reutiliza sustrato de HU-16 | MVP1 |

**Total Bloque F: 2 HUs operativas + 0 TTH nuevas.**

F12 (Dashboard ejecutivo) y F13 (Selector de periodo) se fusionaron en HU-16 conforme a DHU-016 subsección I (cohesión semántica: el selector no entrega valor sin el dashboard que gobierna).

F30 (Persistencia de estados históricos) queda inglobada en CA-16.1 a CA-16.3 de HU-16, conforme a la regla cerrada en el Bloque A y al patrón establecido por F31 inglobada en CA-08.1 de HU-08.

F15, F16 (drill-down y exportación) son MVP2, fuera del alcance del Bloque F.

---

## Cambios aplicados a documentos previos como consecuencia del Bloque F

Durante la redacción del Bloque F, las siguientes modificaciones se aplicaron a documentos previos:

1. **Cierre de DHU-016** (`DECISIONS_HU.md`): consolida las decisiones de redacción del Bloque F (numeración compactada desde HU-15, F30 inglobada como CAs en HU-16, fuente operacional independiente y agnóstica en MVP1, definiciones operacionales de los cuatro KPIs cerradas, granularidad de treinta segundos del histórico, presets del selector cerrados, definición de periodo previo equivalente, concurrencia entre Gerentes no aplicable, composición F12+F13 en una sola HU, aplicación de DHU-005 Caso B).

2. **Actualización de fichas en `FEATURE_BACKLOG_DETALLADO.md`:** las fichas de F12, F13, F14 y F30 actualizan su columna "Modelado" para apuntar a HU-16 y HU-17, y recogen las decisiones cerradas en DHU-016 sobre presets, periodos, KPIs operacionales, granularidad y exclusión del trimestre en MVP1.

3. **Actualización de "Próximos pasos" en HU_BLOQUE_A/B/C/D/E.md:** el Bloque F ya cerrado; resta solo el MVP2.

4. **Actualización de `LEAN_INCEPTION_CEREBROVIAL.md`:** se incorpora la referencia a `HU_BLOQUE_F.md` en la lista de documentos relacionados.

No se introducen modificaciones al contenido sustantivo de las HUs MVP1 redactadas en bloques previos (HU-01 a HU-15), ni a las TTH (TTH-01 a TTH-11), ni a las decisiones técnicas (D-001 a D-009), ni a las decisiones metodológicas previas (DHU-001 a DHU-015).

---

## Decisiones que aplicaron a este bloque

Durante la redacción del Bloque F se cerró la siguiente decisión formal (en `DECISIONS_HU.md`):

- **DHU-016:** decisiones de redacción del Bloque F (consolidadas en subsecciones A a J). Cubre numeración compactada desde HU-15, F30 inglobada como CAs en HU-16 (no TTH), fuente operacional independiente y agnóstica en MVP1 (Opción A entre las tres evaluadas), definiciones operacionales cerradas de los cuatro KPIs (tiempo promedio de espera, longitud máxima de cola por dirección, throughput, demora promedio acumulada), granularidad del histórico de treinta segundos sin política de retención automática en MVP1, cinco modos del selector (cuatro presets + rango personalizado) con convenciones inequívocas y exclusión del trimestre en MVP1, definición de "periodo previo equivalente" para los cinco modos, concurrencia entre Gerentes declarada como no aplicable (HUs read-only), fusión de F12 + F13 en una sola HU sin dashboard integrador dedicado, y aplicación de DHU-005 Caso B al motor de cálculo de indicadores y a la persistencia del histórico.

Adicionalmente, el cierre del Bloque F implica el cierre formal del **MVP1 redactado**: con HU-17 cerrada, las 16 HUs operativas MVP1 (HU-01 a HU-08, HU-10 a HU-17) y las 11 TTH (TTH-01 a TTH-11) están redactadas y aprobadas. Resta solo el MVP2 (cuatro HUs adicionales: F15, F16, F19, F28; HU-09 ya redactada como única HU MVP2 anticipada).

---

## Próximos pasos

Esta sesión cerró el Bloque F y, con ello, la redacción del MVP1 del Product Backlog. El MVP2 también fue cerrado posteriormente el 2026-05-16 (DHU-017, con HU-18 a HU-21 + HU-09 ya anticipada en el Bloque B). **Con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 21 HUs operativas (HU-01 a HU-21) y 11 TTH (TTH-01 a TTH-11).** Los siguientes pasos del proyecto, fuera del alcance del Product Backlog funcional, en sesiones futuras:

1. **Documento de Requisitos Funcionales y No Funcionales (RF/RNF).** Pendiente desde DHU-007: consolidar los "Candidatos a RNF" de todas las HUs (HU-01 a HU-21) en un documento único aprobado, numerando cada RNF y reemplazando los umbrales hardcodeados en las HUs por referencias al documento formal. Esta es una sesión dedicada futura.

2. **Ceremonias de estimación y priorización.** Una vez completo el backlog (MVP1 + MVP2 + RF/RNF), se ejecutarán **Planning Poker** sobre todas las HUs y TTH, y **MoSCoW** para priorización formal del orden de construcción.

3. **Implementación SCRUM del MVP1.** Construcción de las 16 HUs operativas y 11 TTH del MVP1 en sprints, conforme al cronograma del Bloque 7 del MVP Canvas del Inception. Sprint Goals derivados de los bloques del Sequencer. El MVP2 (5 HUs adicionales) entra al sprint si hay holgura de cronograma, conforme a la semántica refinada por DHU-012.

4. **SDD (Software Design Document)**, siguiente entregable académico mayor del proyecto.

---

## Documentos relacionados

- `HU_BLOQUE_A.md` — Bloque A del Product Backlog (acceso al sistema, 1 HU operativa + 3 TTH).
- `HU_BLOQUE_B.md` — Bloque B del Product Backlog (8 HUs: HU-02 a HU-09).
- `HU_BLOQUE_C.md` — Bloque C del Product Backlog (3 HUs operativas: HU-10, HU-11, HU-12 + 2 TTH).
- `HU_BLOQUE_D.md` — Bloque D del Product Backlog (3 HUs operativas: HU-13, HU-14, HU-15).
- `HU_BLOQUE_E.md` — Bloque E del Product Backlog (0 HUs operativas; mapeo a TTH-07 a TTH-11).
- `HU_MVP2.md` — MVP2 del Product Backlog (HU-18, HU-19, HU-20, HU-21; HU-09 reside en `HU_BLOQUE_B.md`). HU-18 (drill-down) y HU-19 (exportación) acceden desde HU-16 y HU-17 de este bloque.
- `DECISIONS_HU.md` — Decisiones metodológicas sobre HUs (DHU-001 a DHU-022).
- `DECISIONS.md` — Decisiones técnicas del producto (D-001 a D-009).
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-01 a TTH-11.
- `LEAN_INCEPTION_CEREBROVIAL.md` — Inception completo aplicado al proyecto.
- `FEATURE_BACKLOG_DETALLADO.md` — Detalle completo de las 41 features identificadas (29 MVP1 + 5 MVP2 + 7 Trabajos Futuros).
- `EVOLUCION_TESIS.md` — Narrativa de las 4 fases del proyecto.
- `LEAN_INCEPTION_INVESTIGACION.md` — Fundamentación bibliográfica.
- `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` — Documento normativo denso con catálogo de 22 RF y 53 RNF clasificados según ISO/IEC 25010:2023, redactado el 2026-05-18 ejecutando DHU-007 según las decisiones metodológicas consolidadas en DHU-019. `RF_RNF_LITE.md` — versión lite de lectura humana, derivada conforme al modelo de dos documentos cerrado en DHU-019 subsección H.
