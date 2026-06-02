# Historias de Usuario — Bloque B (Cerradas)

> Segunda entrega del Product Backlog del proyecto CerebroVial.
>
> **Estado:** Bloque B cerrado y aprobado. Bloques A, C, D, E y F del MVP1 cerrados, y MVP2 también cerrado el 2026-05-16 (DHU-017). **Con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 23 HU operativas (HU-01 a HU-23) y 13 TTH (TTH-01 a TTH-13), tras las ampliaciones DHU-028 (HU-22 + TTH-12) y DHU-029 (HU-23 + TTH-13).** Pendiente: documento RF/RNF (DHU-007), Planning Poker, MoSCoW, implementación SCRUM del MVP1. HU-09 fue redactada en este bloque como única HU MVP2 anticipada bajo la antigua semántica "fuera del sprint", suavizada por DHU-012 a "candidata a construcción condicional a holgura del cronograma"; HU-09 conserva su ubicación en este documento sin trasladarse a `HU_MVP2.md`, conforme a DHU-017 subsección B.
>
> **Fecha de cierre:** 2026-05-13
> **Fecha de actualización v2:** 2026-05-17 (DHU-018 aplicada retroactivamente: Resumen ejecutivo en HU-02 a HU-09)

---

## Contexto

Este documento contiene las Historias de Usuario del **Bloque B — Operador, núcleo de monitoreo** del Sequencer del Lean Inception (ver `LEAN_INCEPTION_CEREBROVIAL.md`, sección 9, y `FEATURE_BACKLOG_DETALLADO.md`, Bloque B).

Las HUs se redactan en el formato del documento de referencia académica (`Desarrollo_Agil.pdf`, Tablas 9 y 13): "Como X, quiero Y, para Z" con criterios de aceptación Given-When-Then.

Las HUs del Bloque B siguen las reglas metodológicas establecidas y refinadas durante la redacción:

- **DHU-001 a DHU-004** (del Bloque A): sujetos válidos, exclusión del Equipo de Desarrollo, TTH como categoría separada.
- **DHU-005** (refinada en este bloque): principio de robustez ante interrupción de fuente, con Casos A y B.
- **DHU-006:** HUs agnósticas a la implementación.
- **DHU-007:** RNF declarados como tales en sección específica al final de cada HU.
- **DHU-018** (aplicada retroactivamente el 2026-05-17): patrón "Resumen ejecutivo" al inicio de cada HU para uniformidad de lectura. Aditiva, no modifica contenido sustantivo.

Ver `DECISIONS_HU.md` para fundamentación completa.

---

## Mapeo de features del Bloque B

Las 10 features del Bloque B (F02 a F11) se mapearon a 8 HUs según el siguiente criterio (Nivel B de granularidad: una HU por elemento de información del dashboard):

> **Nota sobre el conteo del Bloque B:** existen dos formas válidas de contar las features de este bloque que conviven en la documentación. (a) El **Sequencer del Inception** (`LEAN_INCEPTION_CEREBROVIAL.md` sección 9) declara *"Bloque B — Operador, monitoreo (9 features): F02 a F10"*, porque el Sequencer cuenta únicamente features MVP1 dentro de cada bloque y F11 está clasificada como MVP2. (b) Los documentos del backlog (`HU_BLOQUE_B.md`, `FEATURE_BACKLOG_DETALLADO.md`) cuentan 10 features (F02 a F11), porque agrupan las features por **afinidad temática** independientemente de su clasificación MVP1/MVP2: F11 (Módulo de notas del Operador) es soporte al monitoreo del Operador y pertenece temáticamente a este bloque, aunque su construcción está condicionada a la holgura del cronograma. Las dos convenciones son consistentes; este documento adopta la convención (b) y marca explícitamente cada HU con su clasificación MVP en el resumen del bloque.

| HU | Título | Features que cubre |
|---|---|---|
| HU-02 | Monitoreo del estado actual de la intersección en tiempo real | F03 + F04 (F02 cubierto por composición) |
| HU-03 | Visualización de predicción de congestión a corto plazo | F05 |
| HU-04 | Vista combinada del estado actual y la predicción de tráfico | F06 |
| HU-05 | Visualización de la estrategia de control activa | F07 |
| HU-06 | Explicación de la razón de selección de estrategia | F08 |
| HU-07 | Notificación de cambios de estrategia del motor | F09 |
| HU-08 | Consulta del historial de decisiones del motor | F10 (+ F31 inglobada como CA) |
| HU-09 | Registro de notas e incidencias del turno | F11 (MVP2) |

**Total Bloque B:** 10 HUs (7 MVP1 + 1 MVP2 + 2 ampliaciones: HU-22 habilitada por TTH-12, HU-23 habilitada por TTH-13). F02 (Dashboard principal) queda cubierto por la composición visual de HU-02, HU-03, HU-04, HU-05 y HU-06 en una vista única, sin generar HU propia (es contenedor, no funcionalidad independiente).

---

## HU-02 — Monitoreo del estado actual de la intersección en tiempo real

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | visualizar en tiempo real el flujo vehicular y la longitud de cola en cada acceso de la intersección |
| **Para** | detectar de inmediato condiciones anómalas (congestión incipiente, colas excesivas, flujos desbalanceados) y reaccionar antes de que escalen a problemas mayores |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** F03 (Visualización de flujo vehicular) + F04 (Visualización de cola por dirección). Componente visual integrador F02 cubierto por la composición con HU-03, HU-04 y HU-05.

### Resumen ejecutivo

**Qué entrega:** vista de monitoreo en tiempo real para el Operador. Muestra, por cada acceso de la intersección, el flujo vehicular y la longitud de cola con indicador de color verde/amarillo/rojo según umbrales configurables. Actualización automática sin recarga.

**CAs críticos:** CA-02.1 (presentación de flujo y cola por acceso), CA-02.2 (actualización ≤ 5 s sin recarga), CA-02.4 (robustez ante pérdida de fuente de medición — DHU-005 Caso A), CA-02.5 (redirección al login si no hay sesión).

**Dependencias:** HU es agnóstica a la fuente de datos (DHU-006); en MVP1 la fuente operacional es el entorno simulado (D-008), en operación real sería la salida del módulo sensor (TTH-08). Los umbrales de color son parametrizables vía HU-15 (familia "Visualización del estado del tráfico"). Aplica DHU-005 Caso A.

**Notas clave:** F02 (dashboard principal) queda cubierto por composición visual con HU-03, HU-04, HU-05; no se redacta como HU propia. Geometría de N accesos (típicamente 4) configurable. CA-02.4 introduce la marca pasiva del Bloque B (DHU-005 Caso A: "desactualizado"), complementaria a la alerta activa transversal de HU-10 según DHU-009.

### Descripción

El Operador necesita un panel de monitoreo que le muestre el estado actual del tráfico en la intersección sin tener que ejecutar consultas ni cambiar de vista. El panel se actualiza automáticamente a medida que llegan nuevas mediciones del tráfico. El Operador debe poder identificar de un vistazo si hay una dirección con flujo inusualmente alto o con una cola que está creciendo más rápido de lo habitual.

El panel muestra dos elementos por cada acceso de la intersección:

1. **Flujo vehicular** (vehículos por minuto en el último intervalo de medición).
2. **Longitud de cola** (vehículos esperando en ese acceso, con indicador de nivel según umbrales configurables).

### Criterios de aceptación

- **CA-02.1:** Dado que el Operador ha iniciado sesión, cuando ingresa a la vista de monitoreo, entonces el sistema muestra, para cada acceso de la intersección, el flujo vehicular actual (en vehículos por minuto) y la longitud de cola actual (en número de vehículos).

- **CA-02.2:** Dado que el Operador tiene la vista de monitoreo abierta, cuando llega una nueva medición del tráfico, entonces los valores mostrados se actualizan automáticamente sin necesidad de recargar la página, con una latencia máxima de 5 segundos desde que la medición se genera.

- **CA-02.3:** Dado que el Operador está observando la longitud de cola en un acceso, cuando esa cola supera el umbral configurado como "alto", entonces el indicador visual asociado a ese acceso cambia a color rojo. Cuando está entre el umbral "medio" y "alto", se muestra en amarillo. Cuando está por debajo del umbral "medio", se muestra en verde.

- **CA-02.4:** Dado que el sistema deja de recibir mediciones del tráfico (por cualquier causa), cuando el Operador está observando el panel, entonces el sistema mantiene en pantalla los últimos valores conocidos, los marca visualmente como "desactualizados" e indica el tiempo transcurrido desde la última actualización (DHU-005 Caso A).

- **CA-02.5:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder a la vista de monitoreo, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Origen de los datos del tráfico:** La HU es agnóstica a la fuente. En el entorno de operación real, las mediciones provienen de los componentes técnicos definidos en el Bloque E. En el entorno de validación académica, las mediciones se generan a partir del entorno de validación documentado en D-008. El frontend consume una API estable independiente del origen.
- **Mecanismo de actualización:** SSE (Server-Sent Events) o WebSocket. Decisión técnica a cerrar en el sprint correspondiente.
- **Umbrales de cola:** Los valores que disparan los colores verde/amarillo/rojo son parametrizables y se configuran a través de la HU del Bloque D que cubre la configuración del motor. Para MVP1 se aceptan valores por defecto, con la decisión documentada como "configurable en iteraciones futuras".
- **Geometría de la intersección:** Las direcciones (4 accesos como caso típico) se definen en la configuración del sistema. El código debe soportar N accesos.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** latencia máxima de actualización ≤ 5 s (CA-02.2). → RNF-PERF-01 (Actualización en tiempo real de la presentación operativa).
- **RNF de robustez:** comportamiento ante pérdida de fuente de datos (CA-02.4, DHU-005 Caso A). → RNF-REL-01 (Robustez ante interrupción de fuente).
- **RNF de usabilidad:** umbrales de color verde/amarillo/rojo en indicador de cola (CA-02.3). → RNF-INT-01 (Usabilidad operativa de las presentaciones del Operador), RNF-INT-02 (Accesibilidad WCAG 2.1 nivel AA).

---

## HU-03 — Visualización de predicción de congestión a corto plazo

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | visualizar la predicción del nivel de congestión en los próximos minutos para cada acceso de la intersección |
| **Para** | anticiparme a la formación de congestión y disponer de tiempo de reacción antes de que el problema se materialice |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** F05 (Panel de predicción de congestión).

### Resumen ejecutivo

**Qué entrega:** vista de predicción del nivel de congestión a corto plazo por cada acceso de la intersección, expresado en escala ordinal 0-5 sobre el horizonte futuro configurado. Resaltado visual cuando el nivel predicho supera el umbral configurado.

**CAs críticos:** CA-03.1 (presentación de niveles 0-5 por acceso), CA-03.2 (actualización ≤ 5 s sin recarga), CA-03.4 (robustez ante interrupción de predicciones — DHU-005 Caso B), CA-03.5 (redirección al login).

**Dependencias:** consume el modelo predictivo definido en TTH-09 (D-006 GRU univariado, D-009 jam level 0-5) de forma agnóstica. El horizonte de predicción y el umbral de congestión son parametrizables vía HU-15 (familia "Predicción y evaluación del modelo"). Aplica DHU-005 Caso B (componente interno decisor → "no confirmada"), DHU-006.

**Notas clave:** la escala 0-5 es el constructo "jam level" de Waze adoptado en D-009 — permite intercambiabilidad de fuente sin reentrenar el modelo. La HU no nombra Waze ni GRU (DHU-006); usa "nivel de congestión 0-5" como excepción declarada en DHU-006. Valor por defecto del umbral de resaltado: nivel ≥ 3.

### Descripción

El Operador necesita ver no solo lo que pasa ahora (cubierto por HU-02), sino lo que el sistema anticipa que va a pasar en los próximos minutos. La predicción se expresa en una escala ordinal de nivel de congestión que va desde 0 (flujo libre) hasta 5 (vía cerrada), siguiendo el estándar de la industria documentado en D-009. Para cada acceso de la intersección, el panel muestra el nivel de congestión proyectado en el horizonte futuro configurado.

Esta escala unificada permite que el sistema sea agnóstico a la fuente de datos: tanto el entorno de validación como una fuente de datos real producen la misma variable de estado, garantizando portabilidad sin reentrenamiento del modelo.

### Criterios de aceptación

- **CA-03.1:** Dado que el Operador ha iniciado sesión y existe un horizonte de predicción configurado, cuando ingresa a la vista de predicción, entonces el sistema muestra, para cada acceso de la intersección, el nivel de congestión predicho (en escala 0–5) proyectado hasta el horizonte configurado.

- **CA-03.2:** Dado que el Operador tiene la vista de predicción abierta, cuando el sistema genera una nueva predicción, entonces los valores mostrados se actualizan automáticamente sin necesidad de recargar la página, con una latencia máxima de 5 segundos desde que la predicción se genera.

- **CA-03.3:** Dado que el Operador está observando la predicción, cuando para algún acceso el nivel predicho supera el umbral configurado como "congestión" (por defecto, nivel ≥ 3), entonces ese acceso se resalta visualmente para alertar al Operador del problema anticipado.

- **CA-03.4:** Dado que el sistema deja temporalmente de generar predicciones (por cualquier causa), cuando el Operador está observando el panel, entonces el sistema mantiene en pantalla las últimas predicciones conocidas, las marca visualmente como "no confirmadas" e indica el tiempo transcurrido desde la última confirmación (DHU-005 Caso B).

- **CA-03.5:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder a la vista de predicción, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Variable predicha:** nivel de congestión en escala 0–5, según el constructo definido en D-009. El sistema es agnóstico a la fuente de datos: en validación se calcula a partir de las mediciones del entorno de simulación, y en operación real desde el feed correspondiente.
- **Granularidad temporal:** el sistema entrega la predicción como una proyección en el horizonte (puntos a intervalos o valor agregado del horizonte; decisión a cerrar en el sprint).
- **Horizonte configurable:** parámetro de configuración del sistema; valor por defecto a cerrar en el sprint.
- **Umbral de congestión:** configurable en la HU del Bloque D dedicada a configuración del motor; valor por defecto nivel ≥ 3.
- **Relación con la vista panorámica de red (HU-22):** esta HU es la **vista de detalle de intersección** (predicción de congestión por acceso de una intersección concreta). La **vista panorámica de red** (HU-22, ampliación posterior — DHU-028) es complementaria: muestra la congestión de toda la red por tramo de vía sobre un mapa geográfico. Son dos niveles de zoom del mismo sistema de monitoreo, no sustitutos. HU-22 usa la unidad "tramo de vía/arista" en lugar del "acceso", precisamente para permitir la representación geográfica continua de la red (ver DHU-028).

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** latencia máxima de actualización ≤ 5 s (CA-03.2). → RNF-PERF-01 (Actualización en tiempo real de la presentación operativa).
- **RNF de robustez:** comportamiento ante interrupción de predicciones (CA-03.4, DHU-005 Caso B). → RNF-REL-01 (Robustez ante interrupción de fuente).
- **RNF de calidad de predicción:** umbrales de error aceptable del modelo (MAE/RMSE sobre el ratio velocidad/free-flow). Pertenece al componente predictivo (Bloque E), no a la vista. → RNF-FUN-02 (Calidad del modelo predictivo), RNF-COM-02 (Interoperabilidad mediante constructo unificado de nivel de congestión).

---

## HU-04 — Vista combinada del estado actual y la predicción de tráfico

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | ver en una sola pantalla el estado actual del tráfico y la predicción a corto plazo de manera integrada |
| **Para** | comparar visualmente "lo que está pasando" con "lo que va a pasar" sin tener que cambiar de vista, y así detectar discrepancias o tendencias preocupantes antes de actuar |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** F06 (Vista combinada estado actual + predicción).

### Resumen ejecutivo

**Qué entrega:** vista única que integra el estado actual del tráfico (de HU-02) y la predicción a corto plazo (de HU-03) sobre los mismos accesos de la intersección, para que el Operador detecte de un vistazo discrepancias entre "lo que está pasando" y "lo que va a pasar".

**CAs críticos:** CA-04.1 (presentación integrada por acceso), CA-04.2 (resaltado de discrepancia estado-predicción), CA-04.3 (actualización ≤ 5 s sin recarga), CA-04.4 (robustez independiente para cada fuente — DHU-005 Casos A y B).

**Dependencias:** consume las mismas APIs que HU-02 (estado actual) y HU-03 (predicción), sin duplicar fuentes. Aplica DHU-005 Caso A para mediciones y Caso B para predicciones simultáneamente. La detección de discrepancia (CA-04.2) reutiliza los umbrales de cola de HU-02 y el umbral de congestión de HU-03.

**Notas clave:** según el backlog detallado de F06, esta es la feature más distintiva del Operador. El diseño visual concreto (línea temporal continua, paneles lado a lado o heatmap) se cierra mediante prototipado en sprint; la HU es agnóstica al diseño. Composición de presentación, no nueva fuente de datos.

### Descripción

Esta es la feature más distintiva del Operador, según el backlog detallado. El Operador necesita ver **simultáneamente** lo que está pasando (variables observadas de HU-02: flujo y cola) y lo que va a pasar (predicción de HU-03: nivel de congestión proyectado). La vista combinada permite al Operador detectar:

- **Convergencia esperable:** el estado actual ya es malo y la predicción confirma que seguirá empeorando.
- **Divergencia preocupante:** el estado actual parece estable pero la predicción anticipa congestión en breve.
- **Divergencia tranquilizadora:** el estado actual parece tenso pero la predicción anticipa que se aliviará.

La HU define el qué (las dos fuentes de información integradas y comparables) pero deja explícitamente abierto el diseño visual exacto, que se cierra mediante prototipado antes de implementar.

### Criterios de aceptación

- **CA-04.1:** Dado que el Operador ha iniciado sesión, cuando ingresa a la vista combinada, entonces el sistema muestra en una misma pantalla, para cada acceso de la intersección, el estado actual del tráfico (flujo y cola) y el nivel de congestión predicho en el horizonte configurado, organizados de manera que el Operador pueda comparar "ahora" y "futuro" de un vistazo.

- **CA-04.2:** Dado que el Operador está observando la vista combinada, cuando para algún acceso el nivel de congestión predicho supera el umbral configurado pero el estado actual está dentro de la normalidad, entonces el sistema resalta visualmente esa discrepancia para llamar la atención del Operador sobre la divergencia.

- **CA-04.3:** Dado que el Operador tiene la vista combinada abierta, cuando llega una nueva medición del estado actual o una nueva predicción, entonces los elementos correspondientes se actualizan automáticamente sin necesidad de recargar la página, con una latencia máxima de 5 segundos desde que el dato se genera, manteniendo la alineación temporal entre "ahora" y "futuro".

- **CA-04.4:** Dado que el sistema deja temporalmente de recibir mediciones del estado actual, o deja temporalmente de generar predicciones, cuando el Operador está observando la vista combinada, entonces el sistema indica visualmente cuál de las dos fuentes está desactualizada o no confirmada e informa el tiempo transcurrido desde la última actualización de cada una de forma independiente (DHU-005 Caso A para mediciones, Caso B para predicciones).

- **CA-04.5:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder a la vista combinada, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Decisión de diseño visual:** El backlog detallado documenta tres opciones de wireframe (línea temporal continua con punto "ahora" marcado y proyección futura; dos paneles lado a lado con escalas alineadas; heatmap con tiempo y direcciones). La decisión final se cierra mediante prototipado en papel o Figma antes de codear, no es parte de la HU. La HU es agnóstica al diseño concreto.
- **Composición sobre HU-02 y HU-03:** Esta vista reutiliza los componentes de monitoreo y predicción. No duplica fuentes de datos; consume las mismas APIs que las HUs anteriores. La integración es a nivel de presentación.
- **Coherencia de escalas:** Si el diseño elegido alinea "ahora" y "futuro" en una serie temporal, la unidad de tiempo en el eje debe ser consistente. Si usa visualizaciones independientes (paneles lado a lado), las métricas mostradas pueden ser distintas (flujo y cola observados vs jam level predicho) pero la asociación visual entre ambos debe ser clara para el mismo acceso.
- **Detección de discrepancia (CA-04.2):** La lógica que define "estado actual dentro de la normalidad" se basa en los mismos umbrales de cola usados en HU-02 (verde/amarillo/rojo). Si la cola está en verde o amarillo pero la predicción está en nivel ≥ 3, hay discrepancia y se resalta. La regla exacta se afina en el sprint, pero el principio queda anclado en la HU.
- **Relación con la vista panorámica de red (HU-22):** esta HU es la **vista de detalle de intersección** (estado actual + predicción integrados por acceso de una intersección concreta). La **vista panorámica de red** (HU-22, ampliación posterior — DHU-028) es complementaria: muestra la congestión de toda la red por tramo de vía sobre un mapa geográfico. Son dos niveles de zoom del mismo sistema de monitoreo, no sustitutos. HU-22 usa la unidad "tramo de vía/arista" en lugar del "acceso", para permitir la representación geográfica continua de la red (ver DHU-028).

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** latencia máxima de actualización ≤ 5 s, independiente para cada fuente (CA-04.3). → RNF-PERF-01 (Actualización en tiempo real de la presentación operativa).
- **RNF de robustez:** comportamiento ante desconexión de cualquiera de las dos fuentes de forma independiente (CA-04.4). → RNF-REL-01 (Robustez ante interrupción de fuente).
- **RNF de usabilidad:** alineación temporal entre estado actual y predicción debe ser visualmente inmediata (CA-04.1). Probablemente se valida con pruebas de usuario. → RNF-INT-01 (Usabilidad operativa de las presentaciones del Operador).

---

## HU-05 — Visualización de la estrategia de control activa

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | visualizar cuál es la estrategia de control que el sistema está aplicando actualmente en la intersección y los parámetros activos de esa estrategia |
| **Para** | entender qué decisión de control automático está vigente en cada momento y poder evaluar si es coherente con el estado del tráfico que observo |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** F07 (Panel del motor adaptativo — estrategia activa).

### Resumen ejecutivo

**Qué entrega:** vista pasiva del motor adaptativo. Muestra al Operador qué estrategia de control está vigente en este momento, con qué parámetros (tiempos de verde por acceso), y desde cuándo lleva aplicándose, para que pueda evaluar la coherencia entre la decisión automática del sistema y el estado del tráfico que él observa.

**CAs críticos:** CA-05.1 (nombre de estrategia + tiempos de verde por acceso), CA-05.2 (timestamp de activación de la estrategia), CA-05.4 (robustez ante caída del motor adaptativo — DHU-005 Caso B), CA-05.5 (redirección al login).

**Dependencias:** consume las decisiones expuestas por el motor adaptativo (TTH-10) de forma agnóstica. El nombre interno de la estrategia se mapea a etiqueta legible para el Operador (DHU-006). HU-06 cubre el "por qué" de la decisión; HU-07 cubre el aviso activo de cambio. Aplica DHU-005 Caso B; la alerta activa transversal ante caída del motor es responsabilidad de HU-10 (Bloque C, DHU-009).

**Notas clave:** la HU es agnóstica a la implementación del motor (Webster, Max Pressure, MTC se mencionan en TTH-10 y `CONTROL.md`, no aquí). El Operador ve etiquetas autoexplicativas, no códigos técnicos. Es vista pasiva: "qué está activo ahora"; la consulta de historial vive en HU-08, el aviso activo vive en HU-07.

### Descripción

El sistema selecciona automáticamente entre múltiples estrategias de control semafórico según el estado predicho y observado del tráfico (esta es la naturaleza adaptativa del sistema, núcleo del Objetivo 3 del producto). Pero el Operador necesita saber **qué estrategia se está aplicando en este momento** y **con qué parámetros**, por dos razones operativas:

1. **Trazabilidad de la operación:** si reporta un comportamiento anómalo del sistema, debe poder decir "en ese momento se estaba aplicando la estrategia X con tiempos Y".
2. **Coherencia percibida:** si el Operador ve colas largas pero la estrategia activa es la que se usa para flujo libre, hay una incoherencia que debe poder identificar.

El panel muestra el nombre de la estrategia vigente (sin exponer detalles internos del motor) y los parámetros activos: tiempos de verde asignados a cada acceso de la intersección. El panel también muestra cuándo se aplicó la estrategia actual, para que el Operador sepa cuánto tiempo lleva vigente.

### Criterios de aceptación

- **CA-05.1:** Dado que el Operador ha iniciado sesión y el sistema tiene una estrategia de control vigente, cuando ingresa al panel de la estrategia activa, entonces el sistema muestra el nombre de la estrategia actualmente aplicada y los tiempos de verde asignados a cada acceso de la intersección.

- **CA-05.2:** Dado que el Operador está observando el panel de la estrategia activa, cuando el sistema muestra la estrategia vigente, entonces también indica el timestamp en que esa estrategia se activó, permitiendo al Operador conocer cuánto tiempo lleva aplicándose.

- **CA-05.3:** Dado que el Operador tiene el panel abierto, cuando el sistema cambia la estrategia activa o ajusta sus parámetros, entonces los valores mostrados se actualizan automáticamente sin necesidad de recargar la página, con una latencia máxima de 5 segundos desde que el cambio se produce.

- **CA-05.4:** Dado que el sistema no puede determinar la estrategia vigente por cualquier causa (por ejemplo, porque el motor adaptativo no está respondiendo), cuando el Operador está observando el panel, entonces el sistema mantiene en pantalla la última estrategia conocida, la marca visualmente como "no confirmada" e indica el tiempo transcurrido desde la última confirmación (DHU-005 Caso B).

- **CA-05.5:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder al panel de la estrategia activa, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Nombres de estrategias:** El conjunto de estrategias disponibles está definido en las decisiones técnicas del producto (D-001 y referencias del motor adaptativo). La HU no fija los nombres internos; el frontend muestra etiquetas legibles para el Operador. La asociación entre identificador interno y etiqueta visible se cierra en el sprint, pero el principio es que **el Operador ve nombres autoexplicativos**, no códigos técnicos.
- **Estructura de parámetros:** Los tiempos de verde se muestran como una tabla o lista con un par "acceso → segundos". La unidad temporal es la misma para todos los accesos. Si en el futuro se agregan otros parámetros (por ejemplo, tiempo de amarillo, offset entre fases), el panel debe poder extenderse sin romper el diseño.
- **Composición en el dashboard principal:** Este panel forma parte del dashboard único del Operador (composición visual F02). Su tamaño y posición exacta se definen en el prototipado del dashboard, no en esta HU.
- **Relación con HU-06 (explicación) y HU-07 (notificación de cambio):** Esta HU cubre solo "qué estrategia está activa ahora". El "por qué" se cubre en HU-06. El "aviso de cambio" se cubre en HU-07. Las tres se entregan integradas pero son HUs distintas con CAs distintos.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** latencia máxima de actualización ≤ 5 s (CA-05.3). → RNF-PERF-01 (Actualización en tiempo real de la presentación operativa).
- **RNF de robustez:** comportamiento ante caída del motor adaptativo (CA-05.4, DHU-005 Caso B). La alerta activa al Operador ante caída del componente es responsabilidad transversal del Bloque C. → RNF-REL-01 (Robustez ante interrupción de fuente).
- **RNF de usabilidad:** los nombres de estrategias deben ser autoexplicativos para el Operador sin requerir entrenamiento técnico. → RNF-INT-01 (Usabilidad operativa de las presentaciones del Operador).

---

## HU-06 — Explicación de la razón de selección de estrategia

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | leer una explicación breve y comprensible de por qué el sistema seleccionó la estrategia de control que está aplicando actualmente |
| **Para** | confiar en las decisiones automáticas del sistema y poder justificar ante terceros (supervisores, ciudadanos) por qué se está aplicando una determinada configuración de control |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** F08 (Explicación de razón de selección de estrategia).

### Resumen ejecutivo

**Qué entrega:** explicación textual breve y comprensible al Operador sobre por qué el sistema seleccionó la estrategia de control actualmente vigente. Texto plantillado con sustitución de variables del estado del tráfico que dispararon la selección, en lenguaje del dominio operativo.

**CAs críticos:** CA-06.1 (texto explicativo con valores del estado), CA-06.2 (actualización ≤ 5 s al cambiar estrategia), CA-06.3 (texto genérico de respaldo si la combinación no tiene plantilla), CA-06.4 (robustez ante caída del componente de explicación — DHU-005 Caso B).

**Dependencias:** consume la salida del componente que produce las plantillas (asociado al motor adaptativo TTH-10). HU-07 deriva una versión condensada para la notificación de cambio; HU-12 cubre la explicación equivalente para el modo degradado del Bloque C. Aplica DHU-005 Caso B, DHU-006.

**Notas clave:** explicación de "nivel mínimo" cerrada en el Inception (decisión #9). **NO** se usa procesamiento de lenguaje natural, **NO** se usa explicabilidad de IA, **NO** se usan modelos generativos: es sistema de plantillas estático con catálogo acotado (5-10 textos sugeridos en backlog detallado). CA-06.3 garantiza que el Operador nunca vea un panel vacío.

### Descripción

Un sistema que toma decisiones automáticas sin explicarlas erosiona la confianza del Operador y compromete la auditabilidad operativa. El Operador necesita entender, en lenguaje propio del dominio del tráfico (no técnico ni algorítmico), por qué el sistema eligió la estrategia que está vigente.

El alcance de la explicación es deliberadamente **mínimo**: se trata de un texto breve, predefinido para cada combinación típica de estrategia y condición disparadora, con sustitución de variables del estado que motivaron la selección (por ejemplo, longitud de cola en una dirección específica). **No** se trata de explicabilidad de inteligencia artificial (XAI), interpretación de redes neuronales, ni análisis causal: es texto plantillado con valores observados, suficiente para que el Operador entienda la razón sin requerir conocimiento técnico del motor.

Ejemplos del tipo de texto que se entrega (el catálogo exacto se cierra en el sprint):

- *"Estrategia A activada porque la cola en la dirección Norte (38 vehículos) supera el umbral configurado de 25 vehículos."*
- *"Estrategia B activada porque el flujo está balanceado entre las cuatro direcciones."*

### Criterios de aceptación

- **CA-06.1:** Dado que el Operador ha iniciado sesión y existe una estrategia vigente en la intersección, cuando ingresa al panel de explicación, entonces el sistema muestra un texto breve que describe la razón por la que la estrategia actual fue seleccionada, incluyendo los valores del estado del tráfico que dispararon esa selección.

- **CA-06.2:** Dado que el Operador está observando la explicación, cuando el sistema cambia la estrategia activa, entonces la explicación se actualiza automáticamente para reflejar la razón de la nueva selección, con una latencia máxima de 5 segundos desde que el cambio se produce.

- **CA-06.3:** Dado que el sistema selecciona una estrategia y no encuentra una plantilla de explicación que cubra esa combinación de estrategia y condición disparadora, cuando se muestra la explicación, entonces el sistema presenta un texto genérico que indica al menos la estrategia activa y los valores del estado relevantes, sin dejar el panel vacío.

- **CA-06.4:** Dado que el componente que produce la explicación deja de responder por cualquier causa, cuando el Operador está observando el panel, entonces el sistema mantiene en pantalla la última explicación conocida, la marca visualmente como "no confirmada" e indica el tiempo transcurrido desde la última confirmación (DHU-005 Caso B).

- **CA-06.5:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder al panel de explicación, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Naturaleza del componente de explicación:** sistema de plantillas de texto con sustitución de variables. NO se usa procesamiento de lenguaje natural, NO se usa explicabilidad de IA, NO se usan modelos generativos. La decisión está cerrada y documentada en el backlog detallado.
- **Catálogo de plantillas:** se define un conjunto acotado de plantillas (estimado en el backlog detallado: 5–10) que cubren los casos típicos de selección. Cubrir todos los casos posibles no es objetivo del MVP1.
- **Fallback (CA-06.3):** la existencia de un texto genérico de respaldo garantiza que el Operador nunca vea un panel vacío. Esto protege frente a casos no anticipados durante el diseño del catálogo.
- **Variables disponibles para sustitución:** las plantillas pueden referenciar los valores del estado que estaban presentes en el momento de la selección (cola por dirección, jam level, flujo, etc.). El conjunto exacto de variables disponibles se cierra en el sprint, pero el principio es que el texto debe ser **autocontenido**: el Operador no debería tener que mirar otros paneles para entender la explicación.
- **Componente subyacente:** la explicación es producida por el motor adaptativo (o un componente asociado a él), pero la HU es agnóstica al diseño técnico: lo único relevante es que existe un endpoint que devuelve la explicación actual.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** latencia máxima de actualización ≤ 5 s (CA-06.2). → RNF-PERF-01 (Actualización en tiempo real de la presentación operativa).
- **RNF de robustez:** comportamiento ante caída del componente de explicación (CA-06.4, DHU-005 Caso B). → RNF-REL-01 (Robustez ante interrupción de fuente).
- **RNF de usabilidad:** la explicación debe ser comprensible por un Operador sin formación técnica en el motor adaptativo. Probablemente se valida con prueba de usuario. → RNF-INT-05 (Comprensibilidad de explicaciones textuales sin formación técnica).
- **RNF de mantenibilidad:** el catálogo de plantillas debe ser extensible sin requerir cambios en el código del frontend ni del motor (debería poder agregarse una plantilla nueva como dato de configuración). → RNF-MNT-01 (Extensibilidad de catálogos de plantillas como datos de configuración).

---

## HU-07 — Notificación de cambios de estrategia del motor

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | recibir una notificación visual inmediata cada vez que el sistema cambia la estrategia de control activa |
| **Para** | enterarme del cambio en el momento exacto en que ocurre, aunque mi atención esté en otro panel del dashboard, y poder evaluar si la transición es coherente con lo que estoy observando del tráfico |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** F09 (Notificación visual de cambio de estrategia).

### Resumen ejecutivo

**Qué entrega:** notificación visual temporal y poco intrusiva al Operador cada vez que el motor cambia la estrategia de control activa. Aparece en cualquier vista del sistema, contiene la información mínima (hora, estrategia anterior, estrategia nueva, razón breve) y desaparece sola tras unos segundos.

**CAs críticos:** CA-07.1 (presentación de la notificación con sus 4 campos), CA-07.2 (auto-descarte tras tiempo configurado, default 10 s), CA-07.3 (agrupamiento ante cambios encadenados), CA-07.5 (robustez ante caída del canal de eventos — DHU-005 Caso B aplicado al canal).

**Dependencias:** consume el mismo motor adaptativo (TTH-10) que HU-05 y HU-06; la "razón breve" deriva de HU-06 en versión condensada. Distinta de HU-05 (vista pasiva del estado actual) y de HU-08 (consulta histórica): HU-07 cubre el "aviso activo" en el momento exacto del cambio. Aplica DHU-005 Caso B, DHU-006.

**Notas clave:** la notificación es efímera por diseño; una notificación perdida no es crítica porque el cambio queda registrado en HU-08 y la estrategia vigente se ve siempre en HU-05. El mecanismo de entrega (SSE o WebSocket) se cierra en sprint. Agrupamiento de CA-07.3 evita saturar al Operador con notificaciones encadenadas.

### Descripción

Existe una diferencia operativa importante entre **consultar** el estado de la estrategia (HU-05, vista pasiva) y **enterarse** de que la estrategia cambió (HU-07, evento activo). Si el Operador está observando otra parte del dashboard cuando el motor decide cambiar de estrategia, sin una notificación activa podría no enterarse hasta varios minutos después, al volver al panel correspondiente. Esto retrasa cualquier evaluación del Operador sobre la decisión y degrada su capacidad de supervisión.

La notificación es **temporal y poco intrusiva**: aparece cuando ocurre el cambio, contiene la información mínima para que el Operador entienda qué pasó, y desaparece sola tras unos segundos. No requiere acción del Operador para descartarla, pero permite acción si el Operador quiere profundizar (por ejemplo, abrir HU-05 o HU-06 para ver detalles).

La notificación incluye al menos: hora del cambio, estrategia anterior, estrategia nueva, y una razón breve (la misma que HU-06 produciría para la nueva estrategia, en versión condensada).

### Criterios de aceptación

- **CA-07.1:** Dado que el Operador ha iniciado sesión y tiene una vista del sistema abierta, cuando el motor cambia la estrategia de control activa, entonces el sistema muestra una notificación visual temporal que indica la hora del cambio, la estrategia anterior, la estrategia nueva y una razón breve del cambio.

- **CA-07.2:** Dado que se está mostrando una notificación de cambio de estrategia, cuando transcurre el tiempo configurado para auto-descarte (por defecto 10 segundos), entonces la notificación desaparece automáticamente sin requerir acción del Operador.

- **CA-07.3:** Dado que el motor cambia de estrategia múltiples veces en un intervalo corto, cuando ocurre el segundo o subsiguiente cambio dentro de un mismo intervalo configurado (por defecto 30 segundos), entonces el sistema agrupa las notificaciones en una sola entrada acumulada en lugar de generar una notificación por cada cambio, para evitar saturar visualmente al Operador.

- **CA-07.4:** Dado que el Operador está viendo cualquier vista del sistema (monitoreo, predicción, vista combinada, panel de estrategia, etc.), cuando se dispara una notificación, entonces se muestra de manera visible sin interferir con la lectura de los paneles principales (posición no central, sin bloquear contenido).

- **CA-07.5:** Dado que el sistema no puede emitir notificaciones por cualquier causa (por ejemplo, el canal de eventos en tiempo real está caído), cuando el Operador está observando el sistema, entonces existe un indicador visual persistente que comunica que la entrega de notificaciones está degradada, para que el Operador no asuma que la ausencia de notificaciones significa ausencia de cambios (DHU-005 Caso B aplicado a canal de eventos).

- **CA-07.6:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder al sistema, entonces el sistema lo redirige a la pantalla de login (las notificaciones no se entregan a usuarios no autenticados).

### Notas técnicas

- **Naturaleza temporal de la notificación:** la notificación es efímera por diseño. No genera entrada permanente en el log de decisiones (eso lo cubre HU-08). Una notificación perdida (porque el Operador no estaba mirando en ese instante) no es un problema crítico: el cambio queda registrado en el log y la estrategia vigente se ve siempre en HU-05.
- **Mecanismo de entrega:** Server-Sent Events (SSE) o WebSocket, sobre el mismo canal usado para actualizaciones de datos en tiempo real. Decisión técnica concreta a cerrar en el sprint.
- **Tiempo de auto-descarte (CA-07.2):** valor por defecto 10 segundos según el backlog detallado. Parametrizable.
- **Agrupamiento (CA-07.3):** la lógica exacta de agrupamiento (cuántos cambios juntan en una notificación, qué texto se muestra cuando son varios) se cierra en el sprint. El principio es: no abusar de la atención del Operador con notificaciones encadenadas.
- **Estilo visual:** toast o banner en esquina superior derecha es el patrón estándar de la industria; la decisión visual exacta se cierra en el prototipado del dashboard.
- **Razón breve (CA-07.1):** versión condensada de la explicación de HU-06. Si HU-06 entrega *"Estrategia B activada porque la cola en la dirección Norte (38 vehículos) supera el umbral de 25"*, la notificación puede mostrar *"Cola Norte alta (38 veh)"*.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** latencia máxima entre cambio de estrategia y notificación visible ≤ 5 s (implícito en CA-07.1; conviene explicitar en el documento RF/RNF). → RNF-PERF-01 (Actualización en tiempo real de la presentación operativa).
- **RNF de usabilidad:** la notificación no debe interferir con la lectura de los paneles principales (CA-07.4). Probablemente se valida con prueba de usuario. → RNF-INT-01 (Usabilidad operativa de las presentaciones del Operador).
- **RNF de robustez:** comportamiento ante caída del canal de notificaciones (CA-07.5). → RNF-REL-01 (Robustez ante interrupción de fuente).
- **RNF de configurabilidad:** tiempo de auto-descarte e intervalo de agrupamiento deben ser parametrizables sin redeploy del sistema. → RNF-MNT-02 (Parametrización sin redeploy de tiempos y umbrales operativos).

---

## HU-08 — Consulta del historial de decisiones del motor

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | consultar el historial cronológico de las decisiones que el sistema tomó en periodos pasados, con sus razones y parámetros |
| **Para** | reconstruir lo que el sistema hizo durante mi turno o turnos anteriores, investigar incidentes reportados, y disponer de evidencia auditable de la operación |

**Tipo:** HU de Persona (Operador). El Administrador también consulta este log para análisis técnico (referencia en backlog detallado F10).
**Feature(s) origen:** F10 (Log de decisiones del motor adaptativo). Ingloba F31 (Persistencia de decisiones del motor) como CA, según regla cerrada en el Bloque A.

### Resumen ejecutivo

**Qué entrega:** vista paginada con filtros para consultar cronológicamente las decisiones que el motor adaptativo tomó en periodos pasados, con su razón, parámetros y estrategia anterior. Sustenta auditoría y reconstrucción de turnos. Ingloba el sustrato técnico de persistencia de decisiones (F31) como CA.

**CAs críticos:** CA-08.1 (sustrato técnico F31 inglobada — persistencia de cada decisión con timestamp, estrategia, razón, parámetros, estrategia anterior), CA-08.2 (paginación obligatoria), CA-08.5 (resiliencia: la operación del motor nunca se detiene por fallos del registro de auditoría), CA-08.6 (redirección al login).

**Estructura de CAs:** persistencia F31 inglobada (CA-08.1) → presentación al Operador (CA-08.2 a CA-08.4) → resiliencia operativa (CA-08.5) → control de acceso (CA-08.6).

**Dependencias:** consume el motor adaptativo (TTH-10) que produce las decisiones. El Administrador hereda la misma vista por compartir endpoint en MVP1. HU-18 (MVP2, drill-down del Gerente) consume el mismo registro de decisiones como carril del motor adaptativo. F31 inglobada en CA-08.1 conforme a la regla del Bloque A (igual patrón que F30 en HU-16). Aplica DHU-005 (resiliencia de persistencia), DHU-006.

**Notas clave:** un sistema de control de tráfico real **no puede detener la operación del semáforo porque falle el log de auditoría** (CA-08.5): la operación es prioritaria, la auditoría es complementaria. Mecanismo de respaldo (cola/archivo de fallback) ante fallo temporal de escritura. Política de retención indefinida en MVP1; exportación a CSV fuera de alcance.

### Descripción

Las HUs anteriores (HU-05 a HU-07) cubren la operación **en tiempo presente**: qué estrategia está vigente ahora, por qué, y cómo enterarse de cambios cuando ocurren. Pero el Operador también necesita reconstruir lo que pasó **en el pasado**: durante un turno completo, en respuesta a un incidente, o para auditar una decisión cuestionada por terceros.

La consulta opera sobre un registro persistente de cada decisión tomada por el motor. Cada entrada del registro contiene al menos: marca de tiempo, estrategia aplicada, razón de la selección (la misma plantilla que HU-06 produce en tiempo real), parámetros activos aplicados, y la estrategia anterior (si la hubo).

El Operador accede al registro mediante una vista paginada con filtros básicos por rango de fechas y por estrategia, suficientes para reconstruir periodos de interés sin saturar la vista con todo el historial.

### Criterios de aceptación

- **CA-08.1:** Dado que el motor ha tomado decisiones durante un periodo, cuando el Operador consulta el historial de decisiones, entonces el sistema muestra una lista cronológica donde cada entrada contiene al menos la marca de tiempo de la decisión, la estrategia aplicada, la razón de la selección, los parámetros activos aplicados y la estrategia anterior. Esto presupone que el sistema persiste cada decisión del motor con esos campos en el momento en que la decisión se produce **(persistencia inglobada, F31)**.

- **CA-08.2:** Dado que el volumen de decisiones registradas puede ser alto, cuando el Operador abre el historial, entonces los resultados se presentan paginados con un tamaño de página configurable (valor por defecto sugerido: 50 entradas por página), y el Operador puede navegar entre páginas sin cargar todo el historial en memoria.

- **CA-08.3:** Dado que el Operador necesita acotar la búsqueda, cuando aplica un filtro por rango de fechas, por estrategia, o por ambos, entonces la lista mostrada se limita a las decisiones que cumplen el filtro, manteniendo el orden cronológico y la paginación.

- **CA-08.4:** Dado que el Operador no ha modificado los filtros, cuando abre el historial por primera vez en su sesión, entonces el sistema muestra por defecto las decisiones del periodo más reciente (sugerencia: últimas 24 horas) ordenadas de la más reciente a la más antigua.

- **CA-08.5:** Dado que el sistema persiste cada decisión, cuando ocurre un fallo temporal en la escritura al registro (por ejemplo, base de datos no disponible momentáneamente), entonces la decisión del motor se aplica de todas formas a la operación del semáforo y se registra en un mecanismo de respaldo (cola, archivo de fallback) para ser persistida cuando el registro vuelva a estar disponible. La operación del motor nunca se detiene por una falla del registro de auditoría.

- **CA-08.6:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder al historial de decisiones, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **F31 inglobada en CA-08.1:** según la regla cerrada en el Bloque A, la persistencia de decisiones no se redacta como HU separada. CA-08.1 formaliza el requisito: la consulta solo es posible si el sistema persiste cada decisión en el momento en que se produce. El esquema de la tabla debe contener, como mínimo: timestamp, estrategia, razón (texto), parámetros (estructura serializada), estrategia anterior.
- **Volumen esperado:** depende de la frecuencia de cambios del motor. En operación estable, decenas de decisiones por día. En periodos de alta variabilidad de tráfico, podría llegar a centenas. La paginación (CA-08.2) es obligatoria; no se acepta una vista sin paginación.
- **Política de retención:** durante MVP1 se asume retención indefinida del registro (volumen acotado, no es crítico). Una política de purga o archivado se evalúa en MVP2 o trabajo futuro, no entra acá.
- **Ordenamiento por defecto (CA-08.4):** "más reciente primero" es el patrón estándar para registros de auditoría, alineado con la práctica del Operador (lo más probable es que quiera ver lo último que pasó).
- **CA-08.5 (resiliencia de persistencia):** este criterio refleja un principio operativo crítico: **un sistema de control de tráfico real no puede detener la operación del semáforo porque falle el log de auditoría**. La operación es prioritaria; la auditoría es complementaria. El mecanismo de respaldo (cola en memoria, archivo local, etc.) se elige en el sprint, pero el principio queda anclado en el CA.
- **Exportación:** el backlog detallado menciona "considerar exportación a CSV en el futuro, no incluir en MVP1". La HU no incluye CA de exportación. Si más adelante se decide incluirla, entra como extensión.
- **Acceso del Administrador:** el backlog detallado indica que el Administrador también consulta este historial para análisis. En MVP1 ambos roles ven la misma vista; en iteraciones futuras se pueden agregar capacidades adicionales para el Administrador (filtros avanzados, exportación, etc.). Por ahora, la HU cubre el caso del Operador y el Administrador hereda la misma vista por compartir el endpoint.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** tiempo de respuesta para abrir una página del historial ≤ 2 s con volúmenes esperados de MVP1.
- **RNF de persistencia / auditoría:** cada decisión del motor debe persistirse de forma durable en el momento en que se produce. La pérdida de decisiones por fallo de escritura es inaceptable (CA-08.5 cubre esto con mecanismo de respaldo).
- **RNF de robustez:** la operación del motor no se detiene por fallos del registro de auditoría (CA-08.5).
- **RNF de retención:** política de retención del registro, a cerrar en MVP2 o trabajo futuro.
- **RNF de trazabilidad:** cada entrada del registro debe ser unívocamente identificable y no modificable después de ser escrita (inmutabilidad del log de auditoría).

---

## HU-09 — Registro de notas e incidencias del turno

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | registrar notas o incidencias durante mi turno, asociadas a un momento específico, y consultarlas posteriormente |
| **Para** | dejar constancia escrita de eventos relevantes que observé durante la operación (accidentes, comportamientos inusuales del tráfico, decisiones del motor que llamaron mi atención) y poder revisarlas yo mismo o transmitirlas al siguiente turno |

**Tipo:** HU de Persona (Operador).
**Clasificación MVP:** **MVP2 — candidata a construcción condicional a holgura del cronograma tras cerrar MVP1** (semántica refinada por DHU-012; ver Notas técnicas). Esta HU se redactó al cierre del Bloque B bajo la antigua semántica "fuera del sprint" cerrada en el Lean Inception, y fue suavizada por DHU-012; conserva su ubicación física en este documento por decisión de DHU-017 subsección B.
**Feature(s) origen:** F11 (Módulo de notas/incidencias del Operador).

### Resumen ejecutivo

**Qué entrega:** módulo simple de registro de notas e incidencias del turno para el Operador. Texto libre asociado a una marca de tiempo, persistente, con identidad del autor. Listado paginado con filtros básicos. Todos los Operadores ven todas las notas, sosteniendo el caso de uso de transmisión entre turnos.

**CAs críticos:** CA-09.1 (creación de nota con persistencia), CA-09.2 (listado paginado cronológico inverso), CA-09.4 (ventana de edición acotada por valor de auditoría), CA-09.5 (resiliencia: el sistema informa al Operador ante fallo de escritura sin perder contenido escrito), CA-09.6 (redirección al login).

**Estructura de CAs:** creación y consulta (CA-09.1 a CA-09.3) → política de edición (CA-09.4) → resiliencia y control de acceso (CA-09.5, CA-09.6).

**Dependencias:** HU autocontenida (no depende de otros componentes del sistema operativo). HU-21 (MVP2, escalamiento) hereda el patrón de alcance mínimo viable de esta HU. Aplica DHU-005 (resiliencia de persistencia), DHU-006, DHU-012 (semántica MVP2 refinada).

**Notas clave:** alcance MVP2 acotado: texto libre, sin categorías predefinidas, sin adjuntos, sin asociación automática a decisiones del motor. Visibilidad cruzada entre Operadores (a diferencia de HU-21, donde cada Operador ve solo sus propios escalamientos). Inmutabilidad parcial: nota editable solo dentro de ventana corta (24 h sugerido) para preservar valor de auditoría.

### Descripción

Durante un turno de operación, el Operador observa eventos que no están capturados automáticamente por el sistema: accidentes en accesos cercanos a la intersección que afectan el flujo, decisiones del motor que parecen incoherentes con la situación observada, comportamientos inusuales (manifestaciones, eventos masivos, obras), o cualquier otra observación relevante. Sin un mecanismo de registro, estas observaciones se pierden o quedan en notas personales del Operador en papel.

La HU provee un módulo simple de registro de notas asociadas a un momento. Cada nota contiene un texto libre y se persiste con la marca de tiempo de creación y la identidad del Operador que la registró. El Operador puede consultar el listado de notas (propias y de otros turnos) con filtros básicos.

**Razón de la clasificación MVP2:** durante el Lean Inception se priorizó el núcleo del sistema (monitoreo, predicción, motor adaptativo) sobre el soporte operativo al Operador. Esta HU no realiza directamente ninguno de los 4 Objetivos del Producto; es soporte al trabajo del Operador. Queda documentada como HU completa con criterios de aceptación y entra al sprint si el cronograma permite holgura tras cerrar las HUs MVP1, conforme a la semántica refinada por DHU-012.

### Criterios de aceptación

- **CA-09.1:** Dado que el Operador ha iniciado sesión, cuando crea una nota mediante un formulario que solicita un texto libre, entonces el sistema persiste la nota con su contenido, la marca de tiempo de creación y la identidad del Operador autor.

- **CA-09.2:** Dado que existen notas registradas, cuando el Operador accede a la vista de notas, entonces el sistema muestra el listado en orden cronológico (más reciente primero) con paginación, mostrando para cada nota: marca de tiempo, autor, y el texto (truncado si excede un largo razonable, con opción de expandir).

- **CA-09.3:** Dado que el Operador quiere acotar la búsqueda, cuando aplica un filtro por rango de fechas, por autor, o por ambos, entonces el listado se limita a las notas que cumplen el filtro, manteniendo el orden cronológico y la paginación.

- **CA-09.4:** Dado que el Operador creó una nota con error tipográfico o información incorrecta, cuando edita la nota dentro de un periodo configurado posterior a su creación (sugerencia: 24 horas), entonces el sistema permite la edición y registra la marca de tiempo de modificación además de la original. Pasado ese periodo, la nota queda inmutable para preservar valor de auditoría.

- **CA-09.5:** Dado que el sistema persiste cada nota, cuando ocurre un fallo temporal en la escritura, entonces el sistema informa al Operador que la nota no se guardó y permite reintentar sin perder el contenido escrito.

- **CA-09.6:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder al módulo de notas, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Política de construcción MVP2 (refinada por DHU-012):** Esta HU se documenta como Historia de Usuario completa y se considera candidata a entrar al sprint condicional a la holgura del cronograma tras cerrar las HUs MVP1. No es entregable comprometido del MVP1, pero tampoco está descartada a priori. Si el cronograma permite, se asignan sprints y puntos de historia normalmente.
- **Alcance mínimo:** texto libre. NO se incluyen en MVP2 categorías predefinidas, adjuntos de archivos, asociación explícita a una decisión específica del motor, ni vinculación geográfica a un acceso. Estas extensiones son trabajo futuro adicional.
- **Política de edición (CA-09.4):** la ventana de edición es un compromiso entre flexibilidad (el Operador puede corregir errores recientes) y valor de auditoría (las notas no deben poder ser modificadas a discreción mucho después de su creación). El valor exacto (24 h sugerido) se cierra al implementar.
- **Visibilidad entre Operadores:** todos los Operadores ven todas las notas (no hay notas privadas). Esto sostiene el caso de uso de transmisión entre turnos. Si en el futuro se considera privacidad por Operador, entra como extensión.
- **Sin asociación automática a decisiones del motor:** aunque el backlog detallado lo mencionaba como pregunta abierta, el alcance MVP2 deja la nota como texto libre independiente. La asociación explícita con entradas del log de HU-08 queda como trabajo futuro si se prioriza.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de persistencia:** notas durables, no pueden perderse por fallo temporal del sistema (CA-09.5 cubre el caso del usuario; el RNF cubre la durabilidad de la BD).
- **RNF de auditoría:** las notas no pueden eliminarse después de creadas (no hay CA explícito de "no borrado" porque no se contempla borrado en MVP2; conviene explicitarlo cuando se redacte el documento RF/RNF).
- **RNF de rendimiento:** tiempo de respuesta para abrir el listado paginado ≤ 2 s con volúmenes esperados.
- **RNF de inmutabilidad parcial:** las notas son editables solo dentro de una ventana temporal (CA-09.4); pasado ese periodo, inmutables.

---

## HU-22 — Mapa de congestión de la red

> Ampliación del Product Backlog del proyecto CerebroVial, posterior al cierre del componente funcional (HU-01 a HU-21 + TTH-01 a TTH-11, cerrado 2026-05-16, DHU-017).
>
> **Estado:** Redactada. Pendiente de implementación.
> **Fecha de redacción:** 2026-06-02
> **Bloque:** B — Operador, núcleo de monitoreo (ampliación).
> **Fundamento de la ampliación:** ver DHU-028 en `DECISIONS_HU.md` (registro de por qué el backlog declarado completo recibe una HU adicional, y de la transición de la unidad de análisis de "acceso de intersección" a "tramo de vía / arista" para la vista de red).

### Contexto

Esta HU se redacta en el formato del documento de referencia académica (`Desarrollo_Agil.pdf`, Tablas 9 y 13): "Como X, quiero Y, para Z" con criterios de aceptación Given-When-Then, y sigue las reglas metodológicas vigentes:

- **DHU-003** (sujetos válidos): el sujeto es el Operador de Tráfico Municipal, una de las tres Personas del producto.
- **DHU-006** (HUs agnósticas a la implementación): la HU describe el qué observable, no el cómo. No nombra tecnologías ni fuentes concretas de datos. La única excepción de vocabulario técnico permitida es la escala ordinal de nivel de congestión 0-5 (ya autónoma del sistema, estándar de la industria).
- **DHU-007** (RNF declarados): los requisitos no funcionales se declaran al final como "Candidatos a RNF", referenciados por ID cuando el documento RF/RNF los consolide.
- **DHU-018** (resumen ejecutivo): la HU abre con un resumen ejecutivo para uniformidad de lectura.
- **DHU-028** (esta ampliación): registra la incorporación de HU-22 y la unidad de análisis "tramo de vía" para la vista panorámica de red.

**Relación con HU-03 y HU-04 (vista de detalle de intersección):** HU-03 (predicción de congestión por acceso) y HU-04 (vista combinada estado+predicción por acceso) describen la **vista de detalle** de una intersección individual — el nivel de zoom donde el Operador examina una intersección concreta y sus accesos. HU-22 describe la **vista panorámica de red** — el nivel de zoom donde el Operador observa la congestión de toda la red sobre un mapa geográfico. Son dos niveles de zoom complementarios del mismo sistema de monitoreo, no sustitutos. HU-22 no reemplaza a HU-03/HU-04; se redacta primero por su valor de visión general.

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | ver sobre un mapa de la red vial un indicador de calor que muestre el nivel de congestión actual de cada tramo de vía, actualizándose en vivo |
| **Para** | identificar de un vistazo dónde se está formando congestión en la red y priorizar dónde intervenir, sin tener que revisar intersección por intersección |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** ampliación (vista panorámica de red; ver DHU-028). Relacionada con la familia funcional de monitoreo operativo en tiempo real.

### Resumen ejecutivo

**Qué entrega:** una vista de mapa geográfico de la red vial donde cada tramo de vía se colorea según su nivel de congestión actual (escala 0-5), actualizándose automáticamente en vivo. Permite al Operador una lectura panorámica e inmediata de dónde está la congestión en toda la red.

**CAs críticos:** CA-22.1 (mapa con tramos coloreados por nivel 0-5), CA-22.2 (actualización en vivo ≤ 5 s sin recarga), CA-22.4 (robustez ante interrupción de la fuente de congestión — DHU-005 Caso A), CA-22.6 (redirección al login).

**Dependencias:** consume la fuente de datos de congestión por tramo de vía de forma agnóstica (la HU no nombra la fuente). Habilitada por TTH-12 (infraestructura de datos de congestión por tramo de vía: geometría de la red + feed de estado por tramo persistido). Sin TTH-12 cerrada, no hay datos que pintar. Aplica DHU-005 Caso A (fuente externa de medición → "desactualizado" con antigüedad), DHU-006.

**Notas clave:** la escala 0-5 es el constructo de nivel de congestión adoptado en D-009 — permite intercambiabilidad de la fuente de datos sin alterar la vista. La HU es agnóstica a qué llena esa fuente. El alcance de esta HU cubre el **estado actual** de congestión; la proyección futura (predicción) sobre el mismo mapa es una extensión posterior fuera del alcance de esta versión (ver "Alcance y extensiones futuras").

### Descripción

El Operador necesita una lectura panorámica del estado de la red, no solo de intersecciones individuales. HU-02 le da el estado de una intersección y HU-03/HU-04 le dan estado y predicción por acceso de una intersección concreta; pero ninguna de esas vistas le permite ver, de un vistazo, dónde está la congestión a lo largo de toda la red vial.

Esta HU entrega una vista de mapa geográfico donde la red vial se representa como un conjunto de tramos de vía, y cada tramo se colorea según su nivel de congestión actual en una escala ordinal de 0 (flujo libre) a 5 (vía cerrada), siguiendo el estándar documentado en D-009. El mapa se actualiza automáticamente conforme llega nueva información de congestión, de modo que el Operador ve la evolución del estado de la red en vivo.

La unidad de análisis de esta vista es el **tramo de vía** (cada segmento dirigido de calle de la red), no el acceso de una intersección individual. Esto permite representar la congestión de manera continua sobre la geografía de la red, que es lo que un mapa de calor requiere.

La fuente de los datos de congestión es externa a esta vista y la HU es agnóstica a ella: la vista consume un indicador de nivel de congestión por tramo y lo representa, sin depender de cómo se produjo ese indicador. Esto permite intercambiar la fuente sin alterar la vista.

### Criterios de aceptación

- **CA-22.1:** Dado que el Operador ha iniciado sesión, cuando ingresa a la vista de mapa de congestión de la red, entonces el sistema muestra un mapa geográfico de la red vial donde cada tramo de vía se representa con un indicador visual de calor correspondiente a su nivel de congestión actual en escala 0-5, de manera que el Operador pueda leer el estado de toda la red de un vistazo.

- **CA-22.2:** Dado que el Operador tiene la vista de mapa abierta, cuando llega nueva información de congestión, entonces los tramos afectados actualizan su indicador de calor automáticamente sin necesidad de recargar la página, con una latencia máxima de 5 segundos desde que la información de congestión se genera.

- **CA-22.3:** Dado que el Operador observa el mapa, cuando para algún tramo el nivel de congestión actual supera el umbral configurado (por defecto, nivel ≥ 3), entonces ese tramo se distingue visualmente con suficiente claridad para que el Operador identifique los puntos críticos de la red sin ambigüedad, con una diferenciación que no dependa exclusivamente del color.

- **CA-22.4:** Dado que el sistema deja temporalmente de recibir información de congestión, cuando el Operador observa el mapa, entonces el sistema indica visualmente que la información de congestión está desactualizada e informa el tiempo transcurrido desde la última actualización conocida, manteniendo visibles los últimos niveles conocidos marcados como "desactualizados" (DHU-005 Caso A).

- **CA-22.5:** Dado que el Operador observa el mapa de la red, cuando consulta un tramo de vía concreto, entonces el sistema le permite identificar de qué tramo se trata y su nivel de congestión actual (por ejemplo, al señalarlo o seleccionarlo), de manera que la lectura panorámica pueda complementarse con el dato puntual de un tramo de interés.

- **CA-22.6:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder a la vista de mapa de congestión de la red, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Decisión de diseño visual:** la codificación visual concreta del calor (paleta de colores, intensidad, grosor de los tramos, leyenda) se cierra mediante prototipado antes de codear; no es parte de la HU. La HU es agnóstica al diseño concreto, salvo el requisito de CA-22.3 de que la diferenciación de niveles no dependa exclusivamente del color (coherente con accesibilidad, RNF-INT-02).

- **Composición sobre la infraestructura de datos (TTH-12):** esta vista no produce datos; consume el feed de congestión por tramo y la geometría de la red que TTH-12 provee. La integración es a nivel de presentación.

- **Unidad de análisis:** el "tramo de vía" de esta HU corresponde al segmento dirigido de calle de la red (la arista del grafo vial). La elección de esta unidad, en lugar del "acceso de intersección" usado por HU-03/HU-04, está registrada en DHU-028 y es lo que permite la representación geográfica continua del mapa de calor.

### Alcance y extensiones futuras

- **En alcance de esta versión:** representación del **estado de congestión actual** por tramo de vía sobre el mapa, en vivo.

- **Fuera de alcance (extensión futura):** la proyección de la congestión **predicha** a un horizonte futuro sobre el mismo mapa (análoga a lo que HU-03 hace por acceso, pero por tramo y geográfica). Esta extensión, cuando se redacte, reutilizará la infraestructura de esta HU y la capa de predicción por tramo, y se documentará como ampliación con su propia decisión registrada.

- **Fuera de alcance (extensión futura):** la integración con la vista de detalle de intersección (HU-03/HU-04) mediante zoom — pasar de la vista panorámica de red al detalle de una intersección concreta. Es la convergencia natural de los dos niveles de zoom; se redacta cuando ambas vistas estén implementadas.

### Candidatos a RNF

> Conforme a DHU-007, los siguientes son candidatos a requisitos no funcionales derivados de esta HU. Se consolidan en el documento RF/RNF (`REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md`) con su ID `RNF-XXX-NN` cuando este se actualice; aquí se declaran como candidatos.

- **Candidato a RNF de performance (actualización en tiempo real):** la actualización del calor de los tramos en el mapa ocurre con latencia máxima de 5 segundos desde que la información de congestión se genera (CA-22.2). Consolidable con el RNF transversal de tiempo real ≤ 5 s (RNF-PERF de actualización de tiempo real, DHU-019 subsección C.1), que ya agrupa diez HUs con el mismo umbral. → candidato a `RNF-PERF` (actualización de tiempo real).

- **Candidato a RNF de robustez ante interrupción (DHU-005 Caso A):** ante interrupción de la fuente de congestión, la vista marca los datos como desactualizados con su antigüedad en lugar de mostrarlos como vigentes (CA-22.4). Consolidable con el RNF transversal de robustez ante interrupción (DHU-019 subsección C.2). → candidato a `RNF-REL` (robustez ante interrupción, Caso A).

- **Candidato a RNF de usabilidad (legibilidad panorámica):** el Operador identifica los puntos críticos de la red de un vistazo, con diferenciación de niveles que no depende exclusivamente del color (CA-22.3). Probablemente se valida con prueba de usuario midiendo el tiempo de identificación de los tramos congestionados; validación cualitativa por revisión del diseño cuando una prueba formal no sea factible en el alcance académico. → candidato a `RNF-INT` (usabilidad / legibilidad operativa).

- **Candidato a RNF de accesibilidad (WCAG 2.1 AA):** la distinción de niveles de congestión no depende exclusivamente del color; se acompaña de al menos un código visual redundante (intensidad, patrón, etiqueta, grosor de línea), conforme a RNF-INT-02. → candidato a `RNF-INT-02` (accesibilidad).

- **Candidato a RNF de performance (carga del mapa):** la vista de mapa de la red completa se carga en un tiempo razonable para una vista de monitoreo de apertura (el umbral concreto se cierra en el documento RF/RNF; sugerido coherente con las vistas de apertura del Operador). → candidato a `RNF-PERF` (tiempo de apertura de vista).

---

## HU-23 — Recorrido temporal de la congestión de la red

> Ampliación del Product Backlog del proyecto CerebroVial, posterior a la ampliación de la vista panorámica de red (HU-22 + TTH-12, DHU-028).
>
> **Estado:** Redactada. Pendiente de implementación.
> **Fecha de redacción:** 2026-06-02
> **Bloque:** B — Operador, núcleo de monitoreo (ampliación).
> **Fundamento de la ampliación:** ver DHU-029 en `DECISIONS_HU.md` (registro de por qué la vista panorámica recibe un eje temporal de recorrido como HU separada del Operador, distinta de las HU de análisis del Gerente y complementaria a la voz "actual/en vivo" de HU-22).

### Contexto

Esta HU se redacta en el formato del documento de referencia académica (`Desarrollo_Agil.pdf`, Tablas 9 y 13): "Como X, quiero Y, para Z" con criterios de aceptación Given-When-Then, y sigue las reglas metodológicas vigentes:

- **DHU-003** (sujetos válidos): el sujeto es el Operador de Tráfico Municipal, una de las tres Personas del producto.
- **DHU-006** (HUs agnósticas a la implementación): la HU describe el qué observable, no el cómo. No nombra tecnologías ni fuentes concretas de datos. La única excepción de vocabulario técnico permitida es la escala ordinal de nivel de congestión 0-5 (ya autónoma del sistema, estándar de la industria).
- **DHU-007** (RNF declarados): los requisitos no funcionales se declaran al final como "Candidatos a RNF".
- **DHU-018** (resumen ejecutivo): la HU abre con un resumen ejecutivo para uniformidad de lectura.
- **DHU-029** (esta ampliación): registra la incorporación de HU-23 y el alcance temporal histórico/actual del recorrido.

**Relación con HU-22 (vista panorámica en vivo):** HU-22 entrega la congestión **actual** de la red, actualizándose en vivo, para "intervenir ahora". HU-23 entrega el **recorrido temporal** de esa misma congestión a lo largo del día, para entender cómo se formó la situación actual **antes** de intervenir. Son complementarias: HU-22 es la foto del ahora; HU-23 es la película del día sobre el mismo mapa. HU-23 no reemplaza a HU-22 ni modifica su comportamiento en vivo.

**Relación con las HU del Gerente (HU-16/17/18):** las vistas del Gerente analizan periodos prolongados (semana/mes), comparan periodos y hacen drill-down de KPIs agregados de desempeño, sobre el dashboard. HU-23 es del Operador, su unidad es el tramo de vía sobre el mapa geográfico, y su horizonte es el día como contexto operativo inmediato, no el análisis de planificación. No hay solape de Persona, unidad ni medio.

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | recorrer sobre el mapa de la red vial la evolución del nivel de congestión de cada tramo de vía a lo largo del día, mediante un control temporal que avanza desde el inicio del día hasta el momento actual |
| **Para** | entender cómo se formó la congestión que observo en el presente —si viene creciendo o disipándose— y disponer de ese contexto reciente antes de decidir dónde y cómo intervenir |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** ampliación (recorrido temporal de la vista panorámica de red; ver DHU-029). Relacionada con la familia funcional de monitoreo operativo del Operador.

### Resumen ejecutivo

**Qué entrega:** sobre el mismo mapa geográfico de la red de HU-22, un control temporal que el Operador desplaza para recorrer la evolución de la congestión por tramo de vía a lo largo del día seleccionado. Al mover el control, el mapa repinta el nivel de congestión 0-5 de cada tramo correspondiente al instante elegido. Permite al Operador leer la trayectoria reciente de la congestión, no solo su estado actual.

**CAs críticos:** CA-23.1 (recorrido temporal que repinta el mapa por instante), CA-23.2 (control limitado al rango con datos, bloqueado más allá), CA-23.4 (rótulo honesto de la naturaleza del dato recorrido), CA-23.6 (coexistencia con el modo en vivo de HU-22), CA-23.8 (redirección al login).

**Dependencias:** consume el estado de congestión por tramo en un instante arbitrario y el rango temporal disponible, de forma agnóstica a la fuente. Habilitada por TTH-13 (consulta histórica de congestión por arista: estado por instante + rango disponible), que a su vez reutiliza la geometría y la persistencia de TTH-12. Reutiliza el mapa y la codificación visual de HU-22. Sin TTH-13 cerrada, no hay estado por instante que pintar.

**Notas clave:** la escala 0-5 es el constructo de D-009, idéntico a HU-22. El recorrido cubre el estado **histórico y actual** del día; la proyección a futuro (predicción) sobre el mismo eje es extensión posterior fuera de alcance (ver "Alcance y extensiones futuras"). La HU es agnóstica a qué llena la fuente; en la versión demostrable, el recorrido es sobre un día de replay simulado, lo que la vista rotula explícitamente.

### Descripción

El Operador, antes de intervenir sobre la congestión que observa en vivo (HU-22), necesita el contexto de cómo llegó la red a ese estado: una congestión que viene creciendo desde hace una hora exige una respuesta distinta de una que ya se está disipando. HU-22 le da la foto del presente, pero no la trayectoria.

Esta HU agrega al mapa de la red un control temporal —un deslizador sobre el eje del día— que el Operador desplaza para recorrer la evolución de la congestión. Cada posición del control corresponde a un instante del día; al moverlo, el mapa repinta el nivel de congestión 0-5 de cada tramo de vía correspondiente a ese instante, usando la misma codificación visual de calor de HU-22. El recorrido va desde el inicio del día hasta el momento con datos más reciente.

El control se ofrece sobre la misma vista de mapa de HU-22, de modo que el Operador alterna entre observar la red en vivo (HU-22) y recorrer su evolución del día (HU-23) sin cambiar de pantalla. La unidad de análisis es el **tramo de vía**, idéntica a HU-22.

La fuente de los datos de congestión es externa a esta vista y la HU es agnóstica a ella: la vista consume un nivel de congestión por tramo para un instante dado y lo representa, sin depender de cómo se produjo ni de qué día concreto se recorre.

### Criterios de aceptación

- **CA-23.1:** Dado que el Operador ha iniciado sesión y tiene la vista de mapa de la red abierta, cuando activa el recorrido temporal y desplaza el control a una posición del día, entonces el sistema repinta el indicador de calor de cada tramo de vía con el nivel de congestión 0-5 correspondiente al instante seleccionado, de manera que el Operador pueda leer el estado de toda la red en ese momento del día.

- **CA-23.2:** Dado que el Operador recorre un día cuyos datos solo cubren hasta cierto instante (por ejemplo, el día en curso con datos hasta la hora actual), cuando intenta desplazar el control más allá del último instante con datos disponibles, entonces el sistema impide el desplazamiento más allá de ese punto (el control queda limitado al rango con datos), de manera que el Operador nunca recorra un tramo del día sin información en lugar de mostrarse vacío o engañoso.

- **CA-23.3:** Dado que el Operador desplaza el control temporal, cuando el mapa repinta para el instante seleccionado, entonces el sistema indica de forma visible a qué momento del día corresponde la representación mostrada, de manera que el Operador sepa en todo momento qué instante está observando.

- **CA-23.4:** Dado que el Operador recorre la evolución de la congestión, cuando observa el mapa en modo de recorrido temporal, entonces el sistema indica explícitamente la naturaleza del dato que se está recorriendo (por ejemplo, que corresponde a un día de muestra o registro reproducido), de manera que el recorrido no se interprete como un registro operativo en tiempo real distinto del que ofrece la vista en vivo.

- **CA-23.5:** Dado que el Operador observa el mapa en un instante del recorrido, cuando consulta un tramo de vía concreto, entonces el sistema le permite identificar de qué tramo se trata y su nivel de congestión en el instante seleccionado (por ejemplo, al señalarlo o seleccionarlo), de manera que la lectura panorámica del recorrido pueda complementarse con el dato puntual de un tramo de interés.

- **CA-23.6:** Dado que la vista de mapa ofrece tanto la observación en vivo (HU-22) como el recorrido temporal (esta HU), cuando el Operador activa el recorrido temporal, entonces el sistema deja de actualizar el mapa con la llegada de nueva información en vivo mientras dura el recorrido, y cuando el Operador vuelve al modo en vivo, entonces el sistema retoma la actualización automática del estado actual, de manera que el instante que el Operador está examinando no se vea alterado por actualizaciones en vivo y ambos modos no entren en conflicto.

- **CA-23.7:** Dado que el Operador selecciona el día a recorrer, cuando el día seleccionado no dispone de información de congestión, entonces el sistema comunica explícitamente la ausencia de datos para ese día y deshabilita el control temporal, en lugar de presentar un mapa vacío sin explicación.

- **CA-23.8:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder a la vista de mapa de congestión de la red, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Composición sobre HU-22 y TTH-13:** esta vista no produce datos ni geometría nueva; reutiliza el mapa, la geometría de la red y la codificación visual de HU-22, y consume de TTH-13 el estado de congestión por tramo en un instante arbitrario y el rango temporal disponible por día. La integración es a nivel de presentación.

- **Control temporal como deslizador:** la elección de un deslizador sobre el eje del día (en lugar de la selección de una hora exacta) es una decisión de usabilidad: el recorrido continuo es más ágil que ingresar instantes puntuales. La granularidad efectiva del recorrido (cada cuántos minutos repinta) y el diseño visual concreto del control se cierran mediante prototipado antes de codear; no son parte de la HU.

- **Selección del día:** el día a recorrer lo determina el frontend; el rango temporal disponible dentro de ese día lo reporta el backend (TTH-13), de modo que el control se acote y bloquee según los datos reales y no según una suposición de cobertura.

- **Coherencia visual con HU-22:** el recorrido usa la misma paleta y codificación de calor 0-5 de HU-22 (CA-22.3, diferenciación no dependiente exclusivamente del color), para que el Operador lea ambos modos con el mismo lenguaje visual.

- **Unidad de análisis:** el "tramo de vía" de esta HU es el mismo de HU-22 (segmento dirigido de calle / arista del grafo vial).

### Alcance y extensiones futuras

- **En alcance de esta versión:** recorrido del **estado de congestión histórico y actual** por tramo de vía a lo largo del día (0 a 24 h), sobre el mapa, con el control limitado y bloqueado al rango con datos disponibles.

- **Fuera de alcance (extensión futura):** la **proyección de la congestión predicha** a futuro sobre el mismo eje temporal (por ejemplo, extender el control más allá del instante actual para mostrar los próximos 30 minutos de congestión proyectada por tramo). El control temporal de esta HU se diseña de modo que su rango pueda extenderse hacia el futuro cuando exista la capa de predicción por tramo en escala 0-5; esa capa no existe servida hoy y se aborda como extensión predictiva con su propia HU y su propia TTH de predicción por arista (ver DHU-029).

- **Fuera de alcance (extensión futura):** una vista de análisis histórico geográfico para el Gerente (recorrido o agregación sobre periodos prolongados sobre el mapa), que sería una necesidad de otra Persona con su propia HU.

### Candidatos a RNF

> Conforme a DHU-007, los siguientes son candidatos a requisitos no funcionales derivados de esta HU. Se consolidan en el documento RF/RNF (`REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md`) con su ID `RNF-XXX-NN` cuando este se actualice; aquí se declaran como candidatos.

- **Candidato a RNF de usabilidad (recorrido ágil):** el desplazamiento del control temporal repinta el mapa con una latencia suficientemente baja para que el recorrido se perciba como fluido y no como una sucesión de cargas (CA-23.1). → candidato a RNF-USA-NN.

- **Candidato a RNF de integridad de la representación (no recorrer vacío):** el control nunca permite recorrer instantes sin datos; se limita y bloquea al rango disponible reportado por la fuente (CA-23.2, CA-23.7). → candidato a RNF-FUN-NN.

- **Candidato a RNF de honestidad de la fuente:** la vista comunica la naturaleza del dato recorrido de modo que no se confunda un registro reproducido con un registro operativo en tiempo real (CA-23.4). → candidato a RNF-COM-NN.

- **Candidato a RNF de accesibilidad:** la diferenciación de niveles en el recorrido no depende exclusivamente del color, coherente con HU-22 (CA-22.3) y RNF-INT-02. → candidato a RNF-INT-02.

- **Candidato a RNF de seguridad:** el acceso a la vista y a los datos del recorrido está restringido por rol, coherente con HU-22 (CA-23.8) y el patrón de autorización del proyecto. → candidato a RNF-SEC-NN.

---

## Resumen del Bloque B

| HU | Título | Sujeto | Tipo | Feature(s) origen | Clasif. MVP |
|---|---|---|---|---|---|
| HU-02 | Monitoreo del estado actual de la intersección en tiempo real | Operador | Persona | F03 + F04 | MVP1 |
| HU-03 | Visualización de predicción de congestión a corto plazo | Operador | Persona | F05 | MVP1 |
| HU-04 | Vista combinada del estado actual y la predicción de tráfico | Operador | Persona | F06 | MVP1 |
| HU-05 | Visualización de la estrategia de control activa | Operador | Persona | F07 | MVP1 |
| HU-06 | Explicación de la razón de selección de estrategia | Operador | Persona | F08 | MVP1 |
| HU-07 | Notificación de cambios de estrategia del motor | Operador | Persona | F09 | MVP1 |
| HU-08 | Consulta del historial de decisiones del motor | Operador (+ Admin) | Persona | F10 (+ F31 inglobada) | MVP1 |
| HU-09 | Registro de notas e incidencias del turno | Operador | Persona | F11 | MVP2 |
| HU-22 | Mapa de congestión de la red | Operador | Persona | ampliación (DHU-028) | MVP1 — ampliación |
| HU-23 | Recorrido temporal de la congestión de la red | Operador | Persona | ampliación (DHU-029) | MVP1 — ampliación |

**Total Bloque B: 10 HUs** (7 MVP1 + 1 MVP2 + 2 ampliaciones: HU-22 habilitada por TTH-12, HU-23 habilitada por TTH-13).

F02 (Dashboard principal) queda cubierto por la composición visual de las HUs anteriores en una vista única, sin generar HU propia.

F31 (Persistencia de decisiones del motor) queda inglobada como CA-08.1 de HU-08, conforme a la regla del Bloque A.

---

## Decisiones que aplicaron a este bloque

Durante la redacción del Bloque B se cerraron las siguientes decisiones formales:

- **D-009** (técnica del producto, ver `DECISIONS.md`): adopción del constructo Waze "jam level" (0-5) como variable de estado predicha. Permite intercambiabilidad de fuente de datos sin reentrenamiento del modelo.
- **DHU-005 refinada** (metodológica, ver `DECISIONS_HU.md`): principio de robustez ante interrupción de fuente, con Casos A (fuente externa de medición → "desactualizado") y Caso B (componente interno de decisión → "no confirmada").
- **DHU-006** (metodológica, ver `DECISIONS_HU.md`): HUs agnósticas a la implementación. No se mencionan tecnologías específicas (SUMO, Waze, GRU, Webster, MaxPressure, MTC) en las HUs.
- **DHU-007** (metodológica, ver `DECISIONS_HU.md`): RNF declarados como tales en sección específica al final de cada HU. Anticipa la creación de un documento RF/RNF formal en una sesión futura.

---

## Próximos pasos

Esta sesión cerró el Bloque B. A la fecha actual, los Bloques C, D, E y F del MVP1 están cerrados y el MVP2 también está cerrado (DHU-017, 2026-05-16); **con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 23 HU operativas (HU-01 a HU-23) y 13 TTH (TTH-01 a TTH-13), tras las ampliaciones DHU-028 (HU-22 + TTH-12) y DHU-029 (HU-23 + TTH-13).** Los siguientes pasos del proyecto, en sesiones futuras:

1. **Bloque C — Operador, operación degradada** (F22, F23, F24, F25, F26, F27 → 3 HUs operativas + 2 TTH; ya cerrado con HU-10, HU-11, HU-12, TTH-04, TTH-05). Ver `HU_BLOQUE_C.md`.
2. **Bloque D — Administrador, soporte técnico** (F17, F18, F20 → 3 HUs operativas; ya cerrado con HU-13, HU-14, HU-15). Ver `HU_BLOQUE_D.md`. F21 fue reclasificado a Trabajos Futuros por DHU-012.
3. **Bloque E — Componentes centrales del sistema** (F32, F33, F34, F35 → 0 HUs operativas + 5 TTH: TTH-07 a TTH-11; ya cerrado el 2026-05-15 por DHU-015). Ver `HU_BLOQUE_E.md`.
4. **Bloque F — Gerente, reportería mínima** (F12 + F13 fusionadas en HU-16 con F30 inglobada, F14 en HU-17 → 2 HUs operativas + 0 TTH nuevas; ya cerrado el 2026-05-16 por DHU-016). Ver `HU_BLOQUE_F.md`.
5. **MVP2 — HUs documentadas con construcción condicional a holgura del cronograma tras cerrar MVP1** (F11→HU-09 anticipada en este bloque; F15→HU-18; F16→HU-19; F19→HU-20; F28→HU-21; 5 HUs operativas + 0 TTH nuevas; cerrado el 2026-05-16 por DHU-017). Semántica refinada por DHU-012. Ver `HU_MVP2.md`.

Tras cerrar el MVP2 (ya hecho), los próximos pasos del proyecto, fuera del alcance del Product Backlog funcional, son:

1. **Documento de Requisitos Funcionales y No Funcionales (RF/RNF)** consolidando los "Candidatos a RNF" de todas las HUs (HU-01 a HU-21) en un documento único aprobado, conforme a DHU-007 pendiente. Sesión dedicada futura.
2. **Ceremonias de estimación (Planning Poker) y priorización (MoSCoW)** sobre el backlog completo.
3. **Implementación SCRUM del MVP1** (16 HUs operativas + 11 TTH del MVP1). El MVP2 (5 HUs adicionales) entra al sprint si hay holgura de cronograma, conforme a la semántica refinada por DHU-012.
4. **SDD (Software Design Document)**, siguiente entregable académico mayor del proyecto.

---

## Documentos relacionados

- `HU_BLOQUE_A.md` — Bloque A del Product Backlog (acceso al sistema, 1 HU).
- `HU_BLOQUE_C.md` — Bloque C del Product Backlog (3 HUs operativas: HU-10, HU-11, HU-12).
- `HU_BLOQUE_D.md` — Bloque D del Product Backlog (3 HUs operativas: HU-13, HU-14, HU-15).
- `HU_BLOQUE_E.md` — Bloque E del Product Backlog (0 HUs operativas; mapeo a TTH-07 a TTH-11 y decisiones tomadas durante la redacción).
- `HU_BLOQUE_F.md` — Bloque F del Product Backlog (2 HUs operativas: HU-16, HU-17; F30 inglobada como CAs).
- `HU_MVP2.md` — MVP2 del Product Backlog (HU-18, HU-19, HU-20, HU-21; HU-09 reside en este documento).
- `DECISIONS_HU.md` — Decisiones metodológicas sobre HUs (DHU-001 a DHU-022).
- `DECISIONS.md` — Decisiones técnicas del producto (D-001 a D-009).
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-01 (autenticación), TTH-02 (Docker), TTH-03 (CI), TTH-04 (fallback en cascada), TTH-05 (tiempos preconfigurados degradado nivel 3), TTH-06 (capa DTOs, Trabajos Futuros), TTH-07 (SUMO), TTH-08 (visión), TTH-09 (GRU), TTH-10 (motor adaptativo), TTH-11 (spike de hiperparámetros temporales).
- `LEAN_INCEPTION_CEREBROVIAL.md` — Inception completo aplicado al proyecto.
- `FEATURE_BACKLOG_DETALLADO.md` — Detalle completo de las 41 features (29 MVP1 + 5 MVP2 + 7 Trabajos Futuros).
- `EVOLUCION_TESIS.md` — Narrativa de las 4 fases del proyecto.
