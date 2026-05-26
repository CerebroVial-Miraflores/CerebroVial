# Historias de Usuario — Bloque D (Cerradas)

> Cuarta entrega del Product Backlog del proyecto CerebroVial.
>
> **Estado:** Bloque D cerrado y aprobado. Bloques A, B, C, E y F del MVP1 cerrados, y MVP2 también cerrado el 2026-05-16 (DHU-017). **Con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 21 HUs operativas (HU-01 a HU-21) + 11 TTH (TTH-01 a TTH-11).** Pendiente: documento RF/RNF (DHU-007), Planning Poker, MoSCoW, implementación SCRUM del MVP1. HU-14 (métricas del modelo principal) es ampliada por HU-20 del MVP2 mediante extensión inglobada del registro de predicciones (CA-14.1 extendida a persistir también predicciones del modelo de respaldo, conforme a DHU-017 subsección D); HU-13, HU-14 y HU-15 reciben el badge de incidentes pendientes de HU-21 en la navegación del Administrador.
>
> **Fecha de cierre:** 2026-05-14
> **Fecha de actualización v2:** 2026-05-17 (DHU-018 aplicada retroactivamente: Resumen ejecutivo en HU-13, HU-14, HU-15)

---

## Contexto

Este documento contiene las Historias de Usuario del **Bloque D — Administrador, soporte técnico** del Sequencer del Lean Inception (ver `LEAN_INCEPTION_CEREBROVIAL.md`, sección 9, y `FEATURE_BACKLOG_DETALLADO.md`, Bloque D).

Las HUs se redactan en el formato del documento de referencia académica (`Desarrollo_Agil.pdf`, Tablas 9 y 13): "Como X, quiero Y, para Z" con criterios de aceptación Given-When-Then.

Las HUs del Bloque D siguen las reglas metodológicas establecidas y refinadas durante la redacción de los bloques previos:

- **DHU-001 a DHU-004** (del Bloque A): sujetos válidos, exclusión del Equipo de Desarrollo, TTH como categoría separada.
- **DHU-005** (refinada en el Bloque B): principio de robustez ante interrupción de fuente, con Casos A y B.
- **DHU-006:** HUs agnósticas a la implementación.
- **DHU-007:** RNF declarados como tales en sección específica al final de cada HU.
- **DHU-008** a **DHU-011** (cerradas en el Bloque C): modelo de degradación, marca pasiva vs alerta activa, clasificación TTH del Bloque C, eliminación de HU-13 original del Bloque C.
- **DHU-012** (transversal): auditoría de coherencia documental.
- **DHU-013** (cerrada antes de la redacción): clasificación HU/TTH del Bloque D. Las tres features (F17, F18, F20) son HU operativas del Administrador, sin TTH nuevas; el sustrato técnico de F18 y F20 se ingloba como CAs.
- **DHU-014** (cerrada durante la redacción de este bloque): decisiones de redacción del Bloque D (numeración compactada, sin HU dedicada de dashboard del Administrador, selección concreta de parámetros, métricas exactas, concurrencia, ventana temporal de HU-14 inglobada en HU-15, creación de TTH-06).
- **DHU-018** (aplicada retroactivamente el 2026-05-17): patrón "Resumen ejecutivo" al inicio de cada HU para uniformidad de lectura. Aditiva, no modifica contenido sustantivo.

Ver `DECISIONS_HU.md` para fundamentación completa.

---

## Mapeo de features del Bloque D

Las 3 features del Bloque D (F17, F18, F20) se mapearon a 3 HUs operativas, sin TTH nuevas, conforme a DHU-013 y DHU-014:

| HU | Título | Feature origen |
|---|---|---|
| HU-13 | Vista técnica de salud de los componentes del sistema | F17 |
| HU-14 | Vista de métricas de desempeño del modelo predictivo | F18 (sustrato inglobado como CAs) |
| HU-15 | Configuración de parámetros operativos del sistema | F20 (sustrato inglobado como CAs) |

**Total Bloque D:** 3 HUs operativas + 0 TTH nuevas.

**Sobre F21 (Reentrenamiento del modelo predictivo):** F21 fue reclasificada a Trabajos Futuros por DHU-012 subsección B (no entra en MVP1 del Bloque D). Su ficha vive en `FEATURE_BACKLOG_DETALLADO.md` bajo Trabajos Futuros.

**Sobre la numeración (compactación):** la HU-13 del Bloque C fue eliminada por DHU-011 antes de ser redactada formalmente; el número HU-13 no quedó "ocupado" en ningún documento vigente. Por DHU-014, el Bloque D reutiliza HU-13 = F17 (compactación de la numeración del Product Backlog). La traza histórica de la HU-13 eliminada vive en DHU-011 (`DECISIONS_HU.md`), no en la numeración del backlog.

**Sobre la composición visual del dashboard del Administrador:** por DHU-014, no hay HU dedicada de "dashboard del Administrador" análoga a F02 del Bloque B (dashboard del Operador). La navegación del Administrador da acceso a las tres vistas (HU-13, HU-14, HU-15) y eso es suficiente. La diferencia respecto al Operador es que el Administrador no trabaja con un único objeto en tiempo real (la intersección) sino con objetos distintos en momentos distintos (consulta de componentes, análisis de métricas, ajuste de configuración); integrar visualmente las tres vistas no aporta valor cognitivo y agregaría una HU sin propósito claro.

---

## HU-13 — Vista técnica de salud de los componentes del sistema

| Campo | Contenido |
|---|---|
| **Como** | Administrador del Sistema |
| **Quiero** | consultar en una vista dedicada el estado técnico detallado de cada componente del sistema, incluyendo métricas de respuesta, indicadores de fallos recientes y marcas de tiempo precisas |
| **Para** | diagnosticar rápidamente la causa raíz de un comportamiento anómalo, priorizar qué componentes requieren intervención y mantener la salud técnica del sistema sin depender de la visión simplificada que tienen los Operadores |

**Tipo:** HU de Persona (Administrador del Sistema).
**Feature(s) origen:** F17 (Panel de salud de componentes del sistema, vista del Administrador).

### Resumen ejecutivo

**Qué entrega:** vista técnica detallada de salud de componentes para el Administrador. Por cada componente: nombre legible, identificador interno, estado cualitativo, latencia de respuesta, indicador de fallos recientes, timestamp del último cambio y de la última evaluación exitosa. Resaltado de no-OK. Solo de consulta.

**CAs críticos:** CA-13.1 (presentación con los 7 campos por componente), CA-13.3 (resaltado visual de no-OK equivalente al de HU-11 con riqueza técnica preservada), CA-13.4 (DHU-005 Caso B aplicado al monitor), CA-13.6 (HTTP 403 para roles no-Administrador), CA-13.7 (redirección al login).

**Estructura de CAs:** presentación técnica (CA-13.1 a CA-13.3) → robustez DHU-005 Caso B (CA-13.4) → manejo de datos faltantes (CA-13.5) → control de acceso (CA-13.6, CA-13.7).

**Dependencias:** consume el mismo CT-04.5 de TTH-04 que HU-11 del Operador, con presentación adaptada (DHU-013). El contrato de CT-04.5 se amplió al cerrar HU-13 con los campos técnicos adicionales (latencia, fallos recientes, timestamp de última evaluación exitosa). HU-21 (MVP2) muestra badge de incidentes pendientes en la navegación que incluye esta vista.

**Notas clave:** patrón "un endpoint, dos vistas" con filtrado en presentación (DHU-013). Los campos técnicos no son sensibles (métricas del propio sistema, no datos personales); la política de capa de DTOs específica se documenta como TTH-06 clasificada como Trabajos Futuros. No incluye acciones correctivas (reinicio, recarga, etc.); serían HUs separadas en futuro. Ventana del indicador de fallos recientes en valor único configurable (sugerencia: 5 minutos) sin exposición al Administrador en MVP1.

### Descripción

El Bloque C entregó al Operador una vista cualitativa del estado de los componentes (HU-11): nombres legibles, estado OK / Degradado / Fuera de servicio, marcas de tiempo de último cambio, y resalte visual de los componentes en estado no-OK. Esa vista responde la pregunta del Operador *"¿qué componente está afectando la operación ahora?"*, suficiente para tomar decisiones operativas y escalar.

El Administrador del Sistema tiene una responsabilidad distinta: es la persona que mantiene la salud técnica del sistema. Cuando un componente entra en estado Degradado o Fuera de servicio, el Administrador no necesita saber *que* algo está mal — eso ya lo sabe por escalamiento, por revisión periódica o por la propia vista cualitativa — sino *por qué* está mal, *desde cuándo* exactamente, *con qué frecuencia* viene fallando, y *qué tan grave* es el problema medido cuantitativamente. La vista cualitativa del Operador es insuficiente para diagnóstico técnico.

HU-13 provee la vista técnica detallada que cubre esa brecha. Para cada componente monitoreado, se muestra:

1. **Nombre legible** del componente (consistente con la vista del Operador).
2. **Identificador interno técnico** del componente (útil para correlacionar con logs y configuración).
3. **Estado cualitativo** (OK / Degradado / Fuera de servicio), heredado del mismo monitor que alimenta HU-11.
4. **Latencia de respuesta** del componente en la última evaluación de salud, medida en milisegundos.
5. **Indicador de fallos recientes**, definido como número de evaluaciones fallidas en una ventana temporal configurable (por ejemplo, los últimos 5 minutos).
6. **Marca de tiempo del último cambio de estado** (igual que en HU-11) y, adicionalmente, **marca de tiempo de la última evaluación de salud exitosa**, lo cual permite distinguir "el componente está OK y respondió hace 2 segundos" de "el componente está OK pero la última respuesta fue hace 4 minutos, posible inicio de degradación".

La vista se presenta como una tabla con una fila por componente, con la información anterior visible sin necesidad de expandir filas ni navegar a sub-vistas. El Administrador debe poder leer el estado completo del sistema en una sola pantalla. La vista se actualiza automáticamente a medida que el monitor de componentes detecta cambios, sin necesidad de recargar manualmente.

HU-13 no implementa acciones correctivas (reinicio de componentes, recarga de configuración, etc.); es exclusivamente de consulta y diagnóstico. Las acciones correctivas, si se introducen en el futuro, serían HUs separadas.

### Criterios de aceptación

- **CA-13.1:** Dado que el Administrador ha iniciado sesión, cuando ingresa a la vista de salud técnica de componentes, entonces el sistema muestra una entrada por cada componente monitoreado, con los siguientes datos visibles sin interacción adicional: nombre legible, identificador interno, estado cualitativo, latencia de la última evaluación de salud, indicador de fallos recientes, marca de tiempo del último cambio de estado, y marca de tiempo de la última evaluación de salud exitosa.

- **CA-13.2:** Dado que el Administrador tiene la vista abierta, cuando el monitor de componentes detecta un cambio en el estado de cualquier componente (cambio cualitativo, nueva evaluación con latencia distinta, nuevo fallo registrado), entonces los valores mostrados se actualizan automáticamente sin necesidad de recargar la página, con una latencia máxima de 5 segundos desde que el monitor registra el cambio.

- **CA-13.3:** Dado que el Administrador está observando la vista y algún componente está en estado Degradado o Fuera de servicio, cuando renderiza la lista, entonces las filas correspondientes a los componentes en estado no-OK se resaltan visualmente para que sean identificables de un vistazo, de forma equivalente al resalte que aplica HU-11 para el Operador (CA-11.9), pero conservando la riqueza técnica de los demás campos.

- **CA-13.4:** Dado que el monitor de componentes deja de responder por cualquier causa, cuando el Administrador está observando la vista, entonces el sistema mantiene en pantalla los últimos valores conocidos, los marca visualmente como "no confirmados" e indica el tiempo transcurrido desde la última evaluación recibida del monitor (DHU-005 Caso B aplicado al monitor de componentes). Esta marca pasiva es complementaria a la indicación general de "estado no confirmado" que recibe el Operador transversalmente (CA-10.9 de HU-10), no la sustituye.

- **CA-13.5:** Dado que un componente nunca ha respondido a una evaluación de salud desde el arranque del sistema (por ejemplo, componente recién agregado, configuración mal hecha), cuando el Administrador consulta la vista, entonces ese componente aparece en estado Fuera de servicio, con latencia y marca de última evaluación exitosa marcadas explícitamente como "sin datos" (no como cero ni como timestamp vacío, que podrían interpretarse como datos válidos).

- **CA-13.6:** Dado que un usuario con rol Operador o Gerente intenta acceder a la vista de salud técnica del Administrador, cuando solicita la ruta correspondiente, entonces el sistema deniega el acceso conforme a la política de control de acceso por rol (HU-01 del Bloque A), respondiendo con HTTP 403 si la solicitud llega vía API o redirigiéndolo fuera de la vista si el intento es vía navegación frontend.

- **CA-13.7:** Dado que el Administrador no ha iniciado sesión, cuando intenta acceder a la vista de salud técnica de componentes, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Sustrato técnico:** esta HU consume el mismo endpoint canónico que HU-11 del Operador (CT-04.5 de TTH-04, `GET /system/components/status`), conforme a DHU-013. El contrato original de CT-04.5 cubría nombre legible, estado cualitativo, timestamp del último cambio e identificador interno; al cerrar HU-13 se aplica una ampliación a CT-04.5 dentro de TTH-04 para que el contrato cubra los campos adicionales que esta HU requiere (latencia, indicador de fallos recientes, timestamp de última evaluación exitosa). La ampliación no introduce TTH nueva ni decisión metodológica; es refinamiento del contrato del endpoint existente y se documenta en la sección de cambios de TTH-04.

- **Patrón de consumo (un endpoint, dos vistas):** HU-11 (Operador) y HU-13 (Administrador) consumen el mismo endpoint y filtran en la presentación los campos que cada rol necesita. Los campos técnicos adicionales no son sensibles (son métricas operativas del propio sistema, no datos personales ni credenciales), por lo cual no se justifica un endpoint separado ni un filtrado en backend según el rol del token. La política transversal sobre la introducción de una capa explícita de DTOs en el backend se documenta como TTH-06 (capa de DTOs transversal, clasificada como Trabajos Futuros, ver `TAREAS_TECNICAS_HABILITADORAS.md`).

- **Composición visual:** la vista de HU-13 se accede desde la navegación del Administrador, junto con las vistas de HU-14 (métricas del modelo) y HU-15 (configuración del motor). No hay HU dedicada de "dashboard del Administrador" análoga a F02 del Operador; la navegación del Administrador es la composición. Esta decisión queda registrada en DHU-014 porque el Administrador no opera sobre un único objeto en tiempo real como el Operador; opera sobre objetos distintos en momentos distintos.

- **Ventana temporal del indicador de fallos recientes:** el alcance MVP1 cierra esta ventana en un valor único configurable (sugerencia: 5 minutos) sin permitir al Administrador ajustarla desde la UI. La parametrización de la ventana desde la propia vista es trabajo futuro si se justifica con uso real.

- **Diferencia con HU-11 explicitada:** HU-11 muestra OK/Degradado/Fuera de servicio + resalte. HU-13 muestra lo mismo más latencia, fallos recientes, identificador interno y timestamp de última evaluación exitosa. Ningún campo de HU-13 contradice a HU-11; la diferencia es ortogonal (profundidad técnica añadida, no información alternativa).

- **No incluye acciones correctivas:** HU-13 es solo de consulta. Reiniciar componentes, recargar configuración, forzar nuevas evaluaciones, etc., no son parte del alcance de esta HU. Si se requieren en el futuro, son HUs separadas y probablemente requieren TTH adicional.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de rendimiento:** latencia máxima entre el cambio detectado por el monitor de componentes y su aparición en la vista del Administrador ≤ 5 s (CA-13.2).
- **RNF de usabilidad:** la información de los componentes en estado no-OK debe ser identificable de un vistazo sin lectura detenida (CA-13.3). Combinación color + ícono + texto, no solo color.
- **RNF de accesibilidad:** la distinción visual entre los tres estados (OK / Degradado / Fuera de servicio) no debe depender exclusivamente del color. Probablemente conforme a WCAG 2.1 nivel AA.
- **RNF de seguridad:** los campos técnicos adicionales que HU-13 expone (latencia, fallos recientes, identificador interno) son visibles solo para el rol Administrador. El RBAC se aplica a nivel de la ruta de la vista (CA-13.6). Los campos viajan en el wire incluso para consumidores con otros roles, pero esos consumidores no tienen acceso a la ruta que los renderiza. La decisión de no introducir un DTO específico para el Administrador o un filtrado en backend se basa en que los campos no son sensibles (ver nota técnica sobre el patrón de consumo).
- **RNF de robustez:** comportamiento ante caída del monitor de componentes (CA-13.4, DHU-005 Caso B). Se muestra el último estado conocido marcado como "no confirmado" en lugar de asumir falsamente operación normal o mostrar la vista vacía.
- **RNF de manejabilidad de datos faltantes:** componentes sin historial de evaluaciones (CA-13.5) se muestran de forma inequívocamente distinta a componentes con datos válidos, para evitar diagnósticos erróneos por confundir "sin datos" con "datos válidos en cero".

---

## HU-14 — Vista de métricas de desempeño del modelo predictivo

| Campo | Contenido |
|---|---|
| **Como** | Administrador del Sistema |
| **Quiero** | consultar las métricas de desempeño del modelo predictivo del sistema, comparando sus predicciones recientes contra los niveles de congestión que efectivamente ocurrieron |
| **Para** | evaluar si el modelo sigue siendo confiable, detectar oportunamente una posible degradación de su precisión, y sustentar con datos la decisión de mantener el modelo activo o considerar su sustitución |

**Tipo:** HU de Persona (Administrador del Sistema).
**Feature(s) origen:** F18 (Panel de métricas del modelo predictivo). Ingloba como CAs el sustrato técnico de registro de predicciones y cálculo de métricas, conforme a DHU-013 (patrón equivalente al de F31 inglobada en HU-08 CA-08.1).

### Resumen ejecutivo

**Qué entrega:** vista de métricas de desempeño del modelo predictivo principal para el Administrador. Cuatro elementos: MAE y RMSE sobre el ratio continuo, accuracy y matriz de confusión 6×6 sobre el nivel discreto 0-5. Tooltips de ayuda con la definición operacional de cada métrica. Ventana temporal configurable. Ingloba el sustrato técnico (registro continuo de predicciones, comparación con observaciones, agregación).

**CAs críticos:** CA-14.1 (sustrato inglobado: persistencia continua de predicciones con timestamp y horizonte), CA-14.3 (cálculo de las 4 métricas sobre ventana temporal), CA-14.5 (presentación de los 4 elementos), CA-14.11 (comunicación explícita ante datos insuficientes, no valores en cero), CA-14.12 (DHU-005 Caso B), CA-14.13 (HTTP 403 para roles no-Administrador).

**Estructura de CAs:** sustrato técnico inglobado (CA-14.1 a CA-14.4) → presentación al Administrador con 4 elementos y tooltips (CA-14.5 a CA-14.10) → casos degenerados (CA-14.11, CA-14.12) → control de acceso (CA-14.13, CA-14.14).

**Dependencias:** consume el modelo predictivo TTH-09 (D-006 GRU univariado, D-009 jam level 0-5) de forma agnóstica. La ventana temporal de cálculo es parametrizable vía HU-15 (familia "Predicción y evaluación del modelo"). HU-20 (MVP2) extiende el registro de CA-14.1 con persistencia paralela del modelo de respaldo. F19 (comparativa vs baseline) corresponde a HU-20 (MVP2).

**Notas clave:** doble medición (continuo + discreto) por construcción de D-009: el modelo predice ratio continuo, la discretización ocurre en presentación. Convención de matriz: **filas = real observado, columnas = predicho** (declarada en CA-14.8 y tooltip). Tooltips reducen la barrera cognitiva del perfil del Administrador, que es técnico operativo, no necesariamente experto en evaluación de modelos.

### Descripción

El modelo predictivo del sistema genera predicciones del nivel de congestión a corto plazo, en una escala discreta de 0 a 5, que el Operador consume en tiempo real (HU-03) y que el motor adaptativo usa para tomar decisiones. La confiabilidad de estas predicciones es central para el funcionamiento del sistema: predicciones sesgadas o degradadas se traducen en decisiones operativas incorrectas que el Operador no necesariamente puede detectar a simple vista.

El Administrador del Sistema es la persona responsable de evaluar continuamente el desempeño del modelo. HU-14 le entrega una vista dedicada con las métricas estándar de evaluación de modelos predictivos, calculadas sobre una ventana temporal reciente, comparando para cada predicción registrada el nivel que el modelo predijo contra el nivel que efectivamente ocurrió cuando llegó el horizonte de predicción.

La vista contiene cuatro elementos visuales principales, presentados sin necesidad de navegación adicional:

1. **MAE (Error Absoluto Medio)** sobre la variable continua que el modelo predice antes de su discretización a la escala 0-5. Indica el error típico del modelo en las mismas unidades que la variable predicha.
2. **RMSE (Raíz del Error Cuadrático Medio)** sobre la misma variable. Penaliza más los errores grandes que el MAE; su comparación con el MAE informa sobre la uniformidad de los errores.
3. **Accuracy (Exactitud)** sobre el nivel discreto 0-5. Porcentaje de predicciones donde el nivel predicho coincide exactamente con el nivel real observado.
4. **Matriz de confusión 6×6** del nivel discreto, con filas correspondientes al nivel real observado y columnas al nivel predicho por el modelo. La diagonal principal son los aciertos; las celdas fuera de la diagonal son errores, más graves cuanto más alejadas de la diagonal.

Cada uno de los cuatro elementos cuenta con un ícono de ayuda visible junto al título que, al ser activado por el Administrador, despliega una explicación breve de cómo interpretar la métrica. Esto reduce la carga cognitiva de quien consulta la vista sin necesariamente recordar las definiciones formales de cada métrica.

El sustrato técnico (registro continuo de predicciones realizadas, comparación con las observaciones reales cuando el horizonte llega, agregación de métricas sobre ventana temporal configurable) es responsabilidad de esta misma HU, sin generar TTH adicional, conforme a DHU-013.

### Criterios de aceptación

#### Sustrato técnico (registro y cálculo)

- **CA-14.1:** Dado que el modelo predictivo del sistema genera una predicción para un horizonte futuro, cuando esa predicción se entrega a sus consumidores (Operador, motor adaptativo, otros), entonces el sistema persiste la predicción de forma durable con al menos: marca de tiempo de generación, marca de tiempo del horizonte al que la predicción aplica, valor predicho del ratio continuo, nivel discreto 0-5 correspondiente.

- **CA-14.2:** Dado que ha transcurrido el horizonte de una predicción previamente registrada, cuando el sistema dispone de la observación real de ese momento, entonces persiste la asociación entre la predicción registrada y la observación real, agregando al registro al menos: valor real del ratio continuo, nivel discreto 0-5 real.

- **CA-14.3:** Dado que existen predicciones con observación asociada en el registro, cuando el sistema agrega las métricas sobre la ventana temporal configurada, entonces calcula y mantiene actualizado al menos: MAE sobre el ratio continuo, RMSE sobre el ratio continuo, accuracy sobre el nivel discreto, y la matriz de confusión 6×6 del nivel discreto.

- **CA-14.4:** Dado que el cálculo de métricas opera sobre una ventana temporal, cuando se evalúa qué predicciones entran al cálculo, entonces se incluyen únicamente las predicciones cuyo horizonte ya transcurrió y para las cuales hay observación real registrada. Predicciones cuyo horizonte aún no llega o sin observación correspondiente no se incluyen en el cálculo. La ventana temporal por defecto es configurable; el alcance MVP1 cierra el valor por defecto al implementar (sugerencia: 24 horas). La parametrización de esta ventana es responsabilidad de HU-15 (familia "Predicción y evaluación del modelo").

#### Presentación al Administrador

- **CA-14.5:** Dado que el Administrador ha iniciado sesión, cuando ingresa a la vista de métricas del modelo predictivo, entonces el sistema muestra los cuatro elementos: MAE, RMSE, accuracy y matriz de confusión, calculados sobre la ventana temporal actual.

- **CA-14.6:** Dado que el Administrador está observando la vista, cuando se generan nuevas predicciones con observación asociada que actualizan el cálculo de métricas, entonces los valores mostrados se actualizan automáticamente sin necesidad de recargar la página. La latencia entre la actualización del cálculo y su reflejo en la vista debe ser razonable para una vista de análisis (criterio: ≤ 30 segundos, no requiere tiempo real estricto).

- **CA-14.7:** Dado que el Administrador no recuerda cómo interpretar alguna de las métricas mostradas, cuando activa el ícono de ayuda visible junto al título de la métrica, entonces el sistema despliega una explicación breve y autocontenida que permite al Administrador leer la métrica sin necesidad de consultar documentación externa. Esto aplica a las cuatro métricas: MAE, RMSE, accuracy y matriz de confusión.

- **CA-14.8:** Dado que el Administrador consulta la matriz de confusión, cuando renderiza la matriz, entonces la convención de presentación es **filas = nivel real observado, columnas = nivel predicho por el modelo** (convención académica estándar, equivalente a la del módulo de métricas de scikit-learn). La convención queda declarada en el tooltip de ayuda de la matriz, no solo deducible visualmente.

- **CA-14.9:** Dado que el Administrador consulta la matriz de confusión, cuando renderiza la matriz, entonces se presentan los valores absolutos (conteos) de cada celda junto con los totales de cada fila y de cada columna. El Administrador puede alternar la presentación a porcentajes por fila mediante un control visible (toggle), para neutralizar el efecto del desbalance natural de clases (los niveles bajos de congestión ocurren más frecuentemente que los altos).

- **CA-14.10:** Dado que el Administrador consulta la matriz, cuando se renderiza, entonces la diagonal principal (aciertos del modelo) es identificable visualmente respecto a las celdas fuera de la diagonal (errores), por ejemplo mediante un fondo o intensidad distintos. Esto facilita la lectura sin requerir contar celdas manualmente.

#### Manejo de casos degenerados

- **CA-14.11:** Dado que el sistema acaba de arrancar o la ventana temporal configurada todavía no contiene predicciones con observación asociada, cuando el Administrador consulta la vista, entonces el sistema indica explícitamente "no hay datos suficientes para calcular métricas en esta ventana" en lugar de mostrar valores en cero (que podrían confundirse con métricas reales catastróficas) o métricas calculadas sobre cero muestras.

- **CA-14.12:** Dado que el componente que registra predicciones u observaciones deja de responder, cuando el Administrador consulta la vista, entonces el sistema muestra las últimas métricas calculadas marcadas como "no actualizadas" e indica el tiempo transcurrido desde el último cálculo exitoso (DHU-005 Caso B aplicado al motor de cálculo de métricas).

#### Control de acceso

- **CA-14.13:** Dado que un usuario con rol Operador o Gerente intenta acceder a la vista de métricas del modelo, cuando solicita la ruta correspondiente, entonces el sistema deniega el acceso conforme a la política de control de acceso por rol (HU-01 del Bloque A), respondiendo con HTTP 403 si la solicitud llega vía API o redirigiéndolo fuera de la vista si el intento es vía navegación frontend.

- **CA-14.14:** Dado que el Administrador no ha iniciado sesión, cuando intenta acceder a la vista de métricas del modelo, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Sustrato inglobado, no TTH:** conforme a DHU-013, el sustrato de registro de predicciones y cálculo de métricas se ingloba como CAs de esta HU (CA-14.1 a CA-14.4) en lugar de extraerse como TTH separada. La justificación es que este sustrato es consumido únicamente por esta HU; las TTH se justifican cuando una pieza de lógica autónoma es consumida por múltiples HUs (caso de TTH-04 consumida por HU-10, HU-11, HU-12 y HU-13). Patrón previo equivalente: F31 (persistencia de decisiones del motor) inglobada en CA-08.1 de HU-08, sin generar TTH.

- **Variable continua vs discreta:** el modelo predictivo del sistema produce internamente un valor continuo (el ratio velocidad/free-flow, conforme a D-009) que luego se discretiza a la escala 0-5 para ser consumido por las Personas y por el motor adaptativo. Esta HU consulta las métricas en ambos niveles: MAE y RMSE sobre el ratio continuo (preservan la resolución del entrenamiento del modelo), accuracy y matriz de confusión sobre el nivel discreto (lo que las Personas y el motor efectivamente consumen). La doble medición no es redundante: un modelo puede tener buen MAE pero accuracy peor de lo esperado si la discretización está mal calibrada.

- **Ventana temporal configurable:** el valor por defecto de la ventana (sugerencia: 24 horas) se cierra al implementar. La parametrización de la ventana desde la vista del Administrador es responsabilidad de HU-15 (familia "Predicción y evaluación del modelo"), conforme a DHU-014. HU-14 consume el valor vigente; HU-15 lo configura.

- **Comparativa con baseline:** la comparación de las métricas del modelo activo contra un baseline (por ejemplo, modelo simple anterior) corresponde a F19 (Comparativa de métricas del modelo vs baseline), clasificada como MVP2. HU-14 muestra las métricas del modelo activo; la HU MVP2 las comparará. La separación es intencional para mantener HU-14 enfocada.

- **Tooltips de ayuda:** la decisión de incluir tooltips de ayuda en MAE, RMSE, accuracy y matriz de confusión responde a que el perfil del Administrador del Sistema declarado en el Inception tiene foco operativo y de soporte técnico, no necesariamente formación profunda en evaluación de modelos predictivos. Los tooltips reducen la barrera cognitiva sin contaminar la vista con texto permanente. Textos exactos se cierran al implementar; sugerencia documentada en la sesión de redacción del Bloque D (ver DHU-014).

- **Convención de la matriz declarada explícitamente:** la convención filas = real, columnas = predicho se declara tanto en CA-14.8 como en el tooltip de ayuda de la matriz, para neutralizar la ambigüedad histórica de la literatura (algunos textos invierten filas y columnas). La convención adoptada es la más usada académicamente y la del módulo de métricas de scikit-learn.

- **Frecuencia de cálculo:** las métricas no requieren recalcularse en cada nueva predicción con observación asociada; basta con un recálculo periódico (sugerencia: cada minuto) o incremental. El criterio CA-14.6 cierra la latencia máxima de reflejo en la vista en ≤ 30 segundos; el cálculo subyacente puede ser más lento si se implementa incremental.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de durabilidad:** el registro de predicciones y observaciones (CA-14.1, CA-14.2) debe ser durable; pérdida del registro implica perder la capacidad de evaluar el modelo retroactivamente.
- **RNF de auditabilidad:** las entradas del registro de predicciones no deben modificarse después de escritas (auditoría confiable del desempeño del modelo en el tiempo). En línea con el RNF equivalente de HU-08 y CA-10.7 de HU-10.
- **RNF de rendimiento:** el cálculo de métricas sobre la ventana temporal por defecto no debe degradar la respuesta del sistema; cálculo incremental o asíncrono según se cierre al implementar.
- **RNF de usabilidad:** la vista debe permitir al Administrador interpretar las métricas sin documentación externa, mediante los tooltips de ayuda integrados (CA-14.7).
- **RNF de accesibilidad:** los tooltips deben ser activables tanto con teclado como con dispositivo apuntador. La distinción visual de la diagonal de la matriz (CA-14.10) no debe depender exclusivamente del color.
- **RNF de robustez:** comportamiento ante caída del componente de cálculo de métricas (CA-14.12, DHU-005 Caso B). En lugar de mostrar la vista vacía o métricas en cero, se muestran las últimas métricas conocidas marcadas como no actualizadas.
- **RNF de manejabilidad de datos faltantes:** ventana sin datos suficientes (CA-14.11) se comunica explícitamente, no se muestra como métricas en cero.

---

## HU-15 — Configuración de parámetros operativos del sistema

| Campo | Contenido |
|---|---|
| **Como** | Administrador del Sistema |
| **Quiero** | configurar los parámetros operativos del sistema desde una vista dedicada, organizados por familia de comportamiento, con persistencia de los cambios y registro de auditoría de cada modificación |
| **Para** | ajustar el comportamiento del sistema a las condiciones reales de la intersección sin necesidad de redespliegue, manteniendo trazabilidad de quién cambió qué y cuándo, y pudiendo restaurar configuraciones de referencia si una modificación produce resultados no deseados |

**Tipo:** HU de Persona (Administrador del Sistema).
**Feature(s) origen:** F20 (Configuración de parámetros del motor adaptativo). Ingloba como CAs el sustrato técnico de persistencia y auditoría, conforme a DHU-013 (patrón equivalente al de F31 inglobada en HU-08 CA-08.1 y al sustrato inglobado en HU-14).

### Resumen ejecutivo

**Qué entrega:** vista de configuración de parámetros operativos del sistema para el Administrador. Organizados en tres familias funcionales (visualización del tráfico, predicción y evaluación, monitor de salud). Persistencia y auditoría de cada modificación inglobadas. Aplicación sin redeploy. Restauración a valores por defecto seguros.

**CAs críticos:** CA-15.1 (sustrato inglobado: consulta con timestamp e identidad), CA-15.2 (modificación con auditoría), CA-15.3 (efecto sin redeploy en ≤ 30 s), CA-15.7 (confirmación visual de persistencia con preservación de contenido ante fallo), CA-15.11 (control de concurrencia entre Administradores con last-write-wins explícito), CA-15.12 (HTTP 403 para roles no-Administrador).

**Estructura de CAs:** sustrato técnico inglobado (CA-15.1 a CA-15.4) → presentación y operación (CA-15.5 a CA-15.9) → manejo de casos degenerados con concurrencia (CA-15.10, CA-15.11) → control de acceso (CA-15.12, CA-15.13).

**Dependencias:** cubre parámetros referenciados desde CA-02.3 (umbrales de cola), CA-03.1 (horizonte de predicción), CA-03.3 (umbral de congestión), CA-14.4 (ventana de cálculo de métricas) y CT-04.1 (frecuencia del monitor de salud). **Separada de TTH-05** (tiempos preconfigurados del degradado nivel 3, vive ahí por DHU-013). Aplica DHU-005 Caso B, DHU-006, DHU-013, DHU-014.

**Notas clave:** parámetros internos de las estrategias del motor (Webster, Max Pressure, MTC) quedan internos al sistema en MVP1; su exposición es trabajo futuro condicional. Concurrencia entre Administradores con control optimista basado en timestamp de última modificación (CA-15.11). Restauración de defaults (CA-15.9) preserva auditoría: queda registrada como una modificación más.

### Descripción

El sistema tiene parámetros operativos que afectan cómo se muestra el estado del tráfico al Operador, cómo el modelo predictivo entrega su salida, y con qué frecuencia se evalúa la salud de los componentes. Estos parámetros son valores numéricos sencillos cuya elección concreta depende de las condiciones reales de la intersección (intensidad de tráfico, longitudes físicas, comportamiento observado) y que, en operación real, requieren ajuste periódico sin que ese ajuste implique desplegar nueva versión del software.

El Administrador del Sistema es la persona responsable de mantener estos parámetros calibrados. HU-15 le entrega una vista única donde puede consultar el valor actual de cada parámetro, modificar los valores que requieran ajuste, persistir los cambios para que apliquen al sistema en operación, y consultar el historial de modificaciones recientes. Los parámetros se organizan visualmente en tres familias funcionales, cada una con su propósito propio:

1. **Visualización del estado del tráfico:** parámetros que afectan cómo el Operador percibe el estado de las colas en el dashboard. Específicamente los umbrales que definen cuándo una cola se considera de nivel medio o alto, parámetros referenciados desde CA-02.3 de HU-02.

2. **Predicción y evaluación del modelo:** parámetros que afectan cómo opera el modelo predictivo y cómo se evalúa su desempeño. Incluye el horizonte de predicción (cuánto hacia el futuro proyecta el modelo, referenciado desde CA-03.1 de HU-03), el umbral del nivel de congestión a partir del cual se resalta visualmente la predicción al Operador (referenciado desde CA-03.3 de HU-03, con valor por defecto en nivel ≥ 3 conforme a D-009), y la ventana temporal sobre la cual se calculan las métricas de desempeño del modelo en HU-14 (referenciada desde CA-14.4).

3. **Monitor de salud del sistema:** parámetros que afectan cómo se detecta el estado de los componentes. Específicamente la frecuencia con que el monitor evalúa cada componente, referenciado desde CT-04.1 de TTH-04 (valor por defecto 5 segundos).

Los parámetros internos de las estrategias de control del motor adaptativo (parámetros que afectan cómo cada estrategia decide los tiempos del semáforo) **no son parte del alcance MVP1 de HU-15**. Su exposición al Administrador requeriría conocimiento profundo de ingeniería de tráfico que excede el perfil de la Persona declarada en el Inception y agregaría riesgo operativo sin valor proporcional. Su inclusión en versiones futuras del producto se evalúa cuando exista necesidad concreta y un Administrador con perfil técnico apropiado.

La configuración de los tiempos preconfigurados aplicables cuando el sistema entra en degradado nivel 3 también queda fuera del alcance de HU-15. Esa configuración tiene su propia interfaz dedicada en TTH-05 (CT-05.3), gestionada por el mismo Administrador pero con propósito y vocabulario distintos: HU-15 calibra el comportamiento del sistema en operación normal; TTH-05 define el comportamiento de seguridad cuando los componentes adaptativos no están disponibles. La separación se preserva por cohesión conceptual conforme a DHU-013.

### Criterios de aceptación

#### Sustrato técnico (persistencia y auditoría)

- **CA-15.1:** Dado que existe una tabla persistente en la base de datos para los parámetros operativos del sistema, cuando se consulta el valor actual de cualquier parámetro, entonces el sistema retorna el valor persistido junto con su marca de tiempo de última modificación y la identidad del Administrador que la realizó.

- **CA-15.2:** Dado que el Administrador modifica el valor de uno o varios parámetros y confirma el cambio, cuando el sistema procesa la modificación, entonces persiste los nuevos valores y registra una entrada de auditoría por cada parámetro modificado, con: marca de tiempo, identidad del Administrador, nombre del parámetro, valor anterior y valor nuevo.

- **CA-15.3:** Dado que un parámetro recién persistido afecta el comportamiento de un componente del sistema en operación, cuando la modificación se completa, entonces el componente afectado consume el nuevo valor sin necesidad de reiniciar el sistema ni redesplegar. La latencia máxima entre la persistencia del nuevo valor y su efecto en el componente debe ser razonable para una operación de configuración (criterio: ≤ 30 segundos).

- **CA-15.4:** Dado que el sistema acaba de instalarse o la tabla de parámetros está vacía, cuando un componente consulta un parámetro, entonces el sistema retorna un valor por defecto seguro documentado, equivalente a los valores referenciados desde las HUs y TTH que consumen cada parámetro (CA-02.3, CA-03.1, CA-03.3, CA-14.4, CT-04.1). Esto garantiza que el sistema sea operativo desde el primer arranque sin requerir configuración explícita previa.

#### Presentación al Administrador y operación

- **CA-15.5:** Dado que el Administrador ha iniciado sesión, cuando ingresa a la vista de configuración, entonces el sistema muestra los parámetros organizados en tres secciones visualmente distinguibles correspondientes a las tres familias: "Visualización del estado del tráfico", "Predicción y evaluación del modelo", y "Monitor de salud del sistema". Cada parámetro se muestra con su nombre legible, su valor actual, su unidad cuando aplica, y la marca de tiempo de su última modificación.

- **CA-15.6:** Dado que el Administrador quiere modificar uno o varios parámetros, cuando edita los valores en la vista, entonces el sistema valida en tiempo real que los nuevos valores cumplan los rangos válidos documentados para cada parámetro (por ejemplo, frecuencias positivas, umbrales en rangos razonables, horizontes en intervalos plausibles). Valores fuera de rango se rechazan en el formulario con mensaje claro al Administrador antes de permitir el envío.

- **CA-15.7:** Dado que el Administrador modificó uno o varios parámetros válidos, cuando solicita guardar los cambios, entonces el sistema confirma visualmente que la modificación se persistió correctamente, mostrando el nuevo valor reflejado en la vista y un indicador de éxito. Si la persistencia falla por cualquier causa, el sistema informa al Administrador, conserva los valores ingresados en el formulario para que pueda reintentar, y mantiene los valores anteriores aún vigentes en operación hasta que la persistencia se complete con éxito.

- **CA-15.8:** Dado que el Administrador quiere consultar quién modificó un parámetro y cuándo, cuando solicita el historial de cambios desde la propia vista, entonces el sistema muestra el listado de modificaciones recientes en orden cronológico inverso, con: marca de tiempo, identidad del Administrador autor del cambio, parámetro afectado, valor anterior y valor nuevo. La ventana de historial visible por defecto se cierra al implementar; la consulta más allá de esa ventana es trabajo futuro si se justifica.

- **CA-15.9:** Dado que el Administrador rompió el comportamiento del sistema con una modificación reciente y quiere volver a una configuración conocida, cuando solicita "restaurar valores por defecto" mediante un control explícito de la vista, entonces el sistema solicita confirmación explícita del Administrador (operación destructiva potencial), y al confirmar, sustituye los valores actuales por los valores por defecto seguros documentados (los mismos de CA-15.4), registrando la restauración como una modificación más en el registro de auditoría con autor identificable.

#### Manejo de casos degenerados

- **CA-15.10:** Dado que el componente responsable de la persistencia de parámetros deja de responder, cuando el Administrador intenta consultar o modificar la configuración, entonces el sistema indica explícitamente la indisponibilidad temporal, mostrando los últimos valores conocidos marcados como "no confirmados" y bloqueando las acciones de modificación hasta que el componente vuelva a responder (DHU-005 Caso B aplicado al subsistema de configuración).

- **CA-15.11:** Dado que dos Administradores modifican la configuración simultáneamente, cuando ambos intentan guardar, entonces el sistema aplica el siguiente comportamiento: (a) la primera modificación en guardarse se persiste normalmente y queda registrada en auditoría; (b) cuando el segundo Administrador intenta guardar, el sistema detecta que la configuración fue modificada después del momento en que él cargó la vista, y antes de aceptar la nueva modificación le muestra una advertencia clara que incluye qué Administrador realizó la modificación previa, cuándo, y cuáles parámetros cambió; (c) el segundo Administrador puede entonces confirmar que aún así quiere aplicar sus cambios (en cuyo caso la modificación se persiste sobrescribiendo los valores actuales — last-write-wins — y queda registrada en auditoría como cualquier otra), o cancelar su modificación y recargar la vista con los valores recién actualizados para reevaluar. En ningún caso el segundo guardado sobrescribe valores sin que el segundo Administrador haya visto la advertencia. El registro de auditoría preserva la traza completa de ambas modificaciones, no solo de la última.

#### Control de acceso

- **CA-15.12:** Dado que un usuario con rol Operador o Gerente intenta acceder a la vista de configuración, cuando solicita la ruta correspondiente, entonces el sistema deniega el acceso conforme a la política de control de acceso por rol (HU-01 del Bloque A), respondiendo con HTTP 403 si la solicitud llega vía API o redirigiéndolo fuera de la vista si el intento es vía navegación frontend.

- **CA-15.13:** Dado que el Administrador no ha iniciado sesión, cuando intenta acceder a la vista de configuración, entonces el sistema lo redirige a la pantalla de login.

### Notas técnicas

- **Sustrato inglobado, no TTH:** conforme a DHU-013, el sustrato de persistencia y auditoría de parámetros se ingloba como CAs de esta HU (CA-15.1 a CA-15.4 y CA-15.8) en lugar de extraerse como TTH separada. La justificación es la misma que para HU-14: este sustrato es consumido únicamente por la propia HU; el patrón previo equivalente es F31 inglobada en CA-08.1 de HU-08.

- **Separación con TTH-05 preservada:** la configuración de los tiempos preconfigurados aplicables cuando el sistema entra en degradado nivel 3 vive en TTH-05 (CT-05.3), no en HU-15, conforme a DHU-013 (cierre de la pregunta abierta de TTH-05). El Administrador interactúa con dos interfaces distintas para dos propósitos distintos: HU-15 calibra el comportamiento del sistema en operación normal; TTH-05 define el comportamiento de seguridad cuando los componentes adaptativos no están disponibles. La separación se preserva por cohesión conceptual.

- **Parámetros fuera del alcance MVP1:** los parámetros internos de las estrategias de control del motor (parámetros que afectan cómo cada estrategia decide los tiempos del semáforo) quedan internos al sistema en MVP1, conforme a DHU-014. Su exposición al Administrador es mejora natural si se justifica con (a) necesidad concreta de calibración fina, y (b) un Administrador con perfil técnico apropiado en ingeniería de tráfico. No es MVP2; es trabajo futuro.

- **Granularidad de una sola HU:** HU-15 cubre los tres dominios de configuración (visualización, predicción/evaluación, monitor) en una única HU porque el Administrador trabaja con todos ellos en un mismo flujo, conforme a DHU-013. La organización por familias se hace dentro de la HU con CAs estructurados, no descomponiendo en múltiples HUs.

- **Valores por defecto documentados:** los valores por defecto referenciados en CA-15.4 y restaurables en CA-15.9 son los que aparecen en las CAs y CT de origen (CA-02.3, CA-03.1, CA-03.3, CA-14.4, CT-04.1). El valor por defecto de "ventana temporal de cálculo de métricas" se cierra al implementar (sugerencia: 24 horas, conforme a la nota técnica de HU-14). El valor por defecto del horizonte de predicción y los umbrales de cola se cierran al implementar con criterio de ingeniería de tráfico.

- **Validación de rangos:** los rangos válidos de cada parámetro (CA-15.6) se documentan junto con cada parámetro al implementar. Principios generales: umbrales de cola en número de vehículos razonables para la longitud física del acceso; horizonte de predicción en intervalos para los cuales el modelo está entrenado; frecuencia del monitor en valores que no saturen los componentes monitoreados.

- **Concurrencia entre Administradores (CA-15.11):** mecánicamente se implementa con control de concurrencia optimista con marca de versión: cada lectura de la vista anota el timestamp de la última modificación de la configuración; cada intento de guardado verifica que ese timestamp no haya cambiado en la base de datos desde la lectura; si cambió, se bloquea el guardado y se muestra la advertencia con los detalles de la modificación intermedia. Este patrón es estándar y suficiente para el escenario MVP1 donde la concurrencia será rara. El registro de auditoría preserva ambas modificaciones, lo cual permite reconstruir la historia completa aunque el último guardado haya sobrescrito al anterior.

- **Restaurar valores por defecto (CA-15.9):** este control existe específicamente para el caso "el Administrador rompió algo y necesita volver al punto seguro". No es una operación frecuente; el control se ubica de forma que requiera intención explícita (no accidental) y solicita confirmación. La restauración queda auditada como cualquier otra modificación para preservar trazabilidad.

### Candidatos a RNF (para futuro documento RF/RNF)

- **RNF de durabilidad:** los parámetros persistidos no pueden perderse por fallo temporal del sistema (CA-15.1, CA-15.2). El registro de auditoría tampoco.
- **RNF de auditabilidad:** las entradas del registro de auditoría no deben modificarse después de escritas. En línea con el RNF equivalente de HU-08, HU-10 y HU-14.
- **RNF de continuidad operativa:** las modificaciones de configuración no deben interrumpir la operación del sistema; los componentes afectados consumen los nuevos valores sin necesidad de reinicio (CA-15.3).
- **RNF de seguridad:** la modificación de parámetros está restringida al rol Administrador (CA-15.12). Un usuario sin ese rol no debe poder modificar parámetros bajo ninguna circunstancia, incluso si conoce los endpoints subyacentes.
- **RNF de robustez:** comportamiento ante caída del componente de persistencia de parámetros (CA-15.10, DHU-005 Caso B). En lugar de mostrar la vista vacía o aceptar modificaciones que se perderán, el sistema bloquea las modificaciones y comunica la indisponibilidad temporal.
- **RNF de rendimiento:** la consulta de la vista debe ser razonable para una operación de configuración (criterio: ≤ 2 segundos para abrir la vista con todos los parámetros y el historial reciente).
- **RNF de manejabilidad de concurrencia:** no perder modificaciones silenciosamente cuando dos Administradores trabajan simultáneamente (CA-15.11).
- **RNF de seguridad operativa (valores por defecto):** los valores por defecto referenciados en CA-15.4 y CA-15.9 deben ser tales que el sistema sea operativo y seguro desde el primer arranque o tras una restauración, sin requerir ajuste manual previo.

---

## Resumen del Bloque D

| HU | Título | Sujeto | Tipo | Feature(s) origen | Clasif. MVP |
|---|---|---|---|---|---|
| HU-13 | Vista técnica de salud de los componentes del sistema | Administrador | Persona | F17 | MVP1 |
| HU-14 | Vista de métricas de desempeño del modelo predictivo | Administrador | Persona | F18 (sustrato inglobado) | MVP1 |
| HU-15 | Configuración de parámetros operativos del sistema | Administrador | Persona | F20 (sustrato inglobado) | MVP1 |

**Total Bloque D: 3 HUs operativas + 0 TTH nuevas.**

F18 (sustrato técnico de registro de predicciones y cálculo de métricas) queda inglobada en CA-14.1 a CA-14.4 de HU-14, conforme a DHU-013.

F20 (sustrato técnico de persistencia y auditoría de parámetros) queda inglobada en CA-15.1 a CA-15.4 y CA-15.8 de HU-15, conforme a DHU-013.

F21 (Reentrenamiento del modelo) fue reclasificada a Trabajos Futuros por DHU-012, fuera del alcance MVP1.

---

## Cambios aplicados a documentos previos como consecuencia del Bloque D

Durante la redacción del Bloque D, las siguientes modificaciones se aplicaron a documentos previos:

1. **Ampliación de CT-04.5 dentro de TTH-04** (`TAREAS_TECNICAS_HABILITADORAS.md`): el contrato del endpoint `GET /system/components/status` se amplía para cubrir los campos técnicos adicionales que HU-13 requiere (latencia, indicador de fallos recientes, timestamp de última evaluación exitosa). HU-11 del Operador puede ignorar estos campos; no contradice su contrato previo. La ampliación no introduce TTH nueva ni decisión metodológica.

2. **Creación de TTH-06 — Capa de DTOs transversal al backend** (`TAREAS_TECNICAS_HABILITADORAS.md`), clasificada como **Trabajos Futuros**. Mejora técnica transversal identificada durante la discusión sobre el patrón de consumo de CT-04.5 por HU-11 y HU-13. No bloquea ninguna HU del MVP1.

3. **Cierre de DHU-014** (`DECISIONS_HU.md`): consolida las decisiones de redacción del Bloque D (numeración, sin HU dedicada de dashboard, selección de parámetros, métricas exactas, concurrencia, inglobación de la ventana temporal de HU-14 en HU-15, creación de TTH-06).

---

## Decisiones que aplicaron a este bloque

Durante la redacción del Bloque D se cerró la siguiente decisión formal (en `DECISIONS_HU.md`):

- **DHU-014:** decisiones de redacción del Bloque D (consolidadas). Cubre numeración compactada (HU-13 = F17, HU-14 = F18, HU-15 = F20), sin HU dedicada de dashboard del Administrador, selección concreta de parámetros de F20 en MVP1, métricas exactas en HU-14 (MAE + RMSE + accuracy + matriz de confusión 6×6 con tooltips), concurrencia entre Administradores como last-write-wins con advertencia, inglobación de la ventana temporal de cálculo de métricas de HU-14 en HU-15, y creación de TTH-06 como Trabajos Futuros.

DHU-013 (clasificación HU/TTH del Bloque D) había sido cerrada antes del inicio de la redacción.

---

## Próximos pasos

Esta sesión cerró el Bloque D. A la fecha actual, los Bloques E y F del MVP1 están cerrados y el MVP2 también está cerrado (DHU-017, 2026-05-16); **con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 21 HUs operativas (HU-01 a HU-21) y 11 TTH (TTH-01 a TTH-11).** HU-20 (comparativa vs modelo de respaldo) extiende HU-14 mediante persistencia paralela inglobada. Los siguientes pasos del proyecto, en sesiones futuras:

1. **Bloque E — Componentes centrales del sistema** (F32 SUMO, F33 visión, F34 predictor, F35 motor adaptativo → 0 HUs operativas + 5 TTH: TTH-07 a TTH-11; ya cerrado el 2026-05-15 por DHU-015). Ver `HU_BLOQUE_E.md`.

2. **Bloque F — Gerente, reportería mínima** (F12 + F13 fusionadas en HU-16 con F30 inglobada, F14 en HU-17 → 2 HUs operativas + 0 TTH nuevas; ya cerrado el 2026-05-16 por DHU-016). Ver `HU_BLOQUE_F.md`.

3. **MVP2 — HUs documentadas con construcción condicional a holgura del cronograma tras cerrar MVP1** (F11→HU-09; F15→HU-18; F16→HU-19; F19→HU-20; F28→HU-21; 5 HUs operativas + 0 TTH nuevas; cerrado el 2026-05-16 por DHU-017). Semántica refinada por DHU-012. Ver `HU_MVP2.md`.

Tras cerrar el MVP2 (ya hecho), los próximos pasos del proyecto, fuera del alcance del Product Backlog funcional, son:

1. **Documento de Requisitos Funcionales y No Funcionales (RF/RNF)** consolidando los "Candidatos a RNF" de todas las HUs (HU-01 a HU-21) en un documento único aprobado, conforme a DHU-007 pendiente. Sesión dedicada futura.
2. **Ceremonias de estimación (Planning Poker) y priorización (MoSCoW)** sobre el backlog completo.
3. **Implementación SCRUM del MVP1** (16 HUs operativas + 11 TTH del MVP1). El MVP2 (5 HUs adicionales) entra al sprint si hay holgura de cronograma, conforme a la semántica refinada por DHU-012.
4. **SDD (Software Design Document)**, siguiente entregable académico mayor del proyecto.

---

## Documentos relacionados

- `HU_BLOQUE_A.md` — Bloque A del Product Backlog (acceso al sistema, 1 HU).
- `HU_BLOQUE_B.md` — Bloque B del Product Backlog (8 HUs: HU-02 a HU-09).
- `HU_BLOQUE_C.md` — Bloque C del Product Backlog (3 HUs operativas: HU-10, HU-11, HU-12).
- `HU_BLOQUE_E.md` — Bloque E del Product Backlog (0 HUs operativas; mapeo a TTH-07 a TTH-11 y decisiones tomadas durante la redacción).
- `HU_BLOQUE_F.md` — Bloque F del Product Backlog (2 HUs operativas: HU-16, HU-17; F30 inglobada como CAs).
- `HU_MVP2.md` — MVP2 del Product Backlog (HU-18, HU-19, HU-20, HU-21; HU-09 reside en `HU_BLOQUE_B.md`). HU-20 extiende CA-14.1 inglobando persistencia paralela del modelo de respaldo.
- `DECISIONS_HU.md` — Decisiones metodológicas sobre HUs (DHU-001 a DHU-022).
- `DECISIONS.md` — Decisiones técnicas del producto (D-001 a D-009).
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-01 a TTH-11.
- `LEAN_INCEPTION_CEREBROVIAL.md` — Inception completo aplicado al proyecto.
- `FEATURE_BACKLOG_DETALLADO.md` — Detalle completo de las 41 features identificadas (29 MVP1 + 5 MVP2 + 7 Trabajos Futuros).
- `EVOLUCION_TESIS.md` — Narrativa de las 4 fases del proyecto.
