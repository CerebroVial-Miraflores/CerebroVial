# Requisitos Funcionales y No Funcionales — CerebroVial

> Documento de Requisitos Funcionales (RF) y Requisitos No Funcionales (RNF) del proyecto CerebroVial. Sistema inteligente de control adaptativo de semáforos para la intersección de Miraflores (Lima).
>
> **Versión del documento:** 1 (versión inicial; cierre de DHU-007 pendiente desde el cierre del MVP2).
> **Fecha de redacción:** 2026-05-17.
> **Estado:** Borrador en revisión por bloques (preámbulo + sección 1 redactados; secciones 3, 2, 4, 5, 6 pendientes).

---

## 0. Preámbulo

### 0.1 Propósito de este documento

Este documento consolida los Requisitos Funcionales (RF) y los Requisitos No Funcionales (RNF) del sistema CerebroVial, derivados de las 21 Historias de Usuario operativas (HU-01 a HU-21) y las 11 Tareas Técnicas Habilitadoras (TTH-01 a TTH-11) del Product Backlog cerrado al 2026-05-16.

> **Nota (2026-06-02, DHU-028):** posteriormente al cierre, el Product Backlog se amplió a 22 HU + 12 TTH con HU-22 (mapa de congestión de la red) y TTH-12 (infraestructura de datos de congestión por arista). Los "Candidatos a RNF" de HU-22 (performance de tiempo real, robustez ante interrupción Caso A, usabilidad/legibilidad panorámica, accesibilidad WCAG 2.1 AA — incluida RNF-INT-02 — y performance de carga del mapa) **aún no están consolidados** en este documento; su incorporación al catálogo RF/RNF queda como pasada futura. Por eso este documento sigue refiriéndose a las 21 HU de las que efectivamente deriva su catálogo actual.

El documento ejecuta el trabajo pendiente declarado en DHU-007 (2026-05-13), cuya línea 353 estableció que tras el cierre del Product Backlog se redactaría un documento único que (a) consolidara los "Candidatos a RNF" de todas las HUs, (b) numerara cada RNF con un identificador estable, (c) definiera umbrales formales y (d) habilitara que las HUs referenciaran al RNF correspondiente por código. El presente documento cumple esos cuatro objetivos.

Las decisiones metodológicas que orientan la redacción están consolidadas en **DHU-019** (`DECISIONS_HU.md`). Los lectores que necesiten entender por qué este documento usa una taxonomía específica (ISO/IEC 25010:2023), por qué clasifica los RNF de una manera concreta o por qué los CAs de las HUs no se reescriben deben consultar DHU-019 antes de cuestionar las decisiones de este documento.

### 0.2 Alcance

**Dentro del alcance del documento:**

- Catálogo de **Requisitos Funcionales (RF)** consolidados desde los 130 CAs aproximados de las 21 HUs operativas, mediante composición transversal según DHU-019 subsección E.
- Catálogo de **Requisitos No Funcionales (RNF)** clasificados según las 9 características de calidad de ISO/IEC 25010:2023, derivados de los "Candidatos a RNF" de las 21 HUs y de los criterios técnicos de terminado de las 11 TTH cuando estos exhiben naturaleza no funcional.
- **Matriz de trazabilidad bidireccional** entre RF/RNF y las HUs/TTH/CAs/CTs que los originan, para permitir auditoría académica y mantenimiento futuro.
- **Glosario** de términos del producto referenciados en RF y RNF.
- **Política de mantenimiento** del documento ante evolución del backlog y ante mediciones reales del sistema integrado (calibración futura conforme a D-005).

**Fuera del alcance del documento:**

- **Ceremonia formal de priorización MoSCoW.** Este documento declara una prioridad MoSCoW *sugerida* para cada RF y cada RNF como anclaje argumentado de la ceremonia futura, pero no la ratifica. La ceremonia es sesión dedicada posterior.
- **Ceremonia de estimación Planning Poker.** Las estimaciones de esfuerzo no son contenido de este documento.
- **Validación cuantitativa de los umbrales.** Los umbrales declarados son los esperados al diseño, no los medidos. La calibración con mediciones reales se reporta conforme a D-005 en sesión posterior dedicada.
- **Diseño de implementación de cada RF/RNF.** Las decisiones técnicas de cómo materializar cada requisito viven en `DECISIONS.md` (D-001 a D-009) y en las TTH; este documento describe el *qué* y la *calidad*, no el *cómo*.
- **Reescritura de las HUs.** Los CAs de las 21 HUs preservan su redacción literal con umbrales hardcoded. La política aditiva está declarada en DHU-019 subsección G.
- **Reescritura de las TTH.** Las 11 TTH preservan sus criterios técnicos de terminado. Este documento referencia las CTs como origen de RNFs sin pedir cambios a las TTH.

### 0.3 Distinción entre RF y RNF-FUN como nota orientadora al lector

La palabra "funcional" aparece tanto en "Requisito Funcional (RF)" como en "Functional Suitability (RNF-FUN)" y eso induce confusión recurrente. La distinción es real y se preserva con rigor en este documento:

| Concepto | Pregunta que responde | Ejemplo en CerebroVial |
|---|---|---|
| **Requisito Funcional (RF)** | ¿Qué hace el sistema? ¿Qué comportamiento o servicio ofrece? | *RF-NNN — Consulta de KPIs operativos sobre periodo seleccionable. El sistema calcula y presenta cuatro KPIs (tiempo de espera, longitud de cola, throughput, demora) agregados sobre un periodo elegido por el Gerente.* |
| **Functional Suitability (RNF-FUN)** | El comportamiento que ofrece, ¿lo hace con la corrección, completitud y apropiación necesarias? | *RNF-FUN-NN — Manejo de periodos sin datos. Cuando el periodo seleccionado no contiene datos persistidos, el sistema comunica explícitamente "no hay datos en el periodo seleccionado" en lugar de presentar los KPIs calculados sobre cero filas.* |

El RF declara que el sistema **hace** algo. La Functional Suitability es la **calidad** con que lo hace bien. Las tres subcaracterísticas de Functional Suitability (Functional Completeness, Functional Correctness, Functional Appropriateness) son evaluaciones de calidad sobre el catálogo de RFs; no son RFs adicionales.

El prefijo del código (`RF-` vs `RNF-FUN-`) distingue inequívocamente uno de otro en cualquier referencia cruzada de este documento o de las HUs.

### 0.4 Convenciones del documento

**Codificación de identificadores:**

- **Requisitos Funcionales:** `RF-NNN` donde `NNN` es un correlativo de tres dígitos (RF-001 a RF-999). El correlativo es secuencial por familia funcional pero no garantiza contigüidad entre familias.
- **Requisitos No Funcionales:** `RNF-XXX-NN` donde `XXX` es el código de tres letras de la característica ISO 25010:2023 (FUN, PERF, COM, INT, REL, SEC, MNT, FLX, SAF) y `NN` es un correlativo de dos dígitos dentro de la característica.

**Tablas de cada RF/RNF:** todos siguen la plantilla unificada declarada en DHU-019 subsección D. Cuando un campo no aplica, se declara explícitamente "no aplica" en lugar de omitirse, para preservar la legibilidad estructural.

**Referencias cruzadas:**

- Referencia a una HU: `HU-XX` (por ejemplo `HU-02`).
- Referencia a un CA: `CA-XX.N` (por ejemplo `CA-02.4`).
- Referencia a una TTH: `TTH-XX` (por ejemplo `TTH-04`).
- Referencia a un criterio técnico de TTH: `CT-XX.N` (por ejemplo `CT-04.5`).
- Referencia a una decisión metodológica del backlog: `DHU-XXX` (por ejemplo `DHU-005`).
- Referencia a una decisión técnica del producto: `D-XXX` (por ejemplo `D-009`).
- Referencia a un Objetivo del Producto: por número del 1 al 4 según `BACKLOG_OVERVIEW.md`.
- Referencia entre RFs/RNFs internamente: por código directo (`RF-003`, `RNF-PERF-01`).

**Vocabulario normativo:**

Los RNF se redactan con vocabulario normativo: *"el sistema debe..."*, *"el sistema garantiza..."*, *"no se admite..."*. Cuando un RNF declara un umbral sugerido sujeto a calibración posterior, se usa *"el sistema debería..."* o *"criterio sugerido"*, conforme al vocabulario que ya emplean las HUs origen para esos umbrales no cerrados.

**Vocabulario agnóstico:**

Conforme a DHU-006 (HUs agnósticas a implementación), este documento no menciona tecnologías concretas, frameworks, librerías ni nombres de algoritmos específicos del motor adaptativo o del modelo predictivo. Cuando un RNF deriva de una decisión técnica que sí nombra tecnología (por ejemplo, D-006 menciona GRU; D-009 menciona el jam level de Waze), la referencia se hace por código (`según D-006`, `escala 0-5 según D-009`) sin reproducir la tecnología en el cuerpo del RNF.

### 0.5 Documentos relacionados

Este documento opera dentro de un sistema documental cuya arquitectura completa se describe en `BACKLOG_OVERVIEW.md`. Las relaciones más relevantes son:

- **`BACKLOG_OVERVIEW.md`** — Vista de conjunto del Product Backlog. Punto de entrada para entender las 4 Personas, los 4 Objetivos del Producto y el mapa de las 21 HUs + 11 TTH.
- **`DECISIONS_HU.md`** — Decisiones metodológicas del backlog (DHU-001 a DHU-022). **DHU-007** declara el origen de este documento; **DHU-019** consolida las decisiones de redacción que lo orientan; **DHU-005** declara el principio de robustez ante interrupción consolidado en este documento como RNF transversal.
- **`DECISIONS.md`** — Decisiones técnicas del producto (D-001 a D-009). Consumidas por este documento sin reabrirse.
- **`HU_BLOQUE_A.md`, `HU_BLOQUE_B.md`, `HU_BLOQUE_C.md`, `HU_BLOQUE_D.md`, `HU_BLOQUE_F.md`, `HU_MVP2.md`** — Cuerpo de las 21 HUs operativas con sus CAs y secciones "Candidatos a RNF". Origen principal de los RF y RNF de este documento.
- **`HU_BLOQUE_E.md`** — Bloque E del backlog (0 HUs operativas, mapeo a TTH-07 a TTH-11). Las TTH se redactaron en `TAREAS_TECNICAS_HABILITADORAS.md`.
- **`TAREAS_TECNICAS_HABILITADORAS.md`** — Las 11 TTH con sus criterios técnicos de terminado (CTs). Origen secundario de RNFs cuando los CTs exhiben naturaleza no funcional (por ejemplo, CT-09.7 — objetivo aspiracional accuracy ≥ 80%).
- **`HU_LITE.md`** — Versión corta de las 21 HUs. No es origen de este documento pero útil como lectura paralela.
- **`FEATURE_BACKLOG_DETALLADO.md`** — Origen de las features que se mapean a HUs y TTH. Consumido tangencialmente para entender el origen de cada HU.
- **`LEAN_INCEPTION_CEREBROVIAL.md`** — Personas, Journeys y MVP Canvas del proyecto. Insumo para identificar Personas beneficiarias y Objetivos del Producto referenciados en cada RF.
- **`EVOLUCION_TESIS.md`** — Narrativa de las 4 fases del proyecto académico. Contextualiza algunas decisiones técnicas referenciadas en RNFs (por ejemplo, el rol del modelo de respaldo preservado en TTH-09 declarado en Fase 2).
- **`CONTROL.md`** — Sustentación teórica del motor adaptativo (consumido por TTH-10). Referenciado tangencialmente en RNFs derivados de TTH-10.

**ISO/IEC 25010:2023** — Norma internacional adoptada como taxonomía única de clasificación de RNF, conforme a DHU-019 subsección A.

---

## 1. Marco de referencia

### 1.1 ISO/IEC 25010:2023 — Las 9 características de calidad adoptadas

Este documento clasifica todos los Requisitos No Funcionales según la **norma ISO/IEC 25010:2023 — Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model**, vigente desde 2023-11-01. La elección de versión sobre alternativas (ISO 9126 obsoleta, ISO 25010:2011 superseded) está justificada en DHU-019 subsección A.

ISO 25010:2023 establece **9 características de calidad de producto**, refinando la versión 2011 con dos renombramientos sustantivos (Usability → **Interaction Capability**; Portability → **Flexibility**) y la adición de **Safety** como 9ª característica de primer nivel. Cada característica se descompone en subcaracterísticas operacionales.

Las 9 características adoptadas y sus códigos en este documento:

| # | Característica | Código | Subcaracterísticas principales | Significado en CerebroVial |
|---|---|---|---|---|
| 1 | **Functional Suitability** | RNF-FUN | Functional completeness, functional correctness, functional appropriateness | ¿El catálogo de funciones cubre las tareas de las 3 Personas? ¿Los resultados producidos son correctos incluso en casos límite (datos faltantes, periodos sin cobertura, casos degenerados)? ¿Las funciones facilitan la tarea sin pasos innecesarios? |
| 2 | **Performance Efficiency** | RNF-PERF | Time behaviour, resource utilization, capacity | ¿El sistema responde dentro de los tiempos esperados? ¿Usa recursos razonablemente (paralelización del cálculo, granularidad de persistencia)? ¿Soporta los volúmenes esperados de datos persistidos? |
| 3 | **Compatibility** | RNF-COM | Co-existence, interoperability | ¿Los componentes del sistema co-existen sin interferir entre sí? ¿La integración con fuentes externas (TraCI con SUMO en validación cuantitativa; fuentes de datos reales en operación hipotética) cumple contratos estables? |
| 4 | **Interaction Capability** | RNF-INT | Appropriateness recognizability, learnability, operability, user error protection, UI aesthetics, accessibility, self-descriptiveness | ¿Las Personas reconocen qué hace cada vista sin entrenamiento? ¿Aprenden a usarla en tiempo razonable? ¿Operan sin cometer errores tipográficos o de navegación? ¿Las vistas son legibles, accesibles según WCAG 2.1 nivel AA, y autoexplicativas con tooltips integrados? |
| 5 | **Reliability** | RNF-REL | Faultlessness, availability, fault tolerance, recoverability | ¿El sistema opera correctamente sin fallos? ¿Está disponible cuando se requiere? ¿Tolera fallos de componentes sin perder función operativa (DHU-005, TTH-04 con fallbacks en cascada)? ¿Se recupera de fallos sin pérdida de datos persistidos? |
| 6 | **Security** | RNF-SEC | Confidentiality, integrity, non-repudiation, accountability, authenticity, resistance | ¿La información sensible está protegida (RBAC, autenticación JWT)? ¿Los registros append-only son inmutables tras la escritura? ¿Los logs de auditoría permiten reconstruir quién hizo qué y cuándo? ¿Los endpoints rechazan accesos no autorizados (HTTP 403)? |
| 7 | **Maintainability** | RNF-MNT | Modularity, reusability, analysability, modifiability, testability | ¿La configuración del sistema se modifica sin redeploy? ¿Los catálogos de plantillas se extienden sin tocar código? ¿La cobertura de tests permite auditar el comportamiento? ¿Los módulos del monolito modular (D-001) son analizables independientemente? |
| 8 | **Flexibility** | RNF-FLX | Adaptability, scalability, installability, replaceability | ¿El sistema se despliega en máquina limpia con un solo comando (D-003)? ¿La arquitectura permite escalar a otras intersecciones sin rediseño profundo? ¿Los componentes son reemplazables (modelo principal vs respaldo)? |
| 9 | **Safety** | RNF-SAF | Operational constraint, risk identification, fail safe, hazard warning, safe integration | ¿El sistema falla hacia un estado seguro definido (tiempos preconfigurados de TTH-05 en degradado nivel 3, valores por defecto seguros de CT-05.5 y CA-15.4)? ¿Las restricciones operacionales normativas del MTC peruano se respetan (CT-10.6)? ¿Las alertas comunican adecuadamente situaciones que requieren intervención humana? |

**Nota terminológica importante: Functional Suitability (RNF-FUN) no es sinónimo de Requisito Funcional (RF).**

Esta distinción es la fuente más frecuente de confusión al usar ISO 25010. La distinción se introdujo en la sección 0.3 de este preámbulo y se preserva con rigor en todo el documento. El prefijo del código (`RF-` vs `RNF-FUN-`) distingue inequívocamente uno de otro en toda referencia cruzada.

### 1.2 Derivación de los RF desde las HUs

Los Requisitos Funcionales de este documento se derivan de los 130 Criterios de Aceptación aproximados de las 21 HUs operativas mediante **composición transversal**, no biyección. Esta política está cerrada en DHU-019 subsección E y se resume aquí en tres principios operativos.

**Principio 1 — Un RF puede agrupar CAs de varias HUs cuando describen el mismo comportamiento del sistema.**

Un RF declara qué hace el sistema, independientemente de qué vista lo expone. Cuando varias HUs exhiben el mismo comportamiento desde perspectivas distintas (típicamente una HU del Operador y la HU del Administrador que consume el mismo sustrato técnico, o tres vistas del Operador que muestran las mismas variables desde ángulos distintos), el RF subyacente es uno solo. La presentación específica de cada vista pertenece a la HU, no al RF.

Ejemplo de composición:

> CA-02.1 + CA-03.1 + CA-04.1 → **RF-NNN — Presentación de variables de estado de la intersección por acceso**. El sistema expone, por cada acceso de la intersección, las variables observadas del estado del tráfico (flujo, longitud de cola, nivel de congestión) y las predicciones del nivel de congestión hasta el horizonte configurado.
>
> Los tres CAs alimentan un solo RF porque describen el mismo comportamiento desde tres vistas del Operador (HU-02 monitoreo, HU-03 predicción, HU-04 vista combinada). La presentación específica del dashboard es de las HUs.

**Principio 2 — Control de acceso como dos RFs transversales únicos, no como RFs por HU.**

Las 21 HUs incluyen un CA de redirección al login y un CA de RBAC para roles no autorizados. Estos CAs no producen 42 RFs separados; producen dos RFs transversales:

- **RF-001 — Autenticación al sistema** (consume TTH-01; precondición de cualquier HU operativa).
- **RF-002 — Control de acceso por rol** (deriva de HU-01; aplicable a las 21 HUs con tabla de aplicabilidad).

**Principio 3 — Comportamiento de robustez ante interrupción no produce RFs separados.**

Los CAs de robustez (típicamente CA-XX.4 según DHU-005 Caso A o B) no se vuelven RFs porque la robustez es por naturaleza no funcional. Cada CA de robustez se referencia desde el RF correspondiente (el que captura el comportamiento normal de la vista) y se materializa en el RNF transversal de robustez ante interrupción (RNF-REL-NN consolidado conforme a DHU-019 subsección C.2).

Ejemplo:

> RF-NNN (presentación de variables por acceso) tiene como CAs origen CA-02.1, CA-03.1, CA-04.1 (presentación). Los CAs CA-02.4, CA-03.4, CA-04.4 (robustez) NO entran como CAs origen del RF; entran como aplicabilidad del **RNF-REL-NN — Robustez ante interrupción de fuente**.

**Estimación del catálogo final:** entre 25 y 35 RFs. La cifra exacta se confirma al cerrar la sección 2; no se preestablece para no forzar la consolidación.

### 1.3 Clasificación de los RNF según ISO 25010:2023

Los Requisitos No Funcionales se derivan de tres fuentes y se clasifican mediante una tabla de reasignación cerrada en DHU-019 subsección B.

**Fuentes de los RNF:**

1. **Secciones "Candidatos a RNF" de las 21 HUs operativas.** Cada HU declara entre 3 y 11 candidatos. La consolidación elimina duplicaciones y aplica las decisiones de DHU-019 subsección C (siete inconsistencias detectadas durante la lectura).

2. **Criterios técnicos de terminado (CTs) de las 11 TTH cuando exhiben naturaleza no funcional.** Por ejemplo, CT-09.7 (objetivo aspiracional accuracy ≥ 80%) es un RNF de Functional Correctness con origen en TTH-09; CT-01.2 (bcrypt cost factor ≥ 12) es un RNF de Confidentiality con origen en TTH-01.

3. **Decisiones técnicas D-001 a D-009 cuando declaran propiedades de calidad del sistema.** Por ejemplo, D-003 (deploy local con Docker) origina RNFs de Flexibility / Installability; D-009 (escala 0-5 del nivel de congestión) origina RNFs de Compatibility / Interoperability ante fuentes intercambiables sin reentrenar el modelo.

**Tabla de reasignación de las categorías heterogéneas de DHU-007 a ISO 25010:2023:**

Esta tabla es la referencia normativa para clasificar cada candidato a RNF de las HUs en la característica ISO 25010 correspondiente. Está cerrada en DHU-019 subsección B; aquí se reproduce con las asignaciones más frecuentes para facilitar la lectura del documento.

| Categoría declarada en HUs | Característica ISO 25010:2023 | Código |
|---|---|---|
| Rendimiento, latencia, paralelización | Performance Efficiency | RNF-PERF |
| Robustez, disponibilidad, continuidad operativa, tolerancia a fallos, resiliencia de persistencia | Reliability | RNF-REL |
| Persistencia / durabilidad (datos), retención | Reliability (Recoverability) | RNF-REL |
| Persistencia / durabilidad (cuando aplica seguridad operacional, fail safe) | Safety | RNF-SAF |
| Auditoría, auditabilidad, inmutabilidad de logs, trazabilidad, no-repudiación | Security (Accountability, Integrity, Non-repudiation) | RNF-SEC |
| Seguridad de acceso (RBAC), privacidad, confidencialidad | Security (Authenticity, Confidentiality, Resistance) | RNF-SEC |
| Usabilidad, accesibilidad, coherencia visual entre vistas, identificabilidad | Interaction Capability | RNF-INT |
| Mantenibilidad, configurabilidad, extensibilidad de catálogos, tolerancia parametrizada | Maintainability | RNF-MNT |
| Manejabilidad de datos faltantes, manejabilidad de concurrencia, comparabilidad rigurosa, calidad de predicción, cobertura de catálogos, independencia entre dimensiones | Functional Suitability | RNF-FUN |
| Co-existencia entre componentes, interoperabilidad con fuentes externas | Compatibility | RNF-COM |
| Portabilidad de deploy, escalamiento a múltiples intersecciones, reemplazabilidad de modelo | Flexibility | RNF-FLX |
| Comportamiento ante condiciones que pueden afectar conductores o peatones, fail safe, restricciones normativas del MTC, valores por defecto seguros | Safety | RNF-SAF |

Los casos de doble característica primaria + secundaria están documentados en DHU-019 subsección B con justificación detallada. En este documento, cada RNF lleva un código primario en su identificador `RNF-XXX-NN` y declara la característica secundaria como referencia cruzada en el campo "Característica ISO".

### 1.4 Trazabilidad bidireccional HU/TTH ↔ RF ↔ RNF

Cada RF y cada RNF de este documento incluye en su plantilla los campos *"HUs origen / CAs origen / TTH relacionadas"* o *"HUs/TTH origen"*. Esto permite recorrer el documento en ambas direcciones:

**Dirección HU/TTH → RF/RNF:** desde una HU, identificar qué RFs cubren su comportamiento y qué RNFs aplican a sus CAs.

**Dirección RF/RNF → HU/TTH:** desde un RF/RNF, identificar de qué HUs/CAs/TTH/CTs se derivó y qué Personas y Objetivos del Producto lo motivan.

La **sección 4** de este documento materializa esta trazabilidad como matriz consolidada para consulta agregada. Adicionalmente, conforme a DHU-019 subsección G, las secciones "Candidatos a RNF" de las 21 HUs recibirán una pasada aditiva tras cerrar este documento, agregando una referencia `→ RNF-XXX-NN` por candidato declarado, completando el ciclo de trazabilidad sin modificar el contenido sustantivo de las HUs.

---

## 3. Catálogo de Requisitos No Funcionales

Los RNF se presentan agrupados por característica de calidad de ISO/IEC 25010:2023, en el orden acordado en la sesión de redacción: RNF-PERF, RNF-REL, RNF-SEC, RNF-FUN, RNF-INT, RNF-MNT, RNF-SAF, RNF-COM, RNF-FLX. Cada RNF sigue la plantilla unificada cerrada en DHU-019 subsección D.

### 3.2 Performance Efficiency (RNF-PERF)

La característica de Performance Efficiency cubre el comportamiento del sistema en términos de tiempo de respuesta, utilización de recursos y capacidad bajo las condiciones operativas esperadas. En CerebroVial agrupa los umbrales temporales declarados en los CAs de las 21 HUs, distinguiendo tres familias de tiempo: actualización en tiempo real de vistas operativas, apertura y recálculo de vistas de consulta, y generación de artefactos exportables.

#### RNF-PERF-01 — Actualización en tiempo real de la presentación operativa

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-01 |
| Característica ISO 25010:2023 | Performance Efficiency / Time behaviour |
| Subcaracterística | Time behaviour |
| Descripción normativa | El sistema debe propagar a la presentación operativa cualquier cambio de estado en su fuente correspondiente dentro de un tiempo máximo de 5 segundos, medido desde el instante en que el evento generador se produce hasta el instante en que el dato actualizado es visible en la vista del usuario, sin requerir recarga manual de la página. Este umbral aplica a las presentaciones de tiempo real del Operador y a las vistas técnicas del Administrador que reflejan estado del sistema viviente. |
| Criterio de aceptación medible | Latencia end-to-end (fuente → vista) ≤ 5 s en operación normal sobre los escenarios típicos de validación cuantitativa del sistema integrado. La medición se realiza desde la generación del dato hasta su renderizado en el cliente. |
| Método de validación | Prueba automatizada de latencia end-to-end midiendo el delta entre la marca de tiempo de generación del evento en la fuente y la marca de tiempo de renderizado en la vista. |
| HUs/TTH origen | HU-02 (CA-02.2), HU-03 (CA-03.2), HU-04 (CA-04.3), HU-05 (CA-05.3), HU-06 (CA-06.2), HU-07 (CA-07.1 implícito), HU-10 (banner transversal), HU-11 (CA-11.2), HU-12 (CA-12.2), HU-13 (CA-13.2). |
| DHUs relacionadas | DHU-019 subsección C.1 (consolidación del umbral ≤ 5 s a partir de diez declaraciones equivalentes en los Candidatos a RNF). |
| Prioridad MoSCoW sugerida | Must (realiza el Objetivo 1 del Producto — reducir tiempos de espera — al garantizar que el Operador toma decisiones sobre datos vigentes; realiza el Objetivo 3 — continuidad operativa — al garantizar que las alertas de degradación llegan al Operador en tiempo útil). |
| Aplicabilidad | Diez HUs operativas con actualización automática en tiempo real: HU-02 (monitoreo del estado actual), HU-03 (predicción de congestión), HU-04 (vista combinada), HU-05 (estrategia activa), HU-06 (explicación de selección), HU-07 (notificación de cambio de estrategia), HU-10 (alerta transversal del estado operativo), HU-11 (vista de componentes del Operador), HU-12 (explicación del modo degradado), HU-13 (vista técnica de componentes del Administrador). |
| Excepciones | No aplica a vistas de consulta que no operan en tiempo real (HU-08, HU-14, HU-15, HU-16, HU-17, HU-18, HU-19, HU-20, HU-21), las cuales tienen sus propios umbrales declarados en RNF-PERF-02 a RNF-PERF-07. |
| Notas | El umbral consolida en una sola declaración normativa los umbrales que en las HUs aparecen como CAs individuales con redacciones equivalentes ("con una latencia máxima de 5 segundos desde que la medición se genera", "con una latencia máxima de 5 segundos desde que el cambio se produce", etcétera). La consolidación está justificada en DHU-019 subsección C.1 y no altera los CAs de las HUs (preservación literal según DHU-019 subsección G). El umbral es esperado al diseño; la calibración con mediciones reales se reporta conforme a D-005. La validación cuantitativa puede revelar que la latencia real es menor en operación normal o mayor bajo carga; ambos escenarios se documentan en la sesión de validación, no se preasumen aquí. |

#### RNF-PERF-02 — Apertura de vistas de consulta del Operador

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-02 |
| Característica ISO 25010:2023 | Performance Efficiency / Time behaviour |
| Subcaracterística | Time behaviour |
| Descripción normativa | Las vistas de consulta del Operador que presentan información persistida (no de tiempo real) deben abrirse y mostrar su primer estado utilizable en un tiempo máximo de 2 segundos sobre los volúmenes de datos esperados al cierre del MVP1, medido desde la solicitud del usuario hasta la presentación inicial completa de la primera página o sección visible. |
| Criterio de aceptación medible | Tiempo de respuesta para abrir la vista ≤ 2 s con volúmenes esperados de MVP1 (típicamente: registros acumulados de una jornada operativa simulada con granularidad de cierre, o equivalente). |
| Método de validación | Prueba automatizada de tiempo de respuesta sobre dataset de volumen típico esperado al cierre del MVP1, midiendo el delta entre la solicitud y la presentación inicial. |
| HUs/TTH origen | HU-08 (CA-08.x — consulta del historial de decisiones del motor), HU-09 (CA-09.x — listado paginado de notas). |
| DHUs relacionadas | No aplica de forma específica. |
| Prioridad MoSCoW sugerida | Should para HU-08 (MVP1); Could para HU-09 (MVP2). |
| Aplicabilidad | Dos HUs de consulta del Operador: HU-08 (historial de decisiones) y HU-09 (notas e incidencias del turno, MVP2). |
| Excepciones | No aplica a las vistas de tiempo real del Operador (cubiertas por RNF-PERF-01) ni a las vistas del Gerente (cubiertas por RNF-PERF-04 y RNF-PERF-05). |
| Notas | El umbral aplica al primer renderizado utilizable; operaciones posteriores de paginación, filtrado o búsqueda dentro de la vista pueden tener umbrales distintos no declarados explícitamente en MVP1 (se cierran al implementar). |

#### RNF-PERF-03 — Apertura y operación de la vista de configuración del Administrador

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-03 |
| Característica ISO 25010:2023 | Performance Efficiency / Time behaviour |
| Subcaracterística | Time behaviour |
| Descripción normativa | La vista de configuración del Administrador debe abrirse y mostrar todos los parámetros operativos junto con el historial reciente de modificaciones en un tiempo máximo de 2 segundos, medido desde la solicitud del Administrador hasta el renderizado completo de la vista lista para edición. |
| Criterio de aceptación medible | Tiempo de respuesta para abrir la vista con todos los parámetros y el historial reciente ≤ 2 s. |
| Método de validación | Prueba automatizada de tiempo de respuesta sobre escenario de configuración típica con historial de auditoría representativo. |
| HUs/TTH origen | HU-15 (Candidato a RNF: "consulta de la vista razonable para operación de configuración, ≤ 2 segundos"). |
| DHUs relacionadas | DHU-013 (clasificación de F20 como HU del Administrador con sustrato técnico inglobado), DHU-014 (parámetros concretos cubiertos en MVP1). |
| Prioridad MoSCoW sugerida | Should (operación de configuración del Administrador, no es de tiempo real). |
| Aplicabilidad | HU-15 (configuración de parámetros operativos del sistema). |
| Excepciones | No aplica a las operaciones de modificación de parámetros, las cuales se rigen por RNF-PERF-08 (efecto sin redeploy). |
| Notas | Aunque la vista de configuración es de carácter no tiempo real, el umbral de 2 s preserva una experiencia operativa fluida para el Administrador cuando ajusta parámetros tras detectar una condición que requiere intervención. |

#### RNF-PERF-04 — Apertura de vistas de consulta del Gerente

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-04 |
| Característica ISO 25010:2023 | Performance Efficiency / Time behaviour |
| Subcaracterística | Time behaviour |
| Descripción normativa | Las vistas de consulta del Gerente que presentan indicadores agregados sobre periodos seleccionables deben abrirse y mostrar los indicadores del periodo por defecto en un tiempo máximo de 3 segundos, medido desde el ingreso a la vista hasta el renderizado completo de las cards y gráficos iniciales. |
| Criterio de aceptación medible | Tiempo de apertura de la vista ≤ 3 s con el periodo por defecto ("Esta semana" según CA-16.4). |
| Método de validación | Prueba automatizada de tiempo de respuesta sobre dataset histórico representativo de una semana operativa. |
| HUs/TTH origen | HU-16 (Candidato a RNF: "apertura ≤ 3 segundos"), HU-17 (Candidato a RNF equivalente, aprovechando paralelización de las dos consultas al histórico). |
| DHUs relacionadas | DHU-016 subsección I (composición F12 + F13 en HU-16; HU-17 reutiliza el selector). |
| Prioridad MoSCoW sugerida | Must (realiza el Objetivo 4 del Producto — evidencia gerencial — al permitir consultas fluidas para reportería ejecutiva). |
| Aplicabilidad | HU-16 (consulta de KPIs sobre periodo seleccionable) y HU-17 (vista comparativa entre periodos). |
| Excepciones | El recálculo al cambiar de periodo se rige por RNF-PERF-05 (no por este RNF). |
| Notas | Para HU-17, el umbral se preserva aprovechando que las dos consultas al histórico son paralelizables (una al periodo actual y otra al periodo previo equivalente), conforme a la nota técnica de HU-17 sobre paralelización del cálculo. La degradación bajo carga real se reporta conforme a D-005. |

#### RNF-PERF-05 — Recálculo de vistas del Gerente al cambiar de periodo

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-05 |
| Característica ISO 25010:2023 | Performance Efficiency / Time behaviour |
| Subcaracterística | Time behaviour |
| Descripción normativa | Cuando el Gerente cambia el periodo seleccionado en sus vistas de consulta, el sistema debe completar el recálculo de los indicadores agregados y refrescar la vista en un tiempo máximo de 10 segundos, mostrando un indicador de carga visible durante la operación para que el Gerente sepa que el sistema está trabajando. |
| Criterio de aceptación medible | Tiempo de recálculo al cambiar de periodo ≤ 10 s, con indicador de carga visible durante toda la operación. |
| Método de validación | Prueba automatizada de tiempo de respuesta sobre transiciones entre periodos representativos (semana → mes, mes → trimestre cuando aplique, etcétera). |
| HUs/TTH origen | HU-16 (Candidato a RNF: "recálculo ≤ 10 segundos"), HU-17 (Candidato a RNF equivalente). |
| DHUs relacionadas | DHU-016 subsección I, DHU-016 subsección C (granularidad del histórico). |
| Prioridad MoSCoW sugerida | Must (continuidad de la experiencia del Gerente al explorar distintos periodos). |
| Aplicabilidad | HU-16, HU-17. |
| Excepciones | Para periodos personalizados de duración inusual (rango personalizado largo, cubierto por HU-19 en exportación), el umbral aplica también pero con tolerancia documentada caso a caso al implementar. |
| Notas | El indicador de carga visible es parte integral del cumplimiento: una operación que tarda menos de 10 s sin indicador puede aún percibirse como lenta, mientras que una que tarda 10 s con indicador comunica progreso al Gerente. |

#### RNF-PERF-06 — Apertura y zoom de la vista detallada de drill-down

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-06 |
| Característica ISO 25010:2023 | Performance Efficiency / Time behaviour |
| Subcaracterística | Time behaviour |
| Descripción normativa | La vista detallada de drill-down del Gerente debe abrirse en un tiempo máximo de 5 segundos para periodos cortos (semana o mes típicos) y 15 segundos para periodos largos con volumen alto de registros, paralelizando la carga de los tres carriles temporales que consumen fuentes independientes. La operación de zoom interactivo sobre cualquier sub-intervalo del periodo activo debe responder en un tiempo máximo de 3 segundos, con indicador de carga visible. |
| Criterio de aceptación medible | Apertura ≤ 5 s para periodos cortos; apertura ≤ 15 s para periodos largos con volumen alto; zoom ≤ 3 s sobre cualquier sub-intervalo, con indicador de carga visible. |
| Método de validación | Prueba automatizada de tiempos de respuesta sobre escenarios de drill-down con tres carriles cargados, midiendo tanto apertura como operaciones de zoom interactivo. |
| HUs/TTH origen | HU-18 (Candidato a RNF de apertura y de zoom). |
| DHUs relacionadas | DHU-017 subsección A (clasificación de F15 como HU del Gerente), DHU-017 subsección J (DHU-005 Caso B aplicado independientemente por carril). |
| Prioridad MoSCoW sugerida | Could (HU-18 es MVP2 — candidata a construcción condicional). |
| Aplicabilidad | HU-18 (drill-down del Gerente sobre periodo específico). |
| Excepciones | Cuando uno o más carriles están degradados (DHU-005 Caso B), los umbrales no aplican a los carriles afectados; sí aplican a los carriles operativos. |
| Notas | La paralelización de las tres consultas al abrir la vista no es opcional: es parte del cumplimiento del umbral, declarada como tal en el Candidato a RNF de paralelización del cálculo de HU-18. Sin paralelización, las tres consultas secuenciales superarían el umbral. La degradación bajo carga real se reporta conforme a D-005. |

#### RNF-PERF-07 — Generación de reportes exportables

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-07 |
| Característica ISO 25010:2023 | Performance Efficiency / Time behaviour |
| Subcaracterística | Time behaviour |
| Descripción normativa | La generación de reportes exportables del Gerente debe completarse dentro de los siguientes umbrales según formato y volumen del periodo: para PDF, ≤ 15 segundos en periodos cortos (semana o mes con granularidad típica) y ≤ 60 segundos en rango personalizado largo con volumen alto de datos; para Excel, ≤ 10 segundos en periodos típicos y ≤ 30 segundos en rango personalizado largo. El umbral del Excel es más estricto que el del PDF equivalente porque Excel no requiere renderizado de gráficos. |
| Criterio de aceptación medible | PDF: ≤ 15 s (periodo corto), ≤ 60 s (periodo largo); Excel: ≤ 10 s (periodo típico), ≤ 30 s (periodo largo). |
| Método de validación | Prueba automatizada de tiempo de generación sobre escenarios de exportación con periodos representativos, midiendo desde la solicitud hasta la disponibilidad del archivo descargable. |
| HUs/TTH origen | HU-19 (Candidatos a RNF de generación PDF y Excel). |
| DHUs relacionadas | DHU-017 subsección C (composición de F16 como HU única). |
| Prioridad MoSCoW sugerida | Could (HU-19 es MVP2). |
| Aplicabilidad | HU-19 (exportación de reportes a PDF/Excel del Gerente). |
| Excepciones | Cuando la fuente que alimenta las vistas está caída (DHU-005 Caso B aplicado con política conservadora), la exportación se rechaza en lugar de completarse fuera del umbral; el umbral aplica únicamente a operación normal. Si la realidad medida es peor en MVP2, se reporta conforme a D-005. |
| Notas | El umbral del PDF es menos estricto porque incluye renderizado de gráficos comparativos (semántica visual de mejora/empeoramiento, dos series superpuestas). El umbral del Excel es más estricto porque exporta datos crudos sin gráficos. |

#### RNF-PERF-08 — Efecto de modificaciones de configuración sin redeploy

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-08 |
| Característica ISO 25010:2023 | Performance Efficiency / Time behaviour |
| Subcaracterística | Time behaviour |
| Descripción normativa | Cuando el Administrador modifica un parámetro operativo del sistema y confirma la modificación, el cambio debe surtir efecto en los componentes consumidores del parámetro en un tiempo máximo de 30 segundos desde la confirmación, sin requerir reinicio del sistema ni redespliegue. Los componentes consumidores deben consumir el nuevo valor en su siguiente ciclo de operación. |
| Criterio de aceptación medible | Tiempo entre confirmación de modificación y efecto en componentes consumidores ≤ 30 s, sin reinicio del sistema. |
| Método de validación | Prueba automatizada midiendo el delta entre la persistencia de un nuevo valor de parámetro y la primera operación de un componente consumidor con el nuevo valor. |
| HUs/TTH origen | HU-15 (CA-15.3 — efecto sin redeploy). |
| DHUs relacionadas | DHU-013, DHU-014 subsección C (selección concreta de parámetros). |
| Prioridad MoSCoW sugerida | Must (operación sin redeploy es valor declarado de HU-15). |
| Aplicabilidad | HU-15. Los parámetros configurables cubiertos en MVP1 son: umbrales de cola, horizonte de predicción, umbral de congestión, ventana de cálculo de métricas, frecuencia de evaluación de salud. |
| Notas | El umbral de 30 s es operativamente coherente con la frecuencia de evaluación de salud (default 5 s según CT-04.1) y con los ciclos de operación de los componentes consumidores. |

#### RNF-PERF-09 — Latencia del cálculo de métricas del modelo predictivo

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-09 |
| Característica ISO 25010:2023 | Performance Efficiency / Time behaviour |
| Subcaracterística | Time behaviour |
| Descripción normativa | El cálculo de las métricas de desempeño del modelo predictivo sobre la ventana temporal configurada no debe degradar la respuesta del resto del sistema. La latencia entre la generación de una predicción con horizonte vencido y su incorporación al cálculo de métricas no debe superar 30 segundos. El cálculo puede ser incremental o asíncrono; el método específico se cierra al implementar. |
| Criterio de aceptación medible | Latencia entre vencimiento del horizonte de una predicción y su reflejo en las métricas ≤ 30 s. El cálculo no debe degradar la respuesta de otros componentes del sistema (no causar incrementos perceptibles en la latencia de servicio de predicciones, RNF-PERF-01, ni en la latencia de servicio del motor). |
| Método de validación | Prueba automatizada de latencia entre evento de vencimiento de horizonte y aparición de la métrica actualizada; prueba de no degradación sobre los componentes que comparten infraestructura. |
| HUs/TTH origen | HU-14 (Candidato a RNF: "cálculo de métricas sobre la ventana temporal no degrada la respuesta del sistema"), HU-20 (CA-20.14 y Candidato a RNF: latencia entre cálculo y reflejo ≤ 30 s, coherente con HU-14). |
| DHUs relacionadas | DHU-013 (sustrato técnico de F18 inglobado en HU-14), DHU-014 subsección F (ventana temporal configurable en HU-15). |
| Prioridad MoSCoW sugerida | Should (HU-14 MVP1, HU-20 MVP2). |
| Aplicabilidad | HU-14 (vista de métricas del modelo principal), HU-20 (vista comparativa de métricas del modelo principal vs respaldo). |
| Notas | El umbral de 30 s reconoce que la consulta de métricas no es operación de tiempo real estricta sino de monitoreo gerencial-técnico. La validación incluye verificar no degradación cruzada con RNF-PERF-01. |

#### RNF-PERF-10 — Latencia de actualización del indicador de incidentes pendientes

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-10 |
| Característica ISO 25010:2023 | Performance Efficiency / Time behaviour |
| Subcaracterística | Time behaviour |
| Descripción normativa | Cuando un Operador registra un nuevo incidente escalado al Administrador, el badge numérico de incidentes pendientes visible en la navegación del Administrador debe actualizarse en un tiempo máximo de 30 segundos desde la persistencia del incidente. Este umbral aplica al carácter no-tiempo-real de la vista de gestión de incidentes. |
| Criterio de aceptación medible | Latencia entre persistencia del incidente y actualización del badge ≤ 30 s. |
| Método de validación | Prueba automatizada midiendo el delta entre el registro de un nuevo incidente y la actualización del badge en una sesión activa del Administrador. |
| HUs/TTH origen | HU-21 (CA-21.22 y Candidato a RNF de actualización del indicador de pendientes). |
| DHUs relacionadas | DHU-017 subsección E (composición de F28 con Operador protagonista y Administrador destinatario). |
| Prioridad MoSCoW sugerida | Could (HU-21 es MVP2). |
| Aplicabilidad | HU-21 (escalamiento de incidentes del Operador al Administrador). |
| Notas | El umbral de 30 s es deliberadamente más laxo que el de tiempo real (RNF-PERF-01) porque la gestión de incidentes no requiere reacción inmediata del Administrador; permite implementación con mecanismos de polling o eventos diferidos. |

#### RNF-PERF-11 — Granularidad de persistencia del histórico operacional

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-11 |
| Característica ISO 25010:2023 | Performance Efficiency / Capacity |
| Subcaracterística | Capacity |
| Descripción normativa | El histórico operacional que alimenta las vistas de consulta del Gerente debe persistir los estados observados a una granularidad de 30 segundos por intersección y por dirección. Esta granularidad equilibra la capacidad de detectar variaciones operacionalmente relevantes (cambios de patrón de tráfico en escalas de minutos) con el volumen de datos acumulados a sostener bajo los volúmenes esperados de MVP1. |
| Criterio de aceptación medible | Granularidad de persistencia = 30 segundos por intersección y dirección. Volumen de datos sostenible bajo carga de operación continua sin degradación de RNF-PERF-04 (apertura) ni de RNF-PERF-05 (recálculo). |
| Método de validación | Inspección del esquema de persistencia y prueba de carga sobre dataset acumulado de duración equivalente a operación continua de al menos una semana. |
| HUs/TTH origen | HU-16 (Candidato a RNF: "granularidad de 30 segundos por intersección y dirección, CA-16.2"). |
| DHUs relacionadas | DHU-016 subsección C (F30 inglobada en HU-16 con granularidad cerrada). |
| Prioridad MoSCoW sugerida | Must (parámetro de diseño del sustrato técnico). |
| Aplicabilidad | Persistencia del histórico operacional inglobada en HU-16; consumida por HU-17, HU-18 y HU-19. |
| Notas | La granularidad de 30 segundos es decisión de diseño cerrada al cierre del Bloque F; ajustes posteriores requieren reabrir DHU-016 subsección C. La validación con prueba de carga real puede revelar que la granularidad requiere ajuste; ese ajuste se reporta conforme a D-005 y se aplica en pasada futura. |

#### RNF-PERF-12 — Paralelización del cálculo en vistas multi-fuente

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-12 |
| Característica ISO 25010:2023 | Performance Efficiency / Resource utilization |
| Subcaracterística | Resource utilization |
| Descripción normativa | Las vistas que consumen múltiples fuentes de datos independientes deben ejecutar sus consultas en paralelo, no secuencialmente, para preservar los umbrales de apertura declarados en RNF-PERF-04 y RNF-PERF-06. La paralelización es responsabilidad de la implementación del backend y no requiere intervención del usuario. |
| Criterio de aceptación medible | Las consultas a fuentes independientes se inician simultáneamente; el tiempo de apertura total es aproximadamente el de la consulta más lenta, no la suma de todas. Validable por inspección de logs de tiempo de inicio de cada consulta. |
| Método de validación | Inspección de logs de tiempos de inicio y fin de cada consulta independiente durante una operación de apertura de vista multi-fuente. |
| HUs/TTH origen | HU-17 (nota técnica sobre paralelización de las dos consultas al histórico), HU-18 (Candidato a RNF de paralelización del cálculo sobre los tres carriles temporales). |
| DHUs relacionadas | DHU-016 subsección I, DHU-017 subsección A. |
| Prioridad MoSCoW sugerida | Must (preservación de RNF-PERF-04 y RNF-PERF-06). |
| Aplicabilidad | HU-17 (dos consultas: periodo actual y previo equivalente), HU-18 (tres carriles: tráfico, decisiones del motor, estado operativo). |
| Notas | Sin paralelización, las tres consultas secuenciales de HU-18 sumadas superarían el umbral de RNF-PERF-06. La paralelización es por tanto condición necesaria del cumplimiento de los umbrales de apertura, no una mejora opcional. |

#### RNF-PERF-13 — No degradación por ejecución paralela del modelo de respaldo

| Campo | Contenido |
|---|---|
| Identificador | RNF-PERF-13 |
| Característica ISO 25010:2023 | Performance Efficiency / Resource utilization |
| Subcaracterística | Resource utilization |
| Descripción normativa | La ejecución paralela continua del modelo predictivo de respaldo sobre los mismos inputs operativos que el modelo principal, requerida para alimentar la comparativa de la vista de HU-20, no debe degradar la latencia con la que el modelo principal sirve predicciones a sus consumidores en operación normal. El costo computacional adicional del modelo de respaldo es asumido por el sistema pero no debe impactar el camino crítico del predictor activo. |
| Criterio de aceptación medible | Latencia del modelo principal con modelo de respaldo ejecutándose en paralelo ≤ latencia del modelo principal sin ejecución paralela del respaldo, dentro de un margen razonable de medición (criterio sugerido: no más del 10% de incremento). |
| Método de validación | Prueba comparativa de latencia del modelo principal en dos escenarios: con y sin ejecución paralela del respaldo. |
| HUs/TTH origen | HU-20 (Candidato a RNF de ejecución paralela del modelo de respaldo, CA-20.1). |
| DHUs relacionadas | DHU-017 subsección D (registro paralelo del baseline persistido como extensión de CA-14.1 / CT-09.5). |
| Prioridad MoSCoW sugerida | Could (HU-20 es MVP2). |
| Aplicabilidad | HU-20 (vista comparativa de métricas del modelo principal vs respaldo). |
| Notas | El RNF protege la operación del sistema: aunque HU-20 sea MVP2 y aún no esté en producción, el sustrato técnico que la habilita (extensión del registro de predicciones para persistir también predicciones del modelo de respaldo) se planifica para no degradar el sistema en operación normal. |

### 3.5 Reliability (RNF-REL)

La característica de Reliability cubre la capacidad del sistema para mantener su nivel de servicio bajo las condiciones operativas esperadas y para recuperarse de fallos sin pérdida de información. En CerebroVial agrupa cuatro familias: el principio transversal de robustez ante interrupción de fuente (DHU-005), la disponibilidad de los componentes transversales (alerta del estado operativo, explicaciones), la durabilidad de los registros append-only del sistema y la resiliencia operativa ante fallos de subsistemas auxiliares.

#### RNF-REL-01 — Robustez ante interrupción de fuente

| Campo | Contenido |
|---|---|
| Identificador | RNF-REL-01 |
| Característica ISO 25010:2023 | Reliability / Fault tolerance |
| Subcaracterística | Fault tolerance |
| Descripción normativa | El sistema debe mantener su capacidad operativa frente a la interrupción temporal de cualquier fuente externa de medición o componente interno de decisión, sin presentar al usuario información obsoleta como si fuera vigente y sin asumir falsamente operación normal. Cuando la fuente correspondiente no actualiza un dato en el plazo esperado, la vista afectada debe mantener visible el último valor conocido, marcarlo con un indicador semántico explícito según el modo aplicable, e indicar el tiempo transcurrido desde la última actualización confirmada. Existen dos modos operativos del principio según la naturaleza de la fuente: el modo A para fuentes externas de medición del mundo observado, con marca semántica "desactualizado"; el modo B para componentes internos de decisión del sistema, con marca semántica "no confirmado". |
| Criterio de aceptación medible | Prueba operativa con escenarios de interrupción inducida sobre cada fuente declarada en la aplicabilidad: la vista correspondiente no debe mostrar el dato como vigente; debe mostrar la marca semántica correcta ("desactualizado" para fuentes de medición, "no confirmado" para componentes internos); debe indicar el tiempo transcurrido desde la última actualización; no debe presentar la vista vacía ni sustituir el valor por cero. La marca pasiva por panel afectado es responsabilidad de cada HU; la alerta activa transversal sobre el estado del sistema completo es responsabilidad de HU-10 conforme a DHU-009. |
| Método de validación | Prueba automatizada de interrupción inducida sobre cada fuente declarada en la aplicabilidad, verificando el comportamiento de la vista correspondiente. |
| HUs/TTH origen | Modo A (fuente externa de medición): HU-02 (CA-02.4), HU-04 (CA-04.4 parte de mediciones). Modo B (componente interno de decisión): HU-03 (CA-03.4), HU-04 (CA-04.4 parte de predicciones), HU-05 (CA-05.4), HU-06 (CA-06.4), HU-07 (CA-07.5), HU-10 (CA-10.9), HU-11 (CA-11.7), HU-12 (CA-12.5), HU-13 (CA-13.4), HU-14 (CA-14.12), HU-15 (CA-15.10), HU-16 (CA-16.19), HU-17 (CA-17.14), HU-18 (CA-18.17, CA-18.18, CA-18.19 — independencia por carril), HU-20 (CA-20.18), HU-21 (CA-21.9 — política conservadora de rechazo del disparo, CA-21.28 — marca pasiva en consulta). HU-08 (CA-08.5) y HU-09 (CA-09.5) aplican variantes específicas (operación no se detiene, resiliencia de persistencia). |
| DHUs relacionadas | DHU-005 (origen del principio con sus Casos A y B), DHU-008 (distinción entre componente caído, modo degradado y lógica de fallback), DHU-009 (relación entre marca pasiva del Bloque B y alerta activa del Bloque C), DHU-019 subsección C.2 (consolidación del RNF transversal). |
| Prioridad MoSCoW sugerida | Must (realiza el Objetivo 3 del Producto — garantizar continuidad operativa ante fallos de componentes). |
| Aplicabilidad | Veintiún CAs de robustez distribuidos en diecinueve HUs operativas. La aplicabilidad concreta de cada modo está documentada en la columna "HUs/TTH origen" arriba. La tabla de aplicabilidad operativa por escenario es responsabilidad de la prueba automatizada de validación. |
| Excepciones | **HU-18 — independencia por carril:** las marcas de "no actualizado" se aplican por carril (tráfico, motor adaptativo, estado operativo) y no por vista completa, conforme a CA-18.17, CA-18.18 y CA-18.19. La causa de un carril no implica fallo de los otros. Esto contrasta deliberadamente con CA-17.14 de HU-17, donde la marca aplica simultáneamente a ambos periodos porque la causa raíz es la misma (motor de cálculo único de KPIs). **HU-19 — política conservadora:** la exportación de reportes rechaza la generación cuando la fuente está caída, en lugar de generar con datos no actualizados marcados. La justificación es que un PDF o Excel descargado es un artefacto persistente que circula fuera del sistema; permitir generar reportes sobre datos no confirmados abriría la posibilidad de difundir datos "no actualizados" sin marca visible. **HU-21 — política conservadora en el disparo:** el inicio del escalamiento se rechaza si los endpoints de contexto operativo no responden, en lugar de permitir escalar con campos automáticos vacíos. La justificación es la integridad auditable del registro de incidentes. La consulta del registro sí aplica marca pasiva normal (CA-21.28). |
| Notas | La marca semántica es nominalmente distinta entre los dos modos por razones de claridad para el usuario: "desactualizado" comunica que el dato existe pero no refleja necesariamente lo que está pasando ahora; "no confirmado" comunica que no podemos garantizar que esa decisión siga vigente porque el componente que la confirma no está respondiendo. La distinción operativa es cerrada en DHU-005 y debe preservarse en la implementación de cada HU. La política conservadora de HU-19 y de HU-21 (disparo) representa un patrón de excepción documentado: cuando el artefacto generado puede inducir conclusiones erróneas si lleva datos no actualizados, se prefiere rechazo a marca pasiva. Las HUs que aplican esta política conservadora deben hacerlo explícitamente; las demás HUs aplican el comportamiento estándar de marca pasiva. |

#### RNF-REL-02 — Disponibilidad transversal de la alerta del estado operativo

| Campo | Contenido |
|---|---|
| Identificador | RNF-REL-02 |
| Característica ISO 25010:2023 | Reliability / Availability |
| Subcaracterística | Availability |
| Descripción normativa | La alerta del estado operativo del sistema debe ser visible de manera consistente en todas las vistas del Operador durante el tiempo que el sistema permanezca en un estado distinto a operación normal. No debe haber vistas del sistema donde la alerta no aparezca cuando corresponde. La disponibilidad de la alerta es independiente de qué vista específica esté activa: el Operador la percibe sin importar dónde haya navegado dentro del sistema. |
| Criterio de aceptación medible | Prueba operativa visitando cada vista accesible al Operador durante un escenario de degradación inducida: la alerta debe estar visible en todas las vistas con la misma información (nivel del estado degradado, componente disparador, tiempo transcurrido desde la entrada al estado) y con el mismo código visual. |
| Método de validación | Prueba automatizada que recorre las vistas del Operador durante un escenario de degradación inducida, verificando presencia y consistencia visual de la alerta. |
| HUs/TTH origen | HU-10 (Candidato a RNF: alerta transversal y consistente en todas las vistas), HU-12 (Candidato a RNF: explicación accesible siempre que el sistema esté en estado degradado o de falla total). |
| DHUs relacionadas | DHU-009 (relación entre marca pasiva del Bloque B y alerta activa del Bloque C), DHU-010 (criterios para clasificar trabajo del Bloque C como TTH), DHU-011 (cobertura de F25 por composición). |
| Prioridad MoSCoW sugerida | Must (continuidad operativa visible al Operador). |
| Aplicabilidad | HU-10 (alerta activa transversal del estado operativo) y HU-12 (explicación del modo degradado activo), que son componentes transversales del Bloque C visibles en todas las vistas operativas del Operador. |
| Notas | La disponibilidad transversal es regla del producto que distingue el rol del Bloque C respecto a las marcas pasivas por panel del Bloque B. La duplicación de lógica entre marca pasiva por vista y alerta transversal está cerrada por DHU-009 para evitar redundancia: cada bloque cumple su responsabilidad sin interferir con la del otro. |

#### RNF-REL-03 — Continuidad operativa frente a fallos de subsistemas auxiliares

| Campo | Contenido |
|---|---|
| Identificador | RNF-REL-03 |
| Característica ISO 25010:2023 | Reliability / Fault tolerance |
| Subcaracterística | Fault tolerance |
| Descripción normativa | La operación primaria del sistema sobre el semáforo no debe detenerse ni postergarse por fallos de subsistemas auxiliares de auditoría o registro. Las acciones primarias del sistema deben aplicarse de todos modos a la operación, registrándose en un mecanismo de respaldo cuando el subsistema de persistencia principal no responde, para ser persistidas cuando vuelva a estar disponible. Esta resiliencia aplica a la lógica de cambios de estrategia del motor adaptativo, a las transiciones de estado operativo del sistema completo y a las modificaciones de configuración. |
| Criterio de aceptación medible | Prueba operativa de fallo inducido del subsistema de persistencia de auditoría: las acciones primarias del sistema continúan aplicándose; las entradas de auditoría se acumulan en el mecanismo de respaldo; tras restaurar el subsistema de persistencia, las entradas del respaldo se persisten en el orden cronológico correcto sin pérdida. |
| Método de validación | Prueba automatizada de fallo inducido sobre cada subsistema de persistencia (decisiones del motor, transiciones de estado operativo, parámetros operativos), verificando continuidad de la operación primaria y consistencia eventual del registro. |
| HUs/TTH origen | HU-08 (Candidato a RNF: la operación del motor no se detiene por fallos del registro de auditoría, CA-08.5), HU-10 (Candidato a RNF: la activación de fallbacks y la operación del sistema no se detiene por fallos del registro de auditoría, CA-10.8), HU-15 (Candidato a RNF: las modificaciones de configuración no deben interrumpir la operación del sistema, CA-15.3), HU-21 (Candidato a RNF de resiliencia de persistencia: la operación del motor adaptativo nunca depende del registro de incidentes). |
| DHUs relacionadas | DHU-008 (distinción arquitectónica entre componente caído, modo degradado y lógica de fallback). |
| Prioridad MoSCoW sugerida | Must (separación entre operación primaria y auditoría es regla arquitectónica inegociable). |
| Aplicabilidad | Tres subsistemas de persistencia: registro de decisiones del motor adaptativo (CT-10.9), registro de transiciones de estado operativo (CT-04.3, CT-04.6), registro de auditoría de parámetros (CA-15.4). Adicionalmente la propagación de configuración (CA-15.3) y el registro de incidentes escalados (CA-21.10) están sujetos al mismo principio. |
| Notas | El mecanismo de respaldo concreto (cola en memoria, archivo local) se cierra al implementar conforme a CT-04.6. El principio inegociable es la separación entre operación primaria y auditoría: ningún registro de auditoría puede convertirse en bloqueante de la operación del sistema sobre la infraestructura del tráfico. |

#### RNF-REL-04 — Durabilidad de registros append-only del sistema

| Campo | Contenido |
|---|---|
| Identificador | RNF-REL-04 |
| Característica ISO 25010:2023 | Reliability / Recoverability |
| Subcaracterística | Recoverability |
| Descripción normativa | Los registros append-only del sistema deben ser durables frente a fallos temporales del sistema. La pérdida de filas o entradas por fallo temporal de la infraestructura de persistencia es inaceptable: el sistema debe garantizar que cada escritura confirmada al usuario o al subsistema productor persista efectivamente y sobreviva a reinicios del componente, del sistema completo o del contenedor. |
| Criterio de aceptación medible | Prueba operativa con escritura inducida seguida de reinicio del sistema: las entradas confirmadas antes del reinicio deben estar presentes tras el reinicio, con todos sus campos íntegros y sin alteración. Se prueba sobre cada registro declarado en la aplicabilidad. |
| Método de validación | Prueba automatizada de escritura, reinicio y verificación de integridad sobre cada registro append-only del sistema. |
| HUs/TTH origen | HU-08 (Candidato a RNF de persistencia: pérdida de decisiones por fallo de escritura es inaceptable), HU-09 (Candidato a RNF de persistencia: notas durables, no pueden perderse por fallo temporal del sistema), HU-10 (Candidato a RNF de persistencia: pérdida de transiciones por fallo de escritura es inaceptable), HU-14 (Candidato a RNF de durabilidad: el registro de predicciones y observaciones debe ser durable), HU-15 (Candidato a RNF de durabilidad: los parámetros persistidos no pueden perderse), HU-16 (Candidato a RNF de durabilidad: el histórico persistido no puede perderse), HU-20 (Candidato a RNF de durabilidad: el registro de predicciones extendido con respaldo debe ser durable), HU-21 (Candidato a RNF de durabilidad: el registro de incidentes escalados debe ser durable). |
| DHUs relacionadas | DHU-013 (sustrato técnico inglobado en HUs cuando no hay consumidores múltiples), DHU-016 subsección B (F30 inglobada en HU-16 como persistencia operacional independiente). |
| Prioridad MoSCoW sugerida | Must (precondición de la auditoría, de la reportería gerencial y de la evaluación retroactiva del modelo). |
| Aplicabilidad | Ocho registros append-only declarados en el sistema: registro de decisiones del motor (CT-10.9 / CA-08.1), registro de notas del Operador (HU-09), registro de transiciones de estado operativo (CT-04.3 / CA-10.7), registro de predicciones (CT-09.5 / CA-14.1), registro de auditoría de parámetros (CA-15.4), histórico de estados operacionales (CA-16.1), registro de predicciones del modelo de respaldo (CA-20.2 inglobada como extensión de CT-09.5), registro de incidentes escalados (CA-21.10 inglobada). |
| Notas | La durabilidad es una propiedad distinta de la inmutabilidad: la durabilidad garantiza que lo escrito persista; la inmutabilidad garantiza que lo escrito no se modifique después. Ambas propiedades aplican simultáneamente a los registros append-only del sistema. La inmutabilidad está cubierta por RNF-SEC-01 (Security / Integrity); la durabilidad está cubierta por este RNF (Reliability / Recoverability). |

#### RNF-REL-05 — Resiliencia de persistencia ante fallo de escritura

| Campo | Contenido |
|---|---|
| Identificador | RNF-REL-05 |
| Característica ISO 25010:2023 | Reliability / Fault tolerance |
| Subcaracterística | Fault tolerance |
| Descripción normativa | Ante un fallo temporal de escritura sobre la persistencia, el sistema debe informar al usuario del fallo con un mensaje comprensible, preservar el contenido que el usuario haya escrito o el contexto de la operación pendiente, y permitir reintentar la operación sin pérdida de información. El usuario no debe perder trabajo realizado por causa de un fallo transitorio del subsistema de persistencia. |
| Criterio de aceptación medible | Prueba operativa de fallo inducido de escritura mientras el usuario está creando o modificando una entrada: el sistema muestra mensaje informativo; el contenido escrito por el usuario se preserva en pantalla; al recuperarse el subsistema de persistencia, el reintento aplica la operación sin pérdida. |
| Método de validación | Prueba automatizada de creación de entrada con fallo de persistencia inducido seguido de recuperación y reintento. |
| HUs/TTH origen | HU-09 (CA-09.5 — fallo temporal en escritura de nota), HU-21 (CA-21.13 — fallo de escritura del incidente al confirmar el escalamiento). |
| DHUs relacionadas | No aplica de forma específica. |
| Prioridad MoSCoW sugerida | Should para HU-09 (MVP2); Should para HU-21 (MVP2). |
| Aplicabilidad | Dos operaciones de creación con interacción directa del usuario: creación de notas (HU-09) y creación de incidentes escalados (HU-21). |
| Notas | Este RNF es distinto del RNF-REL-04 (durabilidad) porque cubre el momento de la creación con interacción humana, no la durabilidad post-escritura. El usuario es el bien protegido: su trabajo en proceso no debe perderse. La operación primaria del sistema sobre el semáforo no depende de estas operaciones y por tanto no requiere mecanismo de respaldo análogo a RNF-REL-03. |

#### RNF-REL-06 — Tolerancia a fallos del componente de generación de reportes

| Campo | Contenido |
|---|---|
| Identificador | RNF-REL-06 |
| Característica ISO 25010:2023 | Reliability / Fault tolerance |
| Subcaracterística | Fault tolerance |
| Descripción normativa | Cuando el componente de generación de reportes falla internamente durante la producción de un PDF o Excel, el sistema debe comunicar al Gerente la falla con un mensaje informativo no técnico, sin entregar archivos parciales o corruptos. La política conservadora protege la integridad del artefacto exportado, que circula fuera del sistema y puede ser interpretado como fuente de verdad por destinatarios externos. |
| Criterio de aceptación medible | Prueba operativa de fallo inducido durante la generación: el sistema no entrega archivo; muestra mensaje informativo no técnico al Gerente; permite reintentar la generación. No se entregan archivos con páginas en blanco, con datos parciales sin advertencia, ni con metadatos inconsistentes. |
| Método de validación | Prueba automatizada de fallo inducido en distintos puntos del proceso de generación (consulta de datos, renderizado de gráficos, ensamblaje del archivo), verificando que ningún archivo parcial llega al Gerente. |
| HUs/TTH origen | HU-19 (Candidato a RNF de tolerancia a fallos del componente de generación, CA-19.23). |
| DHUs relacionadas | DHU-017 subsección J (política conservadora aplicada al MVP2). |
| Prioridad MoSCoW sugerida | Could (HU-19 es MVP2). |
| Aplicabilidad | HU-19 (exportación de reportes a PDF/Excel). |
| Notas | El RNF complementa la política conservadora de RNF-REL-01 (rechazo ante fuente caída) con la política conservadora del propio componente generador: ni la entrada ni el proceso pueden producir artefactos no confiables. |

#### RNF-REL-07 — Manejabilidad de concurrencia en operaciones de modificación

| Campo | Contenido |
|---|---|
| Identificador | RNF-REL-07 |
| Característica ISO 25010:2023 | Reliability / Fault tolerance (con característica secundaria: Functional Suitability / Functional correctness, referencia cruzada a RNF-FUN-NN cuando se redacte) |
| Subcaracterística | Fault tolerance |
| Descripción normativa | Cuando múltiples usuarios autorizados modifican simultáneamente el mismo recurso de configuración, el sistema no debe perder modificaciones de forma silenciosa. El sistema debe detectar que el recurso fue modificado por otro usuario entre la lectura y la escritura del segundo modificador, presentar al segundo modificador información sobre la modificación intermedia (identidad del autor, marca de tiempo, parámetros que cambiaron), y permitirle decidir entre confirmar la sobrescritura o cancelar y recargar para reevaluar sobre los valores actuales. El registro de auditoría debe preservar la traza completa de ambas modificaciones, no solo de la última aplicada. |
| Criterio de aceptación medible | Prueba operativa con dos sesiones simultáneas modificando la misma configuración: la primera modificación se persiste normalmente con registro de auditoría; la segunda recibe la advertencia con detalles de la modificación intermedia; al confirmar sobrescritura, la modificación se persiste con su propio registro de auditoría; tras la prueba, el registro de auditoría contiene ambas modificaciones en orden cronológico. |
| Método de validación | Prueba automatizada de modificación concurrente con verificación del registro de auditoría completo. |
| HUs/TTH origen | HU-15 (CA-15.11 — control de concurrencia entre Administradores con last-write-wins y advertencia explícita; Candidato a RNF de manejabilidad de concurrencia). |
| DHUs relacionadas | DHU-014 subsección E (concurrencia entre Administradores en HU-15 con last-write-wins). |
| Prioridad MoSCoW sugerida | Must (no perder modificaciones silenciosamente es regla inegociable de HU-15). |
| Aplicabilidad | HU-15 (configuración de parámetros operativos). Las vistas read-only del Gerente (HU-16, HU-17) no requieren este mecanismo conforme a DHU-016 subsección H (concurrencia entre Gerentes es caso de carga, no de concurrencia funcional). |
| Notas | El mecanismo concreto se implementa con control de concurrencia optimista con marca de versión: cada lectura anota el timestamp de la última modificación de la configuración; cada intento de guardado verifica que ese timestamp no haya cambiado en la base de datos desde la lectura; si cambió, se bloquea el guardado y se muestra la advertencia. Este patrón es estándar y suficiente para el escenario MVP1 donde la concurrencia es rara. |

#### RNF-REL-08 — Atomicidad de las transiciones de estado operativo

| Campo | Contenido |
|---|---|
| Identificador | RNF-REL-08 |
| Característica ISO 25010:2023 | Reliability / Faultlessness |
| Subcaracterística | Faultlessness |
| Descripción normativa | Las transiciones del estado operativo del sistema completo entre los cinco estados declarados (operación normal, degradado nivel 1, degradado nivel 2, degradado nivel 3, falla total) deben ser atómicas: el sistema no queda en estados intermedios o indefinidos durante el cambio. Si una transición no puede completarse íntegramente (por ejemplo, el fallback correspondiente no puede activarse), el sistema permanece en el estado anterior y registra una entrada de error en el registro de transiciones, en lugar de quedar en un estado parcialmente transitado. |
| Criterio de aceptación medible | Prueba operativa de transición con fallo inducido del fallback de destino: el sistema permanece en el estado anterior; el registro de transiciones contiene una entrada de error con causa raíz; la operación sobre el semáforo continúa con la lógica del estado anterior. No se permiten estados híbridos donde algunos componentes han transicionado y otros no. |
| Método de validación | Prueba automatizada que induce fallos en la activación de fallbacks y verifica que el sistema preserva el estado anterior con registro del error. |
| HUs/TTH origen | TTH-04 (CT-04.7 — atomicidad de transiciones). |
| DHUs relacionadas | DHU-008 (distinción arquitectónica entre componente caído, modo degradado y lógica de fallback). |
| Prioridad MoSCoW sugerida | Must (precondición de la cascada de fallback). |
| Aplicabilidad | Lógica de fallback en cascada del sistema (TTH-04), consumida por HU-10, HU-11, HU-12 y HU-13. |
| Notas | La atomicidad protege contra modos de fallo cascadeantes donde un fallback parcialmente activado induce nuevos fallos. La separación arquitectónica entre componente caído (Bloque B), modo degradado (transversal) y lógica de fallback (Bloque C) cerrada en DHU-008 sustenta este RNF. |

#### RNF-REL-09 — Comportamiento por defecto conservador ante fallo del propio mecanismo de detección de salud

| Campo | Contenido |
|---|---|
| Identificador | RNF-REL-09 |
| Característica ISO 25010:2023 | Reliability / Fault tolerance |
| Subcaracterística | Fault tolerance |
| Descripción normativa | Cuando el propio mecanismo de detección de salud de componentes falla y el sistema no puede determinar con certeza el estado operativo agregado, el comportamiento por defecto debe ser conservador: el sistema reporta el estado como "no confirmado" y lo notifica a los consumidores, sin asumir falsamente operación normal. Esto preserva la propiedad de que el Operador nunca recibe falsa tranquilidad cuando el sistema no tiene información confiable. |
| Criterio de aceptación medible | Prueba operativa de fallo inducido del propio mecanismo de detección de salud: el endpoint que expone el estado operativo retorna el flag específico de "no confirmado"; los consumidores (alerta transversal HU-10, explicación HU-12) presentan la marca correspondiente; no se asume falsamente operación normal en ningún punto del sistema. |
| Método de validación | Prueba automatizada de fallo inducido del mecanismo de detección de salud, verificando la propagación correcta del estado "no confirmado". |
| HUs/TTH origen | HU-10 (CA-10.9 — comportamiento ante caída del componente que determina el estado operativo), TTH-04 (CT-04.8 — comportamiento por defecto conservador). |
| DHUs relacionadas | DHU-005 Caso B (aplicado al propio mecanismo de salud), DHU-009 (alerta activa transversal). |
| Prioridad MoSCoW sugerida | Must (precondición de la confiabilidad del sistema de alertas). |
| Aplicabilidad | TTH-04 con propagación a HU-10 y HU-12. |
| Notas | Este RNF es la aplicación recursiva del principio de DHU-005 al propio mecanismo de detección: el monitor que vigila al sistema también debe ser vigilado, y su fallo se trata como una causa raíz declarable. La diferencia con RNF-REL-01 es que aquel cubre fallos de fuentes consumidas por vistas individuales, mientras este cubre el fallo del agregador transversal de estado. |

### 3.6 Security (RNF-SEC)

La característica de Security cubre la protección de la información, la integridad de los registros, la trazabilidad de las acciones y la resistencia frente a accesos no autorizados. En CerebroVial agrupa siete familias: autenticación de usuarios, control de acceso por rol, inmutabilidad de registros append-only, separación de presentación por rol, no exposición de información en respuestas de error, validación dual de restricciones de acceso y no persistencia de artefactos generados.

#### RNF-SEC-01 — Inmutabilidad de registros append-only del sistema

| Campo | Contenido |
|---|---|
| Identificador | RNF-SEC-01 |
| Característica ISO 25010:2023 | Security / Integrity |
| Subcaracterística | Integrity |
| Descripción normativa | Los registros append-only del sistema no deben modificarse después de la escritura inicial. Una vez confirmada la persistencia de una entrada, su contenido debe ser inmutable para preservar la confiabilidad de la consulta retroactiva, la auditabilidad del flujo histórico y la integridad de la evidencia ante revisión académica o gerencial. La inmutabilidad aplica a todas las entradas escritas; no se admiten operaciones de actualización, borrado lógico o reescritura sobre registros confirmados, salvo las excepciones explícitamente declaradas. |
| Criterio de aceptación medible | Prueba operativa de intento de modificación inducida sobre cada registro declarado en la aplicabilidad: el sistema rechaza la operación a nivel de capa de datos o de capa de servicio; no existe endpoint expuesto que permita la modificación; las pruebas automatizadas verifican que tras escribir N entradas en cualquier orden, una consulta posterior recupera las mismas N entradas idénticas a las escritas. |
| Método de validación | Prueba automatizada sobre cada registro append-only que verifica: (a) ausencia de endpoints de modificación, (b) rechazo de operaciones UPDATE/DELETE sobre las tablas correspondientes, (c) integridad de las entradas tras múltiples escrituras y consultas, (d) preservación de campos del momento de la escritura. |
| HUs/TTH origen | HU-08 (Candidato a RNF de trazabilidad: cada entrada del registro debe ser unívocamente identificable y no modificable después de ser escrita), HU-09 (Candidato a RNF de auditoría: las notas no pueden eliminarse después de creadas; Candidato a RNF de inmutabilidad parcial: editables solo dentro de ventana temporal), HU-10 (Candidato a RNF de inmutabilidad: entradas del registro de transiciones de estado operativo no se modifican), HU-14 (Candidato a RNF de auditabilidad: entradas del registro de predicciones no se modifican), HU-15 (Candidato a RNF de auditabilidad: entradas del registro de auditoría no se modifican), HU-16 (Candidato a RNF de auditabilidad: filas del histórico no se modifican), HU-20 (Candidato a RNF de auditabilidad: entradas del registro de predicciones del modelo de respaldo no se modifican), HU-21 (Candidato a RNF de auditabilidad: campos del momento del disparo no se modifican; campos de atención del Administrador se escriben una sola vez y son inmutables después). |
| DHUs relacionadas | DHU-019 subsección C.5 (unificación del RNF de inmutabilidad de logs como principio transversal único). |
| Prioridad MoSCoW sugerida | Must (precondición de la auditoría académica, de la reportería gerencial confiable y de la evaluación retroactiva del modelo). |
| Aplicabilidad | Ocho registros append-only declarados en el sistema, los mismos enumerados en RNF-REL-04 (durabilidad), aplicando ahora la propiedad complementaria de inmutabilidad: registro de decisiones del motor (CT-10.9 / CA-08.1), registro de notas del Operador (HU-09), registro de transiciones de estado operativo (CT-04.3 / CA-10.7), registro de predicciones (CT-09.5 / CA-14.1), registro de auditoría de parámetros (CA-15.4), histórico de estados operacionales (CA-16.1), registro de predicciones del modelo de respaldo (CA-20.2 inglobada como extensión de CT-09.5), registro de incidentes escalados (CA-21.10 inglobada). |
| Excepciones | **HU-09 — inmutabilidad parcial por ventana de edición:** las notas del Operador son editables dentro de una ventana temporal posterior a la creación (sugerencia: 24 horas conforme a CA-09.4), pasada la cual quedan inmutables. La ventana de edición es excepción declarada para preservar el balance entre flexibilidad de corrección y valor de auditoría. **HU-21 — transición de estado del incidente:** los incidentes escalados tienen un campo de estado (Enviado / Atendido) que el Administrador puede transicionar; esta transición es escritura única (de Enviado a Atendido), tras la cual los campos de atención quedan inmutables conforme a CA-21.11. No es modificación de campos existentes; es escritura de campos adicionales asociados al evento de atención. |
| Notas | La inmutabilidad es propiedad complementaria a la durabilidad cubierta por RNF-REL-04. La durabilidad garantiza que lo escrito persista; la inmutabilidad garantiza que lo escrito no se modifique después. Ambas propiedades aplican simultáneamente a los registros append-only del sistema, y juntas sustentan el valor probatorio de los registros frente a evaluación externa. La unificación del RNF a partir de las ocho declaraciones individuales en las HUs está justificada por DHU-019 subsección C.5: el principio es uno solo. |

#### RNF-SEC-02 — Autenticación del usuario al sistema

| Campo | Contenido |
|---|---|
| Identificador | RNF-SEC-02 |
| Característica ISO 25010:2023 | Security / Authenticity |
| Subcaracterística | Authenticity |
| Descripción normativa | El acceso a cualquier funcionalidad operativa del sistema debe estar precedido por la autenticación exitosa del usuario mediante credenciales personales. Las credenciales deben almacenarse hasheadas, nunca en texto plano. El mecanismo de sesión debe emitir tokens firmados con expiración configurable, cuya validez se verifica en cada solicitud al backend. Las solicitudes no autenticadas deben ser rechazadas; los usuarios sin sesión activa deben ser redirigidos al flujo de inicio de sesión. |
| Criterio de aceptación medible | Las contraseñas en la base de datos están hasheadas con bcrypt con factor de costo ≥ 12, nunca almacenadas en texto plano. El token de sesión es un JWT que incluye al menos los claims de identidad del usuario, rol y expiración. El tiempo de expiración por defecto es de 8 horas, configurable por variable de entorno. Cada solicitud autenticada presenta el token vía cabecera estándar; el sistema valida firma y expiración antes de procesar la solicitud. Solicitudes sin token, con token inválido o con token expirado reciben respuesta HTTP 401. |
| Método de validación | Pruebas automatizadas sobre los flujos de autenticación: inicio de sesión con credenciales válidas, inicio de sesión con credenciales inválidas, decodificación de token válido, decodificación de token expirado, decodificación de token con firma inválida. Inspección de la base de datos para verificar ausencia de contraseñas en texto plano. |
| HUs/TTH origen | HU-01 (CA-01.5 — redirección al login cuando no hay sesión; CA-01.6 — desconexión por expiración de token con mensaje informativo), TTH-01 (CT-01.1 a CT-01.5 — endpoint de login, bcrypt cost factor ≥ 12, claims del JWT, dependency de validación, tests de los flujos de autenticación). |
| DHUs relacionadas | DHU-001 (login no es HU, es TTH-01). |
| Prioridad MoSCoW sugerida | Must (precondición transversal de todas las HUs operativas). |
| Aplicabilidad | Transversal a las 21 HUs operativas. Toda HU operativa tiene como precondición implícita "el usuario está autenticado", explicitada en su CA de redirección al login. |
| Notas | El algoritmo de firma del JWT en MVP1 es HS256 con secret en variable de entorno, suficiente para el alcance académico. La evolución a RS256 con par de claves se evalúa en MVP2 si se integra con sistema externo, conforme a la nota técnica de TTH-01. La asignación inicial de roles se hace directamente en la base de datos para MVP1; no hay interfaz de gestión de roles. |

#### RNF-SEC-03 — Control de acceso por rol

| Campo | Contenido |
|---|---|
| Identificador | RNF-SEC-03 |
| Característica ISO 25010:2023 | Security / Confidentiality |
| Subcaracterística | Confidentiality (con característica secundaria: Security / Authenticity) |
| Descripción normativa | Los usuarios autenticados deben acceder únicamente a las funcionalidades correspondientes a su rol asignado. El sistema debe rechazar todo intento de acceso a recursos fuera del rol del usuario, incluso si el usuario conoce los identificadores o rutas internas de los recursos. El rechazo debe aplicarse en el backend en la capa de control de acceso de cada endpoint, no únicamente en el frontend, para que el control sea efectivo aunque el usuario interactúe directamente con el API. |
| Criterio de aceptación medible | Prueba operativa de acceso cruzado para cada par de rol y endpoint: un usuario autenticado con rol distinto al esperado recibe respuesta HTTP 403 al solicitar el endpoint vía API; el endpoint no procesa la solicitud ni retorna el recurso. La verificación cubre todos los endpoints expuestos por el sistema; no existen rutas operativas sin control de acceso por rol. |
| Método de validación | Prueba automatizada que recorre cada endpoint del backend y verifica que solo los roles autorizados reciben respuesta 200; el resto recibe 403. La cobertura debe ser exhaustiva sobre la matriz de endpoints × roles. |
| HUs/TTH origen | HU-01 (CA-01.1 a CA-01.4 — segregación por rol en vistas y en API), HU-15 (CA-15.12 — HTTP 403 para roles no-Administrador), HU-16 (CA-16.20 — restricción al rol Gerente), HU-17 (CA-17.15 — restricción al rol Gerente), HU-18 (CA-18.20 — restricción al rol Gerente), HU-19 (CA-19.27 — restricción al rol Gerente), HU-20 (CA-20.19 — restricción al rol Administrador), HU-21 (CA-21.30 a CA-21.34 — restricciones por rol Operador / Administrador con visibilidad limitada entre Operadores). |
| DHUs relacionadas | DHU-002 (reformulación del valor en HU de acceso diferenciado por rol). |
| Prioridad MoSCoW sugerida | Must (RNF más crítico de seguridad del sistema según declaración explícita en Candidato a RNF de HU-01). |
| Aplicabilidad | Transversal a las 21 HUs operativas. Cada HU declara explícitamente en su CA correspondiente qué roles tienen acceso. La tabla de aplicabilidad agregada (matriz endpoint × rol) se materializa al implementar y se valida con prueba automatizada exhaustiva. |
| Notas | La declaración de HU-01 ("este es probablemente el RNF más crítico de seguridad del sistema") refleja que el control de acceso por rol es la columna vertebral del modelo de seguridad de CerebroVial: protege la información operativa del Operador, los reportes ejecutivos del Gerente y la configuración técnica del Administrador. La aplicación en el backend, no solo en el frontend, es regla inegociable: un atacante con credenciales válidas de un rol no debe poder acceder a recursos de otro rol manipulando solicitudes directas al API. |

#### RNF-SEC-04 — No filtración de información en respuestas de error de control de acceso

| Campo | Contenido |
|---|---|
| Identificador | RNF-SEC-04 |
| Característica ISO 25010:2023 | Security / Confidentiality |
| Subcaracterística | Confidentiality (con característica secundaria: Security / Resistance) |
| Descripción normativa | Las respuestas HTTP 403 emitidas ante intentos de acceso no autorizados no deben filtrar información sobre el recurso solicitado. El cuerpo de la respuesta no debe contener datos del recurso, no debe indicar si el recurso existe o no existe, y no debe distinguir entre "recurso no existe" y "usuario no autorizado para este recurso". Esto evita oráculos de enumeración que permitirían a un atacante con credenciales válidas de un rol descubrir la existencia de recursos accesibles a otros roles. |
| Criterio de aceptación medible | Prueba operativa de enumeración con usuario autenticado de rol limitado: las respuestas HTTP 403 para recursos existentes y para recursos inexistentes son indistinguibles (mismo cuerpo, misma estructura, mismo timing dentro del margen razonable). No se filtran identificadores internos, nombres ni metadatos del recurso solicitado. |
| Método de validación | Prueba automatizada de enumeración que solicita recursos existentes e inexistentes con credenciales de rol limitado y verifica indistinguibilidad de las respuestas 403. |
| HUs/TTH origen | HU-01 (CA-01.4 — HTTP 403 sin filtrar información del recurso solicitado; Candidato a RNF de privacidad: la respuesta HTTP 403 no debe filtrar información del recurso solicitado). |
| DHUs relacionadas | No aplica de forma específica. |
| Prioridad MoSCoW sugerida | Must (defensa de enumeración es complementaria al control de acceso por rol y operativamente necesaria). |
| Aplicabilidad | Todos los endpoints del backend protegidos por control de acceso por rol. |
| Notas | La equivalencia entre "recurso no existe" y "usuario no autorizado" se materializa con respuesta 403 idéntica para ambos casos. La distinción interna entre los dos casos (a nivel de logs del sistema) sí es legítima para diagnóstico, pero no debe filtrarse al cliente. La diferencia con un sistema que retorna 404 para recursos inexistentes y 403 para no autorizados es deliberada en CerebroVial: la separación por código de estado permite al atacante enumerar recursos. |

#### RNF-SEC-05 — Segregación de presentación entre vistas del Operador y del Administrador con el mismo origen de datos

| Campo | Contenido |
|---|---|
| Identificador | RNF-SEC-05 |
| Característica ISO 25010:2023 | Security / Confidentiality |
| Subcaracterística | Confidentiality |
| Descripción normativa | Cuando dos vistas dirigidas a roles distintos consumen el mismo origen de datos en backend, la diferencia entre ambas vistas se materializa en la capa de presentación, no en la capa de datos. El control de acceso a la ruta que renderiza cada vista es el mecanismo efectivo de separación: un usuario con rol del Operador no debe poder navegar a la vista del Administrador, aunque el endpoint subyacente exponga campos adicionales no visibles al Operador. Los campos adicionales que viajan en el payload compartido entre roles deben ser campos no sensibles (métricas operativas del propio sistema, no datos personales ni credenciales). |
| Criterio de aceptación medible | Prueba operativa de acceso a la ruta de la vista del Administrador con credenciales del Operador: el sistema rechaza el acceso a la ruta. Inspección del payload del endpoint compartido: los campos adicionales expuestos son métricas operativas del sistema, no datos sensibles. |
| Método de validación | Prueba automatizada de denegación de acceso por rol a la ruta de la vista del Administrador con credenciales del Operador. Inspección documental del esquema del payload del endpoint compartido contra criterios de sensibilidad. |
| HUs/TTH origen | HU-11 (Candidato a RNF de separación de roles: contenido expuesto en HU-11 Operador y HU-13 Administrador mantenidos separados en presentación), HU-13 (Candidato a RNF de seguridad: campos técnicos adicionales visibles solo para el rol Administrador; RBAC se aplica a nivel de la ruta de la vista). |
| DHUs relacionadas | DHU-013 (clasificación de F17 como HU del Administrador con el mismo sustrato técnico que HU-11), DHU-014 subsección G (patrón "un endpoint, dos vistas" con filtrado en presentación; TTH-06 como Trabajos Futuros). |
| Prioridad MoSCoW sugerida | Must (operacionalmente necesaria para HU-11 y HU-13). |
| Aplicabilidad | Endpoint compartido de estado de componentes (CT-04.5) consumido por HU-11 (Operador) y HU-13 (Administrador). |
| Notas | La declaración de DHU-014 sobre TTH-06 (capa de DTOs transversal al backend) como Trabajos Futuros está alineada con este RNF: en MVP1 el patrón es un endpoint y un DTO completo, con filtrado en presentación; la evolución a DTOs específicos por rol es mejora arquitectónica fuera del alcance académico. Este RNF queda formalizado normativamente conforme a la decisión cerrada en DHU-019 subsección C.3 (reformulación del RNF de seguridad de HU-13 que mezclaba normativa con descriptivo). |

#### RNF-SEC-06 — Validación dual de restricciones de acceso y de datos en frontend y backend

| Campo | Contenido |
|---|---|
| Identificador | RNF-SEC-06 |
| Característica ISO 25010:2023 | Security / Integrity |
| Subcaracterística | Integrity (con característica secundaria: Maintainability / Testability) |
| Descripción normativa | Las restricciones del sistema sobre datos de entrada y sobre acceso a recursos deben validarse tanto en el frontend como en el backend. La validación en el frontend mejora la experiencia del usuario al detectar errores antes de la solicitud al servidor; la validación en el backend es la garantía efectiva contra bypass del frontend mediante solicitudes directas al API. Las dos validaciones cubren el mismo conjunto de restricciones; sus implementaciones son independientes y se prueban por separado. |
| Criterio de aceptación medible | Prueba operativa de bypass del frontend: solicitudes directas al backend con datos que el frontend rechazaría son rechazadas también por el backend con respuesta HTTP apropiada (400 para validación de datos, 403 para validación de acceso). Cobertura de tests automatizados sobre validaciones de datos en backend independiente de los tests de frontend. |
| Método de validación | Prueba automatizada de bypass del frontend sobre cada validación declarada. Cobertura de tests de validación de backend independiente. |
| HUs/TTH origen | HU-15 (CA-15.6 — validación de rangos válidos en frontend y backend), HU-16 (Candidato a RNF de validación dual: restricciones sobre fechas validadas tanto en formulario del frontend como en el backend), HU-21 (Candidato a RNF de validación dual: restricciones de acceso por rol y de visibilidad limitada entre Operadores validadas tanto en frontend como en backend). |
| DHUs relacionadas | No aplica de forma específica. |
| Prioridad MoSCoW sugerida | Must para HU-15 (MVP1) y HU-16 (MVP1); Should para HU-21 (MVP2). |
| Aplicabilidad | HU-15 (validación de rangos de parámetros operativos), HU-16 (validación de restricciones sobre fechas del selector de periodo), HU-21 (validación de visibilidad limitada entre Operadores). |
| Notas | El patrón se materializa así: el frontend valida y muestra mensajes inmediatos al usuario; el backend valida con las mismas reglas y rechaza solicitudes que las violan. La duplicación de lógica es deliberada y aceptada como costo necesario de la defensa en profundidad. Si en el futuro la lógica de validación se vuelve compleja, se evalúa centralizarla en una librería compartida; en MVP1 las validaciones son sencillas y la duplicación tiene costo bajo. |

#### RNF-SEC-07 — No persistencia de reportes generados en el servidor

| Campo | Contenido |
|---|---|
| Identificador | RNF-SEC-07 |
| Característica ISO 25010:2023 | Security / Confidentiality |
| Subcaracterística | Confidentiality |
| Descripción normativa | Los reportes exportables generados por el sistema (PDF y Excel) no se almacenan en el servidor después de su entrega al Gerente. La generación es por demanda y la descarga es directa: el archivo se produce, se transfiere al cliente, y no permanece en el sistema. Esto evita la necesidad de gestión de almacenamiento de archivos generados, preserva trazabilidad temporal limpia del histórico (cada reporte refleja el estado en el momento de su generación), y limita la superficie de exposición de datos persistidos fuera de los registros append-only del sistema. |
| Criterio de aceptación medible | Inspección del sistema de archivos del servidor después de la generación de un reporte: no existen archivos del reporte generado en el servidor. La trazabilidad de "qué reportes se generaron" se preserva mediante el registro de auditoría de las solicitudes de exportación, no mediante la persistencia de los artefactos. |
| Método de validación | Prueba automatizada de generación seguida de inspección del sistema de archivos para verificar ausencia del artefacto en el servidor. |
| HUs/TTH origen | HU-19 (CA-19.24 y Candidato a RNF de no persistencia de reportes generados: los reportes generados no se almacenan en el servidor). |
| DHUs relacionadas | DHU-017 subsección C (composición de F16 como HU única que cubre PDF y Excel). |
| Prioridad MoSCoW sugerida | Could (HU-19 es MVP2). |
| Aplicabilidad | HU-19 (exportación de reportes a PDF o Excel). |
| Notas | La política de no persistencia simplifica la operación: no requiere gestión de retención de archivos generados, no requiere limpieza periódica, no requiere control de acceso adicional sobre artefactos persistidos. El usuario que necesite preservar un reporte lo descarga y lo gestiona fuera del sistema. La trazabilidad gerencial se mantiene mediante el registro de auditoría de las solicitudes de exportación, no de los artefactos en sí. |

### 3.1 Functional Suitability (RNF-FUN)

La característica de Functional Suitability cubre la corrección, completitud y apropiación del conjunto de funciones ofrecidas por el sistema, evaluando si los resultados producidos son los correctos incluso en condiciones límite y si el catálogo de funciones cubre las tareas reales de las Personas del producto. En CerebroVial agrupa cinco familias: manejabilidad de datos faltantes, comparabilidad rigurosa entre modelos, cobertura de catálogos de plantillas, calidad del modelo predictivo, independencia entre dimensiones funcionales.

**Recordatorio terminológico:** los RNF de Functional Suitability (RNF-FUN) no son Requisitos Funcionales (RF). Evalúan la calidad del catálogo de funciones que los RF declaran; no agregan comportamientos al sistema. La distinción está explicada en la sección 0.3 del preámbulo de este documento.

#### RNF-FUN-01 — Manejabilidad de datos faltantes en vistas de consulta y cálculo

| Campo | Contenido |
|---|---|
| Identificador | RNF-FUN-01 |
| Característica ISO 25010:2023 | Functional Suitability / Functional correctness |
| Subcaracterística | Functional correctness |
| Descripción normativa | Cuando las vistas o componentes de cálculo del sistema operan sobre periodos, ventanas temporales o conjuntos de datos que no contienen información, contienen información parcial o presentan condiciones matemáticamente indefinidas (división por cero, agregados sobre cero filas), el sistema debe comunicar explícitamente esa condición al usuario en lugar de presentar valores calculados sobre vacío. La comunicación explícita preserva la corrección funcional: el usuario nunca debe confundir "sin datos" con "datos válidos en cero", ni recibir comparativas espurias sobre cero muestras, ni interpretar carriles vacíos como información operativa. |
| Criterio de aceptación medible | Prueba operativa sobre cada vista y componente declarado en la aplicabilidad, inyectando escenarios de ausencia total de datos, cobertura parcial y condiciones matemáticamente indefinidas: el sistema comunica explícitamente la condición mediante mensaje legible al usuario, no presenta valores numéricos calculados sobre el vacío, y diferencia visualmente "sin datos" de "datos válidos con valor cero" cuando ambos casos pueden coexistir. |
| Método de validación | Prueba automatizada con escenarios sintéticos de ausencia y cobertura parcial sobre cada vista y componente declarado en la aplicabilidad, verificando que la comunicación al usuario es explícita y correcta. |
| HUs/TTH origen | HU-13 (Candidato a RNF: componentes sin historial de evaluaciones se muestran de forma inequívocamente distinta), HU-14 (Candidato a RNF: ventana sin datos suficientes se comunica explícitamente), HU-16 (Candidato a RNF: periodos sin datos o con cobertura parcial), HU-17 (Candidato a RNF: periodos sin datos, variaciones matemáticamente indefinidas, cobertura parcial), HU-18 (Candidato a RNF: periodos sin datos en alguno de los tres carriles, cobertura parcial), HU-19 (Candidato a RNF: periodos sin datos y cobertura parcial en el reporte generado), HU-20 (Candidato a RNF: ventana sin datos suficientes para alguno de los dos modelos o para ninguno), HU-21 (Candidato a RNF: filtros que descartan todos los registros se comunican explícitamente). |
| DHUs relacionadas | DHU-019 subsección C.7 (clasificación de "manejabilidad de datos faltantes" en Functional Correctness). |
| Prioridad MoSCoW sugerida | Must para HUs MVP1 (HU-13, HU-14, HU-16, HU-17); Should para HUs MVP2 (HU-18, HU-19, HU-20, HU-21). |
| Aplicabilidad | Ocho HUs operativas con vistas o componentes de cálculo sobre datos persistidos: HU-13, HU-14, HU-16, HU-17, HU-18, HU-19, HU-20, HU-21. La aplicabilidad concreta por escenario está documentada en los CAs correspondientes de cada HU. |
| Excepciones | No aplica a vistas de tiempo real del Operador (HU-02 a HU-07, HU-10, HU-11, HU-12), donde la ausencia temporal de datos se cubre por el RNF transversal de robustez ante interrupción (RNF-REL-01) con marca pasiva "desactualizado" o "no confirmado". |
| Notas | La distinción operativa entre "manejabilidad de datos faltantes" (este RNF) y "robustez ante interrupción de fuente" (RNF-REL-01) es semántica: el primero cubre la consulta de periodos sin actividad histórica registrada o con condiciones degenerativas matemáticas; el segundo cubre la pérdida temporal de conexión con una fuente operativa. Ambos pueden coexistir en la misma vista. La unificación bajo Functional Correctness está justificada por DHU-019 subsección C.7: el sistema produce resultados correctos comunicando ausencia en lugar de fabricar valores. |

#### RNF-FUN-02 — Calidad del modelo predictivo

| Campo | Contenido |
|---|---|
| Identificador | RNF-FUN-02 |
| Característica ISO 25010:2023 | Functional Suitability / Functional correctness |
| Subcaracterística | Functional correctness |
| Descripción normativa | El modelo predictivo del sistema debe alcanzar un nivel de precisión que sustente su uso operativo como predictor de nivel de congestión. El objetivo aspiracional es alcanzar una exactitud (accuracy) sobre el nivel discreto 0-5 mayor o igual al 80%, evaluada sobre la partición de validación del dataset producido por el componente de simulación del entorno. Adicionalmente, los errores de magnitud sobre el ratio continuo (medido con MAE y RMSE) deben permanecer dentro de rangos compatibles con la utilidad operativa de la predicción para el motor adaptativo y para la vista del Operador. |
| Criterio de aceptación medible | Accuracy sobre el nivel discreto 0-5 medida en la partición de validación: objetivo aspiracional ≥ 80%; criterio de aceptación de Done para el componente: métricas reales reportadas honestamente, sin condicionar Done al cumplimiento del umbral aspiracional. El criterio de aceptación del documento RF/RNF es la disponibilidad de las métricas reales auditadas (MAE, RMSE sobre el ratio, accuracy y matriz de confusión 6×6 sobre el nivel discreto) sobre la partición de validación. |
| Método de validación | Reporte de métricas medidas sobre la partición de validación del dataset producido por TTH-07, computadas según las definiciones operacionales fijadas en HU-14 y CT-09.6, declaradas en el documento de tesis y en la documentación del sprint conforme a D-005. |
| HUs/TTH origen | HU-03 (Candidato a RNF de calidad de predicción: umbrales de error aceptable del modelo; pertenece al componente predictivo, no a la vista — reubicación cerrada en DHU-019 subsección C.4), HU-14 (CA-14.3 a CA-14.5 — definición de las cuatro métricas), TTH-09 (CT-09.6 — evaluación sobre partición de validación; CT-09.7 — objetivo aspiracional accuracy ≥ 80%). |
| DHUs relacionadas | DHU-019 subsección C.4 (reubicación del RNF de calidad de predicción al ámbito del componente predictivo). |
| Prioridad MoSCoW sugerida | Should (el objetivo es aspiracional, no bloqueante de Done según CT-09.7). |
| Aplicabilidad | TTH-09 (modelo predictivo). El RNF tiene origen híbrido: identificado en HU-03 pero materializado en TTH-09. |
| Notas | El objetivo aspiracional del 80% se establece como referencia, no como criterio bloqueante. Si la realidad medida es inferior al objetivo, se reporta honestamente conforme a D-005 (números de tesis: actualizar tras validación real). La separación entre objetivo aspiracional y criterio de Done preserva la honestidad académica: el sistema se evalúa por las métricas reales medidas, no por el cumplimiento ex ante de un umbral. |

#### RNF-FUN-03 — Comparabilidad rigurosa entre modelos predictivos

| Campo | Contenido |
|---|---|
| Identificador | RNF-FUN-03 |
| Característica ISO 25010:2023 | Functional Suitability / Functional correctness |
| Subcaracterística | Functional correctness |
| Descripción normativa | Cuando el sistema presenta una comparativa entre el modelo predictivo principal y el modelo de respaldo, las métricas de ambos modelos deben calcularse sobre exactamente los mismos eventos del registro de predicciones y observaciones. Pares incompletos (eventos para los cuales solo uno de los dos modelos tiene predicción registrada) no entran al cálculo de las métricas comparativas. La integridad de la comparabilidad es inegociable: las métricas reportadas no deben estar sesgadas por diferencias de cobertura entre los dos modelos. |
| Criterio de aceptación medible | Prueba operativa con dataset sintético donde uno de los modelos tenga cobertura parcial: las métricas comparativas se calculan únicamente sobre el subconjunto de eventos cubiertos por ambos modelos; pares incompletos se excluyen y se reportan como métrica de cobertura adicional, separada de las métricas comparativas. |
| Método de validación | Prueba automatizada que inyecta escenarios de cobertura asimétrica entre modelos y verifica que las métricas resultantes son las del subconjunto común. |
| HUs/TTH origen | HU-20 (Candidato a RNF de comparabilidad rigurosa: las métricas de ambos modelos se calculan sobre exactamente los mismos eventos del registro; pares incompletos no entran al cálculo). |
| DHUs relacionadas | DHU-017 subsección D (fuente y alcance del baseline de F19 con registro paralelo). |
| Prioridad MoSCoW sugerida | Could (HU-20 es MVP2). |
| Aplicabilidad | HU-20 (vista comparativa de métricas del modelo predictivo principal vs modelo de respaldo). |
| Notas | La integridad de la comparabilidad es el RNF que sustenta el valor probatorio de la comparativa frente a evaluación académica: cualquier diferencia métrica reportada debe ser atribuible a diferencias de los modelos sobre la misma evidencia, no a diferencias de cobertura. |

#### RNF-FUN-04 — Cobertura de catálogos de plantillas curadas

| Campo | Contenido |
|---|---|
| Identificador | RNF-FUN-04 |
| Característica ISO 25010:2023 | Functional Suitability / Functional completeness |
| Subcaracterística | Functional completeness |
| Descripción normativa | Los catálogos de plantillas curadas que el sistema utiliza para construir textos legibles al usuario (explicación de la razón de selección de estrategia del motor en HU-06, explicación del modo degradado activo en HU-12, textos de impacto operativo de los componentes en HU-11) deben cubrir las combinaciones típicas declaradas como esperadas por el equipo del producto, sin pretender cubrir todas las combinaciones posibles. Las combinaciones no cubiertas caen al texto genérico de respaldo declarado en la HU correspondiente, garantizando que el usuario nunca enfrente un panel vacío. |
| Criterio de aceptación medible | Inspección documental del catálogo de plantillas implementado para HU-06, HU-11 y HU-12: el catálogo cubre las combinaciones típicas declaradas en las notas técnicas de cada HU; las combinaciones no cubiertas activan el texto genérico de respaldo correspondiente sin panel vacío. |
| Método de validación | Revisión documental del catálogo de plantillas en sesión dedicada del equipo. Prueba automatizada que verifica la activación del texto genérico de respaldo cuando se inyecta una combinación no cubierta. |
| HUs/TTH origen | HU-06 (catálogo de plantillas estimado en 5-10), HU-11 (catálogo de textos de impacto operativo, CA-11.3), HU-12 (Candidato a RNF de cobertura: catálogo de plantillas debe cubrir combinaciones típicas; CA-12.4 cubre el resto con texto genérico). |
| DHUs relacionadas | DHU-011 (eliminación de HU-13 original y cobertura de F25 por composición). |
| Prioridad MoSCoW sugerida | Must para HU-06, HU-11, HU-12 (MVP1). |
| Aplicabilidad | HU-06, HU-11, HU-12. |
| Notas | El balance entre cobertura y simplicidad operativa: catálogos pequeños (5-10 textos por HU) son extensibles y mantenibles; catálogos exhaustivos sobre todas las combinaciones posibles son irreales y prematuros. La existencia del texto genérico de respaldo garantiza que la cobertura no exhaustiva no degenera en panel vacío para el usuario. |

#### RNF-FUN-05 — Identificabilidad autosuficiente de artefactos exportados

| Campo | Contenido |
|---|---|
| Identificador | RNF-FUN-05 |
| Característica ISO 25010:2023 | Functional Suitability / Functional appropriateness |
| Subcaracterística | Functional appropriateness |
| Descripción normativa | Los artefactos exportados por el sistema (reportes PDF y Excel descargables por el Gerente) deben ser identificables sin necesidad de abrirlos. La convención de nombres del archivo descargado debe permitir distinguir múltiples reportes acumulados en el sistema de archivos del usuario por el periodo, el tipo de reporte y el momento de generación. El principio operativo es la autosuficiencia del nombre: el nombre debe responder a la pregunta "¿qué es este archivo?" sin requerir abrirlo. |
| Criterio de aceptación medible | Inspección de los nombres generados para múltiples reportes con distintos parámetros: los nombres son distinguibles entre sí; cada nombre comunica al menos el tipo de reporte (KPIs o comparativa), el periodo cubierto y el momento de generación. La convención de nombres está documentada y aplicada consistentemente. |
| Método de validación | Prueba automatizada que genera múltiples reportes con distintos parámetros y verifica unicidad y autosuficiencia de los nombres producidos. |
| HUs/TTH origen | HU-19 (CA-19.25 y Candidato a RNF de identificabilidad del archivo descargado: autosuficiencia del nombre). |
| DHUs relacionadas | DHU-017 subsección C. |
| Prioridad MoSCoW sugerida | Could (HU-19 es MVP2). |
| Aplicabilidad | HU-19 (exportación de reportes a PDF/Excel). |
| Notas | La identificabilidad autosuficiente es propiedad de Functional Appropriateness porque facilita la tarea del Gerente sin pasos innecesarios: el Gerente no debe abrir cada archivo para distinguirlos. La convención concreta de nombres se cierra al implementar; el principio operativo es el RNF. |

#### RNF-FUN-06 — Independencia entre dimensiones funcionales del registro de incidentes

| Campo | Contenido |
|---|---|
| Identificador | RNF-FUN-06 |
| Característica ISO 25010:2023 | Functional Suitability / Functional correctness |
| Subcaracterística | Functional correctness |
| Descripción normativa | El estado de un incidente escalado (Enviado / Atendido, gestionado por el Administrador) y el estado operativo del sistema (gobernado por la lógica de fallback automáticamente) son dimensiones funcionales independientes. La recuperación automática del sistema desde un estado degradado a operación normal no cierra automáticamente los incidentes asociados; el cierre lo decide explícitamente el Administrador. Esta independencia preserva la corrección del registro como evidencia auditable separada de la dinámica operativa del sistema. |
| Criterio de aceptación medible | Prueba operativa con secuencia: (a) Operador escala un incidente durante un estado degradado del sistema; (b) el sistema se recupera automáticamente a operación normal antes de que el Administrador atienda el incidente; (c) el incidente permanece en estado "Enviado" tras la recuperación del sistema; (d) solo el Administrador puede transicionarlo a "Atendido". El registro de auditoría refleja correctamente las dos dimensiones independientes. |
| Método de validación | Prueba automatizada del flujo descrito, verificando que el estado del incidente no cambia automáticamente con la transición del estado operativo del sistema. |
| HUs/TTH origen | HU-21 (CA-21.27 y Candidato a RNF de independencia entre dimensiones: el estado del incidente y el estado del sistema son dimensiones independientes). |
| DHUs relacionadas | DHU-017 subsección E (composición de F28 con Operador protagonista y Administrador destinatario). |
| Prioridad MoSCoW sugerida | Could (HU-21 es MVP2). |
| Aplicabilidad | HU-21 (escalamiento de incidentes del Operador al Administrador). |
| Notas | La independencia funcional preserva el valor probatorio del registro de incidentes como evidencia separada del estado operativo del sistema. Si el cierre fuera automático, el registro perdería la traza de la gestión humana del incidente. |

---

### 3.4 Interaction Capability (RNF-INT)

La característica de Interaction Capability cubre la capacidad del sistema de ser comprendido, aprendido, operado, percibido como atractivo y accesible por las Personas del producto bajo las condiciones operativas esperadas. Es la característica ISO 25010:2023 que renombra a "Usability" de la versión 2011 y amplía su alcance con subcaracterísticas como Self-descriptiveness y Accessibility. En CerebroVial agrupa siete familias: usabilidad de las presentaciones operativas, accesibilidad WCAG 2.1 nivel AA, autoexplicación de tooltips, coherencia visual entre vistas, comprensibilidad de explicaciones textuales, presentación visual de artefactos impresos, ocultación de rutas no accesibles.

#### RNF-INT-01 — Usabilidad operativa de las presentaciones del Operador

| Campo | Contenido |
|---|---|
| Identificador | RNF-INT-01 |
| Característica ISO 25010:2023 | Interaction Capability / Appropriateness recognizability + Operability |
| Subcaracterística | Appropriateness recognizability, Operability |
| Descripción normativa | Las presentaciones del Operador deben permitir el reconocimiento inmediato del estado operativo y de las decisiones del sistema sin requerir entrenamiento técnico ni lectura detenida. Los códigos visuales (umbrales de color de las colas, distinción entre estados degradados, identificación de componentes afectados) deben ser interpretables a primer golpe de vista. Las notificaciones temporales no deben interferir con la lectura de los paneles principales. La alineación temporal entre estado actual y predicción debe ser visualmente inmediata. |
| Criterio de aceptación medible | Prueba de usuario con Operadores reales (o sus proxies durante validación académica) midiendo tiempo de identificación de condiciones operativas relevantes (cola alta, componente caído, transición de estado). El tiempo de identificación debe ser cualitativamente menor con el sistema que sin el sistema. La validación cualitativa también verifica que las notificaciones no interfieren con la lectura de los paneles principales. |
| Método de validación | Prueba de usuario con tareas operativas representativas, midiendo tiempos de identificación y comprensión. Cuando una prueba de usuario formal no es factible en el alcance académico, validación cualitativa por revisión del diseño contra los principios declarados en este RNF. |
| HUs/TTH origen | HU-02 (Candidato a RNF de usabilidad: umbrales de color verde/amarillo/rojo), HU-04 (Candidato a RNF de usabilidad: alineación temporal entre estado actual y predicción), HU-05 (Candidato a RNF de usabilidad: nombres de estrategias autoexplicativos), HU-07 (Candidato a RNF de usabilidad: notificación no interfiere con paneles principales), HU-10 (Candidato a RNF de usabilidad: distinción visual entre los cuatro estados de degradación), HU-11 (Candidatos a RNF de usabilidad: identificación del componente afectado de un vistazo; resalte visual panorámico), HU-13 (Candidato a RNF de usabilidad: información de componentes no-OK identificable de un vistazo). |
| DHUs relacionadas | DHU-012 subsección F (códigos visuales unificados entre vistas que reflejan estado del sistema). |
| Prioridad MoSCoW sugerida | Must (la utilidad operativa del sistema depende de la legibilidad inmediata por el Operador en condiciones de presión de tiempo real). |
| Aplicabilidad | HU-02, HU-04, HU-05, HU-07, HU-10, HU-11, HU-13. |
| Notas | La declaración recurrente "probablemente se valida con prueba de usuario" en los Candidatos a RNF de las HUs reconoce que la usabilidad operativa es difícil de cuantificar con umbrales numéricos universales. La validación cualitativa por revisión del diseño contra los principios declarados es aceptable en el alcance académico cuando una prueba de usuario formal no es factible. |

#### RNF-INT-02 — Accesibilidad WCAG 2.1 nivel AA

| Campo | Contenido |
|---|---|
| Identificador | RNF-INT-02 |
| Característica ISO 25010:2023 | Interaction Capability / Accessibility |
| Subcaracterística | Accessibility |
| Descripción normativa | Las presentaciones del sistema deben cumplir los criterios de accesibilidad de WCAG 2.1 nivel AA. En particular, ninguna distinción visual de estados, categorías o resultados debe depender exclusivamente del color: toda diferenciación basada en color debe acompañarse de al menos un código visual redundante (ícono, etiqueta textual, patrón visual, estilo de línea, marcador, intensidad). Los controles interactivos deben ser activables tanto con teclado como con dispositivo apuntador. Los elementos de información dinámica deben ser interpretables por lectores de pantalla. |
| Criterio de aceptación medible | Auditoría WCAG 2.1 nivel AA sobre las vistas operativas del sistema. Inspección sistemática de cada distinción visual basada en color para verificar la presencia de redundancia visual. Prueba de navegación completa con teclado sobre cada vista. Prueba con lector de pantalla sobre los elementos dinámicos relevantes (badge numérico de incidentes pendientes, alerta transversal del estado operativo, indicador comparativo de mejora/empeoramiento). |
| Método de validación | Auditoría automatizada con herramientas estándar (axe, Lighthouse, WAVE) complementada con inspección manual de los criterios que requieren juicio. |
| HUs/TTH origen | HU-10 (Candidato a RNF de accesibilidad: estado identificable sin depender exclusivamente del color, WCAG 2.1 nivel AA), HU-11 (Candidato a RNF de accesibilidad equivalente), HU-13 (Candidato a RNF de accesibilidad: distinción entre los tres estados no debe depender exclusivamente del color), HU-14 (Candidato a RNF de accesibilidad: tooltips activables con teclado; diagonal de la matriz no depende del color), HU-16 (Candidato a RNF de accesibilidad: tooltips con teclado; distinción agregada/desglosada no solo color), HU-17 (Candidato a RNF de accesibilidad: semántica de mejora/empeoramiento no solo color; series del gráfico distinguibles más allá del color; CA-17.7 redundancia), HU-18 (Candidato a RNF de accesibilidad: códigos visuales no solo color; bandas con patrón/etiqueta/intensidad; marcadores activables con teclado; zoom interactivo con alternativa por teclado), HU-19 (Candidato a RNF de accesibilidad: semántica del PDF no solo color; series del gráfico comparativo con estilo de línea o marcador), HU-20 (Candidato a RNF de accesibilidad: indicador comparativo no solo color con texto explícito; matrices con etiqueta), HU-21 (Candidato a RNF de accesibilidad: modal navegable con teclado; badge interpretable por lectores de pantalla; distinción Enviado/Atendido no solo color). |
| DHUs relacionadas | DHU-012 subsección F (códigos visuales unificados que respetan accesibilidad). |
| Prioridad MoSCoW sugerida | Should (accesibilidad WCAG 2.1 nivel AA es estándar de la industria; obligatoria en el sector público en muchas jurisdicciones; aplicable a un sistema operado por funcionarios municipales). |
| Aplicabilidad | Transversal a todas las vistas operativas del sistema: HU-10 a HU-21. La declaración explícita en los Candidatos a RNF cubre las HUs listadas en "HUs/TTH origen". |
| Notas | La accesibilidad WCAG 2.1 nivel AA aplica al sistema independientemente de la formalización de cada Candidato a RNF de cada HU: si una HU no declara explícitamente accesibilidad, el RNF transversal aplica de todos modos. La declaración en los Candidatos a RNF de las HUs es refuerzo, no condición de aplicabilidad. |

#### RNF-INT-03 — Autoexplicación mediante tooltips integrados

| Campo | Contenido |
|---|---|
| Identificador | RNF-INT-03 |
| Característica ISO 25010:2023 | Interaction Capability / Self-descriptiveness |
| Subcaracterística | Self-descriptiveness |
| Descripción normativa | Las vistas que presentan información técnica o agregada al usuario (métricas del modelo predictivo, indicadores agregados de KPIs, indicadores comparativos entre modelos) deben permitir la interpretación de cada elemento sin necesidad de documentación externa, mediante tooltips de ayuda integrados activables tanto con teclado como con dispositivo apuntador. Los tooltips deben explicar la definición operacional de cada métrica o indicador, no solo su nombre. |
| Criterio de aceptación medible | Inspección de cada vista declarada en la aplicabilidad: todos los elementos métricos tienen tooltip integrado; los tooltips son activables con teclado y con dispositivo apuntador; el contenido del tooltip incluye la definición operacional de la métrica o indicador, no solo su nombre. |
| Método de validación | Inspección manual sistemática sobre cada vista declarada en la aplicabilidad y prueba automatizada de activación de tooltips con teclado y con dispositivo apuntador. |
| HUs/TTH origen | HU-14 (Candidato a RNF de usabilidad: tooltips de ayuda integrados, CA-14.7), HU-16 (Candidato a RNF de usabilidad: tooltips integrados, CA-16.16), HU-19 (Candidato a RNF de usabilidad: definiciones operacionales autocontenidas en el PDF), HU-20 (Candidato a RNF de usabilidad: tooltips heredados de HU-14 y específicos del indicador comparativo). |
| DHUs relacionadas | No aplica de forma específica. |
| Prioridad MoSCoW sugerida | Must para HU-14, HU-16 (MVP1); Could para HU-19, HU-20 (MVP2). |
| Aplicabilidad | HU-14, HU-16, HU-19 (definiciones autocontenidas en el artefacto exportado), HU-20. |
| Notas | La autoexplicación mediante tooltips reduce la barrera cognitiva sin contaminar la vista con texto permanente. En el caso de HU-19, la autoexplicación se materializa en el contenido del PDF generado (definiciones operacionales autocontenidas en el reporte), no en tooltips interactivos del PDF. |

#### RNF-INT-04 — Coherencia visual y textual entre vistas relacionadas

| Campo | Contenido |
|---|---|
| Identificador | RNF-INT-04 |
| Característica ISO 25010:2023 | Interaction Capability / Self-descriptiveness (con característica secundaria: Maintainability / Modifiability) |
| Subcaracterística | Self-descriptiveness |
| Descripción normativa | Las vistas que tratan dominios relacionados deben preservar la coherencia visual y textual entre sí, reduciendo la carga cognitiva del usuario al navegar entre ellas. La coherencia aplica a: el orden de presentación de los indicadores cuando los mismos indicadores aparecen en vistas distintas; los tooltips de definición operacional cuando las mismas métricas aparecen en vistas distintas; los códigos visuales que reflejan estados del sistema cuando esos estados aparecen en vistas distintas; el lenguaje, tono y nivel de detalle de los catálogos de plantillas curadas que el sistema utiliza para textos legibles. |
| Criterio de aceptación medible | Inspección sistemática de las parejas de vistas declaradas en la aplicabilidad: el orden de los indicadores en HU-16 y HU-17 es idéntico; los tooltips de definición operacional son idénticos entre vistas; los códigos visuales de las bandas de estado operativo en HU-18 son los mismos que los aplicados en HU-10; el lenguaje, tono y nivel de detalle de las plantillas de HU-06, HU-11 y HU-12 son consistentes (revisión del catálogo completo de textos en sesión dedicada). |
| Método de validación | Inspección manual sistemática sobre las parejas de vistas declaradas. Revisión del catálogo completo de textos en sesión dedicada al cierre de la implementación. |
| HUs/TTH origen | HU-12 (Candidato a RNF de coherencia textual: plantillas consistentes con HU-06 y HU-11), HU-17 (Candidato a RNF de consistencia entre vistas: orden de indicadores y tooltips consistentes con HU-16), HU-18 (Candidato a RNF de coherencia visual: códigos visuales reutilizan los de HU-10), HU-20 (Candidato a RNF de coherencia con HU-14: métricas, orden, tooltips, ventana temporal consistentes con HU-14). |
| DHUs relacionadas | DHU-012 subsección F (códigos visuales unificados). |
| Prioridad MoSCoW sugerida | Must para HU-17, HU-18 (MVP1); Could para HU-20 (MVP2). HU-12 Must (MVP1). |
| Aplicabilidad | HU-12 ↔ HU-06 ↔ HU-11 (catálogos de plantillas); HU-16 ↔ HU-17 (vistas del Gerente); HU-18 ↔ HU-10 (códigos visuales de estado operativo); HU-20 ↔ HU-14 (vistas del Administrador sobre métricas del modelo). |
| Notas | La coherencia es regla del producto, no decisión local de cada HU. Si los códigos visuales o los textos evolucionan en una iteración posterior, la coherencia debe preservarse simultáneamente en todas las vistas afectadas, no actualizarse parcialmente. La característica secundaria de Maintainability / Modifiability refleja que la coherencia depende de tener los textos y códigos visuales como datos modificables, no hardcoded en componentes individuales. |

#### RNF-INT-05 — Comprensibilidad de explicaciones textuales sin formación técnica

| Campo | Contenido |
|---|---|
| Identificador | RNF-INT-05 |
| Característica ISO 25010:2023 | Interaction Capability / Self-descriptiveness + Learnability |
| Subcaracterística | Self-descriptiveness, Learnability |
| Descripción normativa | Las explicaciones textuales generadas por el sistema (explicación de la razón de selección de estrategia del motor en HU-06, explicación del modo degradado activo en HU-12) deben ser comprensibles por un Operador sin formación técnica en el motor adaptativo ni en la arquitectura del sistema. El vocabulario, nivel de abstracción y estructura del texto deben ser apropiados para un funcionario operativo de tránsito municipal, no para un ingeniero de tráfico ni para un desarrollador del sistema. |
| Criterio de aceptación medible | Prueba de usuario con Operadores reales (o sus proxies durante validación académica) leyendo el catálogo de textos en escenarios típicos: los Operadores entienden el sentido del texto sin asistencia ni consulta de documentación externa. Validación cualitativa por revisión del catálogo de textos contra el perfil de la Persona declarada en `LEAN_INCEPTION_CEREBROVIAL.md`. |
| Método de validación | Prueba de usuario con tareas de comprensión sobre escenarios típicos. Cuando una prueba formal no es factible, validación cualitativa por revisión del catálogo. |
| HUs/TTH origen | HU-06 (Candidato a RNF de usabilidad: explicación comprensible por Operador sin formación técnica), HU-12 (Candidato a RNF de usabilidad: explicación comprensible por Operador sin formación técnica en el sistema). |
| DHUs relacionadas | No aplica de forma específica. |
| Prioridad MoSCoW sugerida | Must (sostiene el Objetivo 2 del Producto — sustentar la decisión técnica con métricas auditables — desde la perspectiva del Operador, que es el que materializa la confianza en el sistema). |
| Aplicabilidad | HU-06, HU-12. |
| Notas | Este RNF es complementario de RNF-INT-04 (coherencia entre catálogos): la comprensibilidad de cada plantilla individual y la coherencia entre plantillas son dos propiedades distintas. La comprensibilidad puede validarse por catálogo individual; la coherencia requiere validación cruzada entre catálogos. |

#### RNF-INT-06 — Presentación visual de artefactos exportados en formato impreso

| Campo | Contenido |
|---|---|
| Identificador | RNF-INT-06 |
| Característica ISO 25010:2023 | Interaction Capability / UI aesthetics + Accessibility |
| Subcaracterística | UI aesthetics, Accessibility |
| Descripción normativa | Los gráficos incluidos en los reportes PDF exportados por el Gerente deben ser legibles cuando el reporte se imprime en papel. La tipografía debe ser suficientemente grande; los ejes deben estar etiquetados explícitamente; la escala numérica debe ser identificable; las series del gráfico deben distinguirse más allá del color (estilo de línea, marcador o etiqueta directa) para preservar la legibilidad en impresiones en escala de grises. |
| Criterio de aceptación medible | Inspección de muestras impresas en papel durante la implementación: los gráficos son legibles; los ejes están etiquetados; la escala es identificable; las series son distinguibles en escala de grises. |
| Método de validación | Inspección manual de muestras impresas durante la implementación. |
| HUs/TTH origen | HU-19 (Candidato a RNF de presentación visual: gráficos del PDF legibles en formato impreso). |
| DHUs relacionadas | No aplica de forma específica. |
| Prioridad MoSCoW sugerida | Could (HU-19 es MVP2). |
| Aplicabilidad | HU-19 (exportación de reportes a PDF). |
| Notas | La legibilidad impresa es propiedad operativa del flujo gerencial: los reportes circulan fuera del sistema y eventualmente se imprimen. La degradación de legibilidad en impresión es defecto de calidad incluso si la legibilidad en pantalla es buena. |

#### RNF-INT-07 — Ocultación de rutas no accesibles al rol del usuario

| Campo | Contenido |
|---|---|
| Identificador | RNF-INT-07 |
| Característica ISO 25010:2023 | Interaction Capability / Operability + Appropriateness recognizability |
| Subcaracterística | Operability, Appropriateness recognizability |
| Descripción normativa | Las rutas de navegación del sistema que no son accesibles al rol del usuario autenticado no deben aparecer como enlaces visibles en la interfaz. La presencia de enlaces a rutas no accesibles confunde al usuario al ofrecerle opciones que el sistema rechazará. La ocultación de rutas no accesibles refuerza el valor cognitivo del control de acceso por rol declarado en HU-01: el usuario trabaja con un menú coherente con su rol, no con un menú universal del que algunas entradas fallarán. |
| Criterio de aceptación medible | Inspección de la navegación visible para cada uno de los tres roles (Operador, Gerente, Administrador): solo aparecen enlaces a rutas accesibles al rol. Las rutas accesibles a otros roles no aparecen ni siquiera como enlaces deshabilitados. |
| Método de validación | Prueba automatizada que carga la navegación para cada rol y verifica el conjunto de enlaces visibles. |
| HUs/TTH origen | HU-01 (Candidato a RNF de usabilidad: las rutas no accesibles no deben aparecer ni siquiera como enlaces). |
| DHUs relacionadas | DHU-002 (reformulación del valor en HU de acceso diferenciado por rol). |
| Prioridad MoSCoW sugerida | Must (precondición de la experiencia operativa coherente del usuario). |
| Aplicabilidad | Transversal a las 21 HUs operativas. |
| Notas | Este RNF es propiedad de Interaction Capability, no de Security: el control efectivo de acceso vive en RNF-SEC-03 (backend). La ocultación de rutas en la navegación visible es propiedad de presentación que reduce confusión del usuario, no defensa de seguridad. Las dos propiedades son complementarias: RNF-SEC-03 garantiza que el bypass del frontend no funciona; RNF-INT-07 garantiza que el usuario no necesite intentar el bypass porque no ve las opciones. |

---

### 3.7 Maintainability (RNF-MNT)

La característica de Maintainability cubre la facilidad con la que el sistema puede ser modificado, analizado, probado y extendido sin introducir defectos. En CerebroVial agrupa dos familias: extensibilidad de los catálogos de plantillas curadas sin tocar código, y parametrización sin redeploy de tiempos y umbrales que pueden requerir ajuste operativo.

#### RNF-MNT-01 — Extensibilidad de catálogos de plantillas como datos de configuración

| Campo | Contenido |
|---|---|
| Identificador | RNF-MNT-01 |
| Característica ISO 25010:2023 | Maintainability / Modifiability |
| Subcaracterística | Modifiability |
| Descripción normativa | Los catálogos de plantillas curadas utilizados por el sistema para construir textos legibles al usuario (explicación de la razón de selección de estrategia del motor en HU-06, textos de impacto operativo de los componentes en HU-11, explicación del modo degradado activo en HU-12) deben ser extensibles sin requerir cambios al código del frontend ni del backend. Agregar una plantilla nueva, modificar una existente o ajustar la asociación entre combinación de inputs y plantilla aplicada debe ser una operación sobre datos de configuración, no sobre código compilable. |
| Criterio de aceptación medible | Inspección de la implementación de los catálogos: los catálogos viven como datos de configuración (archivos, tablas, recursos), no como cadenas hardcoded en el código del frontend ni del backend. Prueba operativa de agregar una plantilla nueva sin recompilar ni redesplegar: la plantilla está disponible para el sistema tras la operación. |
| Método de validación | Inspección del esquema de almacenamiento de los catálogos y prueba operativa de extensión. |
| HUs/TTH origen | HU-06 (Candidato a RNF de mantenibilidad: catálogo extensible sin cambios al código), HU-11 (Candidato a RNF de mantenibilidad: catálogo de textos de impacto operativo extensible), HU-12 (Candidato a RNF de mantenibilidad: catálogo de plantillas extensible). |
| DHUs relacionadas | No aplica de forma específica. |
| Prioridad MoSCoW sugerida | Should (la extensibilidad facilita iteración futura del producto sin requerir nuevos sprints de desarrollo cada vez que el equipo del producto identifica un texto a refinar). |
| Aplicabilidad | HU-06, HU-11, HU-12. |
| Notas | La forma concreta del almacenamiento (archivos JSON o YAML versionados, tabla de base de datos, recursos i18n) se cierra al implementar. El principio operativo es que los textos viven como datos modificables, no como cadenas literales en el código compilable. |

#### RNF-MNT-02 — Parametrización sin redeploy de tiempos y umbrales operativos

| Campo | Contenido |
|---|---|
| Identificador | RNF-MNT-02 |
| Característica ISO 25010:2023 | Maintainability / Modifiability |
| Subcaracterística | Modifiability |
| Descripción normativa | Los tiempos y umbrales operativos del sistema que pueden requerir ajuste durante la operación deben ser parametrizables sin necesidad de redespliegue. La parametrización aplica al menos a: el tiempo de auto-descarte de las notificaciones del Operador, el intervalo de agrupamiento de notificaciones encadenadas, y todos los parámetros operativos cubiertos por la interfaz de configuración del Administrador (HU-15). La modificación de estos parámetros surte efecto en operación dentro del umbral declarado en RNF-PERF-08. |
| Criterio de aceptación medible | Inspección documental de la lista de parámetros operativos y verificación de que cada uno es modificable sin redespliegue. Prueba operativa sobre un subconjunto representativo de parámetros: modificación, persistencia, observación del efecto en componentes consumidores sin reinicio del sistema. |
| Método de validación | Inspección documental y prueba automatizada sobre parámetros representativos. |
| HUs/TTH origen | HU-07 (Candidato a RNF de configurabilidad: tiempo de auto-descarte e intervalo de agrupamiento parametrizables sin redeploy), HU-15 (parametrización de los parámetros operativos del sistema cubiertos por la HU; declarado en CA-15.3 y en la nota técnica). |
| DHUs relacionadas | DHU-013 (sustrato técnico de F20 inglobado en HU-15), DHU-014 subsección C (selección concreta de parámetros). |
| Prioridad MoSCoW sugerida | Must (precondición de la operatividad del sistema sin ciclos de redespliegue). |
| Aplicabilidad | HU-07 (parámetros de notificación), HU-15 (parámetros operativos cubiertos por la interfaz de configuración). |
| Notas | Este RNF es complementario de RNF-PERF-08 (efecto sin redeploy en ≤ 30 segundos): RNF-PERF-08 declara el umbral temporal; RNF-MNT-02 declara la propiedad de que los parámetros viven como datos modificables. La unión de ambos sustenta la promesa operativa del Administrador de calibrar el sistema sin sprints adicionales. |

#### RNF-MNT-03 — Tolerancia configurable del indicador comparativo entre modelos

| Campo | Contenido |
|---|---|
| Identificador | RNF-MNT-03 |
| Característica ISO 25010:2023 | Maintainability / Modifiability |
| Subcaracterística | Modifiability |
| Descripción normativa | La tolerancia de empate aplicada al indicador comparativo entre el modelo predictivo principal y el modelo de respaldo (que determina cuándo el sistema reporta "Empate dentro de la tolerancia configurable" en lugar de "Modelo principal mejor" o "Modelo de respaldo mejor") debe ser configurable internamente por el equipo de desarrollo sin redespliegue. Por defecto, la tolerancia debe estar calibrada de forma que diferencias operacionalmente insignificantes no se reporten como superioridad de uno de los modelos. |
| Criterio de aceptación medible | La tolerancia es leída desde configuración (archivo, variable de entorno, parámetro de TTH-05 o equivalente) al inicializar el componente comparativo. Modificar la tolerancia y reiniciar el componente comparativo aplica el nuevo valor sin tocar código. |
| Método de validación | Inspección de la implementación y prueba operativa de cambio de la tolerancia. |
| HUs/TTH origen | HU-20 (CA-20.10 y Candidato a RNF de tolerancia parametrizada). |
| DHUs relacionadas | DHU-017 subsección D. |
| Prioridad MoSCoW sugerida | Could (HU-20 es MVP2). |
| Aplicabilidad | HU-20 (vista comparativa de métricas del modelo principal vs respaldo). |
| Notas | La tolerancia no se expone al Administrador en HU-15 (es configuración interna del componente comparativo, no parámetro operativo del sistema). La elección concreta del default se cierra al implementar conforme a la nota técnica de HU-20. |

---

### 3.9 Safety (RNF-SAF)

La característica de Safety, agregada como 9ª característica de calidad por ISO 25010:2023, cubre la capacidad del sistema de operar bajo condiciones operativas sin causar daño o riesgo inaceptable a personas, propiedad o el entorno. En CerebroVial este principio aplica de forma directa porque el sistema actúa sobre infraestructura de tránsito vehicular cuya operación incorrecta puede afectar a conductores y peatones. La sección agrupa tres familias: comportamiento fail safe ante fallo del control adaptativo, restricciones normativas operacionales del Manual MTC peruano, valores por defecto seguros desde el primer arranque del sistema.

#### RNF-SAF-01 — Comportamiento fail safe ante fallo del control adaptativo

| Campo | Contenido |
|---|---|
| Identificador | RNF-SAF-01 |
| Característica ISO 25010:2023 | Safety / Fail safe |
| Subcaracterística | Fail safe |
| Descripción normativa | Cuando el motor adaptativo del sistema no puede decidir o no responde, el sistema debe transitar a un estado de operación seguro definido por tiempos preconfigurados del semáforo conservadores, en lugar de detener completamente la operación sobre el semáforo o aplicar decisiones inconsistentes. Los tiempos preconfigurados conservadores deben ser tales que un operador de tráfico clásico aceptaría como razonables para una intersección típica, y deben respetar las restricciones normativas operacionales del Manual MTC peruano. El sistema debe mantener la intersección operativa con esos tiempos preconfigurados hasta que el motor adaptativo se recupere o el Administrador intervenga. |
| Criterio de aceptación medible | Prueba operativa de fallo inducido del motor adaptativo: el sistema transiciona a degradado nivel 3; los tiempos preconfigurados se aplican al semáforo; la intersección continúa operativa con esos tiempos; la transición y la operación se registran en los logs correspondientes. Al recuperarse el motor, el sistema retorna a operación normal sin acción manual. |
| Método de validación | Prueba automatizada de fallo inducido del motor adaptativo con validación end-to-end del comportamiento de transición y recuperación. |
| HUs/TTH origen | HU-12 (CA-12.1 — explicación del fallback aplicado), TTH-04 (CT-04.2 — Nivel 3 con tiempos preconfigurados), TTH-05 (CT-05.1 a CT-05.5 — esquema, endpoint, interfaz, latencia de consulta, valores por defecto seguros). |
| DHUs relacionadas | DHU-008 (distinción arquitectónica entre componente caído, modo degradado y lógica de fallback), DHU-010 (criterios para clasificar trabajo del Bloque C como TTH), DHU-013 (mantener TTH-05 íntegra, no dividir). |
| Prioridad MoSCoW sugerida | Must (precondición de la operación segura del sistema sobre infraestructura de tránsito). |
| Aplicabilidad | Cascada de fallback en degradado nivel 3 (TTH-04 + TTH-05), con propagación a HU-10 (alerta transversal) y HU-12 (explicación del fallback). |
| Notas | Este RNF es la materialización operativa del Objetivo 3 del Producto (garantizar continuidad operativa ante fallos) bajo la característica formal de Safety de ISO 25010:2023. La cascada completa (Nivel 1 sin detección de tráfico → Nivel 2 con predictor de respaldo → Nivel 3 con tiempos preconfigurados → Falla total como último recurso) está cubierta por RNF-REL-01 (robustez ante interrupción) en sus dimensiones de Reliability; este RNF se concentra en la dimensión de Safety que la cascada también cumple. La declaración como RNF-SAF, no como RNF-REL adicional, refleja la naturaleza de impacto sobre personas físicas que distingue Safety de Reliability en ISO 25010:2023. |

#### RNF-SAF-02 — Cumplimiento de restricciones normativas del Manual MTC peruano

| Campo | Contenido |
|---|---|
| Identificador | RNF-SAF-02 |
| Característica ISO 25010:2023 | Safety / Operational constraint |
| Subcaracterística | Operational constraint |
| Descripción normativa | Las decisiones del motor adaptativo aplicadas al semáforo deben respetar las restricciones operacionales documentadas en el Manual MTC peruano R.D. N.° 26-2024-MTC/18 y en manuales internacionales equivalentes (FHWA Traffic Signal Timing Manual 2008). Las restricciones incluyen tiempos mínimos y máximos de cada fase y tiempos de transición obligatorios entre fases. La capa de aplicación normativa del motor adaptativo (capa MTC) debe elevar, recortar o componer la salida de la capa de selección estratégica (Webster o Max Pressure) para garantizar el cumplimiento de las restricciones, incluso cuando la capa estratégica produciría valores fuera de rango. |
| Criterio de aceptación medible | Prueba automatizada sobre la capa MTC del motor adaptativo: inyectar salidas de Webster o Max Pressure con valores fuera de las constantes normativas; verificar que la salida final aplicada al semáforo respeta las cinco constantes declaradas (MIN_GREEN, MAX_GREEN, MIN_YELLOW, ALL_RED, MIN_PEDESTRIAN); verificar que cada elevación o recorte se registra como ajuste con descripción legible. |
| Método de validación | Tests automatizados de la capa MTC con escenarios sintéticos que ejercen las cinco constantes. |
| HUs/TTH origen | TTH-10 (CT-10.6 — cinco constantes normativas con origen en el Manual MTC peruano y FHWA; CT-10.7 — tres operaciones determinísticas de la capa MTC: elevar, recortar, componer; CT-10.8 — output con lista de ajustes; CT-10.9 — persistencia de tiempos antes y después de MTC con lista de ajustes). |
| DHUs relacionadas | DHU-014 subsección C (parámetros internos de las estrategias internos al sistema en MVP1). |
| Prioridad MoSCoW sugerida | Must (cumplimiento normativo es regla inegociable de un sistema que opera sobre tránsito vehicular). |
| Aplicabilidad | TTH-10 (motor adaptativo), con propagación a la persistencia de decisiones consumida por HU-08. |
| Notas | El motor adaptativo no inventa límites operativos; replica restricciones documentadas en normativa vigente y en manuales operativos internacionales, conforme a la trazabilidad regulatoria declarada en CT-10.6 y desarrollada en la sección 6 de `CONTROL.md`. Esta trazabilidad sustenta la defensa académica del componente como aporte de ingeniería respetuoso del marco regulatorio. |

#### RNF-SAF-03 — Valores por defecto seguros desde el primer arranque

| Campo | Contenido |
|---|---|
| Identificador | RNF-SAF-03 |
| Característica ISO 25010:2023 | Safety / Fail safe |
| Subcaracterística | Fail safe |
| Descripción normativa | Los valores por defecto de los parámetros operativos del sistema (umbrales, ventanas temporales, frecuencias) y los tiempos preconfigurados del degradado nivel 3 deben ser tales que el sistema sea operativo y seguro desde el primer arranque o tras una restauración de configuración, sin requerir ajuste manual previo del Administrador. La existencia de valores por defecto seguros garantiza que el sistema no entre en condición patológica por configuración ausente o inválida. |
| Criterio de aceptación medible | Prueba operativa de arranque del sistema sobre una base de datos sin configuración previa: el sistema arranca sin errores; los componentes operativos consumen valores por defecto seguros; el semáforo opera con tiempos por defecto conservadores. Prueba operativa de restauración de configuración por el Administrador (CA-15.9): tras la restauración, el sistema opera correctamente sin requerir ajuste manual adicional. |
| Método de validación | Prueba automatizada de arranque sobre base de datos vacía y prueba operativa de restauración de configuración. |
| HUs/TTH origen | HU-15 (CA-15.4 — valores por defecto referenciados; CA-15.9 — restauración a valores por defecto; Candidato a RNF de seguridad operativa: los valores por defecto deben ser tales que el sistema sea operativo y seguro desde el primer arranque o tras una restauración), TTH-05 (CT-05.5 — valores por defecto seguros conservadores cuando no existe configuración explícita). |
| DHUs relacionadas | DHU-013, DHU-014. |
| Prioridad MoSCoW sugerida | Must (precondición de la operatividad y seguridad del sistema desde el primer arranque). |
| Aplicabilidad | HU-15 (parámetros operativos del sistema), TTH-05 (tiempos preconfigurados del degradado nivel 3). |
| Notas | La elección de "valores por defecto seguros" es decisión de diseño con implicaciones operativas: valores demasiado conservadores degradan la utilidad del sistema en operación normal; valores demasiado agresivos pueden producir condiciones inseguras en escenarios límite. El criterio de "un operador de tráfico clásico aceptaría como razonables" actúa como referencia operativa para la calibración. |

---

### 3.3 Compatibility (RNF-COM)

La característica de Compatibility cubre la capacidad del sistema de co-existir con otros componentes y de intercambiar información con sistemas externos a través de contratos estables. En CerebroVial agrupa dos familias: co-existencia de los componentes del monolito modular, e interoperabilidad con fuentes de datos del estado del tráfico a través de un constructo unificado.

#### RNF-COM-01 — Co-existencia de los componentes del monolito modular

| Campo | Contenido |
|---|---|
| Identificador | RNF-COM-01 |
| Característica ISO 25010:2023 | Compatibility / Co-existence |
| Subcaracterística | Co-existence |
| Descripción normativa | Los componentes del sistema declarados como módulos del monolito modular (módulo de API de gestión central, módulo del dispositivo de borde, módulo de servicio de predicción, frontend, base de datos) deben co-existir en el mismo entorno de despliegue sin interferir entre sí en términos de puertos, recursos compartidos, espacios de nombres ni estado. La comunicación entre componentes debe realizarse por nombres internos resolvibles dentro del entorno de despliegue, sin requerir configuración manual de hosts por parte del operador del despliegue. |
| Criterio de aceptación medible | Prueba operativa de arranque conjunto de todos los componentes en el mismo entorno: los componentes inicializan sin conflictos de puertos; cada componente accede a los demás por nombres internos sin configuración manual; el sistema completo opera end-to-end. |
| Método de validación | Prueba automatizada de despliegue conjunto en máquina limpia siguiendo el procedimiento documentado en el quickstart. |
| HUs/TTH origen | TTH-02 (CT-02.1 a CT-02.6 — declaración de servicios, puertos, dependencias, volúmenes persistentes, documento de quickstart, archivo de variables de entorno de ejemplo). |
| DHUs relacionadas | No aplica de forma específica; consume D-001 (monolito modular como decisión de arquitectura) y D-003 (deploy local con Docker). |
| Prioridad MoSCoW sugerida | Must (precondición de cualquier validación end-to-end del sistema). |
| Aplicabilidad | TTH-02 (arquitectura del despliegue local). |
| Notas | La co-existencia es propiedad del despliegue, no del código fuente: los componentes pueden estar en el mismo repositorio (D-001 monolito modular) pero deben co-existir como servicios separados en el despliegue. La separación de despliegue habilita la portabilidad futura a Pi física (`edge_device`) y servidor central (resto del sistema) declarada en D-004. |

#### RNF-COM-02 — Interoperabilidad mediante constructo unificado de nivel de congestión

| Campo | Contenido |
|---|---|
| Identificador | RNF-COM-02 |
| Característica ISO 25010:2023 | Compatibility / Interoperability |
| Subcaracterística | Interoperability |
| Descripción normativa | El sistema debe utilizar un constructo unificado de nivel de congestión expresado como escala ordinal 0-5 para representar el estado predicho y observado del tráfico, independientemente de la fuente concreta de datos que alimenta cada modo operativo. Esta abstracción permite la intercambiabilidad de fuentes (entorno de simulación en validación cuantitativa; fuentes de datos reales en operación hipotética) sin requerir reentrenamiento del modelo predictivo ni modificación de las vistas operativas del Operador. |
| Criterio de aceptación medible | Inspección de la implementación: las vistas del Operador y el modelo predictivo consumen el nivel de congestión 0-5 como tipo de dato unificado, no la variable bruta de ninguna fuente específica. Cambio de fuente operativa (de la fuente A a la fuente B en condiciones de prueba): el sistema continúa operando sin modificar las vistas ni reentrenar el modelo, asumiendo que la nueva fuente produce el mismo constructo 0-5. |
| Método de validación | Inspección documental del esquema de datos y prueba operativa de cambio de fuente sobre escenario sintético. |
| HUs/TTH origen | HU-03 (CA-03.1 — nivel de congestión en escala 0-5), TTH-07 (CT-07.3 — cálculo del jam level 0-5 derivado del ratio velocidad/free-flow), TTH-09 (CT-09.4 — endpoint que devuelve ratio continuo y nivel discreto 0-5). |
| DHUs relacionadas | DHU-006 (HUs agnósticas a implementación; la escala 0-5 es excepción explícitamente declarada). |
| Prioridad MoSCoW sugerida | Should (la portabilidad de fuente es valor declarado del diseño del sistema, fundamental para la extensibilidad futura). |
| Aplicabilidad | HU-03, HU-04 (vista combinada que consume el constructo 0-5), TTH-07, TTH-09. |
| Notas | La escala 0-5 está documentada en D-009 con su origen como constructo de Waze; el documento RF/RNF no nombra Waze en el cuerpo normativo conforme a DHU-006, pero la trazabilidad técnica vive en D-009. Esta abstracción es uno de los aportes arquitectónicos del sistema y sustenta su extensibilidad futura. |

---

### 3.8 Flexibility (RNF-FLX)

La característica de Flexibility, renombrada desde "Portability" por ISO 25010:2023, cubre la capacidad del sistema de ser adaptado a entornos de despliegue distintos, escalado a volúmenes operativos mayores, instalado en condiciones diversas y de tener componentes reemplazados por otros equivalentes. En CerebroVial agrupa tres familias: portabilidad del despliegue local con contenedores, reemplazabilidad del modelo predictivo y escalabilidad arquitectónica con restricciones declaradas.

#### RNF-FLX-01 — Portabilidad del despliegue local mediante contenedores

| Campo | Contenido |
|---|---|
| Identificador | RNF-FLX-01 |
| Característica ISO 25010:2023 | Flexibility / Installability |
| Subcaracterística | Installability |
| Descripción normativa | El sistema debe ser desplegable localmente en una máquina limpia con un solo comando estándar de orquestación de contenedores, sin requerir configuración manual de hosts, instalación previa de dependencias del lenguaje o configuración manual de la base de datos. El procedimiento de despliegue debe estar documentado en un archivo de quickstart legible por un nuevo miembro del equipo o por un evaluador académico, permitiendo replicar el sistema funcional sin asistencia del autor. |
| Criterio de aceptación medible | Prueba operativa en máquina limpia siguiendo el procedimiento del quickstart: el sistema queda funcional end-to-end sin asistencia. El frontend es accesible vía la URL local declarada; la API responde a su endpoint de salud; la base de datos acepta conexiones desde la API. |
| Método de validación | Prueba en máquina limpia (real o virtual) por una persona ajena a la implementación, ejecutando solo el procedimiento del quickstart. |
| HUs/TTH origen | TTH-02 (CT-02.2 — arranque limpio con un solo comando; CT-02.5 — documento de quickstart suficiente para nuevo miembro o evaluador; CT-02.6 — archivo de variables de entorno de ejemplo). |
| DHUs relacionadas | No aplica de forma específica; consume D-003 (deploy local con Docker, no cloud en MVP1) y D-004 (Pi física como demostración conceptual, no entrega). |
| Prioridad MoSCoW sugerida | Must (precondición de la defensa académica del sistema en máquina limpia). |
| Aplicabilidad | TTH-02 (arquitectura del despliegue local), con propagación a la documentación del proyecto. |
| Notas | La portabilidad del despliegue es propiedad declarada del alcance MVP1 conforme a D-003: el sistema se despliega localmente, no en cloud. La declaración arquitectónica de que el sistema sería desplegable en Pi física o en servidor central queda como demostración conceptual conforme a D-004, sin entrega de hardware. |

#### RNF-FLX-02 — Reemplazabilidad del modelo predictivo

| Campo | Contenido |
|---|---|
| Identificador | RNF-FLX-02 |
| Característica ISO 25010:2023 | Flexibility / Replaceability |
| Subcaracterística | Replaceability |
| Descripción normativa | El modelo predictivo principal del sistema debe ser reemplazable por un modelo de respaldo equivalente cuando el primero no responde, sin requerir modificación del contrato del endpoint de predicciones consumido por el motor adaptativo y por las vistas del Operador. El modelo de respaldo debe producir respuestas con el mismo formato de output que el modelo principal (cuatro direcciones, ratio continuo y nivel discreto por predicción), preservando la portabilidad del sistema ante la indisponibilidad del modelo principal. |
| Criterio de aceptación medible | Inspección de los outputs de ambos modelos: el modelo de respaldo produce respuestas con el mismo esquema que el modelo principal. Prueba operativa de fallo inducido del modelo principal: el sistema activa el modelo de respaldo (Nivel 2 de la cascada de TTH-04); los consumidores del endpoint de predicciones reciben respuestas con el mismo formato sin requerir adaptación. |
| Método de validación | Inspección documental del contrato del endpoint y prueba operativa de fallback al modelo de respaldo. |
| HUs/TTH origen | TTH-04 (CT-04.2 — Nivel 2 con predictor de respaldo), TTH-09 (CT-09.8 — error HTTP estándar cuando el modelo principal no responde; nota técnica sobre preservación del modelo de respaldo con mismo formato de output). |
| DHUs relacionadas | No aplica de forma específica; consume D-002 (modelo predictivo es RNN con fallback temporal) y D-006 (GRU univariado refina D-002). |
| Prioridad MoSCoW sugerida | Must (precondición del Nivel 2 de la cascada de fallback). |
| Aplicabilidad | TTH-09 (modelo predictivo) con propagación a la cascada de fallback de TTH-04. |
| Notas | La reemplazabilidad del modelo predictivo es propiedad declarada del diseño del sistema: el modelo de respaldo se preserva sin modificación funcional, accesible internamente para que TTH-04 lo invoque cuando el modelo principal no responde. Esta reemplazabilidad sustenta el Objetivo 3 del Producto (continuidad operativa) y habilita la comparativa de HU-20 (MVP2). |

#### RNF-FLX-03 — Escalabilidad arquitectónica con restricciones declaradas

| Campo | Contenido |
|---|---|
| Identificador | RNF-FLX-03 |
| Característica ISO 25010:2023 | Flexibility / Scalability |
| Subcaracterística | Scalability |
| Descripción normativa | La arquitectura del sistema debe permitir extensiones declaradas con restricciones explícitas. El crecimiento dentro de una misma intersección a un número mayor de accesos es escalamiento trivial mediante entrenamiento independiente de modelos univariados adicionales. El crecimiento a múltiples intersecciones interrelacionadas como red urbana no es escalamiento trivial de la arquitectura actual: requiere arquitecturas espacio-temporales que capturen dependencia espacial entre intersecciones, declaradas como trabajo futuro del proyecto. Esta distinción debe ser explícita en la documentación del sistema. |
| Criterio de aceptación medible | Inspección documental: la arquitectura del modelo predictivo permite escalamiento trivial a más accesos en la misma intersección sin rediseño. La extensión a múltiples intersecciones está declarada como trabajo futuro arquitectónico, no como escalamiento lineal del componente actual. |
| Método de validación | Revisión documental del diseño y de la nota técnica correspondiente en TTH-09. |
| HUs/TTH origen | TTH-09 (nota técnica sobre topología de cuatro direcciones como limitación de alcance MVP1 y extensión a múltiples intersecciones como trabajo futuro arquitectónico no lineal). |
| DHUs relacionadas | No aplica de forma específica. |
| Prioridad MoSCoW sugerida | Won't (la extensión a múltiples intersecciones es trabajo futuro declarado fuera del alcance del proyecto académico, conforme a F37 de la lista de Trabajos Futuros). |
| Aplicabilidad | TTH-09 (modelo predictivo). El alcance MVP1 cubre una intersección de cuatro accesos. La declaración explícita de las dos vías de escalamiento (trivial vs arquitectónico) sustenta la defensa académica del alcance y la honestidad sobre las extensiones futuras. |
| Notas | Este RNF documenta una restricción del alcance, no una capacidad. La inclusión explícita en el documento RF/RNF preserva la trazabilidad: si en el futuro se aborda la extensión a red urbana, este RNF se modifica para reflejar el cambio de alcance, y el componente preservado en el código actual sirve como punto de partida natural de la nueva arquitectura. |

---

### Cierre de la sección 3

La sección 3 cubre las 9 características de calidad de ISO/IEC 25010:2023 con los RNF derivados de los Candidatos a RNF de las 21 HUs operativas y de los criterios técnicos relevantes de las 11 TTH. El catálogo total de RNF al cierre de esta sección es el siguiente:

| Subsección | Característica ISO 25010:2023 | Código | Cantidad de RNF |
|---|---|---|---|
| 3.1 | Functional Suitability | RNF-FUN | 6 |
| 3.2 | Performance Efficiency | RNF-PERF | 13 |
| 3.3 | Compatibility | RNF-COM | 2 |
| 3.4 | Interaction Capability | RNF-INT | 7 |
| 3.5 | Reliability | RNF-REL | 9 |
| 3.6 | Security | RNF-SEC | 7 |
| 3.7 | Maintainability | RNF-MNT | 3 |
| 3.8 | Flexibility | RNF-FLX | 3 |
| 3.9 | Safety | RNF-SAF | 3 |
| **Total** | | | **53 RNF** |

Cada RNF está clasificado bajo una característica primaria de ISO 25010:2023 con su código de tres letras (FUN, PERF, COM, INT, REL, SEC, MNT, FLX, SAF). Los RNF con doble característica (primaria + secundaria) están declarados explícitamente en el campo "Característica ISO 25010:2023" de su tabla y mantienen su código primario en el identificador.

## 2. Catálogo de Requisitos Funcionales

Los RF se presentan agrupados por familia funcional según la composición natural del backlog declarada en la sección 1.2 y formalizada en DHU-019 subsección E. Cada RF sigue la plantilla unificada cerrada en DHU-019 subsección D. La derivación desde los CAs aplica composición transversal: un RF puede agrupar CAs de varias HUs cuando describen el mismo comportamiento del sistema desde perspectivas distintas.

**Las 7 familias funcionales:**

1. Control de acceso y autenticación.
2. Monitoreo operativo en tiempo real.
3. Decisiones del motor adaptativo.
4. Predicción de tráfico.
5. Soporte técnico y configuración del sistema.
6. Reportería ejecutiva.
7. Soporte al Operador y trazabilidad de incidentes.

### 2.1 Familia 1 — Control de acceso y autenticación

#### RF-001 — Autenticación al sistema

| Campo | Contenido |
|---|---|
| Identificador | RF-001 |
| Familia funcional | Control de acceso y autenticación |
| Descripción | El sistema permite a los usuarios autenticarse mediante credenciales personales. Las solicitudes a recursos protegidos requieren un token de sesión válido emitido tras autenticación exitosa. Los usuarios sin sesión activa son redirigidos al flujo de inicio de sesión; los usuarios con sesión expirada son desconectados con mensaje informativo. |
| HUs origen | HU-01 |
| CAs origen | CA-01.5, CA-01.6 |
| TTH relacionadas | TTH-01 |
| Persona beneficiaria | Transversal a las tres Personas (Operador, Gerente, Administrador) |
| Objetivo del producto | Soporte transversal |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-SEC-02 (autenticación con bcrypt y JWT) |
| Notas | La autenticación es precondición transversal de todas las HUs operativas. El login no es modelado como HU sino como TTH-01 conforme a DHU-001. |

#### RF-002 — Control de acceso por rol

| Campo | Contenido |
|---|---|
| Identificador | RF-002 |
| Familia funcional | Control de acceso y autenticación |
| Descripción | El sistema reconoce el rol del usuario autenticado y le presenta las vistas correspondientes a su rol, restringiendo el acceso a los recursos fuera de su ámbito. Cada uno de los tres roles del producto (Operador, Gerente, Administrador) accede a un conjunto coherente de vistas y endpoints; los intentos de acceso cruzado son rechazados tanto desde el frontend (no se renderizan rutas no accesibles) como desde el backend (respuesta HTTP 403). |
| HUs origen | HU-01 |
| CAs origen | CA-01.1, CA-01.2, CA-01.3, CA-01.4 |
| TTH relacionadas | No aplica |
| Persona beneficiaria | Transversal a las tres Personas |
| Objetivo del producto | Soporte transversal |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-SEC-03 (control de acceso por rol como propiedad de Confidentiality), RNF-SEC-04 (no filtración en respuestas 403), RNF-INT-07 (ocultación de rutas no accesibles) |
| Notas | El control de acceso por rol es transversal a las 21 HUs operativas. Cada HU declara explícitamente en su CA correspondiente qué roles tienen acceso; la matriz agregada se materializa al implementar. |

### 2.2 Familia 2 — Monitoreo operativo en tiempo real

#### RF-003 — Presentación del estado actual del tráfico por acceso

| Campo | Contenido |
|---|---|
| Identificador | RF-003 |
| Familia funcional | Monitoreo operativo en tiempo real |
| Descripción | El sistema expone, por cada acceso de la intersección, las variables observadas del estado actual del tráfico: flujo vehicular y longitud de cola. Los valores se actualizan automáticamente sin recarga manual. Los valores de cola se acompañan de un indicador visual con umbrales que distinguen niveles operativos relevantes. |
| HUs origen | HU-02 |
| CAs origen | CA-02.1, CA-02.2, CA-02.3 |
| TTH relacionadas | TTH-08 (módulo de visión computacional como fuente operacional hipotética), TTH-07 (entorno de simulación como fuente operacional en MVP1) |
| Persona beneficiaria | Operador |
| Objetivo del producto | 1, 3 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-01 (actualización ≤ 5 s), RNF-REL-01 (robustez ante interrupción modo A), RNF-INT-01 (umbrales visuales legibles a primer golpe de vista), RNF-INT-02 (accesibilidad de los códigos visuales) |
| Notas | El sistema es agnóstico a la fuente concreta de datos conforme a DHU-006: en MVP1 la fuente operativa son corridas del entorno de simulación; en operación hipotética sería el módulo de visión computacional. Los umbrales concretos del indicador visual de cola son configurables vía RF-011. |

#### RF-004 — Presentación de la predicción de congestión por acceso

| Campo | Contenido |
|---|---|
| Identificador | RF-004 |
| Familia funcional | Monitoreo operativo en tiempo real |
| Descripción | El sistema expone, por cada acceso de la intersección, la predicción del nivel de congestión en escala 0-5 proyectada hasta el horizonte temporal configurado. Cuando el nivel predicho supera el umbral configurado, el acceso correspondiente se resalta visualmente. Los valores se actualizan automáticamente cuando el modelo predictivo genera nuevas predicciones. |
| HUs origen | HU-03 |
| CAs origen | CA-03.1, CA-03.2, CA-03.3 |
| TTH relacionadas | TTH-09 (modelo predictivo) |
| Persona beneficiaria | Operador |
| Objetivo del producto | 1 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-01 (actualización ≤ 5 s), RNF-REL-01 (robustez ante interrupción modo B), RNF-FUN-02 (calidad del modelo predictivo), RNF-COM-02 (constructo unificado 0-5) |
| Notas | El horizonte de predicción y el umbral de resaltado son configurables vía RF-011. La escala 0-5 está documentada en D-009; este RF es agnóstico al origen del constructo conforme a DHU-006. |

#### RF-005 — Vista combinada del estado actual y la predicción

| Campo | Contenido |
|---|---|
| Identificador | RF-005 |
| Familia funcional | Monitoreo operativo en tiempo real |
| Descripción | El sistema presenta de forma integrada, en una única vista, la información cubierta por RF-003 y RF-004, con alineación temporal explícita entre el estado actual y la predicción. Cuando el estado actual está dentro de la normalidad pero la predicción anticipa congestión, el sistema resalta la discrepancia para alertar al Operador del problema anticipado. |
| HUs origen | HU-04 |
| CAs origen | CA-04.1, CA-04.2, CA-04.3 |
| TTH relacionadas | TTH-08, TTH-09 (consumo indirecto vía RF-003 y RF-004) |
| Persona beneficiaria | Operador |
| Objetivo del producto | 1 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-01 (actualización ≤ 5 s), RNF-REL-01 (robustez con dos modos independientes), RNF-INT-01 (alineación temporal visualmente inmediata), RNF-INT-02 |
| Notas | Esta vista compone visualmente las dos fuentes consumidas por RF-003 y RF-004 sin duplicar lógica de obtención de datos. La integración es a nivel de presentación. |

### 2.3 Familia 3 — Decisiones del motor adaptativo

#### RF-006 — Presentación de la estrategia de control activa

| Campo | Contenido |
|---|---|
| Identificador | RF-006 |
| Familia funcional | Decisiones del motor adaptativo |
| Descripción | El sistema expone al Operador, en tiempo real, qué estrategia de control semafórico está aplicando actualmente el motor adaptativo, los parámetros activos de esa estrategia (tiempos de verde por acceso) y el momento en que la estrategia se activó. Los nombres de estrategias presentados al Operador son autoexplicativos, sin referenciar identificadores técnicos internos. |
| HUs origen | HU-05 |
| CAs origen | CA-05.1, CA-05.2, CA-05.3 |
| TTH relacionadas | TTH-10 (motor adaptativo) |
| Persona beneficiaria | Operador |
| Objetivo del producto | 1, 2 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-01 (actualización ≤ 5 s), RNF-REL-01 (robustez modo B), RNF-INT-01 (nombres autoexplicativos), RNF-SAF-02 (cumplimiento normativo MTC en los tiempos aplicados) |
| Notas | El RF es agnóstico a la implementación del motor adaptativo conforme a DHU-006. Los algoritmos concretos (Webster, Max Pressure, MTC) viven en TTH-10. |

#### RF-007 — Explicación legible de la razón de selección de estrategia

| Campo | Contenido |
|---|---|
| Identificador | RF-007 |
| Familia funcional | Decisiones del motor adaptativo |
| Descripción | El sistema expone al Operador una explicación en lenguaje legible de por qué el motor adaptativo seleccionó la estrategia activa, integrando los valores del estado del tráfico que justifican la decisión. La explicación se construye desde un catálogo curado de plantillas con sustitución de variables; las combinaciones no cubiertas por una plantilla específica activan un texto genérico de respaldo. |
| HUs origen | HU-06 |
| CAs origen | CA-06.1, CA-06.2, CA-06.3 |
| TTH relacionadas | TTH-10 |
| Persona beneficiaria | Operador |
| Objetivo del producto | 2 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-01 (actualización ≤ 5 s), RNF-REL-01 (robustez modo B), RNF-INT-05 (comprensibilidad sin formación técnica), RNF-FUN-04 (cobertura del catálogo de plantillas), RNF-MNT-01 (extensibilidad del catálogo), RNF-INT-04 (coherencia textual con HU-11 y HU-12) |
| Notas | El catálogo de plantillas vive como datos de configuración, no como cadenas hardcoded en el código. Las plantillas no usan procesamiento de lenguaje natural ni explicabilidad de IA; son textos curados por humanos. |

#### RF-008 — Notificación temporal de cambios de estrategia

| Campo | Contenido |
|---|---|
| Identificador | RF-008 |
| Familia funcional | Decisiones del motor adaptativo |
| Descripción | El sistema notifica al Operador, mediante una notificación visual temporal poco intrusiva, cada cambio de estrategia que aplica el motor adaptativo. La notificación indica la hora del cambio, la estrategia anterior, la estrategia nueva y una razón breve. Las notificaciones se auto-descartan tras un tiempo configurado; los cambios encadenados dentro de un intervalo corto se agrupan para no saturar al Operador. |
| HUs origen | HU-07 |
| CAs origen | CA-07.1, CA-07.2, CA-07.3, CA-07.4 |
| TTH relacionadas | TTH-10 |
| Persona beneficiaria | Operador |
| Objetivo del producto | 1, 2 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-01 (latencia entre cambio y notificación ≤ 5 s), RNF-REL-01 (robustez modo B aplicado al canal de eventos), RNF-INT-01 (no interferencia con paneles principales), RNF-MNT-02 (tiempo de auto-descarte y agrupamiento parametrizables) |
| Notas | La notificación es efímera por diseño. Una notificación perdida no es crítica: el cambio queda registrado en el historial (RF-009) y la estrategia vigente se ve siempre en RF-006. |

#### RF-009 — Consulta histórica de decisiones del motor

| Campo | Contenido |
|---|---|
| Identificador | RF-009 |
| Familia funcional | Decisiones del motor adaptativo |
| Descripción | El sistema permite al Operador consultar el historial de decisiones del motor adaptativo en periodos pasados. Cada entrada del historial incluye el momento de la decisión, la estrategia aplicada, los parámetros calculados y la razón legible. El historial es auditable y se persiste de forma durable e inmutable en el momento en que cada decisión se produce. |
| HUs origen | HU-08 |
| CAs origen | CA-08.1, CA-08.2, CA-08.3, CA-08.4, CA-08.5 |
| TTH relacionadas | TTH-10 (CT-10.9 como materialización del sustrato técnico de F31 inglobada) |
| Persona beneficiaria | Operador (consulta), Administrador (acceso técnico) |
| Objetivo del producto | 2 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-02 (apertura ≤ 2 s), RNF-REL-03 (la operación del motor no se detiene por fallos del registro), RNF-REL-04 (durabilidad del registro), RNF-SEC-01 (inmutabilidad del registro) |
| Notas | El sustrato técnico de F31 (persistencia de decisiones del motor) está inglobado en CA-08.1 conforme al cierre del Bloque A. CT-10.9 materializa la persistencia desde el motor adaptativo. |

### 2.4 Familia 4 — Predicción de tráfico

La familia de Predicción de tráfico está cubierta por RF-004 (presentación al Operador) y RF-005 (vista combinada) declarados en la Familia 2, sumados al sustrato de TTH-09 (modelo predictivo) y TTH-11 (calibración de hiperparámetros temporales). El sustrato técnico de la familia se materializa en las TTH; los RF visibles al usuario son los de la Familia 2 que consumen la predicción.

La Familia 4 no introduce RF adicionales; queda documentada como familia cuya superficie funcional visible al usuario está cubierta por RF-004, RF-005 y RF-013 (cuando el modelo principal y de respaldo se comparan). La inclusión de la Familia 4 en este catálogo preserva la trazabilidad declarada en la sección 1.

### 2.5 Familia 5 — Soporte técnico y configuración del sistema

#### RF-010 — Vista técnica de salud de los componentes

| Campo | Contenido |
|---|---|
| Identificador | RF-010 |
| Familia funcional | Soporte técnico y configuración del sistema |
| Descripción | El sistema expone al Administrador una vista técnica detallada del estado de cada componente operativo (módulo de detección de tráfico, módulo predictivo, motor adaptativo, componente de explicación, registro de eventos), incluyendo nombre legible, estado cualitativo (OK / Degradado / Fuera de servicio), timestamp del último cambio de estado, identificador interno, latencia de respuesta en la última evaluación de salud, indicador de fallos recientes y timestamp de la última evaluación exitosa. |
| HUs origen | HU-13 |
| CAs origen | CA-13.1, CA-13.2, CA-13.3, CA-13.4, CA-13.5, CA-13.6 |
| TTH relacionadas | TTH-04 (CT-04.5 con ampliación a 7 campos por cierre del Bloque D) |
| Persona beneficiaria | Administrador |
| Objetivo del producto | 3 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-01 (actualización ≤ 5 s), RNF-REL-01 (robustez modo B), RNF-SEC-03 (control de acceso al rol Administrador), RNF-SEC-05 (segregación de presentación entre HU-11 del Operador y HU-13 del Administrador), RNF-INT-01, RNF-INT-02, RNF-FUN-01 (componentes sin historial se muestran distinguibles) |
| Notas | La vista comparte el sustrato técnico con RF-014 (vista simplificada del Operador) consumiendo el mismo endpoint con presentación distinta conforme a DHU-013. Los campos técnicos adicionales son visibles solo al Administrador por control de acceso a la ruta de la vista. |

#### RF-011 — Configuración de parámetros operativos del sistema

| Campo | Contenido |
|---|---|
| Identificador | RF-011 |
| Familia funcional | Soporte técnico y configuración del sistema |
| Descripción | El sistema permite al Administrador consultar y modificar los parámetros operativos del sistema desde una vista dedicada, organizados en tres familias funcionales (visualización del estado del tráfico, predicción y evaluación del modelo, monitor de salud del sistema). Las modificaciones se persisten de forma durable, son auditadas con identidad del autor y marca de tiempo, y surten efecto en operación sin requerir redespliegue. El sistema soporta la concurrencia entre Administradores con last-write-wins y advertencia explícita al segundo modificador. La restauración a valores por defecto seguros está disponible y queda auditada. |
| HUs origen | HU-15 |
| CAs origen | CA-15.1, CA-15.2, CA-15.3, CA-15.4, CA-15.5, CA-15.6, CA-15.7, CA-15.8, CA-15.9, CA-15.10, CA-15.11, CA-15.12, CA-15.13 |
| TTH relacionadas | No aplica directamente (sustrato técnico inglobado en CAs de la HU); TTH-05 cubre la familia separada de tiempos preconfigurados del degradado nivel 3 |
| Persona beneficiaria | Administrador |
| Objetivo del producto | Soporte técnico |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-03 (apertura ≤ 2 s), RNF-PERF-08 (efecto sin redeploy ≤ 30 s), RNF-REL-04 (durabilidad de parámetros y auditoría), RNF-REL-07 (manejabilidad de concurrencia), RNF-SEC-01 (inmutabilidad del registro de auditoría), RNF-SEC-03 (control de acceso al rol Administrador), RNF-SEC-06 (validación dual frontend y backend), RNF-MNT-02 (parametrización sin redeploy), RNF-SAF-03 (valores por defecto seguros) |
| Notas | Los parámetros operativos cubiertos en MVP1 están declarados en DHU-014 subsección C. Los parámetros internos de las estrategias del motor adaptativo quedan internos al sistema en MVP1. La configuración de tiempos preconfigurados del degradado nivel 3 vive en TTH-05, con propósito y vocabulario distintos. |

#### RF-012 — Vista de métricas de desempeño del modelo predictivo

| Campo | Contenido |
|---|---|
| Identificador | RF-012 |
| Familia funcional | Soporte técnico y configuración del sistema |
| Descripción | El sistema expone al Administrador una vista de las métricas de desempeño del modelo predictivo principal, calculadas sobre una ventana temporal configurable comparando predicciones recientes contra niveles de congestión efectivamente observados. Las métricas presentadas son: MAE y RMSE sobre el ratio continuo, accuracy sobre el nivel discreto 0-5, y matriz de confusión 6×6. Cada métrica está acompañada de un tooltip de ayuda con su definición operacional. |
| HUs origen | HU-14 |
| CAs origen | CA-14.1, CA-14.2, CA-14.3, CA-14.4, CA-14.5, CA-14.6, CA-14.7, CA-14.8, CA-14.9, CA-14.10, CA-14.11, CA-14.12 |
| TTH relacionadas | TTH-09 (registro de predicciones como sustrato de cálculo) |
| Persona beneficiaria | Administrador |
| Objetivo del producto | 2 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-09 (latencia de cálculo ≤ 30 s), RNF-REL-04 (durabilidad del registro de predicciones), RNF-SEC-01 (inmutabilidad del registro), RNF-SEC-03 (control de acceso al rol Administrador), RNF-FUN-01 (manejo de ventanas sin datos suficientes), RNF-FUN-02 (calidad del modelo predictivo), RNF-INT-03 (tooltips autoexplicativos), RNF-INT-02 (accesibilidad de los tooltips y de la diagonal de la matriz) |
| Notas | El sustrato técnico de F18 está inglobado en CA-14.1 a CA-14.4 conforme a DHU-013. La ventana temporal de cálculo es parámetro configurable vía RF-011 conforme a DHU-014 subsección F. |

#### RF-013 — Vista comparativa de métricas del modelo principal vs modelo de respaldo

| Campo | Contenido |
|---|---|
| Identificador | RF-013 |
| Familia funcional | Soporte técnico y configuración del sistema |
| Descripción | El sistema permite al Administrador comparar simultáneamente las métricas de desempeño del modelo predictivo principal contra las del modelo de respaldo, sobre la misma ventana temporal y los mismos eventos del registro. La vista presenta cuatro paneles comparativos (uno por métrica) con valores de ambos modelos lado a lado y un indicador comparativo que comunica cuál modelo es mejor o si la diferencia está dentro de la tolerancia configurable. |
| HUs origen | HU-20 |
| CAs origen | CA-20.1 a CA-20.19 |
| TTH relacionadas | TTH-09 (extensión del registro de predicciones para persistir también predicciones del modelo de respaldo) |
| Persona beneficiaria | Administrador |
| Objetivo del producto | 2 |
| Prioridad MoSCoW sugerida | Could (MVP2) |
| RNF asociados | RNF-PERF-09 (latencia de cálculo), RNF-PERF-13 (no degradación por ejecución paralela del respaldo), RNF-REL-04 (durabilidad del registro extendido), RNF-SEC-01 (inmutabilidad), RNF-SEC-03 (control de acceso al rol Administrador), RNF-FUN-01 (manejo de ventanas sin datos suficientes), RNF-FUN-03 (comparabilidad rigurosa: pares incompletos no entran al cálculo), RNF-MNT-03 (tolerancia de empate configurable), RNF-INT-04 (coherencia con RF-012), RNF-INT-02 |
| Notas | El registro de predicciones se extiende inglobando la persistencia paralela del modelo de respaldo en CA-20.1 a CA-20.4 conforme a DHU-017 subsección D. El esquema de CT-09.5 no se modifica; la extensión consiste en escribir adicionalmente predicciones del modelo de respaldo con identificador de modelo distinto. |

### 2.6 Familia 6 — Reportería ejecutiva

#### RF-014 — Consulta de KPIs operativos sobre periodo seleccionable

| Campo | Contenido |
|---|---|
| Identificador | RF-014 |
| Familia funcional | Reportería ejecutiva |
| Descripción | El sistema permite al Gerente consultar cuatro KPIs operativos agregados (tiempo promedio de espera por vehículo, longitud máxima de cola por dirección, throughput de la intersección, demora promedio acumulada por vehículo) sobre un periodo elegido entre presets predefinidos o rango personalizado. Los KPIs se presentan en una vista única con cards numéricas, controles de desglose opcional por dirección, tooltips de definición operacional y semántica visual coherente. |
| HUs origen | HU-16 |
| CAs origen | CA-16.1 a CA-16.20 (incluye sustrato F30 inglobado) |
| TTH relacionadas | No aplica directamente (sustrato técnico de persistencia inglobado en CAs de la HU) |
| Persona beneficiaria | Gerente |
| Objetivo del producto | 1, 4 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-04 (apertura ≤ 3 s), RNF-PERF-05 (recálculo ≤ 10 s), RNF-PERF-11 (granularidad de 30 s del histórico), RNF-REL-04 (durabilidad del histórico), RNF-SEC-01 (inmutabilidad del histórico), RNF-SEC-03 (control de acceso al rol Gerente), RNF-SEC-06 (validación dual de restricciones sobre fechas), RNF-FUN-01 (manejo de periodos sin datos o con cobertura parcial), RNF-INT-03 (tooltips autoexplicativos), RNF-INT-02 |
| Notas | Esta HU fusiona F12 (Dashboard ejecutivo) y F13 (Selector de periodo) por cohesión semántica conforme a DHU-016 subsección I, e ingloba F30 (Persistencia de estados históricos) como sustrato técnico en sus CAs. |

#### RF-015 — Vista comparativa entre periodos

| Campo | Contenido |
|---|---|
| Identificador | RF-015 |
| Familia funcional | Reportería ejecutiva |
| Descripción | El sistema permite al Gerente comparar los cuatro KPIs operativos entre el periodo seleccionado y el periodo previo equivalente, presentando para cada KPI un panel con gráfico de dos series temporales superpuestas, dos valores agregados y un indicador prominente de variación entre los dos agregados con semántica visual de mejora o empeoramiento según la naturaleza del indicador. |
| HUs origen | HU-17 |
| CAs origen | CA-17.1 a CA-17.16 |
| TTH relacionadas | No aplica directamente (reutiliza el sustrato persistido inglobado en RF-014) |
| Persona beneficiaria | Gerente |
| Objetivo del producto | 1, 4 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-04 (apertura ≤ 3 s), RNF-PERF-05 (recálculo ≤ 10 s), RNF-PERF-12 (paralelización de las dos consultas al histórico), RNF-REL-01 (robustez con marca aplicada simultáneamente a ambos periodos), RNF-SEC-03 (control de acceso al rol Gerente), RNF-FUN-01 (manejo de periodos sin datos, variaciones indefinidas y cobertura parcial), RNF-INT-02 (semántica visual de mejora/empeoramiento no solo color), RNF-INT-04 (coherencia con RF-014) |
| Notas | La semántica visual de mejora/empeoramiento se materializa con tres pistas visuales redundantes: signo numérico, flecha direccional, color. La redundancia es deliberada para no depender exclusivamente del color. |

#### RF-016 — Drill-down sobre periodo específico

| Campo | Contenido |
|---|---|
| Identificador | RF-016 |
| Familia funcional | Reportería ejecutiva |
| Descripción | El sistema permite al Gerente investigar lo que ocurrió durante un periodo específico mediante una vista detallada con tres carriles temporales integrados sobre un eje temporal común: tráfico observado, decisiones del motor adaptativo e intervalos de estado operativo del sistema. La vista soporta zoom interactivo sobre el carril de tráfico y popovers autocontenidos en marcadores y bandas. |
| HUs origen | HU-18 |
| CAs origen | CA-18.1 a CA-18.20 |
| TTH relacionadas | Reutiliza tres registros existentes del MVP1: histórico inglobado en RF-014, persistencia de decisiones del motor (CT-10.9), transiciones de estado operativo (CT-04.3) |
| Persona beneficiaria | Gerente |
| Objetivo del producto | 4 |
| Prioridad MoSCoW sugerida | Could (MVP2) |
| RNF asociados | RNF-PERF-06 (apertura y zoom), RNF-PERF-12 (paralelización de las tres consultas), RNF-REL-01 (robustez independiente por carril, política específica), RNF-SEC-03 (control de acceso al rol Gerente), RNF-FUN-01 (manejo de periodos sin datos en alguno de los carriles), RNF-INT-04 (coherencia visual con RF-014 y los códigos de RF-019), RNF-INT-02 |
| Notas | HU-18 es exclusivamente consultiva: el Gerente no edita decisiones del motor, no modifica el histórico ni cambia el estado operativo. Si surge necesidad futura de anotación, se modela como extensión. |

#### RF-017 — Exportación de reportes a PDF o Excel

| Campo | Contenido |
|---|---|
| Identificador | RF-017 |
| Familia funcional | Reportería ejecutiva |
| Descripción | El sistema permite al Gerente exportar los reportes de RF-014 y RF-015 a un formato presentable (PDF) o a datos crudos (Excel), eligiendo el formato según el destino del reporte. Los reportes generados son autocontenidos: incluyen las definiciones operacionales de los indicadores en el propio artefacto, sin requerir consulta de documentación externa. |
| HUs origen | HU-19 |
| CAs origen | CA-19.1 a CA-19.27 |
| TTH relacionadas | Reutiliza RF-014 y RF-015 como fuentes |
| Persona beneficiaria | Gerente |
| Objetivo del producto | 4 |
| Prioridad MoSCoW sugerida | Could (MVP2) |
| RNF asociados | RNF-PERF-07 (generación PDF y Excel con umbrales diferenciados), RNF-REL-01 (política conservadora de rechazo ante fuente caída), RNF-REL-06 (tolerancia a fallos del componente de generación), RNF-SEC-03 (control de acceso al rol Gerente), RNF-SEC-07 (no persistencia de reportes generados), RNF-FUN-01 (comunicación explícita de periodos sin datos en el reporte), RNF-FUN-05 (identificabilidad autosuficiente del archivo), RNF-INT-02, RNF-INT-03 (definiciones operacionales autocontenidas), RNF-INT-06 (legibilidad impresa de los gráficos del PDF) |
| Notas | La generación es por demanda y la descarga es directa: los reportes no se almacenan en el servidor (RNF-SEC-07). La política de rechazo ante fuente caída es excepción específica de RNF-REL-01 declarada en DHU-019 subsección C.2. |

### 2.7 Familia 7 — Soporte al Operador y trazabilidad de incidentes

#### RF-018 — Vista simplificada de salud de los componentes para el Operador

| Campo | Contenido |
|---|---|
| Identificador | RF-018 |
| Familia funcional | Soporte al Operador y trazabilidad de incidentes |
| Descripción | El sistema expone al Operador una vista simplificada del estado operativo de cada componente del sistema (nombre legible, estado cualitativo y timestamp del último cambio de estado), permitiéndole identificar de un vistazo qué pieza específica está afectada cuando el sistema entra en un estado degradado. La vista incluye textos de impacto operativo curados que explican qué hace el sistema sin cada componente afectado. |
| HUs origen | HU-11 |
| CAs origen | CA-11.1 a CA-11.9 |
| TTH relacionadas | TTH-04 (CT-04.5) |
| Persona beneficiaria | Operador |
| Objetivo del producto | 3 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-01 (actualización ≤ 5 s), RNF-REL-01 (robustez modo B), RNF-INT-01 (identificación de un vistazo, resalte panorámico), RNF-INT-02 (accesibilidad), RNF-FUN-04 (cobertura del catálogo de textos de impacto operativo), RNF-MNT-01 (extensibilidad del catálogo), RNF-SEC-05 (segregación de presentación con RF-010 del Administrador) |
| Notas | Esta vista comparte el sustrato técnico con RF-010 (vista técnica del Administrador) consumiendo el mismo endpoint con presentación simplificada conforme a DHU-013. |

#### RF-019 — Alerta transversal del estado operativo del sistema

| Campo | Contenido |
|---|---|
| Identificador | RF-019 |
| Familia funcional | Soporte al Operador y trazabilidad de incidentes |
| Descripción | El sistema muestra al Operador una alerta visible de manera consistente en todas las vistas operativas cuando el sistema entra en un estado distinto a operación normal. La alerta identifica el nivel del estado degradado (degradado 1, degradado 2, degradado 3, falla total), el componente o condición que disparó el estado y el tiempo transcurrido desde la entrada al estado. Cada transición de estado se persiste de forma durable e inmutable. |
| HUs origen | HU-10 |
| CAs origen | CA-10.1 a CA-10.9 |
| TTH relacionadas | TTH-04 (CT-04.2, CT-04.3, CT-04.4, CT-04.6, CT-04.7, CT-04.8) |
| Persona beneficiaria | Operador |
| Objetivo del producto | 3 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-01 (latencia transición → alerta ≤ 5 s), RNF-REL-01 (robustez modo B), RNF-REL-02 (disponibilidad transversal en todas las vistas), RNF-REL-03 (la activación de fallbacks no se detiene por fallos del registro), RNF-REL-04 (durabilidad del registro de transiciones), RNF-REL-08 (atomicidad de las transiciones), RNF-REL-09 (comportamiento conservador ante fallo del propio mecanismo de detección), RNF-SEC-01 (inmutabilidad del registro), RNF-INT-01 (distinción visual entre los cuatro estados), RNF-INT-02, RNF-SAF-01 (fail safe ante fallo del motor adaptativo) |
| Notas | La alerta transversal es responsabilidad del Bloque C; la marca pasiva por panel afectado es responsabilidad de cada HU del Bloque B conforme a DHU-009. Las dos responsabilidades coexisten sin duplicación. |

#### RF-020 — Explicación del modo degradado activo

| Campo | Contenido |
|---|---|
| Identificador | RF-020 |
| Familia funcional | Soporte al Operador y trazabilidad de incidentes |
| Descripción | El sistema expone al Operador una explicación textual del modo degradado activo cuando el sistema opera en un estado distinto a operación normal, integrando tres elementos: qué disparó el modo, qué fallback está activo y qué capacidad operativa se perdió con su implicación para la supervisión. La explicación se construye desde un catálogo curado de plantillas; las combinaciones no cubiertas activan un texto genérico de respaldo. |
| HUs origen | HU-12 |
| CAs origen | CA-12.1 a CA-12.6 |
| TTH relacionadas | TTH-04 |
| Persona beneficiaria | Operador |
| Objetivo del producto | 3 |
| Prioridad MoSCoW sugerida | Must |
| RNF asociados | RNF-PERF-01 (actualización ≤ 5 s), RNF-REL-01 (robustez modo B), RNF-REL-02 (disponibilidad mientras el sistema esté degradado), RNF-INT-05 (comprensibilidad sin formación técnica), RNF-FUN-04 (cobertura del catálogo), RNF-MNT-01 (extensibilidad del catálogo), RNF-INT-04 (coherencia textual con RF-007 y RF-018) |
| Notas | Esta HU es complementaria de RF-018 (vista de componentes) y de RF-019 (alerta transversal). Las tres juntas responden las tres preguntas del Operador en estado degradado: qué nivel (RF-019), qué componente (RF-018) y qué significa operativamente (RF-020). |

#### RF-021 — Registro de notas e incidencias del turno del Operador

| Campo | Contenido |
|---|---|
| Identificador | RF-021 |
| Familia funcional | Soporte al Operador y trazabilidad de incidentes |
| Descripción | El sistema permite al Operador registrar notas e incidencias durante su turno asociadas a un momento específico, y consultar posteriormente el listado de notas propias y de otros Operadores con filtros por fechas y autor. Las notas son editables dentro de una ventana temporal acotada posterior a su creación; pasada esa ventana quedan inmutables para preservar valor de auditoría. |
| HUs origen | HU-09 |
| CAs origen | CA-09.1 a CA-09.6 |
| TTH relacionadas | No aplica directamente (persistencia inglobada en CAs) |
| Persona beneficiaria | Operador |
| Objetivo del producto | Soporte al Operador |
| Prioridad MoSCoW sugerida | Should (MVP2) |
| RNF asociados | RNF-PERF-02 (apertura del listado ≤ 2 s), RNF-REL-04 (durabilidad de las notas), RNF-REL-05 (resiliencia de persistencia ante fallo de escritura con preservación del contenido escrito), RNF-SEC-01 (inmutabilidad parcial: editables solo dentro de la ventana), RNF-SEC-03 (control de acceso al rol Operador) |
| Notas | Todos los Operadores ven todas las notas, sosteniendo el caso de uso de transmisión entre turnos. La política MVP2 está refinada por DHU-012. |

#### RF-022 — Escalamiento de incidentes del Operador al Administrador

| Campo | Contenido |
|---|---|
| Identificador | RF-022 |
| Familia funcional | Soporte al Operador y trazabilidad de incidentes |
| Descripción | El sistema permite al Operador escalar al Administrador un incidente observado durante operación degradada, capturando automáticamente el contexto operativo del sistema en el momento del escalamiento (estado operativo, componentes afectados) y permitiendo al Operador agregar una descripción libre. El Administrador recibe los escalamientos en una vista de gestión con un badge numérico de pendientes en su navegación, transiciona los incidentes a "Atendido" cuando los gestiona y los escalamientos quedan auditables. |
| HUs origen | HU-21 |
| CAs origen | CA-21.1 a CA-21.34 |
| TTH relacionadas | TTH-04 (CT-04.4 y CT-04.5 consumidos para captura de contexto) |
| Persona beneficiaria | Operador (originador), Administrador (destinatario y gestor) |
| Objetivo del producto | 3 |
| Prioridad MoSCoW sugerida | Could (MVP2) |
| RNF asociados | RNF-PERF-02 (apertura de vistas ≤ 3 s, equivalencia con RF-009), RNF-PERF-10 (latencia del badge ≤ 30 s), RNF-REL-01 (robustez con política de rechazo del disparo y marca pasiva en consulta), RNF-REL-03 (la operación del motor no depende del registro de incidentes), RNF-REL-04 (durabilidad del registro), RNF-REL-05 (resiliencia de escritura con preservación del contenido del modal), RNF-SEC-01 (inmutabilidad de los campos del disparo y de la transición de estado), RNF-SEC-03 (control de acceso por rol), RNF-SEC-06 (validación dual frontend y backend), RNF-FUN-01 (manejo de filtros que descartan todos los registros), RNF-FUN-06 (independencia entre dimensiones funcionales del registro), RNF-INT-01 (botón identificable, modal legible, badge no intrusivo), RNF-INT-02 (modal navegable con teclado, badge interpretable por lectores de pantalla) |
| Notas | El estado del incidente y el estado del sistema son dimensiones independientes (RNF-FUN-06): la recuperación automática del sistema no cierra automáticamente los incidentes. El cierre lo decide explícitamente el Administrador. |

### Cierre de la sección 2

La sección 2 cubre las 7 familias funcionales declaradas en la sección 1.2 con los RF derivados de los CAs de las 21 HUs operativas y los sustratos técnicos relevantes de las 11 TTH. El catálogo total de RF al cierre de esta sección es el siguiente:

| Familia | RFs declarados | Cantidad |
|---|---|---|
| 1. Control de acceso y autenticación | RF-001, RF-002 | 2 |
| 2. Monitoreo operativo en tiempo real | RF-003, RF-004, RF-005 | 3 |
| 3. Decisiones del motor adaptativo | RF-006, RF-007, RF-008, RF-009 | 4 |
| 4. Predicción de tráfico | (cubierta por composición de RF-004, RF-005, RF-013) | 0 RF dedicados |
| 5. Soporte técnico y configuración del sistema | RF-010, RF-011, RF-012, RF-013 | 4 |
| 6. Reportería ejecutiva | RF-014, RF-015, RF-016, RF-017 | 4 |
| 7. Soporte al Operador y trazabilidad de incidentes | RF-018, RF-019, RF-020, RF-021, RF-022 | 5 |
| **Total** | | **22 RF** |

La cantidad de RF (22) está dentro del rango estimado en DHU-019 subsección E (25 a 35). La cifra es menor que la estimación inicial porque la composición transversal fue efectiva en consolidar comportamientos coherentes en RFs únicos: los CAs de robustez se referenciaron como RNF transversales en lugar de generar RFs propios, y los CAs de control de acceso por rol se consolidaron en RF-002 transversal en lugar de generar 21 RFs específicos. Esta consolidación está alineada con los tres principios de derivación declarados en la sección 1.2.

---

## 4. Matriz de trazabilidad consolidada

La matriz de trazabilidad consolidada presenta el catálogo de RF y RNF organizado por las HUs y TTH del Product Backlog, permitiendo el recorrido inverso desde una HU o TTH hacia los RF y RNF que se derivan de ella.

### 4.1 Trazabilidad desde las 21 HUs operativas

| HU | RF que cubren su comportamiento | RNF que aplican a sus CAs |
|---|---|---|
| HU-01 | RF-001, RF-002 | RNF-SEC-02, RNF-SEC-03, RNF-SEC-04, RNF-INT-07 |
| HU-02 | RF-003 | RNF-PERF-01, RNF-REL-01, RNF-INT-01, RNF-INT-02 |
| HU-03 | RF-004 | RNF-PERF-01, RNF-REL-01, RNF-FUN-02, RNF-COM-02 |
| HU-04 | RF-005 | RNF-PERF-01, RNF-REL-01, RNF-INT-01, RNF-INT-02 |
| HU-05 | RF-006 | RNF-PERF-01, RNF-REL-01, RNF-INT-01, RNF-SAF-02 |
| HU-06 | RF-007 | RNF-PERF-01, RNF-REL-01, RNF-INT-05, RNF-FUN-04, RNF-MNT-01, RNF-INT-04 |
| HU-07 | RF-008 | RNF-PERF-01, RNF-REL-01, RNF-INT-01, RNF-MNT-02 |
| HU-08 | RF-009 | RNF-PERF-02, RNF-REL-03, RNF-REL-04, RNF-SEC-01 |
| HU-09 | RF-021 | RNF-PERF-02, RNF-REL-04, RNF-REL-05, RNF-SEC-01, RNF-SEC-03 |
| HU-10 | RF-019 | RNF-PERF-01, RNF-REL-01, RNF-REL-02, RNF-REL-03, RNF-REL-04, RNF-REL-08, RNF-REL-09, RNF-SEC-01, RNF-INT-01, RNF-INT-02, RNF-SAF-01 |
| HU-11 | RF-018 | RNF-PERF-01, RNF-REL-01, RNF-INT-01, RNF-INT-02, RNF-FUN-04, RNF-MNT-01, RNF-SEC-05 |
| HU-12 | RF-020 | RNF-PERF-01, RNF-REL-01, RNF-REL-02, RNF-INT-05, RNF-FUN-04, RNF-MNT-01, RNF-INT-04 |
| HU-13 | RF-010 | RNF-PERF-01, RNF-REL-01, RNF-SEC-03, RNF-SEC-05, RNF-INT-01, RNF-INT-02, RNF-FUN-01 |
| HU-14 | RF-012 | RNF-PERF-09, RNF-REL-04, RNF-SEC-01, RNF-SEC-03, RNF-FUN-01, RNF-FUN-02, RNF-INT-03, RNF-INT-02 |
| HU-15 | RF-011 | RNF-PERF-03, RNF-PERF-08, RNF-REL-04, RNF-REL-07, RNF-SEC-01, RNF-SEC-03, RNF-SEC-06, RNF-MNT-02, RNF-SAF-03 |
| HU-16 | RF-014 | RNF-PERF-04, RNF-PERF-05, RNF-PERF-11, RNF-REL-04, RNF-SEC-01, RNF-SEC-03, RNF-SEC-06, RNF-FUN-01, RNF-INT-03, RNF-INT-02 |
| HU-17 | RF-015 | RNF-PERF-04, RNF-PERF-05, RNF-PERF-12, RNF-REL-01, RNF-SEC-03, RNF-FUN-01, RNF-INT-02, RNF-INT-04 |
| HU-18 | RF-016 | RNF-PERF-06, RNF-PERF-12, RNF-REL-01, RNF-SEC-03, RNF-FUN-01, RNF-INT-04, RNF-INT-02 |
| HU-19 | RF-017 | RNF-PERF-07, RNF-REL-01, RNF-REL-06, RNF-SEC-03, RNF-SEC-07, RNF-FUN-01, RNF-FUN-05, RNF-INT-02, RNF-INT-03, RNF-INT-06 |
| HU-20 | RF-013 | RNF-PERF-09, RNF-PERF-13, RNF-REL-04, RNF-SEC-01, RNF-SEC-03, RNF-FUN-01, RNF-FUN-03, RNF-MNT-03, RNF-INT-04, RNF-INT-02 |
| HU-21 | RF-022 | RNF-PERF-02, RNF-PERF-10, RNF-REL-01, RNF-REL-03, RNF-REL-04, RNF-REL-05, RNF-SEC-01, RNF-SEC-03, RNF-SEC-06, RNF-FUN-01, RNF-FUN-06, RNF-INT-01, RNF-INT-02 |

### 4.2 Trazabilidad desde las 11 TTH

| TTH | RF que habilita | RNF cuyo origen primario es esta TTH |
|---|---|---|
| TTH-01 | RF-001 | RNF-SEC-02 |
| TTH-02 | (transversal a todos) | RNF-COM-01, RNF-FLX-01 |
| TTH-03 | (transversal a todos) | No aplica de forma específica (cobertura de CI es propiedad de Maintainability / Testability documentada en los criterios técnicos de TTH-03 sin RNF formalmente declarado) |
| TTH-04 | RF-010, RF-018, RF-019, RF-020 | RNF-REL-08, RNF-REL-09 |
| TTH-05 | (consumida por TTH-04 en degradado nivel 3) | RNF-SAF-01, RNF-SAF-03 |
| TTH-06 | (Trabajos Futuros, no se construye en MVP1 ni MVP2) | No aplica (Won't) |
| TTH-07 | (provee dataset a TTH-09 y entorno a TTH-10) | RNF-COM-02 |
| TTH-08 | (componente demostrable, no en loop de validación conforme a D-007) | RNF-COM-02 |
| TTH-09 | RF-004, RF-005, RF-012 | RNF-FUN-02, RNF-FLX-02 |
| TTH-10 | RF-006, RF-007, RF-008, RF-009 | RNF-SAF-02 |
| TTH-11 | (provee hiperparámetros temporales a TTH-09) | No aplica de forma específica (sustento bibliográfico y empírico de hiperparámetros, propiedad de Maintainability / Analysability sin RNF formalmente declarado) |

---

## 5. Glosario

Esta sección consolida la terminología específica del producto referenciada en el catálogo de RF y RNF. Los términos del marco ISO/IEC 25010:2023 están cubiertos en la sección 1.1 y no se repiten aquí.

**Acceso (de la intersección).** Una de las direcciones de entrada vehicular a la intersección semaforizada. En la topología genérica de cuatro accesos declarada en TTH-07, los cuatro accesos corresponden a los puntos cardinales de la intersección.

**Componente del sistema.** Pieza arquitectónica con responsabilidad operativa diferenciada, monitoreada por la lógica de detección de salud de TTH-04. Los componentes con impacto operativo perceptible incluyen al menos: módulo de detección de tráfico, módulo predictivo, motor adaptativo, componente de explicación, registro de eventos.

**Estrategia de control.** Política de decisión que el motor adaptativo aplica al semáforo para determinar los tiempos de las fases. El sistema selecciona automáticamente entre estrategias según el estado predicho y observado del tráfico; las estrategias concretas están documentadas en TTH-10 y `CONTROL.md`, no en el documento RF/RNF conforme a DHU-006.

**Estado operativo del sistema.** Una de las cinco condiciones agregadas en las que el sistema puede operar según la lógica de fallback de TTH-04: operación normal, degradado nivel 1, degradado nivel 2, degradado nivel 3, falla total. La transición entre estados es atómica conforme a RNF-REL-08.

**Fail safe.** Comportamiento del sistema ante fallo que transita a un estado de operación seguro definido en lugar de detener la operación o aplicar decisiones inconsistentes. En CerebroVial el fail safe del control adaptativo se materializa con los tiempos preconfigurados del degradado nivel 3 conforme a RNF-SAF-01.

**Fase (del semáforo).** Cada uno de los intervalos de operación del semáforo durante un ciclo, asociado a un conjunto de accesos con derecho de paso simultáneo. Cada fase tiene tiempos de verde, amarillo y all-red sujetos a las constantes normativas declaradas en RNF-SAF-02.

**Fuente operacional.** Origen de los datos que alimentan las vistas en tiempo real del Operador. El sistema es agnóstico a la fuente conforme a DHU-006: en MVP1 la fuente operativa son corridas del entorno de simulación (TTH-07); en operación hipotética sería el módulo de visión computacional (TTH-08). La portabilidad entre fuentes se sustenta con el constructo unificado de RNF-COM-02.

**Histórico operacional.** Registro append-only de los estados observados del tráfico persistido con granularidad de 30 segundos por intersección y dirección (RNF-PERF-11), consumido por las vistas del Gerente (RF-014, RF-015, RF-016) y por la exportación de reportes (RF-017). Materializa F30 inglobada en HU-16 conforme a DHU-016.

**Jam level.** Nivel de congestión expresado como escala ordinal 0-5, derivado del ratio velocidad/free-flow según el mapeo de D-009. Constructo unificado adoptado del estándar de la industria que permite la portabilidad de fuentes operacionales sin reentrenamiento del modelo predictivo (RNF-COM-02).

**Last-write-wins.** Política de resolución de concurrencia entre múltiples modificadores del mismo recurso, donde la última modificación en guardarse sobrescribe los valores anteriores. En CerebroVial se aplica con advertencia explícita al segundo modificador conforme a RNF-REL-07.

**Modelo predictivo principal.** Componente del sistema que produce las predicciones de nivel de congestión consumidas por el Operador y por el motor adaptativo. La arquitectura concreta del modelo está documentada en D-006 y en TTH-09; el documento RF/RNF es agnóstico al detalle técnico conforme a DHU-006.

**Modelo predictivo de respaldo.** Componente preservado en el sistema que produce predicciones cuando el modelo principal no responde, invocado por la lógica de fallback de TTH-04 en degradado nivel 2 (RNF-FLX-02). El registro paralelo de sus predicciones habilita la vista comparativa de RF-013 (HU-20 MVP2).

**Motor adaptativo.** Componente central del sistema que selecciona automáticamente la estrategia de control semafórico aplicable según el estado predicho y observado del tráfico. Su arquitectura está documentada en TTH-10 y `CONTROL.md`. La capa de aplicación normativa del motor (capa MTC) garantiza el cumplimiento de RNF-SAF-02.

**Nivel de congestión.** Variable de estado del tráfico expresada como escala ordinal 0-5, donde 0 representa flujo libre y 5 representa vía cerrada. Equivalente a jam level.

**Periodo previo equivalente.** Periodo temporal de la misma duración que el periodo actualmente seleccionado, ubicado inmediatamente antes en el calendario. Por ejemplo, si el periodo actual es "Esta semana", el periodo previo equivalente es "La semana anterior". Definición cerrada en CA-17.4 conforme a DHU-016 subsección G.

**Robustez ante interrupción.** Principio de comportamiento del sistema cerrado en DHU-005 que distingue dos modos según la naturaleza de la fuente interrumpida: modo A para fuentes externas de medición del mundo observado, con marca semántica "desactualizado"; modo B para componentes internos de decisión del sistema, con marca semántica "no confirmado". Materializa el RNF transversal RNF-REL-01.

**Tiempos preconfigurados del degradado nivel 3.** Conjunto de tiempos de fases del semáforo conservadores que el sistema aplica cuando el motor adaptativo no responde (degradado nivel 3), garantizando que la intersección permanezca operativa hasta la recuperación del motor o intervención del Administrador. Provistos por TTH-05 y aplicados por la lógica de fallback de TTH-04 conforme a RNF-SAF-01.

---

## 6. Cierre y mantenimiento

### 6.1 Cuándo se actualiza el documento

El documento `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` es vivo y se actualiza ante los siguientes eventos:

1. **Modificación sustantiva del Product Backlog.** Si una HU se reabre y sus CAs cambian, o si una TTH se reabre y sus criterios técnicos cambian de naturaleza, los RF y RNF derivados se ajustan en la pasada correspondiente. El catálogo nunca queda desincronizado del backlog que lo origina.

2. **Calibración de umbrales tras validación cuantitativa.** Conforme a D-005 y la subsección I de DHU-019, los umbrales del documento se inicializan idénticos a los del backlog. Cuando la validación cuantitativa del sistema integrado o las mediciones en operación real demuestran que un umbral declarado es irrealizable o demasiado laxo, el umbral se ajusta en este documento con justificación basada en datos reportados, no en juicio ex ante.

3. **Ceremonia formal de priorización MoSCoW posterior.** Las prioridades sugeridas declaradas en este documento son anclaje argumentado para la ceremonia formal. Cuando la ceremonia ratifica o ajusta una prioridad, el documento se actualiza con la prioridad ratificada y se preserva la prioridad sugerida original en notas de mantenimiento.

4. **Identificación de RF o RNF omitidos.** Si durante la implementación se identifica un comportamiento del sistema requerido o una propiedad de calidad necesaria no cubierta por el catálogo actual, el RF o RNF correspondiente se agrega con el próximo número correlativo disponible dentro de su familia o característica.

5. **Refinamiento de las clasificaciones ISO 25010:2023.** Si el equipo del producto identifica que un RNF está mal clasificado (subcaracterística incorrecta, característica primaria errónea), la clasificación se ajusta. Las decisiones de clasificación masiva siguen viviendo en DHU-019 subsección B; ajustes individuales se documentan en notas del RNF afectado.

### 6.2 Cómo se referencia desde las HUs y desde las TTH

Conforme a la subsección G de DHU-019, los CAs de las 21 HUs preservan su redacción literal con umbrales hardcoded. La trazabilidad bidireccional se materializa con dos mecanismos:

1. **Tablas de la sección 4** (matriz de trazabilidad consolidada) declaran, para cada HU y para cada TTH, los RF que se derivan de ella y los RNF que aplican a sus CAs.

2. **Pasada aditiva pendiente sobre las secciones "Candidatos a RNF" de cada HU.** Tras cerrar este documento, cada sección "Candidatos a RNF" de las 21 HUs recibe una pasada aditiva agregando una referencia `→ RNF-XXX-NN` por candidato declarado. La pasada es estrictamente aditiva, no modifica contenido sustantivo de las HUs, y completa el ciclo de trazabilidad bidireccional.

### 6.3 Relación con la ceremonia MoSCoW pendiente

Este documento declara prioridades MoSCoW *sugeridas* para cada RF y cada RNF. La ceremonia formal MoSCoW posterior consume estas sugerencias como anclaje argumentado y produce las prioridades ratificadas que el sprint de implementación honra.

La convención de prioridad sugerida está declarada en DHU-019 subsección F. En resumen: Must para RF y RNF que realizan directamente los Objetivos del Producto y aplican a Personas MVP1 o son soporte transversal del MVP1; Should para RF y RNF que aplican a HUs MVP2 (incluyendo HU-09) o a propiedades de calidad importantes pero no críticas (accesibilidad WCAG, autoexplicación); Could para mejoras operativas; Won't para Trabajos Futuros declarados (TTH-06, escalamiento a múltiples intersecciones).

---

> **Cierre del documento.**
>
> El documento `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` queda cerrado en esta versión inicial con: preámbulo (sección 0), marco de referencia (sección 1), catálogo de 22 RF organizado en 7 familias funcionales (sección 2), catálogo de 53 RNF clasificados por las 9 características de ISO/IEC 25010:2023 (sección 3), matriz de trazabilidad consolidada (sección 4), glosario (sección 5), cierre y mantenimiento (sección 6).
>
> Las próximas acciones derivadas del cierre de este documento son:
>
> 1. **Pasada aditiva** sobre las secciones "Candidatos a RNF" de las 21 HUs operativas, agregando referencias `→ RNF-XXX-NN` por candidato declarado, conforme a DHU-019 subsección G.
>
> 2. **Derivación del documento lite** `RF_RNF_LITE.md` para lectura humana fluida, con la misma estructura pero plantillas simplificadas (sin campos de trazabilidad fina), conforme al acuerdo sobre el modelo de dos documentos (Lectura B confirmada en la sesión de redacción).
>
> 3. **Ceremonia formal MoSCoW** sobre el catálogo completo de RF y RNF, ratificando o ajustando las prioridades sugeridas.
>
> 4. **Ceremonia Planning Poker** sobre los RF y RNF priorizados Must y Should, para estimación de esfuerzo previa al sprint de implementación.
>
> 5. **Implementación SCRUM del MVP1** con el catálogo de RF y RNF como referencia normativa.
