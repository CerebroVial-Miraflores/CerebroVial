# Documento de Diseño de Software (SDD) — CerebroVial

> Sistema inteligente de control adaptativo de semáforos para una intersección de Miraflores, Lima.
> Proyecto de tesis de ingeniería de software · ejecución individual con asistencia de IA.
>
> **Estado del documento:** en construcción incremental, sección por sección.
> **Última actualización:** 2026-05-20 · cuerpo completo (§0–§12 + vista de desarrollo + DHU-021).

---

## 0. Sobre este documento

### 0.1 Propósito y alcance

Este documento describe la **arquitectura objetivo** del sistema CerebroVial: su descomposición en componentes, su modelo de datos, sus vistas de proceso y despliegue, su stack tecnológico, sus atributos de calidad y las decisiones arquitectónicas que lo conforman. Es el puente entre el backlog formal (qué hace el sistema y con qué calidad) y la implementación (cómo está construido).

El SDD adopta una postura **As-designed**: el cuerpo del documento (§1 a §9) describe la arquitectura tal como se diseñó, sin entremezclar en la prosa el grado de avance de cada componente. El estado real de construcción y la brecha entre lo diseñado y lo implementado (≈25% del backlog vivo al momento de redacción) se concentran en dos secciones designadas: §10 (matriz de trazabilidad con estado y delta) y §11 (estado de implementación y brecha). Esta separación mantiene el documento legible como diseño de arquitectura y confina la información de avance —de naturaleza distinta— a su lugar propio.

### 0.2 Audiencia

Tres lectores previstos: el jurado de tesis (verifica rigor arquitectónico y trazabilidad a requisitos), el propio tesista y futuros colaboradores (consultan el diseño para implementar), y agentes de asistencia de código que operan sobre el repositorio (consumen el diseño objetivo y, cuando necesitan el punto de partida, la matriz de §10).

### 0.3 Relación con los demás documentos del proyecto

El SDD no reabre lo ya cerrado en el corpus documental; lo cita y construye sobre él.

| Documento | Rol respecto al SDD |
|---|---|
| `BACKLOG_OVERVIEW.md` y `HU_BLOQUE_*.md`, `HU_MVP2.md` | Origen de las 21 HU y su semántica; fuente de la trazabilidad de §10. |
| `TAREAS_TECNICAS_HABILITADORAS.md` | Origen de las 11 TTH; sustrato técnico de los componentes. |
| `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` y `RF_RNF_LITE.md` | Catálogo de 22 RF y 53 RNF clasificados por ISO/IEC 25010:2023; fuente de §8 (atributos de calidad). |
| `DECISIONS.md` (D-001 a D-009) | Decisiones técnicas del producto; fuente de §2 y del catálogo ADR de §9. |
| `DECISIONS_HU.md` (DHU-001 a DHU-020) | Decisiones metodológicas del backlog; DHU-020 delega al SDD el diseño de la persistencia de estado vigente (§4). |
| `AUDITORIA_HU_CODIGO.md` | Estado real del código por HU/TTH y los 13 deltas; fuente de §10 y §11. |
| `REPORTE_PLANIFICACION_SPRINT_4.md` | Marco de ejecución; el SDD se anticipó allí como entregable metodológico paralelo (riesgo R3). |

### 0.4 Convenciones

Las referencias cruzadas usan los identificadores canónicos del proyecto: decisiones técnicas `D-00N`, decisiones metodológicas `DHU-0NN`, historias de usuario `HU-NN`, tareas técnicas habilitadoras `TTH-NN`, requisitos funcionales `RF-0NN`, requisitos no funcionales `RNF-XXX-NN`, criterios de aceptación `CA-NN.N` (HU) y `CT-NN.N` (TTH), y deltas de auditoría `Delta-NN`. Los nombres de archivos, módulos, clases y rutas del repositorio se escriben en `estilo de código`.

---

---

## 1. Introducción

### 1.1 El problema

El control de semáforos por tiempos fijos —el ingeniero calcula los verdes una vez y los deja iguales— funciona razonablemente bajo demanda estable, pero sirve mal en una ciudad real, donde la demanda cambia con la hora pico, el fin de semana, un evento o un accidente. CerebroVial aborda ese problema con control adaptativo: un sistema que recalcula los tiempos de semáforo en función de la demanda observada y anticipada, sobre una intersección de Miraflores, Lima, como caso de uso.

### 1.2 El sistema

CerebroVial integra cuatro capacidades sobre una arquitectura común. Un **sensor de estado** por visión computacional (YOLO11n en el nodo edge) observa el tráfico y produce métricas de flujo, cola y densidad. Un **modelo predictivo** (GRU univariado por intersección) anticipa la congestión a un horizonte configurable. Un **motor adaptativo** de control —aporte central del sistema— selecciona entre estrategias clásicas (Webster) y reactivas (MaxPressure) y aplica una capa de cumplimiento normativo del Manual MTC peruano que garantiza la seguridad vial de cada decisión. Y un **entorno de validación** con SUMO mide cuantitativamente el aporte del sistema comparándolo contra tiempos fijos. Las tres Personas del producto —Operador, Gerente y Administrador— consumen el sistema a través de vistas diferenciadas por rol.

Un rasgo de diseño atraviesa todo el sistema: está desacoplado de la fuente de su variable de estado (§3.3). La misma variable de congestión puede provenir de la visión en operación, de SUMO en validación o de un feed externo a futuro, sin que el modelo predictivo ni el motor lo noten. Ese desacople es lo que permite validar cuantitativamente sin depender de la visión, y lo que da al diseño su flexibilidad.

### 1.3 Propósito y alcance del documento

Este documento es el Documento de Diseño de Software (SDD) del proyecto: describe la arquitectura objetivo del sistema en el marco de vistas 4+1 de Kruchten, con los atributos de calidad clasificados según ISO/IEC 25010:2023 y las decisiones arquitectónicas registradas como ADR ligeros. Adopta una postura As-designed (§0.1): el cuerpo describe el diseño objetivo; el estado real de construcción —aproximadamente 25% del backlog al momento de redacción— se confina a la matriz de trazabilidad (§10) y al análisis de brecha (§11).

El SDD es un proyecto de tesis de ingeniería de software de ejecución individual con asistencia de IA, enmarcado metodológicamente en Lean Inception adaptado y SCRUM. Corresponde a los artefactos `plan.md` y `data-model.md` del proceso GitHub Spec Kit, adoptado en modo brownfield: el corpus documental ya curado (21 HU, 11 TTH, 22 RF, 53 RNF, 20 DHU, 9 decisiones técnicas) se mapea a las plantillas de Spec Kit, no se regenera.

### 1.4 Estructura del documento

El documento sigue las vistas del modelo 4+1, precedidas por las decisiones fundacionales y seguidas por las secciones de calidad, decisiones y estado. La §2 destila las restricciones arquitectónicas que las decisiones ya cerradas imponen. La §3 (componentes), §4 (datos), §5 (proceso) y §6 (despliegue) son las vistas estáticas y dinámicas del 4+1, complementadas por la vista de desarrollo (organización del código). La §7 inventaria el stack. La §8 trata los atributos de calidad. La §9 cataloga las decisiones arquitectónicas. La §10 y §11 confrontan el diseño con el estado real. La §12 reúne glosario y referencias.

---

---

## 2. Decisiones y restricciones arquitectónicas fundacionales

Esta sección destila qué **restricción arquitectónica** impone cada decisión técnica ya cerrada del proyecto. No reproduce su contexto ni su justificación completa —eso vive en el catálogo ADR de §9 y, en su forma original, en `DECISIONS.md`—; responde a una sola pregunta: *¿qué forma toma forzosamente la arquitectura por las decisiones ya tomadas?* Es el terreno sobre el que se levantan todas las secciones siguientes.

### 2.1 Forma del sistema

**D-001 — Monolito modular.** El sistema es un único sistema desplegable organizado en módulos, no un conjunto de microservicios. Los módulos son `core_management_api` (que aloja predicción, control y consumo de visión), `edge_device` (visión computacional) y `frontend_ui` (presentación), con una base común consolidada en `shared/` como paquete pip local. No existe API HTTP entre los módulos internos del núcleo, y `ia_prediction_service` queda fuera del runtime como herramienta de entrenamiento offline. Esta decisión es la madre de la vista de componentes (§3): fija qué piezas existen y descarta de raíz la complejidad de servicios independientes, coherente con un proyecto de tesis de equipo reducido.

### 2.2 Forma del despliegue

**D-003 — Despliegue local con Docker.** El sistema se despliega en una sola máquina mediante orquestación de contenedores (`docker compose up`), sin nube. La arquitectura desplegable en hardware distribuido se demuestra conceptualmente y se documenta como plan de productivización, no se entrega. Esta decisión fija la vista de despliegue (§6) y sustenta el atributo de portabilidad (RNF-FLX-01).

**D-004 — Edge físico como demostración conceptual.** No se entrega hardware (Raspberry Pi) en la defensa; se demuestra que la arquitectura *es desplegable* en un nodo edge separado, gracias a la contenerización independiente de `edge_device` y a su comunicación con el núcleo por HTTP/SSE. La consecuencia arquitectónica es un mapeo conceptual edge/servidor que §6 documenta: qué módulos correrían en el nodo edge (visión) y cuáles en el servidor central (núcleo, presentación, base de datos).

### 2.3 Naturaleza del modelo predictivo y de su variable de estado

**D-002 — Modelo predictivo de la familia RNN, con respaldo.** El modelo principal pertenece a la familia RNN; un predictor `RandomForest` se conserva como respaldo invocable. La restricción arquitectónica es una **cascada de modelos** con un contrato de endpoint estable, de modo que el respaldo pueda sustituir al principal sin alterar a sus consumidores (motor adaptativo y vistas del Operador). Sustenta RNF-FLX-02 (reemplazabilidad del modelo).

**D-006 — GRU univariado por intersección.** El modelo principal se concreta como GRU univariado, tratando cada intersección de forma independiente. Se descartan arquitecturas espacio-temporales (STGNN), declaradas trabajo futuro. La consecuencia es que el componente de predicción (§3) y el modelo de datos (§4) operan sobre **una serie temporal por dirección**, sin grafo espacial entre intersecciones. Esto acota directamente la escalabilidad declarada en RNF-FLX-03.

**D-009 — Variable de estado: jam level (constructo Waze, escala 0-5).** El sistema adopta el "jam level" ordinal 0-5 como variable de estado y objetivo del modelo. El modelo se entrena sobre el **ratio continuo** velocidad/flujo-libre y la discretización al nivel 0-5 ocurre solo en la capa de presentación. La restricción de diseño es doble: el modelo de datos (§4) persiste el ratio continuo además del nivel discreto, y la arquitectura queda desacoplada de la fuente de datos (SUMO, Waze o visión propia producen la misma variable), lo que habilita la intercambiabilidad de fuente sin reentrenar.

### 2.4 Límites del sistema y fuentes de datos

**D-007 — Visión como sensor demostrable, fuera del loop de validación.** El módulo de visión (`edge_device/src/vision/`) es un sensor de estado en tiempo real, funcional y demostrable, con validación propia mediante métricas de detección (precisión, recall, mAP). No participa en el loop de validación cuantitativa del sistema integrado. La consecuencia arquitectónica es que la visión alimenta al motor en operación, pero la validación del sistema no depende de ella.

**D-008 — SUMO como columna vertebral de datos.** La simulación SUMO genera el dataset de entrenamiento del modelo y los escenarios de validación cuantitativa (con sistema vs. sin sistema), con particiones independientes para evitar fuga de información. La restricción arquitectónica es que SUMO provee, en el entorno de validación, las mismas métricas de estado que la visión proveería en producción —coherente con el desacople de fuente de D-009— y que la integración con SUMO (vía TraCI) es un componente de primera clase del diseño, no un accesorio de validación final.

### 2.5 Restricción heredada de la alineación especificación↔código

**DHU-020 §E — Persistencia de "estado vigente del motor".** No es una decisión técnica numerada (`D-00N`), pero impone una restricción de diseño de datos tan fundacional como las anteriores. La alineación de la especificación con el código (cierre del Delta-08) estableció que HU-05 describe una vista pasiva del estado vigente del motor en producción, y que el backend actual no tiene noción de "estrategia vigente": solo posee una calculadora invocada a demanda. Cumplir HU-05 exige, por tanto, **construir el concepto de estado vigente persistido por intersección con timestamp de activación**. DHU-020 autoriza deliberadamente ese cambio estructural y delega su diseño concreto —entidad, esquema y relación con el histórico de decisiones de HU-08— al modelo de datos de este documento (§4). §2 lo registra aquí para que §4 aparezca con un mandato explícito y no como una novedad sin origen.

### 2.6 Resumen de restricciones

| Decisión | Restricción que impone | Sección del SDD afectada |
|---|---|---|
| D-001 | Monolito modular; módulos sin API interna; `ia_prediction_service` offline | §3 Componentes |
| D-003 | Despliegue local por contenedores, sin nube | §6 Despliegue |
| D-004 | Mapeo conceptual edge/servidor; sin entrega de hardware | §6 Despliegue |
| D-002 | Cascada de modelos con contrato de endpoint estable | §3 Componentes · §8 (RNF-FLX-02) |
| D-006 | Serie temporal univariada por intersección; sin grafo espacial | §3 Componentes · §4 Datos · §8 (RNF-FLX-03) |
| D-009 | Ratio continuo entrenado, nivel 0-5 presentado; fuente desacoplada | §4 Datos · §3 Componentes |
| D-007 | Visión como sensor demostrable fuera del loop de validación | §3 Componentes · límites del sistema |
| D-008 | SUMO provee métricas de estado en validación; TraCI de primera clase | §3 Componentes · §5 Proceso |
| DHU-020 §E | Persistencia de estado vigente por intersección con timestamp | §4 Datos (mandato explícito) |

---

## 3. Vista de componentes

Esta sección describe la descomposición del sistema en sus componentes y las relaciones entre ellos. Sigue la vista lógica/de componentes del modelo 4+1, en dos niveles: el nivel de contenedores (las unidades desplegables) y el nivel de componentes internos del núcleo (la descomposición del `core_management_api`, donde reside la complejidad central del sistema). La estructura interna de cada módulo según Domain-Driven Design se documenta aparte, en la vista de desarrollo.

La descomposición se deriva directamente de la restricción fundacional D-001 (monolito modular): el sistema es un único sistema desplegable organizado en módulos, no un conjunto de microservicios. Las relaciones de cada componente con las historias de usuario y tareas técnicas que realiza se enuncian aquí de forma breve; la trazabilidad completa, con estado de construcción y deltas, vive en la matriz de §10.

### 3.1 Nivel de contenedores

En tiempo de ejecución, el sistema se compone de cuatro contenedores orquestados por Docker Compose (D-003), más un quinto módulo que opera fuera del runtime.

| Contenedor | Tecnología | Responsabilidad | Realiza |
|---|---|---|---|
| `edge_device` | Python, YOLO11n, supervision | Sensor de estado en tiempo real. Procesa video de cámara, detecta y cuenta vehículos por zona, y produce métricas de estado del tráfico (flujo, cola, densidad). Diseñado para correr en un nodo edge separado (D-004). | TTH-08 |
| `core_management_api` | Python, FastAPI | Núcleo del sistema. Aloja la predicción de congestión, el motor adaptativo de control y el consumo del estado de visión. Expone la API que consume el frontend y persiste contra la base de datos. | TTH-09, TTH-10, HU-02 a HU-08, HU-10 a HU-17 |
| `frontend_ui` | React, TypeScript | Capa de presentación. Vistas diferenciadas por rol (Operador, Gerente, Administrador) sobre el estado del tráfico, la predicción, las decisiones del motor, la salud del sistema y la reportería. | HU-01, y todas las HU con interfaz de usuario |
| `db` | PostgreSQL + TimescaleDB + PostGIS | Persistencia. Series temporales (hypertables Timescale) para datos de tráfico y decisiones, datos espaciales (PostGIS) para la topología del grafo vial, y datos relacionales para el resto del dominio. | Sustrato de toda HU/TTH con persistencia |
| `ia_prediction_service` | Python, PyTorch | Herramienta de entrenamiento offline, **fuera del runtime**. Produce el modelo GRU univariado (D-006) que el componente de predicción del núcleo carga y sirve. No es un servicio HTTP ni participa de la operación en vivo. | TTH-09 (entrenamiento), TTH-11 |

La decisión de mantener `ia_prediction_service` fuera del runtime es consecuencia directa de D-001: el entrenamiento del modelo es un pipeline ML offline, no un servicio que responda peticiones en operación. El núcleo carga el artefacto de modelo ya entrenado; no invoca al servicio de entrenamiento en tiempo de ejecución.

Las relaciones principales en runtime: `edge_device` envía el estado observado al `core_management_api`; el `frontend_ui` consume la API del núcleo; el núcleo lee y escribe contra `db`. La comunicación entre el edge y el núcleo es por HTTP (y, en el diseño objetivo del canal de tiempo real, por un mecanismo de eventos que se define en la vista de proceso, §5).

### 3.2 Nivel de componentes — interior del núcleo

El `core_management_api` aloja tres componentes lógicos. Su coexistencia en un mismo desplegable —en lugar de tres servicios separados— es precisamente lo que D-001 establece: no existe API HTTP entre ellos, sino invocación interna directa.

El componente de **predicción** sirve la predicción de congestión. Carga el modelo GRU univariado por intersección (D-006) producido offline por `ia_prediction_service`, y conserva un predictor `RandomForest` como respaldo invocable. Esta dualidad principal/respaldo materializa la cascada de modelos de D-002 con un contrato de endpoint estable, de modo que el respaldo pueda sustituir al principal sin alterar a sus consumidores (atributo de reemplazabilidad, RNF-FLX-02). La variable que predice es el jam level en escala 0-5 (D-009): el modelo se entrena sobre el ratio continuo velocidad/flujo-libre y la discretización al nivel ordinal ocurre en la capa de presentación. Realiza la cadena predictiva de TTH-09 y alimenta a HU-03, HU-04, HU-14 y HU-20.

El componente de **control** es el motor adaptativo, aporte central del sistema. Selecciona entre estrategias de control —Webster y MaxPressure en la capa estratégica— y aplica la capa de restricciones normativas del Manual MTC peruano, que eleva, recorta o compone la salida estratégica para garantizar el cumplimiento de las constantes de seguridad vial (RNF-SAF-02). Produce, junto con la decisión, un razonamiento textual sobre por qué se eligió la estrategia. Realiza TTH-10 y sustenta HU-05 (estrategia vigente), HU-06 (explicación), HU-07 (notificación de cambios) y HU-08 (historial de decisiones). El diseño de la persistencia que este componente requiere —el estado vigente del motor y el historial de decisiones— se define en el modelo de datos (§4), conforme al mandato de DHU-020.

El componente de **consumo de visión** (vision-consumer) recibe del `edge_device` las métricas de estado del tráfico y las pone a disposición del resto del núcleo: la predicción las usa como observación de entrada y el control las usa como estado actual de la intersección. Este componente es el punto de integración entre el sensor de estado (que vive en otro contenedor) y la lógica central. Su existencia refleja la separación física de D-004: el sensor corre en el edge, su consumo en el servidor central.

### 3.3 Frontera con las fuentes de estado

Un rasgo de diseño que conviene hacer explícito, derivado de D-009 y D-008: el sistema está desacoplado de la fuente de la variable de estado. La misma variable (jam level / ratio velocidad-flujo-libre) puede provenir del `edge_device` (visión propia, en operación), de SUMO (en el entorno de validación cuantitativa) o, como extensión futura, de un feed externo tipo Waze. El componente de predicción y el de control operan sobre la variable de estado, no sobre su fuente. Esto permite intercambiar la fuente sin reentrenar el modelo ni alterar la lógica de control, y es la razón por la que SUMO puede sustituir a la visión en el loop de validación (D-007/D-008) sin que el núcleo lo note.

---

## 4. Modelo de datos

Esta sección describe el esquema de persistencia de la arquitectura objetivo. Parte del modelo de datos vigente del proyecto —siete tablas que cubren el grafo vial, los feeds de Waze y la visión computacional— y lo extiende con las dos entidades que la operación del motor adaptativo requiere y que aún no existen: el historial de decisiones del motor y su estado vigente por intersección. La descripción es de diseño: enuncia las entidades, sus relaciones y las restricciones que las gobiernan, sin marcar grado de avance; el estado real de construcción de cada tabla vive en la matriz de §10.

El detalle columna por columna del modelo heredado es canónico en `DATA_MODEL.md`; esta sección lo resume por dominios para situar las extensiones y no lo reproduce íntegro.

### 4.1 El modelo heredado: grafo vial, Waze y visión

El modelo de datos vigente organiza el dominio en tres familias de entidades.

La **topología vial** se modela como un grafo dirigido sobre PostGIS. `graph_nodes` representa cada intersección física como un punto georreferenciado; `graph_edges` representa cada calle dirigida entre dos nodos —una vía de doble sentido es dos aristas— con su geometría, distancia y número de carriles. Esta es la columna vertebral espacial del sistema: todo lo demás se ancla, directa o indirectamente, a un nodo o a una arista. Es estructura estática, sembrada una vez y consultada después.

Los **feeds externos de Waze** se modelan como series temporales. `waze_jams` guarda snapshots de congestión asociados a una arista, donde `congestion_level` (entero 1-5) es la clasificación que Waze produce internamente y que el modelo predictivo adopta como variable objetivo de entrenamiento. `waze_alerts` guarda eventos puntuales (accidentes, peligros, cierres) reportados por usuarios. Ambas son candidatas a hypertable de TimescaleDB, particionadas por su timestamp, por su naturaleza de alto volumen y origen continuo.

La **visión computacional** se modela con `cameras` (metadata espacial de cada cámara: ubicación, orientación, campo de visión) y dos tablas de datos observados, `vision_tracks` (trayectorias individuales de vehículos) y `vision_flows` (flujos direccionales por arista). El diseño contempla además una tabla `vision_aggregates`, alineada con el esquema que el pipeline de visión ya produce hoy (conteos por tipo de vehículo, ocupación y flujo por ventana temporal). La relación entre estas tres tablas refleja una decisión de modelado tomada en la auditoría del modelo de datos (registrada el 2026-05-03): la visión persiste agregados compatibles con su salida real, mientras `vision_tracks` y `vision_flows` quedan modeladas para integración futura del pipeline a base de datos. La separación honra el desacople de fuente que el sistema mantiene (§3.3): el modelo predictivo no depende de la visión para entrenar, lo que permite que `vision_tracks`/`vision_flows` existan en el diseño sin condicionar el loop de validación.

> Las decisiones de visión de esa auditoría se citan aquí por contenido y fecha, no por identificador `D-00N`: la serie `D-006/D-007/D-008` de la auditoría (2026-05-03) colisiona con la serie canónica de `DECISIONS.md` (2026-05-11). El SDD usa exclusivamente la numeración canónica de `DECISIONS.md`.

### 4.2 Extensión para el motor adaptativo

El motor adaptativo (§3.2) impone dos necesidades de persistencia que el modelo heredado no cubre, ambas con mandato explícito. La alineación de la especificación con el código (DHU-020 §E) estableció que HU-05 describe una vista pasiva del *estado vigente* del motor, concepto que el backend actual no posee —solo tiene una calculadora invocada a demanda— y delegó su diseño a esta sección. La auditoría de implementación (Delta-10) estableció que HU-08 requiere un historial de decisiones del motor que tampoco existe. §4 satisface ambos con dos entidades nuevas, ancladas a `graph_nodes`.

#### 4.2.1 `motor_decisions` — historial de decisiones

Registro append-only: cada recomendación que el motor produce se inserta como una fila y no se modifica después. Su esquema reproduce el contrato de salida del motor, de modo que cada decisión sea auditable y reproducible.

| Columna | Tipo | Notas |
|---|---|---|
| `decision_id` | uuid PK | Identificador único de la decisión. |
| `node_id` | string FK → `graph_nodes` | La intersección sobre la que se decide. El motor emite hoy un `intersection_id` opaco y sin restricción (es un componente sin estado que no consulta la base de datos); la capa de persistencia **resuelve y valida** ese identificador contra `graph_nodes.node_id` al escribir la fila. El anclaje al grafo es responsabilidad del write-path, no del contrato del endpoint. |
| `decided_at` | datetime | Momento del cálculo. |
| `mode` | string | Estrategia activada: `webster` o `max_pressure`. |
| `cycle_seconds` | float | Ciclo final, compuesto por la capa MTC. |
| `flow_total` | float | Suma de flujos del input; discriminante peak/off-peak. |
| `y_load_factor` | float, nullable | Factor de carga Y de Webster; nulo en el caso de saturación severa (`webster_infeasible`). |
| `next_phase` | string, nullable | Fase que MaxPressure elige entrar primero; nulo en modo Webster. |
| `reasoning` | text | Razonamiento textual del motor; sustrato directo de HU-06. |
| `phase_timings` | jsonb | Arreglo de `{phase_id, green, yellow, all_red}` por fase. |
| `adjustments` | jsonb | Lista de descripciones de texto de los ajustes normativos que aplicó la capa MTC (el motor las emite como `list[str]`); vacía si no hubo. |
| `inputs_snapshot` | jsonb, nullable | Payload de fases que originó la decisión (`flow`, `saturation`, `queue`, `pedestrian` por fase). Hace la decisión reproducible. |

Se indexa por `(node_id, decided_at DESC)`, que sirve las dos consultas naturales: el historial completo de una intersección (HU-08) y su decisión más reciente.

Tres de estas columnas —`flow_total`, `y_load_factor` e `inputs_snapshot`— no forman parte hoy de la salida HTTP del endpoint del motor: la respuesta vigente (`POST /control/recommend`, envuelta en `{data: …}`) expone `mode`, `cycle_seconds`, `phase_timings`, `next_phase`, `reasoning` y `adjustments`, pero `flow_total` y `y_load_factor` se calculan internamente en el motor sin serializarse, e `inputs_snapshot` no se devuelve. El diseño los conserva en `motor_decisions` deliberadamente, por la auditabilidad y reproducibilidad que §8.3 (RNF-SEC-01) exige: el write-path los captura del cálculo interno del motor y del payload de la petición en el momento de persistir, no del cuerpo de la respuesta. Cerrar esa brecha entre lo que el endpoint emite y lo que la decisión persiste es trabajo del componente de control al construir la persistencia (Delta-10).

Dos rasgos de diseño merecen explicitarse. Primero, el detalle de las fases se persiste como `jsonb` (`phase_timings`, `adjustments`, `inputs_snapshot`) en lugar de normalizarse en una tabla hija de fases. Esto es fiel al sistema real: el motor recibe las fases en el cuerpo de la petición y emite su salida como estructura anidada, no las lee ni las escribe como filas relacionales. Normalizar las fases inventaría una estructura que el código no produce. Segundo, `motor_decisions` se modela como **tabla relacional, no como hypertable**, pese a su naturaleza temporal: el volumen de una intersección piloto —una decisión por ciclo semafórico— no justifica el particionamiento de TimescaleDB, y modelarla relacional mantiene limpias las claves foráneas que dependen de ella. La conversión a hypertable es una operación posterior y de bajo costo si la productivización multiplica el volumen; §11 la registra como deuda consciente del plan de productivización.

#### 4.2.2 `engine_active_state` — estado vigente del motor

Registro mutable: exactamente una fila por intersección, actualizada cuando se activa una estrategia. Materializa el "estado vigente" que DHU-020 §E exige.

| Columna | Tipo | Notas |
|---|---|---|
| `node_id` | string PK FK → `graph_nodes` | La intersección. Como clave primaria, garantiza una única estrategia vigente por intersección. |
| `active_decision_id` | uuid FK → `motor_decisions` | Puntero a la decisión actualmente activada. |
| `activated_at` | datetime | Timestamp de activación que exige DHU-020 §E. |
| `activated_by` | string, nullable | Origen de la activación (operador o automático); soporta HU-05 y HU-07 sin acoplar todavía el modelo de usuarios. |

La distinción entre `engine_active_state` y `motor_decisions` no es redundancia: separa dos eventos distintos. `motor_decisions.decided_at` registra cuándo el motor *calculó* una recomendación; `engine_active_state.activated_at` registra cuándo esa recomendación se *activó* en la intersección. El motor puede calcular recomendaciones que no se activan, y la activación puede ocurrir después del cálculo. Por eso el estado vigente es una entidad propia con puntero a la decisión activada, y no una vista derivada del historial (que solo podría inferir "la última calculada", semántica distinta de "la actualmente vigente").

### 4.3 La frontera grafo↔intersección

El modelo de datos razona en términos de grafo vial dirigido (nodos, aristas, congestión por arista); el motor adaptativo razona en términos de *fases* que agrupan *approaches*, con flujo, saturación y cola por fase. Entre ambos lenguajes existe una traducción —qué cámara observa qué approach, qué approaches componen qué fase, cómo se convierte un nivel ordinal de congestión en un flujo en veh/h— que el diseño del sistema reconoce como un adaptador de dominio.

Este adaptador es **frontera de diseño diferida**: no se esquematiza en tablas en esta sección. La razón es arquitectónica, no de omisión. El motor consume las fases como payload de cada petición, no las persiste ni las lee de base de datos; por tanto la operación vigente del motor —y las dos entidades que §4.2 introduce para soportarla— no requieren un modelo relacional de fases para funcionar. La construcción del adaptador (mapeo cámara→approach→fase y conversión nivel→flujo, ambos dependientes del despliegue físico de cada intersección) es trabajo identificado para una etapa posterior. §4 ancla las decisiones del motor a la intersección (`graph_nodes`) y deja el interior de la intersección —su descomposición en approaches y fases— como extensión futura del modelo, coherente con el alcance vigente.

---

## 5. Vista de proceso

Esta sección describe el comportamiento dinámico del sistema: cómo colaboran los componentes de §3 en tiempo de ejecución, qué flujos de datos los recorren y cómo se propaga un cambio de estado a través del sistema. El modelo 4+1 reserva la vista de proceso para los aspectos de concurrencia, comunicación y secuencia temporal que la vista estática de componentes no captura.

El sistema tiene dos procesos de primera clase, gobernados por restricciones distintas: el **loop operativo**, que va del sensor de estado a la presentación al operador, y el **loop de validación con SUMO**, que sustituye la fuente de estado por el simulador para medir cuantitativamente el aporte del sistema. Ambos comparten el núcleo —predicción y control operan idénticos— y difieren solo en la fuente de la variable de estado, exactamente como la frontera de §3.3 anticipa.

### 5.1 El loop operativo

En operación, el ciclo de vida de una observación recorre cuatro etapas. El `edge_device` procesa el video de cámara y produce métricas de estado del tráfico (flujo, cola, densidad por zona). El componente de consumo de visión del núcleo recibe esas métricas por HTTP y las pone a disposición de la predicción y el control. El componente de control, ante una solicitud de recomendación, recibe las fases de la intersección como payload, calcula la estrategia (Webster u MaxPressure según el umbral de demanda, con la capa MTC componiendo la salida normativa) y emite una decisión. La decisión se persiste en `motor_decisions` (§4) y, cuando se activa, actualiza `engine_active_state`. Finalmente, la presentación expone al operador el estado vigente, la explicación de la decisión y la notificación de cambios.

El sistema **no** opera el motor en un lazo cerrado autónomo, y esto es postura deliberada de la arquitectura objetivo, no un estado interino: el motor calcula a demanda y el operador es quien observa y, donde corresponde, activa. La conexión predicción→motor que cerraría el lazo proactivo queda fuera del alcance del MVP como trabajo futuro identificado (la frontera de §4.3), no como omisión a subsanar dentro de este documento. El código confirma la afirmación: el motor expone una calculadora invocable y sin estado, no un planificador continuo.

### 5.2 El canal de tiempo real (HU-07)

La notificación de cambios de estrategia del motor (HU-07) requiere que el frontend reciba actualizaciones del servidor sin sondeo activo. El diseño adopta **Server-Sent Events (SSE)** como mecanismo del canal de tiempo real. La elección se sigue de la naturaleza del flujo: la comunicación relevante es unidireccional —el servidor empuja al cliente la notificación de que la estrategia vigente cambió—, que es precisamente el caso de uso de SSE. Un canal bidireccional con WebSocket sería capacidad excedente para tráfico que no necesita ida y vuelta del cliente al servidor; el sondeo periódico (polling) desperdiciaría peticiones y añadiría latencia de detección. SSE corre además sobre HTTP plano, coherente con la comunicación edge↔núcleo que D-004 fija por HTTP/SSE y que §2 y §3 ya enuncian.

El estado que el frontend lee —la estrategia vigente de una intersección— se consulta **de la base de datos a través de la API del núcleo** (`engine_active_state`, §4), no de un cache en memoria del núcleo. Para una intersección piloto, la lectura directa desde la base de datos es suficiente y elimina el problema de coherencia que un cache introduce. Un cache de estado vigente en memoria es optimización pertinente solo si la productivización multiplica intersecciones y frecuencia de consulta; §11 lo registra como deuda consciente, no como omisión.

### 5.3 El loop de validación con SUMO

La validación cuantitativa del sistema (D-008) no usa la visión: sustituye la fuente de estado por SUMO, que provee al núcleo las mismas métricas de estado que la visión proveería en producción. El proceso de validación ejecuta dos configuraciones de la misma intersección en el simulador —una con tiempos fijos precalibrados, otra con el motor adaptativo recibiendo el estado del simulador— y compara la demora promedio por vehículo entre ambas. La integración con SUMO es por TraCI, y es componente de primera clase del diseño (D-008): no un accesorio de la validación final, sino el sustrato sobre el que se mide el aporte del sistema.

Este loop es la materialización de la frontera de §3.3: el motor y la predicción operan idénticos en validación y en operación, porque ambos consumen la variable de estado y no su fuente. SUMO ocupa, en validación, el lugar que el `edge_device` ocupa en operación. El desacople de fuente es lo que permite que la validación sea cuantitativamente honesta sin depender de la visión (D-007).

### 5.4 Concurrencia y modos de fallo

El sistema mantiene una postura de degradación graceful ante la pérdida de la fuente de estado, fijada por la lógica de fallback en cascada (TTH-04, TTH-05). Cuando la fuente de estado se interrumpe —cámara caída en operación—, el motor no puede recibir las métricas obligatorias que su contrato requiere (flujo, saturación, cola por fase) y el diseño contempla la caída a un programa de tiempos fijos precargado para esa intersección. El diseño concreto de los niveles de degradado vive en TTH-04/TTH-05; §5 lo registra como el comportamiento dinámico ante fallo de la fuente, no como detalle de implementación.

---

## 6. Vista de despliegue

Esta sección describe cómo se mapean los componentes lógicos de §3 sobre infraestructura física: qué corre dónde, cómo se orquesta y cómo se comunica entre nodos. Está gobernada por dos decisiones fundacionales: el despliegue local con Docker (D-003) y el edge físico como demostración conceptual (D-004).

### 6.1 Topología de despliegue vigente

En el alcance entregable, el sistema se despliega en **una sola máquina** mediante orquestación de contenedores con Docker Compose (D-003). No hay nube. Los cuatro contenedores de runtime —`edge_device`, `core_management_api`, `frontend_ui` y `db`— se levantan con `docker compose up` y se comunican por la red interna de Docker Compose. El quinto módulo, `ia_prediction_service`, no participa del runtime: produce offline el artefacto de modelo que el núcleo carga (D-001, §3.1).

La contenerización independiente de cada módulo es lo que sostiene la afirmación arquitectónica de D-004: que el sistema *es desplegable* en hardware distribuido aunque no se entregue así. Cada contenedor es una unidad de despliegue autónoma; que hoy corran todos en una máquina es una configuración de despliegue, no una propiedad de acoplamiento del diseño.

### 6.2 Mapeo conceptual edge/servidor

D-004 exige documentar cómo se distribuiría el sistema en un despliegue productivo con nodo edge físico, sin entregarlo. El mapeo conceptual es directo y se deriva de la responsabilidad de cada componente:

| Nodo | Módulos que alojaría | Justificación |
|---|---|---|
| Nodo edge (junto a la cámara) | `edge_device` | El sensor de estado corre donde está el video, para no transmitir el flujo de video crudo por la red; solo viajan las métricas de estado ya extraídas. |
| Servidor central | `core_management_api`, `frontend_ui`, `db` | La lógica de predicción y control, la presentación y la persistencia residen centralizadas, consumiendo las métricas que el edge les envía. |

La comunicación entre el nodo edge y el servidor central es por HTTP (métricas de estado) y SSE (canal de eventos), tal como §5 establece. Esta separación es la misma que el componente de consumo de visión refleja en §3.2: el sensor en el edge, su consumo en el servidor.

### 6.3 Persistencia y volúmenes

La base de datos (`db`) es un contenedor PostgreSQL con las extensiones TimescaleDB y PostGIS. Su estado persiste en un volumen de Docker para sobrevivir reinicios del contenedor. Las migraciones de esquema se gestionan con Alembic, incluyendo la exclusión de las tablas internas de PostGIS del autogenerate (configurada en el entorno de migraciones).

### 6.4 Plan de productivización

El paso de la topología vigente (una máquina, Docker Compose) a una productiva (nodo edge físico + servidor) es el contenido del plan de productivización que D-003 y D-004 prometen documentar sin entregar. Sus piezas: desplegar `edge_device` en hardware edge (p. ej. Raspberry Pi) con la cámara real; centralizar el resto en un servidor; reemplazar la red interna de Compose por comunicación de red real HTTP/SSE entre nodos. El diseño no requiere cambios estructurales para ese paso —solo de configuración de despliegue—, lo que es precisamente la propiedad que D-004 busca demostrar. §11 retoma este plan como parte de la brecha entre lo diseñado y lo entregado.

---

## 7. Vista de implementación (stack tecnológico)

Esta sección inventaria el stack tecnológico de cada componente y justifica las elecciones que tienen consecuencia arquitectónica. No es un listado exhaustivo de dependencias —eso vive en los manifiestos del repositorio—, sino el mapa de las tecnologías que definen la forma del sistema. Las versiones y librerías que esta sección consigna están confirmadas contra los manifiestos reales del repo.

### 7.1 Stack por componente

| Componente | Lenguaje | Framework / librerías principales | Rol del stack |
|---|---|---|---|
| `edge_device` | Python | YOLO11n (detección), supervision (tracking) | Visión computacional: detección y conteo de vehículos, extracción de métricas de estado. |
| `core_management_api` | Python | FastAPI, SQLAlchemy / GeoAlchemy2, Alembic | Núcleo: API HTTP, ORM con soporte espacial, migraciones. Aloja predicción, control y consumo de visión. |
| `frontend_ui` | TypeScript | React | Presentación: vistas por rol, consumo de la API y del canal SSE. |
| `db` | — | PostgreSQL, TimescaleDB, PostGIS | Persistencia: relacional, series temporales y datos espaciales. |
| `ia_prediction_service` | Python | PyTorch | Entrenamiento offline del modelo GRU univariado. |
| Validación | Python | SUMO, TraCI | Simulación de escenarios y harness de validación cuantitativa. |
| `shared` | Python | (paquete pip local) | Base común consolidada: modelos ORM, utilidades transversales. |

Dos precisiones sobre el versionado, que difieren por componente y conviene no difuminar. El **frontend fija versiones explícitas** en `frontend_ui/package.json` —React 19.2, Vite 7, Tailwind 4, TypeScript 5.9, Leaflet 1.9—, de modo que esas son versiones reales y pineadas. El **backend, en cambio, no fija versiones**: `core_management_api/requirements.txt` declara FastAPI, SQLAlchemy, GeoAlchemy2 y Alembic sin pin (la versión se resuelve al construir la imagen); el único anclaje de plataforma es Python 3.11 por el `Dockerfile`, y PyTorch lleva cota mínima (`>=2.3.0`) en `ia_prediction_service`. YOLO está fijado por artefacto (`yolo11n.pt`, es decir YOLO11n) aunque la librería `ultralytics` va sin pin. Esta asimetría es estado real del repo, no descuido de la documentación.

La fila de **validación con SUMO/TraCI es diseño objetivo, no código presente**: el repositorio no contiene hoy integración SUMO, configuraciones de simulación ni harness de comparación (TTH-07 figura como no iniciado en §10). Se inventaría en §7 como el stack previsto para esa capa, coherente con su estado de brecha en §11.

### 7.2 Elecciones con consecuencia arquitectónica

Algunas elecciones de stack no son intercambiables sin alterar el diseño, y conviene hacer explícita su razón.

La separación **PyTorch (entrenamiento) / artefacto cargado por el núcleo (inferencia)** es la materialización de D-001: el entrenamiento es un pipeline offline que no corre en runtime. El núcleo no importa el servicio de entrenamiento; carga el modelo ya serializado. Esto permite que `ia_prediction_service` quede fuera del `docker-compose.yml` de runtime sin que el núcleo pierda capacidad de predicción.

Sobre el formato del artefacto, conviene separar diseño objetivo y estado vigente. El modelo que el núcleo sirve **hoy** es el `RandomForest` de respaldo, serializado con `joblib` (`.joblib`) y cargado por el componente de predicción; el GRU principal (D-006) vive entrenado en `ia_prediction_service` —como `state_dict` de PyTorch— pero **aún no está servido por el núcleo** (TTH-09 no iniciado, Delta-01). La cascada de modelos de D-002 es, por tanto, diseño objetivo cuyo lado principal está pendiente: el contrato de endpoint estable existe para que, cuando el GRU se sirva, sustituya al respaldo sin alterar a los consumidores.

La elección de **PostgreSQL con TimescaleDB y PostGIS en una sola base** —en lugar de tres almacenes especializados— es coherente con D-001 (un sistema, no microservicios con bases separadas). Una única instancia sirve los tres tipos de dato (relacional, temporal, espacial) mediante extensiones, evitando la complejidad de sincronizar almacenes heterogéneos.

La cascada de modelos (D-002) impone que el componente de predicción mantenga **dos implementaciones tras un contrato de endpoint estable**: el GRU principal (PyTorch) y el RandomForest de respaldo. El stack de inferencia debe poder servir ambos sin que el consumidor (motor, presentación) note cuál responde.

---

## 8. Atributos de calidad

El backlog clasifica 53 requisitos no funcionales en las nueve características de calidad de ISO/IEC 25010:2023 (`RF_RNF_LITE.md`, documento normativo `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md`). Esta sección no recorre las nueve una por una —eso sería un eco del catálogo—; destila los atributos que la arquitectura *forma*, no solo los que cumple: aquellos cuya satisfacción es consecuencia directa de una decisión de diseño y no de buenas prácticas genéricas. El resto se reconoce en la tabla-resumen de §8.5, que remite al catálogo normativo para el detalle.

### 8.1 Reemplazabilidad del modelo predictivo (RNF-FLX-02)

La cascada de modelos de D-002 es, en sí misma, un atributo de calidad hecho estructura. El componente de predicción mantiene el GRU principal y el `RandomForest` de respaldo tras un contrato de endpoint estable; los consumidores —motor adaptativo y vistas del Operador— no conocen cuál responde. Esto satisface la reemplazabilidad (RNF-FLX-02) por diseño: sustituir el modelo principal no propaga cambios río abajo. La calidad no se añade después; está en la forma del componente.

### 8.2 Seguridad vial y operación fail-safe (RNF-SAF-01, RNF-SAF-02, RNF-SAF-03)

La seguridad de operar sobre infraestructura de tránsito real se materializa en dos piezas de diseño. La **capa MTC** del motor (§3.2) garantiza el cumplimiento de las constantes normativas del Manual MTC peruano (RNF-SAF-02): eleva, recorta o compone la salida estratégica de Webster/MaxPressure para que ningún tiempo viole los mínimos y máximos legales. La separación de esta capa respecto de los algoritmos estratégicos no es estética: permite que la regulación cambie sin tocar la optimización, y hace que el cumplimiento sea auditable —cada ajuste queda registrado como un `adjustment` en la decisión (§4)—. El **comportamiento fail-safe** (RNF-SAF-01) ante caída del motor se sostiene en la lógica de fallback en cascada (TTH-04/TTH-05): el sistema aplica tiempos preconfigurados conservadores en vez de detenerse o aplicar decisiones inconsistentes. Y los valores por defecto seguros desde el primer arranque (RNF-SAF-03) son los mismos tiempos preconfigurados, disponibles sin ajuste manual previo.

### 8.3 Auditabilidad e inmutabilidad de registros (RNF-SEC-01)

La inmutabilidad append-only de los registros del sistema (RNF-SEC-01) es la razón de la forma de `motor_decisions` en §4: una tabla a la que solo se inserta, nunca se modifica. El historial de decisiones del motor (HU-08) es auditable y durable porque su sustrato de datos lo es por construcción. La separación entre `motor_decisions` (historial inmutable) y `engine_active_state` (puntero mutable al vigente) preserva esa inmutabilidad sin sacrificar la noción de estado actual: lo que cambia es el puntero, no el registro histórico.

### 8.4 Portabilidad del despliegue (RNF-FLX-01)

La portabilidad —desplegar en máquina limpia con un solo comando (RNF-FLX-01)— es consecuencia directa de D-003 y de la contenerización de cada módulo (§6). `docker compose up` levanta el sistema completo; el quickstart documentado es suficiente para un evaluador académico. La misma contenerización independiente que da portabilidad es la que sostiene el mapeo conceptual edge/servidor de D-004: portabilidad y desplegabilidad-distribuida son dos caras de la misma decisión.

### 8.5 Las nueve características y su cobertura

| Característica ISO 25010:2023 | RNF | Tratamiento arquitectónico |
|---|---|---|
| Functional Suitability | 6 | Corrección en casos límite (datos faltantes, degeneración matemática del ciclo de Webster); el motor responde 422 ante saturación severa en lugar de producir tiempos absurdos. |
| Performance Efficiency | 13 | Latencias de respuesta y actualización; sustrato de series temporales (hypertables Timescale) para consultas eficientes sobre datos de tráfico. |
| Compatibility | 2 | Coexistencia de los servicios contenerizados; interoperabilidad por contratos HTTP/SSE. |
| Interaction Capability | 7 | Vistas diferenciadas por rol; explicaciones en lenguaje del dominio (catálogo de plantillas, §3.2). |
| Reliability | 9 | Degradación graceful (cascada de fallback), atomicidad de transiciones de estado operativo, comportamiento conservador ante fallo del detector de salud. |
| Security | 7 | Autenticación JWT/bcrypt, control de acceso por rol validado en backend, no filtración en errores 403, inmutabilidad append-only (§8.3). |
| Maintainability | 3 | Catálogos de plantillas como datos (no hardcoded), parametrización sin redespliegue. |
| Flexibility | 3 | Reemplazabilidad del modelo (§8.1), portabilidad (§8.4), escalabilidad con límite declarado (univariado por intersección, D-006). |
| Safety | 3 | Capa MTC, fail-safe a tiempos preconfigurados, defaults seguros (§8.2). |

El detalle de cada RNF, su prioridad MoSCoW y su trazabilidad fina viven en el documento normativo; esta tabla es el mapa de cómo el diseño los aloja.

---

## Vista de desarrollo (organización del código)

El modelo 4+1 reserva la vista de desarrollo para la organización estática del código: cómo se estructuran los módulos internamente, qué convenciones siguen y cómo se reparten las responsabilidades dentro de cada uno. Es el complemento de la vista de componentes (§3): §3 dice qué piezas existen y cómo colaboran; esta vista dice cómo está organizado el código dentro de cada pieza.

### Estructura interna por Domain-Driven Design

La convención predominante de los módulos del backend es una estructura por capas de Domain-Driven Design: `domain` (entidades, reglas de negocio y contratos de repositorio), `application` (casos de uso que orquestan el dominio), `infrastructure` (implementaciones concretas: persistencia, modelos ML, clientes externos) y `presentation` (la capa que expone hacia afuera: routers HTTP, esquemas de entrada/salida). Esta organización mantiene el dominio independiente de la tecnología: las reglas del motor o de la predicción no dependen de FastAPI ni de SQLAlchemy, que viven en `infrastructure` y `presentation`.

La estructura se cumple por completo en dos módulos: el de visión (`edge_device/src/vision/`), donde el contrato de repositorio vive en `domain/repositories.py` y sus implementaciones concretas en `infrastructure/persistence/`, y el de predicción (`core_management_api/src/prediction/`), donde los modelos ML concretos viven en `infrastructure/`. El módulo de control (`core_management_api/src/control/`) es la **desviación**: hoy tiene `domain`, `application` (Webster, MaxPressure, capa MTC) y `presentation`, pero **no** una capa `infrastructure`, porque el motor es una calculadora sin estado que no persiste ni consume servicios externos. Esa capa aparecerá cuando se construya la persistencia de decisiones del motor (`motor_decisions`/`engine_active_state`, §4; Delta-10): el repositorio que escriba esas entidades es precisamente el `infrastructure` que hoy falta. La desviación es, por tanto, reflejo del estado de avance, no una decisión de organización divergente.

### La base común `shared`

El paquete `shared` consolida lo transversal —los modelos ORM del esquema de datos, utilidades comunes— como paquete pip local instalable por los módulos que lo necesitan. Esto materializa D-001: en un monolito modular, la base común se comparte como biblioteca, no como un servicio que se invoque por red. Los modelos ORM del modelo de datos (§4) residen aquí, lo que permite que el núcleo y las migraciones compartan una única definición del esquema.

### Convenciones y restricción de no-refactor

El proyecto mantiene una restricción operativa explícita sobre `edge_device/src/vision/`: no refactorizar ese código sin que el sprint correspondiente lo aborde. Es una salvaguarda de estabilidad sobre el módulo de visión existente, que funciona y tiene cobertura de pruebas (`edge_device/tests/vision/`). La vista de desarrollo la registra como convención vigente; su eventual levantamiento es una decisión de planificación, no de arquitectura.

---

## 9. Catálogo de decisiones arquitectónicas (ADR)

Esta sección registra las decisiones técnicas del producto en formato ADR ligero —contexto, decisión, consecuencias, estado—, sin reproducir la justificación completa que vive en `DECISIONS.md`. El propósito es dar trazabilidad arquitectónica: cada decisión que forma el sistema queda registrada con su razón y sus consecuencias, citables desde el resto del SDD y desde el informe de tesis. Se incluyen las nueve decisiones técnicas canónicas (D-001…D-009) y la restricción heredada DHU-020 §E, por su peso arquitectónico. Las decisiones puramente metodológicas (serie DHU) no entran aquí; se consolidan aparte (DHU-021).

**ADR D-001 — Monolito modular.** *Contexto:* proyecto de tesis individual; la complejidad de microservicios no se justifica. *Decisión:* un único sistema desplegable organizado en módulos sin API HTTP interna; `ia_prediction_service` fuera del runtime. *Consecuencias:* fija la vista de componentes (§3); base común como paquete pip (`shared`); entrenamiento ML offline. *Estado:* vigente.

**ADR D-002 — Modelo predictivo RNN con respaldo.** *Contexto:* el sistema necesita un predictor robusto y reemplazable. *Decisión:* modelo principal de la familia RNN con `RandomForest` de respaldo invocable, tras un contrato de endpoint estable. *Consecuencias:* cascada de modelos (§3.2, §8.1); sustenta RNF-FLX-02. *Estado:* vigente.

**ADR D-003 — Despliegue local con Docker.** *Contexto:* defensa académica sin infraestructura de nube. *Decisión:* despliegue en una máquina con Docker Compose; la arquitectura distribuible se demuestra conceptualmente. *Consecuencias:* fija la vista de despliegue (§6); sustenta portabilidad RNF-FLX-01. *Estado:* vigente.

**ADR D-004 — Edge físico como demostración conceptual.** *Contexto:* no se entrega hardware en la defensa. *Decisión:* demostrar que la arquitectura es desplegable en nodo edge separado vía contenerización independiente y comunicación HTTP/SSE. *Consecuencias:* mapeo conceptual edge/servidor (§6.2); canal SSE (§5.2). *Estado:* vigente.

**ADR D-005 — Números de tesis: actualizar tras validación real.** *Contexto:* integridad académica; reportar cifras no reproducibles en el demo es riesgo alto en la defensa. *Decisión:* los números declarados en el documento de tesis (88.2% de accuracy de detección, 81.3% de accuracy del predictor, latencia <2s) se actualizan a los valores reales medidos durante la validación cuantitativa; si la realidad es peor, se reporta la realidad. *Consecuencias:* condiciona la presentación de métricas (§8 Functional Suitability) y la validación con SUMO (§5.3); la actualización del documento con los números reales es entregable explícito. *Estado:* cerrada (sujeta a confirmación con asesor).

**ADR D-006 — GRU univariado por intersección.** *Contexto:* concreción del modelo principal de D-002. *Decisión:* GRU univariado que trata cada intersección de forma independiente; se descartan arquitecturas espacio-temporales (STGNN) como trabajo futuro. *Consecuencias:* una serie temporal por dirección, sin grafo espacial entre intersecciones (§4); acota RNF-FLX-03. *Estado:* vigente.

**ADR D-007 — Visión como sensor demostrable, fuera del loop de validación.** *Contexto:* la visión es funcional pero su integración cuantitativa al sistema es costosa y no aporta a la tesis. *Decisión:* la visión es sensor de estado en operación, validada por métricas propias (precisión, recall, mAP); no participa del loop de validación cuantitativa. *Consecuencias:* la validación del sistema no depende de la visión (§5.3); frontera de fuentes de estado (§3.3). *Estado:* vigente.

**ADR D-008 — SUMO como columna vertebral de datos.** *Contexto:* sin acceso a datos reales de tráfico de Miraflores ni a la API de Waze. *Decisión:* SUMO genera el dataset de entrenamiento y los escenarios de validación (con/sin sistema), con particiones independientes; integración por TraCI como componente de primera clase. *Consecuencias:* loop de validación (§5.3); SUMO sustituye a la visión en validación sin que el núcleo lo note. *Estado:* vigente.

**ADR D-009 — Ratio continuo entrenado, nivel ordinal presentado, fuente desacoplada.** *Contexto:* la variable de estado debe ser entrenable con precisión y presentable de forma legible. *Decisión:* el modelo se entrena sobre el ratio continuo velocidad/flujo-libre; la discretización al nivel ordinal 0-5 (jam level) ocurre en presentación; el sistema se desacopla de la fuente de la variable. *Consecuencias:* modelo de datos y componentes operan sobre la variable, no sobre su fuente (§3.3, §4); permite intercambiar visión/SUMO/Waze. *Estado:* vigente.

**ADR DHU-020 §E — Persistencia de estado vigente del motor.** *Contexto:* HU-05 describe una vista pasiva del estado vigente, concepto que el backend no poseía. *Decisión:* construir el concepto de estado vigente persistido por intersección con timestamp de activación; diseño delegado al modelo de datos. *Consecuencias:* entidades `engine_active_state` y `motor_decisions` (§4); separación calculado/activado. *Estado:* vigente; diseño concretado en §4 de este documento.

---

## 10. Trazabilidad

Esta sección es la matriz bidireccional que conecta el backlog (HU/TTH) con los componentes del diseño (§3) y con el estado real de construcción y sus deltas. Es el único lugar del SDD —junto con §11— donde el estado de avance entra explícitamente: el cuerpo As-designed (§1–§9) describe el diseño objetivo; aquí se confronta con el punto de partida. El estado y los deltas provienen de la auditoría HU↔código (`AUDITORIA_HU_CODIGO.md`, 2026-05-18). Los deltas se identifican por su ID (`Delta-NN`); su descripción y acción propuesta están en §11 y en la auditoría.

### 10.1 Tareas técnicas habilitadoras

| TTH | Componente (§3) | Estado | Deltas |
|---|---|---|---|
| TTH-01 Autenticación JWT/bcrypt | core (transversal) | No iniciado | Delta-02 |
| TTH-02 Docker Compose | despliegue (§6) | Completo | — |
| TTH-03 Repo Git + CI | transversal | Parcial | Delta-03 |
| TTH-04 Fallback en cascada | core (control) | No iniciado | — |
| TTH-05 Tiempos preconfigurados degradado | core (control) | No iniciado | — |
| TTH-06 Capa de DTOs | transversal | Fuera de scope (Trabajos Futuros) | — |
| TTH-07 Integración SUMO/TraCI | validación (§5.3) | No iniciado | — |
| TTH-08 Módulo de visión | `edge_device` | Parcial | Delta-04, Delta-05 |
| TTH-09 GRU servido vía API | core (predicción) | No iniciado | Delta-01 |
| TTH-10 Motor adaptativo | core (control) | Parcial | Delta-09, Delta-10 |
| TTH-11 Spike hiperparámetros | `ia_prediction_service` | No iniciado | — |

### 10.2 Historias de usuario

| HU | Componente (§3) | Estado | Deltas |
|---|---|---|---|
| HU-01 Autenticación | `frontend_ui` + core | No iniciado | Delta-02 |
| HU-02 Estado del tráfico por acceso | `frontend_ui` + vision-consumer | No iniciado | Delta-06, Delta-07 |
| HU-03 Predicción por acceso | `frontend_ui` + predicción | No iniciado | Delta-01, Delta-07 |
| HU-04 Vista combinada estado+predicción | `frontend_ui` + predicción | No iniciado | Delta-06, Delta-07 |
| HU-05 Estrategia vigente | `frontend_ui` + control | Parcial | Delta-07, Delta-08 |
| HU-06 Explicación de la estrategia | `frontend_ui` + control | Parcial | Delta-07, Delta-09 |
| HU-07 Notificación de cambios | `frontend_ui` + control (SSE) | No iniciado | Delta-07 |
| HU-08 Historial de decisiones | core (control) + `db` | No iniciado | Delta-10 |
| HU-09 Notas operativas | `frontend_ui` + core | No iniciado | — |
| HU-10 Alerta transversal | `frontend_ui` | No iniciado | Delta-11 |
| HU-11, HU-12 Soporte degradado al Operador | `frontend_ui` + core | No iniciado | — |
| HU-13 Vista de salud (Admin) | `frontend_ui` + core | No iniciado | Delta-12 |
| HU-14 Métricas del modelo | `frontend_ui` + predicción | No iniciado | Delta-12 |
| HU-15 Configuración de parámetros | `frontend_ui` + core | No iniciado | — |
| HU-16, HU-17 Reportería del Gerente | `frontend_ui` + core | No iniciado | — |
| HU-18, HU-19, HU-20, HU-21 (MVP2) | varios | No iniciado | Delta-10, Delta-13 |

> **Nota de método:** esta matriz refleja el estado de la auditoría del 2026-05-18 (≈25% del backlog vivo). El propio acto de redactar este SDD —y el Sprint 4— mueven varios de estos estados; la matriz se actualiza con cada auditoría, no con cada commit. Claude Code la revalida contra el repo al confrontar el documento.

---

## 11. Estado de implementación y brecha

Esta sección confronta la arquitectura objetivo (el cuerpo del documento) con el estado real de construcción al momento de redacción, y caracteriza la brecha. No es una sección de diseño: es el reconocimiento honesto del punto de partida, en cumplimiento de la postura del documento (§0.1).

### 11.1 Magnitud de la brecha

La auditoría HU↔código (2026-05-18) clasificó 32 elementos del backlog: 1 completo (TTH-02), 5 parciales (TTH-03, TTH-08, TTH-10, HU-05, HU-06), 25 no iniciados y 1 fuera de scope (TTH-06). La cobertura implementada es de aproximadamente 25% del backlog vivo, lo que constituye una brecha estructural para el MVP1. El diseño que este documento describe es, en su mayor parte, arquitectura objetivo aún por construir; lo construido es el andamiaje de despliegue (Docker), parte del motor (calculadora sin persistencia ni integración) y parte de la visión (pipeline funcional que persiste a CSV).

### 11.2 Naturaleza de los deltas

Los 13 deltas de la auditoría no son homogéneos; se agrupan en cuatro naturalezas, y conviene leerlos por tipo más que por número.

Hay **conflictos entre el código y el backlog**, donde existe implementación pero con semántica distinta de la especificada. El motor de predicción expone hoy un endpoint con contrato divergente del diseño —per-cámara con RandomForest y niveles discretos, en vez de per-intersección con GRU y ratio continuo (Delta-01)—. El dashboard actual es una vista multi-intersección de supervisión de red, no la vista intra-intersección con colas por acceso que el backlog describe (Delta-06). La `ControlView` es un simulador interactivo tipo playground, no la vista pasiva del estado vigente que HU-05 especifica (Delta-08) —exactamente la tensión que motivó el mandato de estado vigente de §4—. El `reasoning` del motor se presenta como log técnico, no en lenguaje del dominio del Operador (Delta-09). La vista de alertas es una vista dedicada, no el banner transversal que HU-10 exige (Delta-11).

Hay **funcionalidad declarada en el backlog pero aún no construida**: la autenticación (Delta-02, con una inconsistencia de nomenclatura de roles entre fuentes a cerrar antes de implementar), la cobertura de CI sobre todos los módulos (Delta-03), la tabla `vision_aggregates` y su cableado (Delta-05), y —centralmente— la ausencia total de infraestructura de tiempo real (Delta-07) y de persistencia de decisiones del motor (Delta-10). Delta-07 y Delta-10 son precisamente las brechas que §5 (canal SSE) y §4 (`motor_decisions`) diseñan para cerrar; el diseño antecede a la construcción, como corresponde a un SDD.

Hay una **tensión entre el backlog y una restricción operativa**: el spec de visión pide reconstruir el módulo desde cero, mientras la regla de no-refactor de `edge_device/src/vision/` lo protege (Delta-04). La resolución registrada (decisión humana del 2026-05-18) es que el refactor se ejecutará cuando el sprint lo aborde, no antes; hasta entonces el código vigente es la base operativa.

Y hay **deuda de UI invertida y features huérfanas**: scaffolding visual construido antes que la integración de datos (Delta-12), y funcionalidades en el frontend sin HU que las respalde (Delta-13). Entre estas últimas, la integración con una API externa de IA generativa (Gemini) para generar reportes de incidentes merece atención particular: no está priorizada por el backlog y tiene implicaciones de privacidad y de envío de datos a un tercero que deben discutirse antes de conservarla. El diseño objetivo de este SDD no la contempla; §11 la señala para decisión metodológica explícita.

### 11.3 Plan de cierre de la brecha

El cierre de la brecha es el contenido del Sprint 4 y siguientes, no de este documento. El SDD aporta el destino —la arquitectura objetivo contra la cual cada delta se resuelve— y la matriz de §10 como mapa del punto de partida. La productivización (paso de la topología de una máquina al despliegue edge/servidor, §6.4) es brecha de otra naturaleza: no de funcionalidad faltante, sino de configuración de despliegue que el diseño ya admite sin cambios estructurales.

---

## 12. Anexos

### 12.1 Glosario

| Término | Definición |
|---|---|
| Approach (acceso) | Una de las ramas que entran a la intersección (norte, sur, este, oeste). |
| Fase | Conjunto de approaches con verde simultáneo. |
| Jam level | Nivel ordinal de congestión 0-5 presentado al usuario; se deriva del ratio continuo velocidad/flujo-libre (D-009). |
| Webster | Algoritmo clásico de cálculo de ciclo óptimo bajo demanda estable. |
| MaxPressure | Algoritmo reactivo que elige la próxima fase por presión de colas. |
| Capa MTC | Capa de cumplimiento normativo del Manual MTC peruano que corrige la salida estratégica (RNF-SAF-02). |
| Estado vigente | La estrategia actualmente activada en una intersección, con su timestamp de activación (`engine_active_state`, §4). |
| Delta | Discrepancia entre lo diseñado/especificado y lo construido, registrada por la auditoría. |
| As-designed | Postura del documento: describe la arquitectura objetivo sin marcar avance en la prosa (§0.1). |

### 12.2 Referencias a documentos del proyecto

Los documentos fuente del corpus (backlog, requisitos, decisiones, auditoría, planificación, modelo de datos, teoría del motor) se enumeran en §0.3. El SDD corresponde a los artefactos `plan.md` + `data-model.md` del proceso Spec Kit adoptado en modo brownfield (`SPECKIT_MAPPING.md`).

### 12.3 Referencias técnicas

- Webster, F. V. (1958). *Traffic signal settings*. Road Research Laboratory.
- Varaiya, P. (2013). *Max pressure control of a network of signalized intersections*. Transportation Research Part C, 36, 177-195.
- Koonce, P., et al. (2008). *Traffic Signal Timing Manual* (FHWA-HOP-08-024). Federal Highway Administration.
- Lopez, P. A., et al. (2018). *Microscopic traffic simulation using SUMO*. IEEE ITSC.
- Manual de Dispositivos de Control del Tránsito (R.D. N.° 26-2024-MTC/18, octubre 2024). Ministerio de Transportes y Comunicaciones del Perú.
- ISO/IEC 25010:2023. *Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model.*
- Kruchten, P. (1995). *Architectural Blueprints — The "4+1" View Model of Software Architecture.* IEEE Software.

---

# Anexo — Decisiones metodológicas de redacción del SDD (DHU-021)

Las meta-decisiones de redacción de este SDD —postura As-designed, proceso Spec Kit, estructura híbrida 4+1 / ISO 25010 / ADR, tratamiento de las colisiones del corpus, y los ajustes de diseño que surgieron al verificar el documento contra el código real— se consolidaron en **DHU-021**, cuyo hogar canónico es `documentation/lean-inception/4-decisiones/DECISIONS_HU.md`. La nomenclatura de roles del sistema (Delta-02) se cerró aparte como **DHU-022**, por ser decisión de producto y no de redacción.

Para evitar la duplicación de fuente, este SDD no reproduce el texto completo de esas decisiones: consúltense DHU-021 (Grupo 1 — 17 decisiones de redacción; Grupo 2 — 4 ajustes V1–V4 derivados de la verificación SDD↔repo) y DHU-022 en `DECISIONS_HU.md`.
