# Documento de Diseño de Software (SDD) — CerebroVial

> Sistema inteligente de control adaptativo de semáforos para una intersección de Miraflores, Lima.
> Proyecto de tesis de ingeniería de software · ejecución individual con asistencia de IA.
>
> **Estado del documento:** en construcción incremental, sección por sección.
> **Última actualización:** 2026-05-20.

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
