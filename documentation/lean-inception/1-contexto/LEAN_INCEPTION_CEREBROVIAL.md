# Lean Inception — CerebroVial

> Documento consolidado del Lean Inception realizado para el proyecto de tesis CerebroVial. Contiene los 7 artefactos generados durante el workshop adaptado al contexto académico.
>
> **Estado:** Versión 1.1 — Inception ejecutado, refinado durante la redacción incremental del backlog.
> **Fecha del workshop original:** 2026-05-11 (semana 6 de 15).
> **Fecha de última actualización:** 2026-05-14 (refinamientos consolidados en DHU-012; ver "Cambios respecto a la versión 1.0" al pie del documento).

---

## Índice

1. Contexto y metodología aplicada
2. Artefacto 1 — Objetivo del Producto
3. Artefacto 2 — Product Vision
4. Artefacto 3 — Matriz Es / No Es / Hace / No Hace
5. Artefacto 4 — Objetivos del Producto
6. Artefacto 5 — Personas
7. Artefacto 6 — User Journeys
8. Artefacto 7 — Feature Backlog (Brainstorming + Revisión)
9. Artefacto 8 — Sequencer (MVP1 / MVP2 / Trabajos Futuros)
10. Artefacto 9 — MVP Canvas para MVP1
11. Decisiones cerradas durante el Inception
12. Parking lot — para conversación con asesor
13. Próximos pasos
14. Cambios respecto a la versión 1.0

---

## 1. Contexto y metodología aplicada

### Marco metodológico

Se aplicó **Lean Inception** (Paulo Caroli, 2018) como marco metodológico para la definición de alcance del proyecto. La elección se justifica por:

- Compatibilidad con la estructura de Incepción Ágil sugerida por el curso (Desarrollo_Agil.pdf).
- Respaldo bibliográfico autorizado (Caroli, Fowler/Thoughtworks, Pichler).
- Producción de artefactos concretos y defendibles.
- Conexión natural con SCRUM como marco de ejecución posterior.

La aplicación es **complementaria, no alternativa, a SCRUM**: Lean Inception define **qué construir** (alcance); SCRUM define **cómo construirlo** (ejecución).

### Adaptaciones al contexto académico

Lean Inception está pensado para workshops colaborativos de 5 días con equipos de producto y stakeholders comerciales. Se aplicaron tres adaptaciones explícitas (documentadas en LEAN_INCEPTION_INVESTIGACION.md, sección 4):

1. **Equipo individual con asistencia de IA.** Las actividades colaborativas se sustituyeron por conversaciones estructuradas tesista–asistente, con validación posterior del asesor.
2. **Inception retroactivo.** El proyecto no parte de cero; tiene 6 semanas de código previo. El Inception se aplica para validar coherencia retroactiva del alcance y redefinir lo que sea necesario.
3. **MVP académico, no comercial.** El "MVP" es el subconjunto del producto que se sustenta en la defensa. Las hipótesis son académicas (validación de KPIs en simulación), no comerciales (adopción, retención).

---

## 2. Artefacto 1 — Objetivo del Producto

> **"Reducir los tiempos de espera y la congestión vehicular en intersecciones del distrito de Miraflores, en beneficio de conductores, peatones y vecinos de la zona."**

**Notas:**
- Beneficiario directo: la ciudadanía (conductores, peatones, vecinos).
- Alcance geográfico: Miraflores.
- El objetivo no contiene la solución (no menciona "control adaptativo", "predicción" ni "semáforos"), lo cual permite que sobreviva cambios técnicos.

---

## 3. Artefacto 2 — Product Vision

> **Para** la Municipalidad de Miraflores **que** requiere reducir los tiempos de espera y la congestión vehicular en sus intersecciones urbanas, **CerebroVial es un** sistema inteligente de control adaptativo de semáforos **que** ajusta dinámicamente los tiempos de luz verde anticipando la congestión y respondiendo al estado real del tráfico observado en la intersección. **A diferencia de** los sistemas de control fijo y de los sistemas actuados por sensores convencionales, **nuestro producto** integra predicción de congestión a corto plazo mediante aprendizaje profundo y selección automática entre múltiples estrategias de control según el estado predicho y observado.

**Plantilla utilizada:** Geoffrey Moore (popularizada por Roman Pichler).

**Notas:**
- Cliente / adoptante: Municipalidad de Miraflores.
- Beneficiario: la ciudadanía (declarado en Artefacto 1).
- La Vision sí menciona el "cómo" (aprendizaje profundo, múltiples estrategias) porque describe el producto concreto.

---

## 4. Artefacto 3 — Matriz Es / No Es / Hace / No Hace

### ES + HACE
- Es un sistema de control adaptativo de semáforos.
- Es un sistema de software, no hardware.
- Es desplegable mediante Docker en infraestructura estándar.
- Predice congestión a corto plazo (15 min) por intersección.
- Selecciona dinámicamente entre Webster, MaxPressure y MTC.
- Observa estado de la intersección mediante visión computacional.
- Valida cuantitativamente su desempeño mediante simulación SUMO.
- Provee dashboard de monitoreo en tiempo real.

### NO ES
- No es un sistema comercial productivizado (es prototipo académico).
- No es SCATS, SCOOT ni un sistema propietario equivalente.
- No es un sistema entrenado con datos reales de tráfico de Lima.
- No es un sistema de control de múltiples intersecciones coordinadas.
- No es una app para conductores ni para ciudadanos finales.
- No es un sistema de detección de infracciones ni de fiscalización.

### NO HACE
- No controla semáforos físicos reales (solo simulados en SUMO).
- No coordina ondas verdes entre intersecciones vecinas.
- No detecta tipos específicos de vehículos para priorización.
- No procesa datos reales de Waze ni de fuentes externas.
- No reconoce placas ni rostros (visión es solo conteo y métricas agregadas).
- No realiza optimización offline multi-día.
- No envía notificaciones a conductores ni a vecinos.

### ES PERO NO HACE
- (Vacío.)

---

## 5. Artefacto 4 — Objetivos del Producto

Los 4 objetivos del producto, en orden cronológico-causal:

1. **Observar el estado actual del tráfico** en intersecciones urbanas, proporcionando visibilidad en tiempo real al operador.

2. **Anticipar la congestión** mediante predicción de corto plazo basada en aprendizaje profundo.

3. **Adaptar el control de semáforos** seleccionando dinámicamente entre estrategias de control según el estado observado y predicho.

4. **Demostrar mejora cuantificable** mediante validación en simulación calibrada con la topología de la intersección de estudio, comparando el sistema propuesto contra control fijo.

**Notas:**
- Los 4 objetivos se alinean con los 4 módulos del sistema (visión, predicción GRU, motor adaptativo, validación SUMO).
- La regla de Caroli es 3 objetivos; se justifica el cuarto por su naturaleza irreducible: cada objetivo representa una capa funcional distinta.
- El orden es deliberado: cada objetivo habilita al siguiente.

---

## 6. Artefacto 5 — Personas

### Persona 1: Operador de Tráfico Municipal

| Rol | Detalles | Necesidad |
|---|---|---|
| **Operador de Tráfico Municipal** | Funcionario responsable del monitoreo en tiempo real del flujo vehicular en intersecciones del distrito. Trabaja desde un centro de operaciones con acceso a múltiples pantallas. Tiene formación técnica básica pero no es ingeniero ni programador. Su jornada incluye picos de actividad en horas pico (7-9am, 6-9pm). | • Visualizar el estado actual del tráfico en tiempo real • Anticipar congestiones antes de que escalen • Confiar en que el sistema toma decisiones de control acertadas sin intervención constante • Entender por qué el sistema tomó una decisión específica cuando algo va mal |

### Persona 2: Gerente de Tránsito Municipal

| Rol | Detalles | Necesidad |
|---|---|---|
| **Gerente de Tránsito Municipal** | Funcionario directivo del área de tránsito de la municipalidad. Responsable de evaluar el desempeño global del sistema de control de tráfico, justificar inversiones y reportar a niveles superiores. No opera el sistema en tiempo real; lo consulta para tomar decisiones estratégicas y reportar gestión. | • Evaluar la eficiencia del sistema de control de tráfico en periodos prolongados • Comparar el desempeño antes y después de la implementación • Justificar la inversión en el sistema con datos cuantitativos • Identificar tendencias o problemas recurrentes para planificación futura |

### Persona 3: Administrador del Sistema

| Rol | Detalles | Necesidad |
|---|---|---|
| **Administrador del Sistema** | Perfil técnico responsable de la configuración, mantenimiento y salud del sistema. A diferencia del Operador, no monitorea el tráfico — monitorea el sistema. Tiene formación en ingeniería de sistemas o software. Interviene de forma esporádica para configuraciones, ajustes de parámetros y supervisión técnica. | • Verificar que los componentes del sistema están operativos • Configurar parámetros del motor adaptativo (umbrales, pesos) • Consultar métricas de desempeño del modelo predictivo • Solicitar el reentrenamiento del modelo cuando los datos disponibles cambien significativamente |

### Fuera de Personas (documentados aparte)

- **Ciudadano de Miraflores** — beneficiario indirecto. No interactúa con el sistema. Documentado en Artefacto 1 (Objetivo) y Artefacto 2 (Vision).
- **Comité Evaluador** — stakeholder del proyecto académico, no usuario del producto. Documentado en sección de Stakeholders del proyecto de tesis si la rúbrica lo exige.

---

## 7. Artefacto 6 — User Journeys

### Journey 1: Operador — Turno de monitoreo en hora pico

| # | Acción | Estado emocional | Touchpoint |
|---|---|---|---|
| 1 | Inicia turno y accede al dashboard | Atento | Login + landing |
| 2 | Verifica el estado actual del tráfico | Atento a normalidad | Dashboard principal |
| 3 | Observa la predicción de congestión | Anticipa problemas | Panel de predicción |
| 4 | Identifica congestión próxima | Preocupación | Vista combinada |
| 5 | Verifica estrategia activa del motor | Confianza/duda | Panel del motor |
| 6 | Sistema cambia automáticamente de estrategia | Alivio | Notificación + log |
| 7 | Observa evolución tras el cambio | Validación visual | Dashboard actualizado |
| 8 | Registra incidencia *(MVP2)* | Necesita reportar | Módulo de notas |

### Journey 2: Gerente — Revisión semanal de desempeño

| # | Acción | Estado emocional | Touchpoint |
|---|---|---|---|
| 1 | Accede al sistema | Atención al detalle | Login |
| 2 | Selecciona periodo de análisis | Quiere comparar | Selector de periodo |
| 3 | Consulta KPIs agregados | Necesita números | Dashboard ejecutivo |
| 4 | Compara con periodos previos | Quiere tendencia | Vista comparativa |
| 5 | Detecta variación importante *(MVP2)* | Curiosidad | Vista detallada |
| 6 | Exporta reporte *(MVP2)* | Necesita formato | Exportación PDF/Excel |

### Journey 3: Administrador — Verificación de salud del sistema

| # | Acción | Estado emocional | Touchpoint |
|---|---|---|---|
| 1 | Accede al panel de administración | Inicia revisión | Login con permisos admin |
| 2 | Verifica estado de componentes | Asegurar operatividad | Vista de health check |
| 3 | Consulta métricas del modelo predictivo | Validar performance | Panel de métricas |
| 4 | Identifica degradación *(MVP2)* | Decisión técnica | Comparativa vs baseline |
| 5 | Ajusta parámetros del motor | Configuración fina | Formulario de configuración |
| 6 | Solicita reentrenamiento *(Trabajos Futuros)* | Acción crítica | Botón "reentrenar" |

### Journey 4: Operador — Operación degradada por falla de componente

| # | Acción | Estado emocional | Touchpoint |
|---|---|---|---|
| 1 | Está en turno normal | Atento | Dashboard normal |
| 2 | Recibe alerta de degradación (nivel 1, 2 o 3) o falla total | Preocupación proporcional | Banner / indicador transversal |
| 3 | Consulta detalle del componente afectado | Necesita entender | Panel de estado de componentes |
| 4 | Verifica modo de operación activo | Decide si interviene | Mensaje explicativo |
| 5 | Observa que el sistema sigue operando con fallback (o que ha cesado, en falla total) | Confianza condicional / escalamiento | Indicación en cada panel |
| 6 | Decide escalamiento al Admin *(MVP2)* | Decisión operativa | Botón de escalamiento |
| 7 | Continúa monitoreando hasta recuperación | Atención sostenida | Dashboard normal restaurado |

**Estados operativos del sistema (modelo refinado por DHU-008):**

| Estado | Disparador | Comportamiento del sistema |
|---|---|---|
| Operación normal | Todos los componentes operativos | El sistema opera con capacidades completas |
| Degradado nivel 1 | Componente periférico de detección de tráfico no responde | Motor adaptativo opera con el resto de información disponible (predicción + observaciones restantes) |
| Degradado nivel 2 | Componente predictivo principal no responde | Sistema activa predictor de respaldo de menor precisión que sigue produciendo predicciones vigentes |
| Degradado nivel 3 | Motor adaptativo no responde | Sistema aplica tiempos preconfigurados al semáforo |
| Falla total | Condición sin fallback aplicable | Sistema no aplica decisiones nuevas al semáforo; el último estado conocido se mantiene hasta intervención del Administrador |

**Propiedad clave:** *"Nunca empeoramos el statu quo."* En degradado nivel 3 (el peor estado con operación activa), el sistema opera como los semáforos actuales de Lima (tiempos fijos preconfigurados). En falla total, el sistema no degrada el estado del semáforo, lo mantiene.

> *Nota: el modelo de estados fue refinado durante la redacción del Bloque C. La versión original del Inception describía 3 niveles de degradación con vocabulario técnico (Falla visión / Falla GRU / Falla motor adaptativo); el modelo actual añade "Falla total" como cuarto estado no normal (DHU-008) y usa vocabulario agnóstico a la implementación (DHU-006). El identificador interno técnico del nivel 3 es `degraded_3` (anteriormente `safe_3` / "modo seguro", renombrado por DHU-012). Detalle técnico en TTH-04 (CT-04.2) del documento `TAREAS_TECNICAS_HABILITADORAS.md`.*

---

## 8. Artefacto 7 — Feature Backlog

Total: **41 features candidatas** identificadas (35 originales del Brainstorming + 6 fichas livianas de Trabajos Futuros formalizadas en DHU-012).

**Distribución por persona (post-DHU-012):**
- Operador: 14 features
- Gerente: 5 features
- Administrador: 5 features
- Sistema (transversal): 8 features
- Trabajos Futuros sin persona principal: 6 features (F36-F41)
- Que cruzan personas: variable

**Lista completa con anotaciones de revisión Técnica / UX / Negocio:** ver hoja de revisión adjunta (Artefacto 7 detallado) en `FEATURE_BACKLOG_DETALLADO.md`. Las features se clasifican en el siguiente artefacto (Sequencer).

---

## 9. Artefacto 8 — Sequencer (MVP1 / MVP2 / Trabajos Futuros)

### MVP1 — Sustentación de tesis (29 features, se construye)

**Bloque A — Infraestructura mínima (4 features):**
F01 Autenticación, F29 Roles y permisos, F30 Persistencia de estados, F31 Persistencia de decisiones.

**Bloque B — Operador, monitoreo (9 features MVP1):**
F02 Dashboard principal, F03 Flujo en tiempo real, F04 Cola por dirección, F05 Panel de predicción, F06 Vista combinada estado+predicción, F07 Panel del motor adaptativo, F08 Explicación de selección (nivel mínimo), F09 Notificación de cambio, F10 Log de decisiones.

> *Nota sobre la afinidad temática de F11 con este bloque:* F11 (Módulo de notas del Operador) no entra en el conteo MVP1 del Bloque B (es MVP2, ver subsección siguiente del Sequencer) pero pertenece temáticamente al Bloque B por ser soporte al monitoreo del Operador. Los documentos del backlog (`HU_BLOQUE_B.md`, `FEATURE_BACKLOG_DETALLADO.md`) la ubican dentro del Bloque B y la mapean a HU-09; el Sequencer la lista aparte porque cuenta features por su clasificación MVP, no por afinidad temática. Las dos convenciones son consistentes.

**Bloque C — Operador, degradación (6 features):**
F22 Indicador de estado degradado, F23 Vista simplificada de componentes, F24 Mensaje explicativo, F25 Indicación por panel, F26 Lógica de fallback en cascada, F27 Configuración de tiempos fijos del nivel 3.

**Bloque D — Administrador (3 features):**
F17 Salud de componentes, F18 Métricas del modelo, F20 Configuración de parámetros del motor.

**Bloque E — Componentes del sistema (4 features):**
F33 Módulo de visión, F34 Módulo predictivo GRU, F35 Motor adaptativo, F32 Integración con SUMO.

**Bloque F — Gerente (3 features):**
F12 Dashboard ejecutivo, F13 Selector de periodo, F14 Vista comparativa entre periodos.

> *Nota sobre el conteo (refinado por DHU-012): la versión original del Sequencer reportaba "26 features MVP1" en este título, lo cual resulta de un error de suma al consolidar los bloques. El conteo correcto es 29 (4+9+6+3+4+3). Algunas de esas 29 features se modelan como HU propia, otras como Criterios de Aceptación inglobados en otras HUs (F02 cubierta por composición visual del Bloque B; F30 inglobada en HUs del Bloque F; F31 inglobada en CA-08.1 de HU-08; F25 cubierta por composición de HU-10+HU-11+HU-12 según DHU-011), y otras como Tareas Técnicas Habilitadoras (F01 → TTH-01, F26 → TTH-04, F27 → TTH-05, además de TTH-02 y TTH-03 transversales). Ver `HU_BLOQUE_*.md` y `TAREAS_TECNICAS_HABILITADORAS.md` para el mapeo concreto.*

### MVP2 — Product Backlog candidato a holgura (5 features)

> *Refinamiento de semántica MVP2 (DHU-012): estas features se documentan como HU completa con criterios de aceptación, y se construyen condicional a la holgura del cronograma tras cerrar las MVP1. No son entregables comprometidos del MVP1, pero tampoco descartadas a priori: si el cronograma permite, entran al sprint.*

- F11 — Módulo de notas del Operador *(ya redactada como HU-09 del Bloque B)*.
- F15 — Vista detallada de periodo específico.
- F16 — Exportación de reportes a PDF/Excel.
- F19 — Comparativa de métricas del modelo vs baseline.
- F28 — Botón de escalamiento al Administrador.

**Justificación de la clasificación:** estas features son legítimas del producto. Tienen complejidad razonable y bajo riesgo de scope creep individual, pero su valor no es esencial para sustentar los 4 Objetivos del Producto. Si hay holgura tras MVP1, son candidatas naturales a entrar al sprint.

### Trabajos Futuros (7 direcciones, se documentan como fichas livianas, NO se construyen)

> *Refinamiento de DHU-012: la categoría originalmente llamada "MVP3" se renombra a "Trabajos Futuros" para reflejar con precisión su naturaleza (direcciones declaradas fuera del alcance del proyecto académico, candidatas a futuras extensiones del producto o de la investigación). Todas tienen ficha en el backlog detallado; ninguna se redacta como HU ni se construye dentro del alcance de la tesis. Se mencionan en el capítulo de trabajo futuro del documento de tesis.*

| ID | Título | Decisión técnica relacionada |
|---|---|---|
| F21 | Reentrenamiento del modelo predictivo (pipeline MLOps) | — |
| F36 | Reconocimiento de tipos de vehículos para priorización | — |
| F37 | Coordinación de ondas verdes entre intersecciones vecinas | D-006 |
| F38 | Procesamiento de datos reales de Waze | D-008 |
| F39 | Despliegue real en Raspberry Pi como dispositivo de borde | D-004 |
| F40 | Notificaciones push y monitoreo proactivo de cámaras | — |
| F41 | Integración cerrada del módulo de visión al loop de validación cuantitativa | D-007 |

> **F41 reafirmada en TTH-08 F9 (2026-05-29).** Sin cambios estructurales: integración vision→predictivo queda fuera de MVP1, ámbito reservado al ciclo F41. Ver `documentation/vision_contract.md` §6 (shape, columna `queue` nullable, calibración direccional postergada) y `DECISIONS_HU.md` DHU-024 addendum F9.

Ver fichas detalladas en `FEATURE_BACKLOG_DETALLADO.md`. Ver `EVOLUCION_TESIS.md` sección 8 para la conexión con el capítulo de trabajo futuro de la tesis.

---

## 10. Artefacto 9 — MVP Canvas para MVP1

### Bloque 1 — Propuesta del MVP

> CerebroVial MVP1 es un sistema integrado de control adaptativo de semáforos que observa el estado actual del tráfico, anticipa congestión a corto plazo mediante predicción con GRU, adapta dinámicamente la estrategia de control entre Webster, MaxPressure y MTC, y demuestra cuantitativamente su mejora frente a control fijo mediante simulación SUMO calibrada con la topología de una intersección de Miraflores. El sistema opera con degradación controlada en cascada ante falla de componentes, manteniendo siempre un comportamiento seguro y predecible.

### Bloque 2 — Personas Segmentadas

- **Personas:** Operador (principal), Administrador (técnico), Gerente (reportería).
- **Segmentación de validación:** Una sola intersección de Miraflores en SUMO. Sin segmentación adicional por turno o subconjunto de operadores.

### Bloque 3 — Jornadas

Las 4 cerradas en Artefacto 6.

### Bloque 4 — Features

Las 29 del MVP1 organizadas en Bloques A-F (ver Artefacto 8).

### Bloque 5 — Resultado Esperado

**Resultado primario:** Demostrar que un sistema integrado de control adaptativo con predicción mediante GRU mejora cuantitativamente los KPIs de tráfico de una intersección urbana frente a control fijo (Webster precalculado), en condiciones de simulación calibrada.

**Resultados secundarios:**
- Viabilidad técnica de integración entre módulo de visión, modelo predictivo, motor adaptativo y simulación.
- Operación con degradación controlada ante falla de componentes, sin empeorar el statu quo.
- Producción de un Product Backlog completo (MVP1 implementado + MVP2 documentado + Trabajos Futuros declarados).

### Bloque 6 — Métricas para validar hipótesis

**Métricas técnicas del sistema integrado (Obj. 4):**
- Tiempo promedio de espera por vehículo (segundos).
- Longitud máxima de cola por dirección (vehículos).
- Throughput de la intersección (vehículos/hora).
- Demora promedio acumulada en periodo de simulación.

**Métricas del modelo predictivo (Obj. 2):**
- MAE (Mean Absolute Error) y RMSE de la predicción de congestión.
- Sobre escenarios SUMO no vistos en entrenamiento.

**Métricas del módulo de visión (Obj. 1, validación independiente):**
- Precisión, recall, mAP de detección sobre dataset etiquetado.

**Métrica de robustez:**
- El sistema continúa operando bajo los 3 niveles de fallback sin degradar KPIs por debajo del control fijo.

**Criterio de éxito de la tesis:**
- **Al menos 2 de las 4 métricas técnicas del sistema integrado mejoran significativamente frente a control fijo.**
- **Ninguna de las 4 métricas empeora significativamente.**
- Métricas del modelo predictivo dentro de rango aceptable según literatura comparable.

### Bloque 7 — Costo y Cronograma

**Costo:** Cero costo monetario directo. Recursos: tiempo del tesista, infraestructura local (Docker, sin cloud — D-003).

**Cronograma:** 9 semanas (semana 6 a semana 15).

| Semana | Hito |
|---|---|
| 6 (actual) | Cierre de Inception. Inicio SUMO en paralelo. |
| 7 | SUMO funcional. Cierre de autenticación. |
| 8 | GRU entrenamiento sobre dataset SUMO. |
| 9 | GRU integración + persistencia. |
| 10 | Integración SUMO ↔ motor adaptativo. |
| 11 | Validación con/sin sistema. |
| 12 | Validación independiente del módulo de visión. |
| 13 | Actualización del documento de tesis. |
| 14 | Buffer + ensayo de defensa. |
| 15 | Entrega final. |

**Datos para validación disponibles al final de semana 11**, lo que da 4 semanas de margen para iterar el modelo o ajustar si los resultados son insuficientes.

---

## 11. Decisiones cerradas durante el Inception

Las siguientes decisiones se tomaron y cerraron durante el Inception (sujetas a confirmación con asesor):

1. **Alcance del producto:** Miraflores (no Lima en general).
2. **Cliente / adoptante:** Municipalidad de Miraflores.
3. **Beneficiario:** ciudadanía (conductores, peatones, vecinos), declarado pero no como persona del producto.
4. **Tres personas del producto:** Operador, Gerente, Administrador.
5. **Comité Evaluador NO es persona del producto** — es stakeholder del proyecto.
6. **Ciudadano NO es persona del producto** — es beneficiario indirecto.
7. **4 objetivos del producto** (no 3) — justificados por irreducibilidad funcional.
8. **Operación degradada con estados explícitos** — diseño formalizado durante el Bloque C (DHU-008): 3 niveles de degradación + falla total. Ver Journey 4 y TTH-04.
9. **F08 (explicación de selección):** nivel mínimo, texto predefinido por estrategia.
10. **5 features quedan en MVP2** — documentadas como HU completa; su construcción es condicional a la holgura del cronograma tras cerrar MVP1 (semántica refinada por DHU-012).
11. **Criterio de éxito:** al menos 2 de 4 métricas técnicas mejoran significativamente, ninguna empeora.

---

## 12. Parking lot — Para conversación con asesor

Items identificados durante el Inception que requieren validación específica del asesor (o del PO si llega a existir):

- ¿Opera sobre una intersección a la vez como definición de producto, o solo como alcance de validación?
- ¿Pi como hardware desplegado cuenta como "es" del producto? D-004 lo deja conceptual.
- ¿Autenticación multi-rol compleja, o login básico es suficiente para MVP1?
- ¿Validación de campo en Miraflores es factible en algún horizonte? Si no, queda como trabajo futuro.
- ¿Se debe incluir alguna jornada adicional (turno nocturno, fin de semana, Gerente preparando reunión)?
- ¿La sección de "Stakeholders del proyecto" debe formar parte del documento de tesis?

---

## 13. Próximos pasos

Tras validación del Showcase con el asesor:

1. **Conversión de las 29 features de MVP1 a HUs en formato "Como X, quiero Y, para Z"** con criterios de aceptación Given-When-Then. Estimadas con Planning Poker y priorizadas con MoSCoW. Algunas features se modelan como Criterios de Aceptación inglobados o como Tareas Técnicas Habilitadoras según corresponda; ver `HU_BLOQUE_*.md` y `TAREAS_TECNICAS_HABILITADORAS.md`.

2. **Conversión de las 5 features de MVP2 a HUs documentadas** con la misma estructura, marcadas como candidatas a holgura del sprint.

3. **Documentación de las 7 direcciones de Trabajos Futuros** como fichas livianas en `FEATURE_BACKLOG_DETALLADO.md`. No se redactan como HU.

4. **Definición de Sprint Goals** para las 9 semanas, siguiendo el cronograma del Bloque 7 del MVP Canvas.

5. **Inicio de SDD (Spec-Driven Development)** sobre las HUs prioritarias del sprint.

---

## 14. Cambios respecto a la versión 1.0

Esta sección documenta los refinamientos al Inception desde la versión 1.0 (workshop original del 2026-05-11). Todos los cambios están consolidados en DHU-012 (`DECISIONS_HU.md`).

### Cambios sustantivos

- **Renombrado de la categoría "MVP3" a "Trabajos Futuros"** en el Sequencer. Las direcciones de trabajo futuro ahora se documentan como fichas de feature (F21, F36-F41) en `FEATURE_BACKLOG_DETALLADO.md`, con asimetría justificada (F21 conserva ficha completa del Brainstorming original; F36-F41 son fichas livianas).

- **Semántica de MVP2 refinada:** ahora se documenta como HU completa y se construye condicional a holgura del cronograma tras cerrar MVP1 (anteriormente: "se documenta pero no se construye").

- **Conteo de features MVP1 corregido a 29** (anteriormente: 26 por error aritmético al consolidar los bloques del Sequencer).

- **Conteo total de features actualizado a 41** (35 originales del Brainstorming + 6 fichas livianas de Trabajos Futuros F36-F41 agregadas para formalizar las direcciones que originalmente vivían como prosa).

- **Journey 4 reescrito con 4 estados** (Degradado nivel 1, Degradado nivel 2, Degradado nivel 3, Falla total) y vocabulario agnóstico a la implementación, según DHU-006 y DHU-008. La versión original describía 3 niveles con vocabulario técnico (Falla visión / Falla modelo / Falla motor adaptativo).

- **Renombrado uniforme "modo seguro" → "degradado nivel 3"** para cohesionar el vocabulario de niveles. Identificador interno técnico `safe_3` → `degraded_3`. Aplica a Journey 4 y a todas las referencias del backlog.

### Cambios de higiene documental

- **Eliminación de referencias al régimen previo al Inception:** se removieron menciones a `PLAN.md`, "Fase N del PLAN", "Bloque K/J/F del TODO" (no confundir con los Bloques A-F del Sequencer del Inception, que son legítimos).
- **Conteo de decisiones técnicas actualizado:** "D-001 a D-008" → "D-001 a D-009" en la sección de Documentos relacionados.
- **Eliminación de la referencia a `tesis/(2).docx`** (era copia temporal del documento de tesis; el documento final no se ha cerrado).

### Cambios menores

- Cabecera con versión 1.1, fecha del workshop original y fecha de última actualización.
- Sección 14 (esta) agregada al pie del documento.
- Decisión cerrada #10 (sección 11) actualizada para reflejar la nueva semántica de MVP2.
- Decisión cerrada #8 (sección 11) actualizada para reflejar el modelo de 4 estados de DHU-008.
- **Aclaración sobre el conteo del Bloque B** (agregada en pase de higiene cruzada con los documentos del backlog): en el Sequencer, Bloque B = 9 features MVP1 (F02 a F10). F11 (Módulo de notas del Operador) pertenece temáticamente a este bloque por ser soporte al monitoreo del Operador, pero no entra en el conteo MVP1 porque está clasificada como MVP2 (ver subsección MVP2 más abajo). Los documentos del backlog (`HU_BLOQUE_B.md`, `FEATURE_BACKLOG_DETALLADO.md`) la ubican dentro del Bloque B; el Sequencer la lista aparte. Ambas convenciones son consistentes.
- **Limpieza de etiquetas en User Journeys (sección 7).** Los pasos de las cuatro jornadas marcados con `(fuera del sprint)` se actualizan a `(MVP2)` para alinearse con la semántica refinada en DHU-012 (MVP2 = HU documentada con construcción condicional a holgura, no descartada a priori). Aplica a Journey 1 paso 8 (HU-09), Journey 2 pasos 5 y 6 (HU-18 y HU-19), Journey 3 paso 4 (HU-20) y Journey 4 paso 6 (HU-21). El paso 6 de Journey 3 (reentrenamiento del modelo, F21) conserva `(Trabajos Futuros)` que ya estaba correcto.

---

## Documentos relacionados

- `DECISIONS.md` — Registro formal de decisiones técnicas (D-001 a D-009).
- `DECISIONS_HU.md` — Decisiones metodológicas sobre la redacción del Product Backlog (DHU-001 a DHU-022).
- `EVOLUCION_TESIS.md` — Narrativa de las 4 fases del proyecto; sección 8 contiene la tabla de Trabajos Futuros.
- `LEAN_INCEPTION_INVESTIGACION.md` — Fundamentación del marco metodológico.
- `documentation/docs/DISCOVERY_2026-05-10.md` — Auditoría inicial del repositorio.
- `FEATURE_BACKLOG_DETALLADO.md` — Detalle completo de las 41 features (29 MVP1 + 5 MVP2 + 7 Trabajos Futuros).
- `HU_BLOQUE_A.md`, `HU_BLOQUE_B.md`, `HU_BLOQUE_C.md`, `HU_BLOQUE_D.md`, `HU_BLOQUE_E.md`, `HU_BLOQUE_F.md` — Product Backlog del MVP1 redactado por bloques. **MVP1 redactado completo al cierre del Bloque F el 2026-05-16 (DHU-016): 16 HUs operativas MVP1 (HU-01 a HU-08 + HU-10 a HU-17) y 11 TTH (TTH-01 a TTH-11).** El Bloque E cerró sin HUs operativas (5 TTH, DHU-015); el Bloque F cierra con 2 HUs operativas + 0 TTH nuevas (F12+F13 fusionadas en HU-16 con F30 inglobada; F14 como HU-17).
- `HU_MVP2.md` — Product Backlog del MVP2 (cerrado el 2026-05-16 por DHU-017): HU-18 (drill-down del Gerente), HU-19 (exportación PDF/Excel), HU-20 (comparativa del modelo vs respaldo, Administrador) y HU-21 (escalamiento del Operador al Administrador). HU-09 (notas e incidencias del Operador, F11 del MVP2) reside físicamente en `HU_BLOQUE_B.md` desde su redacción anticipada en el cierre del Bloque B. **Con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 21 HUs operativas (HU-01 a HU-21) y 11 TTH (TTH-01 a TTH-11); 5 features del MVP2 + 0 TTH nuevas.**
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH transversales (TTH-01 a TTH-03), del Bloque C (TTH-04, TTH-05), Trabajos Futuros (TTH-06) y del Bloque E (TTH-07 a TTH-11).
