# Product Backlog — Vista de Conjunto

> Punto de entrada al Product Backlog del proyecto CerebroVial. Sistema inteligente de control adaptativo de semáforos para la intersección de Miraflores (Lima).
>
> **Para qué sirve este documento:** entender el producto completo en 10 minutos sin tener que abrir las HUs largas. Si quieres lectura más rápida pero con CAs, ver `HU_LITE.md`. Si necesitas el detalle implementable y auditable, ir a `HU_BLOQUE_*.md` y `HU_MVP2.md`.
>
> **Fecha de cierre del backlog (componente funcional):** 2026-05-16.

---

## Las 4 Personas del producto

**Operador de Tráfico Municipal.** Trabaja en turnos vigilando el control del semáforo en tiempo real. Su jornada se centra en observar el estado del tráfico, entender qué hace el motor adaptativo y por qué, reaccionar ante alertas del sistema y dejar constancia de incidencias para el siguiente turno.

**Gerente de Tránsito Municipal.** No trabaja en tiempo real. Consulta el desempeño histórico del sistema para reportar a niveles superiores, justificar decisiones ante áreas pares y sustentar análisis ejecutivos con evidencia.

**Administrador del Sistema.** Responsable técnico del producto. Supervisa la salud del sistema, evalúa el desempeño del modelo predictivo y ajusta parámetros operativos. Recibe escalamientos del Operador cuando algo excede su capacidad de resolución.

**Usuario del sistema (sujeto compuesto).** Cualquiera de los tres anteriores cuando el comportamiento es común. Se usa solo en HU-01 (acceso por rol).

---

## Los 4 Objetivos del producto

1. **Reducir tiempos de espera** en la intersección de Miraflores frente al control fijo actual.
2. **Sustentar la decisión técnica** del control adaptativo con métricas comparativas auditables.
3. **Garantizar continuidad operativa** ante fallos de componentes del sistema (visión, modelo predictivo, motor adaptativo).
4. **Generar evidencia gerencial** que permita extender el producto a otras intersecciones.

---

## Mapa del Product Backlog

**21 HUs operativas** (HU-01 a HU-21) + **11 Tareas Técnicas Habilitadoras** (TTH-01 a TTH-11).
- **MVP1 cerrado:** 16 HUs (HU-01 a HU-08, HU-10 a HU-17) + 11 TTH.
- **MVP2 cerrado:** 5 HUs adicionales (HU-09, HU-18 a HU-21) + 0 TTH nuevas.
- **5 features adicionales** declaradas como Trabajos Futuros (F36 a F41 + F21 reclasificada).

---

## Las 21 Historias de Usuario

### Acceso al sistema

| HU | Título | Persona | Qué hace | Objetivo |
|---|---|---|---|---|
| HU-01 | Acceso diferenciado por rol | Usuario (los 3 roles) | El sistema reconoce al usuario autenticado y le muestra las vistas correspondientes a su rol. Operador, Gerente y Administrador tienen interfaces distintas. | Soporte transversal |

### Operador — núcleo de monitoreo en tiempo real (Bloque B)

| HU | Título | Qué hace | Objetivo |
|---|---|---|---|
| HU-02 | Monitoreo del estado actual de la intersección | El Operador ve el flujo vehicular y la longitud de cola en cada acceso, actualizado automáticamente. | 1, 3 |
| HU-03 | Visualización de predicción de congestión a corto plazo | El Operador ve el nivel de congestión predicho para los próximos minutos. | 1 |
| HU-04 | Vista combinada del estado actual y la predicción | El Operador ve estado presente y predicción en una sola vista integrada, sin cambiar de pantalla. | 1 |
| HU-05 | Visualización de la estrategia de control activa | El Operador ve cuál de las estrategias del motor está vigente en este momento (Webster, MaxPressure o respaldo MTC). | 1, 2 |
| HU-06 | Explicación de la razón de selección de estrategia | El Operador entiende, en lenguaje legible, por qué el motor eligió la estrategia activa. | 2 |
| HU-07 | Notificación de cambios de estrategia del motor | El Operador es notificado pasivamente cuando el motor cambia de estrategia, sin tener que estar monitoreando continuamente. | 1, 2 |
| HU-08 | Consulta del historial de decisiones del motor | El Operador consulta cronológicamente qué decisiones tomó el motor en periodos pasados, con razón y parámetros. Auditable. | 2 |
| HU-09 | Registro de notas e incidencias del turno *(MVP2)* | El Operador registra notas de su turno y las consulta. Todos los Operadores ven todas las notas (transmisión entre turnos). | Soporte al Operador |

### Operador — operación degradada (Bloque C)

| HU | Título | Qué hace | Objetivo |
|---|---|---|---|
| HU-10 | Alerta activa transversal del estado operativo | El sistema muestra una alerta visible en todas las vistas del Operador cuando entra en modo degradado o falla total. No pasa desapercibida. | 3 |
| HU-11 | Vista del estado operativo de los componentes | El Operador ve qué componentes del sistema están funcionando y cuáles no. Vista simplificada (sin detalles técnicos). | 3 |
| HU-12 | Explicación del modo degradado activo | El Operador entiende qué capacidad operativa se perdió, qué estrategia de control se está usando como respaldo, y qué debe hacer él como consecuencia. | 3 |

### Administrador — soporte técnico (Bloque D)

| HU | Título | Qué hace | Objetivo |
|---|---|---|---|
| HU-13 | Vista técnica de salud de los componentes | El Administrador ve el estado técnico detallado de cada componente: latencia, errores recientes, timestamps. Más profundo que HU-11. | 3 |
| HU-14 | Vista de métricas de desempeño del modelo predictivo | El Administrador evalúa qué tan confiable es el modelo principal sobre datos operacionales recientes (MAE, RMSE, accuracy, matriz de confusión). | 2 |
| HU-15 | Configuración de parámetros operativos | El Administrador ajusta umbrales, ventanas temporales y frecuencias del sistema. Cambios auditables. | Soporte técnico |

### Gerente — reportería (Bloque F)

| HU | Título | Qué hace | Objetivo |
|---|---|---|---|
| HU-16 | Consulta de KPIs operativos sobre periodo seleccionable | El Gerente ve los 4 KPIs principales (tiempo de espera, longitud de cola, throughput, demora) sobre un periodo que él elige. | 1, 4 |
| HU-17 | Vista comparativa entre periodos | El Gerente compara el periodo actual con el periodo previo equivalente. Indicadores de mejora o empeoramiento. | 1, 4 |

### MVP2 — extensiones condicionales a holgura de cronograma

| HU | Título | Persona | Qué hace | Objetivo |
|---|---|---|---|---|
| HU-18 | Vista detallada de periodo específico (drill-down) | Gerente | El Gerente investiga qué pasó durante un periodo específico, con tres carriles temporales integrados: tráfico, decisiones del motor, intervalos de estado operativo. | 4 |
| HU-19 | Exportación de reportes a PDF o Excel | Gerente | El Gerente descarga reportes de HU-16 o HU-17 en PDF presentable o Excel con datos crudos. | 4 |
| HU-20 | Comparativa modelo predictivo principal vs respaldo | Administrador | El Administrador compara métricas de los dos modelos predictivos sobre los mismos eventos operacionales. Sustenta decisiones sobre el modelo. | 2 |
| HU-21 | Escalamiento de incidentes del Operador al Administrador | Operador → Administrador | El Operador escala incidentes formalmente al Administrador con captura automática del contexto operativo, trazabilidad y badge de pendientes en navegación. | 3 |

---

## Las 11 Tareas Técnicas Habilitadoras

Las TTH son trabajo técnico necesario para que el producto funcione, pero no entregan valor al usuario por sí solas. Se documentan por separado para que el equipo las estime y construya.

| TTH | Qué cubre | Bloque |
|---|---|---|
| TTH-01 | Autenticación (login con usuario y contraseña; sesiones JWT). | Transversal |
| TTH-02 | Entorno de desarrollo y despliegue local (Docker, configuración base). | Transversal |
| TTH-03 | Pipeline de integración continua (CI). | Transversal |
| TTH-04 | Lógica de fallback en cascada del sistema (cómo el sistema decide en qué modo operativo entrar y cuándo activar respaldos). | Operador degradado |
| TTH-05 | Tiempos preconfigurados del modo degradado nivel 3 (cuando todo falla, se usa control fijo conocido como respaldo). | Operador degradado |
| TTH-06 | Capa de DTOs entre frontend y backend *(reclasificada a Trabajos Futuros)*. | (futuro) |
| TTH-07 | Integración con SUMO para simulación del entorno (validación académica). | Componentes centrales |
| TTH-08 | Módulo de visión computacional que produce métricas del estado del tráfico. | Componentes centrales |
| TTH-09 | Modelo predictivo GRU servido vía API (modelo principal). | Componentes centrales |
| TTH-10 | Motor adaptativo (Webster, MaxPressure, MTC). | Componentes centrales |
| TTH-11 | Spike de calibración de hiperparámetros temporales del modelo predictivo. | Componentes centrales |

---

## Trabajos Futuros declarados

7 features fueron formalizadas como Trabajos Futuros (no se construyen en este proyecto académico; se mencionan en el capítulo de trabajo futuro de la tesis):

- F21: Reentrenamiento periódico del modelo predictivo.
- F36-F40: Notificaciones push, integración con sistemas externos, multi-intersección, dashboard ejecutivo extendido, y otras.
- F41: Reportes recurrentes programados.

Ver `FEATURE_BACKLOG_DETALLADO.md` para el detalle.

---

## Cómo navegar el backlog

Según para qué lo necesites, abre uno u otro documento:

- **Para entender el producto sin detalle implementable** → este documento.
- **Para lectura humana de las HUs en formato corto** → `HU_LITE.md`.
- **Para implementación o auditoría rigurosa** → los documentos `HU_BLOQUE_A.md` a `HU_BLOQUE_F.md` (MVP1) y `HU_MVP2.md` (MVP2). Cada HU tiene criterios de aceptación numerados, notas técnicas y candidatos a RNF.
- **Para razonamiento metodológico** → `DECISIONS_HU.md` (DHU-001 a DHU-024).
- **Para decisiones técnicas del producto** → `DECISIONS.md` (D-001 a D-009).
- **Para el origen de cada feature y su clasificación MVP** → `FEATURE_BACKLOG_DETALLADO.md`.
- **Para las Tareas Técnicas Habilitadoras** → `TAREAS_TECNICAS_HABILITADORAS.md`.
- **Para el contexto del proyecto académico completo** → `LEAN_INCEPTION_CEREBROVIAL.md` y `EVOLUCION_TESIS.md`.

---

## Estado al cierre del Product Backlog

El componente funcional del backlog está completo: las 21 HUs y las 11 TTH están redactadas y aprobadas. Quedan tres entregables del proyecto que dependen de este cierre:

1. **Documento de Requisitos Funcionales y No Funcionales (RF/RNF)** consolidando los candidatos a RNF de todas las HUs.
2. **Ceremonias de estimación (Planning Poker) y priorización (MoSCoW)** sobre el backlog completo.
3. **Implementación SCRUM del MVP1**. El MVP2 entra al sprint si el cronograma permite holgura.
