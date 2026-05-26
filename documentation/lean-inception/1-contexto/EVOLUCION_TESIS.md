# Evolución del Proyecto de Tesis — CerebroVial

> Documento de contexto y narrativa de evolución del proyecto. Captura las decisiones de alcance, los giros arquitectónicos y la consolidación final del trabajo. **Pensado como insumo para el capítulo de introducción/contexto de la tesis y para la sustentación oral.**

**Fecha de redacción:** 2026-05-11 (semana 6 de 15)
**Estado:** Versión consolidada tras cierre arquitectónico

---

## 1. Resumen ejecutivo

CerebroVial es un sistema integrado de control adaptativo de tráfico urbano que combina **predicción de congestión mediante GRU**, **observación de estado mediante visión computacional**, y **selección adaptativa de estrategia de control** (Webster o Max Pressure según condiciones, con una capa de reglas duras MTC que asegura cumplimiento normativo) según el estado predicho y observado. La validación cuantitativa se realiza mediante simulación SUMO comparando el sistema propuesto contra control fijo.

El proyecto pasó por **tres fases conceptuales** antes de llegar a su forma actual. Esta narrativa documenta esa evolución porque cada fase aportó un componente que sobrevive en el sistema final, y porque entender el recorrido es relevante para sustentar la coherencia del alcance final.

---

## 2. Fase 1 — La idea inicial (visión como fuente de predicción)

**Periodo aproximado:** Inicio del proyecto.

**Hipótesis original:** Las cámaras urbanas capturan tráfico en tiempo real → de esa captura se generan series temporales de flujo → un modelo predictivo se entrena sobre esas series → el motor de control consume las predicciones.

**Lo que se construyó:**

- Módulo de visión computacional con YOLO para detección y conteo de vehículos.
- Pipeline de captura desde streams de YouTube (de distintas ciudades, por falta de acceso a cámaras de Lima).
- Arquitectura `edge_device` separada, pensando en Raspberry Pi como dispositivo de borde.
- Estructura inicial de historias de usuario centradas en captura, procesamiento y análisis de imagen.

**Limitaciones encontradas:**

1. Los streams de YouTube no son fiables: pueden apagarse, cerrarse, cambiar de ubicación o ángulo sin aviso.
2. Las cámaras no son de Miraflores, lo cual rompe la coherencia con el caso de estudio declarado.
3. No hay ground truth para validar las detecciones cuantitativamente.
4. Entrenar un modelo predictivo sobre datos derivados de detecciones introduce ruido sobre ruido: errores de detección se propagan al entrenamiento.

**Aporte que sobrevive:** El módulo de visión funcional, con detección y conteo, queda como **componente del sistema en su rol de sensor de estado** (no como fuente de entrenamiento). Se demuestra como MVP en las primeras semanas del proyecto.

---

## 3. Fase 2 — La exploración del modelo predictivo (STGNN y RandomForest)

**Periodo aproximado:** Fase intermedia, durante la consolidación arquitectónica.

**Hipótesis revisada:** El modelo predictivo merece una arquitectura más sofisticada que un baseline simple. Se exploran arquitecturas espacio-temporales (Spatio-Temporal Graph Neural Networks) que han mostrado resultados competitivos en la literatura de predicción de tráfico (METR-LA, PEMS-BAY).

**Lo que se construyó:**

- Exploración de `time_then_space.py`: arquitectura Time-then-Space (encoder lineal + RNN/GRU temporal + DiffConv espacial + MLPDecoder).
- Entrenamiento experimental sobre dataset METR-LA con varios checkpoints generados.
- En paralelo, `RandomForestPredictor` como baseline de fallback para mantener el endpoint `/predictions/predict` operativo mientras la arquitectura neuronal se estabilizaba.

**Limitaciones encontradas:**

1. STGNN requiere definir un **grafo espacial** entre nodos (intersecciones, detectores). La validación de la tesis se enfoca en **una sola intersección de Miraflores**, lo cual hace la componente espacial inaplicable.
2. La complejidad arquitectónica (tsl, PyTorch Lightning, DiffConv) excede el tiempo disponible para debuggear, justificar y defender.
3. METR-LA es un dataset de red urbana de Los Ángeles, no transferible directamente al caso de estudio.

**Aporte que sobrevive:**

- El **RandomForestPredictor** queda en el código como baseline de fallback y como referencia de comparación en la sección de validación de la tesis (modelo simple vs modelo propuesto).
- La exploración de STGNN se documenta como **fundamentación teórica del trabajo futuro**: la línea natural de extensión del trabajo es escalar la predicción a múltiples intersecciones con dependencia espacial.
- La revisión de literatura realizada en esta fase sustenta la elección de GRU como familia arquitectónica para predicción de series temporales de tráfico.

---

## 4. Fase 3 — El motor adaptativo (control sobre la predicción)

**Periodo aproximado:** Semanas previas al cierre arquitectónico (commits del 7-9 de mayo 2026).

**Hipótesis:** El valor del sistema no está solo en predecir congestión, sino en **actuar sobre esa predicción**. Un motor que combina **dos estrategias adaptativas de control clásicas (Webster, Max Pressure)** seleccionadas dinámicamente según el estado predicho y observado, **con una capa de reglas duras (MTC) que asegura cumplimiento del marco normativo peruano**, es el aporte central de ingeniería del trabajo.

**Lo que se construyó:**

- Implementación de **dos estrategias adaptativas + una capa de reglas duras** en `core_management_api/src/control/`:
  - **Webster (estrategia adaptativa, modo off-peak):** asignación de tiempos de verde proporcional a la demanda histórica.
  - **Max Pressure (estrategia adaptativa, modo peak):** estrategia descentralizada basada en presión de colas.
  - **Capa MTC (Manual de Tránsito MTC peruano, reglas duras):** corrige los tiempos calculados por la estrategia activa para cumplir el marco normativo (R.D. N.° 26-2024-MTC/18). No decide tiempos adaptativos; corrige los decididos por Etapa 1 y compone la secuencia final aplicada al semáforo.
- **AdaptiveEngine:** pipeline de dos etapas. Etapa 1 selecciona la estrategia adaptativa según `flow_total` (umbral parametrizable, default 1500 veh/h); Etapa 2 aplica las correcciones de la capa MTC.
- Frontend completo de visualización del motor (`views/control/`, 9 archivos, 1034 líneas).
- Documentación teórica en `CONTROL.md` (552 líneas).
- Tests unitarios y de integración pasando.

> *Nota agregada el 2026-05-15 al cerrar TTH-10 (DHU-015): la descripción original de Fase 3 hablaba de "tres estrategias de control" (Webster, MaxPressure, MTC). La revisión arquitectónica de TTH-10 clarificó que MTC no es una estrategia adaptativa intercambiable con las otras dos, sino una **capa de reglas duras post-procesamiento**. La arquitectura real es de dos etapas (estrategia adaptativa + capa MTC), no un selector tripartita. El componente construido es el mismo; cambia solo la descripción para coherencia con `CONTROL.md` y TTH-10. Ver DHU-015 en `DECISIONS_HU.md`.*

**Reflexión sobre el orden de ejecución:** El motor adaptativo se construyó antes de cerrar el trabajo de autenticación y persistencia de visión. Esto rompe el orden formal originalmente planeado pero es defendible: el motor es el componente más visible y académicamente diferenciador del sistema; trabajarlo temprano permite estabilizarlo y testearlo con tiempo.

**Aporte que sobrevive:** **Todo.** El motor adaptativo es el componente más maduro del sistema al momento del cierre arquitectónico y se mantiene como el corazón del sistema de control.

---

## 5. Fase 4 — El cierre arquitectónico (semana 6, donde estamos hoy)

**Periodo:** 2026-05-11.

**Hipótesis consolidada:** El proyecto se reorganiza alrededor de cuatro componentes claramente definidos, con roles y validaciones separadas:

| Componente | Rol | Fuente de datos | Validación |
|---|---|---|---|
| **Visión computacional** | Sensor de estado en tiempo real (cola, flujo, densidad observados) | Streams/video (demo) | Métricas de detección (precisión, recall) sobre dataset etiquetado |
| **Predictor GRU** | Predicción de congestión a corto plazo | Dataset sintético SUMO | Métricas de predicción (MAE, RMSE) sobre escenarios SUMO no vistos |
| **Motor adaptativo** | Selección de estrategia de control según estado predicho + observado | Predicción GRU + métricas de estado | KPIs de tráfico (tiempo de viaje, colas, demoras) |
| **Validación SUMO** | Comparación cuantitativa con/sin sistema | Topología de Miraflores + patrones de demanda | KPIs comparativos vs Webster fijo |

**Decisiones clave de esta fase (documentadas en DECISIONS.md):**

- **D-006:** GRU univariado por intersección. Se descarta STGNN. Dependencia espacial entre intersecciones → trabajo futuro.
- **D-007:** Visión como componente demostrable, con validación independiente. No participa en el loop de validación cuantitativa.
- **D-008:** SUMO genera **tanto** el dataset de entrenamiento del GRU **como** los escenarios de validación. Datos reales de Waze → trabajo futuro o bono.

**Por qué este cierre es coherente:**

1. **Cada componente tiene un rol único y una validación independiente.** No hay solapamientos ni ambigüedades.
2. **No hay dependencias externas críticas.** El sistema entero vive en un entorno controlable (SUMO + código propio). No depende de Waze, de YouTube, ni de acuerdos administrativos.
3. **La narrativa académica es defendible.** El alcance está explícitamente declarado: validación simulada sobre una intersección, generalización a red urbana como trabajo futuro.
4. **El trabajo previo no se desperdicia.** Visión, RandomForest baseline y exploración de STGNN sobreviven con roles claros: MVP demostrable, baseline de comparación y fundamentación de trabajo futuro respectivamente.

---

## 6. Narrativa para la sustentación

El siguiente texto puede servir como base para el capítulo de introducción/contexto de la tesis o para la apertura de la sustentación oral:

> *"El presente trabajo evolucionó a través de varias iteraciones conceptuales antes de consolidar su alcance final. La idea inicial planteaba el uso de cámaras urbanas como fuente principal de datos para entrenar un modelo predictivo de tráfico. Durante la construcción del módulo de visión computacional —que se mantiene en el sistema final como sensor de estado en tiempo real— se identificaron limitaciones prácticas en la fiabilidad y representatividad de los streams disponibles, lo cual motivó replantear el rol de la visión dentro del sistema.*
>
> *En una segunda fase se exploraron arquitecturas espacio-temporales (STGNN) para la predicción, en línea con la literatura reciente del área. Esta exploración —que se documenta como sustento del trabajo futuro— se acotó al constatar que el alcance de validación del trabajo se centra en una intersección individual, condición bajo la cual la componente espacial no aplica directamente. En su lugar, se adoptó GRU univariado como modelo predictivo, manteniendo RandomForest como baseline de comparación.*
>
> *Paralelamente se desarrolló el componente central del sistema: un motor adaptativo que selecciona dinámicamente entre dos estrategias de control clásicas (Webster y Max Pressure) según el estado predicho por el modelo y el estado observado, complementado por una capa de cumplimiento normativo basada en el Manual de Dispositivos de Control del Tránsito Automotor del Ministerio de Transportes y Comunicaciones del Perú (R.D. N.° 26-2024-MTC/18). Este motor constituye el aporte de ingeniería principal del trabajo.*
>
> *El sistema final integra los tres componentes (visión, predicción GRU, motor adaptativo) y se valida cuantitativamente mediante simulación SUMO, comparando los KPIs de tráfico bajo el sistema propuesto contra los obtenidos con control fijo (Webster). Tanto el dataset de entrenamiento del modelo predictivo como los escenarios de validación se generan en SUMO con particiones independientes, asegurando consistencia metodológica. La calibración del modelo con datos reales de tráfico urbano y la extensión a redes de múltiples intersecciones interrelacionadas se declaran como las dos líneas naturales de trabajo futuro."*

---

## 7. Mapa de aportes que sobreviven (referencia rápida)

| Fase | Aporte | Rol en sistema final |
|---|---|---|
| Fase 1 (visión inicial) | Módulo de visión con YOLO | Sensor de estado / componente demostrable (D-007) |
| Fase 2 (exploración predicción) | RandomForestPredictor | Baseline de comparación en validación |
| Fase 2 (exploración predicción) | Revisión literatura STGNN | Fundamentación de trabajo futuro |
| Fase 3 (motor adaptativo) | Webster + Max Pressure (estrategias adaptativas) + capa MTC (reglas duras) + AdaptiveEngine | Componente central del sistema (aporte de ingeniería) |
| Fase 3 (motor adaptativo) | Frontend control (`views/control/`) | Visualización del estado del sistema |
| Fase 4 (cierre) | GRU univariado | Modelo predictivo final (D-006) |
| Fase 4 (cierre) | SUMO end-to-end | Columna vertebral de datos y validación (D-008) |

---

## 8. Líneas declaradas como trabajo futuro

Las direcciones naturales de extensión del trabajo están explícitamente fuera del alcance académico y se documentan como fichas de feature en `FEATURE_BACKLOG_DETALLADO.md` bajo la categoría **Trabajos Futuros** (renombrado desde "MVP3" según DHU-012). Estas direcciones se mencionan en el capítulo de trabajo futuro del documento de tesis. No se redactan como Historias de Usuario ni se construyen dentro del alcance del proyecto académico.

| ID | Título | Decisión técnica relacionada |
|---|---|---|
| F21 | Reentrenamiento del modelo predictivo (pipeline MLOps) | — |
| F36 | Reconocimiento de tipos de vehículos para priorización | — |
| F37 | Coordinación de ondas verdes entre intersecciones vecinas (extensión a múltiples intersecciones interrelacionadas mediante arquitecturas espacio-temporales tipo STGNN) | D-006 |
| F38 | Procesamiento de datos reales de Waze (calibración del modelo con datos reales de tráfico de Lima vía acuerdo con la municipalidad) | D-008 |
| F39 | Despliegue real en infraestructura distribuida (Raspberry Pi como dispositivo de borde + servidor central) | D-004 |
| F40 | Notificaciones push y monitoreo proactivo de cámaras | — |
| F41 | Integración cerrada del módulo de visión al loop de validación cuantitativa (cuando se disponga de cámaras propias o fuentes de video controlables de Miraflores) | D-007 |
| F42 | Análisis del estado del tráfico asistido por LLM (modo explicativo del motor adaptativo + chat opcional, apoyado en la capa de explicación de HU-06) | D-007 / Art. 21 |

**Sobre F42 (análisis del tráfico asistido por LLM).** Dirección futura explorada y descartada del MVP. Durante el desarrollo se prototiparon dos componentes contra un LLM externo (Gemini): un análisis automático que narraba el estado de la intersección en lenguaje natural, y un asistente conversacional que respondía consultas del operador sobre el tráfico. Ambos se removieron del MVP (SAN-02) por quedar fuera de la arquitectura objetivo (Art. 21 de la constitución) y depender de una API externa. Se conserva como extensión natural: un componente que explique el estado y las decisiones del motor adaptativo en lenguaje natural —apoyándose en la capa de explicación ya prevista en HU-06—, con un modo conversacional opcional. Fuera de alcance del MVP por costo de integración, dependencia de proveedor externo y falta de aporte a la validación cuantitativa de la tesis.

Ver fichas detalladas en `FEATURE_BACKLOG_DETALLADO.md`. La asimetría entre F21 (ficha completa, originada en el Brainstorming del Inception) y F36-F41 (fichas livianas, formalizadas en DHU-012) es histórica y deliberada. F42 se incorpora retroactivamente en saneamiento (SAN-02, 2026-05-26) sin ficha en `FEATURE_BACKLOG_DETALLADO.md`: queda referenciada únicamente en este documento por ser dirección de tesis, no feature del Product Backlog.

---

## 9. Documentos relacionados

- `DECISIONS.md` — Registro formal de decisiones técnicas (D-001 a D-009).
- `DECISIONS_HU.md` — Decisiones metodológicas sobre la redacción del Product Backlog (DHU-001 a DHU-022).
- `LEAN_INCEPTION_CEREBROVIAL.md` — Inception consolidado del proyecto.
- `FEATURE_BACKLOG_DETALLADO.md` — Detalle completo de las 41 features del backlog (MVP1 + MVP2 + Trabajos Futuros).
- `documentation/docs/DISCOVERY_2026-05-10.md` — Auditoría completa del estado del repositorio.
