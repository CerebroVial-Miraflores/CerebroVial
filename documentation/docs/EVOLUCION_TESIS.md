# Evolución del Proyecto de Tesis — CerebroVial

> Documento de contexto y narrativa de evolución del proyecto. Captura las decisiones de alcance, los giros arquitectónicos y la consolidación final del trabajo. **Pensado como insumo para el capítulo de introducción/contexto de la tesis y para la sustentación oral.**

**Fecha de redacción:** 2026-05-11 (semana 6 de 15)
**Estado:** Versión consolidada tras cierre arquitectónico

---

## 1. Resumen ejecutivo

CerebroVial es un sistema integrado de control adaptativo de tráfico urbano que combina **predicción de congestión mediante GRU**, **observación de estado mediante visión computacional**, y **selección adaptativa de estrategia de control** (Webster, MaxPressure, MTC) según el estado predicho y observado. La validación cuantitativa se realiza mediante simulación SUMO comparando el sistema propuesto contra control fijo.

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

**Hipótesis:** El valor del sistema no está solo en predecir congestión, sino en **actuar sobre esa predicción**. Un motor que selecciona dinámicamente entre estrategias de control clásicas (Webster, MaxPressure, MTC) según el estado predicho y observado es el aporte central de ingeniería del trabajo.

**Lo que se construyó:**

- Implementación de tres estrategias de control en `core_management_api/src/control/`:
  - **Webster:** asignación de tiempos de verde proporcional a la demanda histórica.
  - **MaxPressure:** estrategia descentralizada basada en presión de colas.
  - **MTC (Multi-Threshold Control):** control por umbrales sobre el estado actual.
- **AdaptiveEngine:** selector que escoge la estrategia según el estado del sistema.
- Frontend completo de visualización del motor (`views/control/`, 9 archivos, 1034 líneas).
- Documentación teórica en `CONTROL.md` (552 líneas).
- Tests unitarios y de integración pasando.

**Reflexión sobre el orden de ejecución:** El motor adaptativo se construyó antes de cerrar Fase 2 del PLAN.md (autenticación, persistencia de visión). Esto rompe el orden formal del plan pero es defendible: el motor es el componente más visible y académicamente diferenciador del sistema; trabajarlo temprano permite estabilizarlo y testearlo con tiempo.

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
> *Paralelamente se desarrolló el componente central del sistema: un motor adaptativo que selecciona dinámicamente entre estrategias de control clásicas (Webster, MaxPressure, Multi-Threshold Control) según el estado predicho por el modelo y el estado observado por el módulo de visión. Este motor constituye el aporte de ingeniería principal del trabajo.*
>
> *El sistema final integra los tres componentes (visión, predicción GRU, motor adaptativo) y se valida cuantitativamente mediante simulación SUMO, comparando los KPIs de tráfico bajo el sistema propuesto contra los obtenidos con control fijo (Webster). Tanto el dataset de entrenamiento del modelo predictivo como los escenarios de validación se generan en SUMO con particiones independientes, asegurando consistencia metodológica. La calibración del modelo con datos reales de tráfico urbano y la extensión a redes de múltiples intersecciones interrelacionadas se declaran como las dos líneas naturales de trabajo futuro."*

---

## 7. Mapa de aportes que sobreviven (referencia rápida)

| Fase | Aporte | Rol en sistema final |
|---|---|---|
| Fase 1 (visión inicial) | Módulo de visión con YOLO | Sensor de estado / componente demostrable (D-007) |
| Fase 2 (exploración predicción) | RandomForestPredictor | Baseline de comparación en validación |
| Fase 2 (exploración predicción) | Revisión literatura STGNN | Fundamentación de trabajo futuro |
| Fase 3 (motor adaptativo) | Webster + MaxPressure + MTC + AdaptiveEngine | Componente central del sistema (aporte de ingeniería) |
| Fase 3 (motor adaptativo) | Frontend control (`views/control/`) | Visualización del estado del sistema |
| Fase 4 (cierre) | GRU univariado | Modelo predictivo final (D-006) |
| Fase 4 (cierre) | SUMO end-to-end | Columna vertebral de datos y validación (D-008) |

---

## 8. Líneas declaradas como trabajo futuro

Estas líneas están explícitamente fuera del alcance del trabajo pero son extensiones naturales y se mencionan en el capítulo de trabajo futuro:

1. **Extensión a múltiples intersecciones interrelacionadas** mediante arquitecturas espacio-temporales (STGNN, atención sobre vecinos). La exploración inicial de `time_then_space.py` durante el desarrollo sustenta esta línea.
2. **Calibración del modelo predictivo con datos reales de tráfico de Lima** (vía acuerdo con la municipalidad para acceso a datos de Waze for Cities u otras fuentes).
3. **Integración cerrada de visión computacional al loop de validación** (cuando se disponga de cámaras propias o fuentes de video controlables y de Miraflores).
4. **Despliegue real en infraestructura distribuida** (Raspberry Pi en el borde + servidor central), del cual se demuestra la viabilidad arquitectónica pero no la operación física.

---

## 9. Documentos relacionados

- `DECISIONS.md` — Registro formal de decisiones técnicas (D-001 a D-008).
- `documentation/docs/DISCOVERY_2026-05-10.md` — Auditoría completa del estado del repositorio.
- `tesis/(2).docx` — Documento de tesis (requiere actualización del capítulo de alcance para reflejar el cierre arquitectónico de Fase 4).
- `PLAN.md` — Plan de ejecución (requiere actualización del cronograma según semanas 6-15).
