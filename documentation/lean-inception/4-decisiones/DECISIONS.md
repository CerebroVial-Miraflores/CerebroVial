# DECISIONS — CerebroVial

> Registro de decisiones técnicas y de proyecto. Formato ADR ligero. Las decisiones cerradas afectan el código y la documentación; las pendientes (`D-PENDING-*`) son cuestiones abiertas que requieren resolución antes de avanzar a las fases que dependen de ellas.

## Índice

| ID | Estado | Fecha | Título |
|---|---|---|---|
| D-001 | Cerrada | 2026-04-30 | Arquitectura: monolito modular |
| D-002 | Cerrada | 2026-04-30 | Modelo predictivo: RNN |
| D-003 | Cerrada | 2026-04-30 | Deploy: Docker local |
| D-004 | Cerrada | 2026-04-30 | Pi física: demostración conceptual, no entrega |
| D-005 | Cerrada | 2026-04-30 | Números de tesis: actualizar tras validación real |
| D-006 | Cerrada | 2026-05-11 | Modelo predictivo: GRU univariado por intersección |
| D-007 | Cerrada | 2026-05-11 | Módulo de visión: componente demostrable, no en loop de validación |
| D-008 | Cerrada | 2026-05-11 | SUMO end-to-end: datos sintéticos para entrenamiento y validación |
| D-009 | Cerrada | 2026-05-13 | Variable de estado predicha: jam level (constructo Waze) |
| D-010 | Cerrada | 2026-05-31 | Revisión de SAN-01: torch CPU-only en el core para servir el GRU |
| D-011 | Cerrada | 2026-06-01 | Reapertura de D-006: predictor espacio-temporal sobre grafo (STGNN) acotado al corredor de la Av. Larco |
| D-012 | Cerrada | 2026-06-01 | Enmienda a D-011: escenario del track STGNN de corredor Larco a Miraflores completo |
| D-013 | Cerrada | 2026-06-01 | Target del track STGNN: meanTimeLoss/demora (enmienda a D-011, cierre de deuda diferida por D-012) |
| D-014 | Cerrada | 2026-06-03 | Criterio de drenaje net-específico para v2: gate multi-señal por-día (reemplaza la racha sub-8 km/h, v1-específica) |
| D-PENDING-001 | **Resuelta por D-006** | — | Modelo: reutilizar `time_then_space.py` o GRU desde cero |

---

## D-001 — Arquitectura: monolito modular
**Fecha:** 2026-04-30 · **Estado:** Cerrada

**Decisión:** El sistema se organiza como un **monolito modular**, no como microservicios. Las carpetas `core_management_api/`, `edge_device/`, `ia_prediction_service/` se entienden como módulos del mismo sistema. La base común se consolida en `shared/` instalable como paquete pip local.

**Justificación:** El refactor previo separó el código en carpetas que sugieren microservicios, pero (a) los `common/` de `core_management_api` y `edge_device` son byte-idénticos, (b) no existe API real entre ellos, (c) `ia_prediction_service` es pipeline ML offline, no servicio HTTP. Mantener tres servicios desplegables independientes agregaría complejidad sin valor en un proyecto de tesis con un equipo de dos.

**Impacto:** El docker-compose final tiene `db`, `core_management_api` (incluye prediction + control + vision-consumer), `edge_device` y `frontend_ui`. `ia_prediction_service` queda como herramienta de entrenamiento offline.

---

## D-002 — Modelo predictivo: RNN
**Fecha:** 2026-04-30 · **Estado:** Cerrada (refinada por D-006)

**Decisión:** El modelo predictivo del sistema es una **RNN** (alineado al documento de tesis). El `RandomForestPredictor` actual queda como fallback temporal con flag de configuración hasta que la RNN esté servida.

**Justificación:** El documento de tesis declara una arquitectura RNN. Mantener el RandomForest como fallback evita que una falla de carga del modelo neuronal rompa el endpoint de predicción.

**Impacto:** La implementación de la RNN es trabajo del equipo, materializada concretamente como GRU univariado según D-006.

**Nota:** D-006 refina esta decisión especificando GRU como la familia de RNN a utilizar, descartando arquitecturas espacio-temporales (STGNN) por estar fuera del alcance.

---

## D-003 — Deploy: Docker local
**Fecha:** 2026-04-30 · **Estado:** Cerrada

**Decisión:** El sistema se despliega localmente con `docker compose up`. No se usa Azure ni ningún cloud por ahora.

**Justificación:** El alcance de la tesis no incluye productivización. Los recursos disponibles (tiempo + cuenta cloud + presupuesto) no justifican el deploy en Azure. La "arquitectura desplegable en Pi/cloud" se demuestra arquitectónicamente y se documenta como plan de productivización en el README final.

**Impacto:** El README de quickstart asume `docker compose up` + `npm run dev`. La defensa final prueba el sistema en máquina limpia, no en cloud.

---

## D-004 — Pi física: demostración conceptual, no entrega
**Fecha:** 2026-04-30 · **Estado:** Cerrada · **Sujeta a confirmación con asesor**

**Decisión:** No se entrega una Raspberry Pi física en la defensa. Se demuestra que la arquitectura **es desplegable** en Pi (separación de `edge_device` con dependencias mínimas, contenerización separada, comunicación por SSE/HTTP) sin entregar el hardware.

**Justificación:** El proyecto se evalúa por la arquitectura predictiva y la integridad del sistema, no por hardware. La demostración conceptual cubre el espíritu del IoT del documento sin agregar riesgo de hardware roto en la defensa.

**Impacto:** El demo final corre todo en una laptop. El documento de tesis y el video explican qué módulos correrían en Pi (edge_device) y cuáles en servidor central (core_management_api + frontend + db).

**Pendiente:** Confirmar con asesor.

---

## D-005 — Números de tesis: actualizar tras validación real
**Fecha:** 2026-04-30 · **Estado:** Cerrada · **Sujeta a confirmación con asesor**

**Decisión:** Los números declarados en el documento de tesis (88.2% accuracy de detección, 81.3% accuracy del predictor, latencia <2s) se **actualizan a los valores reales** medidos durante la validación cuantitativa. Si la realidad es peor, se reporta la realidad.

**Justificación:** Integridad académica. Reportar números que no se pueden reproducir en el demo es riesgo alto en preguntas de defensa. La tesis se defiende mejor con honestidad sobre limitaciones que con marketing inflado.

**Impacto:** La actualización del documento de tesis con los números reales es un entregable explícito del proyecto. Si los números reales son peores, el README documenta limitaciones del demo (datos sintéticos, validación parcial, etc.).

**Pendiente:** Confirmar con asesor.

---

## D-006 — Modelo predictivo: GRU univariado por intersección
**Fecha:** 2026-05-11 · **Estado:** Cerrada · **Resuelve:** D-PENDING-001 · **Sujeta a confirmación con asesor**

**Decisión:** Se adopta **GRU univariado por intersección** como modelo predictivo de congestión. Se descarta `time_then_space.py` (RNN + DiffConv espacial / STGNN) y los checkpoints asociados (`epoch=79-step=30800.ckpt` y otros). La incorporación de dependencia espacial entre intersecciones (arquitecturas tipo STGNN, vecindad) se declara como **trabajo futuro**.

**Justificación:**

1. **Alcance de validación.** El sistema se valida sobre **una sola intersección** de Miraflores. Una arquitectura espacio-temporal requiere múltiples nodos interrelacionados; no aplica al problema definido.
2. **Cronograma realista.** Con 9 semanas hasta entrega final y dependencias pesadas (SUMO end-to-end e integración completa del sistema), no hay margen para definir grafos espaciales, debuggear pipelines tsl/PyTorch Lightning y entrenar STGNN.
3. **Aporte central de tesis.** La contribución es el **sistema integrado** (predicción + control adaptativo + visión + validación cuantitativa), no la sofisticación arquitectónica del predictor aislado.
4. **Defensa académica.** GRU univariado es estándar de la literatura para predicción de serie temporal de tráfico por sensor/intersección. Es justificable y reproducible.

**Impacto:**

- Se crea `ia_prediction_service/src/models/gru_model.py` (GRU desde cero, simple).
- `time_then_space.py` se mueve a `ia_prediction_service/src/models/legacy/` o se elimina (decisión en limpieza de repo).
- Checkpoints en `ia_prediction_service/notebooks/logs/` se archivan o eliminan.
- METR-LA deja de ser referencia de dataset; los datos vienen de SUMO (ver D-008).
- El capítulo de modelo predictivo de la tesis se reescribe para reflejar GRU univariado y declarar STGNN como trabajo futuro.

**Justificación para la tesis (texto sugerido):**

> *"Se selecciona GRU univariado por intersección dado que la validación del sistema se realiza sobre intersecciones tratadas independientemente. La incorporación de dependencia espacial entre intersecciones (mediante arquitecturas espacio-temporales tipo STGNN o atención sobre vecinos) se identifica como una extensión natural del trabajo y se declara como trabajo futuro, condicionada a validación a escala de red urbana."*

**Pendiente:** Confirmar con asesor.

---

## D-007 — Módulo de visión: componente demostrable, no en loop de validación
**Fecha:** 2026-05-11 · **Estado:** Cerrada · **Sujeta a confirmación con asesor**

**Decisión:** El módulo de visión computacional se implementa como **componente funcional demostrable** del sistema, con validación independiente mediante métricas estándar de detección (precisión, recall, mAP de YOLO sobre un dataset etiquetado representativo). **No participa en el loop de validación cuantitativa del sistema integrado**; en su lugar, SUMO provee directamente las métricas de estado (flujo, cola, densidad) que el módulo de visión proveería en producción.

El rol del módulo en la arquitectura del sistema es de **sensor de estado en tiempo real** que alimenta al motor adaptativo con observación del tráfico observado por cámara. La idea original de "ajuste fino del motor mediante visión" se descarta por requerir literatura adicional fuera del cronograma y por no haber input real confiable (los streams de YouTube usados hoy no son fiables a largo plazo).

**Justificación:**

1. **Input no controlable.** Los streams de YouTube actuales pueden apagarse, no son específicos de Miraflores y no proveen ground truth para validación cuantitativa. Depender de ellos para la validación final es un riesgo de demo evitable.
2. **Consistencia metodológica.** Si la validación se hace en SUMO, las métricas de estado deben venir de SUMO. Mezclar simulación con observación real introduce confusión sobre qué se está validando.
3. **Defensa académica.** Tener un módulo de visión con su propia validación acotada (métricas de detección sobre dataset etiquetado) es metodológicamente más limpio que un módulo cuya validación está acoplada al sistema completo.
4. **Alcance temporal.** "Cómo usar visión para ajuste fino de un motor de control adaptativo" es un tema de investigación completo; no cabe en 9 semanas.

**Impacto:**

- El módulo de visión (`edge_device/src/vision/`) se mantiene y se completa para demostración.
- Validación del módulo: dataset etiquetado pequeño (≥200 frames), métricas de detección reportadas.
- En el loop de validación cuantitativa del sistema integrado (comparación de KPIs con/sin sistema documentada en el capítulo de validación de la tesis), las métricas de estado las provee SUMO.
- El video de demo muestra el módulo de visión operando sobre un stream/video grabado, sin ser parte del experimento cuantitativo.
- El capítulo de validación de la tesis separa explícitamente "validación del módulo de visión" (métricas de detección) y "validación del sistema integrado" (KPIs SUMO).

**Justificación para la tesis (texto sugerido):**

> *"El módulo de visión computacional se implementa como sensor de estado en tiempo real del sistema. Su validación se realiza mediante métricas estándar de detección (precisión, recall, mAP) sobre un dataset etiquetado representativo. Para la validación cuantitativa del sistema integrado (predicción + control adaptativo), se utiliza simulación SUMO que provee directamente las métricas de estado que el módulo de visión proveería en operación. Esta separación asegura consistencia metodológica y aísla las fuentes de error del sistema integrado de las fuentes de error del módulo de detección."*

**Pendiente:** Confirmar con asesor.

**Encuadre operativo de la implementación:** ver DHU-024 en `DECISIONS_HU.md`
(2026-05-27). D-007 fija el rol de la visión (componente demostrable, no en loop
cuantitativo); DHU-024 fija el cómo de la implementación (alcance 11 CTs, arquitectura
DDD, contrato `vision_aggregates`, levantamiento de la regla CLAUDE.md) sin reversar
esta decisión.

---

## D-008 — SUMO end-to-end: datos sintéticos para entrenamiento y validación
**Fecha:** 2026-05-11 · **Estado:** Cerrada · **Sujeta a confirmación con asesor**

**Decisión:** SUMO (Simulation of Urban MObility) es la **columna vertebral del sistema de datos** del proyecto. Se utiliza para:

1. **Generación del dataset de entrenamiento** del modelo GRU (series temporales sintéticas de flujo/velocidad/ocupación por intersección, bajo distintos patrones de demanda).
2. **Validación cuantitativa del sistema integrado**: comparación "con sistema (GRU + motor adaptativo) vs sin sistema (Webster fijo)" mediante KPIs estándar (tiempo de viaje, longitud de cola, demoras).

Las particiones de entrenamiento y validación son **escenarios SUMO distintos** (distintos seeds, patrones de demanda, eventos) para evitar fuga de información. No se utilizan datos reales de Waze ni datasets públicos (PeMS, METR-LA) como fuente principal de entrenamiento. La incorporación de datos reales de tráfico de Lima (vía acuerdo con la municipalidad) se declara como **trabajo futuro** o como **bono académico** si se obtienen antes de la entrega.

**Justificación:**

1. **No hay acceso a datos reales hoy.** No se tiene API key de Waze ni acuerdo con la municipalidad. Depender de obtenerlo en 9 semanas es riesgo terminal.
2. **Consistencia metodológica fuerte.** Entrenar y validar en el mismo mundo simulado evita el problema de transferibilidad entre datasets distintos. La tesis declara explícitamente este alcance.
3. **Control experimental.** SUMO permite generar dataset ilimitado y controlable: días laborales, fines de semana, hora pico, valle, eventos. Calidad y variabilidad están bajo control del tesista.
4. **Eliminación de dependencias externas.** No hay riesgo de que un servicio público apague endpoints, que un acuerdo se caiga, que un dataset cambie.
5. **Defendible académicamente.** Múltiples tesis de control de tráfico operan en este modo. La limitación se declara explícitamente.

**Riesgo conocido:** El jurado puede objetar "se entrena y valida en el mismo simulador". **Respuesta:** se usan particiones distintas (escenarios, seeds), se declara el límite metodológico explícitamente en el capítulo de alcance, y la generalización a datos reales se identifica como trabajo futuro.

**Impacto:**

- La integración con SUMO (F32 del backlog, Bloque E del Sequencer) sube de "validación al final" a **columna vertebral del sistema**. Empieza en semana 6.
- Cronograma: 1-2 semanas para topología de Miraflores en SUMO + escenarios de demanda + generación de dataset.
- El modelo GRU se entrena sobre el dataset SUMO generado, no sobre METR-LA ni Waze.
- El capítulo de alcance de la tesis declara explícitamente la naturaleza simulada de la validación.
- Si el PO de la municipalidad provee datos antes de la entrega: se usan como **validación adicional** ("el modelo entrenado en simulación se evalúa también sobre datos reales de X periodo, mostrando degradación de Y%"), no como reemplazo del flujo principal.

**Justificación para la tesis (texto sugerido):**

> *"La validación del sistema propuesto se realiza mediante simulación SUMO calibrada con la topología de la intersección de estudio en Miraflores. Tanto el dataset de entrenamiento del modelo predictivo como los escenarios de validación cuantitativa se generan a partir de SUMO con particiones independientes (distintos seeds y patrones de demanda) para evitar fuga de información. La obtención de datos de tráfico reales de la ciudad de Lima se identifica como una limitación reconocida del trabajo, y la calibración del modelo con datos reales mediante acuerdo con entidades municipales se declara como trabajo futuro."*

**Pendiente:** Confirmar con asesor.

---

## D-009 — Variable de estado predicha: jam level (constructo Waze)
**Fecha:** 2026-05-13 · **Estado:** Cerrada

**Decisión:** Se adopta el constructo **"jam level"** de Waze (escala ordinal 0-5) como variable de estado del sistema y como objetivo del modelo predictivo. Complementa D-006 (GRU univariado por intersección) especificando la variable objetivo del modelo. No reemplaza decisión previa.

**Definición de niveles (según Waze):**

| Nivel | Significado | Umbral de ratio velocidad/free-flow |
|---|---|---|
| 0 | Flujo libre | ratio ≥ 80% |
| 1 | Bajo | 60% ≤ ratio < 80% |
| 2 | Medio | 40% ≤ ratio < 60% |
| 3 | Alto | 20% ≤ ratio < 40% |
| 4 | Muy alto | 0 < ratio < 20% |
| 5 | Vía cerrada | velocidad = 0 |

> **Nota 2026-05-31 — Alineación a la escala Waze.** Los cortes de jam_level se alinearon a las anclas oficiales de la escala de congestión de Waze (jam 1 = 80% de velocidad libre, jam 4 = 20%), reemplazando los cortes previos (90/70/50/30). Motivo: el modelo predictivo (TTH-09) consume jam_level real de Waze; el dataset de entrenamiento debe usar la misma escala que la fuente de producción para evitar un sesgo sistemático de sobre-reporte de congestión (~1 nivel). Referencia: Waze define jam level 0-5 donde 1 = bajo (80% free-flow) y 4 = muy alto (20% free-flow). Los conteos de cobertura publicados en handoffs de TTH-07 (p. ej. `sustained jam ≥3`) se calcularon con la escala previa y no se recalculan (sprint cerrado); ver nota de reenvío en `documentation/handoffs/tth-07/tth-07-fase2-handoff.md`.

donde `ratio = velocidad_promedio / velocidad_flujo_libre`.

**Anclajes documentados oficialmente:** Los anclajes nivel 1 ≈ 80% del free-flow y nivel 4 ≈ 20% del free-flow están documentados en el paper:

> Carvalho et al. (2022). *JamVis: exploration and visualization of traffic jams.* The European Physical Journal Special Topics. DOI: 10.1140/epjs/s11734-021-00424-2

El paper cita textualmente: *"a jam with level 4 (20% of free-flow speed), while the light orange line represents a jam with level 1 (80% of free-flow speed)"*.

A partir de la realineación del 2026-05-31 (ver nota arriba) las anclas oficiales se adoptan como **bordes de banda directos**: 80% es la frontera flujo-libre/jam 1 y 20% la frontera jam 3/jam 4. Los umbrales intermedios 60% y 40% (fronteras de los niveles 2 y 3) se obtienen por **interpolación lineal** de 20 puntos porcentuales entre las anclas. Es la interpretación natural y se declara explícitamente como tal. Los umbrales pueden ajustarse contra datos reales de Waze cuando se obtenga acceso vía el asesor de Miraflores; el principio del constructo no cambia.

**Mapeo SUMO → jam_level:**

```python
def sumo_to_jam_level(mean_speed_mps, max_speed_mps):
    """Mapea velocidad de SUMO a jam level (constructo Waze)."""
    if mean_speed_mps == 0:
        return 5
    ratio = mean_speed_mps / max_speed_mps
    if ratio >= 0.80: return 0
    if ratio >= 0.60: return 1
    if ratio >= 0.40: return 2
    if ratio >= 0.20: return 3
    return 4
```

- `mean_speed_mps` se obtiene de SUMO vía TraCI: `traci.edge.getLastStepMeanSpeed`.
- `max_speed_mps` se obtiene de `traci.lane.getMaxSpeed` o del archivo de red, como aproximación inicial de velocidad de flujo libre. Para mayor precisión, se puede sustituir por el percentil 85 o 95 del histórico de velocidades del segmento en condiciones de baja demanda (práctica estándar en ingeniería de tráfico).

**Mapeo Waze → jam_level:** Directo, viene en el feed CCP de Waze (campo `level` de cada `jam`). No requiere transformación.

**Justificación:**

1. **Intercambiabilidad de fuente de datos sin reentrenar el modelo:** el entorno de validación (SUMO), una fuente real (Waze) y, eventualmente, la visión computacional propia producen la misma variable de estado. Esto elimina dependencia arquitectónica entre el modelo predictivo y la fuente de datos.

2. **Constructo validado por industria:** Waze procesa cientos de millones de trayectorias diarias usando este algoritmo. La adopción de su variable no es arbitraria; es replicación de un estándar de facto del sector. En sustentación es defendible: "no inventamos un índice; replicamos uno validado por una de las plataformas de tráfico más usadas del mundo".

3. **Coherencia con el HCM:** el principio "velocidad relativa al flujo libre" es la base del Level of Service del *Highway Capacity Manual* para arterias urbanas (medida por velocidad de viaje y razón volumen/capacidad). La adopción del constructo Waze hereda este fundamento académico.

4. **Compatibilidad con D-006:** variable univariada por intersección/segmento. Una sola serie temporal por dirección. Cumple D-006 sin modificarla.

5. **Interpretabilidad operativa:** el Operador entiende inmediatamente "nivel 4 de 5", concepto familiar de su propio Waze cotidiano. No requiere entrenamiento técnico para interpretar la salida del modelo.

**Implicancias para el modelo:**

- Se entrena el modelo para predecir el **ratio continuo** velocidad/free-flow.
- La discretización al nivel discreto (0-5) ocurre solo en la capa de presentación, para preservar resolución del entrenamiento y métricas de evaluación (MAE/RMSE sobre el ratio).
- La variable es univariada por segmento, una serie temporal por dirección, cumpliendo D-006.

**Implicancias para las HUs:**

- **HU-02** (monitoreo en tiempo real, Bloque B) sigue mostrando flujo y cola observados (variables primarias, sin transformación). El Operador conserva la visión física del tráfico en tiempo real.
- **HU-03** (predicción, Bloque B) muestra jam level predicho (0-5). El umbral de "congestión" para resaltar visualmente se establece por defecto en nivel ≥ 3, configurable en la HU del Bloque D dedicada a configuración del motor (origen F20).

**Caveat para la sustentación:**

Waze **no publica oficialmente los umbrales exactos** de cada nivel. Lo que tenemos son:

1. La definición cualitativa de niveles (0 = flujo libre, 5 = vía cerrada).
2. Dos puntos de calibración del paper JamVis (nivel 1 ≈ 80%, nivel 4 ≈ 20%).
3. Confirmación de que el algoritmo es velocidad relativa al flujo libre (documentación oficial de Waze for Cities partners).

Los umbrales intermedios (60% para nivel 2, 40% para nivel 3) son **deducidos por interpolación lineal**, no oficiales. Esto se documenta honestamente: replicamos el principio del algoritmo, calibramos los anclajes documentados, e interpolamos linealmente los intermedios. Es replicación de un constructo público con los detalles documentados disponibles, no ingeniería inversa de un secreto comercial.

**Trabajo futuro asociado:**

Cuando se disponga de acceso a datos históricos de Waze sobre la intersección objetivo (vía acuerdo con la municipalidad de Miraflores), se debe:

1. Calibrar los umbrales intermedios (60%, 40%) contra los niveles observados en el feed real de Waze para esa intersección.
2. Validar que el modelo entrenado sobre SUMO mantiene precisión aceptable cuando se alimenta con Waze.
3. Documentar diferencias sistemáticas entre el constructo replicado y el oficial de Waze, si las hubiera.

El diseño actual permite esta extensión **sin cambios al modelo predictivo**, solo agregando un adaptador de fuente Waze que ya devuelve `jam_level` directamente del feed.

**Referencias:**

- Carvalho, C. et al. (2022). *JamVis: exploration and visualization of traffic jams.* The European Physical Journal Special Topics. DOI: 10.1140/epjs/s11734-021-00424-2
- Waze Data Feed specifications. Google Support (Waze for Cities partners). Disponible en: https://support.google.com/waze/partners/answer/13458165
- Transportation Research Board. *Highway Capacity Manual.* Para fundamentación del LOS basado en velocidad relativa al flujo libre.
- Afrin, T., & Yodo, N. (2020). Survey citado en: *Applications of deep learning in congestion detection, prediction and alleviation: A survey.* ScienceDirect (2021), respaldando la práctica de definir congestión a partir de una sola variable primaria.

**Documentos relacionados:**

- D-006 — GRU univariado por intersección (define el tipo de modelo; D-009 define la variable que predice).
- D-007 — Visión como componente demostrable, con validación independiente.
- D-008 — SUMO genera dataset de entrenamiento y escenarios de validación.
- `HU_BLOQUE_B.md`, HU-02 y HU-03 — primeras HUs operativas que consumen este constructo.
- `EVOLUCION_TESIS.md` — sección de trabajo futuro, donde se declara la integración con Waze.

---

## D-010 — Revisión de SAN-01: torch CPU-only en el core para servir el GRU
**Fecha:** 2026-05-31 · **Estado:** Cerrada · **Revisa:** SAN-01 / C7.5 (regla CLAUDE.md "no torch en core") · **Habilita:** TTH-09 Fase 3

**Decisión:** Se **revisa SAN-01**. Se admite **`torch` CPU-only** como dependencia de `core_management_api`, **exclusivamente para servir el predictor GRU de TTH-09** (inferencia in-process). La clase `GRUMultiOutput` se **vendoriza en el core** (copia de la definición `nn.Module`, no import desde `ia_prediction_service`). El núcleo carga los 4 `.pt` (state_dicts) al arranque y corre la inferencia síncrona dentro del proceso FastAPI. Esto es una **revisión deliberada y acotada** de la regla, no su borrado: la guardia anti-regresión sigue vigente para todo lo demás (ver "Alcance / límite").

**Contexto / por qué:** TTH-09 necesita servir el GRU desde el endpoint `POST /predictions/predict` (ver `documentation/contracts/prediction_contract.md`). Se evaluaron tres caminos:

1. **Opción A — microservicio HTTP** (`ia_prediction_service` como servicio FastAPI+torch, consumido por red desde el core). **Descartada:** rompe el monolito modular (D-001); no hay precedente en el repo de cliente HTTP inter-servicio ni de patrón de fallback por red (la comunicación entre módulos es in-process + BD compartida). Agregaría orquestación (servicio nuevo en docker-compose, health checks, timeouts, manejo de caída remota) sin valor para un proyecto de tesis con un equipo de dos.
2. **Opción B — torch CPU-only in-process en el core + clase vendorizada.** **Elegida.** Servir el GRU síncrono dentro del proceso es lo más simple y lo más coherente con el monolito modular (D-001): el core ya sirve el baseline RandomForest in-process por el mismo path; el GRU lo sustituye sin cambiar el modelo de despliegue.
3. **Opción C — exportar a ONNX** (servir el GRU vía `onnxruntime`, sin torch en el core). **Descartada:** agrega complejidad de export/validación de paridad ONNX↔PyTorch para **evitar un `torch` que, en su variante CPU-only, ya se acepta**. El costo que motivó la evasión (peso de torch) se neutraliza directamente (ver abajo), así que el rodeo ONNX no se justifica.

**Mitigación del costo que originó SAN-01:** el disparador real de SAN-01 / C7.5 fue el **peso de la imagen** — `torch` con CUDA agrega ~1.5 GB innecesarios al core (documentado en `documentation/docs/DISCOVERY_2026-05-10.md` §9.7). Ese costo **no aplica a la inferencia**: no se necesita GPU. Se neutraliza instalando **torch CPU-only** vía `--index-url https://download.pytorch.org/whl/cpu` (~200-300 MB instalado, vs ~2.5 GB de la variante CUDA). El modelo es trivial para CPU: `GRUMultiOutput` es 1 capa GRU `hidden=64` + 1 `Linear(64, 180)`, ~25k parámetros, forward sobre series de 30 pasos; 4 modelos (N/S/E/W) corren en microsegundos. El código de evaluación de `ia_prediction_service` ya carga con `map_location="cpu"`, confirmando que el camino CPU es el natural.

**Lo que SE preserva del principio de SAN-01:** la **separación de responsabilidades** que la regla codificaba se mantiene intacta:
- **Visión** sigue en `edge_device` (YOLO/ultralytics fuera del core).
- **Entrenamiento del GRU** sigue **off-line en `ia_prediction_service`** — el core **no entrena**, solo carga state_dicts y sirve inferencia.
- El **RandomForest baseline se preserva como respaldo Nivel 2** (CT-09.8): una caída del GRU degrada, no tumba, la predicción.
- **`torch` NO entra a `cerebrovial_shared`.** Queda contenido como dependencia del core; el paquete transversal no se acopla a torch (la clase vendorizada vive en el core, no en `shared`).

**Alcance / límite:** esta revisión habilita `torch` en el core **SOLO para inferencia del GRU**. **No** habilita: entrenamiento en el core, `torch` en `cerebrovial_shared`, `torch` con CUDA, ni `ultralytics`/otras dependencias pesadas. La guardia anti-regresión de CLAUDE.md **se mantiene para todo lo demás**: cualquier HU futura que quiera meter otra dependencia pesada (o torch para algo distinto de servir el GRU) debe revisarse igual que antes. D-010 es una excepción nominada y trazable, no un levantamiento general de la regla.

**Relación con precedentes existentes:**
- **Precedente que D-010 ACOTA (no reescribe):** el handoff `documentation/handoffs/tth-07/tth-07-fase0-handoff.md` usó SAN-01 como precedente del **principio general "dependencia pesada ⇒ módulo fuera del core"**, para mantener `simulation/` (SUMO) fuera del núcleo. **Ese principio general SIGUE VIGENTE.** D-010 revisa SAN-01 **únicamente** para el caso torch-CPU-only-inferencia-del-GRU; **no** abre la puerta a meter otras dependencias pesadas en el core, y en particular **no** afecta la decisión de mantener `simulation/`/SUMO fuera del núcleo. El handoff de tth-07 es histórico y **no se edita**; D-010 solo acota explícitamente el alcance de aquel precedente.
- **Precedente del mecanismo CPU-only (refuerza, no inventa):** el patrón `--index-url https://download.pytorch.org/whl/cpu` (~200 MB vs ~2 GB CUDA) **ya está documentado en el repo** para `edge_device`: `DECISIONS_HU.md` (DHU-024 §7) y `documentation/docs/TODO.md` (C7.6, reabierta como F9.z). El método CPU-only que adopta D-010 **no es nuevo** — es un patrón ya reconocido en el proyecto, lo que fortalece la viabilidad de la decisión.
- **SAN-01 sigue "resuelta", no se des-cierra:** `documentation/sdd/SPECKIT_MAPPING.md` marca SAN-01 como resuelto (purga de torch muerto, 2026-05-26). D-010 **no reabre** esa deuda: aquella purga eliminó torch como dependencia *muerta* (código STGNN residual); D-010 reintroduce torch como dependencia *viva y justificada* (servir el GRU). Son eventos distintos sobre la misma regla.

**Impacto:**
- `core_management_api/requirements.txt` sumará `torch` (variante CPU-only, vía `--index-url`/`--extra-index-url`) al implementar TTH-09 Fase 3. Sería la primera dep del core con directiva de índice.
- La clase `GRUMultiOutput` se vendoriza en el core; debe ser **byte-compatible** con el state_dict guardado por `ia_prediction_service` (mismo `nn.Module`), verificada por un **test de paridad** (Fase 3a) — ver `prediction_contract.md` §4.
- CLAUDE.md (regla "no torch en core") y `documentation/ESTADO_Y_PROXIMOS_PASOS.md` (afirmación de guardia anti-regresión) se actualizan para apuntar a D-010 como la excepción registrada.
- CI (ubuntu) y el Docker del core (`python:3.11-slim`, Debian glibc) admiten el wheel CPU-only de torch; el entorno Windows del equipo también tiene wheel CPU-only. Mecanismo de llegada de los `.pt` al contenedor: a resolver al implementar TTH-09 Fase 3 (los `.pt` hoy están gitignored; pesan ~101 KB c/u).

**Referencias:**
- **SAN-01 / C7.5:** CLAUDE.md §"Deuda técnica a respetar"; `documentation/docs/TODO.md` C7.5; `documentation/ESTADO_Y_PROXIMOS_PASOS.md`; `documentation/docs/DISCOVERY_2026-05-10.md` §9.7 (peso de torch en el core).
- **Contrato de predicción:** `documentation/contracts/prediction_contract.md` §4 (mecanismo de carga del GRU) y §7 (RF como respaldo Nivel 2).
- **Decisiones relacionadas:** D-001 (monolito modular), D-006 (GRU univariado por dirección).
- **Precedente del patrón CPU-only:** `DECISIONS_HU.md` DHU-024 §7; `documentation/docs/TODO.md` C7.6/F9.z.
- **Precedente acotado:** `documentation/handoffs/tth-07/tth-07-fase0-handoff.md` (principio "dependencia pesada fuera del core", sigue vigente).
- **TTH-09 Fase 3:** implementación del serving del GRU.

---

## D-011 — Reapertura de D-006: predictor espacio-temporal sobre grafo (STGNN) acotado al corredor de la Av. Larco
**Fecha:** 2026-06-01 · **Estado:** Cerrada · **Revisa:** D-006 · **Notifica:** asesor (Paucar) · **Track:** investigación paralela, no toca producción

**Decisión:** Se reabre D-006 con fundamento empírico. Se autoriza un track de investigación paralelo que migra el predictor de 4 GRUs univariados independientes a un modelo espacio-temporal sobre grafo (STGNN, arquitectura Time-then-Space), **acotado al corredor de la Av. Larco** (3 cruces semaforizados encadenados S→N: Diez Canseco, Schell, Benavides). El track es de investigación: **no modifica el predictor de producción, el core, ni D-010** hasta que una decisión de integración posterior (Fase 5) lo justifique con números. Ambas salidas del track —integrar la STGNN o conservar el GRU— son resultados válidos.

**Qué cambió desde D-006 (por qué se reabre):**

1. **Límite empírico del univariado.** Los 4 GRUs univariados (`gru_N/S/E/W.pt`, TTH-09/PR #38) predicen `jam_level` 0–5 por dirección sin acoplamiento espacial. Las direcciones N/S/E/W son un artefacto del dataset sintético (`miraflores_4way`), no aristas reales. El modelo no captura la propagación de congestión entre tramos, que es la dinámica que el dominio requiere.
2. **El corredor ya es simulable.** Existe `simulation/conf/corredor_larco/corredor_larco.net.xml` (red SUMO real OSM/UTM, 3 cruces encadenados), ya usada para validar Max Pressure (IE05). La premisa #1 de D-006 ("se valida sobre una sola intersección, una arquitectura espacio-temporal no aplica") ya no se sostiene: hay una red multi-nodo real y simulable.
3. **Abstracción de dominio correcta.** La predicción de `jam_level` por arista dirigida (`source→target`, escala 0–5, alineada con cómo Waze modela congestión por segmento) corresponde a `graph_edges`, que ya existe. El univariado por dirección no es esa abstracción.
4. **Riesgo acotado y reuso disponible.** `time_then_space.py` (encoder lineal + RNN/GRU temporal + DiffConv espacial + MLPDecoder) sigue en el repo, funcional como referencia. El GRU actual es reusable como bloque temporal de la STGNN (la arquitectura Time-then-Space es agnóstica a qué representa la serie). Candidata técnica: `tsl` (Torch Spatiotemporal, sobre PyTorch Geometric).

**Condición de éxito (explícita, medible):** La STGNN se adopta **solo si supera al baseline GRU univariado reentrenado sobre el dataset del corredor, en el mismo split de evaluación**, por un margen que justifique su complejidad e integración. Si no lo supera, se conserva el GRU y se documenta el hallazgo. No se asume mejora; se mide.

**Alcance — qué NO toca este track:**

- **No modifica D-010.** `torch` CPU-only permanece en el core justificado exactamente como D-010 lo dejó: para servir el GRU univariado de producción de TTH-09. La STGNN vive **fuera del core**, en el track de investigación. D-010 no se reabre ni se debilita.
- **No toca `core_management_api`, el predictor de producción (`gru_multioutput.py` vendorizado), ni el endpoint `POST /predictions/predict`.**
- La extensión a la red completa de Miraflores (`miraflores.net.xml`, 47 semáforos) queda como trabajo futuro, fuera de este track.

**Riesgos y limitaciones documentados (no resueltos):**

- Red pequeña (3 nodos): no está garantizado que la STGNN supere al GRU. De ahí la condición de éxito medida contra baseline.
- Datos sintéticos sin calibración real: riesgo de sobreajuste, a documentar como limitación de la tesis.
- `tsl`/PyTorch Lightning es la zona de mayor riesgo de tiempo (ya señalado al cerrar D-006).

**Nota de estado sobre la ejecución de D-006 (hallazgos de auditoría 2026-06-01, no resueltos aquí):** El "Impacto" de D-006 quedó parcialmente sin ejecutar: `time_then_space.py` nunca se movió a `legacy/` (el directorio no existe), el checkpoint `epoch=79-step=30800.ckpt` sigue en `notebooks/logs/`, y el artefacto nominal `gru_model.py` nunca se creó con ese nombre (el real es `gru_multioutput.py`). Además, `TAREAS_TECNICAS_HABILITADORAS.md:580` y `HU_BLOQUE_E.md:105` afirman como hecho un movimiento a `legacy/` que no ocurrió. Estas inconsistencias se corrigen como deuda de saneamiento aparte (SAN nuevo a registrar), **no en esta enmienda**. D-011 las deja registradas para no heredarlas como verdad falsa.

**Plan del track (cada fase con stage gate y aprobación explícita):** Fase 1 constructor de grafo desde `corredor_larco.net.xml` + `graph_edges` → `edge_index`; Fase 2 dataset por-arista (tiempo×nodos×canales, escala 0–5); Fase 3 baseline GRU reentrenado sobre el dataset del corredor; Fase 4 STGNN Time-then-Space, métricas en el mismo split; Fase 5 decisión de integración.

> **Enmienda 2026-06-01 (D-013) — target.** El target "escala 0–5" que esta línea fija para Fase 2 y baseline en adelante quedó **enmendado por D-013**: el target del track pasa a demora continua (`meanTimeLoss`); la escala 0–5 queda como capa de presentación derivada de la demora predicha, no como target de entrenamiento. El texto histórico de D-011 se conserva sin cambios; D-013 lo corrige desde afuera.

**Pendiente:** Notificar a Paucar (asesor actual) la reapertura de D-006 — notificación, no solicitud de permiso. La condición original "Sujeta a confirmación con asesor" de D-006 quedó en suspenso bajo un asesor anterior; D-011 la cierra como track autorizado bajo el asesor actual.

---

## D-012 — Enmienda a D-011: escenario del track STGNN de corredor Larco a Miraflores completo
**Fecha:** 2026-06-01 · **Estado:** Cerrada · **Revisa:** D-011 · **Notifica:** asesor (Paucar) · **Track:** investigación paralela, no toca producción

**Decisión:** Se enmienda el **escenario** del track STGNN abierto por D-011: deja de ser el corredor de la Av. Larco (3 cruces encadenados S→N: Diez Canseco, Schell, Benavides) y pasa a ser **Miraflores completo** (`miraflores.net.xml`, 47 cruces semaforizados, ~590 edges vehiculares). La estrategia es **simular la red completa de Miraflores una sola vez** —sin recortarla, para no degradar los `tlLogic` ni perder el tráfico de contexto que entra y sale del subgrafo— y **elegir el subgrafo de análisis de la STGNN sobre los datos a posteriori**, no antes de simular. Todo lo demás que fijó D-011 (track de investigación paralelo; no toca producción ni D-010; ambas salidas —integrar STGNN o conservar GRU— válidas; decisión de integración diferida a la fase final con números) se mantiene sin cambios.

> **Nota 2026-06-01 — Corrección de conteo de edges.** El conteo "~590 edges vehiculares" de la Decisión es incorrecto. Auditoría de la red (`miraflores.net.xml`) establece **381 edges vehiculares** (lanes que permiten `passenger`); la cifra de ~1044 edges "con nombre" incluía 663 edges peatonales/ciclovía. El escenario y la decisión no cambian; solo se corrige la magnitud. Fuente: auditoría read-only de `simulation/conf/network/miraflores.net.xml`, 2026-06-01.

**Qué cambió desde D-011 (por qué se enmienda):**

1. **El corredor Larco no tiene señal espacial empírica explotable.** El corredor es una cadena de 3 cruces; el co-movimiento observado entre ellos proviene de **demanda compartida**, no de una estructura espacial que una STGNN pueda explotar como información de vecindad. Una corrida de prueba de 24 h lo confirmó: no hay señal espacial empírica que justifique el modelo sobre 3 nodos encadenados. La premisa de D-011 ("el corredor ya es simulable, multi-nodo real") era cierta pero insuficiente: simulable ≠ portador de señal espacial aprovechable.
2. **Miraflores completo da topología real con vecindad de grado >1.** 47 cruces y ~590 edges vehiculares ofrecen un grafo con suficiente conectividad para que el contexto de vecinos pueda, en principio, mejorar la predicción por arista. La elección del subgrafo concreto de análisis se hace **después** de ver los datos simulados, no por diseño previo.
3. **Simular completo y recortar después, no recortar antes.** Recortar la red antes de simular degradaría los `tlLogic` de los cruces de borde y eliminaría el tráfico de contexto (rutas que atraviesan el subgrafo). Se simula Miraflores entero una vez y el subgrafo de la STGNN se selecciona sobre el dataset resultante.

Esta enmienda **deja sin efecto** la frase de D-011 en su sección "Alcance — qué NO toca": *"La extensión a la red completa de Miraflores (`miraflores.net.xml`, 47 semáforos) queda como trabajo futuro, fuera de este track."* Lo que D-011 declaró trabajo futuro **es ahora el escenario activo y único** del track. El corredor Larco se descarta como escenario del STGNN (su uso como red de validación de Max Pressure / IE05 no se ve afectado).

**Condición de éxito (actualizada):** Se mantiene el criterio de D-011 —la STGNN se adopta **solo si supera al baseline GRU univariado reentrenado sobre el mismo split de evaluación**, por un margen que justifique su complejidad e integración; si no lo supera, se conserva el GRU y se documenta el hallazgo— **ahora medido sobre el dataset de Miraflores completo y el subgrafo de análisis elegido a posteriori**, no sobre el corredor Larco. El criterio que se evalúa es **correlación espacial** —que el contexto de los vecinos mejore la predicción por arista, al estilo de los benchmarks METR-LA / PEMS-BAY— **no** propagación física de congestión con lag temporal medible entre tramos. No se asume mejora; se mide.

**Alcance — qué NO toca esta enmienda:**

- **No modifica D-010.** `torch` CPU-only permanece en el core justificado exactamente como D-010 lo dejó (servir el GRU univariado de producción de TTH-09). La STGNN sigue **fuera del core**, en el track de investigación.
- **No toca el predictor de producción ni el core:** ni `core_management_api`, ni la clase vendorizada `gru_multioutput.py`, ni el endpoint `POST /predictions/predict`.
- **No resuelve el cambio de target de predicción.** El reemplazo de `jam_level`/ratio por demora/`meanTimeLoss` **NO se decide aquí**; queda **diferido a la Fase 2** del track (generación de dataset) como enmienda de fondo a evaluar contra D-009. Esta entrada cambia **solo el escenario**, no la variable objetivo.
- **No reabre ni reescribe D-011 in-place:** D-011 queda intacta; esta entrada la revisa desde afuera.

**Riesgos documentados (no resueltos):**

- Aun con 47 nodos, no está garantizado que exista señal espacial aprovechable en datos sintéticos sin calibración real; de ahí la condición de éxito medida contra baseline.
- La simulación de Miraflores completo es más pesada (590 edges, 47 `tlLogic`) que el corredor: mayor costo de cómputo y de almacenamiento del dataset.
- La selección de subgrafo a posteriori introduce un grado de libertad metodológico: el criterio de selección debe documentarse para no inducir cherry-picking del subgrafo más favorable.
- `tsl`/PyTorch Lightning sigue siendo la zona de mayor riesgo de tiempo (heredado de D-006/D-011).

**Trabajo futuro / deuda:**

- **(a) Artefactos de Fase 1 atados a escenario obsoleto.** Los 4 archivos `corridor_*` y el handoff de cierre de Fase 1 quedan referidos al escenario Larco, ahora descartado; deberán migrarse o reemplazarse al reconstruir Fase 1 sobre Miraflores (**deuda de Fase 1 → Fase 1.5**).
- **(b) Cambio de target diferido.** El reemplazo `jam_level` → demora/`meanTimeLoss` queda **pendiente de decisión formal en Fase 2** (generación de dataset), a evaluar contra D-009 (que fija jam_level como variable de estado). No se resuelve en esta enmienda. **→ CERRADA por D-013 (2026-06-01):** la decisión formal se tomó — el target del track pasa a demora/`meanTimeLoss` continua; D-009 permanece vigente y sin enmienda para producción (D-013 es excepción acotada del track). Ver § D-013.
- **(c) Decisiones ausentes de la Constitución.** D-010, D-011 y D-012 siguen sin reflejarse en `.specify/memory/constitution.md` (deuda preexistente; **no se resuelve aquí**).

**Pendiente:** Notificar a Paucar (asesor) el cambio de escenario del track —notificación, no solicitud de permiso, en línea con D-011.

---

### Anexo — Cierre del scope del grafo STGNN: decisión a posteriori sobre los datos (2026-06-01)

D-012 difirió la elección del subgrafo de la STGNN a "una decisión a posteriori sobre los datos, no por diseño previo". Esta nota cierra esa decisión con la evidencia de dos auditorías exploratorias read-only sobre el dataset `miraflores_laborable_60d/` y el grafo edge-as-node de Miraflores.

**Decisión:** el grafo de modelado es la componente conexa principal de la red vehicular — **375 nodos (de 381)**. Recorte por conectividad topológica, no por densidad de señal.

**Evidencia que la fundamenta:**

- **Señal espacial confirmada.** Correlación de congestión entre pares de edges, contrastando vecinos-en-el-grafo contra no-vecinos (control). Sobre `speedRelative` (variable primaria, neutral respecto al target diferido): mediana de correlación vecinos-1-salto = 0.46 vs no-vecinos = 0.01 (contraste +0.46). Decaimiento monótono con la distancia de grafo: 1-salto 0.46 → 2-saltos 0.17 → no-vecinos 0.01. Replicado sobre `density` (sin NaN, n comparable entre grupos): contraste +0.38, mismo decaimiento. La concordancia de las dos variables descarta que el resultado sea artefacto del n desigual entre grupos. Conclusión: existe señal espacial local, decreciente con la distancia de grafo — la firma que D-011 fijó como criterio (correlación espacial estilo METR-LA/PEMS-BAY, no propagación física con lag). El track tiene fundamento empírico para un STGNN.
- **Señal distribuida, no concentrada.** Los pares vecinos de alta correlación (>0.5) involucran 269 de 381 edges, sin hubs (reparto plano ~1.5 pares/edge). La señal espacial vive en casi toda la red, no en una zona. Consecuencia: un subgrafo por densidad de señal —recortar a los ~42 edges de alto tráfico— descartaría la mayor parte de la estructura espacial. Esa opción se refuta: sacrificaría el fenómeno que el track quiere medir a cambio de comodidad de datos.
- **Recorte por topología, no por señal.** El grafo completo tiene 2 componentes conexas: una principal de 375 nodos y una islita de 6 edges interconectados (grados 1-3) desconectada del cuerpo vía `<connection>`. Esos 6 edges son además de muy bajo tráfico (todos vacíos en ≥1 de los 8 días auditados). Como están topológicamente desconectados, no comparten señal espacial con el cuerpo principal por construcción y un STGNN no puede propagar contexto hacia/desde ellos. Se excluyen. Este recorte NO contradice la decisión de "no recortar por señal": saca 6 nodos que por definición topológica no tienen señal espacial compartida, no edges de baja densidad dentro de la componente conexa.
- **Lo que se conserva dentro del grafo de 375:** los edges de bajo tráfico o sin señal que estén conectados a la componente principal — incluidos 2 edges estructuralmente vacíos (sin tráfico en los 8 días: `111898821`, `438009517`) que son vecinos topológicos válidos y podrían activarse bajo perfiles de demanda aún no calibrados (finde/feriado/especial). Su vacío se trata como deuda de modelado, no como criterio de recorte.
- **Deuda registrada para Fase 2/3 (esparsidad):** a scale 0.20 la red es muy esparsa — `speedRelative` es NaN (edge vacío, sin vehículos que promediar) en ~82% de las celdas; la mediana de edge está vacía el 91% del día. El NaN es estructural (estado vacío), no dato faltante. El discriminador robusto de vacío es la regla de 3 estados (`density==0 AND speed.isna()`), NO `density==0` sola (esta última invierte el ~0.3-0.6% de celdas que son atascos-parpadeo). El tratamiento del estado vacío en el dataset por-arista y en el loss del modelo queda como decisión de diseño de Fase 2/3 — no se resuelve aquí.
- **Sin relación con el target diferido:** toda la medición se hizo sobre `speedRelative`/`density`, variables neutrales. El cambio de target `jam_level`→`meanTimeLoss` sigue diferido a Fase 2 por D-012; esta nota no lo toca. **→ Cerrado por D-013 (2026-06-01):** el cambio de target quedó decidido — target del track = demora continua (`meanTimeLoss`). Ver § D-013.

**Artefactos:** el grafo de 375 (componente principal) es el canónico de modelado. El grafo completo de 381 se conserva versionado como evidencia del análisis de componentes. Ambos mappings JSON versionados como insumo cross-sesión. Respaldo crudo de los Bloques 0/1 en `documentation/handoffs/stgnn-fase1/REPORTE_CRUDO_BLOQUES_0_1.md`.

---

## D-013 — Target del track STGNN: meanTimeLoss/demora (enmienda a D-011, cierre de deuda diferida por D-012)
**Estado:** Cerrada · 2026-06-01

**Contexto.** D-011 abrió el track STGNN especificando el target como jam_level escala 0–5 (heredado de D-009, derivado del ratio de velocidad). D-012 y su anexo difirieron explícitamente a la Fase 2 la decisión de reemplazar ese target por demora/meanTimeLoss, dejándola como deuda abierta (deuda (b) de D-012; anexo de cierre de Fase 1). Esta decisión cierra esa deuda.

**Decisión.** El target del track STGNN pasa de jam_level (ratio de velocidad) a meanTimeLoss/demora continua. El modelo entrena sobre la demora continua (regresión); la discretización a escala 0–5 estilo Waze queda como capa de presentación —derivada de la demora predicha—, no como target de entrenamiento.

**Fundamento (empírico).** El jam_level derivado de velocidad-media está contaminado por el estado del semáforo: un edge con el semáforo en rojo registra velocidad baja que aparenta congestión alta (jam 4–5) cuando en realidad es el ciclo normal del semáforo, no congestión. La demora (meanTimeLoss) no tiene ese sesgo. Dos verificaciones lo respaldan: (1) la verificación de variables candidatas (cola / demora / velocidad) sobre corrida existente del corredor, que mostró que la demora exhibe estructura espacial limpia y la velocidad no; (2) el bloque exploratorio de Fase 2 sobre los 375 nodos de Miraflores, que reconfirmó señal espacial monótona decreciente en timeLoss (contraste vecinos-1-salto vs no-vecinos +0.14 sobre tráfico real, decaimiento v1>v2>no-vec) y mostró que el timeLoss==0 está 99.8% ocupado por celdas vacías (sin vehículos), confirmando que la señal de demora vive en las celdas con tráfico, no en el cero.

**Alcance — aplica SOLO al track STGNN** (investigación, D-011). D-009 sigue vigente y sin enmienda para el sistema de producción: el predictor GRU de producción (TTH-09) y sus consumidores siguen usando jam_level por ratio de velocidad. Esta decisión NO toca producción, ni el GRU de TTH-09, ni los archivos que consumen jam_level. Es una excepción acotada del track de investigación, justificada porque el track mide demora (lo que el sistema realmente quiere reducir) sin el ruido del semáforo.

**Tratamiento del estado vacío.** Las celdas vacías (regla de 3 estados: density==0 AND speed.isna(), que reproduce exacto el predicado de generación sampledSeconds==0) NO se mapean a demora 0 —eso confundiría "sin tráfico" con "sin congestión"—. Se tratan como ausencia de señal (NaN + máscara de validez), y el modelo las ignora vía máscara. Decisión de modelado heredada a Fase 2/3.

**Baseline de comparación (Fase 3).** El baseline contra el que se evalúa el STGNN en Fase 5 es una GRU univariada nueva, entrenada sobre demora continua sobre el mismo dataset y split que el STGNN — NO el GRU de producción de TTH-09 (que predice otro target sobre otro dataset). La comparación STGNN-vs-baseline aísla la contribución de la componente espacial: ambos comparten target (demora), features y split; el STGNN agrega solo la vecindad del grafo.

**Enmienda.** Corrige D-011 (donde especificaba "escala 0–5" para Fase 2 y baseline) y cierra la deuda (b) de D-012 y el anexo de cierre de Fase 1 (que dejaban el cambio diferido). El plan de fases del track se lee ahora con target = demora continua de Fase 2 en adelante.

**Deuda registrada.** (i) La calibración del dataset (scale=0.20) se ancló pensando en jam_level-velocidad; reverificar que sigue siendo apropiada para demora como target. (ii) Los cortes de discretización 0–5 para demora (capa de presentación) no existen aún; se calibrarán contra datos de Waze reales cuando estén disponibles, no se inventan desde SUMO. (iii) Enriquecimiento de features de entrada (density, flow, speedRelative además de la propia serie de demora) diferido como posible mejora del modelo tras la comparación baseline-vs-STGNN.

**Enmienda 2026-06-01 — corrección del target: timeLoss total, no meanTimeLoss por-vehículo**

D-013 fijó el target del track en "meanTimeLoss/demora". Una verificación read-only sobre el dataset (los 375 nodos del LCC, day_seed042 + cross-check contra el edgeData crudo) mostró que la columna timeLoss del Parquet es el timeLoss TOTAL agregado por arista por intervalo (sumado sobre vehículos), no el promedio por vehículo, y que derivar el promedio por vehículo es inviable e indeseable:

**El promedio por-vehículo está indefinido en el régimen de máxima congestión.** En celdas de atasco (cola estática, entered≈0, nadie completa la traversía), timeLoss_total / n_veh diverge — valores de hasta 3.240.000 s por división por casi-cero. El conteo exacto (entered) también es 0 en esas celdas, así que el problema es intrínseco al per-vehicle, no un artefacto del denominador estimado. El target debe ser máximo en el atasco, no indefinido.

**El total es estable y preserva el orden.** Es finito y monótono en toda la cola (máx 1253 s). Su correlación de Spearman con el promedio por-vehículo es 0.94, y el volumen es bajo en el grueso de las celdas (mediana 1 vehículo/bin, 86% ≤2 vehículos), así que el promedio y el total casi coinciden donde hay poco tráfico y el total se comporta mejor donde hay atasco.

**Regenerar no es opción gratis ni resuelve el problema.** El edgeData crudo se descartó para 59 de los 60 días (solo sobrevive una muestra), así que obtener el promedio "exacto" exigiría re-correr las 60 simulaciones SUMO — y aun así el entered=0 en atascos reproduciría la misma indefinición.

**Corrección.** El target del track es timeLoss total por arista por intervalo de 60 s (la columna timeLoss del Parquet, ya disponible). Sin regeneración, sin recompactado, sin derivación per-vehículo. Todo lo demás de D-013 se mantiene: el target sigue siendo demora (no velocidad/jam_level, evitando la contaminación del semáforo que motivó D-013), continuo, con discretización 0–5 solo en presentación, alcance solo-track, producción intacta.

**Deuda registrada.** El timeLoss total mezcla intensidad de congestión con volumen de tráfico (confounding acotado: Spearman 0.94 con el per-vehículo indica reordenamiento moderado, concentrado en celdas de alto volumen). Es un sesgo medible y documentado, preferible a la indefinición catastrófica del per-vehículo en el atasco. Si una fase futura quisiera un target per-vehículo, requeriría regeneración + una política de regularización para las colas estáticas — no resuelto aquí.

---

## D-014 — Criterio de drenaje net-específico para v2: gate multi-señal por-día (reemplaza la racha sub-8 km/h, v1-específica)
**Estado:** Cerrada · 2026-06-03 · **Revisa:** criterio de drenaje de `analyze24.py` (C2, v1-específico) · **Habilita:** gate de aceptación por-día de B3.2 · **Cierra:** deuda de método de B2 (C3, candidato a ADR)

**Contexto.** El veredicto binario de drenaje de `analyze24.py` —"racha ≥3h con velocidad media < 8 km/h"— está calibrado al **colapso duro del net v1**, donde la velocidad se clavaba sostenida en 3–4 km/h. El net v2 (distrito completo, 1664 edges ≈ 4.4×, LCC 1660 nodos) **colapsa "suave"**: en su peor punto medido (scale 1.5) la velocidad media ponderada por viaje toca **~11 km/h**, nunca 3–4. Consecuencia: el criterio sub-8 marca "drena" **incluso en scales que colapsan en v2** (a 1.5: 920 teleports y +43% de duración de viaje, y el binario igual dice "drena"). El criterio v1 no traslada a v2.

**Evidencia (C3, `simulation/data/datasets/miraflores_laborable_60d/calibracion/SWEEP_C3_RESULTS.md`).** Barrido de 10 scales sobre v2, seed 42, 24h continua, control fijo. La rodilla del colapso es **gradual** y el cliff cae **entre 1.2 y 1.3**:

| señal | 1.0 | 1.1 (operación) | 1.2 | 1.3 (onset colapso) | 1.5 (colapso) |
|---|--:|--:|--:|--:|--:|
| teleports | 0 | 11 | 36 | 137 | 920 |
| Δdur vs 1.0 | — | +2% | +5.5% | +15% | +43% |
| dip (km/h mín) | →25.2 | →23.7 incipiente | →21.3 recupera | sub-20 ancho | →11.2 |

**Decisión.** Para v2 el drenaje se evalúa con un **criterio multi-señal por-día**, no con la racha sub-8. Un día de simulación v2 **drena si y solo si se cumplen las tres**:

1. **Teleports ≤ 50** — *señal primaria* (la más limpia: sale de `stats.xml`, no contaminada por stops de semáforo en tramos cortos). Referencia: 1.1 = 11 teleports; onset de colapso 1.3 = 137.
2. **Δduración media de viaje ≤ +10 %** sobre el baseline ~254 s (es decir, **≤ ~280 s**), de `tripinfo.xml`. Referencia: 1.1 = +2 %; 1.3 = +15 %.
3. **Dip acotado** — la velocidad media de red **NO** permanece **bajo 20 km/h por más de 15 minutos consecutivos** en ninguno de los dos picos (AM 07-09 / PM 18-20). El doble criterio duración-Y-profundidad distingue 1.2 (toca 21.3 y **recupera** → pasa) de 1.3 (**sub-20 ancho sostenido** → falla). Requiere serie de velocidad sub-horaria (`edgeData freq=60` la soporta; el bucketing horario de `analyze24.py` es demasiado grueso para el test de 15 min).

**Regla de combinación.** El día **falla** si `teleports > 50` **O** dispara cualquiera de las otras dos. Ante señales en desacuerdo, **manda teleports** (es la menos contaminada por el ciclo de semáforo en tramos cortos).

**Consecuencias.**
- `analyze24.py` **conserva la racha sub-8 intacta** por compatibilidad con C2, pero se interpreta con el caveat de que es **v1-específica**; el **veredicto autoritativo para v2 es este criterio multi-señal**. Este ADR no reescribe el código.
- **B3.2 usa este criterio como gate de aceptación por-día** sobre los 60 seeds del dataset regenerado a scale 1.1. Un día que falla el criterio es **bandera** (a investigar / posible descarte), **no se absorbe en silencio**.
- **Evaluador pendiente (trabajo de B3.2, no de B3.1.5).** Evaluar este criterio requiere un **evaluador multi-señal sub-horario** que lea `edgeData freq=60` + `stats.xml` + `tripinfo.xml` y emita el veredicto de las tres señales por-día. **`analyze24.py` no lo soporta** (buckets horarios, no puede testear "15 min consecutivos" ni leer la distribución de duración). Hasta que B3.2 lo implemente, **este criterio es normativo, no automatizado**: define la vara, no la herramienta que la mide.
- **Robustez a seed.** El criterio se fijó sobre seed 42 (único del C3). El margen de los umbrales respecto del régimen de 1.1 (teleports 11 vs corte 50; Δdur +2 % vs +10 %) está **dimensionado para que la variación de seed de los 60 días de B3.2 no cruce el umbral espuriamente**. Si en B3.2 una fracción material de los 60 días falla el gate, eso es señal de que **el margen o el scale necesitan revisión** — queda como **trigger explícito**, no como sorpresa.

**Alcance / límite.** Define el criterio de aceptación de drenaje del **dataset v2** (B3.2 en adelante). No toca producción, ni el GRU de TTH-09, ni la variable predicha (D-009/D-013). No modifica `analyze24.py`. Los cortes son net-específicos de v2: una reconstrucción futura del net los invalida (re-localizar el cliff, como hizo C3).

**Referencias.**
- Evidencia: `simulation/data/datasets/miraflores_laborable_60d/calibracion/SWEEP_C3_RESULTS.md` (barrido v2, cliff 1.2–1.3, hallazgo del colapso suave ~11 km/h).
- Señales: `stats.xml` (`<teleports total>` y `vehicleTripStatistics @duration`, la media agregada de duración de viaje), `edgeData freq=60` (serie de velocidad de red sub-horaria).
- **Nota de implementación (B3.2.a, 2026-06-03):** la señal #2 (Δduración media) se computa de la **media agregada** `vehicleTripStatistics @duration` de `stats.xml`; **`tripinfo.xml` no se genera ni se persiste**, porque el corte es sobre la media, no sobre la distribución (percentiles). La mención a `tripinfo.xml` como fuente era referencia de origen del dato, no dependencia operativa.
- Relacionadas: D-008 (SUMO end-to-end), D-013 (target meanTimeLoss del track STGNN, mismo dataset).
- Scale de operación: C3 = 1.1 (fijado en B2; `gen_day.sh` lo adopta en B3.2).

---

## D-PENDING-001 — Modelo: reutilizar `time_then_space.py` o GRU desde cero
**Estado:** **Resuelta por D-006** (2026-05-11)

**Contexto histórico:** El archivo `ia_prediction_service/src/models/time_then_space.py` implementa una arquitectura **Time-then-Space**: encoder lineal + RNN(cell='gru') temporal + DiffConv espacial + MLPDecoder. La celda recurrente ya era GRU por defecto. Existían 5 checkpoints entrenados en `ia_prediction_service/notebooks/logs/`.

**Resolución:** Ver D-006. Se descarta `time_then_space.py` por exceder el alcance temporal y metodológico del trabajo. Se implementa GRU univariado desde cero.

**Acción de archivo:** Esta entrada se mantiene como traza histórica de la decisión. No mover.
