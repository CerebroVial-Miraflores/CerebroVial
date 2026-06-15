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
| D-015 | Cerrada | 2026-06-03 | Revalidación del veredicto STGNN sobre v2 (1660): STGNN supera al GRU en severo; D-011/Fase 5 revertido (adopción abierta) |
| D-016 | Cerrada | 2026-06-05 | `intersections` como entidad de primera clase; cámara accesorio; puente al grafo vía `intersection_edges` (Fase A) |
| D-017 | Cerrada | 2026-06-07 | Refundación del módulo de visión (contenido refundado, tubería reusada previo saneamiento) |
| D-018 | Cerrada | 2026-06-07 | Arquitectura del muestreador de visión (scheduler único, modelo compartido, instancia dueña de cámaras) |
| D-019 | Pendiente registro | 2026-06 | Tap de detecciones por-frame (`latest_detections` → overlay del front, Fase 4 Mitad A); ya referenciado en código (`multi_camera`, `detections`, serializers). ADR formal pendiente |
| D-020 | Cerrada | 2026-06-12 | Edge como servicio de inferencia centralizada sobre streams remotos (enmienda D-004: se descarta el edge computing IoT distribuido) |
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

**Enmienda 2026-06-03 — señal #3: de "racha consecutiva sub-20" a "fracción de intervalos sub-20"**

La señal #3 se fijó (arriba) como *"la velocidad media de red no permanece bajo 20 km/h por más de 15 min consecutivos"*. Al operacionalizarla en B3.2.a (evaluador `simulation/scripts/evaluate_drenaje.py`: mean_kmh de red por intervalo de 60 s ponderado por `sampledSeconds`) resultó **inerte** y se redefine.

**Qué decía y por qué no sobrevivió.** La racha consecutiva máxima de intervalos sub-20 es **1–2 min en AMBOS regímenes** —el que drena (s11/scale 1.1) y el que colapsa (s13/scale 1.3)—, muy por debajo del corte de 15 min. El dip de v2 es **intermitente-frecuente, no sostenido**: no existe un bloque de 15 min consecutivos sub-20 ni siquiera en el día que colapsa. "Racha consecutiva" no discrimina.

**El método del C3 no era reproducible en código.** Sus valores absolutos no coinciden con la serie ponderada por-60 s: C3 caracterizó el dip de s11 como "incipiente →23.7" (nunca sub-20), pero la media de red ponderada por `sampledSeconds` por-60 s toca **18.8 km/h** en el pico PM de s11. C3 midió con otra resolución/scope (no documentada). Por eso esta enmienda **no recalibra contra C3**: redefine la operacionalización sobre **evidencia nueva** —caracterización read-only sobre los fixtures versionados s11/s13 (`simulation/tests/fixtures/drenaje_c3/`), 2026-06-03.

**Nueva regla.** La señal #3 pasa a: **el día falla si la fracción de intervalos con mean_kmh < 20 km/h supera el 10 % en cualquiera de las dos ventanas de pico** (AM 07-09 h / PM 18-20 h). mean_kmh = velocidad de red ponderada por `sampledSeconds` por intervalo de 60 s (igual que C3). El umbral de velocidad (20 km/h) se mantiene; lo que cambia es el corte: "15 min consecutivos" → ">10 % de intervalos".

**Evidencia (caracterización, fixtures s11/s13, ventanas de pico, frac = nº sub-v / nº con tráfico):**

| ventana | frac@18 | frac@20 | frac@22 | min | mediana |
|---|--|--|--|--:|--:|
| s11-AM | 0/120 · 0.0% | 1/120 · 0.8% | 13/120 · 10.8% | 19.79 | 25.30 |
| s11-PM | 0/120 · 0.0% | 2/120 · 1.7% | 27/120 · 22.5% | 18.81 | 24.23 |
| s13-AM | 3/120 · 2.5% | 21/120 · 17.5% | 56/120 · 46.7% | 16.63 | 22.87 |
| s13-PM | 2/120 · 1.7% | 24/120 · 20.0% | 54/120 · 45.0% | 16.35 | 22.37 |

**Por qué 20 km/h.** Es el único de los tres umbrales que separa con margen a ambos lados. @18 la señal es débil (s13 apenas registra, 1.7–2.5 %); @22 contamina (s11-PM salta a 22.5 %, solapando el rango de s13@20); @20 separa **~10×** con hueco ancho (s11 0.8–1.7 % vs s13 17.5–20.0 %).

**Por qué 10 %.** Parte el hueco [2 %, 17 %] de forma asimétrica conservadora: **~6× de margen** sobre el peor del día que drena (s11-PM 1.7 %) y **~1.75–2× de margen** bajo el peor del que colapsa (s13-PM 20 %, AM 17.5 %). Más holgura del lado del día sano —para que la variación de seed de los 60 días no empuje un día que drena por encima del corte— y suficiente del lado del que colapsa. Mismo principio de margen que teleports/duración.

**Por qué fracción-de-cola y no media/mediana.** Las medianas de s11 y s13 difieren solo **~2 km/h** (24–25 vs 22–23), pero la fracción sub-20 difiere **~10×**. La congestión de v2 vive en la **cola** de la distribución de intervalos, no en el centro: el día que colapsa no tiene una red sistemáticamente más lenta, tiene **más intervalos puntualmente congestionados**. Esto explica por qué "racha consecutiva" falló (buscaba un bloque sostenido que no existe) y por qué "fracción" funciona (mide la extensión de la cola — que es lo que "ancho" significaba en la caracterización cualitativa del C3).

**Regla por ventana.** "Cualquiera de las dos ventanas dispara" se mantiene; PM es sistemáticamente la peor, así el criterio agarra naturalmente la ventana más severa. teleports/duración siguen mandando en los casos limítrofes; el dip recupera su rol espacial-temporal como tercera condición sin volverse la señal frágil que decide al filo.

**Alcance de la enmienda.** Redefine **solo la operacionalización de la señal #3**. Las señales #1 (teleports ≤ 50) y #2 (Δduración media ≤ 280 s), la regla de combinación (AND; teleports primaria ante desacuerdo) y el resto de D-014 no cambian.

---

## D-015 — Revalidación del veredicto del track STGNN sobre el universo v2 (1660): el STGNN supera al GRU en régimen severo (D-011/Fase 5 se revierte como veredicto técnico; adopción abierta)
**Estado:** Cerrada · 2026-06-03 · **Revisa:** cierre de Fase 5 del track STGNN (veredicto sobre 375) y el criterio de adopción de D-011/D-012 · **Track:** investigación; NO decide adopción en producción

**Contexto.** El cierre de Fase 5 (2026-06-02, registrado en `ESTADO_Y_PROXIMOS_PASOS.md`) aplicó el criterio de adopción de D-011/D-012 —"el STGNN se adopta solo si supera al baseline GRU en el régimen severo; si no, se conserva el GRU y se documenta el hallazgo"— sobre el **universo de 375 nodos / 504 aristas** y conservó el GRU: el STGNN ganaba en régimen normal y agregado pero **perdía en congestión máxima** (test_081 MAE@30 STGNN 24.02 vs GRU 23.20). B4 reentrenó ambos modelos sobre el **universo real v2 (1660 nodos / 2948 aristas, grafo 5.8× más denso)** —el universo sobre el que el sistema realmente opera— para revalidar ese veredicto.

**Validez de la revalidación.** Config de entrenamiento **idéntica** a la corrida de 375 (épocas, horizonte MAE@30, hiperparámetros; solo cambian N y el split estratificado B4.2). Ambos modelos sobre el **mismo universo de evaluación** (scaler train-only idéntico 6.820/26.581, `severe_test=[71,85,90]`, conteos de ventanas idénticos por corte). Ambos convergieron limpio (GRU early-stop ep 12, STGNN ep 28; sin divergencia). La única variable entre el veredicto viejo y este es **la densidad del grafo** —que es lo que la revalidación aísla.

**Hallazgo — D-011/Fase 5 se revierte como veredicto técnico.** Sobre 1660 el STGNN **supera al GRU en el régimen severo**, el régimen sobre el que el veredicto viejo había fallado a favor del GRU:

| corte (@30 min) | métrica | GRU-1660 | STGNN-1660 |
|---|---|--:|--:|
| severe_dia {71,85,90} (primario) | MAE | 6.006 | **5.882** |
| | RMSE | 19.831 | **18.000** |
| | R² | 0.700 | **0.753** |
| severe_pico 18-20h (afilado) | MAE | 7.970 | **7.421** |
| | RMSE | 27.858 | **24.634** |
| | R² | 0.737 | **0.794** |

El STGNN gana también en `test_all` y `test_normal` (los cuatro cortes), pero lo decisivo es que **gana donde antes perdía**: la congestión severa.

**Interpretación — la utilidad de la componente espacial escala con la densidad del grafo.** En 375/504 el grafo era demasiado ralo para que la vecindad aportara, y el STGNN perdía en severo; en 1660/2948 (5.8× aristas) la correlación espacial entre tramos vecinos —la apuesta original del track (D-011/D-012, señal confirmada: contraste vecinos-1-salto vs no-vecinos)— sí tiene señal explotable. La magnitud, con honestidad: **en MAE@30 severo el resultado es casi un empate (5.882 vs 6.006 = 0.12 s sobre ~6 s, ~2%), diferencia que podría no ser significativa frente a la varianza de entrenamiento. La ventaja robusta y consistente del STGNN está en RMSE (~9%, 18.0 vs 19.8) y R² (+0.05, 0.753 vs 0.700) —los errores grandes—, no en el MAE medio.** El STGNN penaliza mejor los outliers, coherente con que la información espacial ayuda cuando la congestión se **propaga entre tramos**. Por horizonte: el STGNN gana en 15/20/25/30 min (donde la propagación espacial tiene tiempo de manifestarse) y el GRU empata/gana marginal en 5/10 min (corto plazo, dominado por la dinámica temporal local). **El veredicto "STGNN gana en severo" se sostiene por RMSE / R² / horizontes largos, no por el MAE medio** —y decirlo así es más defendible, no más débil. El track STGNN tenía fundamento; lo que faltaba en Fase 5 no era el modelo sino un universo con densidad espacial suficiente.

**Lo que este veredicto NO decide — adopción en producción ABIERTA.** El criterio de D-011/D-012 condicionaba la adopción a *"un margen que justifique su complejidad e integración; si no lo supera, se conserva el GRU"*. B4 establece la **primera mitad**: el STGNN supera al GRU en severo sobre el universo real. Pero si ese margen —**modesto en MAE, robusto en RMSE/R²**— justifica el **costo operativo** (CPU-bound, ~4× más lento de entrenar —~2 h vs ~36 min—, dependencia `tsl` con venv separado, servido in-process no resuelto) es precisamente la **decisión de adopción que queda abierta**. El veredicto técnico es **condición necesaria pero no suficiente** para adoptar. Honestidad sobre el alcance: el dato sostiene *"el STGNN gana técnicamente en severo sobre el universo real"*, no *"se adopta el STGNN"*. La decisión de adopción es de producto/ingeniería y se toma con estos datos sobre la mesa, no aquí.

**Alcance / límite.** Track de investigación; NO toca producción (TTH-09, D-009/D-010 intactos). **Limitación metodológica: el veredicto se apoya en una corrida por modelo (n=1, sin validación multi-seed)** —a diferencia del gate de drenaje (60 seeds) y de IE05 (9/10 seeds), donde el rigor multi-seed fue exigido. La consistencia del resultado a través de RMSE, R² y los horizontes largos sugiere que **no es artefacto de inicialización**, pero una **confirmación multi-seed robustecería el veredicto antes de cualquier decisión de adopción** —se registra como condición para cerrar la adopción, no para el veredicto técnico direccional. El resultado es net-específico de v2: una reconstrucción futura del net podría volver a mover la densidad y con ella el resultado. No reescribe D-011/D-012/Fase 5 in-place: esta entrada los revisa desde afuera (mismo patrón que D-012 sobre D-011).

**Referencias.** Métricas: `ia_prediction_service/scripts/miraflores_{baseline,stgnn}_metrics.json` (retrain B4.3, commit `467c5106`). Split estratificado: `ia_prediction_service/src/data/miraflores_split.py` (B4.2). Corte de régimen severo derivado del split: `severe_in()` + `PM_PEAK_*` en el mismo módulo. Contexto histórico (375, no contendiente): `ESTADO_Y_PROXIMOS_PASOS.md` §Fase 5. Relacionadas: D-011 (apertura del track), D-012 (escenario Miraflores completo), D-013 (target timeLoss), D-014 (gate de drenaje v2).

---

## D-016 — `intersections` como entidad de primera clase; cámara como accesorio; puente al grafo vía `intersection_edges`
**Fecha:** 2026-06-05 · **Estado:** Cerrada · **Fase:** A (modelo de datos de intersecciones)

**Contexto.** Hasta Fase A, las cámaras anclaban directamente a `graph_nodes` (`cameras.node_id` FK) y se sembraban 4 placeholders friendly-name. No existía una entidad de primera clase para las intersecciones semaforizadas del PMU de Miraflores: el mapeo canónico (`documentation/contracts/mapeo_pmu_edges_v2.yaml`, 11 semaforizadas + ovalo_gutierrez) vivía solo como YAML, sin materializarse en BD. El control (`motor_decisions`/`engine_active_state`) ancla a `graph_nodes.node_id` y es independiente de las cámaras.

**Decisión.**
1. **`intersections` es entidad de primera clase.** `intersection_id` (= `nombre` del mapeo) PK; `junction_id` (junction SUMO, opaco), `lat`/`lon`, `geom` POINT 4326, `los_pmu` y `tls_id` nullable. Las cámaras, el control y el puente al grafo cuelgan de acá.
2. **La cámara es accesorio de la intersección.** `cameras` pierde `node_id` (FK→graph_nodes) y gana `intersection_id` (FK→intersections, nullable) + `stream_url`. Ya no ancla al grafo.
3. **El puente al grafo va por `intersection_edges`**, no por una FK directa intersección→nodo. `intersection_edges(intersection_id FK→intersections, edge_id FK→graph_edges, direction)`, PK compuesta. `junction_id` queda como dato opaco; la resolución al grafo se hace por aristas (`edge_id` SUMO crudo → `graph_edges`). Esto acopla el seed al net real: orden `invoke seed` → `build_graph_geometry.py` → `invoke seed-intersections`, con pre-check fail-fast.
4. **`tls_id` solo se puebla cuando está verificado** contra una config de control real. Hoy: `larco_benavides` (= su `junction_id`, idéntico en el mapeo y en `corredor_adaptive.py`). El resto NULL — no se mete a la base un `tls_id` sin verificar.
5. **`intersections.geom` sin índice GIST** (`spatial_index=False`), consistente con los GIST comentados desde la migración inicial. Las queries `ST_DWithin` corren igual.

**Deudas nombradas (registradas en `documentation/docs/TODO.md`).**
- **DEUDA-CAM-GEO** — la asociación cámara-Claro ↔ intersección es nominal (stream_url asignado 1:1 arbitrariamente); falta concordancia geográfica real.
- **DEUDA-CTRL-TLS** — 10 de 11 intersecciones sin `tls_id`/nodo de control. El modelo lo soporta; falta poblarlo en una fase de control futura. `arequipa_angamos` es el caso "casi listo" (ya es nodo de control sembrado; falta solo verificar su `tls_id` SUMO).

**Consecuencias.** El único consumidor de `cameras.node_id` (`/api/intersections`) pasa a derivar el nombre de `intersection_id`. El frontend (`CameraDetailView.tsx`) y el edge (`run_server.py`) usan `camera_id`, no `node_id`: se ajustan en fases B/C. El control no se toca.

**Referencias.** Migración `f3a9c1d2e4b7_intersections_cameras_bridge.py`; ORM `shared/cerebrovial_shared/database/models.py` (`IntersectionDB`, `IntersectionEdgeDB`, `CameraDB`); seed `scripts/seed_intersections.py`; contrato `documentation/contracts/intersections_contract.md`; schema `documentation/docs/DATA_MODEL.md`. Fuente: `documentation/contracts/mapeo_pmu_edges_v2.yaml`.

---

## D-017 — Refundación del módulo de visión (contenido refundado, tubería reusada previo saneamiento)
**Fecha:** 2026-06-07 · **Estado:** Cerrada

**Contexto.** El módulo de visión preexistente (TTH-08) fue **exploratorio**: sirvió para descubrir qué métricas eran viables sobre los streams, no para producir datos válidos. Dos auditorías read-only de esta sesión establecieron:
- **`vision_aggregates` es write-only:** se escribe desde el edge, pero nadie la lee — ni endpoint, ni frontend, ni el GRU. Hoy está vacía salvo corridas en vivo.
- **Computa y persiste cuatro métricas no confiables:** `mean_occupancy`, `density_vehicles_per_km`, `mean_speed_kmh`, `flow_vehicles_per_hour`. Todas dependen de una calibración que no existe. `flow_vehicles_per_hour` es **presencia disfrazada de flujo** (ya identificado en `documentation/docs/CIERRE-metricas-vision-flujo.md`).
- **Las zonas son polígonos en píxeles sin normalizar**, definidos en el YAML del edge (`conf/vision/default.yaml`), con `camera_id` (`"CAM_001".."CAM_004"`) que ni siquiera matchea los `cam_<intersection_id>` reales de la BD.
- **La tubería de infraestructura** (`MultiCameraManager`, `AsyncVisionPipeline`, `AsyncTrafficAggregator`, `PostgresTrafficRepository`, `RealtimeBroadcaster`, `FrameAnalysis`) es **estructuralmente sana pero arrastra 5 bugs confirmados** (B1–B5, abajo).

CLAUDE.md rotula al subsistema de visión como *"subsistema mejor armado, con tests reales"* (regla protegida, levantada en TTH-08 Fase 2). La realidad auditada matiza fuerte ese rótulo: write-only, sin lector, métricas no confiables y un montaje frankenstein heredado de la fase exploratoria.

**Decisión.** Se **refunda** el módulo de visión.

1. **Contenido refundado (sale, no se difiere).**
   - Las cuatro métricas no confiables (`occupancy`, `density`, `speed`, `flow_vph`) **salen del alcance y del cómputo**. No se persisten más. Cada una se reincorpora solo cuando se cumpla su precondición de calibración (ver *Decisiones diferidas*).
   - Las zonas en YAML/píxeles sin normalizar **salen**. Las reemplaza una entidad **`vision_zones` en BD** (geometría normalizada, FK a cámara, estado de calibración). Diseño en fase posterior.
   - El **código muerto** de la fase exploratoria se elimina (auditoría de código muerto pendiente; ya detectados: el CLI `edge_device/src/main.py` con `analysis.total_count` inexistente, y el stub no-op `set_pipeline`).
   - En el **espíritu de D-005** (honestidad de datos: se reporta la realidad, no una versión inflada) y del **lema del equipo *«no verde con asterisco»***: no se persiste una magnitud física que el dato no entrega.

2. **Tubería reusada *previo saneamiento* (no se reescribe lo sano; se arreglan los bugs confirmados).**
   - **B1 (crítico):** `_run_camera_pipeline` **bloquea el event loop** — itera un generador síncrono con `queue.get`/`time.sleep` dentro del loop async. Impide multi-cámara real. Requiere re-arquitecturar el cruce sync/async.
   - **B2:** `aggregator.stop()` **nunca se llama** → fuga de thread garantizada en cada baja de cámara.
   - **B3:** `activate_camera` **sin lock** → la invariante de instancias (un solo modelo vivo) no se sostiene bajo requests concurrentes.
   - **B4:** subscriber SSE **colgado/fugado** tras baja de cámara o ante excepción distinta de `CancelledError`.
     - **Estado (2026-06-07, B1 Paso 0):** el **disparador por idle quedó eliminado** — el watchdog ya no llama `remove_camera` (ahora solo apaga el render), así que ninguna cámara desaparece bajo un subscriber por inactividad. **Pendiente:** la desuscripción de SSE en **bajas explícitas** (`DELETE /cameras/{id}` y cambio de fuente en `activate_camera`) sigue sin hacerse. **Trigger / fix acotado:** `RealtimeBroadcaster.disconnect_camera(camera_id)` (encola un sentinel a los subscribers de esa cámara) + cierre del generador en `streaming.py`, llamado desde `remove_camera`. Diferido a propósito para no diluir la verificación negativa de B1 Paso 0.
   - **B5:** CLI legacy `main.py` con `analysis.total_count` inexistente (`AttributeError` latente) — se va con la limpieza de código muerto.

3. **Alcance de runtime.** **11 cámaras contando en simultáneo a 1 Hz** (muestreador), con **aislamiento real** (bajar una no afecta a las otras). Validado en cómputo por **Benchmark 1** (11 inferencias @320, peor caso ~0.6 s < 1 s a 1 thread). El **MJPEG full-FPS (640)** queda **single-slot**: solo la cámara que el operador está mirando. Las 11 cuentan; una se ve en detalle. La arquitectura actual **no** implementa esto (monta 11 pipelines pesados que bloquean el loop, B1); el saneamiento debe converger hacia **un muestreador que recorra las 11**, no 11 pipelines independientes.

4. **Calibración por operador (fase posterior, diseño no cerrado acá).** CRUD de cámaras con datos de calibración (lat/long, ángulo, distancia cámara–línea); definición de **tramos entre puntos** con su valor en metros asignado por el operador; **dibujo de zonas/líneas desde el front** (hoy el front manda `zones: {}` vacío — no existe UI).

5. **Estado de sensor con NULL-con-motivo.** `vision_aggregates` debe distinguir **"0 vehículos medidos"** de **"no medido"**, y en este último caso el **motivo**: cámara caída, sin stream disponible, o en recalibración. La honestidad de datos se lleva al estado del sensor: YOLO **no persiste datos por las puras** cuando no hay señal válida.

**Principio de migraciones (aplica de acá en adelante).**
- Migraciones **aditivas y no destructivas**: agregar tablas/columnas, nunca recrear tablas existentes. Las columnas nuevas nacen **nullable o con default**; agregar una columna no borra data existente.
- **DROP solo en migraciones explícitamente marcadas como destructivas**, nunca como efecto colateral de otro cambio.
- **Excepción puntual de esta refundación:** el DROP/recreación de `vision_aggregates` y la baja de las zonas YAML **son destructivos a propósito**, y son seguros porque `vision_aggregates` está **vacía** (sin data real que perder). De ahí en adelante, régimen aditivo estricto.
- Las columnas de las métricas diferidas (`occupancy`/`density`/`speed`/`flow`) se prevén en el esquema como **NULL-honesto**: existen, pero nadie las escribe hasta la calibración. Reincorporarlas es **activar escritura, no migrar estructura**.

**Decisiones diferidas (con trigger).**
- **Hot-reload vs. reinicio al calibrar.** Ideal: **hot-reload** (cambiar la config de zonas sin parar captura/inferencia, dado que las zonas se consumen en el cómputo espacial posterior, no en la captura). Viabilidad **no confirmada**: depende de si `ZoneCounter` consume las zonas desde una estructura mutable-en-caliente o si están horneadas en el pipeline. **Trigger:** auditoría dirigida a `ZoneCounter` en Fase 0. El modelo de datos se diseña **agnóstico**: en ambos casos la cámara entra en estado `recalibrating` (NULL-con-motivo) durante la calibración; solo cambia cuánto dura y si el MJPEG parpadea.
- **Reincorporación de cada métrica diferida.** **Trigger** = su precondición de calibración respectiva: `occupancy`/`density` → zonas + denominador en metros; `speed` → homografía / `pixels_per_meter` validado; flujo-por-cruce → tracking estable + línea calibrada (ya diferido en `CIERRE-metricas-vision-flujo.md`).

**Lo que NO es esta refundación.**
- No se reescribe la tubería sana (captura, agregación, broadcast, repo): se **reusa previo saneamiento** de B1–B5.
- **No se mergea a master.** Vive en rama propia post-demo.
- **No compite con el demo** (predictivo con GRU); arranca **después** del demo.

**Plan por fases (referencia; cada fase con su propio gate).**
- **Fase 0:** saneamiento de tubería (B1–B5) + auditoría de `ZoneCounter` (resuelve hot-reload) + decisión de arquitectura del muestreador.
- **Fase 1:** modelo de datos (`vision_zones`, calibración, tramos, estado de sensor NULL-con-motivo) — agnóstico a hot-reload/reinicio.
- **Fase 2:** canal vivo (tap por-frame read-only desde `FrameAnalysis`, SSE sin BD) + canal agregado (`vision_aggregates` reducido a presencia: `count_mean`/`max`/`min`, `sample_count`, ventana 60 s).
- **Fase 3:** UI de calibración (CRUD cámaras, dibujo de zonas/líneas, tramos).
- **Transversal:** limpieza de código muerto (con auditoría dedicada previa).

**Consecuencias.** **CLAUDE.md debe corregirse:** su rótulo del módulo de visión (*"subsistema mejor armado, con tests reales"*) refleja el estado **pre-refundación** y contradice este record. La corrección va en la **limpieza transversal de documentación** — junto con los `DATA_MODEL.md` obsoletos que aún documentan `vision_aggregates` como tabla *"a crear"* y `vision_tracks`/`vision_flows` ya dropeadas. Esta tarea **no toca CLAUDE.md ni `DATA_MODEL.md`**; solo deja registrada la acción pendiente.

**Referencias.** Auditorías read-only de esta sesión (tubería + esquema de visión). Componentes citados: `edge_device/src/vision/application/services/multi_camera.py`, `.../pipelines/async_pipeline.py`, `.../aggregators/async_aggregator.py`, `.../infrastructure/persistence/postgres_repository.py`, `.../infrastructure/broadcast/realtime_broadcaster.py`, `.../domain/entities.py` (`FrameAnalysis`); ORM `shared/cerebrovial_shared/database/models.py` (`VisionAggregateDB`, `CameraDB`); seed `scripts/seed_intersections.py`; config `edge_device/conf/vision/default.yaml`. Decisiones relacionadas: D-005 (honestidad de datos), D-007 (visión como componente demostrable), D-016 (cámara como accesorio de intersección). Cierre previo: `documentation/docs/CIERRE-metricas-vision-flujo.md`.

---

## D-018 — Arquitectura del muestreador de visión (scheduler único, modelo compartido, instancia dueña de cámaras)
**Fecha:** 2026-06-07 · **Estado:** Cerrada

**Contexto.** Resuelve la decisión de arquitectura que el D-017 dejó como núcleo de Fase 0. Tres fuerzas la motivan:
1. **El bug B1** — el event loop se bloquea porque `_run_camera_pipeline` itera un generador síncrono (`queue.get` + `time.sleep`) en el thread del loop, y el cableado actual monta **11 pipelines independientes** (11 modelos, 22 threads, 11 aggregators).
2. **El requisito de muestreo permanente** — las 11 cámaras deben muestrear de forma continua, no on-demand.
3. **La visión de producto escalable** — a futuro, muchas más cámaras que las 11.

Una auditoría read-only de Fase 0 estableció que el modelo YOLO es **stateless por llamada** (modo predict, no `track(persist=)`), por lo que un único modelo puede inferir frames de varias cámaras en secuencia **sin contaminación de estado**. Un spike de carga sostenida (5 min, 11 clips reales, 1 modelo, 1 thread, con persistencia) confirmó la premisa física.

**Validación por spike (números medidos, no asumidos).**
- 11 cámaras a 1 Hz @imgsz 320, sostenido: **ciclo medio 0.118 s (~12 % del presupuesto de 1 s)**, p95 0.123 s, **0/300 desbordes**.
- **Sin deriva temporal** en 5 min (Δ entre primeros y últimos 30 ciclos ≈ −0.002 s).
- **Persistencia no es cuello:** 11 escrituras/ventana ≈ 18 ms con engine pooleado (~1.7 ms c/u).
- **Margen** (extrapolación conservadora con el peor caso del Benchmark 1, 26 ms/inf @320): techo holgado en torno a **varias decenas de cámaras por instancia** a 320.
- A **640 también entra** (0.324 s, ~32 % del presupuesto).

**Decisión.**
1. **Un scheduler único con UN modelo YOLO compartido por instancia de edge.** El scheduler recorre sus cámaras a 1 Hz e infiere con el modelo único en secuencia. Reemplaza los 11 pipelines independientes y el generador síncrono bloqueante (**sanea B1**: el scheduler cede el loop correctamente).
2. **La instancia de edge es dueña de un CONJUNTO CONFIGURABLE de cámaras** (hoy las 11). El código asume *"mis cámaras"*, no *"todas las cámaras"* — costura para escalar horizontalmente a K instancias sin reescritura.
3. **Dos salidas por cámara:** tap vivo por-frame (conteo instantáneo, SSE, sin DB) y agregado por-ventana (persistido). Ambas derivan del mismo muestreo.
4. **Estado por-cámara LIVIANO** (tracker ByteTrack, buffer de ventana) se mantiene por-cámara; **el modelo NO se duplica**.
5. **El MJPEG a 640 con boxes visibles es un consumidor adicional single-slot** sobre la cámara que el operador está mirando. Las 11 muestrean a 320 sin render (background); solo la activa renderiza a 640. No compiten.
6. **`imgsz` y frecuencia de muestreo CONFIGURABLES, no horneados** (hoy no existe `imgsz`, corre a 640 default). Son las palancas de densidad de cámaras por modelo.
7. **CONJUNTO DE CÁMARAS MUTABLE EN CALIENTE:** agregar o quitar una cámara del muestreo activo es operación de primera clase — el scheduler la incluye/excluye en el ciclo siguiente, **sin reiniciar ni reconstruir el modelo**. La cámara nueva aporta su propio estado liviano. (Contraparte runtime del conjunto configurable.)

**Advertencia para la implementación de B1 (mordida anticipada, contemplar sí o sí).** El `POST /cameras/{id}` actual (del track de cámaras, modelo single-slot) hoy significa *"activá la única cámara, bajá las demás"*. Al pasar al scheduler debe **redefinirse** como *"sumá esta cámara al conjunto del scheduler"*, sin afectar a las otras. Si el diseño de B1 no contempla este cambio de semántica, hay choque seguro entre un endpoint que asume single-slot y un scheduler multi-cámara.

**Riesgos conocidos con trigger (lo que el spike NO cubrió).**
- **Captura HLS en vivo NO medida.** El spike usó `.mp4` local (captura 10 ms/11 frames). En producción son 11 streams HLS de Claro en vivo, con latencia y jitter de red. **Conclusión clave: el cuello de botella de escala de esta arquitectura es la I/O de red, NO el cómputo** (la inferencia tiene margen de sobra). **Trigger:** spike de captura HLS concurrente en vivo antes de producción. La palanca ante desborde de red no es `imgsz` (la inferencia no es el problema) sino el manejo de captura/concurrencia de red.
- **RSS creció +388 MB en 5 min** (~1.3 MB/ciclo); el Δ por corrida decrece (388→152→27), consistente con caché de arranque que se aplana, y el tiempo de ciclo **NO derivó**. No concluyente: 5 min no descartan un leak lento. **Trigger:** corrida larga (horas) antes de deploy 24/7.
- **Sesión-por-escritura del aggregator:** barata hoy con pool caliente (1.7 ms c/u), pero escala con cámaras × zonas. Revisar a mayor densidad.

**Diferido con trigger.**
- **Orquestación multi-instancia** (qué instancia maneja qué cámaras, balanceo, descubrimiento): **trigger** = cuando una sola instancia no alcance. Las costuras (instancia-dueña-de-cámaras, `imgsz`/frecuencia configurables) quedan listas ahora; el orquestador no se construye todavía.
- **Alta de cámara ENTIDAD-NUEVA** (URL de stream nueva + datos de calibración, no sembrada en `cameras`): depende del modelo de datos (Fase 1) y del CRUD de calibración (Fase 3). El hot-reload de **runtime** (que el scheduler tome una cámara sin reiniciar) se diseña ahora; el alta de entidad completa llega con esas fases.

**Relación.** Implementa el saneamiento de **B1 del D-017**. Precede al modelo de datos de **Fase 1** (el scheduler define dónde viven las zonas indexadas por cámara, lo que condiciona el hot-reload de zonas — ver auditoría de Fase 0). **Referencias:** auditoría de Fase 0 (ZoneCounter + acoplamiento del loop) y spike de carga sostenida, ambos de esta sesión. Benchmark 1 (costo de inferencia) en `documentation/docs/CIERRE-metricas-vision-flujo.md`.

**Revisión (Fase 2, 2026-06-12) — default a 640.** El rationale original de arriba fijó **320** como default del muestreo por presupuesto de cómputo (decisión 6: `imgsz` configurable, no horneado). Esta sección NO se reescribe; se registra que el default **se revisa con evidencia posterior**. El benchmark de Fase 2 (sobre la cámara viva de Claro, no el spike de `.mp4`) mostró que **320 pierde ~la mitad de las detecciones** (320 → ~6 cajas; 640 → ~12-14 sobre vehículos chicos/lejanos de ángulo alto) y que el presupuesto a **640 es holgado**: **~183 ms/ronda de 11 cámaras** a 1 Hz (≪ 1000 ms), consistente con lo que el propio D-018 ya intuía ("A 640 también entra"). Conclusión: el 320 era **innecesariamente conservador para la calidad de detección**. Se mueve el default a **640** (un solo punto: `DEFAULT_IMGSZ` en `application/processors/smart_detection.py`). La palanca de configurabilidad (decisión 6) queda intacta: `imgsz` sigue siendo per-cámara vía `cfg.vision.model.imgsz` (hot-reload per-tick), y el techo (768 / GPU en deploy centralizado) se explora con datos cuando se quiera. El device auto-detect (cpu/mps/cuda en `yolo_detector.py`) no se toca; en Docker da `cpu`, correcto.

---

## D-020 — Edge como servicio de inferencia centralizada sobre streams remotos (enmienda a D-004)
**Fecha:** 2026-06-12 · **Estado:** Cerrada

**Decisión.** CerebroVial adopta **inferencia centralizada sobre streams remotos** (HLS de Claro) y **descarta el edge computing con dispositivos IoT distribuidos** (Raspberry Pi por cámara o grupo). El "edge" deja de ser un *lugar físico* (un dispositivo en la calle) y pasa a ser un *rol de software*: el componente que infiere (`edge_device/`), corriendo en contenedores/servicios del centro, escalable horizontalmente.

**Contexto.** El diseño original (y el espíritu IoT del documento de tesis) contemplaba hardware en campo haciendo inferencia local por cámara. Se evaluó como **ineficiente e innecesario**: los streams HLS de las cámaras de Claro ya son accesibles por red, así que la inferencia puede correr en el centro sin desplegar ni mantener hardware nuevo. **Fase 1 (B-fase1) ya materializó el giro**: `HlsKeyframeSource` consume el stream HLS remoto de Claro por red (captura keyframe-only), no una cámara conectada localmente a un dispositivo.

**Justificación.**
- **Cero hardware nuevo** que aprovisionar, calibrar o mantener en campo. El sistema corre íntegro en infraestructura central contenerizada (consistente con D-003, Docker local).
- **El cuello de botella de escala ya estaba identificado como I/O de red, no cómputo** (D-018: la inferencia tiene margen holgado de decenas de cámaras por instancia). Centralizar no agrava ese límite — lo concentra en un lugar gestionable.
- **Robustez operacional:** un servicio centralizado se monitorea, actualiza y escala con prácticas estándar de cloud; una flota de dispositivos distribuidos multiplica la superficie de fallo y mantenimiento.

**Impacto.**
- **El término "edge" en el proyecto refiere de acá en más al servicio de inferencia, no a un dispositivo.** El crecimiento se absorbe en el centro vía **réplicas de software**. Esto da contexto a la **épica futura de auto-scaling horizontal** del edge (profiling de capacidad por contenedor + partición de cámaras entre réplicas) — cuyas costuras D-018 ya dejó listas: *"la instancia de edge es dueña de un conjunto CONFIGURABLE de cámaras"*, `imgsz`/frecuencia configurables, conjunto mutable en caliente. El orquestador multi-instancia sigue diferido (trigger D-018: cuando una instancia no alcance).
- **Coherencia narrativa:** conviene que CLAUDE.md, el SDD y la comunicación con stakeholders/jurado hablen del edge como servicio, no como Raspberry Pi en la calle. La reconciliación de los documentos que aún dicen lo contrario está **señalada al cierre de esta decisión** (barrido aparte, no ejecutado acá).

**Trade aceptado.** La inferencia centralizada **depende de la disponibilidad de los streams remotos** (Claro hoy) y de la capacidad del centro; **pierde la resiliencia teórica del cómputo distribuido en campo** (un nodo caído ≠ sistema entero caído). Se acepta porque la **simplicidad operacional y el cero-hardware superan ese beneficio** para el alcance del proyecto (tesis/demo, no operación 24/7 de misión crítica).

**Relación.** **Enmienda a D-004** ("Pi física: demostración conceptual, no entrega"): D-004 conservaba el modelo edge-en-Pi como arquitectura-objetivo desplegable (solo no se entregaba el hardware); D-020 va más lejos y **descarta ese modelo como objetivo** — la inferencia centralizada sobre streams remotos no es un fallback sino la arquitectura. La premisa de despliegue de D-004 (y los docs derivados) queda **superada y flaggeada** para reconciliación. Construye sobre **D-018** (scheduler único, instancia dueña de un conjunto configurable de cámaras) y **D-003** (Docker local). Coherente con la materialización ya hecha en **Fase 1** (`HlsKeyframeSource`).

**Deuda de documentación señalada (para reconciliación posterior, NO reescrita en esta decisión):**
- `documentation/lean-inception/4-decisiones/DECISIONS.md` — **D-004**: "qué módulos correrían en Pi (edge_device)".
- `documentation/sdd/SDD_CEREBROVIAL.md` — §3/§6/§11 y el bloque ADR D-004 (≈ líneas 83, 257, 267, 282, 388): mapeo "edge físico/servidor" y "desplegar `edge_device` en hardware edge (p. ej. Raspberry Pi)".
- `documentation/docs/ARCHITECTURE_TARGET.md` (≈ línea 190) — "Edge Device | Raspberry Pi 4 (5 unidades para piloto)".
- `documentation/docs/RNF02_LATENCY_REPORT.md` (≈ líneas 14, 23) — diagrama "[Cámara] → [YOLOv8 Edge Device]".
- `CLAUDE.md` — **no** llama al edge "dispositivo IoT" (describe `edge_device/` como *módulo* de visión); único residuo es el nombre de carpeta, ya coherente con el rol-de-software. Sin cambio urgente.

---

## D-PENDING-001 — Modelo: reutilizar `time_then_space.py` o GRU desde cero
**Estado:** **Resuelta por D-006** (2026-05-11)

**Contexto histórico:** El archivo `ia_prediction_service/src/models/time_then_space.py` implementa una arquitectura **Time-then-Space**: encoder lineal + RNN(cell='gru') temporal + DiffConv espacial + MLPDecoder. La celda recurrente ya era GRU por defecto. Existían 5 checkpoints entrenados en `ia_prediction_service/notebooks/logs/`.

**Resolución:** Ver D-006. Se descarta `time_then_space.py` por exceder el alcance temporal y metodológico del trabajo. Se implementa GRU univariado desde cero.

**Acción de archivo:** Esta entrada se mantiene como traza histórica de la decisión. No mover.
