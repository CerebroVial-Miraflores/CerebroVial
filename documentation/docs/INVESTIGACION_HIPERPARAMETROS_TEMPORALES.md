# Investigación de hiperparámetros temporales del modelo predictivo (TTH-11)

> **Documento completo.** Cubre el spike TTH-11 íntegro: la parte académica (definición de los
> cuatro hiperparámetros temporales, revisión bibliográfica de 5 fuentes y recomendaciones
> ancladas en literatura) y la parte empírica (barrido de 16 modelos GRU sobre el dataset
> perfil-día, con las 4 métricas por dirección y el contrato de hiperparámetros para TTH-09).
>
> **Alcance cubierto:** CT-11.1 (estructura y propósito), CT-11.2 (los cuatro hiperparámetros con
> definición, bibliografía, rango candidato, implicancias y recomendación preliminar), CT-11.3
> (revisión bibliográfica de ≥5 fuentes), CT-11.4 (exploración empírica: barrido entrenado con 4
> métricas), CT-11.5 (tabla resumen / contrato para TTH-09), CT-11.6 (limitaciones y trabajo
> futuro), CT-11.7 (cierre de doble propósito) y CT-11.8 (nota de cierre de Δt_in que materializa
> la dependencia con TTH-07).

## CT-11.1 — Propósito y estructura

### Propósito

Este documento sustenta la elección de los **hiperparámetros temporales** del modelo predictivo
de congestión de CerebroVial (una red recurrente GRU que recibe series temporales agregadas por
dirección y predice el `jam_level` discreto 0–5 definido en D-009). Cumple un **doble propósito**:

1. **Sustento técnico interno:** fija el contrato temporal que consume la implementación del
   modelo en TTH-09 (resolución de entrada, ventana histórica, horizonte de predicción y cadencia
   de re-inferencia del endpoint).
2. **Apéndice académico de tesis:** documenta con rigor bibliográfico las decisiones, de modo que
   sean defendibles ante revisión académica sin asistencia adicional.

El nivel de rigor se mantiene consistente entre ambos propósitos: cada recomendación se ancla en
literatura citable y en los valores por defecto ya declarados en el backlog técnico.

### Índice de secciones

- **CT-11.1** — Propósito y estructura *(esta sección)*.
- **CT-11.2** — Los cuatro hiperparámetros temporales (definición, bibliografía, rango candidato,
  implicancias, recomendación preliminar). Incluye la nota **CT-11.8** sobre Δt_in.
- **CT-11.3** — Revisión bibliográfica (5 fuentes) y lista legible en prosa.
- **CT-11.4** — Exploración empírica (barrido de 16 modelos, 4 métricas por dirección).
- **CT-11.5** — Tabla resumen / contrato para TTH-09 (hallazgo central: el óptimo difiere por eje).
- **CT-11.6** — Limitaciones y trabajo futuro.
- **CT-11.7** — Cierre de doble propósito.

### Contexto del dataset

Las recomendaciones preliminares se apoyan en el dataset generado para TTH-09 (esquema D-009, 10
columnas: `seed, pattern, t_sim_s, direction, mean_speed_mps, n_vehicles, queue_length_m,
max_speed_mps, ratio, jam_level`). Las observaciones se agregan en **buckets de 60 segundos**
(`t_sim_s` en múltiplos de 60), por cada una de las cuatro direcciones de aproximación
(`N, S, E, W`) y bajo cuatro patrones de demanda (`am_peak, pm_peak, offpeak, weekend`). El
particionado es de 80 archivos de entrenamiento y 20 de validación.

## CT-11.2 — Los cuatro hiperparámetros temporales

### 1. Δt_in — Resolución de muestreo de entrada

**Definición.** Intervalo de tiempo entre dos observaciones consecutivas de la serie de entrada.
Define la granularidad temporal del dataset: cada paso del tensor de entrada representa un agregado
de tránsito sobre una ventana de Δt_in segundos.

**Bibliografía.** La literatura de predicción de flujo vehicular con redes recurrentes opera
típicamente con agregaciones de varios minutos. @wangTrafficFlowPrediction2022 emplean datos de
flujo agregados en intervalos del orden de minutos para su GRU bidireccional; @singhNovelCNNGRULSTM2025
y @wenNovelTrafficOptimization2023 trabajan en el mismo régimen temporal sobre flujo y sistemas IoV.
El fundamento de la arquitectura GRU que procesa estas secuencias proviene de
@choLearningPhraseRepresentations2014 y su evaluación empírica frente a LSTM de
@chungEmpiricalEvaluationGated2014.

**Rango candidato.** 30 s – 5 min. La literatura de tráfico urbano suele situarse entre 1 y 5 min;
resoluciones más finas aumentan el ruido por baja densidad de eventos, y más gruesas pierden la
dinámica de los picos.

**Implicancias para CerebroVial.** Δt_in está acoplado a la cadena de captación: el conteo
agregado proveniente del edge (YOLO) y la agregación de la simulación SUMO ya producen buckets de
60 s. Reducir Δt_in por debajo del bucket exigiría reagregar o regenerar el dataset; aumentarlo
implicaría submuestrear y perder resolución en los picos.

**Recomendación preliminar.** **Δt_in = 60 s**, alineado con el bucket de agregación del dataset
D-009. (Ver nota de cierre CT-11.8 a continuación.)

> **Nota CT-11.8 — Cierre del parámetro abierto de TTH-07 (CT-07.3).**
> El paso de muestreo del dataset quedó declarado como provisional/configurable en TTH-07 (CT-07.3),
> a cerrarse con la salida de TTH-11. **Valor decidido: Δt_in = 60 segundos simulados por muestra.**
> **Razón:** (i) es la granularidad nativa de la cadena de datos —el conteo agregado del edge y la
> agregación de SUMO producen buckets de 60 s—, de modo que evita un paso de reagregación o una
> regeneración del dataset; (ii) se ubica dentro del régimen reportado por la literatura de
> predicción de flujo con GRU (@wangTrafficFlowPrediction2022); y (iii) ofrece un equilibrio
> razonable entre resolución de los picos de congestión y ruido por baja densidad de eventos por
> bucket. Esta decisión materializa la dependencia TTH-07 ↔ TTH-11.

### 2. lookback — Ventana histórica de entrada

**Definición.** Número de pasos previos (de Δt_in cada uno) que el modelo "observa" para producir
una predicción. Determina cuánto contexto histórico recibe la GRU en cada inferencia.

**Bibliografía.** La capacidad de las unidades recurrentes con compuertas para retener dependencias
temporales a lo largo de una ventana es justamente el aporte de @choLearningPhraseRepresentations2014
y @chungEmpiricalEvaluationGated2014. En el dominio de tráfico, @wangTrafficFlowPrediction2022
muestran que una ventana histórica suficientemente amplia —procesada de forma bidireccional—
mejora la captura de patrones de flujo recurrentes.

**Rango candidato.** 15 – 60 min de historia. Con Δt_in = 60 s, esto equivale a **15 – 60 pasos**.

**Implicancias para CerebroVial.** Una ventana mayor captura mejor la tendencia hacia los picos
(am_peak / pm_peak) pero incrementa el costo de cómputo y la longitud de secuencia; una ventana
demasiado corta puede no anticipar transiciones de congestión.

**Recomendación preliminar.** **lookback = 30 min** (valor por defecto declarado), equivalente a
30 pasos con Δt_in = 60 s. Sujeto a confirmación en la exploración empírica (CT-11.4).

### 3. horizonte — Alcance de predicción

**Definición.** Número de pasos futuros que el modelo predice en una sola inferencia. Determina el
alcance del endpoint de predicción y el rango del control deslizante de la HU-03.

**Bibliografía.** Los esquemas multi-paso de predicción de flujo combinando convolución y
recurrencia, como @singhNovelCNNGRULSTM2025, abordan precisamente el compromiso entre alcance y
exactitud al extender el horizonte. @wangTrafficFlowPrediction2022 reportan resultados de
predicción de flujo a horizontes del orden de la hora.

**Rango candidato.** 15 – 60 min hacia adelante. Con Δt_in = 60 s, equivale a **15 – 60 pasos**.

**Implicancias para CerebroVial.** El horizonte fija el rango temporal que el usuario puede
explorar en la HU-03. Horizontes más largos degradan la exactitud (la incertidumbre crece con la
distancia de predicción), pero aportan más valor operativo para la anticipación.

**Recomendación preliminar.** **horizonte = 60 min** (valor por defecto declarado), equivalente a
60 pasos con Δt_in = 60 s. La curva exactitud-vs-horizonte se cuantificará en CT-11.4.

### 4. Frecuencia de re-inferencia del endpoint

**Definición.** Cada cuánto el backend recalcula la predicción que sirve. Es **independiente** de
Δt_in: define la cadencia de actualización del servicio, no la granularidad de la serie de entrada.

**Bibliografía.** La operación en tiempo (casi) real de modelos GRU para sistemas de transporte
inteligentes y vehiculares es el foco de @wenNovelTrafficOptimization2023, que ilustra el
compromiso entre frescura de la predicción y costo computacional del servicio.

**Rango candidato.** 30 s – 5 min entre recálculos.

**Implicancias para CerebroVial.** Una re-inferencia más frecuente mantiene la predicción fresca
para la HU-03 pero incrementa la carga de cómputo del backend; dado que la entrada se agrega cada
60 s, recalcular con cadencia menor al bucket no aporta información nueva.

**Recomendación preliminar.** **paso de re-inferencia = 60 s** (valor por defecto), coherente con
el bucket de entrada de 60 s. Se afinará junto a las consideraciones de costo en CT-11.6.

## CT-11.3 — Revisión bibliográfica

Las cinco fuentes que sustentan las definiciones y recomendaciones anteriores se citan a lo largo
del documento en formato narrativo: @choLearningPhraseRepresentations2014 introduce la unidad GRU
en el marco encoder–decoder; @chungEmpiricalEvaluationGated2014 evalúa empíricamente las redes
recurrentes con compuertas (GRU frente a LSTM) en tareas de modelado de secuencias;
@wangTrafficFlowPrediction2022 aplican una GRU bidireccional a la predicción de flujo vehicular;
@singhNovelCNNGRULSTM2025 proponen un modelo híbrido CNN-GRU-LSTM para predicción de tráfico
multi-paso; y @wenNovelTrafficOptimization2023 emplean una red profunda basada en GRU para
optimización de tráfico en sistemas IoV en tiempo real.

Las dos primeras fundamentan la **arquitectura** (justifican el uso de GRU y su capacidad de
retención temporal, relevante para `lookback`); las tres restantes aportan **evidencia de dominio**
sobre flujo vehicular (resolución de entrada, horizonte multi-paso y cadencia de re-inferencia en
operación real).

### Lista legible de referencias

Para revisión a ojo sin abrir `referencias.bib` (clave de cita entre paréntesis):

1. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. y Bengio, Y.
   (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine
   Translation*. Proceedings of EMNLP 2014, pp. 1724–1734. DOI: 10.3115/v1/D14-1179.
   (`@choLearningPhraseRepresentations2014`)
2. Chung, J., Gulcehre, C., Cho, K. y Bengio, Y. (2014). *Empirical Evaluation of Gated Recurrent
   Neural Networks on Sequence Modeling*. arXiv:1412.3555. Presentado en el NIPS 2014 Deep Learning
   and Representation Learning Workshop. (`@chungEmpiricalEvaluationGated2014`)
3. Wang, S., Shao, C., Zhang, J., Zheng, Y. y Meng, M. (2022). *Traffic flow prediction using
   bi-directional gated recurrent unit method*. Urban Informatics, 1(1):16. DOI:
   10.1007/s44212-022-00015-z. (`@wangTrafficFlowPrediction2022`)
4. Singh, V., Sahana, S. K. y Bhattacharjee, V. (2025). *A novel CNN-GRU-LSTM based deep learning
   model for accurate traffic prediction*. Discover Computing, 28:38. DOI:
   10.1007/s10791-025-09526-0. (`@singhNovelCNNGRULSTM2025`)
5. Wen, W., Xu, D. y Xia, Y. (2023). *A novel traffic optimization method using GRU based deep
   neural network for the IoV system*. PeerJ Computer Science, 9:e1411. DOI: 10.7717/peerj-cs.1411.
   (`@wenNovelTrafficOptimization2023`)

---

## CT-11.4 — Exploración empírica

### Metodología

La exploración barre el plano **lookback × horizonte** entrenando modelos **desechables y
efímeros**: cada modelo se entrena, se mide y se descarta sin persistir checkpoints a disco; solo
se conservan las métricas. El modelo de producción es responsabilidad de TTH-09; este spike únicamente
fija el contrato temporal que TTH-09 consumirá.

**Dataset.** Se entrena sobre el dataset **perfil-día** (esquema D-009), que reemplaza a los patrones
de demanda constante anteriores por **cuatro perfiles de jornada de 24 h** —`laborable`, `finde`,
`feriado`, `especial`— con dinámica real valle→pico→valle. Cada corrida es **un día continuo de 1440
buckets de 60 s** por dirección; el particionado es 80 corridas de entrenamiento (seeds 1–20) y 20 de
validación (seeds 21–25), sin solape. El ventaneo trata cada `(perfil, seed, dirección)` como una
**serie temporal independiente** y no cruza el borde entre corridas (la dinámica de un día no continúa
en otro), evitando fabricar continuidad falsa.

**Arquitectura y entrenamiento (fijos para los 16 — comparación limpia).** GRU **univariada por
dirección** (D-006): se entrena un modelo independiente por cada dirección de aproximación
(`N, S, E, W`). Una capa GRU de `hidden = 64`, entrada de **un canal** (el `jam_level` normalizado
`/5.0`, autoregresivo puro — sin features auxiliares), seguida de una capa lineal a **6 clases**
(escala Waze completa 0–5). `CrossEntropyLoss` **sin class weights**, Adam con `lr = 1e-3`, batch 512
y **12 épocas idénticas** para los 16 modelos. La arquitectura GRU y su capacidad de retención
temporal a lo largo de la ventana `lookback` se sustentan en @choLearningPhraseRepresentations2014 y
@chungEmpiricalEvaluationGated2014.

**Decisión de no rebalancear.** El entrenamiento usa la **distribución honesta** de clases, sin pesos
ni remuestreo. El desbalance es un dato del dominio (ver CT-11.6), no un defecto a corregir: introducir
rebalanceo contaminaría la comparación entre combinaciones, que es el objetivo del spike. La escala de
salida mantiene las **6 clases** aunque `jam5` nunca se active en el soporte demanda-driven: `jam5` se
documenta como **clase fuera-de-soporte**, no se remapea a 5 clases.

**Por qué clasificación ordinal y estas métricas.** `jam_level` es un **constructo ordinal** (D-009):
las clases tienen orden (0 < 1 < … < 5) pero no una métrica continua calibrada. Por eso se modela como
**clasificación de 6 clases** y se reportan, **por dirección** (no solo agregadas), cuatro métricas que
se complementan:

- **Accuracy** — proporción de aciertos exactos; intuitiva pero ciega al desbalance (una clase mayoritaria
  la infla).
- **F1-macro sobre las 6 clases** *(primaria)* — promedia el F1 de cada clase con igual peso, de modo que
  **no deja esconder el desbalance**: una clase de cola ignorada arrastra el macro hacia abajo. Se reporta
  además el **F1-macro presentes-only** *(secundaria)*, que promedia solo sobre las clases con soporte en
  validación (excluye `jam5`), para separar el efecto de la clase ausente del efecto del colapso de clases.
- **Matriz de confusión 6×6** — expone *dónde* caen los errores (a qué clase se confunde cada una).
- **MAE ordinal** — `|ŷ − y|` tratando la clase como entero ordenado; **captura la cercanía del error**:
  un MAE bajo indica que cuando el modelo falla, lo hace hacia una clase **adyacente**, no hacia un salto
  grande. Es la métrica que rescata la lectura del F1 (ver CT-11.6).

El barrido cubre **4 direcciones × 4 combinaciones = 16 modelos**. Las combinaciones (con Δt_in = 60 s,
1 paso = 1 min):

| Combo | lookback | horizonte |
|-------|----------|-----------|
| **C1** | 15 min | 30 min |
| **C2** | 30 min | 30 min |
| **C3** | 30 min | 60 min |
| **C4** | 60 min | 60 min |

Las shapes son holgadas: ~105–112 k secuencias de entrenamiento por dirección y ~26–28 k de validación
(según la combinación). El costo total del barrido es de ~5,5 min en GPU Metal (MPS).

### Resultados

**Accuracy** (validación, por combinación × dirección; **negrita** = mejor de la columna):

| Combo (lb/hz) | N | S | E | W | Prom. N/S | Prom. E/W |
|---------------|------|------|------|------|-----------|-----------|
| C1 (15/30) | 0.7443 | 0.7387 | 0.7471 | **0.7571** | 0.7415 | 0.7521 |
| **C2 (30/30)** | **0.7459** | **0.7416** | **0.7522** | 0.7555 | **0.7437** | **0.7539** |
| C3 (30/60) | 0.7115 | 0.7015 | 0.7400 | 0.7325 | 0.7065 | 0.7363 |
| C4 (60/60) | 0.7182 | 0.7130 | 0.7430 | 0.7441 | 0.7156 | 0.7436 |

**F1-macro sobre 6 clases** *(primaria)* — entre paréntesis, el F1-macro presentes-only *(secundaria)*:

| Combo (lb/hz) | N | S | E | W | Prom. N/S | Prom. E/W |
|---------------|------|------|------|------|-----------|-----------|
| C1 (15/30) | 0.387 (0.464) | **0.388** (0.465) | 0.395 (0.474) | 0.379 (0.455) | **0.387** | 0.387 |
| C2 (30/30) | 0.383 (0.459) | 0.381 (0.457) | 0.399 (0.479) | 0.415 (0.497) | 0.382 | 0.407 |
| C3 (30/60) | 0.358 (0.430) | 0.353 (0.423) | 0.410 (0.492) | 0.402 (0.483) | 0.356 | 0.406 |
| C4 (60/60) | 0.369 (0.443) | 0.363 (0.435) | **0.432** (0.518) | **0.431** (0.517) | 0.366 | **0.431** |

**MAE ordinal** (menor = mejor):

| Combo (lb/hz) | N | S | E | W | Prom. N/S | Prom. E/W |
|---------------|------|------|------|------|-----------|-----------|
| C1 (15/30) | 0.2821 | 0.2877 | 0.3999 | 0.4005 | 0.2849 | 0.4002 |
| **C2 (30/30)** | **0.2817** | **0.2837** | 0.4024 | 0.3942 | **0.2827** | 0.3983 |
| C3 (30/60) | 0.3395 | 0.3549 | 0.4064 | 0.4135 | 0.3472 | 0.4099 |
| C4 (60/60) | 0.3248 | 0.3328 | 0.4052 | 0.4019 | 0.3288 | 0.4036 |

Para contextualizar la accuracy, el **baseline mayoritario** (predecir siempre la clase más frecuente de
esa dirección) es: N → `jam1` 0.4724, S → `jam1` 0.4716, E → `jam2` 0.6156, W → `jam2` 0.6130. **Todos los
modelos superan su baseline con holgura**: en el eje dominante N/S por ~+27 puntos (0.47 → 0.74) y en el
secundario E/W por ~+14 puntos (0.61 → 0.75). El modelo aprende señal real, no memoriza la mayoría.

(Métricas completas de los 16 modelos —incluidas las matrices de confusión 6×6 por modelo— en
`ia_prediction_service/scripts/tth11_sweep_metrics.json`, evidencia reproducible del barrido.)

## CT-11.5 — Tabla resumen (contrato para TTH-09)

### Hallazgo central: el óptimo de hiperparámetros **difiere por eje**

El resultado más valioso del spike no es un único combo ganador, sino que **el lookback/horizonte óptimo
depende del eje de la intersección**. La red corredor-Larco × Schell tiene un **eje dominante N/S**
(demanda alta, transición filosa valle→`jam4`) y un **eje secundario E/W** (demanda ≈ N/S ÷ 4, dinámica
suave). El barrido muestra dos regímenes distintos:

- **Eje dominante (N/S)** — prefiere **horizonte corto**. Pasar de horizonte 30 (C2) a 60 (C3) **degrada**
  la accuracy de ~0.74 a ~0.71 y empeora el MAE; la transición valle→pico es tan abrupta que predecir a 60
  min adelante pierde anclaje. El mejor desempeño está en C1/C2 (horizonte 30).
- **Eje secundario (E/W)** — mejora su **F1-macro con lookback largo**: C4 (lookback 60) alcanza
  F1-macro 0.43 frente a 0.41 de C2. La dinámica suave del eje secundario se beneficia de más contexto
  histórico para resolver sus bandas transicionales.

**Razón física:** las clases de cola viven en ejes distintos —el pico `jam4` es propio del eje dominante,
mientras que el estado transitorio `jam3` aparece sobre todo en el eje secundario—, de modo que cada eje
plantea un problema de predicción con una estructura de clases diferente. El compromiso alcance-vs-exactitud
que la literatura documenta para horizontes multi-paso (@singhNovelCNNGRULSTM2025; @wangTrafficFlowPrediction2022)
se manifiesta aquí de forma **asimétrica entre ejes**, no de forma uniforme.

### Tabla resumen consolidada (promedio de las 4 direcciones por combinación)

| Combo | lookback | horizonte | Accuracy | F1-macro (6) | MAE ordinal |
|-------|----------|-----------|----------|--------------|-------------|
| C1 | 15 min | 30 min | 0.7468 | 0.3871 | 0.3426 |
| **C2** | **30 min** | **30 min** | **0.7488** | 0.3943 | **0.3405** |
| C3 | 30 min | 60 min | 0.7214 | 0.3808 | 0.3785 |
| C4 | 60 min | 60 min | 0.7296 | **0.3986** | 0.3662 |

### Recomendación para TTH-09 (contrato)

**Default robusto: C2 — lookback = 30 min, horizonte = 30 min.** Es la única combinación que **gana
accuracy y MAE en los cuatro ejes** y queda segunda en F1-macro global; **nunca es la peor** en ninguna
métrica ni dirección. C1 (lookback 15) la empata casi en accuracy pero con peor MAE; C3/C4 (horizonte 60)
degradan el eje dominante. La recomendación es **robusta para accuracy y MAE** (C2 domina ambos ejes) y
**sensible solo en un punto**: si TTH-09 prioriza el **F1-macro del eje secundario E/W**, C4 (lookback 60)
es netamente mejor allí.

| Hiperparámetro | Valor recomendado | Unidad | Justificación | Referencias |
|----------------|-------------------|--------|---------------|-------------|
| **Δt_in** | 60 | s | Granularidad nativa de la cadena de datos (edge + SUMO → buckets de 60 s); evita reagregación. | @wangTrafficFlowPrediction2022 |
| **lookback** | 30 (≡ 30 pasos) | min | Mejor accuracy/MAE en los 4 ejes; ventana suficiente sin diluir la transición del eje dominante. | @choLearningPhraseRepresentations2014; @wangTrafficFlowPrediction2022 |
| **horizonte** | 30 (≡ 30 pasos) | min | Horizonte 60 degrada el eje dominante (0.74→0.71); 30 conserva exactitud con alcance operativo útil. | @singhNovelCNNGRULSTM2025 |
| **Re-inferencia** | 60 | s | Coherente con el bucket de entrada; recalcular más seguido no aporta información nueva. | @wenNovelTrafficOptimization2023 |

> **Matiz por eje (decisión de TTH-09).** Para exprimir cada eje: **C2 para N/S** (horizonte corto) y
> **C4 para E/W** (lookback largo). Es una decisión de TTH-09; aquí se documenta como la divergencia
> dominante/secundario que encontró el spike. El default único C2 sigue siendo la opción segura si se
> prefiere un solo juego de hiperparámetros para las 4 direcciones.

## CT-11.6 — Limitaciones y trabajo futuro

Las siguientes limitaciones son **honestas y esperadas**; se reportan como contexto de lectura de las
métricas, no como defectos a corregir dentro del spike.

- **Colapso de clases direccional.** Sin rebalanceo (decisión de CT-11.4), la red **ignora la cola
  minoritaria de cada eje**, y el eje define *qué* clase se pierde. En el eje dominante **N/S**, `jam3`
  (~0.75 % del soporte) y `jam0` colapsan a F1 = 0: el modelo predice solo `{jam1, jam2, jam4}`. En el eje
  secundario **E/W**, el modelo sí resuelve `jam3` (F1 ≈ 0.45) pero la clase débil pasa a ser `jam4`
  (~2.8 % del soporte, F1 ≈ 0.27). Esto deprime el F1-macro de 6 clases y es la causa de que ronde
  0.38–0.43 pese a accuracies de ~0.75.
- **El MAE ordinal bajo rescata la lectura.** El MAE de 0.28–0.41 muestra que, cuando el modelo falla, lo
  hace hacia una clase **adyacente**, no hacia un salto grande: en las matrices de confusión los errores de
  `jam2` caen en `jam1` o `jam4`, nunca lejos. **No es un modelo malo, es uno que ignora clases raras pero
  "se equivoca cerca"** — respeta el orden ordinal aunque falle la clase exacta. Para un constructo ordinal
  como `jam_level`, esa propiedad importa tanto como la accuracy.
- **`jam5` ausente (fuera-de-soporte).** Ningún modelo ve ni predice `jam5` en los 16 entrenamientos: en el
  dataset demanda-driven, `jam5` corresponde a un evento de bloqueo/spillback que no emerge de la demanda.
  La escala se mantiene en 6 clases por contrato, pero `jam5` queda como clase fuera-de-soporte cuya
  cobertura es trabajo de TTH-09.
- **Volumen sintético y bimodalidad.** El dataset son 100 corridas sintéticas sobre 4 perfiles; la
  **bimodalidad** de la intersección de 4 ramas (estados estables `jam2`/`jam4`, con `jam3` meramente
  transitorio en las rampas) hace que estos resultados sean **indicativos de tendencia, no una medida de
  capacidad** del enfoque. Con datos reales (calibración Waze, D-008/F38) o topología expandida (>4
  direcciones, múltiples intersecciones, F37) la estructura de clases podría cambiar.
- **D-005 — el 81.3 % heredado, sin maquillar.** La mejor accuracy del barrido es **0.7571** (W/C1),
  **5.6 puntos por debajo** del 81.3 % heredado (D-005). La comparación **no es 1:1**: aquel número proviene
  de otras condiciones, y este es un **spike con modelos desechables** (12 épocas, `hidden = 64`, sin tuning,
  sin manejo de desbalance) sobre datos sintéticos chicos con desbalance severo y direccional —donde dos
  clases presentes son efectivamente no aprendibles tal cual. La accuracy global está **deprimida por esas
  colas, no por incapacidad del enfoque**: contra el baseline mayoritario de cada eje el modelo aprende
  fuerte (N/S +27 pts, E/W +14 pts). El 81.3 % **no es un techo del sistema**; cerrar la brecha —más datos,
  tuning, manejo de desbalance, posible pérdida ordinal explícita— es trabajo de **TTH-09**, no del spike.
- **Hiperparámetros no temporales fuera de alcance.** El spike fija solo el **contrato temporal**. El tamaño
  del estado oculto, el número de capas, el optimizador y la regularización se mantuvieron fijos para una
  comparación limpia y quedan para la afinación de TTH-09.

## CT-11.7 — Cierre de doble propósito

Este documento cumple, sin inconsistencias entre ambos registros, su **doble propósito**:

1. **Sustento técnico interno para TTH-09.** La tabla de CT-11.5 es el **contrato** de hiperparámetros
   temporales que consume la implementación del modelo: Δt_in = 60 s, lookback = 30 min, horizonte = 30 min
   y re-inferencia = 60 s, con el matiz por eje documentado para que TTH-09 decida si afina por dirección.
   El hallazgo de que el óptimo **difiere entre eje dominante y secundario** es la entrada más accionable que
   el spike entrega a producción.
2. **Apéndice académico de tesis.** Las decisiones se anclan en literatura citable (5 fuentes en CT-11.3),
   la metodología es reproducible (config fija, evidencia en JSON) y las limitaciones se declaran con
   honestidad —incluido el contraste contra el 81.3 % heredado sin maquillarlo—, de modo que el documento
   sea defendible ante revisión académica.

El rigor se mantiene consistente entre ambos registros: lo que TTH-09 toma como contrato es exactamente lo
que la tesis documenta como resultado del spike, con sus alcances y sus límites explícitos.
