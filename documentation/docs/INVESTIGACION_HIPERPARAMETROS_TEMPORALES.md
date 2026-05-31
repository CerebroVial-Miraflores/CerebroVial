# Investigación de hiperparámetros temporales del modelo predictivo (TTH-11)

> **Documento parcial.** Cubre la parte académica del spike TTH-11: definición de los
> cuatro hiperparámetros temporales, revisión bibliográfica (5 fuentes) y recomendaciones
> preliminares basadas en literatura y en los valores por defecto declarados.
>
> **Alcance cubierto en esta versión:** CT-11.1 (estructura y propósito), CT-11.2 (los cuatro
> hiperparámetros con definición, bibliografía, rango candidato, implicancias y recomendación
> preliminar), CT-11.3 (revisión bibliográfica de ≥5 fuentes) y CT-11.8 (nota de cierre de Δt_in
> que materializa la dependencia con TTH-07).
>
> **Pendiente (próxima tarea — exploración empírica con entrenamiento):** CT-11.4 (≥3 combinaciones
> entrenadas con 4 métricas), CT-11.5 (tabla resumen / contrato para TTH-09), CT-11.6 (limitaciones
> y trabajo futuro) y CT-11.7 (cierre de doble propósito). Estas secciones quedan como *placeholders*
> explícitos más abajo y **no** deben completarse hasta disponer de los resultados de entrenamiento.

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
- **CT-11.4** — Exploración empírica *(placeholder — próxima tarea)*.
- **CT-11.5** — Tabla resumen / contrato para TTH-09 *(placeholder — próxima tarea)*.
- **CT-11.6** — Limitaciones y trabajo futuro *(placeholder — próxima tarea)*.
- **CT-11.7** — Cierre de doble propósito *(placeholder — próxima tarea)*.

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

`[PENDIENTE TAREA 3: exploración empírica]`

Se entrenarán **al menos 3 combinaciones** de hiperparámetros temporales (combinaciones razonadas,
no aleatorias) con modelos completos, reportando **4 métricas**: MAE y RMSE sobre la variable
continua, y exactitud (accuracy) más matriz de confusión sobre el `jam_level` discretizado (0–5).
No completar hasta disponer de los resultados de entrenamiento.

## CT-11.5 — Tabla resumen (contrato para TTH-09)

`[PENDIENTE TAREA 3: tabla resumen]`

Tabla final con, por hiperparámetro: valor recomendado, unidad, justificación breve (1–2 líneas) y
referencias bibliográficas que respaldan el valor. Esta tabla es el "contrato" que consume TTH-09.

## CT-11.6 — Limitaciones y trabajo futuro

`[PENDIENTE TAREA 3: limitaciones]`

Cubrirá: (a) qué podría cambiar con datos reales de Waze (D-008, F38); (b) qué ocurre si la
topología se expande más allá de 4 direcciones o a múltiples intersecciones (F37, notas de TTH-09);
y (c) los hiperparámetros **no temporales** fuera de alcance (tamaño del estado oculto, número de
capas, optimizador, regularización).

## CT-11.7 — Cierre de doble propósito

`[PENDIENTE TAREA 3: cierre de doble propósito]`

Nota final que confirma la consistencia del documento como sustento interno para TTH-09 y como
apéndice académico de tesis, sin inconsistencias entre ambos registros.
