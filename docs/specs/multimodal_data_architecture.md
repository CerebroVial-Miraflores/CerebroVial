# Arquitectura de Datos Multimodal para la Predicción Espacio-Temporal de Congestión Vehicular: Especificación Exhaustiva e Integración de Telemetría Waze y Visión Computacional

## 1. Introducción: El Paradigma de la Predicción Espacio-Temporal en Sistemas de Transporte Inteligente

La gestión moderna de la movilidad urbana ha transitado desde enfoques reactivos hacia modelos predictivos que anticipan la formación de cuellos de botella antes de que alcancen niveles críticos de saturación. En este contexto, la construcción de un dataset robusto para el entrenamiento de modelos de Inteligencia Artificial, particularmente aquellos basados en Redes Neuronales de Grafos (GNN) y arquitecturas convolucionales espacio-temporales (ST-GCN), constituye el cimiento fundamental de cualquier estrategia de predicción exitosa. La literatura contemporánea y la práctica en ingeniería de datos de tráfico sugieren que los modelos univariados, que dependen únicamente de la historia temporal de un sensor aislado, son insuficientes para capturar la dinámica compleja de la congestión urbana, donde el estado de una intersección está intrínsecamente acoplado al estado de sus vecinas espaciales.

Este reporte técnico desarrolla una especificación exhaustiva para la construcción de un dataset "raw" (crudo) ideal, diseñado específicamente para alimentar modelos que integren componentes espaciales y temporales. La arquitectura de datos propuesta fusiona dos fuentes de información disjuntas pero complementarias: la inteligencia colectiva y telemetría pasiva proporcionada por el programa Waze for Cities (anteriormente Connected Citizens Program - CCP) y la granularidad microscópica obtenida mediante sensores de visión computacional procesados con algoritmos de detección de objetos como YOLO (You Only Look Once).

El objetivo central de esta especificación es preservar la integridad de los datos en su forma más pura, permitiendo que las arquitecturas de aprendizaje profundo infieran relaciones latentes durante el entrenamiento, en lugar de imponer sesgos a través de un preprocesamiento destructivo prematuro. Se abordará la anatomía completa de los feeds de datos, la taxonomía de los campos disponibles, su relevancia teórica para el modelado de grafos de tráfico y las metodologías para la conflación espacio-temporal de fuentes heterogéneas.

## 2. Anatomía del Ecosistema de Datos Waze (Waze for Cities Data Feed)

El programa Waze for Cities ofrece a las entidades asociadas un acceso privilegiado a datos en tiempo real generados por millones de usuarios activos. A diferencia de los sensores inductivos tradicionales que miden flujo en un punto fijo, Waze proporciona datos lagrangianos, es decir, mediciones basadas en la trayectoria de vehículos sonda que se mueven a través de la red. La entrega de estos datos se realiza típicamente mediante un feed JSON o XML, actualizado con una frecuencia de dos minutos, lo cual define la resolución temporal base del componente macroscópico del dataset.

Para construir un modelo predictivo que entienda la causalidad y la propagación de la congestión, es imperativo capturar y almacenar la totalidad de la información contenida en tres objetos JSON primarios: Jams (Atascos), Alerts (Alertas) e Irregularities (Irregularidades). A continuación, se detalla la estructura profunda de cada objeto y la relevancia crítica de sus campos para el entrenamiento del modelo.

### 2.1 El Objeto "Jams": La Variable Objetivo y el Estado de la Red

El objeto jams constituye el núcleo del dataset para la predicción de congestión. Representa segmentos de la red vial donde la velocidad de flujo ha caído por debajo de un umbral significativo en comparación con la velocidad de flujo libre o la velocidad histórica promedio. Desde la perspectiva del modelado, los campos contenidos en este objeto actúan tanto como features (variables de entrada para instantes pasados $t-n$) como targets (variables objetivo para instantes futuros $t+n$).

#### 2.1.1 Identificación y Persistencia Temporal

*   **uuid (Universal Unique Identifier):** Este campo alfanumérico es la clave primaria lógica para el seguimiento de eventos. En un dataset de entrenamiento de series temporales, la persistencia del uuid es vital. Un error común en la ingeniería de datos de tráfico es tratar cada snapshot del feed de Waze como eventos independientes. El uuid permite encadenar observaciones a lo largo de múltiples descargas del feed, permitiendo al modelo aprender la duración y la tasa de disipación de un atasco. Si un atasco persiste durante 40 minutos, aparecerá en 20 archivos JSON consecutivos con el mismo uuid pero con métricas (speed, length) evolutivas.
*   **pubMillis (Timestamp de Publicación):** Representado como un entero largo (milisegundos desde la época Unix), este campo define el eje temporal del dataset. Es fundamental para la sincronización con los datos de visión computacional. Dado que Waze agrega datos con cierta latencia, el pubMillis no representa el instante exacto de la medición del GPS, sino el momento de la agregación en el servidor. El modelo debe ser entrenado para tolerar esta ligera varianza, utilizando ventanas de tiempo (e.g., bins de 5 minutos) en lugar de emparejamiento exacto al milisegundo.

#### 2.1.2 Métricas de Estado del Tráfico (Variables Cuantitativas)

*   **speed (Velocidad Actual):** Expresada en metros por segundo (m/s), es la medida directa de la fricción en el arco del grafo. Para el entrenamiento de GNNs, la velocidad nodal o de arista suele ser la característica principal de entrada. Es crucial almacenar el valor en m/s (precisión nativa) y no solo la versión convertida a km/h (speedKMH), para evitar errores de redondeo acumulativos.
*   **level (Nivel de Severidad):** Un valor entero discreto del 0 al 5 que categoriza la intensidad del atasco.
    *   *Importancia para el Modelo:* Mientras que speed es una variable continua, level es una variable ordinal que encapsula una normalización implícita respecto al tipo de vía. Una velocidad de 20 km/h puede ser level 1 en una calle residencial pero level 4 en una autopista. Incluir este campo permite al modelo aprender la relatividad de la congestión sin necesidad de que la red neuronal "memorice" los límites de velocidad de cada segmento vial individualmente.
*   **delay (Retraso Absoluto):** Medido en segundos, representa el tiempo adicional que un vehículo tarda en cruzar el segmento comparado con condiciones de flujo libre.
    *   *Importancia Crítica:* En funciones de pérdida (loss functions) personalizadas para el entrenamiento del modelo, el delay debe usarse como factor de ponderación. Un error de predicción en un segmento con delay: 1200 (20 minutos de retraso) debe ser penalizado mucho más severamente que un error en un segmento con delay: 10. Un valor de -1 indica un bloqueo total o cierre, lo cual debe ser tratado como una bandera de estado especial o un valor atípico (outlier) en el preprocesamiento numérico.
*   **length (Longitud de Cola):** La longitud del atasco en metros. Esta variable es esencial para modelar el fenómeno de spillback (cuando la cola de un atasco crece hasta bloquear la intersección anterior). En un modelo espacio-temporal, la correlación entre el incremento de length en la arista $E_i$ y la caída abrupta de speed en la arista anterior $E_{i-1}$ es el mecanismo causal que la GNN debe aprender.

#### 2.1.3 Geometría y Topología (Variables Espaciales)

*   **line (Polilínea):** Una lista de objetos de coordenadas `{x: Longitude, y: Latitude}`.
    *   *Función en el Dataset:* A diferencia de sistemas que solo reportan un punto central, Waze entrega la geometría completa del segmento afectado. Para construir el dataset "raw" ideal, se debe almacenar esta lista completa (frecuentemente serializada como GeoJSON o WKT). Esta geometría es la única verdad terreno que permite realizar el Map Matching hacia el grafo base de la ciudad. Sin el campo line, es imposible asignar correctamente la congestión a las aristas direccionales específicas de una intersección compleja.
*   **roadType (Clase Funcional de la Vía):** Un entero que indica la jerarquía de la vía. Según la documentación técnica de Waze, los valores clave incluyen:
    *   1: Streets (Calles urbanas)
    *   2: Primary Street (Avenidas principales)
    *   3: Freeways (Autopistas)
    *   4: Ramps (Rampas de acceso/salida)
    *   6: Primary (Carreteras primarias)
    *   7: Secondary (Carreteras secundarias)
    *   *Relevancia:* Esta es una variable categórica estática fundamental. La dinámica de flujo es radicalmente distinta entre una autopista (flujo continuo) y una calle (flujo interrumpido). El modelo predictivo utilizará esto para segregar pesos o utilizar mecanismos de atención diferenciados según el tipo de vía.
*   **turnType (Tipo de Giro):** Describe si el atasco está asociado a una maniobra específica (ej. Left, Right, Exit).
    *   *Relevancia:* Identifica cuellos de botella específicos en intersecciones. Un atasco etiquetado con turnType: Left sugiere que el carril de giro es el problema, no necesariamente los carriles pasantes. Esto añade una resolución sub-segmento al dataset.

### 2.2 El Objeto "Alerts": Variables Explicativas y Perturbaciones Exógenas

Mientras que los jams son el síntoma, las alerts suelen describir la causa. Son reportes activos generados por usuarios ("Wazers") sobre condiciones en la vía. En el contexto de la predicción, estas actúan como variables exógenas o choques estocásticos al sistema. Un modelo robusto debe ingerir estas alertas para explicar cambios repentinos en la velocidad que no responden a patrones cíclicos diarios.

#### 2.2.1 Taxonomía de Incidentes

El campo type y su refinamiento subtype son las características categóricas más ricas del dataset. Es imperativo almacenar ambos niveles jerárquicos.
*   **ACCIDENT:** Subtipos como ACCIDENT_MINOR vs. ACCIDENT_MAJOR permiten al modelo estimar la gravedad del impacto en la capacidad de la vía. Un accidente mayor implica una reducción drástica de capacidad y un tiempo de despeje (clearance time) largo.
*   **HAZARD:** Incluye HAZARD_ON_ROAD (objeto en la vía), HAZARD_ON_SHOULDER (vehículo en el arcén) y HAZARD_WEATHER.
    *   *Insight Meteorológico:* Los subtipos HAZARD_WEATHER_HEAVY_RAIN, HAZARD_WEATHER_FLOOD o HAZARD_WEATHER_FOG actúan como sensores meteorológicos hiperlocales. En lugar de depender de una estación meteorológica general para toda la ciudad, Waze proporciona puntos específicos donde el clima está afectando la conducción. El dataset debe preservar estos subtipos como features ambientales críticas.
*   **ROAD_CLOSED:** Este tipo de evento es estructuralmente diferente a los demás, ya que implica una modificación temporal de la topología del grafo (la arista "desaparece" o su peso se vuelve infinito). El dataset debe etiquetar estos eventos claramente para que, durante el entrenamiento, el modelo aprenda a redirigir el flujo virtualmente.

#### 2.2.2 Metadatos de Calidad y Confianza

Dado que Waze es una plataforma de crowdsourcing, la veracidad de los datos varía. El dataset raw debe incluir los metadatos de calidad para permitir el filtrado o ponderación durante el entrenamiento.
*   **reliability (Confiabilidad):** Un valor entero (1-10) calculado por Waze basado en la reputación del usuario informante y corroboraciones posteriores.
*   **confidence (Confianza):** Un valor entero (0-5) que indica la certeza del sistema sobre la existencia del evento.
*   **nThumbsUp:** Número de confirmaciones por otros usuarios.
    *   *Estrategia para el Dataset:* No se debe filtrar datos con baja confiabilidad en la etapa raw. El modelo predictivo puede aprender a ignorar ruido si se le proporciona la reliability como una feature de entrada. Por ejemplo, una red neuronal podría aprender que ACCIDENT + reliability: 2 tiene poco impacto predictivo, mientras que ACCIDENT + reliability: 10 garantiza un atasco futuro.

#### 2.2.3 Direccionalidad del Reporte

*   **magvar (Magnetic Variation / Heading):** Indica el rumbo del usuario al momento del reporte (grados 0-359).
    *   *Relevancia:* Es crucial para asignar el incidente al lado correcto de la vía. En una avenida de doble sentido, un accidente en rumbo 0° (Norte) no bloquea físicamente el flujo en rumbo 180° (Sur), aunque pueda causar ralentización por "efecto mirón". El dataset debe permitir esta distinción espacial precisa.

### 2.3 El Objeto "Irregularities": Detección de Anomalías

Este feed contiene eventos de tráfico que desvían significativamente del patrón histórico para ese día de la semana y hora.
*   **regularSpeed vs. speed:** La comparación entre la velocidad histórica almacenada por Waze (regularSpeed) y la actual permite calcular el residual de congestión. Incluir esto en el dataset ahorra la necesidad de construir una línea base histórica masiva desde cero, ya que Waze ya proporciona la "expectativa" normativa.
*   **trend:** Un valor numérico que indica si la congestión está creciendo o decreciendo. Es una derivada de primer orden pre-calculada sumamente valiosa para la predicción a corto plazo (short-term forecasting).

## 3. Anatomía de los Datos de Visión Computacional (YOLO + Cámaras)

Para complementar la visión macroscópica de Waze, se integra una capa de datos microscópica obtenida mediante cámaras. Waze informa que el tráfico es lento; las cámaras explican por qué (volumen excesivo, bloqueo por camión, etc.) y proporcionan el conteo exacto de flujo que Waze solo puede estimar indirectamente. Se asume el uso de modelos de detección de objetos de la familia YOLO (e.g., YOLOv8, YOLOv11) aplicados sobre flujos de video de cámaras fijas en intersecciones.

### 3.1 El Output "Raw" de Inferencia (Nivel Frame)

El dataset ideal no almacena video (ineficiente y costoso), sino los metadatos de inferencia generados cuadro a cuadro.

#### 3.1.1 Identificación y Sincronización

*   **camera_id:** Identificador único del sensor. Vincula los datos de píxeles con la ubicación geoespacial definida en la tabla de metadatos de sensores.
*   **frame_timestamp:** Marca de tiempo de alta precisión (ms) de la captura del cuadro. Es la clave de unión (join key) con el campo pubMillis de Waze.
*   **frame_seq:** Número secuencial del cuadro, vital para calcular deltas temporales entre frames consecutivos (necesario para estimación de velocidad visual).

#### 3.1.2 Detección y Clasificación (Bounding Boxes)

Para cada objeto detectado en un frame, YOLO genera un vector de datos que debe ser almacenado:
*   **class_id:** Entero que representa la clase del objeto según el dataset de entrenamiento (usualmente COCO Dataset para modelos pre-entrenados).
    *   *Clases Relevantes:* El dataset debe preservar la distinción entre clases vehiculares. Según el estándar COCO: 2: car, 3: motorcycle, 5: bus, 7: truck.
    *   *Importancia:* La composición del tráfico es fundamental. Un flujo de 10 autobuses satura una intersección mucho más rápido que 10 coches. Esta heterogeneidad es invisible para los sensores inductivos simples y a menudo para Waze, pero explícita para YOLO.
*   **bounding_box:** Cuarteto de valores `[x_center, y_center, width, height]`, típicamente normalizados entre 0 y 1 relativos a las dimensiones de la imagen.
    *   *Importancia:* Permite calcular la Ocupación Espacial Visual. Más allá del simple conteo, la suma de las áreas de las bounding boxes en una Región de Interés (ROI) proporciona una métrica directa de la densidad física en la intersección.
*   **confidence:** Probabilidad de la detección (0.0 - 1.0). Almacenar este valor permite ajustar el umbral de sensibilidad durante la etapa de modelado sin necesidad de re-procesar el video.

#### 3.1.3 Seguimiento de Objetos (Object Tracking)

La detección simple cuenta objetos por frame, pero no cuenta flujo (vehículos únicos pasando un punto). Para un dataset de predicción de tráfico, es esencial aplicar un algoritmo de Multi-Object Tracking (MOT) como DeepSORT o ByteTrack sobre las detecciones de YOLO.
*   **track_id (o object_id):** Un identificador entero que persiste para el mismo vehículo a través de múltiples frames mientras permanece en el campo de visión.
    *   *Utilidad Crítica:* Permite calcular trayectorias y vectores de desplazamiento. Al conocer la posición del track_id: 105 en el tiempo $t$ y en $t+\Delta$, se puede estimar su velocidad y dirección de giro (izquierda, derecha, recto), enriqueciendo la topología de flujo en la intersección.

### 3.2 Metadatos de Calibración: La Matriz de Homografía

Los datos de YOLO son nativamente bidimensionales (espacio de imagen: píxeles). Para integrarlos con Waze (espacio geográfico: latitud/longitud), el dataset debe incluir una tabla estática de calibración para cada cámara.
*   **Matriz de Homografía ($H$):** Una matriz de $3 \times 3$ que mapea coordenadas planas de la imagen a coordenadas del plano terrestre (Bird's Eye View o coordenadas GPS locales).
*   **Ecuación de Transformación:** $P_{world} = H \cdot P_{image}$.
*   **Función:** Permite saber que la bounding box en los píxeles `` corresponde a la coordenada GPS [-34.60, -58.38], facilitando la asignación espacial del vehículo a un carril específico o segmento de Waze.

## 4. Estrategia de Integración y Construcción del Grafo Espacio-Temporal

La premisa central de su solicitud es modelar cómo las intersecciones se influyen mutuamente. Esto requiere estructurar el dataset no como tablas planas independientes, sino como un grafo donde la información fluye a través de una Matriz de Adyacencia.

### 4.1 Definición Topológica del Grafo ($G$)

El dataset debe contener una definición estática de la red vial, basada fundamentalmente en la topología física (obtenida de OpenStreetMap u otra fuente cartográfica) y enriquecida con los sensores.
*   **Tabla de Nodos ($V$):** Representan las intersecciones y puntos de decisión.
    *   `node_id`: Identificador único.
    *   `lat`, `lon`: Ubicación geográfica.
    *   `has_camera`: Booleano que indica si este nodo dispone de datos de visión computacional.
*   **Tabla de Aristas ($E$):** Representan los segmentos viales que conectan los nodos. Es aquí donde ocurre la fusión de datos.
    *   `edge_id`: Identificador único direccional (del Nodo A al Nodo B).
    *   `length`: Distancia física en metros.
    *   `waze_segment_ids`: Lista de IDs de segmentos de Waze que se solapan con esta arista física (resultado del Map Matching).

### 4.2 La Matriz de Adyacencia Ponderada ($A$)

Para entrenar modelos como ST-GCN, el dataset debe materializar la matriz que define las relaciones espaciales.
*   **Conectividad Física:** $A_{ij} = 1$ si existe una calle directa de $i$ a $j$.
*   **Ponderación por Distancia:** $A_{ij} = \exp(-\frac{dist(i,j)^2}{\sigma^2})$. Pondera la influencia basada en la cercanía física.
*   **Ponderación por Flujo:** Si se dispone de datos históricos, $A_{ij}$ puede ponderarse por la probabilidad de que un vehículo en $i$ vaya a $j$.

El dataset ideal debe proporcionar esta matriz pre-calculada o los datos de distancias necesarios para generarla dinámicamente durante el entrenamiento.

### 4.3 Metodología de Conflación (Unión de Fuentes)

La integración de Waze y Cámaras no es trivial debido a sus diferentes naturalezas espaciales y temporales.

#### 4.3.1 Conflación Espacial (Map Matching)

El proceso para asignar datos de Waze a las aristas del grafo ($E$) debe basarse en la geometría.
*   **Algoritmo:** Utilizar algoritmos de proyección Point-to-Curve o Modelos Ocultos de Markov (HMM) para mapear la polilínea del campo line de los objetos jams de Waze a la línea central de las aristas del grafo.
*   **Buffer Espacial:** Dado el error GPS, se debe considerar un buffer (ej. 10-15 metros) alrededor de la arista del grafo para capturar eventos de Waze y alertas que no caen exactamente en la línea central.

#### 4.3.2 Sincronización Temporal (Resampling)

*   **Waze:** Frecuencia ~2 minutos (irregular).
*   **Cámaras:** Frecuencia ~33 ms (30 FPS).
*   **Estrategia para el Dataset:** Se recomienda una estrategia de Ventanas de Agregación (Time Windows). El dataset "raw" mantiene los timestamps originales, pero para el entrenamiento se define una ventana común, por ejemplo, 5 minutos.
    *   Para la ventana $T_i$: Se promedian las velocidades de Waze reportadas en ese intervalo. Se suman los conteos vehiculares únicos (track_ids) vistos por YOLO en ese intervalo.

Este enfoque alinea ambas fuentes en una serie temporal regular estructurada por grafos: $X_t \in \mathbb{R}^{N \times F}$, donde $N$ es el número de nodos/aristas y $F$ es el número de características fusionadas (velocidad Waze + flujo Cámara).

## 5. Especificación del Dataset Ideal (Schema Definition)

A continuación, se presenta la especificación técnica de las tablas que compondrán el dataset completo. Este diseño sigue principios de bases de datos analíticas (OLAP) optimizadas para la ingestión masiva y la posterior consulta para entrenamiento. Se recomienda el uso de formatos columnares como Parquet o HDF5 para el almacenamiento físico debido a su eficiencia en lectura de grandes volúmenes numéricos.

### 5.1 Tabla: WAZE_JAMS_RAW

Almacena la historia evolutiva de la congestión.

| Campo | Tipo de Dato | Origen | Descripción |
| :--- | :--- | :--- | :--- |
| `event_uuid` | String | Waze | Identificador único del evento de tráfico. |
| `snapshot_timestamp` | Timestamp | Waze | Momento de captura del feed (pubMillis). |
| `edge_id` | String | Calculado | ID de la arista del grafo tras Map Matching. |
| `waze_line_geometry` | WKT/GeoJSON | Waze | Geometría original del atasco (polilínea). |
| `speed_mps` | Float | Waze | Velocidad media en m/s. |
| `delay_seconds` | Integer | Waze | Retraso respecto a flujo libre. |
| `congestion_level` | Integer | Waze | Nivel 0-5. |
| `jam_length_m` | Integer | Waze | Longitud física del atasco. |
| `road_type` | Integer | Waze | ID del tipo de vía. |
| `turn_type` | String | Waze | Contexto de giro (Left, Right, etc). |

### 5.2 Tabla: WAZE_ALERTS_RAW

Almacena eventos puntuales y perturbaciones.

| Campo | Tipo de Dato | Origen | Descripción |
| :--- | :--- | :--- | :--- |
| `alert_uuid` | String | Waze | ID único de la alerta. |
| `timestamp` | Timestamp | Waze | pubMillis. |
| `edge_id` | String | Calculado | ID de la arista más cercana. |
| `alert_type` | String | Waze | Categoría mayor (ACCIDENT, HAZARD). |
| `alert_subtype` | String | Waze | Categoría detallada. |
| `reliability` | Integer | Waze | Score 1-10. |
| `confidence` | Integer | Waze | Score 0-5. |
| `magvar` | Integer | Waze | Rumbo en grados. |
| `report_location` | Point (WKT) | Waze | Coordenada original. |

### 5.3 Tabla: VISION_TRACKS_RAW

Almacena el movimiento vehicular detectado. (Nivel de agregación recomendado: Track/Vehículo, no Frame/Caja, para optimizar espacio sin perder información de flujo).

| Campo | Tipo de Dato | Origen | Descripción |
| :--- | :--- | :--- | :--- |
| `track_uuid` | String | YOLO+DeepSORT | ID único del vehículo en la sesión. |
| `camera_id` | String | Config | Sensor que capturó el objeto. |
| `class_id` | Integer | YOLO | Tipo de vehículo (Car, Bus, Truck). |
| `entry_timestamp` | Timestamp | Sistema | Hora de primera detección. |
| `exit_timestamp` | Timestamp | Sistema | Hora de última detección. |
| `trajectory_wkt` | LineString | Calculado | Trayectoria proyectada (GPS) tras homografía. |
| `avg_speed_px` | Float | Calculado | Velocidad visual media en píxeles/frame. |
| `direction_vector` | String | Calculado | Vector de movimiento (ej. "North-to-East"). |

### 5.4 Tabla: GRAPH_TOPOLOGY_STATIC

Define la estructura espacial para las GNN.

| Campo | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `edge_id` | String | Clave primaria. |
| `source_node` | String | ID Nodo Origen. |
| `target_node` | String | ID Nodo Destino. |
| `distance_m` | Float | Peso de distancia. |
| `lanes` | Integer | Capacidad teórica (número de carriles). |
| `adjacency_list` | Array | Lista de edge_ids que impactan a este segmento (upstream/downstream). |
