# Comparación de modelos para la predicción de demora vehicular: GRU frente a STGNN

## 1. El problema que se quiere resolver

El sistema necesita anticipar la congestión de la red vial de Miraflores antes de que ocurra. Concretamente, para cada tramo de calle (cada arista de la red, 1660 en total) se quiere predecir cuánta demora acumulará el tránsito en los próximos minutos. La variable que se predice es el *timeLoss*: el tiempo total que los vehículos pierden en un tramo respecto a circular sin obstáculos, medido en intervalos de 60 segundos. Predecir esta demora con anticipación es lo que permite que el mapa muestre no solo cómo está el tránsito ahora, sino cómo estará en los siguientes 5 a 30 minutos.

Para esa tarea se evaluaron dos modelos de predicción, y este documento explica por qué uno de ellos resultó mejor, con qué evidencia, y qué significa cada número de la comparación.

## 2. Los dos modelos comparados

El primer modelo es una **GRU** (*Gated Recurrent Unit*), un tipo de red neuronal diseñada para datos que evolucionan en el tiempo. La GRU mira la historia reciente de cada tramo y, a partir de ese patrón temporal, proyecta hacia adelante. Es un modelo sólido y bien establecido, pero tiene una limitación: trata a cada tramo de forma relativamente aislada, sin aprovechar de manera explícita cómo se influyen entre sí los tramos vecinos. Funciona como la línea base, el punto de referencia contra el cual se mide cualquier alternativa más compleja.

El segundo modelo es un **STGNN** (*Spatio-Temporal Graph Neural Network*, red neuronal de grafo espacio-temporal). Además de mirar la historia temporal de cada tramo, este modelo modela explícitamente la red vial como un grafo: sabe qué tramos son vecinos y aprende cómo la congestión en un tramo se relaciona con la de los tramos conectados. La hipótesis es que esa información espacial —el hecho de que el tránsito de una avenida afecta a las calles que desembocan en ella— debería mejorar la predicción. El costo de esa capacidad es que el STGNN es más complejo y más caro de entrenar.

La pregunta de la evaluación fue directa: **¿la información espacial del STGNN mejora la predicción lo suficiente como para justificar su mayor complejidad, especialmente cuando el tránsito está congestionado?**

## 3. Cómo se midió: las métricas, explicadas

Para comparar los modelos se usaron tres métricas. Cada una mide una cosa distinta sobre la calidad de una predicción, y conviene entender qué dice cada número antes de mirar la tabla de resultados.

**MAE (Error Absoluto Medio).** Es la métrica más intuitiva. Toma cada predicción, mide cuánto se equivocó respecto al valor real, y promedia todos esos errores. Como la variable predicha es demora en segundos, un MAE de 5.7 significa que, en promedio, el modelo se equivoca por unos 5.7 segundos de demora en cada predicción. **Cuanto más bajo, mejor.** Su ventaja es que es fácil de interpretar: está en las mismas unidades que lo que se predice (segundos). Su característica es que trata todos los errores por igual, sin castigar especialmente a los grandes.

**RMSE (Raíz del Error Cuadrático Medio).** Es parecido al MAE —también mide el error promedio en segundos, y cuanto más bajo mejor— pero con una diferencia importante: penaliza más fuerte los errores grandes. Antes de promediar, eleva cada error al cuadrado, lo que hace que un error grande pese mucho más que varios errores pequeños. Por eso el RMSE siempre es mayor o igual que el MAE, y la diferencia entre ambos indica cuánto pesan los errores grandes. En la práctica, el RMSE importa porque un modelo que ocasionalmente comete errores enormes (predecir tránsito fluido cuando en realidad hay un congestionamiento severo) es peligroso para un sistema de control de tránsito, y el RMSE captura precisamente esa clase de falla.

**R² (Coeficiente de Determinación).** A diferencia de las dos anteriores, esta métrica no está en segundos sino en una escala de 0 a 1 (puede ser negativa si el modelo es muy malo). Mide qué proporción de la variación real del tránsito logra explicar el modelo. Un R² de 1.0 sería una predicción perfecta; un R² de 0 significaría que el modelo no predice mejor que simplemente decir siempre el promedio. Un R² de 0.75 quiere decir que el modelo captura el 75% de la dinámica real del tránsito. **Cuanto más alto, mejor.** Es útil porque da una noción global de qué tan bien el modelo sigue el comportamiento del sistema, independiente de las unidades.

*(Una cuarta métrica que aparece a menudo en este tipo de trabajos es el MAPE, el Error Porcentual Absoluto Medio, que expresa el error como un porcentaje del valor real. No se usó como métrica de decisión en esta evaluación porque la demora tiene muchos valores cercanos a cero —tramos sin congestión— y dividir por valores muy chicos infla artificialmente el porcentaje, volviéndolo poco confiable para esta variable en particular.)*

## 4. En qué condiciones se midió: el régimen severo

Un modelo de tránsito es fácil de acertar cuando no pasa nada: si las calles están vacías, predecir "sin demora" acierta casi siempre. Lo que realmente importa para un sistema de gestión de tránsito es cómo se comporta el modelo **cuando hay congestión**, que es justo cuando una buena predicción tiene valor y cuando los errores son costosos.

Por eso la evaluación separó el desempeño en varios regímenes, y la decisión se ancló en los dos más exigentes:

- **Régimen severo de día** (`severe_dia`): los períodos de congestión alta durante la jornada. Es el corte primario de la decisión.
- **Régimen severo de pico** (`severe_pico`): los momentos más afilados de congestión, las horas pico. Es el corte más exigente de todos.

Medir ahí, y no en el promedio general, es lo que hace la comparación honesta: se evalúa a los modelos en el escenario donde de verdad tienen que rendir.

## 5. Los resultados

La comparación se realizó con cinco repeticiones independientes de cada modelo (cinco *seeds*, es decir, cinco entrenamientos con distinta inicialización aleatoria), para asegurar que el resultado no dependiera de la suerte de una única corrida. Los valores que siguen son la media sobre los cinco entrenamientos, con su desviación, a un horizonte de predicción de 30 minutos.

### Régimen severo de día (corte primario)

| Métrica | GRU (línea base) | STGNN | Diferencia | Mejora |
|---|---|---|---|---|
| MAE | 5.991 ± 0.089 | **5.743 ± 0.100** | −0.248 s | ~4% |
| RMSE | 19.846 ± 0.100 | **18.092 ± 0.094** | −1.754 s | ~9% |
| R² | 0.700 ± 0.003 | **0.751 ± 0.003** | +0.051 | — |

### Régimen severo de pico (corte más exigente)

| Métrica | GRU (línea base) | STGNN | Diferencia | Mejora |
|---|---|---|---|---|
| MAE | 7.963 ± 0.095 | **7.418 ± 0.078** | −0.545 s | ~7% |
| RMSE | 27.851 ± 0.310 | **24.861 ± 0.274** | −2.990 s | ~11% |
| R² | 0.737 ± 0.006 | **0.791 ± 0.005** | +0.054 | — |

Cómo leer estas tablas: en cada fila, el STGNN tiene menor error (MAE y RMSE más bajos) y mayor capacidad explicativa (R² más alto) que la GRU. La mejora es de aproximadamente 4% a 7% en el error promedio (MAE) y de 9% a 11% en el error que castiga los fallos grandes (RMSE). El que la ventaja en RMSE sea mayor que en MAE es significativo: indica que el STGNN no solo acierta mejor en promedio, sino que comete menos errores grandes, que es precisamente la clase de falla más costosa en congestión.

Ampliando la mirada a toda la evaluación —cuatro regímenes, tres métricas y seis horizontes de predicción, lo que da 72 combinaciones— el STGNN resultó mejor en las 72, sin una sola excepción a favor de la GRU.

## 6. Por qué el resultado es confiable y no producto del azar

Que un modelo dé mejores números en promedio no basta: hay que descartar que la diferencia sea casualidad de un entrenamiento afortunado. La evidencia de robustez se apoya en dos pilares.

### Separación sin solapamiento (el argumento principal)

Este es el argumento más fuerte y el más fácil de verificar. Al observar los cinco entrenamientos individuales de cada modelo, en todos los cortes severos **el peor entrenamiento del STGNN sigue siendo mejor que el mejor entrenamiento de la GRU**. Por ejemplo, en el MAE del régimen severo de día, los cinco resultados de la GRU se ubicaron entre 5.910 y 6.156, mientras que los del STGNN se ubicaron entre 5.651 y 5.882. El peor STGNN (5.882) es mejor que el mejor GRU (5.910): los dos rangos no se tocan.

Esto importa porque elimina la duda de la suerte: no hay ningún escenario, entre los evaluados, donde la GRU iguale al STGNN. Las brechas entre modelos (entre 0.25 y 0.55 segundos) son varias veces mayores que la dispersión interna de cada modelo entre entrenamientos (alrededor de 0.08 a 0.10), de modo que la diferencia es estructural, no ruido.

### Respaldo estadístico formal (con su salvedad honesta)

Como respaldo, se aplicó una prueba estadística estándar, el **test de Wilcoxon pareado de rangos con signo**, que compara los dos modelos seed por seed y calcula la probabilidad de que la diferencia observada se deba al azar. Esa probabilidad es el llamado *p-value*: un valor bajo (por convención, menor a 0.05) indica que la diferencia es muy improbablemente fruto de la casualidad.

El test arrojó el estadístico más fuerte posible (W = 0.0 en todos los cortes severos), que significa que las cinco diferencias favorecieron al STGNN sin una sola inversión. El p-value de dos colas fue 0.0625.

Aquí va la salvedad, que conviene explicar con claridad porque es fácil de malinterpretar: ese 0.0625 **no baja del umbral de 0.05, pero no por debilidad del resultado, sino por una limitación matemática del test con solo cinco muestras.** Con cinco pares que apuntan todos en la misma dirección, el menor p-value de dos colas que el test puede producir es exactamente 2/2⁵ = 0.0625; el test no puede dar un número más bajo aunque la separación entre los modelos fuera infinita. Dicho de otro modo, con cinco seeds el test ya está dando su veredicto más contundente posible. En su versión de una cola (apropiada cuando la hipótesis es direccional, "el STGNN es mejor"), el p-value es 0.0312, que sí cruza el umbral de 0.05.

Por eso el respaldo estadístico se presenta como complemento, no como argumento central. El argumento central es la separación sin solapamiento descrita arriba, que es independiente del tamaño de muestra y visualmente concluyente. El Wilcoxon confirma la dirección y la consistencia del resultado dentro de lo que cinco muestras permiten afirmar.

## 7. Veredicto y alcance

La conclusión técnica es clara: **el STGNN supera a la GRU en la predicción de demora bajo congestión**, en todas las métricas, en todos los regímenes, con una mejora de entre 4% y 11% según la métrica, y con una separación entre entrenamientos que descarta que el resultado sea casualidad. La información espacial que el STGNN modela explícitamente —cómo se relacionan los tramos vecinos de la red— se traduce en predicciones medibles y consistentemente mejores, sobre todo en la reducción de los errores grandes, que son los más costosos para un sistema de control de tránsito.

Conviene precisar el alcance de esta conclusión. Lo que la evaluación establece es la **superioridad técnica** del STGNN en la tarea de predicción. La decisión de adoptarlo como modelo en producción es una cuestión separada, que debe sopesar esa ventaja de calidad contra su mayor costo operativo: el STGNN tarda aproximadamente cuatro veces más en entrenarse, depende de una librería adicional con un entorno de ejecución propio, y su integración en el servicio en tiempo real plantea consideraciones de ingeniería que exceden esta comparación de desempeño. Este documento responde la pregunta de cuál modelo predice mejor; la pregunta de si esa mejora justifica el costo de integración es una decisión de producto e ingeniería que se apoya en este resultado pero no se agota en él.
