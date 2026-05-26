# Motor Adaptativo de Control Semafórico — Teoría completa

> Documento de referencia para defensa. Explica los dos algoritmos del motor (Webster y Max Pressure), cómo se combinan, qué hace cada parámetro, cuándo el sistema falla, y cómo se conectará con el modelo predictivo de congestión (HU-12) en SP4.

---

## 1. El problema que resuelve el motor

Una intersección semaforizada tiene que decidir, ciclo tras ciclo, dos cosas:

1. **Cuánto dura cada turno de verde** (los segundos que cada approach tiene paso libre).
2. **En qué orden entran los turnos** (cuál fase abre primero el ciclo).

La forma tradicional de programar semáforos es con tiempos fijos: el ingeniero observa el tráfico, calcula una vez los verdes, y los deja iguales para siempre. Eso funciona razonablemente bien cuando la demanda es estable. Pero en una ciudad real la demanda cambia: hora pico, fin de semana, evento, accidente, lluvia. Un semáforo de tiempos fijos sirve mal en todos esos momentos.

La alternativa es un **control adaptativo**: el semáforo recalcula sus tiempos en función de la demanda actual. El motor que construimos en SP3 es un control adaptativo de dos niveles:

- **Webster (1958)** — algoritmo clásico. Calcula el ciclo óptimo bajo asunción de demanda estable. Sirve cuando la intersección está holgada (off-peak).
- **Max Pressure (Varaiya 2013)** — algoritmo reactivo. Mira las colas instantáneas y elige cuál fase entra primero. Sirve cuando la intersección está estresada (peak).

El motor decide cuál de los dos usar según la demanda total observada.

---

## 2. Vocabulario mínimo

Definiciones cortas, todas usadas más abajo:

| Término | Significado |
|---|---|
| **Approach** | Una de las "ramas" que entran a la intersección (norte, sur, este, oeste). |
| **Fase** | Conjunto de approaches que tienen verde simultáneo. En un cruce típico, fase NS (Norte+Sur juntos) y fase EW (Este+Oeste juntos). |
| **Ciclo** | Una vuelta completa del semáforo: NS verde → amarillo → all-red → EW verde → amarillo → all-red → vuelve a NS. |
| **All-red** | Intervalo corto donde todas las fases están en rojo. Sirve para que la intersección "se vacíe" entre fases. |
| **Flujo (q)** | Cuántos autos quieren pasar por hora por una fase. Se mide en veh/h. Es la demanda. |
| **Flujo de saturación (s)** | Cuántos autos podrían pasar por hora si tuvieran verde permanente. Es la capacidad teórica. Se mide también en veh/h. |
| **Cola** | Cuántos autos están detenidos esperando en este momento. |
| **Verde efectivo (g)** | Los segundos del verde "útiles" para mover autos, descontando el tiempo perdido al arrancar y la fracción del amarillo que los autos extienden el cruce. |
| **Tiempo perdido (L)** | Segundos del ciclo que efectivamente NO se aprovechan para mover autos. Es un parámetro teórico, no se mide directamente. |
| **Y (factor de carga)** | Una métrica que mide cuán saturada está la intersección. Es la suma de q/s sobre las fases. Si Y se acerca a 1, la intersección está al límite de su capacidad. |

---

## 3. Algoritmo 1 — Webster (1958)

### 3.1 La pregunta que responde

Dada una demanda estable, ¿cuál es el ciclo más corto posible que aún logra mover toda la demanda con la menor demora promedio para los conductores?

### 3.2 La intuición

Pensá en una balanza. De un lado tenés "ciclos cortos" (los autos esperan poco entre verdes, pero se pierde mucho tiempo en transiciones amarillo+all-red). Del otro lado tenés "ciclos largos" (menos transiciones por hora, pero los autos esperan más en cada rojo). En el medio hay un punto óptimo. Webster derivó la fórmula que encuentra ese punto.

La fórmula clásica es:

$$C_{\text{óptimo}} = \frac{1{,}5 \cdot L + 5}{1 - Y}$$

donde:

- **L** es el tiempo perdido por ciclo (parámetro de diseño, típicamente 4-12 segundos para 2-4 fases).
- **Y** es el factor de carga (se calcula desde la demanda).

Las constantes 1.5 y 5 las encontró Webster empíricamente en su paper original analizando intersecciones reales en Inglaterra. Llevan más de 60 años en uso y se mantienen en los manuales actuales (FHWA Traffic Signal Timing Manual, 2008).

### 3.3 Cómo se calcula Y (el factor de carga)

Para cada fase i, calculás el cociente q_i / s_i (demanda / capacidad). Después sumás esos cocientes sobre todas las fases:

$$Y = \sum_{i} \frac{q_i}{s_i}$$

**Ejemplo concreto** (caso del demo, preset "Off-peak típico"):

- Fase NS: flujo q_NS = 750 veh/h, saturación s_NS = 1800 veh/h → q/s = 0.417
- Fase EW: flujo q_EW = 600 veh/h, saturación s_EW = 1800 veh/h → q/s = 0.333
- **Y = 0.417 + 0.333 = 0.750**

Interpretación: la intersección está usando el 75% de su capacidad agregada.

### 3.4 Cálculo del ciclo óptimo

Sustituyendo en la fórmula con L = 8 segundos:

$$C_{\text{óptimo}} = \frac{1{,}5 \cdot 8 + 5}{1 - 0{,}75} = \frac{17}{0{,}25} = 68 \text{ s}$$

### 3.5 Reparto del verde

Una vez calculado el ciclo, hay que decidir cuántos segundos de verde le toca a cada fase. Webster reparte el "verde total efectivo" (ciclo total menos tiempo perdido) proporcionalmente al peso de cada fase en Y:

$$g_i = (C - L) \cdot \frac{q_i / s_i}{Y}$$

**Ejemplo continuando el caso:**

- Verde efectivo total disponible = 68 − 8 = **60 segundos**
- g_NS = 60 × (0.417 / 0.750) = 60 × 0.556 = **33.3 s**
- g_EW = 60 × (0.333 / 0.750) = 60 × 0.444 = **26.7 s**

Verificación: g_NS + g_EW = 60 ✅

### 3.6 Cuándo Webster funciona y cuándo falla

**Funciona bien cuando:**
- La demanda es estable (no fluctúa fuerte de minuto a minuto).
- La intersección no está saturada: Y < 0.95.
- Las saturations están bien medidas (geometría conocida).

**Falla cuando Y ≥ 0.95.** Mirá la fórmula del ciclo: si Y se acerca a 1, el denominador (1 − Y) se acerca a 0 y el ciclo tiende a infinito. En la práctica, cuando Y ≥ 0.95 el ciclo "óptimo" sería de varios minutos, lo cual es absurdo (los conductores no esperan tres minutos en rojo). Webster levanta una excepción `WebsterInfeasible` y delega al motor decidir qué hacer.

Por eso el motor responde HTTP 422 con `code=webster_infeasible` cuando `Y ≥ 0.95`. Es un caso de **saturación severa**: la intersección está demasiado cargada para que un ciclo de tiempos óptimos exista. La solución matemática no existe; hay que reducir demanda, aumentar capacidad, o intervenir manualmente.

### 3.7 Por qué `lost_time` no es la suma visual de amarillos+all-reds

Este es un detalle conceptual sutil que vale conocer porque es contraintuitivo.

En tu intersección, los amarillos físicos son 3s y los all-reds 2s, lo que da 5s por cambio de fase. Para 2 fases con 2 cambios por ciclo, la suma física es 10 segundos. Pero el slider del demo usa `lost_time = 8`.

La razón es que **no todo el amarillo es tiempo perdido para mover autos**. Cuando el semáforo cambia de verde a amarillo, los autos que ya iniciaron el cruce lo terminan — durante el amarillo y a veces mordiendo el inicio del all-red. Esa fracción cuenta como "productiva" en términos de Webster.

La descomposición típica para una fase:

```
Intergreen físico = yellow + all_red       = 3 + 2 = 5 s
Lost time efectivo per fase                ≈ 3-4 s
Extension (parte productiva del amarillo)  ≈ 1-2 s
```

Para 2 fases, intergreen físico total = 10 s, lost time efectivo total ≈ 6-8 s.

**Implicancia práctica.** Webster calcula el ciclo óptimo con `lost_time = 8 s` (parámetro teórico). Pero el ciclo real que devuelve el motor es 70 s, no 68 s, porque la capa MTC agrega los amarillos+all-reds físicos completos (10 s) por encima de los verdes efectivos (60 s). La diferencia de 2 segundos es el "stretch" entre el lost_time teórico y el intergreen físico.

Para defensa: "Webster usa `lost_time` como parámetro teórico de su fórmula. La capa MTC compone los amarillos y all-reds físicos según las constantes regulatorias peruanas. La diferencia representa la fracción del amarillo que los conductores aprovechan extendiendo el cruce."

---

## 4. Algoritmo 2 — Max Pressure (Varaiya 2013)

### 4.1 La pregunta que responde

Dado el estado actual de las colas, ¿cuál fase debería entrar primero al próximo ciclo para evitar que las colas se descontrolen?

### 4.2 La intuición

Imaginate cada approach como un caño de agua que llega a una intersección. El "agua" es el tráfico. Si en un caño se acumula presión (cola larga), conviene abrirlo primero. Max Pressure literalmente calcula la "presión" de cada fase y elige la de mayor presión como próxima.

Es un algoritmo **reactivo**: no asume demanda estable, mira lo que está pasando ahora.

### 4.3 Cómo se calcula la presión

La fórmula rigurosa de Varaiya (2013) es compleja porque considera flujos upstream, downstream, y matrices de doblar:

$$P(\varphi) = \sum_{l \in \varphi} s_l \cdot \left(x_{\text{up}}(l) - \sum_m \tau_{l,m} \cdot x_{\text{down}}(m)\right)$$

donde:
- φ es la fase candidata
- l son los links (carriles) que la fase activa
- s_l es el flujo de saturación del link
- x_up es la cola upstream, x_down la downstream
- τ_l,m son las fracciones de tráfico que doblan del link l al link m

En el POC del SP3 usamos una **versión simplificada**: la presión de una fase es la cola de la fase ponderada por su capacidad.

$$P(\varphi) \approx \text{cola}(\varphi) \cdot \frac{s(\varphi)}{3600}$$

(La división por 3600 es para normalizar a unidades comparables — autos por segundo.)

**Ejemplo concreto** (preset "Peak normal"):
- Fase NS: cola = 15 vehículos, saturación = 3000 veh/h → P_NS = 15 × (3000/3600) = **12.5**
- Fase EW: cola = 5 vehículos, saturación = 3000 veh/h → P_EW = 5 × (3000/3600) = **4.17**

P_NS > P_EW → la próxima fase a entrar es **NS**.

### 4.4 Round-robin alfabético cuando todas las colas son cero

Si todas las fases tienen cola = 0 (raro, pero posible cuando el intervalo de medición justo cayó entre ráfagas), la fórmula da empate (todas las presiones son cero). En ese caso el motor entra en **round-robin alfabético**: elige la fase cuyo `phase_id` viene primero en orden alfabético. Para NS/EW, gana EW (porque "EW" < "NS"). Es un tie-breaker determinístico que evita comportamiento errático en momentos de baja demanda.

### 4.5 Por qué Max Pressure necesita un ciclo base

Max Pressure decide **cuál fase entra primero**, pero no cuánto dura cada verde. Para definir los verdes hace falta un ciclo base. El motor lo resuelve así:

- **Si Webster es feasible** (Y < 0.95) en peak: usa el ciclo de Webster como base. Es decir, MP elige el orden y Webster define las duraciones.
- **Si Webster es infeasible** (Y ≥ 0.95) en peak: usa un ciclo fijo de 60 segundos como `default_cycle`, y reparte verdes proporcionalmente a las colas.

Esto es lo que viste en el preset "Peak saturado": flow_total = 2000, saturations bajas hacen Y ≈ 1.11 ≥ 0.95, entonces Webster cae y MP usa los 60s default.

### 4.6 Por qué Max Pressure tiene garantías matemáticas

Varaiya demostró en 2013 (paper *"Max pressure control of a network of signalized intersections"*) que MP tiene una propiedad atractiva: **estabilidad**. En términos simples, si la demanda total es menor a la capacidad agregada de la red, MP garantiza que las colas no crezcan indefinidamente. No promete colas cortas — promete que NO explotan.

Esto es valioso para defensa: MP no es un algoritmo ad-hoc, es una contribución académica con propiedades formales demostradas.

---

## 5. Cómo el motor combina ambos algoritmos

### 5.1 El umbral peak

El motor usa un umbral simple para decidir qué algoritmo activar: **flow_total > 1500 veh/h**.

`flow_total` es la suma de los flujos de todas las fases del input. Si la suma excede 1500, el motor considera la intersección en **peak** y activa Max Pressure. Si está debajo, considera **off-peak** y activa Webster.

**Por qué 1500.** Es heurístico, no físico. Se justifica así:

- Una intersección urbana típica de 2 fases con 1-2 carriles por approach tiene capacidad teórica agregada ~3000-3600 veh/h.
- 1500 representa ~40-50% de esa capacidad — el punto donde la demanda empieza a estresar el ciclo de tiempos fijos.
- Por debajo, Webster funciona bien y produce ciclos cortos eficientes.
- Por encima, hace falta una estrategia reactiva que responda a colas instantáneas en lugar de optimizar contra demanda promedio.

El umbral es **parametrizable**. En SP4 se calibra contra simulaciones SUMO. En producción se ajusta por intersección según observación real.

### 5.2 Los cuatro casos del motor

Combinando los dos criterios (flow_total vs 1500, Y vs 0.95) salen cuatro casos posibles:

| flow_total | Y (saturación) | Modo activado | Notas |
|---|---|---|---|
| ≤ 1500 | < 0.95 | **Webster (off-peak)** | Caso normal de baja demanda. Ciclo óptimo, verdes proporcionales. |
| ≤ 1500 | ≥ 0.95 | **422 webster_infeasible** | Caso raro: demanda baja pero saturación severa (saturations patológicamente bajas, ej. obras o un carril cerrado). |
| > 1500 | < 0.95 | **Max Pressure (peak normal)** | MP elige próxima fase, Webster define duraciones. |
| > 1500 | ≥ 0.95 | **Max Pressure (peak saturado)** | MP usa ciclo default 60s, reparte verdes proporcionalmente a colas. Caso defensivo, no óptimo. |

### 5.3 Por qué Webster se sigue calculando en peak

Aunque en peak el "modo activo" es Max Pressure, internamente Webster sigue calculándose. Tres razones:

1. **MP necesita un ciclo base.** Si Webster es feasible, ese ciclo es mejor que cualquier default arbitrario.
2. **Validación cruzada.** Si Y se acerca a 0.95 sin cruzarlo, podemos detectar que la intersección está al borde del límite operativo y emitir alertas (no implementado en SP3, pendiente para SP4).
3. **Fallback robusto.** Si MP falla por algún motivo (ej. todas las colas son cero, todas las saturations son patológicas), Webster sigue siendo la opción segura.

---

## 6. Cumplimiento normativo: la capa MTC

### 6.1 Por qué existe esta capa

Webster y Max Pressure son algoritmos teóricos que pueden producir tiempos que no cumplen con la regulación peruana. Por ejemplo:

- Webster con baja demanda en una fase puede asignar verde de 2 segundos. Eso viola el mínimo de 7 segundos del MTC peruano (Manual de Dispositivos de Control del Tránsito, R.D. 26-2024-MTC/18) y el mínimo peatonal de 7 segundos (un peatón no alcanza a cruzar 2s).
- Webster con saturación alta puede asignar verde de 90 segundos. Eso es operativamente inviable (los conductores en la fase opuesta no esperan tanto).

La **capa MTC** toma los verdes "ideales" calculados por Webster o MP y los corrige para que cumplan con los límites legales y operativos.

### 6.2 Las constantes MTC

| Constante | Valor | Significado |
|---|---|---|
| `MIN_GREEN` | 7 s | Verde mínimo por fase (criterio operativo + permite cruce peatonal básico). |
| `MAX_GREEN` | 60 s | Verde máximo por fase (criterio operativo: evita esperas largas en fase opuesta). |
| `MIN_YELLOW` | 3 s | Amarillo mínimo (criterio MTC: tiempo de reacción del conductor). |
| `ALL_RED` | 2 s | All-red obligatorio entre fases (criterio MTC: limpia la intersección). |
| `MIN_PEDESTRIAN` | 7 s | Verde mínimo cuando la fase tiene cruce peatonal activo. |

Las constantes son **parametrizables** en el backend (`settings`). En SP4 se podrán ajustar por intersección si la regulación local lo justifica (ej. cruces escolares con tiempo peatonal extendido).

### 6.3 Cómo modifica los outputs

La capa MTC recibe los verdes calculados por Webster o MP y aplica tres correcciones:

1. **Eleva si está bajo mínimo.** Si Webster asignó g_NS = 4.2 s, MTC lo eleva a 7 s y registra un *adjustment* `"NS: verde elevado de 4.2s a 7s (MIN_GREEN)"`.
2. **Recorta si está sobre máximo.** Si MP asignó g_NS = 75 s, MTC lo recorta a 60 s con adjustment `"NS: verde recortado de 75s a 60s (cycle_capped)"`.
3. **Compone amarillos y all-reds físicos.** Toma los verdes corregidos y agrega 3s de amarillo + 2s de all-red por cada fase. El ciclo final = suma de todos los segmentos.

Por eso el ciclo final del demo (70 s) puede ser distinto del ciclo "óptimo" calculado por Webster (68 s). La diferencia son los amarillos+all-reds físicos que MTC compone.

### 6.4 Por qué separamos esta capa

La separación de responsabilidades es deliberada:

- **Webster y MP** son algoritmos puros, optimizan capacidad. No saben de regulación local.
- **MTC** es una capa de cumplimiento. Toma los outputs de los algoritmos y los hace legales/operativos.

Si mañana cambia la regulación peruana o adoptamos otra norma (chilena, colombiana, internacional), reemplazamos la capa MTC sin tocar los algoritmos. Buena ingeniería de software, además de ser pedagógicamente claro para defensa.

---

## 7. Conexión con HU-12 (predicción de congestión)

### 7.1 Qué da HU-12

El módulo de predicción HU-12 (`CongestionPredictor.predict_congestion`) recibe datos de una cámara y devuelve una predicción de congestión a 15-30-45 minutos. La salida es un **nivel ordinal de 1 a 5**:

| Nivel | Etiqueta | Descripción |
|---|---|---|
| 1 | Bajo | Tráfico fluido, casi sin congestión. |
| 2 | Ligero | Tráfico levemente más denso, sin demoras significativas. |
| 3 | Moderado | Congestión perceptible, demoras leves a moderadas. |
| 4 | Alto | Congestión significativa, demoras notorias. |
| 5 | Severo | Casi colapso, demoras muy largas. |

La predicción es **por cámara**, no por approach ni por fase. Una cámara puede mirar uno o varios approaches según su ubicación y ángulo.

### 7.2 Qué necesita el motor

El motor de control adaptativo necesita inputs muy distintos:

| Lo que da HU-12 | Lo que necesita el motor |
|---|---|
| Nivel ordinal 1-5 | Flujo numérico continuo en veh/h |
| Por cámara | Por fase (que agrupa approaches) |
| Horizonte 15-30-45 min | Estado actual (instantáneo) |

Hay un **gap semántico real**. No se puede conectar HU-12 al motor con un import directo porque las dimensiones no calzan.

### 7.3 El adapter de SP4

Para conectar HU-12 con el motor hace falta un componente intermedio (un *adapter de dominio*) que traduzca:

1. **Cámara → approach → fase.** Una tabla de mapeo dice qué cámara mira qué approach, y qué fase agrupa qué approaches. Este mapeo es por intersección y depende del despliegue físico.

2. **Nivel de congestión → flujo veh/h.** Hay que convertir el ordinal 1-5 en un flujo numérico. La forma más simple es una tabla de lookup calibrada con observaciones reales:

| Nivel | Rango de flujo equivalente (veh/h) |
|---|---|
| 1 (Bajo) | 200-500 |
| 2 (Ligero) | 500-900 |
| 3 (Moderado) | 900-1400 |
| 4 (Alto) | 1400-1800 |
| 5 (Severo) | 1800-2500+ |

Estos rangos son **sugeridos para POC**. En producción se calibran contra mediciones reales por approach. Un nivel 4 en una avenida principal no significa lo mismo que un nivel 4 en una calle secundaria.

3. **Horizonte 15-30-45 min → flujo instantáneo.** La predicción es de demanda futura, no del estado actual. Hay dos estrategias:
   - **Anticipación pura:** usar directamente el flujo predicho como si fuera el actual. El motor "ve el futuro".
   - **Suavizado:** combinar el flujo medido ahora (cámara YOLOv8) con el predicho (HU-12) en un promedio ponderado: `flow = α × flow_actual + (1−α) × flow_predicho`, con α típicamente 0.7-0.8 (más peso al actual).

El adapter completo es trabajo de SP4 (HU explícita pendiente en backlog refinado).

### 7.4 Cómo cambia el comportamiento del motor con predicción

Hoy el motor opera en **modo reactivo**: ve la demanda actual y reacciona. Ciclo a ciclo, las decisiones son óptimas pero "ciegas al futuro". Si en 10 minutos llega una ola de demanda, el motor recién la atenderá cuando los autos lleguen.

Con HU-12 conectado, el motor pasa a **modo proactivo**:

1. **Detección anticipada del cambio peak/off-peak.** Si el motor está en off-peak (Webster) pero la predicción indica que en 15 min la demanda subirá a peak, puede empezar a alargar los verdes de las fases que se cargarán, anticipando la ola.

2. **Suavizado de transiciones.** En lugar de saltar bruscamente de Webster a MP cuando la demanda cruza 1500, puede transicionar gradualmente — alargando ciclos sin cambiar de algoritmo todavía, después cambiando de algoritmo cuando la demanda real lo confirma.

3. **Asignación preemptiva de verde.** Si la predicción dice que la fase NS tendrá nivel 4 en 15 min, el motor puede ir reasignando verde a NS gradualmente para que la cola no crezca cuando llegue la ola.

4. **Alertas preventivas al operador C4.** Si la predicción para 30 min apunta a saturación (nivel 5 en varias fases), el motor puede emitir una alerta operativa: *"saturación predicha en INT_001, considerar intervención manual o rerouting"*.

### 7.5 Por qué SP3 no incluye HU-12

Decisiones de scope:

1. **Adapter no trivial.** El adapter cámara→approach + nivel→flujo necesita su propia HU con calibración empírica. No se puede improvisar dentro de SP3.

2. **Stateless en SP3.** El motor SP3 es stateless — no tiene memoria entre llamadas, no puede mantener una serie temporal de predicciones. La integración real con HU-12 se beneficia de un motor que recuerde decisiones recientes.

3. **Validación necesaria.** Antes de cablear HU-12 al motor en producción, hay que validar la calidad de la predicción contra realidad observada. Esa validación es trabajo de SP4 con SUMO.

Por eso el DTO del motor expone `predicted_demand` como **campo opcional**. En SP3 lo recibe pero lo ignora. Cuando SP4 cierre el adapter, basta con dejar de enviarlo en None y el motor empieza a usarlo. El contrato HTTP no cambia.

---

## 8. Casos de fallo y cómo los maneja el motor

### 8.1 Webster infeasible (Y ≥ 0.95)

**Síntoma.** Y se acerca o supera 0.95. La fórmula del ciclo óptimo daría un número absurdo (cientos de segundos).

**Causas posibles.**
- Demanda real mayor a la capacidad de la intersección.
- Saturations mal medidas (carril obstruido, obra, lluvia que reduce capacidad).
- Phase splits mal definidos (movimientos que no deberían estar en la misma fase).

**Cómo lo maneja el motor.**
- En off-peak: levanta excepción `WebsterInfeasible`, el motor responde HTTP 422 con `code=webster_infeasible`. Delega la decisión al operador C4.
- En peak: Max Pressure cae a su `default_cycle = 60s` y reparte verdes proporcionalmente a las colas. No es óptimo pero es operativo.

**Mensaje para defensa.** *"El motor reconoce que la matemática del control adaptativo tiene un límite operativo. Cuando la intersección está demasiado saturada, el motor delega al operador en lugar de inventar una solución matemáticamente inválida."*

### 8.2 Datos inválidos del input

**Síntoma.** El frontend envía un IntersectionState con valores que rompen las constraints Pydantic (saturation_flow ≤ 0, flow < 0, phases vacías, etc.).

**Cómo lo maneja el motor.**
- Pydantic los rechaza antes de entrar al endpoint.
- El motor responde HTTP 422 con shape `{detail: [...]}` (array de errores Pydantic).
- El frontend lo mapea a un banner genérico "Datos inválidos. Revisa el formulario."

### 8.3 Errores de dominio post-Pydantic

**Síntoma.** El input pasa la validación Pydantic pero la lógica interna del motor detecta inconsistencias (ej. phase_ids duplicados, matrices de turn_ratios mal formadas en el futuro).

**Cómo lo maneja el motor.** El motor levanta un `ValueError` que se traduce a HTTP 422 con `code=invalid_state`. El frontend lo mapea a un banner naranja con el mensaje específico del error.

Este caso no se dispara con los inputs actuales del demo (Pydantic ya valida todo lo cubierto), pero el código está listo para cuando el dominio crezca.

### 8.4 Datos faltantes (HU-12 no disponible)

**Síntoma.** En SP4, si HU-12 no responde a tiempo o devuelve un nivel `null`.

**Cómo lo maneja el motor.** `predicted_demand` es opcional. Si no llega, el motor opera en modo reactivo (como hoy en SP3). La predicción es un *enhancement*, no un *requirement*.

---

## 9. Preguntas anticipables del jurado y respuestas cortas

### "¿Por qué dos algoritmos en lugar de uno?"

Webster es óptimo bajo demanda estable pero falla bajo demanda variable. Max Pressure es robusto bajo demanda variable pero no optimiza demora promedio. Combinarlos da lo mejor de ambos: eficiencia en off-peak, robustez en peak. Es una arquitectura híbrida estándar en literatura de ITS.

### "¿De dónde sacaron las constantes de Webster (1.5 y 5)?"

Son las constantes empíricas del paper original de F.V. Webster (1958), *"Traffic signal settings"*, derivadas analizando datos reales de intersecciones inglesas. Llevan más de 60 años en uso y siguen siendo la base del FHWA Traffic Signal Timing Manual (2008), referente normativo internacional.

### "¿Por qué el umbral peak es 1500 veh/h y no otro número?"

Es heurístico, no físico. Representa ~40-50% de la capacidad teórica agregada de una intersección urbana de 2 fases (~3000-3600 veh/h). Por debajo, Webster funciona bien. Por encima, hace falta una estrategia reactiva. El valor exacto es parametrizable y se calibra en SP4 con SUMO contra simulaciones de Miraflores.

### "¿Qué pasa si la intersección tiene 4 fases en lugar de 2?"

El motor soporta N fases. Webster suma sobre N términos en el cálculo de Y, y reparte verde sobre N fases proporcionalmente. Max Pressure elige cuál de las N fases entra primero por presión. La capa MTC aplica las mismas constantes a cada fase. El demo del frontend usa 2 fases (NS/EW) como caso paradigmático de libro; el caso de 4 fases es lo que se va a simular en SP4 si la intersección piloto de Miraflores lo amerita.

### "¿Qué pasa si las cámaras YOLOv8 fallan y no dan inputs?"

`predicted_demand` es opcional. Pero `flow`, `saturation_flow` y `queue` son obligatorios — si no llegan, el motor responde 422. La estrategia operativa en SP4 es: si una cámara falla por más de 60 segundos, el motor cae a un programa de tiempos fijos pre-cargado para esa intersección. Esto es trabajo de HU explícita pendiente.

### "¿Cómo se valida que el motor mejora frente a tiempos fijos?"

En SP4 se simula la misma intersección en SUMO con dos configuraciones: (a) tiempos fijos pre-calibrados, (b) motor adaptativo recibiendo inputs sintéticos del simulador. Se compara la demora promedio por vehículo (KPI principal del IE05). El criterio de éxito es RD% ≥ 15%.

### "¿Cumple con la regulación peruana?"

El motor incorpora explícitamente las restricciones del Manual MTC R.D. 26-2024-MTC/18 (octubre 2024) vía la capa MTC: mínimos de verde, amarillos, all-reds y mínimos peatonales. Las constantes son parametrizables si la regulación cambia. La justificación regulatoria está documentada en Cap 2 (Marco regulatorio peruano).

### "¿Qué pasa con los peatones?"

Cada fase tiene un flag `has_pedestrian` que activa el mínimo de verde peatonal (7 s). El motor garantiza que si la fase tiene cruce peatonal activo, el verde nunca baje de ese mínimo, incluso si Webster lo calcularía menor. La detección automática de demanda peatonal es trabajo futuro (HU explícita pendiente para SP4).

### "¿Por qué no usaron una red neuronal directamente?"

Hay tres razones. **Defensibilidad**: Webster y MP tienen 60+ y 10+ años de literatura respectivamente, con propiedades formales demostradas (estabilidad de MP). Una red neuronal sería caja negra y difícil de defender ante un asesor de tránsito. **Datos**: una red neuronal de control semafórico necesitaría miles de horas de datos reales por intersección, que no tenemos. **Regulación**: la capa MTC tiene que ser auditable; un controlador neural opaco es problemático ante una eventual investigación de la Defensoría del Pueblo (caso Vía Expresa Lima Sur, sept 2025).

### "¿Cómo se conecta con el modelo predictivo?"

Hoy no se conecta — `predicted_demand` es opcional y el motor lo ignora en SP3. La conexión real requiere un *adapter de dominio* que mapee (a) cámara → approach → fase y (b) nivel ordinal 1-5 → flujo veh/h. Este adapter es una HU explícita pendiente para SP4. Cuando se cierre, el motor pasa de modo reactivo a modo proactivo: detecta anticipadamente cambios peak/off-peak, asigna verde preemptivamente a fases que se cargarán, y emite alertas predictivas al operador.

---

## 10. Referencias bibliográficas

- **Webster, F. V. (1958).** *Traffic signal settings* (Technical Paper No. 39). Road Research Laboratory. Documento fundacional del cálculo de ciclos óptimos.
- **Varaiya, P. (2013).** *Max pressure control of a network of signalized intersections*. Transportation Research Part C: Emerging Technologies, 36, 177-195. Demuestra estabilidad teórica de MP.
- **Koonce, P., et al. (2008).** *Traffic Signal Timing Manual* (FHWA-HOP-08-024). Federal Highway Administration. Manual operativo internacional, base de las constantes prácticas.
- **Lopez, P. A., et al. (2018).** *Microscopic traffic simulation using SUMO*. IEEE ITSC 2018. Justifica el harness de simulación que se usa en SP4.
- **Manual de Dispositivos de Control del Tránsito Automotor para Calles y Carreteras** (R.D. N.° 26-2024-MTC/18, octubre 2024). Ministerio de Transportes y Comunicaciones del Perú. Norma vigente que define las restricciones legales aplicadas en la capa MTC.

---

## Anexo — Caso completo trabajado paso a paso

Para fijar todo lo anterior, te paso el cálculo completo del preset "Off-peak típico" del demo, desde inputs hasta outputs.

**Inputs del usuario:**

```
intersection_id: INT_001
lost_time: 8 s
phases:
  NS: { flow: 750 veh/h, saturation: 1800 veh/h, queue: 0, pedestrian: false }
  EW: { flow: 600 veh/h, saturation: 1800 veh/h, queue: 0, pedestrian: false }
```

**Paso 1: el motor calcula el flow_total.**

```
flow_total = 750 + 600 = 1350 veh/h
```

**Paso 2: decide modo según umbral peak.**

```
1350 ≤ 1500 → off-peak → activa Webster
```

**Paso 3: Webster calcula Y.**

```
Y = q_NS/s_NS + q_EW/s_EW
  = 750/1800 + 600/1800
  = 0.4167 + 0.3333
  = 0.7500
```

**Paso 4: Webster verifica feasibility.**

```
Y = 0.75 < 0.95 → feasible, continúa
```

**Paso 5: Webster calcula el ciclo óptimo.**

```
C_opt = (1.5 × L + 5) / (1 − Y)
      = (1.5 × 8 + 5) / (1 − 0.75)
      = (12 + 5) / 0.25
      = 17 / 0.25
      = 68 segundos
```

**Paso 6: Webster reparte el verde efectivo.**

```
verde_efectivo_total = C_opt − L = 68 − 8 = 60 segundos

g_NS = 60 × (q_NS/s_NS) / Y = 60 × 0.4167 / 0.75 = 60 × 0.5556 = 33.3 s
g_EW = 60 × (q_EW/s_EW) / Y = 60 × 0.3333 / 0.75 = 60 × 0.4444 = 26.7 s

verificación: g_NS + g_EW = 33.3 + 26.7 = 60.0 ✅
```

**Paso 7: la capa MTC valida y compone.**

```
g_NS = 33.3 ≥ MIN_GREEN (7) y ≤ MAX_GREEN (60) → OK, sin ajuste
g_EW = 26.7 ≥ 7 y ≤ 60 → OK, sin ajuste

amarillo_NS = 3 s (MIN_YELLOW)
all_red_NS  = 2 s (ALL_RED)
amarillo_EW = 3 s
all_red_EW  = 2 s

ciclo_final = (33.3 + 3 + 2) + (26.7 + 3 + 2) = 38.3 + 31.7 = 70.0 s
```

**Paso 8: el motor devuelve la recomendación.**

```json
{
  "intersection_id": "INT_001",
  "mode": "webster",
  "cycle_seconds": 70.0,
  "phase_timings": [
    { "phase_id": "NS", "green": 33.3, "yellow": 3.0, "all_red": 2.0 },
    { "phase_id": "EW", "green": 26.7, "yellow": 3.0, "all_red": 2.0 }
  ],
  "next_phase": null,
  "reasoning": "Off-peak (Σ flow = 1350 veh/h < 1500). Webster optimal cycle = 68.0s; MTC constraints applied.",
  "adjustments": []
}
```

**Lo que ve el operador en el frontend:**

- Modo activo: **Webster (off-peak)** — pill verde en la fila de modos.
- Ciclo: **70.0 s**.
- Métricas: flow_total = 1350, Y = 0.750, estado = OFF-PEAK.
- Semáforo NS animando: 33.3s verde → 3s amarillo → 2s all-red.
- Semáforo EW animando: 26.7s verde → 3s amarillo → 2s all-red.
- Razonamiento amigable explicando la decisión, log técnico debajo en mono.

Esa es la cadena completa desde un click en "Recomendar" hasta la animación del semáforo. El motor no es una caja negra: cada paso es derivable, defendible, y trazable contra teoría publicada.
