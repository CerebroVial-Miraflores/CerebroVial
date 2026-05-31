# Síntesis — Control adaptativo de semáforos en el corredor Av. Larco

> Documento legible de cierre del estudio del corredor Larco. Audiencia: un ingeniero
> (de software o de cualquier disciplina) que quiera entender **qué se probó, qué funcionó y
> por qué**, sin ser especialista en ingeniería de tránsito. Cada término se explica en su
> primer uso; la matemática es mínima. Los números son los **reales** de los experimentos
> (semillas 42–51, demanda determinista); las configuraciones están **verificadas contra los
> archivos del repo**, no de memoria.

---

## 1. Resumen ejecutivo

Se construyó un banco de pruebas en SUMO (un simulador de tráfico microscópico) para decidir
**cómo deben temporizar** los semáforos de un corredor real de Miraflores —Av. Larco, en las
intersecciones con **Benavides** y **Schell**— bajo hora pico. Se compararon cinco estrategias de
control contra un baseline de tiempos fijos. El ganador es un control adaptativo simple llamado
**Max Pressure "per-node"** (cada cruce decide solo, mirando sus propias colas): reduce la demora
total del conductor en **~15.7 %**. Las cuatro alternativas más sofisticadas —coordinar los cruces
con "onda verde", imponer un ciclo común, y darle a cada cruce información de su vecino de aguas
abajo ("Max Pressure de red", probado en cuatro intensidades)— **no mejoraron**, y varias
**empeoraron**. El hallazgo central, contraintuitivo y bien medido: en este corredor **mirar al
vecino perjudica**, porque el cuello de botella es estructural (el cruce de Schell) y retener autos
arriba solo **mueve la cola a la entrada**, donde el conductor espera más. La estrategia simple es
la óptima.

---

## 2. El corredor y el objetivo

**El corredor.** Un tramo de la Av. José Larco que va de sur a norte y cruza dos avenidas
transversales con semáforo: primero **Benavides**, después **Schell**. Más al norte hay un tercer
cruce (Diez Canseco) que en este modelo descarga libre (no genera conflicto, no se controla). Entre
Benavides y Schell hay un **tramo interno corto** de ~97 m: el "link interno". Es la pieza clave de
toda la historia.

**Qué controla un semáforo aquí.** En cada cruce hay dos **fases** (grupos de movimientos que
reciben verde a la vez):
- **LARCO** — el flujo recto sur→norte por la avenida Larco.
- **TRANSV** — el flujo de la calle transversal (Benavides o Schell).

El controlador decide, ciclo a ciclo, **cuánto verde** le da a cada fase (y, en Max Pressure,
**cuál** sirve primero).

**Qué optimizamos.** La **demora total del conductor** de punta a punta: cuánto tarde de más, en
promedio, respecto de un viaje sin tráfico. Menos demora = mejor.

**El régimen.** Hora pico **sobre-saturada**: llega más tráfico del que el corredor puede evacuar.
En este régimen el cuello de botella **estructural** es el cruce de **Schell**: Benavides puede
empujar autos, pero Schell no los absorbe al mismo ritmo. Ese detalle explica casi todos los
resultados.

---

## 3. Cómo medimos (y por qué así)

**La métrica: demora puerta-a-puerta.** El tiempo del conductor tiene dos partes y contamos las dos:
- **w_wait** — *espera para entrar*: el auto está en la cola de ingreso al corredor y todavía no
  arrancó.
- **w_inside** — *tiempo adentro*: ya circula por el corredor (avanzando o frenado en cola interna).

La demora total es **W_RED = w_wait + w_inside**. Contar las dos partes es esencial: una estrategia
puede "vaciar" la calle de adentro simplemente **no dejando entrar** a los autos — se vería rápida
por dentro mientras la cola de ingreso explota. Solo sumando la espera de entrada se ve la
experiencia real. (Técnicamente W_RED se calcula con la ley de Little sobre todos los autos del
sistema, los que circulan y los que esperan; así no se "pierden" los autos que nunca llegan a entrar.)

**Por qué pareamos por semilla.** Cada corrida usa una **semilla** aleatoria (42 a 51, diez réplicas)
que fija el patrón exacto de llegada de autos. Comparamos cada estrategia **contra otra con la misma
semilla**: misma demanda, mismos autos, misma secuencia — la **única** diferencia es el control. Eso
cancela la variabilidad entre réplicas y deja ver el efecto puro de la estrategia. El veredicto se
toma sobre la **diferencia pareada** Δ (estrategia A − estrategia B) por semilla, no sobre promedios
sueltos que se solapan.

**Por qué la reproducción exacta del control es una prueba de integridad.** El control "per-node"
es nuestro punto de referencia. Cuando una corrida nueva del per-node reproduce su número histórico
**exacto al dígito**, eso garantiza que el simulador, la demanda y el flujo de autos son **idénticos**
entre sesiones — así, cualquier diferencia que veamos en otra estrategia es del **control**, no de un
cambio accidental en el banco de pruebas. Lo verificamos antes de cada experimento.

---

## 4. Los experimentos (qué se probó y por qué dio lo que dio)

**4.1 Max Pressure per-node — ADOPTADO (+15.7 %).**
"Max Pressure" es una regla clásica: darle verde a la fase con más **presión** (cola × capacidad).
"Per-node" = cada cruce decide solo, con sus propias colas. Resultado: **−15.7 %** de demora vs
fijo (10 semillas, 9/10 favorables). El beneficio es físico y honesto: la espera **para entrar** cae
fuerte (de 36.8 s a 12.2 s), mientras el tiempo **adentro** queda casi igual. No descongestiona la
calle —el tapón se **relocaliza** al link interno Benavides→Schell— pero hace **esperar mucho menos
para entrar**. Es el sistema adoptado.

**4.2 Onda verde / offsets — DESCARTADO.**
"Onda verde" = desfasar el arranque de los verdes entre cruces para que un pelotón viaje sin frenar.
Se barrió el desfase de 0 a 80 s. **El óptimo es offset = 0** (no coordinar): cualquier desfase
empeora la demora (+50 a +123 %) y dispara autos que no logran entrar. Por qué: en un corredor
sobre-saturado con link interno **corto** (97 m), mandar el pelotón sincronizado hacia Schell **llena
el link más rápido** y agrava el atasco. Manda la gestión de cola, no la progresión.

**4.3 Ciclo común fijo — DESCARTADO.**
Imponer a ambos cruces el mismo **ciclo** (duración total de la vuelta de semáforo) fijo de 90 s. No
mejora al ciclo variable: vs el per-node de ciclo libre da **−5.3 % ± 18.8** (el intervalo cruza
cero — empate con enorme dispersión; en una semilla llega a −47 %). Fijar el ciclo le quita al motor
la libertad de elegir su largo; a veces ayuda, a veces lo arruina. No generaliza.

**4.4 Max Pressure de red ("mirar al vecino") — REFUTADO.**
La idea: como el per-node relocaliza el tapón al link interno, darle a Benavides la **cola del link
interno** (su vecino de aguas abajo) para que **retenga** cuando ese link está lleno. Fórmula:
presión de la fase = capacidad × (cola_local − **τ** × cola_del_vecino), donde **τ** (turn ratio)
gradúa cuánto pesa el vecino. Resultado: **empeora**. A acoplamiento pleno (τ=1.0): **Δ = +35.07 s**
peor que per-node, intervalo de confianza **[+21.6, +48.6]** (excluye el cero), Wilcoxon p=0.002,
**0/10** semillas a favor; queda en **−9.2 % vs fijo**. Por qué: el término **sí** alivia el link
interno (la cola media de #1 baja ~22 %), pero lo hace **reteniendo Benavides**, lo que **devuelve la
cola a la entrada** (de 144.6 a 172.2 m) y **cuadruplica la espera para entrar** (12.2 → 48.2 s). Como
la métrica cuenta esa espera, el neto es peor. Es el régimen capacidad-limitado: Schell es el cuello;
retener arriba no aumenta cuánto pasa, solo mueve la cola a un lugar peor.

**4.5 Barrido de τ — per-node (τ=0) es el óptimo del eje.**
Para descartar que τ=1.0 fuera simplemente "demasiado", se barrió τ en 0, 0.5, 0.75, 1.0 (τ=0 ≡
per-node, sin mirar al vecino). El eje es **monótono**: cuanto más se mira al vecino, peor. Ningún τ
supera al per-node (todos los intervalos de confianza dan peor, excluyen el cero). Confirma que el
problema no es la sintonía de τ sino la idea misma en este régimen.

**4.6 Barrido de capacidad — cuánto de la demora es arreglable por temporizado y cuánto es irreducible.**
Los experimentos anteriores comparan estrategias a **una** demanda (la pico, scale = 1.0). Para situar
ese punto y separar lo que el control **puede** arreglar de lo que impone la **capacidad** física del
corredor, se barrió la demanda en ocho niveles (×0.6 a ×2.0, escalando todos los flujos en proporción,
preservando los turn ratios), con los dos brazos —**fijo** y **per-node**— pareados por semilla
(42–46, cinco réplicas; basta para la forma de la curva, cuyo techo es casi determinista). No es un
"controlador perfecto" (contestable): es la demanda la que revela el techo. La compuerta de integridad
(per-node a scale = 1.0 reproduce IE05 **exacto al dígito**, semilla a semilla) pasó antes de aceptar
la curva. Figura: `simulation/data/corredor_larco/capacity_sweep/capacity_sweep.png`; tabla:
`capacity_sweep_summary.csv`.

Dos lecturas, una por panel:

- **El techo de capacidad (panel B).** Se mide el **flujo de descarga por Schell** —los autos que
  cruzan la línea de pare de Schell hacia el norte, contados con el detector del link Schell→Diez
  Canseco, no derivados de un ratio de completados (que se sesga al cierre de ventana). Sube con la
  demanda hasta scale = 1.0 y **ahí se aplana**: μ̂ ≈ **1740 veh/h** para el per-node, plano de 1.0 a
  2.0. Ese es el techo: por más demanda que entre, no descargan más autos por Schell. Confirma —desde
  otro ángulo y de forma cuantitativa— el cuello estructural de Etapa 1 (~1650–1700 veh/h clavados).
  La demanda del corredor ofrecida a scale = 1.0 (λ_corr ≈ 1790 veh/h) ya **iguala** ese techo: **el
  punto de operación de IE05 está justo en la rodilla de capacidad** (s* = 1.0).

- **La demora y su descomposición (panel A).** Por debajo de capacidad (≤0.8) la espera-para-entrar
  (w_wait) es **cero** y la demora es modesta y sensible al temporizado: el per-node gana (a 0.6,
  36.9 s vs 40.9 s). En la rodilla (1.0) aparece el número de IE05: per-node 124 s vs fijo 142 s,
  −13 % — y **todo ese ahorro vive en w_wait** (el per-node vacía la entrada mejor: 14 s vs 38 s),
  no en el tiempo adentro (≈igual, ~105–110 s). Pasada la capacidad, w_wait **explota**: 38 → 182 →
  822 s a medida que la demanda supera el techo, y **domina** la demora total (el tiempo adentro
  queda acotado, ~110–148 s en todo el rango). Esa explosión es la parte **irreducible por
  capacidad**: es el exceso (λ − μ̂) que se acumula en la cola de ingreso, ~lineal en la demanda, y
  **ningún temporizado la evita** —fijo y per-node la siguen casi pegados (dentro de ~10 %)—. La línea
  punteada de la figura es el piso teórico determinístico de sobre-saturación (λ−μ̂)·T/2λ; ambos
  brazos quedan **por encima** (el corredor real tiene varias colas acopladas), pero acotados por el
  mismo muro.

**Honestidad del techo.** SUMO podría inflar el techo teletransportando autos atascados (los saca del
tapón y los manda río abajo), lo que subiría el throughput y bajaría la demora justo en las escalas
altas. No pasó: **teleports = 0 en todas las escalas, ambos brazos**. La sobre-saturación se manifestó
como **backlog de ingreso** (autos que nunca entran: 0 → 1080 a scale 2.0), que la métrica cuenta
honestamente dentro de w_wait. El techo medido no es un artefacto.

**El cruce en sobre-saturación profunda — honesto y consistente.** Hay un cruce cerca de scale ≈ 1.1:
por debajo, el per-node gana (régimen de IE05); por encima (≥1.2), el **fijo** queda algo mejor en
demora (a 2.0: 891 s vs 970 s) porque sostiene un poco más de descarga por Schell (1964 vs 1740 a
2.0). Es esperable: el per-node de Etapa 1 (sin término de aguas abajo) balancea presión **local**, y
cuando el link interno se satura no prioriza la descarga del recto de Larco tan agresivamente como el
plan fijo. Es exactamente el modo de falla de alta demanda que el término "mirar al vecino" (MP de red,
§4.4) apuntaba a corregir —y que §4.4/§4.5 **refutaron** como net-negativo, porque retener arriba mueve
la cola a la entrada—. Importa leer la escala: ese cruce vive en el régimen sobre-saturado profundo,
donde la demora ya se multiplicó por ~8 por puro exceso de demanda; la diferencia entre controles
(±10 %) es **de segundo orden** frente al muro de capacidad. En el punto de operación y por debajo, el
per-node es el óptimo.

**El número titular de la caracterización.** En el corredor Larco, a la demanda pico, lo **arreglable
por temporizado** es la fracción de demora que el mejor control quita en o por debajo de la capacidad:
~13–16 % (el per-node, todo vía la espera de ingreso). Lo **irreducible por capacidad** es el resto y,
pasada la rodilla, **casi todo**: la descarga por Schell está topada en ~1740 veh/h y el per-node ya
está en ese techo desde scale = 1.0. No hay temporizado que pase más autos por Schell —eso es
capacidad, no control—.

---

## 5. Benchmark (números reales, 10 semillas pareadas)

W_RED = demora puerta-a-puerta media (s); RD% vs fijo = reducción relativa (positivo = mejor).

| Estrategia | W_RED (s) | RD% vs fijo | w_wait (s) | w_inside (s) | cola link interno benSch (media, m) |
|---|---:|---:|---:|---:|---:|
| **fijo** (baseline) | 141.37 | — | 36.8 | 104.5 | bajo (~20–28 pico; corredor desacoplado) |
| **per-node (τ=0)** ✅ adoptado | **119.26** | **+15.7 %**¹ | 12.2 | 107.1 | 59.7 |
| MP-red τ=0.5 | 137.88 | +2.5 % | 30.6 | 107.3 | 53.6 |
| MP-red τ=0.75 | 144.98 | −2.6 % | 38.2 | 106.8 | 50.1 |
| MP-red τ=1.0 | 154.33 | −9.2 % | 48.2 | 106.2 | 46.7 |

Onda verde (offsets) y ciclo común fijo no entran en la tabla porque se midieron en otro brazo;
resumen: **offset óptimo = 0** (todo desfase empeora +50…+123 %); **ciclo fijo −5.3 % ± 18.8 vs el
per-node** (empate con dispersión enorme). Ambos **descartados**.

Lectura de la tabla: bajar la cola del link interno (columna benSch: 59.7 → 46.7 al subir τ)
**no compra nada** — la espera para entrar (w_wait) sube más rápido (12.2 → 48.2) y arrastra la
demora total para arriba. El per-node, que **no** toca el link interno, es el que menos hace esperar.

¹ +15.7 % es la media de las reducciones por semilla (número titular de IE05). Sobre los promedios
agregados de la tabla da +15.6 %; es la misma señal, distinta forma de promediar.

---

## 6. Glosario de variables

- **SUMO** — simulador de tráfico microscópico (cada auto se modela individualmente).
- **Fase** — grupo de movimientos que reciben verde simultáneamente. Aquí: **LARCO** (recto S→N) y
  **TRANSV** (transversal).
- **Ciclo** — duración total de una vuelta completa del semáforo (todas las fases).
- **W_RED** — demora puerta-a-puerta media por conductor, en segundos = w_wait + w_inside.
- **w_wait** — parte de la demora esperando **para entrar** al corredor (cola de ingreso).
- **w_inside** — parte de la demora ya **dentro** del corredor (circulando o en cola interna).
- **RD%** — reducción relativa de demora vs el control fijo; positivo = mejora.
- **Δ (delta pareado)** — diferencia de W_RED entre dos estrategias en la **misma** semilla. El
  veredicto se toma sobre la distribución de Δ.
- **IC 95 % (intervalo de confianza)** — rango plausible de la media de Δ. Si **no** incluye el 0, el
  efecto es estadísticamente real.
- **Wilcoxon signed-rank** — prueba no-paramétrica (no asume forma de la distribución) que confirma el
  IC sin depender de outliers.
- **Max Pressure** — regla de control: dar verde a la fase de mayor **presión** = capacidad × cola.
- **per-node** — cada cruce decide con sus propias colas (sin mirar vecinos). Equivale a τ=0.
- **MP de red** — variante que resta la cola del vecino de aguas abajo: presión = capacidad ×
  (cola_local − τ × cola_vecino).
- **τ (turn ratio)** — peso del término del vecino (0 = ignorarlo; 1 = acoplamiento pleno).
- **link interno / benSch** — el tramo de ~97 m entre Benavides y Schell (edge `279893875#1`); donde
  se relocaliza el tapón.
- **larcoS** — la cola de **entrada** sur a Benavides (edge `129466113#0`).
- **semillas 42–51** — diez réplicas con distinto patrón aleatorio de llegada; misma demanda.
- **régimen capacidad-limitado** — llega más tráfico del que el corredor evacúa; el cuello de botella
  manda.

---

## 7. Supuestos

- **Régimen capacidad-limitado** con **Schell como cuello estructural**: validado en Etapa 1 (la
  demanda extra que cruza Benavides queda clavada en ~1650–1700 veh/h y no llega a Schell). Toda la
  interpretación depende de este régimen; en uno no saturado el resultado podría diferir.
- **Modelo de demanda supuesto** (no calibrado con campo): hora pico S→N, totales Larco 1800 /
  Benavides 800 / Schell 500 veh/h; una sola configuración OD. Reemplazable cuando haya conteos reales.
- **Alcance del corredor**: dos cruces controlados (Benavides, Schell). Diez Canseco descarga libre
  (sin transversal entrante) → no se controla.
- **Único enlace de acoplamiento activo**: Benavides→Schell (el link interno). Schell→Diez Canseco no
  acopla porque Diez Canseco es sumidero libre.
- **Simplificación del "downstream compartido"** (MP de red): el link interno recibe ~89 % de su flujo
  del recto de Larco y ~11 % de giros de la fase TRANSV; el término solo se le restó a LARCO, así que
  la mis-atribución es ≤~11 %. Suficientemente chico para que el resultado negativo se lea como régimen
  capacidad-limitado y no como artefacto.
- **Warmup de 600 s**: los primeros 600 s de cada corrida (de 1800 s) se descartan para medir en
  régimen estacionario, no en el llenado inicial.

---

## 8. Configuraciones (verificadas contra los archivos)

- **Restricciones de tiempo (MTC)** — `mtc_constraints.py`: verde mínimo 7 s, verde máximo 60 s,
  amarillo 3 s, todo-rojo 2 s, peatonal mínimo 7 s, **ciclo máximo 120 s**.
- **Ruteo del motor** — `adaptive_engine.py`: si la suma de flujos de las fases < **1500 veh/h** usa
  Webster (método clásico off-peak); si ≥ 1500 usa **Max Pressure** (peak). En estas corridas pico el
  ruteo cae en Max Pressure.
- **Ciclo** — variable: el motor elige el largo por ciclo (base Webster; si es inviable, **60 s** por
  defecto). No se impuso ciclo fijo (salvo en el experimento 4.3, descartado).
- **Tiempo perdido** — el adaptador pasa **lost_time = 10 s** por ciclo (amarillo + todo-rojo de 2
  fases).
- **Capacidad (saturación)** — `SAT_PER_LANE = 1800 veh/h/carril`. Benavides LARCO 2 carriles=3600,
  TRANSV 4=7200; Schell LARCO 2=3600, TRANSV 3=5400.
- **Demanda** — `demand_params.yaml`, `scale = 1.0` (determinista): Larco S→N 1800, Benavides 800
  (Este 500 + Oeste 300), Schell 500 (solo Este). ~3100 veh/h en total.
- **Corridas** — `peak_s_n.sumocfg`: inicio 0, fin **1800 s**, paso 1 s; **warmup 600 s**; semillas
  **42–51**.
- **MP de red** — link downstream de Benavides-LARCO = `279893875#1` (Benavides→Schell); τ barrido en
  {0, 0.5, 0.75, 1.0}. El término es **opcional y aditivo** en el motor: con τ=0 / sin downstream, la
  decisión es **byte-idéntica** al per-node (retrocompatibilidad probada con tests golden y con la
  reproducción exacta semilla a semilla).

---

## 9. Conclusión y trabajo futuro

**Conclusión.** En el corredor Larco, bajo hora pico sobre-saturada, **el control adaptativo simple
(Max Pressure per-node) es el óptimo**: −15.7 % de demora vs fijo. Toda forma de "ser más listo"
—coordinar (onda verde), rigidizar (ciclo fijo) o mirar al vecino (MP de red, en todo el rango de
τ)— **no mejora o empeora**. La razón es estructural: el cuello es Schell; cualquier estrategia que
retenga o sincronice aguas arriba termina **moviendo la cola a la entrada**, donde el conductor espera
más. La lección general: en un corredor capacidad-limitado con un cuello claro, la gestión local de
cola le gana a la coordinación. El **barrido de capacidad** (§4.6) lo cierra con número: la descarga
por Schell está topada en ~1740 veh/h, el punto de operación cae justo en esa rodilla, y el per-node ya
entrega ese techo — lo arreglable por temporizado (~13–16 %) ya está arreglado; lo que queda, pasada la
capacidad, es irreducible y ningún control lo evita.

**Por qué el resultado es confiable.** Comparación pareada por semilla; el control de referencia
reproduce su número histórico **exacto al dígito** (el flujo de autos es idéntico entre brazos); el MP
de red pasó un **pre-flight** que verificó **en ejecución** que el término realmente operaba (atrapó,
de hecho, un servidor desactualizado que lo ignoraba y habría dado un falso "no cambia nada"); y los
veredictos se sostienen con dos pruebas estadísticas independientes (IC pareado + Wilcoxon).

**Trabajo futuro (no imprescindible).**
- **Max Pressure de red riguroso (por-movimiento)**: la versión probada resta la cola del link interno
  solo a la fase LARCO; la formulación completa de Varaiya también pondería los **giros de la
  transversal** que alimentan ese link (~11 % del flujo). Improbable que cambie el signo dado el
  régimen, pero cerraría la pregunta.
- **Demanda calibrada con campo** (conteos y tiempos reales de la Subgerencia) en lugar del modelo
  supuesto, y una corrida de sensibilidad a la saturación transversal.
- **Otros regímenes**: el resultado es específico de la saturación con cuello en Schell; un escenario
  no saturado podría reordenar el ranking.
