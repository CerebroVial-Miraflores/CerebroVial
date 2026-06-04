# Muros encontrados en la validación del control adaptativo: del nodo aislado al corredor coordinado

## Encuadre

Este documento reúne las limitaciones —los "muros"— que aparecieron al validar el motor de control adaptativo de CerebroVial contra semáforos de tiempos fijos en simulación SUMO. Es importante leerlo con su alcance correcto: **toda la validación realizada hasta la fecha es sobre *partes* del mapa de Miraflores, nunca sobre la red completa.** Primero sobre una intersección sintética aislada, después sobre un corredor real de tres cruces (Av. José Larco). La extensión a Miraflores completo (99 cruces semaforizados) es trabajo pendiente y queda fuera de lo que aquí se reporta.

Un muro, en este contexto, no es un fracaso: es un resultado. Cuando una corrida "sale mal", el valor está en explicar *por qué* salió mal y *qué enseña* sobre el problema. Algunos de los muros que siguen resultaron ser límites físicos reales del tránsito; otros, errores de medición que escondían un sistema que en realidad funcionaba. Distinguir unos de otros es el aporte central de este capítulo. Todas las cifras se recomputaron desde los datos crudos de las simulaciones (archivos `tripinfo`/`summary` por corrida), no de resúmenes previos.

## 1. El primer muro: el nodo aislado no se puede medir bien (TTH-07)

La primera validación se hizo sobre una intersección sintética de cuatro ramas, un cruce de manual con dos fases (Norte-Sur y Este-Oeste). El resultado fue desalentador: el control adaptativo **empeoraba** el tránsito respecto a los tiempos fijos en hora pico, y empataba fuera de pico.

Las cifras son contundentes. En el pico de la mañana, el tiempo de viaje subió de 56,5 a 68,7 segundos (+21,6%) y la demora acumulada casi se duplicó (+83%). En el pico de la tarde fue peor: el tiempo de viaje pasó de 57,0 a 79,0 segundos (+38,6%) y la demora se multiplicó por más de dos (+137%). La cola máxima en el approach Norte llegó a triplicarse. Fuera de pico, en cambio, el adaptativo y el fijo eran indistinguibles (diferencias de −3% y −0,8%, dentro del ruido).

A primera vista, esto condenaba al motor. Pero el diagnóstico reveló otra cosa: **el problema no estaba en el control, sino en cómo se lo medía.** El sensor leía el estado de la intersección en una ventana de tiempo fija (30 segundos), mientras que el ciclo del semáforo adaptativo variaba de duración. Cuando la ventana de medición se desalineaba del ciclo, la lectura de presión colapsaba y el motor "hambreaba" sistemáticamente a una de las fases. Era un problema de *aliasing*: la herramienta de medición batía contra el fenómeno medido y producía basura.

Este es el primer tipo de muro: **un muro de instrumentación.** No enseña nada sobre si el control adaptativo sirve; enseña que para medir un control de ciclo variable hay que sensar *sobre el ciclo completo*, no sobre una ventana de reloj fija. La corrección de este muro es lo que habilitó la segunda etapa.

## 2. La corrección, y el segundo muro escondido detrás de una métrica (corredor Larco)

Resuelto el aliasing —ahora el sensor promedia sobre el ciclo anterior completo— la validación se mudó a un escenario realista: el corredor de la Av. José Larco, tres cruces consecutivos (Diez Canseco, Schell, Benavides) importados de la geometría real de OpenStreetMap. Aquí apareció un segundo muro, más sutil que el primero, y de naturaleza completamente distinta.

La primera medición del corredor dio un **empate**: el control adaptativo mejoraba apenas +1,0% respecto al fijo, con una dispersión que cruzaba el cero (±7,5%, solo 4 de 10 simulaciones favorables, intervalo de confianza [−6,5%, +8,5%]). Estadísticamente, nada. El motor no parecía aportar valor.

El problema, de nuevo, no era el motor: era la métrica. La medición original contaba el tiempo perdido **solo de los vehículos que completaban su viaje dentro de la red**. Pero en un corredor saturado, el control de tiempos fijos deja muchos autos que *nunca llegan a entrar* — se quedan en la cola de inserción y la simulación termina sin que arranquen. Esos autos no aparecen en las estadísticas de viajes completados. La métrica vieja, en efecto, *premiaba* al control que dejaba más gente afuera, porque no contaba sus esperas.

La magnitud del sesgo es grande: el control fijo abandonaba en promedio **68 autos sin insertar**; el adaptativo, solo **23**. Al corregir la métrica para que cuente puerta a puerta —la espera para entrar al corredor más el tiempo adentro— el resultado se da vuelta por completo:

Misma data, dos métricas. Con la métrica censurada: **+1,0%** (empate). Con la métrica robusta puerta a puerta: **+15,68% ± 8,07, 9 de 10 simulaciones favorables**, intervalo de confianza [+7,61%, +23,75%] que ya no cruza el cero. El sistema siempre había funcionado; la métrica vieja no lo veía.

Este es el segundo tipo de muro: **un muro de medición.** Enseña que en régimen de saturación, cualquier métrica de demora que ignore a los vehículos que no logran entrar está sistemáticamente sesgada a favor del peor control. Es una lección metodológica que trasciende este proyecto.

## 3. El resultado positivo —y el muro físico que esconde (IE05)

Con la métrica corregida, el corredor Larco es el resultado positivo del trabajo: el control adaptativo (Max Pressure por nodo, de ciclo variable) reduce la demora de red en **+15,7%** frente a tiempos fijos, cumpliendo el umbral de éxito del proyecto (RD% ≥ 15%). El cumplimiento es ajustado —la media menos una desviación estándar cae por debajo del 15%— y así se reporta, sin inflarlo.

Pero el desglose de *dónde* viene esa mejora revela el tercer muro, y es el más importante de todos porque es físico, no instrumental. La mejora de +15,7% se compone así: la espera para *entrar* al corredor se desploma un 67% (de 36,8 a 12,2 segundos), mientras que el tiempo *adentro* del corredor prácticamente no cambia (+2,5%, de 104,5 a 107,1 segundos).

Dicho en criollo: **el control adaptativo no elimina la congestión, la relocaliza.** Vacía la cola de la entrada —por eso entran más autos y la espera de ingreso cae— pero esa demanda extra que ahora circula se acumula río abajo, en el tramo interno entre Benavides y Schell. La cola se mueve de lugar; no desaparece. Esto se confirma en las 10 simulaciones sin excepción: en todas, el adaptativo drena la entrada y a la vez llena el link interno.

Este es el muro físico, y es un resultado válido y esperable: en un corredor donde el cuello de botella estructural está aguas abajo (la capacidad de descarga de Schell), ningún control local puede crear capacidad que no existe. Lo mejor que puede hacer es administrar *dónde* espera la gente. Que ese reacomodo dé un saldo neto positivo de +15,7% es el aporte real; pretender que "elimina" la congestión sería deshonesto.

## 4. Dos intentos de derribar el muro físico, ambos descartados

Si la cola se relocaliza al tramo interno, la pregunta natural es: ¿se puede atacar esa cola relocalizada coordinando los semáforos? Se exploraron dos formas de hacerlo. Ninguna funcionó, y entender por qué es instructivo.

**Onda verde (coordinación de offsets).** La idea clásica: sincronizar los semáforos para que un pelotón de autos encuentre verdes encadenados. Se barrió el desfase entre cruces de 0 a 80 segundos. El resultado fue inequívoco: **el desfase cero —no coordinar— es el óptimo.** Siete de los ocho desfases probados empeoraron la demora entre 50% y 123% (el peor, un desfase de 30 segundos, la empeoró 122,8%). Solo un desfase (60 segundos) quedó en casi-empate (−3,0%), pero ninguno mejoró. Y los autos sin insertar se dispararon: de 3 con desfase cero a 171 con desfase 30. El mecanismo físico es claro: el tramo interno entre cruces es muy corto (97 metros); sincronizar el pelotón hacia el cruce siguiente lo llena *más rápido*, agravando el desborde en lugar de aliviarlo. En un corredor sobresaturado con links cortos, lo que manda es gestionar la cola, no progresar el pelotón.

**Mirar al vecino (Max Pressure de red).** El segundo intento fue más sofisticado: que cada cruce, al decidir, tenga en cuenta la cola del tramo de aguas abajo (que Benavides "vea" que el link hacia Schell está lleno y retenga). Se extendió el motor con ese término y se corrió en paralelo contra el control por nodo independiente. **Refutado, y con contundencia estadística:** mirar al vecino *empeora* la demora de red en +35,07 segundos por simulación en promedio (intervalo de confianza [+21,6, +48,6], que no incluye el cero; test de Wilcoxon p=0,002), y **ninguna de las 10 simulaciones** favoreció a la versión de red. Frente al fijo, mirar al vecino rinde −9,2% (peor que el fijo), contra el +15,7% del control por nodo. ¿Por qué? Retener en Benavides para no llenar el tramo interno simplemente devuelve la cola a la entrada: la espera de ingreso se cuadruplica (de 12,2 a 48,2 segundos) mientras el tiempo adentro queda igual. Es el mismo muro físico otra vez —la cola se relocaliza— pero ahora hacia el lado peor.

Estos dos descartes son **decisiones de diseño fundamentadas**, no fracasos. Documentan que, en este régimen, la coordinación entre cruces no solo no ayuda sino que perjudica, y por una razón física entendible. El término de "mirar al vecino" quedó implementado en el código, apagado por defecto, disponible por si un régimen distinto (demanda menor, links más largos) lo justificara.

## 5. El muro estructural que motivó todo el recorrido

Conviene cerrar con el muro que, cronológicamente, vino primero como intuición y se confirmó al final: **no se puede estimar la mejora del control adaptativo sobre un solo semáforo aislado.** La ventaja de Max Pressure es estructuralmente una propiedad de *red* —depende de la presión que un cruce ejerce sobre sus vecinos aguas abajo— y en un nodo aislado ese término es cero. Sin vecinos, Max Pressure se vuelve matemáticamente indistinguible de un control fijo bien sintonizado.

Esto es lo que motivó el salto de la intersección aislada (TTH-07) al corredor (Larco): no fue un capricho, fue una necesidad estructural. Y es también lo que motiva el paso siguiente, todavía pendiente: validar sobre la red completa de Miraflores, donde la estructura de vecindad es de grado mayor que en una cadena lineal de tres cruces. El recorrido nodo → corredor → red es la línea natural del trabajo, y cada muro empujó al siguiente paso.

## Apéndice — Tabla de muros por categoría

| Categoría | Caso | Qué pasó | Qué enseña | Veredicto |
|---|---|---|---|---|
| Instrumentación | TTH-07, nodo aislado | Adaptativo empeora en pico (+83%/+137% demora) | Medir control de ciclo variable exige sensar sobre el ciclo, no ventana fija | Artefacto corregido en Larco |
| Medición | "Empate +1,0%" de IE05 | Métrica censurada esconde el beneficio (+1,0% vs +15,7% real) | En saturación, la métrica debe contar autos que no logran entrar | Artefacto; métrica corregida |
| Físico / régimen | IE05, relocalización de cola | La cola se mueve de la entrada al link interno, no se elimina (10/10) | Ningún control local crea capacidad aguas abajo; administra dónde se espera | Resultado válido, +15,7% neto |
| Diseño | Onda verde / offsets | Coordinar empeora 50–123%; offset=0 óptimo | Links cortos sobresaturados: sincronizar llena el tramo antes | Descartado con fundamento |
| Diseño | Mirar al vecino (MP-red) | Empeora +35s (p=0,002, 0/10); devuelve la cola a la entrada | Coordinar aguas abajo relocaliza al lado peor en este régimen | Descartado; código apagado |
| Estructural | Nodo aislado vs red | MP en nodo aislado = fijo bien sintonizado | La ventaja de MP es de red; exige vecindad para medirse | Motivó nodo → corredor → red |
