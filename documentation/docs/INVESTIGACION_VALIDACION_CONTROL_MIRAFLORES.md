# Validación del control adaptativo sobre la red completa de Miraflores

> **Naturaleza del documento.** Reporte de investigación del experimento de control
> fijo-vs-adaptativo sobre la red completa de Miraflores (DHU-030, fases F1–F10, rama
> `feature/validacion-red-completa`). **Todas las cifras salen de artefactos versionados**
> (`multiseed_54tls_resultado.{csv,json}`, `f4_{franjas_global,zonas_kpi}.csv`,
> `f4_resultado.json`, `mapeo_pmu_edges_v2.yaml`, `miraflores.net.xml`). Las tablas embebidas
> son la fuente para los gráficos del informe final. Donde una cifra no proviene de un
> artefacto, se marca explícitamente.

---

## 0. Resumen ejecutivo

Se validó el motor de control adaptativo de semáforos de CerebroVial sobre la **red completa
de Miraflores** (net v2, 99 semáforos), comparándolo contra el control de tiempos fijos bajo
una demanda laborable realista de día completo (~33 557 vehículos, seed051). El control
adaptativo reduce la demora de red (`net_timeLoss`) en **+20,67 % ± 1,81** (media ± SD sobre
10 semillas pareadas), con **10/10 semillas favorables**, intervalo de confianza 95 %
**[+21,30 %, +24,21 %]** (no cruza cero) y Wilcoxon **p = 0,00195**. El alcance del control son
los **54 de 99 semáforos** que son de dos fases (~55 % de la red); los 45 restantes (31
mono-fase, 14 multifase) quedan fijos por las razones de §3. La medición por zona sobre los 3
cruces saturados del PMU que caen en el alcance reveló el hallazgo más fino del trabajo: el
beneficio **no es uniforme** — se concentra en el cruce más saturado (arequipa_angamos, E/H,
**+30,7 %**) y tiene un **costo local de fluidez** en dos cruces secundarios D/F, aunque el
tiempo de cola (waitingTime) **mejora en los tres**.

---

## 1. Antecedentes — la línea evolutiva del trabajo

La validación cuantitativa del motor recorrió tres escalas crecientes. Las dos primeras están
desarrolladas en detalle en
[INVESTIGACION_MUROS_VALIDACION_CONTROL.md](INVESTIGACION_MUROS_VALIDACION_CONTROL.md); acá se
las menciona con cifras para encuadrar este reporte, sin re-desarrollarlas.

**(a) Intersección aislada (TTH-07) — banco de validación de la integración TraCI↔motor.** La
primera prueba se hizo sobre un cruce sintético de cuatro ramas con dos fases. El adaptativo
**empeoraba** en hora pico (+83 % / +137 % de demora) y empataba fuera de pico. El diagnóstico
mostró que el problema no era el control sino la **medición**: el sensor leía en una ventana
fija de 30 s mientras el ciclo adaptativo variaba de duración — *aliasing* entre la ventana de
sensado y el ciclo. Corregido (el sensor pasó a promediar sobre el ciclo anterior completo). La
conclusión metodológica: **un nodo aislado bajo demanda estable es el peor escenario para
mostrar valor**, porque ahí los tiempos fijos ya están cerca del óptimo y no hay vecindad que
gestionar.

**(b) Corredor Larco (IE05 / DHU-027) — tres cruces, Max Pressure per-node.** Sobre tres cruces
consecutivos reales (Diez Canseco, Schell, Benavides). Reveló dos cosas. Primero, que una
**métrica censurada** escondía el beneficio: medido solo sobre vehículos completados daba un
empate (+1,0 % ± 7,5 %, 4/10), pero la métrica robusta puerta-a-puerta dio **+15,68 % ± 8,07**
(9/10 semillas, IC [+7,61 %, +23,75 %]). Segundo, que la mejora viene de **relocalizar la
cola**: la espera para *entrar* al corredor cae 67 % (36,8 → 12,2 s) mientras el tiempo *adentro*
casi no cambia (+2,5 %). Dos extensiones se **descartaron** con contundencia estadística: la
**onda verde** (offset óptimo = 0; 7/8 desfases empeoraron entre +50 % y +123 %) y el **Max
Pressure de red** (mirar la cola del vecino: +35,07 s/corrida, Wilcoxon p = 0,002, 0/10
favorables; **−9,2 % vs el fijo**, contra el +15,7 % del per-node). Ambos descartes son
antecedentes directos del trabajo futuro (§9).

**(c) Miraflores red completa (este reporte) — la extensión a escala distrito.** El recorrido
nodo → corredor → red es la línea natural: cada muro empujó al siguiente paso. Este experimento
lleva la validación de 3 cruces a **99 semáforos**, donde la estructura de vecindad es de grado
mayor que en una cadena lineal.

---

## 2. Diseño experimental

**Red.** Miraflores v2 (`simulation/conf/network/miraflores.net.xml`): **99 semáforos**,
**1664 edges vehiculares** (no internos), componente conexo mayor estimado **1660** nodos
(fuente: `mapeo_pmu_edges_v2.yaml`, campos `n_traffic_light`, `n_edges_total_no_internas`,
`lcc_n_estimado`). Es el net reconstruido en el PR #44 (2026-06-03), distinto del net previo
(la cifra "47 cruces" de D-012 corresponde al net viejo y no se reusa).

**Demanda.** Día laborable B2 **seed051** (`routes/miraflores_seed051_laborable.rou.xml`),
**33 557 vehículos cargados** (`runs/*/metrics.json`, campo `loaded`), **scale 1.1** (gate
D-014), perfil laborable de **10 fases horarias**. Corresponde al "lunes 8 de junio" de la demo
(seed051 = day_idx 9; mapeo `2026-06-08 ↔ seed051` en
`core_management_api/src/congestion/.../routes.py`). Detalle y regeneración determinística en el
[README del experimento](../../simulation/conf/miraflores_red_completa/README.md) y en
`REGENERACION_DEMANDA_B2.md`.

**Dos brazos, misma demanda, única diferencia = el control.**
- **Fijo** — los `tlLogic` de netconvert embebidos en el net, **tal cual**. Se los nombra
  honestamente como **tiempos heurísticos de netconvert**: ni planes de campo, ni Webster
  optimizado, sino una red **sin optimizar** (un tercer punto). Sintetizar el fijo con Webster
  fresco simularía una red optimizada a hoy (irreal para Miraflores) y compararía el adaptativo
  contra una versión de sí mismo (Webster es su estrategia base).
- **Adaptativo** — los **54 semáforos de dos fases**, control **per-node** vía TraCI alimentado
  por el motor (`/control`), que decide ciclo/reparto con Webster / Max Pressure / MTC. Ciclo
  variable (el motor decide el ratio), sin offset (per-node puro, sin coordinación cableada).

**Métrica.** `net_timeLoss` = demora media por vehículo sobre **toda** la corrida (tiempo
perdido respecto de viajar a velocidad libre), leída de `tripinfo`. **Por qué no door-to-door:**
en IE05 hizo falta una métrica robusta a censura porque el corredor saturaba y dejaba autos sin
insertar; acá el **diagnóstico de censura dio ~0 %** (1 vehículo never-inserted de 33 557 en
ambos brazos, **0,003 %**), así que `net_timeLoss` no está censurado y basta.

**Protocolo.** Multi-seed **pareado**: 10 semillas (42–51), cada una corre **ambos** brazos con
la **misma** demanda regenerada (comparación pareada; única diferencia = el control), día
completo **86 400 s**. Estadística estilo IE05: vector de Δ pareados, IC 95 % por t de Student
(df = 9), Wilcoxon signed-rank exacto.

**Coherencia con el mapa del frontend.** El brazo fijo es **bit-a-bit el mismo régimen** (net +
`tlLogic` + seed051 + `time-to-teleport=300` + `collision.action=warn`) con que se generó el
dataset de 60 días que alimenta el mapa de congestión. Es decir: **el mapa muestra el baseline
que el adaptativo mejora**.

---

## 3. Alcance de control: por qué 54 de 99

El núcleo del adaptador per-node asume **dos fases por nodo**. Clasificando los 99 semáforos por
número de fases verdes en su `tlLogic` (verificado parseando `miraflores.net.xml`):

| Clase | Fases verdes | # TLS | Tratamiento |
|---|---|---:|---|
| **Bifásico** | 2 | **54** | **Controlado** (adaptativo per-node) |
| Mono-fase | 1 | 31 | Fijo — sin fases en conflicto, nada que optimizar |
| Multifase | 3–4 | 14 | Diferido — derivación con solape de movimientos |

(Los 14 multifase son 2 con 3 fases verdes + 12 con 4.) Los **mono-fase** no tienen fases en
conflicto: no hay nada que un control de reparto pueda optimizar, quedan fijos por definición.
Los **multifase** requieren resolver la atribución de edges cuando un movimiento aparece como
protegido en una fase y permisivo en otra (solape) antes de poder derivar el reparto — ingeniería
adicional, diferida a trabajo futuro (§9).

**Cruces saturados del PMU (nivel de servicio E/F) y cobertura.** El *Plan de Movilidad Urbana
de Miraflores 2017-2020* marca el nivel de servicio (LOS) de las intersecciones. La escala es
**A–F** (A = flujo libre, F = saturación); el código del campo `nivel_los` es **`X/Y` = pico
AM / pico PM**, y aparece un código **`H`** no estándar que denota **saturación extrema** (peor
que F; coincide siempre con notas "crítico"/"muy saturada" en el YAML). Considerando saturado a
todo cruce con E, F o H en algún pico, de los cruces PMU semaforizados saturados:

**3 caen entre los 54 bifásicos (medibles, este experimento):**

| Cruce | LOS (AM/PM) | junction_id | aproches (edges) |
|---|---|---|---|
| arequipa_angamos | E/H | `cluster_108177589_..._#2more` | `1023395811#2`, `1148678976#1`, `435657506#1`, `784296866#0` |
| 28julio_lapaz | D/F | `133936090` | `-427240760#3`, `653650077#9` |
| ricardopalma_paseo | D/F | `cluster_114612822_..._#2more` | `1148860369`, `437344447#1`, `653439619#0` |

**5 quedan FUERA del alcance — zonas críticas no cubiertas** (se declaran explícitamente):

| Cruce | LOS | Clase | Por qué fuera |
|---|---|---|---|
| pardo_espinar | F/H | multifase (4) | requiere derivación multifase |
| espinar_angamos | F/F | multifase (4) | requiere derivación multifase |
| paseo_angamos | E/H | multifase (4) | requiere derivación multifase |
| paseo_benavides | D/E | multifase (4) | requiere derivación multifase |
| 28julio_reducto | E/H | mono-fase (1) | sin fases en conflicto (nada que optimizar) |

Las dos rotondas del gazetteer PMU (`ovalo_miraflores`, `ovalo_gutierrez`) tienen `nivel_los`
nulo y no entran en el cómputo de saturación.

---

## 4. Resultado global

Fuente: `multiseed_54tls_resultado.json` / `.csv` (F5, 10 seeds pareados, día completo).

- **RD% (reducción de demora) = +20,67 % ± 1,81** (media ± SD, 10 seeds).
- **Δ pareado (fijo − adaptativo) = +22,76 s ± 2,04**, IC 95 % **[+21,30 s, +24,21 s]** (df = 9,
  t = 2,262) — **no cruza cero**.
- **Wilcoxon signed-rank exacto: W⁺ = 55,0, p = 0,00195** (dos colas).
- **10/10 semillas favorables** al adaptativo.
- Umbral de éxito del proyecto (RD% ≥ 15 %, heredado de IE05/DHU-027): **superado con holgura**.

**Tabla completa por semilla** (`net_timeLoss` medio, s):

| seed | fijo | adaptativo | Δ (f−a) | RD% | teleports f→a |
|---:|---:|---:|---:|---:|---:|
| 42 | 109,0906 | 86,1945 | 22,8961 | +20,99 % | 0 → 5 |
| 43 | 109,4819 | 85,6827 | 23,7992 | +21,74 % | 1 → 2 |
| 44 | 112,8021 | 87,6478 | 25,1543 | +22,30 % | 31 → 0 |
| 45 | 112,8461 | 91,0879 | 21,7582 | +19,28 % | 22 → 5 |
| 46 | 106,5200 | 84,8437 | 21,6763 | +20,35 % | 1 → 12 |
| 47 | 105,5364 | 82,2710 | 23,2654 | +22,04 % | 5 → 0 |
| 48 | 104,9806 | 82,9745 | 22,0061 | +20,96 % | 2 → 0 |
| 49 | 113,7515 | 94,7965 | 18,9550 | +16,66 % | 8 → 5 |
| 50 | 111,4221 | 89,5921 | 21,8300 | +19,59 % | 5 → 1 |
| 51 | 115,2048 | 88,9665 | 26,2383 | +22,78 % | 19 → 10 |

**Salud de las corridas.** Censura ~0 % en ambos brazos (1 never-inserted de 33 557 por corrida,
0,003 %). Teleports agregados: **fijo Σ = 94, adaptativo Σ = 40** (el adaptativo teleporta menos).
Vehículos completados ~33 525 por corrida (99,9 %).

---

## 5. Desglose por franja horaria

Fuente: `f4_franjas_global.csv` (40 filas: 10 seeds × 4 franjas) y `f4_resultado.json`
(agregado). Franjas ancladas a la curva B2: **pico AM 07–09 h** (25 200–32 400 s), **valle
10–17 h** (36 000–61 200 s), **pico PM 18–20 h** (64 800–72 000 s), **día completo** (0–86 400 s).

**Agregado (10 seeds, `net_timeLoss` medio/veh, s):**

| franja | fijo | adaptativo | RD% | ± SD |
|---|---:|---:|---:|---:|
| pico AM 07–09 | 122,51 | 98,16 | **+19,79 %** | 4,42 |
| valle 10–17 | 101,10 | 78,56 | **+22,29 %** | 0,48 |
| pico PM 18–20 | 125,76 | 102,54 | **+18,51 %** | 3,37 |
| **día completo** | 110,16 | 87,41 | **+20,67 %** | 1,81 |

**Interpretación.** El **valle es la franja más fuerte y más estable** (+22,29 % ± 0,48): en
carga media el re-timing tiene espacio para optimizar. Los **picos rinden algo menos y con más
dispersión** (AM +19,79 % ± 4,42; PM +18,51 % ± 3,37): en saturación todo está al límite y hay
menos margen para que el control ayude. El día completo (+20,67 %) coincide exactamente con el
titular de F5 — confirma que el reproceso por franja mide el mismo experimento.

**Aproximación documentada.** Las franjas globales agrupan a cada vehículo por su **cohorte de
salida** (`tripinfo_depart`), no por dónde acumula la demora: un auto que sale al final de una
franja derrama demora a la siguiente. Es la aproximación inevitable con `tripinfo` (registra un
único `depart` por vehículo, no su traza temporal). Las franjas **por zona** (§6) no tienen esta
limitación.

<details><summary>Tabla por seed × franja (40 filas) — fuente <code>f4_franjas_global.csv</code></summary>

| seed | franja | fijo | adaptativo | RD% |
|---:|---|---:|---:|---:|
| 42 | AM | 121,18 | 97,32 | +19,69 % |
| 42 | valle | 100,83 | 77,99 | +22,65 % |
| 42 | PM | 125,33 | 101,18 | +19,27 % |
| 42 | día | 109,09 | 86,19 | +20,99 % |
| 43 | AM | 120,73 | 95,25 | +21,11 % |
| 43 | valle | 100,57 | 78,26 | +22,19 % |
| 43 | PM | 125,42 | 100,48 | +19,88 % |
| 43 | día | 109,48 | 85,68 | +21,74 % |
| 44 | AM | 134,31 | 99,33 | +26,04 % |
| 44 | valle | 100,15 | 78,10 | +22,02 % |
| 44 | PM | 127,05 | 102,93 | +18,98 % |
| 44 | día | 112,80 | 87,65 | +22,30 % |
| 45 | AM | 126,65 | 99,29 | +21,61 % |
| 45 | valle | 100,28 | 78,42 | +21,80 % |
| 45 | PM | 139,68 | 120,40 | +13,81 % |
| 45 | día | 112,85 | 91,09 | +19,28 % |
| 46 | AM | 111,97 | 88,06 | +21,35 % |
| 46 | valle | 100,73 | 78,19 | +22,38 % |
| 46 | PM | 116,65 | 97,82 | +16,15 % |
| 46 | día | 106,52 | 84,84 | +20,35 % |
| 47 | AM | 111,29 | 91,35 | +17,92 % |
| 47 | valle | 100,97 | 78,62 | +22,14 % |
| 47 | PM | 116,78 | 88,32 | +24,37 % |
| 47 | día | 105,54 | 82,27 | +22,04 % |
| 48 | AM | 116,73 | 91,70 | +21,44 % |
| 48 | valle | 101,35 | 77,77 | +23,27 % |
| 48 | PM | 111,65 | 92,67 | +17,00 % |
| 48 | día | 104,98 | 82,97 | +20,96 % |
| 49 | AM | 125,24 | 112,23 | +10,39 % |
| 49 | valle | 101,98 | 80,00 | +21,55 % |
| 49 | PM | 131,14 | 112,53 | +14,19 % |
| 49 | día | 113,75 | 94,80 | +16,66 % |
| 50 | AM | 121,84 | 103,49 | +15,07 % |
| 50 | valle | 102,19 | 79,16 | +22,53 % |
| 50 | PM | 130,17 | 105,55 | +18,91 % |
| 50 | día | 111,42 | 89,59 | +19,59 % |
| 51 | AM | 135,14 | 103,63 | +23,32 % |
| 51 | valle | 101,92 | 79,13 | +22,36 % |
| 51 | PM | 133,68 | 103,52 | +22,56 % |
| 51 | día | 115,20 | 88,97 | +22,78 % |

</details>

---

## 6. KPI por zona — los 3 cruces PMU saturados

Fuente: `f4_zonas_kpi.csv` (120 filas: 10 seeds × 3 zonas × 4 franjas) y `f4_resultado.json`
(agregado). Medición vía `<edgeData>` (freq 3600 s) sobre los 9 edges-aproche, agregada a las
franjas. **A diferencia de las franjas globales, esto es exacto por franja**: edgeData mide el
intervalo de tiempo, no la cohorte de salida. RD% > 0 = el adaptativo mejora.

**Agregado (10 seeds). Demora total y por-vehículo en s, RD% ± SD; throughput de contexto.**

| zona (LOS) | franja | TL total RD% ±SD | TL/veh RD% ±SD | wait/veh RD% ±SD | thru f→a |
|---|---|---:|---:|---:|---:|
| **arequipa_angamos** (E/H) | AM | +28,88 ±3,55 | +28,96 ±3,54 | +40,23 ±3,88 | 1006 → 1007 |
| | valle | +30,58 ±2,57 | +30,57 ±2,59 | +41,92 ±2,66 | 2031 → 2030 |
| | PM | +29,93 ±4,23 | +30,02 ±4,07 | +41,38 ±4,26 | 1047 → 1049 |
| | **día** | **+30,67 ±2,33** | **+30,68 ±2,33** | **+42,07 ±2,47** | 5934 → 5935 |
| **28julio_lapaz** (D/F) | AM | −3,48 ±14,54 | −2,86 ±14,16 | +28,71 ±12,86 | 300 → 302 |
| | valle | −23,37 ±7,17 | −23,43 ±7,19 | +12,88 ±8,04 | 604 → 604 |
| | PM | −12,69 ±28,49 | −12,46 ±28,56 | +22,25 ±22,86 | 311 → 312 |
| | **día** | **−17,51 ±9,40** | **−17,49 ±9,40** | **+16,99 ±8,85** | 1783 → 1783 |
| **ricardopalma_paseo** (D/F) | AM | −19,12 ±16,95 | −18,90 ±16,90 | +14,65 ±14,94 | 1072 → 1074 |
| | valle | −17,58 ±8,06 | −17,57 ±8,03 | +14,76 ±6,61 | 2192 → 2192 |
| | PM | −11,93 ±15,01 | −11,56 ±14,70 | +22,30 ±12,88 | 1114 → 1117 |
| | **día** | **−13,98 ±9,03** | **−13,95 ±9,03** | **+19,35 ±7,64** | 6384 → 6385 |

**Niveles (día completo, contexto para magnitud).** arequipa: TL total 71 791 → 49 731 s,
TL/veh 12,10 → 8,38 s, wait/veh 8,77 → 5,07 s. 28julio: TL total 9 569 → 11 218 s, TL/veh
5,37 → 6,29 s, wait/veh 3,75 → 3,10 s. ricardopalma: TL total 21 674 → 24 602 s, TL/veh
3,39 → 3,85 s, wait/veh 2,49 → 2,00 s. (Per-seed completo en `f4_zonas_kpi.csv`.)

**El hallazgo central.**

**(a) arequipa_angamos (E/H) domina.** El cruce más saturado del alcance mejora **+30,67 % ±
2,33** en timeLoss (día), parejo en las 4 franjas y con SD ajustada. TL/veh ≈ TL total (porque
el throughput está pareado), así que no es efecto de volumen. **El sistema concentra el
beneficio donde la saturación es peor.**

**(b) Los dos cruces D/F tienen costo local en timeLoss.** 28julio_lapaz **−17,51 %** y
ricardopalma_paseo **−13,98 %** en el día, con **signo negativo consistente en los 10 seeds** —
no es ruido de una semilla. La **SD es ancha** (9–28) porque son cruces chicos (TL total un
orden de magnitud menor que arequipa): las celdas de pico son **no concluyentes** (SD > |media|,
p.ej. 28julio PM −12,69 ± 28,49), pero el día completo está a ~1,5–1,9 SD de cero (sugestivo de
un costo real, no demostrado a alta confianza).

**(c) El trade-off cola-vs-fluidez (lo destrabó waitingTime).** En **waitingTime los tres cruces
mejoran** (+13 % a +42 % por vehículo), **incluso los dos D/F donde el timeLoss empeora**. La
lectura precisa: el re-timing **reduce el tiempo parado en cola en todos los cruces**; en los
D/F el tiempo *total* empeora porque los autos pasan más tiempo **moviéndose lento** (verdes más
cortos/frecuentes → menos cola pero más stop-and-go). Se cambia **espera-detenido por
circulación-lenta**, no se penaliza la cola. Es un matiz fino, no una contradicción.

**Chequeo anti-artefacto.** El **throughput está pareado** entre brazos en las 120 celdas
(divergencia máxima 10 vehículos sobre ~1000) — confirma que las diferencias de demora son
**reasignación real de verde, no artefacto de volumen**.

---

## 7. Caracterización del control: Webster online, no Max Pressure

Confirmado multi-seed (`multiseed_54tls_resultado.json`, `modes_total`): de **1 468 193**
decisiones de control sobre los 10 seeds, solo **23 fueron Max Pressure** (1 468 170 Webster) —
**0,0016 %**. El umbral de activación de Max Pressure (~1500 vph por intersección) rara vez se
cruza con la demanda **repartida sobre 1660 aristas**, a diferencia de Larco, donde el pico está
concentrado en un eje. **Implicación honesta:** el +20,67 % es **re-timing Webster ganándole al
fijo de netconvert**, no coordinación ni Max Pressure. A la demanda real de Miraflores, el
sistema opera como **Webster adaptativo per-node**. El resultado valida el lazo
adaptativo-online, no el término Max Pressure (que queda sin ejercer a esta demanda).

---

## 8. Limitaciones y honestidad metodológica

1. **Alcance 54/99 (~55 % de la red).** 5 cruces PMU saturados quedan sin cubrir (4 multifase +
   1 mono-fase; §3). El resultado no habla de esas zonas críticas.
2. **Costo local en los 2 cruces D/F** (§6b): el adaptativo per-node, sin visión de red, deja
   que cruces secundarios cedan verde al neto global.
3. **Aproximación por cohorte en las franjas globales** (§5): agrupa por `depart`, no por dónde
   se acumula la demora. Las franjas por zona (§6) sí son exactas.
4. **Un solo día-tipo.** seed051 es **laborable**; no se probó fin de semana ni feriado.
5. **Baseline fijo = heurístico netconvert**, no planes de campo 2014 ni Webster optimizado
   (§2). Es un punto "sin optimizar", rotulado como tal.
6. **Max Pressure no ejercido a esta demanda** (§7): el resultado valida **Webster online**, no
   Max Pressure.

---

## 9. Trabajo futuro

**(a) Control coordinado / aprendido (RL / GNN).** Motivado empíricamente por **tres hallazgos
convergentes**: el muro de MP-red en Larco (coordinación cableada con τ fijo empeora en régimen
saturado, −9,2 %), Max Pressure inactivo a la demanda de Miraflores (el per-node no explota
coordinación), y el costo local en los D/F (un control con visión de red podría evitar que
cruces secundarios cedan). La diferencia con MP-red: **no coordinación cableada con τ fijo, sino
política aprendida**.

**(b) Ingeniería de derivación multifase.** Los 14 semáforos de 3–4 fases requieren resolver la
atribución de edges cuando un movimiento es protegido en una fase y permisivo en otra (solape)
antes de poder controlarlos. Es el camino para subir del 55 % hacia el 100 % de la red, e
incorporaría 4 de los 5 cruces saturados hoy fuera de alcance.

**(c) Días no-laborables.** Fin de semana y feriado, para caracterizar el control fuera del
perfil laborable.

**(d) Validación estructural contra el PMU.** Contrastar los niveles de servicio medidos contra
los del *Plan de Movilidad Urbana 2017-2020*.

---

## Apéndice — Artefactos versionados (procedencia de las cifras)

| Artefacto | Qué aporta |
|---|---|
| `simulation/conf/miraflores_red_completa/multiseed_54tls_resultado.{csv,json}` | Resultado global F5 (§4): RD%, Δ, IC, Wilcoxon, modos, por-seed |
| `simulation/conf/miraflores_red_completa/f4_franjas_global.csv` | Franjas globales por seed (§5) |
| `simulation/conf/miraflores_red_completa/f4_zonas_kpi.csv` | KPI por zona por seed (§6) |
| `simulation/conf/miraflores_red_completa/f4_resultado.json` | Agregados + hallazgos en prosa (§5, §6) |
| `simulation/conf/miraflores_red_completa/zonas_pmu_f4.json` | Definición de zonas, edges-aproche, franjas (§3, §6) |
| `documentation/contracts/mapeo_pmu_edges_v2.yaml` | Métricas del net (99 TLS, 1664 edges, LCC 1660), LOS PMU (§2, §3) |
| `simulation/conf/network/miraflores.net.xml` | Clasificación de fases 31/54/14 (§3, verificada por parseo) |
| `documentation/docs/INVESTIGACION_MUROS_VALIDACION_CONTROL.md` | Detalle de TTH-07 y Larco/IE05 (§1) |
| `documentation/lean-inception/4-decisiones/DECISIONS_HU.md` (DHU-030) | Alcance y cierre del experimento |

**Cómo se regeneran los consolidados:** `python simulation/scripts/reprocess_f4_zonas.py`
(lee `runs/f4_*` + `zonas_pmu_f4.json`). Los crudos (tripinfo/edgeData) viven en `runs/`
(gitignoreado).
