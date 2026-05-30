# Corredor Larco — Etapa 2 / IE05 Fase 1: cierre (baseline fijo + barrido de demanda + MP adaptativo vs fijo)

**Rama:** `feature/corredor-larco-max-pressure`. Sin push, sin PR, sin merge.
**Alcance:** ejecución del experimento sobre la red base de Etapa 1. **No toca el motor**
(`core_management_api/` intacto; consumido por HTTP en `:8001`). Solo `simulation/` y este doc.
**Escenario fijado:** demanda peak S→N, `scale=1.0`, `seed` variable, `end=1800`, `warmup=600`,
mismos 18 detectores E2, mismo `.sumocfg`. La única variable entre fijo y adaptativo es el control.

> **⚠️ ACTUALIZACIÓN (resultado final de IE05): el "empate" de abajo era un artefacto de
> medición.** Con la métrica de demora correcta (puerta-a-puerta robusta a censura) la
> adaptación local **SÍ alcanza el umbral: RD% RED = +15.7% ± 8.1 (9/10 semillas)**. Ver
> §"Actualización: el empate era de medición". La narrativa de Fase 1 de abajo se conserva como
> el recorrido honesto (medimos mal → corregimos → el sistema rinde); lo que cambia es la
> conclusión, no los hechos.

## Resultado (IE05 Fase 1 — adaptación local) — lectura ORIGINAL, corregida más abajo

**IE05 sigue EN PROGRESO. Lo que cierra esta etapa es su Fase 1 (adaptación local per-node), no
el indicador entero.** IE05 (DHU-027) exige RD% ≥ 15% sobre demora de RED; **medido con la
métrica vieja (timeLoss DENTRO de la red, solo vehículos completados) la adaptación local, sola,
parecía NO alcanzar el umbral**: RD% = **empate estadístico — +1.0% ± 7.5% (SD)** sobre 10
semillas (media±SD = −6.5%…+8.5%, cruza el cero; 4/10 positivas). **Esta lectura quedó SUPERADA**
(la métrica escondía el beneficio — ver actualización).

**La Fase 2 (coordinación de offsets / onda verde) se ejecutó y quedó DESCARTADA** (offset=0
óptimo; coordinar perjudica en este régimen sobresaturado — ver actualización). El mecanismo
estructural robusto (10/10 semillas, ver §"Mecanismo") sigue siendo el hallazgo físico que
sustenta el resultado.

## Actualización: el empate era de medición (resultado final de IE05)

El RD% de Fase 1 se midió con **timeLoss dentro de la red, solo sobre vehículos completados**.
Esa métrica está **censurada**: en saturación, el control fijo deja autos que **nunca llegan a
insertarse** (se quedan en la cola de entrada y la simulación termina sin que arranquen) →
**no aparecen en `tripinfo`**. El control que mete más autos "carga" con sus esperas; el que los
deja afuera parece más rápido. Magnitud (media 10 semillas): el fijo abandona **68** autos sin
insertar, el MP adaptativo solo **23**. La métrica vieja no contaba a los abandonados del fijo →
empate espurio.

**Métrica corregida (puerta-a-puerta robusta a censura).** Tiempo total del conductor = espera
para ENTRAR (la cola de inserción) + tiempo DENTRO de la red. Para que la censura no engañe se
usa el **tiempo total en el sistema** vía Little's law: `W = mean(running + waiting)_post-warmup
÷ λ` (λ = tasa de generación), que cuenta a **todos** los autos —los que circulan y los que
esperan entrar— sin depender de quién completa. Es la experiencia real del conductor, no un
cambio de vara.

**Resultado final IE05 — RED (número titular):**

| métrica | RD% (media ± SD, 10 semillas) | ¿cruza 0? | semillas >0 |
|---|---|---|---|
| vieja (timeLoss en red, completados) | +1.0% ± 7.5 | SÍ → empate | 4/10 |
| **puerta-a-puerta robusta (RED)** | **+15.7% ± 8.1** | **NO → señal** | **9/10** |
| puerta-a-puerta (CORREDOR S→N, complementario) | +18.1% ± 9.5 | NO | 9/10 |

Misma corrida, misma comparación pareada — **la única diferencia es cómo se mide**. El número
titular es **RED (+15.7%)**: IE05 está definido sobre demora de red, es la experiencia de todos
los conductores (no solo los de la avenida) y es el más robusto a la censura (cuenta a todos). El
**corredor S→N (+18.1%)** va como complementario, con la salvedad de que es menos robusto a la
censura (solo completados + ratio de completación). Se encabeza con el más conservador y
conceptualmente correcto a propósito.

**Framing honesto del +15.7%:** reducción media del 15.7% sobre 10 réplicas, **cumpliendo el
objetivo del 15% en promedio** — pero es un **cumplimiento ajustado, no holgado**: con la
dispersión (±8.1 SD; media−SD = +7.6%, per-semilla de −0.6% en la 44 a +29.8% en la 42, mediana
≈16%), el umbral del 15% cae **dentro** del margen. Es una **mejora robusta y significativa
(9/10 semillas, lejos de cero)**, con la dispersión entre réplicas reportada explícitamente.

**El beneficio es físico (desglose espera vs adentro, media de las semillas):**

```
ESPERA para entrar:  fijo  36.8 s → adaptativo  12.2 s   (−67 %)
MANEJAR adentro:     fijo 104.5 s → adaptativo 107.1 s   (+2.5 %, tapón relocalizado)
```

La historia para la defensa: **el sistema no descongestiona la calle (el tiempo adentro queda
casi igual — el tapón se relocaliza al link interno, ver §"Mecanismo"), pero hace esperar mucho
menos para entrar al corredor.** Ese desglose blinda contra el "cambiaste la vara": el +15.7% no
sale de redefinir nada, sale de contar la espera de entrada y los autos abandonados que la
métrica vieja ignoraba.

### Fase 2 (onda verde) y ciclo fijo — explorados y DESCARTADOS

Sobre el mismo adaptador (cero core), se probó coordinar Benavides↔Schell con **ciclo común fijo
90 s + offset** (la demora relocalizada al link interno era el blanco natural):

- **Onda verde (offset) — DESCARTADA.** Barrido offset 0–80 (seed 42, ciclo fijo 90): **offset=0
  (sin coordinar) es óptimo**; todo offset ≠0 empeora la demora robusta (+50…+123%) y **dispara
  autos sin insertar** (off=0 deja 3 afuera; off 20–30 dejan 117–171). En un corredor
  **sobresaturado** con link interno corto (96.9 m), sincronizar el pelotón hacia Schell llena el
  link más rápido y agrava el spillback; lo que manda es la gestión de cola, no la progresión.
- **Ciclo común fijo (offset=0) — DESCARTADO.** Multi-semilla (10): ciclo-fijo vs fijo +12.2% ±
  10.6 (9/10), pero **vs MP de ciclo variable −5.3% ± 18.8** (cruza 0, 4/10, dispersión enorme;
  seed 51 −47.5%). La ventaja aparente en seed 42 (W 99.5 vs 122.4) **no generalizó** —
  caso de manual de por qué se exige multi-semilla. Fijar el ciclo le quita al motor la libertad
  de elegir su largo; entre semillas a veces ayuda y a veces lo arruina.

**Sistema final de IE05: MP per-node de ciclo VARIABLE** (el motor decide ciclo y split por
presión, sin coordinación), **+15.7% RED**. La adaptación local **alcanza el umbral en la media**
(marginal); coordinación de offsets y ciclo fijo quedan descartados.

### Cómo reproducir la métrica corregida (re-proceso, NO re-corre SUMO)

```bash
cd simulation
# IE05 (10 semillas) y barrido de offsets, con métrica puerta-a-puerta robusta:
.venv/bin/python scripts/reprocess_door2door.py            # ie05 + offsets
.venv/bin/python scripts/reprocess_door2door.py fixedcycle # confirmación ciclo fijo
```

La métrica vive en `kpis/collect_corredor.py` (campos `w_red_door2door_s_postwarm` con desglose
espera/adentro, y bloque de censura loaded/inserted/never_inserted). Las corridas ciclo-fijo se
generan con `scripts/sweep_offsets.py fixedcycle` (reusa el adaptador, offset=0).

## Qué se ejecutó (3 hallazgos encadenados)

1. **Baseline fijo (gate de spillback).** Demanda OD explícita (11 pares, verificados contra la
   tabla `<connection>`): Larco S→N 1800, Benavides 800 (E+O), Schell 500 veh/h (`scale=1.0` =
   3100 veh/h). Control = tlLogic estático del net.xml (defaults de netconvert). Bajo fijo: la
   entrada sur a Benavides satura (~99%, cola media 217 m) pero **NO hay spillback inter-cruce**
   sostenido; throughput 2788 veh/h, completion 0.90, teleports 0. Congestión funcional, sin
   acoplamiento Benavides→Schell.

2. **Barrido de demanda (¿el desacople es de demanda o estructural?).** Escala global 1.0→1.5.
   El flujo que cruza Benavides hacia el link interno queda **clavado en ~1650–1700 veh/h** en
   todos los niveles; la demanda extra se acumula como backlog de inserción en la entrada sur
   (wait 72→520) y **nunca llega a Schell**. Ningún nivel produce acoplamiento sostenido; ni
   siquiera gridlockea (teleports 0 hasta 1.5). **Desacople ESTRUCTURAL confirmado:** el cuello
   está aguas arriba (Benavides descarga < capacidad de Schell), no es problema de demanda.

3. **MP adaptativo per-node vs fijo (IE05).** Benavides y Schell controlados por el motor
   (POST `/control/recommend`, Max Pressure / Webster según el motor; MP elegido en la gran
   mayoría de ciclos bajo peak). Diez Canseco queda fijo (paso libre, sin conflicto que
   adaptar). Sensado ciclo-alineado (ventana = ciclo completo, decisión en borde de ciclo) →
   sin el aliasing de TTH-07. 10 semillas pareadas → RD% empate (arriba) + mecanismo robusto.

## Mecanismo (10/10 semillas — el hallazgo robusto)

| | semillas |
|---|---|
| Drenaje del cuello de Benavides (cola media entrada sur: adaptativo < fijo) | **10/10** |
| Relocalización al link interno Benavides→Schell (pico benSch: adaptativo > fijo) | **10/10** |
| Ambos a la vez | **10/10** |

En **todas** las semillas: MP reasigna verde a la fase Larco en Benavides → la cola media de
entrada sur cae (≈216 m fijo → 115–163 m adaptativo) → ese flujo extra llega a Schell (binding
constraint estructural) → el link interno Benavides→Schell se llena (≈20–28 m fijo → 88–96 m,
~99%) y **aparece el spillback inter-cruce que la demanda sola no pudo producir**. Throughput
sube de forma consistente (+1…+3%, 10/10).

**Lectura:** la adaptación local **conserva y relocaliza** la demora (de la entrada al link
interno), no la elimina; y al hacerlo **convierte un corredor desacoplado (fijo) en uno acoplado
(spillback Benavides→Schell)**. La demora relocalizada al link interno es precisamente lo que una
optimización de offsets podría atacar y que bajo fijo no existía. **La adaptación local no alcanza
el umbral ≥15%, pero demuestra empíricamente por qué la coordinación (Fase 2 de IE05) es el
siguiente paso necesario.**

## Diferencias de control declaradas (no confounds ocultos)

- **all-red:** el programa adaptativo incluye `all_red=2s` por fase (MTC), que el fijo de
  netconvert no tiene (solo amarillos) → ~278 s de tiempo perdido estructural extra para el
  adaptativo. Reportado junto al RD% bruto: el RD%≈0 se compone de un beneficio de asignación de
  verde enmascarado por la relocalización a Schell + este overhead.
- **Saturación transversal de Benavides** (parámetro del motor, calibrable): modelada como
  4 carriles (Este 2 generales —carril 0 bus— + Oeste 2) = 7200 veh/h, la elección que NO
  sub-estima la transversal. Sensible: candidata a corrida de sensibilidad (7200 vs 3600) si se
  reabre IE05.
- **Diez Canseco:** sin transversal entrante (deuda de Etapa 1) → paso libre, queda con control
  fijo. El acoplamiento se ejercita en el único link interno con conflicto (Benavides→Schell).

## Parámetros físicos y supuestos

- `vType car` (length 5, maxSpeed 13.89). Carril 0 del eje Larco = bus/bici (excluido de la
  capacidad de descarga del tráfico general → 2 carriles efectivos para Larco). `SAT_PER_LANE`
  = 1800 veh/h/carril (parametrizable; el cálculo a mano del baseline usó 1900 — no es confound,
  el control fijo no consume `saturation_flow`).
- Cola para MP = **media del ciclo** (no instantánea en el borde: tendría sesgo de fase a favor
  de Larco). Parametrizable por si se quiere una corrida de sensibilidad con la instantánea.
- node_id: `larco_benavides`, `larco_schell` (ya en el seed; sin tocar core).

## Deudas y follow-ups

- **IE05 alcanzado en la media con adaptación local (marginal).** Con métrica puerta-a-puerta
  robusta, el MP per-node de ciclo variable da **RD% RED +15.7% ± 8.1** (9/10) — cumplimiento
  **ajustado** del umbral 15%. Onda verde (offsets) y ciclo común fijo **explorados y descartados**
  (ver §"Actualización"). Sistema final = MP per-node de ciclo variable.
- **Mirar-al-vecino (network-aware MP) — EXPLORACIÓN, plan-first (toca el motor).** Como el +15.7%
  es ajustado, sumar conciencia del vecino aguas abajo (presión del link interno Benavides→Schell
  en la decisión de Benavides) podría correr el cumplimiento de **marginal a holgado** —segunda
  razón concreta, además del interés teórico. No imprescindible; si se aborda, va plan-first
  porque modifica el core del motor.
- Sensibilidad a la saturación transversal de Benavides (7200 vs 3600) queda como variante opcional.
- **Una sola configuración de demanda** (peak S→N supuesto). Tiempos fijos y matriz OD siguen
  SUPUESTOS (Etapa 1); reemplazo cuando la Subgerencia provea tiempos 2014 + conteos.
- Diez Canseco sin conflicto transversal (deuda de Etapa 1, sin cambios).

## Reproducibilidad

```bash
cd simulation
# 1. Baseline fijo (gate de spillback)
bash scripts/run_corredor_baseline.sh --seed 42 --end 1800 --warmup 600
# 2. Barrido de demanda (desacople estructural)
.venv/bin/python scripts/sweep_demand.py 1.0 1.1 1.2 1.3 1.4 1.5
# 3. Adaptativo vs fijo (requiere motor: `invoke up` en la raíz, :8001)
bash scripts/run_corredor_adaptive.sh --seed 42 --end 1800 --warmup 600
# 4. Robustez multi-semilla (IE05)
.venv/bin/python scripts/sweep_seeds.py 42 43 44 45 46 47 48 49 50 51
```
Outputs en `data/corredor_larco/` (gitignored, regenerables y determinísticos por semilla).

## Commits de la rama (Etapa 2)

- `ETAPA 2 F1: demanda peak S→N + detectores spillback para corredor Larco`
- `ETAPA 2 F2: runner headless + collector de KPIs/spillback del corredor`
- `ETAPA 2: ignorar simulation/data/corredor_larco (outputs regenerables del barrido)`
- `ETAPA 2: herramienta de barrido de demanda (--scale + sweep_demand.py, exploración)`
- `ETAPA 2 F1 (IE05): adaptador Max Pressure per-node + runner + comparador`
- `ETAPA 2 (IE05): barrido multi-semilla del RD% adaptativo vs fijo (herramienta)`
- `ETAPA 2 (IE05): cierre — baseline + desacople estructural + RD% empate + mecanismo 10/10`
- `F1 onda verde: ciclo común fijo + offset en el adaptador`
- `Métrica puerta-a-puerta robusta a censura + re-proceso de KPIs`
- `Confirmación ciclo fijo: subcomandos fixedcycle (runner + comparación)`
