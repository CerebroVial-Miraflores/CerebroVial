# Corredor Larco — Etapa 2 / IE05: cierre (baseline fijo + barrido de demanda + MP adaptativo vs fijo)

**Rama:** `feature/corredor-larco-max-pressure`. Sin push, sin PR, sin merge.
**Alcance:** ejecución del experimento sobre la red base de Etapa 1. **No toca el motor**
(`core_management_api/` intacto; consumido por HTTP en `:8001`). Solo `simulation/` y este doc.
**Escenario fijado:** demanda peak S→N, `scale=1.0`, `seed` variable, `end=1800`, `warmup=600`,
mismos 18 detectores E2, mismo `.sumocfg`. La única variable entre fijo y adaptativo es el control.

## Resultado de IE05 (honesto)

**IE05 (DHU-027) exige RD% ≥ 15% sobre demora de RED. NO se alcanza.**
El RD% (reducción de demora de red, post-warmup, comparación pareada) es un **empate
estadístico**: **+1.0% ± 7.5% (SD)** sobre 10 semillas (media±SD = −6.5%…+8.5%, cruza el cero;
4/10 semillas positivas, la media positiva la arrastra un único outlier). La adaptación local
per-node, sola, **no reduce la demora de red** en este corredor.

Pero el experimento **no es un nulo**: revela un mecanismo estructural robusto (10/10 semillas)
que reformula el hallazgo y motiva el siguiente paso (coordinación). Ver §"Mecanismo".

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
optimización de offsets podría atacar y que bajo fijo no existía. **IE05 no cumple su umbral
≥15%, pero demuestra empíricamente por qué la coordinación es el siguiente paso necesario.**

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

- **Coordinación (offsets Benavides↔Schell):** siguiente paso natural — el mecanismo robusto dice
  que ahí está el valor que la adaptación local no captura.
- **IE05 ≥15% no alcanzado a nivel nodo-local.** Reabrir requeriría: sensibilidad a la saturación
  transversal, o evaluar el RD% bajo control coordinado (no per-node aislado).
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
