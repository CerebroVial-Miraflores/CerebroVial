# C3 — Recalibración del scale para `miraflores_v2.net.xml` (distrito completo). ELEGIDO: 1.1

**Fecha:** 2026-06-03 · scratch (corridas), doc versionado. Semilla=42 (igual que C2).
**Cierra:** B2 (recalibración del scale tras reconstruir el net al distrito completo).

## Contexto y objetivo

El net pasó de `miraflores.net.xml` (viejo, 381 edges vehiculares, 51 lane-km) a
`miraflores_v2.net.xml` (distrito completo: **1664 edges ≈ 4.4×**, **260 lane-km ≈ 5.1×**,
LCC ~1660 nodos). El **scale=0.20 de C2 es inválido para v2**: estaba calibrado a la
capacidad del net viejo. Con ~5× de calzada cargando la misma demanda (el conteo de trips es
función de `scale` solamente — `scale · Σ(eh-bh)·vph` = `scale · 30500`, **net-independiente**),
v2 drena holgado a 0.20. Hay que **re-localizar el cliff** y elegir el scale de operación (C3).

## Método

Idéntico a C2: 24h continua headless, **control fijo (sin TraCI)**, **seed 42**,
`edgeData freq=60`, criterio = ¿la red **drena**? Maquinaria promovida y parametrizada por
`--net` en B2/F0 (`simulation/scripts/sweep24.sh` + `analyze24.py` + `generate_b1_demand.py`).
Barrido en tres etapas:
- **Grueso** `{0.50, 0.35, 0.20, 0.10}` → los cuatro drenan holgado (cliff > 0.50).
- **Grueso-arriba** `{1.5, 1.0, 0.75}` → cliff entre 1.0 y 1.5.
- **Fino** `{1.3, 1.2, 1.1}` → cliff entre 1.2 y 1.3 (rodilla gradual).

## ⚠️ HALLAZGO CRÍTICO — el criterio binario de C2 NO es trasladable a v2

El veredicto **`racha ≥3h sub-8 km/h`** de `analyze24.py` está calibrado al **colapso del net
viejo**, donde la velocidad se **clavaba a 3–4 km/h** sostenido. **v2 colapsa "suave" en km/h
ponderada por viaje**: en colapso toca **~11 km/h, no 3–4**. Resultado: el binario marca
**"drena" incluso a 1.5**, que colapsa con **920 teleports** y **+43% de duración de viaje**.

**Para v2 el drenaje NO se lee de la racha sub-8.** Se decide por el **conjunto** de señales:
1. **Teleports** (la más limpia): salto de ~0–decenas (drena) a **centenas** (colapso).
2. **Δduración de viaje**: salto material sobre el baseline ~254s (≤1.0).
3. **Dip bimodal sostenido**: dips **sub-20 km/h en franjas anchas** AM/PM (vs incipiente/ausente).

Esto es **deuda de método** a tener presente en B3/B4, **formalizada en el ADR D-014**: el umbral de
drenaje es **net-específico**; `analyze24.py` necesita un veredicto recalibrado (o multi-señal)
para v2. (`racha sub-8` se deja intacta en el código por compatibilidad con C2; se interpreta
con este caveat.)

## Tabla completa del barrido (10 scales, v2, seed 42)

| scale | inserted | teleports (tot · Jam/Yield/WL) | dur. viaje | Δdur vs 1.0 | dip bimodal | km/h glob | jam | **veredicto** |
|------:|------:|:--|--:|--:|:--|--:|--:|:--|
| 0.10 | 3 051 | 5 · 3/0/2 | 251s | −1% | ausente | 29.7 | 3 | drena holgado |
| 0.20 | 6 100 | 2 · 2/0/0 | 248s | −2% | ausente | 29.7 | 2 | drena holgado |
| 0.35 | 10 682 | 2 · 2/0/0 | 249s | −2% | ausente | 29.6 | 2 | drena holgado |
| 0.50 | 15 253 | 1 · 1/0/0 | 248s | −2% | ausente | 29.6 | 1 | drena holgado |
| 0.75 | 22 877 | 2 · 2/0/0 | 250s | −2% | ausente | 29.5 | 2 | drena holgado |
| 1.0 | 30 503 | 0 · 0/0/0 | 254s | — | ausente (→25.2) | 29.2 | 0 | drena limpio |
| **1.1** | **33 556** | **11 · 2/9/0** | **259s** | **+2%** | **incipiente (→23.7)** | **29.1** | **2** | **drena limpio ← C3** |
| 1.2 | 36 603 | 36 · 17/19/0 | 268s | +5.5% | moderado (→21.3, recupera) | 28.8 | 17 | drena (borde superior) |
| 1.3 | 39 653 | 137 · 53/84/0 | 292s | +15% | pronunciado (sub-20 ancho) | 28.6 | 53 | **colapsa (onset)** |
| 1.5 | 45 754 | 920 · 579/286/55 | 364s | +43% | fuerte (→11.2) | 27.9 | 579 | **colapsa** |

(WL = Wrong Lane. Colisiones 0–2 en todos, no sumadas al total de teleports. Insertados ≈
generados en todos: 0 descartes, `waiting=0` → v2 absorbe lo que se genera sin backlog.)

**Cliff = entre 1.2 y 1.3.** Rodilla **gradual, no escalón**: teleports 11→36→**137**→920;
Δdur +2%→+5.5%→**+15%**→+43%; el dip cruza **sub-20 km/h en franjas anchas** recién en 1.3.

## Scale elegido: C3 = 1.1 (gate de Cesar)

**1.1 es el scale más alto que drena con margen al cliff.** Justificación:
- **Margen al cliff** (que cae entre 1.2 y 1.3): 1.1 está dos pasos finos por debajo del onset
  de colapso → colchón para que la **variación de seed en B3 no cruce al colapso** (robustez).
- **Señal espacial completa**: el *excess aditivo* local (~0.13, ver abajo) **ya está completo
  a 1.1**. Subir no compra más señal local explotable.
- **1.2 evaluado como alternativa agresiva y descartado**: aporta más congestión *global*
  (más teleports, dip más profundo) pero **0 ganancia de señal local neta**, y acerca al cliff
  → **riesgo de seed-crossing sin beneficio**. 1.1 domina a 1.2 para el propósito de B3.

## Caracterización de señal espacial (insumo metodológico, NO control de resultado)

Medición **pre-entrenamiento** sobre los troncales PMU (`mapeo_pmu_edges_v2.yaml`):
serie `meanTimeLoss` por tramo, correlación de **vecinos adyacentes** (lag 0 y ±1..±3) vs
**control aleatorio no-adyacente**, en tres scales.

| scale | r_vecino (lag0) | control no-ady. | ratio vec/ctrl | excess aditivo (vec−ctrl) | lag≠0 (% pares) |
|------:|--:|--:|--:|--:|--:|
| 0.50 | 0.192 | 0.056 | 3.46× | 0.136 | bajo (lag0 dominante) |
| 1.0 | 0.227 | 0.097 | 2.35× | 0.130 | incipiente |
| 1.5 | 0.251 | 0.122 | 2.05× | 0.129 | **49%** (modo +1 = 60s) |

**Hallazgos honestos:**
- **(a) Hay señal espacial local genuina.** Vecinos > control en todos los scales; el control
  casi-nulo a baja carga (0.056) **descarta un confound diurnal global** — la varianza de
  timeLoss es local/de-junction, y la adyacencia la levanta.
- **(b) La propagación direccional con lag emerge post-cliff.** A 1.5, **49% de los pares
  vecinos pican en lag≠0**, con **modo +1 (= 60s)** = colas encadenadas tramo-a-tramo. Es la
  firma física que el STGNN modela. **Pero NO se vuelve dominante** (lag0 sigue siendo el modo
  individual #1; margen r_best−r_lag0 ≈ +0.04).
- **(c) El excess local neto NO crece con la congestión** (~**0.13 constante** en 0.50/1.0/1.5).
  Lo que crece al saturar es el componente **global sincrónico** (control 0.056→0.122), que un
  baseline por-nodo con feature de hora ya captura. Por eso el **ratio cae** (3.46×→2.05×)
  aunque r_vecino suba.

**Predicción honesta (registrada, no es un fracaso):** la ventaja explotable de un **STGNN
sobre un baseline por-nodo** es de **magnitud moderada y estable**, coherente con dominio
**arterial urbano** (no autopista; no METR-LA/PEMS-BAY con r_vecino 0.6–0.8). **El margen
STGNN vs baseline en B4 será probablemente modesto, y eso es consistente con esta
caracterización, no una falla del modelo.** Se recomienda **re-confirmar** la señal sobre el
dataset final de B3 si el scale o la demanda cambian.

## No-alcance de B2 / handoff a B3

B2 **NO** regeneró dataset, **NO** promovió v2, **NO** tocó `gen_day.sh`. Entregables de B2:
la maquinaria del sweep versionada y parametrizada por `--net` (F0), el cliff localizado y el
scale C3=1.1 (este doc).

**B3 toma C3=1.1** para alimentar `gen_day.sh` cuando promueva v2 a la ruta de net operativa.
**Deuda arrastrada a B3/B4:**
- **N del tensor STGNN cambia con v2**: el 375/N del net viejo ya no aplica (LCC ~1660 nodos);
  **el N autoritativo lo fija B3** vía `miraflores_graph_builder.build_miraflores_graph`.
- **Umbral de drenaje net-específico** (HALLAZGO CRÍTICO arriba): `analyze24.py` necesita
  veredicto recalibrado para v2 — **formalizado en D-014**.
- **Re-confirmar la caracterización de señal espacial** sobre el dataset final de B3 si el
  scale o la distribución de demanda cambian.
