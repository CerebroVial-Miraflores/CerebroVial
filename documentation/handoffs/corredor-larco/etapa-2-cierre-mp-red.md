# Corredor Larco — Etapa 2 / MP de red (downstream del link interno): cierre — REFUTACIÓN

**Rama:** `feature/corredor-larco-mp-red`. Sin push, sin PR, sin merge.
**Resultado titular:** el **Max Pressure de red** (darle al motor la cola del link interno
Benavides→Schell como término downstream, `P(φ) = s·(x_local − Σ τ·x_down)`) **NO mejora** sobre el
MP per-node en este corredor — **lo empeora**, y el hallazgo es estadísticamente claro. La hipótesis
de "mirar-al-vecino" (follow-up abierto en DHU-027 / IE05) queda **refutada** para este régimen.
**Alcance del cambio:** Etapa 1 extendió el motor (`core_management_api`, opcional/aditivo, retrocompat
bit-a-bit, ya cerrada — ver commit `ac7f1c83` + tests golden). Etapa 2 (este doc) cableó el adaptador
(`simulation/`) y corrió el experimento. El motor en `:8001` corre el código de Etapa 1 (deployado vía
rebuild del contenedor; ver nota de runtime).

## Hipótesis y diseño

**Hipótesis (a confirmar/refutar):** como el per-node relocaliza el tapón al link interno
Benavides→Schell (`279893875#1`, ~99% en 10/10 semillas de IE05), darle a Benavides la cola de ese
link debería hacerlo **retener** cuando el link está lleno → menos tapón interno → ¿menos demora neta?

**Diseño:** 3 brazos pareados por semilla 42–51, misma demanda determinista (`scale=1.0`, `end=1800`,
`warmup=600`), única variable = el control.
- **fijo** — control estático del net.xml.
- **per-node** — MP simplificado `P=s·q` (downstream OFF). Reproduce IE05.
- **MP-red** — MP de red `P=s·(q − τ·q_down)` (downstream ON; Benavides lee `279893875#1`, **τ=1.0**).

**Veredicto = delta pareado** `Δ = W_RED(MP-red) − W_RED(per-node)` por semilla (RED puerta-a-puerta
robusta, `w_red_door2door_s_postwarm`). Se decide por si el **IC del Δ excluye el cero** (t pareado
df=9) + **Wilcoxon signed-rank exacto** — no por medias marginales (que se solapan) ni por conteo a ojo.
`Δ<0` ⇒ MP-red mejor.

## Resultado — REFUTACIÓN (el downstream EMPEORA la demora RED)

`Δ` por semilla (s): `[39.1, 31.5, 37.8, 31.3, 48.8, 20.2, 2.0, 18.7, 66.4, 54.9]`

| estadístico | valor |
|---|---|
| media Δ | **+35.07 s** ± SD 18.85 (SE 5.96) |
| IC 95% pareado (t, df=9) | **[+21.59, +48.55] → EXCLUYE el 0** |
| Wilcoxon signed-rank exacto | W+ = 55.0, **p (two-sided) = 0.0020** |
| semillas con Δ<0 (MP-red mejor) | **0 / 10** |

Ambos métodos (paramétrico y no-paramétrico) coinciden: el efecto es real y va en contra de la
hipótesis. **MP-red sube la demora RED ~35 s respecto al per-node.**

**W_RED por brazo (contexto — las medias marginales se solapan, NO son el veredicto):**

| brazo | W_RED (media ± SD) | RD% vs fijo |
|---|---|---|
| fijo | 141.37 s ± 2.79 | — |
| per-node | 119.26 s ± 12.11 | **+15.7 %** |
| MP-red | 154.33 s ± 16.56 | **−9.2 %** |

MP-red no solo pierde contra per-node: queda **peor que el fijo**.

## No-regresión de IE05 (compuerta dura — PASÓ)

El brazo per-node (downstream OFF) reproduce el W_RED histórico de IE05 **semilla a semilla, exacto al
dígito** (no "dentro de ±8.1"):

```
seed 42 122.4=122.4 · 43 121.6=121.6 · 44 141.1=141.1 · 45 122.0=122.0 · 46 114.8=114.8
seed 47 123.6=123.6 · 48 129.9=129.9 · 49 110.6=110.6 · 50 109.8=109.8 · 51  96.8= 96.8
agregado per-node = 119.26 = histórico 119.26   →  +15.7% vs fijo reproducido
```

Como el motor es bit-idéntico sin downstream (Etapa 1) y el harness/configs/SUMO/demanda no cambiaron,
el match exacto confirma que el veredicto del delta pareado mide **solo** el efecto del término
downstream, no una deriva del harness.

## Mecanismo medido (el trade-off — por qué empeora, y por qué NO es artefacto)

El término **sí opera** y **sí alivia** el link interno; el costo neto es el problema:

| señal (media de semillas) | per-node → MP-red | lectura |
|---|---|---|
| tapón interno benSch (mean) | 59.7 → **46.7 m (−22 %)** | ✓ alivia el link, como predijo la hipótesis |
| tapón interno benSch (max) | 94.9 → 89.6 m | apenas baja el pico |
| espera para entrar `w_wait` | **12.2 → 48.2 s (×4)** | ✗ explota la espera de inserción |
| adentro de red `w_inside` | 107.1 → 106.2 s | ≈ plano |
| entrada Benavides `larcoS` (mean) | 144.6 → **172.2 m** | cola **relocalizada a la entrada** |

**Mecanismo:** régimen **capacidad-limitado** con Schell como **binding constraint** estructural (la
demanda extra que Benavides empuja no la absorbe Schell). Darle a Benavides la cola del link interno lo
hace **retener** → alivia un poco `#1` (−22% mean) **pero devuelve la cola a la entrada sur de Benavides**
(larcoS 144.6→172.2 m) y **cuadruplica la espera para entrar** (12.2→48.2 s). Como la métrica
puerta-a-puerta cuenta esa espera, el neto es peor. Retener arriba no sube throughput; solo mueve la
cola a un lugar que la métrica (correctamente) penaliza más.

**No es el no-op de cableado (b'):** el pre-flight probó que el término opera (payload fiel + decisiones
que cambian) y benSch efectivamente bajó. El empeoramiento es **real**, no un cable muerto.

**Señal bang-bang (τ=1.0):** en MP-red benSch tiene `max ≫ mean` (89.6 vs 46.7, gap ~43 m) más marcado
que en per-node (94.9 vs 59.7, gap ~35 m) → la ocupación de `#1` burstea más, consistente con el umbral
de switcheo duro (`x_local = x_down`) que anticipamos para τ=1.0: Schell drena → Larco positivo → sirve
→ `#1` se llena → suprime → repite.

## Nota metodológica — el pre-flight atrapó un artefacto (fortaleza del método)

Antes del barrido, un **pre-flight de 1 semilla / 3 brazos con aserciones EN EJECUCIÓN** descartó el
modo de falla que el análisis de fin de sweep **no puede** ver: un `Δ≈0` tiene dos causas
indistinguibles desde la tabla — (a) null real (Schell binding, retener no ayuda) y **(b') no-op de
cableado** (el término no enchufa → MP-red corre idéntico al per-node → `Δ=0` artefacto leído como null).

El pre-flight **cazó un (b') real**: el contenedor `core_management_api` en `:8001` llevaba 31 h
corriendo (imagen pre-Etapa-1). Aceptaba el payload con `downstream` (Pydantic descarta campos extra)
pero **lo ignoraba** → A1 (payload fiel a `#1`) pasaba, pero **A2 (≥1 decisión cambia) fallaba: 0/34**,
y W_RED salía idéntico al per-node. Sin el pre-flight, las 10 semillas habrían dado `Δ≈0` y lo
habríamos reportado como **null falso**. Fix: rebuild del contenedor con el código de Etapa 1 (ya
commiteado; sin migración) → verificado que el término opera (flip de fase end-to-end) → pre-flight
verde (A0 per-node exacto, A1 payload fiel 35/35 con queue>0, A2 32/34 decisiones cambian) → barrido.

Esto se documenta como **fortaleza del método** (la disciplina de verificar el cable en ejecución, no
solo el agregado), no como error a esconder. Herramienta: `simulation/scripts/preflight_mp_red.py`.

## Matiz dimensionado — downstream compartido (declarado, no asertado)

`x_local` (`129466113#0`) es de **un solo sentido** S→N → la resta `x_local − τ·x_down` es limpia.
Pero `#1` también recibe **giros de la fase TRANSV** (`344159559#2` der., `406007422#0` izq.), así que
x_down(#1) es un downstream **compartido**. Split de inflow medido en la demanda: **~89% LARCO-recta /
~11% giros de TRANSV** (1530 vs 190 veh/h de 1720). Restarlo solo a LARCO mis-atribuye ≤~11% → la
simplificación es menor y el null/refutación se lee como régimen capacidad-limitado, no como artefacto
de TRANSV inundando `#1`. El término por-movimiento (Varaiya riguroso) queda como future work.

## Conclusión y decisión

- **MP de red (downstream del link interno) DESCARTADO en este corredor.** Empeora la demora RED por
  relocalización de cola a la entrada en régimen capacidad-limitado (IC pareado excluye cero, Wilcoxon
  p=0.002, 0/10 favorables). **Sistema adoptado = MP per-node (τ=0), +15.7% RED** (IE05, sin cambios).
- **El motor conserva la capacidad** (término opcional/aditivo, retrocompat bit-a-bit): no se elimina,
  queda disponible y desactivado por defecto. La extensión es código probado, no deuda.
- **Future work:** (1) término **por-movimiento** (Varaiya riguroso) que también pondere los giros de
  TRANSV que alimentan `#1`; (2) barrido **τ ∈ {0.5, 0.75}** por si un acoplamiento más débil cambia el
  signo (improbable dado el régimen, pero acota la sensibilidad). Ninguno imprescindible.

## Reproducibilidad

```bash
cd simulation
# motor vivo en :8001 CON Etapa 1 deployada (invoke up-build --service=core_management_api)
# 1. Pre-flight (1 semilla, 3 compuertas de cableado) — ANTES del barrido
.venv/bin/python scripts/preflight_mp_red.py 42
# 2. Barrido 3-brazos pareado (veredicto: delta pareado + IC + Wilcoxon + no-regresión + trade-off)
.venv/bin/python scripts/sweep_seeds_downstream.py 42 43 44 45 46 47 48 49 50 51
```

Outputs en `data/corredor_larco/` (gitignored, regenerables y determinísticos por semilla). El brazo
per-node fresco va a `peak_s_n_mpred_pernode_seed*` (no pisa el histórico `peak_s_n_adaptive_seed*`,
que sirve de referencia para la no-regresión).

## Nota de runtime

`:8001` corre ahora `feature/corredor-larco-mp-red` (Etapa-1 deployada vía rebuild del contenedor
`core_management_api`). La ruta **sin** downstream es byte-idéntica al per-node (probado: no-regresión
semilla a semilla exacta); **sin migración de BD** (el campo `downstream` solo agrega una clave al
`inputs_snapshot` JSON). **Sin mergear a master.**

## Commits de la rama (Etapa 2)

- `feat(simulation): MP de red — downstream wiring del adaptador + preflight de cableado + sweep pareado 3-brazos (corredor Larco)`
- `docs(handoffs): cierre Etapa 2 MP de red — refutación + trade-off medido + nota de pre-flight`

(Etapa 1, motor — rama misma: `feat(control): término downstream opcional en Max Pressure (MP de red, Varaiya); ruta sin-downstream bit-idéntica al per-node, retrocompat IE05 cubierta por golden`.)
