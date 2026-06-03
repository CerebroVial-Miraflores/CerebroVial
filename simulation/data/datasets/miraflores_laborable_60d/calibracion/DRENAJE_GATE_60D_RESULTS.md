# Gate de drenaje D-014 — dataset v2 60d @ scale 1.1 (B3.2.c)

**Fecha:** 2026-06-03 · **Net:** `miraflores.net.xml` (v2, 1664 edges vehiculares; LCC 1660).
**Cadena:** `gen_day.sh` (randomTrips ratio 5, scale 1.1) → `batch_days.sh` seeds 42..101 (60 días,
secuencial, 56 min, exit 0, sin fallos de infra) → `evaluate_drenaje.py --no-cleanup`.
**HEAD de la cadena preparada:** `3adf7798` (c.1). **Veredicto:** scale 1.1 **ACEPTADO** (ver juicio D-014 abajo).

> **Nota de naturaleza del gate (clave, ver juicio):** sobre este dataset el gate D-014 **NO
> separa "día válido / día roto"** — los 60 días generaron bien (exit 0, sin fallos de infra).
> Separa **"congestión moderada / congestión severa"**. Los 12 "colapsos" son la **cola superior
> de la distribución de congestión de pico PM**, no días a descartar. Es un **caracterizador de
> régimen**, no un filtro de validez.

## Criterio (D-014, v2)
El día **drena** si pasa las tres señales (AND); **colapsa** si dispara cualquiera:
1. **Teleports** `≤ 50` (total de `stats.xml`). Señal primaria.
2. **Duración media de viaje** `≤ 280 s` (baseline ~254 s, +10%).
3. **Dip acotado**: fracción de intervalos de 60 s con `mean_kmh < 20` **≤ 10 %** en cada ventana
   de pico (AM 07-09h `[25200,32400)`, PM 18-20h `[64800,72000)`), ponderada por `sampledSeconds`.

Umbrales en `simulation/scripts/evaluate_drenaje.py` (`TELEPORTS_MAX=50`, `DURATION_MAX_S=280`,
`DIP_SPEED_KMH=20`, `DIP_FRAC_MAX=0.10`). Net-específicos de v2.

## Resultado — tabla de los 60 veredictos

```
seed |  tel |  dur_s |  AM frac/min |  PM frac/min | veredicto
--------------------------------------------------------------
  42 |    0 |  258.5 |  0.8%/ 19.4 |  0.8%/ 19.9 | drena
  43 |    1 |  261.0 |  0.0%/ 20.3 |  0.0%/ 20.2 | drena
  44 |   31 |  262.6 |  5.0%/ 17.7 |  0.8%/ 19.1 | drena
  45 |   22 |  263.9 |  2.5%/ 19.5 |  5.0%/ 19.3 | drena
  46 |    1 |  258.6 |  0.0%/ 21.2 |  0.0%/ 20.6 | drena
  47 |    5 |  256.2 |  0.0%/ 20.1 |  0.0%/ 20.9 | drena
  48 |    2 |  255.5 |  0.0%/ 20.9 |  0.0%/ 20.9 | drena
  49 |    8 |  266.3 |  1.7%/ 19.7 |  1.7%/ 19.6 | drena
  50 |    5 |  262.7 |  0.0%/ 20.4 |  0.8%/ 19.9 | drena
  51 |   19 |  267.4 |  4.2%/ 18.9 |  4.2%/ 18.7 | drena
  52 |    1 |  257.3 |  0.8%/ 19.9 |  0.8%/ 19.8 | drena
  53 |   46 |  271.8 |  5.0%/ 18.2 | 12.5%/ 17.3 | colapsa
  54 |   28 |  266.9 |  3.3%/ 17.4 |  5.8%/ 17.8 | drena
  55 |   70 |  284.1 | 20.8%/ 16.8 | 27.5%/ 16.1 | colapsa
  56 |   15 |  258.5 |  0.8%/ 19.8 |  2.5%/ 18.4 | drena
  57 |    4 |  256.9 |  0.8%/ 19.6 |  0.8%/ 20.0 | drena
  58 |   31 |  271.1 |  7.5%/ 17.9 | 15.0%/ 17.4 | colapsa
  59 |   15 |  258.0 |  0.8%/ 20.0 |  0.8%/ 18.0 | drena
  60 |   38 |  269.9 |  5.0%/ 19.1 | 13.3%/ 17.7 | colapsa
  61 |    8 |  263.5 |  0.0%/ 20.1 |  2.5%/ 18.9 | drena
  62 |   80 |  291.1 | 20.8%/ 17.7 | 26.7%/ 17.0 | colapsa
  63 |   44 |  269.6 | 13.3%/ 17.8 | 15.8%/ 17.4 | colapsa
  64 |    7 |  261.7 |  0.0%/ 20.2 |  0.0%/ 20.1 | drena
  65 |    8 |  259.6 |  1.7%/ 19.3 |  2.5%/ 19.5 | drena
  66 |    9 |  262.6 |  0.0%/ 20.5 |  2.5%/ 19.5 | drena
  67 |    2 |  259.5 |  1.7%/ 19.4 |  0.8%/ 18.6 | drena
  68 |    1 |  259.5 |  0.0%/ 20.1 |  0.0%/ 20.0 | drena
  69 |   15 |  261.2 |  1.7%/ 19.6 |  0.8%/ 20.0 | drena
  70 |   14 |  259.5 |  1.7%/ 19.9 |  1.7%/ 19.3 | drena
  71 |   44 |  271.0 | 11.7%/ 18.2 | 15.0%/ 17.5 | colapsa
  72 |    3 |  257.8 |  0.0%/ 20.9 |  0.0%/ 20.7 | drena
  73 |    5 |  266.1 |  0.8%/ 20.0 |  0.0%/ 20.3 | drena
  74 |   15 |  258.5 |  0.0%/ 20.3 |  3.3%/ 19.2 | drena
  75 |    8 |  261.1 |  2.5%/ 19.0 |  2.5%/ 18.6 | drena
  76 |   20 |  263.6 |  0.8%/ 19.3 |  5.8%/ 18.4 | drena
  77 |    7 |  261.7 |  4.2%/ 19.6 |  0.0%/ 20.3 | drena
  78 |   35 |  270.2 |  2.5%/ 18.9 |  8.3%/ 18.4 | drena
  79 |   25 |  266.3 |  3.3%/ 17.9 |  4.2%/ 18.3 | drena
  80 |   31 |  267.2 |  0.8%/ 19.7 |  6.7%/ 17.7 | drena
  81 |   21 |  269.8 |  1.7%/ 19.6 |  1.7%/ 19.4 | drena
  82 |   31 |  265.2 |  7.5%/ 17.9 |  2.5%/ 19.4 | drena
  83 |   60 |  280.2 |  7.5%/ 17.7 | 17.5%/ 16.1 | colapsa
  84 |    4 |  256.8 |  0.0%/ 20.5 |  0.0%/ 20.3 | drena
  85 |   92 |  288.3 | 14.2%/ 17.9 | 25.8%/ 17.4 | colapsa
  86 |    7 |  255.9 |  0.0%/ 21.8 |  0.0%/ 21.0 | drena
  87 |   22 |  267.2 |  1.7%/ 19.7 |  2.5%/ 19.0 | drena
  88 |    1 |  258.3 |  0.8%/ 20.0 |  0.0%/ 20.9 | drena
  89 |    5 |  257.1 |  0.8%/ 19.2 |  0.0%/ 20.6 | drena
  90 |   26 |  267.7 |  1.7%/ 19.1 | 10.8%/ 18.2 | colapsa
  91 |   14 |  266.1 |  2.5%/ 16.2 |  3.3%/ 18.9 | drena
  92 |    0 |  254.1 |  0.0%/ 21.8 |  0.0%/ 20.2 | drena
  93 |    2 |  255.0 |  0.0%/ 20.3 |  0.0%/ 20.9 | drena
  94 |   16 |  262.7 |  0.0%/ 20.1 |  7.5%/ 18.9 | drena
  95 |    3 |  259.2 |  0.0%/ 20.7 |  0.0%/ 20.6 | drena
  96 |    3 |  253.6 |  0.0%/ 21.3 |  0.0%/ 20.6 | drena
  97 |   51 |  272.9 |  5.0%/ 17.9 | 14.2%/ 16.2 | colapsa
  98 |    4 |  256.0 |  0.0%/ 20.6 |  1.7%/ 19.1 | drena
  99 |   48 |  276.0 |  7.5%/ 19.0 | 13.3%/ 17.5 | colapsa
 100 |    0 |  253.9 |  0.0%/ 21.7 |  0.0%/ 21.2 | drena
 101 |    4 |  256.3 |  0.0%/ 22.4 |  0.0%/ 21.1 | drena
--------------------------------------------------------------
drenan 48/60.  Banderas (no-drena): [53, 55, 58, 60, 62, 63, 71, 83, 85, 90, 97, 99]
```

## Desglose factual

**48 drenan / 12 colapsan (20 %).** Los 12: `53, 55, 58, 60, 62, 63, 71, 83, 85, 90, 97, 99`.

- **El dip es el gatillo universal**: los 12 colapsos disparan el dip (frac sub-20 > 10 %) en al
  menos una ventana. **PM es la ventana vinculante** — en casi todos PM > AM. Coherente con que
  el pico PM es el momento de máxima congestión de Miraflores.
- **Teleports > 50**: 5 seeds — 55 (70), 62 (80), 83 (60), 85 (92), 97 (51, marginal).
- **Duración > 280 s**: 4 seeds — 55 (284.1), 62 (291.1), 83 (280.2, marginal), 85 (288.3).
- **Colapsos "duros" (las tres señales)**: `55, 62, 85` (+ `83` con duración marginal 280.2).
  Días genuinamente severos: PM 25-27 % sub-20, teleports 60-92.
- **Colapsos solo-por-dip** (teleports y duración OK): `53, 58, 60, 63, 71, 90, 99`. Congestión
  de pico intensa pero la red sigue funcionando (sin teleports masivos, duración acotada).
- **Continuo en el filo (8-13 % PM)**: la separación drena/colapsa **no es un abismo**, es un
  continuo con el corte en 10 %. `90` colapsa apenas (PM 10.8 %), `53` (12.5 %), `99` (13.3 %);
  del lado que drena, `78` (8.3 %), `94` (7.5 %), `80` (6.7 %), `76`/`54` (5.8 %). El umbral 10 %
  cae **en medio de un continuo**, no parte un cluster bimodal. → el dataset tiene un **gradiente
  de congestión de pico**, no dos poblaciones discretas (riqueza de señal para B4).
- **Drenan con holgura amplia** (~21 días, 0%/0% dip, teleports de un dígito): 42, 43, 46, 47,
  48, 52, 57, 64, 68, 72, 73, 84, 86, 88, 89, 92, 93, 95, 96, 100, 101.

**Ningún día alcanza el régimen de colapso-franco del C3.** El peor (seed 85, 92 teleports) está
**por debajo del onset de 1.3** (137 teleports, "colapsa (onset)") y un orden de magnitud bajo
1.5 (920 teleports). Ref: `calibracion/SWEEP_C3_RESULTS.md` (progresión teleports 11→36→137→920
por scale {1.1,1.2,1.3,1.5}; cliff entre 1.2 y 1.3; 1.1 "dos pasos finos por debajo del onset").
Scale 1.1 transfirió del sweep por-fase a gen_day randomTrips con el comportamiento esperado, con
margen al cliff.

## Juicio D-014 (decisión humana)

**Scale 1.1 ACEPTADO para el dataset v2.** 48/60 días drenan, 12/60 (20 %) colapsan por el
criterio D-014, con el dip PM como gatillo dominante. Los 12 colapsos **NO son días rotos ni se
descartan**: son la **cola superior de la distribución de congestión de pico PM**, señal necesaria
para un predictor de demora — un STGNN entrenado solo sobre días que drenan limpio sería ciego a
la congestión severa, que es el caso que más importa predecir. Los 12 colapsos son **señal, no
contaminación. Entran al tensor.**

Ningún día alcanza el régimen de colapso-franco del C3 (el peor, seed 85, tiene 92 teleports vs
los 137 del onset 1.3 y 920 de 1.5) — scale 1.1 transfirió del sweep por-fase a gen_day randomTrips
con el comportamiento esperado, con margen al cliff.

El gate D-014, **sobre este dataset, no separa "válido/roto"** (los 60 son válidos: generaron bien,
exit 0, sin fallos de infra) **sino "congestión moderada/severa"** — es un **caracterizador de
régimen**, no un filtro de validez. Nos dice qué fracción del dataset es congestión severa (20 %).

**El trigger de D-014 ("revisar scale/margen") NO se dispara**: el 20 % de colapsos no indica un
scale mal elegido, indica un scale que **captura la cola del fenómeno como se diseñó**. Si se
hubiera visto ~45/60 colapsando, o algún día en régimen de colapso-franco (teleports en centenas),
ahí sí se revisaría. No es el caso.

## Consecuencia para B4 — split estratificado por régimen de congestión

El split train/val/test **debe estratificarse por régimen de congestión**, NO aleatorio. Los 12
días de congestión severa —

`53, 55, 58, 60, 62, 63, 71, 83, 85, 90, 97, 99`

— deben distribuirse conscientemente entre train/val/test. Si caen todos en test, el modelo se
entrena ciego a la congestión severa y se evalúa solo sobre ella (desastre); si caen todos en
train, se evalúa sin congestión severa (métrica engañosamente buena). Es la generalización del
trato que el README v1 daba al seed-081 (un día severo, distribuido a conciencia), pero ahora son
**12, no 1**. **Deuda para B4 (el split).**

## Conexión con la recaracterización Sanity diferida (post-B3.2.c)

Esta caracterización es el **insumo de la narrativa Sanity de v2** (diferida, ver banner en el
README del dataset). La narrativa Sanity de v2 es justamente ésta: *el dataset tiene 12/60 días de
congestión severa de pico PM, distribuidos en `{53,55,58,60,62,63,71,83,85,90,97,99}`, con un
gradiente continuo (no bimodal) en el filo, y hay que estratificar el split por régimen.* El gate
de este documento + el `aggregate_sanity.py` (con su hardcode 381/548640 corregido) corrido sobre
la data v2 son los dos insumos de esa recaracterización.
