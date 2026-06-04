# Reporte crudo — Bloques 0 y 1 (track STGNN Miraflores, Fase 1)

**Fecha:** 2026-06-01 · **Naturaleza:** respaldo de trazabilidad (data cruda, no prosa
interpretativa). La interpretación vive en la nota de cierre anexada a **D-012**
(`documentation/lean-inception/4-decisiones/DECISIONS.md`). Este documento solo vuelca los
números EXACTOS tal como los emitieron los scripts de auditoría —sin redondeo, reinterpretación
ni resumen— para que cualquiera pueda rastrear de dónde sale cada cifra de la decisión.

> **Nota posterior — NO es parte del volcado crudo (agregada 2026-06-03, B3.1).** Los números de
> abajo son del net **v1** (381 nodos), exactos a la fecha de medición (2026-06-01) y se preservan
> sin tocar. El net fue reconstruido a **v2** y promovido a producción en B3.0/B3.1: sobre v2,
> N_LCC=**1660** y grafo completo=**1664**. Esta nota es solo un puntero-hacia-adelante; el volcado
> de abajo sigue siendo el dato crudo original, sin reinterpretación.

## Setup de medición

- **Dataset:** `simulation/data/datasets/miraflores_laborable_60d/` (parquets `day_seed*.parquet`).
- **Grafo:** `ia_prediction_service/src/data/artifacts/miraflores_graph_mapping.json` (381 nodos,
  509 aristas dirigidas → 504 no-dirigidas; 2 componentes conexas).
- **Días auditados:** 8 → `042, 081, 060, 043, 044, 045, 046, 047` (042 base, 081 congestión
  máxima, 060 azar; +5 extra). El Bloque 1 y el cruce de componentes usan los 8; el **Bloque 0
  (salud de señal) corre sobre el subconjunto de 3** (`042, 081, 060`), que es lo que ejecuta el
  script — se reporta tal cual.
- **Variables:** `speedRelative` (primaria), `density` (robustez), `speed` (apoyo).
- **Correlación (Bloque 1):** Pearson con exclusión pairwise de NaN, umbral mínimo
  `n_conjunto >= 50`; vecinos en grafo no-dirigido (distancia de saltos); no-vecinos por muestreo
  determinista (`SEED=12345`, 20 000 candidatos). NaN en `speedRelative` ⟹ no había tráfico
  (Bloque 0).
- **Reproducibilidad:** scripts read-only, no escriben nada fuera de stdout. Fuente:
  `scratch/b0_signal_health/{health.py, spatial_corr.py, components.py}`. Re-ejecutables con el
  venv del repo (`.venv/bin/python`).

---

## Bloque 0 — salud de señal (`health.py`, días 042 / 081 / 060)

```
Bloque 0 — salud de señal. venv: /Users/rasec/Tesis/CerebroVial/.venv/bin/python

==============================================================================
 DIA seed042  —  day_seed042.parquet
==============================================================================
shape: 548640 filas | edges=381 (esp 381) | timesteps=1440 (esp 1440) | consistente=OK

-- (1) NaN global --
   speedRelative  NaN global =  82.19%
   density        NaN global =   0.00%
   speed          NaN global =  82.19%

-- (1b) Concentracion de NaN en speedRelative --
   edges con >50% NaN  : 339 / 381  (candidatos 'casi sin senal')
   edges con >90% NaN  : 201 / 381
   edges con ~100% NaN :  16 / 381  (siempre-vacios)
   distrib frac-NaN/edge (min,25,50,75,90,max): 0.28, 0.69, 0.91, 0.97, 0.99, 1.00
   frac-NaN por timestep (sobre 381 edges): min=0.59 med=0.80 max=1.00

-- (2) Varianza / rango (sobre valores no-NaN) --
   speedRelative  rango=[0.0000, 1.4100] | edges var~0 (con >=2 datos) =   0 | edges todo-NaN =  13
   density        rango=[0.0000, 533.3300] | edges var~0 (con >=2 datos) =  13 | edges todo-NaN =   0
   speed          rango=[0.0000, 25.3100] | edges var~0 (con >=2 datos) =   0 | edges todo-NaN =  13

-- (3) Perfil temporal: promedio sobre edges por timestep --
   momento         ts(s)  speedRelative        density          speed
   madrugada_03h   10800         0.5078         0.3010         8.1356
   picoAM_08h      28800         0.6968         2.1877         8.9495
   mediodia_12h    43200         0.6856         2.0492         9.3165
   picoPM_18h      64800         0.6996         2.7849         8.9889
   pm_tail_20h     72000         0.7649         2.0142         9.8417
   noche_22h       79200         0.7502         0.5981         8.8744
   speedRelative perfil: min=0.0100@0h max=0.9995@2h
   jam%% (speedRel<0.5 entre presentes) por hora clave: 3h=24.4, 8h=24.7, 9h=24.1, 10h=24.6, 12h=23.8, 18h=23.2, 20h=23.2, 22h=21.0

-- (4) Equivalencia  speedRelative.isna()  <->  density==0 --
   speedRel NaN & density==0  :  450910 (82.187%)  [vacio coherente]
   speedRel NaN & density>0   :       0 ( 0.000%)  [NaN c/presencia -> gridlock?]
   speedRel !NaN & density==0 :    1839 ( 0.335%)  [vel sin densidad -> raro]
   EQUIVALENCIA EXACTA: NO — hay casos raros (ver arriba)
     -> !NaN&density0: speedRel[med=0.180] speed[med=2.460]

==============================================================================
 DIA seed081  —  day_seed081.parquet
==============================================================================
shape: 548640 filas | edges=381 (esp 381) | timesteps=1440 (esp 1440) | consistente=OK

-- (1) NaN global --
   speedRelative  NaN global =  81.40%
   density        NaN global =   0.00%
   speed          NaN global =  81.40%

-- (1b) Concentracion de NaN en speedRelative --
   edges con >50% NaN  : 327 / 381  (candidatos 'casi sin senal')
   edges con >90% NaN  : 182 / 381
   edges con ~100% NaN :  11 / 381  (siempre-vacios)
   distrib frac-NaN/edge (min,25,50,75,90,max): 0.19, 0.71, 0.89, 0.98, 0.99, 1.00
   frac-NaN por timestep (sobre 381 edges): min=0.59 med=0.79 max=1.00

-- (2) Varianza / rango (sobre valores no-NaN) --
   speedRelative  rango=[0.0000, 1.3300] | edges var~0 (con >=2 datos) =   0 | edges todo-NaN =   8
   density        rango=[0.0000, 533.3300] | edges var~0 (con >=2 datos) =   8 | edges todo-NaN =   0
   speed          rango=[0.0000, 22.4600] | edges var~0 (con >=2 datos) =   0 | edges todo-NaN =   8

-- (3) Perfil temporal: promedio sobre edges por timestep --
   momento         ts(s)  speedRelative        density          speed
   madrugada_03h   10800         0.6992         0.0736         7.3975
   picoAM_08h      28800         0.6280         7.1177         7.7701
   mediodia_12h    43200         0.6088         8.2207         7.5296
   picoPM_18h      64800         0.5540         5.5982         7.0702
   pm_tail_20h     72000         0.3363        26.8909         3.7445
   noche_22h       79200         0.5717         5.2981         6.8776
   speedRelative perfil: min=0.0100@1h max=1.1400@0h
   jam%% (speedRel<0.5 entre presentes) por hora clave: 3h=21.8, 8h=29.7, 9h=31.6, 10h=30.9, 12h=29.3, 18h=31.2, 20h=65.8, 22h=34.5

-- (4) Equivalencia  speedRelative.isna()  <->  density==0 --
   speedRel NaN & density==0  :  446618 (81.405%)  [vacio coherente]
   speedRel NaN & density>0   :       0 ( 0.000%)  [NaN c/presencia -> gridlock?]
   speedRel !NaN & density==0 :    3275 ( 0.597%)  [vel sin densidad -> raro]
   EQUIVALENCIA EXACTA: NO — hay casos raros (ver arriba)
     -> !NaN&density0: speedRel[med=0.030] speed[med=0.320]

==============================================================================
 DIA seed060  —  day_seed060.parquet
==============================================================================
shape: 548640 filas | edges=381 (esp 381) | timesteps=1440 (esp 1440) | consistente=OK

-- (1) NaN global --
   speedRelative  NaN global =  81.73%
   density        NaN global =   0.00%
   speed          NaN global =  81.73%

-- (1b) Concentracion de NaN en speedRelative --
   edges con >50% NaN  : 333 / 381  (candidatos 'casi sin senal')
   edges con >90% NaN  : 200 / 381
   edges con ~100% NaN :  14 / 381  (siempre-vacios)
   distrib frac-NaN/edge (min,25,50,75,90,max): 0.20, 0.69, 0.91, 0.98, 0.99, 1.00
   frac-NaN por timestep (sobre 381 edges): min=0.59 med=0.80 max=1.00

-- (2) Varianza / rango (sobre valores no-NaN) --
   speedRelative  rango=[0.0000, 1.3800] | edges var~0 (con >=2 datos) =   0 | edges todo-NaN =  12
   density        rango=[0.0000, 533.3300] | edges var~0 (con >=2 datos) =  12 | edges todo-NaN =   0
   speed          rango=[0.0000, 22.9000] | edges var~0 (con >=2 datos) =   0 | edges todo-NaN =  12

-- (3) Perfil temporal: promedio sobre edges por timestep --
   momento         ts(s)  speedRelative        density          speed
   madrugada_03h   10800         0.7500         0.1101         9.3468
   picoAM_08h      28800         0.7373         4.3605         9.2225
   mediodia_12h    43200         0.5770         3.1876         7.3446
   picoPM_18h      64800         0.6926         3.9580         8.9120
   pm_tail_20h     72000         0.6685         4.8058         8.2035
   noche_22h       79200         0.6149         2.0525         7.4512
   speedRelative perfil: min=0.0600@0h max=1.0662@0h
   jam%% (speedRel<0.5 entre presentes) por hora clave: 3h=23.0, 8h=24.8, 9h=26.8, 10h=27.7, 12h=25.4, 18h=26.2, 20h=29.6, 22h=28.0

-- (4) Equivalencia  speedRelative.isna()  <->  density==0 --
   speedRel NaN & density==0  :  448404 (81.730%)  [vacio coherente]
   speedRel NaN & density>0   :       0 ( 0.000%)  [NaN c/presencia -> gridlock?]
   speedRel !NaN & density==0 :    1956 ( 0.357%)  [vel sin densidad -> raro]
   EQUIVALENCIA EXACTA: NO — hay casos raros (ver arriba)
     -> !NaN&density0: speedRel[med=0.170] speed[med=2.440]

[fin] análisis read-only. nada escrito al dataset/grafo.
```

---

## Bloque 1 — correlación espacial vecinos-vs-no-vecinos (`spatial_corr.py`, 8 días)

```
Bloque 1 — correlación espacial.
grafo: 381 nodos, 504 aristas no-dirigidas (de 509 dirigidas) | componentes conexas: 2
grupos de pares: vecinos-1=504  vecinos-2=809  no-vecinos(muestra)=20000

cargando matriz speedRelative...
días usados: 8 -> ['042', '081', '060', '043', '044', '045', '046', '047'] | matriz (11520, 381) (timesteps x edges)
NaN global speedRelative en matriz: 81.94%

##############################################################################
# VARIABLE: speedRelative   (umbral n>=50)
##############################################################################

-- vecinos-1 --  candidatos=504  sobreviven(corr def & n>=50)=447
   corr: min=-0.235 p25=0.253 MED=0.463 p75=0.654 p90=0.814 max=0.959 | mean=0.448
   n (sobrevivientes): min=51 p25=205 med=796 p75=2456 max=7581
   n (todos candidatos): min=0 p25=122 med=530 p75=1791 max=7581

-- vecinos-2 --  candidatos=809  sobreviven(corr def & n>=50)=631
   corr: min=-0.467 p25=0.028 MED=0.171 p75=0.398 p90=0.579 max=0.937 | mean=0.222
   n (sobrevivientes): min=50 p25=165 med=626 p75=1704 max=7519
   n (todos candidatos): min=0 p25=60 med=281 p75=1318 max=7519

-- no-vecinos --  candidatos=20000  sobreviven(corr def & n>=50)=12334
   corr: min=-0.438 p25=-0.039 MED=0.007 p75=0.056 p90=0.128 max=0.885 | mean=0.014
   n (sobrevivientes): min=50 p25=117 med=297 p75=874 max=6984
   n (todos candidatos): min=0 p25=19 med=97 p75=428 max=6984

-- CONTRASTE (speedRelative) --
   mediana(v1) - mediana(no-vec) = 0.463 - 0.007 = +0.456
   media(v1)   - media(no-vec)   = 0.448 - 0.014 = +0.434
   decaimiento: MED v1=0.463 > v2=0.171 > no-vec=0.007 ?

-- CONCENTRACIÓN (speedRelative): pares vecinos-1 con corr>0.5 --
   pares v1 con corr>0.5: 202 de 447 sobrevivientes (45.2%)
   edges distintos involucrados en pares corr>0.5: 269
   top-15 edges por apariciones en pares corr>0.5:
      node 318  426670758#0             aparece en 4 pares
      node 340  460123287               aparece en 4 pares
      node 375  892765650               aparece en 4 pares
      node 319  426670758#1             aparece en 4 pares
      node 172  24252825#1              aparece en 3 pares
      node  43  11986321#3              aparece en 3 pares
      node  58  1230462976              aparece en 3 pares
      node  60  1230462977#1            aparece en 3 pares
      node 158  24252820                aparece en 3 pares
      node 153  24252819#2              aparece en 3 pares
      node 171  24252825#0              aparece en 3 pares
      node 157  24252819#6              aparece en 3 pares
      node 330  435707312#0             aparece en 3 pares
      node 283  344160285#3             aparece en 3 pares
      node 196  24315678#1              aparece en 3 pares

cargando matriz density...
matriz density (11520, 381) | NaN: 0.00%

##############################################################################
# VARIABLE: density   (umbral n>=50)
##############################################################################

-- vecinos-1 --  candidatos=504  sobreviven(corr def & n>=50)=502
   corr: min=-0.020 p25=0.147 MED=0.395 p75=0.677 p90=0.862 max=0.976 | mean=0.416
   n (sobrevivientes): min=11520 p25=11520 med=11520 p75=11520 max=11520
   n (todos candidatos): min=11520 p25=11520 med=11520 p75=11520 max=11520

-- vecinos-2 --  candidatos=809  sobreviven(corr def & n>=50)=806
   corr: min=-0.024 p25=0.034 MED=0.122 p75=0.319 p90=0.539 max=0.966 | mean=0.206
   n (sobrevivientes): min=11520 p25=11520 med=11520 p75=11520 max=11520
   n (todos candidatos): min=11520 p25=11520 med=11520 p75=11520 max=11520

-- no-vecinos --  candidatos=20000  sobreviven(corr def & n>=50)=19789
   corr: min=-0.025 p25=0.004 MED=0.020 p75=0.042 p90=0.078 max=0.942 | mean=0.036
   n (sobrevivientes): min=11520 p25=11520 med=11520 p75=11520 max=11520
   n (todos candidatos): min=11520 p25=11520 med=11520 p75=11520 max=11520

-- CONTRASTE (density) --
   mediana(v1) - mediana(no-vec) = 0.395 - 0.020 = +0.375
   media(v1)   - media(no-vec)   = 0.416 - 0.036 = +0.380
   decaimiento: MED v1=0.395 > v2=0.122 > no-vec=0.020 ?

-- CONCENTRACIÓN (density): pares vecinos-1 con corr>0.5 --
   pares v1 con corr>0.5: 200 de 502 sobrevivientes (39.8%)
   edges distintos involucrados en pares corr>0.5: 265
   top-15 edges por apariciones en pares corr>0.5:
      node 375  892765650               aparece en 4 pares
      node 219  315218164#2             aparece en 3 pares
      node 343  511800018#2             aparece en 3 pares
      node  94  1365587180#1            aparece en 3 pares
      node  96  1365587180#3            aparece en 3 pares
      node 293  345820035#6             aparece en 3 pares
      node 153  24252819#2              aparece en 3 pares
      node 154  24252819#3              aparece en 3 pares
      node 232  319655874#4             aparece en 3 pares
      node 301  39441587#1              aparece en 3 pares
      node   2  -129822384#2            aparece en 2 pares
      node  20  1102478359#3            aparece en 2 pares
      node  23  1102478359#6            aparece en 2 pares
      node  37  1152311682#1            aparece en 2 pares
      node  41  11986321#1              aparece en 2 pares

[fin] read-only. nada escrito al dataset/grafo.
```

---

## Bloque 2B-cierre — componentes, aislados, cruce con siempre-vacíos (`components.py`, 8 días)

Respalda el recorte topológico: 2 componentes conexas (375 + 6), la islita de 6 edges (grados
1-3, desconectada del cuerpo vía `<connection>`), y los 2 edges estructuralmente vacíos
(`111898821`, `438009517`) que SÍ son vecinos topológicos válidos dentro de la componente
principal (comp0, deg=1) — por eso se conservan en el grafo de 375.

```
=== (1) COMPONENTES CONEXAS: 2 ===
  componente 0: 375 nodos
  componente 1: 6 nodos
    sumo_edges:
      node  38  deg=1  11985865#0
      node  39  deg=3  11985865#1
      node  90  deg=1  1364346888
      node  91  deg=3  1364346889
      node 132  deg=1  24252389
      node 143  deg=1  24252402

=== (2) NODOS AISLADOS (grado 0): 0 ===

=== (3) SIEMPRE-VACÍOS (all-NaN speedRelative) ===
  all-NaN en LOS 8 días (estructural): 2
  all-NaN en >=1 día (referencia):     30

  [all-NaN en los 8 días] total=2 | aislados=0 | por componente={0: 2}
      node  25  111898821              comp0(deg=1)
      node 339  438009517              comp0(deg=1)

  [all-NaN en >=1 día] total=30 | aislados=0 | por componente={1: 6, 0: 24}
      node  25  111898821              comp0(deg=1)
      node  38  11985865#0             comp1(deg=1)
      node  39  11985865#1             comp1(deg=3)
      node  64  1230462981#0           comp0(deg=3)
      node  65  1230462981#1           comp0(deg=1)
      node  76  1310962223#0           comp0(deg=1)
      node  77  1310962223#1           comp0(deg=2)
      node  83  1363209677             comp0(deg=1)
      node  85  1364169023             comp0(deg=3)
      node  89  1364169032             comp0(deg=2)
      node  90  1364346888             comp1(deg=1)
      node  91  1364346889             comp1(deg=3)
      node  92  1364346891             comp0(deg=3)
      node 121  19070397#1             comp0(deg=2)
      node 122  19070444               comp0(deg=2)
      node 132  24252389               comp1(deg=1)
      node 143  24252402               comp1(deg=1)
      node 147  24252525#3             comp0(deg=1)
      node 170  24252824               comp0(deg=2)
      node 190  24315671               comp0(deg=2)
      node 243  320470478              comp0(deg=2)
      node 299  37749666#2             comp0(deg=2)
      node 304  39441587#4             comp0(deg=2)
      node 305  406007412#0            comp0(deg=1)
      node 307  406007417#0            comp0(deg=1)
      node 321  426670758#3            comp0(deg=1)
      node 335  437191060              comp0(deg=1)
      node 339  438009517              comp0(deg=1)
      node 347  511823826              comp0(deg=2)
      node 380  977112067              comp0(deg=2)

[fin] read-only. grafo intacto.
```
