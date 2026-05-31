# TTH-11 — Handoff de cierre: spike de hiperparámetros temporales COMPLETO

**Rama**: `feature/tth-11-hiperparametros-temporales`.
**Fecha de cierre**: 2026-05-31.
**Estado al cierre**: **TTH-11 COMPLETO (8/8 CTs)**. El spike entregó el contrato de
hiperparámetros temporales para TTH-09. El próximo arranque **ya no continúa TTH-11**:
parte de él (lo hereda como insumo). Sin merge (lo hace Cesar).

> Este handoff continúa al de dataset
> [tth-11-dataset-perfil-dia-handoff.md](tth-11-dataset-perfil-dia-handoff.md), que dejó
> la Fase 1 del loader como próximo paso. Esa fase y las siguientes ya están hechas.

---

## 1. TTH-11 completo — los 8 CTs

Doc final: [INVESTIGACION_HIPERPARAMETROS_TEMPORALES.md](../../docs/INVESTIGACION_HIPERPARAMETROS_TEMPORALES.md)
(marcado **documento completo**).

- **Académico** (commit `e9ff8d9a`): **CT-11.1** propósito/estructura, **CT-11.2** los 4
  hiperparámetros (Δt_in, lookback, horizonte, re-inferencia), **CT-11.3** revisión
  bibliográfica (5 fuentes), **CT-11.8** cierre de Δt_in = 60 s (cerró CT-07.3 de TTH-07).
- **Empírico** (esta sesión): **CT-11.4** barrido entrenado con 4 métricas por dirección,
  **CT-11.5** tabla resumen / contrato para TTH-09, **CT-11.6** limitaciones honestas,
  **CT-11.7** cierre de doble propósito.

## 2. El resultado del spike (lo que TTH-09 hereda como contrato)

Barrido de **16 modelos** GRU univariada por dirección (4 dir × 4 combos), modelos
**efímeros** (ningún checkpoint; solo métricas). Métricas reproducibles en
[tth11_sweep_metrics.json](../../../ia_prediction_service/scripts/tth11_sweep_metrics.json).

**Contrato recomendado:**

| Hiperparámetro | Valor | Nota |
|----------------|-------|------|
| Δt_in | **60 s** | cierra el param abierto de TTH-07 (CT-07.3) |
| lookback | **30 min** | default robusto (C2) |
| horizonte | **30 min** | default robusto (C2) |
| re-inferencia | **60 s** | coherente con el bucket de entrada |

**Recomendación: C2 (lookback 30 / horizonte 30)** como default robusto — gana accuracy y
MAE en los 4 ejes, 2º en F1-macro global, **nunca el peor**.

**HALLAZGO CENTRAL — el óptimo difiere por eje** (el resultado más valioso del spike):
- **Eje dominante N/S** (transición filosa valle→jam4): prefiere **horizonte corto**.
  Alargar a horizonte 60 lo degrada (accuracy 0.74 → 0.71).
- **Eje secundario E/W** (dinámica suave, demanda ≈ N/S ÷ 4): mejora su **F1-macro con
  lookback largo** (C4 = lookback 60: F1 0.43 vs 0.41 de C2).
- *Razón física*: las clases de cola viven en ejes distintos (pico jam4 en N/S; jam3
  transitorio en E/W).
- **Decisión que queda para TTH-09**: usar **C2 default** para las 4 direcciones, o
  **C2 para N/S + C4 para E/W** si prioriza el F1 del eje secundario.

## 3. Decisiones de fondo heredadas (que TTH-09 debe conocer)

De esta sesión y la previa (ver también §2 del handoff de dataset):

- **jam_level alineado a escala Waze** (D-009, cortes 80/60/40/20) — paridad con la
  fuente de producción de TTH-09.
- **Bug edge-vacío arreglado**: bucket sin observación = **jam 0** (calle vacía = flujo
  libre), **no jam 5**. Distingue de velocidad-cero-genuina-con-vehículos (jam 5 real).
- **Bimodalidad de la 4-way**: solo **jam 2** (bajo capacidad) y **jam 4** (sobre
  capacidad) son estables; **jam 3 es transitorio** (cliff de capacidad). La **clase
  escasa es direccional**: jam3 ~0.75 % en N/S (la red colapsa esa clase a F1 = 0),
  jam4 ~2.8 % en E/W (F1 ≈ 0.27). El MAE ordinal bajo (0.28–0.41) muestra que los
  errores caen en clases **adyacentes**, no en saltos — el modelo "se equivoca cerca".
- **Dataset perfil-día** (24 h, 1440 buckets/corrida, dinámica real), **6 clases** de
  salida con **jam 5 fuera-de-soporte** (no se remapea; no emerge de demanda).
- **El 81.3 % heredado (D-005) NO fue reproducido por el spike**: mejor accuracy 0.7571
  (−5.6 pts). **No es comparable 1:1 ni es techo del sistema** — el spike es desechable,
  sin tuning ni manejo de desbalance, sobre datos sintéticos chicos. Contra el baseline
  mayoritario de cada eje el modelo aprende fuerte (N/S +27 pts, E/W +14 pts).

## 4. Deuda / horizonte para TTH-09 (el modelo de producción)

- **Cerrar la brecha del 81.3 %**: más datos, tuning y **manejo de desbalance**
  (class weights / rebalanceo). El spike **NO lo hizo a propósito** —distribución honesta
  para una comparación limpia de hiperparámetros—; es precisamente la palanca que TTH-09
  tiene disponible.
- **jam 5**: requiere un **mecanismo de bloqueo** (spillback forzado / incidente / cierre),
  no demanda. Modelar como **evento**, no como cola de la distribución demanda-driven.
- **Migración a red real multi-intersección**: el mapa OSM
  [miraflores.net.xml](../../../simulation/conf/network/miraflores.net.xml) **ya está
  versionado** (commit `f76bb60c`). El pipeline actual está **acoplado a la 4-way**
  (naming `*_in`/`*_out`, esquema NS/EW, detectores LA_*); migrarlo a la topología OSM es
  trabajo de TTH-09.
- **`test_b2` de TTH-07 en `xfail(strict)`**: los patrones de flujo constante no llegan a
  jam ≥ 3 bajo la escala Waze. Recalibrarlos es **tarea aparte** (desacople consciente de
  TTH-07).
- **Bibliografía**: 4 refs (Chung 2014, Wang 2022, Singh 2025, Wen 2023) agregadas **a
  mano** a `markdownToDocx/referencias.bib` → **se pierden en el próximo re-export de
  Zotero**. Cargarlas en Zotero antes de re-exportar.

## 5. Estado git

Rama `feature/tth-11-hiperparametros-temporales`, **sin merge** (lo hace Cesar), sin push/PR.

**Commits de TTH-11** (previos + esta sesión, del más reciente al más antiguo):

```
f76bb60c  feat(simulation): mapa OSM real (miraflores.net.xml) para TTH-09
ef6a6b19  TTH-11: completar CT-11.4/11.5/11.6/11.7 con resultados del barrido
595069df  TTH-11: barrido exploratorio lookback×horizonte (CT-11.4, spike)
13e907b9  TTH-11: loader temporal para dataset perfil-día (Fase 1)
fc487704  docs(tth-11): handoff de cierre — dataset perfil-día + fixes de pipeline
29718d51  fix(simulation): spillback per-lane en cobertura perfil-día
c9e744ef  test(simulation): cobertura del perfil-día con criterio temporal
dac02261  feat(simulation): escenarios SUMO 24h del perfil-día + repunte de partitions
ee889539  feat(simulation): perfiles-día de demanda variable 24h (TTH-11)
3c324829  Nota de reenvío en handoff TTH-07 por realineación de escala
664ff904  Tests de regresión: cortes nuevos + ausente-vs-cero; xfail cobertura
5af501b0  Distinguir bucket sin dato de velocidad cero en jam_level
1ebdaf0a  Alinear cortes de jam_level a la escala Waze (80/60/40/20)
e9ff8d9a  docs(tth-11): documento de hiperparámetros temporales (parte académica)
```

**Insumo versionado para TTH-09**: `miraflores.net.xml` (mapa OSM) ya commiteado
(`f76bb60c`) — antes untracked, ahora parte del repo a pedido del equipo. Los parquets del
dataset perfil-día siguen en `data/` (gitignored, regenerables). TTH-07 sin tocar.
