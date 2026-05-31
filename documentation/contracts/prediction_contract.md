# Contrato canónico — `POST /predictions/predict` (predictor GRU, TTH-09)

> Contrato del módulo de predicción que **TTH-09 va a implementar**. Es el
> **blanco de diseño**, no la documentación de algo ya construido: al día de
> redacción el path `POST /predictions/predict` lo sirve el baseline
> RandomForest con un contrato divergente (Delta-01), y el GRU principal aún
> **no está servido por el núcleo** (TTH-09 No iniciado). Este documento fija
> el shape estable contra el que se codean los consumidores (frontend HU-03,
> HU-14) para que el GRU sustituya al baseline sin alterarlos.
>
> **Arquitectura de producción: clasificador multi-output directo** que
> extiende la línea del spike de TTH-11 (no regresor de ratio). Los
> hiperparámetros temporales (lookback, horizonte, Δt, cadencia) vienen
> **cerrados de TTH-11 (CT-11.5)** y no se re-derivan acá.

**Estado al redactarse:** TTH-09 **No iniciado** (Delta-01). Contrato de diseño; las secciones describen el comportamiento objetivo en presente de especificación.
**Última actualización:** 2026-05-31 (apertura de TTH-09; eje arquitectónico redefinido a clasificador multi-output).
**Autoridad:** `TAREAS_TECNICAS_HABILITADORAS.md` CT-09.1 a CT-09.9 (`documentation/lean-inception/2-backlog/TAREAS_TECNICAS_HABILITADORAS.md` líneas 542–558) + cierre de hiperparámetros de TTH-11 (`documentation/handoffs/tth-11/tth-11-cierre-handoff.md` §2) + D-006 (GRU univariado por dirección) y D-009 (jam level 0-5, escala Waze).

---

## 1. Endpoint

```
POST http://localhost:8001/predictions/predict
Content-Type: application/json
```

**Mismo path que el baseline RandomForest existente** ([core_management_api/src/prediction/presentation/api/routes.py:25](../../core_management_api/src/prediction/presentation/api/routes.py#L25), router con `prefix="/predictions"`). TTH-09 es un **reemplazo total del contrato**, no una extensión: se reusa el path y se descarta el shape per-cámara/niveles-discretos del baseline (ver §9 y Delta-01). El RandomForest no se borra: se preserva como respaldo invocable por TTH-04 (Nivel 2, ver §7).

Puerto 8001 lockeado para el núcleo (consistente con [engine_recommend_contract.md](engine_recommend_contract.md) §1, [docker-compose.yml](../../docker-compose.yml)).

---

## 2. Request — series temporales por dirección

Input **per-intersección**. Para cada una de las **4 direcciones de la 4-way** (`N`, `S`, `E`, `W` — naming real del pipeline de simulación, ver [schema.py](../../simulation/src/cerebrovial_simulation/dataset/schema.py) y D-009) se envía la **ventana de entrada (lookback)** de la serie univariada del acceso.

La serie de entrada es de **`jam_level`** (entero 0-5, escala Waze de D-009), **no de ratio**. Esto es coherente con lo que el modelo consume: la GRU se alimenta con `jam_level / 5.0` normalizado (ver §4); el `ratio` velocidad/free-flow es el insumo **upstream** del que D-009 deriva el `jam_level` (en el sensor/dataset, vía `ratio_to_jam_level`), pero **no** entra al modelo ni viaja en el request. El backend normaliza `jam_level / 5.0` al recibir la serie.

```json
{
  "intersection_id": "larco_schell",
  "timestamp": "2026-05-31T10:00:00Z",
  "series": [
    {
      "direction": "N",
      "jam_levels": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2,
                     2, 3, 3, 2, 2, 2, 1, 1, 2, 2,
                     3, 4, 4, 3, 3, 2, 2, 2, 1, 1]
    },
    { "direction": "S", "jam_levels": [/* 30 enteros 0-5 */] },
    { "direction": "E", "jam_levels": [/* 30 enteros 0-5 */] },
    { "direction": "W", "jam_levels": [/* 30 enteros 0-5 */] }
  ]
}
```

Campos del top-level:

| Campo | Tipo | Notas |
|-------|------|-------|
| `intersection_id` | string | Identificador único de la intersección; ≥ 1 char. |
| `timestamp` | string | ISO-8601. Marca de tiempo del **fin de la ventana** (el bucket más reciente de `jam_levels`). |
| `series` | `list[DirectionSeries]` | **Exactamente 4 entradas**, una por dirección (`N`, `S`, `E`, `W`). Sin direcciones repetidas ni faltantes. |

`DirectionSeries`:

| Campo | Tipo | Notas |
|-------|------|-------|
| `direction` | `"N"` \| `"S"` \| `"E"` \| `"W"` | Acceso de la 4-way. Naming idéntico al de D-009/simulación (letra simple, **no** `north_approach`). |
| `jam_levels` | `list[int]` | **Ventana de lookback = 30 pasos** (= 30 min a Δt = 60 s). Orden **cronológico ascendente** (más antiguo → más reciente; el último elemento corresponde a `timestamp`). Cada valor es el `jam_level` **observado** (entero **0-5**, D-009). Cardinalidad fija: exactamente 30 elementos. El backend lo normaliza `/5.0` antes de la inferencia (ver §4). |

**Una llamada lleva las 4 ventanas**; el endpoint corre una inferencia por dirección sobre su serie.

---

## 3. Response — horizonte por dirección

Objeto con **una entrada por dirección**. Cada entrada es el **array de horizonte = 30 predicciones** (t+1 … t+30, paso 60 s). Cada predicción lleva el **nivel discreto 0-5** (el `argmax` de la clasificación) **y** el vector `probs` de 6 probabilidades softmax (distribución sobre las clases). **No hay campo `ratio`**: el modelo clasifica, no regresa un ratio continuo.

```json
{
  "intersection_id": "larco_schell",
  "generated_at": "2026-05-31T10:00:00Z",
  "model_version": "gru-clf-multioutput-v1",
  "predictions": [
    {
      "direction": "N",
      "horizon": [
        { "step": 1,  "level": 1, "probs": [0.05, 0.74, 0.18, 0.02, 0.01, 0.00] },
        { "step": 2,  "level": 2, "probs": [0.02, 0.21, 0.61, 0.14, 0.02, 0.00] },
        { "step": 30, "level": 3, "probs": [0.00, 0.04, 0.19, 0.55, 0.22, 0.00] }
      ]
    },
    { "direction": "S", "horizon": [/* 30 predicciones */] },
    { "direction": "E", "horizon": [/* 30 predicciones */] },
    { "direction": "W", "horizon": [/* 30 predicciones */] }
  ]
}
```

Campos del top-level:

| Campo | Tipo | Notas |
|-------|------|-------|
| `intersection_id` | string | Eco del request. |
| `generated_at` | string | ISO-8601. Momento de generación de la predicción (≡ `timestamp` del registro persistido, ver §8). |
| `model_version` | string | Identificador/versión del modelo que produjo la respuesta. Se persiste por predicción (§8) y discrimina principal vs respaldo (HU-20). |
| `predictions` | `list[DirectionPrediction]` | **Exactamente 4 entradas**, una por dirección. |

`DirectionPrediction`:

| Campo | Tipo | Notas |
|-------|------|-------|
| `direction` | `"N"` \| `"S"` \| `"E"` \| `"W"` | Acceso. |
| `horizon` | `list[HorizonStep]` | **Exactamente 30 pasos** (t+1 … t+30), orden cronológico ascendente. Cubre el horizonte completo que el slider del Operador (HU-03) recorre — sin llamadas adicionales. |

`HorizonStep`:

| Campo | Tipo | Notas |
|-------|------|-------|
| `step` | int ∈ [1, 30] | Paso futuro relativo a `generated_at`; `step = k` ⇒ instante `generated_at + k·60 s`. |
| `level` | int ∈ [0, 5] | **Nivel de congestión predicho** = `argmax(probs)`. Es **salida directa del clasificador**, **no** se deriva de un ratio (no hay ratio en la salida; `ratio_to_jam_level` **no** participa de la inferencia). Por construcción del entrenamiento (§5) `level` nunca es 5. |
| `probs` | `list[float]` (6) | Distribución softmax sobre las 6 clases (0-5), en orden de clase. **Suma ≈ 1**. Se expone para **incertidumbre en frontend** (HU-03 puede mostrar confianza por paso) e **insumo de métricas** (HU-14). `probs[5]` es estructuralmente ~0 (§5). |

---

## 4. Modelo — clasificador GRU multi-output por dirección (D-006)

- **4 modelos GRU univariados independientes**, uno por dirección (`N`/`S`/`E`/`W`). Cada uno consume **su propia serie univariada** (los `jam_levels` de su acceso, normalizados `/5.0`) y es ciego al resto (CT-09.1, D-006). `input_size = 1` (una feature por paso).
- **Clasificador, no regresor.** Pérdida `CrossEntropyLoss`; **6 clases** (escala Waze 0-5). El target de entrenamiento es la **clase `jam_level` entera 0-4** (jam 5 fuera de soporte, §5).
- **Salida multi-output directa (variante b1).** La GRU produce las **30 distribuciones de horizonte en UNA sola inferencia**, no autoregresiva y no seq2seq. La cabeza es una **Linear ancha** `Linear(hidden, horizonte × n_classes)` = `Linear(hidden, 30 × 6 = 180)`, cuya salida se **reshape a `(30, 6)`** y se le aplica softmax sobre la dimensión de clase → **30 vectores de 6 probabilidades**. El `level` de cada paso es el `argmax` de su distribución. Una llamada al endpoint = 4 inferencias (una por dirección) que cubren las 4 direcciones × horizonte completo.
  - **NO single-step** (la GRU no emite un único punto a `+horizonte`).
  - **NO autoregresivo** (no realimenta su propia predicción; consume solo la ventana observada).
  - **seq2seq queda como trabajo futuro** (no se implementa en TTH-09).
- **Entrenamiento del target vectorial:** cada secuencia de entrada (ventana de 30 pasos) se aparea con un **vector target de 30 clases** (`jam_level` en t+1 … t+30). La pérdida es `CrossEntropyLoss` promediada sobre los 30 pasos del horizonte (y sobre el batch).
- **Persistencia de modelos**: los **4 artefactos** entrenados (uno por dirección) se serializan a una **ubicación fija documentada** bajo `ia_prediction_service/` (el path exacto lo fija CT-09.3 al implementar TTH-09). El endpoint del núcleo los carga desde esa ubicación al inicializarse (CT-09.4.a). Los binarios van por **Git LFS** (convención del repo para `.pt`/`.joblib`/`.ckpt`).
- **Script de entrenamiento** reproducible (CT-09.3): consume el dataset de TTH-07 (CT-07.3) con las particiones de CT-07.4, entrena los 4 GRU con los hiperparámetros temporales de TTH-11 (§10) + los no-temporales que el equipo documente, persiste los modelos y reporta las métricas de §6 sobre validación. Reejecutable y determinista; sin pipeline MLOps.

---

## 5. El nivel 5 (jam 5) — estructuralmente fuera de soporte

El esquema de salida **admite `level` ∈ [0, 5]** y `probs` tiene **6 componentes** (incluida `probs[5]`), por completitud de la escala Waze. Pero el modelo **se entrena sobre 0-4 únicamente**, y el argumento es estructural (no derivado de un ratio):

- La cabeza del clasificador **podría** en principio emitir la clase 5 (tiene 6 logits de salida). Sin embargo, el target de entrenamiento **nunca contiene un ejemplo de jam 5**: el dataset perfil-día de TTH-11 deja jam 5 **fuera de soporte** a propósito.
- jam 5 es un **evento de bloqueo exógeno** (spillback forzado, incidente, cierre de vía), **no demanda aprendible**. En D-009 solo emerge de velocidad cero con vehículos presentes (`ratio ≤ 0`); no aparece en una distribución demanda-driven.
- Como la clase 5 **nunca recibe gradiente positivo** durante el entrenamiento, su logit queda deprimido: **`probs[5]` es estructuralmente ~0** y el **`argmax` nunca cae en 5**.

**Consecuencia contractual, sin ambigüedad:** TTH-09 **nunca predice `level = 5`**. La clase 5 permanece en el rango del schema (y en `probs`) por completitud de la escala y para que un consumidor downstream la reciba si una **capa futura de detección de eventos** la inyecta — pero esa capa **no es parte de TTH-09**. Queda registrada como **deuda**: modelar el bloqueo como evento (detección/inyección), no como cola de la distribución de demanda.

---

## 6. Métricas y evaluación (CT-09.6)

El modelo se evalúa sobre la **partición de validación del dataset de TTH-07** (CT-07.4: seeds y patrones distintos de entrenamiento, sin fuga de información, D-008). Por ser **clasificador**, las métricas son de clasificación, computadas **por paso del horizonte** y **consolidadas**, **por dirección** y **global**:

- **Accuracy** sobre el nivel discreto 0-5.
- **F1-macro** (las 6 clases; reportar también F1-macro sobre clases presentes, como hizo el spike, es opcional y documentable).
- **Matriz de confusión 6×6** (filas = nivel real, columnas = nivel predicho; convención scikit-learn de CA-14.8).
- **MAE ordinal** sobre la clase (distancia `|nivel_pred − nivel_real|`) — métrica ordinal honesta que el spike ya calculaba; **opcional**, complementa accuracy mostrando si los errores caen en clases adyacentes.

> **Nota de alcance (D-005) — divergencia respecto a la letra de CT-09.6.** CT-09.6 enumera, entre las cuatro métricas de HU-14, **MAE y RMSE sobre el ratio continuo**. Esa premisa asume un **regresor de ratio**, arquitectura que **quedó obsoleta**: la producción es un **clasificador** sin salida de ratio, por lo que **MAE/RMSE-sobre-ratio no aplica** y no se reporta. Se reportan las métricas de clasificación de arriba (accuracy, F1-macro, matriz 6×6, MAE ordinal). HU-14 compara el **nivel predicho** contra el **`jam_level` observado** cuando llega el horizonte. Esta es una divergencia consciente entre la arquitectura cerrada y el texto literal de CT-09.6 / CA-14.x, a reconciliar documentalmente en el cierre de TTH-09 / HU-14.

El **objetivo aspiracional** sigue siendo **accuracy ≥ 80%** sobre el nivel 0-5 (CT-09.7), **no bloqueante de Done** (D-005): se reporta honestamente lo medido.

---

## 7. Errores

| Código | Cuándo | Cuerpo |
|--------|--------|--------|
| `200 OK` | Inferencia exitosa sobre las 4 direcciones. | Shape §3. |
| `422 Unprocessable Entity` | Request mal formado (≠ 4 direcciones, `jam_levels` con largo ≠ 30, valor fuera de 0-5, dirección inválida). | Validación Pydantic estándar. |
| `5xx` | **Modelo no responde**: proceso caído, modelo no cargado, timeout interno (CT-09.8). | Cuerpo descriptivo del error (5xx según el caso). |

**El fallback al RandomForest (Nivel 2) NO es responsabilidad de este módulo.** Este endpoint expone **únicamente el GRU principal**. La detección de la caída y la activación del predictor de respaldo son responsabilidad de **TTH-04** (cascada de fallback, Nivel 2), que invoca por separado al `RandomForestPredictor` preservado. TTH-09 no implementa lógica de fallback interno; emite 5xx y delega la cascada a TTH-04 (CT-09.8). Implicancia: el sistema de predicción puede seguir respondiendo (con RandomForest, vía TTH-04) aunque el GRU esté caído — estado **degradado**, no caído del todo.

---

## 8. Persistencia de predicciones (CT-09.5)

Cada predicción generada se **registra de forma durable** en el momento de generarse, con al menos:

- `timestamp` de generación de la predicción,
- `direction`,
- `step` futuro al que corresponde (t+1 … t+30),
- `level` discreto predicho (la clase `argmax`),
- distribución `probs` (las 6 probabilidades) — recomendado, habilita métricas de incertidumbre/calibración en HU-14,
- identificador/versión del modelo (`model_version`).

Este registro es la **fuente de datos de HU-14**: cuando llega el horizonte de cada predicción, se asocia con el `jam_level` observado y se calculan las métricas de §6 (accuracy, F1-macro, matriz de confusión 6×6, MAE ordinal). El registro es **independiente** del de decisiones del motor (CA-08.1 / CT-10.9) y del de transiciones de estado (CT-04.3).

> **Nota (D-005):** CT-09.5 lista, entre los campos a persistir, el **ratio continuo predicho**. Como la producción es clasificador sin salida de ratio, ese campo **no aplica**; se persiste el `level` predicho (y `probs`). Misma reconciliación documental que §6.

El **esquema exacto de persistencia** (tabla, migración Alembic, índices) se fija al implementar CT-09.5 y se referencia desde acá cuando exista; este contrato no lo enumera de forma exhaustiva. HU-20 (MVP2) extiende este mismo registro para persistir en paralelo las predicciones del modelo de respaldo con `model_version` como discriminante, **sin modificar el esquema** (DHU-017 §D).

---

## 9. Consumidores

| Consumidor | Acoplamiento |
|------------|--------------|
| **Frontend — `predictionService.ts` + vista de predicción (HU-03)** | Consume `POST /predictions/predict` para alimentar el panel/slider de predicción por acceso (nivel 0-5 a lo largo del horizonte, con `probs` para confianza). **Requiere migración** desde el shape del baseline RF (per-cámara, niveles `Normal`/`High`/`Heavy` a 15/30/45 min) al shape de §2–§3 (per-intersección, 4 direcciones, `level` + `probs`, horizonte de 30 pasos). Esta migración es parte del cierre de Delta-01. |
| **HU-14 (métricas del modelo)** | Consume el **registro persistente de predicciones** (§8) como fuente para comparar `level` predicho vs `jam_level` observado y computar las métricas de §6. |
| **TTH-04 (cascada de fallback)** | Detecta la caída del GRU (5xx, §7) y activa el Nivel 2 (RandomForest de respaldo). |

**El baseline RandomForest se preserva como respaldo Nivel 2 (CT-09.8), no se borra.** La única coordinación con TTH-04 es asegurar que la ruta interna que invoca el respaldo produce salida **compatible con el contrato de §3** (4 direcciones, `level` por paso de horizonte), con su `model_version` propio para distinguir el origen en el registro (§8). El baseline hoy emite niveles categóricos a horizontes fijos (15/30/45 min); adaptarlo al shape de §3 es parte de la integración de TTH-04, fuera de este módulo.

---

## 10. Hiperparámetros temporales — cerrados de TTH-11 (CT-11.5)

Vienen **fijados por el spike TTH-11**, recomendación **C2 única en los 4 ejes** (`tth-11-cierre-handoff.md` §2). No se re-derivan ni se discuten en TTH-09:

| Hiperparámetro | Valor | Materialización en el contrato |
|----------------|-------|--------------------------------|
| Δt_in (resolución de entrada) | **60 s** | cada paso de `jam_levels` y de `horizon` = 1 bucket de 60 s. |
| lookback (ventana de entrada) | **30 min = 30 pasos** | largo fijo de `jam_levels` (§2). |
| horizonte (ventana de salida) | **30 min = 30 pasos** | largo fijo de `horizon` (§3), t+1 … t+30; fija el `horizonte` de la cabeza `Linear(hidden, 30×6)` (§4). |
| re-inferencia (cadencia) | **60 s** | el endpoint se invoca cada 60 s en operación (coherente con el bucket de entrada). |

> TTH-11 halló que el óptimo difiere por eje (N/S prefiere horizonte corto; E/W mejora F1 con lookback largo, C4). El contrato fija **C2 para las 4 direcciones** como default robusto; usar C4 (lookback 60) en E/W es una decisión opcional de tuning de TTH-09 que **no cambia el shape** de este contrato (solo el largo de `jam_levels` del eje afectado, si se adopta).

---

## 11. Relación con el spike de TTH-11 — la producción lo extiende

La arquitectura de producción **extiende** la del spike (CT-11.4), no la contradice. Verificado contra [tth11_sweep.py](../../ia_prediction_service/scripts/tth11_sweep.py) y [tth11_temporal_loader.py](../../ia_prediction_service/src/data/tth11_temporal_loader.py):

**Lo que se conserva del spike** (misma línea):
- **Clasificador** con `CrossEntropyLoss` ([tth11_sweep.py:76](../../ia_prediction_service/scripts/tth11_sweep.py#L76)).
- **Mismo input**: `jam_level / 5.0` normalizado, 1 feature por paso ([tth11_sweep.py:63-67](../../ia_prediction_service/scripts/tth11_sweep.py#L63-L67)).
- **Misma escala**: 6 clases (0-5), jam 5 fuera de soporte ([tth11_sweep.py:34](../../ia_prediction_service/scripts/tth11_sweep.py#L34)).
- **Mismo dimensionamiento temporal**: lookback/horizonte 30/30 (C2).

**Lo que cambia — la cabeza de salida**:
- **Spike = single-step.** `Linear(hidden, 6)` sobre el último hidden state (`fc(h[-1])` → `(batch, 6)`), con **target escalar** = la clase a `+horizonte` (`window_series` devuelve un solo `y` por secuencia, [tth11_temporal_loader.py:153-156](../../ia_prediction_service/src/data/tth11_temporal_loader.py#L153)). Predice **un punto**.
- **Producción = multi-output directo (b1).** `Linear(hidden, 30×6)` con reshape a `(30, 6)`, **target vectorial** de 30 clases (t+1 … t+30), 30 softmax en una inferencia (§4).

**Consecuencia honesta a registrar (D-005):**
- La métrica del spike (`tth11_sweep_metrics.json`, mejor accuracy **0.7571**) midió un **single-step** y **no es comparable** con la de un multi-output. **TTH-09 reporta métricas propias desde cero** (§6), sin arrastrar el número del spike.
- El **81.3 % heredado (D-005)** queda **descartado** como referencia: su origen no es confiable ni reproducible, y no es techo del sistema.
- Las **palancas que el spike no usó a propósito** — class weights / rebalanceo del desbalance, tuning de hiperparámetros no-temporales, más datos — están **disponibles para TTH-09** (`tth-11-cierre-handoff.md` §4).

---

## 12. Migración a red real (4-way → OSM) — fuera de este contrato

El pipeline de datos está hoy **acoplado a la intersección 4-way genérica** (naming `*_in`/`*_out`, esquema NS/EW, detectores `LA_*`). El mapa OSM real de Miraflores ([miraflores.net.xml](../../simulation/conf/network/miraflores.net.xml)) ya está versionado (commit `f76bb60c`) pero **el pipeline aún no lo usa**. Migrar a la topología OSM multi-intersección es trabajo de TTH-09, pero **no altera este contrato de endpoint**: el shape per-intersección × 4 direcciones se mantiene; cambia el origen de las series, no su forma. Si la migración introduce intersecciones con != 4 accesos, el contrato deberá revisarse en bloque junto a los consumidores (ver §13).

---

## 13. Versionado del contrato

Este contrato es el **blanco de diseño de TTH-09** al 2026-05-31, con la arquitectura redefinida a **clasificador multi-output directo** y los hiperparámetros temporales cerrados por TTH-11. Cuando TTH-09 se implemente, el **shape vivo (schemas Pydantic)** debe **transcribirse acá** reemplazando los ejemplos de diseño por la referencia al código real, como hacen [vision_contract.md](vision_contract.md) y [engine_recommend_contract.md](engine_recommend_contract.md). Cualquier cambio posterior al shape del endpoint requiere actualizar este archivo **y** los consumidores (`frontend_ui/.../predictionService.ts`, HU-14) en bloque.

Cross-refs:
- **CTs canónicos**: [TAREAS_TECNICAS_HABILITADORAS.md](../lean-inception/2-backlog/TAREAS_TECNICAS_HABILITADORAS.md) CT-09.1 a CT-09.9 (líneas 542–558).
- **Hiperparámetros / contrato de TTH-11**: [tth-11-cierre-handoff.md](../handoffs/tth-11/tth-11-cierre-handoff.md) §2–§4; métricas del spike (single-step) en [tth11_sweep_metrics.json](../../ia_prediction_service/scripts/tth11_sweep_metrics.json).
- **Arquitectura del spike (clasificador single-step)**: [tth11_sweep.py](../../ia_prediction_service/scripts/tth11_sweep.py); ventaneo en [tth11_temporal_loader.py](../../ia_prediction_service/src/data/tth11_temporal_loader.py).
- **Delta-01 (contrato divergente del baseline)**: [AUDITORIA_HU_CODIGO.md](../lean-inception/planificacion/AUDITORIA_HU_CODIGO.md) §Delta-01 (línea 396); [SDD_CEREBROVIAL.md](../sdd/SDD_CEREBROVIAL.md) §(Delta-01).
- **D-006 (GRU univariado) / D-009 (jam level Waze 0-5)**: [DECISIONS.md](../lean-inception/4-decisiones/DECISIONS.md).
- **Mapeo `ratio_to_jam_level` (upstream del dataset, no en la inferencia)**: [jam_level.py](../../simulation/src/cerebrovial_simulation/jam_level.py).
- **Baseline RF preservado como respaldo Nivel 2**: [core_management_api/src/prediction/](../../core_management_api/src/prediction/); cascada en TTH-04.
- **Precedentes de "contrato canónico"** en `documentation/contracts/`: [vision_contract.md](vision_contract.md), [engine_recommend_contract.md](engine_recommend_contract.md).
</content>
