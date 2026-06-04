# Loader de inferencia model-agnostic — Fase 1 (cierre)

**Fecha:** 2026-06-03 · **Rama:** `feature/inference-loader` (desde `03153666`, HEAD de
`feature/multiseed-d015`) · **Venv:** `ia_prediction_service/.venv` (Python 3.11, tsl 0.9.6,
torch 2.9.1, CPU).

## Qué entregó

Módulo nuevo `ia_prediction_service/src/inference/` (greenfield, separado del clúster
Lightning muerto) que carga cualquiera de los dos `.pt` del track —GRU baseline
(`MirafloresGRUBaseline`) o STGNN (`TimeThenSpaceModel`)— por **ruta explícita** y expone
un contrato de inferencia **idéntico para ambos modelos**:

```
predict(window [30, 1660, 2]) -> [1660, 30]   # timeLoss en segundos por arista × 30 pasos
```

- `window [30, 1660, 2]`: 30 pasos de lookback, 1660 aristas, 2 canales (timeLoss + validez).
- salida `[1660, 30]`: eje 0 = `node_index` 0..1659 (mapeable a `sumo_edge` vía
  `miraflores_graph_lcc_mapping.json`), eje 1 = 30 pasos de horizonte; en segundos.

Las **tres asimetrías** entre modelos quedan ENTERAMENTE dentro de cada adaptador, no en el
caller:

| | GRU (`GruAdapter`) | STGNN (`StgnnAdapter`) |
|---|---|---|
| forward | `model(x)` | `model(x, edge_index, edge_weight)` (grafo LCC cacheado) |
| layout | 1660 series `[1660,30,2]` (nodo en batch) | snapshot `[1,30,1660,2]` (nodos en eje N) |
| reconstrucción | `Model(**arch)` directo | arch remapeado (`hidden→hidden_size`, `emb→emb_size`, `horizonte→horizon`; `rnn_cell`/`dropout` a defaults) |

### Archivos
- `src/inference/preprocessing.py` — `standardize_window` + `destandardize` (lógica del eval
  RE-EXTRAÍDA, no importada de los scripts de training).
- `src/inference/adapters.py` — `InferenceAdapter` (ABC) + `GruAdapter` + `StgnnAdapter`.
- `src/inference/registry.py` — `str → adaptador`; listo para alias de una línea (ver deuda).
- `src/inference/loader.py` — `load_inference_adapter(ckpt_path, device="cpu")`, entrada única.
- `src/inference/__init__.py` — API pública.
- `tests/test_inference_loader.py` — gate de reproducción + orden + unitarios.

No se tocaron: modelos congelados (`miraflores_gru_baseline.py`, `time_then_space.py`),
scripts de training, ni el core. No se entrenó.

## Gate de Fase 1 (condición de cierre) — verde y honesto

`cd ia_prediction_service && .venv/bin/python -m pytest tests/test_inference_loader.py -v`
→ **9 passed, 1 skipped** (3.6 s). Con `RUN_SLOW=1`, el test agregado opt-in: **+1 passed** (23 s).

El loader reproduce el `predict_collect` del eval offline existente (mismas predicciones),
con dos blindajes que hacen que el verde pruebe algo:

1. **Alineación de orden de nodos (GRU).** Antes del `allclose`, el test asierta
   `np.array_equal(Xb, transpose(window))` —que el índice GRU del oráculo (construido
   explícito `(day,start,node)` en orden `node_index`) y el window del adaptador recorren las
   1660 aristas en el mismo orden **por construcción**. Si fallara, el test revienta ahí
   (desalineo localizado), no en el `allclose`.
2. **CPU en ambos lados.** Loader en `device="cpu"`; oráculo forzado a CPU
   (`train_miraflores_baseline.DEV = "cpu"`, `dev="cpu"` al STGNN). Tolerancia estricta
   `atol=1e-5, rtol=1e-4` sin aflojar → no se persigue un falso negativo CPU-vs-MPS.

Reproducción **doble**: per-predicción (gate, K=8 snapshots) + métricas agregadas del loader
sobre todo el test set vs `miraflores_baseline_metrics.json` (opt-in `RUN_SLOW=1`, tol 1e-2).
Ambas verdes. Más: integridad del mapping (`node_index==posición`, arista 0 = `-1098384939`),
order-equivariance del GRU, registry/errores/shapes.

## Deuda registrada al cierre (ninguna de código)

1. **Rename futuro `TimeThenSpaceModel → MirafloresSTGNNModel`.** El registry ya queda listo:
   agregar `"MirafloresSTGNNModel": StgnnAdapter` en `registry.py` (una línea) hace que ambos
   strings —checkpoints viejos y nuevos— resuelvan al mismo adaptador. El remapeo de `arch` y
   la decisión grafo/no-grafo viven en el adaptador, así que el alias no toca nada más.

2. **Setup del venv de training/inferencia — pytest (gap de arranque).** El gate STGNN importa
   `tsl`, que vive solo en `ia_prediction_service/.venv`; ese venv no traía `pytest` (estaba
   solo en el venv raíz). Se instaló `pytest>=8.0.0` en el venv de ia (dev-dep declarada en
   `requirements-dev.txt`). **Este cambio es estado local del venv, NO lo captura el commit.**
   Un clon limpio necesita instalar las dev-deps en el venv de training para correr el gate.
   Esto **dispara y materializa** la "Deuda pytest" ya registrada en ESTADO (2026-06-01), que
   anticipaba exactamente este caso ("cuando un test del módulo dependa de tsl, instalar pytest
   en el venv de training"). Follow-up de doc: que el README/setup de `ia_prediction_service`
   mencione el requisito de dev-deps en el venv de training para los tests.

3. **Suite de `ia_prediction_service` — 2 fallas + 1 error preexistentes, ajenos al loader.**
   Al correr `pytest tests/` completo aparecen: `test_data_loader.py` (2 fallas,
   `ValueError: Invalid frequency: 5T` — pandas deprecó el alias `T`→`min`) y
   `test_miraflores_dataset_builder.py` (1 error de colección, `ModuleNotFoundError: pyarrow`,
   deuda de venvs ya conocida). **Verificado que fallan SIN el código del loader**
   (`test_data_loader.py` falla en aislamiento). Se registran para que una corrida futura de la
   suite completa no atribuya estas roturas al loader. No se arreglan en esta fase (fuera de
   scope). Disparadores: pandas `5T→5min` en `test_data_loader.py`; instalar `pyarrow` en el
   venv para los tests de dataset builder.

## Siguiente (Fase 2)

Conversión `timeLoss` (segundos) → nivel 0-5. No existe en ningún lado todavía; incluye la
sub-decisión de los **cortes** (qué demora mapea a cada nivel), a resolver con la
**distribución real de `timeLoss`**, no a ojo. Se planea aparte.
