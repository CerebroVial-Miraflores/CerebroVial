# TTH-09 Fase 3b — Endpoint de producción GRU + 5xx (handoff)

## Alcance entregado

Se expone el GRU en `POST /predictions/predict` con el contrato de
`documentation/contracts/prediction_contract.md` (§2-§3): request per-intersección con
4 series (N/S/E/W) de 30 `jam_levels` (enteros 0-5); response de 4 direcciones × 30
pasos, cada paso `{step, level=argmax, probs[6]=softmax}`, con `model_version`. El
backend normaliza `jam_level/5.0` antes de inferir (§2).

El GRU se carga eager al arranque (`GRUModelEngine.load_models()`, Fase 3a) y se sirve
in-process con `torch` CPU-only (D-010). El RandomForest baseline (`CongestionPredictor`,
`TrafficModelEngine`) **se preservó intacto** — no se tocó una línea de su código.

## Decisiones (alineadas al contrato)

- **Reemplazo total del path (§1):** el shape RF viejo dejó de exponerse en `/predict`;
  ese path es ahora exclusivamente GRU. El RF queda invocable internamente (lo usa
  `GET /predictions/history`, y lo invocará TTH-04 como respaldo Nivel 2).
- **Fallback CT-09.8 = solo 5xx (§7/§9):** este endpoint expone únicamente el GRU. Si el
  GRU no puede servir, emite 5xx; **la cascada al RF NO se cablea acá** — es TTH-04. Dos
  causas distinguibles por el body (para que TTH-04 discrimine en la cascada):
  - `503 "GRU model unavailable: …"` — modelo no cargado / falta alguna dirección
    (`GruModelUnavailableError`), o servicio DI no inicializado.
  - `500 "GRU inference failed: …"` — modelo cargado pero el forward falló.
- **Input malformado → 422** (validación Pydantic: ≠4 direcciones, len≠30, valor fuera 0-5).
- `model_version = "gru-clf-multioutput-v1"` (literal del contrato §3). Refinamiento futuro
  posible: leerlo del checkpoint; hoy es constante.

## Archivos

**Nuevos**
- `core_management_api/src/prediction/presentation/api/gru_schemas.py` — schemas `Gru*` (§2-§3).
- `core_management_api/src/prediction/application/gru_predictor.py` — `GruInferenceService`,
  `GruModelUnavailableError`, `GRU_MODEL_VERSION`.
- `core_management_api/tests/prediction/test_gru_routes.py` — tests del endpoint GRU.

**Modificados**
- `core_management_api/src/prediction/presentation/api/routes.py` — handler GRU en `/predict`
  + DI espejo (`init_gru_service`/`get_gru_service`); `GET /history` y el DI del RF intactos.
- `core_management_api/src/main.py` — wiring aditivo del GRU (instancia engine + `load_models()`
  + `init_gru_service(...)`), al lado del RF sin tocarlo.
- `core_management_api/tests/prediction/test_routes.py` — retiro de tests del shape RF viejo
  (ver abajo); conserva `test_get_history_success`.

**Preservados sin tocar:** `application/predictor.py` (`CongestionPredictor`),
`infrastructure/engine.py` (`TrafficModelEngine`), `presentation/api/schemas.py` (RF),
`infrastructure/gru_engine.py` y `infrastructure/gru_model.py` (Fase 3a).

## Cambios de tests (cobertura trasladada, no perdida)

| Test retirado (`test_routes.py`) | Qué probaba | Reemplazo (`test_gru_routes.py`) |
|---|---|---|
| `test_predict_traffic_success` | shape RF viejo en `/predict` | `test_predict_happy_path_shape` (shape GRU §3) |
| `test_predict_service_unavailable` | 503 DI no inicializado (RF) | `test_service_not_initialized_503` (503 DI GRU) |

`test_get_history_success` se conserva intacto (history no cambió). Tests nuevos
adicionales: `test_validation_{missing_direction,wrong_length,out_of_range}_422`,
`test_model_unavailable_503`, `test_inference_failure_500`.

## Dependencias / deuda declarada (no silenciosa)

- **HU-03 / Delta-01 — frontend roto:** `frontend_ui/src/services/predictionService.ts`
  → `CameraDetailView.tsx` (y `TrafficHistoryWidget.tsx` para el sub-objeto de predicción)
  consumen el shape RF viejo de `POST /predictions/predict` y **quedan rotos** hasta que
  HU-03 migre el cliente al shape de §2-§3. Ruptura conocida y documentada (también en el
  docstring del handler en `routes.py`), no silenciosa.
- **TTH-04:** cascada de fallback GRU→RF, adaptador RF→shape-§3 y `model_version` de respaldo.
- **Fase 3c:** deploy de los `.pt` al runtime del core (hoy el default apunta repo-relativo
  a `ia_prediction_service/models/`; override con `GRU_MODEL_DIR`).

## Verificación

- `python -m pytest tests/` desde `core_management_api/` (pendiente de correr por el usuario).
- `ruff check .` desde la raíz.
- Smoke 200: `POST /predictions/predict` con el ejemplo del contrato §2 → shape de §3.
- Smoke 5xx: arrancar con `GRU_MODEL_DIR` vacío → `/predict` responde 503 descriptivo
  (CT-09.8); el core bootea igual y `GET /predictions/history` (RF) sigue 200.
