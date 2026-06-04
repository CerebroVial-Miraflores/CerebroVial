# `core_management_api/models/` — artefactos servidos por el core

Binarios horneados a la imagen del core vía `COPY core_management_api/ .` (Dockerfile).
Versionados por Git LFS; no se entrenan en el core (D-010: torch CPU-only solo inferencia).

## Predicción servida (Fase 3 — congestion/)

### `miraflores_gru_baseline.pt` (~64 KB)
Checkpoint del **GRU baseline univariado** (`model="MirafloresGRUBaseline"`, regresión
continua de `timeLoss`, por arista, 30 pasos). `arch={input_size:2, hidden:64, horizonte:30}`,
scaler embebido (`ckpt["scaler"]`). Servido por `GET /congestion/prediction`.

- **Origen**: `ia_prediction_service/models/miraflores_gru_baseline.pt` (track STGNN, Fase 1
  de inferencia). Copia manual a este directorio (mismo trato que los `gru/*.pt` de TTH-09).
- **Sync**: si se reentrena el baseline, recopiar acá. El gate de reproducción
  (`tests/congestion/test_prediction_window_gate.py`) verifica que el camino vendorizado
  reproduce el loader de Fase 1 sobre el mismo checkpoint.

### `day_seed051_tensor.npz` (~1.8 MB)
Slice **auto-contenido** del día **seed051** (`day_idx=9`) del dataset de training: `tensor`
`[1440,1660,1]` (canal 0 = `timeLoss` s, NaN en vacío) + `mask` `[1440,1660]` + `node_order`
`[1660]` (sumo_edge en orden `node_index`) + `seed`/`day_idx`. Es la fuente fiel de la ventana
de inferencia (el modelo se entrenó sobre este tensor).

- **Procedencia**: `day_idx=9 ↔ seed051` (verificado: `metadata.json` del `.npz` completo +
  array `seeds`; regla día N ↔ seed 42+N). El lunes 8 jun de la demo = seed051.
- **Por qué un slice y no el `.npz` completo**: el `.npz` completo (574 MB descomprimido)
  excede el límite de 512 MB del core y está gitignored. El slice descomprime a ~12 MB en RAM.
- **Regenerar** (desde la raíz del repo, con el root `.venv` que tiene numpy):
  ```bash
  .venv/bin/python ia_prediction_service/scripts/extract_day_slice.py
  ```
  El script corre el **stage-gate 1** (byte-idéntico al slice del `.npz` completo, incl.
  `node_order`) y aborta si no coincide. Requiere el `.npz` completo presente localmente
  (gitignored/regenerable vía `miraflores_dataset_builder`).
