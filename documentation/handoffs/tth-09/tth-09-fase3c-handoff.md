# TTH-09 Fase 3c — Deploy de los modelos GRU al runtime del core (handoff)

## Problema que cierra

El cargador GRU tenía como default `<repo>/ia_prediction_service/models` (computado con
`parents[4]`). Esa ruta **no existe dentro del contenedor del core** (el Dockerfile solo
copia `core_management_api/` a `/app`), así que los 4 `.pt` nunca se encontraban en deploy
y el core caía al RandomForest en silencio (la carga es tolerante a faltantes). El GRU
**jamás servía en contenedor**. Deuda declarada en el handoff de Fase 3b (§Dependencias).

## Solución (replica al pie del patrón RandomForest)

Los `.pt` se commitean por Git LFS **dentro de** `core_management_api/`, se hornean a la
imagen por el `COPY core_management_api/ .` existente, y se hallan por un default relativo
al CWD — exactamente como el RF sirve sus `.joblib` desde `models/`.

- **Sincronización (opción A, manual):** el entrenamiento (`tth09_train.py`) no se modifica;
  sigue produciendo los `.pt` en `ia_prediction_service/models/`. Para deploy, los 4 `.pt`
  + `metadata.json` se **copian** (no move, el origen se preserva) a
  `core_management_api/models/gru/` y se commitean por LFS. Proceso documentado en
  `core_management_api/models/gru/README.md`.
- **Ubicación:** subcarpeta `core_management_api/models/gru/`, separada de los `.joblib`
  del RF que viven en `core_management_api/models/`.

## Archivos

**Nuevos**
- `core_management_api/models/gru/{gru_N,gru_S,gru_E,gru_W}.pt` — copia LFS de los checkpoints.
- `core_management_api/models/gru/metadata.json` — procedencia del modelo servido (blob de
  texto normal, no LFS; el cargador no lo lee, cada `.pt` es autocontenido).
- `core_management_api/models/gru/README.md` — proceso de copia manual tras re-entrenar.

**Modificados**
- `core_management_api/src/prediction/infrastructure/gru_engine.py` — **solo el default**:
  `_DEFAULT_GRU_MODEL_DIR = "models/gru"` (antes repo-relativo vía `parents[4]`). Cleanup del
  `_REPO_ROOT` y el import `Path`, ahora muertos. La lógica de carga (`__init__` precedencia,
  `load_models`) y el override `GRU_MODEL_DIR` quedan intactos. Resuelve en contenedor
  (`/app/models/gru`) y en local (`core_management_api/models/gru`).
- `core_management_api/.gitignore` — `!models/gru/*.pt` para des-ignorar los `.pt` del GRU
  (la regla general `*.pt` sigue vigente para el resto). El `.gitattributes` raíz
  (`*.pt filter=lfs`) los cubre recursivamente.
- `tasks.py` (`check_lfs`) — segundo sentinela LFS `core_management_api/models/gru/gru_N.pt`
  junto al `.joblib` del RF, para que `invoke up` valide que los `.pt` están materializados.

**Preservados sin tocar:** Dockerfile (el `COPY` ya arrastra `models/gru/`), `engine.py` (RF),
`tth09_train.py` (entrenamiento), `tests/prediction/test_gru_engine.py` (paridad — sigue
leyendo de `ia_prediction_service/models/`, preservado).

## Verificación

- `ls core_management_api/models/gru/` → 4 `.pt` + `metadata.json` + `README.md`; `cmp` de los
  `.pt` contra `ia_prediction_service/models/` → idénticos.
- `git check-ignore core_management_api/models/gru/gru_N.pt` → sin output (des-ignorado);
  un `.pt` fuera de `models/gru/` sigue ignorado.
- Cargador local: `GRUModelEngine().model_dir == "models/gru"` y `load_models()` puebla N/S/E/W.
- `python -m pytest tests/` desde `core_management_api/` y `ruff check .` desde la raíz → verde.
- **Gate (LFS, lo crítico de 3c):** `git lfs ls-files | grep models/gru` lista los 4 `.pt`;
  `git show :core_management_api/models/gru/gru_N.pt` empieza con `version https://git-lfs`
  (entra como puntero LFS, no blob). `invoke check-lfs` pasa con el nuevo sentinela.
- Container real (opcional): `invoke up-build --service=core_management_api` → logs muestran
  `GRU {N,S,E,W} cargado desde models/gru/gru_*.pt` en vez del warning de faltante.
