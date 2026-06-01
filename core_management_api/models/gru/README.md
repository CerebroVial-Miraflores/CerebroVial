# Modelos GRU servidos por el core (TTH-09 Fase 3c)

Este directorio contiene los 4 checkpoints GRU direccionales (`gru_{N,S,E,W}.pt`)
y su `metadata.json`, **horneados a la imagen del core** para que el predictor GRU
sirva en runtime de deploy.

## Por qué viven acá

El cargador ([`src/prediction/infrastructure/gru_engine.py`](../../src/prediction/infrastructure/gru_engine.py))
busca los `.pt` en `models/gru/` **relativo al CWD** (`/app/models/gru` en contenedor,
`core_management_api/models/gru/` en local) — mismo patrón que el RandomForest baseline
(`models/*.joblib`, ver `engine.py`). El `COPY core_management_api/ .` del
[`Dockerfile`](../../Dockerfile) los arrastra a la imagen automáticamente; no hace falta
un `COPY` explícito. Los `.pt` se versionan por **Git LFS** (regla `*.pt filter=lfs`
en el `.gitattributes` raíz), des-ignorados vía `!models/gru/*.pt` en
[`core_management_api/.gitignore`](../../.gitignore).

> El cargador **no lee** `metadata.json` — cada `.pt` es un checkpoint autocontenido
> (hyperparams, horizonte, n_classes, class_weights). `metadata.json` se versiona acá
> solo por procedencia/auditoría del modelo servido; se commitea como blob de texto
> normal (no LFS).

## Proceso de copia tras re-entrenar (manual — sincronización opción A)

El entrenamiento (`ia_prediction_service/scripts/tth09_train.py`) **no se modifica**:
sigue produciendo los `.pt` en `ia_prediction_service/models/`, que es la ubicación
fija del flujo de entrenamiento/evaluación y la que lee el test de paridad.

Tras re-entrenar el GRU, **copiar manualmente** los artefactos a este directorio y
commitearlos por LFS — igual que se hace con los `.joblib` del RF:

```sh
# desde la raíz del repo (CerebroVial/)
cp ia_prediction_service/models/gru_{N,S,E,W}.pt \
   ia_prediction_service/models/metadata.json \
   core_management_api/models/gru/

git add core_management_api/models/gru/
git lfs ls-files | grep models/gru   # confirmar que los 4 .pt quedan tracked por LFS
git commit -m "actualizar modelos GRU servidos por el core"
```

Es **copia, no move**: el original en `ia_prediction_service/models/` se preserva.
El proceso es manual y deliberado (no automatizado) — queda registrado acá para que
no dependa de la memoria de nadie.
