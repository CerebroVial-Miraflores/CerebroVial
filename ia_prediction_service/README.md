# ia_prediction_service

## Qué es
Pipeline de entrenamiento offline para los modelos de predicción de
tráfico. Basado en PyTorch Lightning + tsl. **NO es un servicio HTTP.**
No tiene FastAPI, no expone endpoints, no se ejecuta como contenedor
permanente.

## Por qué no está en docker-compose.yml
Porque no es un servicio. Se invoca manualmente cuando hace falta
(re)entrenar modelos. Mantenerlo en `docker compose up` rompía el
arranque del sistema entero.

## Cómo correrlo

### Opción A — Sin Docker (desarrollo local)
El venv de **entrenamiento** vive en `ia_prediction_service/.venv` — AISLADO del venv
core+visión de la raíz (`.venv`). Están separados a propósito: tsl ancla `numpy<2` y
choca con opencv/visión (ver `documentation/ESTADO_Y_PROXIMOS_PASOS.md`).

Lo más simple, desde la raíz del repo:
```bash
invoke setup-train
```
O a mano (equivalente):
```bash
cd ia_prediction_service
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt   # instalar desde acá para que `-e ../shared` resuelva
```
Después, desde la raíz del repo:
```bash
ia_prediction_service/.venv/bin/python \
  ia_prediction_service/scripts/train_miraflores_baseline.py --quick
```

### Opción B — Con Docker (reproducible, recomendado para CI)
```bash
# Build (desde la raíz del repo)
docker build -f ia_prediction_service/Dockerfile \
             -t cerebrovial-trainer \
             ia_prediction_service/

# Run
docker run --rm \
  -v $(pwd)/models:/app/models \
  cerebrovial-trainer
```

El volumen `-v models:/app/models` es para que el modelo entrenado
quede en el host, no se pierda al cerrar el contenedor.

## Estado actual del modelo
Hoy este pipeline entrena un STGNN. Según `docs/DECISIONS.md`, se va
a reemplazar por GRU en Fase 3. El código del STGNN se mantiene como
referencia hasta que el GRU esté validado.

## Salidas
- Checkpoints en `notebooks/logs/`
- Modelo final en `models/` (montado como volumen si se usa Docker)
