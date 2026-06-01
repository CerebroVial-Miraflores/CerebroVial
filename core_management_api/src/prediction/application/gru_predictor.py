"""Servicio de inferencia GRU (TTH-09 Fase 3b).

Orquesta el ``GRUModelEngine`` (Fase 3a): recibe la request validada del contrato §2,
normaliza ``/5.0``, corre los 4 forwards (N/S/E/W), deriva ``level = argmax`` y
``probs = softmax`` por paso, y arma la ``GruPredictResponse`` del contrato §3.

Análogo a ``CongestionPredictor`` (RF), pero SEPARADO: el RF se preserva intacto como
respaldo Nivel 2 (no se importa ni se toca aquí). Este servicio expone ÚNICAMENTE el
GRU principal. Si el GRU no puede servir, lanza ``GruModelUnavailableError`` (-> 503 en
el handler) o propaga el error de inferencia (-> 500); la cascada al RF (CT-09.8) es
responsabilidad de TTH-04, no de este módulo.

El idiom de inferencia (tensor ``(1, lookback, 1)`` normalizado ``/5.0`` + ``eval`` +
``no_grad`` -> logits ``(1, 30, 6)``) es el mismo que verificó el test de paridad de
Fase 3a (``tests/prediction/test_gru_engine.py``), extendido con softmax/argmax.
"""

from __future__ import annotations

import torch

from src.prediction.infrastructure.gru_engine import GRUModelEngine
from src.prediction.presentation.api.gru_schemas import (
    GruDirectionPrediction,
    GruHorizonStep,
    GruPredictRequest,
    GruPredictResponse,
)

# Literal del contrato §3. Discrimina principal vs respaldo en el registro (§8 / HU-20).
GRU_MODEL_VERSION = "gru-clf-multioutput-v1"

_DIRECTIONS = ("N", "S", "E", "W")


class GruModelUnavailableError(RuntimeError):
    """El GRU no puede servir: engine sin modelos o falta una dirección requerida.

    Se mapea a 503 en el handler (CT-09.8: modelo no cargado -> 5xx). El body lo
    distingue del 500 de inferencia para que TTH-04 discrimine la causa en la cascada.
    """


class GruInferenceService:
    """Inferencia GRU end-to-end sobre los 4 modelos direccionales cargados en memoria."""

    def __init__(self, engine: GRUModelEngine):
        self.engine = engine

    def predict(self, req: GruPredictRequest, generated_at: str) -> GruPredictResponse:
        """Corre los 4 forwards y arma la response del contrato §3.

        ``generated_at`` lo inyecta el handler (no se usa reloj interno acá) para que la
        inferencia sea determinista y testeable.
        """
        if not self.engine.available:
            raise GruModelUnavailableError(
                "GRU engine sin modelos cargados (ningún gru_<dir>.pt disponible)"
            )
        missing = [d for d in _DIRECTIONS if d not in self.engine.models]
        if missing:
            raise GruModelUnavailableError(
                f"GRU engine sin modelo(s) para dirección(es) {missing}; "
                "se requieren las 4 (N/S/E/W) para el contrato §3"
            )

        series_by_dir = {s.direction: s for s in req.series}
        predictions: list[GruDirectionPrediction] = []
        for direction in _DIRECTIONS:
            jam_levels = series_by_dir[direction].jam_levels
            # (1, lookback, 1) normalizado /5.0 — idiom validado en test_gru_engine.py.
            x = torch.tensor(jam_levels, dtype=torch.float32).reshape(
                1, len(jam_levels), 1
            ) / 5.0
            with torch.no_grad():
                logits = self.engine.models[direction](x)  # (1, 30, 6)
            probs = torch.softmax(logits, dim=-1)[0]  # (30, 6)
            levels = probs.argmax(dim=-1)  # (30,) — level = argmax(probs), §3
            horizon = [
                GruHorizonStep(
                    step=k + 1,
                    level=int(levels[k].item()),
                    probs=[float(p) for p in probs[k].tolist()],
                )
                for k in range(probs.shape[0])
            ]
            predictions.append(
                GruDirectionPrediction(direction=direction, horizon=horizon)
            )

        return GruPredictResponse(
            intersection_id=req.intersection_id,
            generated_at=generated_at,
            model_version=GRU_MODEL_VERSION,
            predictions=predictions,
        )
