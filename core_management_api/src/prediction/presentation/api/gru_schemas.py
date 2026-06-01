"""Schemas Pydantic del contrato GRU (TTH-09 §2-§3, Fase 3b).

Viven AL LADO de ``schemas.py`` (RF baseline), que se preserva intacto. Nombres
prefijados ``Gru*`` para no colisionar con ``PredictionInput`` / ``PredictionOutput``
del RF. El shape es el reemplazo total del contrato (§1): per-intersección, 4
direcciones (N/S/E/W), 30 pasos de horizonte, ``level`` + ``probs`` por paso.

El backend normaliza ``jam_level / 5.0`` antes de inferir (§2); aquí solo se valida
que la serie de entrada sea ``jam_level`` entero 0-5 (D-009), 30 pasos exactos.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_DIRECTIONS = ("N", "S", "E", "W")
_LOOKBACK = 30
_HORIZON = 30
_N_CLASSES = 6


class GruDirectionSeries(BaseModel):
    """Serie univariada de una dirección: 30 ``jam_levels`` (enteros 0-5, D-009)."""

    direction: Literal["N", "S", "E", "W"]
    jam_levels: List[int] = Field(
        ...,
        description="Lookback de 30 pasos, cronológico ascendente; cada valor entero 0-5.",
    )

    @field_validator("jam_levels")
    @classmethod
    def _exactly_30_in_range(cls, v: List[int]) -> List[int]:
        if len(v) != _LOOKBACK:
            raise ValueError(
                f"jam_levels debe tener exactamente {_LOOKBACK} elementos, recibió {len(v)}"
            )
        for x in v:
            if not (0 <= x <= 5):
                raise ValueError(f"jam_level fuera de rango [0,5]: {x}")
        return v


class GruPredictRequest(BaseModel):
    """Request del contrato §2: intersección + 4 series (N/S/E/W) de 30 jam_levels."""

    intersection_id: str = Field(..., min_length=1)
    timestamp: str = Field(..., description="ISO-8601; fin de la ventana de lookback.")
    series: List[GruDirectionSeries]

    @model_validator(mode="after")
    def _exactly_4_directions(self) -> "GruPredictRequest":
        dirs = [s.direction for s in self.series]
        if len(dirs) != 4 or set(dirs) != set(_DIRECTIONS):
            raise ValueError(
                f"series debe tener exactamente 4 direcciones {_DIRECTIONS} sin "
                f"duplicados ni faltantes, recibió {dirs}"
            )
        return self


class GruHorizonStep(BaseModel):
    """Un paso del horizonte: nivel argmax + distribución softmax sobre 6 clases."""

    step: int = Field(..., ge=1, le=_HORIZON)
    level: int = Field(..., ge=0, le=5)
    probs: List[float] = Field(..., min_length=_N_CLASSES, max_length=_N_CLASSES)


class GruDirectionPrediction(BaseModel):
    """Predicción de una dirección: 30 pasos de horizonte (t+1 … t+30)."""

    direction: str
    horizon: List[GruHorizonStep]


class GruPredictResponse(BaseModel):
    """Response del contrato §3: 4 direcciones, cada una con 30 pasos."""

    intersection_id: str
    generated_at: str
    model_version: str
    predictions: List[GruDirectionPrediction]
