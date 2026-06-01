"""Persistence repository for GRU predictions (TTH-09 Fase 5, CT-09.5).

``PredictionsRepo`` is append-only by construction: it exposes only
``insert_batch(...)``. The absence of ``update``/``delete`` is enforced as an
invariant of the application layer (mirrors ``MotorDecisionsRepo``), not as a
DB-side constraint.

DECISIÓN CONSCIENTE de capas (no accidente): este repo (infraestructura) importa
``GruPredictResponse`` de la capa de presentación. Es un leve smell, pero el diseño
acordado es que ``insert_batch`` reciba la *response* completa de la inferencia y la
aplane a las 120 filas en una sola transacción; el repo solo consume el DTO de lectura,
no lo construye. La alternativa (pasar primitivos desde el handler) duplicaría el
aplanado 4×30 en presentación. Se acepta el acoplamiento a cambio de un handler delgado.
"""
from datetime import datetime

from sqlalchemy.orm import Session

from cerebrovial_shared.database.models import PredictionDB

from ..presentation.api.gru_schemas import GruPredictResponse


class PredictionsRepo:
    """Append-only repository for ``predictions`` (CT-09.5). No update/delete."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def insert_batch(self, response: GruPredictResponse) -> int:
        """Persiste las 4×30 = 120 filas de una inferencia en un solo flush.

        Reusa el ``generated_at`` de la response (contrato §8: el timestamp del
        registro persistido ≡ ``generated_at`` devuelto al cliente), parseando el
        ISO-8601 tz-aware a ``datetime``. Devuelve el conteo de filas insertadas.
        """
        generated_at = datetime.fromisoformat(response.generated_at)
        rows = [
            PredictionDB(
                intersection_id=response.intersection_id,
                direction=pred.direction,
                step=hstep.step,
                level=hstep.level,
                probs=list(hstep.probs),
                model_version=response.model_version,
                generated_at=generated_at,
            )
            for pred in response.predictions
            for hstep in pred.horizon
        ]
        self.session.add_all(rows)
        self.session.flush()
        return len(rows)
