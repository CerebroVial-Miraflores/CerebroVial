"""Cargador de los modelos GRU direccionales (TTH-09 Fase 3a, D-010).

Andamiaje GRU que vive *al lado* del baseline RandomForest (``engine.py``), sin
tocarlo. Carga eager los 4 checkpoints ``gru_<dir>.pt`` en memoria (patrón análogo
a ``TrafficModelEngine.load_models``) y los deja en ``.eval()`` listos para servir.

Alcance Fase 3a: SOLO cargar a memoria + verificar compatibilidad con la clase
vendorizada (``gru_model.GRUMultiOutput``). El endpoint, el wiring a ``main.py`` y el
fallback formal GRU->RF (CT-09.8) son Fase 3b. Por eso este cargador NO se engancha
al arranque todavía; lo ejercita el test de paridad.

Tolerante a faltantes (CT-09.8): si un .pt no existe, es un puntero LFS sin
materializar o falla la carga, se loguea y se omite —el core puede bootear igual y
caer al RandomForest (el fallback se cablea en 3b). El RF queda como respaldo Nivel 2.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from cerebrovial_shared.lfs_check import assert_real_binary

from src.prediction.infrastructure.gru_model import GRUMultiOutput

logger = logging.getLogger(__name__)

# Path por defecto de los .pt. HONESTO: hoy los checkpoints viven en el módulo de
# entrenamiento (``ia_prediction_service/models/``), que no es runtime del core. Este
# default repo-relativo apunta a esa ubicación real para que 3a/local cargue de
# verdad; el deploy de los .pt al runtime del core (path en contenedor) es deuda
# explícita de Fase 3c. Override en runtime con la env var ``GRU_MODEL_DIR``.
#   gru_engine.py -> infrastructure[0] -> prediction[1] -> src[2]
#                 -> core_management_api[3] -> <repo root>[4]
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_GRU_MODEL_DIR = str(_REPO_ROOT / "ia_prediction_service" / "models")


class GRUModelEngine:
    """Carga y custodia los 4 modelos GRU direccionales en memoria (CPU, eval).

    Espeja la forma de ``TrafficModelEngine``: ``__init__`` fija el directorio,
    ``load_models()`` puebla ``self.models`` (dict ``direction -> GRUMultiOutput``).
    """

    DIRECTIONS = ("N", "S", "E", "W")

    def __init__(self, model_dir: str | None = None):
        # Precedencia: argumento explícito > env GRU_MODEL_DIR > default repo-relativo.
        self.model_dir = (
            model_dir
            or os.environ.get("GRU_MODEL_DIR")
            or _DEFAULT_GRU_MODEL_DIR
        )
        self.models: dict[str, GRUMultiOutput] = {}

    def load_models(self) -> None:
        """Carga eager cada ``gru_<dir>.pt`` disponible. Tolerante a faltantes."""
        for direction in self.DIRECTIONS:
            path = os.path.join(self.model_dir, f"gru_{direction}.pt")
            if not os.path.exists(path):
                logger.warning(
                    "GRU %s no encontrado en %s — se omite (fallback a RF en 3b)",
                    direction,
                    path,
                )
                continue
            try:
                # Guarda LFS: rechaza punteros sin materializar (mismo patrón que el RF).
                assert_real_binary(path)
                # weights_only=False es NECESARIO: el checkpoint es un dict autocontenido
                # con metadata no-tensor (version, hyperparams, class_weights, counts), no
                # solo el state_dict. PyTorch 2.6+ default weights_only=True lo rechazaría.
                # Es seguro: el artefacto lo produce nuestro propio entrenamiento
                # (ia_prediction_service/scripts/tth09_train.py), fuente confiable; no se
                # cargan checkpoints de terceros.
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                model = GRUMultiOutput(
                    hidden=int(ckpt["hyperparams"]["hidden"]),
                    horizonte=int(ckpt["horizonte"]),
                    n_classes=int(ckpt["n_classes"]),
                )
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                self.models[direction] = model
                logger.info("GRU %s cargado desde %s", direction, path)
            except Exception as e:  # noqa: BLE001 - loguea y sigue, no crashea
                logger.error("Falló carga GRU %s desde %s: %s", direction, path, e)

    @property
    def loaded_directions(self) -> tuple[str, ...]:
        """Direcciones efectivamente cargadas (introspección / tests)."""
        return tuple(d for d in self.DIRECTIONS if d in self.models)

    @property
    def available(self) -> bool:
        """True si hay al menos un modelo GRU cargado."""
        return bool(self.models)
