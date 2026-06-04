"""Modelo GRU baseline univariado para regresión continua de ``timeLoss`` (Fase 3).

VENDORIZADO de ``ia_prediction_service/src/models/miraflores_gru_baseline.py`` (D-010,
patrón TTH-09). Copia byte-a-byte de la arquitectura. NO importar de
``ia_prediction_service`` (no es pip-instalable) ni mover a ``cerebrovial_shared`` (no
acoplar shared a torch). Mantener en sync con el original; el gate de reproducción
(``tests/congestion/test_prediction_window_gate.py``) verifica que esta copia carga el
state_dict real y produce la misma salida que la clase de origen.

Diferencia con el GRU 4-way de TTH-09 (``prediction/infrastructure/gru_model.py``): aquel
clasifica jam_level 0-5 por intersección (cabeza ``horizonte*n_classes``, logits); éste
hace REGRESIÓN continua de ``timeLoss`` por arista (cabeza ``Linear(hidden, horizonte)``,
sin softmax). Son dos predictores distintos.
"""
from __future__ import annotations

import torch
from torch import nn


class MirafloresGRUBaseline(nn.Module):
    """GRU univariada multi-output para regresión continua de ``timeLoss``.

    forward(x): x shape ``(batch, lookback, 2)`` → ``(batch, horizonte)``.
    Devuelve la demora predicha (continua) de los ``horizonte`` pasos; sin softmax.
    """

    def __init__(self, hidden: int = 64, horizonte: int = 30, input_size: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.horizonte = horizonte
        self.input_size = input_size
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden, num_layers=1, batch_first=True
        )
        # Cabeza ancha: hidden -> horizonte (regresión, p. ej. 64 -> 30).
        self.head = nn.Linear(hidden, horizonte)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)        # h: (num_layers, batch, hidden)
        return self.head(h[-1])   # (batch, horizonte)
