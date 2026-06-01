"""Modelo GRU univariado multi-output por dirección (TTH-09, arquitectura b1).

VENDORIZADO de ``ia_prediction_service/src/models/gru_multioutput.py`` (D-010, TTH-09
Fase 3a). Copia byte-a-byte de la arquitectura b1. NO importar de
``ia_prediction_service`` (ese módulo no es pip-instalable) ni mover a
``cerebrovial_shared`` (no acoplar shared a torch). Mantener en sync con el original;
el test de paridad (``tests/prediction/test_gru_engine.py``) verifica que esta copia
carga los state_dicts reales y produce la misma salida que la clase de origen.

----------------------------------------------------------------------------------
Una capa GRU univariada (input_size=1, jam_level normalizado /5.0) con una cabeza
Linear ancha que proyecta el último estado oculto a ``horizonte * n_classes`` y se
reshapea a ``(batch, horizonte, n_classes)``. Multi-output DIRECTO: predice las
``horizonte`` clases de un solo forward, sin autoregresión ni seq2seq.

Se entrena un modelo independiente por dirección (D-006: N/S/E/W).
"""

from __future__ import annotations

import torch
from torch import nn


class GRUMultiOutput(nn.Module):
    """GRU univariada con cabeza ancha multi-output (b1).

    forward(x): x shape (batch, lookback, 1) -> logits (batch, horizonte, n_classes).
    Devuelve logits crudos (sin softmax); el softmax es implícito en CrossEntropyLoss.
    """

    def __init__(
        self, hidden: int = 64, horizonte: int = 30, n_classes: int = 6
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.horizonte = horizonte
        self.n_classes = n_classes
        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden, num_layers=1, batch_first=True
        )
        # Cabeza ancha: hidden -> horizonte * n_classes (p. ej. 64 -> 30*6 = 180).
        self.head = nn.Linear(hidden, horizonte * n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)  # h: (num_layers, batch, hidden)
        logits = self.head(h[-1])  # (batch, horizonte * n_classes)
        return logits.view(-1, self.horizonte, self.n_classes)  # (batch, horizonte, n_classes)
