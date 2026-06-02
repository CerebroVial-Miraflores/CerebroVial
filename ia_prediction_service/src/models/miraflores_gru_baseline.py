"""Baseline GRU univariado solo-temporal del track STGNN (Fase 3).

Espejo estructural del GRU de producción (``gru_multioutput.py``, TTH-09) pero para
**regresión continua** de ``timeLoss`` (NO clasificación, NO softmax): una capa GRU
con cabeza Linear ancha que proyecta el último estado oculto a los ``horizonte``
pasos de salida. Multi-output DIRECTO (un solo forward, sin autoregresión).

Diferencias con el de producción (D-013 — target = demora continua):
- ``input_size=2``: canal 0 = ``timeLoss`` (estandarizado por el trainer), canal 1 =
  indicador de validez (1=válido, 0=vacío).
- cabeza ``Linear(hidden, horizonte)`` → salida ``[batch, horizonte]`` continua.

**Pesos compartidos**: un único modelo para los 375 nodos (cada nodo ve sólo su
propia serie; el modelo no distingue de qué nodo viene la ventana). tsl-free.
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
