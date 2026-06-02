"""
Time-then-Space Spatiotemporal Graph Neural Network model.

STGNN base B (Fase 4): reusa únicamente capas de ``tsl`` (NodeEmbedding, RNN,
DiffConv, MLPDecoder). El resto del andamiaje (split, scaler, índice de ventanas,
loop y métricas) es propio del baseline de Fase 3, para que la única variable nueva
respecto al GRU sea la componente espacial DiffConv.

Predice **solo** ``timeLoss`` (1 canal de salida), independientemente de cuántos
canales de entrada tenga (entrada Miraflores = 2: timeLoss + indicador de validez).
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from tsl.nn.blocks import RNN, MLPDecoder
from tsl.nn.layers import NodeEmbedding, DiffConv


class TimeThenSpaceModel(nn.Module):
    """
    Spatiotemporal Graph Neural Network con arquitectura Time-then-Space.

    Flujo: ``[B, T, N, F]`` → (embedding de nodo + encoder lineal) → RNN temporal
    (último estado) → DiffConv espacial → (embedding de nodo + decoder MLP) →
    ``[B, horizon, N, output_size]``.
    """

    def __init__(
        self,
        input_size: int = 2,
        n_nodes: int = 375,
        horizon: int = 30,
        emb_size: int = 10,
        hidden_size: int = 64,
        rnn_layers: int = 1,
        gnn_kernel: int = 2,
        rnn_cell: str = 'gru',
        dropout: float = 0.0,
        output_size: int = 1,
    ):
        super(TimeThenSpaceModel, self).__init__()

        self.input_size = input_size
        self.n_nodes = n_nodes
        self.horizon = horizon
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        # Salida desacoplada de la entrada: el modelo predice solo timeLoss (1 canal),
        # NO el indicador de validez. No acoplar a input_size.
        self.output_size = output_size

        # Node embeddings
        self.node_embeddings = NodeEmbedding(n_nodes, emb_size)

        # Encoder
        self.encoder = nn.Linear(input_size + emb_size, hidden_size)

        # Temporal processor: RNN
        self.time_nn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=rnn_layers,
            cell=rnn_cell,
            return_only_last_state=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        # Spatial processor: Diffusion Convolution
        self.space_nn = DiffConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            k=gnn_kernel,
            root_weight=True
        )

        # Decoder
        self.decoder = MLPDecoder(
            input_size=hidden_size + emb_size,
            hidden_size=2 * hidden_size,
            output_size=output_size,
            horizon=horizon,
            n_layers=1,
            dropout=dropout
        )

    def forward(self, x, edge_index, edge_weight):
        """Forward pass of the model."""
        b, t, n, f = x.size()

        # Concatenate node embeddings to input
        emb = self.node_embeddings(expand=(b, t, -1, -1))
        x_emb = torch.cat([x, emb], dim=-1)

        # Encode
        x_enc = self.encoder(x_emb)

        # Temporal processing
        h = self.time_nn(x_enc)

        # Spatial processing
        z = self.space_nn(h, edge_index, edge_weight)

        # Decode
        emb = self.node_embeddings(expand=(b, -1, -1))
        z_emb = torch.cat([z, emb], dim=-1)
        x_out = self.decoder(z_emb)

        return x_out


def create_model(config: Dict[str, Any], n_nodes: int = 375, input_size: int = 2) -> TimeThenSpaceModel:
    """
    Factory para crear el modelo desde una configuración.

    Defaults Miraflores (grafo LCC 375 nodos, horizonte 30, entrada 2 canales).
    El decoder fija ``output_size=1`` (predice solo timeLoss).

    Args:
        config: diccionario de configuración del modelo
        n_nodes: número de nodos del grafo (default 375 — LCC Miraflores)
        input_size: número de features de entrada (default 2 — timeLoss + validez)

    Returns:
        Modelo inicializado.
    """
    model = TimeThenSpaceModel(
        input_size=input_size,
        n_nodes=n_nodes,
        horizon=config.get('horizon', 30),
        emb_size=config.get('emb_size', 10),
        hidden_size=config.get('hidden_size', 64),
        rnn_layers=config.get('rnn_layers', 1),
        gnn_kernel=config.get('gnn_kernel', 2),
        output_size=config.get('output_size', 1),
    )
    return model
