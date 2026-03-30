"""Transformer helpers for SG temporal modeling."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_pe(seq_len: int, d_model: int) -> Tensor:
    """Create standard sinusoidal positional encodings."""
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.detach()


class SGTransformerEncoder(nn.Module):
    """Temporal encoder over stacked frame-wise node embeddings."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, node_embeds: Tensor) -> Tensor:
        """Encode ``[T, num_nodes, hidden_dim]`` into ``[T, d_model]``."""
        seq_len, num_nodes, hidden_dim = node_embeds.shape
        flattened = node_embeds.reshape(seq_len, num_nodes * hidden_dim)
        projected = self.input_proj(flattened)
        pe = sinusoidal_pe(seq_len, self.d_model).to(projected.device)
        encoded = self.encoder((projected + pe).unsqueeze(0)).squeeze(0)
        return encoded


class SGTransformerDecoder(nn.Module):
    """Temporal decoder from latent vectors back to node-level predictions."""

    def __init__(
        self,
        latent_dim: int = 64,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        num_vehicles: int = 4,
        node_feat_dim: int = 4,
        node_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.node_out = nn.Linear(d_model, num_vehicles * node_feat_dim)
        self.node_hidden_out = nn.Linear(d_model, num_vehicles * node_hidden_dim)
        self.d_model = d_model
        self.num_vehicles = num_vehicles
        self.node_feat_dim = node_feat_dim
        self.node_hidden_dim = node_hidden_dim

    def forward(self, z: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        """Decode a latent vector into node features and node hidden states."""
        if z.dim() == 1:
            repeated = z.unsqueeze(0).repeat(seq_len, 1)
            memory = self.latent_proj(repeated)
            pe = sinusoidal_pe(seq_len, self.d_model).to(memory.device)
            memory = (memory + pe).unsqueeze(0)
            decoded = self.decoder(tgt=memory, memory=memory).squeeze(0)
            final_token = decoded[-1]

            node_pred = self.node_out(final_token).reshape(self.num_vehicles, self.node_feat_dim)
            node_hidden = self.node_hidden_out(final_token).reshape(
                self.num_vehicles,
                self.node_hidden_dim,
            )
            return node_pred, node_hidden

        batch_size = z.size(0)
        repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        memory = self.latent_proj(repeated)
        pe = sinusoidal_pe(seq_len, self.d_model).to(memory.device).unsqueeze(0)
        memory = memory + pe
        decoded = self.decoder(tgt=memory, memory=memory)
        final_token = decoded[:, -1, :]

        node_pred = self.node_out(final_token).reshape(batch_size, self.num_vehicles, self.node_feat_dim)
        node_hidden = self.node_hidden_out(final_token).reshape(
            batch_size,
            self.num_vehicles,
            self.node_hidden_dim,
        )
        return node_pred, node_hidden
