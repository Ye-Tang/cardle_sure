"""Latent sampling and edge decoding utilities for VGAE."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VAESampleLayer(nn.Module):
    """Reparameterized Gaussian latent sampler."""

    def __init__(self, d_model: int = 128, latent_dim: int = 64) -> None:
        super().__init__()
        self.mu_proj = nn.Linear(d_model, latent_dim)
        self.logvar_proj = nn.Linear(d_model, latent_dim)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return mu, logvar, z


class EdgeDecoder(nn.Module):
    """Decode edge probabilities and edge attributes from node embeddings."""

    def __init__(self, node_dim: int = 64, edge_feat_dim: int = 5) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, edge_feat_dim),
        )

    def edge_prob_forward(self, node_embeds: Tensor) -> Tensor:
        if node_embeds.dim() == 2:
            logits = node_embeds @ node_embeds.t()
            probs = torch.sigmoid(logits)
            mask = 1.0 - torch.eye(probs.size(0), device=probs.device, dtype=probs.dtype)
            return probs * mask

        logits = torch.matmul(node_embeds, node_embeds.transpose(-1, -2))
        probs = torch.sigmoid(logits)
        num_nodes = probs.size(-1)
        mask = 1.0 - torch.eye(num_nodes, device=probs.device, dtype=probs.dtype).unsqueeze(0)
        return probs * mask

    def edge_feat_forward(self, node_embeds: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return torch.empty(
                (0, self.edge_mlp[-1].out_features),
                dtype=node_embeds.dtype,
                device=node_embeds.device,
            )

        src = edge_index[0]
        dst = edge_index[1]
        pair_embeds = torch.cat([node_embeds[src], node_embeds[dst]], dim=-1)
        return self.edge_mlp(pair_embeds)
