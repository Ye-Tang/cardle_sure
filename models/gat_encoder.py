"""Graph attention encoder blocks for scenario graphs."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class DualGATEncoder(nn.Module):
    """Two-layer GATv2 encoder for single-frame scenario graphs."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")

        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.conv1 = GATv2Conv(
            in_channels=node_feat_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=edge_feat_dim,
            concat=True,
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            edge_dim=edge_feat_dim,
            concat=False,
        )
        self.activation = nn.ReLU()

    def forward(self, data: Data) -> Tensor:
        """Encode one frame into per-node hidden embeddings."""
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if data.edge_attr.numel() > 0 else data.edge_attr

        if edge_index.numel() == 0:
            # Keep empty-edge graphs numerically stable.
            return torch.zeros(
                (x.size(0), self.hidden_dim),
                dtype=x.dtype,
                device=x.device,
            )

        hidden = self.activation(self.conv1(x, edge_index, edge_attr))
        hidden = self.activation(self.conv2(hidden, edge_index, edge_attr))
        return hidden
