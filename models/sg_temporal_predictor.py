"""Full SG temporal predictor based on GAT + Transformer + VAE."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data

from models.gat_encoder import DualGATEncoder
from models.transformer_module import SGTransformerDecoder, SGTransformerEncoder
from models.vgae import EdgeDecoder, VAESampleLayer


class SGTemporalPredictor(nn.Module):
    """Temporal scenario-graph predictor with stochastic latent sampling."""

    def __init__(
        self,
        node_feat_dim: int = 4,
        edge_feat_dim: int = 5,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        transformer_d_model: int = 128,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 3,
        latent_dim: int = 64,
        num_vehicles: int = 4,
    ) -> None:
        super().__init__()
        self.num_vehicles = num_vehicles
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.gat_hidden = gat_hidden
        self.latent_dim = latent_dim

        self.register_buffer("node_mean", torch.zeros(node_feat_dim, dtype=torch.float32))
        self.register_buffer("node_std", torch.ones(node_feat_dim, dtype=torch.float32))
        self.register_buffer("edge_mean", torch.zeros(edge_feat_dim, dtype=torch.float32))
        self.register_buffer("edge_std", torch.ones(edge_feat_dim, dtype=torch.float32))

        self.gat = DualGATEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=gat_hidden,
            heads=gat_heads,
        )
        self.transformer_enc = SGTransformerEncoder(
            input_dim=gat_hidden * num_vehicles,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
        )
        self.vae = VAESampleLayer(
            d_model=transformer_d_model,
            latent_dim=latent_dim,
        )
        self.transformer_dec = SGTransformerDecoder(
            latent_dim=latent_dim,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            num_vehicles=num_vehicles,
            node_feat_dim=node_feat_dim,
            node_hidden_dim=gat_hidden,
        )
        self.edge_decoder = EdgeDecoder(node_dim=gat_hidden, edge_feat_dim=edge_feat_dim)

    def set_normalization_stats(
        self,
        node_mean: Tensor,
        node_std: Tensor,
        edge_mean: Tensor,
        edge_std: Tensor,
    ) -> None:
        self.node_mean.copy_(node_mean.detach().float().to(self.node_mean.device))
        self.node_std.copy_(node_std.detach().float().clamp_min(1e-6).to(self.node_std.device))
        self.edge_mean.copy_(edge_mean.detach().float().to(self.edge_mean.device))
        self.edge_std.copy_(edge_std.detach().float().clamp_min(1e-6).to(self.edge_std.device))

    def _normalize_nodes(self, x: Tensor) -> Tensor:
        return (x.float() - self.node_mean) / self.node_std

    def _denormalize_nodes(self, x: Tensor) -> Tensor:
        return x * self.node_std + self.node_mean

    def _normalize_edges(self, edge_attr: Tensor) -> Tensor:
        if edge_attr.numel() == 0:
            return edge_attr.float()
        return (edge_attr.float() - self.edge_mean) / self.edge_std

    def _denormalize_edges(self, edge_attr: Tensor) -> Tensor:
        if edge_attr.numel() == 0:
            return edge_attr.float()
        return edge_attr * self.edge_std + self.edge_mean

    def _normalize_graph(self, graph: Data) -> Data:
        normalized = graph.clone()
        normalized.x = self._normalize_nodes(graph.x)
        normalized.edge_attr = self._normalize_edges(graph.edge_attr)
        return normalized

    def encode_sequence(self, sg_seq: list[Data]) -> tuple[Tensor, Tensor, Tensor]:
        normalized_seq = [self._normalize_graph(graph) for graph in sg_seq]
        node_embeds = [self.gat(graph) for graph in normalized_seq]
        node_embed_seq = torch.stack(node_embeds, dim=0)
        h_seq = self.transformer_enc(node_embed_seq)
        h = h_seq.mean(dim=0)
        return node_embed_seq, h_seq, h

    def forward(self, sg_seq: list[Data]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        node_embed_seq, _, h = self.encode_sequence(sg_seq)
        mu, logvar, z = self.vae(h)
        normalized_last_graph = self._normalize_graph(sg_seq[-1])
        node_delta, node_hidden_delta = self.transformer_dec(z, seq_len=len(sg_seq))
        node_pred = node_delta + normalized_last_graph.x
        node_hidden_pred = node_hidden_delta + node_embed_seq[-1]
        edge_prob = self.edge_decoder.edge_prob_forward(node_hidden_pred)
        pred_edge_index = self._edge_index_from_prob(edge_prob)
        edge_feat_pred = self.edge_decoder.edge_feat_forward(node_hidden_pred, pred_edge_index)
        return mu, logvar, node_pred, edge_prob, edge_feat_pred, pred_edge_index

    def _edge_index_from_prob(self, edge_prob: Tensor, threshold: float = 0.5) -> Tensor:
        mask = edge_prob > threshold
        mask.fill_diagonal_(False)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        if edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=edge_prob.device)
        return edge_index

    def _target_adjacency(self, sg_target: Data) -> Tensor:
        adj = torch.zeros(
            (self.num_vehicles, self.num_vehicles),
            dtype=torch.float32,
            device=sg_target.x.device,
        )
        if sg_target.edge_index.numel() > 0:
            adj[sg_target.edge_index[0], sg_target.edge_index[1]] = 1.0
        adj.fill_diagonal_(0.0)
        return adj

    def _edge_feature_loss(
        self,
        edge_feat_pred: Tensor,
        pred_edge_index: Tensor,
        sg_target: Data,
    ) -> Tensor:
        if pred_edge_index.numel() == 0 or sg_target.edge_index.numel() == 0:
            return edge_feat_pred.new_tensor(0.0)

        target_map = {
            (int(src), int(dst)): idx
            for idx, (src, dst) in enumerate(sg_target.edge_index.t().tolist())
        }
        pred_features: list[Tensor] = []
        tgt_features: list[Tensor] = []
        for pred_idx, (src, dst) in enumerate(pred_edge_index.t().tolist()):
            key = (int(src), int(dst))
            if key not in target_map:
                continue
            pred_features.append(edge_feat_pred[pred_idx])
            tgt_features.append(sg_target.edge_attr[target_map[key]])

        if not pred_features:
            return edge_feat_pred.new_tensor(0.0)

        pred_tensor = torch.stack(pred_features, dim=0)
        tgt_tensor = torch.stack(tgt_features, dim=0).to(pred_tensor.device).float()
        return F.mse_loss(pred_tensor, tgt_tensor)

    def compute_loss(
        self,
        outputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        sg_target: Data,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
    ) -> Tensor:
        mu, logvar, node_pred, edge_prob, edge_feat_pred, pred_edge_index = outputs
        target_adj = self._target_adjacency(sg_target)
        target_graph = self._normalize_graph(sg_target)

        node_loss = F.mse_loss(node_pred, target_graph.x.float())
        edge_loss = F.binary_cross_entropy(
            edge_prob.clamp(1e-7, 1 - 1e-7).flatten(),
            target_adj.flatten(),
        )
        edge_feat_loss = self._edge_feature_loss(edge_feat_pred, pred_edge_index, target_graph)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return node_loss + alpha * edge_loss + beta * edge_feat_loss + gamma * kl_loss

    @torch.no_grad()
    def sample(self, sg_seq: list[Data], k: int = 50) -> list[Data]:
        node_embed_seq, _, h = self.encode_sequence(sg_seq)
        normalized_last_graph = self._normalize_graph(sg_seq[-1])
        mu, logvar, _ = self.vae(h)
        std = torch.exp(0.5 * logvar)
        z_batch = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            k,
            std.numel(),
            device=std.device,
            dtype=std.dtype,
        )
        node_delta_batch, node_hidden_delta_batch = self.transformer_dec(z_batch, seq_len=len(sg_seq))
        node_pred_batch = node_delta_batch + normalized_last_graph.x.unsqueeze(0)
        node_hidden_pred_batch = node_hidden_delta_batch + node_embed_seq[-1].unsqueeze(0)
        edge_prob_batch = self.edge_decoder.edge_prob_forward(node_hidden_pred_batch)

        samples: list[Data] = []
        for sample_idx in range(k):
            edge_prob = edge_prob_batch[sample_idx]
            pred_edge_index = self._edge_index_from_prob(edge_prob)
            edge_feat_pred = self.edge_decoder.edge_feat_forward(
                node_hidden_pred_batch[sample_idx],
                pred_edge_index,
            )
            samples.append(
                Data(
                    x=self._denormalize_nodes(node_pred_batch[sample_idx]).detach().cpu(),
                    edge_index=pred_edge_index.detach().cpu(),
                    edge_attr=self._denormalize_edges(edge_feat_pred).detach().cpu(),
                )
            )
        return samples
