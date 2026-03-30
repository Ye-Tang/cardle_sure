"""Manual validation script for Phase 3 model shapes and losses."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sg_temporal_predictor import SGTemporalPredictor


def make_mock_sg() -> Data:
    return Data(
        x=torch.randn(4, 4),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 5),
    )


if __name__ == "__main__":
    sg_seq = [make_mock_sg() for _ in range(10)]
    target = make_mock_sg()

    model = SGTemporalPredictor(
        node_feat_dim=4,
        edge_feat_dim=5,
        gat_hidden=64,
        gat_heads=4,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_num_layers=3,
        latent_dim=64,
    )

    mu, logvar, node_pred, edge_prob, edge_feat_pred, pred_ei = model(sg_seq)
    assert mu.shape == (64,), f"mu 形状错误: {mu.shape}"
    assert logvar.shape == (64,), f"logvar 形状错误: {logvar.shape}"
    assert node_pred.shape == (4, 4), f"node_pred 形状错误: {node_pred.shape}"
    assert edge_prob.shape == (4, 4), f"edge_prob 形状错误: {edge_prob.shape}"
    assert edge_prob.min() >= 0.0, "edge_prob 应 >= 0"
    assert edge_prob.max() <= 1.0, "edge_prob 应 <= 1"
    print("[OK] forward 形状验证")

    loss = model.compute_loss(
        (mu, logvar, node_pred, edge_prob, edge_feat_pred, pred_ei),
        target,
        alpha=1.0,
        beta=1.0,
        gamma=0.1,
    )
    assert loss.item() > 0, "loss 应 > 0"
    assert not torch.isnan(loss), "loss 不应为 NaN"
    loss.backward()
    print("[OK] compute_loss 和反向传播")

    samples = model.sample(sg_seq, k=50)
    assert len(samples) == 50, f"sample 应返回 50 个 Data，得到 {len(samples)}"
    for sample in samples:
        assert sample.x.shape == (4, 4), "sample 节点特征形状错误"
        assert sample.edge_attr.shape[1] == 5, "sample 边特征维度错误"
    print("[OK] sample(k=50) 形状验证")
