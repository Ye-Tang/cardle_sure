"""Manual validation script for Phase 8 risk predictors."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from application.risk_predictor import GTFRiskPredictor, LSTMRiskPredictor


if __name__ == "__main__":
    lstm = LSTMRiskPredictor()
    batch_size = 4
    time_steps = 10
    x = torch.randn(batch_size, time_steps, 16)
    risk = lstm(x)
    assert risk.shape == (batch_size, 1), f"LSTM 输出形状错误: {risk.shape}"
    assert torch.all((risk >= 0) & (risk <= 1)), "风险值应在 [0,1]"
    print("[OK] LSTMRiskPredictor 前向传播")

    loss_fn = torch.nn.BCELoss()
    labels = torch.ones(batch_size, 1)
    loss = loss_fn(risk, labels)
    assert not torch.isnan(loss), "LSTM loss 不应为 NaN"
    loss.backward()
    print("[OK] LSTMRiskPredictor 损失计算和反向传播")

    model = LSTMRiskPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_train = torch.randn(20, 10, 16)
    y_train = torch.cat([torch.ones(10, 1), torch.zeros(10, 1)])
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        pred = model(x_train)
        train_loss = loss_fn(pred, y_train)
        train_loss.backward()
        optimizer.step()
        losses.append(float(train_loss.item()))
    assert losses[-1] < losses[0], f"loss 应下降: {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"[OK] LSTM 训练收敛: {losses[0]:.4f} → {losses[-1]:.4f}")

    gtf = GTFRiskPredictor(
        node_feat_dim=4,
        edge_feat_dim=5,
        gat_hidden=64,
        gat_heads=4,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_num_layers=3,
        freeze_encoder=False,
        pretrained_state_path=None,
    )
    sg_seq = [
        Data(
            x=torch.randn(4, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.randn(2, 5),
        )
        for _ in range(10)
    ]
    risk_gtf = gtf(sg_seq)
    assert risk_gtf.shape == (1,), f"GTF 输出形状错误: {risk_gtf.shape}"
    assert 0.0 <= risk_gtf.item() <= 1.0, "GTF 风险值应在 [0,1]"
    print("[OK] GTFRiskPredictor 前向传播")
