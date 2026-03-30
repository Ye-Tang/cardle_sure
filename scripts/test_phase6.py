"""Manual validation script for Phase 6 PPO setup."""

from __future__ import annotations

from pathlib import Path
import sys

import copy

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.acg_builder import make_acg_type1
from models.sg_temporal_predictor import SGTemporalPredictor
from rl.environment import ScenarioGenEnv
from rl.ppo_agent import make_ppo_model
from rl.train_ppo import load_config


def make_mock_sequence(length: int = 50) -> list[Data]:
    return [
        Data(
            x=torch.randn(4, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.randn(2, 5),
        )
        for _ in range(length)
    ]


if __name__ == "__main__":
    cfg = copy.deepcopy(load_config())
    cfg["rl"]["ppo_n_steps"] = 32
    cfg["rl"]["ppo_batch_size"] = 16
    cfg["rl"]["episode_length"] = 20

    model_vgae = SGTemporalPredictor(
        node_feat_dim=4,
        edge_feat_dim=5,
        gat_hidden=64,
        gat_heads=4,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_num_layers=3,
        latent_dim=64,
    )
    model_vgae.eval()

    seed_seqs = [make_mock_sequence(50) for _ in range(20)]
    env = ScenarioGenEnv(
        vgae_model=model_vgae,
        seed_sequences=seed_seqs,
        acg_gt=make_acg_type1(),
        lane_bounds=(0.0, 10.5),
        config=cfg,
    )

    ppo = make_ppo_model(env, cfg, device="cpu")
    assert ppo is not None
    print("[OK] make_ppo_model 构造成功")

    obs, _ = env.reset()
    action, _ = ppo.predict(obs, deterministic=False)
    action = int(action)
    assert 0 <= action < 50, f"动作应在 [0,50): {action}"
    print(f"[OK] ppo.predict 返回动作: {action}")

    ppo.learn(total_timesteps=100, progress_bar=False)
    print("[OK] 100步训练不崩溃")
