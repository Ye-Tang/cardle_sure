"""Manual validation script for Phase 5 Gymnasium environment."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from gymnasium.utils.env_checker import check_env
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.acg_builder import make_acg_type1
from models.sg_temporal_predictor import SGTemporalPredictor
from rl.environment import ScenarioGenEnv, violation_penalty


def make_mock_seq(length: int = 10) -> list[Data]:
    return [
        Data(
            x=torch.randn(4, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.randn(2, 5),
        )
        for _ in range(length)
    ]


def make_sg(x_pos: float, y_pos: float, vx: float = 25.0) -> Data:
    x = torch.tensor(
        [
            [x_pos, y_pos, vx, 0.0],
            [50.0, 0.0, 20.0, 0.0],
            [5.0, 3.5, 25.0, 0.0],
            [55.0, 3.5, 20.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return Data(
        x=x,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 5), dtype=torch.float32),
    )


if __name__ == "__main__":
    with (PROJECT_ROOT / "configs" / "config.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

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
    model.eval()

    seed_seqs = [make_mock_seq(50) for _ in range(20)]
    acg_gt = make_acg_type1()
    lane_bounds = (0.0, 10.5)

    env = ScenarioGenEnv(
        vgae_model=model,
        seed_sequences=seed_seqs,
        acg_gt=acg_gt,
        lane_bounds=lane_bounds,
        config=cfg,
    )

    assert env.observation_space.shape == (160,), f"obs 形状错误: {env.observation_space.shape}"
    assert env.action_space.n == 50, f"动作空间大小错误: {env.action_space.n}"
    print("[OK] observation_space 和 action_space")

    obs, info = env.reset()
    assert obs.shape == (160,), f"reset obs 形状错误: {obs.shape}"
    assert obs.dtype == np.float32, "obs 应为 float32"
    assert isinstance(info, dict), "info 应为 dict"
    print("[OK] reset()")

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (160,), f"step obs 形状错误: {obs.shape}"
        assert isinstance(reward, float), f"reward 应为 float: {type(reward)}"
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    print("[OK] step() 接口验证（5 步）")

    prev = make_sg(x_pos=0.0, y_pos=1.75, vx=25.0)
    normal = make_sg(x_pos=1.0, y_pos=1.75, vx=25.0)
    assert violation_penalty(prev, normal, lane_bounds) == 0
    print("[OK] 正常情况：无违规")

    reversed_dir = make_sg(x_pos=-1.0, y_pos=1.75, vx=25.0)
    assert violation_penalty(prev, reversed_dir, lane_bounds) == 1
    print("[OK] 违规1：运动方向相反")

    out_of_lane = make_sg(x_pos=1.0, y_pos=-0.5)
    assert violation_penalty(prev, out_of_lane, lane_bounds) == 1
    print("[OK] 违规2：驶出车道边界")

    large_lateral = make_sg(x_pos=1.0, y_pos=2.1)
    assert violation_penalty(prev, large_lateral, lane_bounds) == 1
    print("[OK] 违规3：横向位移超限")

    large_longitudinal = make_sg(x_pos=4.0, y_pos=1.75)
    assert violation_penalty(prev, large_longitudinal, lane_bounds) == 1
    print("[OK] 违规4：纵向位移超限")

    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        step += 1
    assert step == cfg["rl"]["episode_length"] - cfg["rl"]["n_steps_input"], f"episode 长度错误: {step}"
    assert "sj" in info, "info 应包含 sj"
    print(f"[OK] 完整 episode ({step} 步), total_reward={total_reward:.3f}, sj={info['sj']:.3f}")

    check_env(env, skip_render_check=True)
    print("[OK] gymnasium check_env 通过")
