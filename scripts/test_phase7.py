"""Manual validation script for Phase 7 evaluator functions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import authenticity_score, diversity_score, rationality_scores


if __name__ == "__main__":
    time_steps, num_traj = 50, 10
    identical_trajs = [np.zeros((time_steps, 4, 2)) for _ in range(num_traj)]
    d = diversity_score(identical_trajs, x_range=(0, 200), y_range=(0, 10.5))
    assert d < 5.0, f"完全相同轨迹多样性应接近 0, 得到 {d}"
    print(f"[OK] 多样性（相同轨迹）= {d:.2f}%")

    diverse_trajs = []
    for i in range(200):
        traj = np.zeros((time_steps, 4, 2))
        traj[:, 0, 0] = i
        traj[:, 0, 1] = i % 10
        diverse_trajs.append(traj)
    d2 = diversity_score(diverse_trajs, x_range=(0, 200), y_range=(0, 10.5))
    assert d2 > 5.0, f"均匀分布轨迹多样性应较高, 得到 {d2}"
    print(f"[OK] 多样性（均匀分布）= {d2:.2f}%")

    smooth_trajs = [
        np.column_stack([np.linspace(0, 200, time_steps), np.ones(time_steps) * 1.75]).reshape(time_steps, 1, 2).repeat(4, axis=1)
        for _ in range(20)
    ]
    _, mean_rmse = rationality_scores(smooth_trajs)
    assert mean_rmse < 0.01, f"平滑轨迹 RMSE 应接近 0, 得到 {mean_rmse}"
    print(f"[OK] 合理性（平滑轨迹）mean_RMSE = {mean_rmse:.4f}m")

    zigzag = [
        np.column_stack([np.arange(time_steps), np.array([0.0 if i % 2 == 0 else 0.5 for i in range(time_steps)])]).reshape(time_steps, 1, 2).repeat(4, axis=1)
        for _ in range(20)
    ]
    _, mean_rmse2 = rationality_scores(zigzag)
    assert mean_rmse2 > 0.05, f"锯齿轨迹 RMSE 应较大, 得到 {mean_rmse2}"
    print(f"[OK] 合理性（锯齿轨迹）mean_RMSE = {mean_rmse2:.4f}m")

    np.random.seed(42)
    n = 100
    same_trajs_a = [
        np.random.randn(50, 4, 4) * np.array([1, 0, 25, 0]) + np.array([0, 0, 25, 0])
        for _ in range(n)
    ]
    same_trajs_b = [
        np.random.randn(50, 4, 4) * np.array([1, 0, 25, 0]) + np.array([0, 0, 25, 0])
        for _ in range(n)
    ]
    result = authenticity_score(same_trajs_a, same_trajs_b)
    assert result["velocity_kl"] < 0.5, f"相似分布 KL 应较低: {result['velocity_kl']}"
    print(f"[OK] 真实性（相似分布）velocity_kl = {result['velocity_kl']:.4f}")

    diff_trajs = [
        np.random.randn(50, 4, 4) * np.array([1, 0, 5, 0]) + np.array([0, 0, 5, 0])
        for _ in range(n)
    ]
    result2 = authenticity_score(diff_trajs, same_trajs_a)
    assert result2["velocity_kl"] > result["velocity_kl"], "不同分布 KL 应更高"
    print(f"[OK] 真实性（不同分布）velocity_kl = {result2['velocity_kl']:.4f}")
