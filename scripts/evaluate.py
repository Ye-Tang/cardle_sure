"""Evaluation entrypoint for generated scenario databases."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import authenticity_score, diversity_score, rationality_scores


def is_valid_generated_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        data = torch.load(path, weights_only=False)
    except Exception:
        return False
    if not isinstance(data, list) or not data:
        return False
    return any(isinstance(item, dict) and "trajectory" in item for item in data)


def resolve_generated_path(acg_type: int) -> Path | None:
    candidates = sorted(
        (PROJECT_ROOT / "data" / "generated").glob(f"type{acg_type}_*.pt"),
        key=lambda path: int(path.stem.rpartition("_")[2]) if path.stem.rpartition("_")[2].isdigit() else -1,
        reverse=True,
    )
    for path in candidates:
        if is_valid_generated_file(path):
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=int, choices=[1, 2], default=None)
    return parser.parse_args()


def load_generated(acg_type: int) -> list[dict]:
    path = resolve_generated_path(acg_type)
    if path is None:
        print(f"Warning: generated file missing for type {acg_type} under data/generated/")
        return []
    data = torch.load(path, weights_only=False)
    if not isinstance(data, list):
        print(f"Warning: generated file has unexpected format for type {acg_type}")
        return []
    return data


def load_highd_sequences(limit: int = 1000) -> list[np.ndarray]:
    path = PROJECT_ROOT / "data" / "processed" / "sg_sequences.pt"
    sequences = torch.load(path, weights_only=False)
    result: list[np.ndarray] = []
    for sequence in sequences[:limit]:
        result.append(np.stack([graph.x.detach().cpu().float().numpy() for graph in sequence], axis=0))
    return result


def save_diversity_curve(acg_type: int, xy_trajectories: list[np.ndarray], output_path: Path) -> None:
    if not xy_trajectories:
        return
    x_values = np.concatenate([traj[:, :, 0].reshape(-1) for traj in xy_trajectories])
    y_values = np.concatenate([traj[:, :, 1].reshape(-1) for traj in xy_trajectories])
    x_range = (float(np.min(x_values)), float(np.max(x_values)))
    y_range = (float(np.min(y_values)), float(np.max(y_values)))
    ns = list(range(100, len(xy_trajectories) + 1, 100))
    if len(xy_trajectories) not in ns:
        ns.append(len(xy_trajectories))
    dks = [diversity_score(xy_trajectories[:n], x_range=x_range, y_range=y_range) for n in ns]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, dks, marker="o")
    ax.set_xlabel("Number of Scenarios")
    ax.set_ylabel("Diversity Index Dk (%)")
    ax.set_title(f"Diversity vs Number of Scenarios (Type {acg_type})")
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_collision_density(acg_type: int, trajectories: list[np.ndarray], output_path: Path) -> None:
    collision_points: list[np.ndarray] = []
    for traj in trajectories:
        if traj.shape[0] == 0:
            continue
        positions = traj[:, :, :2]
        pairwise_points: list[tuple[float, float, float]] = []
        for t in range(positions.shape[0]):
            min_dist = float("inf")
            collision_point = None
            for i in range(positions.shape[1]):
                for j in range(i + 1, positions.shape[1]):
                    dist = np.linalg.norm(positions[t, i] - positions[t, j])
                    if dist < min_dist:
                        min_dist = dist
                        collision_point = (positions[t, i] + positions[t, j]) / 2.0
            if collision_point is not None:
                pairwise_points.append((min_dist, collision_point[0], collision_point[1]))
        if pairwise_points:
            _, x, y = min(pairwise_points, key=lambda item: item[0])
            collision_points.append(np.array([x, y], dtype=float))

    if not collision_points:
        return

    points = np.stack(collision_points, axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(points[:, 0], points[:, 1], s=12, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Collision Density (Type {acg_type})")
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_type(acg_type: int) -> None:
    generated_items = load_generated(acg_type)
    if not generated_items:
        print(f"=== 评估结果：事故类型 {acg_type} ===")
        print("生成数据为空，跳过完整评估。")
        return

    trajectories = [np.asarray(item["trajectory"], dtype=float) for item in generated_items]
    xy_trajectories = [traj[:, :, :2] for traj in trajectories]
    x_values = np.concatenate([traj[:, :, 0].reshape(-1) for traj in xy_trajectories])
    y_values = np.concatenate([traj[:, :, 1].reshape(-1) for traj in xy_trajectories])
    x_range = (float(np.percentile(x_values, 1)), float(np.percentile(x_values, 99)))
    y_range = (float(np.min(y_values)), float(np.max(y_values)))

    dk = diversity_score(xy_trajectories, x_range=x_range, y_range=y_range)
    _, mean_rmse = rationality_scores(xy_trajectories)
    highd_trajs = load_highd_sequences()
    auth = authenticity_score(trajectories, highd_trajs)

    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    save_diversity_curve(acg_type, xy_trajectories, checkpoints_dir / f"fig13_diversity_type{acg_type}.png")
    save_collision_density(acg_type, trajectories, checkpoints_dir / f"fig17_collision_density_type{acg_type}.png")

    print(f"=== 评估结果：事故类型 {acg_type} ===")
    print(f"多样性 (Dk):         {dk:.3f}%")
    print(f"平均合理性 RMSE:      {mean_rmse:.3f}m")
    print(f"速度 KL 散度:         {auth['velocity_kl']:.3f}")
    print(f"加速度 KL 散度:       {auth['acceleration_kl']:.3f}")


if __name__ == "__main__":
    args = parse_args()
    types = [args.type] if args.type else [1, 2]
    for acg_type in types:
        evaluate_type(acg_type)
