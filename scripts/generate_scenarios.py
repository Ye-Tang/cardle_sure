"""CLI for PPO training and scenario generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.train_ppo import load_config, train_and_collect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=int, choices=[1, 2], default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--max-collect-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
    types = [args.type] if args.type else [1, 2]

    for acg_type in types:
        print(f"\n=== 开始训练事故类型 {acg_type} ===")
        train_and_collect(
            acg_type=acg_type,
            cfg=cfg,
            total_episodes=args.episodes,
            target_count=args.target_count,
            max_collect_episodes=args.max_collect_episodes,
            device=args.device,
        )
        print(f"=== 类型 {acg_type} 完成 ===")
