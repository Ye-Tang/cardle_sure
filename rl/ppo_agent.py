"""Stable-Baselines3 PPO factory."""

from __future__ import annotations

import torch.nn as nn
from stable_baselines3 import PPO


def make_ppo_model(env, cfg: dict, device: str = "cpu") -> PPO:
    """Build a PPO model from config."""
    rl_cfg = cfg["rl"]
    policy_kwargs = {
        "net_arch": [256, 256],
        "activation_fn": nn.ReLU,
    }
    return PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=int(rl_cfg["ppo_n_steps"]),
        batch_size=int(rl_cfg["ppo_batch_size"]),
        clip_range=float(rl_cfg["ppo_clip_range"]),
        ent_coef=float(rl_cfg["ppo_ent_coef"]),
        vf_coef=0.5,
        learning_rate=3e-4,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
    )
