"""Gymnasium environment for scenario generation."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from scipy.signal import savgol_filter
from torch_geometric.data import Data

from knowledge_graph.acg_builder import AccidentCausationGraph, infer_acg, jaccard_similarity


def violation_penalty(
    prev_sg: Data,
    next_sg: Data,
    lane_bounds: tuple[float, float],
    max_lateral_per_step: float = 0.3,
    max_longitudinal_per_step: float = 3.0,
) -> int:
    """Return 1 when the candidate graph violates any motion constraint."""
    prev_x = prev_sg.x.detach().cpu().float().numpy()
    next_x = next_sg.x.detach().cpu().float().numpy()

    delta_x = next_x[:, 0] - prev_x[:, 0]
    delta_y = next_x[:, 1] - prev_x[:, 1]
    prev_vx = prev_x[:, 2]

    reversed_motion = (np.abs(delta_x) > 0.01) & (delta_x * prev_vx < 0.0)
    if np.any(reversed_motion):
        return 1

    if np.any((next_x[:, 1] < lane_bounds[0]) | (next_x[:, 1] > lane_bounds[1])):
        return 1

    if np.any(np.abs(delta_y) > max_lateral_per_step):
        return 1

    if np.any(np.abs(delta_x) > max_longitudinal_per_step):
        return 1

    return 0


class ScenarioGenEnv(gym.Env):
    """Scenario-graph generation environment with VGAE sampling."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        vgae_model,
        seed_sequences: Sequence[Sequence[Data]],
        acg_gt: AccidentCausationGraph,
        lane_bounds: tuple[float, float],
        config: dict,
    ) -> None:
        super().__init__()
        if not seed_sequences:
            raise ValueError("seed_sequences must not be empty")

        self.vgae_model = vgae_model.eval()
        self.seed_sequences = [list(seq) for seq in seed_sequences]
        self.acg_gt = acg_gt
        self.lane_bounds = lane_bounds
        self.config = config

        self.n_steps_input = int(config["rl"]["n_steps_input"])
        self.episode_length = int(config["rl"]["episode_length"])
        self.n_candidates = int(config["rl"]["n_candidates"])
        self.reward_a = float(config["rl"]["reward_a"])
        self.reward_b = float(config["rl"]["reward_b"])
        self.reward_c = float(config["rl"]["reward_c"])
        self.max_lateral_per_step = float(config["rl"]["max_lateral_per_step"])
        self.max_longitudinal_per_step = float(config["rl"]["max_longitudinal_per_step"])

        self.node_feat_dim = int(config["model"]["node_feat_dim"])
        self.num_vehicles = int(config["data"]["num_vehicles"])
        self.vehicle_ids = list(range(1, self.num_vehicles + 1))
        self.max_steps = self.episode_length - self.n_steps_input

        obs_dim = self.n_steps_input * self.num_vehicles * self.node_feat_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_candidates)

        self.current_sg_seq: list[Data] = []
        self.full_trajectory: list[np.ndarray] = []
        self.step_count = 0
        self.candidates: list[Data] = []
        self._np_random: np.random.Generator | None = None
        self._sample_calls = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed if seed is not None else self.np_random.integers(0, 2**31 - 1))

        seed_sequence = self.seed_sequences[int(self._np_random.integers(0, len(self.seed_sequences)))]
        if len(seed_sequence) < self.n_steps_input:
            raise ValueError("each seed sequence must have length >= n_steps_input")

        self.current_sg_seq = [graph.clone() for graph in seed_sequence[: self.n_steps_input]]
        self.full_trajectory = [graph.x.detach().cpu().float().numpy().copy() for graph in self.current_sg_seq]
        self.step_count = 0
        self._sample_calls = 0
        self.candidates = self._sample_candidates()
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.current_sg_seq:
            raise RuntimeError("reset() must be called before step()")
        if not self.candidates:
            self.candidates = self._sample_candidates()

        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self.candidates):
            raise IndexError(f"action {action_idx} out of range for {len(self.candidates)} candidates")

        next_sg = self.candidates[action_idx].clone()
        penalty = violation_penalty(
            self.current_sg_seq[-1],
            next_sg,
            self.lane_bounds,
            max_lateral_per_step=self.max_lateral_per_step,
            max_longitudinal_per_step=self.max_longitudinal_per_step,
        )
        rconstraint = -self.reward_c * float(penalty)

        self.full_trajectory.append(next_sg.x.detach().cpu().float().numpy().copy())
        self.current_sg_seq = self.current_sg_seq[1:] + [next_sg]
        rsmoothness = -self.reward_b * self._smoothness_rmse()

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False

        sj = 0.0
        rsimilarity = 0.0
        acg0 = None
        if terminated:
            trajectory = np.stack(self.full_trajectory, axis=0)
            acg0 = infer_acg(
                trajectory,
                self.vehicle_ids,
                delta_x=float(self.config["causal"]["delta_x"]),
                delta_y1=float(self.config["causal"]["delta_y1"]),
                delta_y2=float(self.config["causal"]["delta_y2"]),
                delta_collision=float(self.config["causal"]["delta_collision"]),
                min_interval=int(self.config["causal"]["min_interval_frames"]),
            )
            sj = float(jaccard_similarity(acg0, self.acg_gt))
            rsimilarity = self.reward_a * sj
        else:
            self.candidates = self._sample_candidates()

        reward = float(rsimilarity + rsmoothness + rconstraint)
        info = {
            "sj": float(sj),
            "reward_total": reward,
            "reward_similarity": float(rsimilarity),
            "reward_smoothness": float(rsmoothness),
            "reward_constraint": float(rconstraint),
        }
        if terminated:
            info["trajectory"] = np.stack(self.full_trajectory, axis=0)
            info["acg0"] = acg0
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        stacked = np.stack(
            [graph.x.detach().cpu().float().numpy() for graph in self.current_sg_seq],
            axis=0,
        )
        return np.nan_to_num(stacked.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _sample_candidates(self) -> list[Data]:
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        sample_seed = int(self._np_random.integers(0, 2**31 - 1))
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        self._sample_calls += 1
        samples = list(self.vgae_model.sample(self.current_sg_seq, k=self.n_candidates))
        if not samples:
            raise RuntimeError("VGAE sample returned no candidates")

        while len(samples) < self.n_candidates:
            samples.append(samples[-1].clone())
        return samples[: self.n_candidates]

    def _smoothness_rmse(self) -> float:
        trajectory = np.stack(self.full_trajectory, axis=0)
        if trajectory.shape[0] < 11:
            return 0.0

        x_positions = trajectory[:, :, 0]
        window = min(11, trajectory.shape[0] if trajectory.shape[0] % 2 == 1 else trajectory.shape[0] - 1)
        if window < 3:
            return 0.0

        smoothed = savgol_filter(x_positions, window_length=window, polyorder=min(3, window - 1), axis=0)
        rmse = np.sqrt(np.mean((x_positions - smoothed) ** 2))
        return float(rmse)
