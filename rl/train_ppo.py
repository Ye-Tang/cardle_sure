"""PPO training and scenario collection utilities."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import SGSequenceDataset
from knowledge_graph.acg_builder import make_acg_type1, make_acg_type2
from models.sg_temporal_predictor import SGTemporalPredictor
from rl.environment import ScenarioGenEnv
from rl.ppo_agent import make_ppo_model


class RewardLoggingCallback(BaseCallback):
    """Record per-episode reward components from env infos."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.rewards_log: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.rewards_log.append(
                {
                    "total": float(info.get("reward_total", 0.0)),
                    "similarity": float(info.get("reward_similarity", 0.0)),
                    "smoothness": float(info.get("reward_smoothness", 0.0)),
                    "constraint": float(info.get("reward_constraint", 0.0)),
                    "sj": float(info.get("sj", 0.0)),
                }
            )
        return True


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else PROJECT_ROOT / "configs" / "config.yaml"
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_seed_sequences(cfg: dict) -> list[list]:
    dataset = SGSequenceDataset(
        sequences_path=str(PROJECT_ROOT / cfg["data"]["processed_path"]),
        n_input=int(cfg["rl"]["n_steps_input"]),
    )
    return dataset.sequences


def infer_lane_bounds(seed_sequences: list[list]) -> tuple[float, float]:
    y_min = math.inf
    y_max = -math.inf
    for sequence in seed_sequences:
        for graph in sequence:
            y_values = graph.x[:, 1].detach().cpu().float()
            y_min = min(y_min, float(y_values.min().item()))
            y_max = max(y_max, float(y_values.max().item()))
    if not math.isfinite(y_min) or not math.isfinite(y_max):
        raise RuntimeError("failed to infer lane bounds from seed sequences")
    margin = 0.5
    return (y_min - margin, y_max + margin)


def load_vgae_model(cfg: dict, checkpoint_path: str | Path | None = None, device: str = "cpu") -> SGTemporalPredictor:
    model = SGTemporalPredictor(
        node_feat_dim=int(cfg["model"]["node_feat_dim"]),
        edge_feat_dim=int(cfg["model"]["edge_feat_dim"]),
        gat_hidden=int(cfg["model"]["gat_hidden"]),
        gat_heads=int(cfg["model"]["gat_heads"]),
        transformer_d_model=int(cfg["model"]["transformer_d_model"]),
        transformer_nhead=int(cfg["model"]["transformer_nhead"]),
        transformer_num_layers=int(cfg["model"]["transformer_num_layers"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        num_vehicles=int(cfg["data"]["num_vehicles"]),
    )
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else PROJECT_ROOT / "checkpoints" / "vgae_best.pt"
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def make_target_acg(acg_type: int):
    if acg_type == 1:
        return make_acg_type1()
    if acg_type == 2:
        return make_acg_type2()
    raise ValueError(f"unsupported acg_type: {acg_type}")


def make_env(cfg: dict, acg_type: int, device: str = "cpu") -> ScenarioGenEnv:
    seed_sequences = load_seed_sequences(cfg)
    lane_bounds = infer_lane_bounds(seed_sequences)
    vgae_model = load_vgae_model(cfg, device=device)
    return ScenarioGenEnv(
        vgae_model=vgae_model,
        seed_sequences=seed_sequences,
        acg_gt=make_target_acg(acg_type),
        lane_bounds=lane_bounds,
        config=cfg,
    )


def load_ppo_model(checkpoint_path: str | Path, env: ScenarioGenEnv, device: str = "cpu") -> PPO:
    """Load an existing PPO checkpoint for scenario collection."""
    return PPO.load(str(checkpoint_path), env=env, device=device)


def plot_reward_curve(rewards_log: list[dict[str, float]], output_path: Path) -> None:
    if not rewards_log:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.flat, ["total", "similarity", "smoothness", "constraint"]):
        values = [item[key] for item in rewards_log]
        ax.plot(values)
        ax.set_title(key)
        ax.set_xlabel("Episode")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def generated_output_path(acg_type: int, target_count: int) -> Path:
    """Return the generated-scenarios output path for the requested target count."""
    return PROJECT_ROOT / "data" / "generated" / f"type{acg_type}_{target_count}.pt"


def resolve_resume_path(output_path: Path) -> Path | None:
    """Find the most suitable existing scenario file to resume from."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    stem = output_path.stem
    if "_" not in stem:
        return None

    prefix, _, suffix = stem.rpartition("_")
    try:
        target_count = int(suffix)
    except ValueError:
        return None

    candidates = []
    for candidate in output_path.parent.glob(f"{prefix}_*.pt"):
        if not candidate.exists() or candidate.stat().st_size == 0:
            continue
        try:
            count = int(candidate.stem.rpartition("_")[2])
        except ValueError:
            continue
        if count <= target_count:
            candidates.append((count, candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_match = candidates[0][1]
    return best_match


def load_existing_scenarios(output_path: Path, target_count: int, strategy: str) -> list[dict]:
    """Load previously collected scenarios so collection can resume safely."""
    resume_path = resolve_resume_path(output_path)
    if resume_path is None:
        return []

    try:
        loaded = torch.load(resume_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive resume path
        print(f"Warning: failed to load existing scenarios from {resume_path}: {exc}")
        return []

    if not isinstance(loaded, list):
        print(f"Warning: unexpected scenario file format in {resume_path}; ignoring existing file.")
        return []

    scenarios = [item for item in loaded if isinstance(item, dict) and "sj" in item]
    if len(scenarios) != len(loaded):
        print(f"Warning: dropped {len(loaded) - len(scenarios)} malformed scenarios from {output_path}.")

    if strategy == "topk":
        scenarios.sort(key=lambda item: float(item["sj"]), reverse=True)
        scenarios = scenarios[:target_count]

    if scenarios:
        best_sj = max(float(item["sj"]) for item in scenarios)
        worst_sj = min(float(item["sj"]) for item in scenarios)
        print(
            f"[collect] resumed {len(scenarios)} existing scenarios from {resume_path} "
            f"(best_sj={best_sj:.3f}, worst_kept_sj={worst_sj:.3f})"
        )

    return scenarios


def collect_scenarios(
    model,
    env: ScenarioGenEnv,
    output_path: Path,
    target_count: int = 1000,
    max_episodes: int | None = None,
    min_sj: float = 0.75,
    progress_every: int = 10,
    save_every: int = 25,
    strategy: str = "threshold",
) -> list[dict]:
    scenarios = load_existing_scenarios(output_path, target_count=target_count, strategy=strategy)
    initial_count = len(scenarios)
    episodes = 0

    if strategy == "threshold" and len(scenarios) >= target_count:
        print(f"[collect] existing scenario file already satisfies target_count={target_count}.")
        return scenarios

    while True:
        obs, _ = env.reset()
        terminated = False
        truncated = False
        last_info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, last_info = env.step(action)

        episodes += 1
        sj = float(last_info.get("sj", 0.0))
        candidate = {
            "trajectory": last_info["trajectory"],
            "acg0": last_info["acg0"],
            "sj": sj,
        }

        if strategy == "threshold":
            if sj >= min_sj:
                scenarios.append(candidate)
                if len(scenarios) % max(1, save_every) == 0:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(scenarios, output_path)
        elif strategy == "topk":
            scenarios.append(candidate)
            scenarios.sort(key=lambda item: float(item["sj"]), reverse=True)
            if len(scenarios) > target_count:
                scenarios = scenarios[:target_count]
            if episodes % max(1, save_every) == 0:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(scenarios, output_path)
        else:
            raise ValueError(f"unsupported collection strategy: {strategy}")

        if episodes % max(1, progress_every) == 0 or (
            strategy == "threshold" and len(scenarios) == target_count
        ):
            accepted = len(scenarios) if strategy == "threshold" else min(len(scenarios), target_count)
            newly_collected = max(0, accepted - initial_count)
            hit_rate = newly_collected / max(1, episodes)
            best_sj = max((float(item["sj"]) for item in scenarios), default=float("nan"))
            worst_sj = min((float(item["sj"]) for item in scenarios), default=float("nan"))
            print(
                f"[collect] strategy={strategy} episodes={episodes} "
                f"accepted={accepted}/{target_count} (+{newly_collected}) last_sj={sj:.3f} "
                f"threshold={min_sj:.3f} hit_rate={hit_rate:.3%} "
                f"best_sj={best_sj:.3f} worst_kept_sj={worst_sj:.3f}"
            )

        if max_episodes is not None and episodes >= max_episodes:
            if strategy == "threshold" and len(scenarios) < target_count:
                print(
                    f"Warning: collected {len(scenarios)} / {target_count} scenarios "
                    f"after {episodes} episodes; saving partial results."
                )
            else:
                print(
                    f"[collect] reached max_episodes={max_episodes}; saving top {min(len(scenarios), target_count)} scenarios."
                )
            break

        if strategy == "threshold" and len(scenarios) >= target_count:
            break

        if strategy == "topk" and max_episodes is None and episodes >= target_count:
            print(
                f"[collect] no max_episodes provided; gathered initial top-{target_count} "
                f"scenes over {episodes} episodes."
            )
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scenarios, output_path)
    return scenarios


def train_and_collect(
    acg_type: int,
    cfg: dict,
    total_episodes: int | None = None,
    target_count: int = 1000,
    max_collect_episodes: int | None = None,
    device: str = "cpu",
    min_sj: float = 0.75,
    progress_every: int = 10,
    save_every: int = 25,
    strategy: str = "threshold",
) -> dict:
    env = make_env(cfg, acg_type=acg_type, device=device)
    model = make_ppo_model(env, cfg, device=device)
    callback = RewardLoggingCallback()

    steps_per_episode = int(cfg["rl"]["episode_length"]) - int(cfg["rl"]["n_steps_input"])
    train_episodes = int(total_episodes if total_episodes is not None else cfg["rl"]["total_episodes"])
    total_timesteps = train_episodes * steps_per_episode

    print(f"Training PPO for type {acg_type}: total_timesteps={total_timesteps}")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoints_dir / f"ppo_type{acg_type}.zip"
    model.save(model_path)

    reward_curve_path = checkpoints_dir / f"fig9_reward_curve_type{acg_type}.png"
    plot_reward_curve(callback.rewards_log, reward_curve_path)

    generated_path = generated_output_path(acg_type, target_count)
    scenarios = collect_scenarios(
        model=model,
        env=env,
        output_path=generated_path,
        target_count=target_count,
        max_episodes=max_collect_episodes,
        min_sj=min_sj,
        progress_every=progress_every,
        save_every=save_every,
        strategy=strategy,
    )

    print(f"Saved PPO checkpoint to {model_path}")
    print(f"Saved reward curve to {reward_curve_path}")
    print(f"Saved {len(scenarios)} scenarios to {generated_path}")
    return {
        "model_path": model_path,
        "reward_curve_path": reward_curve_path,
        "generated_path": generated_path,
        "rewards_log": callback.rewards_log,
        "num_scenarios": len(scenarios),
    }


def collect_with_checkpoint(
    acg_type: int,
    cfg: dict,
    target_count: int = 1000,
    max_collect_episodes: int | None = None,
    device: str = "cpu",
    min_sj: float = 0.75,
    progress_every: int = 10,
    save_every: int = 25,
    strategy: str = "threshold",
) -> dict:
    """Skip PPO training and only collect scenarios from an existing checkpoint."""
    env = make_env(cfg, acg_type=acg_type, device=device)
    model_path = PROJECT_ROOT / "checkpoints" / f"ppo_type{acg_type}.zip"
    model = load_ppo_model(model_path, env=env, device=device)
    generated_path = generated_output_path(acg_type, target_count)
    print(
        f"Collecting scenarios for type {acg_type} from {model_path} "
        f"with strategy={strategy} min_sj={min_sj:.3f}"
    )
    scenarios = collect_scenarios(
        model=model,
        env=env,
        output_path=generated_path,
        target_count=target_count,
        max_episodes=max_collect_episodes,
        min_sj=min_sj,
        progress_every=progress_every,
        save_every=save_every,
        strategy=strategy,
    )
    print(f"Saved {len(scenarios)} scenarios to {generated_path}")
    return {
        "model_path": model_path,
        "generated_path": generated_path,
        "num_scenarios": len(scenarios),
    }
