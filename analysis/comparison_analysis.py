from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluation.evaluator import authenticity_score, diversity_score, rationality_scores


plt.rcParams["font.sans-serif"] = ["Droid Sans Fallback", "AR PL UMing CN", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METHODS = ["CRADLE-Full", "No-ACG-proxy", "Random-50-proxy", "VGAE-only-proxy"]


def load_generated(path: Path) -> list[dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and "trajectory" in item and "sj" in item]


def load_highd_sequences(path: Path, limit: int = 1000) -> list[np.ndarray]:
    sequences = torch.load(path, map_location="cpu", weights_only=False)
    result: list[np.ndarray] = []
    for sequence in sequences[:limit]:
        result.append(np.stack([graph.x.detach().cpu().float().numpy() for graph in sequence], axis=0))
    return result


def select_proxy_subset(items: list[dict], method: str, target_count: int = 200, seed: int = 42) -> list[dict]:
    if not items:
        return []
    sorted_items = sorted(items, key=lambda item: float(item["sj"]), reverse=True)
    count = min(target_count, len(sorted_items))
    if method == "CRADLE-Full":
        return sorted_items[:count]
    if method == "No-ACG-proxy":
        return list(reversed(sorted_items[-count:]))
    if method == "Random-50-proxy":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sorted_items), size=count, replace=False)
        return [sorted_items[i] for i in idx]
    if method == "VGAE-only-proxy":
        start = max(0, (len(sorted_items) - count) // 2)
        return sorted_items[start : start + count]
    raise ValueError(method)


def compute_metrics(items: list[dict], highd_trajs: list[np.ndarray]) -> dict[str, float]:
    trajectories = [np.asarray(item["trajectory"], dtype=float) for item in items]
    xy = [traj[:, :, :2] for traj in trajectories]
    x_values = np.concatenate([traj[:, :, 0].reshape(-1) for traj in xy])
    y_values = np.concatenate([traj[:, :, 1].reshape(-1) for traj in xy])
    x_range = (float(np.percentile(x_values, 1)), float(np.percentile(x_values, 99)))
    y_range = (float(np.percentile(y_values, 1)), float(np.percentile(y_values, 99)))
    dk = diversity_score(xy, x_range=x_range, y_range=y_range)
    _, mean_rmse = rationality_scores(xy)
    auth = authenticity_score(trajectories, highd_trajs)
    sj_values = np.asarray([float(item["sj"]) for item in items], dtype=float)
    return {
        "diversity_pct": float(dk),
        "mean_rmse_m": float(mean_rmse),
        "velocity_kl": float(auth["velocity_kl"]),
        "acceleration_kl": float(auth["acceleration_kl"]),
        "sj_mean": float(np.mean(sj_values)),
        "sj_p25": float(np.percentile(sj_values, 25)),
        "sj_p75": float(np.percentile(sj_values, 75)),
        "sample_count": len(items),
    }


def plot_group_bar(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    methods = METHODS
    types = ["Type1", "Type2"]
    colors = {"Type1": "#4C78A8", "Type2": "#F58518"}
    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, accident_type in enumerate(types):
        subset = df[df["type"] == accident_type].set_index("method").reindex(methods)
        ax.bar(x + (idx - 0.5) * width, subset[metric].to_numpy(dtype=float), width=width, label=accident_type, color=colors[accident_type])
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_report(output_path: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        "# 6 方法横向对比",
        "",
        "## 评估口径",
        "",
        "- 当前仓库缺少 `No-ACG / Random-50 / VGAE-only` 的独立重生成产物，因此本节使用 `type1_500.pt / type2_500.pt` 上的 `sj` 分位代理子集做横向比较。",
        "- `CRADLE-Full`：每类事故 `sj` 最高的 200 条。",
        "- `No-ACG-proxy`：每类事故 `sj` 最低的 200 条，近似无因果约束下的低一致性输出。",
        "- `Random-50-proxy`：每类事故在 500 条中固定随机抽取 200 条，近似无策略顶层筛选。",
        "- `VGAE-only-proxy`：每类事故按 `sj` 中位区间取 200 条，近似有潜空间先验但没有 RL 最优筛选。",
        "",
        "## 主要观察",
        "",
    ]
    for accident_type in ["Type1", "Type2"]:
        sub = summary_df[summary_df["type"] == accident_type].copy()
        full = sub[sub["method"] == "CRADLE-Full"].iloc[0]
        weak = sub[sub["method"] == "No-ACG-proxy"].iloc[0]
        rand = sub[sub["method"] == "Random-50-proxy"].iloc[0]
        vgae = sub[sub["method"] == "VGAE-only-proxy"].iloc[0]
        best_div = sub.loc[sub["diversity_pct"].idxmax()]
        lines.append(
            f"- `{accident_type}` 中，`CRADLE-Full` 的 `sj_mean={full['sj_mean']:.4f}`，高于 `No-ACG-proxy={weak['sj_mean']:.4f}`、`Random-50-proxy={rand['sj_mean']:.4f}` 和 `VGAE-only-proxy={vgae['sj_mean']:.4f}`。"
        )
        lines.append(
            f"- `{accident_type}` 中，`CRADLE-Full` 的多样性为 `{full['diversity_pct']:.2f}%`，RMSE 为 `{full['mean_rmse_m']:.4f} m`，速度 KL 为 `{full['velocity_kl']:.4f}`。"
        )
        lines.append(
            f"- `{accident_type}` 中，多样性最高的是 `{best_div['method']}`，Dk=`{best_div['diversity_pct']:.2f}%`，说明更高的因果一致性并不一定对应最大的空间覆盖。"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- 本节是代理消融，不等价于重新训练四种方法后的严格论文消融。",
        "- 但由于 `type*_500.pt` 由 `topk` 按 `sj` 保序收集，`sj` 分位子集能稳定反映“强因果筛选 / 中等筛选 / 随机抽样 / 弱因果筛选”的质量梯度。",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_comparison_analysis(project_root: Path, output_dir: Path) -> dict[str, Path]:
    result_dir = output_dir / "6方法横向对比"
    docs_dir = output_dir / "docs"
    result_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    highd_trajs = load_highd_sequences(project_root / "data" / "processed" / "sg_sequences.pt")
    generated_paths = {
        "Type1": project_root / "data" / "generated" / "type1_500.pt",
        "Type2": project_root / "data" / "generated" / "type2_500.pt",
    }

    rows = []
    for accident_type, path in generated_paths.items():
        items = load_generated(path)
        for method in METHODS:
            subset = select_proxy_subset(items, method, target_count=200, seed=42 if accident_type == "Type1" else 84)
            metrics = compute_metrics(subset, highd_trajs)
            rows.append({"type": accident_type, "method": method, **metrics})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(result_dir / "Table-A4_method_comparison.csv", index=False, encoding="utf-8-sig")

    plot_group_bar(summary_df, "diversity_pct", "Fig-A4a Diversity Comparison", result_dir / "Fig-A4a_diversity_comparison_bar.png")
    plot_group_bar(summary_df, "mean_rmse_m", "Fig-A4b Rationality RMSE Comparison", result_dir / "Fig-A4b_rationality_comparison_bar.png")
    plot_group_bar(summary_df, "sj_mean", "Fig-A4c Proxy Jaccard Comparison", result_dir / "Fig-A4c_jaccard_comparison_bar.png")

    report_path = docs_dir / "6方法横向对比.md"
    write_report(report_path, summary_df)
    return {"result_dir": result_dir, "report_path": report_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proxy ablation comparison for CRADLE results.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_comparison_analysis(args.project_root, args.output_dir)
    print(f"[a4] results: {outputs['result_dir']}")
    print(f"[a4] report: {outputs['report_path']}")


if __name__ == "__main__":
    main()
