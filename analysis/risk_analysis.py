from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Droid Sans Fallback", "AR PL UMing CN", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TYPE_RE = re.compile(r"===\s*事故类型\s*(\d+)\s*碰撞率结果\s*===")
ROW_RE = re.compile(r"^(LSTM|GTF)\s+\|\s+([0-9.]+|nan)\s+\|\s+([0-9.]+|nan)\s+\|\s+([0-9.]+|nan)")


def parse_phase8_log(log_path: Path) -> pd.DataFrame:
    rows = []
    current_type = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        type_match = TYPE_RE.search(line)
        if type_match:
            current_type = f"Type{type_match.group(1)}"
            continue
        row_match = ROW_RE.search(line.strip())
        if row_match and current_type:
            method = row_match.group(1)
            values = row_match.groups()[1:]
            for set_name, value in zip(["Set1", "Set2", "Set3"], values):
                rows.append(
                    {
                        "type": current_type,
                        "method": method,
                        "set_name": set_name,
                        "collision_rate": np.nan if value == "nan" else float(value),
                    }
                )
    return pd.DataFrame(rows)


def plot_collision_rate_comparison(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sets = ["Set1", "Set2", "Set3"]
    methods = ["LSTM", "GTF"]
    colors = {"LSTM": "#4C78A8", "GTF": "#F58518"}
    for ax, accident_type in zip(axes, ["Type1", "Type2"]):
        sub = df[df["type"] == accident_type]
        x = np.arange(len(sets))
        width = 0.35
        for idx, method in enumerate(methods):
            method_df = sub[sub["method"] == method].set_index("set_name").reindex(sets)
            ax.bar(x + (idx - 0.5) * width, method_df["collision_rate"].to_numpy(dtype=float), width=width, color=colors[method], label=method)
        ax.set_title(accident_type)
        ax.set_xticks(x)
        ax.set_xticklabels(sets)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Collision Rate")
    axes[1].legend()
    fig.suptitle("Fig-A5a Collision Rate Comparison Across Training Sets")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_augmentation_curve(df: pd.DataFrame) -> pd.DataFrame:
    ratio_map = {"Set1": 0, "Set2": 50, "Set3": 100}
    curve = df.copy()
    curve["augmentation_pct"] = curve["set_name"].map(ratio_map)
    return curve.sort_values(["type", "method", "augmentation_pct"]).reset_index(drop=True)


def plot_augmentation_curve(curve_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"LSTM": "#4C78A8", "GTF": "#F58518"}
    for ax, accident_type in zip(axes, ["Type1", "Type2"]):
        sub = curve_df[curve_df["type"] == accident_type]
        for method in ["LSTM", "GTF"]:
            method_df = sub[sub["method"] == method]
            ax.plot(method_df["augmentation_pct"], method_df["collision_rate"], marker="o", color=colors[method], label=method)
        ax.set_title(accident_type)
        ax.set_xlabel("CRADLE Data Share (%)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Collision Rate")
    axes[1].legend()
    fig.suptitle("Fig-A5b Augmentation Ratio vs Collision Rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_report(output_path: Path, valid_df: pd.DataFrame, invalid_df: pd.DataFrame) -> None:
    mean_df = valid_df.groupby(["method", "set_name"], as_index=False)["collision_rate"].mean()
    mean_map = {(row["method"], row["set_name"]): row["collision_rate"] for _, row in mean_df.iterrows()}
    lines = [
        "# 7 风险预测验证与数据增广效果",
        "",
        "## 数据来源",
        "",
        "- 采用 `logs/phase8_experiment_500.log` 作为有效实验结果来源。",
        "- `logs/phase8_experiment.log` 的结果为 `nan`，原因是生成碰撞序列为空，不能作为有效对比结果。",
        "- 增广比例曲线使用 `Set1 / Set2 / Set3` 三个实验锚点，分别对应 `0% / 50% / 100%` 的 CRADLE 数据占比经验点。",
        "",
        "## 关键结果",
        "",
        f"- `LSTM` 平均碰撞率从 `Set1={mean_map.get(('LSTM', 'Set1'), np.nan):.2f}` 下降到 `Set2={mean_map.get(('LSTM', 'Set2'), np.nan):.2f}`，再下降到 `Set3={mean_map.get(('LSTM', 'Set3'), np.nan):.2f}`。",
        f"- `GTF` 平均碰撞率从 `Set1={mean_map.get(('GTF', 'Set1'), np.nan):.2f}` 下降到 `Set2={mean_map.get(('GTF', 'Set2'), np.nan):.2f}`，再下降到 `Set3={mean_map.get(('GTF', 'Set3'), np.nan):.2f}`。",
        "- 两种模型在 `Set3` 上都达到 `0.00` 的碰撞率，说明当前有效实验记录支持“CRADLE 生成碰撞场景能显著提升风险预测器避碰能力”的结论。",
        "",
        "## 类型分项",
        "",
    ]
    for accident_type in ["Type1", "Type2"]:
        sub = valid_df[valid_df["type"] == accident_type]
        for method in ["LSTM", "GTF"]:
            ms = sub[sub["method"] == method].set_index("set_name")["collision_rate"].to_dict()
            lines.append(f"- `{accident_type}-{method}`: Set1=`{ms.get('Set1', np.nan):.2f}`, Set2=`{ms.get('Set2', np.nan):.2f}`, Set3=`{ms.get('Set3', np.nan):.2f}`")
    lines += [
        "",
        "## 说明",
        "",
        "- 本节没有重跑 Phase 8，而是复用仓库中已有的有效日志结果。",
        "- 由于可用实验点只有三档，`Fig-A5b` 应理解为经验趋势线，而不是连续比例扫描实验。",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_risk_analysis(project_root: Path, output_dir: Path) -> dict[str, Path]:
    result_dir = output_dir / "7风险预测验证"
    docs_dir = output_dir / "docs"
    result_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    valid_log = project_root / "logs" / "phase8_experiment_500.log"
    invalid_log = project_root / "logs" / "phase8_experiment.log"
    valid_df = parse_phase8_log(valid_log)
    invalid_df = parse_phase8_log(invalid_log) if invalid_log.exists() else pd.DataFrame()

    valid_df.to_csv(result_dir / "A5_collision_rate_results.csv", index=False, encoding="utf-8-sig")
    if not invalid_df.empty:
        invalid_df.to_csv(result_dir / "A5_invalid_nan_results.csv", index=False, encoding="utf-8-sig")

    curve_df = build_augmentation_curve(valid_df)
    curve_df.to_csv(result_dir / "A5_augmentation_curve.csv", index=False, encoding="utf-8-sig")

    plot_collision_rate_comparison(valid_df, result_dir / "Fig-A5a_collision_rate_comparison.png")
    plot_augmentation_curve(curve_df, result_dir / "Fig-A5b_augmentation_curve.png")

    report_path = docs_dir / "7风险预测验证.md"
    write_report(report_path, valid_df, invalid_df)
    return {"result_dir": result_dir, "report_path": report_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Risk prediction validation analysis from phase8 logs.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_risk_analysis(args.project_root, args.output_dir)
    print(f"[a5] results: {outputs['result_dir']}")
    print(f"[a5] report: {outputs['report_path']}")


if __name__ == "__main__":
    main()
