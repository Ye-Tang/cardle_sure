from __future__ import annotations

import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Droid Sans Fallback", "AR PL UMing CN", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCENE_RE = re.compile(r"\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}")


@dataclass
class SceneMeta:
    scene_id: str
    month: int
    split: str
    raw_type: str


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("，", ",").replace(" ", "")


def read_scene_meta(scene_dir: Path, generated_month_start: int) -> SceneMeta | None:
    txt_path = scene_dir / f"{scene_dir.name}.txt"
    if not txt_path.exists():
        return None
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    month = int(scene_dir.name.split("_")[1])
    return SceneMeta(
        scene_id=scene_dir.name,
        month=month,
        split="generated" if month >= generated_month_start else "original",
        raw_type=normalize_text(rows[0].get("事故类型", "")) or "未知",
    )


def build_signature(snapshot_df: pd.DataFrame, collision_dist_threshold: float = 8.0) -> set[str]:
    signature: set[str] = set()
    if snapshot_df.empty:
        return signature
    x_min = snapshot_df.groupby("vehicle_id")["pos_x"].agg(["min", "max"])
    lane_change_count = int(((x_min["max"] - x_min["min"]).abs() >= 2.8).sum())
    if lane_change_count > 0:
        signature.add("lane_change")

    snapshot_df = snapshot_df.sort_values(["vehicle_id", "ts"]).copy()
    snapshot_df["prev_speed"] = snapshot_df.groupby("vehicle_id")["speed_mps"].shift(1)
    snapshot_df["dt"] = snapshot_df.groupby("vehicle_id")["ts_sec"].diff()
    snapshot_df["accel"] = (snapshot_df["speed_mps"] - snapshot_df["prev_speed"]) / snapshot_df["dt"].replace(0, np.nan)
    if (snapshot_df["accel"] < -3.0).sum() > 0:
        signature.add("hard_brake")
    if (snapshot_df["speed_mps"] < 2.0).sum() > 0:
        signature.add("slow_obstacle")

    grouped = snapshot_df.groupby("ts", sort=False)
    rear_end_risk = 0
    multi_conflict = 0
    for _, group in grouped:
        if len(group) < 2:
            continue
        g = group.copy()
        g["lane_bin"] = np.floor((g["pos_x"] + 20.0) / 3.75)
        for _, lane_group in g.groupby("lane_bin"):
            if len(lane_group) < 2:
                continue
            lane_group = lane_group.sort_values("pos_y").reset_index(drop=True)
            follower = lane_group.iloc[:-1]
            leader = lane_group.iloc[1:]
            lane_conflicts = 0
            for idx in range(len(follower)):
                gap = float(leader.iloc[idx]["pos_y"] - follower.iloc[idx]["pos_y"])
                rel_speed = float(follower.iloc[idx]["speed_mps"] - leader.iloc[idx]["speed_mps"])
                if gap <= 0:
                    continue
                if rel_speed > 0.1:
                    ttc = gap / rel_speed
                    if ttc < 2.0:
                        rear_end_risk += 1
                        lane_conflicts += 1
            if lane_conflicts >= 2:
                multi_conflict += 1

        coords = g[["pos_x", "pos_y"]].to_numpy(dtype=float)
        if len(coords) >= 2:
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    if float(np.linalg.norm(coords[i] - coords[j])) < collision_dist_threshold:
                        signature.add("cross_lane_conflict")
                        break
    if rear_end_risk > 0:
        signature.add("rear_end_risk")
    if multi_conflict > 0:
        signature.add("multi_vehicle_conflict")
    return signature


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def nearest_prior_value(series_df: pd.DataFrame, target_sec: float, value_col: str) -> float:
    if series_df.empty:
        return float("nan")
    subset = series_df[series_df["rel_sec"] <= target_sec]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[-1][value_col])


def analyze_scene(scene_dir: Path, meta: SceneMeta, max_csv_per_scene: int = 140) -> dict[str, object] | None:
    csv_paths = [p for p in sorted(scene_dir.rglob("*.csv")) if p.name != "事故信息.csv" and "_status" not in p.stem]
    if len(csv_paths) > max_csv_per_scene:
        selected = np.linspace(0, len(csv_paths) - 1, num=max_csv_per_scene, dtype=int)
        csv_paths = [csv_paths[idx] for idx in selected]

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, usecols=["time_stamp", "id", "pos_x", "pos_y", "spd_x", "spd_y"])
        except ValueError:
            continue
        if len(df) < 2:
            continue
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.dropna(subset=["time_stamp", "pos_x", "pos_y", "spd_x", "spd_y"]).sort_values("time_stamp").reset_index(drop=True)
        if len(df) < 2:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "ts": df["time_stamp"],
                    "vehicle_id": int(df["id"].iloc[0]),
                    "pos_x": df["pos_x"].astype(float),
                    "pos_y": df["pos_y"].astype(float),
                    "speed_mps": df["spd_y"].astype(float) / 3.6,
                }
            )
        )
    if not frames:
        return None

    snapshot_df = pd.concat(frames, ignore_index=True).sort_values(["ts", "vehicle_id"]).reset_index(drop=True)
    ts0 = snapshot_df["ts"].min()
    snapshot_df["ts_sec"] = (snapshot_df["ts"] - ts0).dt.total_seconds()
    # Coarsen timestamps slightly to keep propagation metrics tractable on large scenes.
    snapshot_df["ts_bucket"] = (snapshot_df["ts_sec"] * 5).round() / 5.0
    snapshot_df = (
        snapshot_df.sort_values(["ts_bucket", "vehicle_id", "ts"])
        .drop(columns=["ts_sec"])
        .groupby(["ts_bucket", "vehicle_id"], as_index=False)
        .last()
        .rename(columns={"ts_bucket": "ts_sec"})
    )
    snapshot_df["ts"] = ts0 + pd.to_timedelta(snapshot_df["ts_sec"], unit="s")
    t_max = float(snapshot_df["ts_sec"].max())
    if not np.isfinite(t_max) or t_max <= 0:
        return None

    collision_rel_sec = np.nan
    collision_x = np.nan
    collision_y = np.nan
    delta_v = np.nan
    best_dist = np.inf
    best_pair: tuple[int, int] | None = None

    ttc_rows: list[dict[str, float]] = []
    for ts, group in snapshot_df.groupby("ts", sort=True):
        rel_sec = float((ts - ts0).total_seconds())
        g = group.copy()
        g["lane_bin"] = np.floor((g["pos_x"] + 20.0) / 3.75)

        min_ttc = np.inf
        for _, lane_group in g.groupby("lane_bin"):
            if len(lane_group) < 2:
                continue
            lane_group = lane_group.sort_values("pos_y").reset_index(drop=True)
            follower = lane_group.iloc[:-1]
            leader = lane_group.iloc[1:]
            for idx in range(len(follower)):
                gap = float(leader.iloc[idx]["pos_y"] - follower.iloc[idx]["pos_y"])
                rel_speed = float(follower.iloc[idx]["speed_mps"] - leader.iloc[idx]["speed_mps"])
                if gap > 0 and rel_speed > 0.1:
                    min_ttc = min(min_ttc, gap / rel_speed)

        coords = g[["vehicle_id", "pos_x", "pos_y", "speed_mps"]].to_numpy(dtype=float)
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = float(np.linalg.norm(coords[i, 1:3] - coords[j, 1:3]))
                if dist < best_dist:
                    best_dist = dist
                    collision_rel_sec = rel_sec
                    collision_x = float((coords[i, 1] + coords[j, 1]) / 2.0)
                    collision_y = float((coords[i, 2] + coords[j, 2]) / 2.0)
                    delta_v = abs(float(coords[i, 3] - coords[j, 3]))
                    best_pair = (int(coords[i, 0]), int(coords[j, 0]))
        ttc_rows.append({"rel_sec": rel_sec, "min_ttc": float(min_ttc if np.isfinite(min_ttc) else np.nan)})

    if not np.isfinite(collision_rel_sec):
        return None
    ttc_df = pd.DataFrame(ttc_rows).sort_values("rel_sec").drop_duplicates("rel_sec")
    pre_collision_ttc = ttc_df[ttc_df["rel_sec"] <= collision_rel_sec].copy()

    ttc_1s = nearest_prior_value(pre_collision_ttc, collision_rel_sec - 1.0, "min_ttc")
    ttc_2s = nearest_prior_value(pre_collision_ttc, collision_rel_sec - 2.0, "min_ttc")
    ttc_3s = nearest_prior_value(pre_collision_ttc, collision_rel_sec - 3.0, "min_ttc")
    low_ttc = pre_collision_ttc[(pre_collision_ttc["min_ttc"] < 3.0) & np.isfinite(pre_collision_ttc["min_ttc"])]
    propagation_start = float(low_ttc["rel_sec"].iloc[0]) if not low_ttc.empty else np.nan
    propagation_duration = float(collision_rel_sec - propagation_start) if np.isfinite(propagation_start) else np.nan

    full_signature = build_signature(snapshot_df)
    evolution_rows = []
    for frac in [0.4, 0.6, 0.8, 1.0]:
        cutoff = t_max * frac
        partial = snapshot_df[snapshot_df["ts_sec"] <= cutoff].copy()
        partial_signature = build_signature(partial)
        evolution_rows.append({"fraction": frac, "proxy_jaccard": jaccard_similarity(partial_signature, full_signature)})

    if np.isfinite(delta_v) and (delta_v > 10.0 or (np.isfinite(ttc_1s) and ttc_1s < 1.0) or (np.isfinite(propagation_duration) and propagation_duration < 1.0)):
        severity = "high"
    elif np.isfinite(delta_v) and delta_v < 5.0 and (np.isfinite(ttc_1s) and ttc_1s > 2.0) and (np.isfinite(propagation_duration) and propagation_duration > 2.0):
        severity = "low"
    else:
        severity = "mid"

    return {
        "scene_row": {
            "scene_id": meta.scene_id,
            "raw_type": meta.raw_type,
            "collision_rel_sec": collision_rel_sec,
            "scene_duration_sec": t_max,
            "collision_progress_pct": 100.0 * collision_rel_sec / t_max,
            "collision_x": collision_x,
            "collision_y": collision_y,
            "delta_v_mps": delta_v,
            "ttc_1s": ttc_1s,
            "ttc_2s": ttc_2s,
            "ttc_3s": ttc_3s,
            "propagation_start_sec": propagation_start,
            "propagation_duration_sec": propagation_duration,
            "proxy_complexity": len(full_signature),
            "proxy_signature": ",".join(sorted(full_signature)),
            "severity_proxy": severity,
        },
        "evolution_rows": [
            {"scene_id": meta.scene_id, "severity_proxy": severity, **row}
            for row in evolution_rows
        ],
    }


def scene_worker(scene_dir_str: str, generated_month_start: int) -> dict[str, object] | None:
    scene_dir = Path(scene_dir_str)
    meta = read_scene_meta(scene_dir, generated_month_start)
    if meta is None or meta.split != "generated":
        return None
    return analyze_scene(scene_dir, meta)


def plot_evolution(evolution_df: pd.DataFrame, out_path: Path) -> None:
    agg = evolution_df.groupby("fraction", as_index=False)["proxy_jaccard"].mean()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=agg, x="fraction", y="proxy_jaccard", marker="o", color="#4C78A8")
    plt.xticks([0.4, 0.6, 0.8, 1.0], ["40%", "60%", "80%", "100%"])
    plt.ylim(0, 1.05)
    plt.title("Fig-A3a Proxy Causal Evolution Over Time")
    plt.xlabel("Observed Trajectory Fraction")
    plt.ylabel("Mean Proxy Jaccard")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_collision_cdf(scene_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    available = ["high", "low"] if (scene_df["severity_proxy"] == "low").any() else ["high", "mid"]
    colors = {"high": "#E45756", "low": "#54A24B", "mid": "#4C78A8"}
    for severity in available:
        subset = scene_df[scene_df["severity_proxy"] == severity]["collision_progress_pct"].dropna().sort_values()
        if len(subset) == 0:
            continue
        y = np.arange(1, len(subset) + 1) / len(subset)
        plt.step(subset, y, where="post", label=severity, color=colors[severity])
    plt.title("Fig-A3b Collision-Time CDF by Severity Group")
    plt.xlabel("Collision Progress in Scene (%)")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_ttc(scene_df: pd.DataFrame, out_path: Path) -> None:
    available = ["high", "low"] if (scene_df["severity_proxy"] == "low").any() else ["high", "mid"]
    palette = {"high": "#E45756", "low": "#54A24B", "mid": "#4C78A8"}
    plot_df = scene_df.melt(
        id_vars=["scene_id", "severity_proxy"],
        value_vars=["ttc_1s", "ttc_2s", "ttc_3s"],
        var_name="horizon",
        value_name="ttc_value",
    ).dropna()
    plot_df = plot_df[plot_df["severity_proxy"].isin(available)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    horizon_labels = {"ttc_1s": "1s before", "ttc_2s": "2s before", "ttc_3s": "3s before"}
    for ax, horizon in zip(axes, ["ttc_1s", "ttc_2s", "ttc_3s"]):
        subset = plot_df[plot_df["horizon"] == horizon]
        sns.boxplot(data=subset, x="severity_proxy", y="ttc_value", hue="severity_proxy", order=available, palette=palette, legend=False, ax=ax)
        ax.set_title(horizon_labels[horizon])
        ax.set_xlabel("")
        ax.set_ylabel("TTC (s)")
    fig.suptitle("Fig-A3c TTC Distribution Before Collision")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_delta_v(scene_df: pd.DataFrame, out_path: Path) -> None:
    available = ["high", "low"] if (scene_df["severity_proxy"] == "low").any() else ["high", "mid"]
    palette = {"high": "#E45756", "low": "#54A24B", "mid": "#4C78A8"}
    plot_df = scene_df[scene_df["severity_proxy"].isin(available)].copy()
    plt.figure(figsize=(7, 5))
    sns.violinplot(data=plot_df, x="severity_proxy", y="delta_v_mps", hue="severity_proxy", order=available, palette=palette, legend=False)
    plt.title("Fig-A3d Relative Speed at Collision")
    plt.xlabel("")
    plt.ylabel("Delta v (m/s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_propagation(scene_df: pd.DataFrame, out_path: Path) -> None:
    available = ["high", "low"] if (scene_df["severity_proxy"] == "low").any() else ["high", "mid"]
    palette = {"high": "#E45756", "low": "#54A24B", "mid": "#4C78A8"}
    plot_df = scene_df[scene_df["severity_proxy"].isin(available)].copy()
    plt.figure(figsize=(8, 5))
    sns.histplot(data=plot_df, x="propagation_duration_sec", hue="severity_proxy", bins=16, kde=True, palette=palette, alpha=0.5)
    plt.title("Fig-A3e Propagation Duration Distribution")
    plt.xlabel("Propagation Duration (s)")
    plt.ylabel("Scene Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_report(output_path: Path, summary_df: pd.DataFrame, evolution_df: pd.DataFrame, scene_df: pd.DataFrame) -> None:
    summary = summary_df.set_index("metric")["value"].to_dict()
    evo = evolution_df.groupby("fraction", as_index=False)["proxy_jaccard"].mean()
    fallback_note = "- 当前样本中未识别出满足严格低严重程度阈值的场景，因此图表中的分组对比使用 `high` vs `mid` 作为补充展示。" if int(summary.get("low_severity_scene_count", 0)) == 0 else "- 当前样本同时包含高/低严重程度场景，图表按 `high` vs `low` 对比。"
    lines = [
        "# 5 事故严重程度传播特征分析",
        "",
        "## 评估口径",
        "",
        "- 仅使用 `2024-09` 到 `2024-11` 的生成场景。",
        "- 高严重程度：满足 `delta_v > 10 m/s`、或 `ttc_1s < 1 s`、或 `propagation_duration < 1 s` 中任一条件。",
        "- 低严重程度：同时满足 `delta_v < 5 m/s`、`ttc_1s > 2 s`、`propagation_duration > 2 s`。",
        "- 其余场景记为 `mid`，不用于高低分组对比，但仍参与整体时序演化统计。",
        fallback_note,
        "",
        "## 关键结果",
        "",
        f"- 生成场景总数：`{int(summary.get('generated_scene_count', 0))}`；高严重程度：`{int(summary.get('high_severity_scene_count', 0))}`；低严重程度：`{int(summary.get('low_severity_scene_count', 0))}`；中间带：`{int(summary.get('mid_severity_scene_count', 0))}`。",
        f"- 高严重程度平均碰撞相对速度：`{summary.get('high_mean_delta_v_mps', float('nan')):.4f} m/s`；低严重程度：`{summary.get('low_mean_delta_v_mps', float('nan')):.4f} m/s`。",
        f"- 高严重程度平均传播时长：`{summary.get('high_mean_propagation_duration_sec', float('nan')):.4f} s`；低严重程度：`{summary.get('low_mean_propagation_duration_sec', float('nan')):.4f} s`。",
        f"- 碰撞前 1 秒的平均 TTC：高严重程度 ` {summary.get('high_mean_ttc_1s', float('nan')):.4f} s`，低严重程度 ` {summary.get('low_mean_ttc_1s', float('nan')):.4f} s`。",
        "",
        "## 时序演化",
        "",
    ]
    for _, row in evo.iterrows():
        lines.append(f"- {int(row['fraction'] * 100)}% 轨迹长度下，平均 proxy Jaccard 为 `{row['proxy_jaccard']:.4f}`")
    lines += [
        "",
        "## 解释",
        "",
        "- `proxy Jaccard` 用于替代 plan 中的 ACG 完整率，表示部分轨迹截断后可恢复的交互因果模式与全轨迹模式的一致程度。",
        "- `proxy_complexity` 表示每个场景被识别出的交互事件类型数，用于近似因果链复杂度。",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_propagation_analysis(data_dir: Path, output_dir: Path, generated_month_start: int = 9) -> dict[str, Path]:
    result_dir = output_dir / "5事故严重程度传播特征"
    docs_dir = output_dir / "docs"
    result_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and SCENE_RE.fullmatch(p.name)])
    scene_rows: list[dict[str, object]] = []
    evolution_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(8, max(1, (os.cpu_count() or 2) - 1), max(1, len(scene_dirs)))) as executor:
        futures = [executor.submit(scene_worker, str(scene_dir), generated_month_start) for scene_dir in scene_dirs]
        for future in as_completed(futures):
            payload = future.result()
            if payload is None:
                continue
            scene_rows.append(payload["scene_row"])
            evolution_rows.extend(payload["evolution_rows"])

    scene_df = pd.DataFrame(scene_rows).sort_values("scene_id").reset_index(drop=True)
    evolution_df = pd.DataFrame(evolution_rows)
    scene_df.to_csv(result_dir / "A3_scene_metrics.csv", index=False, encoding="utf-8-sig")
    evolution_df.to_csv(result_dir / "A3_evolution_curve.csv", index=False, encoding="utf-8-sig")

    plot_evolution(evolution_df, result_dir / "Fig-A3a_proxy_causal_evolution.png")
    plot_collision_cdf(scene_df, result_dir / "Fig-A3b_collision_cdf.png")
    plot_ttc(scene_df[scene_df["severity_proxy"].isin(["high", "low"])], result_dir / "Fig-A3c_ttc_distribution.png")
    plot_delta_v(scene_df, result_dir / "Fig-A3d_delta_v_violin.png")
    plot_propagation(scene_df, result_dir / "Fig-A3e_propagation_duration.png")

    summary_rows = [
        {"metric": "generated_scene_count", "value": int(len(scene_df))},
        {"metric": "high_severity_scene_count", "value": int((scene_df["severity_proxy"] == "high").sum())},
        {"metric": "low_severity_scene_count", "value": int((scene_df["severity_proxy"] == "low").sum())},
        {"metric": "mid_severity_scene_count", "value": int((scene_df["severity_proxy"] == "mid").sum())},
    ]
    for severity in ["high", "low"]:
        subset = scene_df[scene_df["severity_proxy"] == severity]
        summary_rows.extend(
            [
                {"metric": f"{severity}_mean_delta_v_mps", "value": float(subset["delta_v_mps"].mean())},
                {"metric": f"{severity}_mean_propagation_duration_sec", "value": float(subset["propagation_duration_sec"].mean())},
                {"metric": f"{severity}_mean_ttc_1s", "value": float(subset["ttc_1s"].mean())},
                {"metric": f"{severity}_mean_collision_progress_pct", "value": float(subset["collision_progress_pct"].mean())},
                {"metric": f"{severity}_mean_proxy_complexity", "value": float(subset["proxy_complexity"].mean())},
            ]
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(result_dir / "A3_summary_metrics.csv", index=False, encoding="utf-8-sig")

    report_path = docs_dir / "5事故严重程度传播特征.md"
    write_report(report_path, summary_df, evolution_df, scene_df)
    return {"result_dir": result_dir, "report_path": report_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propagation analysis for generated accident scenes.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-month-start", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_propagation_analysis(args.data_dir, args.output_dir, generated_month_start=args.generated_month_start)
    print(f"[a3] results: {outputs['result_dir']}")
    print(f"[a3] report: {outputs['report_path']}")


if __name__ == "__main__":
    main()
