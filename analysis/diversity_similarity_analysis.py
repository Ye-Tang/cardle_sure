from __future__ import annotations

import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from evaluation.evaluator import segment_primitives


sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Droid Sans Fallback", "AR PL UMing CN", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCENE_RE = re.compile(r"\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}")
FILE_RE = re.compile(
    r"(?P<ts>\d{14})_y(?P<y1>-?\d+)-(?P<y2>-?\d+)_d(?P<d>-?\d+)_v(?P<v1>-?\d+)-(?P<v2>-?\d+)_id(?P<vid>\d+)",
    re.IGNORECASE,
)
PRIMITIVES = ("following", "lane_change", "accelerating", "decelerating")
TYPE1 = "rear_end"
TYPE2 = "lane_change_conflict"


@dataclass
class SceneMeta:
    scene_id: str
    month: int
    split: str
    accident_type_raw: str
    analysis_type: str | None


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("，", ",").replace(" ", "")


def classify_analysis_type(raw_type: str) -> str | None:
    text = normalize_text(raw_type)
    if not text:
        return None
    if "变道" in text or "碰撞" in text or "刮蹭" in text or "护栏" in text:
        return TYPE2
    if "追尾" in text:
        return TYPE1
    return None


def read_scene_meta(scene_dir: Path, generated_month_start: int) -> SceneMeta | None:
    txt_path = scene_dir / f"{scene_dir.name}.txt"
    if not txt_path.exists():
        return None
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    raw_type = normalize_text(rows[0].get("事故类型", ""))
    month = int(scene_dir.name.split("_")[1])
    split = "generated" if month >= generated_month_start else "original"
    return SceneMeta(
        scene_id=scene_dir.name,
        month=month,
        split=split,
        accident_type_raw=raw_type or "未知",
        analysis_type=classify_analysis_type(raw_type),
    )


def resample_polyline(points: np.ndarray, n_points: int = 60) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.zeros((n_points, 2), dtype=float)
    deltas = np.diff(arr, axis=0)
    seg_lengths = np.sqrt((deltas**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total <= 1e-8:
        return np.repeat(arr[:1], n_points, axis=0)
    targets = np.linspace(0.0, total, n_points)
    out = np.zeros((n_points, 2), dtype=float)
    for dim in range(2):
        out[:, dim] = np.interp(targets, cumulative, arr[:, dim])
    return out


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    a = np.asarray(seq_a, dtype=float)
    b = np.asarray(seq_b, dtype=float)
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n + m, 1))


def build_scene_payload(scene_dir: Path, meta: SceneMeta, max_csv_per_scene: int = 220) -> dict[str, object]:
    csv_paths = [p for p in sorted(scene_dir.rglob("*.csv")) if p.name != "事故信息.csv" and "_status" not in p.stem]
    if len(csv_paths) > max_csv_per_scene:
        selected = np.linspace(0, len(csv_paths) - 1, num=max_csv_per_scene, dtype=int)
        csv_paths = [csv_paths[idx] for idx in selected]

    primitive_counter = {name: 0 for name in PRIMITIVES}
    trajectory_points: list[np.ndarray] = []
    snapshot_frames: list[pd.DataFrame] = []
    scene_speed_means: list[float] = []
    scene_lane_changes = 0

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, usecols=["time_stamp", "id", "pos_x", "pos_y", "spd_x", "spd_y"])
        except ValueError:
            continue
        if df.empty:
            continue
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.dropna(subset=["time_stamp", "pos_x", "pos_y", "spd_x", "spd_y"]).sort_values("time_stamp").reset_index(drop=True)
        if len(df) < 2:
            continue

        x = df["pos_x"].to_numpy(dtype=float)
        y = df["pos_y"].to_numpy(dtype=float)
        vx = df["spd_x"].to_numpy(dtype=float) / 3.6
        vy = df["spd_y"].to_numpy(dtype=float) / 3.6
        state = np.column_stack([x, y, vx, vy]).astype(float)
        scene_speed_means.append(float(np.mean(vy)))
        scene_lane_changes += int(abs(x[-1] - x[0]) >= 2.8)
        trajectory_points.append(np.column_stack([x, y]))

        for primitive in segment_primitives(state):
            primitive_counter[primitive["type"]] += max(1, primitive["end"] - primitive["start"])

        snapshot_frames.append(
            pd.DataFrame(
                {
                    "ts": df["time_stamp"],
                    "vehicle_id": int(df["id"].iloc[0]),
                    "pos_x": x,
                    "pos_y": y,
                    "speed_mps": vy,
                }
            )
        )

    collision_x = np.nan
    collision_y = np.nan
    conflict_min_dist = np.nan
    representative_path = np.zeros((60, 2), dtype=float)
    pair_count = 0

    if snapshot_frames:
        snapshot_df = pd.concat(snapshot_frames, ignore_index=True)
        snapshot_df = snapshot_df.dropna(subset=["pos_x", "pos_y"]).copy()
        best_pair: tuple[int, int] | None = None
        best_midpoints: list[list[float]] = []
        best_dist = np.inf

        for _, group in snapshot_df.groupby("ts", sort=False):
            if len(group) < 2:
                continue
            pair_count += len(group) * (len(group) - 1) // 2
            values = group[["vehicle_id", "pos_x", "pos_y"]].to_numpy()
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    dist = float(np.linalg.norm(values[i, 1:3] - values[j, 1:3]))
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (int(values[i, 0]), int(values[j, 0]))
                        collision_x = float((values[i, 1] + values[j, 1]) / 2.0)
                        collision_y = float((values[i, 2] + values[j, 2]) / 2.0)

        if best_pair is not None:
            subset = snapshot_df[snapshot_df["vehicle_id"].isin(best_pair)].copy()
            ts_groups = subset.groupby("ts", sort=True)
            for _, group in ts_groups:
                if len(group) != 2:
                    continue
                best_midpoints.append([float(group["pos_x"].mean()), float(group["pos_y"].mean())])
            if len(best_midpoints) >= 2:
                representative_path = resample_polyline(np.asarray(best_midpoints, dtype=float), n_points=60)
        conflict_min_dist = float(best_dist) if np.isfinite(best_dist) else np.nan

    primitive_total = sum(primitive_counter.values())
    primitive_props = {
        f"primitive_prop_{name}": float(primitive_counter[name] / primitive_total) if primitive_total > 0 else 0.0
        for name in PRIMITIVES
    }
    feature_row = {
        "scene_id": meta.scene_id,
        "analysis_type": meta.analysis_type,
        "month": meta.month,
        "split": meta.split,
        "raw_type": meta.accident_type_raw,
        "trajectory_count": len(trajectory_points),
        "mean_speed_mps": float(np.mean(scene_speed_means)) if scene_speed_means else np.nan,
        "lane_change_vehicle_count": scene_lane_changes,
        "collision_x": collision_x,
        "collision_y": collision_y,
        "conflict_min_dist": conflict_min_dist,
        "pair_count": pair_count,
        **primitive_props,
    }
    return {
        "feature_row": feature_row,
        "point_clouds": trajectory_points,
        "representative_path": representative_path,
        "collision_point": (collision_x, collision_y),
    }


def scene_worker(scene_dir_str: str, generated_month_start: int) -> dict[str, object] | None:
    scene_dir = Path(scene_dir_str)
    meta = read_scene_meta(scene_dir, generated_month_start=generated_month_start)
    if meta is None or meta.split != "generated" or meta.analysis_type is None:
        return None
    return build_scene_payload(scene_dir, meta)


def compute_diversity_curve(point_clouds: list[list[np.ndarray]], steps: list[int], x_range: tuple[float, float], y_range: tuple[float, float], nx: int = 200, ny: int = 10) -> pd.DataFrame:
    rows = []
    x_min, x_max = x_range
    y_min, y_max = y_range
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    for n in steps:
        occupied = np.zeros((nx, ny), dtype=bool)
        for scene_points in point_clouds[:n]:
            for points in scene_points:
                for x, y in points:
                    if x < x_min or x > x_max or y < y_min or y > y_max:
                        continue
                    ix = min(nx - 1, max(0, int(np.floor((x - x_min) / dx))))
                    iy = min(ny - 1, max(0, int(np.floor((y - y_min) / dy))))
                    occupied[ix, iy] = True
        rows.append({"n_scenes": n, "diversity_pct": float(occupied.sum() / (nx * ny) * 100.0)})
    return pd.DataFrame(rows)


def plot_diversity_curves(type1_df: pd.DataFrame, type2_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=type1_df, x="n_scenes", y="diversity_pct", marker="o", color="#4C78A8", ax=ax, label="Rear-end")
    ax.set_title("Fig-A2a Diversity Growth Curve: Rear-end")
    ax.set_xlabel("Number of Scenes")
    ax.set_ylabel("Diversity Dk (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "Fig-A2a_type1_diversity_curve.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=type2_df, x="n_scenes", y="diversity_pct", marker="o", color="#F58518", ax=ax, label="Lane-change conflict")
    ax.set_title("Fig-A2b Diversity Growth Curve: Lane-change Conflict")
    ax.set_xlabel("Number of Scenes")
    ax.set_ylabel("Diversity Dk (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "Fig-A2b_type2_diversity_curve.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=type1_df, x="n_scenes", y="diversity_pct", marker="o", color="#4C78A8", ax=ax, label="Rear-end")
    sns.lineplot(data=type2_df, x="n_scenes", y="diversity_pct", marker="o", color="#F58518", ax=ax, label="Lane-change conflict")
    ax.set_title("Fig-A2c Diversity Growth Comparison")
    ax.set_xlabel("Number of Scenes")
    ax.set_ylabel("Diversity Dk (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "Fig-A2c_diversity_comparison.png", dpi=220)
    plt.close(fig)


def plot_collision_density(feature_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = feature_df.dropna(subset=["collision_x", "collision_y"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    type_to_title = {
        TYPE1: "Rear-end",
        TYPE2: "Lane-change conflict",
    }
    for ax, analysis_type in zip(axes, [TYPE1, TYPE2]):
        subset = plot_df[plot_df["analysis_type"] == analysis_type]
        if subset.empty:
            ax.set_title(type_to_title[analysis_type])
            ax.set_xlabel("Collision X")
            ax.set_ylabel("Collision Y")
            continue

        x_min = float(subset["collision_x"].min())
        x_max = float(subset["collision_x"].max())
        y_min = float(subset["collision_y"].min())
        y_max = float(subset["collision_y"].max())
        x_pad = max(1.0, 0.1 * max(x_max - x_min, 1.0))
        y_pad = max(5.0, 0.1 * max(y_max - y_min, 1.0))

        if len(subset) >= 3:
            xy = subset[["collision_x", "collision_y"]].to_numpy().T
            kde = gaussian_kde(xy)
            x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 80)
            y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 120)
            xx, yy = np.meshgrid(x_grid, y_grid)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=12, cmap="YlOrRd", alpha=0.75)
        ax.scatter(subset["collision_x"], subset["collision_y"], s=30, color="#1f1f1f", alpha=0.8)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title(type_to_title[analysis_type])
        ax.set_xlabel("Collision X")
        ax.set_ylabel("Collision Y")
    fig.suptitle("Fig-A2d Collision Point Density Heatmaps")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tsne(feature_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    feat_cols = [
        "trajectory_count",
        "mean_speed_mps",
        "lane_change_vehicle_count",
        "collision_x",
        "collision_y",
        "conflict_min_dist",
        "pair_count",
        "primitive_prop_following",
        "primitive_prop_lane_change",
        "primitive_prop_accelerating",
        "primitive_prop_decelerating",
    ]
    data = feature_df[feat_cols].copy().fillna(feature_df[feat_cols].median(numeric_only=True))
    scaled = StandardScaler().fit_transform(data)
    perplexity = max(3, min(10, len(feature_df) - 1))
    embedded = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto").fit_transform(scaled)
    tsne_df = feature_df[["scene_id", "analysis_type"]].copy()
    tsne_df["tsne_x"] = embedded[:, 0]
    tsne_df["tsne_y"] = embedded[:, 1]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=tsne_df, x="tsne_x", y="tsne_y", hue="analysis_type", palette={TYPE1: "#4C78A8", TYPE2: "#F58518"}, s=70)
    plt.title("Fig-A2e t-SNE of Generated Scene Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return tsne_df


def plot_primitive_composition(feature_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for analysis_type in [TYPE1, TYPE2]:
        subset = feature_df[feature_df["analysis_type"] == analysis_type]
        for primitive in PRIMITIVES:
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "primitive": primitive,
                    "proportion": float(subset[f"primitive_prop_{primitive}"].mean()) if len(subset) else 0.0,
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    titles = {TYPE1: "Rear-end", TYPE2: "Lane-change conflict"}
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for ax, analysis_type in zip(axes, [TYPE1, TYPE2]):
        subset = plot_df[plot_df["analysis_type"] == analysis_type]
        ax.pie(subset["proportion"], labels=subset["primitive"], autopct="%1.1f%%", colors=colors, startangle=90)
        ax.set_title(titles[analysis_type])
    fig.suptitle("Fig-A2f Primitive Composition by Accident Type")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return plot_df


def compute_dtw_table(scene_payloads: list[dict[str, object]], max_pairs: int = 50) -> pd.DataFrame:
    by_type: dict[str, list[dict[str, object]]] = {TYPE1: [], TYPE2: []}
    for payload in scene_payloads:
        by_type[payload["feature_row"]["analysis_type"]].append(payload)

    rows = []
    rng = np.random.default_rng(42)
    for analysis_type in [TYPE1, TYPE2]:
        items = by_type[analysis_type]
        pairs = list(combinations(range(len(items)), 2))
        if not pairs:
            continue
        rng.shuffle(pairs)
        for i, j in pairs[:max_pairs]:
            dist = dtw_distance(items[i]["representative_path"], items[j]["representative_path"])
            rows.append({"pair_group": f"intra_{analysis_type}", "distance": dist})

    inter_pairs = [(i, j) for i in range(len(by_type[TYPE1])) for j in range(len(by_type[TYPE2]))]
    rng.shuffle(inter_pairs)
    for i, j in inter_pairs[:max_pairs]:
        dist = dtw_distance(by_type[TYPE1][i]["representative_path"], by_type[TYPE2][j]["representative_path"])
        rows.append({"pair_group": "inter_type", "distance": dist})

    raw_df = pd.DataFrame(rows)
    summary = raw_df.groupby("pair_group", as_index=False).agg(
        sample_count=("distance", "count"),
        mean_distance=("distance", "mean"),
        median_distance=("distance", "median"),
        std_distance=("distance", "std"),
    )
    return raw_df, summary


def write_report(output_path: Path, summary_df: pd.DataFrame, mapping_df: pd.DataFrame, diversity_df: pd.DataFrame, dtw_summary: pd.DataFrame) -> None:
    summary = summary_df.set_index("metric")["value"].to_dict()
    inter_mean = float(summary.get("inter_type_dtw_mean", np.nan))
    rear_mean = float(summary.get("intra_rear_end_dtw_mean", np.nan))
    lane_mean = float(summary.get("intra_lane_change_conflict_dtw_mean", np.nan))
    if np.isfinite(inter_mean) and np.isfinite(rear_mean) and np.isfinite(lane_mean):
        if inter_mean > rear_mean and inter_mean > lane_mean:
            dtw_comment = "跨类型 DTW 高于两类类内 DTW，说明两类场景在轨迹结构上具有较清晰的分离性。"
        elif inter_mean > lane_mean and inter_mean <= rear_mean:
            dtw_comment = "跨类型 DTW 高于 `lane_change_conflict` 类内距离，但低于 `rear_end` 类内距离，说明 `rear_end` 家族内部异质性更强。"
        else:
            dtw_comment = "跨类型 DTW 未显著高于两类类内距离，说明仅凭当前代表轨迹的形状距离，类型分离度有限。"
    else:
        dtw_comment = "DTW 结果样本不足，暂不对类型分离度做强结论。"
    lines = [
        "# 4 多样性和相似性分析",
        "",
        "## 评估口径",
        "",
        "- 当前数据没有显式的 `Type1/Type2` 标签，因此本节对生成数据中的两类核心事故原型做替代分析。",
        f"- `rear_end`：追尾、连续追尾、连环追尾相关场景，共 `{int(summary.get('rear_end_scene_count', 0))}` 个生成场景。",
        f"- `lane_change_conflict`：变道碰撞、侧向冲突、护栏碰撞相关场景，共 `{int(summary.get('lane_change_conflict_scene_count', 0))}` 个生成场景。",
        f"- 其余未能稳定映射到两类核心原型的生成场景 `{int(summary.get('excluded_scene_count', 0))}` 个，未纳入双类型对照。",
        "",
        "## 关键结果",
        "",
        f"- `rear_end` 最终多样性 Dk 为 `{summary.get('rear_end_final_diversity_pct', float('nan')):.2f}%`。",
        f"- `lane_change_conflict` 最终多样性 Dk 为 `{summary.get('lane_change_conflict_final_diversity_pct', float('nan')):.2f}%`。",
        f"- 跨类型 DTW 均值为 `{summary.get('inter_type_dtw_mean', float('nan')):.4f}`；`rear_end` 类内 DTW 均值为 `{summary.get('intra_rear_end_dtw_mean', float('nan')):.4f}`；`lane_change_conflict` 类内 DTW 均值为 `{summary.get('intra_lane_change_conflict_dtw_mean', float('nan')):.4f}`。",
        f"- {dtw_comment}",
        "",
        "## 类型映射",
        "",
    ]
    for _, row in mapping_df.iterrows():
        lines.append(f"- `{row['analysis_type']}` <- `{row['raw_type']}` ({int(row['scene_count'])} scenes)")
    lines += [
        "",
        "## 多样性曲线终值",
        "",
    ]
    for _, row in diversity_df.sort_values(["analysis_type", "n_scenes"]).groupby("analysis_type").tail(1).iterrows():
        lines.append(f"- `{row['analysis_type']}`: `{row['diversity_pct']:.2f}%` at `{int(row['n_scenes'])}` scenes")
    lines += [
        "",
        "## DTW 汇总",
        "",
    ]
    for _, row in dtw_summary.iterrows():
        lines.append(f"- `{row['pair_group']}`: mean `{row['mean_distance']:.4f}`, median `{row['median_distance']:.4f}`, n=`{int(row['sample_count'])}`")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_diversity_similarity_analysis(data_dir: Path, output_dir: Path, generated_month_start: int = 9) -> dict[str, Path]:
    result_dir = output_dir / "4多样性和相似性"
    docs_dir = output_dir / "docs"
    result_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and SCENE_RE.fullmatch(p.name)])
    payloads: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(8, max(1, (os.cpu_count() or 2) - 1), max(1, len(scene_dirs)))) as executor:
        futures = [executor.submit(scene_worker, str(scene_dir), generated_month_start) for scene_dir in scene_dirs]
        for future in as_completed(futures):
            payload = future.result()
            if payload is not None:
                payloads.append(payload)

    payloads.sort(key=lambda item: item["feature_row"]["scene_id"])
    feature_df = pd.DataFrame([payload["feature_row"] for payload in payloads]).sort_values("scene_id").reset_index(drop=True)
    feature_df.to_csv(result_dir / "A2_scene_features.csv", index=False, encoding="utf-8-sig")

    mapping_df = feature_df.groupby(["analysis_type", "raw_type"], as_index=False).agg(scene_count=("scene_id", "count")).sort_values(["analysis_type", "scene_count"], ascending=[True, False])
    mapping_df.to_csv(result_dir / "A2_type_mapping.csv", index=False, encoding="utf-8-sig")

    all_points = np.vstack([points for payload in payloads for scene_points in payload["point_clouds"] for points in [scene_points] if len(points) > 0])
    x_range = (float(np.percentile(all_points[:, 0], 1)), float(np.percentile(all_points[:, 0], 99)))
    y_range = (float(np.percentile(all_points[:, 1], 1)), float(np.percentile(all_points[:, 1], 99)))

    type_rows = []
    diversity_frames = []
    for analysis_type in [TYPE1, TYPE2]:
        type_payloads = [payload for payload in payloads if payload["feature_row"]["analysis_type"] == analysis_type]
        point_clouds = [payload["point_clouds"] for payload in type_payloads]
        steps = sorted(set(np.linspace(1, len(type_payloads), num=min(len(type_payloads), 6), dtype=int).tolist()))
        curve = compute_diversity_curve(point_clouds, steps, x_range=x_range, y_range=y_range)
        curve["analysis_type"] = analysis_type
        diversity_frames.append(curve)
        type_rows.append({"metric": f"{analysis_type}_scene_count", "value": len(type_payloads)})
        type_rows.append({"metric": f"{analysis_type}_final_diversity_pct", "value": float(curve["diversity_pct"].iloc[-1]) if len(curve) else np.nan})
    diversity_df = pd.concat(diversity_frames, ignore_index=True)
    diversity_df.to_csv(result_dir / "A2_diversity_curves.csv", index=False, encoding="utf-8-sig")
    plot_diversity_curves(
        diversity_df[diversity_df["analysis_type"] == TYPE1],
        diversity_df[diversity_df["analysis_type"] == TYPE2],
        result_dir,
    )

    plot_collision_density(feature_df, result_dir / "Fig-A2d_collision_density.png")
    tsne_df = plot_tsne(feature_df, result_dir / "Fig-A2e_tsne.png")
    tsne_df.to_csv(result_dir / "A2_tsne_embedding.csv", index=False, encoding="utf-8-sig")
    primitive_df = plot_primitive_composition(feature_df, result_dir / "Fig-A2f_primitive_composition.png")
    primitive_df.to_csv(result_dir / "A2_primitive_composition.csv", index=False, encoding="utf-8-sig")

    dtw_raw, dtw_summary = compute_dtw_table(payloads, max_pairs=50)
    dtw_raw.to_csv(result_dir / "Table-A2_dtw_pairs.csv", index=False, encoding="utf-8-sig")
    dtw_summary.to_csv(result_dir / "Table-A2_dtw_summary.csv", index=False, encoding="utf-8-sig")
    for _, row in dtw_summary.iterrows():
        type_rows.append({"metric": f"{row['pair_group']}_dtw_mean", "value": row["mean_distance"]})
    excluded = 0
    for scene_dir in scene_dirs:
        meta = read_scene_meta(scene_dir, generated_month_start)
        if meta is not None and meta.split == "generated" and meta.analysis_type is None:
            excluded += 1
    type_rows.append({"metric": "excluded_scene_count", "value": excluded})
    summary_df = pd.DataFrame(type_rows)
    summary_df.to_csv(result_dir / "A2_summary_metrics.csv", index=False, encoding="utf-8-sig")

    report_path = docs_dir / "4多样性和相似性.md"
    write_report(report_path, summary_df, mapping_df, diversity_df, dtw_summary)
    return {"result_dir": result_dir, "report_path": report_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diversity and similarity analysis for generated accident scenes.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-month-start", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_diversity_similarity_analysis(args.data_dir, args.output_dir, generated_month_start=args.generated_month_start)
    print(f"[a2] results: {outputs['result_dir']}")
    print(f"[a2] report: {outputs['report_path']}")


if __name__ == "__main__":
    main()
