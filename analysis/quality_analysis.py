from __future__ import annotations

import argparse
import os
import csv
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.stats import entropy

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
EVENTS = (
    "lane_change",
    "hard_brake",
    "rear_end_risk",
    "multi_vehicle_conflict",
    "slow_obstacle",
    "cross_lane_conflict",
)


@dataclass
class SceneMeta:
    scene_id: str
    month: int
    split: str
    accident_type_raw: str
    accident_type_merged: str
    lane_count: int | None
    accident_lane: str


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("，", ",").replace(" ", "")


def merge_accident_type(raw_type: str) -> str:
    t = normalize_text(raw_type)
    if not t:
        return "未知"
    if "连环追尾" in t or "连续追尾" in t:
        return "连环追尾"
    if "追尾" in t:
        return "追尾"
    if "碰撞" in t:
        return "碰撞"
    if "刮蹭" in t:
        return "刮蹭"
    if "抛锚" in t or "故障" in t or "爆胎" in t or "打滑" in t or "异常停车" in t or "抛洒物" in t:
        return "故障/抛锚/打滑"
    if "护栏" in t:
        return "碰撞/护栏"
    return "其他"


def safe_int(value: object) -> int | None:
    text = normalize_text(value)
    if not text:
        return None
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def read_scene_meta(scene_dir: Path, generated_month_start: int) -> SceneMeta | None:
    txt_path = scene_dir / f"{scene_dir.name}.txt"
    if not txt_path.exists():
        return None
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]
    month = int(scene_dir.name.split("_")[1])
    split = "generated" if month >= generated_month_start else "original"
    raw_type = normalize_text(row.get("事故类型", ""))
    return SceneMeta(
        scene_id=scene_dir.name,
        month=month,
        split=split,
        accident_type_raw=raw_type or "未知",
        accident_type_merged=merge_accident_type(raw_type),
        lane_count=safe_int(row.get("道路车道数", "")),
        accident_lane=normalize_text(row.get("事故车道", "")),
    )


def compute_rmse_xy(xy: np.ndarray, window: int = 11, polyorder: int = 3) -> float:
    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 5:
        return 0.0
    window = min(window if window % 2 == 1 else window + 1, arr.shape[0] if arr.shape[0] % 2 == 1 else arr.shape[0] - 1)
    if window < 5:
        return 0.0
    smoothed = np.empty_like(arr, dtype=float)
    for coord_idx in range(2):
        smoothed[:, coord_idx] = savgol_filter(arr[:, coord_idx], window_length=window, polyorder=min(polyorder, window - 1))
    return float(np.sqrt(np.mean((arr - smoothed) ** 2)))


def lane_bin_from_x(x: np.ndarray, lane_width: float = 3.75, offset: float = 20.0) -> np.ndarray:
    return np.floor((np.asarray(x, dtype=float) + offset) / lane_width)


def extract_signature_from_scene(scene_features: dict[str, object]) -> set[str]:
    signature: set[str] = set()
    if scene_features["lane_change_count"] > 0:
        signature.add("lane_change")
    if scene_features["hard_brake_count"] > 0:
        signature.add("hard_brake")
    if scene_features["rear_end_risk_count"] > 0:
        signature.add("rear_end_risk")
    if scene_features["conflict_pair_count"] >= 2:
        signature.add("multi_vehicle_conflict")
    if scene_features["slow_vehicle_count"] > 0:
        signature.add("slow_obstacle")
    if scene_features["lane_change_count"] > 0 and scene_features["rear_end_risk_count"] > 0:
        signature.add("cross_lane_conflict")
    return signature


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def kl_divergence(p_values: np.ndarray, q_values: np.ndarray, bins: int = 50) -> float:
    if p_values.size == 0 or q_values.size == 0:
        return float("nan")
    value_min = float(min(np.min(p_values), np.min(q_values)))
    value_max = float(max(np.max(p_values), np.max(q_values)))
    if not np.isfinite(value_min) or not np.isfinite(value_max) or value_max <= value_min:
        return 0.0
    p_hist, bin_edges = np.histogram(p_values, bins=bins, range=(value_min, value_max))
    q_hist, _ = np.histogram(q_values, bins=bin_edges)
    p_dist = (p_hist + 1e-10) / (p_hist.sum() + 1e-10 * len(p_hist))
    q_dist = (q_hist + 1e-10) / (q_hist.sum() + 1e-10 * len(q_hist))
    return float(entropy(p_dist, q_dist))


def percentile_text(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, q))


def infer_reference_signature(
    prototypes: dict[str, Counter],
    global_counter: Counter,
    scene_type: str,
    n_original_scenes: int,
    type_scene_counts: dict[str, int],
) -> set[str]:
    counter = prototypes.get(scene_type)
    if not counter or type_scene_counts.get(scene_type, 0) == 0:
        threshold = max(1, math.ceil(n_original_scenes / 2))
        return {name for name, cnt in global_counter.items() if cnt >= threshold}
    threshold = max(1, math.ceil(type_scene_counts[scene_type] / 2))
    return {name for name, cnt in counter.items() if cnt >= threshold}


def analyze_scene(
    scene_dir: Path,
    meta: SceneMeta,
    trajectory_rows: list[dict[str, object]],
    primitive_rows: list[dict[str, object]],
    max_csv_per_scene: int | None = None,
) -> dict[str, object]:
    snapshot_frames: list[pd.DataFrame] = []
    lane_change_count = 0
    hard_brake_count = 0
    slow_vehicle_count = 0
    total_violation_steps = 0
    total_steps = 0

    csv_paths = [p for p in sorted(scene_dir.rglob("*.csv")) if p.name != "事故信息.csv" and "_status" not in p.stem]
    if max_csv_per_scene is not None and len(csv_paths) > max_csv_per_scene:
        selected_indices = np.linspace(0, len(csv_paths) - 1, num=max_csv_per_scene, dtype=int)
        csv_paths = [csv_paths[idx] for idx in selected_indices]

    for csv_path in csv_paths:
        meta_match = FILE_RE.search(csv_path.stem)
        vehicle_id = int(meta_match.group("vid")) if meta_match else -1

        try:
            df = pd.read_csv(csv_path, usecols=["time_stamp", "id", "pos_x", "pos_y", "spd_x", "spd_y"])
        except ValueError:
            continue
        if df.empty:
            continue
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.dropna(subset=["time_stamp", "pos_x", "pos_y", "spd_x", "spd_y"]).sort_values("time_stamp").reset_index(drop=True)
        if df.empty:
            continue

        x = df["pos_x"].to_numpy(dtype=np.float32)
        y = df["pos_y"].to_numpy(dtype=np.float32)
        vx = (df["spd_x"].to_numpy(dtype=np.float32) / 3.6).astype(np.float32)
        vy = (df["spd_y"].to_numpy(dtype=np.float32) / 3.6).astype(np.float32)
        xy = np.column_stack([x, y]).astype(np.float32)
        state = np.column_stack([x, y, vx, vy]).astype(np.float32)

        ts = df["time_stamp"].astype("int64").to_numpy()
        if len(ts) >= 2:
            dt = np.diff(ts) / 1e9
            dt = dt[np.isfinite(dt) & (dt > 1e-4)]
            dt_sec = float(np.median(dt)) if len(dt) else 0.08
        else:
            dt_sec = 0.08

        rmse = compute_rmse_xy(xy)
        dy = np.diff(y) if len(y) >= 2 else np.array([], dtype=np.float32)
        direction_flips = int(np.sum(dy < -0.5))
        lane_bins = lane_bin_from_x(x)
        lane_changes = int(np.sum(np.diff(lane_bins) != 0)) if len(lane_bins) >= 2 else 0
        lane_change_flag = int((abs(x[-1] - x[0]) >= 2.8) or (lane_changes > 0))
        accel_long = np.diff(vy, prepend=vy[0]) / max(dt_sec, 1e-3)
        accel_lat = np.diff(vx, prepend=vx[0]) / max(dt_sec, 1e-3)
        accel_limit_hits = int(np.sum(np.abs(accel_long) > 8.0) + np.sum(np.abs(accel_lat) > 5.0))
        lane_bound_hits = int(np.sum((x < -15.0) | (x > 15.0)))
        violation_steps = direction_flips + accel_limit_hits + lane_bound_hits

        lane_change_count += lane_change_flag
        hard_brake_count += int(np.min(accel_long) < -3.0)
        slow_vehicle_count += int(np.mean(vy) < 8.0 or np.min(vy) < 2.0)
        total_violation_steps += violation_steps
        total_steps += len(df)

        trajectory_rows.append(
            {
                "scene_id": meta.scene_id,
                "split": meta.split,
                "month": meta.month,
                "vehicle_id": int(df["id"].iloc[0]) if "id" in df else vehicle_id,
                "trajectory_path": str(csv_path),
                "n_points": int(len(df)),
                "rmse_m": rmse,
                "mean_speed_mps": float(np.mean(vy)),
                "mean_speed_kmh": float(np.mean(vy) * 3.6),
                "max_speed_kmh": float(np.max(vy) * 3.6),
                "mean_abs_lat_speed_kmh": float(np.mean(np.abs(vx)) * 3.6),
                "lane_change_flag": lane_change_flag,
                "hard_brake_flag": int(np.min(accel_long) < -3.0),
                "violation_steps": violation_steps,
                "accident_type_merged": meta.accident_type_merged,
            }
        )

        for primitive in segment_primitives(state):
            primitive_rows.append(
                {
                    "scene_id": meta.scene_id,
                    "split": meta.split,
                    "primitive": primitive["type"],
                    "mean_velocity_mps": primitive["mean_velocity"],
                    "mean_acceleration_mps2": primitive["mean_acceleration"],
                }
            )

        snapshots = pd.DataFrame(
            {
                "scene_id": meta.scene_id,
                "ts_sec": df["time_stamp"],
                "vehicle_id": int(df["id"].iloc[0]) if "id" in df else vehicle_id,
                "pos_x": x.astype(float),
                "pos_y": y.astype(float),
                "speed_mps": vy.astype(float),
                "lane_bin": lane_bins.astype(float),
            }
        )
        snapshot_frames.append(snapshots)

    scene_features = {
        "scene_id": meta.scene_id,
        "split": meta.split,
        "month": meta.month,
        "accident_type_merged": meta.accident_type_merged,
        "lane_change_count": lane_change_count,
        "hard_brake_count": hard_brake_count,
        "slow_vehicle_count": slow_vehicle_count,
        "rear_end_risk_count": 0,
        "conflict_pair_count": 0,
        "violation_steps": total_violation_steps,
        "total_steps": total_steps,
    }

    if snapshot_frames:
        snapshot_df = pd.concat(snapshot_frames, ignore_index=True)
        snapshot_df = snapshot_df.dropna(subset=["pos_x", "pos_y", "speed_mps", "lane_bin"]).copy()
        if not snapshot_df.empty:
            grouped = snapshot_df.groupby(["ts_sec", "lane_bin"], sort=False)
            conflict_pairs = 0
            rear_end_risk_count = 0
            for (_, _), group in grouped:
                if len(group) < 2:
                    continue
                group = group.sort_values("pos_y").reset_index(drop=True)
                follower = group.iloc[:-1]
                leader = group.iloc[1:]
                for idx in range(len(follower)):
                    gap = float(leader.iloc[idx]["pos_y"] - follower.iloc[idx]["pos_y"])
                    if gap <= 0:
                        continue
                    rel_speed = float(follower.iloc[idx]["speed_mps"] - leader.iloc[idx]["speed_mps"])
                    ttc = np.inf
                    if rel_speed > 0.1:
                        ttc = gap / rel_speed
                    if ttc < 2.0:
                        rear_end_risk_count += 1
                        conflict_pairs += 1
            scene_features["rear_end_risk_count"] = rear_end_risk_count
            scene_features["conflict_pair_count"] = conflict_pairs

    scene_features["signature"] = extract_signature_from_scene(scene_features)
    return scene_features


def analyze_scene_worker(scene_dir_str: str, generated_month_start: int) -> dict[str, object] | None:
    scene_dir = Path(scene_dir_str)
    meta = read_scene_meta(scene_dir, generated_month_start=generated_month_start)
    if meta is None:
        return None
    trajectory_rows: list[dict[str, object]] = []
    primitive_rows: list[dict[str, object]] = []
    scene_features = analyze_scene(scene_dir, meta, trajectory_rows, primitive_rows, max_csv_per_scene=300)
    return {
        "meta": {
            "scene_id": meta.scene_id,
            "split": meta.split,
            "accident_type_merged": meta.accident_type_merged,
        },
        "trajectory_rows": trajectory_rows,
        "primitive_rows": primitive_rows,
        "scene_features": scene_features,
    }


def build_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    result_dir = output_dir / "3生成质量评估"
    docs_dir = output_dir / "docs"
    result_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, docs_dir


def plot_rmse(trajectory_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=trajectory_df, x="split", y="rmse_m", hue="split", palette={"original": "#4C78A8", "generated": "#F58518"}, legend=False)
    plt.xlabel("")
    plt.ylabel("RMSE (m)")
    plt.title("Fig-A1a RMSE Comparison: Original vs Generated")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_proxy_jaccard(scene_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    subset = scene_df[scene_df["split"] == "generated"].copy()
    sns.histplot(subset["proxy_jaccard"], bins=20, color="#F58518", alpha=0.8)
    plt.axvline(0.8, color="black", linestyle="--", linewidth=1.2, label="0.8")
    plt.xlabel("Proxy Jaccard")
    plt.ylabel("Scene Count")
    plt.title("Fig-A1b Proxy Causal Consistency of Generated Scenes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_authenticity(kl_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = kl_df[kl_df["primitive"] != "overall"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    sns.barplot(data=plot_df, x="primitive", y="velocity_kl", hue="primitive", palette="Blues_d", legend=False, ax=axes[0])
    axes[0].axhline(0.1, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Velocity KL")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("KL Divergence")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=plot_df, x="primitive", y="acceleration_kl", hue="primitive", palette="Oranges_d", legend=False, ax=axes[1])
    axes[1].axhline(0.1, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Acceleration KL")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("KL Divergence")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Fig-A1c Authenticity KL Divergence vs Original Reference")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_report(
    output_path: Path,
    summary_df: pd.DataFrame,
    kl_df: pd.DataFrame,
    scene_df: pd.DataFrame,
    generated_month_start: int,
) -> None:
    summary = summary_df.set_index("metric")["value"].to_dict()
    overall = kl_df[kl_df["primitive"] == "overall"].iloc[0].to_dict()
    generated = scene_df[scene_df["split"] == "generated"].copy()
    generated["proxy_jaccard"] = pd.to_numeric(generated["proxy_jaccard"], errors="coerce")
    lines = [
        "# 3 生成质量评估",
        "",
        "## 评估范围与假设",
        "",
        f"- 数据来源：`/home/jcz/sure/2事故轨迹数据`。",
        f"- 分组假设：`2024-06` 到 `2024-08` 视为原始参考数据，`2024-{generated_month_start:02d}` 到 `2024-11` 视为生成数据。",
        "- `analysis_plan.md` 中的 HighD 在本次分析里由你指定的原始轨迹数据替代。",
        "- 计算策略：轨迹层指标按场景做均匀抽样，每个事故场景最多读取 300 条车辆轨迹，以控制全量 12 万级文件的读盘成本；场景层比较仍覆盖全部事故场景。",
        "- 由于当前数据不是 CRADLE 标准 `.pt` 场景格式，因果一致性采用“场景交互代理签名”的 Jaccard 指标替代原论文中的目标 ACG Jaccard。",
        "",
        "## 关键结论",
        "",
        f"- 原始轨迹均值 RMSE 为 `{summary.get('original_mean_rmse_m', float('nan')):.4f} m`，生成轨迹均值 RMSE 为 `{summary.get('generated_mean_rmse_m', float('nan')):.4f} m`。",
        f"- 生成轨迹整体速度 KL 为 `{overall.get('velocity_kl', float('nan')):.4f}`，整体加速度 KL 为 `{overall.get('acceleration_kl', float('nan')):.4f}`。",
        f"- 生成场景代理因果一致性均值为 `{summary.get('generated_proxy_jaccard_mean', float('nan')):.4f}`，`SJ>0.8` 占比为 `{summary.get('generated_proxy_jaccard_gt_08_ratio', float('nan')):.2f}%`。",
        f"- 原始约束合规率为 `{summary.get('original_compliance_rate_pct', float('nan')):.2f}%`，生成约束合规率为 `{summary.get('generated_compliance_rate_pct', float('nan')):.2f}%`。",
        "",
        "## 指标说明",
        "",
        "- 合理性：对每条车辆轨迹的 `(pos_x, pos_y)` 使用 Savitzky-Golay 平滑后计算 RMSE。",
        "- 真实性：按 `following / lane_change / accelerating / decelerating` 四类驾驶原语提取速度与加速度分布，并相对原始数据计算 KL 散度。",
        "- 代理因果一致性：对每个场景抽取 `lane_change / hard_brake / rear_end_risk / multi_vehicle_conflict / slow_obstacle / cross_lane_conflict` 六类交互事件，和同事故类型的原始场景原型做 Jaccard。",
        "- 约束合规率：基于逆向位移、车道越界、纵横向异常加速度三类违规步统计。",
        "",
        "## 生成场景代理因果一致性分位数",
        "",
    ]
    quantiles = generated["proxy_jaccard"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    for q, v in quantiles.items():
        lines.append(f"- P{int(q * 100):02d}: `{v:.4f}`")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_quality_analysis(data_dir: Path, output_dir: Path, generated_month_start: int = 9) -> dict[str, Path]:
    result_dir, docs_dir = build_output_dirs(output_dir)

    trajectory_rows: list[dict[str, object]] = []
    primitive_rows: list[dict[str, object]] = []
    scene_rows: list[dict[str, object]] = []
    original_signature_counter: dict[str, Counter] = defaultdict(Counter)
    original_global_counter: Counter = Counter()
    type_scene_counts: dict[str, int] = defaultdict(int)

    scene_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and SCENE_RE.fullmatch(p.name)])
    max_workers = min(8, max(1, (os.cpu_count() or 2) - 1), max(1, len(scene_dirs)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_scene_worker, str(scene_dir), generated_month_start) for scene_dir in scene_dirs]
        for future in as_completed(futures):
            payload = future.result()
            if payload is None:
                continue
            meta = payload["meta"]
            scene_features = payload["scene_features"]
            trajectory_rows.extend(payload["trajectory_rows"])
            primitive_rows.extend(payload["primitive_rows"])
            if meta["split"] == "original":
                type_scene_counts[meta["accident_type_merged"]] += 1
                for event_name in scene_features["signature"]:
                    original_signature_counter[meta["accident_type_merged"]][event_name] += 1
                    original_global_counter[event_name] += 1
            scene_rows.append(scene_features)

    trajectory_df = pd.DataFrame(trajectory_rows)
    primitive_df = pd.DataFrame(primitive_rows)
    scene_df = pd.DataFrame(scene_rows)

    n_original_scenes = int((scene_df["split"] == "original").sum()) if not scene_df.empty else 0
    proxy_scores: list[float] = []
    for idx, row in scene_df.iterrows():
        if row["split"] != "generated":
            proxy_scores.append(float("nan"))
            continue
        ref_signature = infer_reference_signature(
            prototypes=original_signature_counter,
            global_counter=original_global_counter,
            scene_type=row["accident_type_merged"],
            n_original_scenes=n_original_scenes,
            type_scene_counts=type_scene_counts,
        )
        proxy_scores.append(jaccard_similarity(set(row["signature"]), ref_signature))
    if not scene_df.empty:
        scene_df["proxy_jaccard"] = proxy_scores
        scene_df["signature_text"] = scene_df["signature"].map(lambda x: ",".join(sorted(x)))

    kl_rows: list[dict[str, object]] = []
    for primitive in PRIMITIVES:
        gen_subset = primitive_df[(primitive_df["split"] == "generated") & (primitive_df["primitive"] == primitive)]
        ref_subset = primitive_df[(primitive_df["split"] == "original") & (primitive_df["primitive"] == primitive)]
        kl_rows.append(
            {
                "primitive": primitive,
                "generated_samples": int(len(gen_subset)),
                "reference_samples": int(len(ref_subset)),
                "velocity_kl": kl_divergence(
                    gen_subset["mean_velocity_mps"].to_numpy(dtype=float),
                    ref_subset["mean_velocity_mps"].to_numpy(dtype=float),
                ),
                "acceleration_kl": kl_divergence(
                    gen_subset["mean_acceleration_mps2"].to_numpy(dtype=float),
                    ref_subset["mean_acceleration_mps2"].to_numpy(dtype=float),
                ),
            }
        )
    kl_df = pd.DataFrame(kl_rows)
    kl_df = pd.concat(
        [
            kl_df,
            pd.DataFrame(
                [
                    {
                        "primitive": "overall",
                        "generated_samples": int((primitive_df["split"] == "generated").sum()),
                        "reference_samples": int((primitive_df["split"] == "original").sum()),
                        "velocity_kl": float(kl_df["velocity_kl"].mean()),
                        "acceleration_kl": float(kl_df["acceleration_kl"].mean()),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    summary_rows = [
        {"metric": "original_trajectory_count", "value": int((trajectory_df["split"] == "original").sum())},
        {"metric": "generated_trajectory_count", "value": int((trajectory_df["split"] == "generated").sum())},
        {"metric": "original_scene_count", "value": int((scene_df["split"] == "original").sum())},
        {"metric": "generated_scene_count", "value": int((scene_df["split"] == "generated").sum())},
        {"metric": "original_mean_rmse_m", "value": float(trajectory_df.loc[trajectory_df["split"] == "original", "rmse_m"].mean())},
        {"metric": "generated_mean_rmse_m", "value": float(trajectory_df.loc[trajectory_df["split"] == "generated", "rmse_m"].mean())},
        {
            "metric": "original_compliance_rate_pct",
            "value": float(
                100.0
                * (1.0 - trajectory_df.loc[trajectory_df["split"] == "original", "violation_steps"].sum()
                / max(trajectory_df.loc[trajectory_df["split"] == "original", "n_points"].sum(), 1))
            ),
        },
        {
            "metric": "generated_compliance_rate_pct",
            "value": float(
                100.0
                * (1.0 - trajectory_df.loc[trajectory_df["split"] == "generated", "violation_steps"].sum()
                / max(trajectory_df.loc[trajectory_df["split"] == "generated", "n_points"].sum(), 1))
            ),
        },
        {"metric": "generated_proxy_jaccard_mean", "value": float(scene_df.loc[scene_df["split"] == "generated", "proxy_jaccard"].mean())},
        {
            "metric": "generated_proxy_jaccard_gt_08_ratio",
            "value": float(100.0 * np.mean(scene_df.loc[scene_df["split"] == "generated", "proxy_jaccard"].fillna(0) > 0.8)),
        },
        {"metric": "generated_proxy_jaccard_p25", "value": percentile_text(scene_df.loc[scene_df["split"] == "generated", "proxy_jaccard"], 25)},
        {"metric": "generated_proxy_jaccard_p50", "value": percentile_text(scene_df.loc[scene_df["split"] == "generated", "proxy_jaccard"], 50)},
        {"metric": "generated_proxy_jaccard_p75", "value": percentile_text(scene_df.loc[scene_df["split"] == "generated", "proxy_jaccard"], 75)},
    ]
    summary_df = pd.DataFrame(summary_rows)

    trajectory_df.to_csv(result_dir / "A1_trajectory_metrics.csv", index=False, encoding="utf-8-sig")
    primitive_df.to_csv(result_dir / "A1_primitive_features.csv", index=False, encoding="utf-8-sig")
    scene_df.drop(columns=["signature"]).to_csv(result_dir / "A1_scene_metrics.csv", index=False, encoding="utf-8-sig")
    kl_df.to_csv(result_dir / "A1_authenticity_kl.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(result_dir / "A1_summary_metrics.csv", index=False, encoding="utf-8-sig")

    plot_rmse(trajectory_df, result_dir / "Fig-A1a_rmse_boxplot.png")
    plot_proxy_jaccard(scene_df, result_dir / "Fig-A1b_proxy_jaccard_hist.png")
    plot_authenticity(kl_df, result_dir / "Fig-A1c_authenticity_kl.png")

    report_path = docs_dir / "3生成质量评估.md"
    write_report(report_path, summary_df, kl_df, scene_df, generated_month_start=generated_month_start)

    return {
        "result_dir": result_dir,
        "report_path": report_path,
        "summary_csv": result_dir / "A1_summary_metrics.csv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality assessment for original vs generated accident trajectories.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the trajectory dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to the analysis output root.")
    parser.add_argument("--generated-month-start", type=int, default=9, help="Month index treated as generated-data start.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_quality_analysis(args.data_dir, args.output_dir, generated_month_start=args.generated_month_start)
    print(f"[quality] results: {outputs['result_dir']}")
    print(f"[quality] report: {outputs['report_path']}")
    print(f"[quality] summary: {outputs['summary_csv']}")


if __name__ == "__main__":
    main()
