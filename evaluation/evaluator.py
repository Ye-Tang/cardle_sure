"""Scenario database evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import entropy


def diversity_score(
    trajectories: list[np.ndarray],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    nx: int = 200,
    ny: int = 10,
) -> float:
    """Compute occupancy-grid diversity percentage."""
    if not trajectories or nx <= 0 or ny <= 0:
        return 0.0

    x_min, x_max = x_range
    y_min, y_max = y_range
    if x_max <= x_min or y_max <= y_min:
        return 0.0

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    if dx <= 0 or dy <= 0:
        return 0.0

    occupied = np.zeros((nx, ny), dtype=bool)
    for traj in trajectories:
        coords = np.asarray(traj, dtype=float)
        if coords.ndim != 3 or coords.shape[-1] < 2:
            continue
        points = coords.reshape(-1, coords.shape[-1])
        for x, y in points[:, :2]:
            if x < x_min or x > x_max or y < y_min or y > y_max:
                continue
            ix = min(nx - 1, max(0, int(np.floor((x - x_min) / dx))))
            iy = min(ny - 1, max(0, int(np.floor((y - y_min) / dy))))
            occupied[ix, iy] = True

    return float(occupied.sum() / (nx * ny) * 100.0)


def rationality_scores(
    trajectories: list[np.ndarray],
    savgol_window: int = 11,
    savgol_polyorder: int = 3,
) -> tuple[np.ndarray, float]:
    """Compute per-scenario rationality scores and mean RMSE."""
    if not trajectories:
        return np.zeros(0, dtype=float), float("nan")

    rmses: list[float] = []
    for traj in trajectories:
        arr = np.asarray(traj, dtype=float)
        if arr.ndim != 3 or arr.shape[-1] < 2:
            continue
        if arr.shape[0] <= savgol_window:
            rmses.append(0.0)
            continue

        window = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        window = min(window, arr.shape[0] if arr.shape[0] % 2 == 1 else arr.shape[0] - 1)
        if window < 3:
            rmses.append(0.0)
            continue

        smoothed = np.empty_like(arr[:, :, :2], dtype=float)
        for vehicle_idx in range(arr.shape[1]):
            for coord_idx in range(2):
                smoothed[:, vehicle_idx, coord_idx] = savgol_filter(
                    arr[:, vehicle_idx, coord_idx],
                    window_length=window,
                    polyorder=min(savgol_polyorder, window - 1),
                )
        rmse = float(np.sqrt(np.mean((arr[:, :, :2] - smoothed) ** 2)))
        rmses.append(rmse)

    if not rmses:
        return np.zeros(0, dtype=float), float("nan")

    rmse_array = np.asarray(rmses, dtype=float)
    rmse_min = float(rmse_array.min())
    rmse_max = float(rmse_array.max())
    if rmse_max - rmse_min < 1e-8:
        scores = np.ones_like(rmse_array)
    else:
        normalized_error = (rmse_array - rmse_min) / (rmse_max - rmse_min + 1e-8)
        scores = 1.0 - normalized_error
    return scores, float(rmse_array.mean())


def segment_primitives(
    trajectory: np.ndarray,
    lat_threshold: float = 0.8,
    accel_threshold: float = 1.0,
) -> list[dict[str, Any]]:
    """Segment one vehicle trajectory into coarse scenario primitives."""
    arr = np.asarray(trajectory, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return []

    vx = arr[:, 2]
    accel = np.diff(vx, prepend=vx[0]) * 25.0
    y = arr[:, 1]

    def classify(start: int, end: int) -> str:
        lat_disp = abs(y[end - 1] - y[start])
        mean_accel = float(accel[start:end].mean())
        if lat_disp > lat_threshold:
            return "lane_change"
        if mean_accel > accel_threshold:
            return "accelerating"
        if mean_accel < -accel_threshold:
            return "decelerating"
        return "following"

    primitives: list[dict[str, Any]] = []
    start = 0
    current_type = classify(0, min(arr.shape[0], 2))

    for t in range(1, arr.shape[0]):
        step_type = classify(max(0, t - 1), t + 1)
        if step_type != current_type:
            segment = arr[start:t]
            primitives.append(
                {
                    "type": current_type,
                    "start": start,
                    "end": t,
                    "mean_velocity": float(segment[:, 2].mean()),
                    "mean_acceleration": float(accel[start:t].mean()),
                }
            )
            start = t
            current_type = step_type

    segment = arr[start:]
    primitives.append(
        {
            "type": current_type,
            "start": start,
            "end": arr.shape[0],
            "mean_velocity": float(segment[:, 2].mean()),
            "mean_acceleration": float(accel[start:].mean()),
        }
    )
    return primitives


def authenticity_score(
    generated_trajs: list[np.ndarray],
    highd_trajs: list[np.ndarray],
    num_bins: int = 50,
) -> dict[str, float]:
    """Compare primitive-level speed and acceleration distributions via KL."""

    def extract_features(trajs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        velocities: list[float] = []
        accelerations: list[float] = []
        for traj in trajs:
            arr = np.asarray(traj, dtype=float)
            if arr.ndim != 3 or arr.shape[-1] < 4:
                continue
            for vehicle_idx in range(arr.shape[1]):
                for primitive in segment_primitives(arr[:, vehicle_idx, :4]):
                    velocities.append(float(primitive["mean_velocity"]))
                    accelerations.append(float(primitive["mean_acceleration"]))
        return np.asarray(velocities, dtype=float), np.asarray(accelerations, dtype=float)

    def kl_divergence(p_values: np.ndarray, q_values: np.ndarray) -> float:
        if p_values.size == 0 or q_values.size == 0:
            return float("nan")
        value_min = float(min(p_values.min(), q_values.min()))
        value_max = float(max(p_values.max(), q_values.max()))
        if value_max <= value_min:
            return 0.0
        p_hist, bins = np.histogram(p_values, bins=num_bins, range=(value_min, value_max), density=False)
        q_hist, _ = np.histogram(q_values, bins=bins, density=False)
        p_dist = (p_hist + 1e-10) / (p_hist.sum() + 1e-10 * len(p_hist))
        q_dist = (q_hist + 1e-10) / (q_hist.sum() + 1e-10 * len(q_hist))
        return float(entropy(p_dist, q_dist))

    gen_vel, gen_acc = extract_features(generated_trajs)
    highd_vel, highd_acc = extract_features(highd_trajs)
    return {
        "velocity_kl": kl_divergence(gen_vel, highd_vel),
        "acceleration_kl": kl_divergence(gen_acc, highd_acc),
    }
