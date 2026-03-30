"""Trajectory preprocessing utilities for scenario-graph sequence generation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


REQUIRED_HIGH_D_TRACK_COLUMNS = [
    "frame",
    "id",
    "x",
    "y",
    "xVelocity",
    "yVelocity",
    "laneId",
]

REQUIRED_ACCIDENT_TRACK_COLUMNS = [
    "time_stamp",
    "id",
    "pos_x",
    "pos_y",
    "spd_x",
    "spd_y",
]


def build_logistic_weight(d: float, L: float, k: float, d0: float) -> float:
    """Compute the interaction weight defined by the logistic function."""
    return float(L / (1.0 + math.exp(-k * (d - d0))))


def build_scenario_graph(
    frame_df: pd.DataFrame,
    threshold_phi: float = 0.3,
    interaction_range: float = 50.0,
    L: float = 1.0,
    k: float = 0.1,
    d0: float = 25.0,
) -> Data:
    """Convert a single-frame vehicle table into a PyG ``Data`` graph."""
    required_columns = ["trackId", "x", "y", "xVelocity", "yVelocity", "laneId"]
    missing = [column for column in required_columns if column not in frame_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for scenario graph: {missing}")

    ordered = frame_df.loc[:, required_columns].sort_values("trackId").reset_index(drop=True)

    node_features = torch.tensor(
        ordered[["x", "y", "xVelocity", "yVelocity"]].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    lane_ids = torch.tensor(ordered["laneId"].to_numpy(dtype="int64"), dtype=torch.long)

    edge_pairs: list[list[int]] = []
    edge_features: list[list[float]] = []

    positions = ordered[["x", "y"]].to_numpy(dtype="float32")
    velocities = ordered[["xVelocity", "yVelocity"]].to_numpy(dtype="float32")

    num_nodes = len(ordered)
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst:
                continue

            dx = float(positions[dst, 0] - positions[src, 0])
            dy = float(positions[dst, 1] - positions[src, 1])
            distance = math.hypot(dx, dy)
            if distance >= interaction_range:
                continue

            phi = build_logistic_weight(distance, L=L, k=k, d0=d0)
            if phi <= threshold_phi:
                continue

            dvx = float(velocities[dst, 0] - velocities[src, 0])
            dvy = float(velocities[dst, 1] - velocities[src, 1])
            edge_pairs.append([src, dst])
            edge_features.append([dx, dy, dvx, dvy, phi])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 5), dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, lane_ids=lane_ids)


def _select_compact_vehicle_subset(
    frame_df: pd.DataFrame,
    num_vehicles: int,
    num_lanes: int,
    threshold_phi: float,
    interaction_range: float,
    L: float,
    k: float,
    d0: float,
) -> pd.DataFrame | None:
    """Pick a lane-diverse subset that preserves interactions."""
    if len(frame_df) < num_vehicles:
        return None

    ordered = frame_df.sort_values(["x", "trackId"]).reset_index(drop=True)
    best_subset: pd.DataFrame | None = None
    best_edges = -1
    best_span: float | None = None

    for start_idx in range(0, len(ordered) - num_vehicles + 1):
        subset = ordered.iloc[start_idx : start_idx + num_vehicles].copy()
        if subset["laneId"].nunique() < num_lanes:
            continue

        edge_count = _estimate_candidate_edges(
            subset,
            threshold_phi=threshold_phi,
            interaction_range=interaction_range,
            L=L,
            k=k,
            d0=d0,
        )
        span = float(subset["x"].max() - subset["x"].min())
        if edge_count > best_edges or (edge_count == best_edges and (best_span is None or span < best_span)):
            best_subset = subset
            best_edges = edge_count
            best_span = span

    if best_subset is None:
        return None
    return best_subset.sort_values("trackId").reset_index(drop=True)


def _estimate_candidate_edges(
    frame_df: pd.DataFrame,
    threshold_phi: float,
    interaction_range: float,
    L: float,
    k: float,
    d0: float,
) -> int:
    """Estimate how many directed edges a candidate subset would produce."""
    positions = frame_df[["x", "y"]].to_numpy(dtype=float)
    edge_count = 0
    num_nodes = len(frame_df)
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst:
                continue
            distance = math.hypot(
                float(positions[dst, 0] - positions[src, 0]),
                float(positions[dst, 1] - positions[src, 1]),
            )
            if distance >= interaction_range:
                continue
            phi = build_logistic_weight(distance, L=L, k=k, d0=d0)
            if phi > threshold_phi:
                edge_count += 1
    return edge_count


def _load_highd_track_file(tracks_path: Path) -> pd.DataFrame:
    """Load the subset of HighD columns required for SG construction."""
    tracks_df = pd.read_csv(tracks_path, usecols=REQUIRED_HIGH_D_TRACK_COLUMNS)
    tracks_df = tracks_df.rename(columns={"id": "trackId"})
    return tracks_df.sort_values(["frame", "trackId"]).reset_index(drop=True)


def process_highd(
    raw_dir: str,
    output_path: str,
    window_size: int = 50,
    stride: int = 1,
    num_vehicles: int = 4,
    num_lanes: int = 3,
    threshold_phi: float = 0.3,
    interaction_range: float = 50.0,
    L: float = 1.0,
    k: float = 0.1,
    d0: float = 25.0,
) -> None:
    """Process HighD CSV recordings into SG sequence tensors."""
    raw_path = Path(raw_dir)
    tracks_files = sorted(raw_path.glob("*_tracks.csv"))
    if not tracks_files:
        raise FileNotFoundError(
            f"No HighD track files found in '{raw_dir}'. Expected files like '01_tracks.csv'."
        )

    sequences: list[list[Data]] = []
    total_edges = 0
    total_graphs = 0

    for tracks_file in tracks_files:
        recording_id = tracks_file.stem.replace("_tracks", "")
        print(f"Processing HighD recording {recording_id}...")

        tracks_df = _load_highd_track_file(tracks_file)
        per_frame_selection: dict[int, pd.DataFrame] = {}

        for frame, frame_df in tracks_df.groupby("frame", sort=True):
            normalized = _select_compact_vehicle_subset(
                frame_df,
                num_vehicles=num_vehicles,
                num_lanes=num_lanes,
                threshold_phi=threshold_phi,
                interaction_range=interaction_range,
                L=L,
                k=k,
                d0=d0,
            )
            if normalized is not None:
                per_frame_selection[int(frame)] = normalized

        total_edges, total_graphs = _append_sequences_from_frames(
            per_frame_selection=per_frame_selection,
            sequences=sequences,
            total_edges=total_edges,
            total_graphs=total_graphs,
            window_size=window_size,
            stride=stride,
            threshold_phi=threshold_phi,
            interaction_range=interaction_range,
            L=L,
            k=k,
            d0=d0,
        )

    _save_sequences(sequences, output_path, total_edges, total_graphs)


def _append_sequences_from_frames(
    per_frame_selection: dict[int, pd.DataFrame],
    sequences: list[list[Data]],
    total_edges: int,
    total_graphs: int,
    window_size: int,
    stride: int,
    threshold_phi: float,
    interaction_range: float,
    L: float,
    k: float,
    d0: float,
) -> tuple[int, int]:
    """Materialize SG windows from per-frame vehicle selections."""
    frames = sorted(per_frame_selection.keys())
    if len(frames) < window_size:
        return total_edges, total_graphs

    for start_idx in range(0, len(frames) - window_size + 1, stride):
        window_frames = frames[start_idx : start_idx + window_size]
        if any(curr - prev != 1 for prev, curr in zip(window_frames, window_frames[1:])):
            continue

        sequence: list[Data] = []
        for frame in window_frames:
            graph = build_scenario_graph(
                per_frame_selection[frame],
                threshold_phi=threshold_phi,
                interaction_range=interaction_range,
                L=L,
                k=k,
                d0=d0,
            )
            sequence.append(graph)
            total_edges += graph.num_edges
            total_graphs += 1

        sequences.append(sequence)

    return total_edges, total_graphs


def _read_accident_case_meta(case_dir: Path) -> dict[str, object]:
    """Read one accident case's metadata."""
    meta_path = case_dir / f"{case_dir.name}.txt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    meta_df = pd.read_csv(meta_path)
    if meta_df.empty:
        raise ValueError(f"Metadata file is empty: {meta_path}")

    meta_row = meta_df.iloc[0].to_dict()
    meta_row["道路车道数"] = int(meta_row["道路车道数"])
    return meta_row


def _infer_lane_centers(values: np.ndarray, lane_count: int) -> np.ndarray:
    """Infer lane centers from lateral positions with 1D k-means."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("Cannot infer lane centers from empty position array")

    if lane_count <= 1:
        return np.array([float(np.median(data))], dtype=float)

    low = float(np.percentile(data, 2))
    high = float(np.percentile(data, 98))
    if math.isclose(low, high):
        return np.linspace(low, high + 1e-3, lane_count)

    centers = np.quantile(data, np.linspace(0.05, 0.95, lane_count)).astype(float)
    for _ in range(30):
        distances = np.abs(data[:, None] - centers[None, :])
        labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for idx in range(lane_count):
            members = data[labels == idx]
            if members.size:
                new_centers[idx] = float(members.mean())
        new_centers = np.sort(new_centers)
        if np.allclose(new_centers, centers, atol=1e-3):
            break
        centers = new_centers

    return centers


def _load_accident_case_tracks(case_dir: Path, lane_count: int) -> pd.DataFrame | None:
    """Load one accident case and normalize it into frame-wise rows."""
    candidate_files = sorted(case_dir.rglob("*.csv"))
    if not candidate_files:
        return None

    track_files: list[Path] = []
    medians: list[float] = []
    skipped_invalid = 0
    for track_file in candidate_files:
        try:
            track_df = pd.read_csv(track_file, usecols=["pos_x"])
        except Exception:
            skipped_invalid += 1
            continue

        track_files.append(track_file)
        medians.append(float(track_df["pos_x"].median()))

    if skipped_invalid:
        print(f"  [WARN] Skipped {skipped_invalid} invalid CSV files in {case_dir.name}")

    if not track_files:
        return None

    lane_centers = _infer_lane_centers(np.asarray(medians, dtype=float), lane_count=lane_count)
    rows: list[pd.DataFrame] = []

    for track_idx, track_file in enumerate(track_files, start=1):
        try:
            track_df = pd.read_csv(track_file, usecols=REQUIRED_ACCIDENT_TRACK_COLUMNS)
        except Exception:
            continue
        track_df["time_stamp"] = pd.to_datetime(track_df["time_stamp"], format="mixed")
        track_df["trackId"] = track_idx

        lateral = track_df["pos_x"].to_numpy(dtype=float)
        lane_ids = np.abs(lateral[:, None] - lane_centers[None, :]).argmin(axis=1) + 1

        normalized = pd.DataFrame(
            {
                "time_bucket": track_df["time_stamp"].dt.round("100ms"),
                "trackId": track_df["trackId"].to_numpy(dtype=int),
                # Align with the HighD convention: x is longitudinal and y is lateral.
                "x": track_df["pos_y"].to_numpy(dtype=float),
                "y": lateral,
                "xVelocity": track_df["spd_y"].to_numpy(dtype=float),
                "yVelocity": track_df["spd_x"].to_numpy(dtype=float),
                "laneId": lane_ids.astype(int),
            }
        )
        rows.append(normalized)

    all_tracks = pd.concat(rows, ignore_index=True)
    frame_map = {
        timestamp: idx
        for idx, timestamp in enumerate(sorted(all_tracks["time_bucket"].dropna().unique()))
    }
    all_tracks["frame"] = all_tracks["time_bucket"].map(frame_map).astype(int)
    return all_tracks.drop(columns=["time_bucket"]).sort_values(["frame", "trackId"]).reset_index(drop=True)


def process_accident_dataset(
    raw_dir: str,
    output_path: str,
    window_size: int = 50,
    stride: int = 1,
    num_vehicles: int = 4,
    num_lanes: int = 3,
    threshold_phi: float = 0.3,
    interaction_range: float = 50.0,
    L: float = 1.0,
    k: float = 0.1,
    d0: float = 25.0,
) -> None:
    """Process the accident trajectory dataset into SG sequences."""
    raw_path = Path(raw_dir)
    case_dirs = sorted([path for path in raw_path.iterdir() if path.is_dir()])
    if not case_dirs:
        raise FileNotFoundError(f"No accident case directories found in '{raw_dir}'.")

    sequences: list[list[Data]] = []
    total_edges = 0
    total_graphs = 0
    processed_cases = 0
    skipped_cases = 0

    for case_dir in case_dirs:
        meta = _read_accident_case_meta(case_dir)
        lane_count = int(meta["道路车道数"])
        try:
            case_tracks = _load_accident_case_tracks(case_dir, lane_count=lane_count)
        except Exception as exc:
            print(f"Skipping accident case {case_dir.name}: {exc}")
            skipped_cases += 1
            continue
        if case_tracks is None or case_tracks.empty:
            print(f"Skipping accident case {case_dir.name}: no trajectory CSV files")
            skipped_cases += 1
            continue

        print(f"Processing accident case {case_dir.name}...")
        per_frame_selection: dict[int, pd.DataFrame] = {}
        for frame, frame_df in case_tracks.groupby("frame", sort=True):
            normalized = _select_compact_vehicle_subset(
                frame_df,
                num_vehicles=num_vehicles,
                num_lanes=num_lanes,
                threshold_phi=threshold_phi,
                interaction_range=interaction_range,
                L=L,
                k=k,
                d0=d0,
            )
            if normalized is not None:
                per_frame_selection[int(frame)] = normalized

        total_edges, total_graphs = _append_sequences_from_frames(
            per_frame_selection=per_frame_selection,
            sequences=sequences,
            total_edges=total_edges,
            total_graphs=total_graphs,
            window_size=window_size,
            stride=stride,
            threshold_phi=threshold_phi,
            interaction_range=interaction_range,
            L=L,
            k=k,
            d0=d0,
        )
        processed_cases += 1

    print(f"Processed accident cases: {processed_cases}")
    print(f"Skipped accident cases:   {skipped_cases}")
    _save_sequences(sequences, output_path, total_edges, total_graphs)


def detect_dataset_format(raw_dir: str) -> str:
    """Detect which raw trajectory dataset layout is present."""
    raw_path = Path(raw_dir)
    if sorted(raw_path.glob("*_tracks.csv")):
        return "highd"

    if any(path.is_dir() and (path / f"{path.name}.txt").exists() for path in raw_path.iterdir()):
        return "accident"

    raise FileNotFoundError(
        f"Could not detect a supported dataset format in '{raw_dir}'."
    )


def process_dataset(
    raw_dir: str,
    output_path: str,
    window_size: int = 50,
    stride: int = 1,
    num_vehicles: int = 4,
    num_lanes: int = 3,
    threshold_phi: float = 0.3,
    interaction_range: float = 50.0,
    L: float = 1.0,
    k: float = 0.1,
    d0: float = 25.0,
) -> None:
    """Dispatch preprocessing based on the detected raw dataset format."""
    dataset_format = detect_dataset_format(raw_dir)
    print(f"Detected dataset format: {dataset_format}")

    common_kwargs = dict(
        raw_dir=raw_dir,
        output_path=output_path,
        window_size=window_size,
        stride=stride,
        num_vehicles=num_vehicles,
        num_lanes=num_lanes,
        threshold_phi=threshold_phi,
        interaction_range=interaction_range,
        L=L,
        k=k,
        d0=d0,
    )

    if dataset_format == "highd":
        process_highd(**common_kwargs)
        return

    if dataset_format == "accident":
        process_accident_dataset(**common_kwargs)
        return

    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def _save_sequences(
    sequences: list[list[Data]],
    output_path: str,
    total_edges: int,
    total_graphs: int,
) -> None:
    """Persist processed sequences and print summary statistics."""
    average_edges = total_edges / total_graphs if total_graphs else 0.0
    print(f"Total sequences extracted: {len(sequences)}")
    print(f"Average edges per graph:   {average_edges:.2f}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sequences, output)
    print(f"Saved to {output_path}")
