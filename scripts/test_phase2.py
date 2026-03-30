"""Manual tests for Phase 2 preprocessing components."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import pandas as pd
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import SGSequenceDataset
from data.highd_processor import (
    build_logistic_weight,
    build_scenario_graph,
    detect_dataset_format,
    process_accident_dataset,
    process_dataset,
    process_highd,
)


def test_build_logistic_weight() -> None:
    assert abs(build_logistic_weight(25.0, 1.0, 0.1, 25.0) - 0.5) < 1e-6
    assert build_logistic_weight(0.0, 1.0, 0.1, 25.0) < 0.5
    assert build_logistic_weight(50.0, 1.0, 0.1, 25.0) > 0.5
    assert build_logistic_weight(100.0, 1.0, 0.1, 25.0) < 1.0
    assert build_logistic_weight(0.0, 1.0, 0.1, 25.0) > 0.0
    print("[OK] build_logistic_weight")


def test_build_scenario_graph() -> None:
    frame_df = pd.DataFrame(
        {
            "trackId": [1, 2, 3, 4],
            "x": [0.0, 20.0, 40.0, 60.0],
            "y": [0.0, 0.0, 3.5, 3.5],
            "xVelocity": [25.0, 20.0, 25.0, 20.0],
            "yVelocity": [0.0, 0.0, 0.0, 0.0],
            "laneId": [1, 1, 2, 2],
        }
    )
    sg = build_scenario_graph(frame_df, threshold_phi=0.3)

    assert sg.x.shape == (4, 4), f"Unexpected node feature shape: {sg.x.shape}"
    assert sg.edge_index.shape[0] == 2
    assert sg.edge_attr.shape[1] == 5
    assert sg.edge_index.shape[1] == sg.edge_attr.shape[0]
    assert sg.edge_index.max() < 4
    assert sg.edge_index.min() >= 0
    assert sg.num_edges > 0
    print(f"[OK] build_scenario_graph: {sg.num_edges} edges")


def test_sg_sequence_dataset() -> None:
    mock_seq = [
        [
            Data(
                x=torch.randn(4, 4),
                edge_index=torch.zeros(2, 2, dtype=torch.long),
                edge_attr=torch.randn(2, 5),
            )
            for _ in range(50)
        ]
        for _ in range(100)
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_path = Path(tmp_dir) / "mock_sg.pt"
        torch.save(mock_seq, mock_path)

        dataset = SGSequenceDataset(str(mock_path), n_input=10)
        assert len(dataset) == 100 * (50 - 10)
        input_seq, target = dataset[0]
        assert len(input_seq) == 10
        assert isinstance(target, Data)

    print("[OK] SGSequenceDataset")


def test_process_highd_with_mock_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output_path = tmp_path / "processed" / "sg_sequences.pt"

        rows: list[dict[str, float | int]] = []
        for frame in range(1, 6):
            rows.extend(
                [
                    {
                        "frame": frame,
                        "id": 1,
                        "x": 0.0 + frame,
                        "y": 0.0,
                        "xVelocity": 25.0,
                        "yVelocity": 0.0,
                        "laneId": 1,
                    },
                    {
                        "frame": frame,
                        "id": 2,
                        "x": 10.0 + frame,
                        "y": 3.5,
                        "xVelocity": 24.0,
                        "yVelocity": 0.0,
                        "laneId": 2,
                    },
                    {
                        "frame": frame,
                        "id": 3,
                        "x": 20.0 + frame,
                        "y": 7.0,
                        "xVelocity": 23.0,
                        "yVelocity": 0.0,
                        "laneId": 3,
                    },
                    {
                        "frame": frame,
                        "id": 4,
                        "x": 30.0 + frame,
                        "y": 3.5,
                        "xVelocity": 22.0,
                        "yVelocity": 0.0,
                        "laneId": 2,
                    },
                ]
            )

        pd.DataFrame(rows).to_csv(raw_dir / "01_tracks.csv", index=False)

        process_highd(
            raw_dir=str(raw_dir),
            output_path=str(output_path),
            window_size=3,
            stride=1,
            num_vehicles=4,
            num_lanes=3,
            threshold_phi=0.3,
            interaction_range=50.0,
            L=1.0,
            k=0.1,
            d0=25.0,
        )

        sequences = torch.load(output_path, weights_only=False)
        assert isinstance(sequences, list)
        assert len(sequences) == 3
        assert len(sequences[0]) == 3
        assert sequences[0][0].x.shape == (4, 4)
        assert sequences[0][0].edge_attr.shape[1] == 5

    print("[OK] process_highd")


def test_process_highd_empty_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            process_highd(
                raw_dir=tmp_dir,
                output_path=str(Path(tmp_dir) / "out.pt"),
            )
        except FileNotFoundError:
            print("[OK] process_highd empty dir")
            return

    raise AssertionError("Expected FileNotFoundError for empty raw_dir")


def test_process_accident_dataset_with_mock_case() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_dir = tmp_path / "2024_01_01__00_00_00"
        case_dir.mkdir(parents=True)
        (case_dir / "2024_01_01__00_00_00.txt").write_text(
            "时期时间,天气状况,事故类型,严重程度,事故车道,道路等级,道路车道数,道路线性,是否接近匝道\n"
            "2024_01_01__00_00_00,晴,追尾,一般事故,2,高速公路,4,直线路段,否\n",
            encoding="utf-8",
        )

        track_dir = case_dir / "2024-01" / "2024-01-01" / "K000+000" / "10" / "00"
        track_dir.mkdir(parents=True)
        base_times = pd.date_range("2024-01-01 10:00:00", periods=55, freq="100ms")

        lateral_positions = [-1.5, 2.0, 6.0, 10.0]
        longitudinal_starts = [100.0, 112.0, 124.0, 136.0]
        for track_idx in range(4):
            df = pd.DataFrame(
                {
                    "time_stamp": base_times,
                    "id": [track_idx + 1] * len(base_times),
                    "idx_radar": list(range(len(base_times))),
                    "pos_x": [lateral_positions[track_idx]] * len(base_times),
                    "pos_y": [longitudinal_starts[track_idx] + i * 2.5 for i in range(len(base_times))],
                    "spd_x": [0.0] * len(base_times),
                    "spd_y": [25.0 - track_idx] * len(base_times),
                }
            )
            df.to_csv(track_dir / f"track_{track_idx + 1}.csv", index=False)

        output_path = tmp_path / "processed" / "sg_sequences.pt"
        process_accident_dataset(
            raw_dir=str(tmp_path),
            output_path=str(output_path),
            window_size=10,
            stride=5,
            num_vehicles=4,
            num_lanes=3,
        )

        sequences = torch.load(output_path, weights_only=False)
        assert len(sequences) > 0
        assert len(sequences[0]) == 10
        assert sequences[0][0].x.shape == (4, 4)
        assert sequences[0][0].edge_attr.shape[1] == 5

    print("[OK] process_accident_dataset")


def test_dataset_format_detection() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "01_tracks.csv").write_text("frame,id,x,y,xVelocity,yVelocity,laneId\n", encoding="utf-8")
        assert detect_dataset_format(str(tmp_path)) == "highd"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_dir = tmp_path / "2024_01_01__00_00_00"
        case_dir.mkdir()
        (case_dir / "2024_01_01__00_00_00.txt").write_text("时期时间,道路车道数\n2024_01_01__00_00_00,4\n", encoding="utf-8")
        assert detect_dataset_format(str(tmp_path)) == "accident"

    print("[OK] detect_dataset_format")


def test_process_dataset_dispatch() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output_path = tmp_path / "processed" / "sg_sequences.pt"

        rows: list[dict[str, float | int]] = []
        for frame in range(1, 6):
            rows.extend(
                [
                    {"frame": frame, "id": 1, "x": 0.0 + frame, "y": 0.0, "xVelocity": 25.0, "yVelocity": 0.0, "laneId": 1},
                    {"frame": frame, "id": 2, "x": 10.0 + frame, "y": 3.5, "xVelocity": 24.0, "yVelocity": 0.0, "laneId": 2},
                    {"frame": frame, "id": 3, "x": 20.0 + frame, "y": 7.0, "xVelocity": 23.0, "yVelocity": 0.0, "laneId": 3},
                    {"frame": frame, "id": 4, "x": 30.0 + frame, "y": 3.5, "xVelocity": 22.0, "yVelocity": 0.0, "laneId": 2},
                ]
            )
        pd.DataFrame(rows).to_csv(raw_dir / "01_tracks.csv", index=False)

        process_dataset(
            raw_dir=str(raw_dir),
            output_path=str(output_path),
            window_size=3,
            stride=1,
            num_vehicles=4,
            num_lanes=3,
        )
        sequences = torch.load(output_path, weights_only=False)
        assert len(sequences) == 3

    print("[OK] process_dataset")


if __name__ == "__main__":
    test_build_logistic_weight()
    test_build_scenario_graph()
    test_sg_sequence_dataset()
    test_process_highd_with_mock_csv()
    test_process_highd_empty_dir()
    test_process_accident_dataset_with_mock_case()
    test_dataset_format_detection()
    test_process_dataset_dispatch()
    print("[PASS] Phase 2 unit tests passed")
