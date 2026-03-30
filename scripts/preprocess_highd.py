"""Phase 2 HighD preprocessing entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.highd_processor import process_dataset


if __name__ == "__main__":
    with (PROJECT_ROOT / "configs" / "config.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_cfg = cfg["data"]
    process_dataset(
        raw_dir=str(PROJECT_ROOT / data_cfg["raw_dir"]),
        output_path=str(PROJECT_ROOT / data_cfg["processed_path"]),
        window_size=data_cfg["window_size"],
        stride=data_cfg.get("stride", 1),
        num_vehicles=data_cfg["num_vehicles"],
        num_lanes=data_cfg["num_lanes"],
        threshold_phi=data_cfg["phi_threshold"],
        interaction_range=data_cfg["interaction_range"],
        L=data_cfg["logistic_L"],
        k=data_cfg["logistic_k"],
        d0=data_cfg["logistic_d0"],
    )
