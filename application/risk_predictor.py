"""Risk prediction models and experiment helpers."""

from __future__ import annotations

import copy
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data

from data.highd_processor import build_scenario_graph
from models.gat_encoder import DualGATEncoder
from models.sg_temporal_predictor import SGTemporalPredictor
from models.transformer_module import SGTransformerEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LSTMRiskPredictor(nn.Module):
    """LSTM-based binary traffic risk predictor."""

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        _, (hidden, _) = self.lstm(x.float())
        risk = torch.sigmoid(self.head(hidden[-1]))
        return risk


class GTFRiskPredictor(nn.Module):
    """GAT + Transformer risk predictor."""

    def __init__(
        self,
        node_feat_dim: int = 4,
        edge_feat_dim: int = 5,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        transformer_d_model: int = 128,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 3,
        num_vehicles: int = 4,
        freeze_encoder: bool = True,
        pretrained_state_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.gat = DualGATEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=gat_hidden,
            heads=gat_heads,
        )
        self.transformer = SGTransformerEncoder(
            input_dim=gat_hidden * num_vehicles,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(transformer_d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if pretrained_state_path is not None and Path(pretrained_state_path).exists():
            self._load_pretrained_encoder(pretrained_state_path)

        if freeze_encoder:
            for parameter in self.gat.parameters():
                parameter.requires_grad = False
            for parameter in self.transformer.parameters():
                parameter.requires_grad = False

    def _load_pretrained_encoder(self, pretrained_state_path: str | Path) -> None:
        state_dict = torch.load(pretrained_state_path, map_location="cpu", weights_only=False)
        predictor = SGTemporalPredictor()
        predictor.load_state_dict(state_dict)
        self.gat.load_state_dict(predictor.gat.state_dict())
        self.transformer.load_state_dict(predictor.transformer_enc.state_dict())

    def forward(self, sg_seq: list[Data]) -> Tensor:
        node_embeds = [self.gat(graph) for graph in sg_seq]
        encoded = self.transformer(torch.stack(node_embeds, dim=0))
        pooled = encoded.mean(dim=0)
        risk = torch.sigmoid(self.classifier(pooled)).reshape(1)
        return risk


def _min_pairwise_distance(frame: np.ndarray) -> float:
    min_dist = float("inf")
    for i in range(frame.shape[0]):
        for j in range(i + 1, frame.shape[0]):
            dist = float(np.linalg.norm(frame[i, :2] - frame[j, :2]))
            min_dist = min(min_dist, dist)
    return min_dist


def _trajectory_has_collision(traj: np.ndarray, threshold: float = 3.0) -> bool:
    return any(_min_pairwise_distance(frame) <= threshold for frame in traj)


def _sequence_to_array(sequence: list[Data]) -> np.ndarray:
    return np.stack([graph.x.detach().cpu().float().numpy() for graph in sequence], axis=0)


def _trajectory_to_lstm_feature(traj: np.ndarray, n_input: int) -> np.ndarray:
    return traj[:n_input].reshape(n_input, -1).astype(np.float32)


def _infer_lane_ids(y_values: np.ndarray) -> np.ndarray:
    rounded = np.round(y_values.astype(float), 1)
    unique = np.unique(rounded)
    mapping = {value: idx + 1 for idx, value in enumerate(np.sort(unique))}
    return np.asarray([mapping[value] for value in rounded], dtype=np.int64)


def _frame_to_graph(frame: np.ndarray, track_ids: np.ndarray) -> Data:
    frame_df = pd.DataFrame(
        {
            "trackId": track_ids,
            "x": frame[:, 0],
            "y": frame[:, 1],
            "xVelocity": frame[:, 2],
            "yVelocity": frame[:, 3],
            "laneId": _infer_lane_ids(frame[:, 1]),
        }
    )
    return build_scenario_graph(frame_df)


def _trajectory_to_graph_sequence(traj: np.ndarray, n_input: int) -> list[Data]:
    track_ids = np.arange(1, traj.shape[1] + 1)
    return [_frame_to_graph(traj[t], track_ids) for t in range(n_input)]


def prepare_datasets(
    generated_data_path: str,
    highd_sequences_path: str,
    n_input: int = 10,
) -> dict[str, tuple[list, list[int]]]:
    """Prepare Set1/Set2/Set3/val splits for risk prediction."""
    highd_sequences: list[list[Data]] = torch.load(highd_sequences_path, weights_only=False)
    non_collision: list[np.ndarray] = []
    for sequence in highd_sequences:
        arr = _sequence_to_array(sequence)
        if arr.shape[0] < n_input:
            continue
        if not _trajectory_has_collision(arr):
            non_collision.append(arr)

    generated_items = []
    generated_path = Path(generated_data_path)
    if generated_path.exists():
        generated_items = torch.load(generated_path, weights_only=False)
    collision = [np.asarray(item["trajectory"], dtype=np.float32) for item in generated_items if "trajectory" in item]

    if len(non_collision) < 1000:
        warnings.warn(f"Only found {len(non_collision)} non-collision sequences; using all available.")
    if len(collision) < 200:
        warnings.warn(f"Only found {len(collision)} generated collision sequences; results may be degenerate.")

    rng = random.Random(42)
    rng.shuffle(non_collision)
    rng.shuffle(collision)

    set1_non_collision = non_collision[: min(1000, len(non_collision))]
    val_collision = collision[-200:] if len(collision) >= 200 else collision[:]
    train_collision_pool = collision[:-200] if len(collision) > 200 else collision[:]

    set2_non_collision = set1_non_collision[: min(500, len(set1_non_collision))]
    set2_collision = train_collision_pool[: min(500, len(train_collision_pool))]
    set3_collision = train_collision_pool[: min(1000, len(train_collision_pool))]

    return {
        "set1": (set1_non_collision, [0] * len(set1_non_collision)),
        "set2": (set2_non_collision + set2_collision, [0] * len(set2_non_collision) + [1] * len(set2_collision)),
        "set3": (set3_collision, [1] * len(set3_collision)),
        "val": (val_collision, [1] * len(val_collision)),
    }


def _train_lstm_model(
    train_data: list[np.ndarray],
    train_labels: list[int],
    epochs: int = 50,
    patience: int = 10,
    device: str = "cpu",
) -> LSTMRiskPredictor:
    model = LSTMRiskPredictor().to(device)
    if not train_data:
        return model

    x_tensor = torch.tensor(np.stack(train_data, axis=0), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(np.asarray(train_labels, dtype=np.float32).reshape(-1, 1), device=device)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=min(32, len(train_data)), shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    stale = 0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = F.binary_cross_entropy(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_x)
        mean_loss = epoch_loss / len(train_data)
        if mean_loss < best_loss - 1e-6:
            best_loss = mean_loss
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _train_gtf_model(
    train_data: list[list[Data]],
    train_labels: list[int],
    cfg: dict,
    epochs: int = 50,
    patience: int = 10,
    device: str = "cpu",
) -> GTFRiskPredictor:
    pretrained_path = PROJECT_ROOT / "checkpoints" / "vgae_best.pt"
    model = GTFRiskPredictor(
        node_feat_dim=int(cfg["model"]["node_feat_dim"]),
        edge_feat_dim=int(cfg["model"]["edge_feat_dim"]),
        gat_hidden=int(cfg["model"]["gat_hidden"]),
        gat_heads=int(cfg["model"]["gat_heads"]),
        transformer_d_model=int(cfg["model"]["transformer_d_model"]),
        transformer_nhead=int(cfg["model"]["transformer_nhead"]),
        transformer_num_layers=int(cfg["model"]["transformer_num_layers"]),
        num_vehicles=int(cfg["data"]["num_vehicles"]),
        freeze_encoder=True,
        pretrained_state_path=pretrained_path,
    ).to(device)
    if not train_data:
        return model

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    best_loss = float("inf")
    stale = 0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        losses: list[float] = []
        order = list(range(len(train_data)))
        random.shuffle(order)
        for idx in order:
            optimizer.zero_grad()
            seq = [graph.to(device) for graph in train_data[idx]]
            label = torch.tensor(float(train_labels[idx]), dtype=torch.float32, device=device)
            pred = model(seq).reshape(())
            loss = F.binary_cross_entropy(pred, label)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss - 1e-6:
            best_loss = mean_loss
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def compute_collision_rate(
    model: nn.Module,
    val_data: list,
    threshold: float = 0.5,
    model_type: str = "lstm",
) -> float:
    """Compute missed-warning rate over all-positive validation data."""
    if not val_data:
        return float("nan")

    model.eval()
    misses = 0
    total = 0
    with torch.no_grad():
        if model_type == "lstm":
            device = next(model.parameters()).device
            x_tensor = torch.tensor(np.stack(val_data, axis=0), dtype=torch.float32, device=device)
            pred = model(x_tensor).detach().cpu().numpy().reshape(-1)
            misses = int(np.sum(pred < threshold))
            total = len(pred)
        else:
            device = next(model.parameters()).device
            for seq in val_data:
                risk = float(model([graph.to(device) for graph in seq]).item())
                misses += int(risk < threshold)
                total += 1
    return misses / total if total > 0 else float("nan")


def run_experiment(acg_type: int, cfg: dict) -> dict[str, dict[str, float]]:
    """Run the full risk-prediction experiment for one accident type."""
    generated_path = PROJECT_ROOT / "data" / "generated" / f"type{acg_type}_1000.pt"
    highd_path = PROJECT_ROOT / cfg["data"]["processed_path"]
    datasets = prepare_datasets(
        generated_data_path=str(generated_path),
        highd_sequences_path=str(highd_path),
        n_input=int(cfg["rl"]["n_steps_input"]),
    )

    n_input = int(cfg["rl"]["n_steps_input"])
    set_features_lstm = {
        key: ([_trajectory_to_lstm_feature(traj, n_input) for traj in data], labels)
        for key, (data, labels) in datasets.items()
    }
    set_features_gtf = {
        key: ([_trajectory_to_graph_sequence(traj, n_input) for traj in data], labels)
        for key, (data, labels) in datasets.items()
    }

    results: dict[str, dict[str, float]] = {"LSTM": {}, "GTF": {}}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for set_name in ("set1", "set2", "set3"):
        x_train_lstm, y_train_lstm = set_features_lstm[set_name]
        lstm_model = _train_lstm_model(x_train_lstm, y_train_lstm, device=device)
        results["LSTM"][set_name] = compute_collision_rate(
            lstm_model,
            set_features_lstm["val"][0],
            model_type="lstm",
        )

        x_train_gtf, y_train_gtf = set_features_gtf[set_name]
        gtf_model = _train_gtf_model(x_train_gtf, y_train_gtf, cfg=cfg, device=device)
        results["GTF"][set_name] = compute_collision_rate(
            gtf_model,
            set_features_gtf["val"][0],
            model_type="gtf",
        )

    print(f"=== 事故类型 {acg_type} 碰撞率结果 ===")
    print("Method   | Set1   | Set2   | Set3")
    print("---------|--------|--------|------")
    for method in ("LSTM", "GTF"):
        print(
            f"{method:<8} | "
            f"{results[method]['set1']:.2f}   | "
            f"{results[method]['set2']:.2f}   | "
            f"{results[method]['set3']:.2f}"
        )
    return results


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else PROJECT_ROOT / "configs" / "config.yaml"
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)
