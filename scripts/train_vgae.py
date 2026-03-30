"""VGAE training entrypoint."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import SGSequenceDataset
from models.sg_temporal_predictor import SGTemporalPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Use a small subset for quick validation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collate_samples(batch: list[tuple[list[torch.Tensor], torch.Tensor]]) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
    return batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_graph(graph, device: torch.device):
    return graph.to(device)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_samples)


def compute_feature_stats(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_sum = None
    node_sq_sum = None
    edge_sum = None
    edge_sq_sum = None
    node_count = 0
    edge_count = 0

    for input_seq, target_sg in dataset:
        for graph in [*input_seq, target_sg]:
            x = graph.x.float()
            node_sum = x.sum(dim=0) if node_sum is None else node_sum + x.sum(dim=0)
            node_sq_sum = (x * x).sum(dim=0) if node_sq_sum is None else node_sq_sum + (x * x).sum(dim=0)
            node_count += x.size(0)

            edge_attr = graph.edge_attr.float()
            if edge_attr.numel() == 0:
                continue
            edge_sum = edge_attr.sum(dim=0) if edge_sum is None else edge_sum + edge_attr.sum(dim=0)
            edge_sq_sum = (
                (edge_attr * edge_attr).sum(dim=0)
                if edge_sq_sum is None
                else edge_sq_sum + (edge_attr * edge_attr).sum(dim=0)
            )
            edge_count += edge_attr.size(0)

    if node_sum is None or node_sq_sum is None or node_count == 0:
        raise RuntimeError("failed to compute node feature statistics from dataset")

    node_mean = node_sum / node_count
    node_var = (node_sq_sum / node_count) - node_mean.pow(2)
    node_std = node_var.clamp_min(1e-6).sqrt()

    if edge_sum is None or edge_sq_sum is None or edge_count == 0:
        edge_mean = torch.zeros(5, dtype=torch.float32)
        edge_std = torch.ones(5, dtype=torch.float32)
    else:
        edge_mean = edge_sum / edge_count
        edge_var = (edge_sq_sum / edge_count) - edge_mean.pow(2)
        edge_std = edge_var.clamp_min(1e-6).sqrt()

    return node_mean, node_std, edge_mean, edge_std


def run_epoch(
    model: SGTemporalPredictor,
    loader: DataLoader,
    optimizer: Adam | None,
    device: torch.device,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        if training:
            optimizer.zero_grad()

        batch_loss = torch.tensor(0.0, device=device)
        for input_seq, target_sg in batch:
            seq_on_device = [move_graph(graph, device) for graph in input_seq]
            target_on_device = move_graph(target_sg, device)
            outputs = model(seq_on_device)
            loss = model.compute_loss(outputs, target_on_device, alpha=alpha, beta=beta, gamma=gamma)
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / max(1, len(batch))
        if training:
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += float(batch_loss.item()) * len(batch)
        total_samples += len(batch)

    return total_loss / max(1, total_samples)


def save_sampling_plot(
    model: SGTemporalPredictor,
    dataset: SGSequenceDataset,
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()
    input_seq, target = dataset[0]
    seq_on_device = [move_graph(graph, device) for graph in input_seq]
    with torch.no_grad():
        samples = model.sample(seq_on_device, k=50)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for vehicle_idx in range(4):
        ax = axes[vehicle_idx]
        xs = [sample.x[vehicle_idx, 0].item() for sample in samples]
        ys = [sample.x[vehicle_idx, 1].item() for sample in samples]
        ax.scatter(xs, ys, alpha=0.5, s=10, label="samples")
        ax.scatter(
            [target.x[vehicle_idx, 0].item()],
            [target.x[vehicle_idx, 1].item()],
            marker="*",
            s=100,
            c="red",
            label="true",
        )
        ax.set_title(f"Vehicle {vehicle_idx + 1}")
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with (PROJECT_ROOT / "configs" / "config.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset = SGSequenceDataset(
        sequences_path=str(PROJECT_ROOT / cfg["data"]["processed_path"]),
        n_input=cfg["rl"]["n_steps_input"],
    )
    if args.debug:
        debug_size = min(500, len(dataset))
        dataset = Subset(dataset, list(range(debug_size)))

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = make_loader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_set, batch_size=batch_size, shuffle=False)

    model = SGTemporalPredictor(
        node_feat_dim=cfg["model"]["node_feat_dim"],
        edge_feat_dim=cfg["model"]["edge_feat_dim"],
        gat_hidden=cfg["model"]["gat_hidden"],
        gat_heads=cfg["model"]["gat_heads"],
        transformer_d_model=cfg["model"]["transformer_d_model"],
        transformer_nhead=cfg["model"]["transformer_nhead"],
        transformer_num_layers=cfg["model"]["transformer_num_layers"],
        latent_dim=cfg["model"]["latent_dim"],
        num_vehicles=cfg["data"]["num_vehicles"],
    ).to(device)

    node_mean, node_std, edge_mean, edge_std = compute_feature_stats(train_set)
    model.set_normalization_stats(node_mean, node_std, edge_mean, edge_std)
    print(
        "Feature stats:"
        f" node_mean={node_mean.tolist()}"
        f" node_std={node_std.tolist()}"
        f" edge_mean={edge_mean.tolist()}"
        f" edge_std={edge_std.tolist()}"
    )

    optimizer = Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=1e-5)
    alpha = cfg["training"]["vgae_loss_alpha"]
    beta = cfg["training"]["vgae_loss_beta"]
    gamma = cfg["training"]["vgae_loss_gamma"]

    epochs = args.epochs if args.epochs is not None else (15 if args.debug else 100)
    patience = 10 if args.debug else 20
    best_val = float("inf")
    stale_epochs = 0

    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / "vgae_best.pt"
    last_path = checkpoints_dir / "vgae_last.pt"

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, alpha, beta, gamma)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, device, alpha, beta, gamma)

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        torch.save(model.state_dict(), last_path)
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best val_loss: {best_val:.4f}")
    print(f"Saved to {best_path}")
    print(f"Saved to {last_path}")

    if len(dataset) > 0:
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
        save_sampling_plot(
            model=model,
            dataset=dataset.dataset if isinstance(dataset, Subset) else dataset,
            device=device,
            output_path=checkpoints_dir / "fig7_sampling_space.png",
        )
        print(f"Saved to {checkpoints_dir / 'fig7_sampling_space.png'}")


if __name__ == "__main__":
    main()
