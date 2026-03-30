"""Dataset wrappers for scenario-graph sequences."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class SGSequenceDataset(Dataset):
    """Expose SG sequences as ``(input_seq, target_sg)`` samples."""

    def __init__(self, sequences_path: str, n_input: int = 10) -> None:
        if n_input <= 0:
            raise ValueError("n_input must be positive")

        self.sequences_path = Path(sequences_path)
        self.n_input = n_input
        self.sequences: list[list[Data]] = torch.load(self.sequences_path, weights_only=False)
        self.index_map: list[tuple[int, int]] = []

        for sequence_idx, sequence in enumerate(self.sequences):
            for offset in range(max(0, len(sequence) - self.n_input)):
                self.index_map.append((sequence_idx, offset))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[list[Data], Data]:
        sequence_idx, offset = self.index_map[idx]
        sequence = self.sequences[sequence_idx]
        input_seq = sequence[offset : offset + self.n_input]
        target_sg = sequence[offset + self.n_input]
        return input_seq, target_sg
