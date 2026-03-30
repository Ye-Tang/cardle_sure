"""Scenario graph geometry helpers."""

from __future__ import annotations

import numpy as np


def longitudinal_dist(xi: float, xj: float) -> float:
    """Return absolute longitudinal distance along the driving axis."""
    return abs(float(xi) - float(xj))


def lateral_disp(y_t: float, y_t_prev: float) -> float:
    """Return absolute lateral displacement between two timesteps."""
    return abs(float(y_t) - float(y_t_prev))


def euclidean_dist(pos_i: np.ndarray, pos_j: np.ndarray) -> float:
    """Return Euclidean distance between two 2D positions."""
    pos_i = np.asarray(pos_i, dtype=float)
    pos_j = np.asarray(pos_j, dtype=float)
    return float(np.linalg.norm(pos_i - pos_j))
