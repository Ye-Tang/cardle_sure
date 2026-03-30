"""Accident causation graph structures and inference rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np

from knowledge_graph.scenario_graph import euclidean_dist, lateral_disp, longitudinal_dist

CENTER_NODE = "C"
NodeId = Union[int, str]
Edge = tuple[NodeId, NodeId]


@dataclass
class AccidentCausationGraph:
    """Directed accident-causation graph with a dedicated collision node."""

    vehicle_ids: list[int]
    edges: set[Edge] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.vehicle_ids = list(self.vehicle_ids)
        self.nodes = [*self.vehicle_ids, CENTER_NODE]

    def add_causal_edge(self, src: NodeId, dst: NodeId) -> None:
        if src == dst:
            return
        if (dst, src) in self.edges:
            raise ValueError(f"bidirectional causal edge is not allowed: {(dst, src)} already exists")
        self.edges.add((src, dst))

    def get_edge_set(self) -> set[Edge]:
        return set(self.edges)

    def get_vehicle_ids(self) -> list[int]:
        return list(self.vehicle_ids)

    def __repr__(self) -> str:
        node_repr = ",".join(str(node) for node in self.nodes)
        edge_repr = ",".join(str(edge) for edge in sorted(self.edges, key=str))
        return f"ACG(nodes=[{node_repr}], edges={{{edge_repr}}})"


def make_acg_type1() -> AccidentCausationGraph:
    acg = AccidentCausationGraph([1, 2, 3, 4])
    for src, dst in ((2, 1), (4, 3), (1, CENTER_NODE), (3, CENTER_NODE)):
        acg.add_causal_edge(src, dst)
    return acg


def make_acg_type2() -> AccidentCausationGraph:
    acg = AccidentCausationGraph([1, 2, 3, 4])
    for src, dst in ((2, 1), (3, 1), (1, CENTER_NODE), (4, CENTER_NODE)):
        acg.add_causal_edge(src, dst)
    return acg


def infer_acg(
    trajectory: np.ndarray,
    vehicle_ids: list[int],
    delta_x: float = 40.0,
    delta_y1: float = 0.5,
    delta_y2: float = 0.8,
    delta_collision: float = 1.5,
    min_interval: int = 25,
) -> AccidentCausationGraph:
    """Infer an ACG from a multi-vehicle trajectory tensor."""
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 3 or traj.shape[-1] < 4:
        raise ValueError("trajectory must have shape [T, num_vehicles, 4+]")
    if traj.shape[1] != len(vehicle_ids):
        raise ValueError("vehicle_ids length must match trajectory vehicle dimension")

    time_steps, num_vehicles, _ = traj.shape
    acg = AccidentCausationGraph(vehicle_ids)

    for t in range(time_steps):
        for i in range(num_vehicles):
            for j in range(i + 1, num_vehicles):
                if euclidean_dist(traj[t, i, :2], traj[t, j, :2]) < delta_collision:
                    acg.add_causal_edge(vehicle_ids[i], CENTER_NODE)
                    acg.add_causal_edge(vehicle_ids[j], CENTER_NODE)

    if time_steps <= min_interval:
        return acg

    for t in range(min_interval, time_steps - 1):
        for leading_idx in range(num_vehicles):
            for following_idx in range(num_vehicles):
                if leading_idx == following_idx:
                    continue

                x_leading = traj[t, leading_idx, 0]
                x_following = traj[t, following_idx, 0]
                if x_leading <= x_following:
                    continue

                vx_leading = traj[t, leading_idx, 2]
                vx_following = traj[t, following_idx, 2]
                y_following = traj[t, following_idx, 1]
                y_following_prev = traj[t - min_interval, following_idx, 1]

                cond_a = vx_following > vx_leading
                cond_b = longitudinal_dist(x_following, x_leading) < delta_x
                cond_c = lateral_disp(y_following, y_following_prev) < delta_y1
                cond_d = np.any(
                    np.abs(traj[t + 1 :, following_idx, 1] - y_following) > delta_y2
                )

                if cond_a and cond_b and cond_c and cond_d:
                    acg.add_causal_edge(vehicle_ids[leading_idx], vehicle_ids[following_idx])

    return acg


def jaccard_similarity(acg0: AccidentCausationGraph, acg_gt: AccidentCausationGraph) -> float:
    """Return Jaccard similarity between inferred and ground-truth ACGs."""
    edge_set_0 = acg0.get_edge_set()
    edge_set_gt = acg_gt.get_edge_set()
    union = edge_set_0 | edge_set_gt
    if not union:
        return 0.0
    return len(edge_set_0 & edge_set_gt) / len(union)
