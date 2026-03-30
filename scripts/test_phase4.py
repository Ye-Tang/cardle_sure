"""Manual validation script for Phase 4 accident causation graph logic."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.acg_builder import (
    AccidentCausationGraph,
    infer_acg,
    jaccard_similarity,
    make_acg_type1,
    make_acg_type2,
)


def make_type1_like_trajectory() -> np.ndarray:
    time_steps = 200
    num_vehicles = 4
    traj = np.zeros((time_steps, num_vehicles, 4), dtype=float)

    traj[0, 0] = [0.0, 0.0, 25.0, 0.0]
    traj[0, 1] = [30.0, 0.0, 20.0, 0.0]
    traj[0, 2] = [5.0, 3.5, 25.0, 0.0]
    traj[0, 3] = [35.0, 3.5, 20.0, 0.0]

    for t in range(1, time_steps):
        traj[t, :, 0] = traj[t - 1, :, 0] + traj[t - 1, :, 2] / 25.0
        traj[t, :, 1] = traj[t - 1, :, 1]
        traj[t, :, 2] = traj[t - 1, :, 2]
        traj[t, :, 3] = traj[t - 1, :, 3]

    for t in range(90, 110):
        traj[t, 0, 1] = traj[t - 1, 0, 1] + 0.15
        traj[t, 2, 1] = traj[t - 1, 2, 1] - 0.15

    traj[150:, 0, :2] = [200.0, 1.5]
    traj[150:, 2, :2] = [200.5, 1.6]
    return traj


if __name__ == "__main__":
    acg = AccidentCausationGraph([1, 2, 3, 4])
    acg.add_causal_edge(2, 1)
    acg.add_causal_edge(1, "C")
    assert (2, 1) in acg.get_edge_set()
    assert (1, "C") in acg.get_edge_set()
    assert len(acg.get_edge_set()) == 2
    print("[OK] AccidentCausationGraph 基本操作")

    try:
        acg.add_causal_edge(1, 2)
        raise AssertionError("应抛出 ValueError")
    except ValueError:
        print("[OK] 双向边检测")

    acg.add_causal_edge(1, 1)
    assert (1, 1) not in acg.get_edge_set(), "自环应被忽略"
    print("[OK] 自环忽略")

    acg1 = make_acg_type1()
    assert acg1.get_edge_set() == {(2, 1), (4, 3), (1, "C"), (3, "C")}
    acg2 = make_acg_type2()
    assert acg2.get_edge_set() == {(2, 1), (3, 1), (1, "C"), (4, "C")}
    print("[OK] 预定义 ACG 类型1和类型2")

    acg_a = make_acg_type1()
    acg_b = make_acg_type1()
    assert abs(jaccard_similarity(acg_a, acg_b) - 1.0) < 1e-6
    print("[OK] Jaccard = 1.0（完全相同）")

    acg_c = AccidentCausationGraph([1, 2, 3, 4])
    acg_c.add_causal_edge(1, 2)
    acg_d = AccidentCausationGraph([1, 2, 3, 4])
    acg_d.add_causal_edge(3, 4)
    assert abs(jaccard_similarity(acg_c, acg_d) - 0.0) < 1e-6
    print("[OK] Jaccard = 0.0（完全不同）")

    acg_e = AccidentCausationGraph([1, 2, 3, 4])
    for edge in ((2, 1), (4, 3), (1, "C")):
        acg_e.add_causal_edge(*edge)
    acg_f = make_acg_type1()
    assert abs(jaccard_similarity(acg_e, acg_f) - 0.75) < 1e-6
    print("[OK] Jaccard = 0.75（部分重叠）")

    acg_empty1 = AccidentCausationGraph([1, 2, 3, 4])
    acg_empty2 = AccidentCausationGraph([1, 2, 3, 4])
    assert jaccard_similarity(acg_empty1, acg_empty2) == 0.0
    print("[OK] Jaccard = 0.0（两个空 ACG）")

    vehicle_ids = [1, 2, 3, 4]
    trajectory = make_type1_like_trajectory()
    acg_gt = make_acg_type1()

    acg_short = infer_acg(trajectory[:40], vehicle_ids)
    sj_short = jaccard_similarity(acg_short, acg_gt)
    print(f"前40帧 Jaccard = {sj_short:.3f}")

    acg_mid = infer_acg(trajectory[:90], vehicle_ids)
    print(f"前90帧推断 ACG: {acg_mid.get_edge_set()}")

    acg_full = infer_acg(trajectory, vehicle_ids)
    sj_full = jaccard_similarity(acg_full, acg_gt)
    print(f"完整序列 Jaccard = {sj_full:.3f}")
    assert sj_full > 0.25, f"完整序列 Jaccard 太低: {sj_full}"
    print("[OK] infer_acg 因果推断测试")
