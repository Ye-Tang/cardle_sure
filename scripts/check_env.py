"""Environment validation script for Phase 1."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_imports() -> list[str]:
    failures: list[str] = []
    libs = [
        ("torch", "torch"),
        ("torch_geometric", "torch_geometric"),
        ("stable_baselines3", "stable_baselines3"),
        ("gymnasium", "gymnasium"),
        ("networkx", "networkx"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
    ]
    for module_name, package_name in libs:
        try:
            __import__(module_name)
            print(f"  [OK] {package_name}")
        except ImportError:
            print(f"  [FAIL] {package_name}")
            failures.append(package_name)
    return failures


def check_cuda() -> None:
    try:
        import torch
    except ImportError:
        print("  [SKIP] torch unavailable, skipping CUDA check")
        return

    if torch.cuda.is_available():
        print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  [WARN] CUDA not available, will use CPU")


def check_pyg() -> None:
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        print("  [SKIP] torch or torch_geometric unavailable, skipping PyG check")
        return

    x = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn(2, 5)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    assert data.num_nodes == 4
    assert data.num_edges == 2
    print("  [OK] PyTorch Geometric Data object creation")


def check_config() -> None:
    import yaml

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required_sections = ["data", "model", "training", "rl", "causal"]
    for section in required_sections:
        assert section in config, f"Missing section: {section}"
    print("  [OK] configs/config.yaml loaded with all required sections")


if __name__ == "__main__":
    print("=== Phase 1 环境验证 ===")
    print("[1] 检查库导入...")
    failures = check_imports()
    print("[2] 检查 CUDA...")
    check_cuda()
    print("[3] 检查 PyTorch Geometric...")
    check_pyg()
    print("[4] 检查 config.yaml...")
    check_config()
    if failures:
        print(f"\n[FAIL] 以下库未安装: {failures}")
        sys.exit(1)

    print("\n[PASS] Phase 1 环境验证通过")
