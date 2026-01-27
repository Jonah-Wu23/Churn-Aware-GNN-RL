from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        pytest.skip(f"Parquet writer unavailable: {exc}")


def _load_script(path: str):
    script_path = Path(__file__).resolve().parents[1] / path
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_cpo_train_import_and_env_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "baselines" / "PyTorch-CPO").exists():
        pytest.skip("baselines/PyTorch-CPO not present")
    monkeypatch.chdir(repo_root)

    nodes = pd.DataFrame({"gnn_node_id": [1, 2], "emb_geo_0": [0.1, 0.2]})
    edges = pd.DataFrame({"source": [1, 2], "target": [2, 1], "travel_time_sec": [10.0, 10.0]})
    od = pd.DataFrame(
        {
            "pickup_stop_id": [1, 2],
            "dropoff_stop_id": [2, 1],
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:05:00"]),
        }
    )
    nodes_path = tmp_path / "nodes.parquet"
    edges_path = tmp_path / "edges.parquet"
    od_path = tmp_path / "od.parquet"
    _write_parquet(nodes, nodes_path)
    _write_parquet(edges, edges_path)
    _write_parquet(od, od_path)

    cfg = {
        "env": {
            "max_horizon_steps": 5,
            "max_requests": 2,
            "num_vehicles": 2,
            "vehicle_capacity": 6,
            "mask_alpha": 1.5,
            "od_glob": str(od_path),
            "graph_nodes_path": str(nodes_path),
            "graph_edges_path": str(edges_path),
            "time_split_mode": None,
        },
        "cpo_train": {"num_vehicles": 2},
    }

    module = _load_script("scripts/run_cpo_train.py")
    env_config = module.build_env_config(cfg, seed=7)
    assert env_config.num_vehicles == 2
    env = module.make_env(cfg, seed=7, neighbor_k=2)
    assert env.observation_space is not None
