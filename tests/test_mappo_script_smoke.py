from __future__ import annotations

import importlib.util
import sys
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


@pytest.mark.parametrize("n_rollout_threads", [1, 2])
def test_run_mappo_train_smoke_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_rollout_threads: int) -> None:
    if not (importlib.util.find_spec("gymnasium") or importlib.util.find_spec("gym")):
        pytest.skip("gymnasium/gym not available")

    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "baselines" / "on-policy").exists():
        pytest.skip("baselines/on-policy not present")
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
            "max_requests": 2,
            "num_vehicles": 2,
            "vehicle_capacity": 6,
            "mask_alpha": 1.5,
            "od_glob": str(od_path),
            "graph_nodes_path": str(nodes_path),
            "graph_edges_path": str(edges_path),
            "time_split_mode": None,
        },
        "mappo_train": {"num_vehicles": 2, "fast_inactive_obs": True},
    }
    cfg_path = tmp_path / "cfg.yaml"
    try:
        import yaml

        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except Exception:
        pytest.skip("pyyaml not available")

    run_dir = tmp_path / "run"
    module = _load_script("scripts/run_mappo_train.py")
    monkeypatch.setattr(module, "tqdm", None)
    num_env_steps = 5 * int(n_rollout_threads)
    argv = [
        "run_mappo_train.py",
        "--config",
        str(cfg_path),
        "--run_dir",
        str(run_dir),
        "--num_env_steps",
        str(num_env_steps),
        "--episode_length",
        "5",
        "--n_rollout_threads",
        str(n_rollout_threads),
        "--neighbor_k",
        "2",
        "--log_interval",
        "1",
        "--save_interval",
        "999999",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
