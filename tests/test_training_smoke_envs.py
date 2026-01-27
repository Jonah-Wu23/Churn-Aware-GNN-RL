from __future__ import annotations

from pathlib import Path
from typing import Tuple

import importlib.util

import numpy as np
import pandas as pd
import pytest


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        pytest.skip(f"Parquet writer unavailable: {exc}")


def _build_minimal_data(tmp_path: Path) -> Tuple[Path, Path, Path]:
    nodes = pd.DataFrame(
        {
            "gnn_node_id": [1, 2],
            "emb_geo_0": [0.1, 0.2],
        }
    )
    edges = pd.DataFrame(
        {
            "source": [1, 2],
            "target": [2, 1],
            "travel_time_sec": [10.0, 10.0],
        }
    )
    od = pd.DataFrame(
        {
            "pickup_stop_id": [1, 2],
            "dropoff_stop_id": [2, 1],
            "tpep_pickup_datetime": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 00:05:00"]
            ),
        }
    )

    nodes_path = tmp_path / "nodes.parquet"
    edges_path = tmp_path / "edges.parquet"
    od_path = tmp_path / "od.parquet"
    _write_parquet(nodes, nodes_path)
    _write_parquet(edges, edges_path)
    _write_parquet(od, od_path)
    return nodes_path, edges_path, od_path


def _load_module(path: str):
    script_path = Path(__file__).resolve().parents[1] / path
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_env_config(nodes_path: Path, edges_path: Path, od_path: Path, num_vehicles: int):
    from src.env.gym_env import EnvConfig

    return EnvConfig(
        max_horizon_steps=5,
        max_requests=2,
        seed=7,
        num_vehicles=num_vehicles,
        od_glob=str(od_path),
        graph_nodes_path=str(nodes_path),
        graph_edges_path=str(edges_path),
        time_split_mode=None,
    )


def test_mappo_env_wrapper_smoke(tmp_path: Path) -> None:
    if not (importlib.util.find_spec("gymnasium") or importlib.util.find_spec("gym")):
        pytest.skip("gymnasium/gym not available")

    nodes_path, edges_path, od_path = _build_minimal_data(tmp_path)
    env_config = _build_env_config(nodes_path, edges_path, od_path, num_vehicles=2)
    from src.env.mappo_env_wrapper import MAPPOEnvConfig, MAPPOEnvWrapper

    env = MAPPOEnvWrapper(
        MAPPOEnvConfig(env_config=env_config, neighbor_k=2, max_episode_steps=5, fast_inactive_obs=True)
    )
    obs, share_obs, available = env.reset()
    assert obs.shape[0] == 2
    assert available.shape[0] == 2


def test_mappo_env_wrapper_reset_is_fast(tmp_path: Path) -> None:
    if not (importlib.util.find_spec("gymnasium") or importlib.util.find_spec("gym")):
        pytest.skip("gymnasium/gym not available")

    import time

    nodes_path, edges_path, od_path = _build_minimal_data(tmp_path)
    env_config = _build_env_config(nodes_path, edges_path, od_path, num_vehicles=100)
    from src.env.mappo_env_wrapper import MAPPOEnvConfig, MAPPOEnvWrapper

    env = MAPPOEnvWrapper(
        MAPPOEnvConfig(env_config=env_config, neighbor_k=2, max_episode_steps=5, fast_inactive_obs=True)
    )
    t0 = time.time()
    env.reset()
    elapsed = time.time() - t0
    assert elapsed < 2.0


def test_cpo_env_wrapper_smoke(tmp_path: Path) -> None:
    if not (importlib.util.find_spec("gymnasium") or importlib.util.find_spec("gym")):
        pytest.skip("gymnasium/gym not available")

    nodes_path, edges_path, od_path = _build_minimal_data(tmp_path)
    env_config = _build_env_config(nodes_path, edges_path, od_path, num_vehicles=2)
    from src.env.cpo_env_wrapper import CPOEnvConfig, CPOEnvWrapper

    env = CPOEnvWrapper(CPOEnvConfig(env_config=env_config, neighbor_k=2, include_noop=True))
    obs = env.reset()
    assert obs is not None
