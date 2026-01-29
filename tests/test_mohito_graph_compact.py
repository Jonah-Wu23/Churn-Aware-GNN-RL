import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        pytest.skip(f"Parquet writer unavailable: {exc}")


def test_build_mohito_graph_compact_has_small_edge_space(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch_geometric") is None:
        pytest.skip("torch_geometric not available")

    from src.env.gym_env import EnvConfig, EventDrivenEnv
    from src.baselines.mohito_adapter import build_mohito_graph

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

    env = EventDrivenEnv(
        EnvConfig(
            max_horizon_steps=5,
            max_requests=2,
            seed=7,
            num_vehicles=100,
            od_glob=str(od_path),
            graph_nodes_path=str(nodes_path),
            graph_edges_path=str(edges_path),
            time_split_mode=None,
        )
    )
    features = env.get_feature_batch()
    graph, edge_space, action_space = build_mohito_graph(env, features, vehicle_idx=0, grid_size=10, mode="compact")
    assert len(edge_space) == len(action_space)
    assert graph.x.shape[0] <= 100 + 2 * (len(action_space)) + 1

