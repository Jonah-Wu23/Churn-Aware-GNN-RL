from pathlib import Path

import pandas as pd

from src.env.gym_env import EnvConfig, EventDrivenEnv


def _write_minimal_graph(tmp_path: Path) -> tuple[str, str, str]:
    nodes = pd.DataFrame({"gnn_node_id": [0, 1]})
    edges = pd.DataFrame(
        {
            "source": [0],
            "target": [1],
            "travel_time_sec": [10.0],
        }
    )
    nodes_path = tmp_path / "layer2_nodes.parquet"
    edges_path = tmp_path / "layer2_edges.parquet"
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    embeddings = pd.DataFrame(
        {
            "gnn_node_id": [0, 1],
            "emb_geo_0": [0.1, 0.5],
            "emb_geo_1": [0.2, 0.4],
        }
    )
    emb_path = tmp_path / "node2vec_embeddings.parquet"
    embeddings.to_parquet(emb_path, index=False)
    return str(nodes_path), str(edges_path), str(emb_path)


def _write_minimal_od(tmp_path: Path) -> str:
    od = pd.DataFrame(
        {
            "tpep_pickup_datetime": [pd.Timestamp("2025-01-01T00:00:00")],
            "pickup_stop_id": [0],
            "dropoff_stop_id": [1],
        }
    )
    path = tmp_path / "od.parquet"
    od.to_parquet(path, index=False)
    return str(path)


def test_mask_blocks_over_budget(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)

    env = EventDrivenEnv(
        EnvConfig(
            max_horizon_steps=5,
            mask_alpha=1.0,
            walk_threshold_sec=600,
            max_requests=1,
            seed=1,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    env.current_time = 0.0
    env.current_stop = 0
    env.onboard = [
        {
            "dropoff_stop_id": 1,
            "pickup_time_sec": 0.0,
            "t_max_sec": 5.0,
            "direct_time_sec": 1.0,
        }
    ]

    actions, mask = env.get_action_mask()
    assert actions == [1, 0]
    assert mask == [False, True]
