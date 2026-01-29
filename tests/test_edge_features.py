from pathlib import Path

import pandas as pd

import pytest

from src.env.gym_env import EnvConfig, EventDrivenEnv


def _write_graph(tmp_path: Path) -> tuple[str, str, str]:
    nodes = pd.DataFrame({"gnn_node_id": [0, 1, 2]})
    edges = pd.DataFrame(
        {
            "source": [0, 0, 1],
            "target": [1, 2, 2],
            "travel_time_sec": [10.0, 5.0, 20.0],
        }
    )
    embeddings = pd.DataFrame(
        {
            "gnn_node_id": [0, 1, 2],
            "emb_geo_0": [0.1, 0.2, 0.3],
            "emb_geo_1": [0.0, 0.1, 0.2],
        }
    )
    nodes_path = tmp_path / "layer2_nodes.parquet"
    edges_path = tmp_path / "layer2_edges.parquet"
    emb_path = tmp_path / "node2vec_embeddings.parquet"
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    embeddings.to_parquet(emb_path, index=False)
    return str(nodes_path), str(edges_path), str(emb_path)


def _write_minimal_od(tmp_path: Path) -> str:
    od = pd.DataFrame(
        {
            "tpep_pickup_datetime": [pd.Timestamp("2025-01-01T00:00:00")],
            "pickup_stop_id": [0],
            "dropoff_stop_id": [2],
        }
    )
    path = tmp_path / "od.parquet"
    od.to_parquet(path, index=False)
    return str(path)


def test_edge_feature_definitions(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            onboard_churn_tol_sec=0,
            onboard_churn_beta=0.1,
        )
    )
    env.active_vehicle_id = 0
    env.vehicles[0].current_stop = 0
    env.current_time = 0.0
    env.onboard = [
        {
            "dropoff_stop_id": 2,
            "pickup_time_sec": 0.0,
            "t_max_sec": 15.0,
            "direct_time_sec": 5.0,
        }
    ]

    batch = env.get_feature_batch()
    actions = batch["actions"].tolist()
    idx = actions.index(1)
    edge = batch["edge_features"][idx]

    curr_eta = env._shortest_time(0, 2)
    new_eta = 10.0 + env._shortest_time(1, 2)
    expected_delta = max(0.0, new_eta - curr_eta)

    prob_before = env._onboard_churn_prob(max(0.0, curr_eta - 5.0))
    prob_after = env._onboard_churn_prob(max(0.0, new_eta - 5.0))
    expected_delta_cvar = prob_after - prob_before

    assert edge[0] == pytest.approx(expected_delta)
    assert edge[1] == pytest.approx(expected_delta_cvar)
    assert edge[2] == pytest.approx(1.0)
    assert edge[3] == pytest.approx(10.0)
