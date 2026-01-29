from pathlib import Path

import pandas as pd

from src.env.gym_env import EnvConfig, EventDrivenEnv


def _write_chain_graph(tmp_path: Path) -> tuple[str, str, str]:
    nodes = pd.DataFrame({"gnn_node_id": [0, 1, 2, 3]})
    edges = pd.DataFrame(
        {
            "source": [0, 1, 2],
            "target": [1, 2, 3],
            "travel_time_sec": [10.0, 10.0, 10.0],
        }
    )
    embeddings = pd.DataFrame(
        {
            "gnn_node_id": [0, 1, 2, 3],
            "emb_geo_0": [0.0, 0.1, 0.2, 0.3],
            "emb_geo_1": [0.0, 0.0, 0.0, 0.0],
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
            "dropoff_stop_id": [3],
        }
    )
    path = tmp_path / "od.parquet"
    od.to_parquet(path, index=False)
    return str(path)


def test_k_hop_subgraph_batch(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_chain_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            num_vehicles=1,
        )
    )
    env.active_vehicle_id = 0
    env.vehicles[0].current_stop = 0
    batch = env.get_feature_batch(k_hop=2)
    assert batch["current_node_index"].tolist() == [0]
    assert set(batch["node_ids"].tolist()) == {0, 1, 2}
    assert batch["actions"].tolist() == [1]
    assert batch["action_node_indices"].tolist() == [1]
    assert batch["graph_edge_index"].shape[0] == 2
    assert batch["graph_edge_features"].shape[1] == 4
