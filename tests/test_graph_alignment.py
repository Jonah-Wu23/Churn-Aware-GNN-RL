from pathlib import Path

import pandas as pd

from src.env.gym_env import EnvConfig, EventDrivenEnv


def _write_graph(tmp_path: Path) -> tuple[str, str, str]:
    nodes = pd.DataFrame({"gnn_node_id": [0, 1, 2]})
    edges = pd.DataFrame(
        {
            "source": [0, 0, 1],
            "target": [1, 2, 2],
            "travel_time_sec": [10.0, 20.0, 5.0],
        }
    )
    nodes_path = tmp_path / "layer2_nodes.parquet"
    edges_path = tmp_path / "layer2_edges.parquet"
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    embeddings = pd.DataFrame(
        {
            "gnn_node_id": [0, 1, 2],
            "emb_geo_0": [0.3, 0.1, 0.2],
            "emb_geo_1": [0.0, 0.1, 0.0],
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
            "dropoff_stop_id": [2],
        }
    )
    path = tmp_path / "od.parquet"
    od.to_parquet(path, index=False)
    return str(path)


def test_actions_from_layer2_edges(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            num_vehicles=1,
        )
    )
    env.active_vehicle_id = 0
    env.vehicles[0].current_stop = 0
    actions, mask = env.get_action_mask()
    assert set(actions) == {1, 2}
    assert all(mask)


def test_travel_time_from_layer2_edges(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            num_vehicles=1,
        )
    )
    env.active_vehicle_id = 0
    env.vehicles[0].current_stop = 0
    env.step(1)
    departure = next(
        entry for entry in reversed(env.event_log) if entry["event_type"] == "VehicleDeparture"
    )
    assert departure["payload"]["travel_time_sec"] == 10.0
    assert env._shortest_time(0, 2) == 15.0
