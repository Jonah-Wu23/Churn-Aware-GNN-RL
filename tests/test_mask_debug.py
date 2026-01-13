from pathlib import Path

import pytest
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
    embeddings = pd.DataFrame(
        {
            "gnn_node_id": [0, 1],
            "emb_geo_0": [0.1, 0.2],
            "emb_geo_1": [0.0, 0.3],
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
            "dropoff_stop_id": [1],
        }
    )
    path = tmp_path / "od.parquet"
    od.to_parquet(path, index=False)
    return str(path)


def _set_violating_onboard(env: EventDrivenEnv) -> None:
    env.active_vehicle_id = 0
    vehicle = env.vehicles[0]
    vehicle.current_stop = 0
    env.current_time = 0.0
    vehicle.onboard = [
        {
            "request_id": 99,
            "pickup_time_sec": 0.0,
            "t_max_sec": 5.0,
            "direct_time_sec": 1.0,
            "dropoff_stop_id": 1,
            "status": "onboard",
        }
    ]


def test_mask_debug_entries_deterministic(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env1 = EventDrivenEnv(
        EnvConfig(
            seed=7,
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    _set_violating_onboard(env1)
    actions, mask = env1.get_action_mask(debug=True)
    assert actions == [1]
    assert mask == [False]
    assert env1.last_mask_debug
    entry = env1.last_mask_debug[0]["violations"][0]
    assert entry["type"] == "hard_mask"
    assert entry["request_id"] == 99
    assert entry["dropoff_stop_id"] == 1
    assert entry["t_max_sec"] == pytest.approx(5.0)
    assert entry["eta_sec"] == pytest.approx(10.0)
    assert entry["over_by_sec"] == pytest.approx(5.0)

    env2 = EventDrivenEnv(
        EnvConfig(
            seed=7,
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    _set_violating_onboard(env2)
    env2.get_action_mask(debug=True)
    entry2 = env2.last_mask_debug[0]["violations"][0]
    assert entry2 == entry


def test_step_populates_mask_debug_when_enabled(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            seed=7,
            debug_mask=True,
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    _set_violating_onboard(env)
    with pytest.raises(ValueError, match="Action violates hard mask constraints"):
        env.step(1)
    assert env.last_mask_debug
