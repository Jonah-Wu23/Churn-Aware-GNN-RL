from pathlib import Path

import numpy as np
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
            "emb_geo_0": [0.1, 0.2],
            "emb_geo_1": [0.0, 0.3],
        }
    )
    emb_path = tmp_path / "node2vec_embeddings.parquet"
    embeddings.to_parquet(emb_path, index=False)
    return str(nodes_path), str(edges_path), str(emb_path)


def _write_minimal_od(tmp_path: Path, structural_unreachable: bool = False) -> str:
    od = pd.DataFrame(
        {
            "tpep_pickup_datetime": [pd.Timestamp("2025-01-01T00:00:00")],
            "pickup_stop_id": [0],
            "dropoff_stop_id": [1],
            "structural_unreachable": [structural_unreachable],
        }
    )
    path = tmp_path / "od.parquet"
    od.to_parquet(path, index=False)
    return str(path)


def test_churn_sigmoid_shape(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            churn_tol_sec=10,
            churn_beta=1.0,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    p_low = env._waiting_churn_prob(0.0)
    p_mid = env._waiting_churn_prob(10.0)
    p_high = env._waiting_churn_prob(100.0)
    assert p_low < 0.5
    assert 0.49 <= p_mid <= 0.51
    assert p_high > 0.99


def test_waiting_onboard_param_split(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            waiting_churn_tol_sec=5,
            waiting_churn_beta=2.0,
            onboard_churn_tol_sec=20,
            onboard_churn_beta=0.5,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    wait_prob = env._waiting_churn_prob(5.0)
    onboard_prob = env._onboard_churn_prob(5.0)
    assert 0.49 <= wait_prob <= 0.51
    assert onboard_prob < 0.5


def test_waiting_churn_reproducible(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            seed=123,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    env.waiting = {0: [], 1: []}
    env.current_time = 500.0
    env.rng = np.random.default_rng(123)
    env.waiting[0].append(
        {"request_id": 1, "request_time_sec": 0.0, "pickup_stop_id": 0, "status": "waiting"}
    )
    env.waiting[0].append(
        {"request_id": 2, "request_time_sec": 100.0, "pickup_stop_id": 0, "status": "waiting"}
    )
    churned, timeout, _, _, _, _, churned_ids, _, _ = env._apply_churn()

    env2 = EventDrivenEnv(
        EnvConfig(
            seed=123,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    env2.waiting = {0: [], 1: []}
    env2.current_time = 500.0
    env2.rng = np.random.default_rng(123)
    env2.waiting[0].append(
        {"request_id": 1, "request_time_sec": 0.0, "pickup_stop_id": 0, "status": "waiting"}
    )
    env2.waiting[0].append(
        {"request_id": 2, "request_time_sec": 100.0, "pickup_stop_id": 0, "status": "waiting"}
    )
    churned2, timeout2, _, _, _, _, churned_ids2, _, _ = env2._apply_churn()

    assert churned == churned2
    assert timeout == timeout2
    assert churned_ids == churned_ids2
    assert all(req["status"] in {"waiting", "churned_waiting"} for req in env.waiting[0])


def test_onboard_churn_reproducible(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            seed=7,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    vehicle = env.vehicles[0]
    env.current_time = 200.0
    env.rng = np.random.default_rng(7)
    vehicle.onboard = [
        {
            "request_id": 10,
            "pickup_time_sec": 0.0,
            "direct_time_sec": 50.0,
            "dropoff_stop_id": 1,
            "status": "onboard",
        }
    ]
    churned, _, _, churned_ids, _ = env._apply_onboard_churn(vehicle)

    env2 = EventDrivenEnv(
        EnvConfig(
            seed=7,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    vehicle2 = env2.vehicles[0]
    env2.current_time = 200.0
    env2.rng = np.random.default_rng(7)
    vehicle2.onboard = [
        {
            "request_id": 10,
            "pickup_time_sec": 0.0,
            "direct_time_sec": 50.0,
            "dropoff_stop_id": 1,
            "status": "onboard",
        }
    ]
    churned2, _, _, churned_ids2, _ = env2._apply_onboard_churn(vehicle2)

    assert churned == churned2
    assert churned_ids == churned_ids2


def test_structural_unserviceable_excluded(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path, structural_unreachable=True)
    env = EventDrivenEnv(
        EnvConfig(
            max_horizon_steps=1,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    env.current_stop = 0
    obs, reward, done, info = env.step(1)
    assert info["structural_unserviceable"] == 1.0
    assert info["algorithmic_churned"] == 0.0
