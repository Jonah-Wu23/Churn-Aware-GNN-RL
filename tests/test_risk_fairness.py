from pathlib import Path

import pandas as pd

from src.env.gym_env import EnvConfig, EventDrivenEnv


def _write_graph_with_coords(tmp_path: Path) -> tuple[str, str, str]:
    nodes = pd.DataFrame(
        {
            "gnn_node_id": [0, 1, 2],
            "lon": [0.0, 1.0, 4.0],
            "lat": [0.0, 0.0, 0.0],
        }
    )
    edges = pd.DataFrame(
        {
            "source": [0, 1],
            "target": [1, 2],
            "travel_time_sec": [10.0, 10.0],
        }
    )
    nodes_path = tmp_path / "layer2_nodes.parquet"
    edges_path = tmp_path / "layer2_edges.parquet"
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    embeddings = pd.DataFrame(
        {
            "gnn_node_id": [0, 1, 2],
            "emb_geo_0": [0.1, 0.2, 0.3],
            "emb_geo_1": [0.0, 0.1, 0.2],
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


def test_cvar_upper_tail(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_graph_with_coords(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            cvar_alpha=0.8,
        )
    )
    probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert env._cvar(probs) == 0.5


def test_fairness_weight_monotonic(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_graph_with_coords(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            fairness_gamma=1.0,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    weights = env.fairness_weight
    assert weights[2] > weights[0]
    assert weights[0] > weights[1]


def test_waiting_risk_aggregation(tmp_path: Path):
    nodes_path, edges_path, emb_path = _write_graph_with_coords(tmp_path)
    od_path = _write_minimal_od(tmp_path)
    env = EventDrivenEnv(
        EnvConfig(
            waiting_churn_tol_sec=10,
            waiting_churn_beta=1.0,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    env.current_time = 10.0
    env.waiting = {0: [], 1: []}
    env.waiting[0].append({"request_id": 1, "request_time_sec": 0.0, "pickup_stop_id": 0, "status": "waiting"})
    risks = env._compute_waiting_risks()
    risk_mean, risk_cvar, count = risks[0]
    assert count == 1
    assert 0.49 <= risk_mean <= 0.51
    assert 0.49 <= risk_cvar <= 0.51
