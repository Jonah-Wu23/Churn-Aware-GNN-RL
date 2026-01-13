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
            "emb_geo_0": [0.2, 0.4],
            "emb_geo_1": [0.1, 0.3],
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


def test_reward_components(tmp_path: Path, monkeypatch):
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)

    env = EventDrivenEnv(
        EnvConfig(
            max_horizon_steps=5,
            mask_alpha=1.0,
            walk_threshold_sec=600,
            max_requests=1,
            seed=1,
            reward_service=2.0,
            reward_churn_penalty=3.0,
            reward_travel_cost_per_sec=0.5,
            reward_tacc_weight=0.2,
            reward_onboard_delay_weight=1.0,
            reward_cvar_penalty=4.0,
            reward_fairness_weight=5.0,
            cvar_alpha=0.95,
            fairness_gamma=1.0,
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    env.current_stop = 0
    env.current_time = 0.0
    env.onboard = []

    def _fake_advance():
        env._step_stats = env._init_step_stats()
        env._step_stats["served"] = 3.0
        env._step_stats["waiting_churn_prob_sum"] = 0.5
        env._step_stats["waiting_churn_prob_weighted_sum"] = 0.7
        env._step_stats["waiting_churn_cvar"] = 0.2
        env._step_stats["tacc_gain"] = 100.0
        env.active_vehicle_id = 0

    monkeypatch.setattr(env, "_advance_until_ready", _fake_advance)

    obs, reward, done, info = env.step(1)

    expected = (
        2.0 * 3
        - 3.0 * 0.5
        - 5.0 * 0.7
        - 4.0 * 0.2
        - 0.5 * 10.0
        - 1.0 * 0.0
        + 0.2 * 100.0
    )
    assert reward == expected
