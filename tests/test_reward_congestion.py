from __future__ import annotations

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


def _make_env(tmp_path: Path, monkeypatch, density_value: float) -> EventDrivenEnv:
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_minimal_od(tmp_path)

    env = EventDrivenEnv(
        EnvConfig(
            max_horizon_steps=5,
            mask_alpha=1.0,
            walk_threshold_sec=600,
            max_requests=1,
            seed=1,
            num_vehicles=2,
            reward_congestion_penalty=1.0,
            use_fleet_potential=True,
            fleet_potential_mode="hybrid",
            od_glob=str(od_path),
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        )
    )
    env.current_stop = 0
    env.current_time = 0.0
    env.onboard = []

    def _fake_advance() -> None:
        env._step_stats = env._init_step_stats()
        env.active_vehicle_id = 0

    def _fake_density() -> dict[int, float]:
        return {0: 0.0, 1: float(density_value)}

    monkeypatch.setattr(env, "_advance_until_ready", _fake_advance)
    monkeypatch.setattr(env, "_compute_fleet_density_by_stop", _fake_density)
    return env


def test_reward_congestion_penalty_info_and_monotonic(tmp_path: Path, monkeypatch):
    env_low = _make_env(tmp_path, monkeypatch, density_value=1.0)
    _, _, _, info_low = env_low.step(1)

    for key in ("reward_congestion_penalty", "dst_density_raw", "fleet_potential_dst"):
        assert key in info_low

    env_high = _make_env(tmp_path, monkeypatch, density_value=3.0)
    _, _, _, info_high = env_high.step(1)

    assert info_high["fleet_potential_dst"] >= info_low["fleet_potential_dst"]
    assert info_high["reward_congestion_penalty"] >= info_low["reward_congestion_penalty"]
