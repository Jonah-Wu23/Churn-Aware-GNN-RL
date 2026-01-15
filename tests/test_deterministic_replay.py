"""Integration tests for deterministic replay with fixed seed.

Tests cover:
1. Fixed seed produces identical event trace digest
2. Multi-vehicle configuration is deterministic
3. Capacity constraints are handled deterministically
4. Key metrics (served, churned, TACC) are identical across runs
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.env.gym_env import EnvConfig, EventDrivenEnv


def _write_graph_fixtures(tmp_path: Path) -> Tuple[str, str, str]:
    """Create graph fixtures with multiple nodes for realistic testing."""
    nodes = pd.DataFrame({"gnn_node_id": [0, 1, 2, 3, 4]})
    edges = pd.DataFrame(
        {
            "source": [0, 1, 2, 3, 4, 0, 1, 2, 3],
            "target": [1, 2, 3, 4, 0, 2, 3, 4, 0],
            "travel_time_sec": [10.0, 15.0, 12.0, 8.0, 20.0, 25.0, 18.0, 14.0, 22.0],
        }
    )
    nodes_path = tmp_path / "layer2_nodes.parquet"
    edges_path = tmp_path / "layer2_edges.parquet"
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    embeddings = pd.DataFrame(
        {
            "gnn_node_id": [0, 1, 2, 3, 4],
            "emb_geo_0": [0.1, 0.2, 0.3, 0.4, 0.5],
            "emb_geo_1": [0.0, 0.3, 0.1, 0.2, 0.4],
        }
    )
    emb_path = tmp_path / "node2vec_embeddings.parquet"
    embeddings.to_parquet(emb_path, index=False)
    return str(nodes_path), str(edges_path), str(emb_path)


def _write_od_fixtures(tmp_path: Path, num_requests: int = 20) -> str:
    """Create OD requests with diverse pickup/dropoff stops."""
    rng = np.random.default_rng(123)
    od = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                pd.Timestamp("2025-01-01T00:00:00") + pd.Timedelta(seconds=i * 5)
                for i in range(num_requests)
            ],
            "pickup_stop_id": rng.choice([0, 1, 2, 3, 4], size=num_requests).tolist(),
            "dropoff_stop_id": rng.choice([0, 1, 2, 3, 4], size=num_requests).tolist(),
            "structural_unreachable": [False] * num_requests,
        }
    )
    path = tmp_path / "od.parquet"
    od.to_parquet(path, index=False)
    return str(path)


def _run_episode(env: EventDrivenEnv, max_steps: int = 50) -> Dict[str, float]:
    """Run a complete episode with deterministic action selection.
    
    Uses a simple deterministic policy: always select the first feasible action.
    This ensures the same action sequence for identical states.
    """
    env.reset()

    for _ in range(max_steps):
        if env.done:
            break
        actions, mask = env.get_action_mask()
        if not actions:
            break
        # Deterministic policy: pick first feasible action, or first action if none feasible
        feasible = [a for a, m in zip(actions, mask) if m]
        action = feasible[0] if feasible else actions[0]
        env.step(action)

    # Extract key metrics
    return {
        "served": float(env.served),
        "waiting_churned": float(env.waiting_churned),
        "onboard_churned": float(env.onboard_churned),
        "waiting_timeouts": float(env.waiting_timeouts),
        "structurally_unserviceable": float(env.structurally_unserviceable),
        "event_trace_digest": env._event_trace_digest(),
        "event_log_length": len(env.event_log),
        "state_log_length": len(env.state_log),
    }


# ============================================================================
# Test 1: Fixed seed produces identical event trace
# ============================================================================

def test_fixed_seed_identical_event_trace(tmp_path: Path):
    """Verify same seed produces identical event trace digest."""
    nodes_path, edges_path, emb_path = _write_graph_fixtures(tmp_path)
    od_path = _write_od_fixtures(tmp_path)

    seed = 12345

    # Run 1
    env1 = EventDrivenEnv(
        EnvConfig(
            seed=seed,
            max_horizon_steps=100,
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            num_vehicles=1,
            vehicle_capacity=4,
            realtime_request_rate_per_sec=0.0,
            realtime_request_count=0,
        )
    )
    metrics1 = _run_episode(env1)

    # Run 2 with same seed
    env2 = EventDrivenEnv(
        EnvConfig(
            seed=seed,
            max_horizon_steps=100,
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            num_vehicles=1,
            vehicle_capacity=4,
            realtime_request_rate_per_sec=0.0,
            realtime_request_count=0,
        )
    )
    metrics2 = _run_episode(env2)

    # Verify identical event trace digest
    assert metrics1["event_trace_digest"] == metrics2["event_trace_digest"], (
        f"Event trace digests differ: {metrics1['event_trace_digest']} vs {metrics2['event_trace_digest']}"
    )

    # Verify event log lengths match
    assert metrics1["event_log_length"] == metrics2["event_log_length"]
    assert metrics1["state_log_length"] == metrics2["state_log_length"]


def test_different_seeds_different_traces(tmp_path: Path):
    """Verify different seeds produce different event traces (sanity check)."""
    nodes_path, edges_path, emb_path = _write_graph_fixtures(tmp_path)
    od_path = _write_od_fixtures(tmp_path, num_requests=30)

    # Run with seed 111
    env1 = EventDrivenEnv(
        EnvConfig(
            seed=111,
            max_horizon_steps=100,
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            num_vehicles=1,
            vehicle_capacity=3,
            realtime_request_rate_per_sec=0.1,
            realtime_request_count=10,
        )
    )
    metrics1 = _run_episode(env1)

    # Run with seed 222
    env2 = EventDrivenEnv(
        EnvConfig(
            seed=222,
            max_horizon_steps=100,
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            num_vehicles=1,
            vehicle_capacity=3,
            realtime_request_rate_per_sec=0.1,
            realtime_request_count=10,
        )
    )
    metrics2 = _run_episode(env2)

    # Event traces should differ (with high probability due to different RNG)
    # Note: This is a probabilistic test, but with realistic request generation
    # and churn, different seeds should produce different traces
    assert metrics1["event_trace_digest"] != metrics2["event_trace_digest"], (
        "Different seeds produced identical traces (unexpected)"
    )


# ============================================================================
# Test 2: Multi-vehicle configuration is deterministic
# ============================================================================

def test_multi_vehicle_deterministic(tmp_path: Path):
    """Verify multi-vehicle episodes are deterministic with fixed seed."""
    nodes_path, edges_path, emb_path = _write_graph_fixtures(tmp_path)
    od_path = _write_od_fixtures(tmp_path, num_requests=25)

    seed = 54321
    num_vehicles = 3
    capacity = 4

    def run_multi_vehicle_episode():
        env = EventDrivenEnv(
            EnvConfig(
                seed=seed,
                max_horizon_steps=80,
                od_glob=od_path,
                graph_nodes_path=nodes_path,
                graph_edges_path=edges_path,
                graph_embeddings_path=emb_path,
                num_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                realtime_request_rate_per_sec=0.0,
                realtime_request_count=0,
            )
        )
        return _run_episode(env)

    metrics1 = run_multi_vehicle_episode()
    metrics2 = run_multi_vehicle_episode()

    # All metrics must match exactly
    assert metrics1["event_trace_digest"] == metrics2["event_trace_digest"]
    assert metrics1["served"] == metrics2["served"]
    assert metrics1["waiting_churned"] == metrics2["waiting_churned"]
    assert metrics1["onboard_churned"] == metrics2["onboard_churned"]


def test_multi_vehicle_initial_positions_deterministic(tmp_path: Path):
    """Verify initial vehicle positions are deterministic."""
    nodes_path, edges_path, emb_path = _write_graph_fixtures(tmp_path)
    od_path = _write_od_fixtures(tmp_path)

    seed = 99999
    num_vehicles = 4

    env1 = EventDrivenEnv(
        EnvConfig(
            seed=seed,
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            num_vehicles=num_vehicles,
            realtime_request_rate_per_sec=0.0,
            realtime_request_count=0,
        )
    )
    env1.reset()
    positions1 = [v.current_stop for v in env1.vehicles]

    env2 = EventDrivenEnv(
        EnvConfig(
            seed=seed,
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
            num_vehicles=num_vehicles,
            realtime_request_rate_per_sec=0.0,
            realtime_request_count=0,
        )
    )
    env2.reset()
    positions2 = [v.current_stop for v in env2.vehicles]

    assert positions1 == positions2, (
        f"Initial vehicle positions differ: {positions1} vs {positions2}"
    )


# ============================================================================
# Test 3: Capacity constraint handling is deterministic
# ============================================================================

def test_capacity_constraint_deterministic(tmp_path: Path):
    """Verify capacity-constrained boarding is deterministic."""
    nodes_path, edges_path, emb_path = _write_graph_fixtures(tmp_path)
    # Create high-demand scenario at specific stop
    od = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                pd.Timestamp("2025-01-01T00:00:00") + pd.Timedelta(seconds=i)
                for i in range(15)
            ],
            "pickup_stop_id": [0] * 15,  # All requests from stop 0
            "dropoff_stop_id": [1] * 15,
            "structural_unreachable": [False] * 15,
        }
    )
    od_path = tmp_path / "od_capacity.parquet"
    od.to_parquet(od_path, index=False)

    seed = 77777
    small_capacity = 2  # Force capacity constraints

    def run_capacity_episode():
        env = EventDrivenEnv(
            EnvConfig(
                seed=seed,
                max_horizon_steps=60,
                od_glob=str(od_path),
                graph_nodes_path=nodes_path,
                graph_edges_path=edges_path,
                graph_embeddings_path=emb_path,
                num_vehicles=1,
                vehicle_capacity=small_capacity,
                realtime_request_rate_per_sec=0.0,
                realtime_request_count=0,
            )
        )
        return _run_episode(env)

    metrics1 = run_capacity_episode()
    metrics2 = run_capacity_episode()

    assert metrics1["event_trace_digest"] == metrics2["event_trace_digest"]
    assert metrics1["served"] == metrics2["served"]
    assert metrics1["waiting_churned"] == metrics2["waiting_churned"]


# ============================================================================
# Test 4: Key metrics are identical across runs
# ============================================================================

def test_key_metrics_identical(tmp_path: Path):
    """Verify all key metrics are identical for fixed seed runs."""
    nodes_path, edges_path, emb_path = _write_graph_fixtures(tmp_path)
    od_path = _write_od_fixtures(tmp_path, num_requests=30)

    seed = 31415
    num_runs = 3

    all_metrics: List[Dict[str, float]] = []

    for _ in range(num_runs):
        env = EventDrivenEnv(
            EnvConfig(
                seed=seed,
                max_horizon_steps=100,
                od_glob=od_path,
                graph_nodes_path=nodes_path,
                graph_edges_path=edges_path,
                graph_embeddings_path=emb_path,
                num_vehicles=2,
                vehicle_capacity=5,
                realtime_request_rate_per_sec=0.0,
                realtime_request_count=0,
            )
        )
        all_metrics.append(_run_episode(env))

    # Compare all runs to the first run
    baseline = all_metrics[0]
    for i, metrics in enumerate(all_metrics[1:], start=2):
        assert metrics["served"] == baseline["served"], (
            f"Run {i} served count differs: {metrics['served']} vs {baseline['served']}"
        )
        assert metrics["waiting_churned"] == baseline["waiting_churned"], (
            f"Run {i} waiting_churned differs"
        )
        assert metrics["onboard_churned"] == baseline["onboard_churned"], (
            f"Run {i} onboard_churned differs"
        )
        assert metrics["waiting_timeouts"] == baseline["waiting_timeouts"], (
            f"Run {i} waiting_timeouts differs"
        )
        assert metrics["structurally_unserviceable"] == baseline["structurally_unserviceable"], (
            f"Run {i} structurally_unserviceable differs"
        )
        assert metrics["event_trace_digest"] == baseline["event_trace_digest"], (
            f"Run {i} event trace digest differs"
        )


def test_churn_sampling_deterministic(tmp_path: Path):
    """Verify churn probability sampling is deterministic."""
    nodes_path, edges_path, emb_path = _write_graph_fixtures(tmp_path)
    od_path = _write_od_fixtures(tmp_path, num_requests=20)

    seed = 88888

    # Configure aggressive churn to ensure churn events occur
    config_kwargs = dict(
        seed=seed,
        max_horizon_steps=80,
        od_glob=od_path,
        graph_nodes_path=nodes_path,
        graph_edges_path=edges_path,
        graph_embeddings_path=emb_path,
        num_vehicles=1,
        vehicle_capacity=3,
        churn_tol_sec=5,  # Low tolerance -> high churn probability
        churn_beta=0.5,
        realtime_request_rate_per_sec=0.0,
        realtime_request_count=0,
    )

    env1 = EventDrivenEnv(EnvConfig(**config_kwargs))
    metrics1 = _run_episode(env1)

    env2 = EventDrivenEnv(EnvConfig(**config_kwargs))
    metrics2 = _run_episode(env2)

    # Churn counts must be identical
    assert metrics1["waiting_churned"] == metrics2["waiting_churned"]
    assert metrics1["onboard_churned"] == metrics2["onboard_churned"]
    assert metrics1["event_trace_digest"] == metrics2["event_trace_digest"]


def test_realtime_request_generation_deterministic(tmp_path: Path):
    """Verify realtime request generation is deterministic with fixed seed."""
    nodes_path, edges_path, emb_path = _write_graph_fixtures(tmp_path)
    od_path = _write_od_fixtures(tmp_path, num_requests=10)

    seed = 11111

    config_kwargs = dict(
        seed=seed,
        max_horizon_steps=100,
        od_glob=od_path,
        graph_nodes_path=nodes_path,
        graph_edges_path=edges_path,
        graph_embeddings_path=emb_path,
        num_vehicles=1,
        vehicle_capacity=4,
        realtime_request_rate_per_sec=0.2,
        realtime_request_count=15,
        realtime_request_end_sec=200.0,
    )

    env1 = EventDrivenEnv(EnvConfig(**config_kwargs))
    env1.reset()
    requests1 = [(r["request_id"], r["request_time_sec"], r["pickup_stop_id"]) 
                 for r in env1.requests if r.get("source") == "realtime"]

    env2 = EventDrivenEnv(EnvConfig(**config_kwargs))
    env2.reset()
    requests2 = [(r["request_id"], r["request_time_sec"], r["pickup_stop_id"]) 
                 for r in env2.requests if r.get("source") == "realtime"]

    assert requests1 == requests2, (
        "Realtime requests differ between runs with same seed"
    )
