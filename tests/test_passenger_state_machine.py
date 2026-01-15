"""Unit tests for passenger state machine.

Tests cover:
1. Exactly-one-state invariant: each request is in exactly one state at any time
2. Valid state transitions: legal transitions work correctly
3. Invalid state transitions: illegal transitions raise ValueError
4. Terminal states: no further transitions allowed from terminal states
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.env.gym_env import EnvConfig, EventDrivenEnv


def _write_minimal_graph(tmp_path: Path) -> tuple[str, str, str]:
    """Create minimal graph fixtures for testing."""
    nodes = pd.DataFrame({"gnn_node_id": [0, 1, 2]})
    edges = pd.DataFrame(
        {
            "source": [0, 1, 2, 0],
            "target": [1, 2, 0, 2],
            "travel_time_sec": [10.0, 10.0, 10.0, 15.0],
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
            "emb_geo_1": [0.0, 0.3, 0.1],
        }
    )
    emb_path = tmp_path / "node2vec_embeddings.parquet"
    embeddings.to_parquet(emb_path, index=False)
    return str(nodes_path), str(edges_path), str(emb_path)


def _write_od_requests(tmp_path: Path, num_requests: int = 5) -> str:
    """Create OD requests fixture for testing."""
    od = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                pd.Timestamp("2025-01-01T00:00:00") + pd.Timedelta(seconds=i * 10)
                for i in range(num_requests)
            ],
            "pickup_stop_id": [0] * num_requests,
            "dropoff_stop_id": [1] * num_requests,
            "structural_unreachable": [False] * num_requests,
        }
    )
    path = tmp_path / "od.parquet"
    od.to_parquet(path, index=False)
    return str(path)


def _create_env(tmp_path: Path, **config_overrides) -> EventDrivenEnv:
    """Create environment with minimal fixtures."""
    nodes_path, edges_path, emb_path = _write_minimal_graph(tmp_path)
    od_path = _write_od_requests(tmp_path)
    config = EnvConfig(
        seed=42,
        max_horizon_steps=50,
        od_glob=str(od_path),
        graph_nodes_path=nodes_path,
        graph_edges_path=edges_path,
        graph_embeddings_path=emb_path,
        num_vehicles=1,
        vehicle_capacity=6,
        realtime_request_rate_per_sec=0.0,
        realtime_request_count=0,
        **config_overrides,
    )
    return EventDrivenEnv(config)


# ============================================================================
# Test 1: Exactly-one-state invariant
# ============================================================================

VALID_STATES = {
    "waiting",
    "onboard",
    "served",
    "churned_waiting",
    "churned_onboard",
    "structurally_unserviceable",
}


def test_exactly_one_state_invariant(tmp_path: Path):
    """Verify each request is in exactly one state at any time during episode."""
    env = _create_env(tmp_path)
    env.reset()

    # Run a complete episode
    max_steps = 20
    for _ in range(max_steps):
        if env.done:
            break
        actions, mask = env.get_action_mask()
        if not actions:
            break
        feasible = [a for a, m in zip(actions, mask) if m]
        if not feasible:
            action = actions[0] if actions else None
        else:
            action = feasible[0]
        if action is not None:
            env.step(action)

    # Verify exactly-one-state invariant for all requests
    for req in env.requests:
        status = req.get("status")
        # Status must be one of valid states or None (not yet processed, but should be processed)
        assert status in VALID_STATES or status is None, (
            f"Request {req['request_id']} has invalid status: {status}"
        )
        # If status is set, it must be exactly one state
        if status is not None:
            assert status in VALID_STATES, (
                f"Request {req['request_id']} is in invalid state: {status}"
            )


def test_state_log_shows_unique_states_per_request(tmp_path: Path):
    """Verify state_log entries show each request transitions through valid states."""
    env = _create_env(tmp_path)
    env.reset()

    # Run episode
    for _ in range(15):
        if env.done:
            break
        actions, mask = env.get_action_mask()
        if not actions:
            break
        feasible = [a for a, m in zip(actions, mask) if m]
        action = feasible[0] if feasible else (actions[0] if actions else None)
        if action is not None:
            env.step(action)

    # Group state transitions by request_id
    transitions_by_req = {}
    for entry in env.state_log:
        req_id = entry["request_id"]
        if req_id not in transitions_by_req:
            transitions_by_req[req_id] = []
        transitions_by_req[req_id].append((entry["from_state"], entry["to_state"]))

    for req_id, transitions in transitions_by_req.items():
        # Check each transition has valid to_state
        for from_state, to_state in transitions:
            assert to_state in VALID_STATES, (
                f"Request {req_id} transitioned to invalid state: {to_state}"
            )
        # Check chain: each transition's to_state is next transition's from_state
        for i in range(len(transitions) - 1):
            assert transitions[i][1] == transitions[i + 1][0], (
                f"Request {req_id} has broken state chain: "
                f"{transitions[i][1]} != {transitions[i + 1][0]}"
            )


# ============================================================================
# Test 2: Valid state transitions
# ============================================================================

def test_valid_transition_none_to_waiting(tmp_path: Path):
    """Verify transition from None to waiting is allowed."""
    env = _create_env(tmp_path)
    env.reset()

    # Create a mock request with status=None
    req = {"request_id": 999, "status": None}
    # This should not raise
    env._transition(req, "waiting", reason="test")
    assert req["status"] == "waiting"


def test_valid_transition_none_to_structurally_unserviceable(tmp_path: Path):
    """Verify transition from None to structurally_unserviceable is allowed."""
    env = _create_env(tmp_path)
    env.reset()

    req = {"request_id": 999, "status": None}
    env._transition(req, "structurally_unserviceable", reason="test")
    assert req["status"] == "structurally_unserviceable"


def test_valid_transition_waiting_to_onboard(tmp_path: Path):
    """Verify transition from waiting to onboard is allowed."""
    env = _create_env(tmp_path)
    env.reset()

    req = {"request_id": 999, "status": "waiting"}
    env._transition(req, "onboard", reason="boarded")
    assert req["status"] == "onboard"


def test_valid_transition_waiting_to_churned_waiting(tmp_path: Path):
    """Verify transition from waiting to churned_waiting is allowed."""
    env = _create_env(tmp_path)
    env.reset()

    req = {"request_id": 999, "status": "waiting"}
    env._transition(req, "churned_waiting", reason="timeout")
    assert req["status"] == "churned_waiting"


def test_valid_transition_onboard_to_served(tmp_path: Path):
    """Verify transition from onboard to served is allowed."""
    env = _create_env(tmp_path)
    env.reset()

    req = {"request_id": 999, "status": "onboard"}
    env._transition(req, "served", reason="dropoff")
    assert req["status"] == "served"


def test_valid_transition_onboard_to_churned_onboard(tmp_path: Path):
    """Verify transition from onboard to churned_onboard is allowed."""
    env = _create_env(tmp_path)
    env.reset()

    req = {"request_id": 999, "status": "onboard"}
    env._transition(req, "churned_onboard", reason="onboard_churn")
    assert req["status"] == "churned_onboard"


# ============================================================================
# Test 3: Invalid state transitions raise ValueError
# ============================================================================

@pytest.mark.parametrize(
    "from_state,to_state",
    [
        ("waiting", "served"),  # waiting -> served (must go through onboard)
        ("waiting", "structurally_unserviceable"),  # waiting -> structurally_unserviceable
        ("onboard", "waiting"),  # onboard -> waiting (backward)
        ("onboard", "churned_waiting"),  # onboard -> churned_waiting (wrong churn type)
        (None, "served"),  # None -> served (must go through waiting then onboard)
        (None, "onboard"),  # None -> onboard (must go through waiting first)
        (None, "churned_waiting"),  # None -> churned_waiting
        (None, "churned_onboard"),  # None -> churned_onboard
    ],
)
def test_invalid_transitions_raise_error(tmp_path: Path, from_state: str, to_state: str):
    """Verify illegal state transitions raise ValueError."""
    env = _create_env(tmp_path)
    env.reset()

    req = {"request_id": 999, "status": from_state}
    with pytest.raises(ValueError, match="Illegal transition"):
        env._transition(req, to_state, reason="test")


# ============================================================================
# Test 4: Terminal states have no further transitions
# ============================================================================

TERMINAL_STATES = {"served", "churned_waiting", "churned_onboard", "structurally_unserviceable"}


@pytest.mark.parametrize("terminal_state", list(TERMINAL_STATES))
def test_terminal_states_no_further_transitions(tmp_path: Path, terminal_state: str):
    """Verify terminal states cannot transition to any other state."""
    env = _create_env(tmp_path)
    env.reset()

    req = {"request_id": 999, "status": terminal_state}

    # Try transitioning to all valid states
    for target_state in VALID_STATES:
        with pytest.raises(ValueError, match="Illegal transition"):
            env._transition(req, target_state, reason="test")


def test_state_log_recorded_correctly(tmp_path: Path):
    """Verify state transitions are recorded in state_log."""
    env = _create_env(tmp_path)
    env.reset()

    initial_log_len = len(env.state_log)

    req = {"request_id": 888, "status": None}
    env._transition(req, "waiting", reason="request_arrival", time_sec=100.0)

    assert len(env.state_log) == initial_log_len + 1
    last_entry = env.state_log[-1]
    assert last_entry["request_id"] == 888
    assert last_entry["from_state"] is None
    assert last_entry["to_state"] == "waiting"
    assert last_entry["reason"] == "request_arrival"
    assert last_entry["time_sec"] == 100.0
