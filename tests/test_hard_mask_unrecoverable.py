from __future__ import annotations

import types

import pytest

from src.env.gym_env import EnvConfig, EventDrivenEnv, VehicleState
from src.utils.hard_mask import compute_hard_mask_gate


def test_hard_mask_gate_skips_unrecoverable() -> None:
    gate = compute_hard_mask_gate(
        pickup_time_sec=0.0,
        t_max_sec=10.0,
        current_time_sec=0.0,
        best_remaining_sec=20.0,
        slack_sec=0.0,
        skip_unrecoverable=True,
    )
    assert gate.enforce is False
    assert gate.baseline_over_by_sec > 0


def test_get_action_mask_does_not_collapse_on_unrecoverable_onboard_deadline() -> None:
    env = EventDrivenEnv.__new__(EventDrivenEnv)
    env.config = EnvConfig(
        debug_mask=False,
        hard_mask_skip_unrecoverable=True,
        hard_mask_slack_sec=0.0,
    )

    # One passenger is already unrecoverable: even the best route from current_stop violates by 1s.
    pax = {
        "request_id": 8,
        "pickup_time_sec": 100.0,
        "t_max_sec": 10.0,
        "dropoff_stop_id": 180,
        "direct_time_sec": 6.0,
    }
    env.vehicles = [VehicleState(vehicle_id=0, current_stop=160, onboard=[pax])]
    env.active_vehicle_id = 0
    env.current_time = 111.0
    env.waiting = {180: [], 179: [], 161: []}
    env.neighbors = {160: [(180, 1.0), (179, 2.0), (161, 3.0)]}
    env.last_mask_debug = []

    shortest = {(160, 180): 0.0, (179, 180): 0.0, (180, 180): 0.0, (161, 180): 0.0}

    def _shortest_time(self: EventDrivenEnv, src: int, dst: int) -> float:
        return float(shortest.get((int(src), int(dst)), 0.0))

    env._shortest_time = types.MethodType(_shortest_time, env)

    actions, mask = EventDrivenEnv.get_action_mask(env, debug=True)

    # With the unrecoverable gate enabled, hard_mask should not collapse actions to a singleton noop.
    assert len(actions) == 3
    assert all(bool(v) for v in mask)


def test_get_action_mask_still_blocks_recoverable_deadline_violations() -> None:
    env = EventDrivenEnv.__new__(EventDrivenEnv)
    env.config = EnvConfig(
        debug_mask=False,
        hard_mask_skip_unrecoverable=True,
        hard_mask_slack_sec=0.0,
    )

    pax = {
        "request_id": 1,
        "pickup_time_sec": 0.0,
        "t_max_sec": 10.0,
        "dropoff_stop_id": 180,
        "direct_time_sec": 6.0,
    }
    env.vehicles = [VehicleState(vehicle_id=0, current_stop=160, onboard=[pax])]
    env.active_vehicle_id = 0
    env.current_time = 0.0
    env.waiting = {180: [], 179: []}
    env.neighbors = {160: [(180, 5.0), (179, 20.0)]}
    env.last_mask_debug = []

    shortest = {(160, 180): 0.0, (179, 180): 0.0, (180, 180): 0.0}

    def _shortest_time(self: EventDrivenEnv, src: int, dst: int) -> float:
        return float(shortest.get((int(src), int(dst)), 0.0))

    env._shortest_time = types.MethodType(_shortest_time, env)

    actions, mask = EventDrivenEnv.get_action_mask(env, debug=True)

    # Action 179 violates deadline: eta_total=20 > t_max=10, so it must be masked out.
    assert actions == [180, 179]
    assert mask == [True, False]

