from types import SimpleNamespace

import numpy as np

from src.env.gym_env import EventDrivenEnv


def test_advance_until_ready_pops_ready_vehicle_when_active_none() -> None:
    env = EventDrivenEnv.__new__(EventDrivenEnv)
    env.active_vehicle_id = None
    env.ready_vehicle_ids = [7, 8]
    env.done = False

    env._advance_until_ready()

    assert env.active_vehicle_id == 7
    assert env.ready_vehicle_ids == [8]


def test_advance_until_ready_keeps_existing_active_vehicle() -> None:
    env = EventDrivenEnv.__new__(EventDrivenEnv)
    env.active_vehicle_id = 3
    env.ready_vehicle_ids = [7, 8]
    env.done = False

    env._advance_until_ready()

    assert env.active_vehicle_id == 3
    assert env.ready_vehicle_ids == [7, 8]


def test_get_feature_batch_advances_when_no_active_vehicle() -> None:
    env = EventDrivenEnv.__new__(EventDrivenEnv)
    env.active_vehicle_id = None
    env.ready_vehicle_ids = []
    env.done = False
    env.stop_ids = [0]
    env.geo_embedding_scalar = {0: 0.0}
    env.fairness_weight = {0: 1.0}
    env.graph_edge_index = np.zeros((2, 0), dtype=np.int64)
    env.graph_edge_features = np.zeros((0, 4), dtype=np.float32)
    env.config = SimpleNamespace(use_fleet_potential=False, debug_mask=False)
    env._compute_waiting_risks = lambda: {0: (0.0, 0.0, 0.0)}

    called = {"value": False}

    def _advance_until_ready() -> None:
        called["value"] = True

    env._advance_until_ready = _advance_until_ready
    env._get_active_vehicle = lambda: None

    batch = EventDrivenEnv.get_feature_batch(env)

    assert called["value"] is True
    assert batch["action_mask"].size == 0


def test_get_action_mask_adds_noop_when_all_infeasible() -> None:
    env = EventDrivenEnv.__new__(EventDrivenEnv)
    env.config = SimpleNamespace(
        allow_stop_when_actions_exist=False,
        vehicle_capacity=1,
        debug_mask=False,
    )
    env.waiting = {1: [{}], 2: [{}]}
    env.neighbors = {0: [(1, 1.0), (2, 1.0)]}
    env.current_time = 0.0
    env._shortest_time = lambda _src, _dst: 1.0
    env.active_vehicle_id = 0
    env.vehicles = [
        SimpleNamespace(
            vehicle_id=0,
            current_stop=0,
            onboard=[{"dropoff_stop_id": 0, "pickup_time_sec": 0.0, "t_max_sec": 10.0}],
        )
    ]

    actions, mask = EventDrivenEnv.get_action_mask(env)

    assert int(env.vehicles[0].current_stop) in actions
    stop_idx = actions.index(int(env.vehicles[0].current_stop))
    assert mask[stop_idx] is True
