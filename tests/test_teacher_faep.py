from __future__ import annotations

import numpy as np

from src.train.runner import _greedy_policy


def _build_features(
    counts: list[float],
    fleet_potentials: list[float],
) -> dict[str, np.ndarray]:
    num_actions = len(counts)
    node_features = np.zeros((num_actions, 5), dtype=np.float32)
    for idx, count in enumerate(counts):
        node_features[idx, 0] = 1.0  # risk_mean
        node_features[idx, 1] = 0.0  # risk_cvar
        node_features[idx, 2] = float(count)
        node_features[idx, 3] = 1.0  # fairness
        node_features[idx, 4] = 0.0

    edge_features = np.zeros((num_actions, 5), dtype=np.float32)
    for idx, pot in enumerate(fleet_potentials):
        edge_features[idx, 4] = float(pot)

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "actions": np.array([10, 20], dtype=np.int64),
        "action_mask": np.array([True, True], dtype=bool),
        "action_node_indices": np.array([0, 1], dtype=np.int64),
    }


def test_teacher_faep_prefers_low_congestion():
    features = _build_features(counts=[1.0, 1.0], fleet_potentials=[1.0, 0.0])
    action = _greedy_policy(features, congestion_weight=0.5, coverage_weight=0.3)
    assert action == 20


def test_teacher_faep_count_scale_protection():
    features = _build_features(counts=[1000.0, 1.0], fleet_potentials=[1.0, 0.0])
    action = _greedy_policy(features, congestion_weight=0.5, coverage_weight=0.3)
    assert action == 20
