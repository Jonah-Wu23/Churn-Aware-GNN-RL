"""Feature utilities for distillation student models."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.env.gym_env import EventDrivenEnv


GLOBAL_FEATURE_NAMES = [
    "waiting_ratio",
    "onboard_ratio",
    "time_ratio",
    "coverage_ratio",
]


def build_global_summary(env: EventDrivenEnv) -> np.ndarray:
    waiting_total = float(sum(len(q) for q in env.waiting.values()))
    onboard_total = float(sum(len(v.onboard) for v in env.vehicles))
    max_requests = max(1.0, float(env.config.max_requests))
    capacity_total = max(1.0, float(env.config.num_vehicles * env.config.vehicle_capacity))

    waiting_ratio = waiting_total / max_requests
    onboard_ratio = onboard_total / capacity_total

    time_ratio = 0.0
    if env.config.max_sim_time_sec is not None and float(env.config.max_sim_time_sec) > 0:
        time_ratio = float(env.current_time) / float(env.config.max_sim_time_sec)

    coverage_ratio = 0.0
    visited = getattr(env, "_visited_stops", None)
    if visited is not None and len(env.stop_ids) > 0:
        coverage_ratio = float(len(visited)) / float(len(env.stop_ids))

    return np.array([waiting_ratio, onboard_ratio, time_ratio, coverage_ratio], dtype=np.float32)


def build_action_vectors(
    features: Dict[str, np.ndarray],
    env: EventDrivenEnv,
) -> Tuple[np.ndarray, List[str]]:
    node_features = features["node_features"].astype(np.float32)
    edge_features = features["edge_features"].astype(np.float32)
    action_nodes = features["action_node_indices"].astype(np.int64)

    current_idx = int(features["current_node_index"][0]) if len(features["current_node_index"]) else -1
    if current_idx < 0 or current_idx >= node_features.shape[0]:
        current_node_feat = np.zeros(5, dtype=np.float32)
    else:
        current_node_feat = node_features[current_idx]

    global_summary = build_global_summary(env)
    action_count = int(action_nodes.shape[0])
    if action_count == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    current_stack = np.repeat(current_node_feat.reshape(1, -1), action_count, axis=0)
    action_node_feat = node_features[action_nodes]
    action_vectors = np.concatenate(
        [current_stack, action_node_feat, edge_features, np.repeat(global_summary.reshape(1, -1), action_count, axis=0)],
        axis=1,
    )
    feature_names = (
        [f"current_{name}" for name in _node_feature_names()]
        + [f"action_{name}" for name in _node_feature_names()]
        + _edge_feature_names(edge_features.shape[1])
        + list(GLOBAL_FEATURE_NAMES)
    )
    return action_vectors.astype(np.float32), feature_names


def _node_feature_names() -> List[str]:
    return ["risk_mean", "risk_cvar", "waiting_count", "fairness_weight", "geo_embedding"]


def _edge_feature_names(edge_dim: int) -> List[str]:
    base = ["delta_eta_max", "delta_cvar", "count_violation", "travel_time"]
    if edge_dim >= 5:
        base.append("fleet_potential_phi")
    return base
