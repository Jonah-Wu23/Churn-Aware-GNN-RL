"""Unified evaluator for policies and paper metrics."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

import numpy as np
import pandas as pd
import torch

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.train.dqn import build_hashes
from src.utils.hashing import sha256_file
from src.utils.fairness import compute_service_volume_gini
from src.utils.feature_spec import get_edge_dim, validate_checkpoint_edge_dim

# Lazy imports to avoid dependency issues
EdgeQGNN = None  # Loaded on demand if policy=edgeq
MAPPOEnvConfig = None  # Loaded on demand if policy=mappo
MAPPOEnvWrapper = None
DiscretePolicy = None  # Loaded on demand if policy=cpo


@dataclass(frozen=True)
class EvalConfig:
    episodes: int = 5
    seed: int = 7
    policy: str = "random"
    model_path: Optional[str] = None
    device: str = "cpu"
    max_steps: Optional[int] = None





def _haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    rad = np.pi / 180.0
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(6371000.0 * c)


def _compute_wait_times(env: EventDrivenEnv) -> List[float]:
    waits = []
    for req in env.requests:
        pickup = req.get("pickup_time_sec")
        if pickup is None:
            continue
        wait = float(pickup) - float(req["request_time_sec"])
        waits.append(max(0.0, wait))
    return waits


def _compute_metrics(env: EventDrivenEnv, total_tacc: float) -> Dict[str, float]:
    """计算episode级别的统一指标。
    
    H2: 复用runner的公共函数，不自造分母。
    """
    # Import runner's public functions for unified metrics
    from src.train.runner import (
        compute_eligible,
        compute_service_rate_simple,
        _compute_service_rate,
        verify_request_conservation,
    )
    
    total_requests = float(len(env.requests))
    structural = float(env.structurally_unserviceable)
    waiting_churned = float(env.waiting_churned)
    waiting_timeouts = float(env.waiting_timeouts)
    onboard_churned = float(env.onboard_churned)
    served = float(env.served)
    
    # Get remaining counts
    waiting_remaining = float(sum(len(q) for q in env.waiting.values()))
    onboard_remaining = float(sum(len(v.onboard) for v in env.fleet.values()))
    
    # Build episode log for unified functions
    episode_log = {
        "served": served,
        "waiting_churned": waiting_churned,
        "onboard_churned": onboard_churned,
        "waiting_timeouts": waiting_timeouts,
        "waiting_remaining": waiting_remaining,
        "onboard_remaining": onboard_remaining,
        "structural_unserviceable": structural,
        "total_requests": total_requests,
    }
    
    # Use unified functions
    eligible_total = compute_eligible(episode_log)
    service_rate_full = _compute_service_rate(episode_log)  # 与runner一致的完整口径
    service_rate_simple = compute_service_rate_simple(episode_log)  # 仅终态请求
    
    # 原有逻辑保留（用于论文对比）：分母 = total - structural
    non_structural = max(0.0, total_requests - structural)
    waiting_total = waiting_churned + waiting_timeouts
    algorithmic = waiting_total + onboard_churned
    
    service_rate_legacy = served / non_structural if non_structural > 0 else 0.0
    waiting_churn_rate = waiting_total / non_structural if non_structural > 0 else 0.0
    onboard_churn_rate = onboard_churned / non_structural if non_structural > 0 else 0.0
    algorithmic_churn_rate = algorithmic / non_structural if non_structural > 0 else 0.0
    structural_rate = structural / total_requests if total_requests > 0 else 0.0

    wait_times = _compute_wait_times(env)
    wait_p95 = float(np.percentile(wait_times, 95)) if wait_times else 0.0

    # Use aligned vector (all Layer-2 stops) for reproducible cross-baseline Gini
    gini = compute_service_volume_gini(env.service_count_by_stop, env.stop_ids)
    
    # H3: Conservation check
    if not verify_request_conservation(episode_log):
        import logging
        LOG = logging.getLogger(__name__)
        LOG.warning(
            "⚠️ evaluator守恒校验失败: eligible=%.0f, structural=%.0f, total=%.0f",
            eligible_total, structural, total_requests
        )

    return {
        # Raw counts
        "total_requests": total_requests,
        "served": served,
        "waiting_churned": waiting_churned,
        "waiting_timeouts": waiting_timeouts,
        "onboard_churned": onboard_churned,
        "waiting_remaining": waiting_remaining,
        "onboard_remaining": onboard_remaining,
        "structural_unserviceable": structural,
        # Unified metrics (与runner一致)
        "eligible_total": eligible_total,
        "service_rate": float(service_rate_full),  # 主指标：统一口径
        "service_rate_simple": float(service_rate_simple),  # 仅终态
        "service_rate_legacy": float(service_rate_legacy),  # 原有口径（用于对比）
        # Churn rates (保留原有)
        "waiting_churn_rate": float(waiting_churn_rate),
        "onboard_churn_rate": float(onboard_churn_rate),
        "algorithmic_churn_rate": float(algorithmic_churn_rate),
        "structural_unserviceable_rate": float(structural_rate),
        # Other metrics
        "tacc_total": float(total_tacc),
        "wait_time_p95_sec": float(wait_p95),
        "service_gini": float(gini),
    }


def _random_policy(features: Dict[str, np.ndarray], rng: np.random.Generator) -> Optional[int]:
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    idx = int(rng.choice(valid))
    return int(actions[idx])


def _greedy_policy(features: Dict[str, np.ndarray]) -> Optional[int]:
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    node_features = features["node_features"]
    action_nodes = features["action_node_indices"].astype(np.int64)
    edge_features = features["edge_features"]
    scores = []
    for idx in valid:
        node_idx = int(action_nodes[idx])
        risk_mean = float(node_features[node_idx, 0])
        risk_cvar = float(node_features[node_idx, 1])
        count = float(node_features[node_idx, 2])
        fairness = float(node_features[node_idx, 3])
        edge = edge_features[idx]
        travel_time = float(edge[3])
        delta_eta = float(edge[0])
        delta_cvar = float(edge[1])
        score = (risk_mean + risk_cvar) * fairness + 0.1 * count - 0.001 * travel_time - 0.1 * delta_eta - 0.1 * delta_cvar
        scores.append((score, idx))
    best_idx = max(scores, key=lambda item: item[0])[1]
    return int(actions[best_idx])


def _edgeq_policy(
    features: Dict[str, np.ndarray],
    model: EdgeQGNN,
    device: torch.device,
) -> Optional[int]:
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None

    x = torch.tensor(features["node_features"], dtype=torch.float32, device=device)
    graph_edge_index = torch.tensor(features["graph_edge_index"], dtype=torch.long, device=device)
    graph_edge_features = torch.tensor(features["graph_edge_features"], dtype=torch.float32, device=device)
    dst = torch.tensor(features["action_node_indices"], dtype=torch.long, device=device)
    src = torch.full_like(dst, int(features["current_node_index"][0]), dtype=torch.long, device=device)
    action_edge_index = torch.stack([src, dst], dim=0)
    action_edge_attr = torch.tensor(features["edge_features"], dtype=torch.float32, device=device)
    data = {
        "node_features": x,
        "graph_edge_index": graph_edge_index,
        "graph_edge_features": graph_edge_features,
        "action_edge_index": action_edge_index,
        "edge_features": action_edge_attr,
    }
    with torch.no_grad():
        q = model(data).detach().cpu().numpy()
    q_masked = np.copy(q)
    q_masked[~mask] = -1e9
    idx = int(np.argmax(q_masked))
    return int(actions[idx])


def _mappo_policy(
    env: EventDrivenEnv,
    features: Dict[str, np.ndarray],
    mappo_cfg: Dict[str, Any],
    actor: Any,
    rnn_states: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
) -> Tuple[Optional[int], np.ndarray]:
    """
    MAPPO actor policy for evaluation.
    
    Args:
        env: The EventDrivenEnv instance
        features: Feature dict from env.get_feature_batch()
        mappo_cfg: MAPPO configuration dict
        actor: Loaded R_Actor model
        rnn_states: RNN hidden states
        masks: Agent masks
        device: Torch device
        
    Returns:
        action: Selected stop ID or None
        rnn_states: Updated RNN states
    """
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None, rnn_states
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None, rnn_states
    
    neighbor_k = int(mappo_cfg.get("neighbor_k", 8))
    
    # Build observation from features (flattened)
    node_features = features["node_features"]
    edge_features = features["edge_features"]
    current_idx = int(features["current_node_index"][0])
    
    # Current node features
    current_node_feat = node_features[current_idx]  # [5]
    
    # Edge features padded to neighbor_k
    edge_feat_padded = np.zeros((neighbor_k, 4), dtype=np.float32)
    n_edges = min(len(edge_features), neighbor_k)
    if n_edges > 0:
        edge_feat_padded[:n_edges] = edge_features[:n_edges]
    edge_feat_flat = edge_feat_padded.flatten()  # [neighbor_k * 4]
    
    # Onboard summary
    vehicle = env._get_active_vehicle()
    if vehicle:
        onboard_count = len(vehicle.onboard) / 10.0
        capacity_ratio = len(vehicle.onboard) / max(1, env.config.vehicle_capacity)
    else:
        onboard_count = 0.0
        capacity_ratio = 0.0
    onboard_summary = np.array([onboard_count, 0.0, 0.0, capacity_ratio], dtype=np.float32)
    
    # Position embedding placeholder
    pos_emb_dim = 16
    pos_emb = np.zeros(pos_emb_dim, dtype=np.float32)
    pos_emb[0] = node_features[current_idx, 4] if current_idx < len(node_features) else 0.0
    
    # Concatenate observation
    obs = np.concatenate([current_node_feat, edge_feat_flat, onboard_summary, pos_emb])
    obs = obs.astype(np.float32)
    
    # Build available_actions mask
    available_actions = np.zeros(neighbor_k + 1, dtype=np.float32)
    for i, is_valid in enumerate(mask):
        if i < neighbor_k:
            available_actions[i] = 1.0 if is_valid else 0.0
    available_actions[neighbor_k] = 1.0  # NOOP always available
    
    # Convert to tensors
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    rnn_states_t = torch.tensor(rnn_states, dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks, dtype=torch.float32, device=device)
    available_t = torch.tensor(available_actions, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Forward pass through actor
    with torch.no_grad():
        action_out, _, new_rnn_states = actor(
            obs_t, rnn_states_t, masks_t, available_t, deterministic=True
        )
    
    action_idx = int(action_out.cpu().numpy().flatten()[0])
    new_rnn_states = new_rnn_states.cpu().numpy()
    
    # Map action index to actual stop
    if action_idx >= len(actions) or action_idx == neighbor_k:
        # NOOP: select first valid action
        if len(valid) > 0:
            action_idx = int(valid[0])
        else:
            return None, new_rnn_states
    
    return int(actions[action_idx]), new_rnn_states


def _cpo_policy(
    env: EventDrivenEnv,
    features: Dict[str, np.ndarray],
    cpo_cfg: Dict[str, Any],
    policy: Any,
    running_state: Any,
    device: torch.device,
) -> Optional[int]:
    """
    CPO policy for evaluation (deterministic mode using argmax).
    
    Args:
        env: The EventDrivenEnv instance
        features: Feature dict from env.get_feature_batch()
        cpo_cfg: CPO configuration dict
        policy: Loaded DiscretePolicy model
        running_state: ZFilter running state normalization
        device: Torch device
        
    Returns:
        action: Selected stop ID or None
    """
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    
    neighbor_k = int(cpo_cfg.get("neighbor_k", 8))
    
    # Build observation from features (matching cpo_env_wrapper format)
    node_features = features["node_features"]
    edge_features = features["edge_features"]
    current_idx = int(features["current_node_index"][0])
    
    # Current node features [5]
    if current_idx < len(node_features):
        current_node_feat = node_features[current_idx]
    else:
        current_node_feat = np.zeros(5, dtype=np.float32)
    
    # Edge features padded to neighbor_k [neighbor_k * 4]
    edge_feat_padded = np.zeros((neighbor_k, 4), dtype=np.float32)
    n_edges = min(len(edge_features), neighbor_k)
    if n_edges > 0:
        edge_feat_padded[:n_edges] = edge_features[:n_edges]
    edge_feat_flat = edge_feat_padded.flatten()
    
    # Onboard summary [4]
    vehicle = env._get_active_vehicle()
    if vehicle:
        onboard_count = len(vehicle.onboard) / 10.0
        capacity = max(1, env.config.vehicle_capacity)
        capacity_ratio = len(vehicle.onboard) / capacity
        avg_delay = 0.0
        max_delay = 0.0
    else:
        onboard_count = 0.0
        avg_delay = 0.0
        max_delay = 0.0
        capacity_ratio = 0.0
    onboard_summary = np.array([onboard_count, avg_delay, max_delay, capacity_ratio], dtype=np.float32)
    
    # Position embedding [1]
    if current_idx < len(node_features):
        pos_emb = np.array([node_features[current_idx, 4]], dtype=np.float32)
    else:
        pos_emb = np.zeros(1, dtype=np.float32)
    
    # Concatenate observation
    obs = np.concatenate([current_node_feat, edge_feat_flat, onboard_summary, pos_emb])
    obs = obs.astype(np.float32)
    
    # Apply running state normalization if available
    if running_state is not None:
        obs = running_state(obs, update=False)
    
    # Build action mask (padded)
    action_dim = neighbor_k + 1  # +1 for NOOP
    action_mask_padded = np.zeros(action_dim, dtype=bool)
    n_valid = min(len(mask), neighbor_k)
    action_mask_padded[:n_valid] = mask[:n_valid]
    action_mask_padded[-1] = True  # NOOP always valid
    
    # Forward pass
    obs_t = torch.tensor(obs, dtype=torch.float64, device=device).unsqueeze(0)
    with torch.no_grad():
        action_probs = policy(obs_t)  # [1, action_dim]
    
    # Apply mask and select best action (deterministic)
    probs = action_probs.cpu().numpy().flatten()
    probs[~action_mask_padded] = 0.0
    action_idx = int(np.argmax(probs))
    
    # Map action to stop ID
    if action_idx >= len(actions) or action_idx == neighbor_k:
        # NOOP: select first valid action
        if len(valid) > 0:
            action_idx = int(valid[0])
        else:
            return None
    
    return int(actions[action_idx])


def _hcride_policy(
    env: EventDrivenEnv,
    features: Dict[str, np.ndarray],
    hcride_cfg: Dict[str, Any],
) -> Optional[int]:
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None

    alpha = float(hcride_cfg.get("alpha", 1.5))
    lagrange_lambda = float(hcride_cfg.get("lagrange_lambda", 1.0))
    preference_threshold = float(hcride_cfg.get("preference_threshold", 0.1))
    preference_radius_scale_m = float(hcride_cfg.get("preference_radius_scale_m", 1000.0))
    empty_stop_penalty = float(hcride_cfg.get("empty_stop_penalty", 1e6))

    vehicle = env._get_active_vehicle()
    if vehicle is None:
        return None

    total_visits = float(sum(vehicle.visit_counts.values()))
    visit_freq = {
        int(stop_id): float(count) / total_visits for stop_id, count in vehicle.visit_counts.items() if total_visits > 0
    }
    positive = {stop_id for stop_id, freq in visit_freq.items() if freq > preference_threshold}

    edge_features = features["edge_features"]
    best_score = -float("inf")
    best_idx = int(valid[0])

    for idx in valid:
        dest = int(actions[idx])
        travel_time = float(edge_features[idx][3])
        queue = env.waiting.get(dest, [])
        if not queue:
            score = -empty_stop_penalty - (travel_time / 60.0)
            if score > best_score:
                best_score = score
                best_idx = int(idx)
            continue

        dropoffs_at_dst = sum(1 for pax in vehicle.onboard if pax["dropoff_stop_id"] == dest)
        projected_onboard = len(vehicle.onboard) - dropoffs_at_dst
        capacity_left = int(env.config.vehicle_capacity - projected_onboard)
        if capacity_left <= 0:
            continue

        picked = queue[:capacity_left]
        waits_sec = [
            max(0.0, (env.current_time + travel_time) - float(req["request_time_sec"])) for req in picked
        ]
        wt_sec = float(np.mean(waits_sec)) if waits_sec else 0.0
        if env.service_count_by_stop.get(dest, 0) > 0:
            meanwt_sec = env.acc_wait_time_by_stop.get(dest, 0.0) / float(env.service_count_by_stop[dest])
        else:
            meanwt_sec = 0.0

        wt_min = wt_sec / 60.0
        meanwt_min = meanwt_sec / 60.0
        reward = (-wt_min) + (alpha * (-1.0) * abs(wt_min - meanwt_min) / 3.0)

        cost = 0.0
        if positive:
            dest_lon, dest_lat = env.stop_coords.get(dest, (0.0, 0.0))
            nearest = float("inf")
            neutral = False
            for pos in positive:
                pos_lon, pos_lat = env.stop_coords.get(pos, (0.0, 0.0))
                dist = _haversine_meters(dest_lon, dest_lat, pos_lon, pos_lat)
                nearest = min(nearest, dist)
                radius = visit_freq.get(pos, 0.0) * preference_radius_scale_m
                if dist <= radius:
                    neutral = True
                    break
            if not neutral and dest not in positive:
                cost = 0.0 if not np.isfinite(nearest) else float(nearest)

        score = reward - lagrange_lambda * cost
        if score > best_score:
            best_score = score
            best_idx = int(idx)

    return int(actions[best_idx])


def _build_env_config(env_cfg: Dict[str, Any]) -> EnvConfig:
    return EnvConfig(
        max_horizon_steps=int(env_cfg.get("max_horizon_steps", 200)),
        mask_alpha=float(env_cfg.get("mask_alpha", 1.5)),
        walk_threshold_sec=int(env_cfg.get("walk_threshold_sec", 600)),
        max_requests=int(env_cfg.get("max_requests", 2000)),
        seed=int(env_cfg.get("seed", 7)),
        num_vehicles=int(env_cfg.get("num_vehicles", 1)),
        vehicle_capacity=int(env_cfg.get("vehicle_capacity", 6)),
        request_timeout_sec=int(env_cfg.get("request_timeout_sec", 600)),
        realtime_request_rate_per_sec=float(env_cfg.get("realtime_request_rate_per_sec", 0.0)),
        realtime_request_count=int(env_cfg.get("realtime_request_count", 0)),
        realtime_request_end_sec=float(env_cfg.get("realtime_request_end_sec", 0.0)),
        churn_tol_sec=int(env_cfg.get("churn_tol_sec", 300)),
        churn_beta=float(env_cfg.get("churn_beta", 0.02)),
        waiting_churn_tol_sec=env_cfg.get("waiting_churn_tol_sec"),
        waiting_churn_beta=env_cfg.get("waiting_churn_beta"),
        onboard_churn_tol_sec=env_cfg.get("onboard_churn_tol_sec"),
        onboard_churn_beta=env_cfg.get("onboard_churn_beta"),
        reward_service=float(env_cfg.get("reward_service", 1.0)),
        reward_waiting_churn_penalty=float(env_cfg.get("reward_waiting_churn_penalty", 1.0)),
        reward_onboard_churn_penalty=float(env_cfg.get("reward_onboard_churn_penalty", 1.0)),
        reward_travel_cost_per_sec=float(env_cfg.get("reward_travel_cost_per_sec", 0.0)),
        reward_tacc_weight=float(env_cfg.get("reward_tacc_weight", 1.0)),
        reward_onboard_delay_weight=float(env_cfg.get("reward_onboard_delay_weight", 0.1)),
        reward_cvar_penalty=float(env_cfg.get("reward_cvar_penalty", 1.0)),
        reward_fairness_weight=float(env_cfg.get("reward_fairness_weight", 1.0)),
        reward_congestion_penalty=float(env_cfg.get("reward_congestion_penalty", 0.0)),
        reward_scale=float(env_cfg.get("reward_scale", 1.0)),
        reward_step_backlog_penalty=float(env_cfg.get("reward_step_backlog_penalty", 0.0)),
        reward_waiting_time_penalty_per_sec=float(env_cfg.get("reward_waiting_time_penalty_per_sec", 0.0)),
        demand_exhausted_min_time_sec=float(env_cfg.get("demand_exhausted_min_time_sec", 300.0)),
        cvar_alpha=float(env_cfg.get("cvar_alpha", 0.95)),
        fairness_gamma=float(env_cfg.get("fairness_gamma", 1.0)),
        travel_time_multiplier=float(env_cfg.get("travel_time_multiplier", 1.0)),
        debug_mask=bool(env_cfg.get("debug_mask", False)),
        od_glob=env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet"),
        graph_nodes_path=env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"),
        graph_edges_path=env_cfg.get("graph_edges_path", "data/processed/graph/layer2_edges.parquet"),
        graph_embeddings_path=env_cfg.get(
            "graph_embeddings_path",
            "data/processed/graph/node2vec_embeddings.parquet",
        ),
        time_split_mode=env_cfg.get("time_split_mode"),
        time_split_ratio=float(env_cfg.get("time_split_ratio", 0.3)),
        # FAEP configuration
        use_fleet_potential=bool(env_cfg.get("use_fleet_potential", False)),
        fleet_potential_mode=str(env_cfg.get("fleet_potential_mode", "next_stop")),
        fleet_potential_k=int(env_cfg.get("fleet_potential_k", 1)),
        fleet_potential_hybrid_center_weight=float(env_cfg.get("fleet_potential_hybrid_center_weight", 0.5)),
        fleet_potential_hybrid_neighbor_weight=float(env_cfg.get("fleet_potential_hybrid_neighbor_weight", 0.5)),
        fleet_potential_phi=str(env_cfg.get("fleet_potential_phi", "log1p_norm")),
    )


def evaluate(config: Dict[str, Any], config_path: str | Path, run_dir: Optional[Path] = None) -> Path:
    env_cfg = config.get("env", {})
    eval_cfg = config.get("eval", {})
    model_cfg = config.get("model", {})
    hcride_cfg = eval_cfg.get("hcride", {})
    mappo_cfg = eval_cfg.get("mappo", {})
    cpo_cfg = eval_cfg.get("cpo", {})
    mohito_cfg = eval_cfg.get("mohito", {})
    wu2024_cfg = eval_cfg.get("wu2024", {})

    eval_config = EvalConfig(
        episodes=int(eval_cfg.get("episodes", 5)),
        seed=int(eval_cfg.get("seed", env_cfg.get("seed", 7))),
        policy=str(eval_cfg.get("policy", "random")),
        model_path=eval_cfg.get("model_path"),
        device=str(eval_cfg.get("device", "cpu")),
        max_steps=eval_cfg.get("max_steps"),
    )

    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("reports") / "eval" / f"{eval_config.policy}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(eval_config.device)
    model = None
    if eval_config.policy == "edgeq":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for edgeq policy")
        # Dynamic import to avoid torch_geometric dependency when not needed
        from src.models.edge_q_gnn import EdgeQGNN
        
        # Use unified edge_dim function
        env_edge_dim = get_edge_dim(env_cfg)
        use_fleet_potential = bool(env_cfg.get("use_fleet_potential", False))
        
        model = EdgeQGNN(
            node_dim=int(model_cfg.get("node_dim", 5)),
            edge_dim=env_edge_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            dueling=bool(model_cfg.get("dueling", False)),
        )
        
        # Load checkpoint and validate edge_dim compatibility
        checkpoint = torch.load(eval_config.model_path, map_location=device)
        checkpoint_edge_dim = checkpoint.get("edge_dim", 4) if isinstance(checkpoint, dict) and "edge_dim" in checkpoint else 4
        
        # If checkpoint is just state_dict, infer edge_dim from q_head layer
        if not isinstance(checkpoint, dict) or "edge_dim" not in checkpoint:
            # Try to infer from model weights
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
            q_head_key = "q_head.0.weight"
            if q_head_key in state_dict:
                q_head_in = state_dict[q_head_key].shape[1]
                # q_head input = hidden_dim * 2 + edge_dim
                hidden_dim = int(model_cfg.get("hidden_dim", 32))
                checkpoint_edge_dim = q_head_in - hidden_dim * 2
        
        # Validate compatibility
        validate_checkpoint_edge_dim(checkpoint_edge_dim, env_edge_dim, use_fleet_potential)
        
        model.load_state_dict(checkpoint if isinstance(checkpoint, dict) and "edge_dim" not in checkpoint else checkpoint)
        model.to(device)
        model.eval()

    # MAPPO actor loading
    mappo_actor = None
    mappo_rnn_states = None
    mappo_masks = None
    if eval_config.policy == "mappo":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for mappo policy")
        # Dynamically import on-policy modules
        on_policy_path = Path("baselines/on-policy").resolve()
        if str(on_policy_path) not in sys.path:
            sys.path.insert(0, str(on_policy_path))
        from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
        from onpolicy.utils.util import get_shape_from_obs_space
        import argparse
        
        # Build minimal args for R_Actor
        neighbor_k = int(mappo_cfg.get("neighbor_k", 8))
        obs_dim = 5 + neighbor_k * 4 + 4 + 16  # node + edge + onboard + pos_emb
        act_dim = neighbor_k + 1
        
        class MinimalArgs:
            hidden_size = int(mappo_cfg.get("hidden_size", 64))
            use_recurrent_policy = True
            recurrent_N = int(mappo_cfg.get("recurrent_N", 1))
            use_naive_recurrent_policy = False
            use_feature_normalization = False
            layer_N = 1
            use_orthogonal = True
            gain = 0.01
            use_policy_active_masks = True
            stacked_frames = 1
        
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = spaces.Discrete(act_dim)
        
        mappo_actor = R_Actor(MinimalArgs(), obs_space, act_space, device=device)
        actor_state = torch.load(eval_config.model_path, map_location=device)
        mappo_actor.load_state_dict(actor_state)
        mappo_actor.to(device)
        mappo_actor.eval()
        
        # Initialize RNN states and masks
        mappo_rnn_states = np.zeros((1, MinimalArgs.recurrent_N, MinimalArgs.hidden_size), dtype=np.float32)
        mappo_masks = np.ones((1, 1), dtype=np.float32)

    # CPO policy loading
    cpo_policy = None
    cpo_running_state = None
    if eval_config.policy == "cpo":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for cpo policy")
        import pickle
        pytorch_cpo_path = Path("baselines/PyTorch-CPO").resolve()
        if str(pytorch_cpo_path) not in sys.path:
            sys.path.insert(0, str(pytorch_cpo_path))
        from models.discrete_policy import DiscretePolicy
        
        # Load pickled model (policy, value, running_state)
        with open(eval_config.model_path, "rb") as f:
            cpo_policy, _, cpo_running_state = pickle.load(f)
        cpo_policy.to(device)
        cpo_policy.eval()
        if cpo_running_state is not None:
            cpo_running_state.fix = True  # Freeze running state for eval

    # MOHITO actor loading (zero-shot baseline)
    mohito_actor = None
    if eval_config.policy == "mohito":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for mohito policy")
        from src.baselines.mohito_adapter import load_mohito_actor
        mohito_actor = load_mohito_actor(eval_config.model_path, mohito_cfg, device)

    # Wu2024 model loading (architecture placeholder baseline)
    wu2024_model = None
    wu2024_weights_mode = "random_init"
    if eval_config.policy == "wu2024":
        from src.baselines.wu2024_adapter import load_wu2024_model
        wu2024_model, wu2024_weights_mode = load_wu2024_model(
            eval_config.model_path, wu2024_cfg, device
        )

    episode_rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(int(eval_config.seed))

    for ep in range(int(eval_config.episodes)):
        env_seed = int(eval_config.seed) + ep
        env_cfg_episode = dict(env_cfg)
        env_cfg_episode["seed"] = env_seed
        env = EventDrivenEnv(_build_env_config(env_cfg_episode))

        total_tacc = 0.0
        steps = 0
        done = False
        while not done:
            if eval_config.max_steps is not None and steps >= int(eval_config.max_steps):
                break
            features = env.get_feature_batch()
            if eval_config.policy == "random":
                action = _random_policy(features, rng)
            elif eval_config.policy == "greedy":
                action = _greedy_policy(features)
            elif eval_config.policy == "edgeq":
                action = _edgeq_policy(features, model, device)
            elif eval_config.policy == "hcride":
                action = _hcride_policy(env, features, hcride_cfg)
            elif eval_config.policy == "mappo":
                action, mappo_rnn_states = _mappo_policy(
                    env, features, mappo_cfg, mappo_actor, mappo_rnn_states, mappo_masks, device
                )
            elif eval_config.policy == "cpo":
                action = _cpo_policy(
                    env, features, cpo_cfg, cpo_policy, cpo_running_state, device
                )
            elif eval_config.policy == "mohito":
                from src.baselines.mohito_adapter import mohito_policy
                action = mohito_policy(
                    env, features, mohito_actor, mohito_cfg, device
                )
            elif eval_config.policy == "wu2024":
                from src.baselines.wu2024_adapter import wu2024_policy
                action = wu2024_policy(
                    env, features, wu2024_model, wu2024_cfg, device, rng
                )
            else:
                raise ValueError(f"Unknown eval policy: {eval_config.policy}")

            if action is None:
                break
            _, _reward, done, info = env.step(int(action))
            total_tacc += float(info.get("step_tacc_gain", 0.0))
            steps += 1

        metrics = _compute_metrics(env, total_tacc)
        metrics["episode_index"] = float(ep)
        metrics["seed"] = float(env_seed)
        metrics["steps"] = float(steps)
        episode_rows.append(metrics)

    df = pd.DataFrame(episode_rows)
    aggregate = {
        "episodes": float(len(df)),
        "policy": eval_config.policy,
    }
    for col in df.columns:
        if col in {"episode_index", "seed"}:
            continue
        aggregate[f"{col}_mean"] = float(df[col].mean())
        aggregate[f"{col}_std"] = float(df[col].std(ddof=0))

    graph_hashes, od_hashes = build_hashes(env_cfg)
    meta = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(str(config_path)),
        "eval_config": eval_config.__dict__,
        "env_config": env_cfg,
        "model_config": model_cfg,
        "hcride_config": hcride_cfg,
        "wu2024_config": wu2024_cfg,
        "graph_hashes": graph_hashes,
        "od_hashes": od_hashes,
    }

    output = {"meta": meta, "aggregate": aggregate, "episodes": episode_rows}
    output_path = run_dir / "eval_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="ascii")
    df.to_csv(run_dir / "eval_episodes.csv", index=False)
    return output_path
