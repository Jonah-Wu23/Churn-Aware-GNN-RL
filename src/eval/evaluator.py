"""Unified evaluator for policies and paper metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

import numpy as np
import pandas as pd
import torch

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.models.edge_q_gnn import EdgeQGNN
from src.train.dqn import build_hashes
from src.utils.hashing import sha256_file


@dataclass(frozen=True)
class EvalConfig:
    episodes: int = 5
    seed: int = 7
    policy: str = "random"
    model_path: Optional[str] = None
    device: str = "cpu"
    max_steps: Optional[int] = None


def _gini(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    cum = np.cumsum(arr)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(gini)


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
    total_requests = float(len(env.requests))
    structural = float(env.structurally_unserviceable)
    waiting_churned = float(env.waiting_churned)
    waiting_timeouts = float(env.waiting_timeouts)
    onboard_churned = float(env.onboard_churned)
    served = float(env.served)

    non_structural = max(0.0, total_requests - structural)
    waiting_total = waiting_churned + waiting_timeouts
    algorithmic = waiting_total + onboard_churned

    service_rate = served / non_structural if non_structural > 0 else 0.0
    waiting_churn_rate = waiting_total / non_structural if non_structural > 0 else 0.0
    onboard_churn_rate = onboard_churned / non_structural if non_structural > 0 else 0.0
    algorithmic_churn_rate = algorithmic / non_structural if non_structural > 0 else 0.0
    structural_rate = structural / total_requests if total_requests > 0 else 0.0

    wait_times = _compute_wait_times(env)
    wait_p95 = float(np.percentile(wait_times, 95)) if wait_times else 0.0

    gini = _gini([float(v) for v in env.service_count_by_stop.values()])

    return {
        "total_requests": total_requests,
        "served": served,
        "waiting_churned": waiting_churned,
        "waiting_timeouts": waiting_timeouts,
        "onboard_churned": onboard_churned,
        "structural_unserviceable": structural,
        "service_rate": float(service_rate),
        "waiting_churn_rate": float(waiting_churn_rate),
        "onboard_churn_rate": float(onboard_churn_rate),
        "algorithmic_churn_rate": float(algorithmic_churn_rate),
        "structural_unserviceable_rate": float(structural_rate),
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
    )


def evaluate(config: Dict[str, Any], config_path: str | Path, run_dir: Optional[Path] = None) -> Path:
    env_cfg = config.get("env", {})
    eval_cfg = config.get("eval", {})
    model_cfg = config.get("model", {})
    hcride_cfg = eval_cfg.get("hcride", {})

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
    model: Optional[EdgeQGNN] = None
    if eval_config.policy == "edgeq":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for edgeq policy")
        model = EdgeQGNN(
            node_dim=int(model_cfg.get("node_dim", 5)),
            edge_dim=int(model_cfg.get("edge_dim", 4)),
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        model.load_state_dict(torch.load(eval_config.model_path, map_location=device))
        model.to(device)
        model.eval()

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
        "graph_hashes": graph_hashes,
        "od_hashes": od_hashes,
    }

    output = {"meta": meta, "aggregate": aggregate, "episodes": episode_rows}
    output_path = run_dir / "eval_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="ascii")
    df.to_csv(run_dir / "eval_episodes.csv", index=False)
    return output_path
