"""Unified evaluator for policies and paper metrics."""

from __future__ import annotations

import sys
import multiprocessing as mp
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

import numpy as np
import pandas as pd
import torch

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.models.student_mlp import StudentActionMLP
from src.train.dqn import build_hashes
from src.utils.hashing import sha256_file
from src.utils.fairness import compute_service_volume_gini
from src.utils.feature_spec import get_edge_dim, validate_checkpoint_edge_dim
from src.utils.distill_features import build_action_vectors

# Lazy imports to avoid dependency issues
EdgeQGNN = None  # Loaded on demand if policy=edgeq
MAPPOEnvConfig = None  # Loaded on demand if policy=mappo
MAPPOEnvWrapper = None
DiscretePolicy = None  # Loaded on demand if policy=cpo

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

LOG = logging.getLogger(__name__)
_EVAL_WORKER_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


@dataclass(frozen=True)
class EvalConfig:
    episodes: int = 5
    seed: int = 7
    policy: str = "random"
    model_path: Optional[str] = None
    device: str = "cpu"
    max_steps: Optional[int] = None
    parallel_episodes: int = 1
    fast_eval_disable_debug: bool = False
    allow_cuda_parallel: bool = False
    k_hop: Optional[int] = None





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
    fleet = getattr(env, "fleet", None)
    if isinstance(fleet, dict):
        onboard_remaining = float(sum(len(v.onboard) for v in fleet.values()))
    else:
        vehicles = getattr(env, "vehicles", [])
        onboard_remaining = float(sum(len(v.onboard) for v in vehicles))
    
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


def _policy_cache_key(
    policy: str,
    model_path: Optional[str],
    device: torch.device,
    eval_env_cfg: Dict[str, Any],
    mappo_cfg: Dict[str, Any],
    cpo_cfg: Dict[str, Any],
    student_mlp_cfg: Dict[str, Any],
) -> Tuple[Any, ...]:
    edge_dim = int(get_edge_dim(eval_env_cfg))
    neighbor_k_mappo = int(mappo_cfg.get("neighbor_k", 8))
    neighbor_k_cpo = int(cpo_cfg.get("neighbor_k", 8))
    student_hidden = int(student_mlp_cfg.get("hidden_dim", 64))
    student_layers = int(student_mlp_cfg.get("num_layers", 2))
    student_dropout = float(student_mlp_cfg.get("dropout", 0.1))
    return (
        policy,
        str(model_path) if model_path else None,
        str(device),
        edge_dim,
        neighbor_k_mappo,
        neighbor_k_cpo,
        student_hidden,
        student_layers,
        student_dropout,
    )


def _load_policy_objects(
    eval_env_cfg: Dict[str, Any],
    eval_config: EvalConfig,
    model_cfg: Dict[str, Any],
    mappo_cfg: Dict[str, Any],
    cpo_cfg: Dict[str, Any],
    student_mlp_cfg: Dict[str, Any],
    mohito_cfg: Dict[str, Any],
    wu2024_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    key = _policy_cache_key(
        eval_config.policy,
        eval_config.model_path,
        device,
        eval_env_cfg,
        mappo_cfg,
        cpo_cfg,
        student_mlp_cfg,
    )
    if key in _EVAL_WORKER_CACHE:
        return _EVAL_WORKER_CACHE[key]

    policy_objects: Dict[str, Any] = {}
    if eval_config.policy == "edgeq":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for edgeq policy")
        from src.models.edge_q_gnn import EdgeQGNN

        env_edge_dim = get_edge_dim(eval_env_cfg)
        use_fleet_potential = bool(eval_env_cfg.get("use_fleet_potential", False))
        model = EdgeQGNN(
            node_dim=int(model_cfg.get("node_dim", 5)),
            edge_dim=env_edge_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            dueling=bool(model_cfg.get("dueling", False)),
        )

        checkpoint = torch.load(eval_config.model_path, map_location=device)
        checkpoint_edge_dim = (
            checkpoint.get("edge_dim", 4)
            if isinstance(checkpoint, dict) and "edge_dim" in checkpoint
            else 4
        )
        if not isinstance(checkpoint, dict) or "edge_dim" not in checkpoint:
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
            q_head_key = "q_head.0.weight"
            if q_head_key in state_dict:
                q_head_in = state_dict[q_head_key].shape[1]
                hidden_dim = int(model_cfg.get("hidden_dim", 32))
                checkpoint_edge_dim = q_head_in - hidden_dim * 2
        validate_checkpoint_edge_dim(checkpoint_edge_dim, env_edge_dim, use_fleet_potential)

        model.load_state_dict(
            checkpoint if isinstance(checkpoint, dict) and "edge_dim" not in checkpoint else checkpoint
        )
        model.to(device)
        model.eval()
        policy_objects.update({"edgeq_model": model})

    elif eval_config.policy == "mappo":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for mappo policy")
        on_policy_path = Path("baselines/on-policy").resolve()
        if str(on_policy_path) not in sys.path:
            sys.path.insert(0, str(on_policy_path))
        from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
        from gymnasium import spaces

        neighbor_k = int(mappo_cfg.get("neighbor_k", 8))
        actor_state = torch.load(eval_config.model_path, map_location=device)
        obs_dim = _infer_mappo_obs_dim(actor_state)
        if obs_dim is None:
            edge_dim = int(get_edge_dim(eval_env_cfg))
            obs_dim = 5 + neighbor_k * edge_dim + 4 + 16
        mappo_edge_dim = _infer_edge_dim_from_obs_dim(int(obs_dim), neighbor_k)
        if mappo_edge_dim is None:
            mappo_edge_dim = int(get_edge_dim(eval_env_cfg))
        act_dim = neighbor_k + 1

        class MinimalArgs:
            hidden_size = int(mappo_cfg.get("hidden_size", 64))
            use_ReLU = True
            use_recurrent_policy = True
            recurrent_N = int(mappo_cfg.get("recurrent_N", 1))
            use_naive_recurrent_policy = False
            use_feature_normalization = False
            layer_N = 1
            use_orthogonal = True
            gain = 0.01
            use_policy_active_masks = True
            stacked_frames = 1
            algorithm_name = "mappo"

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = spaces.Discrete(act_dim)
        mappo_actor = R_Actor(MinimalArgs(), obs_space, act_space, device=device)
        mappo_actor.load_state_dict(actor_state)
        mappo_actor.to(device)
        mappo_actor.eval()
        policy_objects.update(
            {
                "mappo_actor": mappo_actor,
                "mappo_edge_dim": mappo_edge_dim,
                "mappo_hidden_size": MinimalArgs.hidden_size,
                "mappo_recurrent_N": MinimalArgs.recurrent_N,
            }
        )

    elif eval_config.policy == "cpo":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for cpo policy")
        import pickle
        pytorch_cpo_path = Path("baselines/PyTorch-CPO").resolve()
        if str(pytorch_cpo_path) not in sys.path:
            sys.path.insert(0, str(pytorch_cpo_path))
        from models.discrete_policy import DiscretePolicy  # noqa: F401

        with open(eval_config.model_path, "rb") as f:
            cpo_policy, _, cpo_running_state = pickle.load(f)
        cpo_policy.to(device)
        cpo_policy.eval()
        if cpo_running_state is not None:
            cpo_running_state.fix = True
        cpo_obs_dim = _infer_cpo_obs_dim(cpo_policy)
        cpo_edge_dim = None
        if cpo_obs_dim is not None:
            cpo_edge_dim = _infer_cpo_edge_dim_from_obs_dim(
                int(cpo_obs_dim), int(cpo_cfg.get("neighbor_k", 8))
            )
        policy_objects.update(
            {"cpo_policy": cpo_policy, "cpo_running_state": cpo_running_state, "cpo_edge_dim": cpo_edge_dim}
        )

    elif eval_config.policy == "student_mlp":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for student_mlp policy")
        checkpoint = torch.load(eval_config.model_path, map_location=device)
        config = {}
        if isinstance(checkpoint, dict):
            config = dict(checkpoint.get("config", {}))
            state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            state_dict = checkpoint
        input_dim = int(config.get("input_dim", 0))
        if input_dim <= 0:
            raise ValueError("student_mlp checkpoint missing input_dim in config")
        student_model = StudentActionMLP(
            input_dim=input_dim,
            hidden_dim=int(config.get("hidden_dim", student_mlp_cfg.get("hidden_dim", 64))),
            num_layers=int(config.get("num_layers", student_mlp_cfg.get("num_layers", 2))),
            dropout=float(config.get("dropout", student_mlp_cfg.get("dropout", 0.1))),
            use_layer_norm=bool(config.get("use_layer_norm", student_mlp_cfg.get("use_layer_norm", False))),
        )
        student_model.load_state_dict(state_dict)
        student_model.to(device)
        student_model.eval()
        policy_objects.update({"student_mlp": student_model})

    elif eval_config.policy == "mohito":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for mohito policy")
        from src.baselines.mohito_adapter import load_mohito_actor

        mohito_actor = load_mohito_actor(eval_config.model_path, mohito_cfg, device)
        policy_objects.update({"mohito_actor": mohito_actor})

    elif eval_config.policy == "wu2024":
        from src.baselines.wu2024_adapter import load_wu2024_model

        wu2024_model, wu2024_weights_mode = load_wu2024_model(
            eval_config.model_path, wu2024_cfg, device
        )
        policy_objects.update(
            {"wu2024_model": wu2024_model, "wu2024_weights_mode": wu2024_weights_mode}
        )

    _EVAL_WORKER_CACHE[key] = policy_objects
    return policy_objects


def _evaluate_single_episode(
    env_cfg_episode: Dict[str, Any],
    eval_config: EvalConfig,
    model_cfg: Dict[str, Any],
    hcride_cfg: Dict[str, Any],
    mappo_cfg: Dict[str, Any],
    cpo_cfg: Dict[str, Any],
    student_mlp_cfg: Dict[str, Any],
    mohito_cfg: Dict[str, Any],
    wu2024_cfg: Dict[str, Any],
    policy_objects: Dict[str, Any],
    device: torch.device,
    episode_index: int,
    show_step_bar: bool,
) -> Dict[str, float]:
    env = EventDrivenEnv(_build_env_config(env_cfg_episode))
    start_time = float(getattr(env, "current_time", 0.0))
    sim_time_limit = env_cfg_episode.get("max_sim_time_sec")
    if sim_time_limit is not None:
        sim_time_limit = float(sim_time_limit)
    sim_time_epsilon = 0.5

    total_tacc = 0.0
    steps = 0
    done = False
    end_reason = "unknown"
    rng = np.random.default_rng(int(eval_config.seed) + int(episode_index))

    step_iter = None
    if show_step_bar and eval_config.max_steps is not None and tqdm is not None:
        step_iter = tqdm(
            total=int(eval_config.max_steps),
            desc=f"Eval ep {episode_index}",
            unit="step",
            dynamic_ncols=True,
            leave=False,
        )

    mappo_rnn_states = None
    mappo_masks = None
    if eval_config.policy == "mappo":
        hidden_size = int(policy_objects.get("mappo_hidden_size", 64))
        recurrent_N = int(policy_objects.get("mappo_recurrent_N", 1))
        mappo_rnn_states = np.zeros((1, recurrent_N, hidden_size), dtype=np.float32)
        mappo_masks = np.ones((1, 1), dtype=np.float32)

    while not done:
        if eval_config.max_steps is not None and steps >= int(eval_config.max_steps):
            end_reason = "max_steps"
            break
        if sim_time_limit is not None:
            current_time = float(getattr(env, "current_time", 0.0))
            elapsed = current_time - start_time
            if elapsed >= sim_time_limit:
                end_reason = "max_sim_time"
                break
        if eval_config.k_hop is None:
            features = env.get_feature_batch()
        else:
            try:
                features = env.get_feature_batch(k_hop=eval_config.k_hop)
            except TypeError:
                LOG.warning("Env get_feature_batch does not accept k_hop; falling back to full graph.")
                features = env.get_feature_batch()
        if eval_config.policy == "random":
            action = _random_policy(features, rng)
        elif eval_config.policy == "greedy":
            action = _greedy_policy(features)
        elif eval_config.policy == "edgeq":
            action = _edgeq_policy(features, policy_objects["edgeq_model"], device)
        elif eval_config.policy == "hcride":
            action = _hcride_policy(env, features, hcride_cfg)
        elif eval_config.policy == "mappo":
            action, mappo_rnn_states = _mappo_policy(
                env,
                features,
                mappo_cfg,
                policy_objects["mappo_actor"],
                mappo_rnn_states,
                mappo_masks,
                device,
                edge_dim=policy_objects.get("mappo_edge_dim"),
            )
        elif eval_config.policy == "cpo":
            action = _cpo_policy(
                env,
                features,
                cpo_cfg,
                policy_objects["cpo_policy"],
                policy_objects.get("cpo_running_state"),
                device,
                edge_dim_override=policy_objects.get("cpo_edge_dim"),
            )
        elif eval_config.policy == "student_mlp":
            action = _student_mlp_policy(env, features, policy_objects["student_mlp"], device)
        elif eval_config.policy == "mohito":
            from src.baselines.mohito_adapter import mohito_policy

            action = mohito_policy(
                env, features, policy_objects["mohito_actor"], mohito_cfg, device
            )
        elif eval_config.policy == "wu2024":
            from src.baselines.wu2024_adapter import wu2024_policy

            action = wu2024_policy(
                env, features, policy_objects["wu2024_model"], wu2024_cfg, device, rng
            )
        else:
            raise ValueError(f"Unknown eval policy: {eval_config.policy}")

        if action is None:
            end_reason = "action_none"
            break
        _, _reward, done, info = env.step(int(action))
        total_tacc += float(info.get("step_tacc_gain", 0.0))
        steps += 1
        if sim_time_limit is not None:
            current_time = float(getattr(env, "current_time", 0.0))
            elapsed = current_time - start_time
            if elapsed > sim_time_limit + sim_time_epsilon and not done:
                raise RuntimeError(
                    f"Eval exceeded max_sim_time_sec={sim_time_limit:.3f} (elapsed={elapsed:.3f})"
                )
        if done and end_reason == "unknown":
            end_reason = str(info.get("done_reason", "env_done"))
        if step_iter is not None:
            step_iter.update(1)

    if step_iter is not None:
        step_iter.close()
    if done and end_reason == "unknown":
        end_reason = "env_done"

    metrics = _compute_metrics(env, total_tacc)
    metrics["episode_index"] = float(episode_index)
    metrics["seed"] = float(env_cfg_episode.get("seed", eval_config.seed + episode_index))
    metrics["steps"] = float(steps)
    end_time = float(getattr(env, "current_time", 0.0))
    metrics["sim_time_sec"] = end_time
    metrics["sim_time_elapsed_sec"] = max(0.0, end_time - start_time)
    metrics["end_reason"] = end_reason
    return metrics


def _eval_worker(payload: Dict[str, Any]) -> Dict[str, float]:
    eval_config = EvalConfig(**payload["eval_config"])
    device = torch.device(eval_config.device)
    policy_objects = _load_policy_objects(
        payload["eval_env_cfg"],
        eval_config,
        payload["model_cfg"],
        payload["mappo_cfg"],
        payload["cpo_cfg"],
        payload["student_mlp_cfg"],
        payload["mohito_cfg"],
        payload["wu2024_cfg"],
        device,
    )
    return _evaluate_single_episode(
        payload["env_cfg_episode"],
        eval_config,
        payload["model_cfg"],
        payload["hcride_cfg"],
        payload["mappo_cfg"],
        payload["cpo_cfg"],
        payload["student_mlp_cfg"],
        payload["mohito_cfg"],
        payload["wu2024_cfg"],
        policy_objects,
        device,
        payload["episode_index"],
        show_step_bar=False,
    )

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
    edge_dim = int(edge_features.shape[1]) if edge_features.ndim == 2 else 4
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


def _student_mlp_policy(
    env: EventDrivenEnv,
    features: Dict[str, np.ndarray],
    model: StudentActionMLP,
    device: torch.device,
) -> Optional[int]:
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    action_vectors, _feature_names = build_action_vectors(features, env)
    if action_vectors.size == 0:
        return None
    with torch.no_grad():
        logits = model(torch.tensor(action_vectors, dtype=torch.float32, device=device)).detach().cpu().numpy()
    logits_masked = np.copy(logits)
    logits_masked[~mask] = -1e9
    idx = int(np.argmax(logits_masked))
    return int(actions[idx])


def _mappo_policy(
    env: EventDrivenEnv,
    features: Dict[str, np.ndarray],
    mappo_cfg: Dict[str, Any],
    actor: Any,
    rnn_states: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    edge_dim: Optional[int] = None,
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
    if edge_dim is None:
        edge_dim = int(edge_features.shape[1]) if edge_features.ndim == 2 else 4
    current_idx = int(features["current_node_index"][0])
    
    # Current node features
    current_node_feat = node_features[current_idx]  # [5]
    
    # Edge features padded to neighbor_k
    edge_feat_padded = np.zeros((neighbor_k, edge_dim), dtype=np.float32)
    n_edges = min(len(edge_features), neighbor_k)
    if n_edges > 0:
        feat_dim = int(edge_features.shape[1]) if edge_features.ndim == 2 else 0
        if feat_dim > 0:
            edge_feat_padded[:n_edges, : min(edge_dim, feat_dim)] = edge_features[:n_edges, : min(edge_dim, feat_dim)]
    edge_feat_flat = edge_feat_padded.flatten()  # [neighbor_k * edge_dim]
    
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
    edge_dim_override: Optional[int] = None,
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
    
    if edge_dim_override is not None:
        edge_dim = int(edge_dim_override)
    else:
        edge_dim = int(edge_features.shape[1]) if edge_features.ndim == 2 else 4
    # Edge features padded to neighbor_k [neighbor_k * edge_dim]
    edge_feat_padded = np.zeros((neighbor_k, edge_dim), dtype=np.float32)
    n_edges = min(len(edge_features), neighbor_k)
    if n_edges > 0:
        feat_dim = int(edge_features.shape[1]) if edge_features.ndim == 2 else 0
        if feat_dim > 0:
            edge_feat_padded[:n_edges, : min(edge_dim, feat_dim)] = edge_features[:n_edges, : min(edge_dim, feat_dim)]
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
        try:
            rs_mean = getattr(running_state, "rs", None)
            mean_shape = getattr(rs_mean, "mean", None)
            mean_shape = getattr(mean_shape, "shape", None)
            if mean_shape is not None and len(mean_shape) == 1 and mean_shape[0] != obs.shape[0]:
                running_state = None
            else:
                obs = running_state(obs, update=False)
        except Exception:
            running_state = None
    
    # Build action mask (padded)
    action_dim = neighbor_k + 1  # +1 for NOOP
    action_mask_padded = np.zeros(action_dim, dtype=bool)
    n_valid = min(len(mask), neighbor_k)
    action_mask_padded[:n_valid] = mask[:n_valid]
    action_mask_padded[-1] = True  # NOOP always valid
    
    # Forward pass
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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


def _infer_mappo_obs_dim(state_dict: Dict[str, Any]) -> Optional[int]:
    if not isinstance(state_dict, dict):
        return None
    for key in ("base.mlp.fc1.0.weight", "base.mlp.fc1.weight"):
        weight = state_dict.get(key)
        if hasattr(weight, "shape") and len(weight.shape) == 2:
            return int(weight.shape[1])
    return None


def _infer_edge_dim_from_obs_dim(obs_dim: int, neighbor_k: int) -> Optional[int]:
    if neighbor_k <= 0:
        return None
    base_dim = 5 + 4 + 16  # node + onboard + pos_emb
    edge_total = obs_dim - base_dim
    if edge_total <= 0:
        return None
    if edge_total % neighbor_k != 0:
        return None
    edge_dim = int(edge_total / neighbor_k)
    return edge_dim if edge_dim > 0 else None


def _infer_cpo_obs_dim(policy: Any) -> Optional[int]:
    weight = None
    try:
        layers = getattr(policy, "affine_layers", None)
        if layers is not None and len(layers) > 0:
            weight = getattr(layers[0], "weight", None)
    except Exception:
        weight = None
    if hasattr(weight, "shape") and len(weight.shape) == 2:
        return int(weight.shape[1])
    return None


def _infer_cpo_edge_dim_from_obs_dim(obs_dim: int, neighbor_k: int) -> Optional[int]:
    if neighbor_k <= 0:
        return None
    base_dim = 5 + 4 + 1  # node + onboard + pos_emb
    edge_total = obs_dim - base_dim
    if edge_total <= 0:
        return None
    if edge_total % neighbor_k != 0:
        return None
    edge_dim = int(edge_total / neighbor_k)
    return edge_dim if edge_dim > 0 else None


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
        max_sim_time_sec=env_cfg.get("max_sim_time_sec"),
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
        allow_demand_exhausted_termination=bool(env_cfg.get("allow_demand_exhausted_termination", True)),
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
    student_mlp_cfg = eval_cfg.get("student_mlp", {})
    mohito_cfg = eval_cfg.get("mohito", {})
    wu2024_cfg = eval_cfg.get("wu2024", {})

    eval_config = EvalConfig(
        episodes=int(eval_cfg.get("episodes", 5)),
        seed=int(eval_cfg.get("seed", env_cfg.get("seed", 7))),
        policy=str(eval_cfg.get("policy", "random")),
        model_path=eval_cfg.get("model_path"),
        device=str(eval_cfg.get("device", "cpu")),
        max_steps=eval_cfg.get("max_steps"),
        parallel_episodes=int(eval_cfg.get("parallel_episodes", 1) or 1),
        fast_eval_disable_debug=bool(eval_cfg.get("fast_eval_disable_debug", False)),
        allow_cuda_parallel=bool(eval_cfg.get("allow_cuda_parallel", False)),
        k_hop=eval_cfg.get("k_hop"),
    )
    env_overrides = dict(eval_cfg.get("env_overrides", {}))
    eval_env_cfg = dict(env_cfg)
    if env_overrides:
        eval_env_cfg.update(env_overrides)
    if eval_config.fast_eval_disable_debug:
        eval_env_cfg["debug_mask"] = False
        eval_env_cfg["debug_abort_on_alert"] = False

    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("reports") / "eval" / f"{eval_config.policy}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(eval_config.device)
    policy_objects = _load_policy_objects(
        eval_env_cfg,
        eval_config,
        model_cfg,
        mappo_cfg,
        cpo_cfg,
        student_mlp_cfg,
        mohito_cfg,
        wu2024_cfg,
        device,
    )

    episode_rows: List[Dict[str, float]] = []
    use_parallel = eval_config.parallel_episodes > 1 and eval_config.episodes > 1
    if use_parallel and device.type != "cpu" and not eval_config.allow_cuda_parallel:
        LOG.warning(
            "parallel_episodes=%s requested but device=%s; falling back to serial. Set eval.allow_cuda_parallel=true to override.",
            eval_config.parallel_episodes,
            device,
        )
        use_parallel = False

    if use_parallel:
        payloads = []
        for ep in range(int(eval_config.episodes)):
            env_seed = int(eval_config.seed) + ep
            env_cfg_episode = dict(eval_env_cfg)
            env_cfg_episode["seed"] = env_seed
            payloads.append(
                {
                    "episode_index": ep,
                    "eval_config": eval_config.__dict__,
                    "env_cfg_episode": env_cfg_episode,
                    "eval_env_cfg": eval_env_cfg,
                    "model_cfg": model_cfg,
                    "hcride_cfg": hcride_cfg,
                    "mappo_cfg": mappo_cfg,
                    "cpo_cfg": cpo_cfg,
                    "student_mlp_cfg": student_mlp_cfg,
                    "mohito_cfg": mohito_cfg,
                    "wu2024_cfg": wu2024_cfg,
                }
            )
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(eval_config.parallel_episodes)) as pool:
            results_iter = pool.imap_unordered(_eval_worker, payloads)
            if tqdm is not None:
                results_iter = tqdm(
                    results_iter,
                    total=int(eval_config.episodes),
                    desc="Eval",
                    unit="ep",
                    dynamic_ncols=True,
                )
            episode_rows = list(results_iter)
        episode_rows.sort(key=lambda row: row.get("episode_index", 0))
    else:
        ep_range = range(int(eval_config.episodes))
        ep_iter = tqdm(ep_range, desc="Eval", unit="ep", dynamic_ncols=True) if tqdm is not None else ep_range
        for ep in ep_iter:
            env_seed = int(eval_config.seed) + ep
            env_cfg_episode = dict(eval_env_cfg)
            env_cfg_episode["seed"] = env_seed
            metrics = _evaluate_single_episode(
                env_cfg_episode,
                eval_config,
                model_cfg,
                hcride_cfg,
                mappo_cfg,
                cpo_cfg,
                student_mlp_cfg,
                mohito_cfg,
                wu2024_cfg,
                policy_objects,
                device,
                ep,
                show_step_bar=True,
            )
            episode_rows.append(metrics)
            if tqdm is not None and hasattr(ep_iter, "set_postfix"):
                ep_iter.set_postfix(steps=int(metrics.get("steps", 0)), refresh=False)

    df = pd.DataFrame(episode_rows)
    aggregate = {
        "episodes": float(len(df)),
        "policy": eval_config.policy,
    }
    for col in df.columns:
        if col in {"episode_index", "seed"}:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        aggregate[f"{col}_mean"] = float(df[col].mean())
        aggregate[f"{col}_std"] = float(df[col].std(ddof=0))

    graph_hashes, od_hashes = build_hashes(eval_env_cfg)
    meta = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(str(config_path)),
        "eval_config": eval_config.__dict__,
        "env_config": eval_env_cfg,
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
