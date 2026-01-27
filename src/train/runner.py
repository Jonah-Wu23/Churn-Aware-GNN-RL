"""Training runner with structured curriculum and service-rate-based transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import hashlib
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.train.curriculum import StageSpec, default_stages, generate_stage, load_nodes, load_od_frames, stress_stages
from src.train.dqn import DQNConfig, DQNTrainer, build_hashes, write_run_meta
from src.train.reward_ramp import RampConfig, build_ramp_config, compute_ramped_weights, DEFAULT_RAMP_FIELDS
from src.train.evaluation_checkpointer import EvalConfig, EvaluationCheckpointer, compute_env_cfg_hash
from src.utils.config import load_config
from src.utils.feature_spec import get_edge_dim

LOG = logging.getLogger(__name__)


class StageFailedError(Exception):
    """Raised when a stage fails to reach trigger_service_rate after max extensions."""
    def __init__(self, stage: str, service_rate: float, trigger_service_rate: float):
        self.stage = stage
        self.service_rate = service_rate
        self.trigger_service_rate = trigger_service_rate
        super().__init__(
            f"Stage {stage} failed: service_rate={service_rate:.4f} < trigger={trigger_service_rate:.4f}"
        )


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning with service-rate-gated transitions."""
    stages: List[str] = field(default_factory=lambda: ["L0", "L1", "L2", "L3"])
    trigger_service_rate: float = 0.5  # Changed from 0.8 to 0.5
    gamma: float = 1.0
    stage_max_steps: int = 50000
    stage_min_episodes: int = 3
    stage_steps: Dict[str, int] = field(default_factory=dict)
    
    # Service-rate-gated transition settings
    service_rate_window_size: int = 5
    require_service_rate_transition: bool = True
    stage_extension_steps: int = 30000
    max_stage_extensions: int = 2
    fail_policy: str = "fail_fast"  # "fail_fast" or "forced"
    service_rate_warning_threshold: float = 0.35  # 70% of trigger_service_rate (0.5)
    stage_require_service_rate: Dict[str, bool] = field(default_factory=dict)
    service_rate_gate_source: str = "auto"  # "auto", "eval", "train", "max"
    stage_service_rate_gate_source: Dict[str, str] = field(default_factory=dict)
    
    # Collapse protection
    collapse_drop_delta: float = 0.10
    collapse_min_rho: float = 0.15
    collapse_patience: int = 2
    epsilon_cap_on_collapse: float = 0.3
    cap_steps: int = 5000
    lr_mult_on_collapse: float = 0.5
    
    # Evaluation settings
    eval_enabled: bool = True
    eval_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1000])
    eval_interval_steps: int = 5000
    
    # Reward ramp settings
    reward_ramp_steps: int = 10000
    ramp_fields: List[str] = field(default_factory=lambda: DEFAULT_RAMP_FIELDS.copy())

    @property
    def trigger_rho(self) -> float:
        return float(self.trigger_service_rate)

    @trigger_rho.setter
    def trigger_rho(self, value: float) -> None:
        self.trigger_service_rate = float(value)

    @property
    def rho_warning_threshold(self) -> float:
        return float(self.service_rate_warning_threshold)

    @rho_warning_threshold.setter
    def rho_warning_threshold(self, value: float) -> None:
        self.service_rate_warning_threshold = float(value)

    @property
    def rho_window_size(self) -> int:
        return int(self.service_rate_window_size)

    @rho_window_size.setter
    def rho_window_size(self, value: int) -> None:
        self.service_rate_window_size = int(value)

    @property
    def require_rho_transition(self) -> bool:
        return bool(self.require_service_rate_transition)

    @require_rho_transition.setter
    def require_rho_transition(self, value: bool) -> None:
        self.require_service_rate_transition = bool(value)


def _build_env_config(env_cfg: Dict[str, Any]) -> EnvConfig:
    return EnvConfig(
        max_horizon_steps=int(env_cfg.get("max_horizon_steps", 200)),
        max_sim_time_sec=env_cfg.get("max_sim_time_sec"),
        allow_stop_when_actions_exist=bool(env_cfg.get("allow_stop_when_actions_exist", False)),
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
        reward_service_transform=str(env_cfg.get("reward_service_transform", "none")),
        reward_service_transform_scale=float(env_cfg.get("reward_service_transform_scale", 1.0)),
        reward_waiting_churn_penalty=float(env_cfg.get("reward_waiting_churn_penalty", 1.0)),
        reward_onboard_churn_penalty=float(env_cfg.get("reward_onboard_churn_penalty", 1.0)),
        reward_travel_cost_per_sec=float(env_cfg.get("reward_travel_cost_per_sec", 0.0)),
        reward_tacc_weight=float(env_cfg.get("reward_tacc_weight", 1.0)),
        reward_tacc_transform=str(env_cfg.get("reward_tacc_transform", "none")),
        reward_tacc_transform_scale=float(env_cfg.get("reward_tacc_transform_scale", 1.0)),
        reward_onboard_delay_weight=float(env_cfg.get("reward_onboard_delay_weight", 0.1)),
        reward_cvar_penalty=float(env_cfg.get("reward_cvar_penalty", 1.0)),
        reward_fairness_weight=float(env_cfg.get("reward_fairness_weight", 1.0)),
        reward_congestion_penalty=float(env_cfg.get("reward_congestion_penalty", 0.0)),
        reward_scale=float(env_cfg.get("reward_scale", 1.0)),
        reward_step_backlog_penalty=float(env_cfg.get("reward_step_backlog_penalty", 0.0)),
        reward_waiting_time_penalty_per_sec=float(env_cfg.get("reward_waiting_time_penalty_per_sec", 0.0)),
        reward_potential_alpha=float(env_cfg.get("reward_potential_alpha", 0.0)),
        reward_potential_alpha_source=str(env_cfg.get("reward_potential_alpha_source", "env_default")),
        reward_potential_lost_weight=float(env_cfg.get("reward_potential_lost_weight", 0.0)),
        reward_potential_scale_with_reward_scale=bool(
            env_cfg.get("reward_potential_scale_with_reward_scale", True)
        ),
        demand_exhausted_min_time_sec=float(env_cfg.get("demand_exhausted_min_time_sec", 300.0)),
        cvar_alpha=float(env_cfg.get("cvar_alpha", 0.95)),
        fairness_gamma=float(env_cfg.get("fairness_gamma", 1.0)),
        debug_mask=bool(env_cfg.get("debug_mask", False)),
        debug_abort_on_alert=bool(env_cfg.get("debug_abort_on_alert", True)),
        debug_dump_dir=str(env_cfg.get("debug_dump_dir", "reports/debug/potential_alerts")),
        od_glob=env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet"),
        graph_nodes_path=env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"),
        graph_edges_path=env_cfg.get("graph_edges_path", "data/processed/graph/layer2_edges.parquet"),
        graph_embeddings_path=env_cfg.get(
            "graph_embeddings_path",
            "data/processed/graph/node2vec_embeddings.parquet",
        ),
        travel_time_multiplier=float(env_cfg.get("travel_time_multiplier", 1.0)),
        time_split_mode=env_cfg.get("time_split_mode"),
        time_split_ratio=float(env_cfg.get("time_split_ratio", 0.3)),
        # FAEP configuration
        use_fleet_potential=bool(env_cfg.get("use_fleet_potential", False)),
        fleet_potential_mode=str(env_cfg.get("fleet_potential_mode", "next_stop")),
        fleet_potential_k=int(env_cfg.get("fleet_potential_k", 1)),
        fleet_potential_hybrid_center_weight=float(env_cfg.get("fleet_potential_hybrid_center_weight", 0.5)),
        fleet_potential_hybrid_neighbor_weight=float(env_cfg.get("fleet_potential_hybrid_neighbor_weight", 0.5)),
        fleet_potential_phi=str(env_cfg.get("fleet_potential_phi", "log1p_norm")),
        reward_terminal_backlog_penalty=float(env_cfg.get("reward_terminal_backlog_penalty", 0.0)),
        hard_mask_skip_unrecoverable=bool(env_cfg.get("hard_mask_skip_unrecoverable", False)),
        hard_mask_slack_sec=float(env_cfg.get("hard_mask_slack_sec", 0.0)),
    )


def _stage_specs_from_config(
    curriculum_cfg: Dict[str, Any],
    strict_stage_params: bool = False,
) -> Tuple[List[StageSpec], Dict[str, Dict]]:
    """Parse stage_params into StageSpec list and env_overrides map.
    
    Args:
        curriculum_cfg: Curriculum configuration dict
        strict_stage_params: If True, raise ValueError for unknown fields; else warning
    
    Returns:
        Tuple of (stage_specs, env_overrides_map)
    """
    # StageSpec accepts only these OD sampling fields
    STAGESPEC_FIELDS = {
        "name", "description", "density_multiplier", "sample_fraction", 
        "time_scale", "center_quantile", "edge_quantile", "short_trip_quantile",
        "long_trip_quantile", "center_ratio", "churn_tol_override_sec", "travel_time_multiplier"
    }
    
    defaults = {spec.name: spec for spec in default_stages()}
    names = curriculum_cfg.get("stages", list(defaults.keys()))
    stage_params = curriculum_cfg.get("stage_params", {})
    
    specs: List[StageSpec] = []
    env_overrides_map: Dict[str, Dict] = {}
    
    for name in names:
        base = defaults.get(name, StageSpec(name=name, description="Custom curriculum stage"))
        overrides = stage_params.get(name, {})
        
        # Split: OD sampling params vs env_overrides
        od_params = {k: v for k, v in overrides.items() if k in STAGESPEC_FIELDS}
        env_overrides_map[name] = overrides.get("env_overrides", {})
        
        # Strong validation: unknown fields not in env_overrides
        for k in overrides:
            if k not in STAGESPEC_FIELDS and k != "env_overrides":
                msg = f"Stage {name}: field '{k}' not in STAGESPEC_FIELDS. Put it under env_overrides."
                if strict_stage_params:
                    raise ValueError(msg)
                else:
                    LOG.warning(msg)
        
        params = {**base.__dict__, **od_params, "name": name}
        specs.append(StageSpec(**params))
    
    return specs, env_overrides_map


def compute_eligible(log: Dict[str, float]) -> float:
    """计算完整 eligible 分母（所有进入系统的可服务请求）。
    
    eligible = served + waiting_churned + onboard_churned + waiting_timeouts + waiting_remaining + onboard_remaining
    
    注意：structural_unserviceable 不计入 eligible，因为它们在结构上不可服务。
    守恒关系：eligible + structural_unserviceable == total_arrived_requests
    """
    served = float(log.get("served", 0.0))
    waiting_churned = float(log.get("waiting_churned", 0.0))
    onboard_churned = float(log.get("onboard_churned", 0.0))
    waiting_timeouts = float(log.get("waiting_timeouts", 0.0))
    waiting_remaining = float(log.get("waiting_remaining", 0.0))
    onboard_remaining = float(log.get("onboard_remaining", 0.0))
    return served + waiting_churned + onboard_churned + waiting_timeouts + waiting_remaining + onboard_remaining


def compute_service_rate_simple(log: Dict[str, float]) -> float:
    """计算简化 service_rate（仅已结束请求，用于论文对比）。
    
    分母：eligible_simple = served + waiting_churned + onboard_churned + waiting_timeouts
    不含 waiting_remaining 和 onboard_remaining。
    
    用途：
    - 与"只统计终态"的既有工作口径对比
    - 论文中称为 "conditional-on-terminated service rate"
    """
    served = float(log.get("served", 0.0))
    waiting_churned = float(log.get("waiting_churned", 0.0))
    onboard_churned = float(log.get("onboard_churned", 0.0))
    waiting_timeouts = float(log.get("waiting_timeouts", 0.0))
    eligible_simple = served + waiting_churned + onboard_churned + waiting_timeouts
    if eligible_simple <= 0:
        return 0.0
    return served / eligible_simple


def verify_request_conservation(log: Dict[str, float], tolerance: float = 1e-6) -> bool:
    """验证请求守恒：eligible_total + structural_unserviceable == total_requests。
    
    确保每个请求最终状态只落在一个终态集合里。
    
    Returns:
        True 如果守恒成立，False 如果存在不一致
    """
    eligible = compute_eligible(log)
    structural = float(log.get("structural_unserviceable", 0.0))
    total_requests = float(log.get("total_requests", 0.0))
    
    # 如果 total_requests 未填充，尝试从 served + churned + remaining + structural 反推
    if total_requests <= 0:
        # 无法验证，默认通过
        return True
    
    computed_total = eligible + structural
    return abs(computed_total - total_requests) <= tolerance


def _compute_service_rate(log: Dict[str, float]) -> float:
    """Compute service rate with backlog in denominator.
    
    Denominator: all requests that entered the system (served + churned + timeouts + backlog).
    Uses compute_eligible() for consistent calculation.
    """
    eligible = compute_eligible(log)
    if eligible <= 0:
        return 0.0
    served = float(log.get("served", 0.0))
    return served / eligible


def _compute_rho(log: Dict[str, float], gamma: float) -> float:
    """Compute rho metric from episode log: service_rate / (1 + gamma * stuckness)."""
    service_rate = _compute_service_rate(log)
    stuckness = float(log.get("stuckness", 0.0))
    return float(service_rate / (1.0 + gamma * stuckness))


def _compute_service_rate_window_mean(service_rate_history: List[float], window_size: int) -> float:
    """Compute mean service_rate over the last window_size episodes."""
    if not service_rate_history:
        return 0.0
    window = service_rate_history[-window_size:]
    return float(np.mean(window))


def _compute_rho_window_mean(rho_history: List[float], window_size: int) -> float:
    """Compute mean rho over the last window_size episodes."""
    if not rho_history:
        return 0.0
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    window = rho_history[-int(window_size):]
    return float(np.mean(window))


def _select_eval_action(
    model: torch.nn.Module,
    features: Dict[str, np.ndarray],
    device: torch.device,
) -> Optional[int]:
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0 or not np.any(mask):
        return None
    obs_idx = int(features["current_node_index"][0]) if len(features["current_node_index"]) else 0
    dst = torch.tensor(features["action_node_indices"].astype(np.int64), device=device)
    src = torch.full_like(dst, int(obs_idx), dtype=torch.long)
    action_edge_index = torch.stack([src, dst], dim=0)
    data = {
        "node_features": torch.tensor(features["node_features"], dtype=torch.float32, device=device),
        "graph_edge_index": torch.tensor(features["graph_edge_index"], dtype=torch.long, device=device),
        "graph_edge_features": torch.tensor(features["graph_edge_features"], dtype=torch.float32, device=device),
        "action_edge_index": action_edge_index,
        "edge_features": torch.tensor(features["edge_features"], dtype=torch.float32, device=device),
    }
    with torch.no_grad():
        q = model(data).detach()
    q_masked = q.clone()
    invalid = torch.tensor(~mask, device=device)
    q_masked[invalid] = -1e9
    idx = int(torch.argmax(q_masked).item())
    return int(actions[idx])


def _merge_env_cfg(
    base_cfg: Dict[str, Any],
    stage_overrides: Dict[str, float | int],
    phase_overrides: Optional[Dict[str, float | int]],
) -> Dict[str, Any]:
    merged = dict(base_cfg)
    merged.update(stage_overrides)
    if phase_overrides:
        merged.update(phase_overrides)
    source = "env_default"
    if "reward_potential_alpha" in stage_overrides:
        source = "stage_override"
    if phase_overrides and "reward_potential_alpha" in phase_overrides:
        source = "phase_override"
    merged["reward_potential_alpha_source"] = source
    return merged


def _build_ramp_config_from_envs(
    phase2_env_cfg: Dict[str, Any],
    phase3_env_cfg: Dict[str, Any],
    reward_ramp_steps: int,
    ramp_fields: List[str],
) -> RampConfig:
    fields = ramp_fields if ramp_fields else DEFAULT_RAMP_FIELDS.copy()
    w2 = {field: float(phase2_env_cfg.get(field, 0.0)) for field in fields}
    w3_target = {field: float(phase3_env_cfg.get(field, w2.get(field, 0.0))) for field in fields}
    return RampConfig(
        reward_ramp_steps=int(reward_ramp_steps),
        ramp_fields=fields,
        w2=w2,
        w3_target=w3_target,
    )


def _resolve_stage_epsilon_schedule(
    stage_name: str,
    schedule_cfg: Dict[str, Any],
    default_decay_steps: int,
) -> Optional[Dict[str, float | int]]:
    if not schedule_cfg:
        return None
    stage_cfg = schedule_cfg.get(stage_name)
    if not isinstance(stage_cfg, dict):
        return None
    start = stage_cfg.get("start")
    end = stage_cfg.get("end")
    decay_steps = int(stage_cfg.get("decay_steps", default_decay_steps))
    if end is None:
        return None
    return {"start": float(start) if start is not None else None, "end": float(end), "decay_steps": decay_steps}


def _behavior_clone(
    trainer: DQNTrainer,
    env: EventDrivenEnv,
    steps: int,
    log_handle,
    log_every: int = 500,
    congestion_weight: float = 0.5,
    coverage_weight: float = 0.3,
) -> None:
    if steps <= 0:
        return
    env.reset()
    losses: List[float] = []
    for step in range(1, int(steps) + 1):
        features = env.get_feature_batch()
        action = _greedy_policy(
            features,
            congestion_weight=congestion_weight,
            coverage_weight=coverage_weight,
        )
        if action is None:
            env.reset()
            continue
        actions = features["actions"].astype(np.int64)
        mask = features["action_mask"].astype(bool)
        if len(actions) == 0 or not np.any(mask):
            env.reset()
            continue
        try:
            target_idx = int(np.where(actions == int(action))[0][0])
        except IndexError:
            env.reset()
            continue

        q = trainer._q_values_single(
            obs=features["node_features"],
            obs_idx=int(features["current_node_index"][0]),
            action_nodes=features["action_node_indices"].astype(np.int64),
            action_edge=features["edge_features"],
            requires_grad=True,
        )
        q_masked = q.clone()
        q_masked[torch.tensor(~mask, device=q.device)] = -1e9
        loss = F.cross_entropy(q_masked.unsqueeze(0), torch.tensor([target_idx], device=q.device))
        trainer.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), float(trainer.config.max_grad_norm))
        trainer.optimizer.step()
        losses.append(float(loss.item()))

        _, _reward, done, _info = env.step(int(action))
        if done:
            env.reset()

        if log_every > 0 and step % int(log_every) == 0:
            avg_loss = float(np.mean(losses)) if losses else 0.0
            log_handle.write(
                json.dumps(
                    {"type": "behavior_cloning", "step": int(step), "avg_loss": avg_loss},
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_handle.flush()
            losses.clear()


def _greedy_policy(
    features: Dict[str, np.ndarray],
    congestion_weight: float = 0.5,
    coverage_weight: float = 0.3,
    debug: bool = False,
) -> Optional[int]:
    """Greedy teacher policy with FAEP-aware scoring.
    
    A1: 修正行为克隆 teacher，把"避免蜂拥"写进贪心打分。
    
    Args:
        features: Feature batch from env.get_feature_batch()
        congestion_weight: λ_cong for congestion penalty (A1.2)
        coverage_weight: λ_cov for marginal coverage reward (A1.3)
        debug: Enable alignment assertion checks (A1.6)
    
    Score components:
        - (risk_mean + risk_cvar) * fairness: Base risk-weighted attraction
        - 0.1 * log1p(count): Demand count (log1p for scale protection, A1.4)
        - -0.001 * travel_time: Travel cost
        - -0.1 * delta_eta: ETA increase penalty
        - -0.1 * delta_cvar: CVaR increase penalty
        - -λ_cong * fleet_potential: Congestion penalty (A1.2, FAEP only)
        - +λ_cov * risk_mean * (1 - fleet_potential): Marginal contribution (A1.3, FAEP only)
    """
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return int(actions[0])
    node_features = features["node_features"]
    action_nodes = features["action_node_indices"].astype(np.int64)
    edge_features = features["edge_features"]
    
    # Check if FAEP enabled (edge_features has 5 dimensions)
    edge_dim = edge_features.shape[1] if edge_features.ndim == 2 else 0
    use_faep = edge_dim >= 5
    
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
        
        # A1.4: Use log1p(count) for scale protection (avoid count overwhelming other terms)
        count_term = float(np.log1p(count))
        score = (
            (risk_mean + risk_cvar) * fairness
            + 0.1 * count_term
            - 0.001 * travel_time
            - 0.1 * delta_eta
            - 0.1 * delta_cvar
        )
        
        # A1.2 & A1.3: FAEP-aware scoring
        if use_faep:
            fleet_potential = float(edge[4])  # φ(C(dst)) - congestion at destination
            
            # A1.6: Debug assertion - verify alignment
            if debug:
                dst = int(actions[idx])
                LOG.debug(
                    "Teacher FAEP check: action_idx=%d, dst=%d, fleet_potential=%.4f",
                    idx, dst, fleet_potential
                )
            
            # A1.2: Congestion penalty - other vehicles targeting this stop reduce attractiveness
            score -= congestion_weight * fleet_potential
            
            # A1.3: Marginal contribution reward - high-risk low-coverage stops are more valuable
            score += coverage_weight * risk_mean * (1.0 - fleet_potential)
        
        scores.append((score, idx))
    best_idx = max(scores, key=lambda item: item[0])[1]
    return int(actions[best_idx])


def run_curriculum_training(
    config_path: str | Path,
    run_dir: Optional[Path] = None,
    start_stage: Optional[str] = None,
    init_model_path: Optional[str | Path] = None,
) -> Path:
    cfg = load_config(config_path)
    env_cfg = cfg.get("env", {})
    curriculum_cfg = cfg.get("curriculum", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    viz_cfg = train_cfg.get("viz") if isinstance(train_cfg, dict) else None

    trigger_service_rate = float(curriculum_cfg.get(
        "trigger_service_rate",
        curriculum_cfg.get("trigger_rho", 0.5),
    ))
    service_rate_window_size = int(curriculum_cfg.get(
        "service_rate_window_size",
        curriculum_cfg.get("rho_window_size", 5),
    ))
    require_service_rate_transition = bool(curriculum_cfg.get(
        "require_service_rate_transition",
        curriculum_cfg.get("require_rho_transition", True),
    ))
    service_rate_warning_threshold = float(curriculum_cfg.get(
        "service_rate_warning_threshold",
        curriculum_cfg.get("rho_warning_threshold", 0.35),
    ))
    stage_require_service_rate = dict(curriculum_cfg.get(
        "stage_require_service_rate",
        curriculum_cfg.get("stage_require_rho", {}),
    ))
    service_rate_gate_source = str(curriculum_cfg.get(
        "service_rate_gate_source",
        curriculum_cfg.get("rho_gate_source", "auto"),
    ))
    stage_service_rate_gate_source = dict(curriculum_cfg.get(
        "stage_service_rate_gate_source",
        curriculum_cfg.get("stage_rho_gate_source", {}),
    ))

    curriculum = CurriculumConfig(
        stages=list(curriculum_cfg.get("stages", ["L0", "L1", "L2", "L3"])),
        trigger_service_rate=trigger_service_rate,
        gamma=float(curriculum_cfg.get("gamma", 1.0)),
        stage_max_steps=int(curriculum_cfg.get("stage_max_steps", 50_000)),
        stage_min_episodes=int(curriculum_cfg.get("stage_min_episodes", 3)),
        # Service-rate-gated transition settings
        service_rate_window_size=service_rate_window_size,
        require_service_rate_transition=require_service_rate_transition,
        stage_extension_steps=int(curriculum_cfg.get("stage_extension_steps", 30_000)),
        max_stage_extensions=int(curriculum_cfg.get("max_stage_extensions", 2)),
        fail_policy=str(curriculum_cfg.get("fail_policy", "fail_fast")),
        service_rate_warning_threshold=service_rate_warning_threshold,
        stage_require_service_rate=stage_require_service_rate,
        service_rate_gate_source=service_rate_gate_source,
        stage_service_rate_gate_source=stage_service_rate_gate_source,
        # Collapse protection
        collapse_drop_delta=float(curriculum_cfg.get("collapse_drop_delta", 0.10)),
        collapse_min_rho=float(curriculum_cfg.get("collapse_min_rho", 0.15)),
        collapse_patience=int(curriculum_cfg.get("collapse_patience", 2)),
        epsilon_cap_on_collapse=float(curriculum_cfg.get("epsilon_cap_on_collapse", 0.3)),
        cap_steps=int(curriculum_cfg.get("cap_steps", 5000)),
        lr_mult_on_collapse=float(curriculum_cfg.get("lr_mult_on_collapse", 0.5)),
        # Evaluation settings
        eval_enabled=bool(curriculum_cfg.get("eval_enabled", True)),
        eval_seeds=list(curriculum_cfg.get("eval_seeds", [42, 123, 456, 789, 1000])),
        eval_interval_steps=int(curriculum_cfg.get("eval_interval_steps", 5000)),
        # Reward ramp settings
        reward_ramp_steps=int(curriculum_cfg.get("reward_ramp_steps", 10000)),
        ramp_fields=list(curriculum_cfg.get("ramp_fields", DEFAULT_RAMP_FIELDS)),
        stage_steps=dict(curriculum_cfg.get("stage_steps", {})),
    )
    # Read strict_stage_params and parse stages
    strict = bool(curriculum_cfg.get("strict_stage_params", False))
    stage_specs, stage_env_overrides_map = _stage_specs_from_config(curriculum_cfg, strict_stage_params=strict)

    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("reports") / "runs" / f"curriculum_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    curriculum_log = run_dir / "curriculum_log.jsonl"
    log_handle = curriculum_log.open("w", encoding="utf-8")
    meta_payload = {
        "type": "meta",
        "config_path": str(config_path),
        "curriculum": curriculum.__dict__,
        "stages": [spec.__dict__ for spec in stage_specs],
        "start_stage": start_stage,
        "init_model_path": str(init_model_path) if init_model_path is not None else None,
    }
    log_handle.write(
        json.dumps(
            meta_payload,
            ensure_ascii=False,
        )
        + "\n"
    )
    log_handle.flush()

    base_od = load_od_frames(env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet"))
    nodes = load_nodes(env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"))

    dqn_config = DQNConfig(
        seed=int(train_cfg.get("seed", env_cfg.get("seed", 7))),
        total_steps=int(train_cfg.get("total_steps", 200_000)),
        buffer_size=int(train_cfg.get("buffer_size", 10_000)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        learning_starts=int(train_cfg.get("learning_starts", 2_000)),
        train_freq=int(train_cfg.get("train_freq", 1)),
        gradient_steps=int(train_cfg.get("gradient_steps", 1)),
        target_update_interval=int(train_cfg.get("target_update_interval", 2_000)),
        target_update_tau=float(train_cfg.get("target_update_tau", 0.0)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 10.0)),
        double_dqn=bool(train_cfg.get("double_dqn", True)),
        epsilon_start=float(train_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(train_cfg.get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(train_cfg.get("epsilon_decay_steps", 100_000)),
        log_every_steps=int(train_cfg.get("log_every_steps", 1_000)),
        checkpoint_every_steps=int(train_cfg.get("checkpoint_every_steps", 10_000)),
        device=str(train_cfg.get("device", "cpu")),
        prioritized_replay=bool(train_cfg.get("prioritized_replay", False)),
        replay_alpha=float(train_cfg.get("replay_alpha", 0.6)),
        replay_beta_start=float(train_cfg.get("replay_beta_start", 0.4)),
        replay_beta_frames=int(train_cfg.get("replay_beta_frames", 200_000)),
        replay_eps=float(train_cfg.get("replay_eps", 1e-6)),
    )

    from src.models.edge_q_gnn import EdgeQGNN

    model = EdgeQGNN(
        node_dim=int(model_cfg.get("node_dim", 5)),
        edge_dim=get_edge_dim(env_cfg),  # Use unified tool function
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        dueling=bool(model_cfg.get("dueling", False)),
    )
    model.to(torch.device(dqn_config.device))

    if init_model_path is not None:
        init_path = Path(init_model_path)
        if not init_path.exists():
            raise FileNotFoundError(f"init_model_path does not exist: {init_path}")
        state_dict = torch.load(init_path, map_location=torch.device(dqn_config.device))
        model.load_state_dict(state_dict)
        LOG.info("Loaded init model from %s", init_path)
    elif start_stage is not None:
        LOG.warning("start_stage=%s set without init_model_path; training will start from scratch.", start_stage)

    all_specs = [spec for spec in stage_specs if spec.name in curriculum.stages]
    if start_stage is not None:
        stage_names = [spec.name for spec in all_specs]
        if start_stage not in stage_names:
            raise ValueError(f"start_stage {start_stage} not in active stages {stage_names}")
        start_index = stage_names.index(start_stage)
        run_specs = all_specs[start_index:]
    else:
        start_index = 0
        run_specs = all_specs

    l3_two_stage_cfg = curriculum_cfg.get("l3_two_stage", {})
    l3_two_stage_enabled = bool(l3_two_stage_cfg.get("enabled", False))
    phase1_steps_default = int(curriculum.stage_max_steps * 0.4)
    phase1_steps_cfg = int(l3_two_stage_cfg.get("phase1_steps", phase1_steps_default))
    phase2_steps_cfg = l3_two_stage_cfg.get("phase2_steps")
    phase1_steps = max(0, min(int(curriculum.stage_max_steps), phase1_steps_cfg))
    remaining_steps = int(curriculum.stage_max_steps) - phase1_steps
    if phase2_steps_cfg is None:
        phase2_steps = remaining_steps
        phase3_steps = 0
    else:
        phase2_steps = max(0, min(int(remaining_steps), int(phase2_steps_cfg)))
        phase3_steps = max(0, int(curriculum.stage_max_steps) - phase1_steps - phase2_steps)
    phase1_overrides = l3_two_stage_cfg.get("phase1_env_overrides")
    phase2_overrides = l3_two_stage_cfg.get("phase2_env_overrides")
    phase3_overrides = l3_two_stage_cfg.get("phase3_env_overrides")
    bc_cfg = l3_two_stage_cfg.get("behavior_cloning", {})
    bc_enabled = bool(bc_cfg.get("enabled", False))
    bc_steps = int(bc_cfg.get("steps", 0))
    bc_log_every = int(bc_cfg.get("log_every", 500))
    bc_congestion_weight = float(bc_cfg.get("teacher_congestion_weight", 0.5))
    bc_coverage_weight = float(bc_cfg.get("teacher_coverage_weight", 0.3))
    epsilon_by_stage_cfg = curriculum_cfg.get("epsilon_by_stage", {})
    eval_config = EvalConfig(
        eval_seeds=list(curriculum.eval_seeds),
        eval_interval_steps=int(curriculum.eval_interval_steps),
        eval_epsilon=float(curriculum_cfg.get("eval_epsilon", 0.0)),
        eval_episodes_per_seed=int(curriculum_cfg.get("eval_episodes_per_seed", 1)),
        collapse_drop_delta=float(curriculum.collapse_drop_delta),
        collapse_min_rho=float(curriculum.collapse_min_rho),
        collapse_patience=int(curriculum.collapse_patience),
    )

    def _build_eval_checkpointer(env_cfg: Dict[str, Any]) -> Optional[EvaluationCheckpointer]:
        if not curriculum.eval_enabled:
            return None
        device = torch.device(dqn_config.device)

        def _env_factory() -> EventDrivenEnv:
            return EventDrivenEnv(_build_env_config(env_cfg))

        def _select_action(model: torch.nn.Module, features: Dict[str, np.ndarray]) -> Optional[int]:
            return _select_eval_action(model, features, device)

        def _rho_fn(info: Dict[str, float]) -> float:
            return _compute_rho(info, curriculum.gamma)

        return EvaluationCheckpointer(
            config=eval_config,
            env_factory=_env_factory,
            select_action_fn=_select_action,
            compute_rho_fn=_rho_fn,
            device=device,
            gamma=curriculum.gamma,  # H1: Pass gamma from curriculum config
        )

    for local_idx, spec in enumerate(run_specs):
        current_stage_idx = start_index + local_idx
        stage_budget = int(curriculum.stage_steps.get(spec.name, curriculum.stage_max_steps))
        stage_dir = run_dir / f"stage_{spec.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = stage_dir / "edgeq_model_best.pt"
        
        phase1_steps_default = int(stage_budget * 0.4)
        phase1_steps_cfg = int(l3_two_stage_cfg.get("phase1_steps", phase1_steps_default))
        phase2_steps_cfg = l3_two_stage_cfg.get("phase2_steps")
        phase1_steps = max(0, min(int(stage_budget), phase1_steps_cfg))
        remaining_steps = int(stage_budget) - phase1_steps
        if phase2_steps_cfg is None:
            phase2_steps = remaining_steps
            phase3_steps = 0
        else:
            phase2_steps = max(0, min(int(remaining_steps), int(phase2_steps_cfg)))
            phase3_steps = max(0, int(stage_budget) - phase1_steps - phase2_steps)
        stage_output = generate_stage(
            base_od=base_od,
            nodes=nodes,
            stage=spec,
            output_dir=stage_dir,
            seed=int(dqn_config.seed),
        )
        stage_env_base = dict(env_cfg)
        stage_env_base["od_glob"] = str(stage_output.od_path)

        episode_count = 0
        latest_rho = 0.0
        rho_history: List[float] = []  # 保留 rho 作为诊断指标
        service_rate_history: List[float] = []  # 进入延长轮依据
        eval_rho_history: List[float] = []
        eval_service_rate_history: List[float] = []
        best_service_rate = -1.0
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        trainer: Optional[DQNTrainer] = None
        eval_checkpointer: Optional[EvaluationCheckpointer] = None
        current_phase: str = "main"  # Track current phase for logging
        phase_step: int = 0  # Track steps within phase for ramp
        current_env_cfg: Dict[str, Any] = {}  # Track current env config

        def _reset_eval_for_env(env_cfg_for_eval: Dict[str, Any]) -> None:
            nonlocal eval_checkpointer, eval_rho_history, eval_service_rate_history
            eval_rho_history = []
            eval_service_rate_history = []
            eval_checkpointer = _build_eval_checkpointer(env_cfg_for_eval)

        def _run_eval_if_needed() -> None:
            if eval_checkpointer is None or trainer is None:
                return
            if not eval_checkpointer.should_evaluate(trainer.global_step):
                return
            eval_result = eval_checkpointer.evaluate(
                model=trainer.model,
                global_step=int(trainer.global_step),
                stage=spec.name,
                phase=current_phase,
                alpha=float(current_env_cfg.get("reward_potential_alpha", 0.0)) if current_env_cfg else 0.0,
                env_cfg_hash=compute_env_cfg_hash(current_env_cfg) if current_env_cfg else "",
                replay_size=int(trainer.buffer.size),
                save_dir=stage_dir,
                log_handle=log_handle,
            )
            eval_rho_history.append(float(eval_result.mean_rho))
            eval_service_rate_history.append(float(eval_result.mean_service_rate))

        def _ensure_eval_sample() -> None:
            if not curriculum.eval_enabled:
                return
            if eval_checkpointer is None or trainer is None:
                return
            if eval_service_rate_history:
                return
            eval_result = eval_checkpointer.evaluate(
                model=trainer.model,
                global_step=int(trainer.global_step),
                stage=spec.name,
                phase=current_phase,
                alpha=float(current_env_cfg.get("reward_potential_alpha", 0.0)) if current_env_cfg else 0.0,
                env_cfg_hash=compute_env_cfg_hash(current_env_cfg) if current_env_cfg else "",
                replay_size=int(trainer.buffer.size),
                save_dir=stage_dir,
                log_handle=log_handle,
            )
            eval_rho_history.append(float(eval_result.mean_rho))
            eval_service_rate_history.append(float(eval_result.mean_service_rate))

        def _get_gate_service_rate(stage_name: str) -> Tuple[float, str]:
            source = curriculum.stage_service_rate_gate_source.get(stage_name, curriculum.service_rate_gate_source)
            source = str(source).lower()
            train_mean = _compute_service_rate_window_mean(
                service_rate_history, curriculum.service_rate_window_size
            )
            eval_mean = _compute_service_rate_window_mean(
                eval_service_rate_history, curriculum.service_rate_window_size
            )

            if source == "train":
                return train_mean, "train"
            if source == "eval":
                if curriculum.eval_enabled and eval_service_rate_history:
                    return eval_mean, "eval"
                return train_mean, "train"
            if source == "max":
                if curriculum.eval_enabled and eval_service_rate_history:
                    return max(train_mean, eval_mean), "max"
                return train_mean, "train"
            if curriculum.eval_enabled and eval_service_rate_history:
                return eval_mean, "eval"
            return train_mean, "train"

        def _on_episode_end(ep_log: Dict[str, float]) -> bool:
            nonlocal episode_count, latest_rho, rho_history, service_rate_history, best_service_rate, best_state_dict, trainer, current_phase
            episode_count += 1
            
            # Compute metrics
            service_rate = _compute_service_rate(ep_log)
            rho = _compute_rho(ep_log, curriculum.gamma)
            latest_rho = rho
            rho_history.append(rho)
            service_rate_history.append(float(service_rate))
            
            # Compute window mean for transition decision
            service_rate_window_mean = _compute_service_rate_window_mean(
                service_rate_history, curriculum.service_rate_window_size
            )
            
            # Enhanced logging with global_step and phase info
            record = {
                "type": "episode",
                "stage": spec.name,
                "phase": current_phase,
                "episode_index": int(episode_count),
                "global_step": trainer.global_step if trainer else 0,
                "epsilon": trainer.get_epsilon() if trainer else 1.0,
                "service_rate": float(service_rate),
                "stuckness": float(ep_log.get("stuckness", 0.0)),
                "rho": float(rho),
                "service_rate_window_mean": float(service_rate_window_mean),
                "trigger_service_rate": float(curriculum.trigger_service_rate),
                "env_cfg_hash": compute_env_cfg_hash(current_env_cfg) if current_env_cfg else "",
                "replay_size": trainer.buffer.size if trainer else 0,
                "episode_log": ep_log,
            }
            for metric_key in ("fleet_density_max_mean", "fleet_density_max_p95", "stop_coverage_ratio"):
                if metric_key in ep_log:
                    record[metric_key] = float(ep_log.get(metric_key, 0.0))
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()

            if curriculum.eval_enabled:
                _run_eval_if_needed()
            
            # Update best model based on service_rate (legacy behavior)
            if trainer is not None and service_rate > best_service_rate:
                best_service_rate = float(service_rate)
                best_state_dict = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
                trainer.save_model(best_model_path)
            
            # Warning if approaching max_steps with low rho
            if service_rate_window_mean < curriculum.service_rate_warning_threshold:
                LOG.warning(
                    "Stage %s service_rate_window_mean=%.3f < warning_threshold=%.3f",
                    spec.name,
                    service_rate_window_mean,
                    curriculum.service_rate_warning_threshold,
                )
            
            # Transition check: use service_rate_window_mean (not single episode)
            if episode_count < curriculum.stage_min_episodes:
                return False
            return service_rate_window_mean >= curriculum.trigger_service_rate

        def _apply_stage_epsilon_schedule(
            trainer: DQNTrainer,
            stage_name: str,
            stage_budget: int,
        ) -> None:
            schedule = _resolve_stage_epsilon_schedule(
                stage_name=stage_name,
                schedule_cfg=epsilon_by_stage_cfg,
                default_decay_steps=int(stage_budget),
            )
            if schedule is None:
                return
            start = schedule["start"]
            if start is None:
                start = float(trainer.get_epsilon())
            trainer.set_epsilon_schedule(
                start=float(start),
                end=float(schedule["end"]),
                decay_steps=int(schedule["decay_steps"]),
                start_global_step=int(trainer.global_step),
            )
            log_handle.write(json.dumps({
                "type": "epsilon_schedule",
                "stage": stage_name,
                "global_step": trainer.global_step,
                "start": float(start),
                "end": float(schedule["end"]),
                "decay_steps": int(schedule["decay_steps"]),
            }, ensure_ascii=False) + "\n")
            log_handle.flush()

        if spec.name == "L3" and l3_two_stage_enabled:
            # L3 Multi-Phase Training: Single trainer, switch env between phases
            # This preserves global_step, replay buffer, and optimizer state
            
            # Merge: stage_output.env_overrides < stage_env_overrides_map (from YAML env_overrides)
            combined_stage_overrides = {**stage_output.env_overrides, **stage_env_overrides_map.get(spec.name, {})}
            
            # Phase1: Initialize trainer
            phase1_env_cfg = _merge_env_cfg(stage_env_base, combined_stage_overrides, phase1_overrides)
            current_env_cfg = phase1_env_cfg
            current_phase = "phase1"
            env = EventDrivenEnv(_build_env_config(phase1_env_cfg))
            graph_hashes, od_hashes = build_hashes(phase1_env_cfg)
            
            trainer = DQNTrainer(
                env=env,
                model=model,
                config=dqn_config,
                run_dir=stage_dir,
                graph_hashes=graph_hashes,
                od_hashes=od_hashes,
                global_step_init=model._global_step if hasattr(model, '_global_step') else 0,
                env_cfg=phase1_env_cfg,
                viz_config=viz_cfg,
            )
            _reset_eval_for_env(phase1_env_cfg)
            _apply_stage_epsilon_schedule(trainer, spec.name, stage_budget)
            
            # Log phase1 transition
            log_handle.write(json.dumps({
                "type": "phase_transition",
                "stage": spec.name,
                "from": None,
                "to": "phase1",
                "global_step": trainer.global_step,
                "epsilon": trainer.get_epsilon(),
                "replay_size": trainer.buffer.size,
                "trainer_reinit": True,
                "env_cfg_hash": compute_env_cfg_hash(phase1_env_cfg),
                "env_overrides": phase1_overrides or {},
            }, ensure_ascii=False) + "\n")
            log_handle.flush()
            
            # Behavior cloning (optional)
            if bc_enabled and bc_steps > 0:
                log_handle.write(json.dumps({
                    "type": "stage_phase",
                    "stage": spec.name,
                    "phase": "behavior_cloning",
                    "steps": int(bc_steps),
                }, ensure_ascii=False) + "\n")
                log_handle.flush()
                _behavior_clone(
                    trainer=trainer,
                    env=env,
                    steps=int(bc_steps),
                    log_handle=log_handle,
                    log_every=int(bc_log_every),
                    congestion_weight=bc_congestion_weight,
                    coverage_weight=bc_coverage_weight,
                )
            
            # Phase1 training
            if phase1_steps > 0:
                trainer.train(total_steps=int(phase1_steps), episode_callback=_on_episode_end)
            
            phase2_env_cfg = _merge_env_cfg(stage_env_base, combined_stage_overrides, phase2_overrides)

            # Phase2: Switch env WITHOUT recreating trainer
            if phase2_steps > 0:
                current_env_cfg = phase2_env_cfg
                current_phase = "phase2"
                
                # Create new env and attach to existing trainer
                new_env = EventDrivenEnv(_build_env_config(phase2_env_cfg))
                replay_size_before = trainer.buffer.size
                trainer._attach_env(new_env, phase2_env_cfg)
                _reset_eval_for_env(phase2_env_cfg)
                
                # Log phase2 transition
                log_handle.write(json.dumps({
                    "type": "phase_transition",
                    "stage": spec.name,
                    "from": "phase1",
                    "to": "phase2",
                    "global_step": trainer.global_step,
                    "epsilon": trainer.get_epsilon(),
                    "replay_size_before": replay_size_before,
                    "replay_size_after": trainer.buffer.size,
                    "trainer_reinit": False,
                    "env_cfg_hash": compute_env_cfg_hash(phase2_env_cfg),
                    "env_overrides": phase2_overrides or {},
                }, ensure_ascii=False) + "\n")
                log_handle.flush()
                
                trainer.train(total_steps=int(phase2_steps), episode_callback=_on_episode_end)
            
            # Phase3: Switch env again, with reward ramp support
            if phase3_steps > 0:
                # Build phase3 config with fallback to phase2
                if not phase3_overrides or len(phase3_overrides) == 0:
                    LOG.info("Phase3 overrides empty, inheriting from phase2")
                    effective_phase3_overrides = phase2_overrides
                else:
                    effective_phase3_overrides = phase3_overrides
                
                phase3_env_cfg = _merge_env_cfg(stage_env_base, combined_stage_overrides, effective_phase3_overrides)
                current_env_cfg = phase3_env_cfg
                current_phase = "phase3"
                
                # Create new env and attach to existing trainer
                new_env = EventDrivenEnv(_build_env_config(phase3_env_cfg))
                replay_size_before = trainer.buffer.size
                trainer._attach_env(new_env, phase3_env_cfg)
                _reset_eval_for_env(phase3_env_cfg)

                ramp_config = _build_ramp_config_from_envs(
                    phase2_env_cfg=phase2_env_cfg,
                    phase3_env_cfg=phase3_env_cfg,
                    reward_ramp_steps=int(curriculum.reward_ramp_steps),
                    ramp_fields=list(curriculum.ramp_fields),
                )
                initial_ramped, _ = compute_ramped_weights(0, ramp_config)
                for field_name, value in initial_ramped.items():
                    setattr(new_env.config, field_name, float(value))
                    trainer._env_cfg[field_name] = float(value)

                def _on_phase3_step(_global_step: int, local_step: int, env_obj: EventDrivenEnv) -> None:
                    nonlocal phase_step
                    phase_step = int(local_step)
                    ramped, _alpha = compute_ramped_weights(phase_step, ramp_config)
                    for field_name, value in ramped.items():
                        setattr(env_obj.config, field_name, float(value))
                        trainer._env_cfg[field_name] = float(value)

                # Log phase3 transition
                log_handle.write(json.dumps({
                    "type": "phase_transition",
                    "stage": spec.name,
                    "from": "phase2",
                    "to": "phase3",
                    "global_step": trainer.global_step,
                    "epsilon": trainer.get_epsilon(),
                    "replay_size_before": replay_size_before,
                    "replay_size_after": trainer.buffer.size,
                    "trainer_reinit": False,
                    "env_cfg_hash": compute_env_cfg_hash(phase3_env_cfg),
                    "env_overrides": effective_phase3_overrides or {},
                    "phase3_inherited_from_phase2": (not phase3_overrides or len(phase3_overrides) == 0),
                }, ensure_ascii=False) + "\n")
                log_handle.flush()
                
                trainer.train(
                    total_steps=int(phase3_steps),
                    episode_callback=_on_episode_end,
                    step_callback=_on_phase3_step,
                )
            
            # L3 Extension loop: check service_rate after all phases complete
            extension_count = 0
            stage_passed = False
            forced_transition = False
            require_service_rate_transition = bool(
                curriculum.stage_require_service_rate.get(spec.name, curriculum.require_service_rate_transition)
            )
            
            while not stage_passed:
                _ensure_eval_sample()
                service_rate_window_mean, service_rate_source = _get_gate_service_rate(spec.name)
                
                if (
                    (len(eval_service_rate_history) if service_rate_source == "eval" else len(service_rate_history))
                    >= curriculum.stage_min_episodes
                    and service_rate_window_mean >= curriculum.trigger_service_rate
                ):
                    LOG.info(
                        "Stage L3 PASSED: service_rate_window_mean=%.4f >= trigger=%.4f (source=%s)",
                        service_rate_window_mean,
                        curriculum.trigger_service_rate,
                        service_rate_source,
                    )
                    stage_passed = True
                    break
                
                # Not passed - check if we should extend or fail
                if not require_service_rate_transition:
                    LOG.warning(
                        "Stage L3 FORCED transition: service_rate_window_mean=%.4f < trigger=%.4f (source=%s)",
                        service_rate_window_mean,
                        curriculum.trigger_service_rate,
                        service_rate_source,
                    )
                    forced_transition = True
                    stage_passed = True
                    break
                
                extension_count += 1
                if extension_count > curriculum.max_stage_extensions:
                    if curriculum.fail_policy == "fail_fast":
                        LOG.error(
                            "Stage L3 FAILED: service_rate_window_mean=%.4f < trigger=%.4f after %d extensions (source=%s)",
                            service_rate_window_mean,
                            curriculum.trigger_service_rate,
                            extension_count - 1,
                            service_rate_source,
                        )
                        log_handle.write(json.dumps({
                            "type": "stage_failed",
                            "stage": "L3",
                            "service_rate_window_mean": float(service_rate_window_mean),
                            "trigger_service_rate": float(curriculum.trigger_service_rate),
                            "service_rate_source": service_rate_source,
                            "extensions": extension_count - 1,
                        }, ensure_ascii=False) + "\n")
                        log_handle.flush()
                        raise StageFailedError("L3", service_rate_window_mean, curriculum.trigger_service_rate)
                    else:
                        LOG.warning("Stage L3 FORCED transition after max extensions")
                        forced_transition = True
                        stage_passed = True
                        break
                
                # Log and run extension
                LOG.warning(
                    "Stage L3 extension %d/%d: service_rate_window_mean=%.4f < trigger=%.4f (source=%s)",
                    extension_count,
                    curriculum.max_stage_extensions,
                    service_rate_window_mean,
                    curriculum.trigger_service_rate,
                    service_rate_source,
                )
                log_handle.write(json.dumps({
                    "type": "stage_extension",
                    "stage": "L3",
                    "extension": extension_count,
                    "service_rate_window_mean": float(service_rate_window_mean),
                    "service_rate_source": service_rate_source,
                    "additional_steps": curriculum.stage_extension_steps,
                }, ensure_ascii=False) + "\n")
                log_handle.flush()
                
                trainer.train(total_steps=int(curriculum.stage_extension_steps), episode_callback=_on_episode_end)
            
            # Store final global_step on model for next stage
            model._global_step = trainer.global_step
            trainer.close()
        else:
            # Non-L3 stage training with extension loop
            # Merge: stage_output.env_overrides < stage_env_overrides_map (from YAML env_overrides)
            combined_stage_overrides = {**stage_output.env_overrides, **stage_env_overrides_map.get(spec.name, {})}
            stage_env_cfg = _merge_env_cfg(stage_env_base, combined_stage_overrides, None)
            current_env_cfg = stage_env_cfg
            current_phase = "main"
            env = EventDrivenEnv(_build_env_config(stage_env_cfg))
            graph_hashes, od_hashes = build_hashes(stage_env_cfg)
            trainer = DQNTrainer(
                env=env,
                model=model,
                config=dqn_config,
                run_dir=stage_dir,
                graph_hashes=graph_hashes,
                od_hashes=od_hashes,
                global_step_init=model._global_step if hasattr(model, '_global_step') else 0,
                env_cfg=stage_env_cfg,
                viz_config=viz_cfg,
            )
            _reset_eval_for_env(stage_env_cfg)
            _apply_stage_epsilon_schedule(trainer, spec.name, stage_budget)
            
            # Extension loop for service-rate-gated transitions
            extension_count = 0
            current_budget = int(stage_budget)
            stage_passed = False
            forced_transition = False
            require_service_rate_transition = bool(
                curriculum.stage_require_service_rate.get(spec.name, curriculum.require_service_rate_transition)
            )
            
            while not stage_passed:
                # Train for current budget
                trainer.train(total_steps=current_budget, episode_callback=_on_episode_end)
                
                _ensure_eval_sample()
                service_rate_window_mean, service_rate_source = _get_gate_service_rate(spec.name)
                
                if (
                    (len(eval_service_rate_history) if service_rate_source == "eval" else len(service_rate_history))
                    >= curriculum.stage_min_episodes
                    and service_rate_window_mean >= curriculum.trigger_service_rate
                ):
                    LOG.info(
                        "Stage %s PASSED: service_rate_window_mean=%.4f >= trigger=%.4f (source=%s)",
                        spec.name,
                        service_rate_window_mean,
                        curriculum.trigger_service_rate,
                        service_rate_source,
                    )
                    stage_passed = True
                    break
                
                # Not passed - check if we should extend or fail
                if not require_service_rate_transition:
                    LOG.warning(
                        "Stage %s FORCED transition: service_rate_window_mean=%.4f < trigger=%.4f (source=%s, require_service_rate_transition=False)",
                        spec.name,
                        service_rate_window_mean,
                        curriculum.trigger_service_rate,
                        service_rate_source,
                    )
                    forced_transition = True
                    stage_passed = True
                    break
                
                extension_count += 1
                if extension_count > curriculum.max_stage_extensions:
                    if curriculum.fail_policy == "fail_fast":
                        LOG.error(
                            "Stage %s FAILED: service_rate_window_mean=%.4f < trigger=%.4f after %d extensions (source=%s)",
                            spec.name,
                            service_rate_window_mean,
                            curriculum.trigger_service_rate,
                            extension_count - 1,
                            service_rate_source,
                        )
                        # Log failure event
                        log_handle.write(json.dumps({
                            "type": "stage_failed",
                            "stage": spec.name,
                            "service_rate_window_mean": float(service_rate_window_mean),
                            "trigger_service_rate": float(curriculum.trigger_service_rate),
                            "service_rate_source": service_rate_source,
                            "extensions": extension_count - 1,
                            "fail_policy": curriculum.fail_policy,
                        }, ensure_ascii=False) + "\n")
                        log_handle.flush()
                        raise StageFailedError(spec.name, service_rate_window_mean, curriculum.trigger_service_rate)
                    else:
                        # forced policy
                        LOG.warning(
                            "Stage %s FORCED transition after max extensions: service_rate_window_mean=%.4f < trigger=%.4f",
                            spec.name,
                            service_rate_window_mean,
                            curriculum.trigger_service_rate,
                        )
                        forced_transition = True
                        stage_passed = True
                        break
                
                # Log extension
                LOG.warning(
                    "Stage %s extension %d/%d: service_rate_window_mean=%.4f < trigger=%.4f (source=%s), adding %d steps",
                    spec.name,
                    extension_count,
                    curriculum.max_stage_extensions,
                    service_rate_window_mean,
                    curriculum.trigger_service_rate,
                    service_rate_source,
                    curriculum.stage_extension_steps,
                )
                log_handle.write(json.dumps({
                    "type": "stage_extension",
                    "stage": spec.name,
                    "extension": extension_count,
                    "max_extensions": curriculum.max_stage_extensions,
                    "service_rate_window_mean": float(service_rate_window_mean),
                    "trigger_service_rate": float(curriculum.trigger_service_rate),
                    "service_rate_source": service_rate_source,
                    "additional_steps": curriculum.stage_extension_steps,
                }, ensure_ascii=False) + "\n")
                log_handle.flush()
                
                current_budget = int(curriculum.stage_extension_steps)
            
            model._global_step = trainer.global_step
            trainer.close()
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            torch.save(best_state_dict, best_model_path)
        write_run_meta(
            run_dir,
            model_path_final=(stage_dir / "edgeq_model_final.pt") if (stage_dir / "edgeq_model_final.pt").exists() else None,
            model_path_latest=(stage_dir / "edgeq_model_latest.pt") if (stage_dir / "edgeq_model_latest.pt").exists() else None,
            extra={"stage": spec.name, "stage_dir": str(stage_dir)},
        )
        # Compute final service_rate_window_mean for transition log
        _ensure_eval_sample()
        final_service_rate_window_mean, final_service_rate_source = _get_gate_service_rate(spec.name)
        transition = {
            "type": "stage_transition",
            "from_stage": spec.name,
            "to_stage": all_specs[current_stage_idx + 1].name if current_stage_idx + 1 < len(all_specs) else None,
            "stage_index": int(current_stage_idx),
            "episodes": int(episode_count),
            "last_rho": float(latest_rho),
            "service_rate_window_mean": float(final_service_rate_window_mean),
            "trigger_service_rate": float(curriculum.trigger_service_rate),
            "service_rate_source": final_service_rate_source,
            "passed": final_service_rate_window_mean >= curriculum.trigger_service_rate,
            "forced_transition": 'forced_transition' in dir() and forced_transition,
        }
        log_handle.write(json.dumps(transition, ensure_ascii=False) + "\n")
        log_handle.flush()

    log_handle.close()
    return curriculum_log


def run_stress_tests(config_path: str | Path, run_dir: Optional[Path] = None) -> Path:
    cfg = load_config(config_path)
    env_cfg = cfg.get("env", {})
    model_cfg = cfg.get("model", {})

    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("reports") / "runs" / f"stress_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_od = load_od_frames(env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet"))
    nodes = load_nodes(env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"))
    scenarios = stress_stages()
    metrics: List[Dict[str, float]] = []

    for spec in scenarios:
        scenario_dir = run_dir / f"scenario_{spec.name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        stage_output = generate_stage(
            base_od=base_od,
            nodes=nodes,
            stage=spec,
            output_dir=scenario_dir,
            seed=int(env_cfg.get("seed", 7)),
        )
        scenario_env_cfg = dict(env_cfg)
        scenario_env_cfg["od_glob"] = str(stage_output.od_path)
        scenario_env_cfg.update(stage_output.env_overrides)
        env = EventDrivenEnv(_build_env_config(scenario_env_cfg))

        total_reward = 0.0
        total_tacc = 0.0
        stuck_sum = 0.0
        steps = 0
        done = False
        info: Dict[str, float] = {}
        while not done:
            features = env.get_feature_batch()
            mask = features["action_mask"].astype(bool)
            step_stuckness = float((~mask).mean()) if len(mask) else 1.0
            stuck_sum += step_stuckness
            action = _greedy_policy(features)
            if action is None:
                break
            _, reward, done, info = env.step(int(action))
            total_reward += float(reward)
            total_tacc += float(info.get("step_tacc_gain", 0.0))
            steps += 1

        served = float(info.get("served", 0.0)) if done else float(env.served)
        waiting_churned = float(info.get("waiting_churned", 0.0)) if done else float(env.waiting_churned)
        onboard_churned = float(info.get("onboard_churned", 0.0)) if done else float(env.onboard_churned)
        structural = float(info.get("structural_unserviceable", 0.0)) if done else float(env.structurally_unserviceable)
        waiting_timeouts = float(info.get("waiting_timeouts", 0.0)) if done else float(env.waiting_timeouts)
        eligible = served + waiting_churned + onboard_churned + waiting_timeouts
        service_rate = float(served / eligible) if eligible > 0 else 0.0
        stuckness = float(stuck_sum / max(1, steps))

        metrics.append(
            {
                "scenario": spec.name,
                "description": spec.description,
                "steps": float(steps),
                "episode_return": float(total_reward),
                "served": float(served),
                "waiting_churned": float(waiting_churned),
                "onboard_churned": float(onboard_churned),
                "structural_unserviceable": float(structural),
                "waiting_timeouts": float(waiting_timeouts),
                "algorithmic_churned": float(waiting_churned + onboard_churned),
                "service_rate": float(service_rate),
                "stuckness": float(stuckness),
                "tacc_total": float(total_tacc),
                "service_gini": float(info.get("service_gini", 0.0)) if done else 0.0,
            }
        )

    metrics_path = run_dir / "stress_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="ascii")
    metrics_csv = run_dir / "stress_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False)
    return metrics_path
