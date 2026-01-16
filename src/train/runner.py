"""Training runner with structured curriculum and rho-based transitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import time

import numpy as np
import pandas as pd
import torch

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.models.edge_q_gnn import EdgeQGNN
from src.train.curriculum import StageSpec, default_stages, generate_stage, load_nodes, load_od_frames, stress_stages
from src.train.dqn import DQNConfig, DQNTrainer, build_hashes, write_run_meta
from src.utils.config import load_config

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CurriculumConfig:
    stages: List[str]
    trigger_rho: float
    gamma: float
    stage_max_steps: int
    stage_min_episodes: int


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
        debug_mask=bool(env_cfg.get("debug_mask", False)),
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
    )


def _stage_specs_from_config(curriculum_cfg: Dict[str, Any]) -> List[StageSpec]:
    defaults = {spec.name: spec for spec in default_stages()}
    names = curriculum_cfg.get("stages", list(defaults.keys()))
    stage_params = curriculum_cfg.get("stage_params", {})
    specs: List[StageSpec] = []
    for name in names:
        base = defaults.get(name, StageSpec(name=name, description="Custom curriculum stage"))
        overrides = stage_params.get(name, {})
        params = {**base.__dict__, **overrides}
        params["name"] = name
        specs.append(StageSpec(**params))
    return specs


def _compute_service_rate(log: Dict[str, float]) -> float:
    served = float(log.get("served", 0.0))
    waiting_churned = float(log.get("waiting_churned", 0.0))
    onboard_churned = float(log.get("onboard_churned", 0.0))
    waiting_timeouts = float(log.get("waiting_timeouts", 0.0))
    eligible = served + waiting_churned + onboard_churned + waiting_timeouts
    if eligible <= 0:
        return 0.0
    return served / eligible


def _greedy_policy(features: Dict[str, np.ndarray]) -> Optional[int]:
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

    curriculum = CurriculumConfig(
        stages=list(curriculum_cfg.get("stages", ["L0", "L1", "L2", "L3"])),
        trigger_rho=float(curriculum_cfg.get("trigger_rho", 0.8)),
        gamma=float(curriculum_cfg.get("gamma", 1.0)),
        stage_max_steps=int(curriculum_cfg.get("stage_max_steps", 50_000)),
        stage_min_episodes=int(curriculum_cfg.get("stage_min_episodes", 3)),
    )
    stage_specs = _stage_specs_from_config(curriculum_cfg)

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
    )

    model = EdgeQGNN(
        node_dim=int(model_cfg.get("node_dim", 5)),
        edge_dim=int(model_cfg.get("edge_dim", 4)),
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
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

    for local_idx, spec in enumerate(run_specs):
        current_stage_idx = start_index + local_idx
        stage_dir = run_dir / f"stage_{spec.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_output = generate_stage(
            base_od=base_od,
            nodes=nodes,
            stage=spec,
            output_dir=stage_dir,
            seed=int(dqn_config.seed),
        )
        stage_env_cfg = dict(env_cfg)
        stage_env_cfg["od_glob"] = str(stage_output.od_path)
        stage_env_cfg.update(stage_output.env_overrides)
        env = EventDrivenEnv(_build_env_config(stage_env_cfg))
        graph_hashes, od_hashes = build_hashes(stage_env_cfg)
        trainer = DQNTrainer(env=env, model=model, config=dqn_config, run_dir=stage_dir, graph_hashes=graph_hashes, od_hashes=od_hashes)

        episode_count = 0
        latest_rho = 0.0

        def _on_episode_end(ep_log: Dict[str, float]) -> bool:
            nonlocal episode_count, latest_rho
            episode_count += 1
            service_rate = _compute_service_rate(ep_log)
            stuckness = float(ep_log.get("stuckness", 0.0))
            rho = float(service_rate / (1.0 + curriculum.gamma * stuckness))
            latest_rho = rho
            record = {
                "type": "episode",
                "stage": spec.name,
                "episode_index": int(episode_count),
                "service_rate": float(service_rate),
                "stuckness": float(stuckness),
                "rho": float(rho),
                "trigger_rho": float(curriculum.trigger_rho),
                "min_episodes": int(curriculum.stage_min_episodes),
                "episode_log": ep_log,
            }
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()
            if episode_count < curriculum.stage_min_episodes:
                return False
            return rho >= curriculum.trigger_rho

        trainer.train(total_steps=int(curriculum.stage_max_steps), episode_callback=_on_episode_end)
        trainer.close()
        write_run_meta(
            run_dir,
            model_path_final=(stage_dir / "edgeq_model_final.pt") if (stage_dir / "edgeq_model_final.pt").exists() else None,
            model_path_latest=(stage_dir / "edgeq_model_latest.pt") if (stage_dir / "edgeq_model_latest.pt").exists() else None,
            extra={"stage": spec.name, "stage_dir": str(stage_dir)},
        )
        transition = {
            "type": "stage_transition",
            "from_stage": spec.name,
            "to_stage": all_specs[current_stage_idx + 1].name if current_stage_idx + 1 < len(all_specs) else None,
            "stage_index": int(current_stage_idx),
            "episodes": int(episode_count),
            "last_rho": float(latest_rho),
            "trigger_rho": float(curriculum.trigger_rho),
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
