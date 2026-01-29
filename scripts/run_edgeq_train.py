"""CLI entrypoint for EdgeQ DQN training (no baselines)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.models.edge_q_gnn import EdgeQGNN
from src.train.dqn import DQNConfig, DQNTrainer, build_hashes
from src.train.runner import run_curriculum_training
from src.utils.config import load_config
from src.utils.build_info import get_build_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--start-stage", default=None)
    parser.add_argument("--init-model-path", default=None)
    return parser.parse_args()


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
    )


def _build_dqn_config(train_cfg: Dict[str, Any], env_cfg: Dict[str, Any]) -> DQNConfig:
    return DQNConfig(
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


def _build_model(model_cfg: Dict[str, Any]) -> EdgeQGNN:
    return EdgeQGNN(
        node_dim=int(model_cfg.get("node_dim", 5)),
        edge_dim=int(model_cfg.get("edge_dim", 4)),
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def _use_curriculum(cfg: Dict[str, Any]) -> bool:
    curriculum_cfg = cfg.get("curriculum", {})
    stages = curriculum_cfg.get("stages")
    if isinstance(stages, list) and len(stages) > 0:
        return True
    return False


def _run_single_training(cfg: Dict[str, Any], run_dir: Path) -> Path:
    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    env = EventDrivenEnv(_build_env_config(env_cfg))
    model = _build_model(model_cfg)
    dqn_config = _build_dqn_config(train_cfg, env_cfg)
    model.to(torch.device(dqn_config.device))

    graph_hashes, od_hashes = build_hashes(env_cfg)
    trainer = DQNTrainer(
        env=env, model=model, config=dqn_config, run_dir=run_dir,
        graph_hashes=graph_hashes, od_hashes=od_hashes, env_cfg=env_cfg,
        viz_config=train_cfg.get("viz") if isinstance(train_cfg, dict) else None,
    )
    log_path = trainer.train()
    trainer.close()
    return log_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    build_id = get_build_id()
    logging.info("BUILD_ID=%s", build_id)

    cfg = load_config(args.config)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("reports") / "runs" / f"edgeq_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if _use_curriculum(cfg):
        log_path = run_curriculum_training(
            args.config,
            run_dir=run_dir,
            start_stage=args.start_stage,
            init_model_path=args.init_model_path,
        )
        logging.info("Curriculum training finished: %s", log_path)
    else:
        log_path = _run_single_training(cfg, run_dir)
        logging.info("Training finished: %s", log_path)


if __name__ == "__main__":
    main()
