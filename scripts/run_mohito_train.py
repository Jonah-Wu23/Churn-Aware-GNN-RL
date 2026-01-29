#!/usr/bin/env python
"""MOHITO training script for in-domain training.

This script trains MOHITO from random initialization using the same
EventDrivenEnv, reward function, and training protocol as other baselines.

Usage:
    python scripts/run_mohito_train.py --config configs/manhattan.yaml
    python scripts/run_mohito_train.py --config configs/manhattan.yaml --total-steps 1000  # smoke test
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.train.mohito_trainer import MOHITOTrainer, MOHITOTrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MOHITO baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/manhattan.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override run directory",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override total training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_env_config(config: Dict[str, Any], seed: int) -> EnvConfig:
    """Build EnvConfig from YAML config with training split."""
    env_cfg = config.get("env", {})
    paths_cfg = config.get("paths", {})
    mohito_cfg = config.get("mohito_train", {})
     
    return EnvConfig(
        max_horizon_steps=int(env_cfg.get("max_horizon_steps", 2000)),
        mask_alpha=float(env_cfg.get("mask_alpha", 1.5)),
        walk_threshold_sec=int(env_cfg.get("walk_threshold_sec", 600)),
        max_requests=int(env_cfg.get("max_requests", 1500)),
        seed=seed,
        num_vehicles=int(mohito_cfg.get("num_vehicles", env_cfg.get("num_vehicles", 51))),
        vehicle_capacity=int(env_cfg.get("vehicle_capacity", 6)),
        request_timeout_sec=int(env_cfg.get("request_timeout_sec", 600)),
        debug_mask=bool(env_cfg.get("debug_mask", False)),
        waiting_churn_tol_sec=int(env_cfg.get("waiting_churn_tol_sec", 300)),
        waiting_churn_beta=float(env_cfg.get("waiting_churn_beta", 0.02)),
        onboard_churn_tol_sec=int(env_cfg.get("onboard_churn_tol_sec", 300)),
        onboard_churn_beta=float(env_cfg.get("onboard_churn_beta", 0.02)),
        reward_service=float(env_cfg.get("reward_service", 1.0)),
        reward_waiting_churn_penalty=float(env_cfg.get("reward_waiting_churn_penalty", 1.0)),
        reward_onboard_churn_penalty=float(env_cfg.get("reward_onboard_churn_penalty", 1.0)),
        reward_travel_cost_per_sec=float(env_cfg.get("reward_travel_cost_per_sec", 0.0)),
        reward_tacc_weight=float(env_cfg.get("reward_tacc_weight", 0.01)),
        reward_onboard_delay_weight=float(env_cfg.get("reward_onboard_delay_weight", 0.1)),
        reward_cvar_penalty=float(env_cfg.get("reward_cvar_penalty", 1.0)),
        reward_fairness_weight=float(env_cfg.get("reward_fairness_weight", 1.0)),
        cvar_alpha=float(env_cfg.get("cvar_alpha", 0.95)),
        fairness_gamma=float(env_cfg.get("fairness_gamma", 1.0)),
        od_glob=str(env_cfg.get("od_glob", paths_cfg.get("od_output_dir", "data/processed/od_mapped") + "/*.parquet")),
        graph_nodes_path=str(env_cfg.get("graph_nodes_path", paths_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"))),
        graph_edges_path=str(env_cfg.get("graph_edges_path", paths_cfg.get("graph_edges_path", "data/processed/graph/layer2_edges.parquet"))),
        graph_embeddings_path=str(env_cfg.get("graph_embeddings_path", paths_cfg.get("graph_embeddings_path", "data/processed/graph/node2vec_embeddings.parquet"))),
        # CRITICAL: Use training split
        time_split_mode="train",
        time_split_ratio=float(env_cfg.get("time_split_ratio", 0.3)),
    )


def build_train_config(config: Dict[str, Any], args) -> MOHITOTrainConfig:
    """Build training config from YAML and CLI args."""
    train_cfg = config.get("train", {})
    mohito_cfg = config.get("mohito_train", config.get("eval", {}).get("mohito", {}))
    
    return MOHITOTrainConfig(
        seed=args.seed if args.seed is not None else int(train_cfg.get("seed", 7)),
        total_steps=args.total_steps if args.total_steps is not None else int(train_cfg.get("total_steps", 200000)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        learning_rate=float(mohito_cfg.get("learning_rate", mohito_cfg.get("lr_actor", 0.001))),
        entropy_coef=float(mohito_cfg.get("entropy_coef", mohito_cfg.get("beta", 0.01))),
        value_loss_coef=float(mohito_cfg.get("value_loss_coef", 0.5)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 5.0)),
        log_every_steps=int(train_cfg.get("log_every_steps", 1000)),
        eval_every_steps=int(mohito_cfg.get("eval_every_steps", 10000)),
        checkpoint_every_steps=int(train_cfg.get("checkpoint_every_steps", 10000)),
        device=args.device if args.device is not None else str(train_cfg.get("device", "cuda")),
        feature_len=int(mohito_cfg.get("feature_len", 5)),
        num_layers_actor=int(mohito_cfg.get("num_layers_actor", 20)),
        hidden_dim=int(mohito_cfg.get("hidden_dim", 50)),
        heads=int(mohito_cfg.get("heads", 2)),
        grid_size=int(mohito_cfg.get("grid_size", 10)),
        update_every_steps=int(mohito_cfg.get("update_every_steps", 64)),
        graph_mode=str(mohito_cfg.get("graph_mode", "compact")),
        use_amp=bool(mohito_cfg.get("use_amp", True)),
        amp_dtype=str(mohito_cfg.get("amp_dtype", "fp16")),
    )


def main():
    """Main training function."""
    args = parse_args()
    
    LOG.info("=" * 60)
    LOG.info("MOHITO IN-DOMAIN TRAINING")
    LOG.info("=" * 60)
    LOG.info(f"Config: {args.config}")
    
    # Load config
    config = load_config(args.config)
    
    # Build training config
    train_config = build_train_config(config, args)
    LOG.info(f"Training config: {train_config}")
    
    # Setup run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = PROJECT_ROOT / "reports" / "mohito_train" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Run directory: {run_dir}")
    
    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Build environment with TRAINING split
    env_config = build_env_config(config, train_config.seed)
    LOG.info(f"Environment: num_vehicles={env_config.num_vehicles}, capacity={env_config.vehicle_capacity}")
    LOG.info(f"Data split: mode={env_config.time_split_mode}, ratio={env_config.time_split_ratio}")
    
    env = EventDrivenEnv(env_config)
    
    # Build trainer
    trainer = MOHITOTrainer(
        env=env,
        config=train_config,
        run_dir=run_dir,
        env_cfg=config.get("env", {}),
    )
    
    # Run training
    try:
        results = trainer.train(total_steps=train_config.total_steps)
        LOG.info("Training completed successfully!")
        LOG.info(f"Results: {results}")
    except KeyboardInterrupt:
        LOG.warning("Training interrupted by user")
    finally:
        trainer.close()
    
    LOG.info("=" * 60)
    LOG.info(f"Model saved to: {run_dir}")
    LOG.info("=" * 60)


if __name__ == "__main__":
    main()
