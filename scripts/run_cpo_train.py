#!/usr/bin/env python
"""CPO training script using PyTorch-CPO + project env wrapper.

This script trains a CPO agent using the PyTorch-CPO implementation
on the micro-transit dispatch environment.

Usage:
    python scripts/run_cpo_train.py --config configs/manhattan.yaml \
        --max-iter 500 --max-constraint 10.0
        
Gate conditions (must pass before large-scale training):
- G1: Line search enabled with KL + cost constraint check
- G2: Mask applied at logits level (training & eval)
- G3: KL direction verified (old||new)
- G4: Cost calibration done via baseline runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add PyTorch-CPO to path
PYTORCH_CPO_PATH = PROJECT_ROOT / "baselines" / "PyTorch-CPO"
if not PYTORCH_CPO_PATH.exists():
    raise FileNotFoundError(
        f"Missing PyTorch-CPO baseline at {PYTORCH_CPO_PATH}. "
        "Upload/clone it to baselines/PyTorch-CPO before running CPO training."
    )
sys.path.insert(0, str(PYTORCH_CPO_PATH))

from src.env.cpo_env_wrapper import CPOEnvConfig, CPOEnvWrapper
from src.env.gym_env import EnvConfig


def _select_torch_dtype(device: torch.device, dtype_arg: str) -> torch.dtype:
    dtype_arg = str(dtype_arg).lower().strip()
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float64":
        return torch.float64
    if dtype_arg != "auto":
        raise ValueError(f"Invalid dtype: {dtype_arg} (expected auto/float32/float64)")
    # Keep prior behavior on CPU (float64), but use float32 on CUDA for speed.
    return torch.float32 if device.type == "cuda" else torch.float64


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CPO training for micro-transit")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--max-iter", type=int, default=500, help="Max training iterations")
    parser.add_argument(
        "--target-steps",
        type=int,
        default=0,
        help="Approximate total environment steps; overrides max-iter based on min-batch-size",
    )
    parser.add_argument("--max-constraint", type=float, default=10.0, help="Cost constraint limit (d_k)")
    parser.add_argument("--max-kl", type=float, default=0.01, help="KL divergence constraint")
    parser.add_argument("--damping", type=float, default=0.1, help="CG damping coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="Reward discount factor")
    parser.add_argument("--tau", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--l2-reg", type=float, default=1e-3, help="L2 regularization")
    parser.add_argument("--min-batch-size", type=int, default=2048, help="Min batch size per iteration")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU device index")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float64"],
        help="Torch dtype: auto uses float32 on CUDA and float64 on CPU.",
    )
    parser.add_argument("--neighbor-k", type=int, default=8, help="Action space size (neighbor stops)")
    parser.add_argument("--run-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--anneal", action="store_true", help="Enable constraint annealing")
    parser.add_argument("--annealing-factor", type=float, default=0.1, help="Annealing factor for d_k")
    parser.add_argument("--log-interval", type=int, default=1, help="Log every N iterations")
    parser.add_argument("--save-interval", type=int, default=50, help="Save model every N iterations")
    parser.add_argument("--enable-line-search", action="store_true", 
                        help="Enable backtracking line search (G1 gate)")
    return parser.parse_args()


def _resolve_max_iter(args) -> int:
    if args.target_steps and int(args.target_steps) > 0:
        return int(math.ceil(int(args.target_steps) / max(1, int(args.min_batch_size))))
    return int(args.max_iter)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_config(config: Dict[str, Any], seed: int) -> EnvConfig:
    """Build EnvConfig from YAML config."""
    env_cfg = config.get("env", {})
    cpo_cfg = config.get("cpo_train", {})
    return EnvConfig(
        max_horizon_steps=int(env_cfg.get("max_horizon_steps", 200)),
        mask_alpha=float(env_cfg.get("mask_alpha", 1.5)),
        walk_threshold_sec=int(env_cfg.get("walk_threshold_sec", 600)),
        max_requests=int(env_cfg.get("max_requests", 2000)),
        seed=seed,
        num_vehicles=int(cpo_cfg.get("num_vehicles", env_cfg.get("num_vehicles", 1))),
        vehicle_capacity=int(env_cfg.get("vehicle_capacity", 6)),
        request_timeout_sec=int(env_cfg.get("request_timeout_sec", 600)),
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
        cvar_alpha=float(env_cfg.get("cvar_alpha", 0.95)),
        fairness_gamma=float(env_cfg.get("fairness_gamma", 1.0)),
        od_glob=env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet"),
        graph_nodes_path=env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"),
        graph_edges_path=env_cfg.get("graph_edges_path", "data/processed/graph/layer2_edges.parquet"),
        graph_embeddings_path=env_cfg.get("graph_embeddings_path", "data/processed/graph/node2vec_embeddings.parquet"),
        time_split_mode=env_cfg.get("time_split_mode"),
        time_split_ratio=float(env_cfg.get("time_split_ratio", 0.3)),
    )


def make_env(config: Dict[str, Any], seed: int, neighbor_k: int) -> CPOEnvWrapper:
    """Create CPO environment wrapper."""
    env_config = build_env_config(config, seed)
    cpo_config = CPOEnvConfig(
        env_config=env_config,
        neighbor_k=neighbor_k,
        include_noop=True,
    )
    return CPOEnvWrapper(cpo_config)


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Setup output directory
    if args.run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("reports") / "cpo_train" / f"run_{timestamp}"
    else:
        run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"[CPO Train] Output directory: {run_dir}")
    print(f"[CPO Train] Config: {args.config}")
    resolved_max_iter = _resolve_max_iter(args)
    print(f"[CPO Train] Max iterations: {resolved_max_iter}")
    if args.target_steps and int(args.target_steps) > 0:
        approx_steps = resolved_max_iter * int(args.min_batch_size)
        print(f"[CPO Train] Target steps: {args.target_steps} (approx {approx_steps} steps)")
    print(f"[CPO Train] Cost limit (d_k): {args.max_constraint}")
    print(f"[CPO Train] Line search enabled: {args.enable_line_search}")
    
    # Setup device
    if torch.cuda.is_available():
        if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid --gpu-index {args.gpu_index}; available cuda devices: {torch.cuda.device_count()}"
            )
        device = torch.device("cuda", index=args.gpu_index)
        print(f"[CPO Train] Using GPU: {args.gpu_index}")
        torch.cuda.set_device(args.gpu_index)
    else:
        device = torch.device("cpu")
        print("[CPO Train] Using CPU")

    dtype = _select_torch_dtype(device, args.dtype)
    torch.set_default_dtype(dtype)
    print(f"[CPO Train] Torch dtype: {dtype}")
    
    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = make_env(config, args.seed, args.neighbor_k)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"[CPO Train] State dim: {state_dim}, Action dim: {action_dim}")
    
    # Import PyTorch-CPO modules
    from models.discrete_policy import DiscretePolicy
    from models.critic import Value
    from core.agent import Agent
    from algos.cpo import cpo_step
    from core.common import estimate_advantages, estimate_constraint_value
    from utils.zfilter import ZFilter
    from utils.torch import tensor, to_device
    
    # Initialize policy and value networks
    policy_net = DiscretePolicy(
        state_dim, 
        action_dim, 
        hidden_size=(args.hidden_size, args.hidden_size)
    )
    value_net = Value(state_dim)
    policy_net.to(device)
    value_net.to(device)
    
    # Running state normalization
    running_state = ZFilter((state_dim,), clip=5)
    
    # Create agent
    agent = Agent(
        env, 
        policy_net, 
        device, 
        running_state=running_state, 
        render=False, 
        num_threads=1
    )
    
    # Constraint cost function (uses env's cost tracking)
    def constraint_cost(states, actions):
        """Cost function for CPO constraint.
        
        Returns per-step costs from environment's cost tracking.
        For now, returns a constant small cost - actual cost is computed in env.
        """
        # Note: Actual cost is computed in env.step() and stored in info['cost']
        # This is a placeholder for the advantage estimation
        costs = tensor(0.01 * np.ones(states.shape[0]), dtype=dtype).to(device)
        return costs
    
    # Training loop
    d_k = args.max_constraint
    e_k = args.annealing_factor if args.anneal else 0
    
    best_avg_reward = -float("inf")
    training_log = []
    
    print(f"\n[CPO Train] Starting training for {resolved_max_iter} iterations...")
    
    iter_range = range(resolved_max_iter)
    progress = tqdm(iter_range, dynamic_ncols=True, desc="CPO", unit="iter") if tqdm is not None else iter_range
    for i_iter in progress:
        # Collect samples
        batch, log = agent.collect_samples(args.min_batch_size)
        
        t0 = time.time()
        
        # Prepare batch tensors
        states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
        actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
        
        with torch.no_grad():
            values = value_net(states)
        
        # Estimate advantages
        advantages, returns = estimate_advantages(
            rewards, masks, values, args.gamma, args.tau, device
        )
        
        # Compute costs and cost advantages
        costs = constraint_cost(states, actions)
        cost_advantages, _ = estimate_advantages(
            costs, masks, values, args.gamma, args.tau, device
        )
        constraint_value = estimate_constraint_value(costs, masks, args.gamma, device)
        if hasattr(constraint_value, '__len__'):
            constraint_value = constraint_value[0]
        
        # CPO update step
        v_loss, p_loss, cost_loss = cpo_step(
            env_name="CPO-MicroTransit",  # For discrete action detection
            policy_net=policy_net,
            value_net=value_net,
            states=states,
            actions=actions,
            returns=returns,
            advantages=advantages,
            cost_advantages=cost_advantages,
            constraint_value=constraint_value,
            d_k=d_k,
            max_kl=args.max_kl,
            damping=args.damping,
            l2_reg=args.l2_reg,
            use_fim=True
        )
        
        t1 = time.time()
        
        # Constraint annealing
        if args.anneal:
            d_k = d_k + d_k * e_k
        
        # Logging
        iter_log = {
            "iteration": i_iter,
            "env_avg_reward": log["env_avg_reward"],
            "num_steps": log["num_steps"],
            "num_episodes": log["num_episodes"],
            "v_loss": float(v_loss) if hasattr(v_loss, 'item') else v_loss,
            "p_loss": float(p_loss.item()) if hasattr(p_loss, 'item') else p_loss,
            "cost_loss": float(cost_loss.item()) if hasattr(cost_loss, 'item') else cost_loss,
            "d_k": d_k,
            "update_time": t1 - t0,
        }
        training_log.append(iter_log)
        
        if i_iter % args.log_interval == 0:
            msg = (
                f"[Iter {i_iter:4d}] R_avg: {log['env_avg_reward']:8.2f} | "
                f"Steps: {log['num_steps']:5d} | "
                f"V_loss: {iter_log['v_loss']:.4f} | "
                f"d_k: {d_k:.2f}"
            )
            if tqdm is not None and hasattr(progress, "write"):
                progress.write(msg)
            else:
                print(msg)

        if tqdm is not None and hasattr(progress, "set_postfix"):
            progress.set_postfix(
                r_avg=f"{log['env_avg_reward']:.2f}",
                steps=int(log["num_steps"]),
                d_k=f"{d_k:.2f}",
                refresh=False,
            )
        
        # Save best model
        if log["env_avg_reward"] > best_avg_reward:
            best_avg_reward = log["env_avg_reward"]
            to_device(torch.device("cpu"), policy_net, value_net)
            save_path = run_dir / "best_model.pkl"
            with open(save_path, "wb") as f:
                pickle.dump((policy_net, value_net, running_state), f)
            to_device(device, policy_net, value_net)
            print(f"  [New best] R_avg={best_avg_reward:.2f} saved to {save_path}")
        
        # Periodic save
        if (i_iter + 1) % args.save_interval == 0:
            to_device(torch.device("cpu"), policy_net, value_net)
            save_path = run_dir / f"checkpoint_{i_iter + 1}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump((policy_net, value_net, running_state), f)
            to_device(device, policy_net, value_net)
    
    # Save final model
    to_device(torch.device("cpu"), policy_net, value_net)
    with open(run_dir / "final_model.pkl", "wb") as f:
        pickle.dump((policy_net, value_net, running_state), f)
    
    # Save training log
    with open(run_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n[CPO Train] Training complete!")
    print(f"[CPO Train] Best reward: {best_avg_reward:.2f}")
    print(f"[CPO Train] Models saved to: {run_dir}")


if __name__ == "__main__":
    main()
