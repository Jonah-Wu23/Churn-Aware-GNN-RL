#!/usr/bin/env python
"""MAPPO training script for micro-transit dispatch.

This script trains a MAPPO agent using the on-policy implementation
(https://github.com/marlbenchmark/on-policy) on the micro-transit dispatch
environment.

Usage:
    python scripts/run_mappo_train.py --config configs/manhattan.yaml

    # With custom parameters
    python scripts/run_mappo_train.py --config configs/manhattan.yaml \
        --num_env_steps 1000000 --n_rollout_threads 4

Reference:
    Chao Yu et al. "The Surprising Effectiveness of PPO in Cooperative 
    Multi-Agent Games." NeurIPS 2022. arXiv:2103.01955
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MAPPO training for micro-transit")
    
    # Environment config
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Output directory for logs and models")
    
    # Training hyperparameters (MAPPO paper defaults)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for actor")
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help="Learning rate for critic")
    parser.add_argument("--num_env_steps", type=int, default=100000,
                        help="Total number of environment steps")
    parser.add_argument("--episode_length", type=int, default=200,
                        help="Maximum episode length")
    parser.add_argument("--n_rollout_threads", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--ppo_epoch", type=int, default=10,
                        help="Number of PPO epochs")
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help="Number of mini batches")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="PPO clip parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Hidden layer size")
    parser.add_argument("--recurrent_N", type=int, default=1,
                        help="Number of recurrent layers")
    parser.add_argument("--neighbor_k", type=int, default=8,
                        help="Number of candidate actions (neighbor_k)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="Use CUDA")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save model every N episodes")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="Log every N episodes")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_on_policy_path():
    """Add on-policy to Python path."""
    on_policy_path = Path("baselines/on-policy").resolve()
    if str(on_policy_path) not in sys.path:
        sys.path.insert(0, str(on_policy_path))
    return on_policy_path


def make_args_namespace(args, config: Dict[str, Any]):
    """Create argparse.Namespace for on-policy compatibility."""
    import argparse as ap
    
    env_cfg = config.get("env", {})
    
    all_args = ap.Namespace(
        # Algorithm
        algorithm_name="mappo",
        experiment_name="microtransit",
        env_name="MicroTransit",
        
        # Training
        lr=args.lr,
        critic_lr=args.critic_lr,
        opti_eps=1e-5,
        weight_decay=0,
        num_env_steps=args.num_env_steps,
        episode_length=args.episode_length,
        n_rollout_threads=args.n_rollout_threads,
        n_eval_rollout_threads=1,
        n_render_rollout_threads=1,
        
        # PPO
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        clip_param=args.clip_param,
        entropy_coef=args.entropy_coef,
        value_loss_coef=1.0,
        max_grad_norm=10.0,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        use_gae=True,
        use_proper_time_limits=False,
        
        # Network
        hidden_size=args.hidden_size,
        layer_N=1,
        recurrent_N=args.recurrent_N,
        use_recurrent_policy=True,
        use_naive_recurrent_policy=False,
        use_orthogonal=True,
        gain=0.01,
        use_feature_normalization=False,
        use_policy_active_masks=True,
        stacked_frames=1,
        
        # Value
        use_centralized_V=True,
        use_obs_instead_of_state=False,
        use_valuenorm=True,
        use_linear_lr_decay=True,
        use_popart=False,
        use_clipped_value_loss=True,
        use_huber_loss=True,
        huber_delta=10.0,
        
        # Misc
        seed=args.seed,
        cuda=args.cuda,
        cuda_deterministic=True,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        eval_interval=25,
        use_eval=False,
        use_render=False,
        use_wandb=False,
        model_dir=None,
        
        # Environment specific
        num_vehicles=int(env_cfg.get("num_vehicles", 1)),
        neighbor_k=args.neighbor_k,
    )
    
    return all_args


def main():
    """Main training function."""
    args = get_args()
    config = load_config(args.config)
    
    # Setup paths
    on_policy_path = setup_on_policy_path()
    
    # Create run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("reports/mappo_train") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device("cpu")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Import on-policy modules
    from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
    from onpolicy.utils.shared_buffer import SharedReplayBuffer
    
    # Import our environment wrapper
    from src.env.gym_env import EnvConfig
    from src.env.mappo_env_wrapper import MAPPOEnvConfig, MAPPOEnvWrapper, DummyVecEnv
    
    # Create args namespace
    all_args = make_args_namespace(args, config)
    
    # Save config
    run_meta = {
        "args": vars(args),
        "all_args": vars(all_args),
        "config_path": str(args.config),
        "on_policy_path": str(on_policy_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    # Build environment config
    env_cfg = config.get("env", {})
    env_config = EnvConfig(
        max_horizon_steps=args.episode_length,
        max_requests=int(env_cfg.get("max_requests", 500)),
        seed=args.seed,
        num_vehicles=int(env_cfg.get("num_vehicles", 1)),
        vehicle_capacity=int(env_cfg.get("vehicle_capacity", 6)),
        mask_alpha=float(env_cfg.get("mask_alpha", 1.5)),
        od_glob=env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet"),
        graph_nodes_path=env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"),
        graph_edges_path=env_cfg.get("graph_edges_path", "data/processed/graph/layer2_edges.parquet"),
        graph_embeddings_path=env_cfg.get(
            "graph_embeddings_path",
            "data/processed/graph/node2vec_embeddings.parquet"
        ),
    )
    
    mappo_config = MAPPOEnvConfig(
        env_config=env_config,
        neighbor_k=args.neighbor_k,
        max_episode_steps=args.episode_length,
    )
    
    # Create environment
    def make_env(seed):
        def _init():
            cfg = MAPPOEnvConfig(
                env_config=EnvConfig(**{**env_config.__dict__, "seed": seed}),
                neighbor_k=args.neighbor_k,
                max_episode_steps=args.episode_length,
            )
            return MAPPOEnvWrapper(cfg)
        return _init
    
    env_fns = [make_env(args.seed + i) for i in range(args.n_rollout_threads)]
    envs = DummyVecEnv(env_fns)
    
    print("=" * 60)
    print("MAPPO Training for Micro-Transit Dispatch")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Num agents: {envs.num_agents}")
    print(f"Obs space: {envs.observation_space[0]}")
    print(f"Action space: {envs.action_space[0]}")
    print(f"Num env steps: {args.num_env_steps}")
    print("=" * 60)
    
    # Create policy
    share_observation_space = envs.share_observation_space[0] if all_args.use_centralized_V else envs.observation_space[0]
    
    policy = Policy(
        all_args,
        envs.observation_space[0],
        share_observation_space,
        envs.action_space[0],
        device=device
    )
    
    # Create trainer
    trainer = TrainAlgo(all_args, policy, device=device)
    
    # Create buffer
    buffer = SharedReplayBuffer(
        all_args,
        envs.num_agents,
        envs.observation_space[0],
        share_observation_space,
        envs.action_space[0]
    )
    
    # Training loop
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    
    train_infos = []
    start_time = time.time()
    
    for episode in range(episodes):
        if all_args.use_linear_lr_decay:
            policy.lr_decay(episode, episodes)
        
        # Reset environment
        obs, share_obs, available = envs.reset()
        
        # Initialize buffer
        buffer.obs[0] = obs.copy()
        buffer.share_obs[0] = share_obs.copy()
        buffer.available_actions[0] = available.copy()
        
        # Rollout
        for step in range(args.episode_length):
            # Get actions
            trainer.prep_rollout()
            
            values, actions, action_log_probs, rnn_states, rnn_states_critic = trainer.policy.get_actions(
                np.concatenate(buffer.share_obs[step]),
                np.concatenate(buffer.obs[step]),
                np.concatenate(buffer.rnn_states[step]),
                np.concatenate(buffer.rnn_states_critic[step]),
                np.concatenate(buffer.masks[step]),
                np.concatenate(buffer.available_actions[step])
            )
            
            values = np.array(np.split(values.cpu().numpy(), args.n_rollout_threads))
            actions = np.array(np.split(actions.cpu().numpy(), args.n_rollout_threads))
            action_log_probs = np.array(np.split(action_log_probs.cpu().numpy(), args.n_rollout_threads))
            rnn_states = np.array(np.split(rnn_states.cpu().numpy(), args.n_rollout_threads))
            rnn_states_critic = np.array(np.split(rnn_states_critic.cpu().numpy(), args.n_rollout_threads))
            
            # Step environment
            obs, share_obs, rewards, dones, infos, available = envs.step(actions[:, :, 0])
            
            # Handle episode end
            masks = np.ones((args.n_rollout_threads, envs.num_agents, 1), dtype=np.float32)
            active_masks = np.ones((args.n_rollout_threads, envs.num_agents, 1), dtype=np.float32)
            
            for i in range(args.n_rollout_threads):
                if dones[i].all():
                    # Reset RNN states
                    rnn_states[i] = np.zeros_like(rnn_states[i])
                    rnn_states_critic[i] = np.zeros_like(rnn_states_critic[i])
                    masks[i] = np.zeros_like(masks[i])
            
            # Insert to buffer
            buffer.insert(
                share_obs, obs, rnn_states, rnn_states_critic,
                actions, action_log_probs, values, rewards, masks,
                np.ones_like(masks), available, active_masks
            )
        
        # Compute returns
        trainer.prep_rollout()
        next_values = trainer.policy.get_values(
            np.concatenate(buffer.share_obs[-1]),
            np.concatenate(buffer.rnn_states_critic[-1]),
            np.concatenate(buffer.masks[-1])
        )
        next_values = np.array(np.split(next_values.cpu().numpy(), args.n_rollout_threads))
        buffer.compute_returns(next_values, trainer.value_normalizer)
        
        # Train
        trainer.prep_training()
        train_info = trainer.train(buffer)
        buffer.after_update()
        
        train_infos.append(train_info)
        
        # Log
        if episode % args.log_interval == 0:
            total_steps = (episode + 1) * args.episode_length * args.n_rollout_threads
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            
            print(f"Episode {episode}/{episodes} | Steps: {total_steps} | "
                  f"FPS: {fps:.1f} | "
                  f"Policy Loss: {train_info.get('policy_loss', 0):.4f} | "
                  f"Value Loss: {train_info.get('value_loss', 0):.4f}")
        
        # Save
        if episode % args.save_interval == 0 or episode == episodes - 1:
            models_dir = run_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            torch.save(
                policy.actor.state_dict(),
                models_dir / "actor.pt"
            )
            torch.save(
                policy.critic.state_dict(),
                models_dir / "critic.pt"
            )
            torch.save(
                policy.actor.state_dict(),
                models_dir / f"actor_ep{episode}.pt"
            )
            
            print(f"Saved model checkpoint at episode {episode}")
    
    envs.close()
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Final model saved to: {run_dir / 'models'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
