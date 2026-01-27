"""MAPPO environment wrapper for on-policy compatibility.

This module wraps EventDrivenEnv to provide a multi-agent interface compatible
with the on-policy MAPPO implementation (https://github.com/marlbenchmark/on-policy).

Key Design Decisions:
1. action_dim = neighbor_k + 1 (last action = NOOP)
2. Synchronous step: actions always [n_agents], only active agent effective
3. Team reward: rewards shape [n_agents, 1], all agents receive same reward
4. available_actions: inactive agent only has NOOP available
5. Parameter sharing: all agents share the same policy

Reference:
- Chao Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games."
  NeurIPS 2022. arXiv:2103.01955
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from src.env.gym_env import EnvConfig, EventDrivenEnv


# Action index for "no operation" - vehicle waits at current stop
NOOP_ACTION = -1  # Sentinel value, mapped to last action index


@dataclass
class MAPPOEnvConfig:
    """Configuration for MAPPO environment wrapper."""
    env_config: EnvConfig
    neighbor_k: int = 8  # Fixed number of candidate actions per agent
    obs_include_global: bool = False  # Include global stats in local obs
    max_episode_steps: int = 200  # Maximum steps per episode
    fast_inactive_obs: bool = True  # If True, compute full obs only for active agents


def _normalize_features(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize features to zero mean and unit variance."""
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + eps
    return (arr - mean) / std


class MAPPOEnvWrapper:
    """
    Wrap EventDrivenEnv for on-policy MAPPO compatibility.
    
    Invariants:
    1. action_dim = neighbor_k + 1 (last action = NOOP)
    2. Synchronous step: actions always [n_agents], only active agent effective
    3. Team reward: rewards shape [n_agents, 1], all agents receive same reward
    4. available_actions: inactive agent only has NOOP available
    5. Stage-1 only: decisions happen at stop arrivals (event-driven)
    
    Observation Design:
    - Local obs: flattened features from current k-hop subgraph
    - Share obs: global state summary (fixed dimension)
    
    Action Design:
    - Discrete(neighbor_k + 1): indices 0..neighbor_k-1 map to candidate stops
    - Index neighbor_k = NOOP (no operation, wait at current stop)
    - Padding: if fewer than neighbor_k candidates, padding actions are masked
    """
    
    def __init__(self, config: MAPPOEnvConfig):
        self.config = config
        self.env = EventDrivenEnv(config.env_config)
        
        self.n_agents = config.env_config.num_vehicles
        self.neighbor_k = config.neighbor_k
        self.action_dim = self.neighbor_k + 1  # +1 for NOOP
        
        # Track which agents need to make decisions
        self._active_agent_mask = np.zeros(self.n_agents, dtype=bool)
        self._current_candidates: Dict[int, List[int]] = {}  # agent_id -> candidate stops
        self._current_masks: Dict[int, np.ndarray] = {}  # agent_id -> valid action mask
        
        # Observation dimensions
        # Local obs: node features (5) + edge features (4) per action + position embedding
        self._node_feat_dim = 5
        self._edge_feat_dim = 5 if self.env.config.use_fleet_potential else 4
        self._pos_emb_dim = 16  # Learned position embedding dimension
        self._global_summary_dim = 32  # Global state summary
        
        # Compute observation dimension
        # Local = current node features + candidate edge features + onboard summary
        self._local_obs_dim = (
            self._node_feat_dim  # Current stop features
            + self.neighbor_k * self._edge_feat_dim  # Edge features for each candidate
            + 4  # Onboard summary: count, mean_delay, max_delay, capacity_ratio
            + self._pos_emb_dim  # Position embedding
        )
        
        # Share obs = local obs + global summary + other agents positions
        self._share_obs_dim = (
            self._local_obs_dim
            + self._global_summary_dim
            + (self.n_agents - 1) * (self._node_feat_dim + 2)  # Other agents: features + count + time
        )
        
        # Build spaces (lists for compatibility with on-policy)
        self._build_spaces()
        
    def _build_spaces(self) -> None:
        """Build observation and action spaces for all agents."""
        # All agents share the same spaces (parameter sharing)
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._local_obs_dim,),
            dtype=np.float32
        )
        share_obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._share_obs_dim,),
            dtype=np.float32
        )
        act_space = spaces.Discrete(self.action_dim)
        
        # on-policy expects lists of spaces
        self.observation_space = [obs_space for _ in range(self.n_agents)]
        self.share_observation_space = [share_obs_space for _ in range(self.n_agents)]
        self.action_space = [act_space for _ in range(self.n_agents)]
        
    @property
    def num_agents(self) -> int:
        return self.n_agents
    
    def seed(self, seed: int) -> None:
        """Set random seed."""
        self.env.config.seed = seed
        self.env.rng = np.random.default_rng(seed)
        
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reset environment.
        
        Returns:
            obs: [n_agents, obs_dim] local observations
            share_obs: [n_agents, share_obs_dim] centralized observations
            available_actions: [n_agents, action_dim] action masks
        """
        self.env.reset()
        self._update_active_agents()
        
        obs, share_obs = self._build_observations()
        available_actions = self._get_all_available_actions()
        
        return obs, share_obs, available_actions
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one step.
        
        Args:
            actions: [n_agents] action indices for all agents
            
        Returns:
            obs: [n_agents, obs_dim]
            share_obs: [n_agents, share_obs_dim]
            rewards: [n_agents, 1] team reward (same for all)
            dones: [n_agents, 1] episode done flags
            infos: [n_agents] list of info dicts
            available_actions: [n_agents, action_dim]
        """
        total_reward = 0.0
        step_infos: List[Dict[str, Any]] = [{} for _ in range(self.n_agents)]
        
        # Process actions for active agents
        for agent_id in range(self.n_agents):
            if not self._active_agent_mask[agent_id]:
                continue
                
            action_idx = int(actions[agent_id])
            candidates = self._current_candidates.get(agent_id, [])
            
            # Map action index to actual stop
            if action_idx >= len(candidates) or action_idx == self.neighbor_k:
                # NOOP: stay at current stop (select first valid candidate or do nothing)
                if candidates:
                    actual_action = candidates[0]
                else:
                    continue
            else:
                actual_action = candidates[action_idx]
            
            # Execute step in underlying environment
            # Note: env.step expects the actual stop ID, not index
            obs_dict, reward, done, info = self.env.step(actual_action)
            total_reward += reward
            step_infos[agent_id] = dict(info)
            
            # If episode ended, break
            if self.env.done:
                break
        
        # Update active agents for next step
        if not self.env.done:
            self._update_active_agents()
        
        # Build observations
        obs, share_obs = self._build_observations()
        available_actions = self._get_all_available_actions()
        
        # Team reward: all agents get the same reward
        rewards = np.full((self.n_agents, 1), total_reward, dtype=np.float32)
        
        # Done flags
        dones = np.full((self.n_agents, 1), self.env.done, dtype=bool)
        
        # Infos list
        infos = step_infos
        
        return obs, share_obs, rewards, dones, infos, available_actions
    
    def _update_active_agents(self) -> None:
        """Update which agents need to make decisions."""
        self._active_agent_mask.fill(False)
        self._current_candidates.clear()
        self._current_masks.clear()
        
        if self.env.done:
            return
            
        # In the current event-driven design, only one vehicle decides at a time
        active_vehicle = self.env._get_active_vehicle()
        if active_vehicle is None:
            return
            
        agent_id = active_vehicle.vehicle_id
        self._active_agent_mask[agent_id] = True
        
        # Get candidates for this agent
        actions, mask = self.env.get_action_mask()
        self._current_candidates[agent_id] = list(actions)
        
        # Build fixed-size mask (neighbor_k + 1)
        full_mask = np.zeros(self.action_dim, dtype=bool)
        for i, valid in enumerate(mask):
            if i < self.neighbor_k:
                full_mask[i] = valid
        # NOOP is always available for active agents
        full_mask[self.neighbor_k] = True
        
        self._current_masks[agent_id] = full_mask
        
    def _get_all_available_actions(self) -> np.ndarray:
        """Get available actions for all agents."""
        available = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
        
        for agent_id in range(self.n_agents):
            if self._active_agent_mask[agent_id]:
                # Use computed mask
                available[agent_id] = self._current_masks.get(
                    agent_id, 
                    self._default_inactive_mask()
                ).astype(np.float32)
            else:
                # Inactive: only NOOP available
                available[agent_id] = self._default_inactive_mask().astype(np.float32)
                
        return available
    
    def _default_inactive_mask(self) -> np.ndarray:
        """Default mask for inactive agents: only NOOP."""
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[self.neighbor_k] = True  # Only NOOP
        return mask
    
    def _build_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build local and shared observations for all agents."""
        obs = np.zeros((self.n_agents, self._local_obs_dim), dtype=np.float32)
        share_obs = np.zeros((self.n_agents, self._share_obs_dim), dtype=np.float32)
        
        # Get features from environment ONCE (critical for performance)
        features = self.env.get_feature_batch()
        node_features = features["node_features"]  # [n_stops, 5]

        # Global summary (fixed dim)
        global_summary = self._compute_global_summary(node_features, features)

        # Precompute per-agent coarse state used in share_obs
        # Shape: [n_agents, node_feat_dim + 2]
        agent_state = np.zeros((self.n_agents, self._node_feat_dim + 2), dtype=np.float32)
        for agent_id in range(self.n_agents):
            vehicle = self.env.vehicles[agent_id]
            stop_idx = self.env.stop_index.get(vehicle.current_stop, 0)
            if stop_idx < len(node_features):
                agent_state[agent_id, : self._node_feat_dim] = node_features[stop_idx]
            agent_state[agent_id, self._node_feat_dim] = float(len(vehicle.onboard)) / 10.0
            agent_state[agent_id, self._node_feat_dim + 1] = max(
                0.0, float(vehicle.available_time - self.env.current_time) / 600.0
            )

        # Determine which agents need full observation build
        if self.config.fast_inactive_obs:
            active_ids = np.where(self._active_agent_mask)[0].tolist()
            if not active_ids:
                active_ids = []
        else:
            active_ids = list(range(self.n_agents))

        # Build local obs for chosen agents
        for agent_id in active_ids:
            vehicle = self.env.vehicles[agent_id]
            local_obs = self._build_local_obs(vehicle, features)
            obs[agent_id] = local_obs

        # Build share_obs (local + global + other agents state)
        other_agents_by_agent = self._build_other_agents_obs_all(agent_state)
        for agent_id in range(self.n_agents):
            share_obs[agent_id] = np.concatenate([obs[agent_id], global_summary, other_agents_by_agent[agent_id]])

        return obs, share_obs
    
    def _build_local_obs(self, vehicle, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Build local observation for a single agent."""
        node_features = features["node_features"]
        edge_features = features["edge_features"]
        
        # Current stop features
        stop_idx = self.env.stop_index.get(vehicle.current_stop, 0)
        current_node_feat = node_features[stop_idx]  # [5]
        
        # Edge features for candidates (pad to neighbor_k)
        edge_feat_padded = np.zeros((self.neighbor_k, self._edge_feat_dim), dtype=np.float32)
        n_edges = min(len(edge_features), self.neighbor_k)
        if n_edges > 0:
            edge_feat_padded[:n_edges] = edge_features[:n_edges]
        edge_feat_flat = edge_feat_padded.flatten()  # [neighbor_k * 4]
        
        # Onboard summary
        onboard_count = len(vehicle.onboard)
        capacity_ratio = onboard_count / max(1, self.env.config.vehicle_capacity)
        mean_delay = 0.0
        max_delay = 0.0
        if vehicle.onboard:
            delays = []
            for pax in vehicle.onboard:
                if pax.get("pickup_time_sec") is not None:
                    elapsed = self.env.current_time - pax["pickup_time_sec"]
                    direct = pax.get("direct_time_sec", elapsed)
                    delay = max(0.0, elapsed - direct) / max(1.0, direct)
                    delays.append(delay)
            if delays:
                mean_delay = float(np.mean(delays))
                max_delay = float(np.max(delays))
        onboard_summary = np.array([
            onboard_count / 10.0,  # Normalize
            mean_delay,
            max_delay,
            capacity_ratio
        ], dtype=np.float32)
        
        # Position embedding (use geo embedding from node features)
        # Pad to _pos_emb_dim
        pos_emb = np.zeros(self._pos_emb_dim, dtype=np.float32)
        pos_emb[0] = node_features[stop_idx, 4]  # geo_embedding_scalar
        # Add normalized position in stop list
        pos_emb[1] = stop_idx / max(1, len(self.env.stop_ids))
        # Add time progress
        if self.env.requests:
            max_time = max(r["request_time_sec"] for r in self.env.requests)
            pos_emb[2] = self.env.current_time / max(1.0, max_time)
        
        return np.concatenate([
            current_node_feat,
            edge_feat_flat,
            onboard_summary,
            pos_emb
        ])
    
    def _compute_global_summary(
        self, node_features: np.ndarray, features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute fixed-dimension global state summary."""
        summary = np.zeros(self._global_summary_dim, dtype=np.float32)
        
        # Node feature statistics (mean, max, std for each dimension)
        # risk_mean, risk_cvar, count, fairness, geo_emb
        for i in range(min(5, node_features.shape[1])):
            col = node_features[:, i]
            summary[i * 3] = float(np.mean(col))
            summary[i * 3 + 1] = float(np.max(col))
            summary[i * 3 + 2] = float(np.std(col))
        
        # Fleet summary (starting at index 15)
        fleet_idx = 15
        total_onboard = sum(len(v.onboard) for v in self.env.vehicles)
        total_capacity = self.n_agents * self.env.config.vehicle_capacity
        summary[fleet_idx] = total_onboard / max(1, total_capacity)
        
        # Demand summary
        total_waiting = sum(len(q) for q in self.env.waiting.values())
        summary[fleet_idx + 1] = min(1.0, total_waiting / 100.0)  # Normalize
        
        # Time progress
        summary[fleet_idx + 2] = self.env.steps / max(1, self.config.max_episode_steps)
        
        # Served ratio
        total_requests = len(self.env.requests)
        summary[fleet_idx + 3] = self.env.served / max(1, total_requests)
        
        # Churn ratio
        total_churned = self.env.waiting_churned + self.env.onboard_churned
        summary[fleet_idx + 4] = total_churned / max(1, total_requests)
        
        return summary
    
    def _build_other_agents_obs_all(self, agent_state: np.ndarray) -> np.ndarray:
        """Build other-agent observation blocks for all agents.

        This avoids O(n_agents^2) calls to env.get_feature_batch() which was
        the dominant bottleneck for large fleets.
        """
        other_dim = (self.n_agents - 1) * (self._node_feat_dim + 2)
        other_obs = np.zeros((self.n_agents, other_dim), dtype=np.float32)
        for agent_id in range(self.n_agents):
            if agent_id == 0:
                others = agent_state[1:]
            elif agent_id == self.n_agents - 1:
                others = agent_state[:-1]
            else:
                others = np.concatenate([agent_state[:agent_id], agent_state[agent_id + 1 :]], axis=0)
            other_obs[agent_id] = others.reshape(-1)
        return other_obs
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def render(self, mode: str = "human") -> None:
        """Render environment (not implemented)."""
        pass
    
    def get_active_masks(self) -> np.ndarray:
        """
        Get active agent masks for training.
        
        Returns:
            active_masks: [n_agents, 1] - 1 if agent is active, 0 otherwise
        """
        return self._active_agent_mask.reshape(-1, 1).astype(np.float32)


def make_train_env(config: MAPPOEnvConfig) -> MAPPOEnvWrapper:
    """Factory function to create training environment."""
    return MAPPOEnvWrapper(config)


def make_eval_env(config: MAPPOEnvConfig) -> MAPPOEnvWrapper:
    """Factory function to create evaluation environment."""
    return MAPPOEnvWrapper(config)


# Vectorized environment wrapper for parallel training
class DummyVecEnv:
    """Simple vectorized environment wrapper for compatibility."""
    
    def __init__(self, env_fns: List[callable]):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        
        # Copy spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.share_observation_space = self.envs[0].share_observation_space
        self.action_space = self.envs[0].action_space
        self.num_agents = self.envs[0].num_agents
        
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reset all environments."""
        results = [env.reset() for env in self.envs]
        obs = np.stack([r[0] for r in results])
        share_obs = np.stack([r[1] for r in results])
        available = np.stack([r[2] for r in results])
        return obs, share_obs, available
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List, np.ndarray]:
        """Step all environments."""
        results = [
            env.step(actions[i]) for i, env in enumerate(self.envs)
        ]
        obs = np.stack([r[0] for r in results])
        share_obs = np.stack([r[1] for r in results])
        rewards = np.stack([r[2] for r in results])
        dones = np.stack([r[3] for r in results])
        infos = [r[4] for r in results]
        available = np.stack([r[5] for r in results])
        return obs, share_obs, rewards, dones, infos, available
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()
