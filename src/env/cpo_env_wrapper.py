"""CPO environment wrapper for PyTorch-CPO compatibility.

This module wraps EventDrivenEnv to provide a standard Gym interface
compatible with the PyTorch-CPO implementation.

Design decisions:
- step = one vehicle's decision (single shared policy, async turns)
- action_mask: hard feasibility rule, applied at logits level
- cost: behavioral safety only (not structural unreachability)
- episode boundary: fixed simulation horizon (max_horizon_steps)

Masking is applied DURING TRAINING AND EVALUATION at policy distribution
level (logits masking before softmax), audited via masked_probability_mass.
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


# Sentinel for NOOP action (stay at current stop)
NOOP_ACTION_IDX = -1


@dataclass
class CPOEnvConfig:
    """Configuration for CPO environment wrapper."""
    env_config: EnvConfig
    neighbor_k: int = 8  # Fixed action dimension (padded/clipped)
    include_noop: bool = True  # Whether to include NOOP as last action
    cost_capacity_weight: float = 1.0  # Weight for capacity overflow cost
    cost_churn_weight: float = 1.0  # Weight for onboard churn cost
    cost_violation_weight: float = 1.0  # Weight for service commitment violation


class CPOEnvWrapper(gym.Env):
    """
    Wrap EventDrivenEnv for PyTorch-CPO compatibility.
    
    Provides a standard Gym interface with:
    - observation_space: Box (flattened features)
    - action_space: Discrete (neighbor_k + 1 for NOOP)
    - step() returns: obs, reward, done, info (with 'cost' key)
    
    Cost Signal (enters CPO constraint):
    - Onboard churn events (delay-induced passenger loss)
    - Capacity overflow attempts
    - Service commitment violations (action masked by budget constraint)
    
    NOT in cost (structural, not policy fault):
    - Structurally unreachable OD pairs (graph/data issue)
    - Initial corridor frozen failures
    """
    
    def __init__(self, config: CPOEnvConfig):
        super().__init__()
        self.config = config
        self.env = EventDrivenEnv(config.env_config)
        self.neighbor_k = config.neighbor_k
        self.include_noop = config.include_noop
        
        # Action dimension: neighbor_k candidates + optional NOOP
        self.action_dim = self.neighbor_k + (1 if self.include_noop else 0)
        
        # Observation dimension:
        # - Current node features: 5
        # - Edge features (padded): neighbor_k * 4
        # - Onboard summary: 4 (count_normalized, avg_delay, max_delay, capacity_ratio)
        # - Position embedding: 1 (geo embedding of current stop)
        self.node_feat_dim = 5
        self.edge_feat_dim = 5 if self.env.config.use_fleet_potential else 4
        self.onboard_dim = 4
        self.pos_dim = 1
        self.obs_dim = (
            self.node_feat_dim + 
            self.neighbor_k * self.edge_feat_dim + 
            self.onboard_dim + 
            self.pos_dim
        )
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_dim)
        
        # Tracking
        self._step_count = 0
        self._current_features: Optional[Dict[str, np.ndarray]] = None
        self._action_mask: Optional[np.ndarray] = None
        self._action_to_stop_map: Optional[np.ndarray] = None
        
        # Audit metrics
        self.masked_probability_mass: float = 0.0
        self.episode_cost: float = 0.0
        self.step_costs: List[float] = []
    
    def seed(self, seed: int) -> None:
        """Set random seed."""
        self.env.rng = np.random.default_rng(seed)
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.env.reset()
        self._step_count = 0
        self.episode_cost = 0.0
        self.step_costs = []
        self.masked_probability_mass = 0.0
        
        # Build initial observation
        obs, mask, action_map = self._build_observation()
        self._current_features = self.env.get_feature_batch()
        self._action_mask = mask
        self._action_to_stop_map = action_map
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step (one vehicle's decision).
        
        Args:
            action: Index into padded action space (0 to action_dim-1)
            
        Returns:
            obs: Next observation
            reward: Step reward
            done: Episode termination flag
            info: Dict with 'cost' key for CPO constraint
        """
        # Map action index to actual stop ID
        stop_id = self._map_action_to_stop(action)
        
        # Track if action was masked (for audit)
        was_masked = (
            self._action_mask is not None and 
            action < len(self._action_mask) and 
            not self._action_mask[action]
        )
        
        # Compute step cost BEFORE executing (captures violated constraints)
        step_cost = self._compute_step_cost(action, was_masked)
        
        # Execute action in underlying environment
        if stop_id is not None:
            _, reward, done, info = self.env.step(int(stop_id))
        else:
            # NOOP or invalid action: stay at current stop
            if self._current_features is not None:
                actions = self._current_features["actions"]
                if len(actions) > 0:
                    # Default to first valid action
                    stop_id = int(actions[0])
                    _, reward, done, info = self.env.step(stop_id)
                else:
                    reward = 0.0
                    done = True
                    info = {}
            else:
                reward = 0.0
                done = True
                info = {}
        
        # Add onboard churn cost from step result
        onboard_churned = info.get("onboard_churned_this_step", 0)
        step_cost += self.config.cost_churn_weight * float(onboard_churned)
        
        # Track costs
        self.step_costs.append(step_cost)
        self.episode_cost += step_cost
        
        self._step_count += 1
        
        # Build next observation
        obs, mask, action_map = self._build_observation()
        self._current_features = self.env.get_feature_batch()
        self._action_mask = mask
        self._action_to_stop_map = action_map
        
        # Add cost to info for CPO
        info["cost"] = step_cost
        info["episode_cost"] = self.episode_cost
        info["masked_action"] = was_masked
        
        return obs, float(reward), bool(done), info
    
    def _build_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build flattened observation from environment features.
        
        Returns:
            obs: Flattened observation vector
            action_mask: Boolean mask for valid actions
            action_to_stop: Mapping from action index to stop ID
        """
        features = self.env.get_feature_batch()
        
        # Current node features
        node_features = features["node_features"]  # [num_nodes, 5]
        current_idx = int(features["current_node_index"][0])
        
        if current_idx < len(node_features):
            current_node_feat = node_features[current_idx]  # [5]
        else:
            current_node_feat = np.zeros(self.node_feat_dim, dtype=np.float32)
        
        # Edge features (padded to neighbor_k)
        edge_features = features["edge_features"]  # [num_candidates, 4]
        n_candidates = min(len(edge_features), self.neighbor_k)
        
        edge_feat_padded = np.zeros((self.neighbor_k, self.edge_feat_dim), dtype=np.float32)
        if n_candidates > 0:
            edge_feat_padded[:n_candidates] = edge_features[:n_candidates]
        edge_feat_flat = edge_feat_padded.flatten()  # [neighbor_k * 4]
        
        # Onboard summary
        vehicle = self.env._get_active_vehicle()
        if vehicle:
            onboard_count = len(vehicle.onboard) / 10.0  # Normalized
            capacity = max(1, self.env.config.vehicle_capacity)
            capacity_ratio = len(vehicle.onboard) / capacity
            
            # Calculate delay stats for onboard passengers
            avg_delay = 0.0
            max_delay = 0.0
            if vehicle.onboard:
                delays = []
                for pax in vehicle.onboard:
                    direct_time = pax.get("direct_time_sec", 0)
                    elapsed = self.env.current_time - pax.get("pickup_time_sec", self.env.current_time)
                    delay = max(0, elapsed - direct_time) / 300.0  # Normalized by 5 min
                    delays.append(delay)
                avg_delay = float(np.mean(delays))
                max_delay = float(np.max(delays))
        else:
            onboard_count = 0.0
            avg_delay = 0.0
            max_delay = 0.0
            capacity_ratio = 0.0
        
        onboard_summary = np.array(
            [onboard_count, avg_delay, max_delay, capacity_ratio], 
            dtype=np.float32
        )
        
        # Position embedding (first dimension of geo embedding)
        if current_idx < len(node_features):
            pos_emb = np.array([node_features[current_idx, 4]], dtype=np.float32)
        else:
            pos_emb = np.zeros(self.pos_dim, dtype=np.float32)
        
        # Concatenate all features
        obs = np.concatenate([
            current_node_feat,
            edge_feat_flat,
            onboard_summary,
            pos_emb
        ]).astype(np.float32)
        
        # Action mask (padded)
        raw_mask = features["action_mask"].astype(bool)
        action_mask = np.zeros(self.action_dim, dtype=bool)
        n_valid = min(len(raw_mask), self.neighbor_k)
        action_mask[:n_valid] = raw_mask[:n_valid]
        
        # NOOP is only valid when no feasible actions exist
        if self.include_noop:
            action_mask[-1] = not bool(np.any(action_mask[:n_valid]))
        
        # Action to stop mapping
        actions = features["actions"].astype(np.int64)
        action_to_stop = np.full(self.action_dim, -1, dtype=np.int64)
        n_actions = min(len(actions), self.neighbor_k)
        if n_actions > 0:
            action_to_stop[:n_actions] = actions[:n_actions]
        
        return obs, action_mask, action_to_stop
    
    def _map_action_to_stop(self, action: int) -> Optional[int]:
        """Map action index to actual stop ID."""
        if action < 0 or action >= self.action_dim:
            return None
        
        # NOOP action
        if self.include_noop and action == self.action_dim - 1:
            return None
        
        if self._action_to_stop_map is None:
            return None
        
        stop_id = self._action_to_stop_map[action]
        if stop_id < 0:
            return None
        
        return int(stop_id)
    
    def _compute_step_cost(self, action: int, was_masked: bool) -> float:
        """
        Compute step cost for CPO constraint.
        
        Cost components (behavioral safety, enters CPO constraint):
        1. Capacity overflow attempt (trying to board beyond capacity)
        2. Service commitment violation (action was masked by budget)
        
        NOT included (structural, not policy fault):
        - Structurally unreachable OD pairs
        """
        cost = 0.0
        
        # Cost for taking masked action (budget/commitment violation)
        if was_masked:
            cost += self.config.cost_violation_weight * 1.0
        
        # Check for capacity constraint
        vehicle = self.env._get_active_vehicle()
        if vehicle:
            current_onboard = len(vehicle.onboard)
            capacity = self.env.config.vehicle_capacity
            
            # If at capacity and action leads to stop with waiting passengers
            stop_id = self._map_action_to_stop(action)
            if stop_id is not None and current_onboard >= capacity:
                waiting = self.env.waiting.get(stop_id, [])
                if len(waiting) > 0:
                    # Cannot board anyone, cost proportional to denied passengers
                    cost += self.config.cost_capacity_weight * min(len(waiting), 3) / 3.0
        
        return cost
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get current action mask for policy.
        
        This should be used during training to mask logits BEFORE softmax.
        """
        if self._action_mask is None:
            # All actions valid if no mask available
            return np.ones(self.action_dim, dtype=bool)
        return self._action_mask.copy()
    
    def get_stop_id_for_action(self, action: int) -> Optional[int]:
        """Get the stop ID corresponding to an action index (for logging)."""
        return self._map_action_to_stop(action)
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def render(self, mode: str = "human") -> None:
        """Render environment (not implemented)."""
        pass


def make_cpo_env(config: CPOEnvConfig) -> CPOEnvWrapper:
    """Factory function to create CPO environment."""
    return CPOEnvWrapper(config)
