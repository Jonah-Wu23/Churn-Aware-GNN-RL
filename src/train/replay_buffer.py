"""Experience replay buffer for DQN-style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class BufferSpec:
    num_nodes: int
    node_dim: int
    edge_dim: int
    max_actions: int


class ReplayBuffer:
    """Fixed-size replay buffer with padding for variable action counts."""

    def __init__(self, capacity: int, spec: BufferSpec) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.spec = spec

        self._size = 0
        self._pos = 0

        self.obs = np.zeros((capacity, spec.num_nodes, spec.node_dim), dtype=np.float16)
        self.obs_idx = np.zeros((capacity,), dtype=np.int64)
        self.action_nodes = np.zeros((capacity, spec.max_actions), dtype=np.int64)
        self.action_edge = np.zeros((capacity, spec.max_actions, spec.edge_dim), dtype=np.float16)
        self.action_mask = np.zeros((capacity, spec.max_actions), dtype=bool)
        self.action_count = np.zeros((capacity,), dtype=np.int64)
        self.action_taken = np.zeros((capacity,), dtype=np.int64)

        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=bool)

        self.next_obs = np.zeros((capacity, spec.num_nodes, spec.node_dim), dtype=np.float16)
        self.next_obs_idx = np.zeros((capacity,), dtype=np.int64)
        self.next_action_nodes = np.zeros((capacity, spec.max_actions), dtype=np.int64)
        self.next_action_edge = np.zeros((capacity, spec.max_actions, spec.edge_dim), dtype=np.float16)
        self.next_action_mask = np.zeros((capacity, spec.max_actions), dtype=bool)
        self.next_action_count = np.zeros((capacity,), dtype=np.int64)

    @property
    def size(self) -> int:
        return int(self._size)

    def add(self, transition: Dict[str, np.ndarray]) -> None:
        """Add a transition dict produced by the trainer."""
        pos = self._pos
        spec = self.spec

        obs = transition["obs"].astype(np.float32)
        next_obs = transition["next_obs"].astype(np.float32)
        if obs.shape != (spec.num_nodes, spec.node_dim):
            raise ValueError("obs shape mismatch")
        if next_obs.shape != (spec.num_nodes, spec.node_dim):
            raise ValueError("next_obs shape mismatch")

        a_count = int(transition["action_count"])
        if a_count < 0 or a_count > spec.max_actions:
            raise ValueError("action_count out of range")

        action_nodes = transition["action_node_indices"].astype(np.int64)
        action_edge = transition["action_edge_features"].astype(np.float32)
        action_mask = transition["action_mask"].astype(bool)
        if action_nodes.shape != (a_count,):
            raise ValueError("action_node_indices shape mismatch")
        if action_edge.shape != (a_count, spec.edge_dim):
            raise ValueError("action_edge_features shape mismatch")
        if action_mask.shape != (a_count,):
            raise ValueError("action_mask shape mismatch")

        next_a_count = int(transition["next_action_count"])
        next_action_nodes = transition["next_action_node_indices"].astype(np.int64)
        next_action_edge = transition["next_action_edge_features"].astype(np.float32)
        next_action_mask = transition["next_action_mask"].astype(bool)
        if next_action_nodes.shape != (next_a_count,):
            raise ValueError("next_action_node_indices shape mismatch")
        if next_action_edge.shape != (next_a_count, spec.edge_dim):
            raise ValueError("next_action_edge_features shape mismatch")
        if next_action_mask.shape != (next_a_count,):
            raise ValueError("next_action_mask shape mismatch")

        self.obs[pos] = obs.astype(np.float16)
        self.obs_idx[pos] = int(transition["obs_idx"])

        self.action_nodes[pos].fill(0)
        self.action_edge[pos].fill(0)
        self.action_mask[pos].fill(False)
        self.action_nodes[pos, :a_count] = action_nodes
        self.action_edge[pos, :a_count] = action_edge.astype(np.float16)
        self.action_mask[pos, :a_count] = action_mask
        self.action_count[pos] = a_count
        self.action_taken[pos] = int(transition["action_taken"])

        self.reward[pos] = float(transition["reward"])
        self.done[pos] = bool(transition["done"])

        self.next_obs[pos] = next_obs.astype(np.float16)
        self.next_obs_idx[pos] = int(transition["next_obs_idx"])

        self.next_action_nodes[pos].fill(0)
        self.next_action_edge[pos].fill(0)
        self.next_action_mask[pos].fill(False)
        self.next_action_nodes[pos, :next_a_count] = next_action_nodes
        self.next_action_edge[pos, :next_a_count] = next_action_edge.astype(np.float16)
        self.next_action_mask[pos, :next_a_count] = next_action_mask
        self.next_action_count[pos] = next_a_count

        self._pos = (pos + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        size = self._size
        batch_size = int(batch_size)
        idx = rng.integers(0, size, size=batch_size, dtype=np.int64)
        return {
            "obs": self.obs[idx].astype(np.float32),
            "obs_idx": self.obs_idx[idx],
            "action_node_indices": self.action_nodes[idx],
            "action_edge_features": self.action_edge[idx].astype(np.float32),
            "action_mask": self.action_mask[idx],
            "action_count": self.action_count[idx],
            "action_taken": self.action_taken[idx],
            "reward": self.reward[idx],
            "done": self.done[idx],
            "next_obs": self.next_obs[idx].astype(np.float32),
            "next_obs_idx": self.next_obs_idx[idx],
            "next_action_node_indices": self.next_action_nodes[idx],
            "next_action_edge_features": self.next_action_edge[idx].astype(np.float32),
            "next_action_mask": self.next_action_mask[idx],
            "next_action_count": self.next_action_count[idx],
        }

    def get_state(self) -> Dict[str, object]:
        """获取buffer完整状态用于checkpoint保存"""
        return {
            "_size": self._size,
            "_pos": self._pos,
            "obs": self.obs[:self._size].copy() if self._size > 0 else np.array([]),
            "obs_idx": self.obs_idx[:self._size].copy() if self._size > 0 else np.array([]),
            "action_nodes": self.action_nodes[:self._size].copy() if self._size > 0 else np.array([]),
            "action_edge": self.action_edge[:self._size].copy() if self._size > 0 else np.array([]),
            "action_mask": self.action_mask[:self._size].copy() if self._size > 0 else np.array([]),
            "action_count": self.action_count[:self._size].copy() if self._size > 0 else np.array([]),
            "action_taken": self.action_taken[:self._size].copy() if self._size > 0 else np.array([]),
            "reward": self.reward[:self._size].copy() if self._size > 0 else np.array([]),
            "done": self.done[:self._size].copy() if self._size > 0 else np.array([]),
            "next_obs": self.next_obs[:self._size].copy() if self._size > 0 else np.array([]),
            "next_obs_idx": self.next_obs_idx[:self._size].copy() if self._size > 0 else np.array([]),
            "next_action_nodes": self.next_action_nodes[:self._size].copy() if self._size > 0 else np.array([]),
            "next_action_edge": self.next_action_edge[:self._size].copy() if self._size > 0 else np.array([]),
            "next_action_mask": self.next_action_mask[:self._size].copy() if self._size > 0 else np.array([]),
            "next_action_count": self.next_action_count[:self._size].copy() if self._size > 0 else np.array([]),
        }

    def set_state(self, state: Dict[str, object]) -> None:
        """从checkpoint恢复buffer状态"""
        self._size = int(state["_size"])
        self._pos = int(state["_pos"])
        if self._size > 0:
            self.obs[:self._size] = state["obs"]
            self.obs_idx[:self._size] = state["obs_idx"]
            self.action_nodes[:self._size] = state["action_nodes"]
            self.action_edge[:self._size] = state["action_edge"]
            self.action_mask[:self._size] = state["action_mask"]
            self.action_count[:self._size] = state["action_count"]
            self.action_taken[:self._size] = state["action_taken"]
            self.reward[:self._size] = state["reward"]
            self.done[:self._size] = state["done"]
            self.next_obs[:self._size] = state["next_obs"]
            self.next_obs_idx[:self._size] = state["next_obs_idx"]
            self.next_action_nodes[:self._size] = state["next_action_nodes"]
            self.next_action_edge[:self._size] = state["next_action_edge"]
            self.next_action_mask[:self._size] = state["next_action_mask"]
            self.next_action_count[:self._size] = state["next_action_count"]
