"""Experience replay buffer for DQN-style training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import sys

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


_PER_PATH = Path(__file__).resolve().parents[2] / "per"
if _PER_PATH.exists() and str(_PER_PATH) not in sys.path:
    sys.path.append(str(_PER_PATH))
try:
    from prioritized_memory import Memory
except ImportError:  # pragma: no cover
    Memory = None


class PrioritizedReplayBuffer:
    """Replay buffer backed by per/Memory for prioritized sampling."""

    def __init__(
        self,
        capacity: int,
        spec: BufferSpec,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 200_000,
        epsilon: float = 1e-6,
    ) -> None:
        if Memory is None:
            raise ImportError("PER module not available. Ensure per/ is present and importable.")
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.spec = spec
        self._size = 0
        self._pos = 0
        self._max_error = 1.0

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

        self.memory = Memory(self.capacity)
        self.memory.a = float(alpha)
        self.memory.e = float(epsilon)
        self.memory.beta = float(beta_start)
        if int(beta_frames) > 0:
            self.memory.beta_increment_per_sampling = float((1.0 - float(beta_start)) / float(beta_frames))
        else:
            self.memory.beta_increment_per_sampling = 0.0

    @property
    def size(self) -> int:
        return int(self._size)

    def add(self, transition: Dict[str, np.ndarray]) -> None:
        pos = int(self.memory.tree.write)
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

        self.memory.add(self._max_error, pos)
        self._pos = int(self.memory.tree.write)
        self._size = int(self.memory.tree.n_entries)

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        batch_list, idxs, is_weight = self.memory.sample(int(batch_size))
        data_indices = np.array(batch_list, dtype=np.int64)
        if data_indices.size == 0:
            raise ValueError("Empty batch from PER sampler")

        batch = {
            "obs": self.obs[data_indices].astype(np.float32),
            "obs_idx": self.obs_idx[data_indices],
            "action_node_indices": self.action_nodes[data_indices],
            "action_edge_features": self.action_edge[data_indices].astype(np.float32),
            "action_mask": self.action_mask[data_indices],
            "action_count": self.action_count[data_indices],
            "action_taken": self.action_taken[data_indices],
            "reward": self.reward[data_indices],
            "done": self.done[data_indices],
            "next_obs": self.next_obs[data_indices].astype(np.float32),
            "next_obs_idx": self.next_obs_idx[data_indices],
            "next_action_node_indices": self.next_action_nodes[data_indices],
            "next_action_edge_features": self.next_action_edge[data_indices].astype(np.float32),
            "next_action_mask": self.next_action_mask[data_indices],
            "next_action_count": self.next_action_count[data_indices],
            "weights": np.array(is_weight, dtype=np.float32),
            "indices": np.array(idxs, dtype=np.int64),
        }
        return batch

    def update_priorities(self, indices: Iterable[int], td_errors: Iterable[float]) -> None:
        for idx, err in zip(indices, td_errors):
            self.memory.update(int(idx), float(err))
            if abs(float(err)) > float(self._max_error):
                self._max_error = float(abs(float(err)))

    def get_state(self) -> Dict[str, object]:
        return {
            "_size": int(self._size),
            "_pos": int(self._pos),
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
            "priority_tree": self.memory.tree.tree.copy(),
            "priority_data": self.memory.tree.data.copy(),
            "priority_write": int(self.memory.tree.write),
            "priority_n_entries": int(self.memory.tree.n_entries),
            "priority_beta": float(self.memory.beta),
            "max_error": float(self._max_error),
        }

    def set_state(self, state: Dict[str, object]) -> None:
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

        tree = state.get("priority_tree")
        data = state.get("priority_data")
        if tree is not None and data is not None:
            self.memory.tree.tree[:] = tree
            self.memory.tree.data[:] = data
            self.memory.tree.write = int(state.get("priority_write", 0))
            self.memory.tree.n_entries = int(state.get("priority_n_entries", self._size))
            self.memory.beta = float(state.get("priority_beta", self.memory.beta))
            self._max_error = float(state.get("max_error", 1.0))
        else:
            self.memory.tree.tree.fill(0.0)
            self.memory.tree.data[:] = 0
            self.memory.tree.write = int(self._pos)
            self.memory.tree.n_entries = int(self._size)
            self._max_error = 1.0
