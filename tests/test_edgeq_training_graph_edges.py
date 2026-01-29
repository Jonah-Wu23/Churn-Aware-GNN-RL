from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from src.train.dqn import DQNConfig, DQNTrainer


@dataclass
class DummyEnv:
    stop_ids: list[int]
    neighbors: Dict[int, list[Tuple[int, float]]]
    graph_edge_index: np.ndarray
    graph_edge_features: np.ndarray

    def reset(self) -> Dict[str, float]:
        return {}

    def get_feature_batch(self) -> Dict[str, np.ndarray]:
        return {
            "node_features": np.zeros((2, 5), dtype=np.float32),
            "action_mask": np.array([True], dtype=bool),
            "actions": np.array([1], dtype=np.int64),
            "action_node_indices": np.array([1], dtype=np.int64),
            "edge_features": np.zeros((1, 4), dtype=np.float32),
            "current_node_index": np.array([0], dtype=np.int64),
        }

    def step(self, action: int):
        info = {
            "served": 0,
            "waiting_churned": 0,
            "onboard_churned": 0,
            "structural_unserviceable": 0,
        }
        return {}, 0.0, True, info


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_dim = 5
        self.edge_dim = 4
        self.bias = nn.Parameter(torch.zeros(1))
        self.forward_calls = 0
        self.last_data: Dict[str, torch.Tensor] | None = None

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.forward_calls += 1
        assert "graph_edge_index" in data
        assert "graph_edge_features" in data
        assert "action_edge_index" in data
        assert "edge_features" in data
        self.last_data = data
        num_actions = int(data["action_edge_index"].shape[1])
        return self.bias + torch.zeros(num_actions, device=data["node_features"].device)


def test_dqn_passes_graph_edges_to_model(tmp_path):
    env = DummyEnv(
        stop_ids=[0, 1],
        neighbors={0: [(1, 1.0)], 1: [(0, 1.0)]},
        graph_edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        graph_edge_features=np.array([[0, 0, 0, 1.0], [0, 0, 0, 1.0]], dtype=np.float32),
    )
    model = DummyModel()
    cfg = DQNConfig(
        total_steps=1,
        buffer_size=1,
        batch_size=1,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        log_every_steps=1,
        checkpoint_every_steps=0,
        device="cpu",
    )
    trainer = DQNTrainer(env=env, model=model, config=cfg, run_dir=tmp_path, graph_hashes={}, od_hashes={}, env_cfg={})
    trainer.train(total_steps=1)
    trainer.close()
    assert model.forward_calls > 0
    assert model.last_data is not None
    assert "graph_edge_index" in model.last_data
