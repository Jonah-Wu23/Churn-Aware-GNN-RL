import numpy as np
import torch
from torch import nn

from src.train.dqn import DQNConfig, DQNTrainer


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_dim = 5
        self.edge_dim = 4
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        num_actions = int(data["action_edge_index"].shape[1]) if data["action_edge_index"].numel() > 0 else 0
        return self.bias + torch.zeros(max(1, num_actions), device=self.bias.device)


class DummyEnv:
    def __init__(self) -> None:
        self.stop_ids = [0, 1]
        self.neighbors = {0: list(range(30))}
        self.graph_edge_index = np.array([[0], [1]], dtype=np.int64)
        self.graph_edge_features = np.zeros((1, 4), dtype=np.float32)


def test_replay_buffer_allows_noop_action_slot(tmp_path) -> None:
    env = DummyEnv()
    model = DummyModel()
    cfg = DQNConfig(
        total_steps=1,
        buffer_size=10,
        batch_size=1,
        learning_starts=10,
        train_freq=1,
        gradient_steps=0,
        target_update_interval=9999,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        log_every_steps=1,
        checkpoint_every_steps=0,
        device="cpu",
    )
    trainer = DQNTrainer(
        env=env,
        model=model,
        config=cfg,
        run_dir=tmp_path,
        graph_hashes={},
        od_hashes={},
        env_cfg={},
    )
    assert trainer.buffer.spec.max_actions == 31
    trainer.close()
