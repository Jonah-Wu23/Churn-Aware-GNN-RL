"""Student MLP model for action scoring."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class StudentActionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.use_layer_norm = bool(use_layer_norm)

        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(max(1, self.num_layers - 1)):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(self.hidden_dim))
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            logits = self.net(x).squeeze(-1)
            return logits
        if x.dim() == 3:
            batch, actions, dim = x.shape
            x_flat = x.reshape(batch * actions, dim)
            logits = self.net(x_flat).reshape(batch, actions)
            return logits
        raise ValueError("Input must be 2D or 3D tensor")
