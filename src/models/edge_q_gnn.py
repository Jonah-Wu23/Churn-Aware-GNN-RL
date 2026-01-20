"""Edge-Q GNN model.

This module implements an Edge-Q network with edge-conditioned message passing (ECC-style).
The model outputs Q-values aligned with the provided candidate action edges.
"""

from __future__ import annotations

import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import TransformerConv


class EdgeQGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.dueling = bool(dueling)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList(
            [
                TransformerConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    edge_dim=self.edge_dim,
                    heads=1,
                    concat=False,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.edge_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        if self.dueling:
            self.value_head = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )

    def forward(self, data):
        """Return edge Q values aligned with candidate action edges.

        Expected keys:
        - node_features: [num_nodes, node_dim]
        - graph_edge_index: [2, num_graph_edges] (optional; falls back to edge_index)
        - graph_edge_features: [num_graph_edges, edge_dim] (optional; falls back to edge_features)
        - edge_index: [2, num_action_edges] candidate edges (optional if action_edge_index provided)
        - edge_features: [num_action_edges, edge_dim] candidate edge features
        - action_edge_index: [2, num_action_edges] (optional alternative name)
        """
        x: Tensor = data["node_features"]
        graph_edge_index: Tensor = data.get("graph_edge_index", data.get("edge_index"))
        graph_edge_attr: Tensor = data.get("graph_edge_features", data.get("edge_features"))

        action_edge_index: Tensor = data.get("action_edge_index", data.get("edge_index"))
        action_edge_attr: Tensor = data.get("edge_features", data.get("action_edge_features"))

        h = self.node_encoder(x)
        for conv in self.convs:
            h = conv(h, graph_edge_index, graph_edge_attr)

        src = action_edge_index[0].long()
        dst = action_edge_index[1].long()
        features = torch.cat([h[src], h[dst], action_edge_attr], dim=-1)
        advantage = self.q_head(features).squeeze(-1)
        if not self.dueling:
            return advantage
        if action_edge_index.numel() == 0:
            return advantage
        current_node = int(action_edge_index[0, 0].item())
        pooled = h.mean(dim=0)
        value_input = torch.cat([h[current_node], pooled], dim=-1)
        value = self.value_head(value_input).squeeze(-1)
        adv_mean = advantage.mean()
        return value + (advantage - adv_mean)
