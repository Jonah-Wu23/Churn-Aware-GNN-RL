"""Node-Only GNN 消融变体 (w/o Edge-Encoding)

使用 GATConv 替代 TransformerConv，不使用 edge_dim。
保持相同的 action mask、候选邻居集合和输出接口。

消融目标：验证边特征决策机制的必要性。
"""

from __future__ import annotations

import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import GATConv


class NodeOnlyGNN(nn.Module):
    """消融变体：Node-Only GNN (w/o Edge-Encoding)
    
    核心区别：
    - 使用 GATConv 替代带 edge_dim 的 TransformerConv
    - Q-head 只使用 src/dst 节点嵌入，不使用边特征
    - 保持相同的 action mask 和候选邻居集合
    
    Args:
        node_dim: 节点特征维度
        hidden_dim: 隐藏层维度
        num_layers: GNN 层数
        dropout: Dropout 概率
        heads: GAT 注意力头数
        edge_dim: 边特征维度（用于训练缓冲区对齐，不参与计算）
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        heads: int = 4,
        edge_dim: int = 4,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = int(edge_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.heads = int(heads)
        
        # 确保 hidden_dim 能被 heads 整除
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        
        head_dim = hidden_dim // heads
        
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.ReLU(),
        )
        
        # 使用 GATConv，不传入 edge_dim
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=head_dim,
                    heads=self.heads,
                    concat=True,  # 输出 heads * head_dim = hidden_dim
                    dropout=self.dropout,
                    add_self_loops=True,
                )
            )
        
        # Q-head: 基于 src + dst 节点嵌入（不使用边特征）
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, data) -> Tensor:
        """返回候选动作的 Q 值（与 EdgeQGNN 接口一致）
        
        Expected keys:
        - node_features: [num_nodes, node_dim]
        - graph_edge_index: [2, num_graph_edges] (用于消息传递)
        - action_edge_index 或 edge_index: [2, num_action_edges] (候选动作边)
        
        注意：edge_features 被忽略，这是本消融的核心区别。
        
        Returns:
            Q values: [num_action_edges] 每个候选动作的 Q 值
        """
        x: Tensor = data["node_features"]
        
        # 用于消息传递的图结构边
        graph_edge_index: Tensor = data.get("graph_edge_index", data.get("edge_index"))
        
        # 候选动作边
        action_edge_index: Tensor = data.get("action_edge_index", data.get("edge_index"))
        
        # 节点编码
        h = self.node_encoder(x)
        
        # GNN 消息传递（不使用边特征）
        for conv in self.convs:
            h = conv(h, graph_edge_index)
        
        # 获取候选动作的 src/dst 节点嵌入
        src = action_edge_index[0].long()
        dst = action_edge_index[1].long()
        
        # 拼接 src 和 dst 的节点嵌入（不使用边特征）
        features = torch.cat([h[src], h[dst]], dim=-1)
        
        return self.q_head(features).squeeze(-1)
