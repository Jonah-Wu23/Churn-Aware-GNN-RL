"""消融实验环境包装器

提供三种消融变体的环境配置：
1. RiskAblatedEnv: w/o Risk-Awareness (同时消融 Reward + Feature)
2. create_direct_training_config: w/o Curriculum 配置生成

注意：Node-Only GNN 消融不需要修改环境，只需使用 NodeOnlyGNN 模型。
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.env.gym_env import EnvConfig, EventDrivenEnv


class RiskAblatedEnv(EventDrivenEnv):
    """w/o Risk-Awareness 环境包装器
    
    同时消融 Reward 层和 Feature 层的风险信号：
    
    Reward 层消融：
    - reward_cvar_penalty = 0.0
    - reward_fairness_weight = 0.0
    
    Feature 层消融：
    - node_feature[1] (risk_cvar) 用 risk_mean 替代
    - edge_feature[1] (delta_cvar) 用 0.0 替代
    - fairness_weight 全部固定为 1.0
    
    这确保模型既"看不到"尾部风险信号，也不会因尾部风险受惩罚。
    """
    
    def __init__(self, config: EnvConfig) -> None:
        # 强制禁用 reward 层的风险惩罚
        ablated_config = dataclasses.replace(
            config,
            reward_cvar_penalty=0.0,
            reward_fairness_weight=0.0,
        )
        super().__init__(ablated_config)
        
        # Feature 层：fairness_weight 全部设为 1.0
        self.fairness_weight = {stop_id: 1.0 for stop_id in self.stop_ids}
    
    def _compute_waiting_risks(self) -> Dict[int, Tuple[float, float, int]]:
        """覆写风险计算：用 mean 替代 cvar
        
        返回格式：{stop_id: (risk_mean, risk_mean, count)}
        注意：第二项本应是 risk_cvar，这里用 risk_mean 替代。
        """
        risks: Dict[int, Tuple[float, float, int]] = {}
        for stop_id, queue in self.waiting.items():
            if not queue:
                risks[int(stop_id)] = (0.0, 0.0, 0)
                continue
            waits = [self.current_time - req["request_time_sec"] for req in queue]
            probs = [self._waiting_churn_prob(wait_sec) for wait_sec in waits]
            mean_prob = float(np.mean(probs)) if probs else 0.0
            # CVaR 位置用 mean 替代，保持维度一致
            risks[int(stop_id)] = (mean_prob, mean_prob, len(queue))
        return risks
    
    def get_feature_batch(self, k_hop: int | None = None) -> Dict[str, np.ndarray]:
        """覆写特征生成：edge_feature 中用 0.0 替代 delta_cvar
        
        edge_features 布局：
        - [0] delta_eta_max: 保留
        - [1] delta_cvar: 置零（消融）
        - [2] violation_count: 保留
        - [3] travel_time: 保留
        - [4] fleet_potential: 保留（若FAEP启用，独立于风险消融）
        
        FAEP兼容性：当 use_fleet_potential=True 时，edge_features 为5维，
        第5维 fleet_potential 不受风险消融影响，保持原值。
        """
        batch = super().get_feature_batch(k_hop)
        # edge_features[:, 1] 原为 delta_cvar，置零
        # 注意：只修改第2维，保留其他维度（包括FAEP的第5维）
        if batch["edge_features"].shape[0] > 0:
            batch["edge_features"][:, 1] = 0.0
        return batch


def create_risk_ablated_config(base_config: EnvConfig) -> EnvConfig:
    """创建 w/o Risk-Awareness 的环境配置
    
    使用此配置创建 RiskAblatedEnv 实例。
    """
    return dataclasses.replace(
        base_config,
        reward_cvar_penalty=0.0,
        reward_fairness_weight=0.0,
        fairness_gamma=0.0,  # 所有站点权重统一为 1.0
    )


def create_direct_training_config(base_config: EnvConfig) -> EnvConfig:
    """创建 w/o Curriculum 的环境配置
    
    直接使用真实世界场景（L3 阶段）的参数，跳过课程学习。
    所有约束（Hard Mask）保持不变。
    """
    return dataclasses.replace(
        base_config,
        # L3 阶段默认参数（真实世界场景）
        # 硬约束保持不变
    )


# 消融类型枚举
ABLATION_TYPES = {
    "no_edge": {
        "description": "w/o Edge-Encoding: 使用 NodeOnlyGNN，移除边特征",
        "model_type": "node_only_gnn",
        "env_wrapper": None,  # 使用标准环境
    },
    "no_risk": {
        "description": "w/o Risk-Awareness: 同时消融 Reward 和 Feature 层的风险信号",
        "model_type": "edge_q_gnn",  # 使用原模型
        "env_wrapper": RiskAblatedEnv,
    },
    "no_curriculum": {
        "description": "w/o Curriculum: 直接在 L3 阶段训练，跳过课程学习",
        "model_type": "edge_q_gnn",  # 使用原模型
        "env_wrapper": None,  # 使用标准环境
        "skip_curriculum": True,
    },
}
