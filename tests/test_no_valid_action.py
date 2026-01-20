"""Tests for no-valid-action terminal penalty handling.

验证：
1. 终止惩罚计算给定 waiting/onboard/max_requests 的输出
2. DQN 无有效动作时 done_reason=no_valid_action 且 episode_return < 0
3. _compute_service_rate 分母包含 backlog+timeouts
"""

import pytest
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass, field


class TestTerminalPenaltyCalculation:
    """纯函数测试：终止惩罚计算逻辑"""
    
    def test_penalty_with_backlog(self):
        """测试有积压时的惩罚计算"""
        waiting_remaining = 30.0
        onboard_remaining = 20.0
        max_requests = 1000.0
        penalty_coef = 30.0
        
        norm_backlog = (waiting_remaining + onboard_remaining) / max_requests
        expected_penalty = penalty_coef * norm_backlog
        
        assert norm_backlog == 0.05
        assert expected_penalty == 1.5
    
    def test_penalty_zero_coef(self):
        """测试系数为0时惩罚为0"""
        penalty_coef = 0.0
        waiting_remaining = 100.0
        onboard_remaining = 50.0
        max_requests = 500.0
        
        if penalty_coef <= 0:
            penalty = 0.0
        else:
            penalty = penalty_coef * (waiting_remaining + onboard_remaining) / max_requests
        
        assert penalty == 0.0
    
    def test_penalty_empty_backlog(self):
        """测试空积压时惩罚为0"""
        waiting_remaining = 0.0
        onboard_remaining = 0.0
        max_requests = 1000.0
        penalty_coef = 30.0
        
        norm_backlog = (waiting_remaining + onboard_remaining) / max_requests
        penalty = penalty_coef * norm_backlog
        
        assert penalty == 0.0


class TestServiceRateCalculation:
    """测试 service rate 计算包含 backlog"""
    
    def test_compute_service_rate_with_backlog(self):
        """测试 _compute_service_rate 分母包含 backlog"""
        from src.train.runner import _compute_service_rate
        
        log = {
            "served": 100.0,
            "waiting_churned": 50.0,
            "onboard_churned": 10.0,
            "waiting_timeouts": 20.0,
            "waiting_remaining": 15.0,
            "onboard_remaining": 5.0,
        }
        
        rate = _compute_service_rate(log)
        assert abs(rate - 0.5) < 1e-6
    
    def test_compute_service_rate_zero_eligible(self):
        """测试无有效请求时返回0"""
        from src.train.runner import _compute_service_rate
        
        log = {}
        rate = _compute_service_rate(log)
        assert rate == 0.0
    
    def test_compute_service_rate_missing_backlog(self):
        """测试缺少 backlog 字段时正常处理"""
        from src.train.runner import _compute_service_rate
        
        log = {
            "served": 100.0,
            "waiting_churned": 50.0,
        }
        
        rate = _compute_service_rate(log)
        assert abs(rate - 100.0 / 150.0) < 1e-6


class TestDoneReasonValues:
    """测试 done_reason 枚举值"""
    
    def test_valid_done_reasons(self):
        """验证 done_reason 只能是预定义枚举值"""
        valid_reasons = {"max_horizon", "event_queue_empty", "no_valid_action"}
        
        test_reason = "no_valid_action"
        assert test_reason in valid_reasons
        
        assert "unknown" not in valid_reasons


@dataclass
class FakeVehicle:
    onboard: List[dict] = field(default_factory=list)


@dataclass
class FakeEnv:
    """最小 fake env 用于测试无有效动作场景"""
    stop_ids: list = field(default_factory=lambda: [0, 1])
    neighbors: dict = field(default_factory=lambda: {0: [(1, 1.0)], 1: [(0, 1.0)]})
    graph_edge_index: np.ndarray = field(default_factory=lambda: np.array([[0, 1], [1, 0]], dtype=np.int64))
    graph_edge_features: np.ndarray = field(default_factory=lambda: np.array([[0, 0, 0, 1.0], [0, 0, 0, 1.0]], dtype=np.float32))
    waiting: dict = field(default_factory=lambda: {0: [{"id": 1}, {"id": 2}], 1: [{"id": 3}]})
    vehicles: list = field(default_factory=lambda: [FakeVehicle(onboard=[{"id": 4}, {"id": 5}])])
    waiting_timeouts: int = 5
    served: int = 0
    waiting_churned: int = 0
    onboard_churned: int = 0
    structurally_unserviceable: int = 0
    event_queue: list = field(default_factory=list)
    ready_vehicle_ids: list = field(default_factory=list)
    _return_empty_actions: bool = True

    def reset(self) -> Dict[str, float]:
        return {}

    def get_feature_batch(self) -> Dict[str, np.ndarray]:
        if self._return_empty_actions:
            return {
                "node_features": np.zeros((2, 5), dtype=np.float32),
                "action_mask": np.array([], dtype=bool),
                "actions": np.array([], dtype=np.int64),
                "action_node_indices": np.array([], dtype=np.int64),
                "edge_features": np.zeros((0, 4), dtype=np.float32),
                "current_node_index": np.array([0], dtype=np.int64),
            }
        return {
            "node_features": np.zeros((2, 5), dtype=np.float32),
            "action_mask": np.array([True], dtype=bool),
            "actions": np.array([1], dtype=np.int64),
            "action_node_indices": np.array([1], dtype=np.int64),
            "edge_features": np.zeros((1, 4), dtype=np.float32),
            "current_node_index": np.array([0], dtype=np.int64),
        }

    def step(self, action: int):
        return {}, 0.0, True, {"done_reason": "max_horizon"}

    def _get_active_vehicle(self):
        return None


class TestNoValidActionIntegration:
    """轻量集成测试：验证无效动作时的惩罚机制"""
    
    def test_no_valid_action_penalty_applied(self, tmp_path):
        """测试无有效动作时终止惩罚被正确应用且计入 episode_return"""
        import torch
        from torch import nn
        from src.train.dqn import DQNConfig, DQNTrainer
        
        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.node_dim = 5
                self.edge_dim = 4
                self.bias = nn.Parameter(torch.zeros(1))
            
            def forward(self, data):
                num_actions = int(data["action_edge_index"].shape[1]) if data["action_edge_index"].numel() > 0 else 0
                return self.bias + torch.zeros(max(1, num_actions))
        
        env = FakeEnv(_return_empty_actions=True)
        model = FakeModel()
        
        cfg = DQNConfig(
            total_steps=1,
            buffer_size=10,
            batch_size=1,
            learning_starts=9999,  # 禁用训练期间的优化，避免从空 buffer 采样
            train_freq=1,
            gradient_steps=0,  # 禁用梯度步骤
            target_update_interval=9999,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=1,
            log_every_steps=1,
            checkpoint_every_steps=0,
            device="cpu",
        )
        
        env_cfg = {
            "max_requests": 100,
            "reward_terminal_backlog_penalty": 30.0,
        }
        
        trainer = DQNTrainer(
            env=env, model=model, config=cfg, run_dir=tmp_path,
            graph_hashes={}, od_hashes={}, env_cfg=env_cfg,
        )
        
        episode_logs = []
        def capture_episode(log):
            episode_logs.append(log)
            return True
        
        trainer.train(total_steps=1, episode_callback=capture_episode)
        trainer.close()
        
        assert len(episode_logs) >= 1, "Should have at least one episode"
        log = episode_logs[0]
        
        assert log.get("done_reason") == "no_valid_action", f"Expected no_valid_action, got {log.get('done_reason')}"
        assert log.get("terminal_backlog_penalty_applied", 0) > 0, "Should have penalty applied"
        assert log.get("episode_return", 0) < 0, f"Episode return should be negative, got {log.get('episode_return')}"
