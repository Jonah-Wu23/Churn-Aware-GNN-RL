"""消融实验冒烟测试

使用 CPU 和小规模配置进行轻量级测试，确保流程能打通。
不需要 8G 以上显存，所有测试在 CPU 上运行。

注意: torch_geometric 只在 AutoDL 上安装，相关测试会在本地自动跳过。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 检测 torch_geometric 是否可用
try:
    import torch_geometric
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# ============================================================================
# 测试配置：使用小规模参数以减少资源消耗
# ============================================================================

SMALL_MODEL_CONFIG = {
    "node_dim": 5,
    "edge_dim": 4,
    "hidden_dim": 16,  # 小规模
    "num_layers": 1,   # 单层
    "dropout": 0.0,
    "heads": 2,
}

DEVICE = torch.device("cpu")


# ============================================================================
# NodeOnlyGNN 模型测试
# ============================================================================

@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
class TestNodeOnlyGNN:
    """Node-Only GNN (GAT) 消融变体测试"""
    
    def test_init(self):
        """测试模型初始化"""
        from src.models.node_only_gnn import NodeOnlyGNN
        
        model = NodeOnlyGNN(
            node_dim=SMALL_MODEL_CONFIG["node_dim"],
            hidden_dim=SMALL_MODEL_CONFIG["hidden_dim"],
            num_layers=SMALL_MODEL_CONFIG["num_layers"],
            dropout=SMALL_MODEL_CONFIG["dropout"],
            heads=SMALL_MODEL_CONFIG["heads"],
        )
        
        assert model is not None
        assert model.node_dim == SMALL_MODEL_CONFIG["node_dim"]
        assert model.hidden_dim == SMALL_MODEL_CONFIG["hidden_dim"]
    
    def test_forward_shape(self):
        """测试前向传播输出形状"""
        from src.models.node_only_gnn import NodeOnlyGNN
        
        model = NodeOnlyGNN(
            node_dim=5,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            heads=2,
        )
        model.eval()
        
        # 模拟小规模输入
        num_nodes = 10
        num_graph_edges = 20
        num_action_edges = 5
        
        data = {
            "node_features": torch.randn(num_nodes, 5),
            "graph_edge_index": torch.randint(0, num_nodes, (2, num_graph_edges)),
            "action_edge_index": torch.randint(0, num_nodes, (2, num_action_edges)),
        }
        
        with torch.no_grad():
            q_values = model(data)
        
        assert q_values.shape == (num_action_edges,)
    
    def test_output_matches_edgeq_interface(self):
        """测试 NodeOnlyGNN 输出与 EdgeQGNN 接口一致"""
        from src.models.node_only_gnn import NodeOnlyGNN
        from src.models.edge_q_gnn import EdgeQGNN
        
        node_only = NodeOnlyGNN(
            node_dim=5, hidden_dim=16, num_layers=1, dropout=0.0, heads=2
        )
        edge_q = EdgeQGNN(
            node_dim=5, edge_dim=4, hidden_dim=16, num_layers=1, dropout=0.0
        )
        
        num_nodes = 10
        num_graph_edges = 20
        num_action_edges = 5
        
        data = {
            "node_features": torch.randn(num_nodes, 5),
            "graph_edge_index": torch.randint(0, num_nodes, (2, num_graph_edges)),
            "graph_edge_features": torch.randn(num_graph_edges, 4),
            "action_edge_index": torch.randint(0, num_nodes, (2, num_action_edges)),
            "edge_features": torch.randn(num_action_edges, 4),
        }
        
        with torch.no_grad():
            q_node_only = node_only(data)
            q_edge = edge_q(data)
        
        # 两者输出形状应一致
        assert q_node_only.shape == q_edge.shape == (num_action_edges,)


# ============================================================================
# RiskAblatedEnv 环境测试
# ============================================================================

@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
class TestRiskAblatedEnv:
    """w/o Risk-Awareness 环境包装器测试"""
    
    @pytest.fixture
    def minimal_env_config(self, tmp_path):
        """创建最小化测试环境配置"""
        from src.env.gym_env import EnvConfig
        
        # 使用项目中已有的测试数据（如果存在）
        graph_nodes = Path("data/processed/graph/layer2_nodes.parquet")
        graph_edges = Path("data/processed/graph/layer2_edges.parquet")
        od_glob = "data/processed/od_mapped/*.parquet"
        embeddings = Path("data/processed/graph/node2vec_embeddings.parquet")
        
        if not graph_nodes.exists():
            pytest.skip("测试数据不存在，跳过环境测试")
        
        return EnvConfig(
            max_horizon_steps=10,  # 小规模
            max_requests=20,       # 小规模
            seed=7,
            num_vehicles=2,
            vehicle_capacity=4,
            graph_nodes_path=str(graph_nodes),
            graph_edges_path=str(graph_edges),
            od_glob=od_glob,
            graph_embeddings_path=str(embeddings),
            reward_cvar_penalty=0.5,
            reward_fairness_weight=0.1,
        )
    
    def test_risk_ablated_env_init(self, minimal_env_config):
        """测试 RiskAblatedEnv 初始化"""
        from src.ablation.ablation_env_wrapper import RiskAblatedEnv
        
        env = RiskAblatedEnv(minimal_env_config)
        
        # fairness_weight 应全为 1.0
        assert all(w == 1.0 for w in env.fairness_weight.values())
        
        # reward 配置应被覆盖
        assert env.config.reward_cvar_penalty == 0.0
        assert env.config.reward_fairness_weight == 0.0
    
    def test_risk_ablated_features(self, minimal_env_config):
        """测试 RiskAblatedEnv 特征消融"""
        from src.ablation.ablation_env_wrapper import RiskAblatedEnv
        from src.env.gym_env import EventDrivenEnv
        
        # 对比原始环境和消融环境的特征
        original_env = EventDrivenEnv(minimal_env_config)
        ablated_env = RiskAblatedEnv(minimal_env_config)
        
        original_env.reset()
        ablated_env.reset()
        
        original_features = original_env.get_feature_batch()
        ablated_features = ablated_env.get_feature_batch()
        
        # edge_features[:, 1] (delta_cvar) 应为 0
        if ablated_features["edge_features"].shape[0] > 0:
            assert np.allclose(ablated_features["edge_features"][:, 1], 0.0)


# ============================================================================
# AblationEvaluator 测试
# ============================================================================

@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
class TestAblationEvaluator:
    """消融评估器测试"""
    
    def test_ablation_result_schema(self):
        """测试 AblationResult 数据结构"""
        from src.ablation.ablation_evaluator import AblationResult
        
        result = AblationResult(
            ablation_type="no_edge",
            tacc=1000.0,
            churn_rate=0.1,
            wait_time_p95=120.0,
            suburban_service_rate=0.8,
            waiting_churn_count=10,
            onboard_churn_count=5,
            service_gini=0.2,
            service_rate=0.85,
            served_count=85,
            total_requests=100,
            episodes=10,
        )
        
        result_dict = result.to_dict()
        
        # 验证必需字段
        assert "ablation_type" in result_dict
        assert "main_metrics" in result_dict
        assert "diagnostic_metrics" in result_dict
        
        main = result_dict["main_metrics"]
        assert "tacc" in main
        assert "churn_rate" in main
        assert "wait_time_p95" in main
        assert "suburban_service_rate" in main
    
    def test_comparison_table_format(self):
        """测试对比表格格式"""
        from src.ablation.ablation_evaluator import AblationResult, generate_comparison_table
        
        results = [
            AblationResult(
                ablation_type="full",
                tacc=1000.0, churn_rate=0.1, wait_time_p95=120.0,
                suburban_service_rate=0.8, waiting_churn_count=10,
                onboard_churn_count=5, service_gini=0.2, service_rate=0.85,
                served_count=85, total_requests=100, episodes=10,
            ),
            AblationResult(
                ablation_type="no_edge",
                tacc=800.0, churn_rate=0.2, wait_time_p95=150.0,
                suburban_service_rate=0.7, waiting_churn_count=15,
                onboard_churn_count=10, service_gini=0.3, service_rate=0.75,
                served_count=75, total_requests=100, episodes=10,
            ),
        ]
        
        table = generate_comparison_table(results)
        
        # 验证表格格式
        assert "TACC" in table
        assert "Churn Rate" in table
        assert "Wait P95" in table
        assert "Suburban Rate" in table
        assert "Mobi-Churn (Full)" in table
        assert "w/o Edge-Encoding" in table


# ============================================================================
# Metrics 模块测试
# ============================================================================

class TestMetrics:
    """共享指标模块测试"""
    
    def test_stuckness_ratio(self):
        """测试 stuckness ratio 计算"""
        from src.eval.metrics import compute_stuckness_ratio
        
        # 全部可用
        mask_history = [[True, True, True], [True, True]]
        assert compute_stuckness_ratio(mask_history) == 0.0
        
        # 部分被 mask
        mask_history = [[True, False, True], [False, True]]
        ratio = compute_stuckness_ratio(mask_history)
        assert 0.0 < ratio < 1.0
        
        # 空历史
        assert compute_stuckness_ratio([]) == 0.0


# ============================================================================
# 端到端冒烟测试
# ============================================================================

@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
class TestAblationE2ESmoke:
    """端到端冒烟测试（使用随机权重，不需要预训练模型）"""
    
    @pytest.fixture
    def minimal_env_config(self):
        """创建最小化测试环境配置"""
        from src.env.gym_env import EnvConfig
        
        graph_nodes = Path("data/processed/graph/layer2_nodes.parquet")
        if not graph_nodes.exists():
            pytest.skip("测试数据不存在")
        
        return EnvConfig(
            max_horizon_steps=5,
            max_requests=10,
            seed=7,
            num_vehicles=1,
            vehicle_capacity=4,
            graph_nodes_path="data/processed/graph/layer2_nodes.parquet",
            graph_edges_path="data/processed/graph/layer2_edges.parquet",
            od_glob="data/processed/od_mapped/*.parquet",
            graph_embeddings_path="data/processed/graph/node2vec_embeddings.parquet",
        )
    
    def test_node_only_evaluation_flow(self, minimal_env_config):
        """测试 Node-Only GNN 评估流程"""
        from src.models.node_only_gnn import NodeOnlyGNN
        from src.ablation.ablation_evaluator import AblationEvaluator
        
        model = NodeOnlyGNN(
            node_dim=5, hidden_dim=16, num_layers=1, dropout=0.0, heads=2
        )
        
        evaluator = AblationEvaluator(
            ablation_type="no_edge",
            env_config=minimal_env_config,
            model=model,
            device=DEVICE,
            seed=7,
        )
        
        result = evaluator.evaluate(episodes=1)
        
        assert result.ablation_type == "no_edge"
        assert result.episodes == 1
        assert result.tacc >= 0
    
    def test_risk_ablated_evaluation_flow(self, minimal_env_config):
        """测试 Risk-Ablated 评估流程"""
        from src.models.edge_q_gnn import EdgeQGNN
        from src.ablation.ablation_evaluator import AblationEvaluator
        
        model = EdgeQGNN(
            node_dim=5, edge_dim=4, hidden_dim=16, num_layers=1, dropout=0.0
        )
        
        evaluator = AblationEvaluator(
            ablation_type="no_risk",
            env_config=minimal_env_config,
            model=model,
            device=DEVICE,
            seed=7,
        )
        
        result = evaluator.evaluate(episodes=1)
        
        assert result.ablation_type == "no_risk"
        assert result.churn_rate >= 0
        assert result.churn_rate <= 1.0
