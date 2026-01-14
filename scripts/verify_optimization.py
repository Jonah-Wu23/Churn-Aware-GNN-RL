"""验证优化后的代码可以正常运行"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_model_import():
    """验证模型模块导入"""
    try:
        from src.models.edge_q_gnn import EdgeQGNN
        print("✓ EdgeQGNN模型导入成功")
        return True
    except Exception as e:
        print(f"✗ 模型导入失败: {e}")
        return False

def verify_torch_geometric():
    """验证torch_geometric可用"""
    try:
        from torch_geometric.nn import TransformerConv
        print("✓ torch_geometric.nn.TransformerConv可用")
        return True
    except Exception as e:
        print(f"✗ torch_geometric导入失败: {e}")
        return False

def verify_model_instantiation():
    """验证模型可以实例化"""
    try:
        from src.models.edge_q_gnn import EdgeQGNN
        import torch
        
        model = EdgeQGNN(
            node_dim=5,
            edge_dim=4,
            hidden_dim=256,
            num_layers=3,
            dropout=0.15,
        )
        print(f"✓ 模型实例化成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch = {
            "node_features": torch.randn(10, 5),
            "graph_edge_index": torch.randint(0, 10, (2, 20)),
            "graph_edge_features": torch.randn(20, 4),
            "action_edge_index": torch.randint(0, 10, (2, 5)),
            "edge_features": torch.randn(5, 4),
        }
        output = model(batch)
        print(f"✓ 前向传播成功，输出形状: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ 模型实例化/前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_dqn_trainer():
    """验证DQNTrainer可以导入"""
    try:
        from src.train.dqn import DQNTrainer, DQNConfig
        print("✓ DQNTrainer和DQNConfig导入成功")
        
        # 验证AMP配置存在
        config = DQNConfig(use_amp=True)
        assert hasattr(config, 'use_amp'), "DQNConfig缺少use_amp字段"
        print(f"✓ DQNConfig支持use_amp={config.use_amp}")
        return True
    except Exception as e:
        print(f"✗ DQNTrainer导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_config_file():
    """验证配置文件正确性"""
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "manhattan.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        # 检查关键配置
        assert cfg["train"]["batch_size"] == 2048, "batch_size应为2048"
        assert cfg["train"]["use_amp"] == True, "use_amp应为true"
        assert cfg["model"]["hidden_dim"] == 256, "hidden_dim应为256"
        assert cfg["model"]["num_layers"] == 3, "num_layers应为3"
        assert cfg["train"]["train_freq"] == 8, "train_freq应为8"
        
        print("✓ manhattan.yaml配置正确")
        print(f"  - batch_size: {cfg['train']['batch_size']}")
        print(f"  - hidden_dim: {cfg['model']['hidden_dim']}")
        print(f"  - num_layers: {cfg['model']['num_layers']}")
        print(f"  - use_amp: {cfg['train']['use_amp']}")
        print(f"  - train_freq: {cfg['train']['train_freq']}")
        return True
    except Exception as e:
        print(f"✗ 配置文件验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("代码完整性验证")
    print("=" * 60)
    
    results = []
    
    print("\n1. 验证torch_geometric安装...")
    results.append(verify_torch_geometric())
    
    print("\n2. 验证模型模块导入...")
    results.append(verify_model_import())
    
    print("\n3. 验证模型实例化和前向传播...")
    results.append(verify_model_instantiation())
    
    print("\n4. 验证DQNTrainer模块...")
    results.append(verify_dqn_trainer())
    
    print("\n5. 验证配置文件...")
    results.append(verify_config_file())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ 所有验证通过！代码可以正常运行。")
        print("=" * 60)
        return 0
    else:
        print("✗ 部分验证失败，请检查上述错误。")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
