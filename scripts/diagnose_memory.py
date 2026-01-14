"""诊断训练时显存占用的脚本"""

import sys
from pathlib import Path
import torch
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

def format_bytes(bytes_num):
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.2f} TB"

def diagnose_memory():
    """诊断训练配置和预期显存占用"""
    
    print("=" * 70)
    print("显存占用诊断")
    print("=" * 70)
    
    # 1. 检查CUDA可用性
    if not torch.cuda.is_available():
        print("\n⚠ CUDA不可用，无法诊断显存")
        return
    
    device = torch.device("cuda")
    print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ 总显存: {format_bytes(torch.cuda.get_device_properties(0).total_memory)}")
    
    # 2. 读取配置
    config_path = Path(__file__).parent.parent / "configs" / "manhattan.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    batch_size = cfg["train"]["batch_size"]
    hidden_dim = cfg["model"]["hidden_dim"]
    num_layers = cfg["model"]["num_layers"]
    gradient_steps = cfg["train"]["gradient_steps"]
    use_amp = cfg["train"].get("use_amp", False)
    
    print(f"\n配置参数:")
    print(f"  - batch_size: {batch_size}")
    print(f"  - hidden_dim: {hidden_dim}")
    print(f"  - num_layers: {num_layers}")
    print(f"  - gradient_steps: {gradient_steps}")
    print(f"  - use_amp: {use_amp}")
    
    # 3. 估算显存占用
    print(f"\n显存占用估算:")
    
    # 模型参数
    node_dim = 5
    edge_dim = 4
    
    # TransformerConv参数量估算
    # 每层: in_channels * out_channels * (1 + edge_dim) * heads
    conv_params_per_layer = hidden_dim * hidden_dim * (1 + edge_dim) * 1
    total_conv_params = conv_params_per_layer * num_layers
    
    # node_encoder
    encoder_params = node_dim * hidden_dim + hidden_dim
    
    # q_head
    q_head_params = (hidden_dim * 2 + edge_dim) * hidden_dim + hidden_dim + hidden_dim
    
    total_params = encoder_params + total_conv_params + q_head_params
    model_size = total_params * 4  # FP32
    
    print(f"  模型参数: {total_params:,} ({format_bytes(model_size)})")
    
    # 优化器状态（Adam = 2x参数）
    optimizer_size = model_size * 2
    print(f"  优化器状态: {format_bytes(optimizer_size)}")
    
    # 图结构常驻显存（假设500个节点，2000条边）
    num_nodes = 500
    num_edges = 2000
    graph_node_features = num_nodes * node_dim * 4  # FP32
    graph_edge_features = num_edges * edge_dim * 4
    graph_edge_index = num_edges * 2 * 8  # int64
    graph_size = graph_node_features + graph_edge_features + graph_edge_index
    
    print(f"  图结构: {format_bytes(graph_size)}")
    
    # 批量激活显存（关键！）
    # 每个样本需要存储的中间激活：
    # - GNN每层输出: num_nodes * hidden_dim
    # - Q值计算中间层: 若干
    bytes_per_sample = 2 if use_amp else 4  # FP16或FP32
    
    # 每个样本的激活
    activation_per_sample = (
        num_nodes * hidden_dim * num_layers +  # GNN层输出
        hidden_dim * 2  # Q-head中间层
    ) * bytes_per_sample
    
    # 批量激活 = batch_size * gradient_steps（因为要堆叠）
    effective_batch = batch_size * gradient_steps
    batch_activation_size = activation_per_sample * effective_batch
    
    print(f"  批量激活 (batch_size={batch_size}, gradient_steps={gradient_steps}):")
    print(f"    - 单样本激活: {format_bytes(activation_per_sample)}")
    print(f"    - 有效批量: {effective_batch}")
    print(f"    - 总激活显存: {format_bytes(batch_activation_size)}")
    
    # ReplayBuffer不占显存（在CPU）
    print(f"  ReplayBuffer: 0 (在CPU内存)")
    
    # 总计
    total_estimated = model_size + optimizer_size + graph_size + batch_activation_size
    print(f"\n预计总显存占用: {format_bytes(total_estimated)}")
    
    # 4. 实际测试
    print(f"\n实际显存测试:")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    try:
        from src.models.edge_q_gnn import EdgeQGNN
        
        model = EdgeQGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
        ).to(device)
        
        print(f"  模型已加载: {format_bytes(torch.cuda.memory_allocated())}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print(f"  优化器已创建: {format_bytes(torch.cuda.memory_allocated())}")
        
        # 模拟图数据
        graph_node_features = torch.randn(num_nodes, node_dim, device=device)
        graph_edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        graph_edge_features = torch.randn(num_edges, edge_dim, device=device)
        
        print(f"  图数据已加载: {format_bytes(torch.cuda.memory_allocated())}")
        
        # 模拟批量训练
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        all_q_preds = []
        
        for _ in range(min(effective_batch, 100)):  # 限制测试批量
            # 模拟单个样本的前向传播
            action_edges = 10
            action_edge_index = torch.randint(0, num_nodes, (2, action_edges), device=device)
            action_edge_features = torch.randn(action_edges, edge_dim, device=device)
            
            data = {
                "node_features": graph_node_features,
                "graph_edge_index": graph_edge_index,
                "graph_edge_features": graph_edge_features,
                "action_edge_index": action_edge_index,
                "edge_features": action_edge_features,
            }
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    q = model(data)
            else:
                q = model(data)
            
            all_q_preds.append(q[0])  # 取第一个Q值
        
        print(f"  批量前向完成: {format_bytes(torch.cuda.memory_allocated())}")
        print(f"  峰值显存: {format_bytes(torch.cuda.max_memory_allocated())}")
        
        # 模拟反向传播
        optimizer.zero_grad()
        q_tensor = torch.stack(all_q_preds)
        targets = torch.randn_like(q_tensor)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = torch.nn.functional.mse_loss(q_tensor, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = torch.nn.functional.mse_loss(q_tensor, targets)
            loss.backward()
            optimizer.step()
        
        print(f"  反向传播完成: {format_bytes(torch.cuda.memory_allocated())}")
        print(f"  峰值显存: {format_bytes(torch.cuda.max_memory_allocated())}")
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 建议
    print(f"\n优化建议:")
    
    current_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    target_mem = 20 * 1024**3  # 20GB
    
    if current_mem < target_mem * 0.5:
        print(f"  ⚠ 当前显存利用率极低 ({format_bytes(current_mem)} / 20GB)")
        print(f"  建议:")
        print(f"    1. 确认配置文件已重新加载")
        print(f"    2. 检查训练时是否真的用了batch_size={batch_size}")
        print(f"    3. 可进一步增大: batch_size=16384, hidden_dim=768")
    elif current_mem < target_mem * 0.8:
        print(f"  ✓ 显存利用合理 ({format_bytes(current_mem)} / 20GB)")
        print(f"  可小幅提升: batch_size={int(batch_size * 1.2)}")
    else:
        print(f"  ✓ 显存利用充分 ({format_bytes(current_mem)} / 20GB)")
    
    print("=" * 70)

if __name__ == "__main__":
    diagnose_memory()
