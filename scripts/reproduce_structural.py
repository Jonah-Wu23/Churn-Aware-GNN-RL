"""使用训练环境代码精确复现 structural_unserviceable 计算。

直接使用 EnvConfig 和 EventDrivenEnv 来验证问题。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.gym_env import EnvConfig, EventDrivenEnv
import numpy as np

def main():
    print("=== 使用训练环境代码精确复现 ===\n")
    
    # 使用通用的 OD 数据路径
    od_path = "data/processed/od_mapped/*.parquet"
    
    config = EnvConfig(
        max_horizon_steps=2000,
        mask_alpha=1.5,
        walk_threshold_sec=600,
        max_requests=2000,  # 根据训练日志，每 episode ~2043 请求
        seed=7,
        num_vehicles=50,
        vehicle_capacity=6,
        request_timeout_sec=600,
        churn_tol_sec=300,
        churn_beta=0.02,
        od_glob=od_path,
        graph_nodes_path="data/processed/graph/layer2_nodes.parquet",
        graph_edges_path="data/processed/graph/layer2_edges.parquet",
        graph_embeddings_path="data/processed/graph/node2vec_embeddings.parquet",
    )
    
    print(f"配置: max_requests={config.max_requests}, od_glob={config.od_glob}")
    
    # 创建环境
    print("\n加载环境...")
    env = EventDrivenEnv(config)
    
    print(f"\n=== 环境加载结果 ===")
    print(f"图节点数: {len(env.stop_ids)}")
    print(f"请求总数: {len(env.requests)}")
    
    # 分析 structural
    structural_count = 0
    reasons = {
        "direct_time_inf": 0,
        "structural_unreachable_flag": 0,
        "valid": 0,
    }
    
    inf_pairs = []
    
    for req in env.requests:
        is_structural = req.get("structural_unserviceable", False)
        direct_time = req.get("direct_time_sec", float("inf"))
        structural_flag = req.get("structural_unreachable", False)
        
        if is_structural:
            structural_count += 1
            if not np.isfinite(direct_time):
                reasons["direct_time_inf"] += 1
                if len(inf_pairs) < 20:
                    inf_pairs.append((req["pickup_stop_id"], req["dropoff_stop_id"], direct_time))
            elif structural_flag:
                reasons["structural_unreachable_flag"] += 1
        else:
            reasons["valid"] += 1
    
    print(f"\n=== Structural 分析 ===")
    print(f"structural_unserviceable 总数: {structural_count}")
    print(f"分桶:")
    for k, v in reasons.items():
        print(f"  {k}: {v}")
    
    if inf_pairs:
        print(f"\n前20个 direct_time=inf 的 OD 对:")
        for p, d, t in inf_pairs:
            # 检查这些节点是否在图中
            p_in_graph = p in env.stop_index
            d_in_graph = d in env.stop_index
            print(f"  pickup={p} (in_graph={p_in_graph}), dropoff={d} (in_graph={d_in_graph})")
    
    # 运行一个完整的 episode
    print("\n=== 运行一个 Episode ===")
    obs = env.reset()
    print(f"Reset 后 structurally_unserviceable: {env.structurally_unserviceable}")
    
    # 这是关键！在 reset 时就已经统计了 structural
    # 因为 _schedule_initial_events() 会遍历所有请求并标记 structural

if __name__ == "__main__":
    main()
