"""精确模拟训练环境的 structural_unserviceable 计算逻辑。

复现 gym_env.py 中的 _load_requests() 方法的行为。
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.gym_env import EnvConfig, EventDrivenEnv

def main():
    print("=== 精确模拟训练环境 ===\n")
    
    # 加载图
    nodes = pd.read_parquet("data/processed/graph/layer2_nodes.parquet")
    edges = pd.read_parquet("data/processed/graph/layer2_edges.parquet")
    node_ids = set(nodes["gnn_node_id"].astype(int).tolist())
    print(f"图节点数: {len(node_ids)}")
    print(f"图边数: {len(edges)}")
    
    # 构建有向图计算最短路
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    for _, row in edges.iterrows():
        G.add_edge(int(row["source"]), int(row["target"]), weight=float(row["travel_time_sec"]))
    
    # 加载训练使用的 OD 数据
    od = pd.read_parquet("reports/runs/curriculum_20260114_165103/stage_L1/od_L1.parquet")
    print(f"\nOD 总数: {len(od)}")
    
    # 模拟 max_requests = 2000 (用户日志显示 episode_steps=2000)
    for max_req in [1500, 2000, 2043]:  # 测试不同的 max_requests 值
        # 模拟 gym_env._load_requests() 的逻辑
        od_sample = od.sort_values("tpep_pickup_datetime").head(max_req).copy()
        
        structural_count = 0
        reasons = {
            "structural_unreachable_flag": 0,
            "pickup_not_in_graph": 0,
            "dropoff_not_in_graph": 0,
            "no_directed_path": 0,
            "valid": 0,
        }
        
        # 预计算最短路矩阵（模拟 gym_env 的行为）
        stop_index = {int(stop_id): idx for idx, stop_id in enumerate(sorted(node_ids))}
        n = len(node_ids)
        shortest = np.full((n, n), np.inf, dtype=np.float32)
        np.fill_diagonal(shortest, 0.0)
        
        # 使用 networkx 计算所有最短路
        for src_id in node_ids:
            src_idx = stop_index[int(src_id)]
            lengths = nx.single_source_dijkstra_path_length(G, src_id, weight="weight")
            for dst_id, dist in lengths.items():
                dst_idx = stop_index.get(int(dst_id))
                if dst_idx is not None:
                    shortest[src_idx, dst_idx] = float(dist)
        
        for _, row in od_sample.iterrows():
            pickup = int(row["pickup_stop_id"])
            dropoff = int(row["dropoff_stop_id"])
            structural_flag = row.get("structural_unreachable", False)
            
            # 计算 direct_time_sec（模拟 _shortest_time 函数）
            if pickup == dropoff:
                direct_time = 0.0
            else:
                src_idx = stop_index.get(pickup)
                dst_idx = stop_index.get(dropoff)
                if src_idx is None or dst_idx is None:
                    direct_time = float("inf")
                else:
                    direct_time = float(shortest[src_idx, dst_idx])
            
            # 计算 structural_unserviceable
            is_structural = (not np.isfinite(direct_time)) or structural_flag
            
            if is_structural:
                structural_count += 1
                if structural_flag:
                    reasons["structural_unreachable_flag"] += 1
                elif pickup not in node_ids:
                    reasons["pickup_not_in_graph"] += 1
                elif dropoff not in node_ids:
                    reasons["dropoff_not_in_graph"] += 1
                else:
                    reasons["no_directed_path"] += 1
            else:
                reasons["valid"] += 1
        
        print(f"\n--- max_requests = {max_req} ---")
        print(f"structural_unserviceable: {structural_count}")
        print(f"分桶:")
        for k, v in reasons.items():
            print(f"  {k}: {v}")
    
    # 检查实际的训练配置和运行
    print("\n\n=== 检查实际训练运行 ===")
    
    # 读取训练日志中的元数据
    import json
    log_path = Path("reports/runs/curriculum_20260114_165103/stage_L1/train_log.jsonl")
    if log_path.exists():
        with open(log_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") == "meta":
                    print(f"训练元数据: {json.dumps(data['payload'].get('config', {}), indent=2)[:500]}...")
                    break
    
    # 检查 episode 的统计
    print("\n前几个 episode 的统计:")
    if log_path.exists():
        with open(log_path, "r") as f:
            count = 0
            for line in f:
                data = json.loads(line)
                if data.get("type") == "episode":
                    total = (data.get("served", 0) + 
                             data.get("waiting_churned", 0) + 
                             data.get("onboard_churned", 0) + 
                             data.get("structural_unserviceable", 0))
                    print(f"  step {data['step']}: served={data.get('served')}, "
                          f"waiting_churned={data.get('waiting_churned')}, "
                          f"onboard_churned={data.get('onboard_churned')}, "
                          f"structural={data.get('structural_unserviceable')}, "
                          f"total={total}")
                    count += 1
                    if count >= 5:
                        break

if __name__ == "__main__":
    main()
