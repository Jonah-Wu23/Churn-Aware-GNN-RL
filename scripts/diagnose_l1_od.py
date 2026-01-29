"""诊断 curriculum L1 生成的 OD 数据为什么有大量 structural_unserviceable。

问题：
- reproduce_structural.py 使用原始 od_mapped/*.parquet 只有 102 个 structural
- 训练使用 curriculum 生成的 od_L1.parquet 有 1715 个 structural

这个脚本会分析原始 OD 和 L1 OD 的差异。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import networkx as nx

def main():
    print("=== 诊断 L1 OD 的 structural_unserviceable 问题 ===\n")
    
    # 加载图
    nodes = pd.read_parquet("data/processed/graph/layer2_nodes.parquet")
    edges = pd.read_parquet("data/processed/graph/layer2_edges.parquet")
    node_ids = set(nodes["gnn_node_id"].astype(int).tolist())
    
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    for _, row in edges.iterrows():
        G.add_edge(int(row["source"]), int(row["target"]), weight=float(row["travel_time_sec"]))
    
    # 获取强连通分量
    sccs = list(nx.strongly_connected_components(G))
    print(f"强连通分量数量: {len(sccs)}")
    scc_sizes = sorted([len(scc) for scc in sccs], reverse=True)
    print(f"分量大小: {scc_sizes}")
    
    # 创建节点到分量的映射
    node_to_scc = {}
    for i, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = i
    
    # 加载原始 OD
    print("\n=== 原始 OD 分析 ===")
    od_files = list(Path("data/processed/od_mapped").glob("*.parquet"))
    od = pd.concat([pd.read_parquet(f) for f in od_files], ignore_index=True)
    od = od.sort_values("tpep_pickup_datetime").head(2000)
    
    # 分析跨强连通分量的情况
    cross_scc = 0
    same_scc = 0
    for _, row in od.iterrows():
        pickup = int(row["pickup_stop_id"])
        dropoff = int(row["dropoff_stop_id"])
        if pickup in node_to_scc and dropoff in node_to_scc:
            if node_to_scc[pickup] != node_to_scc[dropoff]:
                cross_scc += 1
            else:
                same_scc += 1
    
    print(f"前 2000 条原始 OD:")
    print(f"  同一强连通分量: {same_scc}")
    print(f"  跨强连通分量: {cross_scc}")
    print(f"  跨分量比例: {cross_scc / (cross_scc + same_scc) * 100:.1f}%")
    
    # 模拟 L1 阶段的 curriculum 生成
    print("\n=== 模拟 L1 curriculum 生成 ===")
    
    # 加载完整 OD
    full_od = pd.concat([pd.read_parquet(f) for f in od_files], ignore_index=True)
    print(f"完整 OD 数量: {len(full_od)}")
    
    # annotate_od: 计算 pickup_dist_center_m 和 trip_dist_m
    nodes_df = nodes.set_index("gnn_node_id")
    center_lon = float(nodes_df["lon"].mean())
    center_lat = float(nodes_df["lat"].mean())
    
    def haversine(lon1, lat1, lon2, lat2):
        rad = np.pi / 180.0
        dlon = (lon2 - lon1) * rad
        dlat = (lat2 - lat1) * rad
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371000.0 * c
    
    # 计算距离
    pickup_lons = nodes_df.loc[full_od["pickup_stop_id"].astype(int), "lon"].values
    pickup_lats = nodes_df.loc[full_od["pickup_stop_id"].astype(int), "lat"].values
    dropoff_lons = nodes_df.loc[full_od["dropoff_stop_id"].astype(int), "lon"].values
    dropoff_lats = nodes_df.loc[full_od["dropoff_stop_id"].astype(int), "lat"].values
    
    full_od["pickup_dist_center_m"] = haversine(pickup_lons, pickup_lats, center_lon, center_lat)
    full_od["trip_dist_m"] = haversine(pickup_lons, pickup_lats, dropoff_lons, dropoff_lats)
    
    # L1 筛选条件
    center_quantile = 0.3
    short_trip_quantile = 0.3
    center_max = float(full_od["pickup_dist_center_m"].quantile(center_quantile))
    short_max = float(full_od["trip_dist_m"].quantile(short_trip_quantile))
    
    print(f"L1 筛选阈值:")
    print(f"  center_max (30% quantile): {center_max:.1f} m")
    print(f"  short_max (30% quantile): {short_max:.1f} m")
    
    # 筛选
    l1_subset = full_od[
        (full_od["pickup_dist_center_m"] <= center_max) &
        (full_od["trip_dist_m"] <= short_max)
    ]
    print(f"\nL1 筛选后的子集大小: {len(l1_subset)}")
    
    # 分析 L1 子集中跨强连通分量的情况
    l1_cross_scc = 0
    l1_same_scc = 0
    for _, row in l1_subset.head(10000).iterrows():
        pickup = int(row["pickup_stop_id"])
        dropoff = int(row["dropoff_stop_id"])
        if pickup in node_to_scc and dropoff in node_to_scc:
            if node_to_scc[pickup] != node_to_scc[dropoff]:
                l1_cross_scc += 1
            else:
                l1_same_scc += 1
    
    print(f"\nL1 子集前 10000 条:")
    print(f"  同一强连通分量: {l1_same_scc}")
    print(f"  跨强连通分量: {l1_cross_scc}")
    print(f"  跨分量比例: {l1_cross_scc / (l1_cross_scc + l1_same_scc) * 100:.1f}%")
    
    # 模拟采样
    density_multiplier = 1.2
    target = int(len(full_od) * density_multiplier)
    print(f"\n采样目标数量: {target}")
    print(f"L1 子集大小: {len(l1_subset)}")
    print(f"采样需要替换: {target > len(l1_subset)}")
    
    # 检查：如果 L1 子集中有大量跨分量的 OD 对，那么采样后也会有大量 structural
    if l1_cross_scc / (l1_cross_scc + l1_same_scc) > 0.5:
        print("\n[!] 问题确认：L1 子集中有大量跨强连通分量的 OD 对！")
        print("    这些 OD 对在图中没有有向路径，所以被标记为 structural_unserviceable。")
        print("    解决方案：在 curriculum 生成时过滤掉跨强连通分量的 OD 对。")

if __name__ == "__main__":
    main()
