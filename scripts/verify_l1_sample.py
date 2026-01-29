"""验证 L1 采样后的前 2000 条 OD 的跨强连通分量比例。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import networkx as nx

def main():
    print("=== 验证 L1 采样后的前 2000 条 OD ===\n")
    
    # 加载图并获取强连通分量
    nodes = pd.read_parquet("data/processed/graph/layer2_nodes.parquet")
    edges = pd.read_parquet("data/processed/graph/layer2_edges.parquet")
    node_ids = set(nodes["gnn_node_id"].astype(int).tolist())
    
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    for _, row in edges.iterrows():
        G.add_edge(int(row["source"]), int(row["target"]), weight=float(row["travel_time_sec"]))
    
    sccs = list(nx.strongly_connected_components(G))
    node_to_scc = {}
    for i, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = i
    
    # 模拟 L1 采样
    print("模拟 L1 curriculum 采样过程...")
    
    # 加载完整 OD
    od_files = list(Path("data/processed/od_mapped").glob("*.parquet"))
    full_od = pd.concat([pd.read_parquet(f) for f in od_files], ignore_index=True)
    
    # annotate_od
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
    
    pickup_lons = nodes_df.loc[full_od["pickup_stop_id"].astype(int), "lon"].values
    pickup_lats = nodes_df.loc[full_od["pickup_stop_id"].astype(int), "lat"].values
    dropoff_lons = nodes_df.loc[full_od["dropoff_stop_id"].astype(int), "lon"].values
    dropoff_lats = nodes_df.loc[full_od["dropoff_stop_id"].astype(int), "lat"].values
    
    full_od["pickup_dist_center_m"] = haversine(pickup_lons, pickup_lats, center_lon, center_lat)
    full_od["trip_dist_m"] = haversine(pickup_lons, pickup_lats, dropoff_lons, dropoff_lats)
    
    # L1 筛选
    center_max = float(full_od["pickup_dist_center_m"].quantile(0.3))
    short_max = float(full_od["trip_dist_m"].quantile(0.3))
    
    l1_subset = full_od[
        (full_od["pickup_dist_center_m"] <= center_max) &
        (full_od["trip_dist_m"] <= short_max)
    ].copy()
    
    print(f"L1 筛选后子集: {len(l1_subset)} 条")
    
    # 采样（模拟 _weighted_sample）
    rng = np.random.default_rng(7)  # 与训练使用相同的 seed
    target = int(len(full_od) * 1.2)  # density_multiplier=1.2
    
    print(f"采样目标: {target} 条")
    print(f"需要替换采样: {target > len(l1_subset)}")
    
    sampled = l1_subset.sample(n=target, replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
    
    # 时间缩放
    time_scale = 0.6
    t0 = sampled["tpep_pickup_datetime"].iloc[0]
    deltas = (sampled["tpep_pickup_datetime"] - t0).dt.total_seconds()
    sampled["tpep_pickup_datetime"] = t0 + pd.to_timedelta(deltas * time_scale, unit="s")
    
    # 按时间排序
    sampled = sampled.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    
    print(f"\n采样后 OD 总数: {len(sampled)}")
    
    # 分析前 2000 条
    first_2000 = sampled.head(2000)
    
    cross_scc = 0
    same_scc = 0
    no_path_count = 0
    
    for _, row in first_2000.iterrows():
        pickup = int(row["pickup_stop_id"])
        dropoff = int(row["dropoff_stop_id"])
        
        if pickup in node_to_scc and dropoff in node_to_scc:
            if node_to_scc[pickup] != node_to_scc[dropoff]:
                cross_scc += 1
                no_path_count += 1
            else:
                same_scc += 1
                # 即使在同一分量内，也可能没有有向路径
                if not nx.has_path(G, pickup, dropoff):
                    no_path_count += 1
    
    print(f"\n=== 前 2000 条采样 OD 分析 ===")
    print(f"同一强连通分量: {same_scc}")
    print(f"跨强连通分量: {cross_scc}")
    print(f"跨分量比例: {cross_scc / (cross_scc + same_scc) * 100:.1f}%")
    print(f"无有向路径总数: {no_path_count}")
    
    if cross_scc > 1000:
        print(f"\n[!] 发现问题！")
        print(f"    采样后的前 2000 条 OD 中有 {cross_scc} 条跨强连通分量")
        print(f"    这就是 structural_unserviceable=1715 的原因！")
        print(f"\n    问题根源：L1 子集虽然只有 ~0.1% 跨分量，")
        print(f"    但经过 13 倍过采样和时间缩放后，在排序时被打乱了。")
        print(f"\n    解决方案：在 curriculum 生成时，过滤掉跨强连通分量的 OD。")

if __name__ == "__main__":
    main()
