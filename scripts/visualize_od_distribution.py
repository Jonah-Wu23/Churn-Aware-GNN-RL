"""生成OD分布可视化图表（IEEE单栏格式）。

该脚本生成两个子图：
(a) OD流量地图：显示站点间的出行流量
(b) 出行距离分布直方图：显示出行距离的频率分布
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config

COLORS = {
    'NC': '#f57c6e',
    'CAAC_MLP': '#f2b56e',
    'RBF': '#fbe79e',
    'BR_AC': '#84c3b7',
    'KRIGING': '#88d7da',
}

IEEE_FONT = 'Times New Roman'
IEEE_FONTSIZE = 8
IEEE_WIDTH_INCH = 3.5
IEEE_DPI = 300


def _haversine_meters(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """计算两点间的Haversine距离（米）。"""
    rad = np.pi / 180.0
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000.0 * c


def load_od_and_stops(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载OD数据和站点数据。"""
    paths_cfg = config.get("paths", {})
    od_dir = Path(paths_cfg.get("od_output_dir", "data/processed/od_mapped"))
    stops_path = Path(paths_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"))
    
    od_files = list(od_dir.glob("*.parquet"))
    if not od_files:
        raise FileNotFoundError(f"未找到OD文件: {od_dir}")
    
    logging.info(f"找到 {len(od_files)} 个OD文件")
    od_dfs = [pd.read_parquet(f) for f in od_files]
    od_df = pd.concat(od_dfs, ignore_index=True)
    
    stops_df = pd.read_parquet(stops_path)
    
    logging.info(f"加载了 {len(od_df)} 条OD记录和 {len(stops_df)} 个站点")
    return od_df, stops_df


def compute_od_matrix(od_df: pd.DataFrame, stops_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算OD矩阵和站点坐标。"""
    valid_od = od_df[~od_df["structural_unreachable"]].copy()
    logging.info(f"有效OD记录: {len(valid_od)} / {len(od_df)} ({100*len(valid_od)/len(od_df):.1f}%)")
    
    od_matrix = valid_od.groupby(["pickup_stop_id", "dropoff_stop_id"]).size().reset_index(name="count")
    
    stop_coords = stops_df[["gnn_node_id", "lon", "lat"]].set_index("gnn_node_id")
    
    od_matrix = od_matrix.merge(
        stop_coords, left_on="pickup_stop_id", right_index=True, how="left"
    ).rename(columns={"lon": "pu_lon", "lat": "pu_lat"})
    
    od_matrix = od_matrix.merge(
        stop_coords, left_on="dropoff_stop_id", right_index=True, how="left"
    ).rename(columns={"lon": "do_lon", "lat": "do_lat"})
    
    od_matrix = od_matrix.dropna()
    
    od_matrix["distance_m"] = _haversine_meters(
        od_matrix["pu_lon"].to_numpy(),
        od_matrix["pu_lat"].to_numpy(),
        od_matrix["do_lon"].to_numpy(),
        od_matrix["do_lat"].to_numpy(),
    )
    od_matrix["distance_km"] = od_matrix["distance_m"] / 1000.0
    
    logging.info(f"OD矩阵包含 {len(od_matrix)} 个唯一OD对")
    return od_matrix, stop_coords


def plot_od_map(ax, od_matrix: pd.DataFrame, stop_coords: pd.DataFrame, bbox: Dict):
    """绘制OD流量地图（子图a）。"""
    flow_threshold = od_matrix["count"].quantile(0.5)
    od_filtered = od_matrix[od_matrix["count"] >= flow_threshold].copy()
    
    logging.info(f"绘制 {len(od_filtered)} 条流量线（阈值: {flow_threshold:.0f} trips）")
    
    max_flow = od_filtered["count"].max()
    min_flow = od_filtered["count"].min()
    
    for _, row in od_filtered.iterrows():
        normalized_flow = (row["count"] - min_flow) / (max_flow - min_flow + 1e-6)
        linewidth = 0.1 + normalized_flow * 1.5
        alpha = 0.1 + normalized_flow * 0.4
        
        ax.plot(
            [row["pu_lon"], row["do_lon"]],
            [row["pu_lat"], row["do_lat"]],
            color=COLORS['BR_AC'],
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )
    
    ax.scatter(
        stop_coords["lon"],
        stop_coords["lat"],
        s=3,
        c=COLORS['NC'],
        alpha=0.6,
        zorder=2,
        edgecolors='none',
    )
    
    ax.set_xlim(bbox["west"], bbox["east"])
    ax.set_ylim(bbox["south"], bbox["north"])
    ax.set_xlabel("Longitude (degrees)", fontsize=IEEE_FONTSIZE, fontfamily=IEEE_FONT)
    ax.set_ylabel("Latitude (degrees)", fontsize=IEEE_FONTSIZE, fontfamily=IEEE_FONT)
    ax.tick_params(labelsize=IEEE_FONTSIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(IEEE_FONT)
    
    ax.text(
        0.02, 0.98, "(a)",
        transform=ax.transAxes,
        fontsize=IEEE_FONTSIZE,
        fontweight='bold',
        fontfamily=IEEE_FONT,
        verticalalignment='top',
    )
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linewidth=0.5)


def plot_distance_histogram(ax, od_matrix: pd.DataFrame):
    """绘制出行距离分布直方图（子图b）。"""
    distances = od_matrix["distance_km"].to_numpy()
    
    distance_95 = np.percentile(distances, 95)
    distances_filtered = distances[distances <= distance_95]
    
    logging.info(f"距离统计: 均值={distances.mean():.2f}km, 中位数={np.median(distances):.2f}km, 95%分位={distance_95:.2f}km")
    
    ax.hist(
        distances_filtered,
        bins=50,
        color=COLORS['CAAC_MLP'],
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax.set_xlabel("Distance (km)", fontsize=IEEE_FONTSIZE, fontfamily=IEEE_FONT)
    ax.set_ylabel("Frequency (trips)", fontsize=IEEE_FONTSIZE, fontfamily=IEEE_FONT)
    ax.tick_params(labelsize=IEEE_FONTSIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(IEEE_FONT)
    
    ax.text(
        0.02, 0.98, "(b)",
        transform=ax.transAxes,
        fontsize=IEEE_FONTSIZE,
        fontweight='bold',
        fontfamily=IEEE_FONT,
        verticalalignment='top',
    )
    
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')


def generate_figure(od_matrix: pd.DataFrame, stop_coords: pd.DataFrame, bbox: Dict, output_path: Path):
    """生成完整的IEEE格式图表。"""
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(IEEE_WIDTH_INCH, 5.0))
    
    plot_od_map(axes[0], od_matrix, stop_coords, bbox)
    plot_distance_histogram(axes[1], od_matrix)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=IEEE_DPI, bbox_inches='tight')
    logging.info(f"图表已保存至: {output_path}")
    plt.close()


def generate_statistics(od_matrix: pd.DataFrame, od_df: pd.DataFrame, output_path: Path):
    """生成统计信息JSON。"""
    valid_od = od_df[~od_df["structural_unreachable"]]
    
    stats = {
        "total_records": int(len(od_df)),
        "valid_records": int(len(valid_od)),
        "structural_unreachable_count": int(od_df["structural_unreachable"].sum()),
        "structural_unreachable_rate": float(od_df["structural_unreachable"].mean()),
        "unique_od_pairs": int(len(od_matrix)),
        "total_trips": int(od_matrix["count"].sum()),
        "distance_stats_km": {
            "mean": float(od_matrix["distance_km"].mean()),
            "median": float(od_matrix["distance_km"].median()),
            "std": float(od_matrix["distance_km"].std()),
            "min": float(od_matrix["distance_km"].min()),
            "max": float(od_matrix["distance_km"].max()),
            "q25": float(od_matrix["distance_km"].quantile(0.25)),
            "q75": float(od_matrix["distance_km"].quantile(0.75)),
            "q95": float(od_matrix["distance_km"].quantile(0.95)),
        },
        "flow_stats": {
            "mean": float(od_matrix["count"].mean()),
            "median": float(od_matrix["count"].median()),
            "std": float(od_matrix["count"].std()),
            "min": int(od_matrix["count"].min()),
            "max": int(od_matrix["count"].max()),
            "q25": float(od_matrix["count"].quantile(0.25)),
            "q75": float(od_matrix["count"].quantile(0.75)),
            "q95": float(od_matrix["count"].quantile(0.95)),
        },
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"统计信息已保存至: {output_path}")
    
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成OD分布可视化图表")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--output", default="reports/figures/od_distribution.png", help="输出图片路径")
    parser.add_argument("--stats-output", default="reports/audit/od_statistics.json", help="统计信息输出路径")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    config = load_config(args.config)
    bbox = config.get("graph", {}).get("bbox", {})
    
    od_df, stops_df = load_od_and_stops(config)
    od_matrix, stop_coords = compute_od_matrix(od_df, stops_df)
    
    output_path = Path(args.output)
    generate_figure(od_matrix, stop_coords, bbox, output_path)
    
    stats_path = Path(args.stats_output)
    stats = generate_statistics(od_matrix, od_df, stats_path)
    
    logging.info("=" * 60)
    logging.info("OD分布统计摘要:")
    logging.info(f"  总记录数: {stats['total_records']:,}")
    logging.info(f"  有效记录数: {stats['valid_records']:,} ({100*stats['valid_records']/stats['total_records']:.1f}%)")
    logging.info(f"  结构不可达: {stats['structural_unreachable_count']:,} ({100*stats['structural_unreachable_rate']:.1f}%)")
    logging.info(f"  唯一OD对: {stats['unique_od_pairs']:,}")
    logging.info(f"  平均距离: {stats['distance_stats_km']['mean']:.2f} km")
    logging.info(f"  中位距离: {stats['distance_stats_km']['median']:.2f} km")
    logging.info(f"  平均流量: {stats['flow_stats']['mean']:.1f} trips/OD")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
