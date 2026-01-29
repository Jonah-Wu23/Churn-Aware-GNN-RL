"""测试环境中的可达性统计脚本

该脚本分析当前环境中的人员可达性统计，包括：
- 结构性不可达人数统计
- Mismatch Rate：欧氏最近站 ≠ 步行路网最近站的比例
- Barrier Impact：因隔离带/河流导致的不可达统计
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config


def load_od_mapping_audit(audit_path: Path) -> Dict:
    """加载OD映射审计报告"""
    if not audit_path.exists():
        raise FileNotFoundError(f"OD映射审计报告不存在: {audit_path}")
    
    with open(audit_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_graph_audit(audit_path: Path) -> Dict:
    """加载图构建审计报告"""
    if not audit_path.exists():
        raise FileNotFoundError(f"图构建审计报告不存在: {audit_path}")
    
    with open(audit_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_total_statistics(audit_data: Dict) -> Dict:
    """计算总体统计信息"""
    files = audit_data.get("files", [])
    
    total_rows = sum(file.get("rows", 0) for file in files)
    total_unreachable = sum(file.get("structural_unreachability_count", 0) for file in files)
    total_barrier_impact = sum(file.get("barrier_impact_count", 0) for file in files)
    
    # 计算加权平均的mismatch rate
    weighted_mismatch_rate = sum(
        file.get("mismatch_rate", 0) * file.get("rows", 0) 
        for file in files
    ) / total_rows if total_rows > 0 else 0
    
    # 计算加权平均的不可达率
    weighted_unreach_rate = sum(
        file.get("structural_unreachability_rate", 0) * file.get("rows", 0) 
        for file in files
    ) / total_rows if total_rows > 0 else 0
    
    # 计算加权平均的barrier impact率
    weighted_barrier_rate = sum(
        file.get("barrier_impact_rate", 0) * file.get("rows", 0) 
        for file in files
    ) / total_rows if total_rows > 0 else 0
    
    return {
        "total_requests": total_rows,
        "total_structural_unreachable": total_unreachable,
        "total_barrier_impacted": total_barrier_impact,
        "total_reachable": total_rows - total_unreachable,
        "weighted_mismatch_rate": weighted_mismatch_rate,
        "weighted_structural_unreach_rate": weighted_unreach_rate,
        "weighted_barrier_impact_rate": weighted_barrier_rate,
        "reachable_percentage": (total_rows - total_unreachable) / total_rows * 100 if total_rows > 0 else 0,
        "unreachable_percentage": total_unreachable / total_rows * 100 if total_rows > 0 else 0
    }


def print_summary(total_stats: Dict, graph_audit: Dict):
    """打印统计摘要"""
    print("=" * 80)
    print("                    可达性测试报告")
    print("=" * 80)
    
    print(f"\n📊 基础统计:")
    print(f"  总请求数: {total_stats['total_requests']:,}")
    print(f"  可达人数: {total_stats['total_reachable']:,} ({total_stats['reachable_percentage']:.2f}%)")
    print(f"  结构性不可达人数: {total_stats['total_structural_unreachable']:,} ({total_stats['unreachable_percentage']:.2f}%)")
    
    print(f"\n🚧 Mismatch Rate (欧氏 vs 路网最近站不一致):")
    print(f"  加权平均Mismatch Rate: {total_stats['weighted_mismatch_rate']:.4f} ({total_stats['weighted_mismatch_rate']*100:.2f}%)")
    print(f"  说明: 证明了Voronoi映射的必要性，{total_stats['weighted_mismatch_rate']*100:.1f}%的请求欧氏最近站与步行路网最近站不同")
    
    print(f"\n🌊 Barrier Impact (隔离带/河流影响):")
    print(f"  受屏障影响请求数: {total_stats['total_barrier_impacted']:,}")
    print(f"  加权平均Barrier Impact Rate: {total_stats['weighted_barrier_impact_rate']:.6f} ({total_stats['weighted_barrier_impact_rate']*100:.4f}%)")
    print(f"  说明: 因地理屏障导致步行距离显著增加的请求比例")
    
    print(f"\n🗺️ 网络统计:")
    print(f"  车站数量: {graph_audit.get('node_count', 'N/A')}")
    print(f"  边数量: {graph_audit.get('edge_count', 'N/A')}")
    print(f"  是否弱连通: {'是' if graph_audit.get('weakly_connected', False) else '否'}")
    print(f"  是否强连通: {'是' if graph_audit.get('strongly_connected', False) else '否'}")
    
    print(f"\n📈 可达性分析:")
    if total_stats['unreachable_percentage'] < 1:
        print("  ✅ 可达性良好: 不可达率低于1%")
    elif total_stats['unreachable_percentage'] < 5:
        print("  ⚠️  可达性一般: 不可达率在1-5%之间")
    else:
        print("  ❌ 可达性较差: 不可达率超过5%")
    
    if total_stats['weighted_mismatch_rate'] > 0.3:
        print("  ✅ Voronoi映射价值高: Mismatch Rate超过30%，证明网络映射必要性")
    else:
        print("  📝 Voronoi映射仍有价值: 即使Mismatch Rate较低，仍能提高精度")
    
    print("\n" + "=" * 80)


def print_detailed_breakdown(audit_data: Dict):
    """打印详细分项统计"""
    print("\n📋 详细分项统计:")
    print("-" * 60)
    
    files = audit_data.get("files", [])
    for i, file in enumerate(files, 1):
        file_name = Path(file.get("input_path", "")).name
        print(f"\n文件 {i}: {file_name}")
        print(f"  请求数: {file.get('rows', 0):,}")
        print(f"  Mismatch Rate: {file.get('mismatch_rate', 0):.4f} ({file.get('mismatch_rate', 0)*100:.2f}%)")
        print(f"  结构性不可达: {file.get('structural_unreachability_count', 0)} ({file.get('structural_unreachability_rate', 0)*100:.3f}%)")
        print(f"    - 上车点不可达: {file.get('pickup_structural_unreachability_count', 0)}")
        print(f"    - 下车点不可达: {file.get('dropoff_structural_unreachability_count', 0)}")
        print(f"  屏障影响: {file.get('barrier_impact_count', 0)} ({file.get('barrier_impact_rate', 0)*100:.4f}%)")
        print(f"    - 上车点屏障: {file.get('pickup_barrier_impact_count', 0)}")
        print(f"    - 下车点屏障: {file.get('dropoff_barrier_impact_count', 0)}")


def analyze_reachability_trends(audit_data: Dict) -> Dict:
    """分析可达性趋势"""
    files = audit_data.get("files", [])
    if len(files) < 2:
        return {"trend": "insufficient_data"}
    
    # 比较第一个和最后一个文件
    first_file = files[0]
    last_file = files[-1]
    
    trends = {
        "mismatch_trend": "stable",
        "unreach_trend": "stable", 
        "barrier_trend": "stable"
    }
    
    # 计算变化
    mismatch_change = last_file.get('mismatch_rate', 0) - first_file.get('mismatch_rate', 0)
    unreach_change = last_file.get('structural_unreachability_rate', 0) - first_file.get('structural_unreachability_rate', 0)
    barrier_change = last_file.get('barrier_impact_rate', 0) - first_file.get('barrier_impact_rate', 0)
    
    # 判断趋势 (变化超过0.1%认为有显著变化)
    if abs(mismatch_change) > 0.001:
        trends["mismatch_trend"] = "increasing" if mismatch_change > 0 else "decreasing"
    if abs(unreach_change) > 0.0001:
        trends["unreach_trend"] = "increasing" if unreach_change > 0 else "decreasing"
    if abs(barrier_change) > 0.00001:
        trends["barrier_trend"] = "increasing" if barrier_change > 0 else "decreasing"
    
    return trends


def main():
    """主函数"""
    config_path = "configs/manhattan.yaml"
    
    try:
        cfg = load_config(config_path)
        paths_cfg = cfg.get("paths", {})
        
        od_audit_path = Path(paths_cfg.get("od_audit_path", "reports/audit/od_mapping.json"))
        graph_audit_path = Path(paths_cfg.get("graph_audit_path", "reports/audit/graph_build.json"))
        
        # 加载审计数据
        od_audit = load_od_mapping_audit(od_audit_path)
        graph_audit = load_graph_audit(graph_audit_path)
        
        # 计算总体统计
        total_stats = calculate_total_statistics(od_audit)
        
        # 打印报告
        print_summary(total_stats, graph_audit)
        print_detailed_breakdown(od_audit)
        
        # 分析趋势
        trends = analyze_reachability_trends(od_audit)
        if trends.get("trend") != "insufficient_data":
            print(f"\n📈 趋势分析:")
            print(f"  Mismatch Rate趋势: {trends['mismatch_trend']}")
            print(f"  不可达率趋势: {trends['unreach_trend']}")
            print(f"  屏障影响趋势: {trends['barrier_trend']}")
        
        print(f"\n📄 详细数据文件:")
        print(f"  OD映射审计: {od_audit_path}")
        print(f"  图构建审计: {graph_audit_path}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n💡 提示: 请确保已运行图构建和OD映射脚本")
        print("   运行命令:")
        print("   python scripts/build_graph.py --config configs/manhattan.yaml")
        print("   python scripts/map_od.py --config configs/manhattan.yaml")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
