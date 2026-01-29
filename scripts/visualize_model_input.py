"""可视化模型输入数据（输入模型前最后一刻的真实数据）。

该脚本拦截训练时输入模型的真实数据，进行统计分析和可视化，
帮助诊断数据质量问题。

生成子图：
(a) 节点特征分布：risk_mean, risk_cvar, count, fairness_weight, geo_embedding
(b) 边特征分布：delta_eta_max, delta_cvar, count_violation, travel_time
(c) 动作掩码有效率
(d) 健康性检查结果
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.utils.config import load_config

# ─────────────────────────────────────────────────────────────
# IEEE 格式常量
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# 数据健康性检查
# ─────────────────────────────────────────────────────────────
@dataclass
class HealthCheckResult:
    """数据健康性检查结果。"""
    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


def check_health(features: Dict[str, np.ndarray]) -> HealthCheckResult:
    """检查模型输入数据的健康性。"""
    issues = []
    warnings = []
    stats = {}
    
    node_feat = features.get("node_features", np.array([]))
    edge_feat = features.get("edge_features", np.array([]))
    action_mask = features.get("action_mask", np.array([]))
    
    # 检查 Shape
    if node_feat.ndim != 2 or node_feat.shape[1] != 5:
        issues.append(f"node_features shape 异常: {node_feat.shape}, 预期 [N, 5]")
    if edge_feat.ndim != 2 or (edge_feat.shape[0] > 0 and edge_feat.shape[1] != 4):
        issues.append(f"edge_features shape 异常: {edge_feat.shape}, 预期 [A, 4]")
    
    stats["node_features_shape"] = list(node_feat.shape)
    stats["edge_features_shape"] = list(edge_feat.shape)
    stats["num_actions"] = int(len(action_mask))
    
    # 检查 NaN/Inf
    if np.any(~np.isfinite(node_feat)):
        issues.append("node_features 包含 NaN 或 Inf!")
    if np.any(~np.isfinite(edge_feat)):
        issues.append("edge_features 包含 NaN 或 Inf!")
    
    # 检查值域
    if node_feat.shape[0] > 0:
        risk_mean = node_feat[:, 0]
        risk_cvar = node_feat[:, 1]
        count = node_feat[:, 2]
        fairness_w = node_feat[:, 3]
        
        if np.any(risk_mean < 0) or np.any(risk_mean > 1):
            warnings.append(f"risk_mean 超出 [0,1]: [{risk_mean.min():.4f}, {risk_mean.max():.4f}]")
        if np.any(risk_cvar < 0) or np.any(risk_cvar > 1):
            warnings.append(f"risk_cvar 超出 [0,1]: [{risk_cvar.min():.4f}, {risk_cvar.max():.4f}]")
        if np.any(count < 0):
            issues.append(f"count 包含负值: min={count.min()}")
        if np.any(fairness_w <= 0):
            issues.append(f"fairness_weight 包含非正值: min={fairness_w.min()}")
        
        # CVaR 应该 >= risk_mean（定义上）
        non_zero_mask = (risk_mean > 0) | (risk_cvar > 0)
        if np.any(non_zero_mask):
            violation = np.sum((risk_cvar < risk_mean - 1e-6) & non_zero_mask)
            if violation > 0:
                warnings.append(f"发现 {violation} 个节点 CVaR < risk_mean")
        
        # 统计
        stats["risk_mean"] = {"min": float(risk_mean.min()), "max": float(risk_mean.max()), "mean": float(risk_mean.mean())}
        stats["risk_cvar"] = {"min": float(risk_cvar.min()), "max": float(risk_cvar.max()), "mean": float(risk_cvar.mean())}
        stats["count"] = {"min": float(count.min()), "max": float(count.max()), "mean": float(count.mean())}
        stats["fairness_weight"] = {"min": float(fairness_w.min()), "max": float(fairness_w.max()), "mean": float(fairness_w.mean())}
    
    if edge_feat.shape[0] > 0:
        delta_eta = edge_feat[:, 0]
        delta_cvar = edge_feat[:, 1]
        violation_count = edge_feat[:, 2]
        travel_time = edge_feat[:, 3]
        
        if np.any(travel_time < 0):
            issues.append(f"travel_time 包含负值: min={travel_time.min()}")
        if np.any(travel_time > 36000):
            warnings.append(f"travel_time 超过 10 小时: max={travel_time.max()}")
        
        stats["delta_eta_max"] = {"min": float(delta_eta.min()), "max": float(delta_eta.max()), "mean": float(delta_eta.mean())}
        stats["delta_cvar"] = {"min": float(delta_cvar.min()), "max": float(delta_cvar.max()), "mean": float(delta_cvar.mean())}
        stats["count_violation"] = {"min": float(violation_count.min()), "max": float(violation_count.max()), "mean": float(violation_count.mean())}
        stats["travel_time"] = {"min": float(travel_time.min()), "max": float(travel_time.max()), "mean": float(travel_time.mean())}
    
    # 检查动作掩码
    if len(action_mask) > 0:
        valid_count = np.sum(action_mask)
        stats["valid_actions"] = int(valid_count)
        stats["total_actions"] = int(len(action_mask))
        stats["valid_ratio"] = float(valid_count / len(action_mask))
        if valid_count == 0:
            issues.append("动作掩码全为无效（无可行动作）!")
    else:
        warnings.append("无候选动作")
    
    passed = len(issues) == 0
    return HealthCheckResult(passed=passed, issues=issues, warnings=warnings, stats=stats)


# ─────────────────────────────────────────────────────────────
# 可视化函数
# ─────────────────────────────────────────────────────────────
def plot_node_features(ax, node_feat: np.ndarray, feature_idx: int, label: str, panel_label: str):
    """绘制节点特征直方图。"""
    data = node_feat[:, feature_idx]
    
    ax.hist(
        data,
        bins=50,
        color=COLORS['CAAC_MLP'],
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax.set_xlabel(label, fontsize=IEEE_FONTSIZE, fontfamily=IEEE_FONT)
    ax.set_ylabel("Frequency (nodes)", fontsize=IEEE_FONTSIZE, fontfamily=IEEE_FONT)
    ax.tick_params(labelsize=IEEE_FONTSIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(IEEE_FONT)
    
    ax.text(
        0.02, 0.98, panel_label,
        transform=ax.transAxes,
        fontsize=IEEE_FONTSIZE,
        fontweight='bold',
        fontfamily=IEEE_FONT,
        verticalalignment='top',
    )
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')


def plot_edge_features(ax, edge_feat: np.ndarray, feature_idx: int, label: str, panel_label: str):
    """绘制边特征直方图。"""
    if edge_feat.shape[0] == 0:
        ax.text(0.5, 0.5, "No edge data", ha='center', va='center', transform=ax.transAxes,
                fontsize=IEEE_FONTSIZE, fontfamily=IEEE_FONT)
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=IEEE_FONTSIZE,
                fontweight='bold', fontfamily=IEEE_FONT, verticalalignment='top')
        return
    
    data = edge_feat[:, feature_idx]
    
    ax.hist(
        data,
        bins=50,
        color=COLORS['BR_AC'],
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax.set_xlabel(label, fontsize=IEEE_FONTSIZE, fontfamily=IEEE_FONT)
    ax.set_ylabel("Frequency (edges)", fontsize=IEEE_FONTSIZE, fontfamily=IEEE_FONT)
    ax.tick_params(labelsize=IEEE_FONTSIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(IEEE_FONT)
    
    ax.text(
        0.02, 0.98, panel_label,
        transform=ax.transAxes,
        fontsize=IEEE_FONTSIZE,
        fontweight='bold',
        fontfamily=IEEE_FONT,
        verticalalignment='top',
    )
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')


def plot_action_mask(ax, action_mask: np.ndarray, panel_label: str):
    """绘制动作掩码有效率饼图。"""
    if len(action_mask) == 0:
        ax.text(0.5, 0.5, "No actions", ha='center', va='center', transform=ax.transAxes,
                fontsize=IEEE_FONTSIZE, fontfamily=IEEE_FONT)
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=IEEE_FONTSIZE,
                fontweight='bold', fontfamily=IEEE_FONT, verticalalignment='top')
        return
    
    valid = np.sum(action_mask)
    invalid = len(action_mask) - valid
    
    colors_pie = [COLORS['BR_AC'], COLORS['NC']]
    labels = [f'Valid ({valid})', f'Invalid ({invalid})']
    
    ax.pie(
        [valid, invalid],
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': IEEE_FONTSIZE, 'fontfamily': IEEE_FONT},
    )
    ax.legend(labels, loc='lower right', fontsize=IEEE_FONTSIZE - 1, framealpha=0.9)
    
    ax.text(
        0.02, 0.98, panel_label,
        transform=ax.transAxes,
        fontsize=IEEE_FONTSIZE,
        fontweight='bold',
        fontfamily=IEEE_FONT,
        verticalalignment='top',
    )


def plot_health_summary(ax, health: HealthCheckResult, panel_label: str):
    """绘制健康性检查摘要。"""
    ax.axis('off')
    
    lines = []
    if health.passed:
        lines.append("✅ Health Check PASSED")
    else:
        lines.append("❌ Health Check FAILED")
    
    lines.append("")
    if health.issues:
        lines.append("Issues:")
        for issue in health.issues:
            lines.append(f"  • {issue}")
    
    if health.warnings:
        lines.append("Warnings:")
        for warn in health.warnings:
            lines.append(f"  ⚠ {warn}")
    
    if not health.issues and not health.warnings:
        lines.append("  No issues or warnings detected.")
    
    text = "\n".join(lines)
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        fontsize=IEEE_FONTSIZE - 1,
        fontfamily='Consolas',  # 等宽字体便于阅读
        verticalalignment='top',
        horizontalalignment='left',
        wrap=True,
    )
    
    ax.text(
        0.02, 0.98, panel_label,
        transform=ax.transAxes,
        fontsize=IEEE_FONTSIZE,
        fontweight='bold',
        fontfamily=IEEE_FONT,
        verticalalignment='top',
    )


def generate_visualization(features: Dict[str, np.ndarray], health: HealthCheckResult, output_path: Path):
    """生成完整的可视化图表。"""
    node_feat = features.get("node_features", np.zeros((0, 5)))
    edge_feat = features.get("edge_features", np.zeros((0, 4)))
    action_mask = features.get("action_mask", np.array([]))
    
    # 3 行 4 列布局
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(IEEE_WIDTH_INCH * 2, 7.0))
    
    # 第一行: 节点特征 (5个)
    node_labels = [
        "Risk Mean (prob)",
        "Risk CVaR (prob)",
        "Wait Count (pax)",
        "Fairness Weight",
        "Geo Embedding",
    ]
    for i in range(5):
        row, col = 0, i if i < 4 else 0
        if i < 4:
            plot_node_features(axes[0, i], node_feat, i, node_labels[i], f"({chr(ord('a') + i)})")
        else:
            plot_node_features(axes[1, 0], node_feat, i, node_labels[i], f"({chr(ord('a') + i)})")
    
    # 第二行: 边特征 (4个)
    edge_labels = [
        "Delta ETA Max (s)",
        "Delta CVaR",
        "Violation Count (pax)",
        "Travel Time (s)",
    ]
    for i in range(4):
        if i == 0:
            continue  # 第一格已被 geo_embedding 占用
        plot_edge_features(axes[1, i], edge_feat, i - 1 if i > 0 else 0, edge_labels[i - 1 if i > 0 else 0], f"({chr(ord('f') + i - 1)})")
    
    # 修正布局：边特征从 (1,1) 开始
    plot_edge_features(axes[1, 1], edge_feat, 0, edge_labels[0], "(f)")
    plot_edge_features(axes[1, 2], edge_feat, 1, edge_labels[1], "(g)")
    plot_edge_features(axes[1, 3], edge_feat, 2, edge_labels[2], "(h)")
    plot_edge_features(axes[2, 0], edge_feat, 3, edge_labels[3], "(i)")
    
    # 第三行: 动作掩码 + 健康性摘要
    plot_action_mask(axes[2, 1], action_mask, "(j)")
    
    # 合并最后两格用于健康性摘要
    axes[2, 2].axis('off')
    axes[2, 3].axis('off')
    # 创建跨格的子图区域
    gs = axes[2, 2].get_gridspec()
    ax_health = fig.add_subplot(gs[2, 2:])
    plot_health_summary(ax_health, health, "(k)")
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=IEEE_DPI, bbox_inches='tight')
    logging.info(f"可视化图表已保存至: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────
def collect_features(config: Dict, stage: str, num_samples: int = 5) -> List[Dict[str, np.ndarray]]:
    """从环境收集模型输入特征。"""
    env_cfg_dict = config.get("env", {})
    
    # 应用课程阶段参数
    curriculum_cfg = config.get("curriculum", {})
    stage_params = curriculum_cfg.get("stage_params", {}).get(stage, {})
    
    # 覆盖 churn_tol 等参数
    if "churn_tol_override_sec" in stage_params:
        env_cfg_dict["churn_tol_sec"] = stage_params["churn_tol_override_sec"]
        env_cfg_dict["waiting_churn_tol_sec"] = stage_params["churn_tol_override_sec"]
        env_cfg_dict["onboard_churn_tol_sec"] = stage_params["churn_tol_override_sec"]
    
    env_config = EnvConfig(**env_cfg_dict)
    env = EventDrivenEnv(env_config)
    
    samples = []
    for i in range(num_samples):
        env.reset()
        features = env.get_feature_batch()
        samples.append(features)
        logging.info(f"收集样本 {i+1}/{num_samples}: node_features={features['node_features'].shape}, "
                     f"edge_features={features['edge_features'].shape}, "
                     f"valid_actions={np.sum(features['action_mask'])}/{len(features['action_mask'])}")
    
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化模型输入数据")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--stage", default="L3", help="课程阶段 (L0/L1/L2/L3)")
    parser.add_argument("--init-model-path", default=None, help="初始模型路径（未使用，仅兼容命令行）")
    parser.add_argument("--output", default="reports/debug/model_input_viz.png", help="输出图片路径")
    parser.add_argument("--stats-output", default="reports/debug/model_input_stats.json", help="统计信息输出路径")
    parser.add_argument("--num-samples", type=int, default=5, help="收集样本数")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    config = load_config(args.config)
    
    logging.info(f"使用配置: {args.config}")
    logging.info(f"课程阶段: {args.stage}")
    logging.info(f"收集 {args.num_samples} 个样本...")
    
    samples = collect_features(config, args.stage, args.num_samples)
    
    # 合并所有样本（用于整体分析）
    all_node_feat = np.concatenate([s["node_features"] for s in samples], axis=0)
    all_edge_feat = np.concatenate([s["edge_features"] for s in samples if s["edge_features"].shape[0] > 0], axis=0) if any(s["edge_features"].shape[0] > 0 for s in samples) else np.zeros((0, 4))
    all_action_mask = np.concatenate([s["action_mask"] for s in samples], axis=0)
    
    merged_features = {
        "node_features": all_node_feat,
        "edge_features": all_edge_feat,
        "action_mask": all_action_mask,
    }
    
    # 健康性检查
    health = check_health(merged_features)
    
    # 打印详细统计
    logging.info("=" * 60)
    logging.info("模型输入数据统计摘要")
    logging.info("=" * 60)
    logging.info(f"node_features shape: {all_node_feat.shape}")
    logging.info(f"edge_features shape: {all_edge_feat.shape}")
    logging.info(f"action_mask length: {len(all_action_mask)}")
    
    for key, val in health.stats.items():
        if isinstance(val, dict):
            logging.info(f"  {key}: min={val['min']:.4f}, max={val['max']:.4f}, mean={val['mean']:.4f}")
        else:
            logging.info(f"  {key}: {val}")
    
    if health.passed:
        logging.info("✅ 健康性检查通过")
    else:
        logging.warning("❌ 健康性检查未通过!")
        for issue in health.issues:
            logging.error(f"  Issue: {issue}")
    
    for warn in health.warnings:
        logging.warning(f"  Warning: {warn}")
    
    logging.info("=" * 60)
    
    # 生成可视化
    output_path = Path(args.output)
    generate_visualization(merged_features, health, output_path)
    
    # 保存统计信息
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_output = {
        "stage": args.stage,
        "num_samples": args.num_samples,
        "health_passed": health.passed,
        "issues": health.issues,
        "warnings": health.warnings,
        "stats": health.stats,
    }
    stats_path.write_text(json.dumps(stats_output, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"统计信息已保存至: {stats_path}")


if __name__ == "__main__":
    main()
