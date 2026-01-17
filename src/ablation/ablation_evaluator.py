"""消融实验评估器

提供独立于 baseline 评估的消融实验评估逻辑。
复用底层指标计算模块确保口径一致。

主表指标 (4个)：
- TACC: 总避免私家车通勤时间
- churn_rate: 系统总流失率
- wait_time_p95: 95百分位等待时间
- suburban_service_rate: 郊区服务率

诊断指标 (3个)：
- waiting_churn_count / onboard_churn_count: churn 分解
- service_gini: 服务公平性
- stuckness_ratio: 课程消融专用
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging

import numpy as np
import torch
from torch import nn

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.utils.fairness import gini_coefficient, compute_service_volume_gini

LOG = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """消融实验结果
    
    包含主表4指标 + 诊断3指标。
    """
    ablation_type: str
    
    # 主表4指标
    tacc: float                    # 总避免私家车通勤时间 ↑
    churn_rate: float              # 系统总流失率 ↓
    wait_time_p95: float           # 95百分位等待时间 ↓
    suburban_service_rate: float   # 郊区服务率 ↑
    
    # 诊断3指标
    waiting_churn_count: int       # 等待阶段流失数
    onboard_churn_count: int       # 车上流失数
    service_gini: float            # 服务公平性 Gini
    
    # 基础指标
    service_rate: float            # 接单率
    served_count: int              # 服务数	
    total_requests: int            # 总请求数
    episodes: int                  # 评估 episode 数
    
    # 课程消融专用（可选）
    stuckness_ratio: Optional[float] = None
    convergence_steps: Optional[int] = None
    
    # 元数据
    seed: int = 7
    model_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "ablation_type": self.ablation_type,
            "main_metrics": {
                "tacc": self.tacc,
                "churn_rate": self.churn_rate,
                "wait_time_p95": self.wait_time_p95,
                "suburban_service_rate": self.suburban_service_rate,
            },
            "diagnostic_metrics": {
                "waiting_churn_count": self.waiting_churn_count,
                "onboard_churn_count": self.onboard_churn_count,
                "service_gini": self.service_gini,
                "stuckness_ratio": self.stuckness_ratio,
                "convergence_steps": self.convergence_steps,
            },
            "basic_metrics": {
                "service_rate": self.service_rate,
                "served_count": self.served_count,
                "total_requests": self.total_requests,
                "episodes": self.episodes,
            },
            "metadata": {
                "seed": self.seed,
                "model_path": self.model_path,
            },
        }


def compute_wait_time_percentile(env: EventDrivenEnv, percentile: float = 95.0) -> float:
    """计算等待时间百分位数
    
    复用环境的请求数据，确保与 baseline 评估口径一致。
    """
    waits: List[float] = []
    for req in env.requests:
        pickup_time = req.get("pickup_time_sec")
        if pickup_time is not None:
            wait = float(pickup_time) - float(req["request_time_sec"])
            waits.append(wait)
    if not waits:
        return float("nan")
    return float(np.percentile(waits, percentile))


def compute_suburban_service_rate(
    env: EventDrivenEnv,
    suburban_threshold: float = 1.2,
) -> float:
    """计算郊区服务率
    
    基于 fairness_weight >= suburban_threshold 定义郊区站点。
    注意：对于 RiskAblatedEnv，fairness_weight 全为 1.0，
    需使用原始配置的 fairness_gamma 来识别郊区。
    
    Args:
        env: 环境实例
        suburban_threshold: 郊区站点的 fairness_weight 阈值
    """
    # 获取原始 fairness_weight（基于地理位置）
    # 如果被消融了，需要重新计算
    if hasattr(env, '_original_fairness_weight'):
        weights = env._original_fairness_weight
    else:
        weights = env.fairness_weight
    
    suburban_stops = set(
        s for s, w in weights.items() 
        if w >= suburban_threshold
    )
    
    if not suburban_stops:
        return float("nan")
    
    # 统计郊区站点的服务数
    suburban_served = sum(
        env.service_count_by_stop.get(s, 0) 
        for s in suburban_stops
    )
    
    # 统计来自郊区站点的请求数
    total_suburban_requests = sum(
        1 for req in env.requests 
        if int(req["pickup_stop_id"]) in suburban_stops
        and req.get("status") != "structurally_unserviceable"
    )
    
    return float(suburban_served) / max(1, total_suburban_requests)


def compute_stuckness_ratio(mask_history: List[List[bool]]) -> float:
    """计算 stuckness ratio（被 mask 的动作比例）
    
    用于 w/o Curriculum 消融的诊断。
    """
    if not mask_history:
        return 0.0
    total_actions = sum(len(m) for m in mask_history)
    masked_actions = sum(sum(1 for v in m if not v) for m in mask_history)
    return float(masked_actions) / max(1, total_actions)


class AblationEvaluator:
    """消融实验评估器（复用底层指标计算）
    
    支持三种消融类型：
    - no_edge: 使用 NodeOnlyGNN
    - no_risk: 使用 RiskAblatedEnv
    - no_curriculum: 使用标准环境，评估训练后的模型
    """
    
    def __init__(
        self,
        ablation_type: str,
        env_config: EnvConfig,
        model: nn.Module,
        device: torch.device,
        seed: int = 7,
    ) -> None:
        self.ablation_type = ablation_type
        self.env_config = env_config
        self.model = model
        self.device = device
        self.seed = seed
        
        # 根据消融类型创建环境
        if ablation_type == "no_risk":
            from src.ablation.ablation_env_wrapper import RiskAblatedEnv
            self.env = RiskAblatedEnv(env_config)
        else:
            self.env = EventDrivenEnv(env_config)
        
        # 保存原始 fairness_weight 用于郊区计算
        if ablation_type == "no_risk":
            # 从原始配置重建
            temp_env = EventDrivenEnv(env_config)
            self.env._original_fairness_weight = temp_env.fairness_weight.copy()
            del temp_env
    
    def evaluate(
        self,
        episodes: int = 10,
        max_steps: Optional[int] = None,
        suburban_threshold: float = 1.2,
    ) -> AblationResult:
        """运行评估并返回结果
        
        Args:
            episodes: 评估 episode 数
            max_steps: 每个 episode 最大步数
            suburban_threshold: 郊区站点的 fairness_weight 阈值
        """
        self.model.eval()
        
        # 累计指标
        total_tacc = 0.0
        total_served = 0
        total_requests = 0
        total_waiting_churned = 0
        total_onboard_churned = 0
        all_wait_times: List[float] = []
        all_suburban_rates: List[float] = []
        all_ginis: List[float] = []
        mask_history: List[List[bool]] = []
        
        for ep in range(episodes):
            ep_tacc = 0.0
            obs = self.env.reset()
            done = False
            steps = 0
            
            while not done:
                if max_steps and steps >= max_steps:
                    break
                
                features = self.env.get_feature_batch()
                action_mask = features["action_mask"]
                actions = features["actions"]
                
                # 记录 mask 用于 stuckness 计算
                mask_history.append(action_mask.tolist())
                
                if len(actions) == 0 or not any(action_mask):
                    break
                
                # 使用模型选择动作
                action = self._select_action(features)
                if action is None:
                    break
                
                obs, reward, done, info = self.env.step(action)
                ep_tacc += info.get("step_tacc_gain", 0.0)
                steps += 1
            
            # 收集 episode 指标
            total_tacc += ep_tacc
            total_served += self.env.served
            total_requests += len([r for r in self.env.requests if r.get("status") != "structurally_unserviceable"])
            total_waiting_churned += self.env.waiting_churned
            total_onboard_churned += self.env.onboard_churned
            
            # 等待时间
            for req in self.env.requests:
                pickup_time = req.get("pickup_time_sec")
                if pickup_time is not None:
                    wait = float(pickup_time) - float(req["request_time_sec"])
                    all_wait_times.append(wait)
            
            # 郊区服务率
            suburban_rate = compute_suburban_service_rate(self.env, suburban_threshold)
            if np.isfinite(suburban_rate):
                all_suburban_rates.append(suburban_rate)
            
            # Gini
            gini = compute_service_volume_gini(
                self.env.service_count_by_stop, 
                self.env.stop_ids
            )
            all_ginis.append(gini)
        
        # 计算汇总指标
        total_churned = total_waiting_churned + total_onboard_churned
        churn_rate = float(total_churned) / max(1, total_requests)
        service_rate = float(total_served) / max(1, total_requests)
        
        wait_p95 = float(np.percentile(all_wait_times, 95)) if all_wait_times else float("nan")
        avg_suburban_rate = float(np.mean(all_suburban_rates)) if all_suburban_rates else float("nan")
        avg_gini = float(np.mean(all_ginis)) if all_ginis else float("nan")
        
        stuckness = compute_stuckness_ratio(mask_history) if self.ablation_type == "no_curriculum" else None
        
        return AblationResult(
            ablation_type=self.ablation_type,
            tacc=total_tacc,
            churn_rate=churn_rate,
            wait_time_p95=wait_p95,
            suburban_service_rate=avg_suburban_rate,
            waiting_churn_count=total_waiting_churned,
            onboard_churn_count=total_onboard_churned,
            service_gini=avg_gini,
            service_rate=service_rate,
            served_count=total_served,
            total_requests=total_requests,
            episodes=episodes,
            stuckness_ratio=stuckness,
            seed=self.seed,
        )
    
    def _select_action(self, features: Dict[str, np.ndarray]) -> Optional[int]:
        """使用模型选择动作"""
        action_mask = features["action_mask"]
        actions = features["actions"]
        
        if len(actions) == 0 or not any(action_mask):
            return None
        
        # 准备模型输入
        data = {
            "node_features": torch.from_numpy(features["node_features"]).float().to(self.device),
            "graph_edge_index": torch.from_numpy(features["graph_edge_index"]).long().to(self.device),
            "graph_edge_features": torch.from_numpy(features["graph_edge_features"]).float().to(self.device),
            "edge_features": torch.from_numpy(features["edge_features"]).float().to(self.device),
            "action_edge_index": torch.stack([
                torch.full((len(actions),), features["current_node_index"][0], dtype=torch.long),
                torch.from_numpy(features["action_node_indices"]).long(),
            ]).to(self.device),
        }
        
        with torch.no_grad():
            q_values = self.model(data)
        
        q_np = q_values.cpu().numpy()
        
        # 应用 mask
        masked_q = np.where(action_mask, q_np, -np.inf)
        best_idx = int(np.argmax(masked_q))
        
        if not action_mask[best_idx]:
            # 如果最佳动作被 mask，选择第一个可用动作
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) == 0:
                return None
            best_idx = valid_indices[0]
        
        return int(actions[best_idx])


def generate_comparison_table(results: List[AblationResult]) -> str:
    """生成论文用对比表格（Markdown 格式）
    
    按照主表4指标布局。
    """
    lines = [
        "| Method | TACC ↑ | Churn Rate ↓ | Wait P95 ↓ | Suburban Rate ↑ |",
        "|--------|--------|--------------|------------|-----------------|",
    ]
    
    for r in results:
        name = {
            "full": "Mobi-Churn (Full)",
            "no_edge": "w/o Edge-Encoding",
            "no_risk": "w/o Risk-Awareness",
            "no_curriculum": "w/o Curriculum",
        }.get(r.ablation_type, r.ablation_type)
        
        lines.append(
            f"| {name} | {r.tacc:.1f} | {r.churn_rate:.3f} | {r.wait_time_p95:.1f}s | {r.suburban_service_rate:.3f} |"
        )
    
    return "\n".join(lines)


def save_results(results: List[AblationResult], output_dir: Path) -> None:
    """保存评估结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 格式
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
    
    # Markdown 表格
    table_path = output_dir / "ablation_comparison.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# 消融实验结果对比\n\n")
        f.write(generate_comparison_table(results))
        f.write("\n\n## 诊断指标\n\n")
        for r in results:
            f.write(f"### {r.ablation_type}\n")
            f.write(f"- Waiting Churn: {r.waiting_churn_count}\n")
            f.write(f"- Onboard Churn: {r.onboard_churn_count}\n")
            f.write(f"- Service Gini: {r.service_gini:.4f}\n")
            if r.stuckness_ratio is not None:
                f.write(f"- Stuckness Ratio: {r.stuckness_ratio:.4f}\n")
            f.write("\n")
    
    LOG.info(f"Results saved to {output_dir}")
