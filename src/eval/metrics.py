"""Evaluation metrics module.

提供消融实验和 baseline 评估共用的指标计算函数，确保口径一致。
"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.env.gym_env import EventDrivenEnv


def compute_tacc(env: "EventDrivenEnv") -> float:
    """Compute Total Avoided Private Car Travel Time (TACC).
    
    统计所有已服务乘客的直达行程时间总和。
    """
    tacc = 0.0
    for req in env.requests:
        if req.get("status") == "served":
            tacc += req.get("direct_time_sec", 0.0)
    return float(tacc)


def compute_churn_rate(env: "EventDrivenEnv") -> float:
    """计算系统总流失率。
    
    流失率 = (等待流失 + 车上流失) / (总请求 - 结构性不可服务)
    """
    total_churned = env.waiting_churned + env.onboard_churned
    serviceable_requests = len([
        r for r in env.requests 
        if r.get("status") != "structurally_unserviceable"
    ])
    return float(total_churned) / max(1, serviceable_requests)


def compute_service_rate(env: "EventDrivenEnv") -> float:
    """计算服务率。
    
    服务率 = 已服务 / (总请求 - 结构性不可服务)
    """
    serviceable_requests = len([
        r for r in env.requests 
        if r.get("status") != "structurally_unserviceable"
    ])
    return float(env.served) / max(1, serviceable_requests)


def compute_wait_times(env: "EventDrivenEnv") -> List[float]:
    """计算所有已服务乘客的等待时间列表。"""
    waits: List[float] = []
    for req in env.requests:
        pickup_time = req.get("pickup_time_sec")
        if pickup_time is not None:
            wait = float(pickup_time) - float(req["request_time_sec"])
            waits.append(wait)
    return waits


def compute_wait_time_percentile(env: "EventDrivenEnv", percentile: float = 95.0) -> float:
    """计算等待时间百分位数。"""
    waits = compute_wait_times(env)
    if not waits:
        return float("nan")
    return float(np.percentile(waits, percentile))


def compute_stuckness_ratio(mask_history: List[List[bool]]) -> float:
    """计算 stuckness ratio（被 mask 的动作比例）。
    
    用于 w/o Curriculum 消融的诊断指标。
    """
    if not mask_history:
        return 0.0
    total_actions = sum(len(m) for m in mask_history)
    masked_actions = sum(sum(1 for v in m if not v) for m in mask_history)
    return float(masked_actions) / max(1, total_actions)

