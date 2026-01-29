"""Reward hacking and invariant alert heuristics for realtime monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import json


@dataclass(frozen=True)
class RewardHackAlertConfig:
    enabled: bool = True
    reward_window: int = 50
    reward_positive_threshold: float = 0.5
    reward_delta_threshold: float = 0.2
    low_service_steps: int = 20
    low_service_threshold: float = 0.0
    loop_window: int = 30
    loop_unique_stops_max: int = 2
    loop_max_served: int = 1
    entropy_floor: float = 0.25
    entropy_patience: int = 20
    epsilon_floor: float = 0.2
    entropy_warning_floor: float = 0.1
    entropy_warning_patience: int = 50
    alert_buffer_size: int = 50
    debug_abort_on_alert: bool = True
    abort_levels: List[str] = field(default_factory=lambda: ["high"])
    abort_codes: List[str] = field(default_factory=lambda: [
        "active_stop_mismatch",
        "time_feature_mismatch",
        "action_collapse_shortcut",
        "overload_deadlock",
    ])
    debug_dump_dir: str = "reports/debug/potential_alerts"
    reward_total_eps: float = 1e-6
    phi_eps: float = 1e-9
    no_service_positive_reward_epsilon: float = 0.1
    # === 可行动作塌缩检测 ===
    action_collapse_threshold: int = 1  # 可行动作数阈值（≤该值视为塌缩）
    action_collapse_patience_medium: int = 20  # Medium级触发步数
    action_collapse_patience_high: int = 50  # High级触发步数
    # === 短路条件 ===
    entropy_collapse_epsilon_high: float = 0.3  # 短路条件epsilon阈值（高于此仍高探索）
    entropy_near_zero_threshold: float = 0.01  # 熵接近零的阈值
    # === action_valid_ratio 突变检测 ===
    action_ratio_drop_threshold: float = 0.5  # 相对下降阈值
    action_ratio_absolute_floor: float = 0.1  # 绝对危险线
    action_ratio_window: int = 50  # 滑窗大小
    action_ratio_min_valid_count: int = 2  # 触发最小可行动作数
    # === 决策效率监控 ===
    efficiency_decay_window: int = 100  # 效率趋势窗口
    efficiency_decay_threshold: float = 0.3  # served_per_decision 下降阈值
    # === 过载锁死检测 ===
    overload_backlog_threshold: float = 100.0  # 过载backlog阈值
    overload_ready_vehicles_threshold: int = 2  # 过载ready_vehicles阈值
    overload_event_queue_threshold: int = 500  # 过载event_queue阈值
    overload_patience: int = 30  # 过载持续步数
    # === 潜势塑形失效检测 ===
    shaping_inactive_patience: int = 500  # 塑形失效检测窗口
    shaping_inactive_alpha_threshold: float = 1e-3
    shaping_inactive_eps: float = 1e-6
    shaping_inactive_min_nonzero_ratio: float = 0.001
    # === self-loop 单点可行动作检测 ===
    self_loop_singleton_patience: int = 300
    check_self_loop_singleton: bool = True
    # === 一致性检测 ===
    check_active_stop_mismatch: bool = True
    check_time_feature_mismatch: bool = True
    check_payload_schema_violation: bool = True
    time_feature_mismatch_eps: float = 1e-6

    @staticmethod
    def from_dict(raw: Optional[Dict[str, Any]]) -> "RewardHackAlertConfig":
        if not raw:
            return RewardHackAlertConfig()
        return RewardHackAlertConfig(
            enabled=bool(raw.get("enabled", True)),
            reward_window=int(raw.get("reward_window", 50)),
            reward_positive_threshold=float(raw.get("reward_positive_threshold", 0.5)),
            reward_delta_threshold=float(raw.get("reward_delta_threshold", 0.2)),
            low_service_steps=int(raw.get("low_service_steps", 20)),
            low_service_threshold=float(raw.get("low_service_threshold", 0.0)),
            loop_window=int(raw.get("loop_window", 30)),
            loop_unique_stops_max=int(raw.get("loop_unique_stops_max", 2)),
            loop_max_served=int(raw.get("loop_max_served", 1)),
            entropy_floor=float(raw.get("entropy_floor", 0.25)),
            entropy_patience=int(raw.get("entropy_patience", 20)),
            epsilon_floor=float(raw.get("epsilon_floor", 0.2)),
            entropy_warning_floor=float(raw.get("entropy_warning_floor", 0.1)),
            entropy_warning_patience=int(raw.get("entropy_warning_patience", 50)),
            alert_buffer_size=int(raw.get("alert_buffer_size", 50)),
            debug_abort_on_alert=bool(raw.get("debug_abort_on_alert", True)),
            abort_levels=list(raw.get("abort_levels", ["high"])),
            abort_codes=list(raw.get(
                "abort_codes",
                [
                    "active_stop_mismatch",
                    "time_feature_mismatch",
                    "action_collapse_shortcut",
                    "overload_deadlock",
                ],
            )),
            debug_dump_dir=str(raw.get("debug_dump_dir", "reports/debug/potential_alerts")),
            reward_total_eps=float(raw.get("reward_total_eps", 1e-6)),
            phi_eps=float(raw.get("phi_eps", 1e-9)),
            no_service_positive_reward_epsilon=float(raw.get("no_service_positive_reward_epsilon", 0.1)),
            # 可行动作塌缩检测
            action_collapse_threshold=int(raw.get("action_collapse_threshold", 1)),
            action_collapse_patience_medium=int(raw.get("action_collapse_patience_medium", 20)),
            action_collapse_patience_high=int(raw.get("action_collapse_patience_high", 50)),
            # 短路条件
            entropy_collapse_epsilon_high=float(raw.get("entropy_collapse_epsilon_high", 0.3)),
            entropy_near_zero_threshold=float(raw.get("entropy_near_zero_threshold", 0.01)),
            # action_valid_ratio 突变检测
            action_ratio_drop_threshold=float(raw.get("action_ratio_drop_threshold", 0.5)),
            action_ratio_absolute_floor=float(raw.get("action_ratio_absolute_floor", 0.1)),
            action_ratio_window=int(raw.get("action_ratio_window", 50)),
            action_ratio_min_valid_count=int(raw.get("action_ratio_min_valid_count", 2)),
            # 决策效率监控
            efficiency_decay_window=int(raw.get("efficiency_decay_window", 100)),
            efficiency_decay_threshold=float(raw.get("efficiency_decay_threshold", 0.3)),
            # 过载锁死检测
            overload_backlog_threshold=float(raw.get("overload_backlog_threshold", 100.0)),
            overload_ready_vehicles_threshold=int(raw.get("overload_ready_vehicles_threshold", 2)),
            overload_event_queue_threshold=int(raw.get("overload_event_queue_threshold", 500)),
            overload_patience=int(raw.get("overload_patience", 30)),
            # 潜势塑形失效检测
            shaping_inactive_patience=int(raw.get("shaping_inactive_patience", 500)),
            shaping_inactive_alpha_threshold=float(raw.get("shaping_inactive_alpha_threshold", 1e-3)),
            shaping_inactive_eps=float(raw.get("shaping_inactive_eps", 1e-6)),
            shaping_inactive_min_nonzero_ratio=float(raw.get("shaping_inactive_min_nonzero_ratio", 0.001)),
            self_loop_singleton_patience=int(raw.get("self_loop_singleton_patience", 300)),
            check_self_loop_singleton=bool(raw.get("check_self_loop_singleton", True)),
            # 一致性检测
            check_active_stop_mismatch=bool(raw.get("check_active_stop_mismatch", True)),
            check_time_feature_mismatch=bool(raw.get("check_time_feature_mismatch", True)),
            check_payload_schema_violation=bool(raw.get("check_payload_schema_violation", True)),
            time_feature_mismatch_eps=float(raw.get("time_feature_mismatch_eps", 1e-6)),
        )


@dataclass
class RewardHackDetector:
    config: RewardHackAlertConfig
    reward_history: deque = field(init=False)
    served_history: deque = field(init=False)
    action_history: deque = field(init=False)
    entropy_history: deque = field(init=False)
    epsilon_history: deque = field(init=False)
    alert_history: deque = field(init=False)
    _no_service_streak: int = field(default=0, init=False)
    _last_action_mask_valid_count: Optional[int] = field(default=None, init=False)
    # === 新增字段 ===
    action_valid_count_history: deque = field(init=False)  # 可行动作数历史
    action_valid_ratio_history: deque = field(init=False)  # action_valid_ratio 历史
    action_count_history: deque = field(init=False)  # action_count 历史
    served_per_decision_history: deque = field(init=False)  # 单位决策服务数历史
    churn_per_decision_history: deque = field(init=False)  # 单位决策流失数历史
    backlog_history: deque = field(init=False)  # backlog 历史
    ready_vehicles_history: deque = field(init=False)  # ready_vehicles 历史
    event_queue_history: deque = field(init=False)  # event_queue_len 历史
    shaping_history: deque = field(init=False)  # 塑形奖励历史
    alpha_history: deque = field(init=False)  # alpha 历史
    phi_delta_history: deque = field(init=False)  # phi_delta 历史
    event_count_history: deque = field(init=False)  # step 事件计数历史
    backlog_delta_history: deque = field(init=False)  # backlog 变化历史
    _action_collapse_streak: int = field(default=0, init=False)  # 连续塌缩计数器
    _overload_streak: int = field(default=0, init=False)  # 过载计数器
    _last_entropy_norm: float = field(default=0.0, init=False)  # 最近熵值
    _last_epsilon: float = field(default=0.0, init=False)  # 最近 epsilon
    _last_active_stop_mismatch: bool = field(default=False, init=False)  # 最近对齐异常
    _last_time_feature_mismatch: bool = field(default=False, init=False)  # 最近时间基准异常
    _self_loop_singleton_streak: int = field(default=0, init=False)  # self-loop 单点可行计数器

    def __post_init__(self) -> None:
        max_len = max(
            2 * self.config.reward_window,
            self.config.loop_window,
            self.config.entropy_patience,
            self.config.low_service_steps,
            self.config.action_collapse_patience_high,
            self.config.action_ratio_window * 2,
            self.config.efficiency_decay_window,
            self.config.overload_patience,
            self.config.shaping_inactive_patience,
            self.config.self_loop_singleton_patience,
        )
        self.reward_history = deque(maxlen=max_len)
        self.served_history = deque(maxlen=max_len)
        self.action_history = deque(maxlen=max_len)
        self.entropy_history = deque(maxlen=max_len)
        self.epsilon_history = deque(maxlen=max_len)
        self.alert_history = deque(maxlen=int(self.config.alert_buffer_size))
        # 新增历史队列
        self.action_valid_count_history = deque(maxlen=max_len)
        self.action_valid_ratio_history = deque(maxlen=max_len)
        self.action_count_history = deque(maxlen=max_len)
        self.served_per_decision_history = deque(maxlen=max_len)
        self.churn_per_decision_history = deque(maxlen=max_len)
        self.backlog_history = deque(maxlen=max_len)
        self.ready_vehicles_history = deque(maxlen=max_len)
        self.event_queue_history = deque(maxlen=max_len)
        self.shaping_history = deque(maxlen=max_len)
        self.alpha_history = deque(maxlen=max_len)
        self.phi_delta_history = deque(maxlen=max_len)
        self.event_count_history = deque(maxlen=max_len)
        self.backlog_delta_history = deque(maxlen=max_len)

    def update(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.config.enabled:
            return []
        reward_total = _safe_float(payload.get("reward_total", 0.0))
        step_served = _safe_float(payload.get("step_served", 0.0))
        action_stop = payload.get("action_stop")
        entropy_norm = payload.get("q_entropy_norm")
        epsilon = _safe_float(payload.get("epsilon", 0.0))
        shaping_raw = _safe_float(payload.get("reward_terms", {}).get("reward_potential_shaping_raw", 0.0))
        alpha = _safe_float(payload.get("reward_potential_alpha", 0.0))
        phi_delta = _safe_float(payload.get("phi_delta", 0.0))
        missing_keys = payload.get("missing_keys", [])
        action_mask_valid_count = payload.get("action_mask_valid_count")
        # 新增数据提取
        action_valid_ratio = _safe_float(payload.get("action_valid_ratio", 0.0))
        invalid_action_ratio = _safe_float(payload.get("invalid_action_ratio", 0.0))
        action_count = int(payload.get("action_count", 0))
        served_per_decision = _safe_float(payload.get("served_per_decision", 0.0))
        churn_per_decision = _safe_float(payload.get("waiting_churn_per_decision", 0.0))
        waiting_remaining = _safe_float(payload.get("waiting_remaining", 0.0))
        onboard_remaining = _safe_float(payload.get("onboard_remaining", 0.0))
        backlog = waiting_remaining + onboard_remaining
        ready_vehicles = int(payload.get("ready_vehicles", 0))
        event_queue_len = int(payload.get("event_queue_len", 0))
        phi_delta = _safe_float(payload.get("phi_delta", 0.0))
        step_served = _safe_float(payload.get("step_served", 0.0))
        step_waiting_churned = _safe_float(payload.get("step_waiting_churned", 0.0))
        step_onboard_churned = _safe_float(payload.get("step_onboard_churned", 0.0))
        step_waiting_timeouts = _safe_float(payload.get("step_waiting_timeouts", 0.0))
        event_count = step_served + step_waiting_churned + step_onboard_churned + step_waiting_timeouts
        waiting_before = _safe_float(payload.get("waiting_remaining_before", waiting_remaining))
        waiting_after = _safe_float(payload.get("waiting_remaining_after", waiting_remaining))
        onboard_before = _safe_float(payload.get("onboard_remaining_before", onboard_remaining))
        onboard_after = _safe_float(payload.get("onboard_remaining_after", onboard_remaining))
        backlog_delta = (waiting_after + onboard_after) - (waiting_before + onboard_before)

        # 更新历史记录
        self.reward_history.append(reward_total)
        self.served_history.append(step_served)
        self.epsilon_history.append(float(epsilon))
        self.action_valid_ratio_history.append(action_valid_ratio)
        self.action_count_history.append(action_count)
        self.served_per_decision_history.append(served_per_decision)
        self.churn_per_decision_history.append(churn_per_decision)
        self.backlog_history.append(backlog)
        self.ready_vehicles_history.append(ready_vehicles)
        self.event_queue_history.append(event_queue_len)
        self.shaping_history.append(shaping_raw)
        self.alpha_history.append(alpha)
        self.phi_delta_history.append(phi_delta)
        self.event_count_history.append(event_count)
        self.backlog_delta_history.append(backlog_delta)

        if action_mask_valid_count is not None:
            try:
                valid_count = int(action_mask_valid_count)
                self._last_action_mask_valid_count = valid_count
                self.action_valid_count_history.append(valid_count)
            except (TypeError, ValueError):
                self._last_action_mask_valid_count = None
        if action_stop is not None:
            self.action_history.append(int(action_stop))
        if entropy_norm is not None:
            self.entropy_history.append(float(entropy_norm))
            self._last_entropy_norm = float(entropy_norm)
        self._last_epsilon = float(epsilon)

        alerts: List[Dict[str, Any]] = []
        if missing_keys:
            alerts.extend(self._check_missing_keys(missing_keys))
        mismatch_alerts, mismatch_flag = self._check_active_stop_mismatch(payload)
        alerts.extend(mismatch_alerts)
        time_alerts, time_flag = self._check_time_feature_mismatch(payload)
        alerts.extend(time_alerts)
        self._last_active_stop_mismatch = bool(mismatch_flag or time_flag)
        self._last_time_feature_mismatch = bool(time_flag)
        alerts.extend(self._check_potential_shaping_consistency(alpha, shaping_raw, phi_delta))
        alerts.extend(self._check_phi_backlog_consistency(payload))
        alerts.extend(self._check_reward_total_consistency(payload))
        alerts.extend(self._check_served_per_decision_consistency(payload))
        alerts.extend(self._check_no_service_positive_reward(reward_total, step_served))
        alerts.extend(self._check_looping())
        alerts.extend(self._check_reward_up_no_service())
        alerts.extend(self._check_entropy_warning())
        alerts.extend(self._check_entropy_collapse())
        # === 新增检测 ===
        alerts.extend(self._check_action_space_collapse())
        alerts.extend(self._check_action_collapse_shortcut())
        alerts.extend(self._check_action_ratio_sudden_drop())
        alerts.extend(self._check_entropy_action_consistency())
        alerts.extend(self._check_valid_invalid_ratio_consistency(action_valid_ratio, invalid_action_ratio))
        alerts.extend(self._check_entropy_collapse_epsilon_high())
        alerts.extend(self._check_decision_efficiency_decay())
        alerts.extend(self._check_overload_deadlock())
        alerts.extend(self._check_singleton_selfloop_persist(payload))
        alerts.extend(self._check_potential_shaping_inactive())

        if alerts:
            dump_paths = self._dump_alerts(payload, alerts)
            for alert, path in zip(alerts, dump_paths):
                alert["dump_path"] = path
            for alert in alerts:
                self.alert_history.append(alert)
            if self.config.debug_abort_on_alert and self._should_abort(alerts):
                raise RuntimeError(f"Alerts raised: {alerts}")
        if self.alert_history:
            return list(self.alert_history)
        return []

    def _check_no_service_positive_reward(
        self,
        reward_total: float,
        step_served: float,
    ) -> List[Dict[str, Any]]:
        if step_served <= 0:
            self._no_service_streak += 1
        else:
            self._no_service_streak = 0

        threshold = float(self.config.no_service_positive_reward_epsilon)
        if self._no_service_streak >= self.config.low_service_steps and reward_total > threshold:
            return [
                {
                    "code": "no_service_positive_reward",
                    "severity": "high",
                    "message": "连续无服务但奖励为正，疑似在刷非服务项奖励。",
                    "stats": {
                        "no_service_steps": self._no_service_streak,
                        "reward_total": reward_total,
                        "threshold": threshold,
                    },
                }
            ]
        return []

    def _extract_active_vehicle_record(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        alerts: List[Dict[str, Any]] = []
        if not self.config.check_active_stop_mismatch and not self.config.check_time_feature_mismatch:
            return None, alerts
        active_vehicle_id = payload.get("active_vehicle_id")
        vehicles_by_id = payload.get("vehicles_by_id")
        vehicles = payload.get("vehicles")
        if active_vehicle_id is None:
            if self.config.check_payload_schema_violation:
                alerts.append({
                    "code": "payload_schema_violation",
                    "severity": "high",
                    "message": "payload 缺少 active_vehicle_id，无法进行对齐校验。",
                    "stats": {"missing": ["active_vehicle_id"]},
                })
            return None, alerts
        if vehicles_by_id is None:
            if isinstance(vehicles, list):
                vehicles_by_id = {
                    str(v.get("vehicle_id")): v
                    for v in vehicles
                    if isinstance(v, dict) and "vehicle_id" in v
                }
            if self.config.check_payload_schema_violation:
                alerts.append({
                    "code": "payload_schema_violation",
                    "severity": "high",
                    "message": "payload 缺少 vehicles_by_id，已使用 vehicles 临时构造映射。",
                    "stats": {"missing": ["vehicles_by_id"]},
                })
        active_key = str(active_vehicle_id)
        vehicle_record = None
        if isinstance(vehicles_by_id, dict):
            vehicle_record = vehicles_by_id.get(active_key) or vehicles_by_id.get(active_vehicle_id)
        if vehicle_record is None:
            if self.config.check_payload_schema_violation:
                alerts.append({
                    "code": "payload_schema_violation",
                    "severity": "high",
                    "message": "vehicles_by_id 缺少 active_vehicle_id 对应记录，无法对齐校验。",
                    "stats": {"active_vehicle_id": active_vehicle_id},
                })
            return None, alerts
        return vehicle_record, alerts

    def _check_active_stop_mismatch(self, payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
        if not self.config.check_active_stop_mismatch:
            return [], False
        alerts: List[Dict[str, Any]] = []
        vehicle_record, schema_alerts = self._extract_active_vehicle_record(payload)
        alerts.extend(schema_alerts)
        if vehicle_record is None:
            return alerts, True if schema_alerts else False
        payload_stop = payload.get("current_stop")
        vehicle_stop = vehicle_record.get("current_stop") if isinstance(vehicle_record, dict) else None
        if payload_stop is None or vehicle_stop is None:
            if self.config.check_payload_schema_violation:
                alerts.append({
                    "code": "payload_schema_violation",
                    "severity": "high",
                    "message": "payload 或 vehicles_by_id 缺少 current_stop 字段，无法对齐校验。",
                    "stats": {
                        "payload_current_stop": payload_stop,
                        "vehicle_current_stop": vehicle_stop,
                    },
                })
            return alerts, True
        if int(payload_stop) != int(vehicle_stop):
            alerts.append({
                "code": "active_stop_mismatch",
                "severity": "high",
                "message": "active_stop 与车辆快照不一致，payload 可能混用 pre/post-step。",
                "stats": {
                    "active_vehicle_id": payload.get("active_vehicle_id"),
                    "payload_current_stop": int(payload_stop),
                    "vehicle_current_stop": int(vehicle_stop),
                },
            })
            return alerts, True
        return alerts, False

    def _check_time_feature_mismatch(self, payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
        if not self.config.check_time_feature_mismatch:
            return [], False
        alerts: List[Dict[str, Any]] = []
        vehicle_record, schema_alerts = self._extract_active_vehicle_record(payload)
        alerts.extend(schema_alerts)
        if vehicle_record is None:
            return alerts, True if schema_alerts else False
        pre_step_time = payload.get("pre_step_time", payload.get("current_time"))
        vehicle_time = vehicle_record.get("available_time") if isinstance(vehicle_record, dict) else None
        if pre_step_time is None or vehicle_time is None:
            if self.config.check_payload_schema_violation:
                alerts.append({
                    "code": "payload_schema_violation",
                    "severity": "high",
                    "message": "payload 或 vehicles_by_id 缺少时间字段，无法进行 time 对齐校验。",
                    "stats": {
                        "pre_step_time": pre_step_time,
                        "vehicle_available_time": vehicle_time,
                    },
                })
            return alerts, True
        diff = abs(float(pre_step_time) - float(vehicle_time))
        if diff > float(self.config.time_feature_mismatch_eps):
            alerts.append({
                "code": "time_feature_mismatch",
                "severity": "high",
                "message": "pre_step_time 与 active vehicle available_time 不一致，时间基准可能错位。",
                "stats": {
                    "active_vehicle_id": payload.get("active_vehicle_id"),
                    "pre_step_time": float(pre_step_time),
                    "vehicle_available_time": float(vehicle_time),
                    "diff": diff,
                },
            })
            return alerts, True
        return alerts, False

    def _check_looping(self) -> List[Dict[str, Any]]:
        if len(self.action_history) < self.config.loop_window:
            return []
        recent_actions = list(self.action_history)[-self.config.loop_window :]
        unique_stops = set(recent_actions)
        served_sum = float(sum(list(self.served_history)[-self.config.loop_window :]))
        if len(unique_stops) <= self.config.loop_unique_stops_max and served_sum <= self.config.loop_max_served:
            return [
                {
                    "code": "looping_small_support",
                    "severity": "medium",
                    "message": "动作在很少的站点间循环且服务量低。",
                    "stats": {
                        "window": self.config.loop_window,
                        "unique_stops": len(unique_stops),
                        "served_sum": served_sum,
                    },
                }
            ]
        return []

    def _check_reward_up_no_service(self) -> List[Dict[str, Any]]:
        window = self.config.reward_window
        if len(self.reward_history) < 2 * window:
            return []
        rewards = list(self.reward_history)
        prev = rewards[-2 * window : -window]
        last = rewards[-window:]
        prev_mean = float(sum(prev) / max(1, len(prev)))
        last_mean = float(sum(last) / max(1, len(last)))
        served_sum = float(sum(list(self.served_history)[-window:]))
        if (
            last_mean > self.config.reward_positive_threshold
            and last_mean - prev_mean > self.config.reward_delta_threshold
            and served_sum <= self.config.low_service_threshold
        ):
            return [
                {
                    "code": "reward_up_no_service",
                    "severity": "medium",
                    "message": "奖励均值上涨但服务量几乎为零。",
                    "stats": {
                        "prev_mean": prev_mean,
                        "last_mean": last_mean,
                        "served_sum": served_sum,
                        "window": window,
                    },
                }
            ]
        return []

    def _check_entropy_warning(self) -> List[Dict[str, Any]]:
        if self._last_action_mask_valid_count is not None and self._last_action_mask_valid_count <= 1:
            return []
        if len(self.entropy_history) < self.config.entropy_warning_patience:
            return []
        recent = list(self.entropy_history)[-self.config.entropy_warning_patience :]
        if all(val < self.config.entropy_warning_floor for val in recent):
            return [
                {
                    "code": "entropy_low_warning",
                    "severity": "medium",
                    "message": "策略熵持续偏低，可能探索不足。",
                    "stats": {
                        "entropy_floor": self.config.entropy_warning_floor,
                        "patience": self.config.entropy_warning_patience,
                    },
                }
            ]
        return []

    def _check_entropy_collapse(self) -> List[Dict[str, Any]]:
        if self._last_action_mask_valid_count is not None and self._last_action_mask_valid_count <= 1:
            return []
        if len(self.entropy_history) < self.config.entropy_patience:
            return []
        recent = list(self.entropy_history)[-self.config.entropy_patience :]
        latest_eps = float(self.epsilon_history[-1]) if self.epsilon_history else 0.0
        if (
            latest_eps <= self.config.epsilon_floor
            and all(val < self.config.entropy_floor for val in recent)
        ):
            return [
                {
                    "code": "entropy_collapse",
                    "severity": "high",
                    "message": "策略熵持续过低，可能出现策略坍缩。",
                    "stats": {
                        "entropy_floor": self.config.entropy_floor,
                        "epsilon": latest_eps,
                        "patience": self.config.entropy_patience,
                    },
                }
            ]
        return []

    def _should_abort(self, alerts: List[Dict[str, Any]]) -> bool:
        abort_codes = {str(code) for code in self.config.abort_codes}
        if abort_codes:
            for alert in alerts:
                if str(alert.get("code", "")) in abort_codes:
                    return True
            return False
        abort_levels = {level.lower() for level in self.config.abort_levels}
        for alert in alerts:
            severity = str(alert.get("severity", "")).lower()
            if severity in abort_levels:
                return True
        return False

    def _check_missing_keys(self, missing_keys: List[str]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        for key in missing_keys:
            alerts.append(
                {
                    "code": f"MISSING_KEY:{key}",
                    "severity": "high",
                    "message": f"Required key missing from payload: {key}",
                    "stats": {"key": key},
                }
            )
        return alerts

    def _check_potential_shaping_consistency(
        self,
        alpha: float,
        shaping_raw: float,
        phi_delta: float,
    ) -> List[Dict[str, Any]]:
        eps = float(self.config.phi_eps)
        alerts: List[Dict[str, Any]] = []
        if abs(alpha) <= eps and abs(shaping_raw) > eps:
            alerts.append(
                {
                    "code": "potential_alpha_zero_but_shaping_nonzero",
                    "severity": "high",
                    "message": "reward_potential_alpha=0 但 shaping_raw 非零。",
                    "stats": {"alpha": alpha, "shaping_raw": shaping_raw},
                }
            )
        if abs(shaping_raw) > eps and abs(phi_delta) <= eps:
            alerts.append(
                {
                    "code": "potential_shaping_phi_delta_mismatch",
                    "severity": "high",
                    "message": "shaping_raw 非零但 phi_delta 为 0。",
                    "stats": {"phi_delta": phi_delta, "shaping_raw": shaping_raw},
                }
            )
        return alerts

    def _check_phi_backlog_consistency(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        eps = float(self.config.phi_eps)
        alerts: List[Dict[str, Any]] = []
        phi_before = _safe_float(payload.get("phi_before", 0.0))
        phi_after = _safe_float(payload.get("phi_after", 0.0))
        phi_delta = _safe_float(payload.get("phi_delta", 0.0))
        phi_backlog_before = _safe_float(payload.get("phi_backlog_before", 0.0))
        phi_backlog_after = _safe_float(payload.get("phi_backlog_after", 0.0))
        lost_total_before = _safe_float(payload.get("lost_total_before", 0.0))
        lost_total_after = _safe_float(payload.get("lost_total_after", 0.0))
        waiting_churned_before = _safe_float(payload.get("waiting_churned_before", 0.0))
        waiting_churned_after = _safe_float(payload.get("waiting_churned_after", 0.0))
        onboard_churned_before = _safe_float(payload.get("onboard_churned_before", 0.0))
        onboard_churned_after = _safe_float(payload.get("onboard_churned_after", 0.0))
        structural_before = _safe_float(payload.get("structural_unserviceable_before", 0.0))
        structural_after = _safe_float(payload.get("structural_unserviceable_after", 0.0))
        waiting_before = _safe_float(payload.get("waiting_remaining_before", 0.0))
        waiting_after = _safe_float(payload.get("waiting_remaining_after", 0.0))
        onboard_before = _safe_float(payload.get("onboard_remaining_before", 0.0))
        onboard_after = _safe_float(payload.get("onboard_remaining_after", 0.0))
        if abs((phi_before - phi_after) - phi_delta) > eps:
            alerts.append(
                {
                    "code": "phi_delta_mismatch",
                    "severity": "high",
                    "message": "phi_delta != phi_before - phi_after",
                    "stats": {"phi_before": phi_before, "phi_after": phi_after, "phi_delta": phi_delta},
                }
            )
        if abs((waiting_before + onboard_before) - phi_backlog_before) > eps:
            alerts.append(
                {
                    "code": "phi_backlog_before_mismatch",
                    "severity": "high",
                    "message": "phi_backlog_before != waiting_before + onboard_before",
                    "stats": {
                        "phi_backlog_before": phi_backlog_before,
                        "waiting_before": waiting_before,
                        "onboard_before": onboard_before,
                    },
                }
            )
        if abs((waiting_after + onboard_after) - phi_backlog_after) > eps:
            alerts.append(
                {
                    "code": "phi_backlog_after_mismatch",
                    "severity": "high",
                    "message": "phi_backlog_after != waiting_after + onboard_after",
                    "stats": {
                        "phi_backlog_after": phi_backlog_after,
                        "waiting_after": waiting_after,
                        "onboard_after": onboard_after,
                    },
                }
            )
        if abs((waiting_churned_before + onboard_churned_before + structural_before) - lost_total_before) > eps:
            alerts.append(
                {
                    "code": "lost_total_before_mismatch",
                    "severity": "high",
                    "message": "lost_total_before != waiting_churned_before + onboard_churned_before + structural_before",
                    "stats": {
                        "lost_total_before": lost_total_before,
                        "waiting_churned_before": waiting_churned_before,
                        "onboard_churned_before": onboard_churned_before,
                        "structural_before": structural_before,
                    },
                }
            )
        if abs((waiting_churned_after + onboard_churned_after + structural_after) - lost_total_after) > eps:
            alerts.append(
                {
                    "code": "lost_total_after_mismatch",
                    "severity": "high",
                    "message": "lost_total_after != waiting_churned_after + onboard_churned_after + structural_after",
                    "stats": {
                        "lost_total_after": lost_total_after,
                        "waiting_churned_after": waiting_churned_after,
                        "onboard_churned_after": onboard_churned_after,
                        "structural_after": structural_after,
                    },
                }
            )
        return alerts

    def _check_reward_total_consistency(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        reward_terms = payload.get("reward_terms", {})
        if not isinstance(reward_terms, dict):
            return []
        total = _safe_float(payload.get("reward_total", 0.0))
        sum_terms = 0.0
        for key, value in reward_terms.items():
            if key == "reward_total" or key.endswith("_raw"):
                continue
            sum_terms += _safe_float(value)
        if abs(total - sum_terms) > float(self.config.reward_total_eps):
            return [
                {
                    "code": "reward_total_mismatch",
                    "severity": "high",
                    "message": "reward_total != sum(reward_components)",
                    "stats": {"reward_total": total, "sum_components": sum_terms},
                }
            ]
        return []

    def _check_served_per_decision_consistency(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        served_per_decision = _safe_float(payload.get("served_per_decision", 0.0))
        step_served = _safe_float(payload.get("step_served", 0.0))
        if step_served > 0.0 and abs(served_per_decision) <= float(self.config.phi_eps):
            return [
                {
                    "code": "served_per_decision_mismatch",
                    "severity": "medium",
                    "message": "step_served>0 but served_per_decision==0",
                    "stats": {"step_served": step_served, "served_per_decision": served_per_decision},
                }
            ]
        return []

    # ========== 新增检测方法 ==========

    def _check_action_space_collapse(self) -> List[Dict[str, Any]]:
        """检测可行动作空间塌缩（两级阈值）。"""
        if self._last_action_mask_valid_count is None:
            return []
        
        threshold = self.config.action_collapse_threshold
        if self._last_action_mask_valid_count <= threshold:
            self._action_collapse_streak += 1
        else:
            self._action_collapse_streak = 0
            return []
        
        alerts: List[Dict[str, Any]] = []
        # High 级别：连续 50 步塌缩
        if self._action_collapse_streak >= self.config.action_collapse_patience_high:
            alerts.append({
                "code": "action_space_collapse",
                "severity": "high",
                "message": f"可行动作数连续 {self._action_collapse_streak} 步 ≤ {threshold}，决策空间严重受限，训练不可学。",
                "stats": {
                    "streak": self._action_collapse_streak,
                    "threshold": threshold,
                    "patience_high": self.config.action_collapse_patience_high,
                    "last_valid_count": self._last_action_mask_valid_count,
                },
            })
        # Medium 级别：连续 20 步塌缩
        elif self._action_collapse_streak >= self.config.action_collapse_patience_medium:
            alerts.append({
                "code": "action_space_collapse",
                "severity": "medium",
                "message": f"可行动作数连续 {self._action_collapse_streak} 步 ≤ {threshold}，可能进入受限态。",
                "stats": {
                    "streak": self._action_collapse_streak,
                    "threshold": threshold,
                    "patience_medium": self.config.action_collapse_patience_medium,
                    "last_valid_count": self._last_action_mask_valid_count,
                },
            })
        if self._last_active_stop_mismatch:
            for alert in alerts:
                alert["severity"] = "medium"
                alert["message"] = f"[可能受 mismatch 污染] {alert['message']}"
                stats = alert.setdefault("stats", {})
                stats["alignment_mismatch"] = True
        return alerts

    def _check_action_collapse_shortcut(self) -> List[Dict[str, Any]]:
        """短路条件：塌缩 + 熵≈0 + epsilon高 → 直接 high。"""
        if self._last_active_stop_mismatch:
            return []
        if self._last_action_mask_valid_count is None:
            return []
        
        threshold = self.config.action_collapse_threshold
        entropy_threshold = self.config.entropy_near_zero_threshold
        epsilon_high = self.config.entropy_collapse_epsilon_high
        
        is_collapsed = self._last_action_mask_valid_count <= threshold
        is_entropy_zero = self._last_entropy_norm < entropy_threshold
        is_epsilon_high = self._last_epsilon > epsilon_high
        
        if is_collapsed and is_entropy_zero and is_epsilon_high:
            return [{
                "code": "action_collapse_shortcut",
                "severity": "high",
                "message": f"短路触发：塌缩(valid={self._last_action_mask_valid_count}) + 熵≈0({self._last_entropy_norm:.4f}) + epsilon高({self._last_epsilon:.3f})，训练信号断流。",
                "stats": {
                    "valid_count": self._last_action_mask_valid_count,
                    "entropy_norm": self._last_entropy_norm,
                    "epsilon": self._last_epsilon,
                    "thresholds": {
                        "collapse": threshold,
                        "entropy": entropy_threshold,
                        "epsilon_high": epsilon_high,
                    },
                },
            }]
        return []

    def _check_action_ratio_sudden_drop(self) -> List[Dict[str, Any]]:
        """检测 action_valid_ratio 突变（相对下降 + 绝对危险线）。"""
        window = self.config.action_ratio_window
        if len(self.action_valid_ratio_history) < 2 * window:
            return []
        if len(self.action_valid_count_history) < 2 * window:
            return []
        
        ratios = list(self.action_valid_ratio_history)
        counts = list(self.action_valid_count_history)
        action_counts = list(self.action_count_history) if self.action_count_history else []
        prev_window = ratios[-2 * window : -window]
        last_window = ratios[-window:]
        prev_count_window = counts[-2 * window : -window]
        last_count_window = counts[-window:]
        last_action_count = None
        if len(action_counts) >= window:
            recent_action_counts = action_counts[-window:]
            last_action_count = sorted(recent_action_counts)[len(recent_action_counts) // 2]
        
        # 使用中位数作为稳健统计
        prev_median = sorted(prev_window)[len(prev_window) // 2]
        last_median = sorted(last_window)[len(last_window) // 2]
        prev_count_median = sorted(prev_count_window)[len(prev_count_window) // 2]
        last_count_median = sorted(last_count_window)[len(last_count_window) // 2]
        
        relative_drop = (prev_median - last_median) / max(prev_median, 0.01)
        absolute_floor = self.config.action_ratio_absolute_floor
        min_valid_count = int(self.config.action_ratio_min_valid_count)
        if last_action_count is not None and last_action_count > 0:
            dynamic_floor = float(min_valid_count) / float(last_action_count)
            absolute_floor = min(absolute_floor, dynamic_floor)
        drop_threshold = self.config.action_ratio_drop_threshold
        
        # 相对下降 > 阈值 且 当前值 < 绝对危险线
        if relative_drop > drop_threshold and last_median < absolute_floor and last_count_median <= min_valid_count:
            return [{
                "code": "action_ratio_sudden_drop",
                "severity": "medium",
                "message": f"action_valid_ratio 突变：{prev_median:.3f} → {last_median:.3f}（下降 {relative_drop*100:.1f}%），低于危险线 {absolute_floor:.3f}。",
                "stats": {
                    "prev_median": prev_median,
                    "last_median": last_median,
                    "prev_valid_count_median": prev_count_median,
                    "last_valid_count_median": last_count_median,
                    "last_action_count_median": last_action_count,
                    "relative_drop": relative_drop,
                    "absolute_floor": absolute_floor,
                    "min_valid_count": min_valid_count,
                    "window": window,
                },
            }]
        return []

    def _check_entropy_action_consistency(self) -> List[Dict[str, Any]]:
        """检测熵与可行动作数的一致性。"""
        if self._last_action_mask_valid_count is None:
            return []
        
        entropy_threshold = self.config.entropy_near_zero_threshold
        alerts: List[Dict[str, Any]] = []
        
        # 情况 A：熵≈0 但可行动作数 > 1 → 熵计算/输出异常
        if self._last_entropy_norm < entropy_threshold and self._last_action_mask_valid_count > 1:
            alerts.append({
                "code": "entropy_action_inconsistency",
                "severity": "high",
                "message": f"熵-可行率矛盾：q_entropy_norm={self._last_entropy_norm:.4f}≈0 但 valid_actions={self._last_action_mask_valid_count}>1，疑似熵计算/遮罩顺序异常。",
                "stats": {
                    "entropy_norm": self._last_entropy_norm,
                    "valid_count": self._last_action_mask_valid_count,
                    "entropy_threshold": entropy_threshold,
                },
            })
        
        # 情况 B：熵 > 阈值 但可行动作数 ≤ 1 → 统计口径异常
        if self._last_entropy_norm > 0.1 and self._last_action_mask_valid_count <= 1:
            alerts.append({
                "code": "entropy_action_inconsistency",
                "severity": "medium",
                "message": f"熵-可行率矛盾：q_entropy_norm={self._last_entropy_norm:.4f}>0 但 valid_actions={self._last_action_mask_valid_count}≤1，疑似统计口径不一致。",
                "stats": {
                    "entropy_norm": self._last_entropy_norm,
                    "valid_count": self._last_action_mask_valid_count,
                },
            })
        return alerts

    def _check_valid_invalid_ratio_consistency(
        self,
        action_valid_ratio: float,
        invalid_action_ratio: float,
    ) -> List[Dict[str, Any]]:
        """检测 valid_ratio 与 invalid_ratio 自洽性。"""
        if invalid_action_ratio > 0.01 and self._last_action_mask_valid_count is not None:
            if self._last_action_mask_valid_count > 5:
                return [{
                    "code": "invalid_action_ratio_high_with_valid_mask",
                    "severity": "medium",
                    "message": f"invalid_ratio={invalid_action_ratio:.4f} 偏高，但 valid_actions={self._last_action_mask_valid_count} 仍充足，疑似 mask/执行链路异常。",
                    "stats": {
                        "action_valid_ratio": action_valid_ratio,
                        "invalid_action_ratio": invalid_action_ratio,
                        "valid_count": self._last_action_mask_valid_count,
                    },
                }]
        # 候选空间狭窄但动作执行合法
        if invalid_action_ratio < 0.01 and action_valid_ratio < 0.1:
            return [{
                "code": "action_space_narrow",
                "severity": "info",
                "message": f"候选空间偏窄：valid_ratio={action_valid_ratio:.4f} 但 invalid_ratio≈0，通常表示仅少数动作可行。",
                "stats": {
                    "action_valid_ratio": action_valid_ratio,
                    "invalid_action_ratio": invalid_action_ratio,
                    "valid_count": self._last_action_mask_valid_count,
                },
            }]
        return []

    def _check_entropy_collapse_epsilon_high(self) -> List[Dict[str, Any]]:
        """熵塌缩持续 + epsilon 高告警。"""
        if len(self.entropy_history) < self.config.entropy_patience:
            return []
        
        recent_entropy = list(self.entropy_history)[-self.config.entropy_patience:]
        entropy_floor = self.config.entropy_warning_floor
        epsilon_high = self.config.entropy_collapse_epsilon_high
        
        all_low = all(e < entropy_floor for e in recent_entropy)
        is_epsilon_high = self._last_epsilon > epsilon_high
        
        if all_low and is_epsilon_high:
            return [{
                "code": "entropy_collapse_epsilon_high",
                "severity": "high",
                "message": f"熵持续低于 {entropy_floor}（{len(recent_entropy)} 步）且 epsilon={self._last_epsilon:.3f} 仍高，策略退化或 mask 锁死。",
                "stats": {
                    "entropy_floor": entropy_floor,
                    "patience": self.config.entropy_patience,
                    "epsilon": self._last_epsilon,
                    "epsilon_threshold": epsilon_high,
                },
            }]
        return []

    def _check_decision_efficiency_decay(self) -> List[Dict[str, Any]]:
        """单位决策效率恶化告警。"""
        window = self.config.efficiency_decay_window
        if len(self.served_per_decision_history) < window:
            return []
        
        half = window // 2
        served_first = list(self.served_per_decision_history)[-window:-half]
        served_last = list(self.served_per_decision_history)[-half:]
        churn_first = list(self.churn_per_decision_history)[-window:-half]
        churn_last = list(self.churn_per_decision_history)[-half:]
        
        served_first_mean = sum(served_first) / len(served_first) if served_first else 0.0
        served_last_mean = sum(served_last) / len(served_last) if served_last else 0.0
        churn_first_mean = sum(churn_first) / len(churn_first) if churn_first else 0.0
        churn_last_mean = sum(churn_last) / len(churn_last) if churn_last else 0.0
        
        threshold = self.config.efficiency_decay_threshold
        served_drop = (served_first_mean - served_last_mean) / max(served_first_mean, 0.001)
        churn_rise = (churn_last_mean - churn_first_mean) / max(churn_first_mean, 0.001)
        
        if served_drop > threshold and churn_rise > 0:
            return [{
                "code": "decision_efficiency_decay",
                "severity": "medium",
                "message": f"决策效率恶化：served_per_decision 下降 {served_drop*100:.1f}%，churn_per_decision 上升 {churn_rise*100:.1f}%。",
                "stats": {
                    "served_first_mean": served_first_mean,
                    "served_last_mean": served_last_mean,
                    "churn_first_mean": churn_first_mean,
                    "churn_last_mean": churn_last_mean,
                    "served_drop": served_drop,
                    "churn_rise": churn_rise,
                    "window": window,
                },
            }]
        return []

    def _check_overload_deadlock(self) -> List[Dict[str, Any]]:
        """过载锁死复合告警。"""
        if len(self.backlog_history) < self.config.overload_patience:
            return []
        
        recent_backlog = list(self.backlog_history)[-self.config.overload_patience:]
        recent_ready = list(self.ready_vehicles_history)[-self.config.overload_patience:]
        recent_queue = list(self.event_queue_history)[-self.config.overload_patience:]
        
        backlog_threshold = self.config.overload_backlog_threshold
        ready_threshold = self.config.overload_ready_vehicles_threshold
        queue_threshold = self.config.overload_event_queue_threshold
        collapse_threshold = self.config.action_collapse_threshold
        
        all_backlog_high = all(b >= backlog_threshold for b in recent_backlog)
        all_ready_low = all(r <= ready_threshold for r in recent_ready)
        all_queue_high = all(q >= queue_threshold for q in recent_queue)
        is_collapsed = (self._last_action_mask_valid_count is not None and 
                       self._last_action_mask_valid_count <= collapse_threshold)
        
        if all_backlog_high and all_ready_low and all_queue_high and is_collapsed:
            self._overload_streak += 1
            if self._overload_streak >= self.config.overload_patience:
                return [{
                    "code": "overload_deadlock",
                    "severity": "high",
                    "message": f"过载锁死：backlog≥{backlog_threshold}、ready_vehicles≤{ready_threshold}、queue≥{queue_threshold}、动作塌缩，系统进入死局。",
                    "stats": {
                        "backlog_mean": sum(recent_backlog) / len(recent_backlog),
                        "ready_vehicles_mean": sum(recent_ready) / len(recent_ready),
                        "event_queue_mean": sum(recent_queue) / len(recent_queue),
                        "valid_count": self._last_action_mask_valid_count,
                        "patience": self.config.overload_patience,
                    },
                }]
        else:
            self._overload_streak = 0
        return []

    def _check_potential_shaping_inactive(self) -> List[Dict[str, Any]]:
        """潜势塑形失效提示。"""
        patience = self.config.shaping_inactive_patience
        if (
            len(self.shaping_history) < patience
            or len(self.alpha_history) < patience
            or len(self.phi_delta_history) < patience
        ):
            return []
        
        recent_alpha = list(self.alpha_history)[-patience:]
        recent_shaping = list(self.shaping_history)[-patience:]
        recent_phi_delta = list(self.phi_delta_history)[-patience:]
        recent_events = list(self.event_count_history)[-patience:] if self.event_count_history else []
        recent_backlog_delta = list(self.backlog_delta_history)[-patience:] if self.backlog_delta_history else []
        
        # alpha > 0 但 shaping 始终 ≈ 0
        alpha_threshold = float(self.config.shaping_inactive_alpha_threshold)
        shaping_eps = float(self.config.shaping_inactive_eps)
        min_nonzero_ratio = float(self.config.shaping_inactive_min_nonzero_ratio)
        alpha_positive = all(a > alpha_threshold for a in recent_alpha)
        shaping_zero = all(abs(s) < shaping_eps for s in recent_shaping)
        nonzero_ratio = sum(1 for s in recent_shaping if abs(s) > shaping_eps) / float(patience)
        phi_eps = float(self.config.phi_eps)
        phi_changed = any(abs(p) > phi_eps for p in recent_phi_delta)
        event_present = any(e > 0.0 for e in recent_events) or any(abs(b) > phi_eps for b in recent_backlog_delta)
        
        alerts: List[Dict[str, Any]] = []
        if alpha_positive:
            if any(abs(p) > phi_eps and abs(s) <= shaping_eps for p, s in zip(recent_phi_delta, recent_shaping)):
                alerts.append({
                    "code": "potential_shaping_zero_with_phi_change",
                    "severity": "high",
                    "message": "phi 发生变化但 shaping 仍为 0，疑似记录点或计算顺序错误。",
                    "stats": {
                        "alpha_mean": sum(recent_alpha) / len(recent_alpha),
                        "patience": patience,
                        "phi_eps": phi_eps,
                        "shaping_eps": shaping_eps,
                    },
                })
            if event_present and shaping_zero and not phi_changed:
                alerts.append({
                    "code": "potential_shaping_inactive",
                    "severity": "medium",
                    "message": f"窗口内存在事件但 phi/shaping 均未变化，潜势塑形可能失效。",
                    "stats": {
                        "alpha_mean": sum(recent_alpha) / len(recent_alpha),
                        "patience": patience,
                        "event_present": True,
                    },
                })
            if nonzero_ratio < min_nonzero_ratio and shaping_zero:
                alerts.append({
                    "code": "potential_shaping_inactive",
                    "severity": "info",
                    "message": "潜势塑形长期接近 0（非零占比过低），建议检查 phi 更新条件。",
                    "stats": {
                        "alpha_mean": sum(recent_alpha) / len(recent_alpha),
                        "nonzero_ratio": nonzero_ratio,
                        "min_nonzero_ratio": min_nonzero_ratio,
                        "patience": patience,
                    },
                })
        return alerts

    def _check_singleton_selfloop_persist(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.config.check_self_loop_singleton:
            return []
        action_mask = payload.get("action_mask")
        action_candidates = payload.get("action_candidates")
        current_stop = payload.get("current_stop")
        flag = False
        if isinstance(action_mask, list) and isinstance(action_candidates, list) and current_stop is not None:
            valid_indices = [idx for idx, keep in enumerate(action_mask) if keep]
            if len(valid_indices) == 1 and valid_indices[0] < len(action_candidates):
                only_action = action_candidates[valid_indices[0]]
                if int(only_action) == int(current_stop):
                    flag = True
        if flag:
            self._self_loop_singleton_streak += 1
        else:
            self._self_loop_singleton_streak = 0
        if self._self_loop_singleton_streak >= self.config.self_loop_singleton_patience:
            return [{
                "code": "action_space_singleton_selfloop_persist",
                "severity": "high",
                "message": f"可行动作长期仅剩 self-loop，持续 {self._self_loop_singleton_streak} 步，疑似局部死锁。",
                "stats": {
                    "streak": self._self_loop_singleton_streak,
                    "patience": self.config.self_loop_singleton_patience,
                    "current_stop": current_stop,
                },
            }]
        return []

    def _dump_alerts(self, payload: Dict[str, Any], alerts: List[Dict[str, Any]]) -> List[str]:
        dump_dir = Path(self.config.debug_dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        global_step = payload.get("global_step", "na")
        ts = int(time())
        for idx, alert in enumerate(alerts):
            filename = f"alert_{global_step}_{ts}_{idx}.json"
            path = dump_dir / filename
            record = {
                "alert": alert,
                "payload": payload,
            }
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            paths.append(str(path))
        return paths


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
