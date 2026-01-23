"""Reward hacking and invariant alert heuristics for realtime monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional
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
    debug_dump_dir: str = "reports/debug/potential_alerts"
    reward_total_eps: float = 1e-6
    phi_eps: float = 1e-9
    no_service_positive_reward_epsilon: float = 0.1

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
            debug_dump_dir=str(raw.get("debug_dump_dir", "reports/debug/potential_alerts")),
            reward_total_eps=float(raw.get("reward_total_eps", 1e-6)),
            phi_eps=float(raw.get("phi_eps", 1e-9)),
            no_service_positive_reward_epsilon=float(raw.get("no_service_positive_reward_epsilon", 0.1)),
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

    def __post_init__(self) -> None:
        max_len = max(
            2 * self.config.reward_window,
            self.config.loop_window,
            self.config.entropy_patience,
            self.config.low_service_steps,
        )
        self.reward_history = deque(maxlen=max_len)
        self.served_history = deque(maxlen=max_len)
        self.action_history = deque(maxlen=max_len)
        self.entropy_history = deque(maxlen=max_len)
        self.epsilon_history = deque(maxlen=max_len)
        self.alert_history = deque(maxlen=int(self.config.alert_buffer_size))

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

        self.reward_history.append(reward_total)
        self.served_history.append(step_served)
        self.epsilon_history.append(float(epsilon))
        if action_mask_valid_count is not None:
            try:
                self._last_action_mask_valid_count = int(action_mask_valid_count)
            except (TypeError, ValueError):
                self._last_action_mask_valid_count = None
        if action_stop is not None:
            self.action_history.append(int(action_stop))
        if entropy_norm is not None:
            self.entropy_history.append(float(entropy_norm))

        alerts: List[Dict[str, Any]] = []
        if missing_keys:
            alerts.extend(self._check_missing_keys(missing_keys))
        alerts.extend(self._check_potential_shaping_consistency(alpha, shaping_raw, phi_delta))
        alerts.extend(self._check_phi_backlog_consistency(payload))
        alerts.extend(self._check_reward_total_consistency(payload))
        alerts.extend(self._check_served_per_decision_consistency(payload))
        alerts.extend(self._check_no_service_positive_reward(reward_total, step_served))
        alerts.extend(self._check_looping())
        alerts.extend(self._check_reward_up_no_service())
        alerts.extend(self._check_entropy_warning())
        alerts.extend(self._check_entropy_collapse())

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
