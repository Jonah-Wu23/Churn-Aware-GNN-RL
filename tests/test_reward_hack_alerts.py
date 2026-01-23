from src.utils.reward_hack_alerts import RewardHackAlertConfig, RewardHackDetector


def _make_payload(
    action_mask_valid_count: int = 5,
    epsilon: float = 0.5,
    entropy: float = 0.5,
    action_valid_ratio: float = 0.5,
    invalid_action_ratio: float = 0.0,
    action_count: int = 10,
    served_per_decision: float = 0.1,
    waiting_churn_per_decision: float = 0.05,
    waiting_remaining: float = 50.0,
    onboard_remaining: float = 20.0,
    ready_vehicles: int = 5,
    event_queue_len: int = 100,
    reward_potential_alpha: float = 0.0,
    reward_potential_shaping_raw: float = 0.0,
    phi_delta: float = 0.0,
    step_waiting_churned: float = 0.0,
    step_onboard_churned: float = 0.0,
    step_waiting_timeouts: float = 0.0,
    waiting_remaining_before: float = 50.0,
    waiting_remaining_after: float = 50.0,
    onboard_remaining_before: float = 20.0,
    onboard_remaining_after: float = 20.0,
    active_vehicle_id: int = 1,
    current_stop: int = 10,
    vehicle_current_stop: int = 10,
    pre_step_time: float = 100.0,
    vehicle_available_time: float = 100.0,
    action_candidates: list | None = None,
    action_mask: list | None = None,
) -> dict:
    if action_candidates is None:
        action_candidates = list(range(int(action_count)))
    if action_mask is None:
        total = int(action_count)
        keep = max(0, min(int(action_mask_valid_count), total))
        action_mask = [True] * keep + [False] * (total - keep)
    return {
        "reward_total": 0.0,
        "step_served": 0.0,
        "step_waiting_churned": float(step_waiting_churned),
        "step_onboard_churned": float(step_onboard_churned),
        "step_waiting_timeouts": float(step_waiting_timeouts),
        "epsilon": float(epsilon),
        "q_entropy_norm": float(entropy),
        "action_mask_valid_count": int(action_mask_valid_count),
        "action_valid_ratio": float(action_valid_ratio),
        "invalid_action_ratio": float(invalid_action_ratio),
        "action_count": int(action_count),
        "action_candidates": action_candidates,
        "action_mask": action_mask,
        "served_per_decision": float(served_per_decision),
        "waiting_churn_per_decision": float(waiting_churn_per_decision),
        "waiting_remaining": float(waiting_remaining),
        "onboard_remaining": float(onboard_remaining),
        "waiting_remaining_before": float(waiting_remaining_before),
        "waiting_remaining_after": float(waiting_remaining_after),
        "onboard_remaining_before": float(onboard_remaining_before),
        "onboard_remaining_after": float(onboard_remaining_after),
        "ready_vehicles": int(ready_vehicles),
        "event_queue_len": int(event_queue_len),
        "reward_potential_alpha": float(reward_potential_alpha),
        "reward_terms": {"reward_potential_shaping_raw": float(reward_potential_shaping_raw)},
        "phi_delta": float(phi_delta),
        "active_vehicle_id": int(active_vehicle_id),
        "current_stop": int(current_stop),
        "pre_step_time": float(pre_step_time),
        "vehicles_by_id": {
            str(int(active_vehicle_id)): {
                "vehicle_id": int(active_vehicle_id),
                "current_stop": int(vehicle_current_stop),
                "available_time": float(vehicle_available_time),
            }
        },
    }


def test_entropy_collapse_ignored_when_single_action() -> None:
    cfg = RewardHackAlertConfig(
        entropy_floor=0.25,
        entropy_patience=5,
        epsilon_floor=0.2,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = []
    for _ in range(cfg.entropy_patience):
        alerts = detector.update(_make_payload(action_mask_valid_count=1, epsilon=0.1, entropy=0.0))
    assert not any(alert.get("code") == "entropy_collapse" for alert in alerts)


def test_entropy_collapse_triggers_when_actions_available() -> None:
    cfg = RewardHackAlertConfig(
        entropy_floor=0.25,
        entropy_patience=5,
        epsilon_floor=0.2,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = []
    for _ in range(cfg.entropy_patience):
        alerts = detector.update(_make_payload(action_mask_valid_count=2, epsilon=0.1, entropy=0.0))
    assert any(alert.get("code") == "entropy_collapse" for alert in alerts)


# ========== 新增测试 ==========


def test_action_collapse_medium_level() -> None:
    """测试连续 20 步塌缩触发 medium 告警。"""
    cfg = RewardHackAlertConfig(
        action_collapse_threshold=1,
        action_collapse_patience_medium=20,
        action_collapse_patience_high=50,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = []
    for _ in range(25):
        alerts = detector.update(_make_payload(action_mask_valid_count=1))
    collapse_alerts = [a for a in alerts if a.get("code") == "action_space_collapse"]
    assert any(a.get("severity") == "medium" for a in collapse_alerts)


def test_action_collapse_high_level() -> None:
    """测试连续 50 步塌缩触发 high 告警。"""
    cfg = RewardHackAlertConfig(
        action_collapse_threshold=1,
        action_collapse_patience_medium=20,
        action_collapse_patience_high=50,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = []
    for _ in range(55):
        alerts = detector.update(_make_payload(action_mask_valid_count=1))
    collapse_alerts = [a for a in alerts if a.get("code") == "action_space_collapse"]
    assert any(a.get("severity") == "high" for a in collapse_alerts)


def test_action_collapse_recovery() -> None:
    """测试恢复后清除塌缩计数。"""
    cfg = RewardHackAlertConfig(
        action_collapse_threshold=1,
        action_collapse_patience_medium=20,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    # 先塌缩 15 步
    for _ in range(15):
        detector.update(_make_payload(action_mask_valid_count=1))
    # 恢复
    detector.update(_make_payload(action_mask_valid_count=5))
    # 再塌缩 15 步，不应触发
    alerts = []
    for _ in range(15):
        alerts = detector.update(_make_payload(action_mask_valid_count=1))
    collapse_alerts = [a for a in alerts if a.get("code") == "action_space_collapse"]
    assert len(collapse_alerts) == 0


def test_action_collapse_shortcut() -> None:
    """测试短路条件：塌缩 + 熵≈0 + epsilon高。"""
    cfg = RewardHackAlertConfig(
        action_collapse_threshold=1,
        entropy_near_zero_threshold=0.01,
        entropy_collapse_epsilon_high=0.3,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = detector.update(_make_payload(
        action_mask_valid_count=1,
        entropy=0.001,
        epsilon=0.8,
    ))
    assert any(a.get("code") == "action_collapse_shortcut" for a in alerts)


def test_active_stop_mismatch_triggers() -> None:
    cfg = RewardHackAlertConfig(debug_abort_on_alert=False)
    detector = RewardHackDetector(cfg)
    alerts = detector.update(_make_payload(
        current_stop=448,
        vehicle_current_stop=506,
    ))
    assert any(a.get("code") == "active_stop_mismatch" for a in alerts)


def test_shortcut_blocked_by_mismatch() -> None:
    cfg = RewardHackAlertConfig(
        action_collapse_threshold=1,
        entropy_near_zero_threshold=0.01,
        entropy_collapse_epsilon_high=0.3,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = detector.update(_make_payload(
        action_mask_valid_count=1,
        entropy=0.001,
        epsilon=0.8,
        current_stop=1,
        vehicle_current_stop=2,
    ))
    assert not any(a.get("code") == "action_collapse_shortcut" for a in alerts)


def test_singleton_selfloop_persist() -> None:
    cfg = RewardHackAlertConfig(
        self_loop_singleton_patience=3,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = []
    for _ in range(cfg.self_loop_singleton_patience):
        alerts = detector.update(_make_payload(
            action_count=3,
            action_candidates=[7, 8, 9],
            action_mask=[False, False, True],
            action_mask_valid_count=1,
            current_stop=9,
        ))
    assert any(a.get("code") == "action_space_singleton_selfloop_persist" for a in alerts)


def test_time_feature_mismatch_triggers() -> None:
    cfg = RewardHackAlertConfig(debug_abort_on_alert=False)
    detector = RewardHackDetector(cfg)
    alerts = detector.update(_make_payload(
        pre_step_time=100.0,
        vehicle_available_time=120.0,
    ))
    assert any(a.get("code") == "time_feature_mismatch" for a in alerts)


def test_action_ratio_drop_with_floor() -> None:
    """测试 ratio 相对下降 + 绝对危险线双检测。"""
    cfg = RewardHackAlertConfig(
        action_ratio_drop_threshold=0.5,
        action_ratio_absolute_floor=0.1,
        action_ratio_window=10,
        action_ratio_min_valid_count=2,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    # 前 10 步高 ratio
    for _ in range(10):
        detector.update(_make_payload(
            action_valid_ratio=0.8,
            action_mask_valid_count=10,
            action_count=10,
        ))
    # 后 10 步低 ratio
    alerts = []
    for _ in range(10):
        alerts = detector.update(_make_payload(
            action_valid_ratio=0.05,
            action_mask_valid_count=1,
            action_count=30,
        ))
    assert any(a.get("code") == "action_ratio_sudden_drop" for a in alerts)


def test_entropy_action_inconsistency_zero_entropy_multiple_actions() -> None:
    """测试熵=0 但多动作触发告警。"""
    cfg = RewardHackAlertConfig(
        entropy_near_zero_threshold=0.01,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = detector.update(_make_payload(
        action_mask_valid_count=10,
        entropy=0.001,
    ))
    assert any(a.get("code") == "entropy_action_inconsistency" for a in alerts)


def test_valid_invalid_ratio_consistency() -> None:
    """测试 valid/invalid ratio 自洽性检查。"""
    cfg = RewardHackAlertConfig(debug_abort_on_alert=False)
    detector = RewardHackDetector(cfg)
    alerts = detector.update(_make_payload(
        action_valid_ratio=0.05,
        invalid_action_ratio=0.0,
    ))
    assert any(a.get("code") == "action_space_narrow" for a in alerts)


def test_entropy_collapse_epsilon_high() -> None:
    """测试熵塌缩 + epsilon高告警。"""
    cfg = RewardHackAlertConfig(
        entropy_warning_floor=0.1,
        entropy_patience=5,
        entropy_collapse_epsilon_high=0.3,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = []
    for _ in range(cfg.entropy_patience):
        alerts = detector.update(_make_payload(
            action_mask_valid_count=5,  # 确保不是单动作跳过
            entropy=0.05,
            epsilon=0.6,
        ))
    assert any(a.get("code") == "entropy_collapse_epsilon_high" for a in alerts)


def test_overload_deadlock() -> None:
    """测试过载锁死复合告警。"""
    cfg = RewardHackAlertConfig(
        overload_backlog_threshold=100.0,
        overload_ready_vehicles_threshold=2,
        overload_event_queue_threshold=500,
        overload_patience=5,
        action_collapse_threshold=1,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = []
    # 需要 2 * patience 步才能触发（先填充历史，再计数 streak）
    for _ in range(cfg.overload_patience * 2 + 1):
        alerts = detector.update(_make_payload(
            action_mask_valid_count=1,
            waiting_remaining=80.0,
            onboard_remaining=50.0,  # backlog=130
            ready_vehicles=1,
            event_queue_len=600,
        ))
    assert any(a.get("code") == "overload_deadlock" for a in alerts)


def test_potential_shaping_inactive() -> None:
    """测试潜势塑形失效告警。"""
    cfg = RewardHackAlertConfig(
        shaping_inactive_patience=10,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = []
    for _ in range(cfg.shaping_inactive_patience + 1):
        alerts = detector.update(_make_payload(
            reward_potential_alpha=0.1,
            reward_potential_shaping_raw=0.0,
            step_waiting_churned=1.0,
            phi_delta=0.0,
        ))
    assert any(a.get("code") == "potential_shaping_inactive" for a in alerts)


def test_potential_shaping_zero_with_phi_change() -> None:
    cfg = RewardHackAlertConfig(
        shaping_inactive_patience=5,
        debug_abort_on_alert=False,
    )
    detector = RewardHackDetector(cfg)
    alerts = []
    for _ in range(cfg.shaping_inactive_patience):
        alerts = detector.update(_make_payload(
            reward_potential_alpha=0.1,
            reward_potential_shaping_raw=0.0,
            phi_delta=1.0,
        ))
    assert any(a.get("code") == "potential_shaping_zero_with_phi_change" for a in alerts)
