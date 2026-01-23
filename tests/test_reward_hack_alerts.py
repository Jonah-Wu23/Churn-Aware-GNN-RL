from src.utils.reward_hack_alerts import RewardHackAlertConfig, RewardHackDetector


def _make_payload(action_mask_valid_count: int, epsilon: float, entropy: float) -> dict:
    return {
        "reward_total": 0.0,
        "step_served": 0.0,
        "epsilon": float(epsilon),
        "q_entropy_norm": float(entropy),
        "action_mask_valid_count": int(action_mask_valid_count),
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
