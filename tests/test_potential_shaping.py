import pytest

from src.env.gym_env import EventDrivenEnv


def test_potential_shaping_alpha_zero() -> None:
    result = EventDrivenEnv.compute_potential_shaping(
        0.0,
        phi_before=10.0,
        phi_after=3.0,
        reward_scale=2.0,
        scale_with_reward_scale=True,
    )
    assert result["reward_potential_shaping_raw"] == 0.0
    assert result["reward_potential_shaping"] == 0.0


def test_potential_shaping_math() -> None:
    result = EventDrivenEnv.compute_potential_shaping(
        0.3,
        phi_before=10.0,
        phi_after=4.0,
        reward_scale=2.0,
        scale_with_reward_scale=True,
    )
    assert result["phi_delta"] == 6.0
    assert result["reward_potential_shaping_raw"] == pytest.approx(1.8)
    assert result["reward_potential_shaping"] == pytest.approx(3.6)


def test_potential_shaping_keys_present() -> None:
    result = EventDrivenEnv.compute_potential_shaping(
        0.1,
        phi_before=2.0,
        phi_after=1.5,
        reward_scale=1.0,
        scale_with_reward_scale=False,
    )
    for key in ("phi_delta", "reward_potential_shaping_raw", "reward_potential_shaping"):
        assert key in result
