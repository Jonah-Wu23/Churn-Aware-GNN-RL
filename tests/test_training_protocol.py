"""Unit tests for the EdgeQ training protocol.

Tests cover:
- Epsilon continuity across phases
- Rho-gated stage transitions
- Reward ramp linear interpolation
- Phase3 empty overrides inheritance
- Checkpoint state restoration
"""

from __future__ import annotations

import numpy as np
import pytest

from src.train.dqn import DQNConfig, _linear_schedule
from src.train.reward_ramp import (
    RampConfig, 
    compute_ramped_weights, 
    build_ramp_config, 
    get_phase3_target_weights,
    DEFAULT_RAMP_FIELDS,
)
from src.train.runner import (
    CurriculumConfig, 
    _compute_rho, 
    _compute_rho_window_mean,
    _compute_service_rate,
)


class TestEpsilonContinuity:
    """Tests for epsilon continuity across phases."""
    
    def test_linear_schedule_basic(self):
        """Test basic epsilon linear schedule."""
        eps = _linear_schedule(1.0, 0.1, 100, 50)
        assert 0.5 < eps < 0.6
    
    def test_linear_schedule_at_boundaries(self):
        """Test epsilon at schedule boundaries."""
        assert _linear_schedule(1.0, 0.1, 100, 0) == 1.0
        assert _linear_schedule(1.0, 0.1, 100, 100) == 0.1
        assert _linear_schedule(1.0, 0.1, 100, 200) == 0.1  # clamp
    
    def test_epsilon_offset_continuity(self):
        """Test that epsilon_offset produces continuous epsilon across phases."""
        # Phase1: steps 0-50
        eps_phase1_end = _linear_schedule(1.0, 0.1, 100, 50)
        
        # Phase2: starts at step 50 with offset=50
        # At local step 0, effective step = 0 + 50 = 50
        eps_phase2_start = _linear_schedule(1.0, 0.1, 100, 0 + 50)
        
        assert abs(eps_phase1_end - eps_phase2_start) < 1e-6, \
            f"Epsilon should be continuous: {eps_phase1_end} vs {eps_phase2_start}"
    
    def test_global_step_consistency(self):
        """Test that global_step produces consistent epsilon across full training."""
        # Simulate training with global_step
        global_step = 0
        eps_history = []
        
        for phase in ["phase1", "phase2", "phase3"]:
            for _ in range(10):
                global_step += 1
                eps = _linear_schedule(1.0, 0.05, 100, global_step)
                eps_history.append(eps)
        
        # Epsilon should monotonically decrease
        for i in range(1, len(eps_history)):
            assert eps_history[i] <= eps_history[i-1], \
                f"Epsilon should not increase: {eps_history[i-1]} -> {eps_history[i]}"


class TestRhoGatedTransition:
    """Tests for rho-gated stage transitions."""
    
    def test_compute_service_rate(self):
        """Test service rate computation."""
        log = {"served": 80, "waiting_churned": 10, "onboard_churned": 5, "waiting_timeouts": 5}
        rate = _compute_service_rate(log)
        assert abs(rate - 0.8) < 1e-6
    
    def test_compute_rho(self):
        """Test rho computation with stuckness penalty."""
        log = {"served": 80, "waiting_churned": 10, "onboard_churned": 5, 
               "waiting_timeouts": 5, "stuckness": 0.1}
        rho = _compute_rho(log, gamma=1.0)
        expected = 0.8 / (1.0 + 0.1)
        assert abs(rho - expected) < 1e-6
    
    def test_rho_window_mean(self):
        """Test rho window mean calculation."""
        history = [0.3, 0.4, 0.5, 0.6, 0.7]
        mean = _compute_rho_window_mean(history, window_size=3)
        assert abs(mean - 0.6) < 1e-6  # (0.5 + 0.6 + 0.7) / 3
    
    def test_rho_window_mean_small_history(self):
        """Test rho window mean with fewer episodes than window size."""
        history = [0.3, 0.4]
        mean = _compute_rho_window_mean(history, window_size=5)
        assert abs(mean - 0.35) < 1e-6  # Uses all available
    
    def test_rho_window_mean_empty(self):
        """Test rho window mean with empty history."""
        assert _compute_rho_window_mean([], window_size=5) == 0.0


class TestRewardRamp:
    """Tests for reward weight linear interpolation."""
    
    def test_ramp_alpha_monotonic(self):
        """Test that alpha monotonically increases."""
        ramp_config = RampConfig(
            reward_ramp_steps=100,
            w2={"reward_service": 4.0, "reward_fairness_weight": 0.5},
            w3_target={"reward_service": 3.0, "reward_fairness_weight": 1.0},
        )
        
        prev_alpha = -1.0
        for step in range(0, 150, 10):
            _, alpha = compute_ramped_weights(step, ramp_config)
            assert alpha >= prev_alpha, f"Alpha should not decrease: {prev_alpha} -> {alpha}"
            prev_alpha = alpha
        
        # Final alpha should be 1.0
        _, final_alpha = compute_ramped_weights(100, ramp_config)
        assert abs(final_alpha - 1.0) < 1e-6
    
    def test_ramp_clamp_nonnegative(self):
        """Test that ramped weights are clamped to non-negative."""
        ramp_config = RampConfig(
            reward_ramp_steps=100,
            w2={"reward_service": 1.0},
            w3_target={"reward_service": -0.5},  # Would go negative
        )
        
        ramped, _ = compute_ramped_weights(100, ramp_config)
        assert ramped["reward_service"] >= 0.0
    
    def test_ramp_interpolation_values(self):
        """Test correct interpolation at midpoint."""
        ramp_config = RampConfig(
            reward_ramp_steps=100,
            w2={"reward_service": 4.0},
            w3_target={"reward_service": 2.0},
        )
        
        ramped, alpha = compute_ramped_weights(50, ramp_config)
        assert abs(alpha - 0.5) < 1e-6
        assert abs(ramped["reward_service"] - 3.0) < 1e-6  # (1-0.5)*4 + 0.5*2 = 3


class TestPhase3Overrides:
    """Tests for phase3 empty overrides inheritance."""
    
    def test_empty_overrides_inherits_phase2(self):
        """Test that empty phase3 overrides inherits from phase2."""
        phase2 = {"reward_service": 4.0, "reward_fairness_weight": 0.5}
        phase3 = {}  # Empty!
        base = {"reward_service": 1.0, "reward_fairness_weight": 0.1}
        
        result = get_phase3_target_weights(phase2, phase3, base)
        
        # Should inherit phase2 values
        assert result["reward_service"] == 4.0
        assert result["reward_fairness_weight"] == 0.5
    
    def test_none_overrides_inherits_phase2(self):
        """Test that None phase3 overrides inherits from phase2."""
        phase2 = {"reward_service": 4.0}
        phase3 = None
        base = {"reward_service": 1.0}
        
        result = get_phase3_target_weights(phase2, phase3, base)
        assert result["reward_service"] == 4.0
    
    def test_explicit_overrides_used(self):
        """Test that explicit phase3 overrides are used."""
        phase2 = {"reward_service": 4.0}
        phase3 = {"reward_service": 5.0}  # Explicit
        base = {"reward_service": 1.0}
        
        result = get_phase3_target_weights(phase2, phase3, base)
        assert result["reward_service"] == 5.0


class TestCurriculumConfig:
    """Tests for CurriculumConfig defaults."""
    
    def test_default_values(self):
        """Test CurriculumConfig default values."""
        cfg = CurriculumConfig()
        
        assert cfg.trigger_rho == 0.5
        assert cfg.rho_window_size == 5
        assert cfg.require_rho_transition == True
        assert cfg.max_stage_extensions == 2
        assert cfg.fail_policy == "fail_fast"
        assert cfg.rho_warning_threshold == 0.35
        assert cfg.eval_enabled == True
    
    def test_warning_threshold_relationship(self):
        """Test that warning threshold is reasonable relative to trigger."""
        cfg = CurriculumConfig()
        assert cfg.rho_warning_threshold < cfg.trigger_rho


class TestServiceRateConsistency:
    """测试 service_rate 与 rho 的口径一致性。"""
    
    def test_service_rate_with_remaining(self):
        """测试含 remaining 的 service_rate 计算。"""
        from src.train.runner import _compute_service_rate, compute_eligible
        log = {
            "served": 33,
            "waiting_churned": 13,
            "onboard_churned": 0,
            "waiting_timeouts": 0,
            "waiting_remaining": 100,
            "onboard_remaining": 47,
        }
        rate = _compute_service_rate(log)
        eligible = compute_eligible(log)
        # 33 / (33+13+0+0+100+47) = 33/193 ≈ 0.171
        assert abs(eligible - 193) < 1e-6
        assert abs(rate - 33/193) < 1e-6
    
    def test_rho_equals_service_rate_when_stuckness_zero(self):
        """当 stuckness=0 时，rho 应等于 service_rate。"""
        log = {
            "served": 80,
            "waiting_churned": 10,
            "onboard_churned": 5,
            "waiting_timeouts": 5,
            "waiting_remaining": 0,
            "onboard_remaining": 0,
            "stuckness": 0.0,
        }
        service_rate = _compute_service_rate(log)
        rho = _compute_rho(log, gamma=1.0)
        assert abs(service_rate - rho) < 1e-6
    
    def test_service_rate_simple_excludes_remaining(self):
        """测试 service_rate_simple 不含 remaining。"""
        from src.train.runner import compute_service_rate_simple
        log = {
            "served": 33,
            "waiting_churned": 13,
            "onboard_churned": 0,
            "waiting_timeouts": 0,
            "waiting_remaining": 100,
            "onboard_remaining": 47,
        }
        rate_simple = compute_service_rate_simple(log)
        # 33 / (33+13+0+0) = 33/46 ≈ 0.717
        assert abs(rate_simple - 33/46) < 1e-6
    
    def test_consistency_check_service_rate_from_rho(self):
        """测试一致性校验派生量：service_rate == rho * (1 + gamma * stuckness)。"""
        log = {
            "served": 80,
            "waiting_churned": 10,
            "onboard_churned": 5,
            "waiting_timeouts": 5,
            "waiting_remaining": 0,
            "onboard_remaining": 0,
            "stuckness": 0.2,
        }
        service_rate = _compute_service_rate(log)
        rho = _compute_rho(log, gamma=1.0)
        stuckness = log["stuckness"]
        
        # 一致性校验
        service_rate_from_rho = rho * (1.0 + 1.0 * stuckness)
        assert abs(service_rate - service_rate_from_rho) < 1e-6
    
    def test_conservation_check_passes(self):
        """测试守恒校验：eligible + structural == total_requests。"""
        from src.train.runner import verify_request_conservation, compute_eligible
        log = {
            "served": 80,
            "waiting_churned": 10,
            "onboard_churned": 5,
            "waiting_timeouts": 5,
            "waiting_remaining": 10,
            "onboard_remaining": 5,
            "structural_unserviceable": 15,
            "total_requests": 130,  # 80+10+5+5+10+5+15 = 130
        }
        assert verify_request_conservation(log) == True
        assert abs(compute_eligible(log) + 15 - 130) < 1e-6
    
    def test_conservation_check_fails(self):
        """测试守恒校验失败情况。"""
        from src.train.runner import verify_request_conservation
        log = {
            "served": 80,
            "waiting_churned": 10,
            "onboard_churned": 5,
            "waiting_timeouts": 5,
            "waiting_remaining": 10,
            "onboard_remaining": 5,
            "structural_unserviceable": 15,
            "total_requests": 100,  # 故意设置错误值
        }
        assert verify_request_conservation(log) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
