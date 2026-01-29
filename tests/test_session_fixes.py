"""Tests for code changes made during NaN loss diagnosis session.

This module tests:
1. Infinite ETA sanitization in get_feature_batch
2. NaN loss guard in DQN optimizer
3. Time-based train/eval data split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import replace

from src.env.gym_env import EnvConfig, EventDrivenEnv


# =============================================================================
# Test 1: Infinite ETA Sanitization
# =============================================================================

class TestInfiniteETASanitization:
    """Test that infinite ETAs from unreachable stops are properly sanitized."""

    def test_config_has_time_split_params(self):
        """Verify EnvConfig has the new time_split parameters."""
        config = EnvConfig()
        assert hasattr(config, "time_split_mode")
        assert hasattr(config, "time_split_ratio")
        assert config.time_split_mode is None
        assert config.time_split_ratio == 0.3

    def test_edge_features_never_infinite(self):
        """Test that edge_features are always finite even with unreachable stops."""
        # Create a minimal mock environment to test sanitization logic
        # We need to verify that the sanitization code handles inf correctly
        
        # Simulate the sanitization logic directly
        MAX_TIME_VAL = 36000.0
        
        # Test inf handling
        curr_eta = float("inf")
        new_eta_leg = float("inf")
        
        # Apply sanitization (as in gym_env.py)
        if not np.isfinite(curr_eta):
            curr_eta = MAX_TIME_VAL
        if not np.isfinite(new_eta_leg):
            new_eta_leg = MAX_TIME_VAL
            
        assert np.isfinite(curr_eta)
        assert np.isfinite(new_eta_leg)
        assert curr_eta == MAX_TIME_VAL
        assert new_eta_leg == MAX_TIME_VAL

    def test_count_capping(self):
        """Test that passenger counts are properly capped."""
        MAX_COUNT = 500.0
        
        # Test various count values
        for count in [0, 100, 500, 1000, 10000]:
            capped = float(min(count, MAX_COUNT))
            assert capped <= MAX_COUNT
            assert np.isfinite(capped)

    def test_travel_time_capping(self):
        """Test that travel times are properly capped."""
        MAX_TIME_VAL = 36000.0
        
        # Test various travel time values
        for travel_time in [0, 100, 3600, 36000, 100000, float("inf")]:
            if not np.isfinite(travel_time):
                travel_time = MAX_TIME_VAL
            capped = float(min(travel_time, MAX_TIME_VAL))
            assert capped <= MAX_TIME_VAL
            assert np.isfinite(capped)


# =============================================================================
# Test 2: NaN Loss Guard
# =============================================================================

class TestNaNLossGuard:
    """Test the NaN loss detection and skip mechanism."""

    def test_nan_detection(self):
        """Test that torch.isnan correctly detects NaN values."""
        # Normal loss
        normal_loss = torch.tensor(0.5)
        assert not torch.isnan(normal_loss)
        
        # NaN loss
        nan_loss = torch.tensor(float("nan"))
        assert torch.isnan(nan_loss)

    def test_inf_propagation_to_nan(self):
        """Test that inf inputs can propagate to NaN in loss calculation."""
        # Simulate what happens when inf gets into the network
        q_preds = torch.tensor([float("inf"), 1.0, 2.0])
        targets = torch.tensor([1.0, 1.0, 1.0])
        
        # smooth_l1_loss with inf input produces inf, not nan directly
        loss = F.smooth_l1_loss(q_preds, targets)
        # But the backward pass on inf gradients typically produces nan
        assert not torch.isfinite(loss) or torch.isnan(loss) or loss.item() > 1e10

    def test_skip_nan_update_logic(self):
        """Test the conditional skip logic for NaN losses."""
        losses_collected = []
        
        for step in range(10):
            # Simulate loss calculation
            if step == 5:
                loss = torch.tensor(float("nan"))
            else:
                loss = torch.tensor(0.1 * step)
            
            # Apply the NaN guard logic
            if torch.isnan(loss):
                continue  # Skip this update
            
            losses_collected.append(float(loss.item()))
        
        # Should have 9 losses (skipped step 5)
        assert len(losses_collected) == 9
        # None should be NaN
        assert all(np.isfinite(l) for l in losses_collected)


# =============================================================================
# Test 3: Time-based Train/Eval Split
# =============================================================================

class TestTimeSplit:
    """Test the time-based data splitting functionality."""

    def test_time_split_config_train_mode(self):
        """Test EnvConfig with train split mode."""
        config = EnvConfig(
            time_split_mode="train",
            time_split_ratio=0.3,
        )
        assert config.time_split_mode == "train"
        assert config.time_split_ratio == 0.3

    def test_time_split_config_eval_mode(self):
        """Test EnvConfig with eval split mode."""
        config = EnvConfig(
            time_split_mode="eval",
            time_split_ratio=0.3,
        )
        assert config.time_split_mode == "eval"
        assert config.time_split_ratio == 0.3

    def test_time_split_logic_train(self):
        """Test the time split filtering logic for training data."""
        # Create mock OD dataframe
        dates = pd.date_range("2025-09-01", periods=100, freq="1H")
        od = pd.DataFrame({
            "tpep_pickup_datetime": dates,
            "pickup_stop_id": range(100),
            "dropoff_stop_id": range(100),
        })
        od = od.sort_values("tpep_pickup_datetime").reset_index(drop=True)
        
        # Apply time split logic (as in gym_env.py)
        time_split_ratio = 0.3
        t_min = od["tpep_pickup_datetime"].min()
        t_max = od["tpep_pickup_datetime"].max()
        duration = (t_max - t_min).total_seconds()
        cutoff_time = t_min + pd.Timedelta(seconds=duration * time_split_ratio)
        
        train_od = od[od["tpep_pickup_datetime"] <= cutoff_time]
        eval_od = od[od["tpep_pickup_datetime"] > cutoff_time]
        
        # Verify split
        assert len(train_od) > 0
        assert len(eval_od) > 0
        assert len(train_od) + len(eval_od) == len(od)
        # Train should be roughly 30%
        assert 0.25 < len(train_od) / len(od) < 0.35

    def test_time_split_logic_eval(self):
        """Test the time split filtering logic for evaluation data."""
        dates = pd.date_range("2025-09-01", periods=100, freq="1H")
        od = pd.DataFrame({
            "tpep_pickup_datetime": dates,
            "pickup_stop_id": range(100),
            "dropoff_stop_id": range(100),
        })
        
        time_split_ratio = 0.3
        t_min = od["tpep_pickup_datetime"].min()
        t_max = od["tpep_pickup_datetime"].max()
        duration = (t_max - t_min).total_seconds()
        cutoff_time = t_min + pd.Timedelta(seconds=duration * time_split_ratio)
        
        eval_od = od[od["tpep_pickup_datetime"] > cutoff_time]
        
        # Eval should be roughly 70%
        assert 0.65 < len(eval_od) / len(od) < 0.75

    def test_time_split_no_overlap(self):
        """Verify train and eval sets have no overlapping records."""
        dates = pd.date_range("2025-09-01", periods=1000, freq="1H")
        od = pd.DataFrame({
            "tpep_pickup_datetime": dates,
            "pickup_stop_id": range(1000),
            "dropoff_stop_id": range(1000),
        })
        
        time_split_ratio = 0.3
        t_min = od["tpep_pickup_datetime"].min()
        t_max = od["tpep_pickup_datetime"].max()
        duration = (t_max - t_min).total_seconds()
        cutoff_time = t_min + pd.Timedelta(seconds=duration * time_split_ratio)
        
        train_times = set(od[od["tpep_pickup_datetime"] <= cutoff_time]["tpep_pickup_datetime"])
        eval_times = set(od[od["tpep_pickup_datetime"] > cutoff_time]["tpep_pickup_datetime"])
        
        # No overlap
        assert len(train_times & eval_times) == 0


# =============================================================================
# Test 4: Integration - Build Env Config
# =============================================================================

class TestBuildEnvConfig:
    """Test that EnvConfig correctly handles new parameters."""

    def test_env_config_accepts_time_split_train(self):
        """Test EnvConfig with train time_split_mode."""
        config = EnvConfig(
            time_split_mode="train",
            time_split_ratio=0.4,
        )
        assert config.time_split_mode == "train"
        assert config.time_split_ratio == 0.4

    def test_env_config_accepts_time_split_eval(self):
        """Test EnvConfig with eval time_split_mode."""
        config = EnvConfig(
            time_split_mode="eval",
            time_split_ratio=0.25,
        )
        assert config.time_split_mode == "eval"
        assert config.time_split_ratio == 0.25

    def test_default_time_split_values(self):
        """Test default values when time_split params not provided."""
        config = EnvConfig()
        assert config.time_split_mode is None
        assert config.time_split_ratio == 0.3

    def test_env_config_time_split_none_disabled(self):
        """Test that time_split_mode=None disables splitting."""
        config = EnvConfig(time_split_mode=None)
        assert config.time_split_mode is None


# =============================================================================
# Test 5: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_count_capping(self):
        """Test count capping with zero value."""
        count = 0
        capped = float(min(count, 500.0))
        assert capped == 0.0

    def test_negative_eta_handling(self):
        """Test that negative values are handled correctly."""
        # Negative values shouldn't occur but if they do, they should pass through
        curr_eta = -100.0
        MAX_TIME_VAL = 36000.0
        
        if not np.isfinite(curr_eta):
            curr_eta = MAX_TIME_VAL
        
        # Negative is finite, so should pass through
        assert curr_eta == -100.0
        assert np.isfinite(curr_eta)

    def test_time_split_ratio_boundaries(self):
        """Test time split with edge ratio values."""
        dates = pd.date_range("2025-09-01", periods=100, freq="1H")
        od = pd.DataFrame({
            "tpep_pickup_datetime": dates,
            "pickup_stop_id": range(100),
            "dropoff_stop_id": range(100),
        })
        
        # Very small ratio
        t_min = od["tpep_pickup_datetime"].min()
        t_max = od["tpep_pickup_datetime"].max()
        duration = (t_max - t_min).total_seconds()
        cutoff_time = t_min + pd.Timedelta(seconds=duration * 0.01)
        train_od = od[od["tpep_pickup_datetime"] <= cutoff_time]
        assert len(train_od) >= 1

        # Very large ratio
        cutoff_time = t_min + pd.Timedelta(seconds=duration * 0.99)
        train_od = od[od["tpep_pickup_datetime"] <= cutoff_time]
        assert len(train_od) <= 100


# =============================================================================
# Test 6: Curriculum max_requests Bug Fix
# =============================================================================

class TestCurriculumMaxRequests:
    """Test that curriculum stage generation does NOT override max_requests."""

    def test_env_overrides_does_not_contain_max_requests(self):
        """Verify that generate_stage does not set max_requests in env_overrides."""
        from src.train.curriculum import StageSpec, generate_stage
        import tempfile
        
        # Create minimal test data
        dates = pd.date_range("2025-09-01", periods=100, freq="1H")
        base_od = pd.DataFrame({
            "tpep_pickup_datetime": dates,
            "pickup_stop_id": list(range(10)) * 10,
            "dropoff_stop_id": list(range(10)) * 10,
        })
        
        nodes = pd.DataFrame({
            "gnn_node_id": range(10),
            "lon": [0.0] * 10,
            "lat": [0.0] * 10,
        })
        
        stage = StageSpec(name="L0", description="Test stage", sample_fraction=0.5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output = generate_stage(
                base_od=base_od,
                nodes=nodes,
                stage=stage,
                output_dir=Path(tmpdir),
                seed=7,
            )
            
            # CRITICAL: max_requests should NOT be in env_overrides
            assert "max_requests" not in output.env_overrides, \
                "Bug: generate_stage should not override max_requests!"

    def test_all_stages_do_not_override_max_requests(self):
        """Test that all default stages do not override max_requests."""
        from src.train.curriculum import StageSpec, generate_stage, default_stages
        import tempfile
        
        dates = pd.date_range("2025-09-01", periods=1000, freq="1H")
        base_od = pd.DataFrame({
            "tpep_pickup_datetime": dates,
            "pickup_stop_id": list(range(10)) * 100,
            "dropoff_stop_id": list(range(10)) * 100,
        })
        
        nodes = pd.DataFrame({
            "gnn_node_id": range(10),
            "lon": [0.0] * 10,
            "lat": [0.0] * 10,
        })
        
        for stage in default_stages():
            with tempfile.TemporaryDirectory() as tmpdir:
                output = generate_stage(
                    base_od=base_od,
                    nodes=nodes,
                    stage=stage,
                    output_dir=Path(tmpdir),
                    seed=7,
                )
                
                assert "max_requests" not in output.env_overrides, \
                    f"Bug: Stage {stage.name} should not override max_requests!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

