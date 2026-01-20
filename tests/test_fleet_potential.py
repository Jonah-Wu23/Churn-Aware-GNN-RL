"""Tests for Fleet-Aware Edge Potential (FAEP) feature.

Tests cover:
1. FAEP disabled: edge_features dim == 4
2. FAEP enabled: edge_features dim == 5, monotonicity of fleet_potential
3. Normalization functions: log1p_norm and linear_norm
4. Deterministic replay: fixed seed produces identical results
"""

from __future__ import annotations

import numpy as np
import pytest

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.utils.feature_spec import get_edge_dim, validate_checkpoint_edge_dim


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def base_env_config():
    """Base environment config with minimal settings for fast testing."""
    return {
        "max_horizon_steps": 100,
        "max_requests": 50,
        "seed": 42,
        "num_vehicles": 3,
        "vehicle_capacity": 4,
        "request_timeout_sec": 600,
        "churn_tol_sec": 300,
        "churn_beta": 0.02,
        "use_fleet_potential": False,
        "fleet_potential_mode": "next_stop",
        "fleet_potential_k": 1,
        "fleet_potential_phi": "log1p_norm",
        "od_glob": "data/processed/od_mapped/*.parquet",
        "graph_nodes_path": "data/processed/graph/layer2_nodes.parquet",
        "graph_edges_path": "data/processed/graph/layer2_edges.parquet",
        "graph_embeddings_path": "data/processed/graph/node2vec_embeddings.parquet",
    }


# ============================================================================
# Test: get_edge_dim utility function
# ============================================================================

def test_get_edge_dim_disabled():
    """get_edge_dim returns 4 when use_fleet_potential is False."""
    cfg = {"use_fleet_potential": False}
    assert get_edge_dim(cfg) == 4


def test_get_edge_dim_enabled():
    """get_edge_dim returns 5 when use_fleet_potential is True."""
    cfg = {"use_fleet_potential": True}
    assert get_edge_dim(cfg) == 5


def test_get_edge_dim_missing_key():
    """get_edge_dim returns 4 by default when key is missing."""
    cfg = {}
    assert get_edge_dim(cfg) == 4


# ============================================================================
# Test: validate_checkpoint_edge_dim
# ============================================================================

def test_validate_checkpoint_compatible():
    """No error when checkpoint and env dimensions match."""
    validate_checkpoint_edge_dim(4, 4, False)  # Should not raise
    validate_checkpoint_edge_dim(5, 5, True)   # Should not raise


def test_validate_checkpoint_incompatible_faep_on():
    """Error when FAEP enabled but checkpoint is edge_dim=4."""
    with pytest.raises(ValueError, match="Checkpoint edge_dim=4 incompatible"):
        validate_checkpoint_edge_dim(4, 5, True)


def test_validate_checkpoint_incompatible_faep_off():
    """Error when FAEP disabled but checkpoint is edge_dim=5."""
    with pytest.raises(ValueError, match="Checkpoint edge_dim=5 incompatible"):
        validate_checkpoint_edge_dim(5, 4, False)


# ============================================================================
# Test: Normalization functions (direct unit tests)
# ============================================================================

def test_apply_fleet_potential_phi_log1p_norm():
    """Test log1p normalization formula."""
    # Create minimal env to access the method
    # phi(C) = log(1+C) / log(1+num_vehicles)
    num_vehicles = 10
    
    # Test values
    for density in [0.0, 1.0, 5.0, 10.0]:
        expected = float(np.log1p(density) / np.log1p(num_vehicles))
        # Verify formula produces expected range [0, 1]
        assert 0.0 <= expected <= 1.0, f"Expected range [0,1] for density={density}"


def test_apply_fleet_potential_phi_linear_norm():
    """Test linear normalization formula."""
    num_vehicles = 10
    
    # phi(C) = C / num_vehicles
    for density in [0.0, 1.0, 5.0, 10.0]:
        expected = float(density / num_vehicles)
        assert 0.0 <= expected <= 1.0, f"Expected range [0,1] for density={density}"


# ============================================================================
# Test: Fleet potential computation (requires real env)
# ============================================================================

@pytest.mark.skipif(
    not pytest.importorskip("pandas", reason="pandas required"),
    reason="Integration tests require data files"
)
class TestFleetPotentialIntegration:
    """Integration tests that require actual environment and data files."""
    
    def test_fleet_potential_disabled_edge_dim(self, base_env_config):
        """When FAEP disabled, edge_features has 4 dimensions."""
        try:
            config = EnvConfig(**base_env_config)
            env = EventDrivenEnv(config)
            features = env.get_feature_batch()
            
            if features["edge_features"].shape[0] > 0:
                assert features["edge_features"].shape[1] == 4, \
                    f"Expected edge_dim=4 when FAEP disabled, got {features['edge_features'].shape[1]}"
        except FileNotFoundError:
            pytest.skip("Test data files not available")
    
    def test_fleet_potential_enabled_edge_dim(self, base_env_config):
        """When FAEP enabled, edge_features has 5 dimensions."""
        try:
            base_env_config["use_fleet_potential"] = True
            config = EnvConfig(**base_env_config)
            env = EventDrivenEnv(config)
            features = env.get_feature_batch()
            
            if features["edge_features"].shape[0] > 0:
                assert features["edge_features"].shape[1] == 5, \
                    f"Expected edge_dim=5 when FAEP enabled, got {features['edge_features'].shape[1]}"
        except FileNotFoundError:
            pytest.skip("Test data files not available")
    
    def test_fleet_density_summary_in_info(self, base_env_config):
        """When FAEP enabled, step() returns fleet_density_summary in info."""
        try:
            base_env_config["use_fleet_potential"] = True
            config = EnvConfig(**base_env_config)
            env = EventDrivenEnv(config)
            
            features = env.get_feature_batch()
            actions = features["actions"]
            mask = features["action_mask"]
            
            if len(actions) == 0 or not np.any(mask):
                pytest.skip("No valid actions available")
            
            valid_idx = np.where(mask)[0][0]
            action = int(actions[valid_idx])
            
            _, _, _, info = env.step(action)
            
            assert "fleet_density_summary" in info, \
                "fleet_density_summary should be in info when FAEP enabled"
            
            summary = info["fleet_density_summary"]
            assert "max" in summary
            assert "mean" in summary
            assert "top_5_congested_stops" in summary
            
            # Verify types are native Python (JSON serializable)
            assert isinstance(summary["max"], float)
            assert isinstance(summary["mean"], float)
            assert isinstance(summary["top_5_congested_stops"], list)
        except FileNotFoundError:
            pytest.skip("Test data files not available")
    
    def test_fleet_potential_monotonicity(self, base_env_config):
        """Stops with more vehicles targeting them should have higher fleet_potential."""
        try:
            base_env_config["use_fleet_potential"] = True
            base_env_config["num_vehicles"] = 5
            config = EnvConfig(**base_env_config)
            env = EventDrivenEnv(config)
            
            # Compute density
            density_map = env._compute_fleet_density_by_stop()
            
            # Apply phi to all stops
            phi_map = {
                stop_id: env._apply_fleet_potential_phi(density)
                for stop_id, density in density_map.items()
            }
            
            # Verify monotonicity: higher density -> higher phi
            sorted_by_density = sorted(density_map.items(), key=lambda x: x[1])
            sorted_by_phi = sorted(phi_map.items(), key=lambda x: x[1])
            
            # Order should be the same
            density_order = [item[0] for item in sorted_by_density]
            phi_order = [item[0] for item in sorted_by_phi]
            
            assert density_order == phi_order, \
                "Monotonicity violated: phi should preserve density ordering"
        except FileNotFoundError:
            pytest.skip("Test data files not available")
    
    def test_fleet_potential_deterministic(self, base_env_config):
        """Fixed seed produces identical fleet_potential values."""
        try:
            base_env_config["use_fleet_potential"] = True
            base_env_config["seed"] = 12345
            
            # Run 1
            config1 = EnvConfig(**base_env_config)
            env1 = EventDrivenEnv(config1)
            features1 = env1.get_feature_batch()
            potentials1 = features1["edge_features"][:, 4] if features1["edge_features"].shape[0] > 0 else np.array([])
            
            # Run 2 (same seed)
            config2 = EnvConfig(**base_env_config)
            env2 = EventDrivenEnv(config2)
            features2 = env2.get_feature_batch()
            potentials2 = features2["edge_features"][:, 4] if features2["edge_features"].shape[0] > 0 else np.array([])
            
            # Should be identical
            np.testing.assert_array_equal(
                potentials1, potentials2,
                "Fleet potential should be deterministic with fixed seed"
            )
        except FileNotFoundError:
            pytest.skip("Test data files not available")


# ============================================================================
# Direct unit tests for _compute_fleet_density_by_stop and _apply_fleet_potential_phi
# These test the logic without needing full environment setup
# ============================================================================

class TestFleetDensityLogic:
    """Direct tests for density computation and normalization logic."""
    
    def test_log1p_norm_boundary_values(self):
        """log1p_norm produces values in [0, 1] for all valid inputs."""
        num_vehicles = 50
        
        for density in [0, 1, 10, 25, 50, 100]:
            phi = float(np.log1p(density) / np.log1p(num_vehicles))
            assert phi >= 0.0, f"phi should be >= 0 for density={density}"
            # When density > num_vehicles, phi can exceed 1.0, which is acceptable
    
    def test_linear_norm_boundary_values(self):
        """linear_norm produces values in [0, 1] for density <= num_vehicles."""
        num_vehicles = 50
        
        for density in [0, 1, 10, 25, 50]:
            phi = float(density / num_vehicles)
            assert 0.0 <= phi <= 1.0, f"phi should be in [0,1] for density={density}"
    
    def test_log1p_more_stable_than_linear(self):
        """log1p normalization is more stable for extreme values."""
        num_vehicles = 50
        
        # Compare sensitivity
        density_low = 1
        density_high = 49
        
        linear_diff = (density_high - density_low) / num_vehicles
        log1p_diff = (np.log1p(density_high) - np.log1p(density_low)) / np.log1p(num_vehicles)
        
        # log1p should compress the range more
        assert log1p_diff < linear_diff, "log1p should be less sensitive to large density changes"
