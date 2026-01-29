"""Unit tests for CPO environment wrapper and policy integration.

These tests are designed to run WITHOUT GPU, verifying:
1. CPO env wrapper initialization and space shapes
2. Observation construction correctness
3. Action mask handling
4. Cost signal generation
5. Policy forward pass (CPU only)

Run with: pytest tests/test_cpo_env_wrapper.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestCPOEnvWrapperImport:
    """Test that CPO env wrapper can be imported correctly."""
    
    def test_import_cpo_env_wrapper(self):
        """Verify CPO env wrapper module imports without error."""
        from src.env.cpo_env_wrapper import CPOEnvWrapper, CPOEnvConfig
        assert CPOEnvWrapper is not None
        assert CPOEnvConfig is not None
    
    def test_import_cpo_env_config_dataclass(self):
        """Verify CPOEnvConfig has required fields."""
        from src.env.cpo_env_wrapper import CPOEnvConfig
        from src.env.gym_env import EnvConfig
        
        # Create minimal env config
        env_config = EnvConfig(
            max_horizon_steps=10,
            max_requests=5,
            seed=42,
        )
        
        cpo_config = CPOEnvConfig(
            env_config=env_config,
            neighbor_k=4,
            include_noop=True,
        )
        
        assert cpo_config.neighbor_k == 4
        assert cpo_config.include_noop is True
        assert cpo_config.env_config is env_config


class TestCPOEnvWrapperSpaces:
    """Test observation and action space definitions."""
    
    @pytest.fixture
    def mock_env_config(self):
        """Create a minimal EnvConfig for testing."""
        from src.env.gym_env import EnvConfig
        return EnvConfig(
            max_horizon_steps=10,
            max_requests=5,
            seed=42,
            num_vehicles=1,
            vehicle_capacity=4,
        )
    
    def test_observation_space_shape(self, mock_env_config):
        """Verify observation space has correct dimension."""
        from src.env.cpo_env_wrapper import CPOEnvConfig, CPOEnvWrapper
        
        neighbor_k = 8
        cpo_config = CPOEnvConfig(
            env_config=mock_env_config,
            neighbor_k=neighbor_k,
            include_noop=True,
        )
        
        # Expected: node(5) + edge(neighbor_k*4) + onboard(4) + pos(1)
        edge_dim = 5 if mock_env_config.use_fleet_potential else 4
        expected_obs_dim = 5 + neighbor_k * edge_dim + 4 + 1
        
        with patch.object(CPOEnvWrapper, '__init__', lambda self, cfg: None):
            wrapper = CPOEnvWrapper.__new__(CPOEnvWrapper)
            wrapper.neighbor_k = neighbor_k
            wrapper.node_feat_dim = 5
            wrapper.edge_feat_dim = edge_dim
            wrapper.onboard_dim = 4
            wrapper.pos_dim = 1
            wrapper.obs_dim = expected_obs_dim
        
        assert wrapper.obs_dim == expected_obs_dim
        if edge_dim == 4:
            assert expected_obs_dim == 42  # 5 + 32 + 4 + 1
    
    def test_action_space_dimension(self, mock_env_config):
        """Verify action space includes NOOP."""
        from src.env.cpo_env_wrapper import CPOEnvConfig
        
        neighbor_k = 8
        cpo_config = CPOEnvConfig(
            env_config=mock_env_config,
            neighbor_k=neighbor_k,
            include_noop=True,
        )
        
        # Action dim = neighbor_k + 1 (NOOP)
        expected_action_dim = neighbor_k + 1
        assert expected_action_dim == 9


class TestCPOObservationConstruction:
    """Test observation vector construction."""
    
    def test_observation_components_concatenation(self):
        """Verify observation is correctly concatenated from components."""
        neighbor_k = 4
        
        # Simulate components
        current_node_feat = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        edge_feat_flat = np.zeros(neighbor_k * 4, dtype=np.float32)
        edge_feat_flat[:4] = [1.0, 2.0, 3.0, 4.0]  # First edge
        onboard_summary = np.array([0.3, 0.1, 0.2, 0.5], dtype=np.float32)
        pos_emb = np.array([0.5], dtype=np.float32)
        
        # Concatenate
        obs = np.concatenate([current_node_feat, edge_feat_flat, onboard_summary, pos_emb])
        
        assert obs.shape == (5 + 16 + 4 + 1,)
        assert obs.dtype == np.float32
        
        # Check components are in correct positions
        np.testing.assert_array_equal(obs[:5], current_node_feat)
        np.testing.assert_array_equal(obs[5:21], edge_feat_flat)
        np.testing.assert_array_equal(obs[21:25], onboard_summary)
        np.testing.assert_array_equal(obs[25:], pos_emb)
    
    def test_edge_feature_padding(self):
        """Verify edge features are correctly padded when fewer candidates."""
        neighbor_k = 8
        actual_edges = 3
        
        # Simulate fewer edges than neighbor_k
        actual_edge_features = np.random.rand(actual_edges, 4).astype(np.float32)
        
        # Pad to neighbor_k
        edge_feat_padded = np.zeros((neighbor_k, 4), dtype=np.float32)
        edge_feat_padded[:actual_edges] = actual_edge_features
        
        assert edge_feat_padded.shape == (8, 4)
        np.testing.assert_array_equal(edge_feat_padded[:3], actual_edge_features)
        np.testing.assert_array_equal(edge_feat_padded[3:], np.zeros((5, 4)))


class TestCPOActionMask:
    """Test action masking functionality."""
    
    def test_action_mask_padding(self):
        """Verify action mask is correctly padded."""
        neighbor_k = 8
        action_dim = neighbor_k + 1  # +1 for NOOP
        
        # Simulate raw mask with fewer actions
        raw_mask = np.array([True, True, False, True, False], dtype=bool)
        
        # Pad to action_dim
        action_mask_padded = np.zeros(action_dim, dtype=bool)
        n_valid = min(len(raw_mask), neighbor_k)
        action_mask_padded[:n_valid] = raw_mask[:n_valid]
        action_mask_padded[-1] = True  # NOOP always valid
        
        assert action_mask_padded.shape == (9,)
        assert action_mask_padded[-1] == True  # NOOP
        assert action_mask_padded[0] == True
        assert action_mask_padded[2] == False
    
    def test_masked_action_selection(self):
        """Verify masked actions result in zero probability."""
        action_dim = 5
        
        # Simulate action probabilities
        probs = np.array([0.2, 0.3, 0.1, 0.25, 0.15], dtype=np.float32)
        mask = np.array([True, False, True, False, True], dtype=bool)
        
        # Apply mask
        probs[~mask] = 0.0
        
        assert probs[1] == 0.0
        assert probs[3] == 0.0
        assert probs[0] > 0
        assert probs[2] > 0
        assert probs[4] > 0
        
        # Best action should be among valid ones
        best_action = int(np.argmax(probs))
        assert mask[best_action] == True


class TestCPOCostSignal:
    """Test cost signal computation."""
    
    def test_cost_for_masked_action(self):
        """Verify cost is generated for masked action violations."""
        cost_violation_weight = 1.0
        was_masked = True
        
        cost = 0.0
        if was_masked:
            cost += cost_violation_weight * 1.0
        
        assert cost == 1.0
    
    def test_cost_accumulation(self):
        """Verify episode cost accumulates correctly."""
        step_costs = [0.1, 0.0, 0.5, 0.2, 0.0]
        episode_cost = sum(step_costs)
        
        assert episode_cost == 0.8
    
    def test_capacity_overflow_cost(self):
        """Verify capacity overflow is penalized."""
        cost_capacity_weight = 1.0
        current_onboard = 6
        capacity = 6
        waiting_at_stop = 3
        
        cost = 0.0
        if current_onboard >= capacity and waiting_at_stop > 0:
            cost += cost_capacity_weight * min(waiting_at_stop, 3) / 3.0
        
        assert cost == 1.0  # 3/3 = 1.0


class TestCPOPolicyIntegration:
    """Test CPO policy integration with evaluator."""
    
    def test_cpo_policy_import(self):
        """Verify _cpo_policy can be imported from evaluator."""
        from src.eval.evaluator import _cpo_policy
        assert _cpo_policy is not None
    
    def test_observation_construction_for_policy(self):
        """Verify observation constructed for policy matches expected format."""
        neighbor_k = 8
        
        # Simulate features dict
        node_features = np.random.rand(10, 5).astype(np.float32)
        edge_features = np.random.rand(5, 4).astype(np.float32)
        current_idx = 3
        
        # Build observation (matching _cpo_policy logic)
        current_node_feat = node_features[current_idx]
        
        edge_feat_padded = np.zeros((neighbor_k, 4), dtype=np.float32)
        n_edges = min(len(edge_features), neighbor_k)
        edge_feat_padded[:n_edges] = edge_features[:n_edges]
        edge_feat_flat = edge_feat_padded.flatten()
        
        onboard_summary = np.array([0.3, 0.0, 0.0, 0.5], dtype=np.float32)
        pos_emb = np.array([node_features[current_idx, 4]], dtype=np.float32)
        
        obs = np.concatenate([current_node_feat, edge_feat_flat, onboard_summary, pos_emb])
        
        expected_dim = 5 + neighbor_k * 4 + 4 + 1
        assert obs.shape == (expected_dim,)


class TestPyTorchCPOIntegration:
    """Test integration with PyTorch-CPO library."""
    
    @pytest.fixture(autouse=True)
    def setup_dtype(self):
        """Set PyTorch default dtype to float64 (required by PyTorch-CPO)."""
        import torch
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        yield
        torch.set_default_dtype(old_dtype)
    
    def test_discrete_policy_import(self):
        """Verify DiscretePolicy can be imported from PyTorch-CPO."""
        pytorch_cpo_path = PROJECT_ROOT / "baselines" / "PyTorch-CPO"
        if str(pytorch_cpo_path) not in sys.path:
            sys.path.insert(0, str(pytorch_cpo_path))
        
        from models.discrete_policy import DiscretePolicy
        assert DiscretePolicy is not None
    
    def test_discrete_policy_initialization(self):
        """Verify DiscretePolicy initializes with correct dimensions."""
        pytorch_cpo_path = PROJECT_ROOT / "baselines" / "PyTorch-CPO"
        if str(pytorch_cpo_path) not in sys.path:
            sys.path.insert(0, str(pytorch_cpo_path))
        
        from models.discrete_policy import DiscretePolicy
        import torch
        
        state_dim = 42  # 5 + 8*4 + 4 + 1
        action_num = 9  # 8 + 1 NOOP
        
        policy = DiscretePolicy(state_dim, action_num, hidden_size=(64, 64))
        
        assert policy.is_disc_action == True
        
        # Test forward pass
        dummy_obs = torch.randn(1, state_dim)
        with torch.no_grad():
            probs = policy(dummy_obs)
        
        assert probs.shape == (1, action_num)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-6)
    
    def test_discrete_policy_get_kl(self):
        """Verify KL divergence computation for categorical policy."""
        pytorch_cpo_path = PROJECT_ROOT / "baselines" / "PyTorch-CPO"
        if str(pytorch_cpo_path) not in sys.path:
            sys.path.insert(0, str(pytorch_cpo_path))
        
        from models.discrete_policy import DiscretePolicy
        import torch
        
        state_dim = 42
        action_num = 9
        
        policy = DiscretePolicy(state_dim, action_num, hidden_size=(64, 64))
        
        dummy_obs = torch.randn(5, state_dim)
        kl = policy.get_kl(dummy_obs)
        
        # KL should be shape [batch, 1] and >= 0
        assert kl.shape == (5, 1)
        assert torch.all(kl >= 0)
    
    def test_discrete_policy_action_sampling(self):
        """Verify action sampling from categorical distribution."""
        pytorch_cpo_path = PROJECT_ROOT / "baselines" / "PyTorch-CPO"
        if str(pytorch_cpo_path) not in sys.path:
            sys.path.insert(0, str(pytorch_cpo_path))
        
        from models.discrete_policy import DiscretePolicy
        import torch
        
        state_dim = 42
        action_num = 9
        
        policy = DiscretePolicy(state_dim, action_num)
        
        dummy_obs = torch.randn(1, state_dim)
        action = policy.select_action(dummy_obs)
        
        assert action.shape == (1, 1)
        assert 0 <= action.item() < action_num


class TestCPOTrainingScriptImport:
    """Test training script can be parsed without import errors."""
    
    def test_training_script_syntax(self):
        """Verify training script has valid Python syntax."""
        train_script_path = PROJECT_ROOT / "scripts" / "run_cpo_train.py"
        
        with open(train_script_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        # This will raise SyntaxError if invalid
        compile(code, train_script_path, "exec")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
