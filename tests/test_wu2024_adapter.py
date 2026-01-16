"""Unit tests for Wu2024 adapter module.

Tests cover:
1. Model initialization and loading
2. Feature mapping correctness (dimension checks)
3. Policy output validity (action mask compliance)
4. Fixed seed reproducibility (action sequence consistency)
"""

import numpy as np
import pytest
import torch

from src.baselines.wu2024_adapter import (
    Wu2024PointerNet,
    load_wu2024_model,
    wu2024_policy,
    _build_static_features,
    _build_dynamic_features,
    _select_candidate_stops,
)


class MockEnv:
    """Minimal mock of EventDrivenEnv for testing."""
    
    def __init__(self):
        self.waiting = {0: [{"dropoff_stop_id": 2}], 1: [], 2: [{"dropoff_stop_id": 0}]}
        self.current_time = 1800.0  # 30 minutes
        self.stop_coords = {
            0: (-73.9, 40.7),
            1: (-73.85, 40.75),
            2: (-73.95, 40.65),
        }
        self._vehicle = MockVehicle()
        
    def _get_active_vehicle(self):
        return self._vehicle
    
    @property
    def config(self):
        return MockConfig()


class MockVehicle:
    def __init__(self):
        self.onboard = [{"id": 1}, {"id": 2}]  # 2 passengers onboard


class MockConfig:
    vehicle_capacity = 6


def _make_features(num_actions: int = 3) -> dict:
    """Create minimal feature dict for testing."""
    return {
        "actions": np.array([0, 1, 2][:num_actions], dtype=np.int64),
        "action_mask": np.array([True, True, False][:num_actions], dtype=bool),
        "node_features": np.random.randn(3, 5).astype(np.float32),
        "edge_features": np.array([
            [0.1, 0.2, 0.3, 120.0],  # travel_time = 120s
            [0.15, 0.25, 0.35, 180.0],
            [0.2, 0.3, 0.4, 240.0],
        ][:num_actions], dtype=np.float32),
        "current_node_index": np.array([0], dtype=np.int64),
    }


class TestModelInitialization:
    """Test model initialization and loading."""
    
    def test_pointer_net_init(self):
        """Test Wu2024PointerNet initializes correctly."""
        model = Wu2024PointerNet(
            static_size=3,
            dynamic_size=3,
            hidden_size=64,
            num_layers=1,
            dropout=0.1,
        )
        assert model is not None
        assert hasattr(model, 'static_encoder')
        assert hasattr(model, 'dynamic_encoder')
        assert hasattr(model, 'pointer')
    
    def test_load_model_random_init(self):
        """Test load_wu2024_model with no pretrained weights."""
        config = {"kmax": 32, "hidden_size": 64}
        device = torch.device("cpu")
        
        model, weights_mode = load_wu2024_model(None, config, device)
        
        assert model is not None
        assert weights_mode == "random_init"
        assert isinstance(model, Wu2024PointerNet)
    
    def test_load_model_invalid_path(self):
        """Test load_wu2024_model with invalid path falls back to random_init."""
        config = {"kmax": 32}
        device = torch.device("cpu")
        
        model, weights_mode = load_wu2024_model("/nonexistent/path.pt", config, device)
        
        assert weights_mode == "random_init"


class TestFeatureMapping:
    """Test feature mapping functions."""
    
    def test_select_candidate_stops_padding(self):
        """Test _select_candidate_stops pads correctly."""
        features = _make_features(2)
        kmax = 8
        
        candidates, mask = _select_candidate_stops(features, kmax)
        
        assert len(candidates) == kmax
        assert len(mask) == kmax
        assert candidates[0] >= 0 or candidates[1] >= 0  # At least one real
        assert candidates[-1] == -1  # Padded
        assert mask[-1] == False  # Padded masked
    
    def test_select_candidate_stops_truncation(self):
        """Test _select_candidate_stops truncates correctly."""
        features = _make_features(3)
        kmax = 2
        
        candidates, mask = _select_candidate_stops(features, kmax)
        
        assert len(candidates) == kmax
        assert len(mask) == kmax
    
    def test_build_static_features_shape(self):
        """Test static feature tensor has correct shape."""
        env = MockEnv()
        features = _make_features()
        kmax = 8
        candidates = [0, 1, 2, -1, -1, -1, -1, -1]
        
        static = _build_static_features(env, features, kmax, candidates)
        
        assert static.shape == (1, 3, kmax)
    
    def test_build_dynamic_features_shape(self):
        """Test dynamic feature tensor has correct shape."""
        env = MockEnv()
        features = _make_features()
        kmax = 8
        candidates = [0, 1, 2, -1, -1, -1, -1, -1]
        
        dynamic = _build_dynamic_features(env, features, kmax, candidates)
        
        assert dynamic.shape == (1, 3, kmax)


class TestPolicyOutput:
    """Test policy output validity."""
    
    def test_policy_respects_mask(self):
        """Test wu2024_policy only returns valid (unmasked) actions."""
        env = MockEnv()
        features = _make_features()
        config = {"kmax": 8, "weights_mode": "random_init"}
        device = torch.device("cpu")
        rng = np.random.default_rng(42)
        
        model, _ = load_wu2024_model(None, config, device)
        
        action = wu2024_policy(env, features, model, config, device, rng)
        
        # Action should be one of the valid actions
        valid_actions = features["actions"][features["action_mask"]]
        assert action in valid_actions
    
    def test_policy_uniform_logits_mode(self):
        """Test uniform_logits mode selects first valid action."""
        env = MockEnv()
        features = _make_features()
        config = {"kmax": 8, "weights_mode": "uniform_logits"}
        device = torch.device("cpu")
        rng = np.random.default_rng(42)
        
        model, _ = load_wu2024_model(None, config, device)
        
        action = wu2024_policy(env, features, model, config, device, rng)
        
        # uniform_logits should select first valid action
        valid_indices = np.where(features["action_mask"])[0]
        expected_action = int(features["actions"][valid_indices[0]])
        assert action == expected_action
    
    def test_policy_empty_actions(self):
        """Test policy returns None for empty actions."""
        env = MockEnv()
        features = {
            "actions": np.array([], dtype=np.int64),
            "action_mask": np.array([], dtype=bool),
            "node_features": np.zeros((0, 5), dtype=np.float32),
            "edge_features": np.zeros((0, 4), dtype=np.float32),
            "current_node_index": np.array([0], dtype=np.int64),
        }
        config = {"kmax": 8, "weights_mode": "random_init"}
        device = torch.device("cpu")
        rng = np.random.default_rng(42)
        
        model, _ = load_wu2024_model(None, config, device)
        
        action = wu2024_policy(env, features, model, config, device, rng)
        
        assert action is None


class TestReproducibility:
    """Test fixed seed reproducibility."""
    
    def test_same_seed_same_action_sequence(self):
        """Test same seed produces identical action sequence."""
        env = MockEnv()
        features = _make_features()
        config = {"kmax": 8, "weights_mode": "random_init", "hidden_size": 64}
        device = torch.device("cpu")
        
        actions1 = []
        actions2 = []
        
        # Run 1: fixed seed
        torch.manual_seed(123)
        np.random.seed(123)
        model1, _ = load_wu2024_model(None, config, device)
        rng1 = np.random.default_rng(42)
        for _ in range(5):
            a = wu2024_policy(env, features, model1, config, device, rng1)
            if a is not None:
                actions1.append(a)
        
        # Run 2: same fixed seed
        torch.manual_seed(123)
        np.random.seed(123)
        model2, _ = load_wu2024_model(None, config, device)
        rng2 = np.random.default_rng(42)
        for _ in range(5):
            a = wu2024_policy(env, features, model2, config, device, rng2)
            if a is not None:
                actions2.append(a)
        
        # Sequences should match
        assert actions1 == actions2, f"Actions differ: {actions1} vs {actions2}"
    
    def test_different_seed_may_differ(self):
        """Test different seeds may produce different results."""
        env = MockEnv()
        features = _make_features()
        config = {"kmax": 8, "weights_mode": "random_init"}
        device = torch.device("cpu")
        
        # Different seeds
        torch.manual_seed(111)
        model1, _ = load_wu2024_model(None, config, device)
        rng1 = np.random.default_rng(111)
        
        torch.manual_seed(999)
        model2, _ = load_wu2024_model(None, config, device)
        rng2 = np.random.default_rng(999)
        
        # Collect multiple calls (different seeds should differ in model weights)
        # This is a weak test - just ensures no crash
        a1 = wu2024_policy(env, features, model1, config, device, rng1)
        a2 = wu2024_policy(env, features, model2, config, device, rng2)
        
        # Both should be valid
        assert a1 is None or a1 in features["actions"]
        assert a2 is None or a2 in features["actions"]


class TestModelForward:
    """Test model forward pass."""
    
    def test_forward_returns_probs(self):
        """Test model forward returns valid probability distribution."""
        kmax = 8
        model = Wu2024PointerNet(
            static_size=3,
            dynamic_size=3,
            hidden_size=64,
        )
        model.eval()
        
        static = torch.randn(1, 3, kmax)
        dynamic = torch.randn(1, 3, kmax)
        mask = torch.ones(1, kmax)
        mask[0, 5:] = 0  # Mask last 3
        
        with torch.no_grad():
            probs = model(static, dynamic, mask)
        
        assert probs.shape == (1, kmax)
        # Probs should sum to ~1 (softmax)
        assert abs(probs.sum().item() - 1.0) < 1e-5
        # Masked positions should have near-zero probability
        assert probs[0, 7].item() < 1e-6
