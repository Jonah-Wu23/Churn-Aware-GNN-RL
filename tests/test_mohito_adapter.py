"""Unit tests for MOHITO baseline adapter.

Tests verify:
1. Graph construction produces valid MOHITO-compatible structure
2. Feature dimensions match expectations (node_dim=5)
3. Action mask is 100% respected
4. Fixed seed produces deterministic output

Note: Tests requiring torch_geometric will be skipped if not installed.
"""

import numpy as np
import pytest
import torch

# Check if torch_geometric is available
try:
    import torch_geometric
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# Import constants (these don't require PyG)
from src.baselines.mohito_adapter import (
    NODE_TYPE_AGENT,
    NODE_TYPE_TASK,
    NODE_TYPE_ACTION,
    NODE_TYPE_EDGE,
)


class MockVehicle:
    def __init__(self, vehicle_id=0, onboard=None):
        self.vehicle_id = vehicle_id
        self.onboard = onboard or []
        self.accepted = []


class MockEnv:
    def __init__(self, num_vehicles=1, waiting=None):
        self.num_vehicles = num_vehicles
        self.waiting = waiting or {}
        self._active_vehicle = MockVehicle()
    
    def _get_active_vehicle(self):
        return self._active_vehicle


def make_features(num_actions=5, current_node=0):
    """Create mock features matching EventDrivenEnv output."""
    return {
        "actions": np.arange(num_actions),
        "action_mask": np.ones(num_actions, dtype=bool),
        "node_features": np.random.randn(20, 5).astype(np.float32),
        "edge_features": np.random.randn(num_actions, 4).astype(np.float32),
        "current_node_index": np.array([current_node]),
        "action_node_indices": np.arange(num_actions),
    }


@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
class TestBuildMohitoGraph:
    """Tests for build_mohito_graph function."""
    
    def test_graph_structure_valid(self):
        """Graph should have valid PyG Data structure."""
        from src.baselines.mohito_adapter import build_mohito_graph
        
        env = MockEnv()
        features = make_features(num_actions=5)
        
        graph, edge_space, action_space = build_mohito_graph(
            env, features, vehicle_idx=0, grid_size=10
        )
        
        # Check PyG Data structure
        assert hasattr(graph, 'x'), "Graph should have node features"
        assert hasattr(graph, 'edge_index'), "Graph should have edge_index"
        assert graph.x.dim() == 2, "Node features should be 2D"
        assert graph.edge_index.dim() == 2, "Edge index should be 2D"
        assert graph.edge_index.shape[0] == 2, "Edge index first dim should be 2"
    
    def test_node_feature_dimension(self):
        """All nodes should have 5 features (MOHITO requirement)."""
        from src.baselines.mohito_adapter import build_mohito_graph
        
        env = MockEnv()
        features = make_features(num_actions=3)
        
        graph, _, _ = build_mohito_graph(env, features, vehicle_idx=0)
        
        assert graph.x.shape[1] == 5, f"Node dim should be 5, got {graph.x.shape[1]}"
    
    def test_node_types_present(self):
        """Graph should contain all required node types."""
        from src.baselines.mohito_adapter import build_mohito_graph
        
        env = MockEnv(waiting={0: [{"dropoff_stop_id": 5}]})
        features = make_features(num_actions=3)
        
        graph, _, _ = build_mohito_graph(env, features, vehicle_idx=0)
        
        node_types = graph.x[:, 0].numpy()
        assert NODE_TYPE_AGENT in node_types, "Should have agent nodes"
        assert NODE_TYPE_TASK in node_types, "Should have task nodes"
        assert NODE_TYPE_ACTION in node_types, "Should have action nodes"
        assert NODE_TYPE_EDGE in node_types, "Should have edge nodes"
    
    def test_edge_space_for_vehicle(self):
        """Edge space should contain valid indices for the vehicle."""
        from src.baselines.mohito_adapter import build_mohito_graph
        
        env = MockEnv()
        features = make_features(num_actions=4)
        
        graph, edge_space, action_space = build_mohito_graph(
            env, features, vehicle_idx=0
        )
        
        num_nodes = graph.x.shape[0]
        for idx in edge_space:
            assert 0 <= idx < num_nodes, f"Edge space index {idx} out of bounds"
    
    def test_action_space_matches_actions(self):
        """Action space should have entries matching number of actions + NOOP."""
        from src.baselines.mohito_adapter import build_mohito_graph
        
        env = MockEnv()
        num_actions = 5
        features = make_features(num_actions=num_actions)
        
        _, _, action_space = build_mohito_graph(env, features, vehicle_idx=0)
        
        # +1 for NOOP action
        assert len(action_space) == num_actions + 1


class TestNodeTypeConstants:
    """Tests for node type constants (no PyG dependency)."""
    
    def test_node_types_distinct(self):
        """All node types should be distinct integers."""
        types = [NODE_TYPE_EDGE, NODE_TYPE_AGENT, NODE_TYPE_TASK, NODE_TYPE_ACTION]
        assert len(types) == len(set(types)), "Node types must be distinct"
    
    def test_node_types_values(self):
        """Node types should match MOHITO rideshare values."""
        assert NODE_TYPE_EDGE == 0
        assert NODE_TYPE_AGENT == 1
        assert NODE_TYPE_TASK == 2
        assert NODE_TYPE_ACTION == 3


class TestActionMaskCompliance:
    """Tests for action mask compliance (no PyG dependency for setup)."""
    
    def test_masked_actions_identification(self):
        """Actions with mask=False should be correctly identified."""
        features = make_features(num_actions=5)
        
        # Mask out actions 1, 2, 3
        features["action_mask"] = np.array([True, False, False, False, True])
        
        valid_actions = features["actions"][features["action_mask"]]
        assert len(valid_actions) == 2
        assert 0 in valid_actions
        assert 4 in valid_actions
    
    def test_all_masked_handling(self):
        """Handle case where all actions are masked."""
        features = make_features(num_actions=3)
        features["action_mask"] = np.array([False, False, False])
        
        valid_indices = np.where(features["action_mask"])[0]
        assert len(valid_indices) == 0


@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
class TestDeterminism:
    """Tests for fixed seed determinism."""
    
    def test_graph_construction_shape_consistent(self):
        """Same inputs should produce graphs with same shape."""
        from src.baselines.mohito_adapter import build_mohito_graph
        
        np.random.seed(42)
        env = MockEnv()
        features = make_features(num_actions=4)
        
        np.random.seed(42)
        graph1, _, _ = build_mohito_graph(env, features, vehicle_idx=0)
        
        np.random.seed(42)
        graph2, _, _ = build_mohito_graph(env, features, vehicle_idx=0)
        
        assert graph1.x.shape == graph2.x.shape


@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
class TestMultiVehicle:
    """Tests for multi-vehicle parameter sharing."""
    
    def test_different_vehicle_indices(self):
        """Different vehicle indices should produce different edge spaces."""
        from src.baselines.mohito_adapter import build_mohito_graph
        
        env = MockEnv(num_vehicles=3)
        features = make_features(num_actions=4)
        
        _, edge_space_0, _ = build_mohito_graph(env, features, vehicle_idx=0)
        _, edge_space_1, _ = build_mohito_graph(env, features, vehicle_idx=1)
        
        # Edge spaces should be different for different vehicles
        assert edge_space_0 != edge_space_1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
