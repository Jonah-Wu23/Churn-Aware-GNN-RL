"""
Unit tests for Gini coefficient implementation.

Tests cover:
- Boundary cases (empty, all-zero, single element, μ=0)
- Known distributions with exact expected values
- Permutation invariance
- Cross-module consistency

Design decisions tested:
- Algorithm: Relative Mean Difference (RMD)
- μ=0 convention: returns 0.0 (no inequality when nothing is served)
- Stop set: ALL Layer-2 stops (including zero-service)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.fairness import gini_coefficient, compute_service_volume_gini


class TestGiniCoefficientBoundary:
    """Boundary case tests."""
    
    def test_empty_list_returns_zero(self):
        """Empty input should return 0.0 (no inequality)."""
        assert gini_coefficient([]) == 0.0
    
    def test_single_element_returns_zero(self):
        """Single element has no inequality."""
        assert gini_coefficient([5]) == 0.0
        assert gini_coefficient([0]) == 0.0
        assert gini_coefficient([100]) == 0.0
    
    def test_all_zeros_returns_zero(self):
        """All zeros (μ=0) convention: returns 0.0."""
        assert gini_coefficient([0, 0, 0]) == 0.0
        assert gini_coefficient([0, 0, 0, 0, 0]) == 0.0
    
    def test_mean_zero_returns_zero(self):
        """When mean is 0, should return 0.0 not NaN."""
        result = gini_coefficient([0] * 10)
        assert result == 0.0
        assert not np.isnan(result)


class TestGiniCoefficientKnownValues:
    """Tests with known expected values."""
    
    def test_perfect_equality(self):
        """All equal values should give Gini = 0."""
        assert gini_coefficient([10, 10, 10]) == 0.0
        assert gini_coefficient([5, 5, 5, 5]) == 0.0
        assert gini_coefficient([1] * 100) == 0.0
    
    def test_maximum_inequality_three_elements(self):
        """[0, 0, 100] should give Gini = 2/3 ≈ 0.6667."""
        result = gini_coefficient([0, 0, 100])
        expected = 2.0 / 3.0
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    
    def test_linear_sequence(self):
        """[1, 2, 3, 4, 5] should give Gini = 0.2667."""
        result = gini_coefficient([1, 2, 3, 4, 5])
        # Manual calculation: 
        # n=5, mean=3, diff_sum = 2*(|1-2|+|1-3|+|1-4|+|1-5|+|2-3|+|2-4|+|2-5|+|3-4|+|3-5|+|4-5|)
        # = 2*(1+2+3+4+1+2+3+1+2+1) = 2*20 = 40
        # G = 40 / (2 * 25 * 3) = 40 / 150 = 4/15 ≈ 0.2667
        expected = 4.0 / 15.0
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    
    def test_binary_distribution(self):
        """[0, 10] should give Gini = 0.5."""
        result = gini_coefficient([0, 10])
        # n=2, mean=5, diff_sum = 2*|0-10| = 20
        # G = 20 / (2 * 4 * 5) = 20 / 40 = 0.5
        expected = 0.5
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"


class TestGiniCoefficientPermutationInvariance:
    """Gini should be order-independent."""
    
    def test_same_result_different_order(self):
        """Same values in different order should give same Gini."""
        values = [1, 5, 10, 2, 8]
        result1 = gini_coefficient(values)
        result2 = gini_coefficient(sorted(values))
        result3 = gini_coefficient(sorted(values, reverse=True))
        result4 = gini_coefficient([5, 1, 8, 10, 2])  # random order
        
        assert result1 == result2 == result3 == result4
    
    def test_random_permutations_consistent(self):
        """Multiple random permutations should give identical results."""
        rng = np.random.default_rng(42)
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected = gini_coefficient(values)
        
        for _ in range(10):
            shuffled = values.copy()
            rng.shuffle(shuffled)
            result = gini_coefficient(shuffled)
            assert result == expected, f"Permutation gave different result: {result} vs {expected}"


class TestComputeServiceVolumeGini:
    """Tests for the aligned vector helper function."""
    
    def test_full_coverage(self):
        """When service dict covers all stops."""
        service = {0: 10, 1: 20, 2: 30}
        all_stops = [0, 1, 2]
        result = compute_service_volume_gini(service, all_stops)
        expected = gini_coefficient([10, 20, 30])
        assert result == expected
    
    def test_partial_coverage_zero_fill(self):
        """Missing stops should be zero-filled."""
        service = {0: 10, 2: 30}  # stop 1 missing
        all_stops = [0, 1, 2, 3]  # 4 stops total
        result = compute_service_volume_gini(service, all_stops)
        expected = gini_coefficient([10, 0, 30, 0])
        assert result == expected
    
    def test_empty_service_dict(self):
        """Empty service dict should give all zeros -> Gini=0."""
        service = {}
        all_stops = [0, 1, 2, 3]
        result = compute_service_volume_gini(service, all_stops)
        assert result == 0.0
    
    def test_single_stop_served(self):
        """Only one stop served out of many."""
        service = {5: 100}
        all_stops = [0, 1, 2, 3, 4, 5]  # 6 stops
        result = compute_service_volume_gini(service, all_stops)
        # values = [0, 0, 0, 0, 0, 100]
        # This should match [0, 0, 100] scaled appropriately for 6 elements
        expected = gini_coefficient([0, 0, 0, 0, 0, 100])
        assert result == expected
        # With 6 elements, 5 zeros and 1 non-zero:
        # n=6, mean=100/6, diff_sum = 2*5*100 = 1000
        # G = 1000 / (2 * 36 * 100/6) = 1000 / 1200 = 5/6 ≈ 0.833
        assert abs(result - 5.0/6.0) < 1e-10


class TestCrossModuleConsistency:
    """Tests that different modules produce consistent results."""
    
    @pytest.fixture
    def sample_graph_data(self, tmp_path: Path) -> tuple:
        """Create sample graph data files."""
        nodes = pd.DataFrame({
            "gnn_node_id": [0, 1, 2, 3, 4],
            "lon": [0.0, 0.1, 0.2, 0.3, 0.4],
            "lat": [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        edges = pd.DataFrame({
            "source": [0, 1, 2, 3],
            "target": [1, 2, 3, 4],
            "travel_time_sec": [10.0, 10.0, 10.0, 10.0],
        })
        od = pd.DataFrame({
            "tpep_pickup_datetime": [pd.Timestamp("2025-01-01T00:00:00")],
            "pickup_stop_id": [0],
            "dropoff_stop_id": [1],
        })
        embeddings = pd.DataFrame({
            "gnn_node_id": [0, 1, 2, 3, 4],
            "emb_geo_0": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        
        nodes_path = tmp_path / "nodes.parquet"
        edges_path = tmp_path / "edges.parquet"
        od_path = tmp_path / "od.parquet"
        emb_path = tmp_path / "emb.parquet"
        
        nodes.to_parquet(nodes_path, index=False)
        edges.to_parquet(edges_path, index=False)
        od.to_parquet(od_path, index=False)
        embeddings.to_parquet(emb_path, index=False)
        
        return str(nodes_path), str(edges_path), str(od_path), str(emb_path)
    
    def test_gym_env_uses_aligned_gini(self, sample_graph_data, tmp_path):
        """Verify gym_env computes Gini with full Layer-2 stop alignment."""
        nodes_path, edges_path, od_path, emb_path = sample_graph_data
        
        from src.env.gym_env import EnvConfig, EventDrivenEnv
        
        env = EventDrivenEnv(EnvConfig(
            od_glob=od_path,
            graph_nodes_path=nodes_path,
            graph_edges_path=edges_path,
            graph_embeddings_path=emb_path,
        ))
        
        # Manually set service counts
        env.service_count_by_stop = {0: 10, 1: 0, 2: 5, 3: 0, 4: 0}
        
        # Verify alignment: stop_ids should be 5 elements
        assert len(env.stop_ids) == 5
        
        # Compute expected Gini with full alignment
        expected = compute_service_volume_gini(env.service_count_by_stop, env.stop_ids)
        
        # The _gini method should now use standardized implementation
        values = [float(env.service_count_by_stop.get(s, 0)) for s in env.stop_ids]
        actual = env._gini(values)
        
        assert actual == expected
    
    def test_fairness_module_matches_aligned_computation(self):
        """Direct fairness module call should match aligned computation."""
        service = {0: 5, 3: 15}
        stops = [0, 1, 2, 3, 4]
        
        # Via compute_service_volume_gini
        result1 = compute_service_volume_gini(service, stops)
        
        # Manual aligned computation
        values = [float(service.get(s, 0)) for s in stops]  # [5, 0, 0, 15, 0]
        result2 = gini_coefficient(values)
        
        assert result1 == result2
