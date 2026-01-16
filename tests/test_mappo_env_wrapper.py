"""Tests for MAPPO environment wrapper.

Tests the key invariants:
1. Fixed action_dim = neighbor_k + 1
2. NOOP + active_masks for async decision handling
3. Team reward (all agents receive same reward)
4. available_actions masks correctly handle inactive agents
5. Deterministic replay with fixed seed
"""

import numpy as np
import pytest

from src.env.gym_env import EnvConfig
from src.env.mappo_env_wrapper import (
    MAPPOEnvConfig,
    MAPPOEnvWrapper,
    DummyVecEnv,
)


@pytest.fixture
def basic_env_config():
    """Create a basic EnvConfig for testing."""
    return EnvConfig(
        max_horizon_steps=50,
        max_requests=100,
        seed=42,
        num_vehicles=2,
        vehicle_capacity=4,
        request_timeout_sec=300,
    )


@pytest.fixture
def mappo_config(basic_env_config):
    """Create MAPPOEnvConfig for testing."""
    return MAPPOEnvConfig(
        env_config=basic_env_config,
        neighbor_k=8,
        max_episode_steps=50,
    )


class TestActionSpace:
    """Test action space invariants."""
    
    def test_fixed_action_dim(self, mappo_config):
        """Action dim should be neighbor_k + 1 (including NOOP)."""
        env = MAPPOEnvWrapper(mappo_config)
        
        assert env.action_dim == mappo_config.neighbor_k + 1
        for space in env.action_space:
            assert space.n == mappo_config.neighbor_k + 1
            
    def test_noop_always_available_for_active(self, mappo_config):
        """NOOP should always be available for active agents."""
        env = MAPPOEnvWrapper(mappo_config)
        obs, share_obs, available = env.reset()
        
        noop_idx = mappo_config.neighbor_k
        for agent_id in range(env.n_agents):
            if env._active_agent_mask[agent_id]:
                assert available[agent_id, noop_idx] == 1.0, \
                    f"NOOP should be available for active agent {agent_id}"
                    
    def test_inactive_only_noop(self, mappo_config):
        """Inactive agents should only have NOOP available."""
        env = MAPPOEnvWrapper(mappo_config)
        obs, share_obs, available = env.reset()
        
        noop_idx = mappo_config.neighbor_k
        for agent_id in range(env.n_agents):
            if not env._active_agent_mask[agent_id]:
                # All actions except NOOP should be unavailable
                for action_idx in range(mappo_config.neighbor_k):
                    assert available[agent_id, action_idx] == 0.0, \
                        f"Action {action_idx} should be unavailable for inactive agent {agent_id}"
                # NOOP should be available
                assert available[agent_id, noop_idx] == 1.0, \
                    f"NOOP should be available for inactive agent {agent_id}"


class TestAsyncDecision:
    """Test asynchronous decision handling with synchronous interface."""
    
    def test_only_one_active_at_start(self, mappo_config):
        """Only one agent should be active initially (event-driven design)."""
        env = MAPPOEnvWrapper(mappo_config)
        env.reset()
        
        active_count = env._active_agent_mask.sum()
        assert active_count <= 1, \
            f"Expected at most 1 active agent, got {active_count}"
            
    def test_active_mask_shape(self, mappo_config):
        """get_active_masks should return correct shape."""
        env = MAPPOEnvWrapper(mappo_config)
        env.reset()
        
        active_masks = env.get_active_masks()
        assert active_masks.shape == (env.n_agents, 1)
        assert active_masks.dtype == np.float32
        
    def test_sync_step_with_all_actions(self, mappo_config):
        """Step should accept actions for all agents."""
        env = MAPPOEnvWrapper(mappo_config)
        obs, share_obs, available = env.reset()
        
        # Create random actions (using available actions)
        actions = np.zeros(env.n_agents, dtype=np.int64)
        for agent_id in range(env.n_agents):
            valid = np.where(available[agent_id] > 0)[0]
            if len(valid) > 0:
                actions[agent_id] = valid[0]
                
        # Should not raise
        obs, share_obs, rewards, dones, infos, available = env.step(actions)
        
        assert obs.shape == (env.n_agents, env._local_obs_dim)
        assert rewards.shape == (env.n_agents, 1)
        assert dones.shape == (env.n_agents, 1)


class TestTeamReward:
    """Test team reward distribution."""
    
    def test_all_agents_same_reward(self, mappo_config):
        """All agents should receive the same team reward."""
        env = MAPPOEnvWrapper(mappo_config)
        obs, share_obs, available = env.reset()
        
        # Take a step
        actions = np.full(env.n_agents, mappo_config.neighbor_k)  # All NOOP
        obs, share_obs, rewards, dones, infos, available = env.step(actions)
        
        # All rewards should be equal
        if env.n_agents > 1:
            for i in range(1, env.n_agents):
                assert rewards[0, 0] == rewards[i, 0], \
                    f"Agent 0 reward {rewards[0, 0]} != Agent {i} reward {rewards[i, 0]}"


class TestObservationSpaces:
    """Test observation space dimensions and consistency."""
    
    def test_obs_space_dimensions(self, mappo_config):
        """Observation spaces should have correct dimensions."""
        env = MAPPOEnvWrapper(mappo_config)
        
        for space in env.observation_space:
            assert len(space.shape) == 1
            assert space.shape[0] == env._local_obs_dim
            
        for space in env.share_observation_space:
            assert len(space.shape) == 1
            assert space.shape[0] == env._share_obs_dim
            
    def test_obs_shape_matches_space(self, mappo_config):
        """Returned observations should match space dimensions."""
        env = MAPPOEnvWrapper(mappo_config)
        obs, share_obs, available = env.reset()
        
        assert obs.shape == (env.n_agents, env._local_obs_dim)
        assert share_obs.shape == (env.n_agents, env._share_obs_dim)
        assert available.shape == (env.n_agents, env.action_dim)
        
    def test_obs_finite(self, mappo_config):
        """Observations should not contain NaN or Inf."""
        env = MAPPOEnvWrapper(mappo_config)
        obs, share_obs, available = env.reset()
        
        assert np.isfinite(obs).all(), "obs contains non-finite values"
        assert np.isfinite(share_obs).all(), "share_obs contains non-finite values"


class TestDeterministicReplay:
    """Test deterministic replay with fixed seed."""
    
    def test_same_seed_same_trajectory(self, basic_env_config):
        """Two runs with same seed should produce identical trajectories."""
        config1 = MAPPOEnvConfig(
            env_config=EnvConfig(**{**basic_env_config.__dict__, "seed": 123}),
            neighbor_k=8,
        )
        config2 = MAPPOEnvConfig(
            env_config=EnvConfig(**{**basic_env_config.__dict__, "seed": 123}),
            neighbor_k=8,
        )
        
        env1 = MAPPOEnvWrapper(config1)
        env2 = MAPPOEnvWrapper(config2)
        
        obs1, share1, avail1 = env1.reset()
        obs2, share2, avail2 = env2.reset()
        
        np.testing.assert_array_almost_equal(obs1, obs2, decimal=5)
        np.testing.assert_array_almost_equal(avail1, avail2, decimal=5)
        
        # Take same actions
        for _ in range(5):
            if env1.env.done or env2.env.done:
                break
                
            # Use NOOP for consistency
            actions = np.full(env1.n_agents, config1.neighbor_k)
            
            result1 = env1.step(actions)
            result2 = env2.step(actions)
            
            np.testing.assert_array_almost_equal(result1[0], result2[0], decimal=5)
            np.testing.assert_array_almost_equal(result1[2], result2[2], decimal=5)


class TestPaddingEdgeCases:
    """Test action padding edge cases."""
    
    def test_candidates_less_than_k(self, mappo_config):
        """Should handle fewer candidates than neighbor_k correctly."""
        env = MAPPOEnvWrapper(mappo_config)
        obs, share_obs, available = env.reset()
        
        for agent_id in range(env.n_agents):
            if env._active_agent_mask[agent_id]:
                candidates = env._current_candidates.get(agent_id, [])
                # Padding actions beyond candidates should be masked
                for action_idx in range(len(candidates), mappo_config.neighbor_k):
                    assert available[agent_id, action_idx] == 0.0, \
                        f"Padding action {action_idx} should be masked"


class TestVecEnv:
    """Test vectorized environment wrapper."""
    
    def test_vec_env_creation(self, mappo_config):
        """Should create multiple parallel environments."""
        env_fns = [lambda: MAPPOEnvWrapper(mappo_config) for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)
        
        assert vec_env.n_envs == 2
        assert vec_env.num_agents == mappo_config.env_config.num_vehicles
        
    def test_vec_env_reset(self, mappo_config):
        """Vectorized reset should return stacked observations."""
        env_fns = [lambda: MAPPOEnvWrapper(mappo_config) for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)
        
        obs, share_obs, available = vec_env.reset()
        
        n_envs = 2
        n_agents = mappo_config.env_config.num_vehicles
        
        assert obs.shape[0] == n_envs
        assert obs.shape[1] == n_agents
        
    def test_vec_env_step(self, mappo_config):
        """Vectorized step should process all environments."""
        env_fns = [lambda: MAPPOEnvWrapper(mappo_config) for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)
        
        obs, share_obs, available = vec_env.reset()
        
        n_envs = 2
        n_agents = mappo_config.env_config.num_vehicles
        
        # All NOOP
        actions = np.full((n_envs, n_agents), mappo_config.neighbor_k)
        
        obs, share_obs, rewards, dones, infos, available = vec_env.step(actions)
        
        assert obs.shape[0] == n_envs
        assert rewards.shape == (n_envs, n_agents, 1)
        assert dones.shape == (n_envs, n_agents, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
