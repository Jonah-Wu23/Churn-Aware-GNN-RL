import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


def _load_collect_samples():
    repo_root = Path(__file__).resolve().parents[1]
    agent_path = repo_root / "baselines" / "PyTorch-CPO" / "core" / "agent.py"
    sys.path.insert(0, str(repo_root / "baselines" / "PyTorch-CPO"))
    spec = importlib.util.spec_from_file_location("cpo_agent", agent_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.collect_samples


def test_cpo_agent_casts_state_to_float32():
    collect_samples = _load_collect_samples()

    class DummyEnv:
        def reset(self):
            return np.array([1.0, 2.0], dtype=np.float64)

        def step(self, action):
            return np.array([1.0, 2.0], dtype=np.float64), 0.0, True, {}

    class DummyPolicy:
        is_disc_action = True

        def select_action(self, state_var):
            if state_var.dtype != torch.float32:
                raise AssertionError(f"expected float32, got {state_var.dtype}")
            return torch.tensor([0]), None

    env = DummyEnv()
    policy = DummyPolicy()
    memory, log = collect_samples(
        pid=0,
        queue=None,
        env=env,
        policy=policy,
        custom_reward=None,
        mean_action=False,
        render=False,
        running_state=None,
        min_batch_size=1,
    )
    assert log["num_steps"] >= 1
