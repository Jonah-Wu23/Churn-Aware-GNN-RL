import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module(path: str):
    script_path = Path(__file__).resolve().parents[1] / path
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mappo_num_vehicles_override() -> None:
    module = _load_module("scripts/run_mappo_train.py")
    args = SimpleNamespace(
        lr=5e-4,
        critic_lr=5e-4,
        num_env_steps=1,
        episode_length=5,
        n_rollout_threads=1,
        ppo_epoch=10,
        num_mini_batch=1,
        clip_param=0.2,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        hidden_size=64,
        use_ReLU=True,
        data_chunk_length=5,
        recurrent_N=1,
        neighbor_k=8,
        seed=1,
        cuda=False,
        save_interval=10,
        log_interval=5,
    )
    config = {
        "env": {"num_vehicles": 3},
        "mappo_train": {"num_vehicles": 100},
    }
    all_args = module.make_args_namespace(args, config)
    assert all_args.num_vehicles == 100


def test_cpo_num_vehicles_override() -> None:
    module = _load_module("scripts/run_cpo_train.py")
    config = {
        "env": {"num_vehicles": 3},
        "cpo_train": {"num_vehicles": 100},
    }
    env_config = module.build_env_config(config, seed=7)
    assert env_config.num_vehicles == 100
