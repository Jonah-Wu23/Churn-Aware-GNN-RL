from types import SimpleNamespace
import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_mappo_train.py"
    spec = importlib.util.spec_from_file_location("run_mappo_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_args() -> SimpleNamespace:
    return SimpleNamespace(
        lr=5e-4,
        critic_lr=5e-4,
        num_env_steps=1000,
        episode_length=200,
        n_rollout_threads=1,
        ppo_epoch=10,
        num_mini_batch=1,
        clip_param=0.2,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        hidden_size=64,
        recurrent_N=1,
        neighbor_k=8,
        seed=1,
        cuda=False,
        save_interval=10,
        log_interval=5,
        use_ReLU=True,
        data_chunk_length=10,
    )


def test_mappo_data_chunk_length_is_capped() -> None:
    args = _build_args()
    args.data_chunk_length = 999
    args.episode_length = 5
    args.n_rollout_threads = 2
    config = {"env": {"num_vehicles": 1}}
    module = _load_module()
    all_args = module.make_args_namespace(args, config)
    assert all_args.data_chunk_length == 10


def test_mappo_args_namespace_has_required_fields() -> None:
    args = _build_args()
    config = {"env": {"num_vehicles": 1}}
    module = _load_module()
    all_args = module.make_args_namespace(args, config)

    required_fields = {
        "algorithm_name",
        "experiment_name",
        "env_name",
        "lr",
        "critic_lr",
        "opti_eps",
        "weight_decay",
        "num_env_steps",
        "episode_length",
        "n_rollout_threads",
        "n_eval_rollout_threads",
        "n_render_rollout_threads",
        "ppo_epoch",
        "num_mini_batch",
        "clip_param",
        "entropy_coef",
        "value_loss_coef",
        "use_max_grad_norm",
        "max_grad_norm",
        "gamma",
        "gae_lambda",
        "use_gae",
        "use_proper_time_limits",
        "hidden_size",
        "layer_N",
        "recurrent_N",
        "use_ReLU",
        "use_recurrent_policy",
        "use_naive_recurrent_policy",
        "use_orthogonal",
        "gain",
        "use_feature_normalization",
        "use_policy_active_masks",
        "use_value_active_masks",
        "stacked_frames",
        "data_chunk_length",
        "use_centralized_V",
        "use_obs_instead_of_state",
        "use_valuenorm",
        "use_linear_lr_decay",
        "use_popart",
        "use_clipped_value_loss",
        "use_huber_loss",
        "huber_delta",
        "seed",
        "cuda",
        "cuda_deterministic",
        "save_interval",
        "log_interval",
        "eval_interval",
        "use_eval",
        "use_render",
        "use_wandb",
        "model_dir",
        "num_vehicles",
        "neighbor_k",
    }

    missing = sorted(field for field in required_fields if not hasattr(all_args, field))
    assert not missing, f"Missing fields in MAPPO args namespace: {missing}"
