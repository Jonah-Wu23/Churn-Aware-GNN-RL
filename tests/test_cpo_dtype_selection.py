import importlib.util
from pathlib import Path

import torch


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_cpo_train.py"
    spec = importlib.util.spec_from_file_location("run_cpo_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_torch_dtype_auto_cpu_is_float64() -> None:
    module = _load_module()
    assert module._select_torch_dtype(torch.device("cpu"), "auto") == torch.float64


def test_select_torch_dtype_auto_cuda_is_float32() -> None:
    module = _load_module()
    assert module._select_torch_dtype(torch.device("cuda"), "auto") == torch.float32


def test_select_torch_dtype_explicit() -> None:
    module = _load_module()
    assert module._select_torch_dtype(torch.device("cpu"), "float32") == torch.float32
    assert module._select_torch_dtype(torch.device("cuda"), "float64") == torch.float64


def test_resolve_max_iter_uses_target_steps() -> None:
    module = _load_module()
    args = type("Args", (), {"target_steps": 200000, "min_batch_size": 2048, "max_iter": 10})()
    assert module._resolve_max_iter(args) == 98
