import json
import logging
from pathlib import Path

import numpy as np

import src.eval.evaluator as evaluator


class DummyVehicle:
    def __init__(self) -> None:
        self.onboard = []


class DummyEnv:
    def __init__(self, config) -> None:
        self.config = config
        self.current_time = 0.0
        self.requests = [{"request_time_sec": 0.0, "pickup_time_sec": 0.0}]
        self.structurally_unserviceable = 0
        self.waiting_churned = 0
        self.waiting_timeouts = 0
        self.onboard_churned = 0
        self.served = 1
        self.waiting = {}
        self.vehicles = [DummyVehicle()]
        self.service_count_by_stop = {0: 1}
        self.stop_ids = [0]
        self.acc_wait_time_by_stop = {0: 0.0}

    def get_feature_batch(self):
        return {
            "actions": np.array([0], dtype=np.int64),
            "action_mask": np.array([True]),
            "node_features": np.zeros((1, 5), dtype=np.float32),
            "edge_features": np.zeros((1, 4), dtype=np.float32),
            "action_node_indices": np.array([0], dtype=np.int64),
            "current_node_index": np.array([0], dtype=np.int64),
            "graph_edge_index": np.zeros((2, 0), dtype=np.int64),
            "graph_edge_features": np.zeros((0, 4), dtype=np.float32),
        }

    def step(self, action):
        self.current_time += 1.0
        info = {"step_tacc_gain": 0.0}
        return None, 0.0, True, info


def _write_dummy_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true\n", encoding="utf-8")
    return config_path


def _base_config():
    return {
        "env": {
            "debug_mask": True,
            "debug_abort_on_alert": True,
            "seed": 7,
        },
        "eval": {
            "episodes": 1,
            "seed": 7,
            "policy": "random",
            "device": "cpu",
            "max_steps": 2,
        },
        "model": {},
    }


def test_fast_eval_disable_debug_applies_to_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "EventDrivenEnv", DummyEnv)
    monkeypatch.setattr(evaluator, "_build_env_config", lambda cfg: cfg)
    monkeypatch.setattr(evaluator, "build_hashes", lambda cfg: ({}, {}))
    monkeypatch.setattr(evaluator, "sha256_file", lambda path: "dummy")

    cfg = _base_config()
    cfg["eval"]["fast_eval_disable_debug"] = True
    run_dir = tmp_path / "eval"
    config_path = _write_dummy_config(tmp_path)
    output_path = evaluator.evaluate(cfg, config_path=str(config_path), run_dir=run_dir)

    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
    meta_env = payload["meta"]["env_config"]
    assert meta_env["debug_mask"] is False
    assert meta_env["debug_abort_on_alert"] is False


def test_parallel_episodes_cuda_falls_back_to_serial(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(evaluator, "EventDrivenEnv", DummyEnv)
    monkeypatch.setattr(evaluator, "_build_env_config", lambda cfg: cfg)
    monkeypatch.setattr(evaluator, "build_hashes", lambda cfg: ({}, {}))
    monkeypatch.setattr(evaluator, "sha256_file", lambda path: "dummy")

    cfg = _base_config()
    cfg["eval"]["episodes"] = 2
    cfg["eval"]["parallel_episodes"] = 2
    cfg["eval"]["device"] = "cuda"
    run_dir = tmp_path / "eval_cuda"
    config_path = _write_dummy_config(tmp_path)

    caplog.set_level(logging.WARNING)
    evaluator.evaluate(cfg, config_path=str(config_path), run_dir=run_dir)
    assert any("falling back to serial" in rec.message for rec in caplog.records)


def test_parallel_episodes_cpu_uses_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "EventDrivenEnv", DummyEnv)
    monkeypatch.setattr(evaluator, "_build_env_config", lambda cfg: cfg)
    monkeypatch.setattr(evaluator, "build_hashes", lambda cfg: ({}, {}))
    monkeypatch.setattr(evaluator, "sha256_file", lambda path: "dummy")

    cfg = _base_config()
    cfg["eval"]["episodes"] = 3
    cfg["eval"]["parallel_episodes"] = 2
    cfg["eval"]["device"] = "cpu"
    run_dir = tmp_path / "eval_cpu"
    config_path = _write_dummy_config(tmp_path)

    called = {"ctx": False, "pool": False}

    class DummyPool:
        def __enter__(self):
            called["pool"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, func, iterable):
            for item in iterable:
                yield func(item)

    class DummyContext:
        def Pool(self, processes=None):
            return DummyPool()

    def dummy_get_context(name):
        called["ctx"] = True
        return DummyContext()

    monkeypatch.setattr(evaluator.mp, "get_context", dummy_get_context)

    evaluator.evaluate(cfg, config_path=str(config_path), run_dir=run_dir)
    assert called["ctx"] is True
    assert called["pool"] is True
