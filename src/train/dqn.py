"""DQN training loop for the event-driven microtransit environment."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import random

import numpy as np
import torch
from torch import nn
import copy
from torch.nn import functional as F

from src.train.replay_buffer import BufferSpec, ReplayBuffer, PrioritizedReplayBuffer
from src.utils.hashing import sha256_file
from src.utils.build_info import get_build_id
from src.utils.realtime_viz import RealtimeVizConfig, RealtimeVizPublisher
from src.utils.reward_hack_alerts import RewardHackAlertConfig, RewardHackDetector

LOG = logging.getLogger(__name__)


def write_run_meta(
    run_dir: Path,
    model_path_final: Optional[Path],
    model_path_latest: Optional[Path],
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    payload: Dict[str, object] = {
        "model_path_final": str(model_path_final) if model_path_final is not None else None,
        "model_path_latest": str(model_path_latest) if model_path_latest is not None else None,
    }
    if extra:
        payload.update(extra)
    output_path = run_dir / "run_meta.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="ascii")
    return output_path


@dataclass(frozen=True)
class DQNConfig:
    seed: int = 7
    total_steps: int = 200_000
    buffer_size: int = 10_000
    batch_size: int = 64
    learning_starts: int = 2_000
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 2_000
    target_update_tau: float = 0.0
    gamma: float = 0.99
    learning_rate: float = 1e-3
    max_grad_norm: float = 10.0
    double_dqn: bool = True
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    log_every_steps: int = 1_000
    checkpoint_every_steps: int = 10_000
    device: str = "cpu"
    use_amp: bool = False
    prioritized_replay: bool = False
    replay_alpha: float = 0.6
    replay_beta_start: float = 0.4
    replay_beta_frames: int = 200_000
    replay_eps: float = 1e-6


@dataclass(frozen=True)
class EpsilonSchedule:
    start: float
    end: float
    decay_steps: int
    start_global_step: int


def _linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    if duration <= 0:
        return float(end)
    if t <= 0:
        return float(start)
    if t >= duration:
        return float(end)
    frac = float(t) / float(duration)
    return float(start + frac * (end - start))


class DQNTrainer:
    def __init__(
        self,
        env,
        model: nn.Module,
        config: DQNConfig,
        run_dir: Path,
        graph_hashes: Dict[str, str],
        od_hashes: Dict[str, str],
        global_step_init: int = 0,
        env_cfg: Optional[Dict[str, object]] = None,
        viz_config: Optional[Dict[str, object]] = None,
    ) -> None:
        self.env = env
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self._env_cfg = env_cfg or {}  # 用于终止惩罚计算
        self.graph_hashes = graph_hashes
        self.od_hashes = od_hashes

        self.model.to(self.device)
        # Re-create target model by deepcopy to preserve architecture.
        self.target_model = copy.deepcopy(model)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.learning_rate))
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and config.device == "cuda" else None

        self.rng = np.random.default_rng(int(config.seed))
        torch.manual_seed(int(config.seed))
        random.seed(int(config.seed))

        # Global step as the single source of truth for epsilon
        self.global_step = int(global_step_init)
        self._epsilon_schedule: Optional[EpsilonSchedule] = None
        
        # Epsilon cap support for collapse protection
        self.epsilon_cap: Optional[float] = None
        self.epsilon_cap_remaining: int = 0

        max_degree = int(max(len(v) for v in env.neighbors.values())) if env.neighbors else 0
        max_actions = max_degree + 1 if max_degree > 0 else 0
        spec = BufferSpec(
            num_nodes=int(len(env.stop_ids)),
            node_dim=int(model.node_dim),
            edge_dim=int(model.edge_dim),
            max_actions=max_actions,
        )
        if config.prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(
                capacity=int(config.buffer_size),
                spec=spec,
                alpha=float(config.replay_alpha),
                beta_start=float(config.replay_beta_start),
                beta_frames=int(config.replay_beta_frames),
                epsilon=float(config.replay_eps),
            )
        else:
            self.buffer = ReplayBuffer(capacity=int(config.buffer_size), spec=spec)

        self.graph_edge_index = torch.tensor(env.graph_edge_index, dtype=torch.long, device=self.device)
        self.graph_edge_features = torch.tensor(env.graph_edge_features, dtype=torch.float32, device=self.device)
        if self.graph_edge_index.numel() == 0:
            raise ValueError("graph_edge_index is empty; training must use the full Layer-2 graph.")
        if self.graph_edge_features.numel() == 0:
            raise ValueError("graph_edge_features is empty; training must use the full Layer-2 graph.")
        if self.graph_edge_index.shape[1] != self.graph_edge_features.shape[0]:
            raise ValueError("graph_edge_index and graph_edge_features size mismatch.")

        self.run_dir = run_dir
        self.log_path = run_dir / "train_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        self.reward_log_path = run_dir / "reward_terms.jsonl"
        self._reward_log_handle = self.reward_log_path.open("w", encoding="utf-8")

        meta = {
            "seed": int(config.seed),
            "config": json.loads(json.dumps(config.__dict__)),
            "graph_hashes": graph_hashes,
            "od_hashes": od_hashes,
        }
        meta_record = json.dumps({"type": "meta", "payload": meta}, ensure_ascii=False) + "\n"
        self._log_handle.write(meta_record)
        self._reward_log_handle.write(meta_record)
        self._log_handle.flush()
        self._reward_log_handle.flush()

        self._viz_cfg = RealtimeVizConfig.from_dict(viz_config)
        self._viz = RealtimeVizPublisher(self._viz_cfg)
        alert_overrides = viz_config.get("alerts") if isinstance(viz_config, dict) else None
        if isinstance(env_cfg, dict):
            alert_overrides = dict(alert_overrides or {})
            alert_overrides.setdefault("debug_abort_on_alert", env_cfg.get("debug_abort_on_alert", True))
            alert_overrides.setdefault("debug_dump_dir", env_cfg.get("debug_dump_dir", "reports/debug/potential_alerts"))
        alert_cfg = RewardHackAlertConfig.from_dict(alert_overrides)
        self._alert_detector = RewardHackDetector(alert_cfg)
        self._viz_window = deque(maxlen=200)
        self._build_id = get_build_id()

    def _fallback_episode_info(self) -> Dict[str, float]:
        env = self.env
        info = {
            "served": float(getattr(env, "served", 0.0)),
            "waiting_churned": float(getattr(env, "waiting_churned", 0.0)),
            "waiting_timeouts": float(getattr(env, "waiting_timeouts", 0.0)),
            "onboard_churned": float(getattr(env, "onboard_churned", 0.0)),
            "structural_unserviceable": float(getattr(env, "structurally_unserviceable", 0.0)),
            "waiting_remaining": float(sum(len(q) for q in getattr(env, "waiting", {}).values())),
            "onboard_remaining": float(sum(len(v.onboard) for v in getattr(env, "vehicles", []))),
        }
        potential = getattr(env, "get_potential_debug", None)
        if callable(potential):
            info.update(potential())
        return info

    def _collect_vehicle_snapshot(self) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
        vehicles: List[Dict[str, object]] = []
        vehicles_by_id: Dict[str, Dict[str, object]] = {}
        for idx, vehicle in enumerate(getattr(self.env, "vehicles", [])):
            vehicle_id = getattr(vehicle, "vehicle_id", idx)
            current_stop = getattr(vehicle, "current_stop", -1)
            available_time = getattr(vehicle, "available_time", getattr(self.env, "current_time", 0.0))
            record = {
                "vehicle_id": int(vehicle_id),
                "current_stop": int(current_stop),
                "available_time": float(available_time),
                "onboard": int(len(getattr(vehicle, "onboard", []))),
            }
            vehicles.append(record)
            vehicles_by_id[str(record["vehicle_id"])] = record
        return vehicles, vehicles_by_id

    def _collect_pre_step_snapshot(
        self,
        episode_index: int,
        episode_steps: int,
        step: int,
    ) -> Dict[str, object]:
        vehicles, vehicles_by_id = self._collect_vehicle_snapshot()
        waiting_remaining = float(sum(len(q) for q in getattr(self.env, "waiting", {}).values()))
        onboard_remaining = float(sum(len(v.onboard) for v in getattr(self.env, "vehicles", [])))
        snapshot_id = f"{int(episode_index)}:{int(episode_steps)}:{int(step)}:{int(self.global_step)}"
        return {
            "snapshot_phase": "pre_step",
            "snapshot_id": snapshot_id,
            "pre_step_time": float(getattr(self.env, "current_time", 0.0)),
            "active_vehicle_id": getattr(self.env, "active_vehicle_id", None),
            "ready_vehicles": int(len(getattr(self.env, "ready_vehicle_ids", []))),
            "event_queue_len": int(len(getattr(self.env, "event_queue", []))),
            "waiting_remaining": waiting_remaining,
            "onboard_remaining": onboard_remaining,
            "vehicles": vehicles,
            "vehicles_by_id": vehicles_by_id,
        }

    def _compute_terminal_backlog_penalty(self) -> Tuple[float, float, float, float]:
        """计算终止积压惩罚（无有效动作时调用）。
        
        Returns:
            Tuple of (penalty, waiting_remaining, onboard_remaining, waiting_timeouts)
        """
        env = self.env
        waiting_remaining = float(sum(len(q) for q in getattr(env, "waiting", {}).values()))
        onboard_remaining = float(sum(len(v.onboard) for v in getattr(env, "vehicles", [])))
        waiting_timeouts = float(getattr(env, "waiting_timeouts", 0))
        max_requests = float(self._env_cfg.get("max_requests", 2000))
        penalty_coef = float(self._env_cfg.get("reward_terminal_backlog_penalty", 0.0))
        
        if penalty_coef <= 0 or max_requests <= 0:
            return 0.0, waiting_remaining, onboard_remaining, waiting_timeouts
        
        norm_backlog = (waiting_remaining + onboard_remaining) / max_requests
        penalty = penalty_coef * norm_backlog
        return penalty, waiting_remaining, onboard_remaining, waiting_timeouts

    def _attach_env(self, new_env, new_env_cfg: Optional[Dict[str, object]] = None) -> None:
        """重新绑定环境（用于 L3 phase 切换）。"""
        self.env = new_env
        if new_env_cfg is not None:
            self._env_cfg = new_env_cfg
        # 刷新图张量缓存
        self.graph_edge_index = torch.tensor(new_env.graph_edge_index, dtype=torch.long, device=self.device)
        self.graph_edge_features = torch.tensor(new_env.graph_edge_features, dtype=torch.float32, device=self.device)
        self.env.reset()

    def close(self) -> None:
        if getattr(self, "_log_handle", None):
            self._log_handle.close()
        if getattr(self, "_reward_log_handle", None):
            self._reward_log_handle.close()
        if getattr(self, "_viz", None):
            self._viz.close()

    def set_epsilon_schedule(
        self,
        start: float,
        end: float,
        decay_steps: int,
        start_global_step: Optional[int] = None,
    ) -> None:
        self._epsilon_schedule = EpsilonSchedule(
            start=float(start),
            end=float(end),
            decay_steps=int(decay_steps),
            start_global_step=int(self.global_step if start_global_step is None else start_global_step),
        )

    def get_epsilon(self) -> float:
        """获取当前epsilon值（考虑cap限制）"""
        schedule = self._epsilon_schedule
        if schedule is None:
            eps = _linear_schedule(
                self.config.epsilon_start,
                self.config.epsilon_end,
                self.config.epsilon_decay_steps,
                self.global_step,
            )
        else:
            local_step = max(0, int(self.global_step - schedule.start_global_step))
            eps = _linear_schedule(
                schedule.start,
                schedule.end,
                schedule.decay_steps,
                local_step,
            )
        if self.epsilon_cap is not None and self.epsilon_cap_remaining > 0:
            eps = min(eps, self.epsilon_cap)
        return eps

    def _soft_update_target(self, tau: float) -> None:
        tau = float(tau)
        if tau <= 0.0:
            return
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

    def _get_rng_states(self) -> Dict[str, object]:
        """获取所有RNG状态用于checkpoint"""
        return {
            "numpy_rng": self.rng.bit_generator.state,
            "python_random": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _set_rng_states(self, states: Dict[str, object]) -> None:
        """从checkpoint恢复RNG状态"""
        self.rng.bit_generator.state = states["numpy_rng"]
        random.setstate(states["python_random"])
        torch.set_rng_state(states["torch_cpu"])
        if states["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda"])

    def _save_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self._write_run_meta()

    def save_model(self, path: Path) -> None:
        """Public wrapper for saving model checkpoints."""
        self._save_model(path)

    def _save_checkpoint(self, path: Path, step: int, episode_index: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "global_step": self.global_step,
            "episode_index": int(episode_index),
            "config": json.loads(json.dumps(self.config.__dict__)),
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_buffer": self.buffer.get_state(),
            "rng_states": self._get_rng_states(),
            "epsilon_cap": self.epsilon_cap,
            "epsilon_cap_remaining": self.epsilon_cap_remaining,
        }
        torch.save(payload, path)

    def save_full_checkpoint(self, path: Path, stage: str = "", phase: str = "", 
                              extra_meta: Optional[Dict[str, object]] = None) -> None:
        """保存完整checkpoint含元信息（用于best模型选择）"""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_step": self.global_step,
            "epsilon": self.get_epsilon(),
            "stage": stage,
            "phase": phase,
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_buffer": self.buffer.get_state(),
            "rng_states": self._get_rng_states(),
        }
        if extra_meta:
            payload["extra_meta"] = extra_meta
        torch.save(payload, path)

    def restore_from_checkpoint(self, path: Path, restore_buffer: bool = True) -> None:
        """从checkpoint恢复trainer状态"""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.target_model.load_state_dict(ckpt["target_model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "global_step" in ckpt:
            self.global_step = int(ckpt["global_step"])
        if restore_buffer and "replay_buffer" in ckpt:
            self.buffer.set_state(ckpt["replay_buffer"])
        if "rng_states" in ckpt:
            self._set_rng_states(ckpt["rng_states"])
        if "epsilon_cap" in ckpt:
            self.epsilon_cap = ckpt["epsilon_cap"]
            self.epsilon_cap_remaining = ckpt.get("epsilon_cap_remaining", 0)
        LOG.info("Restored from checkpoint: global_step=%d, epsilon=%.4f, buffer_size=%d",
                 self.global_step, self.get_epsilon(), self.buffer.size)

    def _write_run_meta(self) -> None:
        latest_path = self.run_dir / "edgeq_model_latest.pt"
        final_path = self.run_dir / "edgeq_model_final.pt"
        meta = {
            "seed": int(self.config.seed),
            "graph_hashes": self.graph_hashes,
            "od_hashes": self.od_hashes,
        }
        write_run_meta(
            self.run_dir,
            model_path_final=final_path if final_path.exists() else None,
            model_path_latest=latest_path if latest_path.exists() else None,
            extra=meta,
        )

    def _q_values_single(self, obs: np.ndarray, obs_idx: int, action_nodes: np.ndarray, action_edge: np.ndarray, model: Optional[nn.Module] = None, requires_grad: bool = True) -> torch.Tensor:
        """计算单个样本的Q值，共享图结构，显存友好。"""
        active_model = self.model if model is None else model
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        dst = torch.tensor(action_nodes, dtype=torch.long, device=self.device)
        src = torch.full_like(dst, int(obs_idx), dtype=torch.long, device=self.device)
        action_edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.tensor(action_edge, dtype=torch.float32, device=self.device)
        data = {
            "node_features": x,
            "graph_edge_index": self.graph_edge_index,
            "graph_edge_features": self.graph_edge_features,
            "action_edge_index": action_edge_index,
            "edge_features": edge_attr,
        }
        if requires_grad:
            return active_model(data)
        else:
            with torch.no_grad():
                return active_model(data).detach()

    def _q_values(self, obs: np.ndarray, obs_idx: int, action_nodes: np.ndarray, action_edge: np.ndarray, model: Optional[nn.Module] = None) -> torch.Tensor:
        active_model = self.model if model is None else model
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        dst = torch.tensor(action_nodes, dtype=torch.long, device=self.device)
        src = torch.full_like(dst, int(obs_idx), dtype=torch.long, device=self.device)
        action_edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.tensor(action_edge, dtype=torch.float32, device=self.device)
        data = {
            "node_features": x,
            "graph_edge_index": self.graph_edge_index,
            "graph_edge_features": self.graph_edge_features,
            "action_edge_index": action_edge_index,
            "edge_features": edge_attr,
        }
        return active_model(data)

    def _select_action(self, features: Dict[str, np.ndarray], epsilon: float) -> Optional[int]:
        actions = features["actions"].astype(np.int64)
        mask = features["action_mask"].astype(bool)
        if len(actions) == 0:
            return None
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return None
        if self.rng.random() < float(epsilon):
            idx = int(self.rng.choice(valid))
            return int(actions[idx])

        with torch.no_grad():
            q = self._q_values(
                obs=features["node_features"],
                obs_idx=int(features["current_node_index"][0]),
                action_nodes=features["action_node_indices"].astype(np.int64),
                action_edge=features["edge_features"],
            ).detach()
            q_masked = q.clone()
            invalid = torch.tensor(~mask, device=q.device)
            q_masked[invalid] = -1e9
            idx = int(torch.argmax(q_masked).item())
            return int(actions[idx])

    def _compute_entropy_stats(
        self,
        features: Dict[str, np.ndarray],
        epsilon: float,
    ) -> Dict[str, float]:
        actions = features["actions"].astype(np.int64)
        mask = features["action_mask"].astype(bool)
        if len(actions) == 0 or not np.any(mask):
            return {}
        with torch.no_grad():
            q = self._q_values(
                obs=features["node_features"],
                obs_idx=int(features["current_node_index"][0]),
                action_nodes=features["action_node_indices"].astype(np.int64),
                action_edge=features["edge_features"],
            ).detach()
        q_masked = q.clone()
        invalid = torch.tensor(~mask, device=q.device)
        q_masked[invalid] = -1e9
        probs = torch.softmax(q_masked, dim=0)
        probs_valid = probs[torch.tensor(mask, device=q.device)]
        q_entropy = float(
            (-probs_valid * torch.log(probs_valid + 1e-10)).sum().item()
        )
        q_mean = float(q_masked[~invalid].mean().item())
        q_std = float(q_masked[~invalid].std(unbiased=False).item())
        q_max = float(q_masked[~invalid].max().item())

        valid_indices = np.where(mask)[0]
        n_valid = int(len(valid_indices))
        if n_valid <= 0:
            eps_entropy = 0.0
        else:
            greedy_idx = int(torch.argmax(q_masked).item())
            probs_eps = np.full(n_valid, float(epsilon) / float(n_valid), dtype=np.float32)
            greedy_pos = int(np.where(valid_indices == greedy_idx)[0][0])
            probs_eps[greedy_pos] += float(1.0 - epsilon)
            eps_entropy = float(-(probs_eps * np.log(probs_eps + 1e-10)).sum())

        if n_valid > 1:
            q_entropy_norm = float(q_entropy / float(np.log(float(n_valid))))
            eps_entropy_norm = float(eps_entropy / float(np.log(float(n_valid))))
        else:
            q_entropy_norm = 0.0
            eps_entropy_norm = 0.0
        return {
            "q_entropy": q_entropy,
            "epsilon_entropy": eps_entropy,
            "q_entropy_norm": q_entropy_norm,
            "epsilon_entropy_norm": eps_entropy_norm,
            "q_mean": q_mean,
            "q_std": q_std,
            "q_max": q_max,
        }

    def _publish_viz(
        self,
        step: int,
        episode_index: int,
        episode_steps: int,
        epsilon: float,
        features: Dict[str, np.ndarray],
        action: Optional[int],
        reward: float,
        done: bool,
        info: Dict[str, float],
        pre_step_snapshot: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self._viz.enabled:
            return
        if self._viz_cfg.publish_every_steps > 0 and step % int(self._viz_cfg.publish_every_steps) != 0:
            if not (done and self._viz_cfg.publish_on_episode_end):
                return
        snapshot = pre_step_snapshot or {}
        current_stop = int(features["current_stop"][0]) if len(features.get("current_stop", [])) else -1
        action_idx = None
        actions = features.get("actions", np.array([], dtype=np.int64)).astype(np.int64)
        mask = features.get("action_mask", np.array([], dtype=bool)).astype(bool)
        if action is not None and len(actions) > 0:
            matches = np.where(actions == int(action))[0]
            if len(matches) > 0:
                action_idx = int(matches[0])
        entropy_stats = self._compute_entropy_stats(features, epsilon) if len(actions) > 0 else {}
        action_valid_ratio = float(mask.mean()) if len(mask) > 0 else 0.0
        if action is None:
            served_delta = 0.0
            waiting_churn_delta = 0.0
            stop_flag = 1.0
            decision_flag = 0.0
        else:
            served_delta = float(info.get("step_served", 0.0))
            waiting_churn_delta = float(info.get("step_waiting_churned", 0.0))
            stop_flag = 0.0
            decision_flag = 1.0
        reward_total = float(info.get("reward_total", reward))
        reward_nonzero = 1.0 if abs(reward_total) > 1e-6 else 0.0
        invalid_flag = float(info.get("invalid_action", 0.0)) if info else 0.0
        self._viz_window.append(
            {
                "decision": decision_flag,
                "stop": stop_flag,
                "served": served_delta,
                "waiting_churned": waiting_churn_delta,
                "invalid_action": invalid_flag,
                "reward_nonzero": reward_nonzero,
            }
        )
        window_len = len(self._viz_window)
        decision_steps = sum(entry["decision"] for entry in self._viz_window)
        stop_count = sum(entry["stop"] for entry in self._viz_window)
        served_sum = sum(entry["served"] for entry in self._viz_window)
        waiting_churn_sum = sum(entry["waiting_churned"] for entry in self._viz_window)
        invalid_count = sum(entry["invalid_action"] for entry in self._viz_window)
        reward_nonzero_count = sum(entry["reward_nonzero"] for entry in self._viz_window)
        stop_ratio = float(stop_count / window_len) if window_len > 0 else 0.0
        served_per_decision = float(served_sum / max(1.0, decision_steps))
        waiting_churn_per_decision = float(waiting_churn_sum / max(1.0, decision_steps))
        invalid_action_ratio = float(invalid_count / window_len) if window_len > 0 else 0.0
        reward_nonzero_ratio = float(reward_nonzero_count / window_len) if window_len > 0 else 0.0
        vehicles = snapshot.get("vehicles")
        vehicles_by_id = snapshot.get("vehicles_by_id")
        if vehicles is None:
            vehicles, vehicles_by_id = self._collect_vehicle_snapshot()
        elif vehicles_by_id is None:
            vehicles_by_id = {
                str(vehicle.get("vehicle_id")): vehicle
                for vehicle in vehicles
                if isinstance(vehicle, dict) and "vehicle_id" in vehicle
            }
        reward_total = float(info.get("reward_total", reward))
        reward_components = info.get("reward_components")
        if isinstance(reward_components, dict) and reward_components:
            reward_terms = dict(reward_components)
        else:
            reward_terms = {
                "reward_base_service": float(info.get("reward_base_service", 0.0)),
                "reward_waiting_churn_penalty": float(info.get("reward_waiting_churn_penalty", 0.0)),
                "reward_fairness_penalty": float(info.get("reward_fairness_penalty", 0.0)),
                "reward_cvar_penalty": float(info.get("reward_cvar_penalty", 0.0)),
                "reward_travel_cost": float(info.get("reward_travel_cost", 0.0)),
                "reward_onboard_delay_penalty": float(info.get("reward_onboard_delay_penalty", 0.0)),
                "reward_onboard_churn_penalty": float(info.get("reward_onboard_churn_penalty", 0.0)),
                "reward_backlog_penalty": float(info.get("reward_backlog_penalty", 0.0)),
                "reward_waiting_time_penalty": float(info.get("reward_waiting_time_penalty", 0.0)),
                "reward_potential_shaping": float(info.get("reward_potential_shaping", 0.0)),
                "reward_congestion_penalty": float(info.get("reward_congestion_penalty", 0.0)),
                "reward_tacc_bonus": float(info.get("reward_tacc_bonus", 0.0)),
            }
        reward_terms["reward_total"] = reward_total
        reward_components_raw = info.get("reward_components_raw")
        if isinstance(reward_components_raw, dict):
            reward_terms.update(reward_components_raw)
        if "reward_potential_shaping_raw" not in reward_terms:
            reward_terms["reward_potential_shaping_raw"] = float(
                info.get("reward_potential_shaping_raw", 0.0)
            )
        alpha = float(info.get("reward_potential_alpha", getattr(self.env.config, "reward_potential_alpha", 0.0)))
        required_keys = [
            "reward_potential_alpha",
            "reward_potential_alpha_source",
            "reward_potential_lost_weight",
            "reward_potential_scale_with_reward_scale",
            "phi_before",
            "phi_after",
            "phi_delta",
            "phi_backlog_before",
            "phi_backlog_after",
            "lost_total_before",
            "lost_total_after",
            "waiting_churned_before",
            "waiting_churned_after",
            "onboard_churned_before",
            "onboard_churned_after",
            "structural_unserviceable_before",
            "structural_unserviceable_after",
            "waiting_remaining_before",
            "waiting_remaining_after",
            "onboard_remaining_before",
            "onboard_remaining_after",
            "reward_potential_shaping",
            "reward_potential_shaping_raw",
        ]
        missing_keys = []
        for key in required_keys:
            if info is None or key not in info:
                missing_keys.append(key)
        if missing_keys:
            potential = getattr(self.env, "get_potential_debug", None)
            if callable(potential):
                info = dict(info) if info else {}
                info.update(potential())
                missing_keys = []
                for key in required_keys:
                    if key not in info:
                        missing_keys.append(key)
        required_reward_terms = [
            "reward_total",
            "reward_potential_shaping",
            "reward_potential_shaping_raw",
        ]
        for key in required_reward_terms:
            if key not in reward_terms:
                missing_keys.append(f"reward_terms.{key}")
        payload = {
            "type": "step",
            "step": int(step),
            "global_step": int(self.global_step),
            "episode_index": int(episode_index),
            "episode_steps": int(episode_steps),
            "epsilon": float(epsilon),
            "build_id": self._build_id,
            "seed": int(self.config.seed),
            "snapshot_phase": str(snapshot.get("snapshot_phase", "unknown")),
            "snapshot_id": snapshot.get("snapshot_id"),
            "pre_step_time": float(snapshot.get("pre_step_time", getattr(self.env, "current_time", 0.0))),
            "current_time": float(snapshot.get("pre_step_time", getattr(self.env, "current_time", 0.0))),
            "active_vehicle_id": snapshot.get("active_vehicle_id", getattr(self.env, "active_vehicle_id", None)),
            "ready_vehicles": int(snapshot.get("ready_vehicles", len(getattr(self.env, "ready_vehicle_ids", [])))),
            "event_queue_len": int(snapshot.get("event_queue_len", len(getattr(self.env, "event_queue", [])))),
            "env_steps": int(getattr(self.env, "steps", 0)),
            "current_stop": int(current_stop),
            "action_stop": int(action) if action is not None else None,
            "action_index": action_idx,
            "done": bool(done),
            "done_reason": info.get("done_reason"),
            "vehicles": vehicles,
            "vehicles_by_id": vehicles_by_id,
            "reward_terms": reward_terms,
            "reward_total": float(reward_terms["reward_total"]),
            "reward_potential_alpha": alpha,
            "reward_potential_alpha_source": str(info.get("reward_potential_alpha_source", "unknown")),
            "reward_potential_lost_weight": float(info.get("reward_potential_lost_weight", 0.0)),
            "reward_potential_scale_with_reward_scale": bool(
                info.get("reward_potential_scale_with_reward_scale", False)
            ),
            "step_served": float(info.get("step_served", 0.0)),
            "step_waiting_timeouts": float(info.get("step_waiting_timeouts", 0.0)),
            "step_waiting_churned": float(info.get("step_waiting_churned", 0.0)),
            "step_onboard_churned": float(info.get("step_onboard_churned", 0.0)),
            "step_waiting_churn_prob_mean": float(info.get("step_waiting_churn_prob_mean", 0.0)),
            "step_onboard_churn_prob_mean": float(info.get("step_onboard_churn_prob_mean", 0.0)),
            "action_count": int(len(actions)),
            "action_valid_ratio": action_valid_ratio,
            "stop_ratio": stop_ratio,
            "served_per_decision": served_per_decision,
            "waiting_churn_per_decision": waiting_churn_per_decision,
            "invalid_action_ratio": invalid_action_ratio,
            "reward_nonzero_ratio": reward_nonzero_ratio,
            "served": float(info.get("served", 0.0)),
            "waiting_churned": float(info.get("waiting_churned", 0.0)),
            "onboard_churned": float(info.get("onboard_churned", 0.0)),
            "waiting_remaining": float(snapshot.get("waiting_remaining", info.get("waiting_remaining", 0.0))),
            "onboard_remaining": float(snapshot.get("onboard_remaining", info.get("onboard_remaining", 0.0))),
            "waiting_remaining_before": float(info.get("waiting_remaining_before", 0.0)),
            "waiting_remaining_after": float(info.get("waiting_remaining_after", 0.0)),
            "onboard_remaining_before": float(info.get("onboard_remaining_before", 0.0)),
            "onboard_remaining_after": float(info.get("onboard_remaining_after", 0.0)),
            "lost_total_before": float(info.get("lost_total_before", 0.0)),
            "lost_total_after": float(info.get("lost_total_after", 0.0)),
            "waiting_churned_before": float(info.get("waiting_churned_before", 0.0)),
            "waiting_churned_after": float(info.get("waiting_churned_after", 0.0)),
            "onboard_churned_before": float(info.get("onboard_churned_before", 0.0)),
            "onboard_churned_after": float(info.get("onboard_churned_after", 0.0)),
            "structural_unserviceable_before": float(info.get("structural_unserviceable_before", 0.0)),
            "structural_unserviceable_after": float(info.get("structural_unserviceable_after", 0.0)),
            "phi_backlog_before": float(info.get("phi_backlog_before", 0.0)),
            "phi_backlog_after": float(info.get("phi_backlog_after", 0.0)),
            "phi_before": float(info.get("phi_before", 0.0)),
            "phi_after": float(info.get("phi_after", 0.0)),
            "phi_delta": float(info.get("phi_delta", 0.0)),
            "action_mask_valid_count": int(mask.sum()) if len(mask) > 0 else 0,
            "action_mask": mask.tolist(),
            "action_candidates": actions.tolist(),
            "mask_debug": getattr(self.env, "last_mask_debug", None),
            "missing_keys": missing_keys,
            **entropy_stats,
        }
        payload["alerts"] = self._alert_detector.update(payload)
        self._viz.publish(payload)

    def _compute_node_embeddings(self, obs: np.ndarray, model: Optional[nn.Module] = None, requires_grad: bool = True) -> torch.Tensor:
        """计算GNN节点嵌入（共享计算，避免重复前向传播）"""
        active_model = self.model if model is None else model
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        if requires_grad:
            h = active_model.node_encoder(x)
            for conv in active_model.convs:
                h = conv(h, self.graph_edge_index, self.graph_edge_features)
        else:
            with torch.no_grad():
                h = active_model.node_encoder(x)
                for conv in active_model.convs:
                    h = conv(h, self.graph_edge_index, self.graph_edge_features)
        return h
    
    def _compute_q_from_embeddings(self, h: torch.Tensor, obs_idx: int, action_nodes: np.ndarray, action_edge: np.ndarray, model: Optional[nn.Module] = None) -> torch.Tensor:
        """从节点嵌入计算Q值（避免重复GNN前向）"""
        active_model = self.model if model is None else model
        dst = torch.tensor(action_nodes, dtype=torch.long, device=self.device)
        src = torch.full_like(dst, int(obs_idx), dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(action_edge, dtype=torch.float32, device=self.device)
        
        features = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        advantage = active_model.q_head(features).squeeze(-1)
        if not getattr(active_model, "dueling", False):
            return advantage
        pooled = h.mean(dim=0)
        value_input = torch.cat([h[int(obs_idx)], pooled], dim=-1)
        value = active_model.value_head(value_input).squeeze(-1)
        return value + (advantage - advantage.mean())

    def _optimize(self, global_step: int) -> Dict[str, float]:
        """优化步骤，GNN只做1次前向传播，批量计算所有样本Q值。"""
        cfg = self.config
        if self.buffer.size < int(cfg.learning_starts):
            return {}
        if global_step % int(cfg.train_freq) != 0:
            return {}

        losses = []
        for _ in range(int(cfg.gradient_steps)):
            if cfg.prioritized_replay:
                batch = self.buffer.sample(int(cfg.batch_size), rng=self.rng)
                batch_weights = batch.get("weights")
                batch_indices = batch.get("indices")
            else:
                batch = self.buffer.sample(int(cfg.batch_size), rng=self.rng)
                batch_weights = None
                batch_indices = None
            
            # 过滤掉无效样本（action_count=0）
            valid_mask = batch["action_count"] > 0
            if not np.any(valid_mask):
                continue
            
            valid_indices = np.where(valid_mask)[0]
            
            # 批量计算当前状态Q值（每个样本obs不同，需逐个GNN前向）
            all_q_preds = []
            all_targets = []
            td_errors: List[float] = []
            priority_indices: List[int] = []
            
            for i in valid_indices:
                a_count = int(batch["action_count"][i])
                
                # 当前状态Q值（每个样本独立计算）
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        q_all = self._q_values_single(
                            batch["obs"][i],
                            int(batch["obs_idx"][i]),
                            batch["action_node_indices"][i, :a_count],
                            batch["action_edge_features"][i, :a_count],
                            requires_grad=True,
                        )
                else:
                    q_all = self._q_values_single(
                        batch["obs"][i],
                        int(batch["obs_idx"][i]),
                        batch["action_node_indices"][i, :a_count],
                        batch["action_edge_features"][i, :a_count],
                        requires_grad=True,
                    )
                q_pred = q_all[int(batch["action_taken"][i])]
                all_q_preds.append(q_pred)
                
                # 下一状态最大Q值（target网络，无梯度）
                n_count = int(batch["next_action_count"][i])
                if n_count == 0:
                    q_next_val = 0.0
                else:
                    next_mask = batch["next_action_mask"][i, :n_count].astype(bool)
                    if not np.any(next_mask):
                        q_next_val = 0.0
                    else:
                        with torch.no_grad():
                            if cfg.double_dqn:
                                # Online网络选择动作
                                online_q = self._q_values_single(
                                    batch["next_obs"][i],
                                    int(batch["next_obs_idx"][i]),
                                    batch["next_action_node_indices"][i, :n_count],
                                    batch["next_action_edge_features"][i, :n_count],
                                    requires_grad=False,
                                )
                                online_q_masked = online_q.clone()
                                online_q_masked[torch.tensor(~next_mask, device=self.device)] = -1e9
                                best = int(torch.argmax(online_q_masked).item())
                                
                                # Target网络评估
                                target_q = self._q_values_single(
                                    batch["next_obs"][i],
                                    int(batch["next_obs_idx"][i]),
                                    batch["next_action_node_indices"][i, :n_count],
                                    batch["next_action_edge_features"][i, :n_count],
                                    model=self.target_model,
                                    requires_grad=False,
                                )
                                q_next_val = float(target_q[best].item())
                            else:
                                # Target网络直接选择最大Q
                                target_q = self._q_values_single(
                                    batch["next_obs"][i],
                                    int(batch["next_obs_idx"][i]),
                                    batch["next_action_node_indices"][i, :n_count],
                                    batch["next_action_edge_features"][i, :n_count],
                                    model=self.target_model,
                                    requires_grad=False,
                                )
                                target_q_masked = target_q.clone()
                                target_q_masked[torch.tensor(~next_mask, device=self.device)] = -1e9
                                q_next_val = float(torch.max(target_q_masked).item())
                
                target = float(batch["reward"][i]) + float(cfg.gamma) * (1.0 - float(batch["done"][i])) * q_next_val
                all_targets.append(target)
                td_errors.append(float(target - float(q_pred.detach().item())))
                if batch_indices is not None:
                    priority_indices.append(int(batch_indices[i]))
            
            if len(all_q_preds) == 0:
                continue
            
            # 批量计算loss并反向传播
            self.optimizer.zero_grad()
            q_preds_tensor = torch.stack(all_q_preds)
            targets_tensor = torch.tensor(all_targets, dtype=torch.float32, device=self.device)
            if batch_weights is not None:
                weights_tensor = torch.tensor(
                    batch_weights[valid_indices], dtype=torch.float32, device=self.device
                )
            else:
                weights_tensor = None
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_vec = F.smooth_l1_loss(q_preds_tensor, targets_tensor, reduction="none")
                    loss = loss_vec.mean() if weights_tensor is None else (loss_vec * weights_tensor).mean()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.max_grad_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_vec = F.smooth_l1_loss(q_preds_tensor, targets_tensor, reduction="none")
                loss = loss_vec.mean() if weights_tensor is None else (loss_vec * weights_tensor).mean()
                if torch.isnan(loss):
                    LOG.warning(f"NaN loss detected at step {global_step}. Skipping optimization.")
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.max_grad_norm))
                self.optimizer.step()
            
            losses.append(float(loss.item()))
            if cfg.prioritized_replay and priority_indices:
                self.buffer.update_priorities(priority_indices, td_errors)
            if float(cfg.target_update_tau) > 0.0:
                self._soft_update_target(float(cfg.target_update_tau))

        if float(cfg.target_update_tau) <= 0.0 and global_step % int(cfg.target_update_interval) == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if not losses:
            return {}
        return {"loss": float(np.mean(losses))}

    def train(
        self,
        total_steps: Optional[int] = None,
        episode_callback: Optional[Callable[[Dict[str, float]], bool]] = None,
        step_callback: Optional[Callable[[int, int, object], None]] = None,
    ) -> Path:
        cfg = self.config
        max_steps = int(total_steps if total_steps is not None else cfg.total_steps)
        episode_return = 0.0
        episode_steps = 0
        stuck_sum = 0.0
        episode_index = 0
        final_step = 0

        self.env.reset()
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover
            tqdm = None

        iterator = range(1, max_steps + 1)
        if tqdm is not None:
            iterator = tqdm(iterator, total=max_steps, desc="train", unit="step")

        last_info: Dict[str, float] = {}
        for step in iterator:
            final_step = int(step)
            # Increment global_step (single source of truth for epsilon)
            self.global_step += 1
            
            # Compute epsilon from global_step (with cap support)
            eps = self.get_epsilon()
            
            # Decrement epsilon_cap_remaining if active
            if self.epsilon_cap_remaining > 0:
                self.epsilon_cap_remaining -= 1
                if self.epsilon_cap_remaining == 0:
                    self.epsilon_cap = None
                    LOG.info("Epsilon cap expired at global_step=%d", self.global_step)

            if step_callback is not None:
                step_callback(self.global_step, int(step), self.env)
            
            features = self.env.get_feature_batch()
            pre_step_snapshot = self._collect_pre_step_snapshot(episode_index, episode_steps, int(step))
            mask = features["action_mask"].astype(bool)
            step_stuckness = float((~mask).mean()) if len(mask) else 1.0
            action = self._select_action(features, epsilon=eps)
            invalid_action = False
            if action is not None:
                actions = features["actions"].astype(np.int64)
                if len(actions) == 0:
                    invalid_action = True
                    action = None
                else:
                    matches = np.where(actions == int(action))[0]
                    if len(matches) == 0 or not bool(mask[int(matches[0])]):
                        invalid_action = True
                        valid = np.where(mask)[0]
                        if len(valid) > 0:
                            action = int(actions[int(valid[0])])
                        else:
                            action = None
            stepped = True
            if action is None:
                actions = features["actions"].astype(np.int64)
                valid = np.where(mask)[0]
                if len(actions) == 0 or len(valid) == 0:
                    # 无有效动作 - 施加终止惩罚（关键修复）
                    terminal_penalty, waiting_rem, onboard_rem, waiting_timeouts = (
                        self._compute_terminal_backlog_penalty()
                    )
                    reward = -terminal_penalty  # 负惩罚作为 reward
                    done = True
                    stepped = False
                    
                    # 构建完整 info
                    info = dict(self._fallback_episode_info())
                    info["done_reason"] = "no_valid_action"
                    info["terminal_backlog_penalty_applied"] = float(terminal_penalty)
                    info["waiting_remaining"] = float(waiting_rem)
                    info["onboard_remaining"] = float(onboard_rem)
                    info["waiting_timeouts"] = float(waiting_timeouts)
                    info["reward_total"] = float(reward)
                    info["reward_components"] = {
                        "reward_terminal_backlog_penalty": float(-terminal_penalty),
                        "reward_potential_shaping": 0.0,
                    }
                    info["reward_components_raw"] = {
                        "reward_potential_shaping_raw": 0.0,
                    }
                    potential = getattr(self.env, "get_potential_debug", None)
                    if callable(potential):
                        info.update(potential())
                    
                    # 【关键】即使 stepped=False，也必须将惩罚计入 episode_return
                    episode_return += float(reward)
                    # 更新 last_info 确保 episode log 包含完整字段
                    last_info = dict(info)
                else:
                    action = int(actions[int(valid[0])])
                    _, reward, done, info = self.env.step(int(action))
            else:
                _, reward, done, info = self.env.step(int(action))

            if stepped:
                stuck_sum += step_stuckness
                episode_return += float(reward)
                episode_steps += 1
                if info:
                    last_info = dict(info)
                if invalid_action and info is not None:
                    info["invalid_action"] = 1.0
                self._publish_viz(
                    step=step,
                    episode_index=episode_index,
                    episode_steps=episode_steps,
                    epsilon=eps,
                    features=features,
                    action=action,
                    reward=float(reward),
                    done=bool(done),
                    info=dict(info) if info else {},
                    pre_step_snapshot=pre_step_snapshot,
                )
            else:
                if invalid_action and last_info is not None:
                    last_info["invalid_action"] = 1.0
                self._publish_viz(
                    step=step,
                    episode_index=episode_index,
                    episode_steps=episode_steps,
                    epsilon=eps,
                    features=features,
                    action=None,
                    reward=float(reward),
                    done=bool(done),
                    info=dict(last_info) if last_info else {},
                    pre_step_snapshot=pre_step_snapshot,
                )

            if step % 50 == 0:
                reward_components = info.get("reward_components") if info else None
                if isinstance(reward_components, dict) and reward_components:
                    reward_terms_payload = dict(reward_components)
                    reward_terms_payload.update(info.get("reward_components_raw", {}))
                    reward_terms_payload["reward_total"] = float(info.get("reward_total", reward))
                else:
                    reward_terms_payload = {
                        "reward_total": float(info.get("reward_total", reward)),
                        "reward_base_service": float(info.get("reward_base_service", 0.0)),
                        "reward_waiting_churn_penalty": float(info.get("reward_waiting_churn_penalty", 0.0)),
                        "reward_fairness_penalty": float(info.get("reward_fairness_penalty", 0.0)),
                        "reward_cvar_penalty": float(info.get("reward_cvar_penalty", 0.0)),
                        "reward_travel_cost": float(info.get("reward_travel_cost", 0.0)),
                        "reward_onboard_delay_penalty": float(info.get("reward_onboard_delay_penalty", 0.0)),
                        "reward_onboard_churn_penalty": float(info.get("reward_onboard_churn_penalty", 0.0)),
                        "reward_backlog_penalty": float(info.get("reward_backlog_penalty", 0.0)),
                        "reward_waiting_time_penalty": float(info.get("reward_waiting_time_penalty", 0.0)),
                        "reward_potential_shaping": float(info.get("reward_potential_shaping", 0.0)),
                        "reward_potential_shaping_raw": float(info.get("reward_potential_shaping_raw", 0.0)),
                        "reward_congestion_penalty": float(info.get("reward_congestion_penalty", 0.0)),
                        "reward_tacc_bonus": float(info.get("reward_tacc_bonus", 0.0)),
                    }
                reward_terms_payload.setdefault(
                    "reward_potential_shaping_raw",
                    float(info.get("reward_potential_shaping_raw", 0.0)),
                )
                reward_payload = {
                    "type": "reward_terms",
                    "step": int(step),
                    "episode_index": int(episode_index),
                    **reward_terms_payload,
                    "reward_scale": float(info.get("reward_scale", 1.0)),
                    "dst_density_raw": float(info.get("dst_density_raw", 0.0)),
                    "fleet_potential_dst": float(info.get("fleet_potential_dst", 0.0)),
                    "step_travel_time_sec": float(info.get("step_travel_time_sec", 0.0)),
                    "step_stuckness": float(step_stuckness),
                    "step_waiting_time_sec": float(info.get("step_waiting_time_sec", 0.0)),
                    "step_waiting_churn_prob_mean": float(info.get("step_waiting_churn_prob_mean", 0.0)),
                    "step_onboard_churn_prob_mean": float(info.get("step_onboard_churn_prob_mean", 0.0)),
                }
                self._reward_log_handle.write(json.dumps(reward_payload, ensure_ascii=False) + "\n")
            if step % int(cfg.log_every_steps) == 0:
                self._reward_log_handle.flush()

            next_features = self.env.get_feature_batch() if stepped else features

            if stepped:
                actions = features["actions"].astype(np.int64)
                action_taken_idx = int(np.where(actions == int(action))[0][0]) if action is not None else 0
                transition = {
                    "obs": features["node_features"],
                    "obs_idx": int(features["current_node_index"][0]),
                    "action_node_indices": features["action_node_indices"].astype(np.int64),
                    "action_edge_features": features["edge_features"],
                    "action_mask": features["action_mask"].astype(bool),
                    "action_count": int(len(actions)),
                    "action_taken": int(action_taken_idx),
                    "reward": float(reward),
                    "done": bool(done),
                    "next_obs": next_features["node_features"],
                    "next_obs_idx": int(next_features["current_node_index"][0]),
                    "next_action_node_indices": next_features["action_node_indices"].astype(np.int64),
                    "next_action_edge_features": next_features["edge_features"],
                    "next_action_mask": next_features["action_mask"].astype(bool),
                    "next_action_count": int(len(next_features["actions"])),
                }
                if int(cfg.buffer_size) > 0:
                    self.buffer.add(transition)

            train_stats = self._optimize(global_step=step)

            if done:
                if not info:
                    info = dict(last_info) if last_info else dict(self._fallback_episode_info())
                else:
                    filled = dict(self._fallback_episode_info())
                    filled.update(info)
                    info = filled
                served = int(info.get("served", 0))
                waiting_churned = int(info.get("waiting_churned", 0))
                onboard_churned = int(info.get("onboard_churned", 0))
                structural = int(info.get("structural_unserviceable", 0))
                log = {
                    "type": "episode",
                    "step": int(step),
                    "global_step": self.global_step,
                    "epsilon": float(eps),
                    "episode_return": float(episode_return),
                    "episode_steps": int(episode_steps),
                    "served": served,
                    "waiting_churned": waiting_churned,
                    "onboard_churned": onboard_churned,
                    "structural_unserviceable": structural,
                    "waiting_timeouts": int(info.get("waiting_timeouts", 0)),
                    "waiting_remaining": int(info.get("waiting_remaining", 0)),
                    "onboard_remaining": int(info.get("onboard_remaining", 0)),
                    "done_reason": info.get("done_reason") or ("no_valid_action" if not stepped and done else "unknown"),
                    "terminal_backlog_penalty_applied": float(info.get("terminal_backlog_penalty_applied", 0.0)),
                    "stuckness": float(stuck_sum / max(1, episode_steps)),
                    "fleet_density_max_mean": float(info.get("fleet_density_max_mean", 0.0)),
                    "fleet_density_max_p95": float(info.get("fleet_density_max_p95", 0.0)),
                    "stop_coverage_ratio": float(info.get("stop_coverage_ratio", 0.0)),
                    "replay_size": self.buffer.size,
                }
                
                # Crash diagnostics for short episodes or event_queue empty
                if episode_steps < 1000 or len(getattr(self.env, "event_queue", [])) == 0:
                    try:
                        vehicle = self.env._get_active_vehicle()
                        features = self.env.get_feature_batch() if vehicle else {}
                        mask = features.get("action_mask", np.array([])).astype(bool) if features else np.array([])
                        log["crash_diag"] = {
                            "current_stop": int(vehicle.current_stop) if vehicle else -1,
                            "out_degree": len(self.env.neighbors.get(vehicle.current_stop, [])) if vehicle else 0,
                            "mask_valid_ratio": float(mask.mean()) if len(mask) > 0 else 0.0,
                            "event_queue_empty": len(getattr(self.env, "event_queue", [])) == 0,
                            "ready_vehicle_count": len(getattr(self.env, "ready_vehicle_ids", [])),
                            "waiting_remaining": int(info.get("waiting_remaining", 0)),
                            "onboard_remaining": int(info.get("onboard_remaining", 0)),
                        }
                    except Exception as e:
                        log["crash_diag"] = {"error": str(e), "type": "diag_failed"}
                
                self._log_handle.write(json.dumps(log, ensure_ascii=False) + "\n")
                self._log_handle.flush()
                stop = False
                if episode_callback is not None:
                    stop = bool(episode_callback(log))
                episode_return = 0.0
                episode_steps = 0
                stuck_sum = 0.0
                episode_index += 1
                self.env.reset()
                if stop:
                    break

            if step % int(cfg.log_every_steps) == 0:
                payload = {"type": "train", "step": int(step), "global_step": self.global_step, "epsilon": float(eps), "buffer_size": int(self.buffer.size), **train_stats}
                self._log_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._log_handle.flush()
                LOG.info("step=%d global=%d eps=%.3f buf=%d %s", step, self.global_step, eps, self.buffer.size, train_stats)
                if tqdm is not None:
                    postfix = {"eps": round(eps, 3), "buf": int(self.buffer.size)}
                    if "loss" in train_stats:
                        postfix["loss"] = round(train_stats["loss"], 4)
                    iterator.set_postfix(**postfix)

            if int(cfg.checkpoint_every_steps) > 0 and step % int(cfg.checkpoint_every_steps) == 0:
                self._save_model(self.run_dir / "edgeq_model_latest.pt")
                self._save_checkpoint(self.run_dir / "checkpoint_latest.pt", step=step, episode_index=episode_index)

        self._save_model(self.run_dir / "edgeq_model_final.pt")
        self._save_checkpoint(self.run_dir / "checkpoint_final.pt", step=final_step, episode_index=episode_index)
        return self.log_path


def build_hashes(env_cfg: Dict[str, object]) -> tuple[Dict[str, str], Dict[str, str]]:
    graph_hashes = {}
    od_hashes = {}
    for key in ("graph_nodes_path", "graph_edges_path", "graph_embeddings_path"):
        path = env_cfg.get(key)
        if isinstance(path, str) and Path(path).exists():
            graph_hashes[key] = sha256_file(path)
    od_glob = env_cfg.get("od_glob")
    if isinstance(od_glob, str):
        for path in sorted(Path().glob(od_glob)):
            od_hashes[str(path)] = sha256_file(path)
    return graph_hashes, od_hashes
