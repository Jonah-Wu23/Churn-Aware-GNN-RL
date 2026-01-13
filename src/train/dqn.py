"""DQN training loop for the event-driven microtransit environment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import nn
import copy
from torch.nn import functional as F

from src.train.replay_buffer import BufferSpec, ReplayBuffer
from src.utils.hashing import sha256_file

LOG = logging.getLogger(__name__)


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
    gamma: float = 0.99
    learning_rate: float = 1e-3
    max_grad_norm: float = 10.0
    double_dqn: bool = True
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    log_every_steps: int = 1_000
    device: str = "cpu"


def _linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    if duration <= 0:
        return float(end)
    frac = min(max(float(t) / float(duration), 0.0), 1.0)
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
    ) -> None:
        self.env = env
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        self.model.to(self.device)
        # Re-create target model by deepcopy to preserve architecture.
        self.target_model = copy.deepcopy(model)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.learning_rate))

        self.rng = np.random.default_rng(int(config.seed))
        torch.manual_seed(int(config.seed))

        spec = BufferSpec(
            num_nodes=int(len(env.stop_ids)),
            node_dim=int(model.node_dim),
            edge_dim=int(model.edge_dim),
            max_actions=int(max(len(v) for v in env.neighbors.values())) if env.neighbors else 0,
        )
        self.buffer = ReplayBuffer(capacity=int(config.buffer_size), spec=spec)

        self.graph_edge_index = torch.tensor(env.graph_edge_index, dtype=torch.long, device=self.device)
        self.graph_edge_features = torch.tensor(env.graph_edge_features, dtype=torch.float32, device=self.device)

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

    def close(self) -> None:
        if getattr(self, "_log_handle", None):
            self._log_handle.close()
        if getattr(self, "_reward_log_handle", None):
            self._reward_log_handle.close()

    def _q_values(self, obs: np.ndarray, obs_idx: int, action_nodes: np.ndarray, action_edge: np.ndarray) -> torch.Tensor:
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
        return self.model(data)

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

    def _optimize(self, global_step: int) -> Dict[str, float]:
        cfg = self.config
        if self.buffer.size < int(cfg.learning_starts):
            return {}
        if global_step % int(cfg.train_freq) != 0:
            return {}

        losses = []
        for _ in range(int(cfg.gradient_steps)):
            batch = self.buffer.sample(int(cfg.batch_size), rng=self.rng)
            obs = batch["obs"]
            obs_idx = batch["obs_idx"]
            action_nodes = batch["action_node_indices"]
            action_edge = batch["action_edge_features"]
            action_mask = batch["action_mask"]
            action_count = batch["action_count"]
            action_taken = batch["action_taken"]

            next_obs = batch["next_obs"]
            next_obs_idx = batch["next_obs_idx"]
            next_action_nodes = batch["next_action_node_indices"]
            next_action_edge = batch["next_action_edge_features"]
            next_action_mask = batch["next_action_mask"]
            next_action_count = batch["next_action_count"]

            rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
            done = torch.tensor(batch["done"].astype(np.float32), dtype=torch.float32, device=self.device)

            q_pred = []
            q_next = []
            for i in range(len(obs)):
                a_count = int(action_count[i])
                n_count = int(next_action_count[i])
                if a_count == 0:
                    q_pred.append(torch.tensor(0.0, device=self.device))
                else:
                    q_values = self._q_values(obs[i], int(obs_idx[i]), action_nodes[i, :a_count], action_edge[i, :a_count])
                    q_pred.append(q_values[int(action_taken[i])])

                if n_count == 0:
                    q_next.append(torch.tensor(0.0, device=self.device))
                    continue

                next_mask = next_action_mask[i, :n_count].astype(bool)
                if not np.any(next_mask):
                    q_next.append(torch.tensor(0.0, device=self.device))
                    continue

                with torch.no_grad():
                    if cfg.double_dqn:
                        online_q = self._q_values(
                            next_obs[i],
                            int(next_obs_idx[i]),
                            next_action_nodes[i, :n_count],
                            next_action_edge[i, :n_count],
                        )
                        online_q[torch.tensor(~next_mask, device=self.device)] = -1e9
                        best = int(torch.argmax(online_q).item())
                        target_q = self._q_values(
                            next_obs[i],
                            int(next_obs_idx[i]),
                            next_action_nodes[i, :n_count],
                            next_action_edge[i, :n_count],
                        )
                        q_next.append(target_q[best].detach())
                    else:
                        target_q = self._q_values(
                            next_obs[i],
                            int(next_obs_idx[i]),
                            next_action_nodes[i, :n_count],
                            next_action_edge[i, :n_count],
                        )
                        target_q[torch.tensor(~next_mask, device=self.device)] = -1e9
                        q_next.append(torch.max(target_q).detach())

            q_pred_t = torch.stack(q_pred)
            q_next_t = torch.stack(q_next)
            target = rewards + float(cfg.gamma) * (1.0 - done) * q_next_t

            loss = F.smooth_l1_loss(q_pred_t, target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.max_grad_norm))
            self.optimizer.step()

            losses.append(float(loss.item()))

        if global_step % int(cfg.target_update_interval) == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return {"loss": float(np.mean(losses)) if losses else 0.0}

    def train(
        self,
        total_steps: Optional[int] = None,
        episode_callback: Optional[Callable[[Dict[str, float]], bool]] = None,
    ) -> Path:
        cfg = self.config
        max_steps = int(total_steps if total_steps is not None else cfg.total_steps)
        episode_return = 0.0
        episode_steps = 0
        stuck_sum = 0.0
        episode_index = 0

        self.env.reset()
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover
            tqdm = None

        iterator = range(1, max_steps + 1)
        if tqdm is not None:
            iterator = tqdm(iterator, total=max_steps, desc="train", unit="step")

        for step in iterator:
            eps = _linear_schedule(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps, step)
            features = self.env.get_feature_batch()
            mask = features["action_mask"].astype(bool)
            step_stuckness = float((~mask).mean()) if len(mask) else 1.0
            stuck_sum += step_stuckness
            action = self._select_action(features, epsilon=eps)
            stepped = True
            if action is None:
                actions = features["actions"].astype(np.int64)
                valid = np.where(mask)[0]
                if len(actions) == 0 or len(valid) == 0:
                    reward = 0.0
                    done = True
                    info = {}
                    stepped = False
                else:
                    action = int(actions[int(valid[0])])
                    _, reward, done, info = self.env.step(int(action))
            else:
                _, reward, done, info = self.env.step(int(action))

            reward_payload = {
                "type": "reward_terms",
                "step": int(step),
                "episode_index": int(episode_index),
                "reward_total": float(info.get("reward_total", reward)),
                "reward_base_service": float(info.get("reward_base_service", 0.0)),
                "reward_waiting_churn_penalty": float(info.get("reward_waiting_churn_penalty", 0.0)),
                "reward_fairness_penalty": float(info.get("reward_fairness_penalty", 0.0)),
                "reward_cvar_penalty": float(info.get("reward_cvar_penalty", 0.0)),
                "reward_travel_cost": float(info.get("reward_travel_cost", 0.0)),
                "reward_onboard_delay_penalty": float(info.get("reward_onboard_delay_penalty", 0.0)),
                "reward_onboard_churn_penalty": float(info.get("reward_onboard_churn_penalty", 0.0)),
                "reward_tacc_bonus": float(info.get("reward_tacc_bonus", 0.0)),
                "step_travel_time_sec": float(info.get("step_travel_time_sec", 0.0)),
                "step_stuckness": float(step_stuckness),
            }
            self._reward_log_handle.write(json.dumps(reward_payload, ensure_ascii=False) + "\n")
            if step % int(cfg.log_every_steps) == 0:
                self._reward_log_handle.flush()

            episode_return += float(reward)
            episode_steps += 1

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
                served = int(info.get("served", 0))
                waiting_churned = int(info.get("waiting_churned", 0))
                onboard_churned = int(info.get("onboard_churned", 0))
                structural = int(info.get("structural_unserviceable", 0))
                log = {
                    "type": "episode",
                    "step": int(step),
                    "epsilon": float(eps),
                    "episode_return": float(episode_return),
                    "episode_steps": int(episode_steps),
                    "served": served,
                    "waiting_churned": waiting_churned,
                    "onboard_churned": onboard_churned,
                    "structural_unserviceable": structural,
                    "stuckness": float(stuck_sum / max(1, episode_steps)),
                }
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
                payload = {"type": "train", "step": int(step), "epsilon": float(eps), **train_stats}
                self._log_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._log_handle.flush()
                LOG.info("step=%d eps=%.3f %s", step, eps, train_stats)
                if tqdm is not None:
                    iterator.set_postfix(loss=train_stats.get("loss", 0.0), eps=round(eps, 3))

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

