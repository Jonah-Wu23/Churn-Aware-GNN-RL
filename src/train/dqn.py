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

        spec = BufferSpec(
            num_nodes=int(len(env.stop_ids)),
            node_dim=int(model.node_dim),
            edge_dim=int(model.edge_dim),
            max_actions=int(max(len(v) for v in env.neighbors.values())) if env.neighbors else 0,
        )
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

    def _fallback_episode_info(self) -> Dict[str, float]:
        env = self.env
        return {
            "served": float(getattr(env, "served", 0.0)),
            "waiting_churned": float(getattr(env, "waiting_churned", 0.0)),
            "waiting_timeouts": float(getattr(env, "waiting_timeouts", 0.0)),
            "onboard_churned": float(getattr(env, "onboard_churned", 0.0)),
            "structural_unserviceable": float(getattr(env, "structural_unserviceable", 0.0)),
        }

    def close(self) -> None:
        if getattr(self, "_log_handle", None):
            self._log_handle.close()
        if getattr(self, "_reward_log_handle", None):
            self._reward_log_handle.close()

    def _save_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self._write_run_meta()

    def _save_checkpoint(self, path: Path, step: int, episode_index: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "episode_index": int(episode_index),
            "config": json.loads(json.dumps(self.config.__dict__)),
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, path)

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
        return active_model.q_head(features).squeeze(-1)

    def _optimize(self, global_step: int) -> Dict[str, float]:
        """优化步骤，GNN只做1次前向传播，批量计算所有样本Q值。"""
        cfg = self.config
        if self.buffer.size < int(cfg.learning_starts):
            return {}
        if global_step % int(cfg.train_freq) != 0:
            return {}

        losses = []
        for _ in range(int(cfg.gradient_steps)):
            batch = self.buffer.sample(int(cfg.batch_size), rng=self.rng)
            
            # 过滤掉无效样本（action_count=0）
            valid_mask = batch["action_count"] > 0
            if not np.any(valid_mask):
                continue
            
            valid_indices = np.where(valid_mask)[0]
            
            # 批量计算当前状态Q值（每个样本obs不同，需逐个GNN前向）
            all_q_preds = []
            all_targets = []
            
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
            
            if len(all_q_preds) == 0:
                continue
            
            # 批量计算loss并反向传播
            self.optimizer.zero_grad()
            q_preds_tensor = torch.stack(all_q_preds)
            targets_tensor = torch.tensor(all_targets, dtype=torch.float32, device=self.device)
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = F.smooth_l1_loss(q_preds_tensor, targets_tensor)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.max_grad_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = F.smooth_l1_loss(q_preds_tensor, targets_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.max_grad_norm))
                self.optimizer.step()
            
            losses.append(float(loss.item()))

        if global_step % int(cfg.target_update_interval) == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if not losses:
            return {}
        return {"loss": float(np.mean(losses))}

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
            eps = _linear_schedule(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps, step)
            features = self.env.get_feature_batch()
            mask = features["action_mask"].astype(bool)
            step_stuckness = float((~mask).mean()) if len(mask) else 1.0
            action = self._select_action(features, epsilon=eps)
            stepped = True
            if action is None:
                actions = features["actions"].astype(np.int64)
                valid = np.where(mask)[0]
                if len(actions) == 0 or len(valid) == 0:
                    reward = 0.0
                    done = True
                    info = dict(self._fallback_episode_info())
                    stepped = False
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

            if step % 50 == 0:
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
                payload = {"type": "train", "step": int(step), "epsilon": float(eps), "buffer_size": int(self.buffer.size), **train_stats}
                self._log_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._log_handle.flush()
                LOG.info("step=%d eps=%.3f buf=%d %s", step, eps, self.buffer.size, train_stats)
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

