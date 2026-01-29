"""MOHITO A2C trainer for in-domain training.

This module trains MOHITO's GAT-based actor from random initialization
using our EventDrivenEnv and A2C algorithm.

The trainer uses the same reward and mask as other baselines (MAPPO/CPO/HCRide),
ensuring fair comparison.

Reference:
    Anil, G., Doshi, P., Redder, D., Eck, A., & Soh, L. K. (2025).
    MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for 
    Task-Open Systems. UAI 2025.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.env.gym_env import EventDrivenEnv
from src.baselines.mohito_adapter import (
    build_mohito_graph,
    _ensure_mohito_imports,
    _ensure_pyg_imports,
)

LOG = logging.getLogger(__name__)


@dataclass
class MOHITOTrainConfig:
    """MOHITO training configuration."""
    seed: int = 7
    total_steps: int = 200_000
    gamma: float = 0.99
    learning_rate: float = 0.001
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 5.0
    log_every_steps: int = 1000
    eval_every_steps: int = 10_000
    checkpoint_every_steps: int = 10_000
    device: str = "cuda"
    # MOHITO-specific
    feature_len: int = 5
    num_layers_actor: int = 20
    hidden_dim: int = 50
    heads: int = 2
    grid_size: int = 10
    # Memory/perf controls (critical for large horizons)
    update_every_steps: int = 64
    graph_mode: str = "compact"  # compact/full, see src.baselines.mohito_adapter.build_mohito_graph
    use_amp: bool = True
    amp_dtype: str = "fp16"  # fp16/bf16 (only used when use_amp and CUDA)


class MLPCritic(nn.Module):
    """Simple MLP critic for value baseline in A2C.
    
    Takes pooled graph features and produces state value.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _get_git_commit() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _compute_hashes(env_cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    """Compute config, graph, and dataset hashes for reproducibility."""
    config_hash = hashlib.sha256(
        json.dumps(env_cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    
    graph_path = env_cfg.get("graph_nodes_path", "")
    if Path(graph_path).exists():
        graph_hash = hashlib.sha256(
            Path(graph_path).read_bytes()
        ).hexdigest()[:8]
    else:
        graph_hash = "missing"
    
    od_glob = env_cfg.get("od_glob", "")
    dataset_hash = hashlib.sha256(od_glob.encode()).hexdigest()[:8]
    
    return config_hash, graph_hash, dataset_hash


class MOHITOTrainer:
    """A2C trainer for MOHITO with in-domain training.
    
    Uses the same EventDrivenEnv and reward/mask protocol as other baselines.
    """
    
    def __init__(
        self,
        env: EventDrivenEnv,
        config: MOHITOTrainConfig,
        run_dir: Path,
        env_cfg: Dict[str, Any],
    ):
        self.env = env
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Import MOHITO components
        _ensure_mohito_imports()
        _ensure_pyg_imports()
        
        from mohitoR.gat import ActorNetwork, ActorGNN
        # Upstream sets anomaly detection True; disable for performance/stability.
        try:
            torch.autograd.set_detect_anomaly(False)
        except Exception:
            pass
        
        # Create actor (random init)
        self.actor = ActorNetwork(
            num_state_features=config.feature_len,
            LR_A=config.learning_rate,
            BETA=config.entropy_coef,
            hidden_dim_actor=config.hidden_dim,
            num_layers=config.num_layers_actor,
            heads=config.heads,
            grad_clip=config.max_grad_norm,
        ).to(self.device)
        
        # Create critic (MLP on pooled graph features)
        # Input: pooled features from actor's GNN output
        critic_input_dim = config.feature_len * config.heads
        self.critic = MLPCritic(critic_input_dim, hidden_dim=64).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate,
        )
        
        # Compute hashes for reproducibility audit
        self.config_hash, self.graph_hash, self.dataset_hash = _compute_hashes(env_cfg)
        self.git_commit = _get_git_commit()
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_service_rate = 0.0
        
        # Logging
        self.train_log_path = self.run_dir / "train_log.jsonl"
        self.reward_terms_path = self.run_dir / "reward_terms.jsonl"
        
        # Episode buffer for A2C
        self.ep_log_probs: List[torch.Tensor] = []
        self.ep_values: List[torch.Tensor] = []
        self.ep_rewards: List[float] = []
        self.ep_entropies: List[torch.Tensor] = []

        try:
            self._scaler = torch.amp.GradScaler(
                device_type="cuda",
                enabled=bool(self.config.use_amp) and self.device.type == "cuda",
            )
        except Exception:
            self._scaler = torch.cuda.amp.GradScaler(
                enabled=bool(self.config.use_amp) and self.device.type == "cuda"
            )

    def _autocast_ctx(self):
        if not (bool(self.config.use_amp) and self.device.type == "cuda"):
            return torch.autocast(device_type="cpu", enabled=False)
        dtype = str(self.config.amp_dtype).lower().strip()
        if dtype == "bf16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        
        LOG.info(f"MOHITOTrainer initialized on {self.device}")
        LOG.info(f"Config hash: {self.config_hash}, Graph hash: {self.graph_hash}")
    
    def _pool_graph_features(self, graph_data: Any, h: torch.Tensor) -> torch.Tensor:
        """Global mean pooling of node features."""
        # Simple mean pooling over all nodes
        return h.mean(dim=0, keepdim=True)
    
    def _select_action(
        self,
        features: Dict[str, np.ndarray],
        epsilon: float = 0.0,
    ) -> Tuple[Optional[int], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Select action using actor, return log_prob, value, and entropy.
        
        Returns:
            action: Selected stop ID or None
            log_prob: Log probability of selected action
            value: State value from critic
            entropy: Policy entropy
        """
        actions = features["actions"].astype(np.int64)
        action_mask = features["action_mask"].astype(bool)
        
        if len(actions) == 0:
            return None, None, None, None
        
        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) == 0:
            return None, None, None, None
        
        # Get vehicle info
        vehicle = self.env._get_active_vehicle() if hasattr(self.env, '_get_active_vehicle') else None
        vehicle_idx = getattr(vehicle, 'vehicle_id', 0) if vehicle else 0
        
        # Build MOHITO graph
        graph_data, edge_space, action_space = build_mohito_graph(
            self.env,
            features,
            vehicle_idx,
            self.config.grid_size,
            mode=str(self.config.graph_mode),
        )
        graph_data = graph_data.to(self.device)
        
        # Forward through actor GNN to get node embeddings
        with self._autocast_ctx():
            h = self.actor.main.straightforward(graph_data, training=True)
        
        # Get logits for edge nodes (action candidates)
        # MOHITO uses sum of edge node features as logits
        edge_space_tensor = torch.tensor(edge_space, dtype=torch.long, device=self.device)
        
        if len(edge_space_tensor) == 0:
            return None, None, None, None
        
        # Compute logits from edge space features
        edge_features = h[edge_space_tensor]
        logits = edge_features.sum(dim=-1)  # Shape: [num_edges]
        
        # Apply action mask: masked actions get -inf logits
        # Map edge indices to action indices (approximate, may need refinement)
        num_actions = len(actions)
        if len(logits) > num_actions:
            logits = logits[:num_actions]
        elif len(logits) < num_actions:
            # Pad with -inf for extra actions
            pad = torch.full(
                (num_actions - len(logits),), -float('inf'), device=self.device
            )
            logits = torch.cat([logits, pad])
        
        # Hard mask: set invalid actions to -inf
        mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
        logits = torch.where(mask_tensor, logits, torch.tensor(-float('inf'), device=self.device))
        
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy (only over unmasked actions)
        valid_probs = probs[mask_tensor]
        if len(valid_probs) > 0:
            entropy = -(valid_probs * torch.log(valid_probs + 1e-10)).sum()
        else:
            entropy = torch.tensor(0.0, device=self.device)
        
        # Sample or greedy
        if epsilon > 0 and np.random.random() < epsilon:
            # Random exploration
            action_idx = np.random.choice(valid_indices)
        else:
            # Sample from distribution
            try:
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample().item()
            except Exception:
                action_idx = int(valid_indices[0])
        
        # Ensure mask compliance
        if not action_mask[action_idx]:
            action_idx = int(valid_indices[0])
        
        # Log probability
        log_prob = torch.log(probs[action_idx] + 1e-10)
        
        # Compute value from critic
        pooled = self._pool_graph_features(graph_data, h)
        with self._autocast_ctx():
            value = self.critic(pooled).squeeze()
        
        return int(actions[action_idx]), log_prob, value, entropy
    
    def _update(self, bootstrap_value: torch.Tensor) -> Dict[str, float]:
        """Perform A2C update on the current rollout buffer.

        This runs frequently (every update_every_steps or on episode end) to
        bound GPU memory usage under long horizons.
        """
        if len(self.ep_rewards) == 0:
            return {}
        
        # Compute returns with bootstrapping from the next state value.
        returns: List[float] = []
        R = float(bootstrap_value.detach().float().item())
        for r in reversed(self.ep_rewards):
            R = float(r) + float(self.config.gamma) * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.stack(self.ep_values).float() if self.ep_values else torch.zeros_like(returns)
        log_probs = torch.stack(self.ep_log_probs).float() if self.ep_log_probs else torch.zeros_like(returns)
        entropies = torch.stack(self.ep_entropies).float() if self.ep_entropies else torch.zeros_like(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Advantage
        advantages = returns - values.detach()
        
        # Actor loss (policy gradient with baseline)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus (encourage exploration)
        entropy_loss = -entropies.mean()
        
        # Total loss
        total_loss = (
            actor_loss
            + self.config.value_loss_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )
        
        # Update actor
        self.actor.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        if self._scaler.is_enabled():
            self._scaler.scale(total_loss).backward()
            self._scaler.unscale_(self.actor.optimizer)
            self._scaler.unscale_(self.critic_optimizer)
        else:
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)

        if self._scaler.is_enabled():
            self._scaler.step(self.actor.optimizer)
            self._scaler.step(self.critic_optimizer)
            self._scaler.update()
        else:
            self.actor.optimizer.step()
            self.critic_optimizer.step()

        self.actor.scheduler.step()
        
        # Clear buffers
        self.ep_log_probs.clear()
        self.ep_values.clear()
        self.ep_rewards.clear()
        self.ep_entropies.clear()
        
        return {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropies.mean().item() if len(entropies) > 0 else 0.0,
            "total_loss": total_loss.item(),
        }

    def _compute_state_value(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute critic value for bootstrapping (no grad)."""
        vehicle = self.env._get_active_vehicle() if hasattr(self.env, "_get_active_vehicle") else None
        vehicle_idx = getattr(vehicle, "vehicle_id", 0) if vehicle else 0
        graph_data, edge_space, action_space = build_mohito_graph(
            self.env,
            features,
            vehicle_idx,
            self.config.grid_size,
            mode=str(self.config.graph_mode),
        )
        graph_data = graph_data.to(self.device)
        with torch.no_grad():
            with self._autocast_ctx():
                h = self.actor.main.straightforward(graph_data, training=False)
                pooled = self._pool_graph_features(graph_data, h)
                value = self.critic(pooled).squeeze()
        return value.detach()
    
    def _save_checkpoint(self, path: Path, is_best: bool = False):
        """Save checkpoint with unified schema."""
        checkpoint = {
            "model_state_dict": self.actor.main.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.actor.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "seed": self.config.seed,
            "config_hash": self.config_hash,
            "graph_hash": self.graph_hash,
            "dataset_hash": self.dataset_hash,
            "git_commit": self.git_commit,
            "best_service_rate": self.best_service_rate,
        }
        torch.save(checkpoint, path)
        LOG.info(f"Saved checkpoint to {path}")
    
    def _log_episode(self, ep_info: Dict[str, Any], loss_info: Dict[str, float]):
        """Log episode metrics."""
        log_entry = {
            "timestamp": time.time(),
            "global_step": self.global_step,
            "episode": self.episode_count,
            **ep_info,
            **loss_info,
        }
        with open(self.train_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def train(
        self,
        total_steps: Optional[int] = None,
        eval_callback: Optional[Callable[[int], Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Run training loop.
        
        Args:
            total_steps: Override total steps from config
            eval_callback: Optional callback for periodic evaluation
            
        Returns:
            Final training statistics
        """
        if total_steps is None:
            total_steps = self.config.total_steps
        
        LOG.info(f"Starting MOHITO training for {total_steps} steps")
        
        # Epsilon schedule (linear decay)
        def get_epsilon(step: int) -> float:
            return max(0.05, 1.0 - step / (total_steps * 0.5))
        
        start_time = time.time()
        self.env.reset()
        features = self.env.get_feature_batch()  # Get proper feature dict
        
        while self.global_step < total_steps:
            epsilon = get_epsilon(self.global_step)
            
            # Select action
            action, log_prob, value, entropy = self._select_action(features, epsilon)
            
            if action is None:
                # No valid action, reset episode
                self.env.reset()
                features = self.env.get_feature_batch()
                continue
            
            # Take step - ONLY use env.step() reward (no custom shaping)
            # Note: EventDrivenEnv.step() returns (obs, reward, done, info) - 4 values
            _, reward, done, info = self.env.step(action)
            
            # Store transition for A2C
            if log_prob is not None:
                self.ep_log_probs.append(log_prob.float())
                self.ep_values.append(value.float())
                self.ep_rewards.append(reward)
                self.ep_entropies.append(entropy.float())
            
            self.global_step += 1
            
            # Get next features from env (step returns obs dict, not feature batch)
            features = self.env.get_feature_batch()
            
            # Update on fixed rollout window to bound memory (and on episode end).
            if done:
                self.episode_count += 1
                loss_info = self._update(torch.tensor(0.0, device=self.device))

                ep_info = {
                    "served": info.get("served", 0),
                    "churned_waiting": info.get("churned_waiting", 0),
                    "churned_onboard": info.get("churned_onboard", 0),
                    "structural_unserviceable": info.get("structural_unserviceable", 0),
                    "total_requests": info.get("total_requests", 0),
                    "tacc_total": info.get("tacc_total", 0.0),
                    "epsilon": epsilon,
                }

                total = ep_info["total_requests"]
                if total > 0:
                    service_rate = ep_info["served"] / total
                    churn_rate = (ep_info["churned_waiting"] + ep_info["churned_onboard"]) / total
                else:
                    service_rate = 0.0
                    churn_rate = 0.0
                ep_info["service_rate"] = service_rate
                ep_info["churn_rate"] = churn_rate

                self._log_episode(ep_info, loss_info)

                if self.global_step % self.config.log_every_steps < 100:
                    LOG.info(
                        f"Step {self.global_step}/{total_steps} | "
                        f"Ep {self.episode_count} | "
                        f"SR: {service_rate:.2%} | "
                        f"Churn: {churn_rate:.2%} | "
                        f"TACC: {ep_info['tacc_total']:.1f} | "
                        f"Loss: {loss_info.get('total_loss', 0):.4f}"
                    )

                if service_rate > self.best_service_rate:
                    self.best_service_rate = service_rate
                    self._save_checkpoint(self.run_dir / "mohito_actor_best.pth", is_best=True)

                if self.global_step % self.config.checkpoint_every_steps < 100:
                    self._save_checkpoint(self.run_dir / f"checkpoint_{self.global_step}.pth")

                if eval_callback and self.global_step % self.config.eval_every_steps < 100:
                    eval_callback(self.global_step)

                self.env.reset()
                features = self.env.get_feature_batch()
            elif int(self.config.update_every_steps) > 0 and len(self.ep_rewards) >= int(self.config.update_every_steps):
                bootstrap = self._compute_state_value(features)
                loss_info = self._update(bootstrap)
                if self.global_step % self.config.log_every_steps < 100:
                    LOG.info(
                        "MOHITO rollout update at step %d: loss=%.4f",
                        self.global_step,
                        float(loss_info.get("total_loss", 0.0)),
                    )
        
        # Save final model
        self._save_checkpoint(self.run_dir / "mohito_actor_final.pth")
        
        # Also save actor weights in evaluator-compatible format
        torch.save(
            self.actor.main.state_dict(),
            self.run_dir / "mohito_actor_weights.pth"
        )
        
        elapsed = time.time() - start_time
        LOG.info(f"Training completed in {elapsed/3600:.2f} hours")
        LOG.info(f"Best service rate: {self.best_service_rate:.2%}")
        
        return {
            "total_steps": self.global_step,
            "total_episodes": self.episode_count,
            "best_service_rate": self.best_service_rate,
            "elapsed_seconds": elapsed,
        }
    
    def close(self):
        """Clean up resources."""
        pass
