"""Wu2024 A2C trainer for in-domain training.

This module trains Wu2024's Pointer Network from random initialization
using our EventDrivenEnv and A2C algorithm.

The trainer uses the same reward and mask as other baselines (MAPPO/CPO/HCRide),
ensuring fair comparison.

Reference:
    Wu, X. et al. (2024). Multi-Agent Deep Reinforcement Learning based
    Real-time Planning Approach for Responsive Customized Bus Routes.
    Computers & Industrial Engineering, 186, 109764.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
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
from src.baselines.wu2024_adapter import (
    Wu2024PointerNet,
    _build_static_features,
    _build_dynamic_features,
    _select_candidate_stops,
)

LOG = logging.getLogger(__name__)


@dataclass
class Wu2024TrainConfig:
    """Wu2024 training configuration."""
    seed: int = 7
    total_steps: int = 200_000
    gamma: float = 0.99
    learning_rate: float = 0.0005
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 5.0
    log_every_steps: int = 1000
    eval_every_steps: int = 10_000
    checkpoint_every_steps: int = 10_000
    device: str = "cuda"
    # Wu2024-specific
    kmax: int = 32
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.1


class SharedEncoderCritic(nn.Module):
    """Critic that shares encoder structure with Wu2024 model.
    
    Uses same Conv1d encoders as the actor to ensure consistent
    feature representation for value estimation.
    """
    
    def __init__(self, static_size: int, dynamic_size: int, hidden_size: int):
        super().__init__()
        self.static_encoder = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.dynamic_encoder = nn.Conv1d(dynamic_size, hidden_size, kernel_size=1)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        static: torch.Tensor,
        dynamic: torch.Tensor,
    ) -> torch.Tensor:
        """Compute state value.
        
        Args:
            static: [batch, static_size, seq_len]
            dynamic: [batch, dynamic_size, seq_len]
            
        Returns:
            value: [batch, 1]
        """
        static_h = self.static_encoder(static)  # [batch, hidden, seq]
        dynamic_h = self.dynamic_encoder(dynamic)
        
        # Global average pooling
        static_pooled = static_h.mean(dim=2)  # [batch, hidden]
        dynamic_pooled = dynamic_h.mean(dim=2)
        
        combined = torch.cat([static_pooled, dynamic_pooled], dim=1)
        return self.value_head(combined)


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


class Wu2024Trainer:
    """A2C trainer for Wu2024 with in-domain training.
    
    Uses the same EventDrivenEnv and reward/mask protocol as other baselines.
    
    Key design decisions (per user requirements):
    - mask = logits hard mask (-inf for invalid actions)
    - entropy only computed over unmasked actions
    - critic shares encoder structure with actor
    """
    
    def __init__(
        self,
        env: EventDrivenEnv,
        config: Wu2024TrainConfig,
        run_dir: Path,
        env_cfg: Dict[str, Any],
    ):
        self.env = env
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Feature dimensions (V0 mapping)
        self.static_size = 3  # x, y, travel_time
        self.dynamic_size = 3  # waiting_count, load, current_time
        
        # Create actor (random init)
        self.actor = Wu2024PointerNet(
            static_size=self.static_size,
            dynamic_size=self.dynamic_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            device=self.device,
        ).to(self.device)
        
        # Create critic (shared encoder structure)
        self.critic = SharedEncoderCritic(
            static_size=self.static_size,
            dynamic_size=self.dynamic_size,
            hidden_size=config.hidden_size,
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate,
        )
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
        
        LOG.info(f"Wu2024Trainer initialized on {self.device}")
        LOG.info(f"Config hash: {self.config_hash}, Graph hash: {self.graph_hash}")
    
    def _select_action(
        self,
        features: Dict[str, np.ndarray],
        epsilon: float = 0.0,
    ) -> Tuple[Optional[int], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Select action using actor, return log_prob, value, and entropy.
        
        Uses hard mask (logits = -inf for invalid actions) per user requirement.
        Entropy computed only over unmasked actions.
        
        Returns:
            action: Selected stop ID or None
            log_prob: Log probability of selected action
            value: State value from critic
            entropy: Policy entropy (unmasked only)
        """
        actions = features["actions"].astype(np.int64)
        mask = features["action_mask"].astype(bool)
        
        if len(actions) == 0:
            return None, None, None, None
        
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return None, None, None, None
        
        kmax = self.config.kmax
        
        # Build candidates
        candidate_stops, action_mask = _select_candidate_stops(features, kmax)
        
        # Build features
        static = _build_static_features(self.env, features, kmax, candidate_stops)
        dynamic = _build_dynamic_features(self.env, features, kmax, candidate_stops)
        
        # Convert to tensors
        static_t = torch.tensor(static, dtype=torch.float32, device=self.device)
        dynamic_t = torch.tensor(dynamic, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(action_mask.astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get logits from actor (before softmax)
        static_hidden = self.actor.static_encoder(static_t)
        dynamic_hidden = self.actor.dynamic_encoder(dynamic_t)
        decoder_input = self.actor.x0.expand(1, -1, -1)
        decoder_hidden = self.actor.decoder(decoder_input)
        
        logits, _ = self.actor.pointer(static_hidden, dynamic_hidden, decoder_hidden, None)
        
        # HARD MASK: set invalid actions to -inf
        # This ensures masked actions have zero probability
        mask_bool = mask_t.squeeze() > 0.5
        logits = logits.squeeze()  # [kmax]
        logits = torch.where(mask_bool, logits, torch.tensor(-float('inf'), device=self.device))
        
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # ENTROPY: only over unmasked actions
        valid_probs = probs[mask_bool]
        if len(valid_probs) > 0 and valid_probs.sum() > 0:
            normalized_probs = valid_probs / (valid_probs.sum() + 1e-10)
            entropy = -(normalized_probs * torch.log(normalized_probs + 1e-10)).sum()
        else:
            entropy = torch.tensor(0.0, device=self.device)
        
        # Sample action
        if epsilon > 0 and np.random.random() < epsilon:
            # Random exploration over valid actions
            valid_kmax = [i for i in range(kmax) if action_mask[i]]
            if valid_kmax:
                action_idx = np.random.choice(valid_kmax)
            else:
                return None, None, None, None
        else:
            # Sample from distribution
            try:
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample().item()
            except Exception:
                # Fallback to first valid
                valid_kmax = [i for i in range(kmax) if action_mask[i]]
                action_idx = valid_kmax[0] if valid_kmax else 0
        
        # Ensure valid action
        if not action_mask[action_idx]:
            valid_kmax = [i for i in range(kmax) if action_mask[i]]
            action_idx = valid_kmax[0] if valid_kmax else 0
        
        # Log probability
        log_prob = torch.log(probs[action_idx] + 1e-10)
        
        # Compute value from critic
        value = self.critic(static_t, dynamic_t).squeeze()
        
        # Map back to stop ID
        selected_stop = candidate_stops[action_idx]
        if selected_stop < 0 or selected_stop not in actions:
            # Fallback to first valid action from original list
            return int(actions[valid_indices[0]]), log_prob, value, entropy
        
        return int(selected_stop), log_prob, value, entropy
    
    def _update(self) -> Dict[str, float]:
        """Perform A2C update at end of episode."""
        if len(self.ep_rewards) == 0:
            return {}
        
        # Compute returns with bootstrapping
        returns = []
        R = 0.0
        for r in reversed(self.ep_rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.stack(self.ep_values) if self.ep_values else torch.zeros_like(returns)
        log_probs = torch.stack(self.ep_log_probs) if self.ep_log_probs else torch.zeros_like(returns)
        entropies = torch.stack(self.ep_entropies) if self.ep_entropies else torch.zeros_like(returns)
        
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
        
        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.max_grad_norm
        )
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
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
    
    def _save_checkpoint(self, path: Path, is_best: bool = False):
        """Save checkpoint with unified schema."""
        checkpoint = {
            "model_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
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
        
        LOG.info(f"Starting Wu2024 training for {total_steps} steps")
        
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
                self.ep_log_probs.append(log_prob)
                self.ep_values.append(value)
                self.ep_rewards.append(reward)
                self.ep_entropies.append(entropy)
            
            self.global_step += 1
            
            # Get next features from env (step returns obs dict, not feature batch)
            features = self.env.get_feature_batch()
            
            # Episode end
            if done:
                self.episode_count += 1
                
                # Perform A2C update
                loss_info = self._update()
                
                # Get episode info
                ep_info = {
                    "served": info.get("served", 0),
                    "churned_waiting": info.get("churned_waiting", 0),
                    "churned_onboard": info.get("churned_onboard", 0),
                    "structural_unserviceable": info.get("structural_unserviceable", 0),
                    "total_requests": info.get("total_requests", 0),
                    "tacc_total": info.get("tacc_total", 0.0),
                    "epsilon": epsilon,
                }
                
                # Compute service rate
                total = ep_info["total_requests"]
                if total > 0:
                    service_rate = ep_info["served"] / total
                    churn_rate = (ep_info["churned_waiting"] + ep_info["churned_onboard"]) / total
                else:
                    service_rate = 0.0
                    churn_rate = 0.0
                
                ep_info["service_rate"] = service_rate
                ep_info["churn_rate"] = churn_rate
                
                # Log
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
                
                # Update best
                if service_rate > self.best_service_rate:
                    self.best_service_rate = service_rate
                    self._save_checkpoint(
                        self.run_dir / "wu2024_model_best.pt", is_best=True
                    )
                
                # Checkpoint
                if self.global_step % self.config.checkpoint_every_steps < 100:
                    self._save_checkpoint(
                        self.run_dir / f"checkpoint_{self.global_step}.pt"
                    )
                
                # Evaluation callback
                if eval_callback and self.global_step % self.config.eval_every_steps < 100:
                    eval_callback(self.global_step)
                
                # Reset
                self.env.reset()
                features = self.env.get_feature_batch()
        
        # Save final model
        self._save_checkpoint(self.run_dir / "wu2024_model_final.pt")
        
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
