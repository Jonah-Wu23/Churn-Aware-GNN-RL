"""Fixed-seed evaluation checkpointer for model selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
import copy
import hashlib
import json
import logging

import numpy as np
import torch
from torch import nn

LOG = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of a fixed-seed evaluation."""
    global_step: int
    stage: str
    phase: str
    alpha: float
    eval_seeds: List[int]
    per_seed_metrics: List[Dict[str, float]]
    mean_rho: float
    median_rho: float
    mean_service_rate: float
    env_cfg_hash: str
    replay_size: int
    
    def to_dict(self) -> Dict[str, object]:
        return {
            "global_step": self.global_step,
            "stage": self.stage,
            "phase": self.phase,
            "alpha": self.alpha,
            "eval_seeds": self.eval_seeds,
            "per_seed_metrics": self.per_seed_metrics,
            "mean_rho": self.mean_rho,
            "median_rho": self.median_rho,
            "mean_service_rate": self.mean_service_rate,
            "env_cfg_hash": self.env_cfg_hash,
            "replay_size": self.replay_size,
        }


@dataclass
class EvalConfig:
    """Configuration for evaluation checkpointer."""
    eval_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1000])
    eval_interval_steps: int = 5000
    eval_epsilon: float = 0.0
    eval_episodes_per_seed: int = 1
    collapse_drop_delta: float = 0.10
    collapse_min_rho: float = 0.15
    collapse_patience: int = 2


class EvaluationCheckpointer:
    """
    Fixed-seed evaluation for model selection.
    
    Evaluates model with epsilon=0 on fixed seeds to select the best checkpoint
    based on mean_rho rather than training episode extremes.
    """
    
    def __init__(
        self,
        config: EvalConfig,
        env_factory: Callable[[], object],
        select_action_fn: Callable[[nn.Module, Dict[str, np.ndarray]], Optional[int]],
        compute_rho_fn: Callable[[Dict[str, float]], float],
        device: torch.device,
        gamma: float = 1.0,  # H1: gamma must be passed in, not hardcoded
    ):
        """
        Args:
            config: Evaluation configuration
            env_factory: Factory function to create eval environments
            select_action_fn: Function to select action given model and features
            compute_rho_fn: Function to compute rho from episode info
            device: Torch device for model inference
            gamma: Gamma parameter for rho calculation (must match runner's gamma)
        """
        self.config = config
        self.env_factory = env_factory
        self.select_action_fn = select_action_fn
        self.compute_rho_fn = compute_rho_fn
        self.device = device
        self.gamma = gamma  # Store gamma for consistency check
        
        self.best_rho: float = -1.0
        self.best_result: Optional[EvalResult] = None
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.history: List[EvalResult] = []
        self.collapse_counter: int = 0
        self._last_eval_global_step: int = 0
    
    def _run_greedy_episode(self, model: nn.Module, env) -> Dict[str, float]:
        """Run a single episode with epsilon=0 (greedy).
        
        Uses runner's public functions for consistent metric calculation.
        """
        # Import runner's public functions for unified metrics
        from src.train.runner import (
            compute_eligible,
            compute_service_rate_simple,
            _compute_service_rate,
            verify_request_conservation,
        )
        
        model.eval()
        env.reset()
        total_reward = 0.0
        done = False
        info: Dict[str, float] = {}
        
        with torch.no_grad():
            while not done:
                features = env.get_feature_batch()
                action = self.select_action_fn(model, features)
                if action is None:
                    # No valid action, episode should end
                    break
                _, reward, done, info = env.step(int(action))
                total_reward += float(reward)
        
        # Extract raw metrics from info
        served = float(info.get("served", 0))
        waiting_churned = float(info.get("waiting_churned", 0))
        onboard_churned = float(info.get("onboard_churned", 0))
        waiting_timeouts = float(info.get("waiting_timeouts", 0))
        waiting_remaining = float(info.get("waiting_remaining", 0))
        onboard_remaining = float(info.get("onboard_remaining", 0))
        structural_unserviceable = float(info.get("structural_unserviceable", 0))
        stuckness = float(info.get("stuckness", 0))
        total_requests = float(info.get("total_requests", 0))
        
        # Use unified functions for metric calculation
        eligible_total = compute_eligible(info)
        service_rate = _compute_service_rate(info)
        service_rate_simple = compute_service_rate_simple(info)
        rho = self.compute_rho_fn(info)
        
        # H1: Consistency check using stored gamma (not hardcoded)
        service_rate_from_rho = rho * (1.0 + self.gamma * stuckness)
        consistency_diff = abs(service_rate - service_rate_from_rho)
        
        if consistency_diff > 1e-4:
            LOG.warning(
                "⚠️ 口径不一致: service_rate=%.6f, service_rate_from_rho=%.6f, diff=%.6f, gamma=%.2f",
                service_rate, service_rate_from_rho, consistency_diff, self.gamma
            )
        
        # H3: Conservation check
        if not verify_request_conservation(info):
            LOG.warning(
                "⚠️ 守恒校验失败: eligible=%.0f, structural=%.0f, total=%.0f",
                eligible_total, structural_unserviceable, total_requests
            )
        
        return {
            # Raw metrics
            "served": served,
            "waiting_churned": waiting_churned,
            "onboard_churned": onboard_churned,
            "waiting_timeouts": waiting_timeouts,
            "waiting_remaining": waiting_remaining,
            "onboard_remaining": onboard_remaining,
            "structural_unserviceable": structural_unserviceable,
            "total_requests": total_requests,
            # Unified metrics
            "eligible_total": eligible_total,
            "service_rate": service_rate,
            "service_rate_simple": service_rate_simple,
            "stuckness": stuckness,
            "rho": rho,
            # Consistency check
            "service_rate_from_rho": service_rate_from_rho,
            "consistency_diff": consistency_diff,
            "episode_return": total_reward,
        }
    
    def evaluate(
        self,
        model: nn.Module,
        global_step: int,
        stage: str,
        phase: str,
        alpha: float,
        env_cfg_hash: str,
        replay_size: int,
        save_dir: Path,
        log_handle=None,
    ) -> EvalResult:
        """
        Run evaluation on all fixed seeds and update best checkpoint.
        
        Args:
            model: Model to evaluate
            global_step: Current global step
            stage: Current stage name
            phase: Current phase name
            alpha: Current ramp alpha
            env_cfg_hash: Hash of current env config
            replay_size: Current replay buffer size
            save_dir: Directory to save best checkpoints
            log_handle: Optional log file handle
        
        Returns:
            EvalResult with evaluation metrics
        """
        self._last_eval_global_step = int(global_step)
        per_seed: List[Dict[str, float]] = []
        
        for seed in self.config.eval_seeds:
            for _ in range(self.config.eval_episodes_per_seed):
                env = self.env_factory()
                env.seed(seed)
                metrics = self._run_greedy_episode(model, env)
                per_seed.append(metrics)
        
        rhos = [m["rho"] for m in per_seed]
        service_rates = [m["service_rate"] for m in per_seed]
        
        result = EvalResult(
            global_step=global_step,
            stage=stage,
            phase=phase,
            alpha=alpha,
            eval_seeds=self.config.eval_seeds,
            per_seed_metrics=per_seed,
            mean_rho=float(np.mean(rhos)),
            median_rho=float(np.median(rhos)),
            mean_service_rate=float(np.mean(service_rates)),
            env_cfg_hash=env_cfg_hash,
            replay_size=replay_size,
        )
        self.history.append(result)
        
        # Log evaluation result
        if log_handle:
            log_handle.write(json.dumps({
                "type": "evaluation",
                **result.to_dict(),
            }, ensure_ascii=False) + "\n")
            log_handle.flush()
        
        LOG.info("Eval at step=%d: mean_rho=%.4f, median_rho=%.4f, service_rate=%.4f",
                 global_step, result.mean_rho, result.median_rho, result.mean_service_rate)
        
        # Update best checkpoint if improved
        if result.mean_rho > self.best_rho:
            self.best_rho = result.mean_rho
            self.best_result = result
            self.best_model_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            self._save_best(model, result, save_dir / "best_rho.pt")
            LOG.info("New best_rho=%.4f at global_step=%d", result.mean_rho, global_step)
            self.collapse_counter = 0
        
        return result
    
    def _save_best(self, model: nn.Module, result: EvalResult, path: Path) -> None:
        """Save best checkpoint with full metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "eval_result": result.to_dict(),
            "global_step": result.global_step,
            "mean_rho": result.mean_rho,
            "stage": result.stage,
            "phase": result.phase,
        }, path)
    
    def check_collapse(self, result: EvalResult) -> bool:
        """
        Check if current evaluation indicates a collapse.
        
        Collapse is detected if:
        - mean_rho < collapse_min_rho
        - OR mean_rho dropped by > collapse_drop_delta from best
        
        Returns True if collapse detected (after patience).
        """
        is_collapse = False
        
        if result.mean_rho < self.config.collapse_min_rho:
            is_collapse = True
            LOG.warning("Collapse signal: rho=%.4f < min=%.4f", 
                       result.mean_rho, self.config.collapse_min_rho)
        elif self.best_rho - result.mean_rho > self.config.collapse_drop_delta:
            is_collapse = True
            LOG.warning("Collapse signal: rho dropped %.4f (%.4f -> %.4f)",
                       self.best_rho - result.mean_rho, self.best_rho, result.mean_rho)
        
        if is_collapse:
            self.collapse_counter += 1
            if self.collapse_counter >= self.config.collapse_patience:
                LOG.error("Collapse confirmed after %d consecutive signals", self.collapse_counter)
                return True
        else:
            self.collapse_counter = 0
        
        return False
    
    def get_rollback_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the best model state for rollback after collapse."""
        return self.best_model_state
    
    def should_evaluate(self, global_step: int) -> bool:
        """Check if evaluation should run at this step."""
        interval = int(self.config.eval_interval_steps)
        if interval <= 0:
            return False
        step = int(global_step)
        return step > 0 and (step - int(self._last_eval_global_step)) >= interval


def compute_env_cfg_hash(env_cfg: Dict[str, object]) -> str:
    """Compute a short hash of env config for auditing."""
    key_fields = [
        "reward_service", "reward_waiting_churn_penalty", "reward_onboard_churn_penalty",
        "reward_fairness_weight", "reward_cvar_penalty", "churn_tol_sec"
    ]
    subset = {k: env_cfg.get(k) for k in key_fields if k in env_cfg}
    return hashlib.md5(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:8]
