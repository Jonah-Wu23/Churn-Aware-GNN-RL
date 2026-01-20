"""Reward weight ramping for smooth phase transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

LOG = logging.getLogger(__name__)

# Default fields that can be linearly interpolated
DEFAULT_RAMP_FIELDS = [
    "reward_service",
    "reward_waiting_churn_penalty",
    "reward_onboard_churn_penalty",
    "reward_fairness_weight",
    "reward_cvar_penalty",
]


@dataclass
class RampConfig:
    """Configuration for reward weight ramping between phases."""
    reward_ramp_steps: int
    ramp_fields: List[str] = field(default_factory=lambda: DEFAULT_RAMP_FIELDS.copy())
    w2: Dict[str, float] = field(default_factory=dict)         # phase2 weights
    w3_target: Dict[str, float] = field(default_factory=dict)  # phase3 target weights


def compute_ramped_weights(phase_step: int, ramp_config: RampConfig) -> Tuple[Dict[str, float], float]:
    """
    Compute interpolated reward weights at a given phase step.
    
    Args:
        phase_step: Current step within the phase
        ramp_config: Ramp configuration with w2 and w3_target
    
    Returns:
        Tuple of (ramped_weights dict, alpha coefficient)
    """
    if ramp_config.reward_ramp_steps <= 0:
        alpha = 1.0
    else:
        alpha = min(1.0, max(0.0, phase_step / ramp_config.reward_ramp_steps))
    
    ramped: Dict[str, float] = {}
    for field_name in ramp_config.ramp_fields:
        w2_val = ramp_config.w2.get(field_name, 0.0)
        # If w3_target is empty or field not in it, inherit from w2
        w3_val = ramp_config.w3_target.get(field_name, w2_val)
        interpolated = (1.0 - alpha) * w2_val + alpha * w3_val
        # Clamp to non-negative (reward weights should be >= 0)
        ramped[field_name] = max(0.0, interpolated)
    
    return ramped, alpha


def get_phase3_target_weights(
    phase2_overrides: Optional[Dict[str, float]],
    phase3_overrides: Optional[Dict[str, float]],
    base_env_cfg: Dict[str, object],
) -> Dict[str, float]:
    """
    Determine phase3 target weights with fallback inheritance.
    
    If phase3_overrides is empty/None, inherit from phase2_overrides.
    If both are empty, use base env config values.
    
    Args:
        phase2_overrides: Phase2 reward weight overrides
        phase3_overrides: Phase3 reward weight overrides
        base_env_cfg: Base environment configuration
    
    Returns:
        Final phase3 target weights
    """
    # Start with base env config
    result: Dict[str, float] = {}
    for field_name in DEFAULT_RAMP_FIELDS:
        if field_name in base_env_cfg:
            result[field_name] = float(base_env_cfg[field_name])
    
    # Apply phase2 overrides as base
    if phase2_overrides:
        for field_name in DEFAULT_RAMP_FIELDS:
            if field_name in phase2_overrides:
                result[field_name] = float(phase2_overrides[field_name])
    
    # Apply phase3 overrides if not empty
    if phase3_overrides and len(phase3_overrides) > 0:
        for field_name in DEFAULT_RAMP_FIELDS:
            if field_name in phase3_overrides:
                result[field_name] = float(phase3_overrides[field_name])
        LOG.info("Phase3 using explicit overrides: %s", result)
    else:
        # Inherit from phase2
        LOG.info("Phase3 overrides empty, inheriting from phase2: %s", result)
    
    return result


def build_ramp_config(
    phase2_overrides: Optional[Dict[str, float]],
    phase3_overrides: Optional[Dict[str, float]],
    base_env_cfg: Dict[str, object],
    reward_ramp_steps: int,
    ramp_fields: Optional[List[str]] = None,
) -> RampConfig:
    """
    Build a RampConfig for phase3 linear interpolation.
    
    Args:
        phase2_overrides: Phase2 reward weight overrides
        phase3_overrides: Phase3 reward weight overrides  
        base_env_cfg: Base environment configuration
        reward_ramp_steps: Number of steps to complete the ramp
        ramp_fields: List of fields to ramp (uses default if None)
    
    Returns:
        RampConfig ready for use in training
    """
    fields = ramp_fields if ramp_fields is not None else DEFAULT_RAMP_FIELDS.copy()
    
    # Compute w2 (phase2 weights)
    w2: Dict[str, float] = {}
    for field_name in fields:
        if phase2_overrides and field_name in phase2_overrides:
            w2[field_name] = float(phase2_overrides[field_name])
        elif field_name in base_env_cfg:
            w2[field_name] = float(base_env_cfg[field_name])
        else:
            w2[field_name] = 0.0
    
    # Compute w3_target (phase3 target weights)
    w3_target = get_phase3_target_weights(phase2_overrides, phase3_overrides, base_env_cfg)
    
    return RampConfig(
        reward_ramp_steps=reward_ramp_steps,
        ramp_fields=fields,
        w2=w2,
        w3_target=w3_target,
    )
