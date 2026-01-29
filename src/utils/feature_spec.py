"""Feature specification utilities.

This module provides a single source of truth for feature dimensions,
ensuring consistency across training, evaluation, and testing.
"""

from __future__ import annotations

from typing import Any, Dict


def get_edge_dim(env_cfg: Dict[str, Any]) -> int:
    """Determine edge_dim based on environment configuration.
    
    This is the ONLY function that should determine edge_dim.
    Do NOT hardcode edge_dim values elsewhere.
    
    Args:
        env_cfg: Environment configuration dictionary containing at minimum
                 the 'use_fleet_potential' key.
    
    Returns:
        4 if FAEP is disabled (default), 5 if FAEP is enabled.
    
    Example:
        >>> get_edge_dim({"use_fleet_potential": False})
        4
        >>> get_edge_dim({"use_fleet_potential": True})
        5
        >>> get_edge_dim({})  # default when key missing
        4
    """
    if env_cfg.get("use_fleet_potential", False):
        return 5
    return 4


def validate_checkpoint_edge_dim(
    checkpoint_edge_dim: int,
    env_edge_dim: int,
    use_fleet_potential: bool,
) -> None:
    """Validate checkpoint edge_dim against environment configuration.
    
    Raises ValueError with clear message if incompatible.
    
    Args:
        checkpoint_edge_dim: edge_dim from saved model checkpoint.
        env_edge_dim: edge_dim derived from current env config.
        use_fleet_potential: whether FAEP is enabled in current config.
    
    Raises:
        ValueError: If dimensions are incompatible.
    """
    if checkpoint_edge_dim != env_edge_dim:
        if use_fleet_potential and checkpoint_edge_dim == 4:
            raise ValueError(
                f"Checkpoint edge_dim={checkpoint_edge_dim} incompatible with "
                f"env edge_dim={env_edge_dim} (use_fleet_potential=true). "
                "Disable FAEP or retrain with FAEP enabled."
            )
        elif not use_fleet_potential and checkpoint_edge_dim == 5:
            raise ValueError(
                f"Checkpoint edge_dim={checkpoint_edge_dim} incompatible with "
                f"env edge_dim={env_edge_dim} (use_fleet_potential=false). "
                "Enable FAEP or use a checkpoint trained without FAEP."
            )
        else:
            raise ValueError(
                f"Checkpoint edge_dim={checkpoint_edge_dim} does not match "
                f"env edge_dim={env_edge_dim}."
            )
