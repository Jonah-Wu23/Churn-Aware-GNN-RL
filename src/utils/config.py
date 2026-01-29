"""Config loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file.

    This helper keeps YAML parsing localized and optional.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyyaml is required to load config files") from exc

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    _validate_potential_config(config)
    return config


def _validate_potential_config(config: Dict[str, Any]) -> None:
    curriculum = config.get("curriculum", {})
    stage_params = curriculum.get("stage_params", {}) if isinstance(curriculum, dict) else {}
    l1 = stage_params.get("L1") if isinstance(stage_params, dict) else None
    if not isinstance(l1, dict):
        return
    env_overrides = l1.get("env_overrides", {})
    if not isinstance(env_overrides, dict):
        raise ValueError("L1 env_overrides must be a dict to configure reward_potential_alpha.")
    if "reward_potential_alpha" not in env_overrides:
        raise ValueError("L1 env_overrides missing reward_potential_alpha (potential shaping not configured).")
    if "reward_potential_lost_weight" not in env_overrides:
        raise ValueError("L1 env_overrides missing reward_potential_lost_weight.")
    alpha = float(env_overrides.get("reward_potential_alpha", 0.0))
    if alpha <= 0.0:
        raise ValueError("L1 reward_potential_alpha must be > 0 when potential shaping is enabled.")
